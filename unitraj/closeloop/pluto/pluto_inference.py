import time
import torch
import numpy as np
from scipy.special import softmax
from unitraj.datasets.Pluto_dataset.Pluto_test_dataset import PlutoTestDataset
from unitraj.models.pluto.pluto_model import PlanningModel


class PlutoInference:
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.dataset = PlutoTestDataset(cfg, is_validation=True)

    def initialize_model(self):
        """Initialize the model for Pluto."""
        import lightning.pytorch.core.saving as saving
        _original_load = saving.torch.load
        # Wrapper to circumvent weights_only constraints
        def _torch_load_no_weights_only(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)
        saving.torch.load = _torch_load_no_weights_only
        self.model = PlanningModel(config=self.cfg)
        model_ckpt = self.cfg.get('ckpt_path', None)
        if model_ckpt is not None and model_ckpt != "null":
            ckpt = torch.load(model_ckpt, map_location=self.device)
            if 'state_dict' in ckpt:
                state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def run_inference(self, current_state, timestep):
        t1 = time.time()
        with torch.no_grad():
            [pluto_feature] = self.dataset.process_scenario(current_state, timestep-1)
            t2 = time.time()
            ref_lines = pluto_feature['reference_line']

            # batch is the standard UniTraj batch_dict: {'batch_size', 'input_dict', ...}
            from unitraj.datasets.Pluto_dataset.pluto_utils import collate_pluto_dicts, to_feature_tensor_dict
            input_dict = collate_pluto_dicts([to_feature_tensor_dict(pluto_feature)])
            t3 = time.time()

            def to_device(data, device):
                if isinstance(data, torch.Tensor):
                    return data.to(device)
                elif isinstance(data, dict):
                    return {k: to_device(v, device) for k, v in data.items()}
                elif isinstance(data, list):
                    return [to_device(v, device) for v in data]
                return data

            input_dict = to_device(input_dict, self.device)
            t4 = time.time()

            out = self.model.forward(input_dict)
            t5 = time.time()

            # candidate_trajectories = out["candidate_trajectories"][0].cpu().numpy()
            candidate_trajectories = out["trajectory"][0].cpu().numpy()
            probability = out["probability"][0].cpu().numpy()

            if len(candidate_trajectories.shape) == 4:
                n_ref, n_mode, T, C = candidate_trajectories.shape
                candidate_trajectories = candidate_trajectories.reshape(-1, T, C)
                probability = probability.reshape(-1)
            topk = self.cfg.get('candidate_max_num', 20)
            sorted_idx = np.argsort(-probability)
            sorted_candidate_trajectories = candidate_trajectories[sorted_idx][:topk]
            sorted_probability = softmax(probability[sorted_idx][:topk])

            input_data = input_dict.data if hasattr(input_dict, 'data') else input_dict

            # Start rule-based evaluation avoiding NuPlan dependencies
            from unitraj.closeloop.pluto.trajectory_evaluator import TrajectoryEvaluator
            evaluator = TrajectoryEvaluator(self.cfg)
            rule_based_scores = evaluator.evaluate(sorted_candidate_trajectories, input_data, out.get("output_prediction"))

            # Combine rule-based and learning-based scores
            # NOTE: rule_based_scores can be large negative penalties (e.g., -100/-200),
            # while probabilities are in [0, 1]. Combine on comparable scales.
            # Default: use log-probability and a soft feasibility transform for rule-based.
            eps = 1e-12
            learning_weight = float(self.cfg.get('learning_based_score_weight', 0.75))
            rule_weight = float(self.cfg.get('rule_based_score_weight', 0.25))
            penalty_scale = float(self.cfg.get('rule_based_penalty_scale', 50.0))

            # Convert negative-penalty score into (0, 1] feasibility.
            # score==0 -> 1.0, score==-100 -> exp(-2) ~ 0.135 when scale=50
            rule_feas = np.exp(np.clip(rule_based_scores, -1e6, 0.0) / max(penalty_scale, 1e-6))

            # Use log-prob to avoid tiny linear contributions.
            logp = np.log(sorted_probability + eps)
            final_scores = rule_weight * np.log(rule_feas + eps) + learning_weight * logp
            best_idx = int(np.argmax(final_scores))

            # Predictions are in the ego-centric coordinate system (centered at ego agent's position and heading),
            # so we need to rotate and translate them back to the global coordinate system.
            origin = None
            angle = None
            if 'origin' in input_data:
                origin = input_data['origin'][0].detach().cpu().numpy()[:2]
            if 'angle' in input_data:
                angle = float(input_data['angle'][0].detach().cpu().numpy())
            if origin is not None and angle is not None:
                rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                sorted_candidate_trajectories[..., :2] = np.matmul(sorted_candidate_trajectories[..., :2], rot_mat.T) + origin
                if sorted_candidate_trajectories.shape[-1] >= 4:
                    # 2, 3 对应 cos_yaw, sin_yaw，可以直接用旋转矩阵变换
                    sorted_candidate_trajectories[..., 2:4] = np.matmul(sorted_candidate_trajectories[..., 2:4], rot_mat.T)
                if sorted_candidate_trajectories.shape[-1] >= 6:
                    # 4, 5 对应 vx, vy，同样接受旋转矩阵变换
                    sorted_candidate_trajectories[..., 4:6] = np.matmul(sorted_candidate_trajectories[..., 4:6], rot_mat.T)

                if ref_lines is not None:
                    import copy
                    ref_lines = copy.deepcopy(ref_lines)
                    ref_lines['position'] = np.matmul(ref_lines['position'], rot_mat.T) + origin
                    if 'orientation' in ref_lines:
                        ref_lines['orientation'] += angle

            if timestep == 0:  # Visualize at the first prediction step (T=0)
            # if timestep :
                import matplotlib.pyplot as plt
                # 1. 绘制道路参考线
                for i in range(ref_lines['position'].shape[0]):
                    ref_x = ref_lines['position'][i, :, 0]
                    ref_y = ref_lines['position'][i, :, 1]
                    plt.plot(ref_x, ref_y)

                # 2. 绘制最优的 top-k 候选轨迹 (红色)
                for i in range(len(input_dict['reference_line'])):
                    plt.plot(sorted_candidate_trajectories[i, 1:, 0], sorted_candidate_trajectories[i, 1:, 1], 'r')

                # 3. 绘制 Ego 历史+未来的完整位置散点 (黑色星号)
                ego_track = current_state['tracks']['ego']['state']['position']
                ego_hist_x = ego_track[:, 0]
                ego_hist_y = ego_track[:, 1]
                plt.plot(ego_hist_x[::5], ego_hist_y[::5], 'black', marker='*', linestyle='None', label='Ego Trajectory')

                # ====== 重点：绘制当前帧（预测起始帧）的自车朝向 ======
                # 在当前数据集中，历史步长设为 20，因此 current_idx = 20 是当前时刻（T=0）
                current_idx = timestep
                ego_now_x = ego_hist_x[current_idx]
                ego_now_y = ego_hist_y[current_idx]
                ego_now_heading = current_state['tracks']['ego']['state']['heading'][current_idx]

                # 设定一个 4 米长的向量用来代表车头前向指示器
                arrow_length = 4.0
                dx = arrow_length * np.cos(ego_now_heading)
                dy = arrow_length * np.sin(ego_now_heading)

                # 绘制带有箭头的朝向指示（从自车当前位置戳向朝向的角度）
                plt.arrow(
                    ego_now_x, ego_now_y,
                    dx, dy,
                    width=0.3,  # 箭杆宽度
                    head_width=1.2,  # 箭头宽度
                    head_length=1.5,  # 箭头长度
                    fc='green', ec='green',  # 绿色表示当前朝向
                    zorder=10  # 保证画在最上层
                )

                plt.axis('equal')  # 保证 X 和 Y 轴比例 1:1，否则角度会有视觉缩放失真！
                plt.legend()
                plt.show()

            # Format output sequence to [num_agents, future_len, 5]
            T_out = sorted_candidate_trajectories.shape[1]
            C_out = sorted_candidate_trajectories.shape[-1]
            pred_traj = np.zeros((16, min(T_out, 80), 6), dtype=np.float32)
            # Use candidate excluding the history prepended '0' frame logic, just taking the 80 steps
            offset = 1 if T_out == 81 else 0
            pred_traj[0, :80, :C_out] = sorted_candidate_trajectories[best_idx, offset:offset+80]

            info = {}
            if 'agent_tokens' in pluto_feature:
                info['agent_tokens'] = pluto_feature['agent_tokens']
            if 'output_prediction' in out:
                pred = out['output_prediction'][0].detach().cpu().numpy()
                if origin is not None and angle is not None:
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                    pred[..., :2] = np.matmul(pred[..., :2], rot_mat.T) + origin
                    if pred.shape[-1] >= 3:
                        pred[..., 2] += angle
                    if pred.shape[-1] >= 5:
                        pred[..., 3:5] = np.matmul(pred[..., 3:5], rot_mat.T)
                info['prediction'] = pred

            t6 = time.time()
            # print(f"[Pluto Profiling] step: {timestep}, process_scenario: {t2-t1:.3f}s, collate&to_tensor: {t3-t2:.3f}s, to_device: {t4-t3:.3f}s, model_forward: {t5-t4:.3f}s, postprocess: {t6-t5:.3f}s")
        return pred_traj, ref_lines, sorted_candidate_trajectories[:, offset:offset+80], info
