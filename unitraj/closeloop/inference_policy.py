import os
from collections import deque
import numpy as np
import yaml
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.scenario.parse_object_state import parse_object_state
from omegaconf import OmegaConf
from unitraj.closeloop.VBD.vbd_inference import VBDInference
from unitraj.closeloop.pluto.pluto_inference import PlutoInference
from unitraj.closeloop.unitraj.unitraj_inference import UnitrajInference
from unitraj.utils.utils import set_seed
from unitraj.utils.history_logger import VehicleHistoryLogger


# --- 控制器参数 ---
LATERAL_PID_GAINS = (0.6, 0.0, 1.2)  # Kp, Ki, Kd
LONGITUDINAL_PID_GAINS = (2, 0, 0)  # Kp, Ki, Kd
FIXED_DT = 0.1  # (s) 仿真步长

class InferencePolicy(EnvInputPolicy):
    """
    一个专业的混合策略，通过继承 ReplayEgoCarPolicy 来实现行为切换
    - 前 WARMUP_STEPS 步: 自动使用父类 ReplayEgoCarPolicy 的 act 方法
    - WARMUP_STEPS 步后: 使用横向和纵向PID控制器，精准跟随模型生成的轨迹。
    """

    def __init__(self, obj, seed):
        super(InferencePolicy, self).__init__(obj, seed)

        self.Sim = None
        self.base_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml")
        with open(self.base_config_path, "r", encoding="utf-8") as f:
            base_raw = yaml.safe_load(f)
        self.base_cfg = OmegaConf.create(base_raw)


        self.WARMUP_STEPS = 20
        self.replan_frequency = 10          # 中频重规划 (比如 5 帧/0.5s 重新推理一次主周车意图)
        self.cbv_switch_frequency = 20     # 低频切换 CBV (比如 25 帧/2.5s 才重新计算最近危险车)
        self.current_cbv_id = None         # 用于记录当前一直锁定的 CBV

        self.ego_method = self.base_cfg.get('ego_method', self.base_cfg.defaults[0]['method'])
        self.ego_cfg = self.get_final_config(self.ego_method)
        print(f"[Policy] Ego Planner: {self.ego_method}")

        # self.EgoSim = self._init_inference_engine(self.ego_cfg)

        self.use_adv = self.base_cfg.adv_control
        if self.use_adv:
            self.adv_method = self.base_cfg.get('adv_method', 'Pluto')
            self.adv_cfg = self.get_final_config(self.adv_method)
            self.AdvSim = self._init_inference_engine(self.adv_cfg)
        else:
            self.AdvSim = None
        print(f"[Policy] Adv Planner: {self.adv_method if self.use_adv else 'None'}")

        # 初始化横向和纵向PID控制器
        self.lateral_pid = PIDController(*LATERAL_PID_GAINS)
        self.longitudinal_pid = PIDController(*LONGITUDINAL_PID_GAINS)
        self.current_state = self.engine.data_manager.current_scenario
        self.sdc_id = str(self.current_state["metadata"]["sdc_id"])

        self.last_match_idx = 0

        self._pred_traj = None
        self.hist_buffer = deque(maxlen=21)

        # 存储周车推理结果
        self.current_adv_trajs = None
        self.adv_agent_ids = []
        self.info = None

        self.history_logger = VehicleHistoryLogger(log_path=os.path.join(os.path.dirname(__file__), '../../outputs/vehicle_history.csv'))

    def get_final_config(self, method_name):
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        method_file = os.path.join(os.path.dirname(self.base_config_path), "method", f"{method_name}.yaml")
        with open(method_file, 'r', encoding='utf-8') as f:
            method_cfg = yaml.safe_load(f)
        cfg = OmegaConf.create(raw)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(method_cfg))
        OmegaConf.set_struct(cfg, False)  # Open the struct
        set_seed(cfg.seed)
        return cfg

    def _init_inference_engine(self, specific_cfg):
        method_name = specific_cfg.get('model_name')
        if method_name == 'GT':
            return None

        if method_name == 'VBD':
            sim = VBDInference(specific_cfg)
        elif method_name == 'Pluto':
            sim = PlutoInference(specific_cfg)
        elif method_name in ['wayformer', 'autobot', 'MTR']:
            sim = UnitrajInference(specific_cfg)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        sim.initialize_model()
        return sim

    def act(self, agent_id: str):
        current_step = self.engine.episode_step
        time_index = current_step - 1
        sdc_track = self.current_state["tracks"][self.sdc_id]
        ego_vehicle = self.control_object
        # 每帧记录自车历史
        self.history_logger.log(
            frame_id=current_step,
            vehicle_id=self.sdc_id,
            position=tuple(ego_vehicle.position),
            heading=ego_vehicle.heading_theta
        )

        # --- 阶段一: 数据采集 (完全模仿 ReplayEgoCarPolicy) ---
        if time_index <= self.WARMUP_STEPS:
            self._handle_warmup(time_index, sdc_track)
            # 把前期的真实回放数据也录入 buffer
            ego_vehicle = self.control_object
            self.get_hist_buffer([ego_vehicle.position, ego_vehicle.heading_theta, ego_vehicle.speed])
            return None
        else:
            if time_index == self.WARMUP_STEPS + 1:
                pass
            else:
                # 记录每帧闭环控制真实状态到历史里
                ego_vehicle = self.control_object
                self.get_hist_buffer([ego_vehicle.position, ego_vehicle.heading_theta, ego_vehicle.speed])

        # --- 阶段二: 闭环规划 ---
        if time_index % self.replan_frequency == 1:
            self._plan_trajectory(time_index, sdc_track)

        # --- 阶段三: 主车闭环控制 (PID跟踪) ---
        ego_control_value = self._compute_control()

        # --- 阶段四: 周车控制 (Adv Control) ---
        if time_index % self.replan_frequency == 1 and self.use_adv:
            self._apply_adv_control(time_index)
        if time_index == 97:
            self.save_history()
        return ego_control_value

    def _handle_warmup(self, time_index, sdc_track):
        """处理预热阶段：回放数据"""
        if time_index == 0:
            self.engine.agents['default_agent'].plan_traj = self.current_state['tracks'][self.sdc_id]['state']["position"][:self.WARMUP_STEPS]
        state = parse_object_state(sdc_track, time_index)

        if state["valid"]:
            self.control_object.set_position(state["position"])
            self.control_object.set_velocity(state["velocity"])
            self.control_object.set_heading_theta(state["heading"])

    def _plan_trajectory(self, time_index, sdc_track):
        """按设定频率进行本车轨迹推理规划"""
        self.current_state = self.update_state(time_index)

        # ================= [1. 主车推理 ] =================
        infer_out = self.Sim.run_inference(self.current_state, time_index)
        self._pred_traj = infer_out[0]
        self.engine.agents['default_agent'].plan_traj = self._pred_traj[0, :40, :2]
        ref_lines = infer_out[1]
        self.engine.head_renderer.set_reference_lines(ref_lines)
        candidates = infer_out[2] if len(infer_out) > 2 else None
        if candidates is not None:
            self.engine.agents['default_agent'].candidate_trajectories = candidates

        self.info = infer_out[3] if len(infer_out) > 3 else {}
        self.last_match_idx = 0

        # --- 从预测中计算空间风险并获取CBV ---
        self._select_cbv(time_index)

        # ================= [2. 周车推理 ] =================
        self.adv_agent_ids = []
        self.current_adv_trajs = None

        if self.AdvSim is not None and self.use_adv:
            self._run_adv_inference(time_index)

        if self.info is not None and 'top_k_ids' in self.info:
            self.engine.pred_agent_ids = self.info['top_k_ids']
        else:
            self.engine.pred_agent_ids = [self.sdc_id]

    def _select_cbv(self, time_index):
        """利用主车推理给出的周车未来预测，通过空间距离计算选出一辆周围背景车CBV"""
        self.info['top_k_ids'] = []
        if 'prediction' in self.info and 'agent_tokens' in self.info and self._pred_traj is not None:
            predictions = self.info['prediction']
            agent_tokens = self.info['agent_tokens']

            if isinstance(agent_tokens, list) and len(agent_tokens) > 0 and isinstance(agent_tokens[0], list):
                tokens = agent_tokens[0]
            else:
                tokens = agent_tokens

            # -------- 低频切换 CBV 逻辑 --------
            need_switch = False
            if self.current_cbv_id is None:
                need_switch = True
            elif time_index % self.cbv_switch_frequency == 1:
                need_switch = True
            elif self.current_cbv_id not in tokens:
                # 如果之前锁定的车已经驶出范围消失，强制重新筛选
                need_switch = True

            if not need_switch:
                self.info['top_k_ids'] = [self.current_cbv_id]
                return
            # -----------------------------------

            if len(tokens) > 1 and predictions is not None and len(predictions) > 0:
                ego_future = self._pred_traj[0, :, :2] # [T, 2]
                min_distances = []

                for i in range(len(predictions)):
                    agent_future = predictions[i, :, :2]
                    min_t = min(len(ego_future), len(agent_future))
                    if min_t == 0:
                        min_distances.append(float('inf'))
                        continue

                    dist = np.linalg.norm(ego_future[:min_t] - agent_future[:min_t], axis=-1)

                    # 防止选择已经路过且远离的agent，空间交互应聚焦未来轨迹而非当前起点的最小距离
                    min_idx = np.argmin(dist)
                    if min_idx <= 5 and min_t > 10 and dist[-1] > dist[0]:
                        min_dist = np.mean(dist[min_t // 2:])
                    else:
                        min_dist = np.min(dist)

                    min_distances.append(min_dist)

                if min_distances:
                    closest_idx = np.argmin(min_distances)
                    if closest_idx + 1 < len(tokens): # ensure bounds, tokens[0] is ego
                        cbv_id = tokens[closest_idx + 1]
                        self.current_cbv_id = cbv_id
                        self.info['top_k_ids'] = [cbv_id]  # 只选拔1辆，完成简单走通
                        print(f"================================> [Policy] Successfully switched and locked new CBV: {cbv_id} at step {time_index}")

    def _run_adv_inference(self, time_index):
        """处理周车推理，加入 Pluto 的键值劫持逻辑狸猫换太子"""
        dynamic_adv_ids = self.info.get('top_k_ids', [])
        target_ids = dynamic_adv_ids if dynamic_adv_ids else getattr(self.base_cfg, 'adv_id', [])
        if not target_ids:
            print("[Policy] Warning: target_ids is empty, AdvSim will not run.")
            return

        all_cbv_trajs = []
        all_cbv_ids = []

        original_ego_track = self.current_state['tracks']['ego']
        scene_to_obj_id = {str(aid)[-5:]: aid for aid in self.current_state['tracks'].keys() if aid != 'ego'}

        for adv_id_str in target_ids:
            real_adv_id = scene_to_obj_id.get(str(adv_id_str)[-5:])
            if not real_adv_id or real_adv_id not in self.current_state['tracks']:
                continue

            cbv_track = self.current_state['tracks'][real_adv_id]

            try:
                # 劫持
                self.current_state['tracks']['real_ego'] = original_ego_track
                self.current_state['tracks']['ego'] = cbv_track
                del self.current_state['tracks'][real_adv_id]

                adv_infer_out = self.AdvSim.run_inference(self.current_state, time_index)
                adv_traj_all = adv_infer_out[0]

                if adv_traj_all.ndim == 3: # (16, T, 5)
                    all_cbv_trajs.append(adv_traj_all[0])
                else:
                    all_cbv_trajs.append(adv_traj_all[0, 0])
                all_cbv_ids.append(real_adv_id)
            finally:
                # 还原
                self.current_state['tracks'][real_adv_id] = cbv_track
                if 'real_ego' in self.current_state['tracks']:
                    del self.current_state['tracks']['real_ego']
                self.current_state['tracks']['ego'] = original_ego_track

        if len(all_cbv_trajs) > 0:
            self.current_adv_trajs = np.stack(all_cbv_trajs)
            self.adv_agent_ids = all_cbv_ids

    def _apply_adv_control(self, time_index):
        """将周车的预测轨迹应用到引擎中"""
        if self.current_adv_trajs is None or self.info is None:
            return

        # 1. 建立 VBD/Pluto 输出 ID 到 Index 的映射 (使用后5位作为 key 进行匹配)
        short_id_map = {aid[-5:]: i for i, aid in enumerate(self.adv_agent_ids)}

        # 2. 建立 ScenarioID 到 ObjID 的映射
        short_id_to_scene_id = {aid[-5:]: oid for oid, aid in self.engine.traffic_manager.obj_id_to_scenario_id.items()}

        # 算关键周车id
        dynamic_adv_ids = self.info.get('top_k_ids', [])
        target_ids = dynamic_adv_ids if dynamic_adv_ids else getattr(self.base_cfg, 'adv_id', [])

        if not hasattr(self, 'adv_expert_trajs_cache'):
            self.adv_expert_trajs_cache = {}

        target_scene_ids = [str(aid)[-5:] for aid in target_ids]
        # 抹除已经过去（不再是 CBV）的对抗车轨迹（排除主车）
        for oid, scene_id in self.engine.traffic_manager.obj_id_to_scenario_id.items():
            if scene_id != self.sdc_id and str(scene_id)[-5:] not in target_scene_ids:
                obj = self.engine.get_objects().get(oid)
                if obj is not None:
                    if hasattr(obj, 'plan_traj'):
                        delattr(obj, 'plan_traj')
                    if hasattr(obj, 'expert_traj'):
                        delattr(obj, 'expert_traj')

        for adv_id_str in target_ids:
            idx = short_id_map.get(adv_id_str[-5:], None)
            if idx is None:
                continue

            adv_traj = self.current_adv_trajs[idx]
            adv_obj_id = short_id_to_scene_id.get(adv_id_str[-5:], None)
            if adv_obj_id is None:
                continue

            adv_obj = self.engine.get_objects().get(adv_obj_id)
            if adv_obj is None:
                continue

            adv_obj.plan_traj = adv_traj[:, :2]  # 可视化更新

            real_adv_id = self.engine.traffic_manager.obj_id_to_scenario_id[adv_obj_id]
            adv_tracks = self.current_state['tracks'][real_adv_id]
            adv_pos = adv_tracks['state']['position']
            adv_heading = adv_tracks['state']['heading']
            adv_velocity = np.array(adv_tracks['state']['velocity'])

            # 使用实例变量对每辆车分别保存真值轨迹，避免多个不同车辆发生轨迹串台乱画
            if real_adv_id not in self.adv_expert_trajs_cache:
                adv_valid_mask = np.array(adv_tracks['state']['valid'], dtype=bool).reshape(-1)
                if adv_valid_mask.any():
                    self.adv_expert_trajs_cache[real_adv_id] = np.copy(adv_tracks['state']["position"][adv_valid_mask])
            adv_obj.expert_traj = self.adv_expert_trajs_cache.get(real_adv_id, None)

            # 处理轨迹长度
            pred_length = len(adv_traj)
            adv_length = len(adv_pos)
            has_heading = adv_traj.shape[1] >= 3
            has_vel = adv_traj.shape[1] >= 5

            if time_index + pred_length < adv_length:
                adv_pos[time_index:time_index + pred_length, :2] = adv_traj[:, :2]
                if has_heading:
                    target_slice = adv_heading[time_index:time_index + pred_length]
                    source_data = adv_traj[:, 2]
                    if target_slice.ndim == 2 and source_data.ndim == 1:
                        source_data = source_data.reshape(-1, 1)
                    adv_heading[time_index:time_index + pred_length] = source_data
                if has_vel:
                    adv_velocity[time_index:time_index + pred_length, :2] = adv_traj[:, -2:]
            else:
                remain_len = adv_length - time_index
                if remain_len > 0:
                    adv_pos[time_index:, :2] = adv_traj[:remain_len, :2]
                    if has_heading:
                        target_slice = adv_heading[time_index:]
                        source_data = adv_traj[:remain_len, 2]
                        if target_slice.ndim == 2 and source_data.ndim == 1:
                            source_data = source_data.reshape(-1, 1)
                        adv_heading[time_index:] = source_data
                    if has_vel:
                        adv_velocity[time_index:, :2] = adv_traj[:remain_len, -2:]


    def _compute_control(self):
        """计算本车 PID 控制动作"""
        pred_traj = self._pred_traj[0, :-1, :]

        ego_vehicle = self.control_object
        current_pos = ego_vehicle.position
        current_heading = ego_vehicle.heading_theta
        current_speed = ego_vehicle.speed

        # ===== RLPlanningPolicy-style tracking =====
        # 1) 用“预瞄点”匹配
        preview_distance = np.clip(current_speed * 0.5, 1.0, 5.0)
        preview_pos = self._preview_point(current_pos, current_heading, preview_distance)

        # 2) 找离预瞄点最近且在车头前方的 match 点
        distances = np.linalg.norm(pred_traj[:, :2] - preview_pos, axis=1)
        match_idx = self.get_match_idx(pred_traj, distances, current_pos, current_heading)
        p_match = pred_traj[match_idx, :2]

        # 3) 选取目标点 (lookahead)
        lookahead_distance = np.clip(current_speed * 0.5, 1.0, 5.0)
        target_idx = self._find_lookahead_idx(pred_traj[:, :2], match_idx, lookahead_distance)
        target_idx = self._ensure_forward_target(pred_traj[:, :2], match_idx, target_idx, current_pos, current_heading)
        p_target = pred_traj[target_idx, :2]

        # ========== 目标点合法性最终校验与外推 ==========
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)
        if np.dot(p_target - current_pos, heading_vec) <= 0:
            # 强行在正前方外推一个目标点，避免横向跟踪崩溃（急打方向盘）
            p_target = current_pos + heading_vec * lookahead_distance
            p_match = current_pos
        self.engine.agents['default_agent'].key_points = [p_target]

        # 4) 算轨迹朝向 (切向)
        K_TANGENT = 3
        idx2 = min(match_idx + K_TANGENT, len(pred_traj) - 1)
        p_dir = pred_traj[idx2, :2]
        vec_traj_dir = p_dir - p_match
        if np.linalg.norm(vec_traj_dir) < 1e-3:
            vec_traj_dir = heading_vec
        else:
            if np.dot(vec_traj_dir, heading_vec) < 0:
                vec_traj_dir = heading_vec

        # 5) 横向与航向误差
        lateral_error_m = self._signed_cross_track_error(p_target, p_match, vec_traj_dir)
        lateral_error = lateral_error_m / max(current_speed, 1.0)

        if pred_traj.shape[-1] >= 4:
            # 索引 2, 3 对应模型输出的 [cos(yaw), sin(yaw)]
            traj_heading = np.arctan2(pred_traj[idx2, 3], pred_traj[idx2, 2])
        else:
            traj_heading = np.arctan2(vec_traj_dir[1], vec_traj_dir[0])
            
        heading_error = traj_heading - current_heading
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
        combined_error = lateral_error + 2.5 * heading_error

        # 6) 纵向误差: 优先从模型预测中直接读取速度以避免震荡
        if pred_traj.shape[-1] >= 6:
            # 索引 4, 5 对应模型输出的 [vx, vy]
            pred_vx = pred_traj[min(match_idx + 1, len(pred_traj) - 1), 4]
            pred_vy = pred_traj[min(match_idx + 1, len(pred_traj) - 1), 5]
            target_speed = float(np.hypot(pred_vx, pred_vy))
        else:
            # 回退到根据距离的空间差分法计算目标速度
            step_diff = 5
            if match_idx + step_diff < len(pred_traj):
                target_speed_idx = match_idx + step_diff
                p_target_speed = pred_traj[target_speed_idx, :2]
                target_speed = float(np.linalg.norm(p_target_speed - p_match) / (step_diff * FIXED_DT))
            else:
                available_steps = len(pred_traj) - 1 - match_idx
                if available_steps > 0:
                    p_target_speed = pred_traj[-1, :2]
                    target_speed = float(np.linalg.norm(p_target_speed - p_match) / (available_steps * FIXED_DT))
                elif len(pred_traj) > 1:
                    target_speed = float(np.linalg.norm(pred_traj[-1, :2] - pred_traj[-2, :2]) / FIXED_DT)
                else:
                    target_speed = current_speed

        steering = self.lateral_pid.get_result(-combined_error)
        speed_error =  current_speed - target_speed
        throttle_brake = self.longitudinal_pid.get_result(speed_error)
        print('[steering, throttle_brake]',[steering, throttle_brake])
        return [steering, throttle_brake]

    def _preview_point(self, current_pos, current_heading, preview_distance: float) -> np.ndarray:
        """Preview point in world frame, aligned with RLPlanningPolicy.preview_point usage."""
        return np.asarray(current_pos, dtype=np.float32) + preview_distance * np.array(
            [np.cos(current_heading), np.sin(current_heading)], dtype=np.float32
        )

    def _find_lookahead_idx(self, traj_xy: np.ndarray, start_idx: int, lookahead_m: float) -> int:
        """Find an index forward along the trajectory by accumulating arc-length from start_idx."""
        n = len(traj_xy)
        if n == 0:
            return 0
        if n == 1:
            return 0
        start_idx = int(np.clip(start_idx, 0, n - 1))

        acc = 0.0
        i = start_idx
        while i < n - 1 and acc < lookahead_m:
            step = float(np.linalg.norm(traj_xy[i + 1] - traj_xy[i]))
            # protect against duplicate points
            if step < 1e-4:
                step = 0.0
            acc += step
            i += 1
        return i

    def _ensure_forward_target(
        self,
        traj_xy: np.ndarray,
        match_idx: int,
        target_idx: int,
        current_pos: np.ndarray,
        current_heading: float,
        max_extra_steps: int = 20,
    ) -> int:
        """Ensure the chosen target point lies in front of the ego (dot>0); otherwise step forward."""
        n = len(traj_xy)
        if n == 0:
            return 0
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)], dtype=np.float32)

        idx = int(np.clip(target_idx, match_idx, n - 1))
        for _ in range(max_extra_steps):
            v = traj_xy[idx] - current_pos
            if float(np.dot(v, heading_vec)) > 0.5:  # require a bit of forward margin
                return idx
            if idx >= n - 1:
                break
            idx += 1
        return idx

    def _signed_cross_track_error(self, p: np.ndarray, p_on_path: np.ndarray, path_tangent: np.ndarray) -> float:
        """Signed lateral distance from point p to the line passing p_on_path with direction path_tangent."""
        t = np.asarray(path_tangent, dtype=np.float32)
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            return 0.0
        t = t / tn
        # left normal of tangent
        n = np.array([-t[1], t[0]], dtype=np.float32)
        return float(np.dot(np.asarray(p, dtype=np.float32) - np.asarray(p_on_path, dtype=np.float32), n))

    def update_state(self, time_index):
        """Return a patched scenario state for inference without mutating the engine's scenario.

        We only patch the ego (sdc) track state arrays over the last `replan_frequency` steps
        with the closed-loop executed states, so that the model sees a consistent history
        while the engine's original expert data remains untouched.
        """
        # Base is the current engine scenario (read-only)
        base_state = self.engine.data_manager.current_scenario
        length = int(base_state.get('length', 0))

        # During warmup we should not patch; also if nothing to patch, return base_state.
        if time_index <= 0 or length <= 0 or len(self.hist_buffer) == 0:
            return base_state

        # Create a lightweight copy:
        # - Deepcopy metadata/map/etc can be very large; we reuse base_state by reference
        # - But we MUST copy the ego arrays to avoid polluting engine memory.
        patched_state = dict(base_state)
        patched_tracks = dict(base_state.get('tracks', {}))
        patched_state['tracks'] = patched_tracks

        base_track = base_state['tracks'][self.sdc_id]
        patched_track = dict(base_track)
        patched_tracks[self.sdc_id] = patched_track

        base_track_state = base_track['state']
        patched_track_state = dict(base_track_state)
        patched_track['state'] = patched_track_state

        # Copy arrays (owning memory) so in-place assignment won't affect engine.
        patched_traj = np.array(base_track_state['position'], copy=True)
        patched_head = np.array(base_track_state['heading'], copy=True)
        patched_vel = np.array(base_track_state['velocity'], copy=True)

        k = len(self.hist_buffer)
        # end_idx = time_index + 1
        end_idx = time_index
        start_idx = end_idx - k
        steps_tail = range(start_idx, end_idx)

        for state, t in zip(self.hist_buffer, steps_tail):
            if 0 <= t < length:
                patched_traj[t, :2] = state['position']
                patched_head[t] = state['heading']
                patched_vel[t] = state['velocity']

        patched_track_state['position'] = patched_traj
        patched_track_state['heading'] = patched_head
        patched_track_state['velocity'] = patched_vel

        return patched_state

    def get_hist_buffer(self, state):
        # 缓存历史
        [current_pos, current_heading, current_speed] = state
        # Convert scalar speed to a 2D velocity vector based on the vehicle heading
        velocity_2d = np.array([current_speed * np.cos(current_heading), current_speed * np.sin(current_heading)])
        state = {
            "position": current_pos,
            "heading": current_heading,
            "velocity": velocity_2d,
            "valid": True
        }
        self.hist_buffer.append(state)

    def get_match_idx(self, pred_traj, distances, current_pos, current_heading):
        # 当前车头方向向量
        heading_vec = np.array([np.cos(current_heading), np.sin(current_heading)])
        vec_to_points = pred_traj[:, :2] - current_pos

        # 只保留车辆前方点（夹角 < 90°）
        dot = np.dot(vec_to_points, heading_vec)
        forward_mask = dot > 0

        # 再在前方点中找最近的点
        if forward_mask.any():
            match_idx = np.argmin(distances * forward_mask + (~forward_mask) * 1e6)
        else:
            match_idx = np.argmin(distances)  # 极端情况备用

        # 平滑更新
        if match_idx < self.last_match_idx:
            match_idx = self.last_match_idx
        self.last_match_idx = match_idx
        return match_idx

    def save_history(self):
        self.history_logger.save()