import time
import torch

from unitraj.datasets.VBD_dataset.VBD_test_dataset import VBDTestDataset
from unitraj.models.vbd.sim_agent.sim_actor import VBDTest

## Parameters
CURRENT_TIME_INDEX = 10
N_SIM_AGENTS = 16  # 32
N_SIMULATION_STEPS = 80

dataset = VBDTestDataset(
    data_dir= None,
    anchor_path='/home/mj/VBD/vbd/data/new_cluster_64_center_dict.pkl',
    max_object=N_SIM_AGENTS,
)

class VBDInference:
    def __init__(self, cfg):
        self.model_path = None
        self.device = "cuda"
        self.cfg = cfg

    def initialize_model(self):
        """Initialize the imitation model."""
        import lightning.pytorch.core.saving as saving
        # 1. 先保留原函数
        _original_load = saving.torch.load

        # 2. 新建包装函数：强制 weights_only=False
        def _torch_load_no_weights_only(*args, **kwargs):
            # 如果调用者又传了 weights_only，直接覆盖掉
            kwargs['weights_only'] = False
            return _original_load(*args, **kwargs)

        # 3. 替换
        saving.torch.load = _torch_load_no_weights_only
        self.model = VBDTest.load_from_checkpoint(self.cfg['ckpt_path'], cfg=self.cfg, map_location = self.device)
        self.model.reset_agent_length(N_SIM_AGENTS)
        self.model.eval()

    def run_inference(self, current_state, timestep, test_mode='diffusion'):
        t1 = time.time()
        with torch.no_grad():
            sample = dataset.process_scenario(current_state, timestep-1, use_log=True)
            batch = dataset.__collate_fn__([sample])
            t2 = time.time()

            if test_mode == 'diffusion':
                pred = self.model.sample_denoiser(batch)
                pred_traj = pred['denoised_trajs'].cpu().numpy()[0]  # [16,80,5]

            elif test_mode == 'prior':
                pred = self.model.inference_predictor(batch)
                scores = pred['goal_scores'][0].softmax(dim=-1)
                trajs = pred['goal_trajs'][0]
                sampled_idx = torch.multinomial(scores, 1).squeeze()  # 分数转概率
                pred_traj = trajs[torch.arange(sampled_idx.shape[0]), sampled_idx].cpu().numpy()
            else:
                raise NotImplementedError
            t3 = time.time()

        # print(f"时间1 (数据处理): {t2 - t1:.4f}s")
        # print(f"时间2 (推理): {t3 - t2:.4f}s")
        # print(f"总时间: {t3 - t1:.4f}s")

        return pred_traj