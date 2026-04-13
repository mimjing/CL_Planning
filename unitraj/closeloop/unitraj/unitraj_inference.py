import time
import os
import torch
from unitraj.models import build_model
os.environ["MPLBACKEND"] = "Agg"   # 离线保存最稳
from unitraj.datasets.unitraj_test_dataset import UnitrajTestDataset
from unitraj.closeloop.unitraj.agent_utils import pred_local_to_world

def to_device(batch, device="cuda"):
    """
    递归地将数据（tensors, dicts, lists）移动到指定的设备（如GPU）。
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    else:
        return batch


class UnitrajInference:
    def __init__(self, cfg):
        self.center_objects = None
        self.cfg = cfg
        self.dataset = UnitrajTestDataset(self.cfg)
        self.imitation_algo = None
        self.device = "cuda"

    def initialize_model(self):
        """Initialize the imitation model and environment."""
        self.imitation_algo = build_model(self.cfg)
        ckpt = torch.load(self.cfg.ckpt_path, map_location=self.device, weights_only=False)
        self.imitation_algo.load_state_dict(ckpt["state_dict"])
        self.imitation_algo = self.imitation_algo.to(self.device)


    def run_inference(self, scenario, current_step):
        """Run inference and return the last batch and prediction."""

        t1 = time.time()
        batch_dict, center_objects = self.dataset.process_scenario(scenario, current_step)
        t2 = time.time()
        batch_dict = to_device(batch_dict, self.device)
        t3 = time.time()

        self.imitation_algo.eval()
        prediction = self.imitation_algo.forward(batch_dict)
        t4 = time.time()

        # print(f"时间1 (数据处理): {t2 - t1:.4f}s")
        # print(f"时间2 (智能体准备): {t3 - t2:.4f}s")
        # print(f"时间3 (推理): {t4 - t3:.4f}s")
        # print(f"总时间: {t4 - t1:.4f}s")

        pred_trajs_local = prediction['predicted_trajectory'].detach().cpu().numpy()[0]
        pred_trajs_world = pred_local_to_world(pred_trajs_local, center_objects)

        return pred_trajs_world[:,:,:2]
