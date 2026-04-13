import torch
from torch import nn

class ProgressReward(nn.Module):
    def __init__(self, min_speed=1.0, weight=0.1):
        """
        Args:
            min_speed (float): 最低速度阈值（米/秒），低于此值会产生惩罚
            weight (float): 奖励系数
        """
        super().__init__()
        self.min_speed = min_speed
        self.weight = weight

    def forward(self, traj_pred: torch.Tensor, weight_mask: torch.Tensor, aoi: list, **kwargs):
        """
        Args:
            traj_pred: 预测轨迹 [B, A, T, D]，必须包含XY坐标（前两维）
            dt: 时间步长（秒），用于计算速度
            **kwargs: 兼容其他参数

        Returns:
            速度奖励 [B, A, T]
        """
        weight_mask = torch.ones_like(weight_mask)
        if aoi is not None:
            traj_pred = traj_pred[:, aoi]
            weight_mask = weight_mask[:, aoi]
        velocities = traj_pred[..., 3:]  # [B, A, T, 2]

        speed = torch.norm(velocities, dim=-1)  # [B, A, T]

        # 基础奖励：速度越大奖励越高
        # reward = speed * weight_mask * 0.1

        # 速度不足惩罚：当速度低于阈值时施加线性惩罚
        penalty = torch.clamp(self.min_speed - speed, min=0) * weight_mask
        return -penalty*10.0
        # print('velocity', (reward - penalty).sum().item())
        # return (reward - penalty) * self.weight
