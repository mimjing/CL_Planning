from unitraj.models.vbd.sim_agent.guidance_metrics import TrackingReward, GoalReward, OverlapReward, OnroadReward, ProgressReward
import torch
from torch import nn

class SafetyBalancedReward(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化基础奖励模块
        self.tracking = TrackingReward()
        self.overlap = OverlapReward()
        self.onroad = OnroadReward()
        self.goal = GoalReward()
        self.progress = ProgressReward()

        # 自适应权重参数
        self.safety_weight = nn.Parameter(torch.tensor(1.0))
        self.goal_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, traj_pred, c, run_overlap=True, run_onroad=False, run_progress=True,
                                    run_track=False, run_goal=True,  **kwargs):
        # 计算基础奖励
        rewards = torch.tensor(0.0, device=traj_pred.device)

        overlap_penalty = self.overlap(traj_pred, c, **kwargs)
        # 计算安全系数（碰撞距离的sigmoid函数）
        min_distance = overlap_penalty.detach().min(dim=-1)[0]
        safety_factor = torch.sigmoid((min_distance - 1.0) * 5)  # [0,1]安全系数

        if run_overlap:
            dynamic_safety_weight = self.safety_weight * (1.5 - 0.5 * safety_factor)
            total_overlap = (overlap_penalty * dynamic_safety_weight.unsqueeze(-1)).sum()
            rewards += total_overlap
            print('overlap', total_overlap.item())
        if run_onroad:
            onroad_penalty = self.onroad(traj_pred, c, **kwargs)
            dynamic_onroad_weight = self.safety_weight * (1.5 - 0.5 * safety_factor)
            total_onroad = (onroad_penalty * dynamic_onroad_weight).sum()
            rewards += total_onroad
            print('onroad', total_onroad.item())
        if run_goal:
            goal_reward = self.goal(traj_pred, **kwargs)
            total_goal = goal_reward.sum()
            rewards += total_goal
            print('goal', total_goal.item())
        if run_track:
            track_reward = self.tracking(traj_pred, **kwargs)
            total_track = track_reward.sum()
            rewards += total_track
            print('track', total_track.item())
        if run_progress:
            speed_reward = self.progress(traj_pred, **kwargs)
            total_speed = speed_reward.sum()
            rewards += total_speed
            print('speed', total_speed.item())

        return rewards


class EnhancedOverlapReward(OverlapReward):
    def __init__(self, safe_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.safe_threshold = safe_threshold

    def forward(self, traj_pred, c, **kwargs):
        orig_reward = super().forward(traj_pred, c, **kwargs)

        # 获取原始距离信息
        signed_distance = self._compute_distance(traj_pred, c)  # [B,A,T,A]
        min_dist = signed_distance.min(dim=-1)[0]  # [B,A,T]

        # 增加渐进式惩罚
        danger_mask = (min_dist < self.safe_threshold).float()
        adaptive_penalty = torch.exp(-min_dist / self.safe_threshold) * danger_mask
        return orig_reward - adaptive_penalty * 2.0


class ProgressiveGoalReward(GoalReward):
    def __init__(self, max_weight=2.0, **kwargs):
        super().__init__(**kwargs)
        self.max_weight = max_weight

    def forward(self, traj_pred, goal, progress_ratio=0.0, **kwargs):
        base_reward = super().forward(traj_pred, goal, **kwargs)
        current_weight = self.max_weight * progress_ratio
        return base_reward * current_weight