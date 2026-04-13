'''Metrics to calculate the signed distance of road map.'''

import numpy as np
import torch
from torch import nn
from torch.autograd import Function


class OnroadReward(nn.Module):
    def __init__(self, weight=1):
        self.weight = weight
        super().__init__()
    
    def forward(
        self, 
        traj_pred: torch.Tensor,
        c: dict,
        roadgraph_points,
        weight: torch.Tensor = 1,
        aoi: list = None,
        **kwargs
    ):
        """
        traj_pred: [B, A, T, D]
        c: dict
        weight: [B, A, T]
        """
        T = traj_pred.shape[-2]
        traj_pred_xy = traj_pred[..., :2]
        traj_pred_yaw = traj_pred[..., 2:3]
        length = c['agents'][..., :16, -1, 5:6].repeat(1, 1, T).unsqueeze(-1)
        width = c['agents'][..., :16, -1, 6:7].repeat(1, 1, T).unsqueeze(-1)
        
        traj_5dof = torch.concatenate([traj_pred_xy, length, width, traj_pred_yaw], dim=-1)
        
        mask = (~c['agents_mask'][:,:16]).unsqueeze(-1).repeat(1, 1, T) # [B, A, T]
        
        if aoi is not None:
            traj_5dof = traj_5dof[:, aoi]
            mask = mask[:, aoi]
        
        # negative means on road
        # signed_distance = distance_offroad(traj_5dof, roadgraph_points) # [B, A, T]
        signed_distance = distance_offroad_polyline(traj_5dof, roadgraph_points) # [B, A, T]


        # Step1: 筛选“起初在路上”的 agent
        onroad_start = (signed_distance[:, :, 0] < self.threshold).unsqueeze(-1)  # [B, A, 1]
        # Step2: 检查是否“离边界太远”→ 认为是出界
        offroad_now = (signed_distance > self.threshold).float()  # [B, A, T]
        cost = offroad_now * onroad_start * mask * weight

        # # filter out already offroad
        # signed_distance = signed_distance * (signed_distance[:, :, 0:1] < 0)
        # # compute cost
        # cost = torch.functional.F.relu(signed_distance)
        # cost = cost * mask * weight
        print('onroad', -cost.sum().item())
        return -cost


# 在onroad_metric.py中修改distance_offroad函数
def distance_offroad_polyline(
    pose_5dof: torch.Tensor,          # [B, A, T, 5]
    polylines: torch.Tensor,         # [B, N, L, 5]
    road_edge_types=(15, 16)
) -> torch.Tensor:
    """
    更高效地计算预测轨迹是否越出边界（支持批量和向量化）
    返回 [B, A, T] 的最小 signed distance（负表示在路上）
    """
    B, A, T, _ = pose_5dof.shape
    device = pose_5dof.device

    # 生成角点 [B, A, T, 4, 2]
    bbox_corners = corners_from_bboxes(pose_5dof)  # 用户已有函数
    flat_corners = bbox_corners.view(B, A*T, 4, 2)  # [B, A*T, 4, 2]

    # 选择 road_edge 类型 polyline
    polyline_types = polylines[:, :, 0, 4].long()  # [B, N]
    edge_types = torch.tensor(road_edge_types, device=device)
    boundary_mask = (polyline_types[..., None] == edge_types).any(-1)  # [B, N]

    min_dists = []

    for b in range(B):
        corners_b = flat_corners[b]  # [A*T, 4, 2]
        poly_b = polylines[b][boundary_mask[b]]  # [n_i, L, 5]

        if poly_b.shape[0] == 0:
            min_dists.append(torch.full((A*T,), 10.0, device=device))  # 默认惩罚
            continue

        # polyline segments：起点 [M, 2], 终点 [M, 2]
        start_pts = poly_b[:, :-1, :2].reshape(-1, 2)  # [M, 2]
        end_pts = poly_b[:, 1:, :2].reshape(-1, 2)     # [M, 2]

        # [A*T*4, 2]
        corners_bt = corners_b.reshape(-1, 2)

        # [A*T*4, M] = 每个角点到所有边界线段的距离
        dists = point_to_segment_distance_batch(corners_bt, start_pts, end_pts)

        # [A*T*4] → [B, A*T, 4] → 每组角点的最小距离
        dists_min = dists.min(dim=1).values.view(A*T, 4).min(dim=1).values  # [A*T]
        min_dists.append(dists_min)

    min_dists = torch.stack(min_dists)  # [B, A*T]
    return min_dists.view(B, A, T)      # [B, A, T]


def point_to_segment_distance_batch(
    points: torch.Tensor,     # [N, 2]
    seg_start: torch.Tensor,  # [M, 2]
    seg_end: torch.Tensor     # [M, 2]
) -> torch.Tensor:
    """
    批量计算 N 个点到 M 条线段的距离，输出 [N, M]
    """
    # [N, M, 2]
    p = points.unsqueeze(1)             # [N, 1, 2]
    a = seg_start.unsqueeze(0)          # [1, M, 2]
    b = seg_end.unsqueeze(0)            # [1, M, 2]

    ab = b - a                          # [1, M, 2]
    ap = p - a                          # [N, M, 2]
    ab_len_sq = (ab ** 2).sum(-1).clamp(min=1e-6)  # [1, M]

    proj = (ap * ab).sum(-1) / ab_len_sq  # [N, M]
    proj = proj.clamp(0, 1).unsqueeze(-1)

    closest = a + proj * ab  # [N, M, 2]
    dist = ((p - closest) ** 2).sum(-1).sqrt()  # [N, M]
    return dist  # [N, M]

def corners_from_bboxes(bbox: torch.Tensor) -> torch.Tensor:
    """
    Computes corners for one 5 dof bbox.
    Args:
        bbox: [..., 5]
    Returns:
        points: [..., 4, 2]  
    """
    # bbox: [..., 5]
    c, s = torch.cos(bbox[..., 4]), torch.sin(bbox[..., 4])
    lc = bbox[..., 2] / 2 * c
    ls = bbox[..., 2] / 2 * s
    wc = bbox[..., 3] / 2 * c
    ws = bbox[..., 3] / 2 * s
    
    dx = torch.stack([lc + ws, lc - ws, -lc - ws, -lc + ws], dim=-1)
    dy = torch.stack([ls - wc, ls + wc, -ls + wc, -ls - wc], dim=-1)
    # [..., 2]
    points = torch.stack([dx, dy], dim=-1)
    
    points += bbox[..., None, :2]
    
    return points


def cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]