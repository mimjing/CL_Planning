import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, l1_loss, smooth_l1_loss


class TrackingReward(nn.Module):
    def __init__(self, loss_fn=smooth_l1_loss):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, traj_pred: torch.Tensor, traj_ref: torch.Tensor, weight_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass of the metrics_before module.

        Args:
            traj_pred (torch.Tensor): The input tensor.
            traj_ref (torch.Tensor): The traj_reference tensor.
            weight_mask (torch.Tensor): The weight_mask tensor.

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight_mask shape is not compatible with traj_ref.

        """
        if weight_mask is None:
            weight_mask = torch.ones_like(traj_ref)
            
        assert traj_pred.shape[:-1] == traj_ref.shape[:-1], f"traj_pred {traj_pred.shape} and traj_ref {traj_ref.shape} must have the same shape"
        d = traj_ref.shape[-1]

        if len(weight_mask.shape) == (len(traj_ref.shape)-1):
            weight_mask = weight_mask.unsqueeze(-1)
        elif len(weight_mask.shape) == len(traj_ref.shape):
            assert weight_mask.shape[-1] == traj_ref.shape[-1], "weight_mask shape must be either (batch, seq) or same as traj_ref"
        else:
            raise ValueError("weight_mask shape must be either (B, A, T) or same as traj_ref")
        
        rewards = -self.loss_fn(input=traj_pred[..., :d], 
                                target=traj_ref[..., :d], reduction="none") * weight_mask
        
        return rewards
    

class GoalReward(nn.Module):
    def __init__(self, loss_fn=smooth_l1_loss):
        self.loss_fn = loss_fn

        super().__init__()
    
    def forward(self, traj_pred: torch.Tensor, goal: torch.Tensor, goal_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass of the metrics_before module.

        Args:
            traj_pred (torch.Tensor): The input tensor.
            traj_ref (torch.Tensor): The traj_reference tensor.
            weight_mask (torch.Tensor): The weight_mask tensor.

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight_mask shape is not compatible with traj_ref.

        """
        if goal_mask is None:
            goal_mask = torch.ones_like(goal)
            
        d = goal.shape[-1]
        look_ahead = kwargs.get("look_ahead", -1)
        rewards = -self.loss_fn(input=traj_pred[..., look_ahead, :d], target=goal, reduction="none") * goal_mask
        
        return rewards
    

class AnchorReward(nn.Module): # Does not work well
    def __init__(self, loss_fn=smooth_l1_loss):
        self.loss_fn = loss_fn
        super().__init__()
    
    def forward(self, traj_pred: torch.Tensor, traj_ref: torch.Tensor, weight_mask: torch.Tensor = None, **kwargs):
        """
        Forward pass of the metrics_before module.

        Args:
            traj_pred (torch.Tensor): The input tensor. [B, A, T, D]
            traj_ref (torch.Tensor): The traj_reference tensor. [B, A, D]
            weight_mask (torch.Tensor): The weight_mask tensor. [B, A] or [B, A, D]

        Returns:
            torch.Tensor: The computed loss tensor.

        Raises:
            AssertionError: If traj_pred and traj_ref do not have the same shape.
            ValueError: If the weight_mask shape is not compatible with traj_ref.

        """
        if weight_mask is None:
            weight_mask = torch.ones_like(traj_ref)
            
        d = traj_ref.shape[-1]
        if len(weight_mask.shape) == (len(traj_ref.shape)-1):
            weight_mask = weight_mask.unsqueeze(-1)
        elif len(weight_mask.shape) == len(traj_ref.shape):
            assert weight_mask.shape[-1] == traj_ref.shape[-1], "weight_mask shape must be either (batch, seq) or same as traj_ref"
        else:
            raise ValueError("weight_mask shape must be either (B, A, T) or same as traj_ref")
         
        traj_ref = traj_ref.unsqueeze(-2).repeat(1, 1, traj_pred.shape[-2], 1)
        weight_mask = weight_mask.unsqueeze(-2).repeat(1, 1, traj_pred.shape[-2], 1)
        
        rewards = -self.loss_fn(
            input=traj_pred[..., :d],
            target=traj_ref[..., :d],
            reduction="none") * weight_mask
        
        rewards, _ = torch.min(torch.sum(rewards, dim=-1), dim=-1)

        return rewards