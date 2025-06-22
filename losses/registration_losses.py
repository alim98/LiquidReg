"""
Loss functions for LiquidReg registration training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class LocalNormalizedCrossCorrelation(nn.Module):
    """
    Local Normalized Cross-Correlation (LNCC) loss.
    """
    
    def __init__(self, window_size: int = 9, eps: float = 1e-8):
        super().__init__()
        self.window_size = window_size
        self.eps = eps
        
        # Create 3D averaging kernel
        self.register_buffer(
            "kernel",
            torch.ones(1, 1, window_size, window_size, window_size) / (window_size ** 3)
        )
    
    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute LNCC between fixed and warped images.
        
        Args:
            fixed: Fixed image (B, 1, D, H, W)
            warped: Warped moving image (B, 1, D, H, W)
            
        Returns:
            lncc: LNCC loss (scalar)
        """

        
        # Pad inputs
        pad = self.window_size // 2
        fixed = F.pad(fixed, [pad] * 6, mode='reflect')
        warped = F.pad(warped, [pad] * 6, mode='reflect')
        
        # Compute local means
        mu_fixed = F.conv3d(fixed, self.kernel, padding=0)
        mu_warped = F.conv3d(warped, self.kernel, padding=0)
        

        
        # Compute local variances and covariance
        mu_fixed_sq = mu_fixed ** 2
        mu_warped_sq = mu_warped ** 2
        mu_fixed_warped = mu_fixed * mu_warped
        
        sigma_fixed_sq = F.conv3d(fixed ** 2, self.kernel, padding=0) - mu_fixed_sq
        sigma_warped_sq = F.conv3d(warped ** 2, self.kernel, padding=0) - mu_warped_sq
        sigma_fixed_warped = F.conv3d(fixed * warped, self.kernel, padding=0) - mu_fixed_warped
        
        # Check for negative variances and fix them
        sigma_fixed_sq = torch.clamp(sigma_fixed_sq, min=self.eps)
        sigma_warped_sq = torch.clamp(sigma_warped_sq, min=self.eps)
        
        # Compute LNCC with better numerical stability
        numerator = sigma_fixed_warped
        
        # More aggressive numerical stability measures
        denominator_sq = sigma_fixed_sq * sigma_warped_sq
        denominator_sq = torch.clamp(denominator_sq, min=self.eps*self.eps)
        denominator = torch.sqrt(denominator_sq)
        denominator = torch.clamp(denominator, min=self.eps)
        
        lncc = numerator / denominator
        
        # Clamp LNCC to reasonable range to prevent extreme values
        lncc = torch.clamp(lncc, min=-10.0, max=10.0)
        
        # Handle NaN/Inf values
        lncc = torch.nan_to_num(lncc, nan=0.0, posinf=1.0, neginf=-1.0)
        

        
        # Return negative mean LNCC (to minimize)
        loss = -lncc.mean()
        

        
        return loss


class MutualInformation(nn.Module):
    """
    Mutual Information loss for multi-modal registration.
    """
    
    def __init__(self, num_bins: int = 64, sigma: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        
    def forward(self, fixed: torch.Tensor, warped: torch.Tensor) -> torch.Tensor:
        """
        Compute mutual information loss.
        
        Args:
            fixed: Fixed image (B, 1, D, H, W)
            warped: Warped moving image (B, 1, D, H, W)
            
        Returns:
            mi_loss: Negative mutual information
        """
        B = fixed.shape[0]
        
        # Flatten images
        fixed_flat = fixed.view(B, -1)
        warped_flat = warped.view(B, -1)
        
        # Normalize to [0, 1]
        fixed_norm = (fixed_flat - fixed_flat.min(dim=1, keepdim=True)[0]) / \
                     (fixed_flat.max(dim=1, keepdim=True)[0] - fixed_flat.min(dim=1, keepdim=True)[0] + 1e-8)
        warped_norm = (warped_flat - warped_flat.min(dim=1, keepdim=True)[0]) / \
                      (warped_flat.max(dim=1, keepdim=True)[0] - warped_flat.min(dim=1, keepdim=True)[0] + 1e-8)
        
        # Create bin centers
        bin_centers = torch.linspace(0, 1, self.num_bins, device=fixed.device)
        
        # Compute histograms using Gaussian kernels
        def soft_histogram(x, bin_centers, sigma):
            x_expanded = x.unsqueeze(-1)  # (B, N, 1)
            bin_centers_expanded = bin_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_bins)
            weights = torch.exp(-0.5 * ((x_expanded - bin_centers_expanded) / sigma) ** 2)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            hist = weights.sum(dim=1)  # (B, num_bins)
            return hist / hist.sum(dim=-1, keepdim=True)
        
        # Compute marginal histograms
        p_fixed = soft_histogram(fixed_norm, bin_centers, self.sigma)
        p_warped = soft_histogram(warped_norm, bin_centers, self.sigma)
        
        # Compute joint histogram
        joint_weights = torch.zeros(B, self.num_bins, self.num_bins, device=fixed.device)
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                weight_fixed = torch.exp(-0.5 * ((fixed_norm - bin_centers[i]) / self.sigma) ** 2)
                weight_warped = torch.exp(-0.5 * ((warped_norm - bin_centers[j]) / self.sigma) ** 2)
                joint_weights[:, i, j] = (weight_fixed * weight_warped).sum(dim=1)
        
        p_joint = joint_weights / joint_weights.sum(dim=[1, 2], keepdim=True)
        
        # Compute mutual information
        p_fixed_expanded = p_fixed.unsqueeze(-1)
        p_warped_expanded = p_warped.unsqueeze(1)
        p_product = p_fixed_expanded * p_warped_expanded
        
        # Avoid log(0)
        mask = (p_joint > 1e-8) & (p_product > 1e-8)
        mi = torch.where(mask, p_joint * torch.log(p_joint / p_product), torch.zeros_like(p_joint))
        mi = mi.sum(dim=[1, 2]).mean()
        
        return -mi  # Return negative MI for minimization


class JacobianDeterminantLoss(nn.Module):
    """
    Jacobian determinant penalty to prevent folding.
    """
    
    def __init__(self, penalty_type: str = "l2"):  # "l2" or "log"
        super().__init__()
        self.penalty_type = penalty_type
    
    def forward(self, jacobian_det: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian determinant penalty.
        
        Args:
            jacobian_det: Jacobian determinant (B, 1, D-2, H-2, W-2)
            
        Returns:
            penalty: Jacobian penalty
        """

        
        if self.penalty_type == "l2":
            # Penalize deviations from 1
            penalty = ((jacobian_det - 1) ** 2).mean()
        elif self.penalty_type == "log":
            # Log-penalty (more stable)
            # Clamp to avoid negative values
            jacobian_det_clamped = jacobian_det.clamp(min=1e-6)
            penalty = (torch.log(jacobian_det_clamped) ** 2).mean()
        else:
            raise ValueError(f"Unknown penalty type: {self.penalty_type}")
        

        
        return penalty


class VelocityRegularization(nn.Module):
    """
    L2 regularization on velocity field.
    """
    
    def __init__(self, norm_type: str = "l2"):
        super().__init__()
        self.norm_type = norm_type
    
    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity regularization.
        
        Args:
            velocity: Velocity field (B, 3, D, H, W)
            
        Returns:
            reg: Regularization term
        """
        if self.norm_type == "l2":
            return (velocity ** 2).mean()
        elif self.norm_type == "l1":
            return velocity.abs().mean()
        else:
            raise ValueError(f"Unknown norm type: {self.norm_type}")


class LiquidStabilityLoss(nn.Module):
    """
    Liquid stability regularization to prevent exploding time constants.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, liquid_params: torch.Tensor) -> torch.Tensor:
        """
        Compute liquid stability regularization.
        
        Args:
            liquid_params: Liquid parameters from hyper-network (B, P)
            
        Returns:
            stability_loss: Stability regularization
        """
        # L2 norm of parameter changes
        param_norm = (liquid_params ** 2).mean()
        
        # Additional penalty on extreme values
        extreme_penalty = F.relu(liquid_params.abs() - 1.0).mean()
        
        return param_norm + extreme_penalty


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation-based evaluation.
    """
    
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(
        self, 
        pred_seg: torch.Tensor, 
        target_seg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred_seg: Predicted segmentation (B, C, D, H, W)
            target_seg: Target segmentation (B, C, D, H, W)
            
        Returns:
            dice_loss: Dice loss
        """
        # Flatten
        pred_flat = pred_seg.view(pred_seg.shape[0], pred_seg.shape[1], -1)
        target_flat = target_seg.view(target_seg.shape[0], target_seg.shape[1], -1)
        
        # Compute intersection and union
        intersection = (pred_flat * target_flat).sum(dim=-1)
        pred_sum = pred_flat.sum(dim=-1)
        target_sum = target_flat.sum(dim=-1)
        
        # Dice coefficient
        dice = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        # Return 1 - dice for loss
        return 1 - dice.mean()


class CompositeLoss(nn.Module):
    """
    Composite loss function combining multiple terms.
    """
    
    def __init__(
        self,
        similarity_loss: str = "lncc",  # "lncc", "mse", "mi"
        lambda_similarity: float = 1.0,
        lambda_jacobian: float = 1.0,
        lambda_velocity: float = 0.01,
        lambda_liquid: float = 0.001,
        lncc_window: int = 9,
        jacobian_penalty: str = "l2",
    ):
        super().__init__()
        
        self.lambda_similarity = lambda_similarity
        self.lambda_jacobian = lambda_jacobian
        self.lambda_velocity = lambda_velocity
        self.lambda_liquid = lambda_liquid
        
        # Similarity loss
        if similarity_loss == "lncc":
            self.similarity_loss = LocalNormalizedCrossCorrelation(window_size=lncc_window)
        elif similarity_loss == "mse":
            self.similarity_loss = nn.MSELoss()
        elif similarity_loss == "mi":
            self.similarity_loss = MutualInformation()
        else:
            raise ValueError(f"Unknown similarity loss: {similarity_loss}")
        
        # Regularization losses
        self.jacobian_loss = JacobianDeterminantLoss(penalty_type=jacobian_penalty)
        self.velocity_reg = VelocityRegularization()
        self.liquid_stability = LiquidStabilityLoss()
    
    def forward(
        self,
        fixed: torch.Tensor,
        warped: torch.Tensor,
        velocity_field: torch.Tensor,
        jacobian_det: torch.Tensor,
        liquid_params: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.
        
        Args:
            fixed: Fixed image (B, 1, D, H, W)
            warped: Warped moving image (B, 1, D, H, W)
            velocity_field: Velocity field (B, 3, D, H, W)
            jacobian_det: Jacobian determinant (B, 1, D-2, H-2, W-2)
            liquid_params: Liquid parameters (B, P)
            
        Returns:
            losses: Dictionary of loss components and total loss
        """

        
        losses = {}
        
        # Similarity loss
        try:
            similarity_loss = self.similarity_loss(fixed, warped)
            losses['similarity'] = self.lambda_similarity * similarity_loss
        except Exception as e:
            print(f"[ERROR] Exception in similarity loss: {e}")
            losses['similarity'] = torch.tensor(0.0, device=fixed.device)
        
        # Jacobian determinant penalty
        try:
            jacobian_loss = self.jacobian_loss(jacobian_det)
            losses['jacobian'] = self.lambda_jacobian * jacobian_loss
        except Exception as e:
            print(f"[ERROR] Exception in jacobian loss: {e}")
            losses['jacobian'] = torch.tensor(0.0, device=fixed.device)
        
        # Velocity regularization
        try:
            velocity_reg = self.velocity_reg(velocity_field)
            losses['velocity_reg'] = self.lambda_velocity * velocity_reg
        except Exception as e:
            print(f"[ERROR] Exception in velocity regularization: {e}")
            losses['velocity_reg'] = torch.tensor(0.0, device=fixed.device)
        
        # Liquid stability
        try:
            liquid_stability = self.liquid_stability(liquid_params)
            losses['liquid_stability'] = self.lambda_liquid * liquid_stability
        except Exception as e:
            print(f"[ERROR] Exception in liquid stability: {e}")
            losses['liquid_stability'] = torch.tensor(0.0, device=fixed.device)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        

        
        return losses 