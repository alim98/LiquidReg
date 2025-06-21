"""
Scaling and Squaring implementation for diffeomorphic transformations.
Based on Arsigny et al. "A Log-Euclidean Framework for Statistics on Diffeomorphisms" MICCAI 2006.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ScalingSquaring(nn.Module):
    """
    Scaling and Squaring layer for computing the exponential map of velocity fields.
    
    Given a stationary velocity field v, computes the deformation φ = exp(v)
    through the recurrence: φ = φ ∘ φ (composition with itself).
    """
    
    def __init__(
        self,
        num_squaring: int = 6,
        mode: str = 'bilinear',
        padding_mode: str = 'border',
        align_corners: bool = True,
    ):
        super().__init__()
        self.num_squaring = num_squaring
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
    
    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Compute deformation field from velocity field using scaling and squaring.
        
        Args:
            velocity: Stationary velocity field (B, 3, D, H, W)
            
        Returns:
            deformation: Deformation field φ (B, 3, D, H, W)
        """
        # Scale velocity by 2^(-num_squaring)
        flow = velocity / (2 ** self.num_squaring)
        
        # Perform squaring iterations
        for _ in range(self.num_squaring):
            flow = self.compose_deformations(flow, flow)
        
        return flow
    
    def compose_deformations(
        self, 
        flow1: torch.Tensor, 
        flow2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose two deformation fields: φ = φ1 ∘ φ2
        
        Args:
            flow1: First deformation field (B, 3, D, H, W)
            flow2: Second deformation field (B, 3, D, H, W)
            
        Returns:
            composed: Composed deformation field (B, 3, D, H, W)
        """
        # Create sampling grid
        grid = self.get_grid(flow1)
        
        # Apply flow2 to get intermediate positions
        sample_grid = grid + flow2.permute(0, 2, 3, 4, 1)
        
        # Normalize grid to [-1, 1] for grid_sample
        sample_grid = self.normalize_grid(sample_grid, flow1.shape[2:])
        
        # Sample flow1 at displaced positions
        flow1_displaced = F.grid_sample(
            flow1,
            sample_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )
        
        # Compose: φ(x) = φ1(x + φ2(x)) + φ2(x)
        composed = flow1_displaced + flow2
        
        return composed
    
    def get_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Generate a coordinate grid matching the input tensor shape.
        
        Args:
            tensor: Reference tensor (B, C, D, H, W)
            
        Returns:
            grid: Coordinate grid (B, D, H, W, 3)
        """
        B, _, D, H, W = tensor.shape
        
        # Create coordinate vectors
        d = torch.arange(D, dtype=tensor.dtype, device=tensor.device)
        h = torch.arange(H, dtype=tensor.dtype, device=tensor.device)
        w = torch.arange(W, dtype=tensor.dtype, device=tensor.device)
        
        # Create meshgrid
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        
        # Stack and expand to batch
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)  # (D, H, W, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, D, H, W, 3)
        
        return grid
    
    def normalize_grid(
        self, 
        grid: torch.Tensor, 
        shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Normalize grid coordinates to [-1, 1] range for grid_sample.
        
        Args:
            grid: Coordinate grid (B, D, H, W, 3)
            shape: Shape of the volume (D, H, W)
            
        Returns:
            normalized: Normalized grid (B, D, H, W, 3)
        """
        D, H, W = shape
        
        # Normalize each dimension
        grid_norm = grid.clone()
        grid_norm[..., 0] = 2 * grid[..., 0] / (W - 1) - 1  # width
        grid_norm[..., 1] = 2 * grid[..., 1] / (H - 1) - 1  # height
        grid_norm[..., 2] = 2 * grid[..., 2] / (D - 1) - 1  # depth
        
        return grid_norm


class SpatialTransformer(nn.Module):
    """
    Spatial transformer for warping images with deformation fields.
    """
    
    def __init__(
        self,
        mode: str = 'bilinear',
        padding_mode: str = 'border',
        align_corners: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
    
    def forward(
        self, 
        image: torch.Tensor, 
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp an image using a deformation field.
        
        Args:
            image: Input image (B, C, D, H, W)
            flow: Deformation field (B, 3, D, H, W)
            
        Returns:
            warped: Warped image (B, C, D, H, W)
        """
        # Generate base grid
        B, C, D, H, W = image.shape
        grid = self.get_grid(image)
        
        # Apply deformation
        sample_grid = grid + flow.permute(0, 2, 3, 4, 1)
        
        # Normalize for grid_sample
        sample_grid = self.normalize_grid(sample_grid, (D, H, W))
        
        # Warp image
        warped = F.grid_sample(
            image,
            sample_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )
        
        return warped
    
    def get_grid(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generate coordinate grid."""
        B, _, D, H, W = tensor.shape
        
        d = torch.arange(D, dtype=tensor.dtype, device=tensor.device)
        h = torch.arange(H, dtype=tensor.dtype, device=tensor.device)
        w = torch.arange(W, dtype=tensor.dtype, device=tensor.device)
        
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        return grid
    
    def normalize_grid(
        self, 
        grid: torch.Tensor, 
        shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Normalize grid to [-1, 1]."""
        D, H, W = shape
        
        grid_norm = grid.clone()
        grid_norm[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
        grid_norm[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
        grid_norm[..., 2] = 2 * grid[..., 2] / (D - 1) - 1
        
        return grid_norm


def compute_jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian determinant of deformation field.
    
    Args:
        flow: Deformation field (B, 3, D, H, W)
        
    Returns:
        jacobian_det: Jacobian determinant (B, 1, D-2, H-2, W-2)
    """
    B, _, D, H, W = flow.shape
    
    # Compute gradients using central differences
    # Note: This reduces spatial dimensions by 2 in each direction
    
    # Gradients in x direction
    dudx = (flow[:, 0, 1:-1, 1:-1, 2:] - flow[:, 0, 1:-1, 1:-1, :-2]) / 2
    dvdx = (flow[:, 1, 1:-1, 1:-1, 2:] - flow[:, 1, 1:-1, 1:-1, :-2]) / 2
    dwdx = (flow[:, 2, 1:-1, 1:-1, 2:] - flow[:, 2, 1:-1, 1:-1, :-2]) / 2
    
    # Gradients in y direction
    dudy = (flow[:, 0, 1:-1, 2:, 1:-1] - flow[:, 0, 1:-1, :-2, 1:-1]) / 2
    dvdy = (flow[:, 1, 1:-1, 2:, 1:-1] - flow[:, 1, 1:-1, :-2, 1:-1]) / 2
    dwdy = (flow[:, 2, 1:-1, 2:, 1:-1] - flow[:, 2, 1:-1, :-2, 1:-1]) / 2
    
    # Gradients in z direction
    dudz = (flow[:, 0, 2:, 1:-1, 1:-1] - flow[:, 0, :-2, 1:-1, 1:-1]) / 2
    dvdz = (flow[:, 1, 2:, 1:-1, 1:-1] - flow[:, 1, :-2, 1:-1, 1:-1]) / 2
    dwdz = (flow[:, 2, 2:, 1:-1, 1:-1] - flow[:, 2, :-2, 1:-1, 1:-1]) / 2
    
    # Add identity matrix
    dudx = dudx + 1
    dvdy = dvdy + 1
    dwdz = dwdz + 1
    
    # Compute determinant
    det = dudx * (dvdy * dwdz - dvdz * dwdy) - \
          dudy * (dvdx * dwdz - dvdz * dwdx) + \
          dudz * (dvdx * dwdy - dvdy * dwdx)
    
    return det.unsqueeze(1) 