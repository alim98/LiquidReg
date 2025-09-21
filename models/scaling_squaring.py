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
        Compute deformation field from velocity field via scaling & squaring.
        
        Args:
            velocity: Velocity field (B, 3, D, H, W)
            
        Returns:
            deformation: Deformation field (B, 3, D, H, W)
        """

        
        # Scale velocity field for integration
        scaled_velocity = velocity / (2 ** self.num_squaring)
        
        # Initial deformation is scaled velocity
        deformation = scaled_velocity
        
        # Squaring steps (compose deformation with itself)
        for _ in range(self.num_squaring):
            deformation = self.compose_deformations(deformation, deformation)
            
            # Check for NaNs and fix them
            deformation = torch.nan_to_num(deformation, nan=0.0)
        

        
        return deformation
    
    def compose_deformations(
        self, 
        flow1: torch.Tensor, 
        flow2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compose two deformation fields.
        
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
        
        # Check for NaNs and fix them
        sample_grid = torch.nan_to_num(sample_grid, nan=0.0)
        
        # Normalize grid to [-1, 1] for grid_sample
        sample_grid = self.normalize_grid(sample_grid, flow1.shape[2:])
        
        # Clamp to valid range with small margin to prevent boundary issues
        sample_grid = torch.clamp(sample_grid, -0.999, 0.999)
        
        # Sample flow1 at displaced positions
        flow1_displaced = F.grid_sample(
            flow1,
            sample_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )
        
        # Check for NaNs and fix them
        flow1_displaced = torch.nan_to_num(flow1_displaced, nan=0.0)
        
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


# at top of the file you already have: from typing import Optional, Tuple
# keep that import

def compute_jacobian_determinant(flow: torch.Tensor, spacing: Optional[torch.Tensor | tuple] = None) -> torch.Tensor:
    """
    Jacobian determinant with anisotropic spacing.
    spacing: (sz, sy, sx) in mm/voxel, or None → assumes (1,1,1)
    Returns: (B, 1, D-2, H-2, W-2)
    """
    # default to isotropic if not provided (keeps old call sites working)
    if spacing is None:
        spacing = flow.new_tensor((1.0, 1.0, 1.0))

    # normalize spacing to device/dtype and split
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.to(device=flow.device, dtype=flow.dtype)
        if spacing.dim() == 1:  # (3,)
            spacing = spacing.view(1, 3)
        sz = spacing[..., 0].view(-1, 1, 1, 1)
        sy = spacing[..., 1].view(-1, 1, 1, 1)
        sx = spacing[..., 2].view(-1, 1, 1, 1)
    else:
        sz, sy, sx = spacing
        sz = torch.tensor(sz, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)
        sy = torch.tensor(sy, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)
        sx = torch.tensor(sx, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)

    # central differences / spacing
    dudx = (flow[:, 0, 1:-1, 1:-1, 2:] - flow[:, 0, 1:-1, 1:-1, :-2]) / (2 * sx)
    dvdx = (flow[:, 1, 1:-1, 1:-1, 2:] - flow[:, 1, 1:-1, 1:-1, :-2]) / (2 * sx)
    dwdx = (flow[:, 2, 1:-1, 1:-1, 2:] - flow[:, 2, 1:-1, 1:-1, :-2]) / (2 * sx)

    dudy = (flow[:, 0, 1:-1, 2:, 1:-1] - flow[:, 0, 1:-1, :-2, 1:-1]) / (2 * sy)
    dvdy = (flow[:, 1, 1:-1, 2:, 1:-1] - flow[:, 1, 1:-1, :-2, 1:-1]) / (2 * sy)
    dwdy = (flow[:, 2, 1:-1, 2:, 1:-1] - flow[:, 2, 1:-1, :-2, 1:-1]) / (2 * sy)

    dudz = (flow[:, 0, 2:, 1:-1, 1:-1] - flow[:, 0, :-2, 1:-1, 1:-1]) / (2 * sz)
    dvdz = (flow[:, 1, 2:, 1:-1, 1:-1] - flow[:, 1, :-2, 1:-1, 1:-1]) / (2 * sz)
    dwdz = (flow[:, 2, 2:, 1:-1, 1:-1] - flow[:, 2, :-2, 1:-1, 1:-1]) / (2 * sz)

    # +I on diagonal
    dudx = dudx + 1
    dvdy = dvdy + 1
    dwdz = dwdz + 1

    # (optional) clip off-diagonals only
    # max_grad = 10.0
    # dudy = torch.clamp(dudy, -max_grad, max_grad)
    # dudz = torch.clamp(dudz, -max_grad, max_grad)
    # dvdx = torch.clamp(dvdx, -max_grad, max_grad)
    # dvdz = torch.clamp(dvdz, -max_grad, max_grad)
    # dwdx = torch.clamp(dwdx, -max_grad, max_grad)
    # dwdy = torch.clamp(dwdy, -max_grad, max_grad)

    det = dudx * (dvdy * dwdz - dvdz * dwdy) - \
          dudy * (dvdx * dwdz - dvdz * dwdx) + \
          dudz * (dvdx * dwdy - dvdy * dwdx)

    return det.unsqueeze(1)


def compute_jacobian_determinant_raw(flow: torch.Tensor, spacing: Optional[torch.Tensor | tuple] = None) -> torch.Tensor:
    """
    Unclamped Jacobian determinant (truthful folding stats).
    spacing: (sz, sy, sx) or None → assumes (1,1,1)
    Returns: (B, 1, D-2, H-2, W-2)
    """
    if spacing is None:
        spacing = flow.new_tensor((1.0, 1.0, 1.0))

    if isinstance(spacing, torch.Tensor):
        spacing = spacing.to(device=flow.device, dtype=flow.dtype)
        if spacing.dim() == 1:
            spacing = spacing.view(1, 3)
        sz = spacing[..., 0].view(-1, 1, 1, 1)
        sy = spacing[..., 1].view(-1, 1, 1, 1)
        sx = spacing[..., 2].view(-1, 1, 1, 1)
    else:
        sz, sy, sx = spacing
        sz = torch.tensor(sz, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)
        sy = torch.tensor(sy, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)
        sx = torch.tensor(sx, device=flow.device, dtype=flow.dtype).view(1, 1, 1, 1)

    dudx = (flow[:, 0, 1:-1, 1:-1, 2:] - flow[:, 0, 1:-1, 1:-1, :-2]) / (2 * sx)
    dvdx = (flow[:, 1, 1:-1, 1:-1, 2:] - flow[:, 1, 1:-1, 1:-1, :-2]) / (2 * sx)
    dwdx = (flow[:, 2, 1:-1, 1:-1, 2:] - flow[:, 2, 1:-1, 1:-1, :-2]) / (2 * sx)

    dudy = (flow[:, 0, 1:-1, 2:, 1:-1] - flow[:, 0, 1:-1, :-2, 1:-1]) / (2 * sy)
    dvdy = (flow[:, 1, 1:-1, 2:, 1:-1] - flow[:, 1, 1:-1, :-2, 1:-1]) / (2 * sy)
    dwdy = (flow[:, 2, 1:-1, 2:, 1:-1] - flow[:, 2, 1:-1, :-2, 1:-1]) / (2 * sy)

    dudz = (flow[:, 0, 2:, 1:-1, 1:-1] - flow[:, 0, :-2, 1:-1, 1:-1]) / (2 * sz)
    dvdz = (flow[:, 1, 2:, 1:-1, 1:-1] - flow[:, 1, :-2, 1:-1, 1:-1]) / (2 * sz)
    dwdz = (flow[:, 2, 2:, 1:-1, 1:-1] - flow[:, 2, :-2, 1:-1, 1:-1]) / (2 * sz)

    dudx = dudx + 1
    dvdy = dvdy + 1
    dwdz = dwdz + 1

    det = dudx * (dvdy * dwdz - dvdz * dwdy) - \
          dudy * (dvdx * dwdz - dvdz * dwdx) + \
          dudz * (dvdx * dwdy - dvdy * dwdx)

    return det.unsqueeze(1)



def foldings_percent(jac_det: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    """
    Percent of voxels with negative det(J).
    jac_det: (B, 1, D-2, H-2, W-2)
    Returns:
        scalar % if per_sample=False, else (B,) tensor with per-sample %.
    """
    neg = (jac_det < 0).float()
    if per_sample:
        return neg.mean(dim=(1, 2, 3, 4)) * 100.0
    return neg.mean() * 100.0
