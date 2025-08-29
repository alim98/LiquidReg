"""
Preprocessing utilities for medical images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def normalize_volume(volume: torch.Tensor, method: str = "zscore") -> torch.Tensor:
    """
    Normalize a 3D volume.
    
    Args:
        volume: Input volume (B, C, D, H, W) or (C, D, H, W)
        method: Normalization method ("zscore", "minmax", "percentile")
        
    Returns:
        normalized: Normalized volume
    """
    if method == "zscore":
        # Z-score normalization
        mean = volume.mean()
        std = volume.std()
        normalized = (volume - mean) / (std + 1e-8)
    
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_val = volume.min()
        max_val = volume.max()
        normalized = (volume - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "percentile":
        # Percentile-based normalization (robust to outliers)
        p1 = torch.quantile(volume, 0.01)
        p99 = torch.quantile(volume, 0.99)
        normalized = torch.clamp((volume - p1) / (p99 - p1 + 1e-8), 0, 1)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


# def resample_volume(
#     volume: torch.Tensor,
#     target_shape: Tuple[int, int, int],
#     mode: str = "trilinear"
# ) -> torch.Tensor:
#     """
#     Resample a volume to target shape.
    
#     Args:
#         volume: Input volume (B, C, D, H, W) or (C, D, H, W)
#         target_shape: Target spatial shape (D, H, W)
#         mode: Interpolation mode
        
#     Returns:
#         resampled: Resampled volume
#     """
#     if volume.dim() == 4:
#         volume = volume.unsqueeze(0)
#         squeeze_batch = True
#     else:
#         squeeze_batch = False
    
#     resampled = F.interpolate(
#         volume,
#         size=target_shape,
#         mode=mode,
#         align_corners=False
#     )
    
#     if squeeze_batch:
#         resampled = resampled.squeeze(0)
    
#     return resampled

# utils/preprocessing.py

def resample_volume(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str = "trilinear"
) -> torch.Tensor:
    """
    Resample a volume to target shape.

    Args:
        volume: (B, C, D, H, W) or (C, D, H, W)
        target_shape: (D, H, W)
        mode: 'trilinear' for images, 'nearest' for labels, etc.
    """
    # Ensure 5D input for interpolate
    squeeze_batch = False
    if volume.dim() == 4:                 # (C, D, H, W)
        volume = volume.unsqueeze(0)      # -> (1, C, D, H, W)
        squeeze_batch = True

    kwargs = dict(size=target_shape, mode=mode)
    # Only set align_corners for linear family modes
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False

    resampled = F.interpolate(volume, **kwargs)

    if squeeze_batch:
        resampled = resampled.squeeze(0)  # back to (C, D, H, W)
    return resampled


def pad_or_crop_to_shape(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int],
    mode: str = "constant",
    value: float = 0.0
) -> torch.Tensor:
    """
    Pad or crop volume to target shape.
    
    Args:
        volume: Input volume (B, C, D, H, W) or (C, D, H, W)
        target_shape: Target spatial shape (D, H, W)
        mode: Padding mode for torch.nn.functional.pad
        value: Fill value for constant padding
        
    Returns:
        output: Padded/cropped volume
    """
    current_shape = volume.shape[-3:]
    target_d, target_h, target_w = target_shape
    current_d, current_h, current_w = current_shape
    
    # Calculate padding/cropping for each dimension
    def calc_pad_crop(current_size, target_size):
        if current_size < target_size:
            # Need padding
            total_pad = target_size - current_size
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            return (pad_before, pad_after, 0, current_size)
        else:
            # Need cropping
            total_crop = current_size - target_size
            crop_before = total_crop // 2
            crop_after = crop_before + target_size
            return (0, 0, crop_before, crop_after)
    
    # Calculate operations for each dimension
    w_pad_before, w_pad_after, w_crop_start, w_crop_end = calc_pad_crop(current_w, target_w)
    h_pad_before, h_pad_after, h_crop_start, h_crop_end = calc_pad_crop(current_h, target_h)
    d_pad_before, d_pad_after, d_crop_start, d_crop_end = calc_pad_crop(current_d, target_d)
    
    # Apply cropping first
    if w_crop_end > 0:
        volume = volume[..., w_crop_start:w_crop_end]
    if h_crop_end > 0:
        volume = volume[..., h_crop_start:h_crop_end, :]
    if d_crop_end > 0:
        volume = volume[..., d_crop_start:d_crop_end, :, :]
    
    # Apply padding
    padding = [w_pad_before, w_pad_after, h_pad_before, h_pad_after, d_pad_before, d_pad_after]
    if any(p > 0 for p in padding):
        volume = F.pad(volume, padding, mode=mode, value=value)
    
    return volume


def apply_augmentation(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    rotation_range: float = 5.0,
    translation_range: float = 0.1,
    scale_range: float = 0.1,
    noise_std: float = 0.01,
    probability: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply data augmentation to image pairs.
    
    Args:
        fixed: Fixed image (B, C, D, H, W)
        moving: Moving image (B, C, D, H, W)
        rotation_range: Max rotation in degrees
        translation_range: Max translation as fraction of image size
        scale_range: Max scaling factor deviation
        noise_std: Standard deviation of Gaussian noise
        probability: Probability of applying augmentation
        
    Returns:
        aug_fixed, aug_moving: Augmented images
    """
    if torch.rand(1).item() > probability:
        return fixed, moving
    
    B, C, D, H, W = fixed.shape
    device = fixed.device
    
    # Generate random transformation parameters
    angles = (torch.rand(B, 3, device=device) - 0.5) * 2 * rotation_range
    translations = (torch.rand(B, 3, device=device) - 0.5) * 2 * translation_range
    scales = 1 + (torch.rand(B, 3, device=device) - 0.5) * 2 * scale_range
    
    # Create affine transformation matrices
    def create_affine_matrix(angle_x, angle_y, angle_z, tx, ty, tz, sx, sy, sz):
        # Convert angles to radians
        ax, ay, az = torch.deg2rad(angle_x), torch.deg2rad(angle_y), torch.deg2rad(angle_z)
        
        # Rotation matrices
        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(ax), -torch.sin(ax), 0],
            [0, torch.sin(ax), torch.cos(ax), 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        Ry = torch.tensor([
            [torch.cos(ay), 0, torch.sin(ay), 0],
            [0, 1, 0, 0],
            [-torch.sin(ay), 0, torch.cos(ay), 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        Rz = torch.tensor([
            [torch.cos(az), -torch.sin(az), 0, 0],
            [torch.sin(az), torch.cos(az), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        # Scale matrix
        S = torch.tensor([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        # Translation matrix
        T = torch.tensor([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ], device=device, dtype=torch.float32)
        
        # Combine transformations: T * Rz * Ry * Rx * S
        return T @ Rz @ Ry @ Rx @ S
    
    # Apply transformations
    aug_fixed = fixed.clone()
    aug_moving = moving.clone()
    
    for b in range(B):
        # Create affine matrix
        affine = create_affine_matrix(
            angles[b, 0], angles[b, 1], angles[b, 2],
            translations[b, 0], translations[b, 1], translations[b, 2],
            scales[b, 0], scales[b, 1], scales[b, 2]
        )
        
        # Extract 3x4 affine matrix for grid_sample
        affine_3x4 = affine[:3, :]
        
        # Generate grid
        grid = F.affine_grid(
            affine_3x4.unsqueeze(0),
            (1, C, D, H, W),
            align_corners=False
        )
        
        # Apply transformation
        aug_fixed[b:b+1] = F.grid_sample(
            fixed[b:b+1],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        aug_moving[b:b+1] = F.grid_sample(
            moving[b:b+1],
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
    
    # Add noise
    if noise_std > 0:
        noise_fixed = torch.randn_like(aug_fixed) * noise_std
        noise_moving = torch.randn_like(aug_moving) * noise_std
        aug_fixed = aug_fixed + noise_fixed
        aug_moving = aug_moving + noise_moving
    
    return aug_fixed, aug_moving


def compute_image_gradients(image: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial gradients of an image.
    
    Args:
        image: Input image (B, C, D, H, W)
        
    Returns:
        gradients: Spatial gradients (B, C, 3, D, H, W)
    """
    # Sobel filters for 3D
    sobel_x = torch.tensor([
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0) / 16.0
    
    sobel_y = torch.tensor([
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0) / 16.0
    
    sobel_z = torch.tensor([
        [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    ], dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0) / 16.0
    
    # Compute gradients
    grad_x = F.conv3d(image, sobel_x, padding=1)
    grad_y = F.conv3d(image, sobel_y, padding=1)
    grad_z = F.conv3d(image, sobel_z, padding=1)
    
    gradients = torch.stack([grad_x, grad_y, grad_z], dim=2)
    return gradients 