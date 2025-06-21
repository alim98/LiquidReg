"""
Patch extraction and augmentation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import random


def extract_patches(
    volume: torch.Tensor,
    patch_size: int,
    stride: int,
    threshold: float = 0.1
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from a volume.
    
    Args:
        volume: Input volume (B, C, D, H, W)
        patch_size: Size of cubic patches
        stride: Stride for patch extraction
        threshold: Minimum intensity threshold for patch selection
        
    Returns:
        patches: List of patches (each patch_size x patch_size x patch_size)
        coordinates: List of patch center coordinates
    """
    B, C, D, H, W = volume.shape
    patches = []
    coordinates = []
    
    for b in range(B):
        vol = volume[b]  # (C, D, H, W)
        
        # Calculate patch positions
        d_positions = range(patch_size // 2, D - patch_size // 2, stride)
        h_positions = range(patch_size // 2, H - patch_size // 2, stride)
        w_positions = range(patch_size // 2, W - patch_size // 2, stride)
        
        for d_center in d_positions:
            for h_center in h_positions:
                for w_center in w_positions:
                    # Extract patch
                    d_start = d_center - patch_size // 2
                    d_end = d_start + patch_size
                    h_start = h_center - patch_size // 2
                    h_end = h_start + patch_size
                    w_start = w_center - patch_size // 2
                    w_end = w_start + patch_size
                    
                    patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    # Check if patch contains enough foreground
                    # Handle all possible data types safely
                    try:
                        if patch.dtype in [torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16]:
                            # For integer data (segmentation labels), check if there are any non-zero values
                            if (patch > 0).sum().item() > 0:
                                patches.append(patch)
                                coordinates.append((d_center, h_center, w_center))
                        else:
                            # For floating point data (intensity images), use mean threshold
                            patch_mean = patch.float().mean().item()
                            if patch_mean > threshold:
                                patches.append(patch)
                                coordinates.append((d_center, h_center, w_center))
                    except Exception as e:
                        # Fallback: just add the patch
                        print(f"Warning: Error in patch validation ({e}), adding patch anyway")
                        patches.append(patch)
                        coordinates.append((d_center, h_center, w_center))
    
    return patches, coordinates


def extract_random_patches(
    volume: torch.Tensor,
    patch_size: int,
    num_patches: int,
    min_intensity: float = 0.1
) -> List[torch.Tensor]:
    """
    Extract random patches from a volume.
    
    Args:
        volume: Input volume (B, C, D, H, W)
        patch_size: Size of cubic patches
        num_patches: Number of patches to extract
        min_intensity: Minimum mean intensity for valid patches
        
    Returns:
        patches: List of extracted patches
    """
    B, C, D, H, W = volume.shape
    patches = []
    
    for b in range(B):
        vol = volume[b]  # (C, D, H, W)
        
        valid_patches = 0
        attempts = 0
        max_attempts = num_patches * 10  # Prevent infinite loop
        
        while valid_patches < num_patches and attempts < max_attempts:
            # Random center position
            d_center = random.randint(patch_size // 2, D - patch_size // 2 - 1)
            h_center = random.randint(patch_size // 2, H - patch_size // 2 - 1)
            w_center = random.randint(patch_size // 2, W - patch_size // 2 - 1)
            
            # Extract patch
            d_start = d_center - patch_size // 2
            d_end = d_start + patch_size
            h_start = h_center - patch_size // 2
            h_end = h_start + patch_size
            w_start = w_center - patch_size // 2
            w_end = w_start + patch_size
            
            patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
            
            # Check validity based on data type
            is_valid = False
            try:
                if patch.dtype in [torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16]:
                    # For integer data (segmentation labels), check if there are any non-zero values
                    is_valid = (patch > 0).sum().item() > 0
                else:
                    # For floating point data (intensity images), use mean threshold
                    is_valid = patch.float().mean().item() > min_intensity
            except Exception as e:
                # Fallback: consider patch valid
                print(f"Warning: Error in patch validation ({e}), considering patch valid")
                is_valid = True
                
            if is_valid:
                patches.append(patch)
                valid_patches += 1
            
            attempts += 1
        
        # If we couldn't find enough valid patches, fill with random ones
        while valid_patches < num_patches:
            d_center = random.randint(patch_size // 2, D - patch_size // 2 - 1)
            h_center = random.randint(patch_size // 2, H - patch_size // 2 - 1)
            w_center = random.randint(patch_size // 2, W - patch_size // 2 - 1)
            
            d_start = d_center - patch_size // 2
            d_end = d_start + patch_size
            h_start = h_center - patch_size // 2
            h_end = h_start + patch_size
            w_start = w_center - patch_size // 2
            w_end = w_start + patch_size
            
            patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
            patches.append(patch)
            valid_patches += 1
    
    return patches


class PatchAugmentor:
    """
    Augmentation for 3D patches.
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.05,
        intensity_prob: float = 0.3,
        intensity_range: float = 0.1,
    ):
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.intensity_prob = intensity_prob
        self.intensity_range = intensity_range
    
    def augment(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a patch.
        
        Args:
            patch: Input patch (C, D, H, W)
            
        Returns:
            augmented: Augmented patch
        """
        patch = patch.clone()
        
        # Random flips
        if random.random() < self.flip_prob:
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[-3])  # Flip depth
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[-2])  # Flip height
            if random.random() < 0.5:
                patch = torch.flip(patch, dims=[-1])  # Flip width
        
        # Random 90-degree rotations
        if random.random() < self.rotation_prob:
            # Random number of 90-degree rotations
            k = random.randint(1, 3)
            patch = torch.rot90(patch, k=k, dims=[-2, -1])
        
        # Add noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(patch) * self.noise_std
            patch = patch + noise
        
        # Intensity variations
        if random.random() < self.intensity_prob:
            intensity_factor = 1 + (random.random() - 0.5) * 2 * self.intensity_range
            patch = patch * intensity_factor
        
        return patch


def reconstruct_from_patches(
    patches: List[torch.Tensor],
    coordinates: List[Tuple[int, int, int]],
    volume_shape: Tuple[int, int, int, int, int],  # (B, C, D, H, W)
    patch_size: int,
    overlap_mode: str = "average"  # "average", "gaussian"
) -> torch.Tensor:
    """
    Reconstruct volume from patches.
    
    Args:
        patches: List of patches
        coordinates: List of patch center coordinates
        volume_shape: Target volume shape (B, C, D, H, W)
        patch_size: Size of patches
        overlap_mode: How to handle overlapping regions
        
    Returns:
        reconstructed: Reconstructed volume
    """
    B, C, D, H, W = volume_shape
    reconstructed = torch.zeros(volume_shape, dtype=patches[0].dtype, device=patches[0].device)
    weight_map = torch.zeros(volume_shape, dtype=patches[0].dtype, device=patches[0].device)
    
    if overlap_mode == "gaussian":
        # Create Gaussian weight for smooth blending
        sigma = patch_size / 6.0
        center = patch_size // 2
        weight_patch = torch.zeros((patch_size, patch_size, patch_size))
        
        for i in range(patch_size):
            for j in range(patch_size):
                for k in range(patch_size):
                    dist_sq = (i - center)**2 + (j - center)**2 + (k - center)**2
                    weight_patch[i, j, k] = np.exp(-dist_sq / (2 * sigma**2))
        
        weight_patch = weight_patch.to(patches[0].device)
    else:
        weight_patch = torch.ones((patch_size, patch_size, patch_size), 
                                device=patches[0].device)
    
    # Place patches
    for patch, (d_center, h_center, w_center) in zip(patches, coordinates):
        d_start = d_center - patch_size // 2
        d_end = d_start + patch_size
        h_start = h_center - patch_size // 2
        h_end = h_start + patch_size
        w_start = w_center - patch_size // 2
        w_end = w_start + patch_size
        
        # Determine which batch this patch belongs to
        # (Assuming patches are in order by batch)
        batch_idx = 0  # Simplified - in practice, you'd track this
        
        # Add patch with weights
        reconstructed[batch_idx, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
            patch * weight_patch.unsqueeze(0)
        weight_map[batch_idx, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
            weight_patch.unsqueeze(0)
    
    # Normalize by weights
    weight_map = torch.clamp(weight_map, min=1e-8)
    reconstructed = reconstructed / weight_map
    
    return reconstructed


def compute_patch_statistics(patches: List[torch.Tensor]) -> dict:
    """
    Compute statistics across patches.
    
    Args:
        patches: List of patches
        
    Returns:
        stats: Dictionary of statistics
    """
    if not patches:
        return {}
    
    # Stack patches
    patch_tensor = torch.stack(patches, dim=0)  # (N, C, D, H, W)
    
    # Convert to float for statistics if needed
    try:
        if patch_tensor.dtype in [torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16]:
            # For integer data (segmentation data), compute different statistics
            stats = {
                "mean": patch_tensor.float().mean().item(),
                "non_zero_fraction": (patch_tensor > 0).float().mean().item(),
                "unique_labels": torch.unique(patch_tensor).numel(),
                "min": patch_tensor.min().item(),
                "max": patch_tensor.max().item(),
                "count": len(patches),
                "shape": patch_tensor.shape[1:],  # (C, D, H, W)
            }
        else:
            # For floating point data (intensity data), compute standard statistics
            stats = {
                "mean": patch_tensor.mean().item(),
                "std": patch_tensor.std().item(),
                "min": patch_tensor.min().item(),
                "max": patch_tensor.max().item(),
                "median": patch_tensor.median().item(),
                "count": len(patches),
                "shape": patch_tensor.shape[1:],  # (C, D, H, W)
            }
    except Exception as e:
        # Fallback statistics
        stats = {
            "error": str(e),
            "count": len(patches),
            "shape": patch_tensor.shape[1:] if len(patches) > 0 else None,
        }
    
    return stats 