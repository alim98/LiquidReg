"""
Patch extraction and augmentation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import random

# --------------------------
# Helper utilities (new)
# --------------------------

def _ensure_5d(volume: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Return (volume_5d, added_batch_flag) accepting 4D (C,D,H,W) or 5D (B,C,D,H,W)."""
    if not torch.is_tensor(volume):
        raise TypeError(f"volume must be torch.Tensor, got {type(volume)}")
    if volume.dim() == 4:          # (C,D,H,W)
        return volume.unsqueeze(0), True
    if volume.dim() == 5:          # (B,C,D,H,W)
        return volume, False
    raise ValueError(f"Expected 4D or 5D tensor, got {volume.dim()}D with shape {tuple(volume.shape)}")

def _gen_positions(size: int, patch_size: int, stride: int) -> List[int]:
    """
    Generate center positions along one dimension. Returns a non-empty list.
    Works even if size < patch_size (falls back to center).
    """
    start = patch_size // 2
    stop  = size - patch_size // 2
    if stop <= start:
        return [max(0, (size - 1) // 2)]
    positions = list(range(start, stop, stride))
    if len(positions) == 0:
        positions = [max(0, (size - 1) // 2)]
    return positions

def _clip_patch_bounds(center: int, size: int, patch_size: int) -> Tuple[int,int]:
    """Given a center, clip the start/end so the patch stays within [0, size)."""
    s = max(0, center - patch_size // 2)
    e = s + patch_size
    if e > size:
        s = max(0, size - patch_size)
        e = size
    return s, e

def _pad_to_size(patch: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Pad a (C,d,h,w) patch to exactly (C,patch_size,patch_size,patch_size)."""
    d = patch.shape[-3]; h = patch.shape[-2]; w = patch.shape[-1]
    pd = max(0, patch_size - d)
    ph = max(0, patch_size - h)
    pw = max(0, patch_size - w)
    if pd or ph or pw:
        # F.pad order for 3D: (wL,wR, hL,hR, dL,dR)
        padding = (pw // 2, pw - pw // 2,
                   ph // 2, ph - ph // 2,
                   pd // 2, pd - pd // 2)
        patch = F.pad(patch, padding, mode='constant', value=0)
    return patch

# --------------------------
# Patch extraction
# --------------------------

def extract_patches(
    volume: torch.Tensor,
    patch_size: int,
    stride: int,
    threshold: float = 0.1
) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from a volume.

    Accepts 4D (C,D,H,W) or 5D (B,C,D,H,W).
    Returns:
        patches: List[Tensor] of shape (C, S, S, S)
        coordinates: List[(d_center, h_center, w_center)]
    """
    volume, _added_batch = _ensure_5d(volume)
    B, C, D, H, W = volume.shape

    patches: List[torch.Tensor] = []
    coordinates: List[Tuple[int, int, int]] = []

    d_positions = _gen_positions(D, patch_size, stride)
    h_positions = _gen_positions(H, patch_size, stride)
    w_positions = _gen_positions(W, patch_size, stride)

    for b in range(B):
        vol = volume[b]  # (C,D,H,W)
        for d_center in d_positions:
            for h_center in h_positions:
                for w_center in w_positions:
                    d_start, d_end = _clip_patch_bounds(d_center, D, patch_size)
                    h_start, h_end = _clip_patch_bounds(h_center, H, patch_size)
                    w_start, w_end = _clip_patch_bounds(w_center, W, patch_size)

                    patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
                    patch = _pad_to_size(patch, patch_size)

                    # dtype-aware validity check
                    try:
                        if patch.dtype in (torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16):
                            is_valid = (patch > 0).sum().item() > 0
                        else:
                            is_valid = patch.float().mean().item() > threshold
                    except Exception:
                        is_valid = True  # fallback if something odd happens

                    if is_valid:
                        patches.append(patch)
                        coordinates.append((d_center, h_center, w_center))

    # Fallback: ensure at least one patch per B if nothing passed the threshold
    if len(patches) == 0:
        dc, hc, wc = D // 2, H // 2, W // 2
        for b in range(B):
            vol = volume[b]
            d_start, d_end = _clip_patch_bounds(dc, D, patch_size)
            h_start, h_end = _clip_patch_bounds(hc, H, patch_size)
            w_start, w_end = _clip_patch_bounds(wc, W, patch_size)
            patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
            patch = _pad_to_size(patch, patch_size)
            patches.append(patch)
            coordinates.append((dc, hc, wc))

    return patches, coordinates

def extract_random_patches(
    volume: torch.Tensor,
    patch_size: int,
    num_patches: int,
    min_intensity: float = 0.1
) -> List[torch.Tensor]:
    """
    Extract random patches from a volume.

    Accepts 4D (C,D,H,W) or 5D (B,C,D,H,W).
    Returns a list of (C, S, S, S) tensors.
    """
    volume, _added_batch = _ensure_5d(volume)
    B, C, D, H, W = volume.shape

    def _rand_center(size: int) -> int:
        # inclusive randint bounds
        lo = max(patch_size // 2, 0)
        hi = max(size - patch_size // 2 - 1, 0)
        if hi < lo:
            return max(0, (size - 1) // 2)
        return random.randint(lo, hi)

    patches: List[torch.Tensor] = []
    for b in range(B):
        vol = volume[b]  # (C,D,H,W)
        valid = 0
        attempts = 0
        max_attempts = max(50, num_patches * 10)

        while valid < num_patches and attempts < max_attempts:
            dc, hc, wc = _rand_center(D), _rand_center(H), _rand_center(W)
            d_start, d_end = _clip_patch_bounds(dc, D, patch_size)
            h_start, h_end = _clip_patch_bounds(hc, H, patch_size)
            w_start, w_end = _clip_patch_bounds(wc, W, patch_size)
            patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
            patch = _pad_to_size(patch, patch_size)

            try:
                if patch.dtype in (torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16):
                    is_valid = (patch > 0).sum().item() > 0
                else:
                    is_valid = patch.float().mean().item() > min_intensity
            except Exception:
                is_valid = True

            if is_valid:
                patches.append(patch)
                valid += 1
            attempts += 1

        # If not enough valid patches, fill with center-ish patches
        while valid < num_patches:
            dc, hc, wc = D // 2, H // 2, W // 2
            d_start, d_end = _clip_patch_bounds(dc, D, patch_size)
            h_start, h_end = _clip_patch_bounds(hc, H, patch_size)
            w_start, w_end = _clip_patch_bounds(wc, W, patch_size)
            patch = vol[:, d_start:d_end, h_start:h_end, w_start:w_end]
            patch = _pad_to_size(patch, patch_size)
            patches.append(patch)
            valid += 1

    return patches

# --------------------------
# Augmentation
# --------------------------

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

        # Random 90-degree rotations (in-plane)
        if random.random() < self.rotation_prob:
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

# --------------------------
# Reconstruction & stats
# --------------------------

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

    # NOTE: batch ownership of each patch is not tracked here; batch_idx=0 is assumed.
    # If you need multi-B reconstruction, pass/track batch indices alongside `coordinates`.
    for patch, (d_center, h_center, w_center) in zip(patches, coordinates):
        d_start = d_center - patch_size // 2
        d_end = d_start + patch_size
        h_start = h_center - patch_size // 2
        h_end = h_start + patch_size
        w_start = w_center - patch_size // 2
        w_end = w_start + patch_size

        batch_idx = 0  # Simplified: all into batch 0

        reconstructed[batch_idx, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
            patch * weight_patch.unsqueeze(0)
        weight_map[batch_idx, :, d_start:d_end, h_start:h_end, w_start:w_end] += \
            weight_patch.unsqueeze(0)

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

    patch_tensor = torch.stack(patches, dim=0)  # (N, C, D, H, W)
    try:
        if patch_tensor.dtype in (torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16):
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
        stats = {
            "error": str(e),
            "count": len(patches),
            "shape": patch_tensor.shape[1:] if len(patches) > 0 else None,
        }
    return stats
