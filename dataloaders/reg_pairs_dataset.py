# file: dataloaders/reg_pairs_dataset.py
import csv
import random
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from utils.preprocessing import normalize_volume
from utils.patch_utils import extract_patches, PatchAugmentor
from utils.preprocessing import resample_volume


def _read_pairs_csv(csv_path: Path) -> List[Dict[str, Optional[str]]]:
    """
    Read path-based registration pairs CSV.

    Expected columns:
        fixed, moving [, fixed_seg, moving_seg]
    """
    rows: List[Dict[str, Optional[str]]] = []
    if not csv_path or not csv_path.exists():
        return rows

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for r in reader:
            rows.append({
                "fixed": r.get("fixed"),
                "moving": r.get("moving"),
                "fixed_seg": r.get("fixed_seg"),
                "moving_seg": r.get("moving_seg"),
            })
    return rows


def _worker_init_fn(worker_id: int):
    """Initialize each worker with a unique seed to avoid duplicate randomness."""
    base_seed = 42
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class RegPairsDataset(Dataset):
    """
    Path-based registration dataset.
    - CSV columns: fixed, moving [, fixed_seg, moving_seg]
    - Each row is repeated `patches_per_pair` times to yield many samples.
    - Uses the same patch extraction / augmentation conventions as OASIS loader.
    """

    def __init__(
        self,
        pairs_csv: str,
        patch_size: int = 64,
        patch_stride: int = 32,
        patches_per_pair: int = 20,
        augment: bool = False,
        use_labels: bool = False,
        seed: int = 42,
        target_size: Optional[Tuple[int,int,int]] = None,
    ):
        self.csv_path = Path(pairs_csv)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

        self.rows = _read_pairs_csv(self.csv_path)
        if len(self.rows) == 0:
            raise RuntimeError(f"No pairs in CSV: {pairs_csv}")

        assert target_size is not None, "target_size must be provided to ensure fixed and moving images are resampled to the same shape for aligned patch grids"

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        self.target_size = target_size

        random.seed(seed)
        np.random.seed(seed)

        self.augmentor = PatchAugmentor() if self.augment else None
        
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_max = 32
        self._patch_cache = OrderedDict()
        self._patch_cache_max = 64


    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_pair

    def _load_volume(self, path: str, is_label: bool = False) -> Optional[np.ndarray]:
        """Load a NIfTI and (optionally) resample once, then cache the result."""
        if not path or not os.path.exists(path):
            return None
        target = tuple(self.target_size) if self.target_size else None
        mode = "nearest" if is_label else "trilinear"
        key = (path, bool(is_label), target, mode)
        # return cached resampled
        if key in self._cache:
            arr = self._cache.pop(key)
            self._cache[key] = arr
            return arr
        # load raw (cache raw too to avoid gz decode repeatedly)
        raw_key = ("RAW", path)
        if raw_key in self._cache:
            raw = self._cache.pop(raw_key); self._cache[raw_key] = raw
        else:
            # nii = nib.load(str(path))
            # raw = nii.get_fdata()
            
            nii = nib.load(str(path))
            # Reorient to canonical RAS to ensure consistent axes
            try:
                nii = nib.as_closest_canonical(nii)
            except Exception:
                pass
            raw = nii.get_fdata()
                        
            # store spacings (sz, sy, sx) on the side for this path
            try:
                zooms = tuple(map(float, nii.header.get_zooms()[:3]))
            except Exception:
                zooms = (1.0, 1.0, 1.0)
            self._cache[("SPACING", path)] = zooms
            raw = raw.astype(np.int32 if is_label else np.float32)
            self._cache[raw_key] = raw
            if len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        arr = raw
        if target is not None:
            t = torch.from_numpy(arr)[None, None]
            arr = resample_volume(t.float(), target, mode=mode).squeeze().cpu().numpy()
            arr = arr.astype(np.int32 if is_label else np.float32)
            # update spacing: physical size preserved ⇒ spacing scales with shape
            old_shape = raw.shape[:3]
            sz, sy, sx = self._cache.get(("SPACING", path), (1.0, 1.0, 1.0))
            oz, oh, ow = old_shape
            nz, nh, nw = target
            new_spacing = (sz * oz / nz, sy * oh / nh, sx * ow / nw)
            self._cache[("SPACING", path)] = new_spacing
        self._cache[key] = arr
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return arr

    def _get_patches_cached(self, volume: torch.Tensor, volume_key: str):
        if volume_key in self._patch_cache:
            patches = self._patch_cache.pop(volume_key)
            self._patch_cache[volume_key] = patches
            return patches
        
        patches, _ = extract_patches(volume, self.patch_size, self.patch_stride)
        self._patch_cache[volume_key] = patches
        if len(self._patch_cache) > self._patch_cache_max:
            self._patch_cache.popitem(last=False)
        return patches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx = idx // self.patches_per_pair
        patch_offset = idx % self.patches_per_pair
        row = self.rows[row_idx]

        f_path, m_path = row["fixed"], row["moving"]
        if not f_path or not m_path:
            raise RuntimeError(f"Bad row in {self.csv_path}: {row}")

        f_img = self._load_volume(f_path, is_label=False)
        m_img = self._load_volume(m_path, is_label=False)

        fixed = torch.from_numpy(f_img).unsqueeze(0).float()
        moving = torch.from_numpy(m_img).unsqueeze(0).float()
        f_spacing = torch.tensor(self._cache.get(("SPACING", f_path), (1.0, 1.0, 1.0)), dtype=torch.float32)
        m_spacing = torch.tensor(self._cache.get(("SPACING", m_path), (1.0, 1.0, 1.0)), dtype=torch.float32)

        fixed = normalize_volume(fixed)
        moving = normalize_volume(moving)

        f_key = (f_path, self.patch_size, self.patch_stride)
        m_key = (m_path, self.patch_size, self.patch_stride)
        fixed_patches = self._get_patches_cached(fixed, f_key)
        moving_patches = self._get_patches_cached(moving, m_key)

        if not fixed_patches or not moving_patches:
            raise RuntimeError("Failed to extract patches from input volumes")

        max_valid_idx = min(len(fixed_patches), len(moving_patches)) - 1
        if max_valid_idx < 0:
            raise RuntimeError("No overlapping valid patches between fixed/moving volumes")

        def _is_valid(p: torch.Tensor) -> bool:
            if p.dtype.is_floating_point:
                pm = p.float()
                p_flat = pm.flatten()
                if p_flat.numel() == 0:
                    return False
                p10 = torch.quantile(p_flat, 0.10).item()
                p90 = torch.quantile(p_flat, 0.90).item()
                return (p90 - p10) > 1e-6
            else:
                return (p > 0).float().mean().item() > 0.10

        valid_indices = [
            i for i in range(max_valid_idx + 1)
            if _is_valid(fixed_patches[i]) and _is_valid(moving_patches[i])
        ]
        
        if not valid_indices:
            valid_indices = list(range(max_valid_idx + 1))

        if self.augment:
            patch_idx = random.choice(valid_indices)
        else:
            patch_idx = valid_indices[patch_offset % len(valid_indices)]

        fixed_patch = fixed_patches[patch_idx]
        moving_patch = moving_patches[patch_idx]

        out = {
            "fixed": fixed_patch,   # (1, S, S, S)
            "moving": moving_patch, # (1, S, S, S)
            "pair_index": row_idx,
            # use fixed spacing as the reference grid for loss/metrics
            "spacing": f_spacing,  # (sz, sy, sx) in mm/voxel
        }

        # Augmentation (synchronized for both)
        # Augmentation (synchronized for both): seed ALL RNGs: random, numpy, torch
        if self.augment and self.augmentor is not None:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            out["fixed"] = self.augmentor.augment(fixed_patch)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            out["moving"] = self.augmentor.augment(moving_patch)

        # Optional segmentation
        if self.use_labels and row.get("fixed_seg") and row.get("moving_seg"):
            f_seg = self._load_volume(row["fixed_seg"], is_label=True)
            m_seg = self._load_volume(row["moving_seg"], is_label=True)
            if f_seg is not None and m_seg is not None:
                fseg_t = torch.from_numpy(f_seg).unsqueeze(0).long()
                mseg_t = torch.from_numpy(m_seg).unsqueeze(0).long()

                fseg_key = (row["fixed_seg"], self.patch_size, self.patch_stride)
                mseg_key = (row["moving_seg"], self.patch_size, self.patch_stride)
                fseg_patches = self._get_patches_cached(fseg_t, fseg_key)
                mseg_patches = self._get_patches_cached(mseg_t, mseg_key)

                if (fseg_patches and mseg_patches and
                        patch_idx < len(fseg_patches) and patch_idx < len(mseg_patches)):
                    out["segmentation_fixed"] = fseg_patches[patch_idx]
                    out["segmentation_moving"] = mseg_patches[patch_idx]

        return out


# --- DDP-aware DataLoader builder for RegPairsDataset ---
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def create_reg_pairs_loaders_ddp(
    train_pairs_csv: str,
    val_pairs_csv: str,
    *,
    batch_size: int,
    patch_size: int,
    patch_stride: int,
    patches_per_pair: int,
    num_workers: int,
    use_labels_train: bool = True,
    use_labels_val: bool = True,
    target_size=None,
    distributed: bool = False,
):
    assert target_size is not None, "target_size must be provided to ensure fixed and moving images are resampled to the same shape for aligned patch grids"
    train_ds = RegPairsDataset(
        train_pairs_csv,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=True,
        use_labels=use_labels_train,
        target_size=target_size,
    )
    val_ds = RegPairsDataset(
        val_pairs_csv,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=False,
        use_labels=use_labels_val,
        target_size=target_size,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if distributed else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, train_sampler, val_sampler
