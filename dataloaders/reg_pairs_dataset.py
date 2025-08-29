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
        resample_mode_img: str = "trilinear",
        resample_mode_lbl: str = "nearest",
    ):
        self.csv_path = Path(pairs_csv)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")

        self.rows = _read_pairs_csv(self.csv_path)
        if len(self.rows) == 0:
            raise RuntimeError(f"No pairs in CSV: {pairs_csv}")

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        self.target_size = target_size
        self.resample_mode_img = resample_mode_img
        self.resample_mode_lbl = resample_mode_lbl

        random.seed(seed)
        np.random.seed(seed)

        self.augmentor = PatchAugmentor() if self.augment else None

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_pair

    def _load_volume(self, path: str, is_label: bool = False) -> Optional[np.ndarray]:
        """Load a NIfTI volume fully into memory."""
        if not path or not os.path.exists(path):
            return None
        arr = nib.load(str(path)).get_fdata()
        
        if self.target_size is not None:
            t = torch.from_numpy(arr)[None, None]  # (1,1,D,H,W)
            mode = self.resample_mode_lbl if is_label else self.resample_mode_img
            # print("DBG", "is_label", is_label, "mode", mode, "shape", t.shape)
            t = resample_volume(t.float(), self.target_size, mode=mode)
            arr = t.squeeze(0).squeeze(0).cpu().numpy()
        
        return arr.astype(np.int32 if is_label else np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx = idx // self.patches_per_pair
        row = self.rows[row_idx]

        f_path, m_path = row["fixed"], row["moving"]
        if not f_path or not m_path:
            raise RuntimeError(f"Bad row in {self.csv_path}: {row}")

        # Load full arrays
        f_img = self._load_volume(f_path, is_label=False)
        m_img = self._load_volume(m_path, is_label=False)

        # Convert to torch tensors (C,D,H,W)
        fixed = torch.from_numpy(f_img).unsqueeze(0).float()
        moving = torch.from_numpy(m_img).unsqueeze(0).float()

        # Normalize
        fixed = normalize_volume(fixed)
        moving = normalize_volume(moving)

        # Extract patches
        fixed_patches, _ = extract_patches(fixed, self.patch_size, self.patch_stride)
        moving_patches, _ = extract_patches(moving, self.patch_size, self.patch_stride)

        if not fixed_patches or not moving_patches:
            raise RuntimeError("Failed to extract patches from input volumes")

        max_valid_idx = min(len(fixed_patches), len(moving_patches)) - 1
        if max_valid_idx < 0:
            raise RuntimeError("No overlapping valid patches between fixed/moving volumes")

        # Brain-coverage filter (same as OASIS)
        def _is_valid(p: torch.Tensor, frac: float = 0.10, thresh: float = 0.15) -> bool:
            if p.dtype.is_floating_point:
                return (p > thresh).float().mean().item() > frac
            else:
                return (p > 0).float().mean().item() > frac

        max_attempts = 20
        patch_idx = None
        for _ in range(max_attempts):
            cand_idx = random.randint(0, max_valid_idx)
            if _is_valid(fixed_patches[cand_idx]) and _is_valid(moving_patches[cand_idx]):
                patch_idx = cand_idx
                break

        if patch_idx is None:
            patch_idx = random.randint(0, max_valid_idx)

        fixed_patch = fixed_patches[patch_idx]
        moving_patch = moving_patches[patch_idx]

        out = {
            "fixed": fixed_patch,   # (1, S, S, S)
            "moving": moving_patch, # (1, S, S, S)
            "pair_index": row_idx,
        }

        # Augmentation (synchronized for both)
        if self.augment and self.augmentor is not None:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            torch.manual_seed(seed)
            out["fixed"] = self.augmentor.augment(fixed_patch)
            torch.manual_seed(seed)
            out["moving"] = self.augmentor.augment(moving_patch)

        # Optional segmentation
        if self.use_labels and row.get("fixed_seg") and row.get("moving_seg"):
            f_seg = self._load_volume(row["fixed_seg"], is_label=True)
            m_seg = self._load_volume(row["moving_seg"], is_label=True)
            if f_seg is not None and m_seg is not None:
                fseg_t = torch.from_numpy(f_seg).unsqueeze(0).long()
                mseg_t = torch.from_numpy(m_seg).unsqueeze(0).long()

                fseg_patches, _ = extract_patches(fseg_t, self.patch_size, self.patch_stride)
                mseg_patches, _ = extract_patches(mseg_t, self.patch_size, self.patch_stride)

                if (fseg_patches and mseg_patches and
                        patch_idx < len(fseg_patches) and patch_idx < len(mseg_patches)):
                    out["segmentation_fixed"] = fseg_patches[patch_idx]
                    out["segmentation_moving"] = mseg_patches[patch_idx]

        return out


def create_loaders_from_pairs(
    train_pairs_csv: str,
    val_pairs_csv: str,
    test_pairs_csv: Optional[str] = None,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4,
    use_labels_train: bool = True,
    use_labels_val: bool = True,
    target_size: Optional[Tuple[int,int,int]] = None, 
):
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

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = None
    if test_pairs_csv:
        test_ds = RegPairsDataset(
            test_pairs_csv,
            patch_size=patch_size,
            patch_stride=patch_stride,
            patches_per_pair=patches_per_pair,
            augment=False,
            use_labels=False,
            target_size=target_size,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

    return train_loader, val_loader, test_loader
