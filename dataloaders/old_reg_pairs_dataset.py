# file: dataloaders/reg_pairs_dataset.py
import csv, os, random
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

def _robust_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (1, D, H, W) float32
    v = x.flatten()
    lo = torch.quantile(v, 0.005)
    hi = torch.quantile(v, 0.995)
    x = torch.clamp(x, lo, hi)
    m = x.mean()
    s = x.std().clamp_min(eps)
    return (x - m) / s

def _load_nii(path: str) -> torch.Tensor:
    # mmap=True keeps memory sane; convert to contiguous float32 tensor
    img = nib.load(str(path), mmap=True).get_fdata(dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0)  # (1,D,H,W)
    return t

def _crop_at(t: torch.Tensor, zyx: Tuple[int,int,int], size: int) -> torch.Tensor:
    # t: (1,D,H,W); zyx is center
    _, D,H,W = t.shape
    zc,yc,xc = zyx
    r = size // 2
    z0, z1 = max(0, zc - r), min(D, zc + r)
    y0, y1 = max(0, yc - r), min(H, yc + r)
    x0, x1 = max(0, xc - r), min(W, xc + r)
    patch = t[:, z0:z1, y0:y1, x0:x1]
    # pad to exact size if near borders
    pad_z = size - patch.shape[1]
    pad_y = size - patch.shape[2]
    pad_x = size - patch.shape[3]
    if pad_z or pad_y or pad_x:
        patch = torch.nn.functional.pad(
            patch,
            (0, pad_x, 0, pad_y, 0, pad_z),  # (W_left,W_right,H_top,H_bottom,D_front,D_back)
            mode="constant", value=0.0
        )
    return patch

def _pick_shared_center(fixed: torch.Tensor, moving: torch.Tensor, size: int) -> Tuple[int,int,int]:
    # choose a center that’s valid in both volumes; prefer brainy regions
    _, Df,Hf,Wf = fixed.shape
    _, Dm,Hm,Wm = moving.shape
    D,H,W = min(Df,Dm), min(Hf,Hm), min(Wf,Wm)
    r = size // 2
    zmin,ymin,xmin = r, r, r
    zmax,ymax,xmax = D - r - 1, H - r - 1, W - r - 1
    zmax, ymax, xmax = max(zmin, zmax), max(ymin, ymax), max(xmin, xmax)

    def is_brain(p: torch.Tensor, thresh=0.10, frac=0.50):
        return (p > thresh).float().mean().item() > frac

    for _ in range(64):
        z = random.randint(zmin, zmax) if zmax >= zmin else r
        y = random.randint(ymin, ymax) if ymax >= ymin else r
        x = random.randint(xmin, xmax) if xmax >= xmin else r
        f_patch = _crop_at(fixed, (z,y,x), size)
        m_patch = _crop_at(moving, (z,y,x), size)
        if is_brain(f_patch) and is_brain(m_patch):
            return (z,y,x)
    # fallback: center
    return (D//2, H//2, W//2)

def _read_pairs_csv(csv_path: Path) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        # expected columns: fixed, moving
        # optional: fixed_seg, moving_seg
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
    Works for OASIS/L2R/IXI uniformly, since it doesn’t assume any folder pattern.
    """
    def __init__(
        self,
        pairs_csv: str,
        patch_size: int = 64,
        patches_per_pair: int = 20,
        augment: bool = False,
        use_labels: bool = False,
        seed: int = 42,
    ):
        self.csv_path = Path(pairs_csv)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Pairs CSV not found: {pairs_csv}")
        self.rows = _read_pairs_csv(self.csv_path)
        if len(self.rows) == 0:
            raise RuntimeError(f"No pairs in CSV: {pairs_csv}")
        self.patch_size = patch_size
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        random.seed(seed)
        np.random.seed(seed)

        # optional: plug your own augmentor here
        self.augmentor = None  # e.g., your SpatialAugmentor(...)

    def __len__(self) -> int:
        return len(self.rows) * self.patches_per_pair

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row_idx = idx // self.patches_per_pair
        row = self.rows[row_idx]
        f_path = row["fixed"]; m_path = row["moving"]
        if not f_path or not m_path:
            raise RuntimeError(f"Bad row in {self.csv_path}: {row}")

        fixed = _load_nii(f_path).float()
        moving = _load_nii(m_path).float()
        fixed = _robust_norm(fixed)
        moving = _robust_norm(moving)

        # choose a shared crop location, then crop both
        center = _pick_shared_center(fixed, moving, self.patch_size)
        fixed_patch  = _crop_at(fixed,  center, self.patch_size)
        moving_patch = _crop_at(moving, center, self.patch_size)

        if self.augment and self.augmentor is not None:
            # ensure same spatial aug for both
            seed = torch.randint(0, 1_000_000, (1,)).item()
            torch.manual_seed(seed)
            fixed_patch  = self.augmentor(fixed_patch)
            torch.manual_seed(seed)
            moving_patch = self.augmentor(moving_patch)

        out = {
            "fixed": fixed_patch.contiguous(),   # (1, S, S, S)
            "moving": moving_patch.contiguous(), # (1, S, S, S)
            "pair_index": row_idx,
        }

        if self.use_labels and row.get("fixed_seg") and row.get("moving_seg"):
            if os.path.exists(str(row["fixed_seg"])) and os.path.exists(str(row["moving_seg"])):
                fseg = torch.from_numpy(nib.load(row["fixed_seg"]).get_fdata().astype(np.int16)).unsqueeze(0)
                mseg = torch.from_numpy(nib.load(row["moving_seg"]).get_fdata().astype(np.int16)).unsqueeze(0)
                # crop labels at the exact same center
                fseg_patch = _crop_at(fseg, center, self.patch_size).long()
                mseg_patch = _crop_at(mseg, center, self.patch_size).long()
                out["segmentation_fixed"] = fseg_patch
                out["segmentation_moving"] = mseg_patch

        return out

def create_loaders_from_pairs(
    train_pairs_csv: str,
    val_pairs_csv: str,
    test_pairs_csv: Optional[str] = None,
    batch_size: int = 4,
    patch_size: int = 64,
    patches_per_pair: int = 20,
    num_workers: int = 4,
    use_labels_train: bool = True,
    use_labels_val: bool = True,
):
    train_ds = RegPairsDataset(
        train_pairs_csv,
        patch_size=patch_size,
        patches_per_pair=patches_per_pair,
        augment=True,
        use_labels=use_labels_train,
    )
    val_ds = RegPairsDataset(
        val_pairs_csv,
        patch_size=patch_size,
        patches_per_pair=patches_per_pair,
        augment=False,
        use_labels=use_labels_val,
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
            patches_per_pair=patches_per_pair,
            augment=False,
            use_labels=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
    return train_loader, val_loader, test_loader
