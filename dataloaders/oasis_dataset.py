import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from utils.preprocessing import normalize_volume
from utils.patch_utils import extract_patches, PatchAugmentor


def _read_pairs_csv_ids(csv_path: Path) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if not csv_path or not csv_path.exists():
        return pairs

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            try:
                fid = int(str(row["fixed"]).strip())
                mid = int(str(row["moving"]).strip())
                pairs.append((fid, mid))
            except (KeyError, ValueError, TypeError):
                continue

    return pairs


class L2RTask3Dataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_size: int = 64,
        patch_stride: int = 32,
        patches_per_pair: int = 20,
        augment: bool = False,
        use_labels: bool = True,
        pairs_csv: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patches_per_pair = patches_per_pair
        self.augment = augment
        self.use_labels = use_labels
        self.pairs_csv = Path(pairs_csv) if pairs_csv else None
        
        random.seed(seed)
        np.random.seed(seed)
        
        if self.augment:
            self.augmentor = PatchAugmentor()
        else:
            self.augmentor = None

        from collections import OrderedDict
        self._volume_cache = OrderedDict()
        self._volume_cache_max = 32
        
        self.volumes = self._discover_volumes()

        if len(self.volumes) == 0:
            # If no volumes found, print diagnostic information
            candidate_imgs = list(self.data_dir.glob("img*.nii.gz"))

        
        if self.split == "val" and self.pairs_csv is not None:
            self.pairs = self._pairs_from_csv()
            if len(self.pairs) == 0:
                self.pairs = self._generate_pairs(len(self.volumes) * 2)
        else:
            self.pairs = self._generate_pairs(len(self.volumes) * 2)

    
    def _discover_volumes(self) -> List[dict]:
        volumes: List[dict] = []

        def add_oasis_subject(subj_dir: Path) -> None:
            # subj_dir: .../OASIS_OAS1_####_MR1
            try:
                sid = int(subj_dir.name.split("_")[2])  # '0001' -> 1
            except (IndexError, ValueError):
                return

            # prefer aligned_norm.nii.gz, else first *.nii*
            img = subj_dir / "aligned_norm.nii.gz"
            if not img.exists():
                nii_files = sorted(subj_dir.glob("*.nii*"))
                if not nii_files:
                    return  # skip subjects with no image
                img = nii_files[0]

            label = None
            if self.use_labels:
                label = subj_dir / "aligned_seg35.nii.gz"
                if not label.exists():
                    seg_files = sorted(subj_dir.glob("seg*.nii*"))
                    label = seg_files[0] if seg_files else None

            volumes.append({"id": sid, "image": img, "label": label})

        if self.split == "train":
            for subj_dir in sorted(self.data_dir.glob("OASIS_OAS1_*_MR1")):
                add_oasis_subject(subj_dir)

        elif self.split in {"val", "test"}:
            # Prefer OASIS-style folders if present...
            subj_dirs = sorted(self.data_dir.glob("OASIS_OAS1_*_MR1"))
            if subj_dirs:
                for subj_dir in subj_dirs:
                    add_oasis_subject(subj_dir)
            else:
                # ...otherwise fall back to L2R-style flat files (img*.nii.gz)
                for img_path in sorted(self.data_dir.glob("img*.nii.gz")):
                    # Path.stem for .nii.gz -> 'img0438.nii' -> strip '.nii'
                    sid_str = img_path.stem.replace("img", "")
                    if sid_str.endswith(".nii"):
                        sid_str = sid_str[:-4]
                    try:
                        sid = int(sid_str)
                    except ValueError:
                        continue
                    label = None
                    if self.use_labels:
                        lp = self.data_dir / f"seg{sid:04d}.nii.gz"
                        label = lp if lp.exists() else None
                    volumes.append({"id": sid, "image": img_path, "label": label})

        else:
            raise ValueError(f"Unknown split: {self.split}")

        return volumes

    
    def _pairs_from_csv(self) -> List[Tuple[int, int]]:
        csv_pairs = _read_pairs_csv_ids(self.pairs_csv)
        id_to_idx = {v["id"]: i for i, v in enumerate(self.volumes)}
        pairs: List[Tuple[int, int]] = []
        for fid, mid in csv_pairs:
            if fid in id_to_idx and mid in id_to_idx:
                pairs.append((id_to_idx[fid], id_to_idx[mid]))
        return pairs
    
    
    def _generate_pairs(self, num_pairs: int) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        n = len(self.volumes)
        for _ in range(num_pairs):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while j == i:
                j = random.randint(0, n - 1)
            pairs.append((i, j))
        return pairs

    
    def _load_volume(self, info: dict) -> dict:
        img_path = str(info["image"])
        label_path = str(info["label"]) if info.get("label") else None
        
        cache_key = (img_path, label_path)
        if cache_key in self._volume_cache:
            cached = self._volume_cache.pop(cache_key)
            self._volume_cache[cache_key] = cached
            return cached
        
        img = nib.load(img_path).get_fdata().astype(np.float32)
        label = None
        if self.use_labels and label_path and Path(label_path).exists():
            label = nib.load(label_path).get_fdata().astype(np.int32)
        
        result = {"image": img, "label": label}
        self._volume_cache[cache_key] = result
        if len(self._volume_cache) > self._volume_cache_max:
            self._volume_cache.popitem(last=False)
        return result

    
    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_pair

    
    def __getitem__(self, idx: int) -> dict:
        pair_idx = idx // self.patches_per_pair
        idx1, idx2 = self.pairs[pair_idx]

        vol1 = self._load_volume(self.volumes[idx1])
        vol2 = self._load_volume(self.volumes[idx2])
            
        fixed = torch.from_numpy(vol1["image"]).float().unsqueeze(0).unsqueeze(0)
        moving = torch.from_numpy(vol2["image"]).float().unsqueeze(0).unsqueeze(0)
        
        fixed = normalize_volume(fixed)
        moving = normalize_volume(moving)
            
        
        fixed_patches, _ = extract_patches(
            fixed,
            self.patch_size,
            self.patch_stride,
        )
        moving_patches, _ = extract_patches(
            moving,
            self.patch_size,
            self.patch_stride,
        )

        if not fixed_patches:
            raise RuntimeError("Failed to extract patches from fixed volume")
            
        if not moving_patches:
            raise RuntimeError("Failed to extract patches from moving volume")
        
        # Ensure we only consider indices that are valid for both patch lists
        max_valid_idx = min(len(fixed_patches), len(moving_patches)) - 1
        if max_valid_idx < 0:
            raise RuntimeError("No valid patches found in both volumes")
        
        # ------------------------------------------------------------------
        # Brain-coverage filter: keep sampling until a patch contains a minimum
        # fraction of voxels above an intensity threshold (i.e. likely tissue).
        # This avoids wasting training iterations on empty background patches.
        # ------------------------------------------------------------------

        def _is_valid(p: torch.Tensor) -> bool:
            try:
                if p.dtype in [torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16]:
                    return (p > 0).float().mean().item() > 0.10
                else:
                    pm = p.float()
                    p_flat = pm.flatten()
                    if p_flat.numel() == 0:
                        return False
                    p10 = torch.quantile(p_flat, 0.10).item()
                    p90 = torch.quantile(p_flat, 0.90).item()
                    return (p90 - p10) > 1e-6
            except Exception as e:
                print(f"Warning: Error in patch validation ({e}), considering patch valid")
                return True

        patch_offset = idx % self.patches_per_pair
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
            
        # Prepare output dictionary - keep channel dimension
        output = {
            "fixed": fixed_patch,  # Keep as (1, 64, 64, 64)
            "moving": moving_patch,  # Keep as (1, 64, 64, 64)
            "pair_idx": pair_idx,
        }

        # Optional data augmentation (train split only)
        if self.augment and self.augmentor is not None:
            # Apply the same augmentation to both images to maintain correspondence
            # First, generate random augmentation parameters
            import torch.nn.functional as F
            
            # Apply same spatial transformations to both patches
            seed = torch.randint(0, 1000000, (1,)).item()
            
            # Set same random seed for both augmentations to ensure same transforms
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            output["fixed"] = self.augmentor.augment(fixed_patch)  # Keep channel dimension
            
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            output["moving"] = self.augmentor.augment(moving_patch)  # Keep channel dimension

        # Attach segmentation labels if available
        if self.use_labels and vol1["label"] is not None and vol2["label"] is not None:
            fixed_label = torch.from_numpy(vol1["label"]).long().unsqueeze(0).unsqueeze(0)
            moving_label = torch.from_numpy(vol2["label"]).long().unsqueeze(0).unsqueeze(0)

            fixed_label_patches, _ = extract_patches(
                fixed_label,
                self.patch_size,
                self.patch_stride,
            )
            moving_label_patches, _ = extract_patches(
                moving_label,
                self.patch_size,
                self.patch_stride,
            )

            # Only add segmentation if we have valid patches for both and the patch_idx is in range
            if (fixed_label_patches and moving_label_patches and 
                patch_idx < len(fixed_label_patches) and patch_idx < len(moving_label_patches)):
                fixed_label_patch = fixed_label_patches[patch_idx]
                moving_label_patch = moving_label_patches[patch_idx]
                output["segmentation_fixed"] = fixed_label_patch  # Keep channel dimension
                output["segmentation_moving"] = moving_label_patch  # Keep channel dimension

        return output




def create_task3_loaders(
    train_dir: str,
    val_dir: str,
    test_dir: Optional[str] = None,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:

    pairs_csv = Path(train_dir) / "pairs_val.csv"
    pairs_csv = pairs_csv if pairs_csv.exists() else None

    train_ds = L2RTask3Dataset(
        train_dir,
        split="train",
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=True,
        use_labels=True,
    )
    
    val_ds = L2RTask3Dataset(
        val_dir,
        split="val",
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        augment=False,
        use_labels=True,
        pairs_csv=str(pairs_csv) if pairs_csv else None,
    )
    
    def _worker_init_fn(worker_id: int):
        base_seed = 42
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    test_loader = None
    if test_dir is not None:
        test_ds = L2RTask3Dataset(
            test_dir,
            split="test",
            patch_size=patch_size,
            patch_stride=patch_stride,
            patches_per_pair=patches_per_pair,
            augment=False,
            use_labels=False,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )

    return train_loader, val_loader, test_loader








OASISDataset = L2RTask3Dataset



def create_oasis_loaders(
    data_root: str,
    batch_size: int = 4,
    patch_size: int = 64,
    patch_stride: int = 32,
    patches_per_pair: int = 20,
    num_workers: int = 4,
):

    root = Path(data_root)
    train_dir = root / "L2R_2021_Task3_train"
    # val_dir = root / "L2R_2021_Task3_val"

    train_dir = root / "OASIS_train"
    val_dir   = root / "OASIS_val"


    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train directory {train_dir} not found.")
    if not val_dir.exists():
        raise FileNotFoundError(f"Expected val directory {val_dir} not found.")

    train_loader, val_loader, _ = create_task3_loaders(
        str(train_dir),
        str(val_dir),
        test_dir=None,
        batch_size=batch_size,
        patch_size=patch_size,
        patch_stride=patch_stride,
        patches_per_pair=patches_per_pair,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader
