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


def _read_pairs_csv(csv_path: Path) -> List[Tuple[int, int]]:
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
    # print(f"[DEBUG] _read_pairs_csv: read {len(pairs)} pairs from '{csv_path}'.")
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
        fixed_pairs: bool = False,
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

        
        self.volumes = self._discover_volumes()
        # print(f"[DEBUG] Split '{self.split}': discovered {len(self.volumes)} volumes in '{self.data_dir}'.")
        # if self.volumes:
        #     print(f"[DEBUG] First 5 volume IDs: {[v['id'] for v in self.volumes[:5]]} ...")

        if len(self.volumes) == 0:
            # If no volumes found, print diagnostic information
            candidate_imgs = list(self.data_dir.glob("img*.nii.gz"))
            # print(f"[DEBUG] glob('img*.nii.gz') returned {len(candidate_imgs)} matches")
            # if candidate_imgs:
            #     print(f"[DEBUG] Example files: {[p.name for p in candidate_imgs[:5]]}")

        
        if self.split == "val" and self.pairs_csv is not None:
            self.pairs = self._pairs_from_csv()
            # print(f"[DEBUG] Attempting to read validation pairs from CSV: {self.pairs_csv}")
            if len(self.pairs) == 0:
                # print("[DEBUG] CSV yielded 0 valid pairs -> falling back to random generation")
                self.pairs = self._generate_pairs(len(self.volumes) * 2)
            # else:
                # print(f"[DEBUG] Loaded {len(self.pairs)} pairs from CSV")
        else:
            self.pairs = self._generate_pairs(len(self.volumes) * 2)
            # print(f"[DEBUG] Generated {len(self.pairs)} random pairs for split '{self.split}'.")

    
    def _discover_volumes(self) -> List[dict]:
        volumes = []
        if self.split == "train":
            for subj_dir in sorted(self.data_dir.glob("OASIS_OAS1_*_MR1")):
                try:
                    sid = int(subj_dir.name.split("_")[2])
                except (IndexError, ValueError):
                    continue

                img = subj_dir / "aligned_norm.nii.gz"
                if not img.exists():
                    nii_files = list(subj_dir.glob("*.nii.gz"))
                    img = nii_files[0] if nii_files else None

                label = subj_dir / "aligned_seg35.nii.gz"
                if self.use_labels and not label.exists():
                    seg_files = list(subj_dir.glob("seg*.nii.gz"))
                    label = seg_files[0] if seg_files else None

                volumes.append({"id": sid, "image": img, "label": label})
        elif self.split in {"val", "test"}:
            for img_path in sorted(self.data_dir.glob("img*.nii.gz")):
                # img_path is '.../imgXXXX.nii.gz'.  Path.stem on a double
                # extension returns 'imgXXXX.nii', so we need to strip the
                # optional '.nii' as well before casting to int.
                try:
                    sid_str = img_path.stem  # 'img0438.nii'
                    sid_str = sid_str.replace("img", "")  # '0438.nii'
                    if sid_str.endswith(".nii"):
                        sid_str = sid_str[:-4]  # drop '.nii'
                    sid = int(sid_str)
                except ValueError:
                    # Skip files with unexpected names
                    continue
                label_path = self.data_dir / f"seg{sid:04d}.nii.gz"
                label = label_path if label_path.exists() else None
                volumes.append({"id": sid, "image": img_path, "label": label})
        else:
            raise ValueError(f"Unknown split: {self.split}")

        return volumes

    
    def _pairs_from_csv(self) -> List[Tuple[int, int]]:
        csv_pairs = _read_pairs_csv(self.pairs_csv)
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
        img = nib.load(str(info["image"])).get_fdata().astype(np.float32)
        label = None
        if self.use_labels and info.get("label") and Path(info["label"]).exists():
            label = nib.load(str(info["label"])).get_fdata().astype(np.int32)
        return {"image": img, "label": label}

    
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
            raise RuntimeError("Failed to extract patches from volume")

        
        # ------------------------------------------------------------------
        # Brain-coverage filter: keep sampling until a patch contains a minimum
        # fraction of voxels above an intensity threshold (i.e. likely tissue).
        # This avoids wasting training iterations on empty background patches.
        # ------------------------------------------------------------------

        def _is_valid(p: torch.Tensor, frac: float = 0.10, thresh: float = 0.15) -> bool:
            """Return True if at least *frac* voxels are > *thresh*."""
            try:
                if p.dtype in [torch.long, torch.int32, torch.int64, torch.int, torch.int8, torch.int16]:
                    # For integer data (segmentation data), check if there are enough non-zero voxels
                    return (p > 0).float().mean().item() > frac
                else:
                    # For floating point data (intensity data), use threshold
                    return (p > thresh).float().mean().item() > frac
            except Exception as e:
                # Fallback: consider patch valid
                print(f"Warning: Error in patch validation ({e}), considering patch valid")
                return True

        max_attempts = 20
        patch_idx = None
        for _ in range(max_attempts):
            cand_idx = random.randint(0, len(fixed_patches) - 1)
            if _is_valid(fixed_patches[cand_idx]):
                patch_idx = cand_idx
                break

        if patch_idx is None:  # fallback if no suitable patch found
            patch_idx = random.randint(0, len(fixed_patches) - 1)

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
            torch.manual_seed(seed)
            output["fixed"] = self.augmentor.augment(fixed_patch)  # Keep channel dimension
            
            torch.manual_seed(seed)  # Reset to same seed
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

            if fixed_label_patches and patch_idx < len(fixed_label_patches):
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
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    val_dir = root / "L2R_2021_Task3_val"

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
