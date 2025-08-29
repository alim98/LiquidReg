#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make deterministic registration pairs from a data root.

- Works with:
  1) OASIS-style: <root>/OASIS_OAS1_####_MR1/{aligned_norm.nii.gz|*.nii*}
     (labels optional: {aligned_seg35.nii.gz|seg*.nii*})
  2) L2R/flat:     <root>/img####.nii.gz (labels: seg####.nii.gz)
  3) IXI-eval:     <root>/<subject_dir>/*T1*.nii[.gz]  (no labels expected)

- Output CSV columns (accepted by RegPairsDataset):
    fixed,moving,fixed_seg,moving_seg

- Deterministic sampling: for each subject i, sample up to K distinct partners j!=i
  using the given --seed.

Usage (as called by run_all.sh):
  python -u scripts/make_pairs.py --root data/OASIS_train --out data/OASIS_train/pairs_train.csv --k 5 --seed 123
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import random
import sys

# ---------------------------
# Discovery helpers
# ---------------------------

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _discover_oasis_subjects(root: Path) -> List[Dict]:
    """Discover OASIS-style subjects directly under root."""
    out = []
    for subj in sorted(root.glob("OASIS_OAS1_*_MR1")):
        try:
            sid = int(subj.name.split("_")[2])  # OASIS_OAS1_0001_MR1 -> 1
        except Exception:
            continue
        img = _first_existing([subj / "aligned_norm.nii.gz"]) or next(
            (p for p in sorted(subj.glob("*.nii*")) if p.is_file()), None
        )
        if not img:
            continue
        seg = _first_existing([subj / "aligned_seg35.nii.gz"]) or next(
            (p for p in sorted(subj.glob("seg*.nii*")) if p.is_file()), None
        )
        out.append({"id": sid, "img": img, "seg": seg})
    return out

def _discover_l2r_flat(root: Path) -> List[Dict]:
    """Discover L2R-style flat files: img####.nii.gz (+ seg####.nii.gz)."""
    out = []
    for img in sorted(root.glob("img*.nii.gz")):
        m = re.search(r"img(\d+)\.nii\.gz$", img.name)
        sid = int(m.group(1)) if m else None
        seg = (root / f"seg{sid:04d}.nii.gz") if sid is not None else None
        seg = seg if (seg and seg.exists()) else None
        out.append({"id": sid, "img": img, "seg": seg})
    return out

import os

def _is_nii_file(p: Path) -> bool:
    """Case-insensitive check for .nii or .nii.gz files."""
    if not p.is_file():
        return False
    name = p.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")

def _pick_subject_image(subj_dir: Path) -> Optional[Path]:
    """
    Recursively search inside a subject directory for a single image to use.
    Preference order:
      1) any file matching *T1* (case-insensitive), non-seg
      2) any non-seg NIfTI
    """
    t1_cands = []
    any_cands = []
    for root, _, files in os.walk(subj_dir):
        for fn in sorted(files):
            p = Path(root) / fn
            if not _is_nii_file(p):
                continue
            low = fn.lower()
            # ignore typical segmentation names
            if low.startswith("seg") or "seg" in low or "_label" in low:
                continue
            if "t1" in low:
                t1_cands.append(p)
            else:
                any_cands.append(p)
    if t1_cands:
        return t1_cands[0]
    if any_cands:
        return any_cands[0]
    return None

import re
from pathlib import Path
from typing import List, Dict, Optional

def _discover_ixi_eval(root: Path) -> List[Dict]:
    """
    Discover IXI eval images under `root` (flat folder of .nii/.nii.gz files).
    """
    out: List[Dict] = []
    for img in sorted(root.iterdir(), key=lambda p: p.name):
        if img.is_file() and (img.suffix == ".nii" or img.suffixes[-2:] == [".nii", ".gz"]):
            m = re.search(r"(\d+)", img.name)
            sid: Optional[int] = int(m.group(1)) if m else None
            out.append({"id": sid, "img": img, "seg": None})
    return out





def _discover_pool(root: Path) -> List[Dict]:
    """
    Try OASIS-style first; if none, try L2R flat; if still none, try IXI-eval style.
    Final fallback: look for any non-seg NIfTI directly under root OR 1-level down.
    """
    oasis = _discover_oasis_subjects(root)
    if oasis:
        return oasis

    l2r = _discover_l2r_flat(root)
    if l2r:
        return l2r

    ixi = _discover_ixi_eval(root)
    if ixi:
        return ixi

    # final fallback
    generic: List[Dict] = []

    # files directly under root
    for p in sorted(root.iterdir()):
        if p.is_file() and _is_nii_file(p) and not p.name.lower().startswith("seg"):
            m = re.search(r"(\d+)", p.stem)
            sid = int(m.group(1)) if m else None
            generic.append({"id": sid, "img": p, "seg": None})

    # one level of subject dirs (not deep recursion here—IXI path already handled that)
    for sub in sorted([d for d in root.iterdir() if d.is_dir()]):
        # Prefer a clearly T1 file if present, otherwise any
        img = _pick_subject_image(sub)
        if img:
            m = re.search(r"(\d+)", sub.name) or re.search(r"(\d+)", img.stem)
            sid = int(m.group(1)) if m else None
            generic.append({"id": sid, "img": img, "seg": None})

    return generic

# ---------------------------
# Pairing
# ---------------------------

def _make_pairs(pool: List[Dict], k: int, seed: int) -> List[Tuple[Dict, Dict]]:
    """
    For each element i, choose up to k distinct partners j != i.
    Deterministic given (pool order, k, seed).
    """
    rng = random.Random(seed)
    n = len(pool)
    pairs = []
    if n < 2:
        return pairs
    indices = list(range(n))
    for i in range(n):
        others = [j for j in indices if j != i]
        # deterministic shuffle via RNG seeded once
        rng.shuffle(others)
        take = min(k, len(others))
        for j in others[:take]:
            pairs.append((pool[i], pool[j]))
    return pairs

# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Create deterministic registration pairs CSV.")
    ap.add_argument("--root", required=True, help="Data root (OASIS split dir, L2R folder, or IXI eval dir)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--k", type=int, default=5, help="Partners per subject")
    ap.add_argument("--seed", type=int, default=123, help="Deterministic sampling seed")
    ap.add_argument("--labels", action="store_true", help="Try to include segmentation columns if available")
    ap.add_argument("--max_pairs", type=int, default=None,help="Maximum number of pairs to output (sampled deterministically if exceeded)")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pool = _discover_pool(root)
    if len(pool) < 2:
        print(f"ERROR: found {len(pool)} images under {root}. Need at least 2.", file=sys.stderr)
        return 2

    pairs = _make_pairs(pool, k=args.k, seed=args.seed)
    if not pairs:
        print("ERROR: no pairs produced.", file=sys.stderr)
        return 2

    # apply max_pairs restriction
    if args.max_pairs is not None and len(pairs) > args.max_pairs:
        rng = random.Random(args.seed)   # keep deterministic
        pairs = rng.sample(pairs, args.max_pairs)

    # Write CSV compatible with RegPairsDataset
    fieldnames = ["fixed", "moving", "fixed_seg", "moving_seg"]
    rows = 0
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a, b in pairs:
            row = {
                "fixed": str(a["img"]),
                "moving": str(b["img"]),
                "fixed_seg": "",
                "moving_seg": "",
            }
            if args.labels:
                fa, fb = a.get("seg"), b.get("seg")
                if fa is not None and fb is not None:
                    row["fixed_seg"] = str(fa)
                    row["moving_seg"] = str(fb)
            w.writerow(row)
            rows += 1

    print(f"[make_pairs] root={root}  n_subjects={len(pool)}  k={args.k}  seed={args.seed}")
    print(f"[make_pairs] wrote {rows} rows → {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
