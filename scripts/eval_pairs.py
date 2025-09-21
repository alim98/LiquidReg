#!/usr/bin/env python3
"""
Batch evaluator over a pairs CSV for LiquidReg.

Outputs (by default):
- metrics.csv: per pair metrics (NCC, SSIM, Dice_macro, HD95_macro, foldings%)
- summary.txt: averages ± std
- TensorBoard logs: <out>/tensorboard
- Optional warped images / flows / Jacobians

Disable features with flags: --no_dice, --no_ssim, --no_hd95, --no_save_jacobian
"""

import os, csv, argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from medpy.metric.binary import hd95
from torch.utils.tensorboard import SummaryWriter

# Make repo root importable when running from scripts/
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
sys.path.append(str(REPO))

from scripts.inference import load_model, load_nifti_volume, preprocess_volume, save_nifti_volume
from models.scaling_squaring import SpatialTransformer
from utils.preprocessing import resample_volume
from models.scaling_squaring import compute_jacobian_determinant


# ------------------------- Metrics helpers -------------------------

def ncc_metric(fixed: torch.Tensor, warped: torch.Tensor, eps: float = 1e-6) -> float:
    f = fixed.flatten().float()
    w = warped.flatten().float()
    f = (f - f.mean()) / (f.std() + eps)
    w = (w - w.mean()) / (w.std() + eps)
    return torch.clamp((f * w).mean(), -1.0, 1.0).item()


def ssim_volume_mean(fixed_np: np.ndarray, warped_np: np.ndarray) -> float:
    """
    Mean 2D SSIM over z-slices (robust & simple). Inputs are (D,H,W) numpy arrays.
    """
    assert fixed_np.ndim == 3 and warped_np.ndim == 3
    D = fixed_np.shape[0]
    dr = max(warped_np.max() - warped_np.min(), 1e-6)
    vals = [ssim(fixed_np[z], warped_np[z], data_range=dr) for z in range(D)]
    return float(np.mean(vals))


def dice_per_label(fseg: torch.Tensor, mseg_warp: torch.Tensor, labels=None) -> dict:
    f = fseg.long().cpu().numpy()
    m = mseg_warp.long().cpu().numpy()
    if labels is None:
        labels = sorted(list(set(np.unique(f)) | set(np.unique(m))))
        labels = [int(l) for l in labels if int(l) != 0]
    out = {}
    for lab in labels:
        A = (f == lab)
        B = (m == lab)
        denom = A.sum() + B.sum()
        out[lab] = np.nan if denom == 0 else 2.0 * (A & B).sum() / denom
    return out


def hd95_per_label(fseg: torch.Tensor, mseg_warp: torch.Tensor, labels=None) -> dict:
    f = fseg.long().cpu().numpy()
    m = mseg_warp.long().cpu().numpy()
    if labels is None:
        labels = sorted(list(set(np.unique(f)) | set(np.unique(m))))
        labels = [int(l) for l in labels if int(l) != 0]
    out = {}
    for lab in labels:
        A = (f == lab)
        B = (m == lab)
        if A.sum() == 0 or B.sum() == 0:
            out[lab] = np.nan
        else:
            # Note: hd95 is in voxel units of the (resampled) grid.
            out[lab] = float(hd95(A, B))
    return out


def detect_dataset(fixed_path: str, moving_path: str) -> str:
    s = f"{fixed_path} {moving_path}".lower()
    if "oasis" in s:
        return "OASIS"
    if "ixi" in s:
        return "IXI"
    if "l2r" in s or "learn2reg" in s:
        return "L2R"
    return "UNKNOWN"


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="CSV with columns fixed,moving[,fixed_seg,moving_seg]")
    ap.add_argument("--model", required=True, help="Path to checkpoint .pth")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--save_warped", action="store_true", help="Save warped moving images")
    ap.add_argument("--save_flow", action="store_true", help="Save deformation fields as NIfTI (dx/dy/dz)")
    ap.add_argument("--no_dice", action="store_true", help="Disable Dice computation")
    ap.add_argument("--no_ssim", action="store_true", help="Disable SSIM computation")
    ap.add_argument("--no_hd95", action="store_true", help="Disable HD95 computation")
    ap.add_argument("--no_save_jacobian", action="store_true", help="Disable saving Jacobian determinant")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(out_dir / "tensorboard")
    device = torch.device(args.device)

    # Load model once
    model, cfg = load_model(args.model, device)
    model_img_size = tuple(cfg["model"]["image_size"])
    model.eval()

    # Transformer for labels (nearest)
    st_nearest = model.spatial_transformer.__class__(
        mode="nearest",
        padding_mode=model.spatial_transformer.padding_mode,
        align_corners=model.spatial_transformer.align_corners
    ).to(device)

    # Read pairs
    with open(args.pairs, "r") as f:
        rows = list(csv.DictReader(f))

    results_rows = []
    dataset_counts = {"OASIS": 0, "IXI": 0, "L2R": 0, "UNKNOWN": 0}

    for i, row in enumerate(tqdm(rows, desc="Evaluating pairs")):
        fixed_path, moving_path = row.get("fixed"), row.get("moving")
        if not fixed_path or not moving_path:
            print(f"[{i}] Skipping row with missing paths: {row}"); continue

        dataset = detect_dataset(fixed_path, moving_path)
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1

        pair_out = out_dir / f"pair_{i:05d}"; pair_out.mkdir(exist_ok=True)

        # Load & preprocess
        fixed_vol, fixed_aff, fixed_hdr = load_nifti_volume(fixed_path)
        moving_vol, moving_aff, moving_hdr = load_nifti_volume(moving_path)
        fixed_pre  = preprocess_volume(fixed_vol,  target_size=model_img_size).to(device)
        moving_pre = preprocess_volume(moving_vol, target_size=model_img_size).to(device)

        # Inference
        start = time.time()
        with torch.no_grad():
            out = model(fixed_pre, moving_pre, return_intermediate=True)
        elapsed = time.time() - start

        warped = out["warped_moving"]     # (1,1,D,H,W)
        flow   = out["deformation_field"] # (1,3,D,H,W)

        # --- spacing of the EVAL GRID (use FIXED image as reference) ---
        # original NIfTI zooms are (sz, sy, sx) for the *original* fixed volume
        f_sz, f_sy, f_sx = tuple(map(float, fixed_hdr.get_zooms()[:3]))
        oz, oh, ow = fixed_vol.shape[:3]
        nz, nh, nw = model_img_size  # you resampled to this size before inference
        # preserve physical size ⇒ spacing scales with shape change
        sz = f_sz * oz / nz
        sy = f_sy * oh / nh
        sx = f_sx * ow / nw
        spacing = flow.new_tensor((sz, sy, sx))

        # spacing-aware Jacobian on the resampled eval grid
        jac = compute_jacobian_determinant(flow, spacing)  # (1,1,D-2,H-2,W-2)
        fold_pct = float((jac <= 0).float().mean().item() * 100.0)

        # Metrics
        ncc = ncc_metric(fixed_pre, warped)

        f = fixed_pre.squeeze().cpu().numpy()   # (D,H,W)
        w = warped.squeeze().cpu().numpy()      # (D,H,W)
        ssim_val = ssim_volume_mean(f, w) if not args.no_ssim else np.nan

        # Dice/HD95 only for OASIS (has labels)
        fseg_path, mseg_path = row.get("fixed_seg"), row.get("moving_seg")
        labeled_pair = (
            dataset == "OASIS" and
            fseg_path and mseg_path and os.path.exists(fseg_path) and os.path.exists(mseg_path)
        )

        # Initialize as blanks for CSV (so IXI/L2R don’t show NaNs)
        dice_macro_csv = ""
        hd95_macro_csv = ""
        num_labels_csv = ""

        if labeled_pair:
            fseg_np = nib.load(fseg_path).get_fdata().astype(np.int16)
            mseg_np = nib.load(mseg_path).get_fdata().astype(np.int16)
            fseg = torch.from_numpy(fseg_np).unsqueeze(0).unsqueeze(0).to(device)
            mseg = torch.from_numpy(mseg_np).unsqueeze(0).unsqueeze(0).to(device)
            fseg_rs = resample_volume(fseg.float(), model_img_size, mode="nearest").long()
            mseg_rs = resample_volume(mseg.float(), model_img_size, mode="nearest").long()
            mseg_warp = st_nearest(mseg_rs.float(), flow).round().long()

            if not args.no_dice:
                dice_map = dice_per_label(fseg_rs.squeeze(0), mseg_warp.squeeze(0))
                vals = [v for v in dice_map.values() if not np.isnan(v)]
                if len(vals) > 0:
                    dice_macro_csv = f"{float(np.mean(vals)):.6f}"
                num_labels_csv = str(len(dice_map))

            if not args.no_hd95:
                hd95_map = hd95_per_label(fseg_rs.squeeze(0), mseg_warp.squeeze(0))
                vals = [v for v in hd95_map.values() if not np.isnan(v)]
                if len(vals) > 0:
                    hd95_macro_csv = f"{float(np.mean(vals)):.6f}"

        # Optional saves
        if args.save_warped:
            save_nifti_volume(warped.cpu(), str(pair_out / "warped_moving.nii.gz"), moving_aff, moving_hdr)
        if args.save_flow:
            flow_cpu = flow.cpu()
            for c, name in enumerate(["dx","dy","dz"]):
                save_nifti_volume(flow_cpu[:, c:c+1], str(pair_out / f"flow_{name}.nii.gz"), moving_aff, moving_hdr)
        if not args.no_save_jacobian:
            # jac is (B,1,D-2,H-2,W-2); save as-is in the fixed-image space
            save_nifti_volume(jac.cpu(), str(pair_out / "jacobian_det.nii.gz"), fixed_aff, fixed_hdr)

        results_rows.append({
            "idx": i,
            "dataset": dataset,
            "fixed": fixed_path,
            "moving": moving_path,
            "NCC": ncc,
            "SSIM": ssim_val if not args.no_ssim else "",
            "foldings_percent": fold_pct,
            "Dice_macro": dice_macro_csv,     # blank for non-OASIS
            "HD95_macro": hd95_macro_csv,     # blank for non-OASIS
            "num_labels": num_labels_csv,     # blank for non-OASIS
            "time_sec": elapsed,
        })

        # TensorBoard
        tb_writer.add_scalar("NCC", ncc, i)
        if not args.no_ssim:
            tb_writer.add_scalar("SSIM", ssim_val, i)
        tb_writer.add_scalar("Foldings_percent", fold_pct, i)
        if labeled_pair and not args.no_dice and dice_macro_csv != "":
            tb_writer.add_scalar("Dice_macro", float(dice_macro_csv), i)
        if labeled_pair and not args.no_hd95 and hd95_macro_csv != "":
            tb_writer.add_scalar("HD95_macro", float(hd95_macro_csv), i)
        tb_writer.add_scalar("Inference_time_sec", elapsed, i)

        # Cleanup
        del fixed_vol, moving_vol, fixed_pre, moving_pre, warped, flow, jac
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Write per-pair metrics
    out_csv = out_dir / "metrics.csv"
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["idx","dataset","fixed","moving","NCC","SSIM","foldings_percent","Dice_macro","HD95_macro","num_labels","time_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(results_rows)

    # Summary
    def mean(xs): return (float(np.mean(xs)), float(np.std(xs))) if xs else (np.nan, np.nan)

    ncc_vals  = [r["NCC"] for r in results_rows if r["NCC"] is not None and not np.isnan(r["NCC"])]
    ssim_vals = [r["SSIM"] for r in results_rows if isinstance(r["SSIM"], (int, float)) and not np.isnan(r["SSIM"])]
    fold_vals = [r["foldings_percent"] for r in results_rows if r["foldings_percent"] is not None and not np.isnan(r["foldings_percent"])]

    # Dice/HD95 only over OASIS rows that have numeric strings
    dice_vals = [float(r["Dice_macro"]) for r in results_rows if r["dataset"] == "OASIS" and isinstance(r["Dice_macro"], str) and r["Dice_macro"] != ""]
    hd95_vals = [float(r["HD95_macro"]) for r in results_rows if r["dataset"] == "OASIS" and isinstance(r["HD95_macro"], str) and r["HD95_macro"] != ""]

    ncc_m, ncc_s   = mean(ncc_vals)
    ssim_m, ssim_s = mean(ssim_vals)
    fold_m, fold_s = mean(fold_vals)
    dice_m, dice_s = mean(dice_vals)
    hd95_m, hd95_s = mean(hd95_vals)

    # Clear, explicit summary header + dataset breakdown
    with open(out_dir / "summary.txt", "w") as f:
        f.write("==== LiquidReg Evaluation Summary ====\n")
        f.write(f"Pairs total: {len(results_rows)}  |  OASIS: {dataset_counts.get('OASIS',0)}  "
                f"IXI: {dataset_counts.get('IXI',0)}  L2R: {dataset_counts.get('L2R',0)}  "
                f"UNKNOWN: {dataset_counts.get('UNKNOWN',0)}\n")
        f.write("(Dice/HD95 computed only for OASIS pairs)\n\n")
        f.write(f"NCC: {ncc_m:.4f} ± {ncc_s:.4f}\n")
        if not args.no_ssim:
            f.write(f"SSIM: {ssim_m:.4f} ± {ssim_s:.4f}\n")
        f.write(f"Foldings %: {fold_m:.3f} ± {fold_s:.3f}\n")
        if not args.no_dice:
            f.write(f"Dice (macro, OASIS only): {dice_m:.4f} ± {dice_s:.4f}\n")
        if not args.no_hd95:
            f.write(f"HD95 (macro, OASIS only): {hd95_m:.4f} ± {hd95_s:.4f}\n")

    print("==== LiquidReg Evaluation Summary ====")
    print(f"Pairs total: {len(results_rows)}  |  OASIS: {dataset_counts.get('OASIS',0)}  "
          f"IXI: {dataset_counts.get('IXI',0)}  L2R: {dataset_counts.get('L2R',0)}  "
          f"UNKNOWN: {dataset_counts.get('UNKNOWN',0)}")
    print("(Dice/HD95 computed only for OASIS pairs)")
    print(f"NCC: {ncc_m:.4f} ± {ncc_s:.4f}")
    if not args.no_ssim:
        print(f"SSIM: {ssim_m:.4f} ± {ssim_s:.4f}")
    print(f"Foldings %: {fold_m:.3f} ± {fold_s:.3f}")
    if not args.no_dice:
        print(f"Dice (macro, OASIS only): {dice_m:.4f} ± {dice_s:.4f}")
    if not args.no_hd95:
        print(f"HD95 (macro, OASIS only): {hd95_m:.4f} ± {hd95_s:.4f}")

    tb_writer.close()


if __name__ == "__main__":
    main()
