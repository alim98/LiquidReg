#!/usr/bin/env python3
"""
Training script for LiquidReg model.
(Enhanced with comprehensive TensorBoard logging.)
"""

import os
import sys
import argparse
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import types
from torch.nn.parallel import DistributedDataParallel as DDP


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.ddp import init_distributed, is_main_process, barrier, cleanup, reduce_mean, get_rank, get_world_size
from models.liquidreg import LiquidReg, LiquidRegLite
from dataloaders.oasis_dataset import create_oasis_loaders
from losses.registration_losses import CompositeLoss
from utils.preprocessing import normalize_volume

# --- TensorBoard viz helpers & optional deps ---
import torch.nn.functional as F  # for image upscaling (viz only)
try:
    from torchvision.utils import make_grid  # optional; used for multi-slice grids
except Exception:
    make_grid = None

amp_dtype = torch.bfloat16

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _select_amp_dtype(device: torch.device):
    if device.type != "cuda":
        return None
    # Force float16 for compatibility with CUDA 11.6
    # (bfloat16 is not supported for grid_sample in older CUDA versions)
    return torch.float16

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(config: dict) -> nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    
    if model_config['name'] == 'LiquidReg':
        model = LiquidReg(
            image_size=tuple(model_config['image_size']),
            encoder_type=model_config['encoder_type'],
            encoder_channels=model_config['encoder_channels'],
            liquid_hidden_dim=model_config['liquid_hidden_dim'],
            liquid_num_steps=model_config['liquid_num_steps'],
            velocity_scale=model_config['velocity_scale'],
            num_squaring=model_config['num_squaring'],
            fusion_type=model_config['fusion_type'],
        )
    elif model_config['name'] == 'LiquidRegLite':
        model = LiquidRegLite(
            image_size=tuple(model_config['image_size']),
            encoder_channels=model_config['encoder_channels'],
            liquid_hidden_dim=model_config['liquid_hidden_dim'],
            liquid_num_steps=model_config['liquid_num_steps'],
            velocity_scale=model_config['velocity_scale'],
            num_squaring=model_config['num_squaring'],
        )
    else:
        raise ValueError(f"Unknown model: {model_config['name']}")
    
    return model


def create_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    train_config = config['training']
    
    if train_config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
    elif train_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config['learning_rate'],
            momentum=0.9,
            weight_decay=train_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: dict):
    """Create learning rate scheduler."""
    train_config = config['training']
    
    if train_config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config['num_epochs'],
            # eta_min=1e-6
            eta_min=1e-5
        )
    elif train_config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5
        )
    elif train_config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    else:
        scheduler = None
    
    return scheduler


# ------------------------- BETTER TENSORBOARD HELPERS -------------------------

def _to_uint01(img: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per-image to [0,1]. Expects (N,1,H,W) or (1,H,W)."""
    if img.ndim == 2:  # (H,W)
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.ndim == 3:  # (1,H,W) or (C,H,W)
        img = img.unsqueeze(0)
    # img: (N,C,H,W)
    mn = img.amin(dim=(2,3), keepdim=True)
    mx = img.amax(dim=(2,3), keepdim=True)
    img = (img - mn) / (mx - mn + 1e-8)
    return img.squeeze(0)  # (C,H,W)

def _slice_from_vol(vol_5d: torch.Tensor, plane: str = "axial", idx: "str|int" = "mid") -> torch.Tensor:
    """
    vol_5d: (N=1,C=1,D,H,W) -> returns (H,W) slice tensor on CPU.
    """
    v = vol_5d.detach().float().cpu()
    assert v.ndim == 5 and v.shape[0] == 1 and v.shape[1] == 1
    _, _, D, H, W = v.shape
    if plane == "axial":   # z
        k = D//2 if idx == "mid" else int(idx)
        sl = v[0,0,k,:,:]
    elif plane == "sag":   # x
        k = W//2 if idx == "mid" else int(idx)
        sl = v[0,0,:,:,k]
    else:                  # "cor" y
        k = H//2 if idx == "mid" else int(idx)
        sl = v[0,0,:,k,:]
    return sl  # (H,W)

def _upscale_chw(chw: torch.Tensor, out_hw: int = 384) -> torch.Tensor:
    """
    CHW -> CHW upscaled (bilinear) for crisper TB display.
    """
    x = chw.unsqueeze(0)             # (1,C,H,W)
    x = F.interpolate(x, size=(out_hw, out_hw), mode="bilinear", align_corners=False)
    return x.squeeze(0)              # (C,H,W)

def _add_images(writer, tag_prefix, fixed, moving, warped, step, *,
                plane="axial", out_hw=384, grid_slices=0):
    """
    Logs mid-slice previews (normalized + upscaled). If grid_slices>0 and
    torchvision is available, also logs a grid across the depth.
    Accepts (N,C,D,H,W) or (N,C,H,W).
    """
    try:
        # Convert to (1,1,D,H,W) if needed and pick the first item
        def _ensure_5d(x):
            x = x.detach()
            if x.ndim == 5:
                return x[:1, :1]
            elif x.ndim == 4:  # (N,C,H,W) -> treat as (N,C,1,H,W)
                return x[:1, :1].unsqueeze(2)
            else:
                raise ValueError(f"Unexpected shape for image logging: {tuple(x.shape)}")

        f5 = _ensure_5d(fixed);  m5 = _ensure_5d(moving);  w5 = _ensure_5d(warped)

        # Single mid-slice for paper figures
        f_sl = _slice_from_vol(f5, plane=plane, idx="mid")
        m_sl = _slice_from_vol(m5, plane=plane, idx="mid")
        w_sl = _slice_from_vol(w5, plane=plane, idx="mid")

        f_img = _upscale_chw(_to_uint01(f_sl), out_hw=out_hw)
        m_img = _upscale_chw(_to_uint01(m_sl), out_hw=out_hw)
        w_img = _upscale_chw(_to_uint01(w_sl), out_hw=out_hw)

        writer.add_image(f"{tag_prefix}/{plane}/fixed",  f_img, step, dataformats="CHW")
        writer.add_image(f"{tag_prefix}/{plane}/moving", m_img, step, dataformats="CHW")
        writer.add_image(f"{tag_prefix}/{plane}/warped", w_img, step, dataformats="CHW")

        # Optional: grid of evenly spaced slices (appendix-friendly)
        if grid_slices and make_grid is not None:
            _, _, D, _, _ = f5.shape
            idxs = torch.linspace(0, D-1, steps=min(grid_slices, D)).long()
            tiles = []
            for k in idxs:
                sl = _slice_from_vol(f5, plane=plane, idx=int(k))
                sl = _to_uint01(sl)
                sl = _upscale_chw(sl, out_hw=out_hw//2)  # slightly smaller tiles
                tiles.append(sl)
            grid = make_grid(torch.stack(tiles, 0), nrow=len(tiles), padding=2)  # (C,H,W)
            writer.add_image(f"{tag_prefix}/{plane}/fixed_grid", grid, step, dataformats="CHW")
    except Exception:
        pass  # never let viz crash training

def _log_param_and_grad_hists(writer, model, step, every_n=500):
    if step % every_n != 0:
        return
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None:
                continue
            try:
                writer.add_histogram(f'params/{name}', p.detach().float().cpu().numpy(), step)
                if p.grad is not None:
                    writer.add_histogram(f'grads/{name}', p.grad.detach().float().cpu().numpy(), step)
            except Exception:
                continue

def _device_mem_stats():
    if not torch.cuda.is_available():
        return {}
    i = torch.cuda.current_device()
    return {
        'cuda/mem_allocated_mb': torch.cuda.memory_allocated(i) / (1024**2),
        'cuda/mem_reserved_mb':  torch.cuda.memory_reserved(i)  / (1024**2),
        'cuda/max_mem_allocated_mb': torch.cuda.max_memory_allocated(i) / (1024**2),
    }

def _field_stats_and_hists(writer, tag_prefix, velocity_field, jacobian_det, step, *, hist_samples=250_000):
    """
    Logs flow magnitude stats and jacobian stats (+ histograms).
    Purely for monitoring; no training effects.
    """
    try:
        # Flow magnitude
        if velocity_field is not None:
            v = velocity_field.detach().float()
            vmag = torch.linalg.vector_norm(v, dim=1)  # (N,D,H,W)
            writer.add_scalar(f"{tag_prefix}/flow_mag_mean", vmag.mean().item(), step)
            writer.add_scalar(f"{tag_prefix}/flow_mag_max",  vmag.max().item(), step)
            # histogram (sample to keep TB light)
            flat = vmag.flatten()
            if flat.numel() > hist_samples:
                idx = torch.randperm(flat.numel(), device=flat.device)[:hist_samples]
                flat = flat[idx]
            writer.add_histogram(f"{tag_prefix}/flow_mag_hist", flat.cpu().numpy(), step)

        # Jacobian stats
        if jacobian_det is not None:
            j = jacobian_det.detach().float()
            nonpos = (j <= 0).float().mean().item() * 100.0
            writer.add_scalar(f"{tag_prefix}/jac_nonpos_percent", nonpos, step)
            writer.add_scalar(f"{tag_prefix}/jac_min", j.min().item(), step)
            writer.add_scalar(f"{tag_prefix}/jac_mean", j.mean().item(), step)
            # histogram (sample)
            flatj = j.flatten()
            if flatj.numel() > hist_samples:
                idx = torch.randperm(flatj.numel(), device=flatj.device)[:hist_samples]
                flatj = flatj[idx]
            writer.add_histogram(f"{tag_prefix}/jac_det_hist", flatj.cpu().numpy(), step)
    except Exception:
        pass

# ---------------------------------------------------------------------------

# ----------------------- GRACEFUL DATALOADER HELPERS ------------------------

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

def is_cuda_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return isinstance(e, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)

class SafeDataset(Dataset):
    """
    Wraps another Dataset and returns None for samples that throw in __getitem__.
    Keeps a set of bad indices to avoid repeating logs.
    """
    def __init__(self, base: Dataset, *, logger: SummaryWriter | None = None, split_name: str = "train"):
        self.base = base
        self.bad = set()
        self.logger = logger
        self.split = split_name

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        try:
            return self.base[idx]
        except Exception as e:
            # remember the index so we don't spam logs
            if idx not in self.bad:
                self.bad.add(idx)
                print(f"[WARN] {self.split}: skipping bad sample idx={idx}: {e}")
                if self.logger is not None:
                    step = len(self.bad)
                    self.logger.add_text(f"errors/{self.split}_sample", f"idx={idx} err={e}", step)
                    self.logger.add_scalar(f"errors/{self.split}_bad_sample_count", len(self.bad), step)
            return None

def safe_collate(batch, *, min_batch: int = 1):
    """
    Drops None items produced by SafeDataset. Returns None if nothing left.
    Uses default_collate for everything else.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) < min_batch:
        return None
    return default_collate(batch)

def make_safe_loader(
    loader: torch.utils.data.DataLoader,
    *,
    split_name: str,
    logger: SummaryWriter,
    num_workers: int,
    pin_memory: bool = True,
    sampler=None
) -> torch.utils.data.DataLoader:
    base_ds = loader.dataset
    safe_ds = SafeDataset(base=base_ds, logger=logger, split_name=split_name)

    shuffle = (sampler is None) and (split_name == "train")
    return torch.utils.data.DataLoader(
        dataset=safe_ds,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        sampler=sampler,                 # <- preserve DDP sharding
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,
        drop_last=loader.drop_last,
        persistent_workers=(num_workers > 0),
        timeout=0,
        **({"prefetch_factor": 2} if num_workers > 0 else {})
    )


# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    config: dict,
    epoch: int,
    writer: SummaryWriter,
    fail_fast = False
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_losses = {}
    num_batches = len(train_loader)

    dropped_batches = 0
    max_dropped = max(5, int(0.2 * num_batches))  # e.g., 20% or at least 5

    start_time_epoch = time.time()

    pbar = tqdm(total=num_batches, desc=f"Epoch {epoch}")
    it = iter(train_loader)
    batch_idx = 0
    while batch_idx < num_batches:
        iter_start = time.time()
        try:
            batch = next(it)
        except StopIteration:
            break
        except Exception as e:
            if fail_fast:
                raise
            # fetching failed before forward — log and skip
            print(f"[WARN] dataloader fetch failed at batch {batch_idx}: {e}")
            writer.add_text("errors/dataloader_fetch",
                            f"epoch={epoch} batch={batch_idx} err={e}",
                            epoch * max(1, num_batches) + batch_idx)
            batch_idx += 1
            pbar.update(1)

            dropped_batches += 1
            if dropped_batches > max_dropped:
                if fail_fast:
                    raise RuntimeError("Too many dropped batches; failing fast.")
                print(f"[WARN] Too many dropped batches ({dropped_batches}/{num_batches}). Continuing, but data may be corrupted.")

            continue

        # SafeDataset+safe_collate may hand us None when all items were bad
        if batch is None:
            batch_idx += 1
            pbar.update(1)
            
            dropped_batches += 1
            if dropped_batches > max_dropped:
                if fail_fast:
                    raise RuntimeError("Too many dropped batches; failing fast.")
                print(f"[WARN] Too many dropped batches ({dropped_batches}/{num_batches}). Continuing, but data may be corrupted.")
            
            continue

        fixed = batch['fixed'].float()
        moving = batch['moving'].float()
        
        # Move to device
        device = next(model.parameters()).device
        fixed = fixed.to(device, non_blocking=True)
        moving = moving.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        use_amp = config['training']['use_amp']

        try:
            # Try new API first
            from torch.amp import autocast
            amp_dtype = _select_amp_dtype(device)
            autocast_context = autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype)
        
        except TypeError:            
            from torch.cuda.amp import autocast
            # old API: omit dtype
            autocast_context = autocast(enabled=use_amp)
        
        try:
            with autocast_context:
                output = model(fixed, moving, return_intermediate=True)
                
                warped = output['warped_moving']
                velocity = output['velocity_field']
                jacobian_det = output['jacobian_det']
                liquid_params = output['liquid_params']
                
                # Compute losses
                losses = criterion(
                    fixed=fixed,
                    warped=warped,
                    velocity_field=velocity,
                    jacobian_det=jacobian_det,
                    liquid_params=liquid_params
                )
                
                loss = losses['total']
                
                loss_log = reduce_mean(loss.detach())

        except Exception as e:
            print(f"[ERROR] Exception in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Log error to TB
            writer.add_text("errors/forward", f"epoch={epoch} batch={batch_idx} err={e}", epoch * num_batches + batch_idx)
            continue
        
        # -------------------- Backward/Step with OOM guard --------------------
        oom_this_batch = False
        step = epoch * num_batches + batch_idx  # for logging

        try:
            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                prev_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
                scaler.scale(loss).backward()

                if config['training']['grad_clip_norm'] > 0.0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['grad_clip_norm']
                    )
                else:
                    grad_norm = float('nan')

                # Optional NaN grad check
                has_nan_grad = any(
                    (p.grad is not None and torch.isnan(p.grad).any())
                    for _, p in model.named_parameters()
                )

                if has_nan_grad:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    new_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
            else:
                loss.backward()
                if config['training']['grad_clip_norm'] > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['grad_clip_norm']
                    )
                else:
                    grad_norm = float('nan')

                has_nan_grad = any(
                    (p.grad is not None and torch.isnan(p.grad).any())
                    for _, p in model.named_parameters()
                )
                if not has_nan_grad:
                    optimizer.step()
                prev_scale = new_scale = 1.0

        except RuntimeError as e:
            if is_cuda_oom(e):
                oom_this_batch = True
                if fail_fast:
                    raise
                print(f"[WARN] OOM at epoch {epoch} batch {batch_idx}: {e}")
                writer.add_scalar('errors/oom', 1, step)
                writer.add_text('errors/oom_detail', f"epoch={epoch} batch={batch_idx} err={e}", step)

                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            else:
                # non-OOM runtime error during backward/step
                if fail_fast:
                    raise
                print(f"[WARN] RuntimeError (non-oom) at epoch {epoch} batch {batch_idx}: {e}")
                writer.add_text('errors/backward_runtime', f"epoch={epoch} batch={batch_idx} err={e}", step)
                optimizer.zero_grad(set_to_none=True)

        if oom_this_batch:
            # skip any per-batch success logging and move to next batch
            continue
        # ----------------------------------------------------------------------

        # Timing & throughput
        iter_time = time.time() - iter_start
        batch_size = fixed.size(0)
        throughput = batch_size / max(iter_time, 1e-8)

        # Accumulate losses
        total_loss += loss.item()
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'grad': f"{grad_norm:.2f}",
            'ips': f"{throughput:.1f}"
        })
        
        # Log to tensorboard (per step)
        if batch_idx % config['logging']['log_interval'] == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss_log.item(), step)
            writer.add_scalar('train/grad_norm', grad_norm, step)
            writer.add_scalar('train/iter_time_s', iter_time, step)
            writer.add_scalar('train/throughput_items_per_s', throughput, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            if config['training']['use_amp']:
                writer.add_scalar('amp/scale', new_scale, step)
                # overflow heuristic: drop in scale
                if new_scale < prev_scale:
                    writer.add_scalar('amp/overflow_events', 1, step)
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value.item(), step)

            # CUDA memory stats (if available)
            for k, v in _device_mem_stats().items():
                writer.add_scalar(k, v, step)

            # --- richer visualizations & stats ---
            _add_images(writer, "train/images", fixed, moving, warped, step,
                        plane="axial", out_hw=384, grid_slices=0)

            _field_stats_and_hists(
                writer, "train/fields",
                velocity_field=velocity,
                jacobian_det=jacobian_det,
                step=step
            )

            # Param/grad histograms occasionally
            _log_param_and_grad_hists(writer, model, step, every_n=max(1, 10 * config['logging']['log_interval']))
    
        batch_idx += 1
        pbar.update(1)
    
    epoch_time = time.time() - start_time_epoch
    writer.add_scalar('train/epoch_time_s', epoch_time, epoch)
    if num_batches > 0:
        # Fallback to config batch_size if we never saw a batch_size variable
        bs = locals().get("batch_size", config['data']['batch_size'])
        denom = max(epoch_time, 1e-8)
        items = (len(train_loader.dataset) if hasattr(train_loader, "dataset") else (num_batches * bs))
        writer.add_scalar('train/epoch_avg_ips', items / denom, epoch)

    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()} if num_batches > 0 else {}
    avg_losses['avg_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_losses

def validate_epoch(
    model: nn.Module,
    val_loader,
    criterion,
    config: dict,
    epoch: int,
    writer: SummaryWriter,
    fail_fast=False
) -> dict:
    """Validate for one epoch."""
    
    base = model.module if hasattr(model, "module") else model

    model.eval()
    
    total_loss = 0.0
    total_losses = {}
    num_batches = len(val_loader)
    
    dropped_batches = 0
    max_dropped = max(5, int(0.2 * num_batches))  # e.g., 20% or at least 5

    
    with torch.no_grad():
        pbar = tqdm(total=num_batches, desc=f"Validation {epoch}")
        it = iter(val_loader)
        batch_idx = 0
        while batch_idx < num_batches:
            try:
                batch = next(it)
            except StopIteration:
                break
            except Exception as e:
                if fail_fast:
                    raise
                print(f"[WARN] val dataloader fetch failed at batch {batch_idx}: {e}")
                writer.add_text("errors/val_dataloader_fetch",
                                f"epoch={epoch} batch={batch_idx} err={e}",
                                epoch * max(1, num_batches) + batch_idx)
                batch_idx += 1
                pbar.update(1)
                
                dropped_batches += 1
                if dropped_batches > max_dropped:
                    if fail_fast:
                        raise RuntimeError("Too many dropped batches; failing fast.")
                    print(f"[WARN] Too many dropped batches ({dropped_batches}/{num_batches}). Continuing, but data may be corrupted.")

                continue



            if batch is None:
                batch_idx += 1
                pbar.update(1)
                
                dropped_batches += 1
                if dropped_batches > max_dropped:
                    if fail_fast:
                        raise RuntimeError("Too many dropped batches; failing fast.")
                    print(f"[WARN] Too many dropped batches ({dropped_batches}/{num_batches}). Continuing, but data may be corrupted.")

                continue

            fixed = batch['fixed'].float()
            moving = batch['moving'].float()
            
            # Move to device
            device = next(model.parameters()).device
            fixed = fixed.to(device, non_blocking=True)
            moving = moving.to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            use_amp = config['training']['use_amp']
            try:
                # Try new API first
                from torch.amp import autocast
                autocast_context = autocast(device_type=device.type, enabled=use_amp)
            except TypeError:
                # Fallback to old API
                from torch.cuda.amp import autocast as cuda_autocast
                autocast_context = cuda_autocast(enabled=use_amp)
            
            try:
                with autocast_context:
                    output = model(fixed, moving, return_intermediate=True)
                    
                    warped = output['warped_moving']
                    velocity = output['velocity_field']
                    jacobian_det = output['jacobian_det']
                    liquid_params = output['liquid_params']
                    deformation = output['deformation_field']

                    # ---- Optional Dice logging if labels are in the batch ----
                    if ('segmentation_fixed' in batch and 'segmentation_moving' in batch
                        and batch['segmentation_fixed'] is not None and batch['segmentation_moving'] is not None):
                        fseg = batch['segmentation_fixed'].to(device, non_blocking=True).long()  # (N,1,D,H,W)
                        mseg = batch['segmentation_moving'].to(device, non_blocking=True).long()

                        # Binary foreground Dice (quick + robust)
                        ffg = (fseg > 0).float()
                        mfg = (mseg > 0).float()

                        # Make 2-class one-hot [bg, fg]
                        m2 = torch.cat([(1.0 - mfg), mfg], dim=1)  # (N,2,D,H,W)

                        # Use the same transformer as images
                        w2 = base.spatial_transformer(m2, deformation)  # bilinear ok for probs
                        wfg = w2[:, 1:2, ...]  # foreground prob

                        eps = 1e-6
                        inter = (wfg * ffg).sum(dim=(1,2,3,4))
                        denom = wfg.sum(dim=(1,2,3,4)) + ffg.sum(dim=(1,2,3,4))
                        dice_fg = ((2 * inter + eps) / (denom + eps)).mean().item()
                        writer.add_scalar('val/dice_fg', dice_fg, epoch)

                    
                    # Compute losses
                    losses = criterion(
                        fixed=fixed,
                        warped=warped,
                        velocity_field=velocity,
                        jacobian_det=jacobian_det,
                        liquid_params=liquid_params
                    )
                    
                    loss = losses['total']

            except RuntimeError as e:
                if is_cuda_oom(e):
                    if fail_fast: raise
                    print(f"[WARN] Val OOM at epoch {epoch} batch {batch_idx}: {e}")
                    writer.add_scalar('errors/val_oom', 1, epoch * max(1, num_batches) + batch_idx)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    batch_idx += 1
                    pbar.update(1)
                    continue
                else:
                    if fail_fast: raise
                    print(f"[WARN] Val runtime error at epoch {epoch} batch {batch_idx}: {e}")
                    writer.add_text('errors/val_forward_runtime', f"epoch={epoch} batch={batch_idx} err={e}",
                                    epoch * max(1, num_batches) + batch_idx)
                    batch_idx += 1
                    pbar.update(1)
                    continue

            # Accumulate losses
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            # Per-batch val logging
            if batch_idx % max(1, config['logging']['log_interval']) == 0:
                step = epoch * num_batches + batch_idx
                writer.add_scalar('val/step_loss', reduce_mean(loss.detach()).item(), step)
                _add_images(writer, "val/images", fixed, moving, warped, step,
                            plane="axial", out_hw=384, grid_slices=0)
                _field_stats_and_hists(
                    writer, "val/fields",
                    velocity_field=velocity,
                    jacobian_det=jacobian_det,
                    step=step
                )
                
            batch_idx += 1
            pbar.update(1)
    
    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()} if num_batches > 0 else {}
    avg_losses['avg_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Log to tensorboard
    writer.add_scalar('val/loss', avg_losses['avg_loss'], epoch)
    for key, value in avg_losses.items():
        if key != 'avg_loss':
            writer.add_scalar(f'val/{key}', value, epoch)

    # --- one full-slice-style preview from the last seen batch (best-effort) ---
    try:
        writer.add_text('val/preview_note', 'Preview uses first item of a validation batch', epoch)
        _add_images(writer, "val/preview_fullslice_like", fixed[:1], moving[:1], warped[:1],
                    step=epoch, plane="axial", out_hw=512, grid_slices=0)
        _add_images(writer, "val/preview_fullslice_like", fixed[:1], moving[:1], warped[:1],
                    step=epoch, plane="sag", out_hw=512, grid_slices=0)
        _add_images(writer, "val/preview_fullslice_like", fixed[:1], moving[:1], warped[:1],
                    step=epoch, plane="cor", out_hw=512, grid_slices=0)
    except Exception:
        pass
    
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    config: dict,
    filepath: str,
    is_best: bool = False,
    writer: SummaryWriter = None
):
    """Save model checkpoint."""
    base = model.module if hasattr(model, "module") else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': base.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

    # Log checkpoint event
    if writer is not None:
        writer.add_scalar('checkpoints/last_epoch', epoch, epoch)
        writer.add_scalar('checkpoints/last_loss', loss, epoch)
        writer.add_text('checkpoints/last_path', str(filepath), epoch)
        if is_best:
            writer.add_text('checkpoints/best_path', str(best_path), epoch)

from pathlib import Path
import shutil

def _atomic_copy(src: str, dst: str):
    tmp = f"{dst}.tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)

def _update_ckpt_aliases(checkpoint_path: str, is_best: bool):
    """Create/overwrite easy-to-find pointers: last.pth and best.pth"""
    ckpt = Path(checkpoint_path)
    base = ckpt.parent  # .../checkpoints
    last_ptr = base / "last.pth"
    _atomic_copy(str(ckpt), str(last_ptr))
    if is_best:
        best_ptr = base / "best.pth"
        _atomic_copy(str(ckpt), str(best_ptr))

import builtins

def make_qprint(is_main: bool):
    def _qprint(*a, **k):
        if is_main:
            builtins.print(*a, **k)
    return _qprint

def main():
    parser = argparse.ArgumentParser(description='Train LiquidReg model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override data root directory')
    parser.add_argument('--small_dataset', action='store_true',
                       help='Use only 1 percent of the dataset for training (to reduce memory usage)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config (to reduce memory usage)')
    parser.add_argument('--lite', action='store_true',
                       help='Use LiquidRegLite model instead of full LiquidReg (to reduce memory usage)')
    parser.add_argument('--reduce_steps', action='store_true',
                       help='Reduce the number of steps in the liquid ODE solver (to reduce memory usage)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing to save memory during training')
    parser.add_argument('--small_encoder', action='store_true',
                       help='Use a smaller encoder with fewer channels (to reduce memory usage)')
    parser.add_argument('--train_pairs', type=str, default=None)
    parser.add_argument('--val_pairs',   type=str, default=None)
    parser.add_argument('--work_dir', type=str, help='Override logging directory')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--resume', type=str, default=None,
                    help='Path to a checkpoint to resume from')
    # add with the other flags
    parser.add_argument('--fail_fast', action='store_true',
                    help='Abort immediately on any batch/data error (default: skip bad batches)')
    parser.add_argument("--ddp", action="store_true", help="Enable multi-GPU via torchrun.")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP mixed precision.")



    args = parser.parse_args()
    
    ## DDP / device
    use_amp = not args.no_amp  # CLI default; we’ll combine with config after load
    
    # Initialize distributed (if launched with torchrun and --ddp)
    is_ddp, local_rank, device = (init_distributed(backend="nccl") if args.ddp
                                else (False, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    main_process = is_main_process()
    rank = get_rank()
    world = get_world_size()

    qprint = make_qprint(main_process)

    qprint(f"[DDP] enabled={is_ddp} world={world} rank={rank} local_rank={local_rank} device={device}")

    # Load configuration
    config = load_config(args.config)
    
    # Fail fast
    FAIL_FAST = bool(args.fail_fast)
    
    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed

    #override work_dir if provided
    if args.work_dir:
        config['logging']['log_dir'] = args.work_dir
        
    # Override data root if provided
    if args.data_root:
        config['data']['data_root'] = args.data_root
        
    # Override batch size if provided
    if args.batch_size is not None:
        print(f"Overriding batch size: {config['data']['batch_size']} → {args.batch_size}")
        config['data']['batch_size'] = args.batch_size
        
    # Use lite model if requested
    if args.lite:
        print("Using LiquidRegLite model instead of full LiquidReg")
        config['model']['name'] = "LiquidRegLite"
        
    # Reduce ODE steps if requested
    if args.reduce_steps:
        original_steps = config['model']['liquid_num_steps']
        config['model']['liquid_num_steps'] = max(2, original_steps // 2)  # At least 2 steps
        print(f"Reducing ODE steps: {original_steps} → {config['model']['liquid_num_steps']}")
        
        # Also reduce squaring steps to save memory
        original_squaring = config['model']['num_squaring']
        config['model']['num_squaring'] = max(2, original_squaring - 2)  # At least 2 steps
        print(f"Reducing squaring steps: {original_squaring} → {config['model']['num_squaring']}")
        
    # Use smaller encoder if requested
    if args.small_encoder:
        original_channels = config['model']['encoder_channels']
        config['model']['encoder_channels'] = original_channels // 2
        print(f"Using smaller encoder: {original_channels} → {config['model']['encoder_channels']} channels")
        
        # Also reduce liquid hidden dimension
        if 'liquid_hidden_dim' in config['model']:
            original_hidden = config['model']['liquid_hidden_dim']
            config['model']['liquid_hidden_dim'] = max(16, original_hidden // 2)
            print(f"Reducing liquid hidden dim: {original_hidden} → {config['model']['liquid_hidden_dim']}")
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    print(f"Using device: {device}")
    # Honor both CLI and config; store back into config so all helpers read the same truth.
    use_amp = (not args.no_amp) and bool(config['training'].get('use_amp', True)) and (device.type == 'cuda')
    config['training']['use_amp'] = use_amp
    # Create output directories
    log_dir = Path(config['logging']['log_dir']) / config['logging']['experiment_name']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup logging
    # one logdir per rank to avoid file contention
    writer = SummaryWriter((log_dir / 'tensorboard' / f"rank{rank}"))


    # Log config & hparams up front
    try:
        writer.add_text('config/yaml', f"```\n{yaml.dump(config)}\n```", 0)
        # hparams: keep it simple (avoid nested dicts)
        flat_hparams = {
            'model_name': config['model']['name'],
            'image_size': str(tuple(config['model']['image_size'])),
            'optimizer': config['training']['optimizer'],
            'lr': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'scheduler': str(config['training']['scheduler']),
            'batch_size': config['data']['batch_size'],
            'patch_size': int(config['data']['patch_size']),
            'use_amp': int(config['training']['use_amp']),
        }
        # add_hparams logs a special summary; provide a dummy metric to render
        writer.add_hparams(flat_hparams, {'hparams/placeholder_metric': 0.0})
    except Exception:
        pass
    
    # Create data loaders
    from dataloaders.reg_pairs_dataset import create_reg_pairs_loaders_ddp

    assert args.train_pairs and args.val_pairs, \
        "This project now trains on pairs CSVs. Provide --train_pairs and --val_pairs."

    train_loader, val_loader, train_sampler, val_sampler = create_reg_pairs_loaders_ddp(
        train_pairs_csv=args.train_pairs,
        val_pairs_csv=args.val_pairs,
        batch_size=config['data']['batch_size'],
        patch_size=config['data']['patch_size'],
        patch_stride=config['data']['patch_stride'],
        patches_per_pair=config['data']['patches_per_pair'],
        num_workers=config['data']['num_workers'],
        use_labels_train=True,
        use_labels_val=True,
        target_size=tuple(config['model']['image_size']),
        distributed=is_ddp,
    )

    from torch.utils.data.distributed import DistributedSampler
    
    if args.small_dataset:
        print("Using small dataset (1 percent of the full dataset)")

        # train subset
        train_size = len(train_loader.dataset)
        small_size = max(1, int(train_size * 0.01))
        idx_train = torch.randperm(train_size)[:small_size]
        train_subset = torch.utils.data.Subset(train_loader.dataset, idx_train)
        if is_ddp:
            train_sampler = DistributedSampler(train_subset, shuffle=True, drop_last=True)
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=config['data']['batch_size'],
                shuffle=False, sampler=train_sampler,
                num_workers=config['data']['num_workers'], pin_memory=True, drop_last=True,
                persistent_workers=(config['data']['num_workers'] > 0),
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_subset, batch_size=config['data']['batch_size'],
                shuffle=True, num_workers=config['data']['num_workers'], pin_memory=True, drop_last=True,
                persistent_workers=(config['data']['num_workers'] > 0),
            )

        # val subset
        val_size = len(val_loader.dataset)
        small_val_size = max(1, int(val_size * 0.01))
        idx_val = torch.randperm(val_size)[:small_val_size]
        val_subset = torch.utils.data.Subset(val_loader.dataset, idx_val)
        if is_ddp:
            val_sampler = DistributedSampler(val_subset, shuffle=False, drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=config['data']['batch_size'],
                shuffle=False, sampler=val_sampler,
                num_workers=config['data']['num_workers'], pin_memory=True, drop_last=False,
                persistent_workers=(config['data']['num_workers'] > 0),
            )
        else:
            val_loader = torch.utils.data.DataLoader(
                val_subset, batch_size=config['data']['batch_size'],
                shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True, drop_last=False,
                persistent_workers=(config['data']['num_workers'] > 0),
            )

    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    writer.add_scalar('data/train_batches', len(train_loader), 0)
    writer.add_scalar('data/val_batches', len(val_loader), 0)

    if not FAIL_FAST:
        train_loader = make_safe_loader(train_loader, split_name="train", logger=writer,
                                        num_workers=config['data']['num_workers'], sampler=train_sampler)
        val_loader   = make_safe_loader(val_loader,   split_name="val",   logger=writer,
                                        num_workers=config['data']['num_workers'], sampler=val_sampler)

    
    # Create model
    model = create_model(config)
    model = model.to(device=device, dtype=torch.float32)
    
    # model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=False)

    
    # Try to log the model graph (best-effort)
    try:
        # Create tiny dummy inputs matching expected shape: (N=1,C=1,D,H,W) or (N,C,H,W)
        img_sz = config['model']['image_size']
        shape = (1, 1, *img_sz) if len(img_sz) == 3 else (1, 1, *img_sz[-2:])
        dummy_fixed  = torch.zeros(shape, device=device)
        dummy_moving = torch.zeros(shape, device=device)
        writer.add_graph(model, (dummy_fixed, dummy_moving), use_strict_trace=False)
    except Exception:
        pass

    # Enable gradient checkpointing if requested (DDP-safe)
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing to save memory")
        _m = model.module if hasattr(model, "module") else model  # <- use the base module

        # Encoder (if it supports HF-style GC)
        if hasattr(_m, 'encoder') and hasattr(_m.encoder, 'encoder'):
            _m.encoder.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for encoder")

        # Liquid core: wrap its forward with torch.utils.checkpoint
        if hasattr(_m, 'liquid_core'):
            import torch.utils.checkpoint as cp
            original_forward = _m.liquid_core.forward  # bound method on the base module

            def checkpointed_forward(self, coords, params):
                # Keep signature; wrap compute in checkpoint
                return cp.checkpoint(lambda c, p: original_forward(c, p),
                                    coords, params,
                                    preserve_rng_state=True)

            _m.liquid_core.forward = types.MethodType(checkpointed_forward, _m.liquid_core)
            print("Gradient checkpointing enabled for liquid_core")


    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    writer.add_scalar('params/total', total_params, 0)
    writer.add_scalar('params/trainable', trainable_params, 0)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create loss function
    criterion = CompositeLoss(
        similarity_loss=config['training']['similarity_loss'],
        lambda_similarity=config['training']['lambda_similarity'],
        lambda_jacobian=config['training']['lambda_jacobian'],
        lambda_velocity=config['training']['lambda_velocity'],
        lambda_liquid=config['training']['lambda_liquid'],
        lncc_window=config['training']['lncc_window'],
        jacobian_penalty=config['training']['jacobian_penalty']
    )
    
    # Mixed precision scaler
    if config['training']['use_amp'] and device.type == 'cuda':
        # Try new API first
        try:
            from torch.amp import GradScaler as AmpGradScaler
        except Exception:
            from torch.cuda.amp import GradScaler as AmpGradScaler
        scaler = AmpGradScaler(enabled=(use_amp and amp_dtype is torch.float16))
    else:
        # For CPU or when AMP is disabled, create a dummy scaler
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
            def unscale_(self, optimizer): pass
            def get_scale(self): return 1.0
            def is_enabled(self): return False
        scaler = DummyScaler()
    
    # Training loop
    early_stopping_counter = 0    
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1
        if 'loss' in ckpt:
            best_loss = float(ckpt['loss'])  # treat stored loss as current best
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_loss={best_loss:.4f})")

    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        
        if is_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch, writer,FAIL_FAST
        )
        
        print(f"Train Loss: {train_losses['avg_loss']:.4f}")
        writer.add_scalar('train/epoch_loss', train_losses['avg_loss'], epoch)
        # log individual loss terms (epoch avg)
        for k, v in train_losses.items():
            if k != 'avg_loss':
                writer.add_scalar(f'train/epoch_{k}', v, epoch)
        
        # Validate
        if epoch % config['validation']['eval_interval'] == 0:
            val_losses = validate_epoch(
                model, val_loader, criterion, config, epoch, writer,FAIL_FAST
            )
            
            print(f"Val Loss: {val_losses['avg_loss']:.4f}")
            writer.add_scalar('val/epoch_loss', val_losses['avg_loss'], epoch)
            for k, v in val_losses.items():
                if k != 'avg_loss':
                    writer.add_scalar(f'val/epoch_{k}', v, epoch)
            
            min_delta = float(config['training'].get('early_stopping_delta', 0.0))
            mon = 'similarity' if 'similarity' in val_losses else 'avg_loss'
            current_loss = val_losses[mon]
            is_best = current_loss < (best_loss - min_delta)
            
            if is_best:
                best_loss = current_loss
                early_stopping_counter = 0
                print(f"New best validation loss: {best_loss:.4f}")
                writer.add_scalar('val/best_loss', best_loss, epoch)
            else:
                early_stopping_counter += 1
                writer.add_scalar('val/no_improve_streak', early_stopping_counter, epoch)
            
            # Save checkpoint
            if main_process and (epoch % config['logging']['save_interval'] == 0 or is_best):
                checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, current_loss, 
                    config, str(checkpoint_path), is_best, writer
                )
            
            # Early stopping
            if (early_stopping_counter >= config['training']['early_stopping_patience'] and
                config['training']['early_stopping_patience'] > 0):
                print(f"Early stopping after {epoch} epochs")
                writer.add_text('early_stopping', f"Stopped at epoch {epoch}", epoch)
                break
        
        # Update scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            mon = 'similarity' if 'val_losses' in locals() and 'similarity' in val_losses else 'avg_loss'
            scheduler.step(val_losses[mon] if 'val_losses' in locals() else train_losses['avg_loss'])
        else:
            scheduler.step()

        
        # Log learning rate again (post-step)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate_post_step', current_lr, epoch)
    
    # Save final model
    if main_process:
        final_path = checkpoint_dir / 'final_model.pth'
        save_checkpoint(
            model, optimizer, scheduler, epoch, 
            val_losses['avg_loss'] if 'val_losses' in locals() else train_losses['avg_loss'], 
            config, str(final_path), writer=writer
        )
    
    # writer.close()
    
    barrier()
    if writer:
        writer.flush(); writer.close()
    cleanup()
    
    print("Training completed!")


if __name__ == '__main__':
    main()
