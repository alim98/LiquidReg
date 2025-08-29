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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.liquidreg import LiquidReg, LiquidRegLite
from dataloaders.oasis_dataset import create_oasis_loaders
from losses.registration_losses import CompositeLoss
from utils.preprocessing import normalize_volume

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
            eta_min=1e-6
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


# ------------------------- NEW: logging helpers -------------------------

def _add_images(writer, tag_prefix, fixed, moving, warped, step, max_images=4):
    """
    Log mid-slice previews (N,C,D,H,W or N,C,H,W supported). Normalizes to [0,1].
    """
    def _to_cpu_img(t):
        t = t.detach().float().cpu()
        if t.dim() == 5:  # N,C,D,H,W -> take middle depth slice
            d = t.shape[2] // 2
            t = t[:, :, d]  # N,C,H,W
        # min-max per image
        t = (t - t.amin(dim=(2,3), keepdim=True)) / (t.amax(dim=(2,3), keepdim=True) - t.amin(dim=(2,3), keepdim=True) + 1e-8)
        return t
    try:
        f = _to_cpu_img(fixed)[:max_images]
        m = _to_cpu_img(moving)[:max_images]
        w = _to_cpu_img(warped)[:max_images]
        writer.add_images(f"{tag_prefix}/fixed",  f, step)
        writer.add_images(f"{tag_prefix}/moving", m, step)
        writer.add_images(f"{tag_prefix}/warped", w, step)
    except Exception as _:
        pass  # images are best-effort; don't crash training


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

# ----------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    config: dict,
    epoch: int,
    writer: SummaryWriter
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_losses = {}
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    start_time_epoch = time.time()

    for batch_idx, batch in enumerate(pbar):
        iter_start = time.time()
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
            autocast_context = autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype)

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
                
                # Compute losses
                losses = criterion(
                    fixed=fixed,
                    warped=warped,
                    velocity_field=velocity,
                    jacobian_det=jacobian_det,
                    liquid_params=liquid_params
                )
                
                loss = losses['total']
        except Exception as e:
            print(f"[ERROR] Exception in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Log error to TB
            writer.add_text("errors/forward", f"epoch={epoch} batch={batch_idx} err={e}", epoch * num_batches + batch_idx)
            continue
        
        # # Backward pass
        # if use_amp and hasattr(scaler, 'scale'):
        #     try:
        #         prev_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
        #         scaler.scale(loss).backward()
        #         scaler.unscale_(optimizer)
                
        #         # Gradient clipping
        #         grad_norm = torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), 
        #             config['training']['grad_clip_norm']
        #         )
                
        #         # Check for NaN gradients
        #         has_nan_grad = False
        #         for name, param in model.named_parameters():
        #             if param.grad is not None and torch.isnan(param.grad).any():
        #                 has_nan_grad = True
        #                 writer.add_text("warnings/nan_grad", f"epoch={epoch} batch={batch_idx} param={name}", epoch * num_batches + batch_idx)
        #                 break
                
        #         if has_nan_grad:
        #             continue
                
        #         scaler.step(optimizer)
        #         scaler.update()
        #         new_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
        #     except ValueError as e:
        #         if "Attempting to unscale FP16 gradients" in str(e):
        #             # Fallback to non-AMP training for this batch
        #             optimizer.zero_grad()
        #             loss.backward()
        #             grad_norm = torch.nn.utils.clip_grad_norm_(
        #                 model.parameters(), 
        #                 config['training']['grad_clip_norm']
        #             )
        #             optimizer.step()
        #             new_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
        #         else:
        #             print(f"[ERROR] Exception in backward pass: {e}")
        #             import traceback
        #             traceback.print_exc()
        #             writer.add_text("errors/backward", f"epoch={epoch} batch={batch_idx} err={e}", epoch * num_batches + batch_idx)
        #             continue
        #     except Exception as e:
        #         print(f"[ERROR] Exception in backward pass: {e}")
        #         import traceback
        #         traceback.print_exc()
        #         writer.add_text("errors/backward", f"epoch={epoch} batch={batch_idx} err={e}", epoch * num_batches + batch_idx)
        #         continue
        # else:
        #     try:
        #         loss.backward()
                
        #         grad_norm = torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), 
        #             config['training']['grad_clip_norm']
        #         )
                
        #         # Check for NaN gradients
        #         has_nan_grad = False
        #         for name, param in model.named_parameters():
        #             if param.grad is not None and torch.isnan(param.grad).any():
        #                 has_nan_grad = True
        #                 writer.add_text("warnings/nan_grad", f"epoch={epoch} batch={batch_idx} param={name}", epoch * num_batches + batch_idx)
        #                 break
                
        #         if has_nan_grad:
        #             continue
                
        #         optimizer.step()
        #         prev_scale = new_scale = 1.0
        #     except Exception as e:
        #         print(f"[ERROR] Exception in backward pass: {e}")
        #         import traceback
        #         traceback.print_exc()
        #         writer.add_text("errors/backward", f"epoch={epoch} batch={batch_idx} err={e}", epoch * num_batches + batch_idx)
        #         continue

        # Backward pass (robust AMP)
        if hasattr(scaler, "is_enabled") and scaler.is_enabled():
            prev_scale = float(getattr(scaler, "get_scale", lambda: 1.0)())
            scaler.scale(loss).backward()

            if config['training']['grad_clip_norm'] > 0.0:
                scaler.unscale_(optimizer)  # grads are FP32 here when scaler is enabled
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip_norm']
                )
            else:
                grad_norm = float('nan')

            # Optional: quick NaN check (after unscale/clip)
            has_nan_grad = any(
                (p.grad is not None and torch.isnan(p.grad).any())
                for _, p in model.named_parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad(set_to_none=True)
                # skip step/update this batch
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

            # Optional: NaN check
            has_nan_grad = any(
                (p.grad is not None and torch.isnan(p.grad).any())
                for _, p in model.named_parameters()
            )
            if not has_nan_grad:
                optimizer.step()
            prev_scale = new_scale = 1.0


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
            writer.add_scalar('train/loss', loss.item(), step)
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

            # Example images (best-effort)
            _add_images(writer, "train/images", fixed, moving, warped, step, max_images=2)

            # Param/grad histograms occasionally
            _log_param_and_grad_hists(writer, model, step, every_n=max(1, 10 * config['logging']['log_interval']))
    
    epoch_time = time.time() - start_time_epoch
    writer.add_scalar('train/epoch_time_s', epoch_time, epoch)
    if num_batches > 0:
        writer.add_scalar('train/epoch_avg_ips', (len(train_loader.dataset) / epoch_time) if hasattr(train_loader, "dataset") else (num_batches * batch_size / epoch_time), epoch)

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
    writer: SummaryWriter
) -> dict:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_losses = {}
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
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
                    # Treat label > 0 as foreground; great for monitoring even if class sets differ.
                    ffg = (fseg > 0).float()
                    mfg = (mseg > 0).float()

                    # Warp moving foreground via probability (one-hot) to avoid NN requirement
                    # Make 2-class one-hot [bg, fg]
                    m2 = torch.cat([(1.0 - mfg), mfg], dim=1)  # (N,2,D,H,W)

                    # Use the same transformer as images
                    w2 = model.spatial_transformer(m2, deformation)  # bilinear ok for probs
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
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            # Per-batch val logging (optional but useful for long epochs)
            if batch_idx % max(1, config['logging']['log_interval']) == 0:
                step = epoch * num_batches + batch_idx
                writer.add_scalar('val/step_loss', loss.item(), step)
                _add_images(writer, "val/images", fixed, moving, warped, step, max_images=2)
    
    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()} if num_batches > 0 else {}
    avg_losses['avg_loss'] = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Log to tensorboard
    writer.add_scalar('val/loss', avg_losses['avg_loss'], epoch)
    for key, value in avg_losses.items():
        if key != 'avg_loss':
            writer.add_scalar(f'val/{key}', value, epoch)
    
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
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
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


    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # pick an AMP dtype; BF16 is safe (no loss-scaling needed) on Ampere+
    use_amp = bool(config['training'].get('use_amp', True) and device.type == 'cuda')

    # Create output directories
    log_dir = Path(config['logging']['log_dir']) / config['logging']['experiment_name']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(log_dir / 'tensorboard')

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
    if args.train_pairs and args.val_pairs:
        from dataloaders.reg_pairs_dataset import create_loaders_from_pairs
        train_loader, val_loader, _ = create_loaders_from_pairs(
            args.train_pairs,
            args.val_pairs,
            batch_size=config['data']['batch_size'],
            patch_size=config['data']['patch_size'],
            patches_per_pair=config['data']['patches_per_pair'],
            num_workers=config['data']['num_workers'],
            patch_stride=config['data']['patch_stride'],
            use_labels_train=True,
            use_labels_val=True,
            target_size=tuple(config['model']['image_size']),
        )
    else:
        from dataloaders.oasis_dataset import create_oasis_loaders
        train_loader, val_loader = create_oasis_loaders(
            data_root=config['data']['data_root'],
            batch_size=config['data']['batch_size'],
            patch_size=config['data']['patch_size'],
            patch_stride=config['data']['patch_stride'],
            patches_per_pair=config['data']['patches_per_pair'],
            num_workers=config['data']['num_workers']
        )

    # If small_dataset flag is set, use only 1% of the dataset
    if args.small_dataset:
        print("Using small dataset (1 percent of the full dataset)")
        # Create subset of the training data (1%)
        train_size = len(train_loader.dataset)
        small_size = max(1, int(train_size * 0.01))  # At least 1 sample
        indices = torch.randperm(train_size)[:small_size]
        train_subset = torch.utils.data.Subset(train_loader.dataset, indices)
        
        # Create new DataLoader with the subset
        train_loader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        # Also reduce validation set size
        val_size = len(val_loader.dataset)
        small_val_size = max(1, int(val_size * 0.01))
        val_indices = torch.randperm(val_size)[:small_val_size]
        val_subset = torch.utils.data.Subset(val_loader.dataset, val_indices)
        
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    writer.add_scalar('data/train_batches', len(train_loader), 0)
    writer.add_scalar('data/val_batches', len(val_loader), 0)
    
    # Create model
    model = create_model(config)
    model = model.to(device=device, dtype=torch.float32)
    
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
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing to save memory")
        # Enable gradient checkpointing for the model
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder'):
            model.encoder.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for encoder")
        if hasattr(model, 'liquid_core'):
            import torch.utils.checkpoint as cp
            original_forward = model.liquid_core.forward  # this is already a *bound* method

            def checkpointed_forward(self, coords, params):
                # call the bound method without passing `self` again
                return cp.checkpoint(
                    lambda c, p: original_forward(c, p),
                    coords, params,
                    preserve_rng_state=True
                )

            model.liquid_core.forward = types.MethodType(checkpointed_forward, model.liquid_core)
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
        # Mixed precision scaler
        try:
            from torch.amp import GradScaler
        except Exception:
            from torch.cuda.amp import GradScaler

        scaler = GradScaler(device="cuda",enabled=(use_amp and amp_dtype is torch.float16))

    else:
        # For CPU or when AMP is disabled, create a dummy scaler
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
            def unscale_(self, optimizer): pass
            def get_scale(self): return 1.0
        scaler = DummyScaler()
    
    # Training loop
    early_stopping_counter = 0    
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'epoch' in ckpt:
            start_epoch = int(ckpt['epoch']) + 1
        if 'loss' in ckpt:
            best_loss = float(ckpt['loss'])  # treat stored loss as current best
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_loss={best_loss:.4f})")

    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch, writer
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
                model, val_loader, criterion, config, epoch, writer
            )
            
            print(f"Val Loss: {val_losses['avg_loss']:.4f}")
            writer.add_scalar('val/epoch_loss', val_losses['avg_loss'], epoch)
            for k, v in val_losses.items():
                if k != 'avg_loss':
                    writer.add_scalar(f'val/epoch_{k}', v, epoch)
            
            # Check for improvement
            current_loss = val_losses['avg_loss']
            is_best = current_loss < best_loss
            
            if is_best:
                best_loss = current_loss
                early_stopping_counter = 0
                print(f"New best validation loss: {best_loss:.4f}")
                writer.add_scalar('val/best_loss', best_loss, epoch)
            else:
                early_stopping_counter += 1
                writer.add_scalar('val/no_improve_streak', early_stopping_counter, epoch)
            
            # Save checkpoint
            if epoch % config['logging']['save_interval'] == 0 or is_best:
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
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses['avg_loss'] if 'val_losses' in locals() else train_losses['avg_loss'])
            else:
                scheduler.step()
        
        # Log learning rate again (post-step)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate_post_step', current_lr, epoch)
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    save_checkpoint(
        model, optimizer, scheduler, epoch, 
        val_losses['avg_loss'] if 'val_losses' in locals() else train_losses['avg_loss'], 
        config, str(final_path), writer=writer
    )
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main()
