#!/usr/bin/env python3
"""
Training script for LiquidReg model.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.liquidreg import LiquidReg, LiquidRegLite
from dataloaders.oasis_dataset import create_oasis_loaders
from losses.registration_losses import CompositeLoss
from utils.preprocessing import normalize_volume


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
    
    for batch_idx, batch in enumerate(pbar):
        fixed = batch['fixed'].float()
        moving = batch['moving'].float()
        
        # Move to device
        device = next(model.parameters()).device
        fixed = fixed.to(device)
        moving = moving.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        use_amp = config['training']['use_amp']
        with autocast(enabled=use_amp):
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
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip_norm']
            )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training']['grad_clip_norm']
            )
            
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for key, value in losses.items():
            if key not in total_losses:
                total_losses[key] = 0.0
            total_losses[key] += value.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'grad': f"{grad_norm:.2f}"
        })
        
        # Log to tensorboard
        if batch_idx % config['logging']['log_interval'] == 0:
            step = epoch * num_batches + batch_idx
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/grad_norm', grad_norm, step)
            
            for key, value in losses.items():
                writer.add_scalar(f'train/{key}', value.item(), step)
    
    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    avg_losses['avg_loss'] = total_loss / num_batches
    
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
            fixed = fixed.to(device)
            moving = moving.to(device)
            
            # Forward pass
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
            
            # Accumulate losses
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    # Average losses
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}
    avg_losses['avg_loss'] = total_loss / num_batches
    
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
    is_best: bool = False
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


def main():
    parser = argparse.ArgumentParser(description='Train LiquidReg model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default=None,
                       help='Override data root directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data root if provided
    if args.data_root:
        config['data']['data_root'] = args.data_root
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    log_dir = Path(config['logging']['log_dir']) / config['logging']['experiment_name']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(log_dir / 'tensorboard')
    
    # Create data loaders
    train_loader, val_loader = create_oasis_loaders(
        data_root=config['data']['data_root'],
        batch_size=config['data']['batch_size'],
        patch_size=config['data']['patch_size'],
        patch_stride=config['data']['patch_stride'],
        patches_per_pair=config['data']['patches_per_pair'],
        num_workers=config['data']['num_workers']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    scaler = GradScaler(enabled=config['training']['use_amp'])
    
    # Training loop
    early_stopping_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config, epoch, writer
        )
        
        print(f"Train Loss: {train_losses['avg_loss']:.4f}")
        
        # Validate
        if epoch % config['validation']['eval_interval'] == 0:
            val_losses = validate_epoch(
                model, val_loader, criterion, config, epoch, writer
            )
            
            print(f"Val Loss: {val_losses['avg_loss']:.4f}")
            
            # Check for improvement
            current_loss = val_losses['avg_loss']
            is_best = current_loss < best_loss
            
            if is_best:
                best_loss = current_loss
                early_stopping_counter = 0
                print(f"New best validation loss: {best_loss:.4f}")
            else:
                early_stopping_counter += 1
            
            # Save checkpoint
            if epoch % config['logging']['save_interval'] == 0 or is_best:
                checkpoint_path = checkpoint_dir / f'epoch_{epoch}.pth'
                save_checkpoint(
                    model, optimizer, scheduler, epoch, current_loss, 
                    config, str(checkpoint_path), is_best
                )
            
            # Early stopping
            if (early_stopping_counter >= config['training']['early_stopping_patience'] and
                config['training']['early_stopping_patience'] > 0):
                print(f"Early stopping after {epoch} epochs")
                break
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_losses['avg_loss'] if 'val_losses' in locals() else train_losses['avg_loss'])
            else:
                scheduler.step()
        
        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
    
    # Save final model
    final_path = checkpoint_dir / 'final_model.pth'
    save_checkpoint(
        model, optimizer, scheduler, epoch, 
        val_losses['avg_loss'] if 'val_losses' in locals() else train_losses['avg_loss'], 
        config, str(final_path)
    )
    
    writer.close()
    print("Training completed!")


if __name__ == '__main__':
    main() 