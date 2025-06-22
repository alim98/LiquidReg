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
import types

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

    from torch.cuda.amp import autocast as cuda_autocast
    autocast_context = cuda_autocast(enabled=use_amp)
    
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
    parser.add_argument('--small_dataset', action='store_true',
                       help='Use only 1% of the dataset for training (to reduce memory usage)')
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
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
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
    
    # If small_dataset flag is set, use only 1% of the dataset
    if args.small_dataset:
        print("Using small dataset (1% of the full dataset)")
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
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing to save memory")
        # Enable gradient checkpointing for the model
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'encoder'):
            model.encoder.encoder.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for encoder")
        if hasattr(model, 'liquid_core'):
            # Custom gradient checkpointing for LiquidODECore
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Monkey patch the forward method to use checkpointing
            original_forward = model.liquid_core.forward
            
            def checkpointed_forward(self, coords, params):
                return torch.utils.checkpoint.checkpoint(
                    create_custom_forward(original_forward),
                    self, coords, params,
                    preserve_rng_state=True
                )
            
            model.liquid_core.forward = types.MethodType(checkpointed_forward, model.liquid_core)
            print("Gradient checkpointing enabled for liquid_core")
    
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
    # Fix deprecated GradScaler API
    if config['training']['use_amp'] and device.type == 'cuda':
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        # For CPU or when AMP is disabled, create a dummy scaler
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
            def unscale_(self, optimizer): pass
        scaler = DummyScaler()
    
    # Training loop
    early_stopping_counter = 0
    best_loss = float('inf')  # Initialize best loss
    
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