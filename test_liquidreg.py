#!/usr/bin/env python3
"""
Test script for LiquidReg model.
"""

import torch
import torch.nn as nn
from models.liquidreg import LiquidReg, LiquidRegLite
from losses.registration_losses import CompositeLoss


def test_liquidreg():
    """Test LiquidReg model with different image sizes."""
    print("Testing LiquidReg model...")
    
    # Test with different patch sizes
    patch_sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    
    # Create model
    model = LiquidReg(
        image_size=(128, 128, 128),  # Default size, should adapt to inputs
        encoder_type="cnn",
        encoder_channels=256,
        liquid_hidden_dim=64,
        liquid_num_steps=8,
        velocity_scale=10.0,
        num_squaring=6,
        fusion_type="concat_pool",
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LiquidReg total parameters: {total_params:,}")
    
    # Create loss function
    criterion = CompositeLoss(
        similarity_loss="lncc",
        lambda_similarity=1.0,
        lambda_jacobian=1.0,
        lambda_velocity=0.01,
        lambda_liquid=0.001,
        lncc_window=9,
        jacobian_penalty="l2"
    )
    
    # Test with different patch sizes
    for D, H, W in patch_sizes:
        print(f"\nTesting with patch size: {D}x{H}x{W}")
        
        # Create random input tensors
        fixed = torch.randn(2, 1, D, H, W)
        moving = torch.randn(2, 1, D, H, W)
        
        # Forward pass
        output = model(fixed, moving, return_intermediate=True)
        
        # Check output shapes
        warped = output['warped_moving']
        deformation = output['deformation_field']
        velocity = output['velocity_field']
        jacobian_det = output['jacobian_det']
        
        print(f"Input shape: {fixed.shape}")
        print(f"Warped shape: {warped.shape}")
        print(f"Deformation field shape: {deformation.shape}")
        print(f"Velocity field shape: {velocity.shape}")
        print(f"Jacobian determinant shape: {jacobian_det.shape}")
        
        # Compute loss
        losses = criterion(
            fixed=fixed,
            warped=warped,
            velocity_field=velocity,
            jacobian_det=jacobian_det,
            liquid_params=output['liquid_params']
        )
        
        print(f"Loss: {losses['total'].item():.4f}")
        
        # Test backpropagation
        losses['total'].backward()
        print("Backpropagation successful!")


def test_liquidreg_lite():
    """Test LiquidRegLite model with different image sizes."""
    print("\nTesting LiquidRegLite model...")
    
    # Create model
    model = LiquidRegLite(
        image_size=(64, 64, 64),  # Default size, should adapt to inputs
        encoder_channels=128,
        liquid_hidden_dim=32,
        liquid_num_steps=4,
        velocity_scale=5.0,
        num_squaring=4,
    )
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LiquidRegLite total parameters: {total_params:,}")
    
    # Test with a single patch size
    D, H, W = 32, 32, 32
    print(f"\nTesting with patch size: {D}x{H}x{W}")
    
    # Create random input tensors
    fixed = torch.randn(2, 1, D, H, W)
    moving = torch.randn(2, 1, D, H, W)
    
    # Forward pass
    output = model(fixed, moving, return_intermediate=True)
    
    # Check output shapes
    warped = output['warped_moving']
    deformation = output['deformation_field']
    velocity = output['velocity_field']
    
    print(f"Input shape: {fixed.shape}")
    print(f"Warped shape: {warped.shape}")
    print(f"Deformation field shape: {deformation.shape}")
    print(f"Velocity field shape: {velocity.shape}")
    
    # Test backpropagation
    loss = torch.mean((fixed - warped) ** 2)
    loss.backward()
    print("Backpropagation successful!")


if __name__ == "__main__":
    test_liquidreg()
    test_liquidreg_lite()
