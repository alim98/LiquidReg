#!/usr/bin/env python3
"""
Simple test script to verify LiquidReg implementation.
"""

import torch
import numpy as np
from models import LiquidReg, LiquidRegLite
from losses import CompositeLoss
from utils import normalize_volume


def test_model_creation():
    """Test model creation and parameter counting."""
    print("Testing model creation...")
    
    # Test LiquidReg
    model = LiquidReg(
        image_size=(64, 64, 64),
        encoder_channels=128,
        liquid_hidden_dim=32,
        liquid_num_steps=4,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LiquidReg parameters: {total_params:,}")
    
    # Test LiquidRegLite
    model_lite = LiquidRegLite(
        image_size=(64, 64, 64),
        encoder_channels=64,
        liquid_hidden_dim=16,
        liquid_num_steps=4,
    )
    
    total_params_lite = sum(p.numel() for p in model_lite.parameters())
    print(f"LiquidRegLite parameters: {total_params_lite:,}")
    
    assert total_params > 0, "Model has no parameters"
    assert total_params_lite < total_params, "Lite model should have fewer parameters"
    print("✓ Model creation test passed")


def test_forward_pass():
    """Test forward pass with synthetic data."""
    print("\nTesting forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = LiquidReg(
        image_size=(64, 64, 64),
        encoder_channels=128,
        liquid_hidden_dim=32,
        liquid_num_steps=4,
    ).to(device)
    
    # Create synthetic data
    fixed = torch.randn(1, 1, 64, 64, 64, device=device)
    moving = torch.randn(1, 1, 64, 64, 64, device=device)
    
    # Normalize images
    fixed = normalize_volume(fixed)
    moving = normalize_volume(moving)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(fixed, moving, return_intermediate=True)
    
    print(f"Warped shape: {output['warped_moving'].shape}")
    print(f"Deformation shape: {output['deformation_field'].shape}")
    print("✓ Forward pass test passed")


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create loss function
    criterion = CompositeLoss(
        similarity_loss="lncc",
        lambda_similarity=1.0,
        lambda_jacobian=0.1,
        lambda_velocity=0.01,
        lambda_liquid=0.001,
    )
    
    # Create synthetic data
    batch_size = 1
    fixed = torch.randn(batch_size, 1, 64, 64, 64, device=device, requires_grad=True)
    warped = torch.randn(batch_size, 1, 64, 64, 64, device=device, requires_grad=True)
    velocity = torch.randn(batch_size, 3, 64, 64, 64, device=device, requires_grad=True)
    jacobian_det = torch.ones(batch_size, 1, 62, 62, 62, device=device, requires_grad=True)  # Smaller due to gradient computation
    liquid_params = torch.randn(batch_size, 1000, device=device, requires_grad=True)
    
    # Compute losses
    losses = criterion(
        fixed=fixed,
        warped=warped,
        velocity_field=velocity,
        jacobian_det=jacobian_det,
        liquid_params=liquid_params,
    )
    
    # Check loss components
    assert 'total' in losses, "Missing total loss"
    assert 'similarity' in losses, "Missing similarity loss"
    assert 'jacobian' in losses, "Missing jacobian loss"
    assert 'velocity' in losses, "Missing velocity loss"
    assert 'liquid' in losses, "Missing liquid loss"
    
    total_loss = losses['total']
    assert torch.isfinite(total_loss), "Total loss is not finite"
    assert total_loss.requires_grad, "Loss should require gradients"
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Similarity loss: {losses['similarity'].item():.4f}")
    print(f"Jacobian loss: {losses['jacobian'].item():.4f}")
    print(f"Velocity loss: {losses['velocity'].item():.4f}")
    print(f"Liquid loss: {losses['liquid'].item():.4f}")
    
    print("✓ Loss computation test passed")


def test_gradient_flow():
    """Test gradient computation and backpropagation."""
    print("\nTesting gradient flow...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss
    model = LiquidReg(
        image_size=(32, 32, 32),  # Smaller for faster computation
        encoder_channels=64,
        liquid_hidden_dim=16,
        liquid_num_steps=2,
    ).to(device)
    
    criterion = CompositeLoss()
    
    # Create synthetic data
    fixed = torch.randn(1, 1, 32, 32, 32, device=device)
    moving = torch.randn(1, 1, 32, 32, 32, device=device)
    
    # Normalize
    fixed = normalize_volume(fixed)
    moving = normalize_volume(moving)
    
    # Forward pass
    output = model(fixed, moving, return_intermediate=True)
    
    # Compute loss
    losses = criterion(
        fixed=fixed,
        warped=output['warped_moving'],
        velocity_field=output['velocity_field'],
        jacobian_det=output['jacobian_det'],
        liquid_params=output['liquid_params'],
    )
    
    # Backward pass
    total_loss = losses['total']
    total_loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm == 0:
                print(f"Warning: Zero gradient for {name}")
    
    assert len(grad_norms) > 0, "No gradients computed"
    avg_grad_norm = np.mean(grad_norms)
    print(f"Average gradient norm: {avg_grad_norm:.4f}")
    print(f"Parameters with gradients: {len(grad_norms)}")
    
    print("✓ Gradient flow test passed")


def main():
    """Run all tests."""
    print("Running LiquidReg tests...\n")
    
    try:
        test_model_creation()
        test_forward_pass()
        test_loss_computation()
        
        print("\n🎉 All tests passed! LiquidReg is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == '__main__':
    main() 