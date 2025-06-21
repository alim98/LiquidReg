#!/usr/bin/env python3
"""
Debug script to check dataset shapes
"""
import torch
import numpy as np
from utils.patch_utils import extract_patches
from utils.preprocessing import normalize_volume

def debug_patch_shapes():
    """Debug the actual shapes in patch extraction process."""
    print("Testing patch extraction shapes...")
    
    # Create mock volume data similar to what OASIS dataset loads
    volume_data = np.random.randn(128, 128, 128).astype(np.float32)
    
    # Convert to tensor and add dimensions as the dataset does
    volume = torch.from_numpy(volume_data).float().unsqueeze(0).unsqueeze(0)
    print(f"Original volume shape: {volume.shape}")  # Should be (1, 1, 128, 128, 128)
    
    # Normalize (as the dataset does)
    volume = normalize_volume(volume)
    print(f"Normalized volume shape: {volume.shape}")
    print(f"Volume mean: {volume.mean().item():.4f}, std: {volume.std().item():.4f}")
    print(f"Volume min: {volume.min().item():.4f}, max: {volume.max().item():.4f}")
    
    # Extract patches with lower threshold
    patches, coords = extract_patches(volume, patch_size=64, stride=32, threshold=0.0)
    
    print(f"Number of patches extracted: {len(patches)}")
    
    if patches:
        print(f"First patch shape: {patches[0].shape}")
        
        # Test what happens with squeeze(0)
        squeezed_patch = patches[0].squeeze(0)
        print(f"After squeeze(0): {squeezed_patch.shape}")
        
        # Test what DataLoader would see
        batch_fixed = torch.stack([squeezed_patch, squeezed_patch])  # Simulate batch
        print(f"Simulated batch shape: {batch_fixed.shape}")
        
        # Test if we need to add channel dimension back
        if len(squeezed_patch.shape) == 3:  # (D, H, W)
            with_channel = squeezed_patch.unsqueeze(0)  # Add channel dim
            print(f"With channel dimension added back: {with_channel.shape}")
            
            batch_with_channel = torch.stack([with_channel, with_channel])
            print(f"Batch with proper channels: {batch_with_channel.shape}")
    
        print("\nTesting augmentation...")
        from utils.patch_utils import PatchAugmentor
        
        augmentor = PatchAugmentor()
        
        test_patch = patches[0]  # Should be (1, 64, 64, 64)
        print(f"Input to augmentor: {test_patch.shape}")
        
        augmented = augmentor.augment(test_patch)
        print(f"Augmented shape: {augmented.shape}")
        
        # Test squeeze
        squeezed_aug = augmented.squeeze(0)
        print(f"Augmented after squeeze(0): {squeezed_aug.shape}")
        
        # Test the actual problem case
        print(f"\nTesting potential 2-channel issue:")
        print(f"Original patch: {test_patch.shape}")
        
        # Check if the augmentor might be causing issues
        if augmented.shape[0] != 1:
            print(f"WARNING: Augmented patch has {augmented.shape[0]} channels instead of 1!")
    else:
        print("No patches extracted - checking why...")
        # Test with very small threshold
        patches2, coords2 = extract_patches(volume, patch_size=64, stride=32, threshold=-1.0)
        print(f"With threshold -1.0: {len(patches2)} patches")
        
        if len(patches2) > 0:
            print(f"Sample patch mean: {patches2[0].mean().item():.4f}")

def debug_dataset_output():
    """Debug the actual dataset output shapes."""
    print("Testing fixed dataset output shapes...")
    
    # Simulate the exact process in the dataset
    volume_data = np.random.randn(128, 128, 128).astype(np.float32)
    volume = torch.from_numpy(volume_data).float().unsqueeze(0).unsqueeze(0)
    volume = normalize_volume(volume)
    
    # Extract patches
    patches, coords = extract_patches(volume, patch_size=64, stride=32, threshold=0.0)
    
    if patches:
        fixed_patch = patches[0]  # (1, 64, 64, 64)
        moving_patch = patches[1] if len(patches) > 1 else patches[0]
        
        print(f"Fixed patch shape: {fixed_patch.shape}")
        print(f"Moving patch shape: {moving_patch.shape}")
        
        # Test what the dataset now returns (without squeeze)
        output = {
            "fixed": fixed_patch,  # Should be (1, 64, 64, 64)
            "moving": moving_patch,  # Should be (1, 64, 64, 64)
        }
        
        print(f"Dataset output fixed shape: {output['fixed'].shape}")
        print(f"Dataset output moving shape: {output['moving'].shape}")
        
        # Test what DataLoader would create
        batch_fixed = torch.stack([output['fixed'], output['fixed']])
        batch_moving = torch.stack([output['moving'], output['moving']])
        
        print(f"DataLoader batch fixed shape: {batch_fixed.shape}")
        print(f"DataLoader batch moving shape: {batch_moving.shape}")
        
        # Test with model
        print("\nTesting with model...")
        from models.liquidreg import LiquidReg
        
        model = LiquidReg(
            image_size=(64, 64, 64),
            encoder_type="cnn",
            encoder_channels=256,
        )
        
        print(f"Model expects input shape: (B, 1, D, H, W)")
        print(f"We're providing: {batch_fixed.shape}")
        
        if batch_fixed.shape[1] == 1:
            print("✅ Channel dimensions match!")
            
            # Test forward pass
            try:
                with torch.no_grad():
                    output = model(batch_fixed, batch_moving)
                print("✅ Model forward pass successful!")
                print(f"Output warped shape: {output['warped_moving'].shape}")
            except Exception as e:
                print(f"❌ Model forward pass failed: {e}")
        else:
            print(f"❌ Channel mismatch! Expected 1, got {batch_fixed.shape[1]}")

if __name__ == "__main__":
    debug_patch_shapes()
    debug_dataset_output() 