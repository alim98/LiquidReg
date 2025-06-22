#!/usr/bin/env python3
"""
Test script for dataset and patch extraction.
"""

import torch
import numpy as np
from pathlib import Path
from utils.patch_utils import extract_patches
from dataloaders.oasis_dataset import L2RTask3Dataset

def test_extract_patches():
    """Test patch extraction with different volume sizes."""
    print("Testing patch extraction...")
    
    # Test with different volume sizes
    volume_sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128)]
    patch_sizes = [32, 64]
    
    for D, H, W in volume_sizes:
        for patch_size in patch_sizes:
            print(f"\nTesting with volume size: {D}x{H}x{W}, patch size: {patch_size}")
            
            # Create random volume
            volume = torch.randn(1, 1, D, H, W)
            
            # Extract patches
            patches, coords = extract_patches(
                volume,
                patch_size=patch_size,
                stride=patch_size // 2,
                threshold=0.0  # Accept all patches
            )
            
            print(f"Number of patches extracted: {len(patches)}")
            if patches:
                print(f"Patch shape: {patches[0].shape}")
            
            # Check that at least one patch is extracted
            assert len(patches) > 0, "No patches extracted"
            
            # Check patch shape
            for patch in patches:
                assert patch.shape[-3] == patch_size, f"Incorrect patch depth: {patch.shape[-3]}"
                assert patch.shape[-2] == patch_size, f"Incorrect patch height: {patch.shape[-2]}"
                assert patch.shape[-1] == patch_size, f"Incorrect patch width: {patch.shape[-1]}"


def create_mock_dataset():
    """Create a mock dataset for testing."""
    # Create temporary directory
    data_dir = Path("temp_data")
    data_dir.mkdir(exist_ok=True)
    
    # Create mock volumes
    subj_dir = data_dir / "OASIS_OAS1_0001_MR1"
    subj_dir.mkdir(exist_ok=True)
    
    # Create random volume data
    D, H, W = 32, 32, 32
    vol_data = np.random.rand(D, H, W).astype(np.float32)
    seg_data = np.zeros((D, H, W), dtype=np.int32)
    seg_data[10:20, 10:20, 10:20] = 1  # Add a segmentation region
    
    # Save as .npy files (simpler than NIfTI for testing)
    np.save(subj_dir / "aligned_norm.npy", vol_data)
    np.save(subj_dir / "aligned_seg35.npy", seg_data)
    
    return data_dir


def test_dataset():
    """Test the dataset with mock data."""
    print("\nTesting dataset...")
    
    # Create mock dataset
    data_dir = create_mock_dataset()
    
    try:
        # Create dataset
        dataset = L2RTask3Dataset(
            str(data_dir),
            split="train",
            patch_size=16,
            patch_stride=8,
            patches_per_pair=2,
            augment=False,
            use_labels=True,
        )
        
        print(f"Dataset length: {len(dataset)}")
        
        # Test __getitem__
        for i in range(min(2, len(dataset))):
            print(f"\nGetting item {i}...")
            item = dataset[i]
            
            print(f"Fixed shape: {item['fixed'].shape}")
            print(f"Moving shape: {item['moving'].shape}")
            
            # Check shapes
            assert item['fixed'].shape == item['moving'].shape, "Fixed and moving shapes don't match"
            
            if 'segmentation_fixed' in item:
                print(f"Segmentation fixed shape: {item['segmentation_fixed'].shape}")
                print(f"Segmentation moving shape: {item['segmentation_moving'].shape}")
                
                # Check shapes
                assert item['segmentation_fixed'].shape == item['segmentation_moving'].shape, \
                    "Fixed and moving segmentation shapes don't match"
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(data_dir)


if __name__ == "__main__":
    test_extract_patches()
    # Uncomment to test dataset with mock data
    # test_dataset() 