#!/usr/bin/env python3
"""
Inference script for LiquidReg model.
"""

import argparse
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.liquidreg import LiquidReg
from utils.preprocessing import normalize_volume, resample_volume, pad_or_crop_to_shape


def load_nifti_volume(filepath: str) -> torch.Tensor:
    """Load NIfTI volume and convert to tensor."""
    nii = nib.load(filepath)
    volume = nii.get_fdata().astype(np.float32)
    
    # Add batch and channel dimensions
    volume = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    
    return volume, nii.affine, nii.header


def save_nifti_volume(
    volume: torch.Tensor, 
    filepath: str, 
    affine: np.ndarray, 
    header=None
):
    """Save tensor as NIfTI volume."""
    # Remove batch and channel dimensions
    if volume.dim() == 5:
        volume = volume.squeeze(0).squeeze(0)
    elif volume.dim() == 4:
        volume = volume.squeeze(0)
    
    volume_np = volume.cpu().numpy()
    
    # Create NIfTI image
    nii = nib.Nifti1Image(volume_np, affine, header)
    nib.save(nii, filepath)


def load_model(checkpoint_path: str, device: torch.device) -> LiquidReg:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration from checkpoint
    config = checkpoint['config']
    model_config = config['model']
    
    # Create model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def preprocess_volume(
    volume: torch.Tensor, 
    target_size: tuple = (128, 128, 128)
) -> torch.Tensor:
    """Preprocess volume for inference."""
    # Normalize
    volume = normalize_volume(volume, method="zscore")
    
    # Resize to target size
    volume = resample_volume(volume, target_size, mode="trilinear")
    
    return volume


def register_images(
    fixed_path: str,
    moving_path: str,
    model_path: str,
    output_dir: str,
    device: torch.device
):
    """Register two images using LiquidReg."""
    
    # Load images
    print("Loading images...")
    fixed_volume, fixed_affine, fixed_header = load_nifti_volume(fixed_path)
    moving_volume, moving_affine, moving_header = load_nifti_volume(moving_path)
    
    print(f"Fixed image shape: {fixed_volume.shape}")
    print(f"Moving image shape: {moving_volume.shape}")
    
    # Preprocess
    print("Preprocessing...")
    original_size = fixed_volume.shape[2:]
    target_size = (128, 128, 128)  # Model input size
    
    fixed_preprocessed = preprocess_volume(fixed_volume, target_size)
    moving_preprocessed = preprocess_volume(moving_volume, target_size)
    
    # Move to device
    fixed_preprocessed = fixed_preprocessed.to(device)
    moving_preprocessed = moving_preprocessed.to(device)
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Perform registration
    print("Performing registration...")
    with torch.no_grad():
        output = model(fixed_preprocessed, moving_preprocessed, return_intermediate=True)
    
    warped_moving = output['warped_moving']
    deformation_field = output['deformation_field']
    velocity_field = output['velocity_field']
    
    # Resize outputs back to original size
    print("Post-processing...")
    warped_moving_resized = resample_volume(warped_moving, original_size, mode="trilinear")
    deformation_resized = resample_volume(deformation_field, original_size, mode="trilinear")
    
    # Scale deformation field
    for i in range(3):
        deformation_resized[:, i] *= original_size[i] / target_size[i]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results
    print("Saving results...")
    
    # Warped moving image
    save_nifti_volume(
        warped_moving_resized,
        str(output_path / "warped_moving.nii.gz"),
        fixed_affine,
        fixed_header
    )
    
    # Deformation field components
    for i, dim in enumerate(['x', 'y', 'z']):
        save_nifti_volume(
            deformation_resized[:, i:i+1],
            str(output_path / f"deformation_{dim}.nii.gz"),
            fixed_affine,
            fixed_header
        )
    
    # Compute and save Jacobian determinant
    from models.scaling_squaring import compute_jacobian_determinant
    jacobian_det = compute_jacobian_determinant(deformation_resized)
    save_nifti_volume(
        jacobian_det,
        str(output_path / "jacobian_determinant.nii.gz"),
        fixed_affine,
        fixed_header
    )
    
    print(f"Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n=== Registration Summary ===")
    print(f"Deformation magnitude (mean): {deformation_resized.norm(dim=1).mean().item():.3f}")
    print(f"Deformation magnitude (max): {deformation_resized.norm(dim=1).max().item():.3f}")
    print(f"Jacobian determinant (mean): {jacobian_det.mean().item():.3f}")
    print(f"Jacobian determinant (min): {jacobian_det.min().item():.3f}")
    print(f"Negative Jacobian ratio: {(jacobian_det < 0).float().mean().item():.3f}")


def main():
    parser = argparse.ArgumentParser(description='LiquidReg Inference')
    parser.add_argument('--fixed', type=str, required=True,
                       help='Path to fixed image (NIfTI)')
    parser.add_argument('--moving', type=str, required=True,
                       help='Path to moving image (NIfTI)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Perform registration
    register_images(
        fixed_path=args.fixed,
        moving_path=args.moving,
        model_path=args.model,
        output_dir=args.output,
        device=device
    )


if __name__ == '__main__':
    main() 