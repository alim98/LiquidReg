"""
Main LiquidReg model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .encoders import SimpleCNN3DEncoder
from .hypernet import HyperNet, FeatureFusion
from .liquid_cell import LiquidODECore
from .scaling_squaring import ScalingSquaring, SpatialTransformer


class LiquidReg(nn.Module):
    """
    LiquidReg: Adaptive ODE Engine for Deformable Medical Image Registration.
    
    Combines:
    1. Liquid Time-constant Networks (LTC) for dynamics
    2. Diffeomorphic transformations via scaling & squaring
    3. Hyper-networks for parameter generation
    4. Continuous-time ODE solvers
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (128, 128, 128),
        encoder_type: str = "cnn",  # "cnn" or "swin"
        encoder_channels: int = 256,
        liquid_hidden_dim: int = 64,
        liquid_num_steps: int = 8,
        velocity_scale: float = 10.0,
        num_squaring: int = 6,
        fusion_type: str = "concat_pool",
    ):
        super().__init__()
        
        self.image_size = image_size
        self.encoder_type = encoder_type
        
        # Shared encoder
        if encoder_type == "cnn":
            self.encoder = SimpleCNN3DEncoder(
                in_channels=1,
                out_channels=encoder_channels,
                num_layers=4
            )
        else:
            raise NotImplementedError(f"Encoder type {encoder_type} not implemented")
        
        if fusion_type in ["attention", "gated"]:
            assert encoder_channels == self.encoder.output_channels, \
                f"For {fusion_type} fusion, encoder_channels ({encoder_channels}) must match encoder output_channels ({self.encoder.output_channels})"
        
        self.fusion = FeatureFusion(
            feature_dim=encoder_channels,
            fusion_type=fusion_type
        )
        
        if fusion_type == "concat_pool":
            fusion_dim = encoder_channels * 2
        else:
            fusion_dim = encoder_channels
        
        # Liquid ODE core
        self.liquid_core = LiquidODECore(
            spatial_dim=3,
            hidden_dim=liquid_hidden_dim,
            num_steps=liquid_num_steps,
            velocity_scale=velocity_scale
        )
        
        # Hyper-network
        param_count = self.liquid_core.get_param_count()
        self.hyper_net = HyperNet(
            input_dim=fusion_dim,
            hidden_dim=256,
            output_dim=param_count,
            num_layers=2
        )
        
        # Scaling & squaring
        self.scaling_squaring = ScalingSquaring(num_squaring=num_squaring)
        
        # Spatial transformer
        self.spatial_transformer = SpatialTransformer()
        
        self._grid_cache = {}
        
    def _generate_coordinate_grid(self, size: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate normalized coordinate grid for the image (cached per size/device/dtype)."""
        key = (size, str(device), dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]
        
        D, H, W = size
        d = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        h = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        w = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        coord_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        self._grid_cache[key] = coord_grid
        return coord_grid
    
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LiquidReg.
        
        Args:
            fixed: Fixed image (B, 1, D, H, W)
            moving: Moving image (B, 1, D, H, W)
            return_intermediate: Whether to return intermediate results
            
        Returns:
            Dictionary containing:
                - warped_moving: Warped moving image
                - deformation_field: Deformation field
                - velocity_field: Velocity field
                - jacobian_det: Jacobian determinant (if return_intermediate)
        """
        B = fixed.shape[0]
        
        # Get actual image dimensions from input
        _, _, D, H, W = fixed.shape
        
        # Encode images
        feat_fixed = self.encoder(fixed)
        feat_moving = self.encoder(moving)
        
        # Fuse features
        fused_features = self.fusion(feat_fixed, feat_moving)
        
        # Generate parameters via hyper-network
        liquid_params = self.hyper_net(fused_features)
        
        # Generate coordinate grid based on actual input dimensions
        coord_grid = self._generate_coordinate_grid((D, H, W), fixed.device, fixed.dtype)
        coord_grid = coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # Generate velocity field using liquid ODE (already scaled by velocity_scale inside core)
        velocity_field = self.liquid_core(coord_grid, liquid_params)

        # Compute deformation field directly from normalized velocity via scaling & squaring
        deformation_field = self.scaling_squaring(velocity_field)

        
        # Warp moving image
        warped_moving = self.spatial_transformer(moving, deformation_field)
        
        # Prepare output
        output = {
            "warped_moving": warped_moving,
            "deformation_field": deformation_field,
            "velocity_field": velocity_field,
        }
        
        if return_intermediate:
            from .scaling_squaring import (
                compute_jacobian_determinant,
                compute_jacobian_determinant_raw,
                foldings_percent,
            )

            # clamped version — keep for loss
            jacobian_det = compute_jacobian_determinant(deformation_field)
            
            output["jacobian_det"] = jacobian_det

            # unclamped version — for logging only
            with torch.no_grad():
                jacobian_det_raw = compute_jacobian_determinant_raw(deformation_field)
                output["jacobian_det_raw"] = jacobian_det_raw
                # both overall scalar and per-sample vector are handy:
                output["foldings_pct_raw"] = foldings_percent(jacobian_det_raw, per_sample=False)
                output["foldings_pct_raw_per_sample"] = foldings_percent(jacobian_det_raw, per_sample=True)

            output["jacobian_det"] = jacobian_det
            output["features_fixed"] = feat_fixed
            output["features_moving"] = feat_moving
            output["fused_features"] = fused_features
            output["liquid_params"] = liquid_params
        
        return output
    
    def get_deformation_only(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute only the deformation field (for inference).
        
        Args:
            fixed: Fixed image (B, 1, D, H, W)
            moving: Moving image (B, 1, D, H, W)
            
        Returns:
            deformation_field: Deformation field (B, 3, D, H, W)
        """
        with torch.no_grad():
            output = self.forward(fixed, moving, return_intermediate=False)
            return output["deformation_field"]
    
    def warp_image(
        self,
        image: torch.Tensor,
        deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp an image using a precomputed deformation field.
        
        Args:
            image: Image to warp (B, C, D, H, W)
            deformation_field: Deformation field (B, 3, D, H, W)
            
        Returns:
            warped: Warped image (B, C, D, H, W)
        """
        return self.spatial_transformer(image, deformation_field)


class LiquidRegLite(nn.Module):
    """
    Lightweight version of LiquidReg for faster inference.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int, int] = (64, 64, 64),
        encoder_channels: int = 128,
        liquid_hidden_dim: int = 32,
        liquid_num_steps: int = 4,
        velocity_scale: float = 5.0,
        num_squaring: int = 4,
    ):
        super().__init__()
        
        self.image_size = image_size
        
        # Lightweight encoder
        self.encoder = SimpleCNN3DEncoder(
            in_channels=1,
            out_channels=encoder_channels,
            num_layers=3
        )
        
        # Simple feature fusion
        self.fusion = FeatureFusion(
            feature_dim=encoder_channels,
            fusion_type="concat_pool"
        )
        
        # Compact liquid core
        self.liquid_core = LiquidODECore(
            spatial_dim=3,
            hidden_dim=liquid_hidden_dim,
            num_steps=liquid_num_steps,
            velocity_scale=velocity_scale
        )
        
        # Hyper-network
        param_count = self.liquid_core.get_param_count()
        self.hyper_net = HyperNet(
            input_dim=encoder_channels * 2,
            hidden_dim=128,
            output_dim=param_count,
            num_layers=2
        )
        
        # Scaling & squaring
        self.scaling_squaring = ScalingSquaring(num_squaring=num_squaring)
        
        # Spatial transformer
        self.spatial_transformer = SpatialTransformer()
        
        self._grid_cache = {}
    
    def _generate_coordinate_grid(self, size: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate normalized coordinate grid (cached per size/device/dtype)."""
        key = (size, str(device), dtype)
        if key in self._grid_cache:
            return self._grid_cache[key]
        
        D, H, W = size
        d = torch.linspace(-1, 1, D, device=device, dtype=dtype)
        h = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        w = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        coord_grid = torch.stack([grid_w, grid_h, grid_d], dim=-1)
        self._grid_cache[key] = coord_grid
        return coord_grid
    
    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B = fixed.shape[0]
        
        # Get actual image dimensions from input
        _, _, D, H, W = fixed.shape
        
        # Encode and fuse
        feat_fixed = self.encoder(fixed)
        feat_moving = self.encoder(moving)
        fused_features = self.fusion(feat_fixed, feat_moving)
        
        # Generate liquid parameters
        liquid_params = self.hyper_net(fused_features)
        
        # Generate coordinate grid based on actual input dimensions
        coord_grid = self._generate_coordinate_grid((D, H, W), fixed.device, fixed.dtype)
        coord_grid = coord_grid.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        # Generate velocity and deformation fields (velocity already scaled by velocity_scale)
        velocity_field = self.liquid_core(coord_grid, liquid_params)
        deformation_field = self.scaling_squaring(velocity_field)

        
        # Warp image
        warped_moving = self.spatial_transformer(moving, deformation_field)
        
        output = {
            "warped_moving": warped_moving,
            "deformation_field": deformation_field,
            "velocity_field": velocity_field,
        }
        
        if return_intermediate and B > 0:
            from .scaling_squaring import compute_jacobian_determinant
            jacobian_det = compute_jacobian_determinant(deformation_field)
            
            output["jacobian_det"] = jacobian_det
            output["features_fixed"] = feat_fixed
            output["features_moving"] = feat_moving
            output["fused_features"] = fused_features
            output["liquid_params"] = liquid_params
            
        return output 