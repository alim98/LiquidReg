"""
Hyper-network module for generating Liquid ODE parameters from image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HyperNet(nn.Module):
    """
    Hyper-network that generates all parameters for the Liquid ODE core
    based on the concatenated features of the fixed and moving images.
    """
    
    def __init__(
        self,
        input_dim: int = 512,  # Concatenated feature dimension
        hidden_dim: int = 256,
        output_dim: int = 50000,  # Approximate parameter count for Liquid ODE
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        in_features = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(in_features, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate parameters from concatenated image features.
        
        Args:
            features: Concatenated features (B, input_dim)
            
        Returns:
            params: Generated parameters (B, output_dim)
        """
        # Generate parameters
        params = self.mlp(features)
        
        # Scale parameters to reasonable range
        params = torch.tanh(params * 0.1)
        
        return params


class FeatureFusion(nn.Module):
    """
    Module for fusing features from fixed and moving images.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        fusion_type: str = "concat_pool",  # Options: concat_pool, attention, gated
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            )
            self.norm = nn.LayerNorm(feature_dim)
        
        elif fusion_type == "gated":
            self.gate_fixed = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
            self.gate_moving = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        feat_fixed: torch.Tensor, 
        feat_moving: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from fixed and moving images.
        
        Args:
            feat_fixed: Fixed image features (B, C, D, H, W)
            feat_moving: Moving image features (B, C, D, H, W)
            
        Returns:
            fused: Fused features (B, fusion_dim)
        """
        B = feat_fixed.shape[0]
        
        if self.fusion_type == "concat_pool":
            # Simple concatenation and global average pooling
            feat_concat = torch.cat([feat_fixed, feat_moving], dim=1)
            fused = F.adaptive_avg_pool3d(feat_concat, 1).view(B, -1)
        
        elif self.fusion_type == "attention":
            # Reshape for attention
            feat_fixed_flat = feat_fixed.flatten(2).permute(0, 2, 1)  # (B, N, C)
            feat_moving_flat = feat_moving.flatten(2).permute(0, 2, 1)
            
            # Cross-attention
            attended, _ = self.attention(
                feat_fixed_flat, 
                feat_moving_flat, 
                feat_moving_flat
            )
            attended = self.norm(attended + feat_fixed_flat)
            
            # Pool
            fused = attended.mean(dim=1)  # (B, C)
        
        elif self.fusion_type == "gated":
            # Gated fusion
            feat_concat = torch.cat([feat_fixed, feat_moving], dim=1)
            gate_f = self.gate_fixed(feat_concat.flatten(2).mean(dim=2))
            gate_m = self.gate_moving(feat_concat.flatten(2).mean(dim=2))
            
            feat_fixed_pool = F.adaptive_avg_pool3d(feat_fixed, 1).view(B, -1)
            feat_moving_pool = F.adaptive_avg_pool3d(feat_moving, 1).view(B, -1)
            
            fused = gate_f * feat_fixed_pool + gate_m * feat_moving_pool
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        return fused 