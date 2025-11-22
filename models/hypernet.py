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
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 50000,
        num_layers: int = 2,
        dropout: float = 0.0,
        param_scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.param_scale = param_scale
        
        if dropout > 0:
            import warnings
            warnings.warn(
                "Dropout in HyperNet is not recommended with batch_size=1. "
                "Set dropout=0 or increase batch_size if using dropout.",
                UserWarning
            )
        
        layers = []
        in_features = input_dim
        
        for i in range(num_layers - 1):
            layer_components = [
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layer_components.append(nn.Dropout(dropout))
            layers.extend(layer_components)
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        params = self.mlp(features)
        params = torch.tanh(params * self.param_scale)
        return params


class FeatureFusion(nn.Module):
    """
    Module for fusing features from fixed and moving images.
    
    WARNING: fusion_type="attention" is computationally infeasible for large spatial
    dimensions (e.g., > 10^4 tokens). Use only with very small feature maps.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        fusion_type: str = "concat_pool",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            import warnings
            warnings.warn(
                "Attention fusion is computationally expensive (O(N^2) memory). "
                "Only use with very small spatial feature maps (< 10^4 tokens). "
                "For typical encoders, use 'concat_pool' or 'gated' instead.",
                UserWarning
            )
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
            feat_fixed_flat = feat_fixed.flatten(2).permute(0, 2, 1)
            feat_moving_flat = feat_moving.flatten(2).permute(0, 2, 1)
            
            N = feat_fixed_flat.shape[1]
            if N > 10000:
                raise RuntimeError(
                    f"Attention fusion with {N} tokens is computationally infeasible. "
                    f"Use 'concat_pool' or 'gated' fusion instead, or reduce encoder spatial output."
                )
            
            attended, _ = self.attention(
                feat_fixed_flat, 
                feat_moving_flat, 
                feat_moving_flat
            )
            attended = self.norm(attended + feat_fixed_flat)
            fused = attended.mean(dim=1)
        
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