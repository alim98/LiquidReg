"""
3D Encoder models for LiquidReg: CNN and Swin-ViT options.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class Conv3DBlock(nn.Module):
    """Basic 3D convolution block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride, 
            padding, 
            bias=False
        )
        
        if norm == "batch":
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm == "group":
            self.norm = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        else:
            self.norm = nn.Identity()
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNN3DEncoder(nn.Module):
    """
    Lightweight 3D CNN encoder for extracting features.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 4,
        blocks_per_level: int = 2,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        current_channels = in_channels
        
        for level in range(num_levels):
            # Calculate output channels
            out_channels = base_channels * (2 ** level)
            
            # Create encoder blocks for this level
            level_blocks = []
            for block in range(blocks_per_level):
                level_blocks.append(
                    Conv3DBlock(
                        current_channels,
                        out_channels,
                        norm=norm,
                        activation=activation,
                        dropout=dropout if block == blocks_per_level - 1 else 0.0,
                    )
                )
                current_channels = out_channels
            
            self.encoders.append(nn.Sequential(*level_blocks))
            
            # Add downsampler (except for last level)
            if level < num_levels - 1:
                self.downsamplers.append(
                    nn.Conv3d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False
                    )
                )
        
        self.output_channels = current_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input image (B, C, D, H, W)
            
        Returns:
            features: Encoded features at coarsest level (B, C_out, D/8, H/8, W/8)
        """
        for level in range(self.num_levels):
            x = self.encoders[level](x)
            
            if level < self.num_levels - 1:
                x = self.downsamplers[level](x)
        
        return x


class SimpleCNN3DEncoder(nn.Module):
    """
    Simplified 3D CNN encoder for computational efficiency.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            if i == 0:
                next_channels = 32
            elif i == num_layers - 1:
                next_channels = out_channels
            else:
                next_channels = min(64 * (2 ** (i-1)), 256)
            
            layers.extend([
                nn.Conv3d(current_channels, next_channels, 3, 2, 1),
                nn.BatchNorm3d(next_channels),
                nn.ReLU(inplace=True),
            ])

            current_channels = next_channels
        
        self.encoder = nn.Sequential(*layers)
        self.output_channels = out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding for Swin-ViT."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96,
        norm: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        self.norm = nn.LayerNorm(embed_dim) if norm else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            patches: Patch embeddings (B, N, embed_dim)
        """
        # Project patches
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        
        # Flatten spatial dimensions
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Apply normalization
        x = self.norm(x)
        
        return x


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer block with window attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor, mask_matrix=None) -> torch.Tensor:
        B, L, C = x.shape
        D = H = W = int(round(L ** (1/3)))
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Cyclic shift if needed
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=mask_matrix)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, *self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, D, H, W)
        
        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        
        x = x.view(B, D * H * W, C)
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class WindowAttention3D(nn.Module):
    """3D window-based multi-head attention."""
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int],
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Mlp(nn.Module):
    """Multi-layer perceptron."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths per sample for regularization."""
    
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """Partition tensor into windows."""
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], D: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partitioning."""
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x 