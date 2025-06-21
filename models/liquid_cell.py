"""
Liquid Time-constant Cell (LTC) implementation for 3D registration.
Based on Hasani et al. "Liquid Time-constant Networks" AAAI 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LiquidCell3D(nn.Module):
    """
    Liquid Time-constant Cell for 3D spatial processing.
    
    The cell dynamics are:
    ḣ = -1/τ(h,u) ⊙ h + σ(W_h h + U_h u + b_h)
    
    where τ is a state-dependent time constant.
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # spatial coordinates
        hidden_dim: int = 64,
        min_tau: float = 0.1,
        max_tau: float = 10.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_tau = min_tau
        self.max_tau = max_tau
        
        # Main dynamics weights
        self.W_h = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.U_h = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time constant network
        self.W_tau = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.U_tau = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.b_tau = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output projection
        self.W_out = nn.Parameter(torch.randn(3, hidden_dim) * 0.1)
        self.b_out = nn.Parameter(torch.zeros(3))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate scales."""
        nn.init.xavier_uniform_(self.W_h)
        nn.init.xavier_uniform_(self.U_h)
        nn.init.xavier_uniform_(self.W_tau)
        nn.init.xavier_uniform_(self.U_tau)
        nn.init.xavier_uniform_(self.W_out)
    
    def compute_tau(
        self, 
        h: torch.Tensor, 
        u: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute state-dependent time constants.
        
        Args:
            h: Hidden state (B, N, hidden_dim)
            u: Input (B, N, input_dim)
            
        Returns:
            tau: Time constants (B, N, hidden_dim)
        """
        tau_raw = torch.matmul(h, self.W_tau.T) + torch.matmul(u, self.U_tau.T) + self.b_tau
        tau = self.min_tau + F.softplus(tau_raw) * (self.max_tau - self.min_tau)
        return tau
    
    def forward(
        self,
        h: torch.Tensor,
        u: torch.Tensor,
        dt: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward dynamics step.
        
        Args:
            h: Current hidden state (B, N, hidden_dim)
            u: Input coordinates (B, N, input_dim)
            dt: Time step
            
        Returns:
            h_new: Updated hidden state
            v: Velocity output (B, N, 3)
        """
        # Compute time constants
        tau = self.compute_tau(h, u)
        
        # Compute cell dynamics
        pre_activation = torch.matmul(h, self.W_h.T) + torch.matmul(u, self.U_h.T) + self.b_h
        f_h = torch.sigmoid(pre_activation)
        
        # Update hidden state using liquid dynamics
        h_dot = -h / tau + f_h
        h_new = h + dt * h_dot
        
        # Compute velocity output
        v = torch.tanh(torch.matmul(h_new, self.W_out.T) + self.b_out)
        
        return h_new, v
    
    def set_params_from_vector(self, params: torch.Tensor):
        """
        Set all parameters from a flat parameter vector.
        Used for hyper-network conditioning.
        """
        # Handle batch dimension - use first sample if batched
        if params.dim() > 1:
            params = params[0]
            
        idx = 0
        param_shapes = {
            'W_h': (self.hidden_dim, self.hidden_dim),
            'U_h': (self.hidden_dim, self.input_dim),
            'b_h': (self.hidden_dim,),
            'W_tau': (self.hidden_dim, self.hidden_dim),
            'U_tau': (self.hidden_dim, self.input_dim),
            'b_tau': (self.hidden_dim,),
            'W_out': (3, self.hidden_dim),
            'b_out': (3,)
        }
        
        for name, shape in param_shapes.items():
            param = getattr(self, name)
            numel = param.numel()
            if idx + numel <= params.numel():
                param.data = params[idx:idx+numel].view(shape)
                idx += numel
            else:
                print(f"Warning: Not enough parameters for {name}, skipping")


class LiquidODECore(nn.Module):
    """
    Multi-step Liquid ODE core for velocity field generation.
    """
    
    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 64,
        num_steps: int = 8,
        dt: float = 0.125,  # 1/8
        velocity_scale: float = 10.0,
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.dt = dt
        self.velocity_scale = velocity_scale
        
        # Single liquid cell (parameters will be set by hypernetwork)
        self.cell = LiquidCell3D(
            input_dim=spatial_dim,
            hidden_dim=hidden_dim
        )
        
    def forward(
        self, 
        spatial_coords: torch.Tensor,
        params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate velocity field using liquid dynamics.
        
        Args:
            spatial_coords: Normalized spatial coordinates (B, D, H, W, 3)
            params: Optional parameter vector from hypernetwork
            
        Returns:
            velocity_field: Output velocity field (B, 3, D, H, W)
        """
        B, D, H, W, _ = spatial_coords.shape
        N = D * H * W
        
        # Reshape spatial coordinates
        coords = spatial_coords.view(B, N, 3)
        
        # Set parameters if provided
        if params is not None:
            self.cell.set_params_from_vector(params)
        
        # Initialize hidden state
        h = torch.zeros(B, N, self.hidden_dim, device=coords.device, dtype=coords.dtype)
        
        # Run liquid dynamics
        velocities = []
        for step in range(self.num_steps):
            h, v = self.cell(h, coords, dt=self.dt)
            velocities.append(v)
        
        # Use final velocity
        velocity = velocities[-1]
        
        # Scale and reshape
        velocity = velocity * self.velocity_scale
        velocity_field = velocity.view(B, D, H, W, 3).permute(0, 4, 1, 2, 3)
        
        return velocity_field
    
    def get_param_count(self) -> int:
        """Get total number of parameters in the cell."""
        return sum(p.numel() for p in self.cell.parameters()) 