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

        
        # Compute raw time constants
        tau_raw = torch.matmul(h, self.W_tau.T) + torch.matmul(u, self.U_tau.T) + self.b_tau
        
        # Apply sigmoid to bound tau in [min_tau, max_tau]
        tau = self.min_tau + torch.sigmoid(tau_raw) * (self.max_tau - self.min_tau)
        

        
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
        try:
            tau = self.compute_tau(h, u)
            
            # Clip extremely small time constants to avoid division by zero
            tau = torch.clamp(tau, min=1e-6)
            
            # Compute cell dynamics
            pre_activation = torch.matmul(h, self.W_h.T) + torch.matmul(u, self.U_h.T) + self.b_h
            f_h = torch.sigmoid(pre_activation)
            

            
            # Update hidden state using liquid dynamics
            h_dot = -h / tau + f_h
            h_new = h + dt * h_dot
            
            # Check for NaNs and fix them
            h_new = torch.nan_to_num(h_new, nan=0.0)
            
            # Compute velocity output
            v_pre = torch.matmul(h_new, self.W_out.T) + self.b_out
            v = torch.tanh(v_pre)
            

            
            return h_new, v
        except Exception as e:
            print(f"[ERROR] Exception in LiquidCell3D forward: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros if exception occurs
            return torch.zeros_like(h), torch.zeros(h.shape[0], h.shape[1], 3, device=h.device, dtype=h.dtype)
    
    def _unpack(self, params: torch.Tensor):
        if params.dim() == 1:
            params = params.unsqueeze(0)
        B = params.shape[0]
        idx = 0
        def take(n):
            nonlocal idx
            out = params[:, idx:idx+n]
            idx += n
            return out

        n_Wh = self.hidden_dim * self.hidden_dim
        n_Uh = self.hidden_dim * self.input_dim
        n_bh = self.hidden_dim
        n_Wt = self.hidden_dim * self.hidden_dim
        n_Ut = self.hidden_dim * self.input_dim
        n_bt = self.hidden_dim
        n_Wo = 3 * self.hidden_dim
        n_bo = 3

        W_h = take(n_Wh).view(B, self.hidden_dim, self.hidden_dim)
        U_h = take(n_Uh).view(B, self.hidden_dim, self.input_dim)
        b_h = take(n_bh)

        W_tau = take(n_Wt).view(B, self.hidden_dim, self.hidden_dim)
        U_tau = take(n_Ut).view(B, self.hidden_dim, self.input_dim)
        b_tau = take(n_bt)

        W_out = take(n_Wo).view(B, 3, self.hidden_dim)
        b_out = take(n_bo)

        return W_h, U_h, b_h, W_tau, U_tau, b_tau, W_out, b_out

    def forward_functional(self, h: torch.Tensor, u: torch.Tensor, params: torch.Tensor, dt: float = 0.1):
        W_h, U_h, b_h, W_tau, U_tau, b_tau, W_out, b_out = self._unpack(params)
        tau_raw = torch.einsum('bnh,bhk->bnk', h, W_tau) + torch.einsum('bni,bki->bnk', u, U_tau) + b_tau.unsqueeze(1)
        tau = self.min_tau + torch.sigmoid(tau_raw) * (self.max_tau - self.min_tau)
        tau = torch.clamp(tau, min=1e-6)
        pre = torch.einsum('bnh,bhk->bnk', h, W_h) + torch.einsum('bni,bki->bnk', u, U_h) + b_h.unsqueeze(1)
        f_h = torch.sigmoid(pre)
        h_dot = -h / tau + f_h
        h_new = h + dt * h_dot
        h_new = torch.nan_to_num(h_new, nan=0.0)
        v_pre = torch.einsum('bnh,bkh->bnk', h_new, W_out) + b_out.unsqueeze(1)
        v = torch.tanh(v_pre)
        return h_new, v

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
        
        if N > 1_000_000:
            raise RuntimeError(
                f"LiquidODECore expects patches, not full volumes. "
                f"Received volume size {D}x{H}x{W} = {N} voxels. "
                f"Use patch-based training with smaller spatial dimensions."
            )
        
        # Reshape spatial coordinates
        coords = spatial_coords.view(B, N, 3)
        
        # Initialize hidden state
        h = torch.zeros(B, N, self.hidden_dim, device=coords.device, dtype=coords.dtype)
        
        # Run liquid dynamics
        velocity = None
        for step in range(self.num_steps):
            try:
                h, v = self.cell.forward_functional(h, coords, params, dt=self.dt)
                
                h = torch.nan_to_num(h, nan=0.0)
                v = torch.nan_to_num(v, nan=0.0)
                _soft = 50.0
                h = _soft * torch.tanh(h / _soft)
                v = _soft * torch.tanh(v / _soft)
                
                velocity = v
            except Exception as e:
                print(f"[ERROR] Exception in liquid dynamics step {step}: {e}")
                import traceback
                traceback.print_exc()
                return torch.zeros(B, 3, D, H, W, device=coords.device, dtype=coords.dtype)
        
        if velocity is None:
            return torch.zeros(B, 3, D, H, W, device=coords.device, dtype=coords.dtype)
        
        # Scale and reshape
        velocity = velocity * self.velocity_scale
        velocity_field = velocity.view(B, D, H, W, 3).permute(0, 4, 1, 2, 3)
        

        
        return velocity_field
    
    def get_param_count(self) -> int:
        """Get total number of parameters in the cell."""
        return sum(p.numel() for p in self.cell.parameters()) 