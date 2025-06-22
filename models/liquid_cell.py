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
        # Debug inputs
        print(f"[DEBUG] compute_tau inputs:")
        print(f"W_tau shape: {self.W_tau.shape}, U_tau shape: {self.U_tau.shape}, b_tau shape: {self.b_tau.shape}")
        print(f"W_tau min/max/mean: {self.W_tau.min().item():.4f}/{self.W_tau.max().item():.4f}/{self.W_tau.mean().item():.4f}")
        print(f"U_tau min/max/mean: {self.U_tau.min().item():.4f}/{self.U_tau.max().item():.4f}/{self.U_tau.mean().item():.4f}")
        print(f"b_tau min/max/mean: {self.b_tau.min().item():.4f}/{self.b_tau.max().item():.4f}/{self.b_tau.mean().item():.4f}")
        
        # Compute raw time constants
        tau_raw = torch.matmul(h, self.W_tau.T) + torch.matmul(u, self.U_tau.T) + self.b_tau
        
        # Debug raw time constants
        print(f"[DEBUG] tau_raw min/max/mean: {tau_raw.min().item():.4f}/{tau_raw.max().item():.4f}/{tau_raw.mean().item():.4f}")
        print(f"tau_raw has NaN: {torch.isnan(tau_raw).any().item()}")
        
        # Check for extreme values
        if tau_raw.abs().max() > 50:
            print(f"[WARNING] Extreme values in tau_raw: {tau_raw.abs().max().item():.4f}")
            tau_raw = torch.clamp(tau_raw, -50, 50)
        
        # Apply softplus and scale
        tau = self.min_tau + F.softplus(tau_raw) * (self.max_tau - self.min_tau)
        
        # Debug final tau
        print(f"[DEBUG] tau final min/max/mean: {tau.min().item():.4f}/{tau.max().item():.4f}/{tau.mean().item():.4f}")
        print(f"tau final has NaN: {torch.isnan(tau).any().item()}")
        
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
        # Debug inputs
        print(f"[DEBUG] LiquidCell3D inputs:")
        print(f"h shape: {h.shape}, u shape: {u.shape}")
        print(f"h min/max/mean: {h.min().item():.4f}/{h.max().item():.4f}/{h.mean().item():.4f}")
        print(f"u min/max/mean: {u.min().item():.4f}/{u.max().item():.4f}/{u.mean().item():.4f}")
        print(f"h has NaN: {torch.isnan(h).any().item()}, u has NaN: {torch.isnan(u).any().item()}")
        
        # Compute time constants
        try:
            tau = self.compute_tau(h, u)
            
            # Debug tau
            print(f"[DEBUG] tau min/max/mean: {tau.min().item():.4f}/{tau.max().item():.4f}/{tau.mean().item():.4f}")
            print(f"tau has NaN: {torch.isnan(tau).any().item()}")
            
            # Clip extremely small time constants to avoid division by zero
            if (tau < 1e-6).any():
                print(f"[WARNING] Very small time constants detected: {(tau < 1e-6).sum().item()} values")
                tau = torch.clamp(tau, min=1e-6)
            
            # Compute cell dynamics
            pre_activation = torch.matmul(h, self.W_h.T) + torch.matmul(u, self.U_h.T) + self.b_h
            f_h = torch.sigmoid(pre_activation)
            
            # Debug activation
            print(f"[DEBUG] pre_activation min/max/mean: {pre_activation.min().item():.4f}/{pre_activation.max().item():.4f}/{pre_activation.mean().item():.4f}")
            print(f"f_h min/max/mean: {f_h.min().item():.4f}/{f_h.max().item():.4f}/{f_h.mean().item():.4f}")
            
            # Update hidden state using liquid dynamics
            h_dot = -h / tau + f_h
            h_new = h + dt * h_dot
            
            # Debug h_dot and h_new
            print(f"[DEBUG] h_dot min/max/mean: {h_dot.min().item():.4f}/{h_dot.max().item():.4f}/{h_dot.mean().item():.4f}")
            print(f"h_new min/max/mean: {h_new.min().item():.4f}/{h_new.max().item():.4f}/{h_new.mean().item():.4f}")
            
            # Check for NaNs or extreme values
            if torch.isnan(h_new).any():
                print(f"[WARNING] NaN values detected in h_new")
                h_new = torch.nan_to_num(h_new, nan=0.0)
            
            # Compute velocity output
            v_pre = torch.matmul(h_new, self.W_out.T) + self.b_out
            v = torch.tanh(v_pre)
            
            # Debug velocity
            print(f"[DEBUG] v_pre min/max/mean: {v_pre.min().item():.4f}/{v_pre.max().item():.4f}/{v_pre.mean().item():.4f}")
            print(f"v min/max/mean: {v.min().item():.4f}/{v.max().item():.4f}/{v.mean().item():.4f}")
            print(f"v has NaN: {torch.isnan(v).any().item()}")
            
            return h_new, v
        except Exception as e:
            print(f"[ERROR] Exception in LiquidCell3D forward: {e}")
            import traceback
            traceback.print_exc()
            # Return zeros if exception occurs
            return torch.zeros_like(h), torch.zeros(h.shape[0], h.shape[1], 3, device=h.device, dtype=h.dtype)
    
    def set_params_from_vector(self, params: torch.Tensor):
        """
        Set all parameters from a flat parameter vector.
        Used for hyper-network conditioning.
        """
        # Debug input
        print(f"[DEBUG] set_params_from_vector:")
        print(f"params shape: {params.shape}")
        print(f"params min/max/mean: {params.min().item():.4f}/{params.max().item():.4f}/{params.mean().item():.4f}")
        print(f"params has NaN: {torch.isnan(params).any().item()}")
        
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
                # Extract and reshape parameter
                param_data = params[idx:idx+numel].view(shape)
                
                # Check for NaNs or extreme values
                if torch.isnan(param_data).any():
                    print(f"[WARNING] NaN values detected in {name}")
                    param_data = torch.nan_to_num(param_data, nan=0.0)
                
                # Clip extreme values
                if param_data.abs().max() > 100:
                    print(f"[WARNING] Extreme values in {name}: {param_data.abs().max().item():.4f}")
                    param_data = torch.clamp(param_data, -100, 100)
                
                # Set parameter
                param.data = param_data
                idx += numel
                
                # Debug parameter
                print(f"[DEBUG] Set {name} shape: {param.shape}, min/max/mean: {param.min().item():.4f}/{param.max().item():.4f}/{param.mean().item():.4f}")
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
        # Debug inputs
        print(f"[DEBUG] LiquidODECore inputs:")
        print(f"spatial_coords shape: {spatial_coords.shape}")
        print(f"spatial_coords min/max/mean: {spatial_coords.min().item():.4f}/{spatial_coords.max().item():.4f}/{spatial_coords.mean().item():.4f}")
        print(f"spatial_coords has NaN: {torch.isnan(spatial_coords).any().item()}")
        if params is not None:
            print(f"params shape: {params.shape}")
            print(f"params min/max/mean: {params.min().item():.4f}/{params.max().item():.4f}/{params.mean().item():.4f}")
            print(f"params has NaN: {torch.isnan(params).any().item()}")
        
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
            try:
                h, v = self.cell(h, coords, dt=self.dt)
                
                # Check for NaNs or extreme values
                if torch.isnan(h).any() or torch.isnan(v).any():
                    print(f"[WARNING] NaN values detected in step {step}")
                    # Replace NaNs with zeros
                    h = torch.nan_to_num(h, nan=0.0)
                    v = torch.nan_to_num(v, nan=0.0)
                
                # Check for extreme values
                if h.abs().max() > 100 or v.abs().max() > 100:
                    print(f"[WARNING] Extreme values detected in step {step}")
                    print(f"h max: {h.abs().max().item():.4f}, v max: {v.abs().max().item():.4f}")
                    # Clip to reasonable range
                    h = torch.clamp(h, -100, 100)
                    v = torch.clamp(v, -100, 100)
                
                velocities.append(v)
            except Exception as e:
                print(f"[ERROR] Exception in liquid dynamics step {step}: {e}")
                import traceback
                traceback.print_exc()
                # Return zero velocity field if exception occurs
                return torch.zeros(B, 3, D, H, W, device=coords.device, dtype=coords.dtype)
        
        # Use final velocity
        velocity = velocities[-1]
        
        # Scale and reshape
        velocity = velocity * self.velocity_scale
        velocity_field = velocity.view(B, D, H, W, 3).permute(0, 4, 1, 2, 3)
        
        # Debug output
        print(f"[DEBUG] LiquidODECore output:")
        print(f"velocity_field shape: {velocity_field.shape}")
        print(f"velocity_field min/max/mean: {velocity_field.min().item():.4f}/{velocity_field.max().item():.4f}/{velocity_field.mean().item():.4f}")
        print(f"velocity_field has NaN: {torch.isnan(velocity_field).any().item()}")
        
        return velocity_field
    
    def get_param_count(self) -> int:
        """Get total number of parameters in the cell."""
        return sum(p.numel() for p in self.cell.parameters()) 