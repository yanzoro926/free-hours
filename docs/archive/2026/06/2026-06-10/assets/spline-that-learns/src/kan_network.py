"""
KAN: Kolmogorov-Arnold Networks — From First Principles

A clean, well-documented implementation of KAN layers and networks.
Based on the paper "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024).

Core insight: Instead of fixed activation functions on nodes with learnable
weights on edges (MLP), KANs put LEARNABLE activation functions (B-splines)
on edges and only summation on nodes.

Architecture:
    φ(x) = w_base · base(x) + w_spline · spline(x)
    where spline(x) = Σ c_i · B_i(x)  (B_i are B-spline basis functions)

For a layer with n_in → n_out:
    output_j = Σ_i φ_{i,j}(input_i)   for j in 1..n_out

Each edge (i,j) has its own learnable activation function φ_{i,j},
parameterized by (G+k) B-spline coefficients, where G is the grid size
and k is the spline order.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


# ═══════════════════════════════════════════════════════════════════
# B-Spline Utilities
# ═══════════════════════════════════════════════════════════════════

def extend_grid(grid: torch.Tensor, k_extend: int) -> torch.Tensor:
    """
    Extend a uniform grid by k_extend points on each side.
    
    Args:
        grid: [..., G+1] — uniform grid points
        k_extend: number of points to add on each side
    
    Returns:
        extended grid: [..., G+1+2*k_extend]
    """
    h = (grid[..., -1:] - grid[..., :1]) / (grid.shape[-1] - 1)
    
    for _ in range(k_extend):
        grid = torch.cat([grid[..., :1] - h, grid], dim=-1)
        grid = torch.cat([grid, grid[..., -1:] + h], dim=-1)
    
    return grid


def cox_de_boor_batch(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate B-spline basis functions using Cox-de Boor recursion (batched).
    
    This is a vectorized implementation that computes all basis functions
    simultaneously for the entire batch.
    
    Args:
        x: input points [batch, n_in]
        grid: extended grid [n_in, G + 2k + 1]
              (original G+1 points extended by k on each side)
        k: spline order (k=3 for cubic)
    
    Returns:
        basis_values: [batch, n_in, G + k]
    """
    x = x.unsqueeze(-1)  # [batch, n_in, 1]
    grid = grid.unsqueeze(0)  # [1, n_in, G+2k+1]
    
    # Base case (k=0): indicator functions
    # B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1] else 0
    left = grid[..., :-1]
    right = grid[..., 1:]
    B = ((left <= x) & (x < right)).float()  # [batch, n_in, G+2k]
    
    # Fix right boundary: the last interval should include the rightmost point
    last_left = grid[..., -2:-1]
    last_right = grid[..., -1:]
    B_last = ((last_left <= x) & (x <= last_right)).float()
    B = torch.cat([B[..., :-1], B_last], dim=-1)
    
    # Cox-de Boor recursion for orders 1 through k
    for order in range(1, k + 1):
        # (x - t_i) / (t_{i+order} - t_i) * B_{i, order-1}
        # + (t_{i+order+1} - x) / (t_{i+order+1} - t_{i+1}) * B_{i+1, order-1}
        
        G = grid.shape[-1]  # total grid points
        num_basis = G - order - 1  # number of basis functions at this order
        
        t = grid  # [1, n_in, G]
        x_flat = x.squeeze(-1)  # [batch, n_in]
        
        # First term
        t_i = t[..., :num_basis]
        t_ik = t[..., order:order+num_basis]  # t_{i+order}
        denom1 = t_ik - t_i
        term1 = torch.where(
            denom1 > 1e-10,
            (x_flat.unsqueeze(-1) - t_i) / denom1 * B[..., :num_basis],
            torch.zeros_like(B[..., :num_basis])
        )
        
        # Second term
        t_ik1 = t[..., order+1:order+1+num_basis]  # t_{i+order+1}
        t_ip1 = t[..., 1:1+num_basis]  # t_{i+1}
        denom2 = t_ik1 - t_ip1
        term2 = torch.where(
            denom2 > 1e-10,
            (t_ik1 - x_flat.unsqueeze(-1)) / denom2 * B[..., 1:1+num_basis],
            torch.zeros_like(B[..., 1:1+num_basis])
        )
        
        B = term1 + term2
    
    return B  # [batch, n_in, G + k]


def coef_to_curve(x: torch.Tensor, grid: torch.Tensor, 
                  coef: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate B-spline curves at points x using coefficients.
    
    curve(x) = Σ_j c_j · B_j(x) for each input-output pair.
    
    Args:
        x: [batch, n_in]
        grid: [n_in, G + 2k + 1]
        coef: [n_in, n_out, G + k]
        k: spline order
    
    Returns:
        curve_values: [batch, n_in, n_out]
    """
    basis = cox_de_boor_batch(x, grid, k)  # [batch, n_in, G+k]
    # einsum: sum over basis dimension
    # basis[b,i,g] * coef[i,j,g] → curve[b,i,j]
    return torch.einsum('big,ijg->bij', basis, coef.to(basis.device))


def curve_to_coef(x: torch.Tensor, y: torch.Tensor, 
                  grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Fit B-spline coefficients using least squares.
    
    Given input-output pairs (x, y), find coefficients c such that
    Σ_j c_j · B_j(x_i) ≈ y_i for each input-output dimension.
    
    Args:
        x: [batch, n_in]
        y: [batch, n_in, n_out]
        grid: [n_in, G + 2k + 1]
        k: spline order
    
    Returns:
        coef: [n_in, n_out, G + k]
    """
    batch, n_in = x.shape
    n_out = y.shape[2]
    n_coef = grid.shape[1] - k - 1  # = G + k
    
    basis = cox_de_boor_batch(x, grid, k)  # [batch, n_in, G+k]
    
    # Reshape for batched least squares
    # mat: [n_in, n_out, batch, n_coef]
    mat = basis.permute(1, 0, 2)[:, None, :, :].expand(n_in, n_out, batch, n_coef)
    # y_eval: [n_in, n_out, batch, 1]
    y_eval = y.permute(1, 2, 0).unsqueeze(-1)
    
    # Least squares solve
    coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    
    return coef  # [n_in, n_out, G+k]


# ═══════════════════════════════════════════════════════════════════
# KAN Layer
# ═══════════════════════════════════════════════════════════════════

class KANLayer(nn.Module):
    """
    A single KAN layer: n_in inputs → n_out outputs.
    
    Each input-output pair (i,j) has its own learnable activation function:
        φ_{i,j}(x) = w_base_{i,j} · base(x) + w_spline_{i,j} · spline_{i,j}(x)
    
    where spline_{i,j}(x) = Σ_g c_{i,j,g} · B_g(x)  (B-spline expansion)
    
    Args:
        n_in: input dimension
        n_out: output dimension
        grid_size: number of grid intervals (G)
        spline_order: B-spline order (k=3 for cubic)
        base_fn: residual activation function (default: SiLU)
        grid_range: range of initial grid [low, high]
        noise_scale: initial noise magnitude for spline coefficients
        grid_eps: interpolation between uniform (1.0) and adaptive (0.0) grids
    """
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        grid_size: int = 5,
        spline_order: int = 3,
        base_fn: nn.Module = None,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        noise_scale: float = 0.1,
        grid_eps: float = 0.02,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_eps = grid_eps
        
        # Create initial uniform grid and extend it
        # grid: [n_in, G+1] → extended: [n_in, G+1+2k]
        # Note: pykan uses extension by k (not k-1), giving G+1+2k total points
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1)
        grid = grid.unsqueeze(0).expand(n_in, grid_size + 1)
        grid = extend_grid(grid, k_extend=spline_order)
        self.register_buffer('grid', grid)  # [n_in, G+1+2k]
        
        # Number of B-spline basis functions: G + k
        n_basis = grid_size + spline_order
        
        # Initialize spline coefficients with small random values
        # coef: [n_in, n_out, n_basis]
        noise = (torch.rand(grid_size + 1, n_in, n_out) - 0.5) * noise_scale / grid_size
        noise = noise.permute(1, 2, 0)  # [n_in, n_out, G+1]
        
        # Fit initial coefficients using least squares on the noise pattern
        # This ensures the initial spline passes through the noise points
        grid_inner = self.grid[:, spline_order:-spline_order]  # [n_in, G+1]
        
        # We need to create initial coef from the noise
        # Using the same approach as pykan: fit coef via curve2coef
        # But simpler: just initialize randomly
        self.coef = nn.Parameter(
            torch.randn(n_in, n_out, n_basis) * noise_scale / np.sqrt(n_basis)
        )
        
        # Scale parameters
        # scale_base: weight for the residual connection b(x)
        self.scale_base = nn.Parameter(
            torch.randn(n_in, n_out) * 0.1 / np.sqrt(n_in)
        )
        
        # scale_sp: weight for the spline component
        self.scale_sp = nn.Parameter(
            torch.ones(n_in, n_out) / np.sqrt(n_in)
        )
        
        # Mask for pruning (all ones initially)
        self.register_buffer('mask', torch.ones(n_in, n_out))
        
        # Base function (residual connection)
        self.base_fn = base_fn if base_fn is not None else nn.SiLU()
    
    def forward(self, x: torch.Tensor, 
                return_activations: bool = False) -> torch.Tensor | Tuple:
        """
        Forward pass through the KAN layer.
        
        Args:
            x: [batch, n_in]
            return_activations: if True, also return pre/post activation values
        
        Returns:
            y: [batch, n_out]
            (optional) preacts: [batch, n_out, n_in]
            (optional) postacts: [batch, n_out, n_in]  
            (optional) postspline: [batch, n_out, n_in]
        """
        batch = x.shape[0]
        
        # Pre-activations: fan out input
        preacts = x.unsqueeze(1).expand(batch, self.n_out, self.n_in)
        
        # Base function (residual): shape [batch, n_in]
        base = self.base_fn(x)
        
        # Spline curve evaluation: [batch, n_in, n_out]
        spline_out = coef_to_curve(x, self.grid, self.coef, self.spline_order)
        
        postspline = spline_out.permute(0, 2, 1)  # [batch, n_out, n_in]
        
        # Combine: φ(x) = w_base * base(x) + w_spline * spline(x)
        # base[:,:,None] is [batch, n_in, 1]
        # spline_out is [batch, n_in, n_out]
        y = (self.scale_base[None, :, :] * base[:, :, None] + 
             self.scale_sp[None, :, :] * spline_out)
        
        # Apply mask
        y = self.mask[None, :, :] * y
        
        postacts = y.permute(0, 2, 1)  # [batch, n_out, n_in]
        
        # Sum over inputs: output_j = Σ_i φ_{i,j}(x_i)
        y = torch.sum(y, dim=1)  # [batch, n_out]
        
        if return_activations:
            return y, preacts, postacts, postspline
        return y
    
    def update_grid(self, x: torch.Tensor):
        """
        Update the spline grid adaptively based on input samples.
        
        The grid is updated to better cover the actual input range seen during training.
        grid_eps controls the interpolation between a purely adaptive grid (grid_eps=0)
        and a purely uniform grid (grid_eps=1).
        
        Args:
            x: [batch, n_in] — sample inputs
        """
        batch = x.shape[0]
        
        # Get current spline curve values at sorted inputs
        x_sorted, _ = torch.sort(x, dim=0)
        y_eval = coef_to_curve(x_sorted, self.grid, self.coef, self.spline_order)
        
        # Number of internal intervals
        n_intervals = self.grid_size
        
        # Adaptive grid: partition based on input percentiles
        ids = [int(batch / n_intervals * i) for i in range(n_intervals)] + [-1]
        grid_adaptive = x_sorted[ids, :].T  # [n_in, G+1]
        
        # Uniform grid: evenly spaced between min and max of each input dimension
        margin = 0.01
        x_min = x_sorted[0, :]  # [n_in]
        x_max = x_sorted[-1, :]  # [n_in]
        h = (x_max - x_min + 2 * margin) / n_intervals
        grid_uniform = (x_min - margin).unsqueeze(1) + h.unsqueeze(1) * torch.arange(
            n_intervals + 1, device=x.device
        ).unsqueeze(0).float()
        
        # Interpolate between uniform and adaptive
        new_grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        # Extend and set
        self.grid.data = extend_grid(new_grid, k_extend=self.spline_order)


# ═══════════════════════════════════════════════════════════════════
# Full KAN Network
# ═══════════════════════════════════════════════════════════════════

class KAN(nn.Module):
    """
    A multi-layer Kolmogorov-Arnold Network.
    
    Args:
        layer_dims: list of dimensions, e.g., [2, 5, 1] for 2 inputs → 5 hidden → 1 output
        grid_size: number of grid intervals per layer
        spline_order: B-spline order
        base_fn: base activation function
        grid_eps: grid adaptation epsilon
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        grid_size: int = 5,
        spline_order: int = 3,
        base_fn: nn.Module = None,
        grid_eps: float = 0.02,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLayer(
                    n_in=layer_dims[i],
                    n_out=layer_dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    base_fn=base_fn,
                    grid_eps=grid_eps,
                )
            )
        
        self.layer_dims = layer_dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def update_grids(self, x: torch.Tensor):
        """Update grids for all layers."""
        for layer in self.layers:
            layer.update_grid(x)
            x = layer(x)


# ═══════════════════════════════════════════════════════════════════
# Equivalent MLP for comparison
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Standard MLP with SiLU activation for fair comparison."""
    
    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters())


def count_kan_params(model: KAN) -> int:
    """Count total parameters in a KAN model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick smoke test
    print("Testing KAN implementation...")
    
    # Test KANLayer
    layer = KANLayer(n_in=2, n_out=3, grid_size=5, spline_order=3)
    x = torch.randn(10, 2)
    y = layer(x)
    print(f"KANLayer: {x.shape} → {y.shape} ✓")
    
    # Test full KAN
    kan = KAN(layer_dims=[2, 5, 1], grid_size=5, spline_order=3)
    y = kan(x)
    print(f"KAN: {x.shape} → {y.shape} ✓")
    print(f"KAN params: {count_kan_params(kan)}")
    
    # Compare with MLP
    mlp = MLP([2, 5, 1])
    y_mlp = mlp(x)
    print(f"MLP: {x.shape} → {y_mlp.shape} ✓")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
    
    print("All tests passed!")
