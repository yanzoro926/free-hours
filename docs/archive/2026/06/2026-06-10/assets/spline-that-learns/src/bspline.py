"""
B-spline basis functions from first principles.
Using the Cox-de Boor recursion formula.

A B-spline of order k (degree k-1) is defined on a knot vector.
For simplicity, we use uniform knots with clamped endpoints
(multiplicity k at each end), which is standard for KANs.
"""

import torch
import torch.nn.functional as F


def extend_grid(grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Extend a uniform grid for clamped B-splines.
    Replicates the first and last points k times.
    grid shape: [num_intervals + 1] → extended: [num_intervals + 2k + 1]
    """
    h = (grid[-1] - grid[0]) / (grid.shape[0] - 1)
    # Prepend k points below and append k points above
    left_ext = grid[0] - h * torch.arange(k, 0, -1, device=grid.device)
    right_ext = grid[-1] + h * torch.arange(1, k + 1, device=grid.device)
    return torch.cat([left_ext, grid, right_ext])


def cox_de_boor(x: torch.Tensor, knots: torch.Tensor, i: int, k: int) -> torch.Tensor:
    """
    Evaluate the i-th B-spline basis function of order k at points x.
    Using the Cox-de Boor recursion:
        B_{i,1}(x) = 1 if knots[i] <= x < knots[i+1] else 0
        B_{i,k}(x) = ((x - knots[i]) / (knots[i+k-1] - knots[i])) * B_{i,k-1}(x)
                   + ((knots[i+k] - x) / (knots[i+k] - knots[i+1])) * B_{i+1,k-1}(x)
    """
    if k == 1:
        return ((knots[i] <= x) & (x < knots[i + 1])).float()
    
    # First term
    denom1 = knots[i + k - 1] - knots[i]
    term1 = torch.where(
        denom1 > 0,
        (x - knots[i]) / denom1 * cox_de_boor(x, knots, i, k - 1),
        torch.zeros_like(x)
    )
    
    # Second term
    denom2 = knots[i + k] - knots[i + 1]
    term2 = torch.where(
        denom2 > 0,
        (knots[i + k] - x) / denom2 * cox_de_boor(x, knots, i + 1, k - 1),
        torch.zeros_like(x)
    )
    
    return term1 + term2


def bspline_basis(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluate all B-spline basis functions of order k on grid at points x.
    
    Args:
        x: input points [batch_size, ...]
        grid: uniform grid points [G + 1] (G intervals)
        k: spline order (e.g., 3 for cubic)
    
    Returns:
        basis values [batch_size, ..., G + k] — one value per basis function
        There are G + k basis functions for a grid with G intervals.
    """
    extended = extend_grid(grid, k)
    num_basis = extended.shape[0] - k  # = G + k
    
    bases = []
    for i in range(num_basis):
        bases.append(cox_de_boor(x, extended, i, k))
    
    return torch.stack(bases, dim=-1)  # [*batch, num_basis]


def bspline_basis_vectorized(x: torch.Tensor, grid: torch.Tensor, k: int) -> torch.Tensor:
    """
    Vectorized (faster) version of B-spline basis evaluation.
    Uses the iterative Cox-de Boor formula.
    
    Args:
        x: input points [batch_size, in_features]
        grid: [G + 1] grid points
        k: spline order
    
    Returns:
        basis [batch_size, in_features, G + k]
    """
    x = x.unsqueeze(-1)  # [batch, in_features, 1]
    extended = extend_grid(grid, k)  # [G + 2k + 1]
    
    # B_{i,1}: indicator on [knots[i], knots[i+1])
    knots_1 = extended[None, None, :]  # [1, 1, G + 2k + 1]
    
    # Build order-1 basis: shape [batch, in_features, G + 2k]
    left = knots_1[..., :-1]  # [1, 1, G + 2k]
    right = knots_1[..., 1:]  # [1, 1, G + 2k]
    B = ((left <= x) & (x < right)).float()  # [batch, in_features, G + 2k]
    
    # Edge case for the last point: include x == last knot
    # This handles the right boundary
    last_left = knots_1[..., -2:-1]
    last_right = knots_1[..., -1:]
    B_last = ((last_left <= x) & (x <= last_right)).float()
    B = torch.cat([B[..., :-1], B_last], dim=-1)
    
    # Cox-de Boor recursion for order 2 to k
    for order in range(2, k + 1):
        B_new = []
        num_B = B.shape[-1] - 1  # one fewer basis per recursion step
        for i in range(num_B):
            # term1: (x - t_i) / (t_{i+order-1} - t_i) * B_{i, order-1}
            t_i = extended[i]
            t_ikm1 = extended[i + order - 1]
            denom1 = t_ikm1 - t_i
            term1 = torch.where(
                denom1 > 0,
                (x.squeeze(-1) - t_i) / denom1 * B[..., i],
                torch.zeros_like(B[..., i])
            )
            
            # term2: (t_{i+order} - x) / (t_{i+order} - t_{i+1}) * B_{i+1, order-1}
            t_ik = extended[i + order]
            t_ip1 = extended[i + 1]
            denom2 = t_ik - t_ip1
            term2 = torch.where(
                denom2 > 0,
                (t_ik - x.squeeze(-1)) / denom2 * B[..., i + 1],
                torch.zeros_like(B[..., i + 1])
            )
            
            B_new.append(term1 + term2)
        
        B = torch.stack(B_new, dim=-1)  # [batch, in_features, num_basis_for_this_order]
    
    return B  # [batch, in_features, G + k]


if __name__ == "__main__":
    # Quick test
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    G = 5  # 5 intervals
    k = 3  # cubic splines
    grid = torch.linspace(-1, 1, G + 1)
    x = torch.linspace(-1, 1, 1000)
    
    # Test vectorized vs recursive
    basis = bspline_basis_vectorized(x.unsqueeze(-1), grid, k).squeeze(1)
    print(f"Basis shape: {basis.shape}")  # Should be [1000, G + k] = [1000, 8]
    
    # Also test recursive version
    basis_rec = bspline_basis(x, grid, k)
    print(f"Recursive basis shape: {basis_rec.shape}")
    print(f"Max difference: {(basis - basis_rec).abs().max().item():.2e}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(basis.shape[1]):
        ax.plot(x.numpy(), basis[:, i].numpy(), label=f'B_{i},{k}')
    for g in grid:
        ax.axvline(x=g.item(), color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'B-spline Basis Functions (G={G}, k={k})')
    ax.legend(fontsize=8, ncol=2)
    ax.set_xlabel('x')
    ax.set_ylabel('B(x)')
    plt.tight_layout()
    plt.savefig('/home/yanyj/VibeCoding/autonomy/2026-06-10/spline-that-learns/visualizations/bspline_basis.png', dpi=150)
    print("Basis visualization saved.")
