#!/usr/bin/env python3
"""Galaxy Automaton — cosmic-web pattern generation via cellular automata.

A creative exploration: can simple local rules produce realistic
cosmic large-scale structure patterns?

Usage:
    conda run -n hermesauto python galaxy_automaton.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import animation

OUT = Path("/home/yanyj/VibeCoding/autonomy/2026-06-01/galaxy_automaton")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 200, 200
STEPS = 60


def init_grid(w, h, seed_density=0.15):
    """Initialize with random overdensity seeds."""
    np.random.seed(42)
    grid = np.zeros((h, w), dtype=float)
    # Seeds
    seeds = np.random.random((h, w)) < seed_density
    grid[seeds] = np.random.uniform(0.5, 1.5, seeds.sum())
    # Add a large-scale gradient (mimics survey depth variation)
    y_grad = np.linspace(0.8, 1.2, h)[:, np.newaxis]
    grid *= y_grad
    return grid


def neighbor_sum(grid):
    """Sum of 8 neighbors with periodic boundary."""
    s = np.zeros_like(grid)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            s += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    return s


def step(grid):
    """One CA step: growth + competition + diffusion.

    Rules inspired by cosmic structure formation:
    1. Rich get richer (gravitational collapse): cells with high density
       draw mass from neighbors
    2. Diffusion: density spreads slightly
    3. Threshold: below a minimum, cells die off (voids)
    """
    ng = neighbor_sum(grid)

    # Growth: overdense regions accrete from neighbors
    accretion = ng * grid * 0.015
    grid = grid + accretion

    # Competition: cells with above-avg neighbors grow faster
    mean_ng = np.mean(ng)
    growth_boost = np.where(ng > mean_ng, 1.02, 0.99)
    grid = grid * growth_boost

    # Diffusion: smooth slightly
    grid = 0.85 * grid + 0.15 * ng / 8.0

    # Floor: prevent negative densities
    grid = np.maximum(grid, 0.001)

    # Normalize to keep total mass roughly constant
    grid = grid / np.mean(grid)

    return grid


def render_frame(grid, step_num, save=False):
    """Render a single frame."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a14')

    ax.imshow(grid, cmap='inferno', origin='lower', aspect='equal',
              vmin=0, vmax=np.percentile(grid, 95))
    ax.set_title(f'Step {step_num}', color='white', fontsize=12)
    ax.axis('off')

    if save:
        fname = OUT / f'frame_{step_num:03d}.png'
        fig.savefig(fname, dpi=100, bbox_inches='tight', facecolor='#0a0a14')
        plt.close(fig)
        return fname
    else:
        return fig


def create_animation():
    """Generate all frames and combine into animation."""
    print("Galaxy Automaton — cosmic web pattern generation")
    print("=" * 50)

    grid = init_grid(W, H)
    frames = [grid.copy()]

    print(f"Step 0: seeds placed, mean={grid.mean():.3f}, max={grid.max():.3f}")

    for i in range(1, STEPS + 1):
        grid = step(grid)
        frames.append(grid.copy())
        if i % 10 == 0:
            print(f"Step {i}: mean={grid.mean():.3f}, max={grid.max():.3f}, "
                  f"fill={(grid > 0.5).mean()*100:.1f}%")

    # Render key frames
    print("\nRendering frames...")
    for i in [0, 5, 10, 20, 30, 40, 50, 60]:
        render_frame(frames[i], i, save=True)
        print(f"  Saved frame_{i:03d}.png")

    # Create final comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a14')
    for ax, step_num in zip(axes.flat, [0, 5, 10, 20, 30, 40, 50, 60]):
        ax.imshow(frames[step_num], cmap='inferno', origin='lower',
                  vmin=0, vmax=np.percentile(frames[-1], 95))
        ax.set_title(f'Step {step_num}', color='white', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(OUT / 'evolution_panel.png', dpi=150, facecolor='#0a0a14')
    plt.close(fig)
    print("  Saved evolution_panel.png")

    # Create animated GIF
    print("\nCreating animated GIF...")
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0a0a14')
    ax.axis('off')

    vmax = np.percentile(frames[-1], 95)
    ims = []
    for i, g in enumerate(frames):
        im = ax.imshow(g, cmap='inferno', origin='lower', animated=True,
                       vmin=0, vmax=vmax)
        t = ax.text(10, 190, f'Step {i}', color='white', fontsize=10,
                     fontfamily='monospace')
        ims.append([im, t])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)
    gif_path = OUT / 'galaxy_automaton.gif'
    ani.save(gif_path, writer='pillow', fps=10, dpi=80)
    plt.close(fig)
    print(f"  Saved galaxy_automaton.gif ({gif_path.stat().st_size / 1024:.0f} KB)")

    # Save final grid data
    np.save(OUT / 'final_grid.npy', frames[-1])
    print(f"  Saved final_grid.npy")

    print(f"\nDone! All outputs in {OUT}")


if __name__ == "__main__":
    create_animation()
