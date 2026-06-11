"""
Training and Visualization for KAN vs MLP.

Benchmark problems:
1. 1D function approximation: f(x) = sin(πx)
2. 1D function approximation: f(x) = e^{-x} · sin(2πx)  (damped oscillation)
3. 2D function: f(x,y) = sin(πx) · cos(πy)
4. 2D classification: make_moons
5. Fitness landscape: f(x) = x² + 3sin(5x)  (multimodal)

For each, we:
- Train both KAN and MLP
- Plot loss curves side by side
- Visualize learned KAN activation functions
- Show how KAN's spline grid adapts
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import time
from sklearn.datasets import make_moons

from kan_network import KAN, KANLayer, MLP, count_kan_params, coef_to_curve, extend_grid

# Output directory
OUT_DIR = Path('/home/yanyj/VibeCoding/autonomy/2026-06-10/spline-that-learns/visualizations')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'figure.dpi': 150,
})

COLORS = ['#58a6ff', '#f78166', '#3fb950', '#d2a8ff', '#ffa657', '#79c0ff']


def train_model(model, x_train, y_train, x_test, y_test,
                epochs=1000, lr=1e-3, weight_decay=1e-5,
                update_grid_every=50, title="model"):
    """Train a model and return loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Update grids periodically (KAN-specific)
        if isinstance(model, KAN) and epoch > 0 and epoch % update_grid_every == 0:
            with torch.no_grad():
                model.update_grids(x_train)
        
        train_losses.append(loss.item())
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = criterion(y_pred_test, y_test)
            test_losses.append(test_loss.item())
            
            if test_loss.item() < best_test_loss:
                best_test_loss = test_loss.item()
    
    train_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'best_test_loss': best_test_loss,
        'train_time': train_time,
    }


# ═══════════════════════════════════════════════════════════════════
# Benchmark 1: 1D Function Approximation
# ═══════════════════════════════════════════════════════════════════

def benchmark_1d(func, func_name, x_range=(-2, 2), n_train=200, n_test=1000):
    """Train KAN and MLP on a 1D function."""
    print(f"\n{'='*60}")
    print(f"Benchmark: 1D Function — {func_name}")
    print(f"{'='*60}")
    
    # Generate data
    x_train = torch.linspace(x_range[0], x_range[1], n_train).unsqueeze(-1)
    y_train = func(x_train)
    x_test = torch.linspace(x_range[0], x_range[1], n_test).unsqueeze(-1)
    y_test = func(x_test)
    
    # KAN: [1, 5, 5, 1] — small hidden layers but learnable activations
    kan = KAN(layer_dims=[1, 5, 5, 1], grid_size=8, spline_order=3)
    # MLP: [1, 20, 20, 1] — more neurons but fixed activations
    mlp = MLP([1, 20, 20, 1])
    
    print(f"KAN params: {count_kan_params(kan)}")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
    
    # Train
    kan_results = train_model(kan, x_train, y_train, x_test, y_test,
                              epochs=2000, lr=1e-2, update_grid_every=100,
                              title=f"KAN_{func_name}")
    mlp_results = train_model(mlp, x_train, y_train, x_test, y_test,
                              epochs=2000, lr=1e-2,
                              title=f"MLP_{func_name}")
    
    print(f"KAN best test loss: {kan_results['best_test_loss']:.2e} (time: {kan_results['train_time']:.1f}s)")
    print(f"MLP best test loss: {mlp_results['best_test_loss']:.2e} (time: {mlp_results['train_time']:.1f}s)")
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Loss curves
    ax_loss = fig.add_subplot(gs[0, :2])
    ax_loss.plot(kan_results['train_losses'], alpha=0.3, color=COLORS[0], label='KAN Train')
    ax_loss.plot(kan_results['test_losses'], color=COLORS[0], linewidth=2, label='KAN Test')
    ax_loss.plot(mlp_results['train_losses'], alpha=0.3, color=COLORS[1], label='MLP Train')
    ax_loss.plot(mlp_results['test_losses'], color=COLORS[1], linewidth=2, label='MLP Test')
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('MSE Loss')
    ax_loss.set_title(f'Training Dynamics — {func_name}')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # Function fits
    ax_fit = fig.add_subplot(gs[0, 2])
    with torch.no_grad():
        x_plot = torch.linspace(x_range[0], x_range[1], 500).unsqueeze(-1)
        y_true = func(x_plot)
        y_kan = kan(x_plot)
        y_mlp = mlp(x_plot)
    
    ax_fit.plot(x_plot.numpy(), y_true.numpy(), 'w-', linewidth=2, label='True')
    ax_fit.plot(x_plot.numpy(), y_kan.numpy(), '--', color=COLORS[0], linewidth=2, label='KAN')
    ax_fit.plot(x_plot.numpy(), y_mlp.numpy(), '--', color=COLORS[1], linewidth=2, label='MLP')
    ax_fit.scatter(x_train.numpy(), y_train.numpy(), c='gray', s=5, alpha=0.5)
    ax_fit.set_xlabel('x')
    ax_fit.set_ylabel('f(x)')
    ax_fit.set_title('Function Fit Comparison')
    ax_fit.legend()
    ax_fit.grid(True, alpha=0.3)
    
    # KAN activation functions visualization
    ax_act = fig.add_subplot(gs[1, :])
    visualize_kan_activations(kan.layers[0], ax_act,
                              x_range=x_range, title='Learned Activation Functions (Layer 1)')
    
    safe_name = func_name.replace(' ', '_').replace('(', '').replace(')', '').replace('*', 'x')
    fig.savefig(OUT_DIR / f'benchmark_1d_{safe_name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: benchmark_1d_{safe_name}.png")
    
    return kan, mlp, kan_results, mlp_results


def visualize_kan_activations(layer: KANLayer, ax, x_range=(-2, 2), title=''):
    """Visualize all learned activation functions in a KAN layer."""
    n_in, n_out = layer.n_in, layer.n_out
    
    with torch.no_grad():
        x = torch.linspace(x_range[0], x_range[1], 500).unsqueeze(-1)  # [500, 1]
        
        # Repeat x for each input dimension
        x_expanded = x.expand(500, n_in)  # [500, n_in]
        
        # Compute activations
        base = layer.base_fn(x_expanded)  # [500, n_in]
        spline_out = coef_to_curve(x_expanded, layer.grid, layer.coef, layer.spline_order)  # [500, n_in, n_out]
        
        activations = (layer.scale_base[None, :, :] * base[:, :, None] + 
                      layer.scale_sp[None, :, :] * spline_out)  # [500, n_in, n_out]
    
    for i in range(n_in):
        for j in range(n_out):
            color_idx = (i * n_out + j) % len(COLORS)
            ax.plot(x.numpy(), activations[:, i, j].numpy(),
                   color=COLORS[color_idx], linewidth=1, alpha=0.7,
                   label=f'φ({i}→{j})' if i == 0 and j < 3 else '')
    
    ax.axhline(y=0, color='#30363d', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('φ(x)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if n_in * n_out <= 15:
        ax.legend(fontsize=6, ncol=min(n_out, 5), loc='upper left')


# ═══════════════════════════════════════════════════════════════════
# Benchmark 2: 2D Function Approximation
# ═══════════════════════════════════════════════════════════════════

def benchmark_2d(func, func_name, x_range=(-1, 1), n_train=500, n_test=2500):
    """Train KAN and MLP on a 2D function."""
    print(f"\n{'='*60}")
    print(f"Benchmark: 2D Function — {func_name}")
    print(f"{'='*60}")
    
    # Generate grid data
    x1 = torch.linspace(x_range[0], x_range[1], int(np.sqrt(n_train)))
    x2 = torch.linspace(x_range[0], x_range[1], int(np.sqrt(n_train)))
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_train = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    y_train = func(x_train)
    
    x1_t = torch.linspace(x_range[0], x_range[1], int(np.sqrt(n_test)))
    x2_t = torch.linspace(x_range[0], x_range[1], int(np.sqrt(n_test)))
    X1_t, X2_t = torch.meshgrid(x1_t, x2_t, indexing='ij')
    x_test = torch.stack([X1_t.flatten(), X2_t.flatten()], dim=-1)
    y_test = func(x_test)
    
    # Models
    kan = KAN(layer_dims=[2, 5, 5, 1], grid_size=8, spline_order=3)
    mlp = MLP([2, 20, 20, 1])
    
    print(f"KAN params: {count_kan_params(kan)}")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
    
    kan_results = train_model(kan, x_train, y_train, x_test, y_test,
                              epochs=2000, lr=1e-2, update_grid_every=100)
    mlp_results = train_model(mlp, x_train, y_train, x_test, y_test,
                              epochs=2000, lr=1e-2)
    
    print(f"KAN best test loss: {kan_results['best_test_loss']:.2e} (time: {kan_results['train_time']:.1f}s)")
    print(f"MLP best test loss: {mlp_results['best_test_loss']:.2e} (time: {mlp_results['train_time']:.1f}s)")
    
    # Plot
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Loss curves
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(kan_results['train_losses'], alpha=0.3, color=COLORS[0], label='KAN Train')
    ax_loss.plot(kan_results['test_losses'], color=COLORS[0], linewidth=2, label='KAN Test')
    ax_loss.plot(mlp_results['train_losses'], alpha=0.3, color=COLORS[1], label='MLP Train')
    ax_loss.plot(mlp_results['test_losses'], color=COLORS[1], linewidth=2, label='MLP Test')
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('MSE Loss')
    ax_loss.set_title(f'Training Dynamics — {func_name}')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)
    
    # 2D function visualizations
    grid_size = int(np.sqrt(n_test))
    with torch.no_grad():
        y_true = func(x_test).reshape(grid_size, grid_size)
        y_kan = kan(x_test).reshape(grid_size, grid_size)
        y_mlp = mlp(x_test).reshape(grid_size, grid_size)
    
    X1_plot = X1_t.numpy()
    X2_plot = X2_t.numpy()
    
    vmin = min(y_true.min().item(), y_kan.min().item(), y_mlp.min().item())
    vmax = max(y_true.max().item(), y_kan.max().item(), y_mlp.max().item())
    
    for idx, (title, data) in enumerate([
        ('True Function', y_true),
        ('KAN Prediction', y_kan),
        ('MLP Prediction', y_mlp)
    ]):
        ax = fig.add_subplot(gs[1, idx])
        im = ax.pcolormesh(X1_plot, X2_plot, data.numpy(), 
                          cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(im, ax=ax)
    
    # Error maps
    kan_error = (y_kan - y_true).abs()
    mlp_error = (y_mlp - y_true).abs()
    err_max = max(kan_error.max().item(), mlp_error.max().item())
    
    for idx, (title, data) in enumerate([
        ('KAN |Error|', kan_error),
        ('MLP |Error|', mlp_error)
    ]):
        ax = fig.add_subplot(gs[2, idx])
        im = ax.pcolormesh(X1_plot, X2_plot, data.numpy(),
                          cmap='hot', shading='auto', vmin=0, vmax=err_max)
        ax.set_title(title)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        plt.colorbar(im, ax=ax)
    
    # Activation visualization
    ax_act = fig.add_subplot(gs[2, 2])
    visualize_kan_activations(kan.layers[0], ax_act, 
                              x_range=(-1.5, 1.5),
                              title='KAN Activations (Layer 1)')
    
    safe_name = func_name.replace(' ', '_').replace('(', '').replace(')', '').replace('*', 'x').replace('·', '_')
    fig.savefig(OUT_DIR / f'benchmark_2d_{safe_name}.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: benchmark_2d_{safe_name}.png")
    
    return kan, mlp, kan_results, mlp_results


# ═══════════════════════════════════════════════════════════════════
# Benchmark 3: Classification (make_moons)
# ═══════════════════════════════════════════════════════════════════

def benchmark_classification(n_train=500, n_test=200):
    """Train KAN and MLP on make_moons classification."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Classification — Make Moons")
    print(f"{'='*60}")
    
    # Generate data
    X, y = make_moons(n_samples=n_train + n_test, noise=0.15, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    
    # Split
    x_train, x_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Models
    kan = KAN(layer_dims=[2, 8, 4, 1], grid_size=8, spline_order=3)
    mlp = MLP([2, 32, 16, 1])
    
    print(f"KAN params: {count_kan_params(kan)}")
    print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
    
    # Train with BCE loss
    def train_classifier(model, x_train, y_train, x_test, y_test, epochs=1500, lr=1e-2):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        
        train_losses = []
        test_losses = []
        test_accs = []
        best_acc = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if isinstance(model, KAN) and epoch > 0 and epoch % 100 == 0:
                with torch.no_grad():
                    model.update_grids(x_train)
            
            train_losses.append(loss.item())
            
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test)
                test_loss = criterion(y_pred_test, y_test)
                test_losses.append(test_loss.item())
                
                acc = ((torch.sigmoid(y_pred_test) > 0.5).float() == y_test).float().mean()
                test_accs.append(acc.item())
                best_acc = max(best_acc, acc.item())
        
        return train_losses, test_losses, test_accs, best_acc
    
    kan_train, kan_test, kan_acc, kan_best = train_classifier(kan, x_train, y_train, x_test, y_test)
    mlp_train, mlp_test, mlp_acc, mlp_best = train_classifier(mlp, x_train, y_train, x_test, y_test)
    
    print(f"KAN best accuracy: {kan_best:.4f}")
    print(f"MLP best accuracy: {mlp_best:.4f}")
    
    # Plot
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(kan_train, alpha=0.3, color=COLORS[0], label='KAN Train')
    ax1.plot(kan_test, color=COLORS[0], linewidth=2, label='KAN Test')
    ax1.plot(mlp_train, alpha=0.3, color=COLORS[1], label='MLP Train')
    ax1.plot(mlp_test, color=COLORS[1], linewidth=2, label='MLP Test')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(kan_acc, color=COLORS[0], linewidth=2, label=f'KAN (best={kan_best:.3f})')
    ax2.plot(mlp_acc, color=COLORS[1], linewidth=2, label=f'MLP (best={mlp_best:.3f})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Decision boundary
    ax3 = fig.add_subplot(gs[0, 2])
    with torch.no_grad():
        xx, yy = torch.meshgrid(
            torch.linspace(-2, 3, 200),
            torch.linspace(-1.5, 2, 200),
            indexing='ij'
        )
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
        z_kan = torch.sigmoid(kan(grid)).reshape(200, 200)
    
    ax3.contourf(xx.numpy(), yy.numpy(), z_kan.numpy(), levels=20, cmap='RdBu', alpha=0.5)
    ax3.scatter(x_train[:, 0], x_train[:, 1], c=y_train.squeeze().numpy(),
               cmap='RdBu', edgecolors='white', s=20, alpha=0.7)
    ax3.set_title(f'KAN Decision Boundary (acc={kan_best:.3f})')
    ax3.set_xlabel('x₁')
    ax3.set_ylabel('x₂')
    
    fig.savefig(OUT_DIR / 'benchmark_classification_moons.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: benchmark_classification_moons.png")
    
    return kan, mlp, kan_best, mlp_best


# ═══════════════════════════════════════════════════════════════════
# Benchmark 4: Grid Extension Study
# ═══════════════════════════════════════════════════════════════════

def benchmark_grid_study():
    """Study how grid size affects KAN performance."""
    print(f"\n{'='*60}")
    print(f"Benchmark: Grid Extension Study")
    print(f"{'='*60}")
    
    # Target function: multimodal fitness landscape
    def f(x):
        return x**2 + 3 * torch.sin(5 * x)
    
    x_train = torch.linspace(-2, 2, 200).unsqueeze(-1)
    y_train = f(x_train)
    x_test = torch.linspace(-2, 2, 500).unsqueeze(-1)
    y_test = f(x_test)
    
    grid_sizes = [3, 5, 8, 12, 20]
    results = []
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    for idx, g in enumerate(grid_sizes):
        kan = KAN(layer_dims=[1, 8, 4, 1], grid_size=g, spline_order=3)
        result = train_model(kan, x_train, y_train, x_test, y_test,
                            epochs=1000, lr=1e-2, update_grid_every=50,
                            title=f"KAN_G={g}")
        results.append((g, result))
        print(f"Grid size G={g}: best test loss = {result['best_test_loss']:.2e}")
    
    # Plot loss curves by grid size
    ax1 = fig.add_subplot(gs[0, :2])
    for g, result in results:
        ax1.plot(result['test_losses'], linewidth=1.5, 
                label=f'G={g} (best={result["best_test_loss"]:.1e})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('KAN Performance vs Grid Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scaling: test loss vs grid size
    ax2 = fig.add_subplot(gs[0, 2])
    gs_plot = [g for g, _ in results]
    losses = [r['best_test_loss'] for _, r in results]
    ax2.loglog(gs_plot, losses, 'o-', color=COLORS[0], markersize=8, linewidth=2)
    ax2.set_xlabel('Grid Size G')
    ax2.set_ylabel('Best Test MSE')
    ax2.set_title('Scaling Law: Loss vs Grid Size')
    ax2.grid(True, alpha=0.3)
    
    # Visualize function fits for best model
    best_idx = np.argmin(losses)
    best_g = gs_plot[best_idx]
    
    ax3 = fig.add_subplot(gs[1, :])
    
    # Train the best model fresh
    best_kan = KAN(layer_dims=[1, 8, 4, 1], grid_size=best_g, spline_order=3)
    _ = train_model(best_kan, x_train, y_train, x_test, y_test,
                   epochs=1500, lr=1e-2, update_grid_every=50)
    
    with torch.no_grad():
        x_plot = torch.linspace(-2, 2, 500).unsqueeze(-1)
        y_true = f(x_plot)
        y_pred = best_kan(x_plot)
    
    ax3.plot(x_plot.numpy(), y_true.numpy(), 'w-', linewidth=2, label='True f(x)')
    ax3.plot(x_plot.numpy(), y_pred.numpy(), '--', color=COLORS[0], linewidth=2, 
            label=f'KAN (G={best_g})')
    ax3.scatter(x_train.numpy()[::10], y_train.numpy()[::10], 
               c='gray', s=10, alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.set_title(f'Best KAN Fit — G={best_g}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.savefig(OUT_DIR / 'benchmark_grid_study.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: benchmark_grid_study.png")
    
    return results


# ═══════════════════════════════════════════════════════════════════
# Activation Function Deep Dive
# ═══════════════════════════════════════════════════════════════════

def activation_deep_dive():
    """Deep visualization of learned KAN activation functions."""
    print(f"\n{'='*60}")
    print(f"Visualization: Activation Function Deep Dive")
    print(f"{'='*60}")
    
    def f(x):
        return torch.sin(np.pi * x) + 0.5 * torch.sin(3 * np.pi * x)
    
    x_train = torch.linspace(-1.5, 1.5, 300).unsqueeze(-1)
    y_train = f(x_train)
    
    # Use a wider KAN for more activations to visualize
    kan = KAN(layer_dims=[1, 8, 1], grid_size=12, spline_order=3)
    
    # Train
    _ = train_model(kan, x_train, y_train, x_train, y_train,
                   epochs=2000, lr=1e-2, update_grid_every=50)
    
    # Plot all activations
    layer = kan.layers[0]  # First layer: 1→8
    n_in, n_out = layer.n_in, layer.n_out
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)
    
    with torch.no_grad():
        x_plot = torch.linspace(-2, 2, 500).unsqueeze(-1)
        
        # Extract each activation function's components
        base = layer.base_fn(x_plot)  # [500, 1]
        spline = coef_to_curve(x_plot, layer.grid, layer.coef, layer.spline_order)  # [500, 1, 8]
        
        for j in range(n_out):
            ax = fig.add_subplot(gs[j // 4, j % 4])
            
            # Base component
            base_comp = (layer.scale_base[0, j].item() * base[:, 0]).numpy()
            # Spline component
            spline_comp = (layer.scale_sp[0, j].item() * spline[:, 0, j]).numpy()
            # Full activation
            full = base_comp + spline_comp
            
            ax.plot(x_plot.numpy(), full, 'w-', linewidth=2, label='φ(x)')
            ax.plot(x_plot.numpy(), base_comp, '--', color=COLORS[2], linewidth=1, 
                   label='base', alpha=0.7)
            ax.plot(x_plot.numpy(), spline_comp, '--', color=COLORS[3], linewidth=1, 
                   label='spline', alpha=0.7)
            ax.axhline(y=0, color='#30363d', linewidth=0.5)
            ax.set_title(f'φ₀→{j}(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('φ(x)')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    
    fig.suptitle('KAN Activation Functions — Anatomy', fontsize=14, y=1.01)
    fig.savefig(OUT_DIR / 'activation_deep_dive.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: activation_deep_dive.png")
    
    return kan


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔" + "═" * 58 + "╗")
    print("║  KAN: Kolmogorov-Arnold Networks — Benchmark Suite       ║")
    print("╚" + "═" * 58 + "╝")
    
    # 1D benchmarks
    benchmark_1d(lambda x: torch.sin(np.pi * x), "sin(πx)")
    benchmark_1d(lambda x: torch.exp(-x.abs()) * torch.sin(2 * np.pi * x), 
                "exp(-|x|)·sin(2πx)")
    
    # 2D benchmark
    benchmark_2d(lambda x: (torch.sin(np.pi * x[:, 0]) * torch.cos(np.pi * x[:, 1])).unsqueeze(-1),
                "sin(πx₁)·cos(πx₂)")
    
    # Classification
    benchmark_classification()
    
    # Grid study
    benchmark_grid_study()
    
    # Activation deep dive
    activation_deep_dive()
    
    print(f"\n{'='*60}")
    print(f"All benchmarks complete. Visualizations saved to {OUT_DIR}")
    print(f"{'='*60}")
