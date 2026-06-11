"""
KAN Pruning and Symbolic Regression Demo

After training a KAN, we can:
1. Prune near-zero activation functions
2. Run symbolic regression on the remaining ones
3. Extract a closed-form formula for the learned function

This is the "scientific discovery" use case from the paper —
the network rediscovers mathematical formulas from data.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kan_network import KAN, MLP, count_kan_params, coef_to_curve

OUT_DIR = Path('/home/yanyj/VibeCoding/autonomy/2026-06-10/spline-that-learns/visualizations')
C = ['#58a6ff','#f78166','#3fb950','#d2a8ff','#ffa657']

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'grid.alpha': 0.5,
    'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d', 'figure.dpi': 150,
})


def train_kan(model, x_tr, y_tr, epochs=1000, lr=1e-2):
    """Train a KAN model."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.MSELoss()
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(x_tr), y_tr); loss.backward(); opt.step(); sch.step()
        if e > 0 and e % 100 == 0:
            with torch.no_grad(): model.update_grids(x_tr)
    return model


def prune_kan(kan, threshold=0.1):
    """
    Prune near-zero activation functions.
    
    An activation function is considered "near-zero" if its total variance
    over the input range is below the threshold relative to the maximum variance.
    """
    x_eval = torch.linspace(-2, 2, 200).unsqueeze(-1)
    
    for layer in kan.layers:
        with torch.no_grad():
            base = layer.base_fn(x_eval.expand(200, layer.n_in))
            spline = coef_to_curve(x_eval.expand(200, layer.n_in), 
                                   layer.grid, layer.coef, layer.spline_order)
            acts = (layer.scale_base[None, :, :] * base[:, :, None] + 
                   layer.scale_sp[None, :, :] * spline)
            
            # Compute variance of each activation
            var = acts.var(dim=0)  # [n_in, n_out]
            max_var = var.max()
            
            if max_var > 0:
                prune_mask = var < (threshold * max_var)
                layer.mask.data = layer.mask.data * (~prune_mask).float()
    
    return kan


def symbolic_regress(kan, x_range=(-2, 2), n_points=200):
    """
    Extract symbolic approximations of learned activation functions.
    
    For each activation function, we:
    1. Evaluate it at n_points
    2. Fit simple functional forms (linear, quadratic, sinusoidal)
    3. Report the best fit
    
    This is a simplified version — a full implementation would use
    genetic programming or RANSAC-based symbolic regression.
    """
    x = torch.linspace(x_range[0], x_range[1], n_points).unsqueeze(-1)
    results = []
    
    for layer_idx, layer in enumerate(kan.layers):
        layer_results = []
        with torch.no_grad():
            base = layer.base_fn(x.expand(n_points, layer.n_in))
            spline = coef_to_curve(x.expand(n_points, layer.n_in),
                                  layer.grid, layer.coef, layer.spline_order)
            acts = (layer.scale_base[None, :, :] * base[:, :, None] + 
                   layer.scale_sp[None, :, :] * spline)
            
            for i in range(layer.n_in):
                for j in range(layer.n_out):
                    if layer.mask[i, j].item() < 0.5:
                        continue
                    
                    y_act = acts[:, i, j].numpy()
                    x_np = x.squeeze(-1).numpy()
                    
                    # Fit various functional forms
                    fits = {}
                    
                    # Linear: y = ax + b
                    A = np.stack([x_np, np.ones_like(x_np)], axis=1)
                    coeffs, _, _, _ = np.linalg.lstsq(A, y_act, rcond=None)
                    fits['linear'] = (coeffs, np.mean((A @ coeffs - y_act)**2))
                    
                    # Quadratic: y = ax² + bx + c
                    A = np.stack([x_np**2, x_np, np.ones_like(x_np)], axis=1)
                    coeffs, _, _, _ = np.linalg.lstsq(A, y_act, rcond=None)
                    fits['quadratic'] = (coeffs, np.mean((A @ coeffs - y_act)**2))
                    
                    # Sinusoidal: y = a·sin(bx + c) + d
                    # Use FFT to find dominant frequency
                    fft = np.fft.rfft(y_act - y_act.mean())
                    freqs = np.fft.rfftfreq(n_points, (x_range[1]-x_range[0])/n_points)
                    dominant_freq = freqs[np.argmax(np.abs(fft[1:])) + 1] if len(fft) > 2 else 1.0
                    
                    # Fit a*sin(bx + c) + d via least squares (approximate)
                    b_guess = 2 * np.pi * dominant_freq
                    A = np.stack([
                        np.sin(b_guess * x_np),
                        np.cos(b_guess * x_np),
                        np.ones_like(x_np)
                    ], axis=1)
                    coeffs, _, _, _ = np.linalg.lstsq(A, y_act, rcond=None)
                    a = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                    c = np.arctan2(coeffs[1], coeffs[0])
                    d = coeffs[2]
                    y_pred_sin = a * np.sin(b_guess * x_np + c) + d
                    fits['sinusoidal'] = (
                        (a, b_guess, c, d), 
                        np.mean((y_pred_sin - y_act)**2)
                    )
                    
                    # Find best fit
                    best_name = min(fits, key=lambda k: fits[k][1])
                    best_params, best_error = fits[best_name]
                    
                    layer_results.append({
                        'edge': f'{i}→{j}',
                        'best_form': best_name,
                        'params': best_params,
                        'mse': best_error,
                    })
        
        results.append(layer_results)
    
    return results


def demo_symbolic_regression():
    """
    Train a KAN on a known formula, then try to recover it via symbolic regression.
    
    Target: f(x) = sin(πx) — simple, should yield interpretable activations.
    """
    print("=== Symbolic Regression Demo ===", flush=True)
    
    # Target function: known formula for validation
    def f_target(x):
        return torch.sin(np.pi * x)
    
    x_tr = torch.linspace(-2, 2, 300).unsqueeze(-1)
    y_tr = f_target(x_tr)
    x_te = torch.linspace(-2, 2, 500).unsqueeze(-1)
    
    # Train KAN
    kan = KAN([1, 5, 5, 1], grid_size=10, spline_order=3)
    kan = train_kan(kan, x_tr, y_tr, epochs=1500, lr=1e-2)
    
    with torch.no_grad():
        test_loss = torch.nn.MSELoss()(kan(x_te), f_target(x_te))
    print(f"Trained KAN: test MSE = {test_loss.item():.2e}", flush=True)
    
    # Prune
    kan = prune_kan(kan, threshold=0.05)
    
    # Count active edges
    active = 0
    for layer in kan.layers:
        active += layer.mask.sum().item()
    print(f"Active edges after pruning: {active}", flush=True)
    
    # Symbolic regression
    formulas = symbolic_regress(kan)
    
    # Visualize
    fig = plt.figure(figsize=(18, 12))
    gs = plt.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)
    
    # Plot the fit
    ax = fig.add_subplot(gs[0, :2])
    with torch.no_grad():
        xp = torch.linspace(-2, 2, 500).unsqueeze(-1)
        y_true = f_target(xp)
        y_pred = kan(xp)
    ax.plot(xp.numpy(), y_true.numpy(), 'w-', lw=2, label='True: sin(πx)')
    ax.plot(xp.numpy(), y_pred.numpy(), '--', color=C[0], lw=2, label=f'KAN (MSE={test_loss.item():.2e})')
    ax.scatter(x_tr.numpy()[::15], y_tr.numpy()[::15], c='gray', s=8, alpha=0.5)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('KAN Approximation of sin(πx)')
    
    # Pruning info
    ax = fig.add_subplot(gs[0, 2:])
    layer_names = [f'Layer {i}: {l.n_in}→{l.n_out}' for i, l in enumerate(kan.layers)]
    total = [l.n_in * l.n_out for l in kan.layers]
    active_counts = [int(l.mask.sum().item()) for l in kan.layers]
    
    x_pos = np.arange(len(layer_names))
    width = 0.35
    ax.bar(x_pos - width/2, total, width, color='#30363d', label='Total edges')
    ax.bar(x_pos + width/2, active_counts, width, color=C[0], label='Active edges')
    ax.set_xticks(x_pos); ax.set_xticklabels(layer_names, fontsize=9)
    ax.set_ylabel('Number of edges'); ax.set_title('Edge Pruning')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    # Visualize activations with symbolic fits
    xp2 = torch.linspace(-2, 2, 500).unsqueeze(-1)
    plot_idx = 0
    
    for layer_idx, (layer, layer_formulas) in enumerate(zip(kan.layers, formulas)):
        if layer_idx > 1: break  # Only show first two layers
        with torch.no_grad():
            base = layer.base_fn(xp2.expand(500, layer.n_in))
            spline = coef_to_curve(xp2.expand(500, layer.n_in), 
                                  layer.grid, layer.coef, layer.spline_order)
            acts = (layer.scale_base[None, :, :] * base[:, :, None] + 
                   layer.scale_sp[None, :, :] * spline)
        
        for i in range(layer.n_in):
            for j in range(layer.n_out):
                if plot_idx >= 8: break
                if layer.mask[i, j].item() < 0.5: continue
                
                ax = fig.add_subplot(gs[1 + plot_idx // 4, plot_idx % 4])
                ax.plot(xp2.numpy(), acts[:, i, j].numpy(), 'w-', lw=2, label=f'φ_{i}→{j}')
                
                # Find the formula for this edge
                formula = None
                for f in layer_formulas:
                    if f['edge'] == f'{i}→{j}':
                        formula = f
                        break
                
                if formula:
                    # Plot the symbolic fit
                    x_np = xp2.squeeze(-1).numpy()
                    if formula['best_form'] == 'linear':
                        a, c = formula['params']
                        y_fit = a * x_np + c
                        label = f'y={a:.3f}x+{c:.3f}'
                    elif formula['best_form'] == 'quadratic':
                        a, b, c = formula['params']
                        y_fit = a * x_np**2 + b * x_np + c
                        label = f'y={a:.3f}x²+{b:.3f}x+{c:.3f}'
                    elif formula['best_form'] == 'sinusoidal':
                        a_s, b_s, c_s, d_s = formula['params']
                        y_fit = a_s * np.sin(b_s * x_np + c_s) + d_s
                        label = f'y={a_s:.2f}sin({b_s:.2f}x+{c_s:.2f})+{d_s:.2f}'
                    
                    ax.plot(x_np, y_fit, '--', color=C[3], lw=1.5, alpha=0.7, label=label)
                
                ax.axhline(y=0, color='#30363d', lw=0.5)
                ax.set_title(f'φ{i}→{j} ({formula["best_form"] if formula else "N/A"})'[:30], fontsize=9)
                ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
                plot_idx += 1
    
    fig.suptitle('KAN Symbolic Regression — Recovering sin(πx)', fontsize=14, y=1.01)
    fig.savefig(OUT_DIR / 'symbolic_regression.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: symbolic_regression.png", flush=True)
    
    # Print discovered formulas
    print("\nDiscovered Activation Formulas:")
    for layer_idx, layer_formulas in enumerate(formulas):
        print(f"\n  Layer {layer_idx}:")
        for f in layer_formulas[:5]:  # Show first 5
            params_str = ', '.join([f'{p:.3f}' for p in f['params']])
            print(f"    Edge {f['edge']}: {f['best_form']} ({params_str}), MSE={f['mse']:.2e}")
    
    return kan, formulas


if __name__ == "__main__":
    demo_symbolic_regression()
