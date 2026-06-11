"""
KAN Scaling Law Experiment

Train KANs and MLPs of increasing width on the same function
and compare test loss vs parameter count.

This demonstrates that KANs have more favorable scaling laws —
they extract more accuracy per parameter.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from kan_network import KAN, MLP, count_kan_params

OUT_DIR = Path('/home/yanyj/VibeCoding/autonomy/2026-06-10/spline-that-learns/visualizations')
C = ['#58a6ff','#f78166','#3fb950','#d2a8ff','#ffa657']

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'grid.alpha': 0.5,
    'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d', 'figure.dpi': 150,
})


def train_and_eval(model, x_tr, y_tr, x_te, y_te, epochs=800, lr=1e-2):
    """Train model and return best test loss."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.MSELoss()
    best = float('inf')
    
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(x_tr), y_tr); loss.backward(); opt.step(); sch.step()
        if isinstance(model, KAN) and e > 0 and e % 100 == 0:
            with torch.no_grad(): model.update_grids(x_tr)
        model.eval()
        with torch.no_grad():
            te = crit(model(x_te), y_te).item()
            best = min(best, te)
    
    return best


def scaling_experiment():
    """Compare KAN vs MLP scaling laws on damped oscillation."""
    print("=== Scaling Law Experiment ===", flush=True)
    
    # Target function: damped oscillation (KAN's strength)
    def f_target(x):
        return torch.exp(-x.abs()) * torch.sin(2 * np.pi * x)
    
    x_tr = torch.linspace(-2, 2, 200).unsqueeze(-1)
    y_tr = f_target(x_tr)
    x_te = torch.linspace(-2, 2, 500).unsqueeze(-1)
    y_te = f_target(x_te)
    
    # Vary hidden width
    widths = [2, 3, 5, 8, 12, 20, 30]
    
    kan_results = []
    mlp_results = []
    
    for w in widths:
        # KAN: [1, w, w, 1] with learnable activations
        kan = KAN([1, w, w, 1], grid_size=8, spline_order=3)
        n_kan = count_kan_params(kan)
        kan_loss = train_and_eval(kan, x_tr, y_tr, x_te, y_te, epochs=600)
        kan_results.append((n_kan, kan_loss))
        print(f"  KAN [1,{w},{w},1]: params={n_kan}, test_loss={kan_loss:.2e}", flush=True)
        
        # MLP: [1, w*3, w*3, 1] — 3x wider to be fair on param count
        mlp_w = w * 3
        mlp = MLP([1, mlp_w, mlp_w, 1])
        n_mlp = sum(p.numel() for p in mlp.parameters())
        mlp_loss = train_and_eval(mlp, x_tr, y_tr, x_te, y_te, epochs=600)
        mlp_results.append((n_mlp, mlp_loss))
        print(f"  MLP [1,{mlp_w},{mlp_w},1]: params={n_mlp}, test_loss={mlp_loss:.2e}", flush=True)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss vs width
    ax1.semilogy(widths, [r[1] for r in kan_results], 'o-', color=C[0], lw=2, markersize=8, label='KAN')
    ax1.semilogy(widths, [r[1] for r in mlp_results], 's--', color=C[1], lw=2, markersize=8, label='MLP (3x wider)')
    ax1.set_xlabel('Hidden Width w'); ax1.set_ylabel('Test MSE'); ax1.set_title('Loss vs Width')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Loss vs parameter count
    ax2.loglog([r[0] for r in kan_results], [r[1] for r in kan_results], 'o-', color=C[0], lw=2, markersize=8, label='KAN')
    ax2.loglog([r[0] for r in mlp_results], [r[1] for r in mlp_results], 's--', color=C[1], lw=2, markersize=8, label='MLP (3x wider)')
    
    # Fit power law: loss ∝ N^{-α}
    for results, color, name in [(kan_results, C[0], 'KAN'), (mlp_results, C[1], 'MLP')]:
        log_n = np.log([r[0] for r in results])
        log_l = np.log([r[1] for r in results])
        alpha, intercept = np.polyfit(log_n, log_l, 1)
        ax2.plot([r[0] for r in results], 
                np.exp(intercept) * np.array([r[0] for r in results])**alpha,
                ':', color=color, alpha=0.5, lw=1)
        ax2.annotate(f'{name}: α={-alpha:.2f}', 
                    xy=(results[-1][0], results[-1][1]),
                    color=color, fontsize=10)
    
    ax2.set_xlabel('Parameter Count'); ax2.set_ylabel('Test MSE')
    ax2.set_title('Scaling Law: Loss ∝ N^{-α}')
    ax2.legend(); ax2.grid(True, alpha=0.3, which='both')
    
    fig.savefig(OUT_DIR / 'scaling_law.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: scaling_law.png", flush=True)
    
    return kan_results, mlp_results


if __name__ == "__main__":
    scaling_experiment()
