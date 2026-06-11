#!/usr/bin/env python3
"""Fast benchmark suite for KAN vs MLP."""
import torch, numpy as np, time, sys, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kan_network import KAN, MLP, count_kan_params, coef_to_curve

OUT_DIR = Path('/home/yanyj/VibeCoding/autonomy/2026-06-10/spline-that-learns/visualizations')
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e', 'ytick.color': '#8b949e',
    'grid.color': '#21262d', 'grid.alpha': 0.5,
    'legend.facecolor': '#161b22', 'legend.edgecolor': '#30363d', 'figure.dpi': 150,
})
C = ['#58a6ff','#f78166','#3fb950','#d2a8ff','#ffa657']

def train(model, x_tr, y_tr, x_te, y_te, epochs=500, lr=1e-2, grid_update=100):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = torch.nn.MSELoss()
    tr_loss, te_loss = [], []
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(x_tr), y_tr); loss.backward(); opt.step(); sch.step()
        if isinstance(model,KAN) and e>0 and e%grid_update==0:
            with torch.no_grad(): model.update_grids(x_tr)
        tr_loss.append(loss.item())
        model.eval()
        with torch.no_grad(): te_loss.append(crit(model(x_te), y_te).item())
        if e % 200 == 0:
            print(f'    epoch {e}: train={loss.item():.5f}, test={te_loss[-1]:.5f}', flush=True)
    return tr_loss, te_loss

# Common data
x_tr = torch.linspace(-2, 2, 200).unsqueeze(-1)
x_te = torch.linspace(-2, 2, 500).unsqueeze(-1)
xp = torch.linspace(-2, 2, 500).unsqueeze(-1)

# === Benchmark 1: sin(πx) ===
print('\n### Benchmark 1: sin(πx) ###', flush=True)
t0 = time.time()
f1 = lambda x: torch.sin(np.pi * x)
kan1 = KAN([1, 5, 5, 1], 8, 3); mlp1 = MLP([1, 20, 20, 1])
k_tr, k_te = train(kan1, x_tr, f1(x_tr), x_te, f1(x_te), 800)
m_tr, m_te = train(mlp1, x_tr, f1(x_tr), x_te, f1(x_te), 800)
print(f'KAN best={min(k_te):.2e}, MLP best={min(m_te):.2e}, time={time.time()-t0:.1f}s', flush=True)

fig = plt.figure(figsize=(16, 5)); gs = GridSpec(1, 3, figure=fig, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
ax.plot(k_tr, alpha=0.3, color=C[0]); ax.plot(k_te, color=C[0], lw=2, label='KAN')
ax.plot(m_tr, alpha=0.3, color=C[1]); ax.plot(m_te, color=C[1], lw=2, label='MLP')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Loss — sin(πx)')
ax = fig.add_subplot(gs[0, 1])
with torch.no_grad():
    ax.plot(xp.numpy(), f1(xp).numpy(), 'w-', lw=2, label='True')
    ax.plot(xp.numpy(), kan1(xp).numpy(), '--', color=C[0], lw=2, label='KAN')
    ax.plot(xp.numpy(), mlp1(xp).numpy(), '--', color=C[1], lw=2, label='MLP')
ax.scatter(x_tr.numpy()[::10], f1(x_tr).numpy()[::10], c='gray', s=5, alpha=0.5)
ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Function Fit')
ax = fig.add_subplot(gs[0, 2])
layer = kan1.layers[0]
xp2 = torch.linspace(-2, 2, 500).unsqueeze(-1)
with torch.no_grad():
    base = layer.base_fn(xp2.expand(500, 1))
    spl = coef_to_curve(xp2.expand(500, 1), layer.grid, layer.coef, layer.spline_order)
    acts = layer.scale_base[None, :, :] * base[:, :, None] + layer.scale_sp[None, :, :] * spl
for j in range(min(5, layer.n_out)):
    ax.plot(xp2.numpy(), acts[:, 0, j].numpy(), color=C[j % len(C)], lw=1, alpha=0.7, label=f'φ→{j}')
ax.axhline(y=0, color='#30363d', lw=0.5); ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_title('Activations (Layer 1)')
fig.savefig(OUT_DIR / 'benchmark_1d_sin_pix.png', bbox_inches='tight'); plt.close()
print('  Saved benchmark_1d_sin_pix.png', flush=True)

# === Benchmark 2: damped oscillation ===
print('\n### Benchmark 2: exp(-|x|)·sin(2πx) ###', flush=True)
t0 = time.time()
f2 = lambda x: torch.exp(-x.abs()) * torch.sin(2 * np.pi * x)
kan2 = KAN([1, 5, 5, 1], 8, 3); mlp2 = MLP([1, 20, 20, 1])
k2_tr, k2_te = train(kan2, x_tr, f2(x_tr), x_te, f2(x_te), 800)
m2_tr, m2_te = train(mlp2, x_tr, f2(x_tr), x_te, f2(x_te), 800)
print(f'KAN best={min(k2_te):.2e}, MLP best={min(m2_te):.2e}, time={time.time()-t0:.1f}s', flush=True)

fig = plt.figure(figsize=(16, 5)); gs = GridSpec(1, 3, figure=fig, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
ax.plot(k2_tr, alpha=0.3, color=C[0]); ax.plot(k2_te, color=C[0], lw=2, label='KAN')
ax.plot(m2_tr, alpha=0.3, color=C[1]); ax.plot(m2_te, color=C[1], lw=2, label='MLP')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Loss — Damped Oscillation')
ax = fig.add_subplot(gs[0, 1])
with torch.no_grad():
    ax.plot(xp.numpy(), f2(xp).numpy(), 'w-', lw=2, label='True')
    ax.plot(xp.numpy(), kan2(xp).numpy(), '--', color=C[0], lw=2, label='KAN')
    ax.plot(xp.numpy(), mlp2(xp).numpy(), '--', color=C[1], lw=2, label='MLP')
ax.scatter(x_tr.numpy()[::10], f2(x_tr).numpy()[::10], c='gray', s=5, alpha=0.5)
ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Function Fit')
ax = fig.add_subplot(gs[0, 2])
layer = kan2.layers[0]
with torch.no_grad():
    base = layer.base_fn(xp2.expand(500, 1))
    spl = coef_to_curve(xp2.expand(500, 1), layer.grid, layer.coef, layer.spline_order)
    acts = layer.scale_base[None, :, :] * base[:, :, None] + layer.scale_sp[None, :, :] * spl
for j in range(min(5, layer.n_out)):
    ax.plot(xp2.numpy(), acts[:, 0, j].numpy(), color=C[j % len(C)], lw=1, alpha=0.7)
ax.axhline(y=0, color='#30363d', lw=0.5); ax.grid(True, alpha=0.3); ax.set_title('Activations (Layer 1)')
fig.savefig(OUT_DIR / 'benchmark_1d_exp-x_sin2pix.png', bbox_inches='tight'); plt.close()
print('  Saved benchmark_1d_exp-x_sin2pix.png', flush=True)

# === Benchmark 3: 2D function ===
print('\n### Benchmark 3: sin(πx₁)·cos(πx₂) ###', flush=True)
t0 = time.time()
n = 25; x1 = torch.linspace(-1, 1, n); x2 = torch.linspace(-1, 1, n)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
x_tr2d = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
f3 = lambda x: (torch.sin(np.pi * x[:, 0]) * torch.cos(np.pi * x[:, 1])).unsqueeze(-1)
n2 = 50; x1t = torch.linspace(-1, 1, n2); x2t = torch.linspace(-1, 1, n2)
X1t, X2t = torch.meshgrid(x1t, x2t, indexing='ij')
x_te2d = torch.stack([X1t.flatten(), X2t.flatten()], dim=-1)
kan3 = KAN([2, 5, 5, 1], 8, 3); mlp3 = MLP([2, 20, 20, 1])
k3_tr, k3_te = train(kan3, x_tr2d, f3(x_tr2d), x_te2d, f3(x_te2d), 600)
m3_tr, m3_te = train(mlp3, x_tr2d, f3(x_tr2d), x_te2d, f3(x_te2d), 600)
print(f'KAN best={min(k3_te):.2e}, MLP best={min(m3_te):.2e}, time={time.time()-t0:.1f}s', flush=True)

fig = plt.figure(figsize=(18, 10)); gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
ax = fig.add_subplot(gs[0, :])
ax.plot(k3_tr, alpha=0.3, color=C[0]); ax.plot(k3_te, color=C[0], lw=2, label='KAN')
ax.plot(m3_tr, alpha=0.3, color=C[1]); ax.plot(m3_te, color=C[1], lw=2, label='MLP')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Loss — 2D Function')
with torch.no_grad():
    yt = f3(x_te2d).reshape(n2, n2); yk = kan3(x_te2d).reshape(n2, n2); ym = mlp3(x_te2d).reshape(n2, n2)
vmin = min(yt.min(), yk.min(), ym.min()); vmax = max(yt.max(), yk.max(), ym.max())
for idx, (title, data) in enumerate([('True', yt), ('KAN', yk), ('MLP', ym)]):
    ax = fig.add_subplot(gs[1, idx])
    im = ax.pcolormesh(X1t, X2t, data.numpy(), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title); plt.colorbar(im, ax=ax)
fig.savefig(OUT_DIR / 'benchmark_2d_sinpix1_cospix2.png', bbox_inches='tight'); plt.close()
print('  Saved benchmark_2d_sinpix1_cospix2.png', flush=True)

# === Benchmark 4: Classification ===
print('\n### Benchmark 4: Classification (make_moons) ###', flush=True)
t0 = time.time()
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=600, noise=0.15, random_state=42)
X = torch.tensor(X, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
x_trc, x_tec = X[:400], X[400:]; y_trc, y_tec = y[:400], y[400:]
kan4 = KAN([2, 8, 4, 1], 8, 3); mlp4 = MLP([2, 32, 16, 1])
crit_bce = torch.nn.BCEWithLogitsLoss()

def train_cls(model, x_tr, y_tr, x_te, y_te, epochs=600, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_l, te_l, te_a = [], [], []
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit_bce(model(x_tr), y_tr); loss.backward(); opt.step(); sch.step()
        if isinstance(model, KAN) and e > 0 and e % 100 == 0:
            with torch.no_grad(): model.update_grids(x_tr)
        tr_l.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred = model(x_te); te_l.append(crit_bce(pred, y_te).item())
            te_a.append(((torch.sigmoid(pred) > 0.5).float() == y_te).float().mean().item())
        if e % 200 == 0:
            print(f'    epoch {e}: loss={loss.item():.4f}, acc={te_a[-1]:.4f}', flush=True)
    return tr_l, te_l, te_a

k4_tr, k4_te, k4_acc = train_cls(kan4, x_trc, y_trc, x_tec, y_tec)
m4_tr, m4_te, m4_acc = train_cls(mlp4, x_trc, y_trc, x_tec, y_tec)
k4_b, m4_b = max(k4_acc), max(m4_acc)
print(f'KAN acc={k4_b:.4f}, MLP acc={m4_b:.4f}, time={time.time()-t0:.1f}s', flush=True)

fig = plt.figure(figsize=(18, 5)); gs = GridSpec(1, 3, figure=fig, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
ax.plot(k4_tr, alpha=0.3, color=C[0]); ax.plot(k4_te, color=C[0], lw=2, label='KAN')
ax.plot(m4_tr, alpha=0.3, color=C[1]); ax.plot(m4_te, color=C[1], lw=2, label='MLP')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('BCE Loss')
ax = fig.add_subplot(gs[0, 1])
ax.plot(k4_acc, color=C[0], lw=2, label=f'KAN ({k4_b:.3f})')
ax.plot(m4_acc, color=C[1], lw=2, label=f'MLP ({m4_b:.3f})')
ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Test Accuracy')
ax = fig.add_subplot(gs[0, 2])
with torch.no_grad():
    xx, yy = torch.meshgrid(torch.linspace(-2, 3, 150), torch.linspace(-1.5, 2, 150), indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    z = torch.sigmoid(kan4(grid)).reshape(150, 150)
ax.contourf(xx.numpy(), yy.numpy(), z.numpy(), levels=20, cmap='RdBu', alpha=0.5)
ax.scatter(x_trc[:, 0], x_trc[:, 1], c=y_trc.squeeze().numpy(), cmap='RdBu', edgecolors='white', s=15, alpha=0.7)
ax.set_title(f'KAN Decision Boundary (acc={k4_b:.3f})')
fig.savefig(OUT_DIR / 'benchmark_classification_moons.png', bbox_inches='tight'); plt.close()
print('  Saved benchmark_classification_moons.png', flush=True)

# === Benchmark 5: Grid scaling ===
print('\n### Benchmark 5: Grid Scaling Study ###', flush=True)
t0 = time.time()
f5 = lambda x: x**2 + 3 * torch.sin(5 * x)
grid_sizes = [3, 5, 8, 12, 20]
results = []
for g in grid_sizes:
    k = KAN([1, 8, 4, 1], g, 3)
    tr, te = train(k, x_tr, f5(x_tr), x_te, f5(x_te), 500)
    results.append((g, min(te)))
    print(f'  G={g}: best_test={min(te):.2e}', flush=True)

fig = plt.figure(figsize=(16, 5)); gs = GridSpec(1, 3, figure=fig, wspace=0.35)
ax = fig.add_subplot(gs[0, 0])
for g, _ in results:
    k = KAN([1, 8, 4, 1], g, 3)
    tr, te = train(k, x_tr, f5(x_tr), x_te, f5(x_te), 500)
    ax.plot(te, lw=1.5, label=f'G={g}')
ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Test Loss by Grid Size')
ax = fig.add_subplot(gs[0, 1])
gs_, ls_ = [r[0] for r in results], [r[1] for r in results]
ax.loglog(gs_, ls_, 'o-', color=C[0], markersize=8, lw=2)
ax.set_xlabel('Grid Size G'); ax.set_ylabel('Best Test MSE'); ax.grid(True, alpha=0.3); ax.set_title('Scaling: Loss ∝ G^{-k}')
ax = fig.add_subplot(gs[0, 2])
best_g = gs_[ls_.index(min(ls_))]
bk = KAN([1, 8, 4, 1], best_g, 3); train(bk, x_tr, f5(x_tr), x_te, f5(x_te), 800)
with torch.no_grad():
    ax.plot(xp.numpy(), f5(xp).numpy(), 'w-', lw=2, label='True')
    ax.plot(xp.numpy(), bk(xp).numpy(), '--', color=C[0], lw=2, label=f'KAN (G={best_g})')
ax.scatter(x_tr.numpy()[::10], f5(x_tr).numpy()[::10], c='gray', s=5, alpha=0.5)
ax.legend(); ax.grid(True, alpha=0.3); ax.set_title(f'Best KAN Fit (G={best_g})')
fig.savefig(OUT_DIR / 'benchmark_grid_study.png', bbox_inches='tight'); plt.close()
print(f'  Saved benchmark_grid_study.png, time={time.time()-t0:.1f}s', flush=True)

# === Deep Dive: Activation Functions ===
print('\n### Deep Dive: Activation Anatomy ###', flush=True)
t0 = time.time()
f6 = lambda x: torch.sin(np.pi * x) + 0.5 * torch.sin(3 * np.pi * x)
x_tr6 = torch.linspace(-1.5, 1.5, 300).unsqueeze(-1)
kan6 = KAN([1, 8, 1], 12, 3); train(kan6, x_tr6, f6(x_tr6), x_tr6, f6(x_tr6), 1000)
layer = kan6.layers[0]
fig = plt.figure(figsize=(16, 10)); gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)
xp3 = torch.linspace(-2, 2, 500).unsqueeze(-1)
with torch.no_grad():
    base = layer.base_fn(xp3)
    spl = coef_to_curve(xp3, layer.grid, layer.coef, layer.spline_order)
    for j in range(8):
        ax = fig.add_subplot(gs[j // 4, j % 4])
        bc = (layer.scale_base[0, j].item() * base[:, 0]).numpy()
        sc = (layer.scale_sp[0, j].item() * spl[:, 0, j]).numpy()
        full = bc + sc
        ax.plot(xp3.numpy(), full, 'w-', lw=2, label='φ(x)')
        ax.plot(xp3.numpy(), bc, '--', color=C[2], lw=1, label='base', alpha=0.7)
        ax.plot(xp3.numpy(), sc, '--', color=C[3], lw=1, label='spline', alpha=0.7)
        ax.axhline(y=0, color='#30363d', lw=0.5); ax.set_title(f'φ₀→{j}(x)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle('KAN Activation Functions — Base + Spline Decomposition', fontsize=14, y=1.01)
fig.savefig(OUT_DIR / 'activation_deep_dive.png', bbox_inches='tight'); plt.close()
print(f'  Saved activation_deep_dive.png, time={time.time()-t0:.1f}s', flush=True)

print('\n' + '='*60)
print('ALL BENCHMARKS COMPLETE')
print('='*60)
