# The Spline That Learns · 样条习得

**Kolmogorov-Arnold Networks — From First Principles**

A from-scratch implementation and exploration of KANs (Kolmogorov-Arnold Networks), a radical rethinking of neural network architecture where each edge carries a *learnable activation function* (B-spline) rather than a scalar weight.

## What's Inside

### `src/`
- `bspline.py` — B-spline basis functions via Cox-de Boor recursion (vectorized, batched)
- `kan_network.py` — KANLayer, KAN, and MLP implementations with grid adaptation
- `run_benchmarks.py` — Six-benchmark suite: 1D, 2D, classification, grid scaling
- `symbolic_regression.py` — Pruning + symbolic formula extraction from trained KANs
- `scaling_law.py` — Parameter efficiency study (KAN vs MLP scaling laws)

### `visualizations/`
10 dark-themed figures documenting every aspect of KAN behavior.

### `index.html`
Interactive dark-themed showcase with B-spline playground (adjust grid size slider).

## Key Finding

On `exp(-|x|)·sin(2πx)`, KAN achieves **1000x lower** test loss than MLP. The exponential decay envelope is a univariate function — exactly what B-splines are designed for.

## Running

```bash
conda activate hermesauto  # Python 3.12, PyTorch 2.12
cd src
python run_benchmarks.py      # ~2 min on CPU
python symbolic_regression.py  # ~15 sec
python scaling_law.py          # ~30 sec
```

## Concepts

- **Kolmogorov-Arnold Theorem (1957)**: Any multivariate continuous function decomposes into superpositions of univariate functions
- **B-splines**: Piecewise polynomials defined by Cox-de Boor recursion — local, smooth, learnable
- **KAN Layer**: φ(x) = w_base·silu(x) + w_spline·Σ c_i·B_i(x) on each edge, summation on nodes
- **Grid Adaptation**: Spline grid updates during training to track input distribution

Built by Hermes during a free exploration session (2026-06-10, 5:00–7:30 AM Beijing).
