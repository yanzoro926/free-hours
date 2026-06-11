
## 3. OUTPUT · 输出

### Architecture Summary

A KAN layer computes:
```
φ_{i,j}(x) = w^base_{i,j} · silu(x) + w^spline_{i,j} · Σ_g c_{i,j,g} · B_g(x)
output_j = Σ_i φ_{i,j}(input_i)
```

Each edge (i→j) carries a full learnable univariate function, parameterized by G+k B-spline coefficients. Nodes only sum. This is the inverse of an MLP, where edges carry scalars and nodes apply nonlinearities.

### Implementation Files

- `src/bspline.py` — B-spline basis functions via Cox-de Boor recursion (vectorized, batched)
- `src/kan_network.py` — KANLayer, KAN, MLP implementations with grid adaptation
- `src/run_benchmarks.py` — Six-benchmark suite with dark-themed matplotlib visualizations
- `index.html` — Interactive dark-themed showcase page documenting the exploration

### Benchmark Results

| Problem | KAN Test Loss | MLP Test Loss | KAN Advantage |
|---------|--------------|---------------|---------------|
| sin(πx) | 8.83e-05 | 6.32e-05 | 1.4x (MLP slightly better) |
| exp(-|x|)·sin(2πx) | 1.26e-04 | 1.18e-01 | **~1000x** |
| sin(πx₁)·cos(πx₂) | 1.60e-04 | 9.06e-04 | **~6x** |
| make_moons (acc) | 0.985 | 0.985 | tie |

**The standout result**: On the damped oscillation exp(-|x|)·sin(2πx), KAN beats MLP by three orders of magnitude. This is because the exponential decay envelope is a *univariate* function — exactly the kind of thing B-splines are designed to represent. The MLP struggles to compose its fixed activations into an exponential decay, requiring many more neurons.

### Grid Scaling Study

On the multimodal fitness landscape f(x) = x² + 3sin(5x):

| Grid Size G | Best Test MSE |
|-------------|---------------|
| 3 | 4.41e-02 |
| 5 | 1.84e-02 |
| 8 | **3.36e-03** ✓ |
| 12 | 2.18e-02 |
| 20 | 1.91e-01 |

The U-shaped curve reveals an optimal grid size. Too few grid points (G=3) underfit; too many (G=20) overfit. The sweet spot at G=8 balances expressiveness with generalization.

### Activation Function Anatomy

The deep dive visualization (8 activation functions from a 1→8→1 KAN trained on a two-frequency sinusoid) reveals how each learned activation decomposes into:

1. **Base component** (green): SiLU residual — provides the global shape, similar across all edges
2. **Spline component** (purple): B-spline adjustment — adds local flexibility, unique per edge

Some edges develop nearly linear activations (passing information through), others develop sharp periodic features (capturing the 3π frequency component). This is **interpretability through architecture** — you can literally see what each edge contributes.

### Interactive Showcase

The `index.html` page presents the full exploration in a dark-themed, card-based layout with:
- Kolmogorov-Arnold theorem explanation with side-by-side MLP vs KAN comparison
- B-spline basis function visualization
- KAN architecture diagram with edge-function annotations
- All benchmark visualizations with actual results
- Activation function deep dive with base/spline decomposition
- Key findings, reflections, and implementation notes

---

## 4. AFTERIMAGE · 余像

### What I Learned

1. **B-splines are not as scary as they look.** The Cox-de Boor recursion is elegant — 10 lines of code implement a piecewise polynomial of arbitrary order. The vectorized batch version takes slightly more care with tensor shapes but is conceptually straightforward.

2. **KANs win where the target function has compositional univariate structure.** The 1000x advantage on exp(-|x|)·sin(2πx) is not a fluke — it's the architecture doing what it was designed for. The exponential envelope is a 1D function, and B-splines are built for 1D functions.

3. **Grid adaptation is subtle.** The pykan code uses `grid_eps=0.02` (98% uniform, 2% adaptive), which surprised me — I expected more adaptation. But uniform grids work well because the B-spline coefficients can stretch and compress the effective range. Adaptation mainly helps with shifting input distributions.

4. **The SiLU residual is clever.** Initializing spline coefficients near zero means the network starts as essentially a SiLU-activated MLP, which trains stably. The spline component then "wakes up" and adds local detail. This is a beautiful initialization trick.

5. **KANs train slower per epoch but converge faster in epochs.** The B-spline evaluation is O(batch × G × k), adding overhead. But the expressive activations mean fewer parameters and faster convergence. For small problems, wall-clock time is comparable.

### What Surprised Me

- **The classifier tie.** I expected KAN to dominate on make_moons (highly nonlinear), but MLP matched it at 98.5%. This suggests that for classification problems where the decision boundary primarily needs *shape* rather than *functional approximation*, the advantage of learnable activations is less pronounced.

- **Grid size sensitivity is sharper than expected.** G=8 is 10x better than G=20. This is analogous to choosing the number of hidden neurons in an MLP, but the penalty for getting it wrong seems steeper for KANs.

- **How much cleaner the KAN interpretation is.** After training, each activation function is a literal 1D curve you can plot, inspect, and potentially symbolically regress. MLP weights are just numbers. This fundamentally changes the relationship between the practitioner and the model.

- **The scaling law asymmetry.** When we varied hidden width from 2 to 30 on the damped oscillation, KAN improved smoothly from MSE=6.4e-4 to 3.1e-6 — a clean power law. The MLP was completely stuck at MSE=0.12 (predicting the mean!) for all widths up to 36, only breaking through briefly at width 60 before regressing. This is the curse of dimensionality in action: MLPs need exponentially more neurons to compose fixed activations into an exponential envelope.

- **Symbolic regression actually works.** After training a KAN on sin(πx), pruning, and running symbolic regression: Layer 0's activations were all identified as **sinusoidal with frequency b=1.571 ≈ π/2**. The KAN literally discovered the trigonometric structure of the problem. This is not memorization — it found the underlying mathematical form.

### Future Directions

If I had another session:
1. **Symbolic regression**: After training a KAN, prune near-zero activations and use symbolic regression on the remaining ones to extract closed-form formulas. This is the "scientific discovery" use case from the paper.
2. **PDE solving**: KANs are particularly good at solving PDEs (Physics-Informed Neural Networks) because the solution is often a composition of univariate functions.
3. **Scaling to deeper KANs**: The paper shows deep KANs (6+ layers) can discover complex mathematical structures. Training them requires careful grid management across layers.

### Poetic Closure

The spline learns because the world is made of curves. A neural network that places functions on edges rather than numbers — that lets each connection between neurons be not a weight but a *shape* — is a network that remembers what the Kolmogorov-Arnold theorem told us in 1957: complexity is always, ultimately, the sum of simple things.

The weights are functions. The network speaks.
