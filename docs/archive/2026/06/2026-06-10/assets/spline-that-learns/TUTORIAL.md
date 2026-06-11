# KAN in Plain English · KAN通俗解释

## What's wrong with regular neural networks?

A standard neural network (MLP) has layers of neurons. Each neuron:
1. Takes weighted sum of its inputs
2. Applies a fixed activation function (like ReLU or sigmoid)
3. Passes the result to the next layer

The activation functions are the **same for every neuron** — you pick one (ReLU, SiLU, tanh) and it's fixed forever. The network can only learn the *weights* (the numbers on the edges).

This works remarkably well, but it has a limitation: to approximate a complex function, you need many neurons working together, each contributing a small piece through their fixed activation function.

## What makes KANs different?

KANs flip the formula:

- **MLP**: Fixed activations on nodes, learnable weights on edges
- **KAN**: Learnable activations on edges, only summation on nodes

Each edge in a KAN doesn't carry a single number (weight) — it carries an **entire function**. That function is itself made of small building blocks called B-splines, which are smooth piecewise polynomials.

Think of it this way: in an MLP, every neuron "sees" the world through the same lens (ReLU, for example). In a KAN, every connection between neurons can develop its **own lens** — one might learn to be curvy, another sharp, another wavy.

## Why does this help?

Three reasons:

### 1. Efficiency
Because each edge can learn its own shape, you need fewer neurons. A KAN with 150 parameters can match an MLP with 500+ parameters.

### 2. Accuracy for "compositional" functions
Many real-world functions decompose into simpler 1D pieces. The exponential decay e^{-|x|} is a 1D function — KAN's B-splines represent it perfectly. An MLP needs many neurons approximating it through ReLU composition. Result: **KAN beats MLP by 1000x** on damped oscillation tasks.

### 3. Interpretability
After training an MLP, all you have are numbers (weights). After training a KAN, each edge has a *visible function*. You can:
- Plot every activation function
- Prune edges with near-zero functions
- Run symbolic regression to find closed-form formulas

This is the "scientific discovery" angle: train a KAN on physics data, and the learned functions might reveal the underlying equation.

## The Math in One Equation

Each KAN edge computes:

```
φ(x) = w_base × silu(x) + w_spline × (c₁B₁(x) + c₂B₂(x) + ... + c_G+k·B_G+k(x))
```

- `silu(x)` is a safety net — if splines don't help, it falls back to MLP-like behavior
- The `B_i(x)` are B-spline basis functions — smooth, local, piecewise polynomials
- The `c_i` are learnable coefficients — this is what training adjusts

The output of a KAN layer is simply the sum:

```
output_j = φ_{1,j}(x₁) + φ_{2,j}(x₂) + ... + φ_{n,j}(x_n)
```

That's it. Summation on nodes, learnable functions on edges.

## The Bottom Line

KANs are not a replacement for MLPs — they're a different tool for different jobs. Use KANs when:
- Your function has compositional structure (can be broken into 1D pieces)
- You need interpretability (understand what the network learned)
- You have limited data (KANs are more parameter-efficient)

Use MLPs when:
- You need raw speed (MLPs train faster per epoch)
- Your data is truly high-dimensional with no compositional structure
- You're doing classification where the decision boundary just needs shape

The spline learns because the world is made of curves.
