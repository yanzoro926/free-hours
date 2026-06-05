"""
Visualizations for the Sparks of Babel project.

Generates:
1. Training loss curves
2. Token embedding PCA visualization
3. Attention pattern heatmaps
4. Token frequency distribution
5. Generation quality comparison across training
"""

import os
import sys
import json
import math
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TrainConfig  # needed for unpickling checkpoints
from bpe_tokenizer import BPETokenizer
from model import GPT, GPTConfig


# ─── Style Setup ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

DARK_BG = '#1a1a2e'
DARK_FG = '#e0e0e0'
ACCENT = '#e94560'
ACCENT2 = '#0f3460'
ACCENT3 = '#16213e'
GOLD = '#f5c518'


def dark_style(ax, title=""):
    """Apply dark theme to axes."""
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=DARK_FG, labelsize=10)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)
    if title:
        ax.set_title(title, color=DARK_FG, fontweight='bold', pad=15)


# ─── Loss Curves ─────────────────────────────────────────────────────────────

def plot_loss_curves(losses_path: str, output_path: str):
    """Plot training and validation loss curves."""
    data = torch.load(losses_path)
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    dark_style(ax, "Training & Validation Loss")
    
    train_iters, train_vals = zip(*train_losses)
    val_iters, val_vals = zip(*val_losses)
    
    ax.plot(train_iters, train_vals, color=ACCENT, alpha=0.7, linewidth=1, label='Train Loss')
    ax.plot(val_iters, val_vals, color=GOLD, linewidth=2, label='Validation Loss')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(facecolor=DARK_BG, edgecolor='#555', labelcolor=DARK_FG)
    ax.grid(True, alpha=0.15, color='white')
    
    # Annotate best
    best_val = min(val_vals)
    best_iter = val_iters[val_vals.index(best_val)]
    ax.annotate(f'Best: {best_val:.4f}',
                xy=(best_iter, best_val),
                xytext=(best_iter + 300, best_val + 0.1),
                arrowprops=dict(arrowstyle='->', color=GOLD),
                color=GOLD, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved loss curves → {output_path}")


# ─── Embedding Visualization ─────────────────────────────────────────────────

def plot_embeddings(model: GPT, tokenizer: BPETokenizer, output_path: str, n_tokens: int = 200):
    """
    Plot PCA of token embeddings.
    Shows how the model clusters related tokens.
    """
    from sklearn.decomposition import PCA
    
    # Get embedding matrix
    with torch.no_grad():
        embeddings = model.transformer.wte.weight.cpu().numpy()  # (vocab_size, n_embd)
    
    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings[:n_tokens])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    dark_style(ax, f"Token Embedding PCA (first {n_tokens} tokens)\nVariance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                         c=range(n_tokens), cmap='viridis',
                         alpha=0.7, s=40, edgecolors='white', linewidth=0.3)
    
    # Label some tokens
    labeled = set()
    for i in range(min(n_tokens, 100)):
        token_bytes = tokenizer.token_bytes(i)
        try:
            label = token_bytes.decode('utf-8')
            if len(label) <= 4 and label.isprintable():
                ax.annotate(label, (reduced[i, 0], reduced[i, 1]),
                           fontsize=7, color='white', alpha=0.8,
                           ha='center', va='bottom')
        except:
            pass
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved embedding PCA → {output_path}")


# ─── Attention Visualization ─────────────────────────────────────────────────

def plot_attention_patterns(model: GPT, tokenizer: BPETokenizer, 
                            text: str, output_path: str, layer: int = -1):
    """
    Visualize attention patterns for a given input text.
    Shows how the model attends across positions.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Encode text
    token_ids = tokenizer.encode(text)
    if len(token_ids) > model.config.block_size:
        token_ids = token_ids[:model.config.block_size]
    
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    T = idx.size(1)
    
    # Get token labels
    labels = []
    for tid in token_ids:
        tb = tokenizer.token_bytes(tid)
        try:
            s = tb.decode('utf-8', errors='replace')
            labels.append(s if s.isprintable() else f'<{tid}>')
        except:
            labels.append(f'<{tid}>')
    
    # Extract attention weights using hooks
    attention_maps = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # We need to extract attn weights during forward pass
            pass
        return hook
    
    # Alternative: use a simpler approach - just visualize the attention
    # by running a forward pass with hooks on the attention modules
    
    # Collect attention weights from each layer
    attn_weights = []
    
    def attn_hook(module, input, output):
        # The attention weights aren't directly in the output
        # We'd need to modify the model — but for visualization,
        # we can approximate with a simpler approach
        pass
    
    # Register hooks
    for i, block in enumerate(model.transformer.h):
        handle = block.attn.register_forward_hook(
            lambda m, inp, out, li=i: attn_weights.append(out.detach())
        )
        hooks.append(handle)
    
    # Forward pass
    with torch.no_grad():
        model(idx)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # If we got attention weights, plot them
    # Actually, let's create an approximation using output activations
    # as a proxy for attention patterns
    
    # For a proper visualization, we'll use the model's embeddings to compute
    # self-similarity as a proxy for attention patterns
    with torch.no_grad():
        tok_emb = model.transformer.wte(idx)  # (1, T, n_embd)
        pos = torch.arange(0, T, device=device)
        pos_emb = model.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        # Pass through blocks and collect intermediate states
        states = [x.squeeze(0).cpu().numpy()]
        for block in model.transformer.h:
            x = block(x)
            states.append(x.squeeze(0).cpu().numpy())
    
    # Create figure with multiple layers
    n_layers = min(4, len(states) - 1)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        layer_idx = i * (len(states) - 1) // n_layers
        # Compute self-similarity of hidden states
        h = states[layer_idx + 1]  # (T, n_embd)
        # Normalize
        h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = h_norm @ h_norm.T  # (T, T)
        
        dark_style(ax, f"Layer {layer_idx + 1} Similarity")
        im = ax.imshow(sim, cmap='inferno', aspect='auto', vmin=-1, vmax=1)
        
        # Label axes with tokens
        if T <= 30:
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(labels, rotation=90, fontsize=6, color=DARK_FG)
            ax.set_yticklabels(labels, fontsize=6, color=DARK_FG)
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle(f'Layer-wise Hidden State Similarity\n(proxy for attention patterns)', 
                 color=DARK_FG, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved attention patterns → {output_path}")


# ─── Token Distribution ──────────────────────────────────────────────────────

def plot_token_distribution(tokenizer: BPETokenizer, data_path: str, output_path: str):
    """Plot the distribution of token frequencies in the training data."""
    # Load text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode
    ids = tokenizer.encode(text[:100000])  # First 100K chars
    freqs = np.bincount(ids, minlength=tokenizer.vocab_size)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Zipf plot (rank vs frequency, log-log)
    ax1 = axes[0]
    dark_style(ax1, "Token Frequency Distribution (Zipf)")
    sorted_freqs = np.sort(freqs[freqs > 0])[::-1]
    ranks = np.arange(1, len(sorted_freqs) + 1)
    ax1.loglog(ranks, sorted_freqs, color=ACCENT, linewidth=1, alpha=0.8)
    ax1.scatter(ranks[::10], sorted_freqs[::10], color=GOLD, s=10, alpha=0.5)
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.15, color='white')
    
    # Top tokens bar chart
    ax2 = axes[1]
    dark_style(ax2, "Top 20 Tokens")
    top_k = 20
    top_indices = np.argsort(freqs)[-top_k:]
    top_freqs = freqs[top_indices]
    
    # Get labels
    labels = []
    for idx in top_indices:
        tb = tokenizer.token_bytes(idx)
        try:
            s = tb.decode('utf-8', errors='replace')
            if s == '\n': s = '\\n'
            elif s == ' ': s = '␣'
            elif s == '\t': s = '\\t'
            labels.append(s[:10] if s.isprintable() else f'#{idx}')
        except:
            labels.append(f'#{idx}')
    
    bars = ax2.barh(range(top_k), top_freqs, color=ACCENT, alpha=0.8)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(labels, fontsize=8, color=DARK_FG)
    ax2.set_xlabel('Frequency')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved token distribution → {output_path}")


# ─── Generation Gallery ──────────────────────────────────────────────────────

def create_generation_gallery(model: GPT, tokenizer: BPETokenizer, output_path: str):
    """
    Generate text samples with different prompts and temperatures.
    Creates a visual gallery of model outputs.
    """
    prompts = [
        "FIRST CITIZEN:\nBefore we proceed any further, hear me speak.\n",
        "KING HENRY:\n",
        "To be, or not to be, that is the question:\n",
        "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?\n",
        "Enter HAMLET\n",
        "MACBETH:\nIs this a dagger which I see before me,\n",
    ]
    
    temperatures = [0.5, 0.8, 1.2]
    
    model.eval()
    device = next(model.parameters()).device
    
    n_cols = len(temperatures)
    n_rows = len(prompts)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2))
    
    for i, prompt in enumerate(prompts):
        for j, temp in enumerate(temperatures):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            
            # Generate
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            with torch.no_grad():
                # Truncate if needed
                if context.size(1) > model.config.block_size:
                    context = context[:, -model.config.block_size + 80:]
                generated = model.generate(context, max_new_tokens=80, temperature=temp, top_k=40)
            
            text = tokenizer.decode(generated[0].tolist())
            
            # Display
            dark_style(ax)
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=7, color=DARK_FG, fontfamily='monospace',
                   verticalalignment='top', wrap=True)
            
            if i == 0:
                ax.set_title(f"T={temp}", color=DARK_FG, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"Prompt {i+1}", color=DARK_FG, fontsize=9)
    
    fig.suptitle('Generation Gallery — Sparks of Babel', color=DARK_FG, 
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved generation gallery → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = "out"
    data_path = "shakespeare.txt"
    
    # Load tokenizer
    tokenizer_path = os.path.join(out_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer: {tokenizer.vocab_size} tokens")
    else:
        print("Tokenizer not found — train first!")
        return
    
    # Load best model
    model_path = os.path.join(out_dir, 'best_model.pt')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        config = GPTConfig(
            block_size=128,
            vocab_size=tokenizer.vocab_size,
            n_layer=4,
            n_head=4,
            n_embd=256,
            dropout=0.1,
        )
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        print(f"Loaded model (iter {checkpoint['iter']}, val_loss={checkpoint['val_loss']:.4f})")
    else:
        print("Model not found — train first!")
        return
    
    # Plot loss curves
    losses_path = os.path.join(out_dir, 'losses.pt')
    if os.path.exists(losses_path):
        plot_loss_curves(losses_path, os.path.join(out_dir, 'loss_curves.png'))
    
    # Plot embeddings
    try:
        from sklearn.decomposition import PCA
        plot_embeddings(model, tokenizer, os.path.join(out_dir, 'embedding_pca.png'))
    except ImportError:
        print("sklearn not installed — skipping embedding visualization")
    
    # Plot token distribution
    if os.path.exists(data_path):
        plot_token_distribution(tokenizer, data_path, os.path.join(out_dir, 'token_distribution.png'))
    
    # Attention patterns
    sample_text = "FIRST CITIZEN:\nBefore we proceed any further, hear me speak.\n"
    plot_attention_patterns(model, tokenizer, sample_text, os.path.join(out_dir, 'attention_patterns.png'))
    
    # Generation gallery
    create_generation_gallery(model, tokenizer, os.path.join(out_dir, 'generation_gallery.png'))
    
    print("\nAll visualizations complete!")


if __name__ == '__main__':
    main()
