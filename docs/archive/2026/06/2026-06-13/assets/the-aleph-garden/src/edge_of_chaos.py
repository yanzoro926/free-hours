#!/usr/bin/env python3
"""
Edge of Chaos · 混沌边缘
Maps the λ parameter space of 2D cellular automata rules.
Visualizes the transition from order → complexity → chaos.

The λ parameter (Langton, 1990) measures the fraction of neighborhood
configurations that lead to a live cell. The "edge of chaos" sits at
the critical point where complex, computation-like behavior emerges.

For 2D Life-like CA, we compute λ for the 18 possible neighborhood sums
(0-8 neighbors, excluding center, for both birth and survival conditions).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

OUTDIR = Path(__file__).parent.parent / 'output'
OUTDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#080c14',
    'axes.facecolor': '#101624',
    'axes.edgecolor': '#1e2848',
    'axes.labelcolor': '#b8c4d8',
    'text.color': '#b8c4d8',
    'xtick.color': '#5a6a8a',
    'ytick.color': '#5a6a8a',
    'grid.color': '#1a1a3e',
    'grid.alpha': 0.4,
})

# ─── Rule Definition ───
# Life-like CA: neighborhoods have 0-8 live neighbors.
# A rule is defined by two sets: B (birth) and S (survival).
# For each neighbor count 0-8:
#   - If the cell is dead: it's born if count ∈ B
#   - If the cell is alive: it survives if count ∈ S

# There are 2^9 * 2^9 = 262,144 possible Life-like rules.
# We can't explore all of them, but we can sample the space intelligently.

def rule_to_lambda(born_set, survive_set):
    """
    Compute λ for a Life-like rule.
    
    In Life-like CA, the outcome depends on:
    - 9 possible dead-cell scenarios (neighbor counts 0-8) → B determines which produce birth
    - 9 possible live-cell scenarios (neighbor counts 0-8) → S determines which survive
    
    λ = fraction of all 18 scenarios that lead to a live cell in the next generation.
    
    For the original Life (B3/S23):
    - Birth: only count=3 → 1/9
    - Survival: counts 2,3 → 2/9
    - λ = (1 + 2) / 18 = 3/18 = 1/6 ≈ 0.1667
    
    Wait, that doesn't seem right for Langton's λ. Let me reconsider.
    
    Langton's original λ is for 1D CA. For Life-like 2D CA, we need a different
    metric. Let's use:
    
    λ_B = |B| / 9  (fraction of neighbor counts that cause birth)
    λ_S = |S| / 9  (fraction of neighbor counts that preserve life)
    
    Combined metric: λ = (|B| + |S|) / 18, with a correction for the overlap
    of B and S (cells that would both be born and survive are just alive).
    """
    total_situations = 18  # 9 for dead center + 9 for live center
    live_outcomes = len(born_set) + len(survive_set)
    return live_outcomes / total_situations


def simulate_rule(born_set, survive_set, size=80, steps=200):
    """Simulate a Life-like rule from random initial state and classify behavior."""
    grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15])
    
    population_history = []
    grid_history = []  # Store periodic snapshots
    
    for step_idx in range(steps):
        # Count neighbors using convolution
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbors = np.zeros((size, size), dtype=int)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbors += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        
        new_grid = np.zeros_like(grid)
        for n in range(9):
            mask = (neighbors == n)
            if n in born_set:
                new_grid[mask & (grid == 0)] = 1
            if n in survive_set:
                new_grid[mask & (grid == 1)] = 1
        
        grid = new_grid
        pop = grid.sum()
        population_history.append(pop)
        
        if step_idx % 20 == 0:
            grid_history.append(grid.copy())
    
    # ─── Classification ───
    pops = np.array(population_history)
    
    if pops[-50:].sum() == 0:
        return 'Extinct', 0.0, grid_history
    
    # Check for periodicity in the population signal
    last_100 = pops[-100:]
    if len(last_100) >= 20:
        # Autocorrelation-like check
        diffs = np.abs(np.diff(last_100))
        if diffs[-20:].max() == 0:
            return 'Static', np.std(last_100), grid_history
        elif diffs[-20:].max() <= 2:
            return 'Oscillating', np.std(last_100), grid_history
    
    # Check for steady growth or decline
    trend = np.polyfit(range(len(pops[-50:])), pops[-50:], 1)[0]
    
    # Complexity measure: variance of population
    variance = np.std(pops)
    cv = variance / (pops.mean() + 1)  # coefficient of variation
    
    if cv < 0.1:
        return 'Stable', cv, grid_history
    elif cv < 0.3:
        return 'Periodic', cv, grid_history
    elif cv < 0.6:
        return 'Complex', cv, grid_history
    else:
        return 'Chaotic', cv, grid_history


def generate_rule_grid():
    """Sample the rule space and classify each rule."""
    # We explore B sets of sizes 0-9 and S sets of sizes 0-9
    # For each combination, we try several random subsets
    
    results = []
    
    # Systematic sampling: all combinations of |B| and |S|
    for b_size in range(10):
        for s_size in range(10):
            # Number of possible B sets of size b_size: C(9, b_size)
            # Too many to enumerate. Sample up to 3 per (b_size, s_size) pair.
            n_samples = min(3, max(1, 50 // ((b_size + 1) * (s_size + 1))))
            
            for _ in range(n_samples):
                all_n = list(range(9))
                b_set = set(np.random.choice(all_n, size=b_size, replace=False).tolist())
                s_set = set(np.random.choice(all_n, size=s_size, replace=False).tolist())
                
                lam = rule_to_lambda(b_set, s_set)
                behavior, cv, _ = simulate_rule(b_set, s_set, size=60, steps=150)
                
                results.append({
                    'b': sorted(b_set),
                    's': sorted(s_set),
                    'lambda': lam,
                    'behavior': behavior,
                    'cv': cv,
                    'b_size': b_size,
                    's_size': s_size,
                })
    
    return results


# ─── Famous Rules ───
FAMOUS_RULES = [
    ('Conway\'s Life', {3}, {2,3}),
    ('HighLife', {3,6}, {2,3}),
    ('Seeds', {2}, set()),
    ('Life w/o Death', {3}, {0,1,2,3,4,5,6,7,8}),
    ('Day & Night', {3,6,7,8}, {3,4,6,7,8}),
    ('Maze', {1}, {1}),
    ('Mazectric', {3}, {1,2,3,4,5}),
    ('Coral', {5,6,7,8}, {4,5,6,7,8}),
    ('3-4 Life', {3,4}, {3,4}),
    ('2x2', {2}, {2,3}),
    ('Replicator', {1,3,5,7}, {1,3,5,7}),
    ('Diamoeba', {3,5,6,7,8}, {5,6,7,8}),
    ('Move', {3,6,8}, {2,4,5}),
    ('Walled Cities', {4,5,6,7,8}, {2,3,4,5}),
]


def build_lambda_phase_diagram(results, famous_rules):
    """Build a publication-quality λ phase diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ─── Plot 1: λ vs Behavior (scatter) ───
    ax = axes[0]
    
    behavior_colors = {
        'Extinct': '#5a5a6a',
        'Static': '#5a6a8a',
        'Stable': '#78a8e8',
        'Periodic': '#78e8a8',
        'Oscillating': '#e8c878',
        'Complex': '#e878a8',
        'Chaotic': '#e87878',
    }
    
    behaviors_seen = set()
    for r in results:
        b = r['behavior']
        behaviors_seen.add(b)
        ax.scatter(r['lambda'], r['cv'], c=behavior_colors.get(b, '#888'),
                  alpha=0.4, s=10, edgecolors='none')
    
    # Plot famous rules
    for name, b_set, s_set in famous_rules:
        lam = rule_to_lambda(b_set, s_set)
        _, cv, gh = simulate_rule(b_set, s_set, size=80, steps=200)
        ax.scatter([lam], [cv], c='white', s=80, edgecolors='#78a8e8', linewidths=1.5, zorder=10)
        ax.annotate(name, (lam, cv), textcoords="offset points", xytext=(0, 10),
                   fontsize=7, color='#b8c4d8', ha='center')
    
    # Legend
    for name, color in behavior_colors.items():
        if name in behaviors_seen:
            ax.scatter([], [], c=color, label=name, s=20)
    ax.legend(loc='upper right', fontsize=7, facecolor='#101624', edgecolor='#1e2848',
             labelcolor='#b8c4d8', markerscale=1.5)
    
    ax.set_xlabel('λ (fraction of configurations → alive)')
    ax.set_ylabel('Coefficient of Variation (complexity)')
    ax.set_title('The Edge of Chaos · λ Phase Diagram', color='#78a8e8', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1.05)
    
    # ─── Plot 2: Behavior distribution by λ ───
    ax = axes[1]
    
    lambda_bins = np.linspace(0, 1, 21)
    behavior_order = ['Extinct', 'Static', 'Stable', 'Periodic', 'Oscillating', 'Complex', 'Chaotic']
    
    bin_data = {b: np.zeros(len(lambda_bins)-1) for b in behavior_order}
    for r in results:
        bin_idx = np.digitize(r['lambda'], lambda_bins) - 1
        if 0 <= bin_idx < len(lambda_bins) - 1:
            bin_data[r['behavior']][bin_idx] += 1
    
    bottom = np.zeros(len(lambda_bins) - 1)
    bar_colors = [behavior_colors[b] for b in behavior_order]
    for bi, b_name in enumerate(behavior_order):
        vals = bin_data[b_name]
        if vals.sum() > 0:
            ax.bar((lambda_bins[:-1] + lambda_bins[1:]) / 2, vals, width=0.045,
                  bottom=bottom, color=bar_colors[bi], alpha=0.85, label=b_name,
                  edgecolor='none')
            bottom += vals
    
    ax.axvline(x=0.167, color='#78e8a8', linestyle='--', linewidth=1, alpha=0.6, label="Life's λ")
    ax.axvline(x=0.273, color='#e878a8', linestyle=':', linewidth=1, alpha=0.6, label="Langton λ* (1D)")
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Number of Rules')
    ax.set_title('Behavior Distribution Across λ', color='#78a8e8', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6, facecolor='#101624', edgecolor='#1e2848',
             labelcolor='#b8c8d4', ncol=2)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, 1.05)
    
    plt.tight_layout()
    outpath = OUTDIR / 'lambda_phase_diagram.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#080c14')
    plt.close()
    print(f'✓ Phase diagram: {outpath}')
    return str(outpath)


def build_famous_rules_grid(famous_rules):
    """Generate evolution snapshots for famous rules."""
    n_rules = len(famous_rules)
    cols = 7
    rows = (n_rules + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes.flatten()
    
    for idx, (name, b_set, s_set) in enumerate(famous_rules):
        ax = axes[idx]
        _, _, grid_history = simulate_rule(b_set, s_set, size=50, steps=200)
        
        # Show the last grid state
        final_grid = grid_history[-1]
        alive_mask = final_grid > 0
        alive_count = alive_mask.sum()
        
        # Create a colored display: alive cells in accent color
        display = np.zeros((*final_grid.shape, 3))
        # Dead cells: dark bg
        display[:, :, 0] = 0.03
        display[:, :, 1] = 0.05
        display[:, :, 2] = 0.08
        # Alive cells
        if alive_count > 0:
            display[alive_mask, 0] = 0.47
            display[alive_mask, 1] = 0.66
            display[alive_mask, 2] = 0.91
        
        ax.imshow(display, interpolation='nearest')
        
        lam = rule_to_lambda(b_set, s_set)
        _, cv, _ = simulate_rule(b_set, s_set, size=60, steps=150)
        behavior, _, _ = simulate_rule(b_set, s_set, size=60, steps=150)
        
        ax.set_title(f'{name}\nλ={lam:.2f} {behavior[:6]}', fontsize=6,
                    color='#b8c4d8', fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(n_rules, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Cellular Automata Rule Atlas', color='#78a8e8', fontsize=14,
                fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = OUTDIR / 'rule_atlas.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#080c14')
    plt.close()
    print(f'✓ Rule atlas: {outpath}')
    return str(outpath)


def build_class_distribution_chart(results):
    """Build a breakdown of behavior classifications with λ correlation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # ─── Pie chart of behaviors ───
    behavior_counts = {}
    for r in results:
        b = r['behavior']
        behavior_counts[b] = behavior_counts.get(b, 0) + 1
    
    sorted_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [b for b, _ in sorted_behaviors]
    values = [v for _, v in sorted_behaviors]
    colors = ['#78a8e8', '#78e8a8', '#e8c878', '#e878a8', '#e87878', '#5a6a8a', '#5a5a6a']
    
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors[:len(labels)],
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'color': '#b8c4d8', 'fontsize': 8})
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(7)
    ax.set_title('Rule Behavior Distribution', color='#78a8e8', fontsize=12, fontweight='bold')
    
    # ─── λ vs |B| scatter ───
    ax = axes[1]
    for r in results:
        ax.scatter(r['b_size'], r['lambda'], c='#78a8e8', alpha=0.15, s=8, edgecolors='none')
    
    # Add famous rules
    for name, b_set, s_set in FAMOUS_RULES:
        lam = rule_to_lambda(b_set, s_set)
        ax.scatter([len(b_set)], [lam], c='white', s=60, edgecolors='#e878a8', linewidths=1.5, zorder=10)
        ax.annotate(name, (len(b_set), lam), textcoords="offset points", xytext=(5, 5),
                   fontsize=6, color='#b8c4d8')
    
    ax.set_xlabel('|B| (birth conditions)')
    ax.set_ylabel('λ')
    ax.set_title('λ vs Birth Set Size', color='#78a8e8', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # ─── Complexity at the edge ───
    ax = axes[2]
    
    # For each λ bin, compute the fraction of complex rules
    lambda_bins = np.linspace(0, 1, 51)
    complex_fractions = []
    total_in_bin = []
    
    for i in range(len(lambda_bins) - 1):
        lo, hi = lambda_bins[i], lambda_bins[i+1]
        bin_rules = [r for r in results if lo <= r['lambda'] < hi]
        total = len(bin_rules)
        complex_count = len([r for r in bin_rules if r['behavior'] in ('Complex', 'Chaotic')])
        complex_fractions.append(complex_count / total if total > 0 else 0)
        total_in_bin.append(total)
    
    midpoints = (lambda_bins[:-1] + lambda_bins[1:]) / 2
    ax.bar(midpoints, complex_fractions, width=0.018, color='#e878a8', alpha=0.8, edgecolor='none')
    ax.axvline(x=0.167, color='#78e8a8', linestyle='--', linewidth=1, alpha=0.6, label="Life's λ")
    
    # Mark the peak complexity region
    max_idx = np.argmax(complex_fractions)
    peak_lam = midpoints[max_idx]
    ax.axvline(x=peak_lam, color='#e8c878', linestyle=':', linewidth=1.5, alpha=0.7,
              label=f'Peak complexity λ≈{peak_lam:.2f}')
    
    ax.set_xlabel('λ')
    ax.set_ylabel('Fraction Complex/Chaotic')
    ax.set_title('Complexity vs λ', color='#78a8e8', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#101624', edgecolor='#1e2848', labelcolor='#b8c4d8')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    outpath = OUTDIR / 'class_distribution.png'
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='#080c14')
    plt.close()
    print(f'✓ Class distribution: {outpath}')
    return str(outpath)


def main():
    print("═" * 60)
    print("  Edge of Chaos · 混沌边缘")
    print("  Cell Automata Rule Space Explorer")
    print("═" * 60)
    
    # Generate results
    print("\n▶ Sampling rule space...")
    results = generate_rule_grid()
    print(f"  Sampled {len(results)} rules")
    
    # Save raw data
    data_path = OUTDIR / 'rule_space_data.json'
    with open(data_path, 'w') as f:
        # Convert sets to lists for JSON
        json_results = []
        for r in results:
            jr = dict(r)
            jr['b'] = sorted(r['b'])
            jr['s'] = sorted(r['s'])
            json_results.append(jr)
        json.dump(json_results, f, indent=2)
    print(f"✓ Data saved: {data_path}")
    
    # Behavior stats
    behavior_counts = {}
    for r in results:
        behavior_counts[r['behavior']] = behavior_counts.get(r['behavior'], 0) + 1
    print("\n  Behavior distribution:")
    for b, c in sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * (c * 40 // len(results))
        print(f"    {b:<12s} {c:>5d}  {bar}")
    
    # Famous rules summary
    print("\n▶ Famous Rules:")
    for name, b_set, s_set in FAMOUS_RULES:
        lam = rule_to_lambda(b_set, s_set)
        behavior, cv, _ = simulate_rule(b_set, s_set, size=60, steps=150)
        b_str = ''.join(str(x) for x in sorted(b_set))
        s_str = ''.join(str(x) for x in sorted(s_set))
        print(f"    {name:<18s} B{b_str}/S{s_str:<10s} λ={lam:.3f}  {behavior}")
    
    # Build charts
    print("\n▶ Building visualizations...")
    build_lambda_phase_diagram(results, FAMOUS_RULES)
    build_famous_rules_grid(FAMOUS_RULES)
    build_class_distribution_chart(results)
    
    print("\n✅ Edge of Chaos analysis complete!")
    print(f"   Output directory: {OUTDIR}")


if __name__ == '__main__':
    main()
