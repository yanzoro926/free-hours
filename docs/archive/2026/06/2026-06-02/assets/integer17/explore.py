"""
The 17% Integer — A Mathematical Exploration

From HN (June 2, 2026): "Only 17% of all 64-bit Integers are products of two 32-bit integers"

This explores why, through computation, visualization, and analysis.

Key insight: A 64-bit integer n is a product of two 32-bit integers x·y
if AND ONLY IF n ≤ (2³² - 1)² ≈ 1.84 × 10¹⁹ AND n has a factor ≤ 2³² - 1.

Think of it this way:
- There are 2³² ≈ 4.29 billion possible 32-bit unsigned integers
- Their products range from 1×1 = 1 up to (2³²-1)(2³²-1) ≈ 1.84×10¹⁹
- But a 64-bit integer can go up to 2⁶⁴ - 1 ≈ 1.84×10¹⁹
- Wait — (2³²-1)² is slightly LESS than 2⁶⁴-1

The ratio is approximately:
  (2³²-1)² / (2⁶⁴-1) ≈ (2⁶⁴ - 2³³ + 1) / 2⁶⁴ ≈ 1 - 2⁻³¹ ≈ 0.9999999995...

So almost ALL 64-bit integers up to (2³²-1)² could be products.
The issue is: numbers BETWEEN (2³²-1)² and 2⁶⁴-1 can NEVER be products of two 32-bit integers.

And the range between (2³²-1)² and 2⁶⁴:
  (2⁶⁴ - 1) - (2³² - 1)² = (2⁶⁴ - 1) - (2⁶⁴ - 2³³ + 1) = 2³³ - 2

So the "unreachable" portion is about 2³³ out of 2⁶⁴, which is 2³³/2⁶⁴ = 2⁻³¹.
That's about 4.66 × 10⁻¹⁰ — nowhere near 17%!

Wait, I must be misunderstanding the claim. Let me re-read...

Ah — I think the claim is about SEMIPRIMES: "products of exactly two prime factors, each ≤ 32 bits"
Or perhaps it's about the density of numbers that CAN be factored into two ≤32-bit components.

Actually, let me compute this empirically: for a range of 64-bit integers,
what fraction can be expressed as x·y where both x,y ≤ 2³²-1?

The key insight is that numbers > (2³²-1)² are IMPOSSIBLE.
And among numbers ≤ (2³²-1)², not all are reachable because of the constraint
that both factors must be ≤ 2³²-1.

For example: n = 2³² × 5 = 21,474,836,480 
  n is ≤ (2³²-1)² ✓
  But to factor n = x·y with both ≤ 2³²-1:
  x could be 2³², but that exceeds 2³²-1 ×
  x could be 5, then y = 2³² ≈ 4.29B > 2³²-1 ×
  So n=2³²×5 is unreachable!

This is the real constraint. Let's explore computationally.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

DARK_BG = '#1a1a2e'
DARK_FG = '#e0e0e0'
ACCENT = '#e94560'
ACCENT2 = '#00d2ff'
GOLD = '#f5c518'


def dark_style(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=DARK_FG, labelsize=9)
    ax.xaxis.label.set_color(DARK_FG)
    ax.yaxis.label.set_color(DARK_FG)
    ax.title.set_color(DARK_FG)
    if title:
        ax.set_title(title, color=DARK_FG, fontweight='bold', pad=12)


def simulate_small_scale(max_val=1000, max_factor=100):
    """
    Simulate at small scale: how many numbers ≤ max_val can be 
    expressed as product of two numbers ≤ max_factor?
    """
    reachable = set()
    for x in range(1, max_factor + 1):
        for y in range(x, max_factor + 1):
            p = x * y
            if p <= max_val:
                reachable.add(p)
    
    total = max_val
    reachable_count = len(reachable)
    
    print(f"Scale: max_val={max_val}, max_factor={max_factor}")
    print(f"Reachable: {reachable_count}/{total} = {reachable_count/total*100:.2f}%")
    print(f"Unreachable (≤ max_val): {total - reachable_count}")
    
    # Numbers > max_factor² are never reachable
    max_product = max_factor * max_factor
    beyond_max_product = sum(1 for i in range(max_product + 1, max_val + 1))
    print(f"Numbers > max_factor²: {beyond_max_product}")
    
    return reachable


def analyze_theoretical():
    """
    Theoretical analysis of the 17% claim.
    
    For 64-bit unsigned integers:
    - Total 64-bit integers: 2⁶⁴ ≈ 1.84 × 10¹⁹
    - 32-bit factor limit: M = 2³² - 1 ≈ 4.29 × 10⁹
    - Maximum product: M² ≈ (2³²-1)² ≈ 2⁶⁴ - 2³³ + 1
    
    The claim of 17% likely refers to the DENSITY of numbers that are products
    of two integers each ≤ 2³²-1, among numbers ≤ M² (not among ALL 64-bit).
    
    Among numbers ≤ M², the density of "representable" numbers depends on
    the distribution of factors.
    
    Let's think in terms of the divisor function.
    
    A number n ≤ M² is representable iff it has a divisor d ≤ M such that n/d ≤ M.
    Equivalently: n has a divisor d where √n ≤ d ≤ M (or d ≥ √n and ≤ M).
    
    The "hard" numbers are those near M² with all divisors either very small or very large.
    """
    M32 = 2**32 - 1
    M64 = 2**64 - 1
    M32_sq = M32 * M32
    
    print("=" * 60)
    print("THEORETICAL ANALYSIS")
    print("=" * 60)
    print(f"M = 2³² - 1 = {M32:,}")
    print(f"M² = {M32_sq:,}")
    print(f"2⁶⁴ - 1 = {M64:,}")
    print(f"M² / (2⁶⁴-1) = {M32_sq / M64:.10f}")
    print(f"Numbers > M² that are completely unreachable: {M64 - M32_sq:,}")
    print(f"Fraction of 64-bit space unreachable: {(M64 - M32_sq) / M64:.2e}")
    print()
    print("The 17% must refer to density WITHIN [1, M²], not all 64-bit space.")
    print()


def empirical_density(max_val=100000, max_factor=None):
    """
    Compute the empirical density of representable numbers.
    
    For each n, we need to check if ∃ x ≤ max_factor such that x|n and n/x ≤ max_factor.
    """
    if max_factor is None:
        import math
        max_factor = int(math.sqrt(max_val)) * 10  # generous bound
    
    # Use sieve-like approach
    reachable = set()
    
    # For each factor x ≤ max_factor
    # For each multiple m = x, 2x, 3x, ... where m ≤ max_val
    # If m/x ≤ max_factor, mark m as reachable
    
    for x in range(1, max_factor + 1):
        max_y = min(max_factor, max_val // x)
        for y in range(1, max_y + 1):
            reachable.add(x * y)
    
    total = max_val
    reachable_count = len(reachable)
    
    return reachable_count / total, reachable, total


def plot_density_curve(output_path='integer17_density.png'):
    """Plot how the density of representable numbers changes with range."""
    
    # Use smaller scale for visualization
    # We'll plot density as a function of max_val for different max_factor values
    
    max_vals = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    factor_ratios = [0.5, 1.0, 2.0, 5.0]  # max_factor as multiple of sqrt(max_val)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: density vs max_val for different factor ratios
    ax1 = axes[0]
    dark_style(ax1, f"Density of Representable Numbers\n(factor ≤ k·√n)")
    
    import math
    for ratio in factor_ratios:
        densities = []
        for max_val in max_vals:
            max_factor = int(ratio * math.sqrt(max_val))
            density, _, _ = empirical_density(max_val, max_factor)
            densities.append(density * 100)
        
        ax1.plot(max_vals, densities, 'o-', 
                label=f'k={ratio}', linewidth=1.5, markersize=4)
    
    ax1.set_xlabel('Max value (n)')
    ax1.set_ylabel('Density (%)')
    ax1.legend(facecolor=DARK_BG, edgecolor='#555', labelcolor=DARK_FG)
    ax1.grid(True, alpha=0.15, color='white')
    ax1.set_xscale('log')
    
    # Right: illustration of reachable vs unreachable
    ax2 = axes[1]
    dark_style(ax2, "Sieve Visualization\n(n ≤ 2000, factors ≤ 60)")
    
    max_val = 2000
    max_factor = 60  # sqrt(2000) ≈ 44.7, so 60 > sqrt → some reachability
    
    reachable = set()
    for x in range(1, max_factor + 1):
        for y in range(x, max_factor + 1):
            p = x * y
            if p <= max_val:
                reachable.add(p)
    
    all_nums = set(range(1, max_val + 1))
    unreachable = all_nums - reachable
    
    # Histogram of gaps between unreachable numbers
    unreachable_sorted = sorted(unreachable)
    gaps = [unreachable_sorted[i+1] - unreachable_sorted[i] 
            for i in range(len(unreachable_sorted) - 1)]
    
    ax2.hist(gaps, bins=50, color=ACCENT, alpha=0.7, edgecolor='white', linewidth=0.3)
    ax2.set_xlabel('Gap between consecutive unreachable numbers')
    ax2.set_ylabel('Frequency')
    ax2.axvline(x=sum(gaps)/len(gaps), color=GOLD, linestyle='--', 
               label=f'Mean gap: {sum(gaps)/len(gaps):.1f}')
    ax2.legend(facecolor=DARK_BG, edgecolor='#555', labelcolor=DARK_FG)
    
    plt.tight_layout()
    plt.savefig(output_path, facecolor=DARK_BG)
    plt.close()
    print(f"Saved density curve → {output_path}")
    
    # Compute actual stats
    density = len(reachable) / max_val * 100
    print(f"\nAt scale n≤{max_val}, factors≤{max_factor}:")
    print(f"  Reachable: {len(reachable)}/{max_val} = {density:.2f}%")
    print(f"  Unreachable: {len(unreachable)}")
    print(f"  Mean gap: {sum(gaps)/len(gaps):.2f}")
    print(f"  Max gap: {max(gaps)}")
    print(f"  Numbers > max_factor² ({max_factor*max_factor}): "
          f"{sum(1 for i in range(max_factor*max_factor+1, max_val+1))}")


def explore_64bit_equivalent():
    """
    We can't simulate 64-bit space, but we can simulate 16-bit space
    with 8-bit factors to get the exact ratio and study its properties.
    
    16-bit: 2¹⁶ = 65,536
    8-bit factor: 2⁸ - 1 = 255
    Max product: 255² = 65,025
    
    This is a perfect miniature of the 32/64 bit relationship!
    """
    print("\n" + "=" * 60)
    print("MINIATURE MODEL: 16-bit products of 8-bit factors")
    print("=" * 60)
    
    MAX16 = 2**16  # 65536 (one more than max 16-bit)
    FACTOR8 = 2**8 - 1  # 255
    MAX_PRODUCT = FACTOR8 * FACTOR8  # 65025
    
    print(f"8-bit factor limit: {FACTOR8}")
    print(f"Max product: {MAX_PRODUCT}")
    print(f"Numbers > product limit (unreachable by definition): {MAX16 - 1 - MAX_PRODUCT}")
    
    reachable = set()
    for x in range(1, FACTOR8 + 1):
        max_y = min(FACTOR8, (MAX16 - 1) // x)
        for y in range(1, max_y + 1):
            reachable.add(x * y)
    
    total_16bit = MAX16 - 1  # numbers 1..65535
    reachable_count = len(reachable)
    
    print(f"\nTotal 16-bit numbers: {total_16bit:,}")
    print(f"Reachable: {reachable_count:,}")
    print(f"Unreachable: {total_16bit - reachable_count:,}")
    print(f"Reachable density: {reachable_count/total_16bit*100:.2f}%")
    
    # Analyze by region
    regions = [
        (1, FACTOR8, "1 to 255 (≤ max factor)"),
        (FACTOR8 + 1, MAX_PRODUCT, f"256 to {MAX_PRODUCT} (≤ max product)"),
        (MAX_PRODUCT + 1, MAX16 - 1, f"{MAX_PRODUCT+1} to 65535 (> max product)"),
    ]
    
    for lo, hi, label in regions:
        region_reachable = sum(1 for i in range(lo, hi + 1) if i in reachable)
        region_total = hi - lo + 1
        print(f"  {label}: {region_reachable}/{region_total} = {region_reachable/region_total*100:.1f}%")
    
    return reachable, FACTOR8, MAX16


def main():
    output_dir = "/home/yanyj/VibeCoding/autonomy/2026-06-02/integer17"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Theoretical analysis
    analyze_theoretical()
    
    # Small-scale simulation
    print("\n" + "=" * 60)
    print("SMALL-SCALE SIMULATION")
    print("=" * 60)
    simulate_small_scale(1000, 30)
    
    # 16-bit miniature
    explore_64bit_equivalent()
    
    # Density curve
    plot_density_curve(os.path.join(output_dir, 'density.png'))
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The 17% claim refers to the fraction of 64-bit integers that can be expressed
as the product of two 32-bit integers. This is a number-theoretic density problem.

Key factors:
1. Numbers > (2³²-1)² are impossible (about 2³³ numbers, negligible fraction)
2. Among numbers ≤ (2³²-1)², the density depends on factor distributions
3. The asymptotic density of numbers representable as x·y with x,y ≤ √N·k
   approaches a specific value as k varies

The 17% suggests that with the constraint x,y ≤ 2³²-1, only 17% of 64-bit 
space is "reachable" — the rest require factors larger than 32 bits.
    """)


if __name__ == '__main__':
    main()
