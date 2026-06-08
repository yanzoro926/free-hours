"""
Speed Frontier — Publication-Quality Charts
============================================
Generates dark-themed SVG visualizations for the speed frontier analysis.
All charts go to charts/ directory.
"""

import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
from pathlib import Path

OUT = Path(__file__).parent / 'charts'
OUT.mkdir(exist_ok=True)

# ============================================================
# STYLE
# ============================================================

BG = '#06060e'
SURFACE = '#0f0f1a'
BORDER = '#1a1a3a'
TEXT = '#b8c5d6'
DIM = '#556688'
ACCENT = '#00d4aa'
ACCENT2 = '#7c6aff'
WARN = '#ff6b6b'
GOLD = '#ffd700'
SPEED_10 = '#ff6b6b'
SPEED_60 = '#ffa94d'
SPEED_200 = '#ffd43b'
SPEED_1000 = '#69db7c'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': BORDER,
    'axes.labelcolor': TEXT,
    'text.color': TEXT,
    'xtick.color': DIM,
    'ytick.color': DIM,
    'grid.color': BORDER,
    'grid.alpha': 0.4,
    'font.family': 'monospace',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 10,
    'savefig.facecolor': BG,
    'savefig.edgecolor': 'none',
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

def save(fig, name):
    fig.savefig(OUT / name, format='svg')
    plt.close(fig)
    print(f"  ✓ charts/{name}")

# ============================================================
# CHART 1: Reasoning Paths vs Speed
# ============================================================

def chart_paths_vs_speed():
    speeds = np.array([5, 10, 30, 60, 100, 200, 500, 1000, 1200])
    paths = np.floor(2.0 / (200 / speeds)).astype(int)
    paths = np.maximum(paths, 1)
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    colors = [SPEED_10 if s < 60 else SPEED_60 if s < 200 else SPEED_200 if s < 500 else SPEED_1000 for s in speeds]
    
    bars = ax.bar(range(len(speeds)), paths, color=colors, alpha=0.8, edgecolor=BG, linewidth=0.5)
    
    # Labels
    for i, (s, p) in enumerate(zip(speeds, paths)):
        ax.text(i, p + 0.3, str(p), ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[i])
    
    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([f'{s}' for s in speeds])
    ax.set_xlabel('Speed (tokens/second)')
    ax.set_ylabel('Reasoning Paths in 2s Budget')
    ax.set_title('Reasoning Paths vs. Inference Speed', fontweight='bold', color=ACCENT, pad=15)
    
    # Annotation
    ax.annotate('Paradigm Shift:\n10× more paths',
                xy=(7, 10), xytext=(5, 9),
                arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5),
                fontsize=8, color=GOLD, fontweight='bold')
    
    ax.set_ylim(0, 13)
    ax.grid(axis='y', alpha=0.3)
    
    save(fig, '01_paths_vs_speed.svg')

# ============================================================
# CHART 2: Best-of-N Confidence Curve
# ============================================================

def chart_best_of_n():
    n_values = np.arange(1, 21)
    confidence = (1 - 0.15**n_values) * 100
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Color gradient based on confidence
    colors = []
    for c in confidence:
        if c < 90:
            colors.append(SPEED_10)
        elif c < 99:
            colors.append(SPEED_60)
        elif c < 99.9:
            colors.append(SPEED_200)
        else:
            colors.append(SPEED_1000)
    
    ax.plot(n_values, confidence, '-', color=ACCENT, linewidth=2, zorder=2)
    ax.scatter(n_values, confidence, c=colors, s=40, zorder=3, edgecolors='white', linewidth=0.5)
    
    # Threshold line at 99%
    ax.axhline(y=99, color=GOLD, linestyle='--', linewidth=1, alpha=0.6)
    ax.text(15, 99.3, '99% threshold', fontsize=7, color=GOLD, va='bottom')
    
    # Labels for key points
    for n, c in [(1, confidence[0]), (3, confidence[2]), (5, confidence[4]), (10, confidence[9])]:
        ax.annotate(f'N={n}: {c:.1f}%', (n, c),
                   textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=7, color=ACCENT)
    
    ax.set_xlabel('Number of Reasoning Paths (N)')
    ax.set_ylabel('P(at least one correct) %')
    ax.set_title('Best-of-N: More Paths = Higher Confidence', fontweight='bold', color=ACCENT, pad=15)
    ax.set_xlim(0.5, 20.5)
    ax.grid(alpha=0.3)
    
    save(fig, '02_best_of_n.svg')

# ============================================================
# CHART 3: Time Saved vs Scenario
# ============================================================

def chart_time_saved():
    scenarios = ['Chat\n10K tok', 'Code Review\n50K tok', 'Codebase\n200K tok', 'Agent\n500K tok', 'Heavy Day\n1M tok']
    base_time_min = [142.9/60, 714.3/60, 2857.1/60, 7142.9/60, 14285.7/60]
    ultra_time_min = [10/60, 50/60, 200/60, 500/60, 1000/60]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    bars1 = ax.bar(x - width/2, base_time_min, width, label='Base Model (70 tok/s)', color=SPEED_60, alpha=0.7)
    bars2 = ax.bar(x + width/2, ultra_time_min, width, label='UltraSpeed (1000 tok/s)', color=SPEED_1000, alpha=0.8)
    
    # Value labels on bars
    for bar, val in zip(bars1, base_time_min):
        label = f'{val:.0f}m' if val >= 1 else f'{val*60:.0f}s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, label,
               ha='center', va='bottom', fontsize=8, color=SPEED_60)
    for bar, val in zip(bars2, ultra_time_min):
        label = f'{val:.0f}m' if val >= 1 else f'{val*60:.0f}s'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, label,
               ha='center', va='bottom', fontsize=8, color=SPEED_1000)
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel('Wait Time (minutes)')
    ax.set_title('Time Saved: Base vs. UltraSpeed', fontweight='bold', color=ACCENT, pad=15)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    save(fig, '03_time_saved.svg')

# ============================================================
# CHART 4: Speed-Quality Landscape
# ============================================================

def chart_landscape():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = [
        ('Human Typing', 5, 95, SPEED_10, 'o'),
        ('GPT-3.5', 40, 55, SPEED_10, 'o'),
        ('Claude Opus 4', 60, 92, SPEED_60, 's'),
        ('DeepSeek-V3', 70, 83, SPEED_60, 'D'),
        ('GPT-4o', 80, 85, SPEED_60, 's'),
        ('Groq LLaMA3', 400, 72, SPEED_200, '^'),
        ('Cerebras CS-3', 550, 70, SPEED_200, '^'),
        ('MiMo UltraSpeed', 1050, 88, SPEED_1000, '*'),
    ]
    
    for name, speed, quality, color, marker in models:
        size = 120 if speed >= 500 else 80
        ax.scatter(speed, quality, c=color, s=size, marker=marker, edgecolors='white',
                  linewidth=0.8, zorder=3, alpha=0.9)
        
        offset = (12, 5) if speed < 200 else (-12, 5)
        ax.annotate(name, (speed, quality), textcoords="offset points",
                   xytext=offset, ha='left' if speed < 200 else 'right',
                   fontsize=8, color=TEXT, fontweight='bold')
    
    # Region shading
    ax.axvspan(0, 60, alpha=0.03, color=SPEED_10)
    ax.axvspan(60, 200, alpha=0.03, color=SPEED_60)
    ax.axvspan(200, 500, alpha=0.03, color=SPEED_200)
    ax.axvspan(500, 1200, alpha=0.06, color=SPEED_1000)
    
    # Region labels
    ax.text(30, 98, 'Slow', fontsize=7, color=SPEED_10, ha='center', alpha=0.6)
    ax.text(130, 98, 'GPT-4 Era', fontsize=7, color=SPEED_60, ha='center', alpha=0.6)
    ax.text(350, 98, 'Fast', fontsize=7, color=SPEED_200, ha='center', alpha=0.6)
    ax.text(850, 98, 'UltraSpeed', fontsize=7, color=SPEED_1000, ha='center', alpha=0.8)
    
    # Paradigm shift arrow
    ax.annotate('Paradigm Shift →', xy=(800, 88), xytext=(550, 78),
               arrowprops=dict(arrowstyle='->', color=GOLD, lw=1.5),
               fontsize=9, color=GOLD, fontweight='bold')
    
    ax.set_xscale('log')
    ax.set_xlabel('Speed (tokens/second, log scale)')
    ax.set_ylabel('Quality (subjective, 0-100)')
    ax.set_title('The Speed-Quality Frontier', fontweight='bold', color=ACCENT, pad=15)
    ax.set_xlim(3, 1500)
    ax.set_ylim(45, 100)
    ax.grid(alpha=0.3)
    
    save(fig, '04_landscape.svg')

# ============================================================
# CHART 5: The Paradigm Shift Visual
# ============================================================

def chart_paradigm():
    """Visual showing what crosses at each threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    thresholds = [
        (5, 'Human Typing', 'Basic text completion', SPEED_10),
        (30, 'Conversational', 'Real-time chat, Q&A', SPEED_10),
        (60, 'GPT-4 Class', 'Code gen, reasoning', SPEED_60),
        (200, 'Fast Frontier', 'Invisible reasoning', SPEED_200),
        (500, 'Agent Speed', 'Real-time coding agents', SPEED_1000),
        (1000, 'UltraSpeed', 'Best-of-N, paradigm shift', SPEED_1000),
    ]
    
    y_positions = [5, 4, 3, 2, 1, 0]
    
    for (speed, name, desc, color), y in zip(thresholds, y_positions):
        ax.barh(y, speed, height=0.6, color=color, alpha=0.7, edgecolor=BG)
        ax.text(speed + 20, y, f'{name}: {speed} tok/s', va='center', fontsize=9,
               fontweight='bold', color=color)
        ax.text(speed + 20, y - 0.25, desc, va='center', fontsize=7, color=DIM)
    
    # Vertical lines for key transitions
    for x, label in [(200, 'Best-of-N begins'), (1000, 'Paradigm Shift')]:
        ax.axvline(x=x, color=GOLD, linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(x + 10, 5.5, label, fontsize=7, color=GOLD, rotation=0, va='bottom')
    
    ax.set_xscale('log')
    ax.set_xlabel('Tokens per Second (log scale)')
    ax.set_yticks([])
    ax.set_title('The Speed Frontier: What Crosses at Each Threshold', fontweight='bold', color=ACCENT, pad=15)
    ax.set_xlim(3, 2000)
    ax.grid(axis='x', alpha=0.3)
    
    save(fig, '05_paradigm_shift.svg')


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Generating publication-quality charts...")
    chart_paths_vs_speed()
    chart_best_of_n()
    chart_time_saved()
    chart_landscape()
    chart_paradigm()
    print(f"\n✓ All charts saved to {OUT}/")
