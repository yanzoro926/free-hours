"""
The Speed Frontier — Analytical Companion
==========================================
Computes and visualizes the token-speed paradigm shift using real metrics.

Key analyses:
1. Time-to-reasoning: How many reasoning paths fit in 2 seconds?
2. Best-of-N confidence: How does N affect confidence?
3. Cost-speed tradeoff: MiMo's 3× cost for 10× speed
4. Frontier crossing: What becomes possible at each threshold
"""

import math
import json
from pathlib import Path

OUT = Path(__file__).parent

# ============================================================
# 1. TIME-TO-REASONING ANALYSIS
# ============================================================

def reasoning_paths_possible(speed_tok_per_sec, path_tokens=200, time_budget_sec=2.0):
    """How many independent reasoning paths can run in the time budget?"""
    time_per_path = path_tokens / speed_tok_per_sec
    paths = max(1, int(time_budget_sec / time_per_path))
    return {
        'speed': speed_tok_per_sec,
        'time_per_path_s': round(time_per_path, 3),
        'paths_in_budget': paths,
        'total_tokens_generated': paths * path_tokens,
    }

def best_of_n_confidence(n_paths, error_rate=0.15):
    """Probability at least one path is correct, given independent errors.
    
    P(at least one correct) = 1 - P(all wrong) = 1 - error_rate^n
    """
    prob_correct = 1 - error_rate ** n_paths
    # Show high precision for near-100% cases
    if prob_correct > 0.999:
        return round(prob_correct * 100, 4)
    return round(prob_correct * 100, 1)

# ============================================================
# 2. COST-SPEED TRADEOFF
# ============================================================

def cost_analysis():
    """Analyze MiMo's 3× price for 10× speed proposition."""
    base_cost_per_1M = 2.0  # hypothetical $/1M tokens for MiMo-V2.5-Pro
    ultra_cost_per_1M = base_cost_per_1M * 3  # 3× price
    
    # How much does it cost to generate 1M tokens?
    # At 1000 tok/s, 1M tokens takes 1000 seconds = 16.7 minutes
    
    scenarios = [
        ('Chat (10K tokens)', 10_000),
        ('Code review (50K tokens)', 50_000),
        ('Full codebase analysis (200K tokens)', 200_000),
        ('Agent session (500K tokens)', 500_000),
        ('Heavy reasoning day (1M tokens)', 1_000_000),
    ]
    
    results = []
    for name, tokens in scenarios:
        base_time = tokens / 70  # base speed ~70 tok/s
        ultra_time = tokens / 1000
        base_cost = (tokens / 1_000_000) * base_cost_per_1M
        ultra_cost = (tokens / 1_000_000) * ultra_cost_per_1M
        time_saved = base_time - ultra_time
        
        results.append({
            'scenario': name,
            'tokens': tokens,
            'base_time_s': round(base_time, 1),
            'ultra_time_s': round(ultra_time, 1),
            'time_saved_s': round(time_saved, 1),
            'speedup': round(base_time / ultra_time, 1),
            'base_cost': round(base_cost, 4),
            'ultra_cost': round(ultra_cost, 4),
            'cost_multiplier': round(ultra_cost / base_cost, 1),
        })
    
    return results

# ============================================================
# 3. FRONTIER THRESHOLDS
# ============================================================

def frontier_analysis():
    """What capabilities cross at each speed threshold?"""
    
    user_read_speed = 300  # words per minute ~ 250-300
    user_tok_per_sec = user_read_speed / 60 * 1.33  # ~6.6 tok/s reading speed
    
    thresholds = [
        {
            'speed': 5,
            'name': 'Human Typing Speed',
            'milestones': [
                'LLM output matches human typing speed',
                'User can read faster than the model generates',
                'Not suitable for real-time applications',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(5),
        },
        {
            'speed': 30,
            'name': 'Conversational Threshold',
            'milestones': [
                'Text appears fast enough for conversational flow',
                'Sub-second responses for short queries (~15 tok)',
                'Minimum viable speed for chatbots',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(30),
        },
        {
            'speed': 60,
            'name': 'GPT-4 Class',
            'milestones': [
                'Most current frontier models operate here',
                '200-token reasoning chain in ~3.3 seconds',
                'Tolerable for most applications, but visible delay',
                'Code generation: ~15 lines of Python in ~3 seconds',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(60),
        },
        {
            'speed': 200,
            'name': 'Fast Frontier',
            'milestones': [
                'Reasoning becomes nearly invisible to the user',
                '200-token CoT in 1 second — instantaneous feel',
                'Enables streaming agents with tool use',
                'Groq/Cerebras custom hardware territory',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(200),
        },
        {
            'speed': 500,
            'name': 'Agent Speed',
            'milestones': [
                'Coding agents generate functions before you read the prompt',
                'Tool-call latency becomes the bottleneck, not generation',
                'Multiple parallel agent actions feasible',
                'Real-time translation and transcription',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(500),
        },
        {
            'speed': 1000,
            'name': 'UltraSpeed (MiMo)',
            'milestones': [
                'Best-of-N reasoning: 10 paths in 2 seconds',
                'Real-time quantitative trading signal generation',
                'Surgical AI: sub-second medical image analysis',
                'FP4 quantization + speculative decoding on commodity GPUs',
                'Speed transmutes into intelligence — paradigm shift',
            ],
            'reasoning_paths_2s': reasoning_paths_possible(1000),
        },
    ]
    
    # Add Best-of-N confidence for each
    for t in thresholds:
        n = t['reasoning_paths_2s']['paths_in_budget']
        t['best_of_n_confidence_pct'] = best_of_n_confidence(n)
    
    return thresholds

# ============================================================
# 4. MAIN REPORT GENERATION
# ============================================================

def generate_report():
    """Generate a comprehensive report."""
    
    frontiers = frontier_analysis()
    costs = cost_analysis()
    
    report = []
    report.append("# The Speed Frontier — Analytical Report")
    report.append("")
    report.append("## 1. Reasoning Paths vs. Speed")
    report.append("")
    report.append("| Speed (tok/s) | Time per 200-tok Path | Paths in 2s Budget | Best-of-N Confidence |")
    report.append("|--------------|----------------------|-------------------|---------------------|")
    
    speeds = [5, 10, 30, 60, 100, 200, 500, 1000, 1200]
    for s in speeds:
        r = reasoning_paths_possible(s)
        conf = best_of_n_confidence(r['paths_in_budget'])
        report.append(f"| {s:>4} | {r['time_per_path_s']:>6.2f}s | {r['paths_in_budget']:>5} | {conf:>5.1f}% |")
    
    report.append("")
    report.append("**Key insight**: At 1000 tok/s, you get 10 independent reasoning paths in 2 seconds.")
    report.append("At 60 tok/s, you get... 1 partial path. This is a qualitative difference, not just quantitative.")
    report.append("")
    
    report.append("## 2. The Best-of-N Advantage")
    report.append("")
    report.append("Assuming each reasoning path has an independent 15% error rate:")
    report.append("")
    report.append("| N paths | P(at least one correct) |")
    report.append("|---------|------------------------|")
    for n in [1, 2, 3, 5, 8, 10, 15, 20]:
        conf = best_of_n_confidence(n)
        report.append(f"| {n:>2} | {conf:>5.1f}% |")
    
    report.append("")
    report.append("At N=10 (what UltraSpeed enables in 2s), confidence reaches ~99.9%.")
    report.append("At N=1 (slow models), confidence is just 85%. Speed IS intelligence.")
    report.append("")
    
    report.append("## 3. Cost-Speed Tradeoff (MiMo UltraSpeed: 3× cost, 10× speed)")
    report.append("")
    report.append("| Scenario | Tokens | Base (70 tok/s) | Ultra (1000 tok/s) | Time Saved | Speedup |")
    report.append("|----------|--------|----------------|-------------------|------------|---------|")
    for c in costs:
        report.append(f"| {c['scenario']} | {c['tokens']:,} | {c['base_time_s']}s | {c['ultra_time_s']}s | {c['time_saved_s']}s | {c['speedup']}× |")
    
    report.append("")
    report.append("## 4. Frontier Thresholds")
    report.append("")
    for f in frontiers:
        report.append(f"### {f['speed']} tok/s — {f['name']}")
        report.append("")
        for m in f['milestones']:
            report.append(f"- {m}")
        report.append(f"- **Best-of-N confidence (2s budget)**: {f['best_of_n_confidence_pct']}%")
        report.append("")
    
    return '\n'.join(report)

# ============================================================
# 5. JSON DATA EXPORT (for web consumption)
# ============================================================

def export_json():
    """Export analysis data as JSON for the web frontend."""
    speeds = [5, 10, 30, 60, 100, 200, 500, 1000, 1200]
    data = {
        'reasoning_paths': [reasoning_paths_possible(s) for s in speeds],
        'best_of_n': [{'n': n, 'confidence': best_of_n_confidence(n)} for n in [1,2,3,5,8,10,15,20,30]],
        'cost_scenarios': cost_analysis(),
        'frontiers': frontier_analysis(),
        'generated': '2026-06-09T05:00:00+08:00',
        'source': 'Xiaomi MiMo-V2.5-Pro-UltraSpeed announcement',
    }
    return data

if __name__ == '__main__':
    # Print report
    print(generate_report())
    
    # Export JSON
    json_path = OUT / 'speed_frontier_data.json'
    json_path.write_text(json.dumps(export_json(), indent=2, ensure_ascii=False))
    print(f"\n[JSON data exported to {json_path}]")
    
    # Quick summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    r1000 = reasoning_paths_possible(1000)
    r60 = reasoning_paths_possible(60)
    print(f"At 1000 tok/s: {r1000['paths_in_budget']} reasoning paths in 2s → {best_of_n_confidence(r1000['paths_in_budget'])}% confidence")
    print(f"At   60 tok/s: {r60['paths_in_budget']} reasoning path  in 2s → {best_of_n_confidence(r60['paths_in_budget'])}% confidence")
    print(f"Gap: {r1000['paths_in_budget'] - r60['paths_in_budget']} more paths = paradigm shift")
