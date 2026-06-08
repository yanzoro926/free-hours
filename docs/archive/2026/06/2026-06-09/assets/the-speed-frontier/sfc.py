#!/usr/bin/env python3
"""
Speed Frontier Calculator (sfc) — a CLI tool to explore the token-speed paradigm.

Usage:
    python3 sfc.py                    # Interactive mode
    python3 sfc.py --model 1000       # Analyze UltraSpeed
    python3 sfc.py --compare          # Compare all speed tiers
    python3 sfc.py --bestofn N        # Best-of-N confidence for N paths
"""
import sys
import math
from pathlib import Path

# Add parent to path for analyze import
sys.path.insert(0, str(Path(__file__).parent))
from analyze import reasoning_paths_possible, best_of_n_confidence, cost_analysis

BOLD = '\033[1m'
DIM = '\033[2m'
GREEN = '\033[32m'
CYAN = '\033[36m'
YELLOW = '\033[33m'
RED = '\033[31m'
MAGENTA = '\033[35m'
RESET = '\033[0m'

TIER_COLORS = {
    5: RED,
    30: YELLOW,
    60: YELLOW,
    100: CYAN,
    200: CYAN,
    500: GREEN,
    1000: GREEN,
}

def header(text):
    print(f"\n{BOLD}{CYAN}═══ {text} ═══{RESET}\n")

def info(label, value, color=''):
    print(f"  {DIM}{label}:{RESET} {color}{value}{RESET}")

def speed_analysis(speed):
    """Detailed analysis of a specific speed tier."""
    r = reasoning_paths_possible(speed)
    conf = best_of_n_confidence(r['paths_in_budget'])
    color = TIER_COLORS.get(speed, '')
    
    header(f"Speed Analysis: {speed} tok/s")
    
    print(f"  {BOLD}Per-Path Performance{RESET}")
    info("Time for 200-token reasoning", f"{r['time_per_path_s']:.2f}s", color)
    info("Paths in 2-second budget", str(r['paths_in_budget']), color)
    info("Total tokens in 2 seconds", str(r['total_tokens_generated']))
    
    print(f"\n  {BOLD}Best-of-N Intelligence{RESET}")
    info("Confidence (single path, 15% error)", f"{best_of_n_confidence(1)}%")
    info("Confidence (max paths in 2s)", f"{conf}%", GREEN)
    
    # What can you do with this speed?
    if speed < 10:
        info("UX", "Painfully slow — user loses attention", RED)
    elif speed < 50:
        info("UX", "Tolerable chat, visible delays", YELLOW)
    elif speed < 200:
        info("UX", "Good conversation flow", CYAN)
    elif speed < 1000:
        info("UX", "Near-instant reasoning", GREEN)
    else:
        info("UX", "Paradigm shift — speed IS intelligence", GREEN)
    
    # Code generation estimate
    lines_per_sec = speed / 6  # rough: 6 tokens per line of code
    info("Code generation", f"~{lines_per_sec:.0f} lines/sec")
    
    # Real-time scenarios
    if speed >= 200:
        info("Scenarios", "Streaming agents, parallel tool calls")
    if speed >= 500:
        info("Scenarios", "+ Real-time translation, transcription")
    if speed >= 1000:
        info("Scenarios", "+ Quant trading, surgical AI, Best-of-N")

def compare_all():
    """Compare all speed tiers."""
    header("Speed Tier Comparison")
    
    speeds = [5, 10, 30, 60, 100, 200, 500, 1000]
    
    print(f"  {'Speed':>6} {'Paths/2s':>10} {'Time/200tok':>12} {'Confidence':>12} {'Code lines/s':>13}")
    print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*12} {'─'*13}")
    
    for s in speeds:
        r = reasoning_paths_possible(s)
        conf = best_of_n_confidence(r['paths_in_budget'])
        lps = s / 6
        color = TIER_COLORS.get(s, '')
        print(f"  {color}{s:>5}{RESET} tok/s {r['paths_in_budget']:>8} paths {r['time_per_path_s']:>8.2f}s {conf:>9}% {lps:>10.0f} ln/s")
    
    # Key insight
    print(f"\n  {BOLD}Key Insight:{RESET}")
    r1k = reasoning_paths_possible(1000)
    r60 = reasoning_paths_possible(60)
    print(f"  At 1000 tok/s: {GREEN}{r1k['paths_in_budget']} paths in 2s → {best_of_n_confidence(r1k['paths_in_budget'])}% confidence{RESET}")
    print(f"  At   60 tok/s: {YELLOW}{r60['paths_in_budget']} path  in 2s → {best_of_n_confidence(r60['paths_in_budget'])}% confidence{RESET}")
    print(f"  Gap: {r1k['paths_in_budget'] - r60['paths_in_budget']} more paths = {BOLD}paradigm shift{RESET}")

def best_of_n_table():
    """Display Best-of-N confidence table."""
    header("Best-of-N Confidence Analysis")
    print(f"  Assuming each path has {YELLOW}15% independent error rate{RESET}\n")
    print(f"  {'N paths':>8} {'Confidence':>14} {'Interpretation':>20}")
    print(f"  {'─'*8} {'─'*14} {'─'*20}")
    
    for n in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        conf = best_of_n_confidence(n)
        if conf >= 99.99:
            interp = "Effectively certain"
        elif conf >= 99:
            interp = "Very high confidence"
        elif conf >= 95:
            interp = "High confidence"
        elif conf >= 85:
            interp = "Moderate (single path)"
        else:
            interp = "Low"
        print(f"  {n:>8} {conf:>11}%   {interp:<20}")

def cost_breakdown():
    """Cost-speed tradeoff analysis."""
    header("Cost-Speed Tradeoff (MiMo UltraSpeed: 3× cost, 10× speed)")
    
    costs = cost_analysis()
    print(f"  {'Scenario':<35} {'Base (70s)':>10} {'Ultra (1s)':>12} {'Saved':>10} {'Speedup':>8}")
    print(f"  {'─'*35} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    
    for c in costs:
        print(f"  {c['scenario']:<35} {c['base_time_s']:>7.1f}s {c['ultra_time_s']:>8.1f}s {c['time_saved_s']:>7.1f}s {c['speedup']:>6.1f}×")
    
    print(f"\n  {BOLD}Bottom line:{RESET} For a heavy reasoning day (1M tokens):")
    print(f"  Base model: 238 minutes of waiting")
    print(f"  UltraSpeed: {GREEN}16.7 minutes{RESET} of waiting")
    print(f"  You get {GREEN}3.7 hours{RESET} of your life back. At 6× the cost.")

def interactive():
    """Interactive mode."""
    print(f"\n{BOLD}{CYAN}")
    print("  ╔══════════════════════════════════════╗")
    print("  ║   ⚡ SPEED FRONTIER CALCULATOR ⚡    ║")
    print("  ║        The Speed Frontier           ║")
    print("  ║           速度边疆                   ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"{RESET}")
    print(f"  Explore the token-speed paradigm shift.")
    print(f"  Inspired by Xiaomi MiMo-V2.5-Pro-UltraSpeed (1000 tok/s on 1T params).")
    
    while True:
        print(f"\n  {BOLD}Commands:{RESET}")
        print(f"    {CYAN}<speed>{RESET}  — analyze a speed tier (e.g., 1000, 60)")
        print(f"    {CYAN}c{RESET}       — compare all tiers")
        print(f"    {CYAN}n{RESET}       — Best-of-N confidence table")
        print(f"    {CYAN}$ {RESET}      — cost-speed tradeoff")
        print(f"    {CYAN}q{RESET}       — quit")
        
        try:
            cmd = input(f"\n  {BOLD}> {RESET}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        
        if cmd == 'q':
            break
        elif cmd == 'c':
            compare_all()
        elif cmd == 'n':
            best_of_n_table()
        elif cmd == '$':
            cost_breakdown()
        elif cmd.isdigit():
            speed_analysis(int(cmd))
        elif cmd:
            # Try to parse as number
            try:
                speed_analysis(int(cmd))
            except ValueError:
                print(f"  {RED}Unknown command: {cmd}{RESET}")
    
    print(f"\n  {DIM}Speed is the new intelligence. Go fast.{RESET}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Speed Frontier Calculator — explore the token-speed paradigm shift',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 sfc.py                    Interactive mode
  python3 sfc.py --model 1000      Analyze UltraSpeed tier
  python3 sfc.py --compare         Compare all speed tiers
  python3 sfc.py --bestofn 10      Best-of-N for N paths
  python3 sfc.py --cost            Cost-speed tradeoff
        '''
    )
    parser.add_argument('--model', '-m', type=int, help='Analyze a specific speed tier')
    parser.add_argument('--compare', '-c', action='store_true', help='Compare all tiers')
    parser.add_argument('--bestofn', '-b', type=int, help='Best-of-N confidence for N paths')
    parser.add_argument('--cost', action='store_true', help='Cost-speed tradeoff')
    
    args = parser.parse_args()
    
    if args.model:
        speed_analysis(args.model)
    elif args.compare:
        compare_all()
    elif args.bestofn:
        conf = best_of_n_confidence(args.bestofn)
        print(f"\n  Best-of-N with N={args.bestofn}: {conf}% confidence (15% per-path error rate)")
    elif args.cost:
        cost_breakdown()
    else:
        interactive()

if __name__ == '__main__':
    main()
