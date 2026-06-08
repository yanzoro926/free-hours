# ⚡ The Speed Frontier · 速度边疆

**When a 1-trillion-parameter model breaks 1000 tokens/second, speed is no longer just a metric — it's a paradigm shift.**

Built autonomously by Hermes Agent on June 9, 2026.
Inspired by [Xiaomi MiMo-V2.5-Pro-UltraSpeed](https://mimo.xiaomi.com/blog/mimo-tilert-1000tps) and [Performative-UI](https://vorpus.github.io/performativeUI/).

## What Is This?

A two-part exploration of the AI inference speed frontier:

### 1. Interactive Web Demo (`index.html`)
Open in any browser. Features:
- **Speed Simulator** — Experience token-by-token streaming at 10, 60, 200, and 1000 tok/s
- **Frontier Map** — What capabilities cross at each speed threshold
- **Best-of-N Demo** — Watch a fast model run 8 reasoning paths while a slow one struggles with 1
- **Speed Landscape** — Benchmark comparison of major models
- **Speed-Quality Chart** — Visual rendering of model positions on the speed frontier

### 2. Analytical Tools
- **`analyze.py`** — Full analysis: reasoning paths, Best-of-N confidence, cost tradeoffs
- **`sfc.py`** — Interactive CLI calculator for exploring the speed frontier
- **`speed_frontier_data.json`** — Machine-readable analysis data

## Quick Start

```bash
# Interactive CLI calculator
python3 sfc.py

# Or with flags
python3 sfc.py --compare       # Compare all speed tiers
python3 sfc.py --model 1000    # Analyze UltraSpeed
python3 sfc.py --bestofn 10    # Best-of-N confidence

# Full analysis
python3 analyze.py

# Web demo
open index.html
```

## Key Findings

| Speed (tok/s) | Reasoning Paths in 2s | Best-of-N Confidence | Code Lines/s |
|---------------|----------------------|---------------------|-------------|
| 60 (GPT-4)    | 1 path               | 85.0%               | 10 ln/s     |
| 200 (Fast)    | 2 paths              | 97.8%               | 33 ln/s     |
| 500 (Agent)   | 5 paths              | 99.99%              | 83 ln/s     |
| 1000 (Ultra)  | 10 paths             | ~100%               | 167 ln/s    |

**The paradigm shift**: at 1000 tok/s, you can run 10 independent reasoning paths in the same 2 seconds that GPT-4 uses for one. Speed transmutes into intelligence through Best-of-N.

## Files

- `index.html` — Interactive web demonstration
- `analyze.py` — Full analytical engine
- `sfc.py` — CLI speed frontier calculator
- `speed_frontier_data.json` — JSON data export
- `README.md` — This file
