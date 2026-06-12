# The Aleph Garden · 阿列夫花园

> *Cellular Automata Exploration — where simple rules bloom into infinite complexity.*

An interactive exploration of cellular automata as a lens for understanding emergence, complexity, and the boundary between order and chaos. Built during the June 13, 2026 autonomous exploration session.

## 🌌 Contents

| File | Description |
|------|-------------|
| `index.html` | **The Aleph Garden** — Interactive CA playground with 10 rule sets, 13 pre-built patterns, draw/pan/zoom, real-time statistics, and PNG export |
| `rule-forge.html` | **The Rule Forge** — Dynamic rule explorer where you toggle individual B/S conditions and watch the λ parameter shift in real-time. 15 preset universes |
| `src/edge_of_chaos.py` | Python analysis engine: samples 191 Life-like rules, computes λ parameters, classifies behavior, generates 3 publication charts |
| `output/lambda_phase_diagram.png` | Phase diagram: λ vs behavioral complexity |
| `output/rule_atlas.png` | 14 famous rules visualized as grid snapshots |
| `output/class_distribution.png` | 3-panel statistical analysis of rule space |
| `output/rule_space_data.json` | Machine-readable data for all sampled rules |

## 🎮 How to Play

Open `index.html` in any browser. No dependencies, no build step.

- **Left-click** on the canvas to draw/erase cells
- **Right-click + drag** to pan
- **Scroll** to zoom
- **Space** to play/pause
- Pick patterns from the library and click to place them
- Switch rule sets to explore different universes

For rule discovery: open `rule-forge.html` and toggle individual birth/survival conditions to create new rules.

## 🧪 Analysis

The `edge_of_chaos.py` script samples the Life-like CA rule space and maps the relationship between λ (fraction of configurations leading to alive cells) and behavioral complexity.

**Key finding:** Complex behavior (Life, HighLife, 3-4 Life) clusters in a narrow band at λ ≈ 0.15–0.25. Only ~1.6% of sampled rules produce complex dynamics. The edge of chaos is razor-thin.

Run: `conda run -n hermesauto python src/edge_of_chaos.py`

## 📐 The λ Parameter

For 2D Life-like cellular automata, λ measures the fraction of the 18 possible "situations" (9 neighbor counts × 2 cell states) that result in a live cell:

```
λ = (|B| + |S|) / 18
```

- λ < 0.1: Most cells die (Extinct/Static)
- λ ≈ 0.15–0.25: Complex behavior (Life, HighLife)
- λ ≈ 0.3–0.5: Oscillating/Periodic
- λ > 0.5: Static fill or chaotic noise

## 🔗 See Also

- [Conway's Game of Life](https://conwaylife.com/) — The comprehensive wiki
- [Langton's Ant & λ parameter](https://en.wikipedia.org/wiki/Langton%27s_ant) — Edge of chaos theory
- [Borges' The Aleph](https://en.wikipedia.org/wiki/The_Aleph_(short_story)) — The literary inspiration

---

*Built autonomously by Hermes Agent. June 13, 2026.*
