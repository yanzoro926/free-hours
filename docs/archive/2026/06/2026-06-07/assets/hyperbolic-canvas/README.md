# Curvature of Thought / 双曲画布

*An interactive hyperbolic geometry explorer — where straight lines bend and triangles remember they're less than 180°.*

## What is this?

This is a complete interactive exploration of **hyperbolic geometry** using the Poincaré disk model, built in a single 2.5-hour free exploration session. It includes:

- **Interactive Canvas** — Draw geodesics, triangles, and measure distances on the hyperbolic plane
- **Tree Embedding Visualizer** — See how hierarchical data naturally inhabits hyperbolic space  
- **Model Comparison** — Same tessellation across Poincaré disk, Klein disk, and Upper Half-Plane
- **Network Graph Generator** — Random geometric graphs in hyperbolic space produce scale-free, small-world networks
- **Tessellation Engine** — Generate {p,q} hyperbolic tessellations using reflection groups (BFS)
- **Computational Exploration** — Python scripts investigating distance distortion, angle defect, parallel lines, and exponential growth
- **6 Publication-Quality Figures** — Dark-themed matplotlib visualizations

## Quick Start

```bash
# Interactive Poincaré canvas
open hyperbolic-canvas/index.html

# Tree-to-hyperbolic embedding visualizer
open hyperbolic-canvas/tree-embedding.html

# Three-model comparison
open hyperbolic-canvas/model-comparison.html

# Hyperbolic network graph generator
open hyperbolic-canvas/network-graphs.html

# Computational exploration (generates figures)
cd hyperbolic-canvas
conda run -n hermesauto python explore.py
```

## Interactive Canvas Features

| Tool | Description |
|------|-------------|
| **Navigate** | Pan and zoom the Poincaré disk |
| **Point** | Place points in hyperbolic space |
| **Geodesic** | Draw straight lines (circular arcs orthogonal to boundary) |
| **Triangle** | Construct hyperbolic triangles — watch the angle sum drop below 180° |
| **Measure** | Measure hyperbolic distance between points |

### Tessellations

Choose from {3,7}, {3,8}, {4,7}, {5,8}, {7,3}, {7,4}, {8,3} — all satisfying the hyperbolic condition (p−2)(q−2) > 4. The {3,7} tessellation was the subject of M.C. Escher's *Circle Limit III*.

## Mathematical Background

The Poincaré disk model maps the entire infinite hyperbolic plane onto the unit disk. Key properties:

1. **Straight lines** appear as circular arcs orthogonal to the disk boundary
2. **Distance** grows exponentially as you approach the boundary: d = arcosh(1 + 2|u−v|²/((1−|u|²)(1−|v|²)))
3. **Triangle angle sum** is always < 180° (π) — the defect equals the triangle's area
4. **Infinite parallels** — through any point not on a line, infinitely many geodesics never intersect that line
5. **Circle circumference** grows exponentially: C = 2π·sinh(r), not 2πr

## Figures

| Figure | Description |
|--------|-------------|
| `distance_distortion.png` | How hyperbolic distance explodes near the boundary |
| `triangle_defect.png` | Angle sum distribution and relationship to perimeter |
| `parallel_lines.png` | Illustration of the infinite parallel lines through a point |
| `tessellation_space.png` | Which {p,q} pairs produce valid hyperbolic tessellations |
| `circle_growth.png` | Exponential growth of circumference and area |
| `poincare_tessellation.png` | Python-rendered {5,4} Poincaré disk tessellation |

## Files

- `index.html` — Interactive hyperbolic canvas (~700 lines JS + HTML)
- `tree-embedding.html` — Tree-to-hyperbolic embedding visualizer (~530 lines)
- `model-comparison.html` — Three-model comparison: Poincaré, Klein, Half-Plane (~400 lines)
- `explore.py` — Computational exploration and figure generation (~550 lines)
- `figures/` — Generated PNG figures (6 total)
- `README.md` — This file

## Why "Curvature of Thought"?

In Euclidean geometry, parallel lines never meet. In hyperbolic geometry, they diverge exponentially. Our minds work the same way — ideas that seem parallel can curve toward each other in the right cognitive space. The Poincaré disk is a metaphor for thought itself: finite in appearance, infinite in depth.

## License

MIT — built as part of a free exploration session, June 7, 2026.
