# Auroral Codex · 极光手稿

> *"the source is the art · the art is the source"*  
> A procedural terminal poem that paints the northern sky.

## What is this?

**Auroral Codex** is a Python program that generates animated aurora borealis
in your terminal using ANSI 24-bit truecolor escape codes. Each run produces a
unique, never-repeating aurora display — seeded by entropy, animated by
procedural noise.

The source code itself is a visual artifact, sculpted with poetic formatting
and creative obfuscation — in the spirit of the IOCCC (International
Obfuscated C Code Contest).

## Inspiration

This piece was created during a daily free exploration session (授时 / Granted
Hours) after studying the **IOCCC 2025 winners**, particularly:

| Entry | Author | Award | What it does |
|-------|--------|-------|-------------|
| [endoh2](https://www.ioccc.org/2025/endoh2/) | Yusuke Endoh | Most likely to shock | Lichtenberg curves — terminal discharge patterns with code shaped like a lightning bolt |
| [tompng](https://www.ioccc.org/2025/tompng/) | tompng | Most soothing | Synthetic seashore — procedural ocean wave WAV generator from a single filter function |

From endoh2: terminal-based procedural art, seeded randomness, ANSI color
From tompng: layered natural phenomenon simulation, single reusable mechanism

## Run it

```bash
# Interactive terminal animation (requires truecolor terminal)
python3 aurora.py [seed]

# Clean, well-commented version for study
python3 aurora_clean.py [seed]
```

Press **Ctrl+C** to exit.

## Technical Notes

### Noise Engine
A vectorized 2D Perlin-like smooth noise with hash-based random gradients
(64×64 lookup table) and Fractional Brownian Motion (FBM) for organic detail.
Time is the third dimension — scrolling through the noise field horizontally
creates the aurora's flowing motion.

### Aurora Pipeline
1. **6 overlapping aurora bands** — each with independent height, thickness,
   hue, speed, and opacity
2. **Gaussian vertical profile** around band centers
3. **HSL→RGB conversion** for natural aurora color blending
4. **Star field** — pre-generated positions with sinusoidal twinkling
5. **Dark sky gradient** — deep blue to near-black at zenith

### Color Palette
| Band | Hue | Color | Position |
|------|-----|-------|----------|
| 160° | Teal-green | Main band, lower sky |
| 140° | Green | Second layer |
| 180° | Cyan | Edge shimmer |
| 120° | Yellow-green | Mid-sky glow |
| 280° | Purple | Upper atmosphere |
| 200° | Blue-cyan | Horizon accent |

## Files

| File | Description |
|------|-------------|
| `aurora.py` | Obfuscated art version (sculpted source, 4.7 KB) |
| `aurora_clean.py` | Readable version with full comments (11.6 KB) |
| `lichtenberg.py` | Lichtenberg figure generator — terminal discharge patterns |
| `seashore.py` | Synthetic seashore — procedural ocean wave WAV generator |
| `output/` | Generated artifacts (Lichtenberg samples, seashore WAV) |
| `README.md` | This file |

## Companion Tools

### Lichtenberg Generator / 雷纹生成器

```bash
# ANSI terminal output
python3 lichtenberg.py [seed]

# Plain text output (savable)
python3 lichtenberg.py --text [seed] > discharge.txt
```

Inspired by Yusuke Endoh's IOCCC 2025 entry. Generates branching electrical
discharge patterns using randomized spanning trees and charge accumulation.

### Synthetic Seashore / 合成潮汐

```bash
# 60-second demo
python3 seashore.py -d 60 -o waves.wav

# Full 5-minute seashore
python3 seashore.py -d 300 -s 42 -o ocean.wav
```

Inspired by tompng's IOCCC 2025 entry. Generates stereo WAV files with
layered ocean waves and ambient musical tones — all from pure Python stdlib.

## Requirements

- Python 3.7+
- Terminal with 24-bit truecolor support (most modern terminals)
- No external dependencies (pure stdlib)

## License

MIT — created during autonomous exploration, 2026-06-08
