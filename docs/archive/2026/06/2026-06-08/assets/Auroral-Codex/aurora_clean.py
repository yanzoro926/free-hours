#!/usr/bin/env python3
"""
Auroral Codex / 极光手稿 — Animated Aurora Borealis in Your Terminal

A procedural terminal art piece that renders animated northern lights
using ANSI 24-bit truecolor and layered Perlin noise. Each run produces
a unique, never-repeating aurora display.

Inspired by the IOCCC 2025 winners:
  - endoh2's "Lichtenberg curves" (terminal-based procedural art)
  - tompng's "Synthetic seashore" (layered natural phenomenon simulation)

Technical notes:
  - Uses time-based noise fields for smooth, organic motion
  - 6 overlapping aurora bands with independent parameters
  - HSL→RGB conversion for natural aurora color gradients
  - Adaptive to terminal width (resize-safe)
  - Pure Python stdlib — no dependencies beyond Python 3.x

Author: Hermes Agent (daily free exploration, 2026-06-08)
License: MIT
"""

import os
import sys
import time
import math
import random
import signal

# ─── Noise Engine ────────────────────────────────────────────────
# Vectorized 2D Perlin-like smooth noise using hash + interpolation.
# Size: 64x64 lookup table with smooth interpolation.
# The "time" dimension scrolls through the noise field horizontally.

class NoiseField:
    """Smooth procedural noise for organic aurora motion."""

    def __init__(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000) % 2**31
        random.seed(seed)
        self.seed = seed
        # Precompute random gradients
        self.size = 64
        self.grad = [[(random.random() * 2 - 1, random.random() * 2 - 1)
                      for _ in range(self.size)] for _ in range(self.size)]

    @staticmethod
    def _fade(t):
        """Smoothstep interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(a, b, t):
        return a + (b - a) * t

    def _dot_grad(self, ix, iy, x, y):
        gx, gy = self.grad[iy % self.size][ix % self.size]
        return gx * (x - ix) + gy * (y - iy)

    def sample(self, x, y, t_offset=0):
        """Sample noise at position (x, y) with time offset for animation."""
        # Scroll y coordinate with time for horizontal aurora motion
        y = y + t_offset * 0.3

        x0 = int(math.floor(x)) % self.size
        x1 = (x0 + 1) % self.size
        y0 = int(math.floor(y)) % self.size
        y1 = (y0 + 1) % self.size

        sx = self._fade(x - math.floor(x))
        sy = self._fade(y - math.floor(y))

        n0 = self._lerp(
            self._dot_grad(x0, y0, x, y),
            self._dot_grad(x1, y0, x, y),
            sx
        )
        n1 = self._lerp(
            self._dot_grad(x0, y1, x, y),
            self._dot_grad(x1, y1, x, y),
            sy
        )
        return self._lerp(n0, n1, sy)

    def fbm(self, x, y, t=0, octaves=4, lacunarity=2.0, gain=0.5):
        """Fractional Brownian Motion — layered noise for natural detail."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0

        for _ in range(octaves):
            value += amplitude * self.sample(x * frequency, y * frequency, t)
            max_value += amplitude
            amplitude *= gain
            frequency *= lacunarity

        return value / max_value


# ─── Aurora Renderer ──────────────────────────────────────────────

class AuroraRenderer:
    """Renders multi-layered aurora borealis on the terminal."""

    # Aurora band configurations: (height_center, thickness, hue, speed, opacity)
    AURORA_BANDS = [
        (0.25, 0.12, 160, 0.8, 1.0),   # Green band (lower)
        (0.30, 0.10, 140, 1.1, 0.9),   # Green-teal
        (0.20, 0.08, 180, 0.6, 0.7),   # Teal
        (0.35, 0.15, 120, 0.9, 0.85),  # Yellow-green
        (0.40, 0.06, 280, 1.3, 0.6),   # Purple (upper)
        (0.15, 0.05, 200, 0.5, 0.5),   # Cyan edge
    ]

    # Star configurations: (density, twinkle_speed)
    STARS_DENSITY = 0.03
    STARS_TWINKLE_SPEED = 2.0

    def __init__(self, seed=None):
        self.noise = NoiseField(seed)
        self.seed = self.noise.seed
        self.width = 0
        self.height = 0
        self.time_start = time.time()
        self.frame_count = 0

        # Pre-generate star positions to avoid per-frame noise
        self.stars = []
        self._generate_stars(max_cols=200, max_rows=80)

    def _generate_stars(self, max_cols, max_rows):
        """Generate random star field."""
        random.seed(self.seed + 42)
        self.stars = []
        for col in range(max_cols):
            for row in range(max_rows):
                if random.random() < self.STARS_DENSITY:
                    phase = random.random() * math.pi * 2
                    self.stars.append((col, row, phase))

    def _hsl_to_rgb(self, h, s, l):
        """Convert HSL to RGB (all in [0, 1]). Returns (r, g, b) in [0, 255]."""
        if s == 0:
            v = int(l * 255)
            return (v, v, v)

        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        h_norm = (h % 360) / 360.0

        r = int(hue_to_rgb(p, q, h_norm + 1/3) * 255)
        g = int(hue_to_rgb(p, q, h_norm) * 255)
        b = int(hue_to_rgb(p, q, h_norm - 1/3) * 255)
        return (r, g, b)

    def _get_terminal_size(self):
        """Get terminal dimensions, falling back to defaults."""
        try:
            cols, rows = os.get_terminal_size()
        except (ValueError, OSError):
            cols, rows = 80, 24
        return max(cols, 20), max(rows, 10)

    def _aurora_intensity(self, row, col, t, band_config):
        """Compute aurora intensity for a specific band at given position."""
        center, thickness, hue, speed, opacity = band_config
        height_fraction = row / max(self.height, 1)

        # Horizontal noise for the aurora curtain shape
        nx = col / (self.width * 0.3)  # Scale noise to terminal width
        ny = t * speed * 0.1  # Time-scrolling for animation

        # FBM for organic aurora curtain
        curtain = self.noise.fbm(nx, ny, t * 0.3, octaves=4)

        # Vertical position: Gaussian-like falloff from center
        distance = abs(height_fraction - center) / thickness
        if distance > 2.0:
            return 0.0

        vertical_factor = math.exp(-distance * distance)

        # Combine curtain shape with vertical profile
        intensity = (curtain * 0.7 + 0.3) * vertical_factor * opacity

        return max(0.0, min(1.0, intensity))

    def render_frame(self):
        """Render one frame of the aurora to a string buffer."""
        self.width, self.height = self._get_terminal_size()
        t = time.time() - self.time_start

        # Build frame in a list of strings for efficiency
        lines = []

        for row in range(self.height):
            line_chars = []
            for col in range(self.width):
                # Accumulate color from all aurora bands
                total_r, total_g, total_b = 0.0, 0.0, 0.0
                total_intensity = 0.0

                for band in self.AURORA_BANDS:
                    intensity = self._aurora_intensity(row, col, t, band)
                    if intensity > 0.01:
                        hue = band[2]
                        r, g, b = self._hsl_to_rgb(hue, 0.8, intensity * 0.7)
                        total_r += r * intensity
                        total_g += g * intensity
                        total_b += b * intensity
                        total_intensity += intensity

                # Star field
                star_brightness = 0
                for sx, sy, phase in self.stars:
                    if sx == col and sy == row:
                        twinkle = (math.sin(t * self.STARS_TWINKLE_SPEED + phase) + 1) / 2
                        star_brightness = twinkle * 0.6
                        break

                # Background: very dark blue gradient
                bg_r = 2
                bg_g = 2
                bg_b = int(5 + (1 - row / max(self.height, 1)) * 10)

                if total_intensity > 0.001:
                    # Blend aurora with background
                    alpha = min(total_intensity, 1.0)
                    if total_intensity > 0:
                        avg_r = total_r / total_intensity
                        avg_g = total_g / total_intensity
                        avg_b = total_b / total_intensity
                    else:
                        avg_r, avg_g, avg_b = 0, 0, 0

                    r = int(bg_r * (1 - alpha) + avg_r * alpha + star_brightness * 255)
                    g = int(bg_g * (1 - alpha) + avg_g * alpha + star_brightness * 255)
                    b = int(bg_b * (1 - alpha) + avg_b * alpha + star_brightness * 200)
                else:
                    r = int(bg_r + star_brightness * 255)
                    g = int(bg_g + star_brightness * 255)
                    b = int(bg_b + star_brightness * 200)

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                # ANSI 24-bit truecolor: \x1b[48;2;R;G;Bm
                line_chars.append(f"\x1b[48;2;{r};{g};{b}m ")

            # End of row: reset and newline
            line_chars.append("\x1b[0m")
            lines.append("".join(line_chars))

        return "".join(lines)

    def run(self, fps=20, max_frames=None):
        """Run the aurora animation loop."""
        # Hide cursor
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()

        def cleanup(sig=None, frame=None):
            sys.stdout.write("\x1b[?25h\x1b[0m\x1b[2J\x1b[H")
            sys.stdout.flush()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup)
        signal.signal(signal.SIGTERM, cleanup)

        try:
            frame_delay = 1.0 / fps
            while True:
                if max_frames and self.frame_count >= max_frames:
                    break

                frame_start = time.time()
                frame = self.render_frame()

                # Clear screen and render
                sys.stdout.write("\x1b[H")  # Move to home
                sys.stdout.write(frame)
                sys.stdout.flush()

                self.frame_count += 1

                # Frame rate limiting
                elapsed = time.time() - frame_start
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)

        finally:
            cleanup()


# ─── Main ─────────────────────────────────────────────────────────

def main():
    """Entry point for aurora terminal animation."""
    print("\x1b[2J\x1b[H", end="", flush=True)  # Clear screen

    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            seed = hash(sys.argv[1]) % 2**31

    renderer = AuroraRenderer(seed)
    print(f"\x1b[38;2;100;200;150m"
          f"  ✦ Auroral Codex / 极光手稿 ✦  seed={renderer.seed}"
          f"\x1b[0m\n"
          f"  Press Ctrl+C to exit\n", flush=True)
    time.sleep(1.5)

    renderer.run(fps=15)


if __name__ == "__main__":
    main()
