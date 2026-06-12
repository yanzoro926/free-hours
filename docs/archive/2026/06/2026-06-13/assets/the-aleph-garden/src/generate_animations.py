#!/usr/bin/env python3
"""
Glider Animation · 滑翔机动画
Generates an animated GIF of Conway's iconic glider traversing the grid.
Output: output/glider.gif
"""

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).parent.parent / 'output'
OUTDIR.mkdir(exist_ok=True)

# ─── Glider pattern ───
GLIDER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
])

def step_life(grid):
    """One step of Conway's Life (B3/S23) — no wrap, padded."""
    h, w = grid.shape
    new = np.zeros_like(grid)
    for y in range(h):
        for x in range(w):
            n = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        n += grid[ny, nx]
            if grid[y, x]:
                new[y, x] = 1 if n in (2, 3) else 0
            else:
                new[y, x] = 1 if n == 3 else 0
    return new

def render_frame(grid, cell_size=12, padding=4):
    """Render a grid as a dark-themed RGBA image."""
    h, w = grid.shape
    img_h = h * cell_size
    img_w = w * cell_size
    
    # Background
    img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    img[:, :, 0] = 8    # R
    img[:, :, 1] = 12   # G
    img[:, :, 2] = 20   # B
    img[:, :, 3] = 255  # A
    
    # Grid lines
    for y in range(0, img_h, cell_size):
        img[y:y+1, :, :3] = [17, 24, 48]
    for x in range(0, img_w, cell_size):
        img[:, x:x+1, :3] = [17, 24, 48]
    
    # Alive cells
    for y in range(h):
        for x in range(w):
            if grid[y, x]:
                y0, y1 = y * cell_size + 1, (y+1) * cell_size - 1
                x0, x1 = x * cell_size + 1, (x+1) * cell_size - 1
                # Gradient fill
                for dy in range(y0, y1):
                    t = (dy - y0) / max(y1 - y0, 1)
                    r = int(80 + 40 * t)
                    g = int(180 + 40 * t)
                    b = int(120 + 20 * t)
                    img[dy, x0:x1, 0] = r
                    img[dy, x0:x1, 1] = g
                    img[dy, x0:x1, 2] = b
    
    return img

def generate_glider_gif():
    """Generate a 32-frame animation of a glider moving diagonally."""
    # Create a 20x20 grid with the glider in the top-left
    grid_size = 20
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[0:3, 0:3] = GLIDER
    
    frames = []
    
    # The glider moves 1 cell right and 1 cell down every 4 generations
    # 32 frames = 8 cycles of the glider pattern
    for i in range(32):
        frame = render_frame(grid, cell_size=16)
        pil_frame = Image.fromarray(frame, 'RGBA')
        frames.append(pil_frame)
        grid = step_life(grid)
        
        # If the glider hits the edge, wrap it
        if grid.sum() < 3:  # glider hit boundary
            grid = np.zeros((grid_size, grid_size), dtype=int)
            # Place glider at top-left again but offset
            offset = (i // 4) % (grid_size - 5)
            grid[offset:offset+3, offset:offset+3] = GLIDER
    
    # Save as GIF
    outpath = OUTDIR / 'glider.gif'
    frames[0].save(
        outpath,
        save_all=True,
        append_images=frames[1:],
        duration=120,  # ms per frame
        loop=0,
        optimize=False,
    )
    print(f'✓ Glider GIF: {outpath} ({len(frames)} frames)')
    return str(outpath)


def generate_pattern_sheet():
    """Generate a static pattern reference sheet showing all 13 patterns."""
    patterns = {
        'Glider': np.array([[0,1,0],[0,0,1],[1,1,1]]),
        'LWSS': np.array([[0,1,1,1,1],[1,0,0,0,1],[0,0,0,0,1],[1,0,0,1,0]]),
        'Blinker': np.array([[1,1,1]]),
        'Toad': np.array([[0,1,1,1],[1,1,1,0]]),
        'Beacon': np.array([[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]),
        'Block': np.array([[1,1],[1,1]]),
        'Beehive': np.array([[0,1,1,0],[1,0,0,1],[0,1,1,0]]),
        'Pulsar': np.array([[0,0,1,1,1,0,0,0,1,1,1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [0,0,1,1,1,0,0,0,1,1,1,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,1,0,0,0,1,1,1,0,0],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [1,0,0,0,0,1,0,1,0,0,0,0,1],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,1,1,1,0,0,0,1,1,1,0,0]]),
    }
    
    # Render each pattern in a grid
    n = len(patterns)
    cols = 4
    rows = (n + cols - 1) // cols
    
    cell_size = 12
    panel_pad = 20
    panel_w = 20 * cell_size  # max pattern width
    panel_h = 20 * cell_size
    
    img_w = cols * (panel_w + panel_pad) + panel_pad
    img_h = rows * (panel_h + panel_pad + 30) + panel_pad
    
    img = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    img[:, :, 0] = 8
    img[:, :, 1] = 12
    img[:, :, 2] = 20
    img[:, :, 3] = 255
    
    for idx, (name, pattern) in enumerate(patterns.items()):
        row, col = divmod(idx, cols)
        px = panel_pad + col * (panel_w + panel_pad)
        py = panel_pad + row * (panel_h + panel_pad + 30)
        
        # Draw name
        # (We'll skip text rendering with pure arrays — just draw the pattern)
        
        ph, pw = pattern.shape
        for y in range(ph):
            for x in range(pw):
                if pattern[y, x]:
                    y0 = py + y * cell_size
                    x0 = px + x * cell_size
                    img[y0:y0+cell_size-1, x0:x0+cell_size-1, 1] = 200
                    img[y0:y0+cell_size-1, x0:x0+cell_size-1, 2] = 140
    
    outpath = OUTDIR / 'pattern_sheet.png'
    pil_img = Image.fromarray(img, 'RGBA')
    pil_img.save(outpath)
    print(f'✓ Pattern sheet: {outpath}')
    return str(outpath)


if __name__ == '__main__':
    print("═" * 40)
    print("  Animations & Extras")
    print("═" * 40)
    generate_glider_gif()
    generate_pattern_sheet()
    print("Done!")
