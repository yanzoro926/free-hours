#!/usr/bin/env python3
"""TinyRay Quick — fast 200x150 render."""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path("/home/yanyj/VibeCoding/autonomy/2026-06-01/tinyray")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 200, 150

def norm(v):
    return v / np.sqrt(np.sum(v*v, axis=-1, keepdims=True))

def hit(o, d, c, r):
    oc = o - c
    a = np.sum(d*d, axis=-1)
    b = 2*np.sum(oc*d, axis=-1)
    disc = b*b - 4*a*(np.sum(oc*oc, axis=-1) - r*r)
    return np.where(disc > 0, (-b - np.sqrt(np.maximum(disc,0))) / (2*a), np.inf)

spheres = [
    (np.array([0.0, -0.2, -2.0]), 0.6, np.array([0.2, 0.4, 1.0]), 0.3),
    (np.array([0.6, 0.2, -2.5]), 0.4, np.array([1.0, 0.3, 0.2]), 0.4),
    (np.array([-0.7, -0.1, -2.2]), 0.35, np.array([0.2, 1.0, 0.3]), 0.3),
    (np.array([0.0, -100.5, -2.0]), 100.0, np.array([0.5, 0.5, 0.5]), 0.1),
    (np.array([-0.3, 0.5, -1.8]), 0.25, np.array([1.0, 0.9, 0.3]), 0.6),
]
light = np.array([2.0, 3.0, 0.0])

screen = np.zeros((H, W, 3), dtype=float)
aspect = W / H

for y in range(H):
    for x in range(W):
        u = (x/W - 0.5) * aspect
        v = -(y/H - 0.5)
        rd = norm(np.array([u, v, -1.0]))
        ro = np.array([0.0, 0.0, 2.0])

        col = np.array([0.1, 0.1, 0.2])
        contrib = np.array([1.0, 1.0, 1.0])
        rp, d = ro.copy(), rd.copy()

        for _ in range(3):
            ht, hs = np.inf, None
            for s in spheres:
                t = hit(rp, d, s[0], s[1])
                if t < ht and t > 0.001:
                    ht, hs = t, s
            if hs is None: break

            hp = rp + d * ht
            n = norm(hp - hs[0])
            ld = norm(light - hp)

            sh = 1.0
            for s in spheres:
                if hit(hp + n*0.001, ld, s[0], s[1]) < np.linalg.norm(light-hp):
                    sh = 0.3; break

            diff = max(0, np.dot(n, ld)) * sh
            col = col + contrib * hs[2] * (0.2 + diff*0.8) * (1-hs[3])
            contrib = contrib * hs[3] * 0.7
            if np.max(contrib) < 0.01: break
            d = d - 2*np.dot(d, n)*n
            rp = hp + n*0.001

        screen[y,x] = np.clip(col, 0, 1)
    if y % 25 == 0:
        print(f"  {y*100//H}%")

print("Saving...")
fig, ax = plt.subplots(figsize=(8,6))
fig.patch.set_facecolor('black')
ax.imshow(screen, origin='upper')
ax.axis('off')
fig.savefig(OUT / 'render.png', dpi=150, bbox_inches='tight', facecolor='black')
plt.close(fig)
np.save(OUT / 'render.npy', screen)
print(f"Done! {OUT / 'render.png'}")
