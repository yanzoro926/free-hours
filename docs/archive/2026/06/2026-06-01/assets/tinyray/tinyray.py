#!/usr/bin/env python3
"""TinyRay — a minimal ray tracer in pure Python (no external deps beyond numpy).

Renders spheres with reflections, shadows, and ambient lighting.
Inspired by the classic "Ray Tracing in One Weekend" but in ~100 lines.

Usage:
    conda run -n hermesauto python tinyray.py
"""

import numpy as np
from pathlib import Path

OUT = Path("/home/yanyj/VibeCoding/autonomy/2026-06-01/tinyray")
OUT.mkdir(parents=True, exist_ok=True)


def normalize(v):
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True))


def ray_sphere_intersect(origin, direction, center, radius):
    """Ray-sphere intersection. Returns t or inf."""
    oc = origin - center
    a = np.sum(direction * direction, axis=-1)
    b = 2.0 * np.sum(oc * direction, axis=-1)
    c = np.sum(oc * oc, axis=-1) - radius * radius
    disc = b * b - 4 * a * c
    t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2.0 * a), np.inf)
    return t


def render_scene(w=800, h=600):
    """Render a scene with multiple colored spheres."""
    aspect = w / h
    screen = np.zeros((h, w, 3), dtype=float)

    # Camera
    for y in range(h):
        for x in range(w):
            # Ray direction
            u = (x / w - 0.5) * aspect
            v = -(y / h - 0.5)
            ray_dir = normalize(np.array([u, v, -1.0]))
            ray_origin = np.array([0.0, 0.0, 2.0])

            # Scene: [center, radius, color, reflectivity]
            spheres = [
                (np.array([0.0, -0.2, -2.0]), 0.6, np.array([0.2, 0.4, 1.0]), 0.3),   # blue
                (np.array([0.6, 0.2, -2.5]), 0.4, np.array([1.0, 0.3, 0.2]), 0.4),    # red
                (np.array([-0.7, -0.1, -2.2]), 0.35, np.array([0.2, 1.0, 0.3]), 0.3), # green
                (np.array([0.0, -100.5, -2.0]), 100.0, np.array([0.5, 0.5, 0.5]), 0.1),  # ground
                (np.array([-0.3, 0.5, -1.8]), 0.25, np.array([1.0, 0.9, 0.3]), 0.6),   # gold
            ]

            # Light position
            light = np.array([2.0, 3.0, 0.0])

            color = np.array([0.1, 0.1, 0.2])  # ambient sky
            ray_pos = ray_origin.copy()
            ray_d = ray_dir.copy()
            contribution = np.array([1.0, 1.0, 1.0])

            for bounce in range(3):  # max 3 bounces
                hit_t = np.inf
                hit_sphere = None

                for sph in spheres:
                    t = ray_sphere_intersect(ray_pos, ray_d, sph[0], sph[1])
                    if t < hit_t and t > 0.001:
                        hit_t = t
                        hit_sphere = sph

                if hit_sphere is None:
                    break

                # Hit point and normal
                hit_point = ray_pos + ray_d * hit_t
                normal = normalize(hit_point - hit_sphere[0])

                # Lighting
                light_dir = normalize(light - hit_point)
                # Shadow check
                shadow = 1.0
                for sph in spheres:
                    t = ray_sphere_intersect(hit_point + normal * 0.001, light_dir, sph[0], sph[1])
                    if t < np.inf and t < np.linalg.norm(light - hit_point):
                        shadow = 0.3
                        break

                diffuse = max(0, np.dot(normal, light_dir)) * shadow
                ambient = 0.2

                obj_color = hit_sphere[2]
                lighting = ambient + diffuse * 0.8

                color = color + contribution * obj_color * lighting * (1.0 - hit_sphere[3])

                # Reflect
                contribution = contribution * hit_sphere[3] * 0.7
                if np.max(contribution) < 0.01:
                    break

                ray_d = ray_d - 2 * np.dot(ray_d, normal) * normal
                ray_pos = hit_point + normal * 0.001

            screen[y, x] = np.clip(color, 0, 1)

        if y % 60 == 0:
            print(f"  Rendering... {y * 100 // h}%")

    return screen


def render():
    print("TinyRay — minimal ray tracer")
    print("=" * 50)
    print("Rendering 800x600 scene with 5 spheres, 3 bounces...")

    img = render_scene(800, 600)

    # Save
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.patch.set_facecolor('black')
    ax.imshow(img, origin='upper')
    ax.axis('off')
    fig.savefig(OUT / 'render.png', dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)

    # Also save a 200x150 quick preview
    print("  Rendering preview...")
    img_small = render_scene(200, 150)
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor('black')
    ax.imshow(img_small, origin='upper')
    ax.axis('off')
    fig.savefig(OUT / 'preview.png', dpi=100, bbox_inches='tight', facecolor='black')
    plt.close(fig)

    # Save raw data
    np.save(OUT / 'render.npy', img)

    size_kb = (OUT / 'render.png').stat().st_size / 1024
    print(f"\nDone! render.png ({size_kb:.0f} KB)")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    render()
