#!/usr/bin/env python3
"""
Generate a side-by-side shading comparison image.
Shows the same scene rendered with Wireframe, Flat, and Gouraud shading.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from renderer import Renderer, create_cube, create_sphere, create_torus, Light, Material
from linalg import Vec3
from PIL import Image
import numpy as np


def render_shading_comparison():
    """Render same scene with 3 shading modes side by side."""
    W, H = 400, 400
    
    def render_one(mode):
        r = Renderer(W, H)
        r.add_light(Light(Vec3(5, 8, 5), Vec3(1.0, 0.95, 0.8), 1.5))
        r.add_light(Light(Vec3(-5, 2, -5), Vec3(0.3, 0.4, 0.8), 0.5))

        # Scene: torus + sphere
        torus = create_torus(0.7, 0.25, 20, 14)
        torus.position = Vec3(0, 0.2, 0)
        torus.rotation = Vec3(0.5, 0, 0)
        torus.material = Material("t", diffuse=Vec3(0.2, 0.5, 0.3),
                                  specular=Vec3(0.7, 0.7, 0.7), shininess=64)
        r.add_mesh(torus)

        sphere = create_sphere(0.35, 14, 10)
        sphere.position = Vec3(1.2, 0.3, 0.3)
        sphere.material = Material("s", diffuse=Vec3(0.8, 0.3, 0.2),
                                   specular=Vec3(0.6, 0.6, 0.6), shininess=32)
        r.add_mesh(sphere)

        r.camera.distance = 3.5
        r.camera.elevation = 0.35
        r.camera.azimuth = 0.4

        if mode == 'wire':
            r.rasterizer.wireframe = True
            r.rasterizer.wireframe_color = (80, 80, 120)
        elif mode == 'flat':
            r.rasterizer.wireframe = False
        elif mode == 'gouraud':
            r.rasterizer.wireframe = False
            # Use per-vertex normals (already computed)
        
        return r.render()

    print("Rendering wireframe...")
    img_wire = render_one('wire')
    print("Rendering flat...")
    img_flat = render_one('flat')
    print("Rendering gouraud...")
    img_gouraud = render_one('gouraud')

    # Create composite
    composite = Image.new('RGB', (W * 3 + 40, H + 60), (10, 10, 25))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    labels = [("Wireframe", 0), ("Flat Shading", 1), ("Gouraud Shading", 2)]
    for label, idx in labels:
        x = 10 + idx * (W + 10)
        draw.text((x + W//2 - 50, H + 15), label, fill=(180, 180, 200), font=font)
    
    composite.paste(img_wire, (10, 10))
    composite.paste(img_flat, (W + 20, 10))
    composite.paste(img_gouraud, (2*W + 30, 10))

    out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'shading_comparison.png')
    composite.save(out_path)
    print(f"✓ Saved: {out_path}")
    return out_path


if __name__ == '__main__':
    render_shading_comparison()
