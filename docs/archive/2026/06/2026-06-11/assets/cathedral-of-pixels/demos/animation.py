#!/usr/bin/env python3
"""
Generate a rotating animation GIF of the 3D scene.
Renders multiple frames with the camera orbiting around.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from renderer import Renderer, create_cube, create_sphere, create_torus, create_pyramid, Light, Material
from linalg import Vec3
from PIL import Image


def render_animation():
    """Render a 360° rotating animation."""
    W, H = 640, 480  # Smaller for GIF
    renderer = Renderer(W, H)

    # Lighting
    renderer.add_light(Light(Vec3(5, 8, 5), Vec3(1.0, 0.95, 0.8), 1.5))
    renderer.add_light(Light(Vec3(-5, 2, -5), Vec3(0.3, 0.4, 0.8), 0.5))

    # Scene objects
    torus = create_torus(0.7, 0.25, 20, 14)
    torus.position = Vec3(0, 0.2, 0)
    torus.rotation = Vec3(0.5, 0, 0)
    torus.material = Material("torus", diffuse=Vec3(0.2, 0.5, 0.3),
                              specular=Vec3(0.7, 0.7, 0.7), shininess=64)
    renderer.add_mesh(torus)

    sphere = create_sphere(0.4, 16, 12)
    sphere.position = Vec3(1.5, 0, 0.5)
    sphere.material = Material("sphere", diffuse=Vec3(0.8, 0.3, 0.2),
                               specular=Vec3(0.6, 0.6, 0.6), shininess=32)
    renderer.add_mesh(sphere)

    cube = create_cube(0.6)
    cube.position = Vec3(-1.5, 0, -0.3)
    cube.rotation = Vec3(0.3, 0.5, 0.1)
    cube.material = Material("cube", diffuse=Vec3(0.2, 0.3, 0.8),
                             specular=Vec3(0.5, 0.5, 0.5), shininess=16)
    renderer.add_mesh(cube)

    pyramid = create_pyramid(0.5)
    pyramid.position = Vec3(0, -0.5, 1.2)
    pyramid.material = Material("pyramid", diffuse=Vec3(0.8, 0.6, 0.1),
                                specular=Vec3(0.9, 0.8, 0.3), shininess=80)
    renderer.add_mesh(pyramid)

    # Render 36 frames (10° each)
    frames = []
    renderer.camera.distance = 4.5
    renderer.camera.elevation = 0.35

    print(f"Rendering {W}x{H} animation (36 frames)...")
    for i in range(36):
        angle = 2 * math.pi * i / 36
        renderer.camera.azimuth = angle
        img = renderer.render()
        frames.append(img)
        if i % 6 == 0:
            print(f"  Frame {i+1}/36 ({(i+1)*100//36}%)")

    # Save as GIF
    out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'rotation_animation.gif')
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # 100ms per frame = ~10fps
        loop=0,
        optimize=True,
    )
    print(f"✓ Saved: {out_path}")
    print(f"  {len(frames)} frames, {os.path.getsize(out_path)//1024} KB")


if __name__ == '__main__':
    render_animation()
