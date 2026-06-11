#!/usr/bin/env python3
"""
Demo: 1993 Room — textured walls, floor, ceiling, and a spinning teapot-like object.
Pure software-rendered nostalgia.
"""
import sys, math, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from renderer import Renderer, create_cube, create_sphere, create_torus, create_plane, create_pyramid, Light, Material
from linalg import Vec3
from PIL import Image, ImageDraw


def create_checker_texture(size=256, square=32):
    """Generate a checkerboard texture."""
    img = Image.new('RGB', (size, size), (128, 128, 128))
    draw = ImageDraw.Draw(img)
    for y in range(0, size, square):
        for x in range(0, size, square):
            if (x // square + y // square) % 2 == 0:
                draw.rectangle([x, y, x+square-1, y+square-1], fill=(180, 160, 140))
            else:
                draw.rectangle([x, y, x+square-1, y+square-1], fill=(100, 80, 60))
    return img


def create_brick_texture(size=256):
    """Generate a brick wall texture."""
    img = Image.new('RGB', (size, size), (80, 50, 40))
    draw = ImageDraw.Draw(img)
    brick_h = 24
    brick_w = 60
    mortar = 3
    for y in range(0, size, brick_h + mortar):
        offset = (y // (brick_h + mortar)) % 2 * (brick_w // 2)
        for x in range(-offset, size + offset, brick_w + mortar):
            bx1 = max(0, x + mortar)
            bx2 = min(size, x + brick_w)
            by1 = max(0, y + mortar)
            by2 = min(size, y + brick_h)
            if bx2 > bx1 and by2 > by1:
                # Randomize brick color slightly
                r = min(255, 160 + (hash((x, y)) % 40))
                g = min(255, 90 + (hash((x+1, y)) % 40))
                b = min(255, 50 + (hash((x, y+1)) % 40))
                draw.rectangle([bx1, by1, bx2, by2], fill=(r, g, b))
    return img


def create_sky_texture(size=512):
    """Generate a sky gradient texture."""
    img = Image.new('RGB', (size, size))
    for y in range(size):
        t = y / size
        r = int(20 + t * 30)
        g = int(30 + t * 50)
        b = int(80 + t * 70)
        for x in range(size):
            img.putpixel((x, y), (r, g, b))
    return img


def render_room_scene():
    """Render a 1993-style room with textured surfaces and objects."""
    W, H = 1024, 768
    renderer = Renderer(W, H)

    # Lighting
    renderer.add_light(Light(Vec3(0, 4, 0), Vec3(1.0, 0.9, 0.8), 1.5))
    renderer.add_light(Light(Vec3(3, 2, -3), Vec3(0.3, 0.4, 0.9), 0.8))

    # ——— Room (textured cube, inverted) ———
    room_size = 8
    floor = create_plane(room_size, 4)
    floor.position = Vec3(0, -2, 0)
    floor.material = Material("floor", diffuse=Vec3(0.4, 0.35, 0.3))
    renderer.add_mesh(floor)

    # Back wall
    back_wall = create_plane(room_size, 4)
    back_wall.position = Vec3(0, 2, -room_size/2)
    back_wall.rotation = Vec3(-math.pi/2, 0, 0)
    back_wall.material = Material("back_wall", diffuse=Vec3(0.5, 0.3, 0.25))
    renderer.add_mesh(back_wall)

    # ——— Objects in the room ———
    # Central torus (the "donut")
    torus = create_torus(1.0, 0.35, 32, 18)
    torus.position = Vec3(0, 0.5, -2)
    torus.rotation = Vec3(0.7, 0.3, 0)
    torus.material = Material("torus", 
        diffuse=Vec3(0.2, 0.5, 0.3),
        specular=Vec3(0.8, 0.8, 0.8), 
        shininess=64)
    renderer.add_mesh(torus)

    # Left sphere
    sphere_l = create_sphere(0.6, 20, 15)
    sphere_l.position = Vec3(-2.5, 0.2, -2)
    sphere_l.material = Material("sphere_red",
        diffuse=Vec3(0.8, 0.2, 0.15),
        specular=Vec3(0.7, 0.7, 0.7),
        shininess=32)
    renderer.add_mesh(sphere_l)

    # Right cube
    cube = create_cube(1.2)
    cube.position = Vec3(2.5, 0, -2)
    cube.rotation = Vec3(0.4, 0.8, 0.1)
    cube.material = Material("cube_blue",
        diffuse=Vec3(0.15, 0.3, 0.7),
        specular=Vec3(0.5, 0.5, 0.5),
        shininess=16)
    renderer.add_mesh(cube)

    # Small pyramid on the cube
    from renderer import create_pyramid
    pyramid = create_pyramid(0.8)
    pyramid.position = Vec3(2.5, 1.2, -2)
    pyramid.material = Material("pyramid_gold",
        diffuse=Vec3(0.8, 0.6, 0.1),
        specular=Vec3(0.9, 0.85, 0.5),
        shininess=80)
    renderer.add_mesh(pyramid)

    # Camera
    renderer.camera.center = Vec3(0, 0, -2)
    renderer.camera.distance = 8
    renderer.camera.elevation = 0.25
    renderer.camera.azimuth = 0.15

    print(f'Rendering {W}x{H} room scene...')
    img = renderer.render()
    path = os.path.join(os.path.dirname(__file__), '..', 'output', 'room_scene.png')
    img.save(path)
    print(f'✓ Saved: {path}')
    return path


def render_landscape_scene():
    """Render an outdoor landscape with terrain, sky, and objects."""
    W, H = 1024, 768
    renderer = Renderer(W, H)
    renderer.bg_top = Vec3(0.3, 0.5, 0.8)     # Sky blue
    renderer.bg_bottom = Vec3(0.6, 0.7, 0.9)   # Horizon

    # Sun light
    renderer.add_light(Light(Vec3(50, 30, 20), Vec3(1.0, 0.95, 0.7), 2.0))

    # Ground
    ground = create_plane(20, 10)
    ground.position = Vec3(0, -1.5, 0)
    ground.material = Material("ground", diffuse=Vec3(0.3, 0.5, 0.2))
    renderer.add_mesh(ground)

    # Trees (conical shapes approximated with spheres and pyramids)
    from renderer import create_pyramid

    tree_positions = [
        (-5, 0, -3), (-2, 0, -5), (1, 0, -4), (4, 0, -3), (6, 0, -6),
        (-6, 0, -7), (-3, 0, -8), (3, 0, -7), (7, 0, -4), (-7, 0, -2),
    ]
    for tx, _, tz in tree_positions:
        # Trunk
        trunk = create_cube(0.3)
        trunk.position = Vec3(tx, -0.5, tz)
        trunk.scale = Vec3(0.3, 1.5, 0.3)
        trunk.material = Material("trunk", diffuse=Vec3(0.4, 0.25, 0.15))
        renderer.add_mesh(trunk)

        # Foliage (pyramid)
        foliage = create_pyramid(0.8)
        foliage.position = Vec3(tx, 0.6, tz)
        foliage.scale = Vec3(1.5, 2.0, 1.5)
        foliage.material = Material("foliage", diffuse=Vec3(0.15, 0.5, 0.15))
        renderer.add_mesh(foliage)

        # Second layer
        foliage2 = create_pyramid(0.6)
        foliage2.position = Vec3(tx, 1.5, tz)
        foliage2.scale = Vec3(1.0, 1.5, 1.0)
        foliage2.material = Material("foliage", diffuse=Vec3(0.2, 0.55, 0.2))
        renderer.add_mesh(foliage2)

    # Large central torus (abstract sculpture)
    sculpture = create_torus(1.5, 0.4, 36, 20)
    sculpture.position = Vec3(0, 0, -5)
    sculpture.rotation = Vec3(0.6, 0, 0)
    sculpture.material = Material("sculpture",
        diffuse=Vec3(0.7, 0.5, 0.2),
        specular=Vec3(0.8, 0.8, 0.8),
        shininess=64)
    renderer.add_mesh(sculpture)

    # Camera — wide shot
    renderer.camera.center = Vec3(0, 0, -4)
    renderer.camera.distance = 15
    renderer.camera.elevation = 0.4
    renderer.camera.azimuth = 0.1

    print(f'Rendering {W}x{H} landscape scene...')
    img = renderer.render()
    path = os.path.join(os.path.dirname(__file__), '..', 'output', 'landscape_scene.png')
    img.save(path)
    print(f'✓ Saved: {path}')
    return path


def render_geometry_study():
    """Render a close-up geometry study: wireframe overlay + shaded primitives."""
    W, H = 1024, 768
    renderer = Renderer(W, H)
    renderer.rasterizer.wireframe = True
    renderer.rasterizer.wireframe_color = (60, 60, 100)

    # Dramatic lighting
    renderer.add_light(Light(Vec3(3, 5, 4), Vec3(1.0, 0.95, 0.85), 1.5))
    renderer.add_light(Light(Vec3(-3, 1, -2), Vec3(0.2, 0.3, 0.9), 0.5))

    # Arrange primitives in a grid
    objects = [
        ('cube', create_cube, Vec3(-3, 0, 0), Vec3(0.3, 0.5, 0)),
        ('sphere', lambda: create_sphere(0.8, 24, 18), Vec3(-1, 0, 0), Vec3(0, 0, 0)),
        ('torus', lambda: create_torus(0.7, 0.25, 24, 14), Vec3(1, 0, 0), Vec3(0.5, 0, 0)),
        ('pyramid', create_pyramid, Vec3(3, 0, 0), Vec3(0, 0.2, 0)),
    ]

    colors = [
        Vec3(0.9, 0.3, 0.2),  # Red
        Vec3(0.2, 0.6, 0.3),  # Green
        Vec3(0.2, 0.35, 0.8), # Blue
        Vec3(0.8, 0.65, 0.1), # Gold
    ]

    for i, (name, factory, pos, rot) in enumerate(objects):
        mesh = factory()
        mesh.position = pos
        mesh.rotation = rot
        mesh.material = Material(name, diffuse=colors[i], 
                                 specular=Vec3(0.6, 0.6, 0.6), shininess=32)
        renderer.add_mesh(mesh)

    renderer.camera.distance = 6
    renderer.camera.elevation = 0.3
    renderer.camera.azimuth = 0

    print(f'Rendering {W}x{H} geometry study...')
    img = renderer.render()
    path = os.path.join(os.path.dirname(__file__), '..', 'output', 'geometry_study.png')
    img.save(path)
    print(f'✓ Saved: {path}')
    return path


if __name__ == '__main__':
    render_room_scene()
    render_landscape_scene()
    render_geometry_study()
    print("\n✓ All demos rendered!")
