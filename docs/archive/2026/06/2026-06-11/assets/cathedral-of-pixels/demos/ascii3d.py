#!/usr/bin/env python3
"""
Terminal ASCII 3D — render rotating 3D objects in your terminal.
Uses the Cathedral of Pixels math engine for transforms,
then maps pixels to ASCII density characters.

Usage:
    python3 ascii3d.py [torus|cube|sphere|pyramid] [--fps 10] [--distance 2.5]
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from linalg import Vec3, Mat4
from renderer import create_cube, create_sphere, create_torus, create_pyramid


DENSITY = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


def get_mesh(name):
    if name == 'torus':
        return create_torus(1.0, 0.35, 20, 14)
    elif name == 'cube':
        return create_cube(1.2)
    elif name == 'sphere':
        return create_sphere(0.9, 16, 12)
    elif name == 'pyramid':
        return create_pyramid(1.2)
    else:
        return create_torus(1.0, 0.35, 20, 14)


def render_frame(mesh, azimuth, elevation, distance, term_w, term_h):
    """Render one ASCII frame using supersampling."""
    # Render at 4x resolution for sub-pixel quality
    scale = 4
    vw = term_w * scale
    vh = term_h * scale

    # Camera
    eye_x = distance * math.cos(elevation) * math.sin(azimuth)
    eye_y = distance * math.sin(elevation)
    eye_z = distance * math.cos(elevation) * math.cos(azimuth)
    eye = Vec3(eye_x, eye_y, eye_z)
    center = Vec3(0, 0, 0)
    up = Vec3(0, 1, 0)

    # View matrix
    f = (center - eye).normalize()
    s = f.cross(up.normalize()).normalize()
    u = s.cross(f)
    view = Mat4.identity()
    view.m[0]=s.x; view.m[4]=s.y; view.m[8]=s.z; view.m[12]=-s.dot(eye)
    view.m[1]=u.x; view.m[5]=u.y; view.m[9]=u.z; view.m[13]=-u.dot(eye)
    view.m[2]=-f.x; view.m[6]=-f.y; view.m[10]=-f.z; view.m[14]=f.dot(eye)

    # Projection
    aspect = vw / max(vh, 1)
    proj = Mat4.perspective(math.radians(60), aspect, 0.1, 50)
    vp = proj * view

    # Z-buffer and char buffer (at virtual resolution)
    zbuf = [[float('inf')] * vw for _ in range(vh)]
    charbuf = [[' '] * vw for _ in range(vh)]
    light = Vec3(0.5, 0.8, 0.3).normalize()

    for face in mesh.faces:
        vi = face[0]
        if len(vi) < 3:
            continue

        # Get world-space vertices
        wv = [mesh.vertices[i] for i in vi[:3]]
        fn = (wv[1] - wv[0]).cross(wv[2] - wv[0]).normalize()

        # Project to virtual screen
        pts = []
        for v in wv:
            ndc = vp.transform_vec3(v)
            sx = int((ndc.x + 1) * vw / 2)
            sy = int((-ndc.y + 1) * vh / 2)
            pts.append((sx, sy, ndc.z))

        # Back-face cull
        (ax, ay, _), (bx, by, _), (cx, cy, _) = pts
        if (bx-ax)*(cy-ay) - (cx-ax)*(by-ay) <= 0:
            continue

        # Lighting
        intensity = max(0.15, fn.dot(light) * 0.7 + 0.3)
        ci = min(int(intensity * (len(DENSITY) - 1)), len(DENSITY) - 1)

        # Bounding box (clamped to virtual canvas)
        min_x = max(0, min(ax, bx, cx))
        max_x = min(vw-1, max(ax, bx, cx))
        min_y = max(0, min(ay, by, cy))
        max_y = min(vh-1, max(ay, by, cy))

        # Edge-function rasterization (standard CCW formula)
        def edge(ax, ay, bx, by, x, y):
            """E_AB(x,y) = (B-A) × (P-A) in 2D. Positive when P is left of edge A→B."""
            return (bx - ax) * (y - ay) - (by - ay) * (x - ax)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                w0 = edge(bx, by, cx, cy, x, y)
                w1 = edge(cx, cy, ax, ay, x, y)
                w2 = edge(ax, ay, bx, by, x, y)

                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    area = w0 + w1 + w2
                    if area > 0:
                        z = (w0*pts[0][2] + w1*pts[1][2] + w2*pts[2][2]) / area
                        if z < zbuf[y][x]:
                            zbuf[y][x] = z
                            charbuf[y][x] = DENSITY[ci]

    # Downsample from virtual to terminal resolution
    lines = []
    for ty in range(term_h):
        row = []
        for tx in range(term_w):
            # Pick the densest character in the 4x4 block
            best_ci = 0
            for dy in range(scale):
                vy = ty * scale + dy
                for dx in range(scale):
                    vx = tx * scale + dx
                    ch = charbuf[vy][vx]
                    idx = DENSITY.find(ch)
                    if idx > best_ci:
                        best_ci = idx
            row.append(DENSITY[best_ci])
        lines.append(''.join(row))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='ASCII 3D Terminal Renderer')
    parser.add_argument('mesh', nargs='?', default='torus',
                        choices=['torus', 'cube', 'sphere', 'pyramid'])
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--distance', type=float, default=2.5)
    args = parser.parse_args()

    mesh = get_mesh(args.mesh)
    term_w = min(100, os.get_terminal_size().columns)
    term_h = min(35, os.get_terminal_size().lines - 2)
    distance = args.distance

    print(f'\x1b[?25l')  # Hide cursor
    try:
        azimuth = 0.0
        while True:
            frame_start = time.time()

            frame = render_frame(mesh, azimuth, 0.35, distance, term_w, term_h)

            # Clear and draw
            sys.stdout.write(f'\x1b[H{frame}\n\x1b[0m')
            sys.stdout.write(f'  {args.mesh.upper()} | Az: {math.degrees(azimuth):.0f}° | Ctrl+C to exit')
            sys.stdout.flush()

            azimuth += 0.05
            if azimuth > 2 * math.pi:
                azimuth -= 2 * math.pi

            # Frame rate control
            elapsed = time.time() - frame_start
            sleep_time = max(0, 1.0/args.fps - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        print(f'\x1b[?25h\x1b[2J')  # Show cursor, clear screen
        print(f"  ✦ Cathedral of Pixels — {args.mesh.upper()} — rendered ~{int(math.degrees(azimuth)/360)} rotations")


if __name__ == '__main__':
    main()
