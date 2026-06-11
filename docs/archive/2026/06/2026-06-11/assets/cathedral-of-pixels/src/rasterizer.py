"""
Rasterizer — The Pixel Alchemist.

Converts projected triangles into pixels using:
- Back-face culling (in screen space)
- Near-plane clipping (Cohen-Sutherland in homogeneous clip space)
- Edge-function rasterization with sub-pixel precision
- Z-buffer depth testing
- Barycentric interpolation for all vertex attributes

This is where the magic of 1993 lives.
"""

import math
from typing import List, Tuple, Optional
try:
    from .linalg import Vec3, Vec4
except ImportError:
    from linalg import Vec3, Vec4


class Vertex:
    """A processed vertex with all interpolatable attributes."""
    __slots__ = ('pos', 'clip', 'normal', 'color', 'uv', 'depth')

    def __init__(self):
        self.pos = Vec3()       # Screen position (x, y, z for z-buffer)
        self.clip = Vec4()      # Clip-space position
        self.normal = Vec3()    # World/view-space normal
        self.color = Vec3(1, 1, 1)  # Vertex color
        self.uv = Vec3(0, 0, 1)     # Texture coordinates (u, v, 1/w)
        self.depth = 0.0        # Screen-space depth [0,1]


class Edge:
    """Edge function coefficients for a triangle edge.
    E(x, y) = A*x + B*y + C
    E > 0 means the point is on the "inside" of the edge.
    """
    __slots__ = ('a', 'b', 'c')

    def __init__(self, v0: Vertex, v1: Vertex):
        self.a = v0.pos.y - v1.pos.y
        self.b = v1.pos.x - v0.pos.x
        self.c = v0.pos.x * v1.pos.y - v1.pos.x * v0.pos.y

    def evaluate(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c


class Rasterizer:
    """Software rasterizer — the pixel-pushing engine."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        # Framebuffers
        self.color_buffer = None     # (height, width, 3) uint8
        self.z_buffer = None         # (height, width) float32
        self.clear_color = (18, 18, 40)  # Dark midnight blue

        # Shading mode
        self.shading_mode = 'flat'   # 'flat', 'gouraud', 'textured'
        self.texture = None          # PIL Image for texture mapping
        self.wireframe = False       # Wireframe overlay
        self.wireframe_color = (255, 255, 255)

        self._alloc_buffers()

    def _alloc_buffers(self):
        """Allocate framebuffers."""
        import numpy as np
        self.color_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.z_buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        self._clear_color_arr = np.array(self.clear_color, dtype=np.uint8)

    def clear(self):
        """Clear framebuffers for new frame."""
        import numpy as np
        self.color_buffer[:] = self._clear_color_arr
        self.z_buffer.fill(np.inf)

    def set_texture(self, texture):
        """Set texture image (PIL Image). Should be RGB."""
        import numpy as np
        self.texture = np.array(texture)
        if self.texture.ndim == 2:
            # Grayscale → RGB
            self.texture = np.stack([self.texture] * 3, axis=-1)
        self.th = self.texture.shape[0]
        self.tw = self.texture.shape[1]

    # ——— Clipping ———

    @staticmethod
    def clip_triangle_near(v0: Vertex, v1: Vertex, v2: Vertex,
                            near: float = 0.001) -> List[Tuple[Vertex, Vertex, Vertex]]:
        """Clip a triangle against the near plane (z = -w in clip space).
        Uses a simple approach: if all vertices are behind near plane, discard.
        If partially behind, clip using homogeneous interpolation.
        
        Returns list of 0-2 triangles.
        """
        # Check if triangle crosses the near plane
        # In clip space, the near plane is z = -w (after projection)
        # Actually, we check in clip space: z_clip >= -w_clip
        def is_inside(v: Vertex) -> bool:
            return v.clip.z >= -v.clip.w

        inside = [is_inside(v0), is_inside(v1), is_inside(v2)]
        count = sum(inside)

        if count == 0:
            return []  # All behind near plane

        if count == 3:
            return [(v0, v1, v2)]  # All in front

        # 1 or 2 vertices inside — clip
        # For simplicity, if any vertex is behind near plane, discard the triangle
        # (this is the 1993 approach — near-plane clipping was expensive)
        # Actually let's do a simple clip: push vertices to near plane
        verts = [v0, v1, v2]
        out_verts = []
        for i in range(3):
            v_curr = verts[i]
            v_next = verts[(i + 1) % 3]

            if inside[i]:
                out_verts.append(v_curr)

            if inside[i] != inside[(i + 1) % 3]:
                # Edge crosses near plane — interpolate
                t = (-v_curr.clip.z - v_curr.clip.w) / (
                    (v_next.clip.z + v_next.clip.w) - (v_curr.clip.z + v_curr.clip.w)
                )
                t = max(0.0, min(1.0, t))
                v_new = Vertex()
                v_new.pos = v_curr.pos.lerp(v_next.pos, t)
                v_new.color = v_curr.color.lerp(v_next.color, t)
                v_new.uv = v_curr.uv.lerp(v_next.uv, t)
                v_new.normal = v_curr.normal.lerp(v_next.normal, t)
                v_new.clip = Vec4(
                    v_curr.clip.x + (v_next.clip.x - v_curr.clip.x) * t,
                    v_curr.clip.y + (v_next.clip.y - v_curr.clip.y) * t,
                    v_curr.clip.z + (v_next.clip.z - v_curr.clip.z) * t,
                    v_curr.clip.w + (v_next.clip.w - v_curr.clip.w) * t,
                )
                out_verts.append(v_new)

        if len(out_verts) < 3:
            return []

        # Fan triangulation from first vertex
        result = []
        for i in range(1, len(out_verts) - 1):
            result.append((out_verts[0], out_verts[i], out_verts[i + 1]))
        return result

    # ——— Rasterization ———

    def rasterize_triangle(self, v0: Vertex, v1: Vertex, v2: Vertex):
        """Rasterize a single triangle using edge functions."""
        x0, y0 = v0.pos.x, v0.pos.y
        x1, y1 = v1.pos.x, v1.pos.y
        x2, y2 = v2.pos.x, v2.pos.y

        # Back-face culling in screen space
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area <= 0:
            return  # Back-facing or degenerate

        # Compute bounding box
        min_x = max(0, int(math.floor(min(x0, x1, x2))))
        max_x = min(self.width - 1, int(math.ceil(max(x0, x1, x2))))
        min_y = max(0, int(math.floor(min(y0, y1, y2))))
        max_y = min(self.height - 1, int(math.ceil(max(y0, y1, y2))))

        if min_x > max_x or min_y > max_y:
            return

        # Edge functions
        e01 = Edge(v0, v1)
        e12 = Edge(v1, v2)
        e20 = Edge(v2, v0)

        # Precompute 1/area for barycentrics
        inv_area = 1.0 / area

        # Top-left fill rule bias
        bias0 = -1e-4 if (e01.a == 0 and e01.b < 0) or (e01.a < 0) else 0
        bias1 = -1e-4 if (e12.a == 0 and e12.b < 0) or (e12.a < 0) else 0
        bias2 = -1e-4 if (e20.a == 0 and e20.b < 0) or (e20.a < 0) else 0

        for y in range(min_y, max_y + 1):
            py = y + 0.5
            # Edge function values at row start
            row_e01 = e01.a * (min_x + 0.5) + e01.b * py + e01.c + bias0
            row_e12 = e12.a * (min_x + 0.5) + e12.b * py + e12.c + bias1
            row_e20 = e20.a * (min_x + 0.5) + e20.b * py + e20.c + bias2

            for x in range(min_x, max_x + 1):
                # Inside test
                if row_e01 >= 0 and row_e12 >= 0 and row_e20 >= 0:
                    # Compute barycentrics
                    w0 = row_e12 * inv_area
                    w1 = row_e20 * inv_area
                    w2 = row_e01 * inv_area

                    # Depth test
                    z = w0 * v0.depth + w1 * v1.depth + w2 * v2.depth
                    if z < self.z_buffer[y, x]:
                        self.z_buffer[y, x] = z

                        # Shade
                        if self.wireframe:
                            # Draw wireframe on edges
                            is_edge = (
                                abs(row_e01) < 1.0 or
                                abs(row_e12) < 1.0 or
                                abs(row_e20) < 1.0
                            )
                            if is_edge:
                                self.color_buffer[y, x] = self.wireframe_color
                            else:
                                color = self._shade_pixel(v0, v1, v2, w0, w1, w2, x, y)
                                self.color_buffer[y, x] = color
                        else:
                            color = self._shade_pixel(v0, v1, v2, w0, w1, w2, x, y)
                            self.color_buffer[y, x] = color

                # Advance edge function values
                row_e01 += e01.a
                row_e12 += e12.a
                row_e20 += e20.a

    def _shade_pixel(self, v0: Vertex, v1: Vertex, v2: Vertex,
                     w0: float, w1: float, w2: float,
                     px: int, py: int) -> Tuple[int, int, int]:
        """Compute pixel color based on shading mode."""
        if self.shading_mode == 'flat':
            # Use first vertex color
            c = v0.color
            r = int(max(0, min(255, c.x * 255)))
            g = int(max(0, min(255, c.y * 255)))
            b = int(max(0, min(255, c.z * 255)))
            return (r, g, b)

        elif self.shading_mode == 'gouraud':
            # Interpolate colors using barycentrics
            c = Vec3(
                w0 * v0.color.x + w1 * v1.color.x + w2 * v2.color.x,
                w0 * v0.color.y + w1 * v1.color.y + w2 * v2.color.y,
                w0 * v0.color.z + w1 * v1.color.z + w2 * v2.color.z,
            )
            r = int(max(0, min(255, c.x * 255)))
            g = int(max(0, min(255, c.y * 255)))
            b = int(max(0, min(255, c.z * 255)))
            return (r, g, b)

        elif self.shading_mode == 'textured' and self.texture is not None:
            # Perspective-correct texture mapping
            # Interpolate u/w and v/w, then divide by 1/w
            w_inv = w0 * v0.uv.z + w1 * v1.uv.z + w2 * v2.uv.z
            if w_inv < 1e-6:
                w_inv = 1e-6
            u = (w0 * v0.uv.x * v0.uv.z + w1 * v1.uv.x * v1.uv.z + w2 * v2.uv.x * v2.uv.z) / w_inv
            v = (w0 * v0.uv.y * v0.uv.z + w1 * v1.uv.y * v1.uv.z + w2 * v2.uv.y * v2.uv.z) / w_inv

            # Sample texture
            tx = int(u * self.tw) % self.tw
            ty = int(v * self.th) % self.th
            pixel = self.texture[ty, tx]
            return (int(pixel[0]), int(pixel[1]), int(pixel[2]))

        else:
            # Fallback: white
            return (255, 255, 255)

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]):
        """Bresenham line for wireframe overlay."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < self.width and 0 <= y0 < self.height:
                self.color_buffer[y0, x0] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def draw_wireframe_triangle(self, v0: Vertex, v1: Vertex, v2: Vertex,
                                 color: Tuple[int, int, int] = None):
        """Draw wireframe triangle edges."""
        if color is None:
            color = self.wireframe_color
        self.draw_line(int(v0.pos.x), int(v0.pos.y),
                       int(v1.pos.x), int(v1.pos.y), color)
        self.draw_line(int(v1.pos.x), int(v1.pos.y),
                       int(v2.pos.x), int(v2.pos.y), color)
        self.draw_line(int(v2.pos.x), int(v2.pos.y),
                       int(v0.pos.x), int(v0.pos.y), color)

    def to_image(self):
        """Convert color buffer to PIL Image."""
        from PIL import Image
        return Image.fromarray(self.color_buffer, 'RGB')
