"""
Renderer — The High Priest of the Cathedral.

Orchestrates the entire rendering pipeline:
  1. Load scene (meshes, materials, lights)
  2. Transform vertices through pipeline
  3. Apply lighting
  4. Rasterize triangles
  5. Output framebuffer
"""

import math
import random
from typing import List, Dict, Tuple, Optional
try:
    from .linalg import Vec3, Vec4, Mat4
    from .pipeline import Pipeline
    from .rasterizer import Rasterizer, Vertex
except ImportError:
    from linalg import Vec3, Vec4, Mat4
    from pipeline import Pipeline
    from rasterizer import Rasterizer, Vertex


class Light:
    """A point light source."""
    def __init__(self, position: Vec3, color: Vec3 = Vec3(1, 1, 1), intensity: float = 1.0):
        self.position = position
        self.color = color
        self.intensity = intensity


class Material:
    """Material properties."""
    def __init__(self, name: str = "default",
                 ambient: Vec3 = None, diffuse: Vec3 = None,
                 specular: Vec3 = None, shininess: float = 32.0):
        self.name = name
        self.ambient = ambient or Vec3(0.1, 0.1, 0.1)
        self.diffuse = diffuse or Vec3(0.7, 0.7, 0.7)
        self.specular = specular or Vec3(0.5, 0.5, 0.5)
        self.shininess = shininess


class Mesh:
    """A triangle mesh with vertices, normals, and optional UVs."""
    def __init__(self, name: str = "mesh"):
        self.name = name
        self.vertices: List[Vec3] = []       # Vertex positions
        self.normals: List[Vec3] = []        # Per-vertex normals (or empty for face normals)
        self.uvs: List[Tuple[float, float]] = []  # Texture coordinates
        self.faces: List[Tuple[List[int], List[int], List[int]]] = []  # (vert_indices, normal_indices, uv_indices)
        self.material: Optional[Material] = None

        # Transform
        self.position = Vec3(0, 0, 0)
        self.rotation = Vec3(0, 0, 0)
        self.scale = Vec3(1, 1, 1)

    def compute_normals(self):
        """Compute per-vertex normals by averaging face normals (smooth shading)."""
        from collections import defaultdict
        if not self.faces:
            return

        # For smooth normals, first compute face normals, then average
        face_normals = []
        for face in self.faces:
            vi = face[0]
            if len(vi) < 3:
                face_normals.append(Vec3(0, 1, 0))
                continue
            v0 = self.vertices[vi[0]]
            v1 = self.vertices[vi[1]]
            v2 = self.vertices[vi[2]]
            n = (v1 - v0).cross(v2 - v0).normalize()
            face_normals.append(n)

        # Average per vertex
        self.normals = [Vec3(0, 0, 0) for _ in range(len(self.vertices))]
        for fi, face in enumerate(self.faces):
            for vi in face[0]:
                self.normals[vi] = self.normals[vi] + face_normals[fi]

        # Normalize
        for i in range(len(self.normals)):
            self.normals[i] = self.normals[i].normalize()


class Camera:
    """Orbit camera for easy scene navigation."""
    def __init__(self):
        self._eye = Vec3(0, 0, 5)
        self.center = Vec3(0, 0, 0)
        self._up = Vec3(0, 1, 0)

        # Orbit angles (radians)
        self._azimuth = 0.0
        self._elevation = 0.3
        self._distance = 5.0

        self._update_from_orbit()

    @property
    def eye(self): return self._eye
    @property
    def up(self): return self._up

    @property
    def azimuth(self): return self._azimuth
    @azimuth.setter
    def azimuth(self, v):
        self._azimuth = v
        self._update_from_orbit()

    @property
    def elevation(self): return self._elevation
    @elevation.setter
    def elevation(self, v):
        self._elevation = v
        self._update_from_orbit()

    @property
    def distance(self): return self._distance
    @distance.setter
    def distance(self, v):
        self._distance = v
        self._update_from_orbit()

    def _update_from_orbit(self):
        """Update eye position from orbit parameters."""
        x = self._distance * math.cos(self._elevation) * math.sin(self._azimuth)
        y = self._distance * math.sin(self._elevation)
        z = self._distance * math.cos(self._elevation) * math.cos(self._azimuth)
        self._eye = self.center + Vec3(x, y, z)
        if abs(self._elevation) > math.pi / 2 - 0.01:
            self._up = Vec3(0, 0, 1 if self._elevation > 0 else -1)
        else:
            self._up = Vec3(0, 1, 0)

    def orbit(self, d_azimuth: float, d_elevation: float):
        """Adjust orbit angles."""
        self.azimuth += d_azimuth
        self.elevation = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01,
                           self._elevation + d_elevation))

    def zoom(self, factor: float):
        """Zoom in/out."""
        self.distance = max(0.5, min(50, self._distance * factor))


class Renderer:
    """The Cathedral's high priest — renders 3D scenes."""

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

        self.pipeline = Pipeline()
        self.rasterizer = Rasterizer(width, height)
        self.camera = Camera()

        # Scene
        self.meshes: List[Mesh] = []
        self.lights: List[Light] = []
        self.ambient_light = Vec3(0.15, 0.15, 0.2)

        # Background gradient colors
        self.bg_top = Vec3(0.05, 0.05, 0.15)     # Dark blue
        self.bg_bottom = Vec3(0.15, 0.1, 0.2)    # Darker purple-blue

        self._setup_pipeline()

    def _setup_pipeline(self):
        """Initialize pipeline with default view/projection."""
        aspect = self.width / self.height
        self.pipeline.set_viewport(self.width, self.height)
        self.pipeline.set_projection(math.radians(60), aspect, 0.1, 100.0)

    def add_mesh(self, mesh: Mesh):
        self.meshes.append(mesh)

    def add_light(self, light: Light):
        self.lights.append(light)

    def render(self):
        """Render one frame."""
        self.rasterizer.clear()

        # Update pipeline with camera
        self.pipeline.set_view(self.camera.eye, self.camera.center, self.camera.up)

        # Draw background gradient
        self._draw_background()

        # Pre-transform all vertices for all meshes
        for mesh in self.meshes:
            self._render_mesh(mesh)

        return self.rasterizer.to_image()

    def _draw_background(self):
        """Draw vertical gradient background."""
        import numpy as np
        for y in range(self.height):
            t = y / self.height
            r = int((self.bg_bottom.x + (self.bg_top.x - self.bg_bottom.x) * t) * 255)
            g = int((self.bg_bottom.y + (self.bg_top.y - self.bg_bottom.y) * t) * 255)
            b = int((self.bg_bottom.z + (self.bg_top.z - self.bg_bottom.z) * t) * 255)
            self.rasterizer.color_buffer[y, :] = [r, g, b]

    def _render_mesh(self, mesh: Mesh):
        """Render a single mesh."""
        # Set up model transform
        self.pipeline.set_model(
            translation=mesh.position,
            rotation=mesh.rotation,
            scale=mesh.scale,
        )

        # Pre-transform all vertices
        screen_vertices = []
        screen_normals = []
        for i, v in enumerate(mesh.vertices):
            sx, sy, depth = self.pipeline.transform_to_screen(v)
            screen_vertices.append((sx, sy, depth))

            # Transform normals if available
            if mesh.normals and i < len(mesh.normals):
                n = self.pipeline.transform_normal(mesh.normals[i])
                screen_normals.append(n)
            else:
                screen_normals.append(Vec3(0, 0, 0))

        # Compute face normals for flat shading
        face_normals = []
        for face in mesh.faces:
            vi = face[0]
            if len(vi) >= 3:
                v0 = mesh.vertices[vi[0]]
                v1 = mesh.vertices[vi[1]]
                v2 = mesh.vertices[vi[2]]
                fn = (v1 - v0).cross(v2 - v0).normalize()
                # Transform normal
                fn = self.pipeline.transform_normal(fn)
                face_normals.append(fn)
            else:
                face_normals.append(Vec3(0, 1, 0))

        # Set shading mode
        use_gouraud = len(mesh.normals) > 0 and len(mesh.normals) == len(mesh.vertices)
        use_flat = not use_gouraud

        # Render each face
        for fi, face in enumerate(mesh.faces):
            vi = face[0]
            ni = face[1] if len(face) > 1 else []
            ti = face[2] if len(face) > 2 else []

            if len(vi) < 3:
                continue

            # Build vertices
            verts = []
            for j in range(len(vi)):
                vtx = Vertex()
                vtx.pos = Vec3(*screen_vertices[vi[j]])

                # Lighting
                if use_gouraud and ni and j < len(ni) and ni[j] < len(screen_normals):
                    normal = screen_normals[ni[j]]
                elif use_flat:
                    normal = face_normals[fi]
                else:
                    normal = Vec3(0, 1, 0)

                vtx.color = self._compute_lighting(
                    mesh.vertices[vi[j]], normal, mesh.material
                )

                # UV
                if ti and j < len(ti) and ti[j] < len(mesh.uvs):
                    u, v = mesh.uvs[ti[j]]
                    # Store 1/w for perspective correction
                    vtx.uv = Vec3(u, v, 1.0 / max(vtx.pos.z, 0.001))
                else:
                    vtx.uv = Vec3(0, 0, 1.0)

                vtx.depth = screen_vertices[vi[j]][2]
                verts.append(vtx)

            # Triangulate face (fan triangulation for quads+)
            for j in range(1, len(verts) - 1):
                v0, v1, v2 = verts[0], verts[j], verts[j + 1]

                # Check if triangle is completely off-screen
                x_coords = [v0.pos.x, v1.pos.x, v2.pos.x]
                y_coords = [v0.pos.y, v1.pos.y, v2.pos.y]
                if (max(x_coords) < 0 or min(x_coords) >= self.width or
                    max(y_coords) < 0 or min(y_coords) >= self.height):
                    continue

                self.rasterizer.draw_wireframe_triangle(v0, v1, v2, (80, 80, 120))
                self.rasterizer.rasterize_triangle(v0, v1, v2)

    def _compute_lighting(self, world_pos: Vec3, normal: Vec3,
                           material: Optional[Material] = None) -> Vec3:
        """Compute Phong-style lighting for a vertex."""
        if material is None:
            mat = Material()
        else:
            mat = material

        # Start with ambient
        color = Vec3(
            mat.ambient.x * self.ambient_light.x,
            mat.ambient.y * self.ambient_light.y,
            mat.ambient.z * self.ambient_light.z,
        )

        # Accumulate from each light
        for light in self.lights:
            # Light direction
            light_dir = (light.position - world_pos).normalize()
            light_dist = (light.position - world_pos).length()

            # Attenuation
            attenuation = light.intensity / (1.0 + light_dist * light_dist * 0.01)

            # Diffuse (Lambertian)
            n_dot_l = normal.dot(light_dir)
            if n_dot_l > 0:
                diffuse = Vec3(
                    mat.diffuse.x * light.color.x * n_dot_l * attenuation,
                    mat.diffuse.y * light.color.y * n_dot_l * attenuation,
                    mat.diffuse.z * light.color.z * n_dot_l * attenuation,
                )
                color = color + diffuse

                # Specular (Blinn-Phong)
                view_dir = (self.camera.eye - world_pos).normalize()
                half_vec = (light_dir + view_dir).normalize()
                n_dot_h = normal.dot(half_vec)
                if n_dot_h > 0:
                    spec = math.pow(n_dot_h, mat.shininess)
                    specular = Vec3(
                        mat.specular.x * light.color.x * spec * attenuation,
                        mat.specular.y * light.color.y * spec * attenuation,
                        mat.specular.z * light.color.z * spec * attenuation,
                    )
                    color = color + specular

        # Clamp
        color.x = min(1.0, max(0.05, color.x))
        color.y = min(1.0, max(0.05, color.y))
        color.z = min(1.0, max(0.05, color.z))

        return color


# ——— Primitive Mesh Generators ———

def create_cube(size: float = 1.0) -> Mesh:
    """Generate a unit cube centered at origin."""
    mesh = Mesh("cube")
    h = size / 2

    # 8 vertices
    verts = [
        Vec3(-h, -h, -h), Vec3( h, -h, -h), Vec3( h,  h, -h), Vec3(-h,  h, -h),
        Vec3(-h, -h,  h), Vec3( h, -h,  h), Vec3( h,  h,  h), Vec3(-h,  h,  h),
    ]
    mesh.vertices = verts

    # 6 faces (2 triangles each), with normals
    faces = [
        # Front (z=-h)
        ([0, 1, 2, 3], [], []),
        # Back (z=+h)
        ([5, 4, 7, 6], [], []),
        # Left (x=-h)
        ([4, 0, 3, 7], [], []),
        # Right (x=+h)
        ([1, 5, 6, 2], [], []),
        # Bottom (y=-h)
        ([4, 5, 1, 0], [], []),
        # Top (y=+h)
        ([3, 2, 6, 7], [], []),
    ]
    mesh.faces = faces
    mesh.compute_normals()
    return mesh


def create_sphere(radius: float = 1.0, segments: int = 16, rings: int = 12) -> Mesh:
    """Generate a UV sphere."""
    mesh = Mesh("sphere")
    
    # Generate vertices
    for r in range(rings + 1):
        phi = math.pi * r / rings  # 0 to pi (top to bottom)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        
        for s in range(segments + 1):
            theta = 2 * math.pi * s / segments
            x = radius * sin_phi * math.cos(theta)
            y = radius * cos_phi
            z = radius * sin_phi * math.sin(theta)
            mesh.vertices.append(Vec3(x, y, z))
            mesh.uvs.append((s / segments, r / rings))

    # Generate faces
    for r in range(rings):
        for s in range(segments):
            a = r * (segments + 1) + s
            b = a + segments + 1
            c = a + 1
            d = b + 1
            mesh.faces.append((
                [a, b, c], [], [a, b, c]
            ))
            mesh.faces.append((
                [c, b, d], [], [c, b, d]
            ))

    mesh.compute_normals()
    return mesh


def create_plane(size: float = 10.0, divisions: int = 10) -> Mesh:
    """Generate a flat ground plane."""
    mesh = Mesh("plane")
    half = size / 2
    step = size / divisions

    # Grid vertices
    for i in range(divisions + 1):
        for j in range(divisions + 1):
            x = -half + j * step
            z = -half + i * step
            mesh.vertices.append(Vec3(x, 0, z))
            mesh.uvs.append((j / divisions, i / divisions))

    # Grid faces
    for i in range(divisions):
        for j in range(divisions):
            a = i * (divisions + 1) + j
            b = a + divisions + 1
            c = a + 1
            d = b + 1
            mesh.faces.append(([a, b, c], [], [a, b, c]))
            mesh.faces.append(([c, b, d], [], [c, b, d]))

    mesh.compute_normals()
    return mesh


def create_pyramid(size: float = 1.0) -> Mesh:
    """Generate a square pyramid."""
    mesh = Mesh("pyramid")
    h = size / 2

    mesh.vertices = [
        Vec3(-h, -h, -h), Vec3( h, -h, -h), Vec3( h, -h,  h), Vec3(-h, -h,  h),  # Base
        Vec3(0, h, 0),  # Apex
    ]

    # Base (2 triangles)
    mesh.faces.append(([0, 2, 1], [], []))
    mesh.faces.append(([0, 3, 2], [], []))
    # Sides
    mesh.faces.append(([0, 1, 4], [], []))
    mesh.faces.append(([1, 2, 4], [], []))
    mesh.faces.append(([2, 3, 4], [], []))
    mesh.faces.append(([3, 0, 4], [], []))

    mesh.compute_normals()
    return mesh


def create_torus(major_r: float = 1.0, minor_r: float = 0.3,
                 major_segments: int = 24, minor_segments: int = 12) -> Mesh:
    """Generate a torus (donut)."""
    mesh = Mesh("torus")
    
    for i in range(major_segments):
        theta = 2 * math.pi * i / major_segments
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            
            r = major_r + minor_r * cos_phi
            x = r * cos_theta
            y = minor_r * sin_phi
            z = r * sin_theta
            
            mesh.vertices.append(Vec3(x, y, z))
            mesh.uvs.append((i / major_segments, j / minor_segments))

    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = ((i + 1) % major_segments) * minor_segments + j
            c = i * minor_segments + (j + 1) % minor_segments
            d = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            mesh.faces.append(([a, b, c], [], [a, b, c]))
            mesh.faces.append(([c, b, d], [], [c, b, d]))

    mesh.compute_normals()
    return mesh
