"""
Transformation Pipeline — Model → World → View → Clip → NDC → Screen.

The sacred journey of every vertex through the graphics pipeline:
  1. Local space → World space (model matrix)
  2. World space → View space (view matrix)
  3. View space → Clip space (projection matrix)
  4. Clip space → NDC (perspective divide)
  5. NDC → Screen space (viewport transform)
"""

from typing import List, Tuple

try:
    from .linalg import Vec3, Vec4, Mat4
except ImportError:
    from linalg import Vec3, Vec4, Mat4


class Pipeline:
    """Manages the transformation chain from model to screen."""

    def __init__(self):
        self.model = Mat4.identity()
        self.view = Mat4.identity()
        self.projection = Mat4.identity()
        self.viewport_matrix = Mat4.identity()

        # Combined matrices (computed on demand)
        self._mv = None
        self._mvp = None
        self._dirty = True

        # Viewport dimensions
        self.viewport_w = 800
        self.viewport_h = 600

    def set_model(self, translation: Vec3 = None, rotation: Vec3 = None, scale: Vec3 = None):
        """Set model transform from translation (Vec3), rotation (Vec3 of Euler angles in radians), scale (Vec3)."""
        m = Mat4.identity()
        if translation:
            m = Mat4.translation(translation.x, translation.y, translation.z) * m
        if rotation:
            m = Mat4.rotation_z(rotation.z) * Mat4.rotation_y(rotation.y) * Mat4.rotation_x(rotation.x) * m
        if scale:
            m = Mat4.scale(scale.x, scale.y, scale.z) * m
        self.model = m
        self._dirty = True

    def set_view(self, eye: Vec3, center: Vec3, up: Vec3 = None):
        """Set view (camera) matrix."""
        if up is None:
            up = Vec3(0, 1, 0)
        self.view = Mat4.look_at(eye, center, up)
        self._dirty = True

    def set_projection(self, fov_y: float, aspect: float, near: float, far: float):
        """Set perspective projection matrix."""
        self.projection = Mat4.perspective(fov_y, aspect, near, far)
        self._dirty = True

    def set_viewport(self, w: int, h: int):
        """Set viewport dimensions for NDC→screen transform."""
        self.viewport_w = w
        self.viewport_h = h
        self._dirty = True

    def _update_matrices(self):
        """Recompute combined matrices if dirty."""
        if not self._dirty:
            return
        self._mv = self.view * self.model
        self._mvp = self.projection * self._mv

        # Viewport matrix: NDC [-1,1] → Screen [0, w] x [0, h]
        hw = self.viewport_w * 0.5
        hh = self.viewport_h * 0.5
        vp = Mat4.identity()
        vp.m[0] = hw
        vp.m[5] = -hh    # Flip Y
        vp.m[10] = 1.0
        vp.m[12] = hw     # X offset
        vp.m[13] = hh     # Y offset
        self.viewport_matrix = vp
        self._dirty = False

    def transform_vertex(self, v: Vec3) -> Vec4:
        """Transform a vertex through the full pipeline: model → view → projection.
        Returns clip-space Vec4 (before perspective divide).
        """
        self._update_matrices()
        return self._mvp.transform_vec4(Vec4(v.x, v.y, v.z, 1.0))

    def transform_to_screen(self, v: Vec3) -> Tuple[float, float, float]:
        """Full pipeline: model → world → view → projection → perspective divide → viewport.
        Returns (screen_x, screen_y, depth) where depth is in [0,1].
        """
        self._update_matrices()
        clip = self._mvp.transform_vec4(Vec4(v.x, v.y, v.z, 1.0))
        ndc = clip.to_vec3()
        screen = self.viewport_matrix.transform_vec3(ndc)
        # Depth: [-1, 1] → [0, 1] (actually post-projection z is in [-1,1])
        depth = ndc.z * 0.5 + 0.5
        return (screen.x, screen.y, depth)

    def transform_normal(self, n: Vec3) -> Vec3:
        """Transform a normal vector using the inverse-transpose of the model-view matrix."""
        self._update_matrices()
        it = self._mv.inverse()
        # Transform with inverse-transpose (row vector), then transpose back
        x = n.x; y = n.y; z = n.z
        return Vec3(
            it.m[0] * x + it.m[1] * y + it.m[2] * z,
            it.m[4] * x + it.m[5] * y + it.m[6] * z,
            it.m[8] * x + it.m[9] * y + it.m[10] * z,
        ).normalize()

    @property
    def model_view(self) -> Mat4:
        self._update_matrices()
        return self._mv

    @property
    def mvp(self) -> Mat4:
        self._update_matrices()
        return self._mvp
