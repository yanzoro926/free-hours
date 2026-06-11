"""
Linear algebra foundation — the cathedral's cornerstone.
Vec3, Vec4, Mat4: pure Python implementations.
No NumPy in the core rendering path — we want to see every multiply.
"""

import math
from typing import Tuple, List


class Vec3:
    """3D vector — the soul of every vertex."""
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vec3':
        inv = 1.0 / scalar
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def __neg__(self) -> 'Vec3':
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: 'Vec3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vec3') -> 'Vec3':
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vec3':
        length = self.length()
        if length < 1e-10:
            return Vec3(0, 0, 0)
        return self / length

    def lerp(self, other: 'Vec3', t: float) -> 'Vec3':
        return Vec3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t,
        )

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Vec4:
    """4D homogeneous vector — for perspective division."""
    __slots__ = ('x', 'y', 'z', 'w')

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, w: float = 1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    @staticmethod
    def from_vec3(v: Vec3, w: float = 1) -> 'Vec4':
        return Vec4(v.x, v.y, v.z, w)

    def to_vec3(self) -> Vec3:
        """Perspective divide — homogeneous to Cartesian."""
        if abs(self.w) < 1e-10:
            return Vec3(self.x, self.y, self.z)
        inv = 1.0 / self.w
        return Vec3(self.x * inv, self.y * inv, self.z * inv)


class Mat4:
    """4x4 transformation matrix — column-major storage.
    
    Layout (column-major):
    m[0]  m[4]  m[8]  m[12]   ← column 0
    m[1]  m[5]  m[9]  m[13]   ← column 1
    m[2]  m[6]  m[10] m[14]   ← column 2
    m[3]  m[7]  m[11] m[15]   ← column 3
    
    This matches OpenGL convention for easy mental porting.
    """

    def __init__(self):
        self.m = [0.0] * 16

    @staticmethod
    def identity() -> 'Mat4':
        m = Mat4()
        m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0
        return m

    @staticmethod
    def translation(x: float, y: float, z: float) -> 'Mat4':
        m = Mat4.identity()
        m.m[12] = x
        m.m[13] = y
        m.m[14] = z
        return m

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> 'Mat4':
        m = Mat4()
        m.m[0] = sx
        m.m[5] = sy
        m.m[10] = sz
        m.m[15] = 1.0
        return m

    @staticmethod
    def rotation_x(angle: float) -> 'Mat4':
        """Rotation around X axis (in radians)."""
        c = math.cos(angle)
        s = math.sin(angle)
        m = Mat4()
        m.m[0] = 1.0
        m.m[5] = c
        m.m[6] = s
        m.m[9] = -s
        m.m[10] = c
        m.m[15] = 1.0
        return m

    @staticmethod
    def rotation_y(angle: float) -> 'Mat4':
        c = math.cos(angle)
        s = math.sin(angle)
        m = Mat4()
        m.m[0] = c
        m.m[2] = -s
        m.m[5] = 1.0
        m.m[8] = s
        m.m[10] = c
        m.m[15] = 1.0
        return m

    @staticmethod
    def rotation_z(angle: float) -> 'Mat4':
        c = math.cos(angle)
        s = math.sin(angle)
        m = Mat4()
        m.m[0] = c
        m.m[1] = s
        m.m[4] = -s
        m.m[5] = c
        m.m[10] = 1.0
        m.m[15] = 1.0
        return m

    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> 'Mat4':
        """Perspective projection matrix.
        
        fov_y: vertical field of view in radians
        aspect: width / height
        near, far: clipping planes
        """
        f = 1.0 / math.tan(fov_y / 2.0)
        m = Mat4()
        m.m[0] = f / aspect
        m.m[5] = f
        m.m[10] = (far + near) / (near - far)
        m.m[11] = -1.0
        m.m[14] = (2.0 * far * near) / (near - far)
        return m

    @staticmethod
    def look_at(eye: Vec3, center: Vec3, up: Vec3) -> 'Mat4':
        """View matrix — camera look-at."""
        f = (center - eye).normalize()
        s = f.cross(up.normalize()).normalize()
        u = s.cross(f)

        m = Mat4()
        m.m[0] = s.x
        m.m[4] = s.y
        m.m[8] = s.z
        m.m[1] = u.x
        m.m[5] = u.y
        m.m[9] = u.z
        m.m[2] = -f.x
        m.m[6] = -f.y
        m.m[10] = -f.z
        m.m[12] = -s.dot(eye)
        m.m[13] = -u.dot(eye)
        m.m[14] = f.dot(eye)
        m.m[15] = 1.0
        return m

    def __mul__(self, other: 'Mat4') -> 'Mat4':
        """Matrix multiplication: self * other."""
        result = Mat4()
        a = self.m
        b = other.m
        r = result.m
        for col in range(4):
            for row in range(4):
                idx = col * 4 + row
                r[idx] = (
                    a[row] * b[col * 4] +
                    a[row + 4] * b[col * 4 + 1] +
                    a[row + 8] * b[col * 4 + 2] +
                    a[row + 12] * b[col * 4 + 3]
                )
        return result

    def transform_vec4(self, v: Vec4) -> Vec4:
        """Transform a Vec4 by this matrix."""
        x = v.x
        y = v.y
        z = v.z
        w = v.w
        return Vec4(
            self.m[0] * x + self.m[4] * y + self.m[8] * z + self.m[12] * w,
            self.m[1] * x + self.m[5] * y + self.m[9] * z + self.m[13] * w,
            self.m[2] * x + self.m[6] * y + self.m[10] * z + self.m[14] * w,
            self.m[3] * x + self.m[7] * y + self.m[11] * z + self.m[15] * w,
        )

    def transform_vec3(self, v: Vec3) -> Vec3:
        """Transform a Vec3 as a position (w=1)."""
        v4 = self.transform_vec4(Vec4(v.x, v.y, v.z, 1.0))
        return v4.to_vec3()

    def transform_direction(self, v: Vec3) -> Vec3:
        """Transform a Vec3 as a direction (w=0, no translation)."""
        v4 = self.transform_vec4(Vec4(v.x, v.y, v.z, 0.0))
        return v4.to_vec3()

    def inverse(self) -> 'Mat4':
        """Compute inverse using Gaussian elimination with partial pivoting."""
        # Augmented matrix [A | I] in row-major form for elimination
        # Convert from column-major to row-major for easier row ops
        m = self.m
        aug = [
            m[0], m[4], m[8],  m[12], 1.0, 0.0, 0.0, 0.0,
            m[1], m[5], m[9],  m[13], 0.0, 1.0, 0.0, 0.0,
            m[2], m[6], m[10], m[14], 0.0, 0.0, 1.0, 0.0,
            m[3], m[7], m[11], m[15], 0.0, 0.0, 0.0, 1.0,
        ]

        def get(r, c):
            return aug[r * 8 + c]

        def set_val(r, c, v):
            aug[r * 8 + c] = v

        # Forward elimination
        for col in range(4):
            # Find pivot
            pivot_row = col
            max_val = abs(get(col, col))
            for row in range(col + 1, 4):
                val = abs(get(row, col))
                if val > max_val:
                    max_val = val
                    pivot_row = row

            if max_val < 1e-12:
                raise ValueError("Matrix is singular")

            # Swap rows
            if pivot_row != col:
                for c in range(8):
                    tmp = get(col, c)
                    set_val(col, c, get(pivot_row, c))
                    set_val(pivot_row, c, tmp)

            # Scale pivot row
            pivot = get(col, col)
            for c in range(8):
                set_val(col, c, get(col, c) / pivot)

            # Eliminate other rows
            for row in range(4):
                if row == col:
                    continue
                factor = get(row, col)
                for c in range(8):
                    set_val(row, c, get(row, c) - factor * get(col, c))

        # Extract right half and convert back to column-major
        result = Mat4()
        for r in range(4):
            for c in range(4):
                # Right half of augmented is columns 4-7
                # Store as column-major: result[c*4 + r] = aug[r*8 + (c+4)]
                result.m[c * 4 + r] = aug[r * 8 + (c + 4)]

        return result

    def __repr__(self):
        lines = []
        for row in range(4):
            vals = [f"{self.m[col * 4 + row]:8.4f}" for col in range(4)]
            lines.append("[" + " ".join(vals) + "]")
        return "Mat4(\n  " + "\n  ".join(lines) + "\n)"
