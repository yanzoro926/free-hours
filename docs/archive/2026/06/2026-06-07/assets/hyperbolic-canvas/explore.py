"""
Curvature of Thought: Computational Exploration of Hyperbolic Geometry
======================================================================

This script explores hyperbolic geometry in the Poincaré disk model
through computation and visualization. We investigate:

1. Distance distortion — how hyperbolic distance maps to Euclidean
2. Triangle angle defect — proof that hyperbolic triangles sum to < 180°
3. Parallel lines — infinite non-intersecting geodesics through a point
4. Tessellation viability — which {p,q} pairs work in hyperbolic space
5. Circle area growth — exponential circumference growth with radius

All figures are saved as dark-themed PNGs for the report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

DARK_BG = '#0a0a14'
DARK_SURFACE = '#12122a'
DARK_TEXT = '#e0e0f0'
DARK_BORDER = '#2a2a44'
ACCENT = '#c4a0ff'
ACCENT2 = '#60d0ff'
GOLD = '#f0c060'
DANGER = '#ff6070'


def dark_style(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.figure.patch.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color(DARK_BORDER)
        spine.set_linewidth(0.5)
    ax.tick_params(colors=DARK_TEXT, labelsize=8)
    ax.xaxis.label.set_color(DARK_TEXT)
    ax.yaxis.label.set_color(DARK_TEXT)
    ax.title.set_color(DARK_TEXT)
    if title:
        ax.set_title(title, color=DARK_TEXT, fontweight='bold', pad=10)


# =================================================================
# CORE GEOMETRY FUNCTIONS
# =================================================================

def hyperbolic_dist(p, q):
    """Poincaré distance between two points in the unit disk."""
    num = (p[0] - q[0])**2 + (p[1] - q[1])**2
    denom = (1 - p[0]**2 - p[1]**2) * (1 - q[0]**2 - q[1]**2)
    if denom <= 0:
        return np.inf
    arg = 1 + 2 * num / denom
    return np.arccosh(max(1, arg))


def euclidean_to_hyperbolic_radius(r_euclidean):
    """Convert Euclidean radius from origin to hyperbolic distance."""
    if r_euclidean >= 1:
        return np.inf
    return 2 * np.arctanh(r_euclidean)


def hyperbolic_to_euclidean_radius(d_hyperbolic):
    """Convert hyperbolic distance from origin to Euclidean radius."""
    return np.tanh(d_hyperbolic / 2)


def compute_geodesic_center(p, q):
    """Compute center and radius of geodesic arc through p and q."""
    pn2 = p[0]**2 + p[1]**2
    qn2 = q[0]**2 + q[1]**2
    cross = p[0]*q[1] - p[1]*q[0]
    if abs(cross) < 1e-10:
        return None, None  # Through origin, straight line

    denom = 2 * cross
    cx = (q[1] * (pn2 + 1) - p[1] * (qn2 + 1)) / denom
    cy = (p[0] * (qn2 + 1) - q[0] * (pn2 + 1)) / denom
    r = np.sqrt(cx**2 + cy**2 - 1)
    if r < 0 or not np.isfinite(r):
        return None, None
    return (cx, cy), r


def triangle_angles(a, b, c):
    """Compute interior angles of hyperbolic triangle using law of cosines."""
    d_ab = hyperbolic_dist(a, b)
    d_bc = hyperbolic_dist(b, c)
    d_ca = hyperbolic_dist(c, a)

    cosA = (np.cosh(d_ab) * np.cosh(d_ca) - np.cosh(d_bc)) / \
           (np.sinh(d_ab) * np.sinh(d_ca))
    cosB = (np.cosh(d_ab) * np.cosh(d_bc) - np.cosh(d_ca)) / \
           (np.sinh(d_ab) * np.sinh(d_bc))
    cosC = (np.cosh(d_bc) * np.cosh(d_ca) - np.cosh(d_ab)) / \
           (np.sinh(d_bc) * np.sinh(d_ca))

    angleA = np.arccos(np.clip(cosA, -1, 1))
    angleB = np.arccos(np.clip(cosB, -1, 1))
    angleC = np.arccos(np.clip(cosC, -1, 1))
    return angleA, angleB, angleC


# =================================================================
# FIGURE 1: DISTANCE DISTORTION
# =================================================================

def plot_distance_distortion(savepath):
    """Show how hyperbolic distance grows as points approach the boundary."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: hyperbolic distance from origin vs Euclidean radius
    ax1 = axes[0]
    dark_style(ax1, "Distance from Origin: Hyperbolic vs Euclidean")

    r_euclidean = np.linspace(0, 0.999, 500)
    d_hyperbolic = [euclidean_to_hyperbolic_radius(r) for r in r_euclidean]

    ax1.plot(r_euclidean, d_hyperbolic, color=ACCENT2, linewidth=2)
    ax1.plot(r_euclidean, r_euclidean, color='#555', linewidth=1, linestyle='--',
             label='Euclidean (y=x)')
    ax1.axhline(y=1, color=GOLD, linewidth=0.8, linestyle=':', alpha=0.5,
                label='h-dist = 1')
    ax1.axhline(y=3, color=DANGER, linewidth=0.8, linestyle=':', alpha=0.5,
                label='h-dist = 3')
    ax1.set_xlabel('Euclidean Distance from Origin')
    ax1.set_ylabel('Hyperbolic Distance')
    ax1.legend(facecolor=DARK_SURFACE, edgecolor=DARK_BORDER,
               labelcolor=DARK_TEXT, fontsize=8)
    ax1.grid(True, alpha=0.1, color='white')

    # Annotate key points
    for r_target in [0.5, 0.8, 0.95, 0.99]:
        hd = euclidean_to_hyperbolic_radius(r_target)
        ax1.annotate(f'r={r_target}\nh={hd:.2f}',
                    xy=(r_target, hd), xytext=(r_target + 0.08, hd + 0.2),
                    color=ACCENT, fontsize=7,
                    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8))

    # Right: how many unit-disk radii correspond to Euclidean radii
    ax2 = axes[1]
    dark_style(ax2, "Exponential Growth: Euclidean Position vs Hyperbolic Distance")

    d_range = np.linspace(0, 6, 500)
    r_range = [hyperbolic_to_euclidean_radius(d) for d in d_range]

    ax2.plot(d_range, r_range, color=ACCENT2, linewidth=2)
    ax2.fill_between(d_range, r_range, alpha=0.1, color=ACCENT2)

    # Mark the "crowding" points
    for hd_label in [1, 2, 3, 5]:
        r_pos = hyperbolic_to_euclidean_radius(hd_label)
        ax2.plot(hd_label, r_pos, 'o', color=GOLD, markersize=6)
        ax2.annotate(f'r={r_pos:.3f}',
                    xy=(hd_label, r_pos), xytext=(hd_label + 0.3, r_pos + 0.03),
                    color=GOLD, fontsize=8)

    ax2.set_xlabel('Hyperbolic Distance from Origin')
    ax2.set_ylabel('Euclidean Distance from Origin')
    ax2.grid(True, alpha=0.1, color='white')

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved distance distortion → {savepath}")


# =================================================================
# FIGURE 2: TRIANGLE ANGLE DEFECT
# =================================================================

def plot_triangle_defect(savepath):
    """Demonstrate that hyperbolic triangles have angle sum < π."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Sample many random triangles at different sizes
    rng = np.random.default_rng(42)
    sizes = []
    defects = []

    for _ in range(2000):
        # Generate three random points in the disk at varying radii
        r = rng.uniform(0.05, 0.95)
        angles = rng.uniform(0, 2*np.pi, 3)
        a = (r * np.cos(angles[0]), r * np.sin(angles[0]))
        b = (r * np.cos(angles[1]), r * np.sin(angles[1]))
        c = (r * np.cos(angles[2]), r * np.sin(angles[2]))

        A, B, C = triangle_angles(a, b, c)
        perimeter = hyperbolic_dist(a, b) + hyperbolic_dist(b, c) + hyperbolic_dist(c, a)
        defect = np.pi - (A + B + C)
        sizes.append(perimeter)
        defects.append(defect)

    # Left: angle sum distribution
    ax1 = axes[0]
    dark_style(ax1, "Triangle Angle Sum Distribution")
    sum_angles = [np.pi - d for d in defects]
    sum_degrees = [s * 180/np.pi for s in sum_angles]

    ax1.hist(sum_degrees, bins=60, color=ACCENT2, alpha=0.7,
             edgecolor=DARK_BORDER, linewidth=0.3)
    ax1.axvline(x=180, color=DANGER, linewidth=1.5, linestyle='--',
                label='Euclidean: 180°')
    ax1.axvline(x=np.mean(sum_degrees), color=GOLD, linewidth=1.5, linestyle='-',
                label=f'Mean: {np.mean(sum_degrees):.1f}°')
    ax1.set_xlabel('Angle Sum (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.legend(facecolor=DARK_SURFACE, edgecolor=DARK_BORDER,
               labelcolor=DARK_TEXT, fontsize=7)

    # Center: defect vs perimeter
    ax2 = axes[1]
    dark_style(ax2, "Angle Defect vs Perimeter")
    sc = ax2.scatter(sizes, [d*180/np.pi for d in defects],
                     c=sizes, cmap='plasma', alpha=0.5, s=3)
    ax2.set_xlabel('Perimeter (hyperbolic)')
    ax2.set_ylabel('Angle Defect (degrees)')
    ax2.grid(True, alpha=0.1, color='white')

    # Right: specific example triangles
    ax3 = axes[2]
    dark_style(ax3, "Specific Triangles")
    examples = [
        (0.2, "tiny"),
        (0.5, "medium"),
        (0.8, "large"),
        (0.95, "near-boundary"),
    ]
    example_defects = []
    example_labels = []
    for r, label in examples:
        angles = [0, 2*np.pi/3, 4*np.pi/3]
        a = (r * np.cos(angles[0]), r * np.sin(angles[0]))
        b = (r * np.cos(angles[1]), r * np.sin(angles[1]))
        c = (r * np.cos(angles[2]), r * np.sin(angles[2]))
        A, B, C = triangle_angles(a, b, c)
        defect = (np.pi - (A + B + C)) * 180 / np.pi
        example_defects.append(defect)
        example_labels.append(f"{label}\n(r={r})")

    colors = [ACCENT2, '#a0d0ff', GOLD, DANGER]
    bars = ax3.bar(example_labels, example_defects, color=colors, alpha=0.8)
    ax3.set_ylabel('Angle Defect (degrees)')
    ax3.axhline(y=0, color='#555', linewidth=0.5)
    for bar, val in zip(bars, example_defects):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}°', ha='center', va='bottom', color=DARK_TEXT, fontsize=8)

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved triangle defect → {savepath}")


# =================================================================
# FIGURE 3: PARALLEL LINES — THE BREAK WITH EUCLID
# =================================================================

def plot_parallel_lines(savepath):
    """Illustrate that infinitely many lines through a point never meet a given line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    dark_style(ax, "Parallel Lines in Hyperbolic Space\n(infinitely many non-intersecting geodesics through P)")

    # Draw the disk
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color=DARK_BORDER, linewidth=1.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Base line: geodesic through (-0.7, 0) and (0.7, 0) — a straight line through origin
    ax.plot([-0.95, 0.95], [0, 0], color=ACCENT2, linewidth=2, label='Base geodesic L')

    # Point P
    P = (0.3, 0.45)
    ax.plot(P[0], P[1], 'o', color=GOLD, markersize=10, label='Point P')
    ax.annotate('P', xy=P, xytext=(P[0]+0.05, P[1]+0.05), color=GOLD, fontsize=10, fontweight='bold')

    # Generate parallel geodesics through P
    # A geodesic through P is determined by its direction at P
    # We want to find directions that produce geodesics that DON'T intersect L
    # In hyperbolic geometry: if the geodesic through P heads "toward" L, it will intersect
    # The boundary between intersecting and non-intersecting is the "limiting parallel"

    # For a line through origin (the x-axis), the condition for a geodesic through P
    # to intersect the x-axis depends on the geodesic's circle center position

    base_colors = [ACCENT, '#a080ff', '#8060ff', '#6040ff', '#4020ff',
                   DANGER, '#ff4060', '#ff2040', '#ff0030']

    # Generate several geodesics through P
    # For each, compute the geodesic arc and draw it
    directions = np.linspace(-np.pi, np.pi, 20)

    for i, direction in enumerate(directions):
        if i % 2 != 0:
            continue  # Skip half for clarity

        # A point Q in the given direction from P
        Q = (P[0] + 0.3 * np.cos(direction), P[1] + 0.3 * np.sin(direction))

        center, radius = compute_geodesic_center(P, Q)

        if center is None:
            # Straight line through origin
            t_max = 1.0
            px_interp = P[0] + (Q[0] - P[0]) * 2
            py_interp = P[1] + (Q[1] - P[1]) * 2
            ax.plot([P[0]-1, P[0]+1], [P[1]-(P[1]/P[0])*(P[0]-1), P[1]+(P[1]/P[0])*(P[0]+1)],
                   color='#444', linewidth=0.6, alpha=0.3)
            continue

        # Draw the geodesic arc
        angP = np.arctan2(P[1] - center[1], P[0] - center[0])
        angQ = np.arctan2(Q[1] - center[1], Q[0] - center[0])
        if angQ < angP:
            angQ += 2*np.pi
        if angQ - angP > np.pi:
            angP, angQ = angQ, angP + 2*np.pi

        # Extend the arc further in both directions
        a1 = angP - 0.4
        a2 = angQ + 0.4

        arc_theta = np.linspace(a1, a2, 200)
        arc_x = center[0] + radius * np.cos(arc_theta)
        arc_y = center[1] + radius * np.sin(arc_theta)

        # Check if this geodesic intersects the x-axis
        intersects = False
        for j in range(len(arc_x) - 1):
            if arc_y[j] * arc_y[j+1] <= 0 and abs(arc_x[j]) < 1:
                intersects = True
                break

        color = DANGER if intersects else '#444'
        alpha = 0.7 if intersects else 0.4
        ax.plot(arc_x, arc_y, color=color, linewidth=1, alpha=alpha)

    # Add the two limiting parallels (ultraparallel boundary)
    # These are harder to compute analytically; approximate
    for sign in [-1, 1]:
        target_x = sign * 1.0
        target_y = 0
        Q = (target_x, target_y)
        center, radius = compute_geodesic_center(P, Q)
        if center is not None:
            angP = np.arctan2(P[1]-center[1], P[0]-center[0])
            angQ = np.arctan2(Q[1]-center[1], Q[0]-center[0])
            if angQ < angP:
                angQ += 2*np.pi
            if angQ - angP > np.pi:
                angP, angQ = angQ, angP + 2*np.pi
            arc_theta = np.linspace(angP, angQ, 200)
            arc_x = center[0] + radius * np.cos(arc_theta)
            arc_y = center[1] + radius * np.sin(arc_theta)
            ax.plot(arc_x, arc_y, color=GOLD, linewidth=1.5, linestyle='--',
                   label='Limiting parallel' if sign == -1 else '')

    ax.legend(facecolor=DARK_SURFACE, edgecolor=DARK_BORDER,
              labelcolor=DARK_TEXT, fontsize=8, loc='lower left')

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved parallel lines → {savepath}")


# =================================================================
# FIGURE 4: TESSELLATION VIABILITY SPACE
# =================================================================

def plot_tessellation_space(savepath):
    """Show which {p,q} Schlaeffli symbols produce valid hyperbolic tessellations."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    dark_style(ax, "Tessellation Viability: The (p-2)(q-2) > 4 Rule\n{regular p-gons, q meeting at each vertex}")

    # Regions
    p_vals = np.arange(3, 15)
    q_vals = np.arange(3, 15)
    P, Q = np.meshgrid(p_vals, q_vals, indexing='ij')

    # Classification
    # Euclidean: (p-2)(q-2) = 4 → {3,6}, {4,4}, {6,3}
    # Spherical: (p-2)(q-2) < 4
    # Hyperbolic: (p-2)(q-2) > 4

    Z = (P - 2) * (Q - 2)
    Z_color = np.where(Z > 4, 2, np.where(Z == 4, 1, 0))

    colors_list = ['#ff6070', GOLD, ACCENT2]
    labels_list = ['Spherical (<4)', 'Euclidean (=4)', 'Hyperbolic (>4)']

    for val in [0, 1, 2]:
        mask = Z_color == val
        if mask.any():
            ax.scatter(P[mask], Q[mask], c=colors_list[val],
                      s=120, alpha=0.8, edgecolors='white', linewidth=0.5,
                      label=labels_list[val])

    # Annotate key tessellations
    key_tess = [(3,7), (3,8), (4,5), (4,7), (5,4), (5,5), (6,4), (7,3), (8,3), (3,6), (4,4), (6,3)]
    for (p, q) in key_tess:
        val = (p-2)*(q-2)
        color = ACCENT2 if val > 4 else (GOLD if val == 4 else DANGER)
        ax.annotate(f'{{{p},{q}}}', xy=(p, q), xytext=(p+0.25, q+0.25),
                   color=color, fontsize=8, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color=color, lw=0.5))

    ax.set_xlabel('p (polygon sides)', fontsize=11)
    ax.set_ylabel('q (meeting at vertex)', fontsize=11)
    ax.set_xticks(p_vals)
    ax.set_yticks(q_vals)
    ax.legend(facecolor=DARK_SURFACE, edgecolor=DARK_BORDER,
              labelcolor=DARK_TEXT, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.1, color='white')
    ax.set_xlim(2.5, 14.5)
    ax.set_ylim(2.5, 14.5)

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved tessellation space → {savepath}")


# =================================================================
# FIGURE 5: EXPONENTIAL GROWTH OF CIRCLES
# =================================================================

def plot_circle_growth(savepath):
    """Show that hyperbolic circle circumference grows exponentially with radius."""
    # In hyperbolic geometry, circumference = 2π sinh(r)
    # In Euclidean geometry, circumference = 2π r

    r = np.linspace(0, 5, 200)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    dark_style(ax, "Circle Circumference Growth")

    ax.plot(r, 2*np.pi * np.sinh(r), color=ACCENT2, linewidth=2.5,
            label='Hyperbolic: C = 2π·sinh(r)')
    ax.plot(r, 2*np.pi * r, color='#555', linewidth=1.5, linestyle='--',
            label='Euclidean: C = 2π·r')

    # Area: A = 2π(cosh(r) - 1) in hyperbolic
    ax2 = ax.twinx()
    ax2.plot(r, 2*np.pi * (np.cosh(r) - 1), color=ACCENT, linewidth=2,
             linestyle=':', label='Hyperbolic Area: A = 2π(cosh(r)-1)')
    ax2.set_ylabel('Area', color=ACCENT)
    ax2.tick_params(colors=ACCENT, labelsize=8)

    ax.set_xlabel('Radius (hyperbolic distance)')
    ax.set_ylabel('Circumference', color=ACCENT2)
    ax.set_title('Circle Circumference Comparison\nHyperbolic: C = 2pi*sinh(r), Euclidean: C = 2pi*r')
    ax.tick_params(colors=DARK_TEXT, labelsize=8)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              facecolor=DARK_SURFACE, edgecolor=DARK_BORDER,
              labelcolor=DARK_TEXT, fontsize=8, loc='upper left')

    ax.grid(True, alpha=0.1, color='white')

    # Annotations
    for r_val in [1, 2, 3]:
        circ = 2 * np.pi * np.sinh(r_val)
        ax.annotate(f'r={r_val}\nC={circ:.1f}',
                   xy=(r_val, circ), xytext=(r_val+0.3, circ*1.5),
                   color=ACCENT2, fontsize=8,
                   arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=0.8))

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved circle growth → {savepath}")


# =================================================================
# FIGURE 6: POINCARE DISK TESSELLATION (Python version)
# =================================================================

def plot_poincare_tessellation(savepath):
    """Draw a Poincaré disk with a {5,4} tessellation using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    dark_style(ax, "Poincaré Disk Tessellation {5,4}\nRegular pentagons, 4 meeting at each vertex")

    # Disk boundary
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=DARK_BORDER, linewidth=1.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # Compute {5,4} tessellation edges
    edges = generate_tessellation_edges(5, 4, max_depth=6, max_edges=5000)

    # Draw edges
    for a, b in edges:
        center, radius = compute_geodesic_center(a, b)
        if center is None:
            ax.plot([a[0], b[0]], [a[1], b[1]],
                   color='rgba(255,255,255,0.15)', linewidth=0.5)
        else:
            ang_a = np.arctan2(a[1]-center[1], a[0]-center[0])
            ang_b = np.arctan2(b[1]-center[1], b[0]-center[0])
            if ang_b < ang_a:
                ang_b += 2*np.pi
            if ang_b - ang_a > np.pi:
                ang_a, ang_b = ang_b, ang_a + 2*np.pi
            arc_theta = np.linspace(ang_a, ang_b, 100)
            arc_x = center[0] + radius * np.cos(arc_theta)
            arc_y = center[1] + radius * np.sin(arc_theta)
            ax.plot(arc_x, arc_y, color='white', linewidth=0.5, alpha=0.15)

    # Redraw boundary on top
    ax.plot(np.cos(theta), np.sin(theta), color=DARK_BORDER, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(savepath, facecolor=DARK_BG)
    plt.close()
    print(f"Saved Poincaré tessellation → {savepath}")


def reflect_across_edge(pt, a, b):
    """Reflect point pt across the geodesic through a and b."""
    center, radius = compute_geodesic_center(a, b)
    if center is None:
        # Reflect across line
        dx, dy = b[0]-a[0], b[1]-a[1]
        len2 = dx*dx + dy*dy
        if len2 < 1e-10:
            return pt
        t = ((pt[0]-a[0])*dx + (pt[1]-a[1])*dy) / len2
        proj = (a[0] + t*dx, a[1] + t*dy)
        return (2*proj[0] - pt[0], 2*proj[1] - pt[1])
    else:
        # Inversion in circle
        dx, dy = pt[0]-center[0], pt[1]-center[1]
        d2 = dx*dx + dy*dy
        if d2 < 1e-10:
            return pt
        factor = radius*radius / d2
        return (center[0] + dx*factor, center[1] + dy*factor)


def generate_tessellation_edges(p, q, max_depth=6, max_edges=8000):
    """Generate edges of a {p,q} hyperbolic tessellation via reflection group BFS."""
    # Compute central polygon
    central_angle = 2 * np.pi / p
    cosR = np.cos(np.pi / p) / np.sin(np.pi / q)
    if cosR < 1:
        return []  # Not hyperbolic
    r = np.arccosh(cosR)
    euclidean_r = np.tanh(r / 2)

    vertices = []
    for i in range(p):
        angle = central_angle * i - np.pi / 2
        vertices.append((euclidean_r * np.cos(angle), euclidean_r * np.sin(angle)))

    def in_disk(pt):
        return pt[0]**2 + pt[1]**2 < 0.99

    edges = []
    visited = set()

    def edge_key(a, b):
        eps = 0.001
        ax = round(a[0]/eps)*eps
        ay = round(a[1]/eps)*eps
        bx = round(b[0]/eps)*eps
        by = round(b[1]/eps)*eps
        if ax < bx or (ax == bx and ay < by):
            return (ax, ay, bx, by)
        return (bx, by, ax, ay)

    def add_polygon(verts, depth):
        if len(edges) >= max_edges or depth > max_depth:
            return

        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            key = edge_key(verts[i], verts[j])
            if key not in visited:
                visited.add(key)
                edges.append((verts[i], verts[j]))

        if depth < max_depth:
            for i in range(len(verts)):
                j = (i + 1) % len(verts)
                new_verts = [reflect_across_edge(v, verts[i], verts[j]) for v in verts]
                if any(not in_disk(v) for v in new_verts):
                    continue
                add_polygon(new_verts, depth + 1)

    # BFS approach
    queue = [(vertices, 0)]
    processed = 0
    while queue and len(edges) < max_edges:
        verts, depth = queue.pop(0)
        add_polygon(verts, depth)
        processed += 1
        if processed > 2000:
            break

    return edges


# =================================================================
# MAIN
# =================================================================

def main():
    import os
    output_dir = "/home/yanyj/VibeCoding/autonomy/2026-06-07/hyperbolic-canvas/figures"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CURVATURE OF THOUGHT — Computational Exploration")
    print("=" * 60)

    # 1. Distance distortion
    plot_distance_distortion(os.path.join(output_dir, "distance_distortion.png"))

    # 2. Triangle angle defect
    plot_triangle_defect(os.path.join(output_dir, "triangle_defect.png"))

    # 3. Parallel lines
    plot_parallel_lines(os.path.join(output_dir, "parallel_lines.png"))

    # 4. Tessellation viability
    plot_tessellation_space(os.path.join(output_dir, "tessellation_space.png"))

    # 5. Circle growth
    plot_circle_growth(os.path.join(output_dir, "circle_growth.png"))

    # 6. Poincaré disk tessellation
    plot_poincare_tessellation(os.path.join(output_dir, "poincare_tessellation.png"))

    print("\nAll figures generated in:", output_dir)
    print("\nKey Insights:")
    print("  1. Hyperbolic distance → ∞ as Euclidean r → 1 (boundary effect)")
    print("  2. Triangle angle sum always < 180°, defect = hyperbolic area")
    print("  3. Infinitely many parallels through a point not on a line")
    print("  4. (p-2)(q-2) > 4 determines hyperbolic tessellation viability")
    print("  5. Circle circumference grows exponentially: C = 2π·sinh(r)")


if __name__ == "__main__":
    main()
