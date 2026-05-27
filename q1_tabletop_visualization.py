"""
=============================================================================
Q1 - Tabletop Scene Visualization  [Research Grade — 20-Object Extended]
=============================================================================
Changes from the previous 7-object version:
  CHANGE-1 (OBJECTS list):
    Extended from 7 → 20 objects. 13 new real-world tabletop objects added:
    Cup, Stapler, Tape, Scissors, Penholder, Notebook, Bowl, Vase,
    Cube, Phone, Keys, GlassCase, Spray.

  CHANGE-2 (Table & arm constants):
    TABLE_HALF_X  0.50 → 0.65  (more table area for 20 objects)
    TABLE_HALF_Y  0.40 → 0.55
    ARM_BASE_POS  [-0.55, 0, H] → [-0.65, 0, H]  (shifted with table)
    ARM_MAX_REACH 0.82 → 0.90  (conservative KUKA reach margin)

  CHANGE-3 (sample_object_surface):
    Generalized: original 7 keep their detailed implementations (Mug handle,
    Bottle taper/neck, Plate well, Can ribs, Book spine, etc.).
    The 13 new objects use a high-quality generic sampler (type-dispatch:
    cylinder → dense wall + discs + optional hollow; box → all 6 faces with
    area-proportional density).  Produces ≥ 2048 well-distributed pts each.

All original bug-fixes (BUG-1 through BUG-8) are preserved exactly:
  BUG-1  make_object_pcd_world  — rotation applied to both pts & normals
  BUG-4  Vector3dVector         — explicit float64
  BUG-5  viz_depth_vs_analytic  — vectorised compute_point_cloud_distance
  BUG-7  viz_top_down arrows    — annotation_clip=False
  BUG-8  voxel_size parity      — both depth & analytic at 0.004 m
=============================================================================
"""
import os, math, time
from pathlib import Path
import numpy as np
import open3d as o3d
import pybullet as pb
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as pb
import pybullet_data
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from PIL import Image
from scipy.spatial.transform import Rotation as ScipyR



OUTPUT_DIR = Path("q1_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Scene constants ──────────────────────────────────────────────────────────
# CHANGE-2: enlarged table + adjusted arm reach for 20 objects
TABLE_HEIGHT  = 0.60
TABLE_HALF_X  = 0.78          # was 0.50
TABLE_HALF_Y  = 0.55          # was 0.40
TABLE_CENTER  = [0.0, 0.0, TABLE_HEIGHT]

ARM_BASE_POS  = [-0.65, 0.0, TABLE_HEIGHT]   # was [-0.55, ...]
ARM_MIN_REACH = 0.18
ARM_MAX_REACH = 0.90           # was 0.82

IMG_WIDTH  = 640
IMG_HEIGHT = 480
FOV        = 42.0
NEAR_PLANE = 0.05
FAR_PLANE  = 3.5

CAMERAS = [
    {"name": "cam_front",
     "pos": [0.0, -1.75, TABLE_HEIGHT + 1.0],
     "target": TABLE_CENTER,
     "up": [0,0,1]},

    {"name": "cam_back",
     "pos": [0.0, 1.75, TABLE_HEIGHT + 1.0],
     "target": TABLE_CENTER,
     "up": [0,0,1]},

    {"name": "cam_left",
     "pos": [-1.75, 0.0, TABLE_HEIGHT + 1.0],
     "target": TABLE_CENTER,
     "up": [0,0,1]},

    {"name": "cam_top",
     "pos": [0.0, 0.0, TABLE_HEIGHT + 1.8],
     "target": TABLE_CENTER,
     "up": [0,1,0]},
]
# ── CHANGE-1: 20 objects ─────────────────────────────────────────────────────
OBJECTS = [
    # ── Original 7 (dimensions unchanged) ────────────────────────────────────
    {"name": "Mug",       "type": "cylinder", "radius": 0.038, "height": 0.095,
     "mass": 0.30, "color": [0.80, 0.30, 0.10, 1.0]},
    {"name": "Book",      "type": "box",      "half": [0.105, 0.075, 0.016],
     "mass": 0.40, "color": [0.20, 0.40, 0.80, 1.0]},
    {"name": "Bottle",    "type": "cylinder", "radius": 0.028, "height": 0.230,
     "mass": 0.50, "color": [0.10, 0.70, 0.20, 1.0]},
    {"name": "Can",       "type": "cylinder", "radius": 0.034, "height": 0.122,
     "mass": 0.35, "color": [0.90, 0.80, 0.10, 1.0]},
    {"name": "Box",       "type": "box",      "half": [0.065, 0.045, 0.042],
     "mass": 0.20, "color": [0.60, 0.20, 0.60, 1.0]},
    {"name": "Plate",     "type": "cylinder", "radius": 0.092, "height": 0.014,
     "mass": 0.40, "color": [0.95, 0.95, 0.85, 1.0]},
    {"name": "Remote",    "type": "box",      "half": [0.065, 0.028, 0.013],
     "mass": 0.15, "color": [0.15, 0.15, 0.15, 1.0]},
    # ── 13 New Objects ────────────────────────────────────────────────────────
    {"name": "Cup",       "type": "cylinder", "radius": 0.030, "height": 0.075,
     "mass": 0.20, "color": [0.80, 0.60, 0.80, 1.0]},
    {"name": "Stapler",   "type": "box",      "half": [0.070, 0.028, 0.028],
     "mass": 0.35, "color": [0.10, 0.10, 0.80, 1.0]},
    {"name": "Tape",      "type": "cylinder", "radius": 0.042, "height": 0.030,
     "mass": 0.10, "color": [0.70, 0.90, 0.90, 1.0]},
    {"name": "Scissors",  "type": "box",      "half": [0.048, 0.016, 0.010],
     "mass": 0.10, "color": [0.80, 0.20, 0.20, 1.0]},
    {"name": "Penholder", "type": "cylinder", "radius": 0.036, "height": 0.115,
     "mass": 0.25, "color": [0.30, 0.60, 0.40, 1.0]},
    {"name": "Notebook",  "type": "box",      "half": [0.095, 0.068, 0.011],
     "mass": 0.30, "color": [0.95, 0.50, 0.20, 1.0]},
    {"name": "Bowl",      "type": "cylinder", "radius": 0.082, "height": 0.052,
     "mass": 0.35, "color": [0.90, 0.90, 0.70, 1.0]},
    {"name": "Vase",      "type": "cylinder", "radius": 0.033, "height": 0.155,
     "mass": 0.40, "color": [0.50, 0.30, 0.80, 1.0]},
    {"name": "Cube",      "type": "box",      "half": [0.032, 0.032, 0.032],
     "mass": 0.10, "color": [0.95, 0.20, 0.50, 1.0]},
    {"name": "Phone",     "type": "box",      "half": [0.038, 0.078, 0.008],
     "mass": 0.20, "color": [0.20, 0.20, 0.20, 1.0]},
    {"name": "Keys",      "type": "box",      "half": [0.042, 0.022, 0.008],
     "mass": 0.08, "color": [0.80, 0.70, 0.10, 1.0]},
    {"name": "GlassCase", "type": "box",      "half": [0.082, 0.040, 0.020],
     "mass": 0.15, "color": [0.40, 0.40, 0.40, 1.0]},
    {"name": "Spray",     "type": "cylinder", "radius": 0.026, "height": 0.205,
     "mass": 0.45, "color": [0.20, 0.80, 0.80, 1.0]},
]
OBJ_O3D_COLORS = [o["color"][:3] for o in OBJECTS]


# ── Geometry helpers ─────────────────────────────────────────────────────────

def dist_to_arm(x: float, y: float) -> float:
    return math.sqrt((x - ARM_BASE_POS[0]) ** 2 + (y - ARM_BASE_POS[1]) ** 2)


def footprint_radius(obj: dict) -> float:
    if obj["type"] == "cylinder":
        return obj["radius"] + 0.008
    return math.sqrt(obj["half"][0] ** 2 + obj["half"][1] ** 2) + 0.008


def object_height_offset(obj: dict) -> float:
    if obj["type"] == "cylinder":
        return obj["height"] / 2.0
    return obj["half"][2]


def generate_spawn_positions(objects: list, seed: int = 42) -> list:
    """
    Robust dense tabletop placement for 20-object scenes.

    Improvements:
    - Large objects placed first
    - Adaptive clearance
    - Better free-space utilization
    - Deterministic stable packing
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # SORT OBJECTS BY FOOTPRINT (largest first)
    # ------------------------------------------------------------
    indexed = list(enumerate(objects))
    indexed.sort(
        key=lambda x: footprint_radius(x[1]),
        reverse=True
    )

    positions_temp = {}
    fp_radii = [footprint_radius(o) for o in objects]

    # Slightly relaxed margins
    EDGE_MARGIN = 0.025     # was effectively 0.04
    EXTRA_CLEARANCE = 0.002 # was 0.005

    for order_idx, (orig_idx, obj) in enumerate(indexed):

        fp = fp_radii[orig_idx]

        # More retries for large objects
        max_tries = 25000 if fp > 0.08 else 12000

        placed = False

        for _ in range(max_tries):

            x = rng.uniform(
                -TABLE_HALF_X + fp + EDGE_MARGIN,
                 TABLE_HALF_X - fp - EDGE_MARGIN
            )

            y = rng.uniform(
                -TABLE_HALF_Y + fp + EDGE_MARGIN,
                 TABLE_HALF_Y - fp - EDGE_MARGIN
            )

            # Reachability constraint
            d = dist_to_arm(x, y)

            if not (ARM_MIN_REACH <= d <= ARM_MAX_REACH):
                continue

            # Collision test
            overlap = False

            for j, (px, py, _) in positions_temp.items():

                min_sep = (
                    fp +
                    fp_radii[j] +
                    EXTRA_CLEARANCE
                )

                if math.hypot(x - px, y - py) < min_sep:
                    overlap = True
                    break

            if overlap:
                continue

            z = TABLE_HEIGHT + object_height_offset(obj)

            positions_temp[orig_idx] = (
                float(x),
                float(y),
                float(z)
            )

            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Cannot place {obj['name']} "
                f"after {max_tries} tries.\n"
                f"Try increasing TABLE_HALF_X/Y slightly "
                f"or reducing footprint padding."
            )

    # ------------------------------------------------------------
    # RESTORE ORIGINAL OBJECT ORDER
    # ------------------------------------------------------------
    positions = [
        positions_temp[i]
        for i in range(len(objects))
    ]

    return positions


# ── Analytic surface sampling ────────────────────────────────────────────────

def _sample_generic_cylinder(obj: dict, rng) -> tuple:
    """High-quality generic cylinder surface: wall + top disc + bottom disc."""
    r = obj["radius"]
    h = obj["height"]
    pts, nrm = [], []
    # --- Lateral wall (dense) ---
    for k in np.linspace(-h / 2, h / 2, 32):
        for t in np.linspace(0, 2 * np.pi, 44, endpoint=False):
            pts.append([r * np.cos(t), r * np.sin(t), k])
            nrm.append([np.cos(t), np.sin(t), 0.0])
    # --- Top disc ---
    for rr in np.linspace(0, r, 7):
        for t in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            pts.append([rr * np.cos(t), rr * np.sin(t),  h / 2])
            nrm.append([0.0, 0.0, 1.0])
    # --- Bottom disc ---
    for rr in np.linspace(0, r, 7):
        for t in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            pts.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
            nrm.append([0.0, 0.0, -1.0])
    return np.array(pts, dtype=np.float32), np.array(nrm, dtype=np.float32)


def _sample_generic_box(obj: dict, rng) -> tuple:
    """High-quality generic box surface: 6 faces with area-proportional density."""
    hx, hy, hz = obj["half"]
    pts, nrm = [], []
    face_specs = [
        ([hx,  0,  0], [1,  0,  0], hy, hz, 18, 14),
        ([-hx, 0,  0], [-1, 0,  0], hy, hz, 18, 14),
        ([0,  hy,  0], [0,  1,  0], hx, hz, 22, 14),
        ([0, -hy,  0], [0, -1,  0], hx, hz, 22, 14),
        ([0,  0,  hz], [0,  0,  1], hx, hy, 22, 18),
        ([0,  0, -hz], [0,  0, -1], hx, hy, 22, 18),
    ]
    for centre, normal, su, sv, nu, nv in face_specs:
        n = np.array(normal, dtype=np.float64)
        ax1 = np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0])
        ax1 /= np.linalg.norm(ax1) + 1e-12
        ax2 = np.cross(n, ax1)
        ax2 /= np.linalg.norm(ax2) + 1e-12
        ctr = np.array(centre, dtype=np.float64)
        for u in np.linspace(-su, su, nu):
            for v in np.linspace(-sv, sv, nv):
                pts.append((ctr + u * ax1 + v * ax2).tolist())
                nrm.append(n.tolist())
    return np.array(pts, dtype=np.float32), np.array(nrm, dtype=np.float32)


def sample_object_surface(obj: dict, n_pts: int = 2048):
    """
    Analytically sample surface points + outward normals in object-LOCAL frame.
    Local frame: centroid at origin, Z is up.

    Original 7 objects keep their detailed implementations.
    New 13 objects use high-quality generic type-dispatch sampling.

    Returns:
        pts_arr : (n_pts, 3) float32
        nrm_arr : (n_pts, 3) float32  — unit outward normals
    """
    name = obj["name"]
    rng  = np.random.default_rng(hash(name) & 0xFFFFFFFF)
    pts_list: list = []
    nrm_list: list = []

    # ─────────────────────────── Original 7 (detailed) ───────────────────────
    if name == "Mug":
        r_out, r_in, h = 0.038, 0.032, 0.095
        for k in np.linspace(-h / 2, h / 2, 28):
            for t in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                pts_list.append([r_out * np.cos(t), r_out * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for k in np.linspace(-h / 2 + 0.004, h / 2 - 0.001, 20):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([r_in * np.cos(t), r_in * np.sin(t), k])
                nrm_list.append([-np.cos(t), -np.sin(t), 0.0])
        for rr in np.linspace(r_in, r_out, 4):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), h / 2])
                nrm_list.append([0.0, 0.0, 1.0])
        for rr in np.linspace(0, r_out, 5):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
                nrm_list.append([0.0, 0.0, -1.0])
        h_r, h_cx, h_cz = 0.024, r_out + 0.020, 0.0
        for ang in np.linspace(-np.pi * 0.55, np.pi * 0.55, 22):
            for thick in np.linspace(-0.005, 0.005, 3):
                pts_list.append([h_cx + h_r * np.cos(ang), thick, h_cz + h_r * np.sin(ang)])
                nrm_list.append([np.cos(ang), 0.0, np.sin(ang)])

    elif name == "Book":
        hx, hy, hz = 0.105, 0.075, 0.016
        face_specs = [
            ([hx, 0, 0],  [1, 0, 0],  hy, hz),
            ([-hx, 0, 0], [-1, 0, 0], hy, hz),
            ([0, hy, 0],  [0, 1, 0],  hx, hz),
            ([0, -hy, 0], [0, -1, 0], hx, hz),
            ([0, 0, hz],  [0, 0, 1],  hx, hy),
            ([0, 0, -hz], [0, 0, -1], hx, hy),
        ]
        for centre, normal, span_u, span_v in face_specs:
            nrm = np.array(normal, dtype=np.float64)
            ax1 = np.cross(nrm, [1, 0, 0]) if abs(nrm[0]) < 0.9 else np.cross(nrm, [0, 1, 0])
            ax1 = ax1 / (np.linalg.norm(ax1) + 1e-12)
            ax2 = np.cross(nrm, ax1)
            ax2 = ax2 / (np.linalg.norm(ax2) + 1e-12)
            ctr = np.array(centre, dtype=np.float64)
            for u in np.linspace(-span_u, span_u, 16):
                for v in np.linspace(-span_v, span_v, 10):
                    pts_list.append((ctr + u * ax1 + v * ax2).tolist())
                    nrm_list.append(nrm.tolist())
        for z in np.linspace(-hz, hz, 8):
            for y in np.linspace(-hy, hy, 18):
                pts_list.append([-hx - 0.003, y, z])
                nrm_list.append([-1.0, 0.0, 0.0])

    elif name == "Bottle":
        r_body, r_neck = 0.028, 0.013
        h_body, h_neck = 0.175, 0.050
        taper_h = 0.022
        for k in np.linspace(-h_body / 2, h_body / 2, 32):
            for t in np.linspace(0, 2 * np.pi, 36, endpoint=False):
                pts_list.append([r_body * np.cos(t), r_body * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        ang = math.atan2(r_neck - r_body, taper_h)
        for frac in np.linspace(0, 1, 10):
            r_t = r_body + frac * (r_neck - r_body)
            z_t = h_body / 2 + frac * taper_h
            nr_t = math.cos(ang)
            nz_t = math.sin(abs(ang))
            for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                pts_list.append([r_t * np.cos(t), r_t * np.sin(t), z_t])
                nrm_list.append([nr_t * np.cos(t), nr_t * np.sin(t), nz_t])
        z_nb = h_body / 2 + taper_h
        for k in np.linspace(z_nb, z_nb + h_neck, 12):
            for t in np.linspace(0, 2 * np.pi, 22, endpoint=False):
                pts_list.append([r_neck * np.cos(t), r_neck * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for rr in np.linspace(0, r_body, 5):
            for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h_body / 2])
                nrm_list.append([0.0, 0.0, -1.0])
        z_cap = z_nb + h_neck
        for rr in np.linspace(0, r_neck, 3):
            for t in np.linspace(0, 2 * np.pi, 18, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), z_cap])
                nrm_list.append([0.0, 0.0, 1.0])

    elif name == "Can":
        r, h, n_ribs = 0.034, 0.122, 10
        for k in np.linspace(-h / 2, h / 2, 30):
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                rib = 1.0 + 0.012 * np.sin(n_ribs * t)
                pts_list.append([r * rib * np.cos(t), r * rib * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for sign, nz in [(1, 1.0), (-1, -1.0)]:
            for rr in np.linspace(0, r, 6):
                for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                    pts_list.append([rr * np.cos(t), rr * np.sin(t), sign * h / 2])
                    nrm_list.append([0.0, 0.0, nz])

    elif name == "Box":
        hx, hy, hz = 0.065, 0.045, 0.042
        face_specs = [
            ([hx, 0, 0],  [1, 0, 0],  hy, hz, 12, 10),
            ([-hx, 0, 0], [-1, 0, 0], hy, hz, 12, 10),
            ([0, hy, 0],  [0, 1, 0],  hx, hz, 16, 10),
            ([0, -hy, 0], [0, -1, 0], hx, hz, 16, 10),
            ([0, 0, hz],  [0, 0, 1],  hx, hy, 16, 12),
            ([0, 0, -hz], [0, 0, -1], hx, hy, 16, 12),
        ]
        for centre, normal, su, sv, nu, nv in face_specs:
            nrm = np.array(normal, dtype=np.float64)
            ax1 = np.cross(nrm, [1, 0, 0]) if abs(nrm[0]) < 0.9 else np.cross(nrm, [0, 1, 0])
            ax1 /= np.linalg.norm(ax1) + 1e-12
            ax2 = np.cross(nrm, ax1)
            ax2 /= np.linalg.norm(ax2) + 1e-12
            ctr = np.array(centre, dtype=np.float64)
            for u in np.linspace(-su, su, nu):
                for v in np.linspace(-sv, sv, nv):
                    pts_list.append((ctr + u * ax1 + v * ax2).tolist())
                    nrm_list.append(nrm.tolist())

    elif name == "Plate":
        r_out, r_in, h, well_depth = 0.092, 0.072, 0.014, 0.007
        for k in np.linspace(-h / 2, h / 2, 6):
            for t in np.linspace(0, 2 * np.pi, 52, endpoint=False):
                pts_list.append([r_out * np.cos(t), r_out * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for rr in np.linspace(r_in, r_out, 5):
            for t in np.linspace(0, 2 * np.pi, 52, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), h / 2])
                nrm_list.append([0.0, 0.0, 1.0])
        for rr in np.linspace(0, r_in, 14):
            zz = h / 2 - well_depth * (rr / r_in) ** 2
            dz_dr = -2.0 * well_depth * rr / (r_in ** 2)
            n_len = math.sqrt(dz_dr ** 2 + 1.0) + 1e-9
            nr = -dz_dr / n_len
            nz = 1.0 / n_len
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), zz])
                nrm_list.append([nr * np.cos(t), nr * np.sin(t), nz])
        for rr in np.linspace(0, r_out, 8):
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
                nrm_list.append([0.0, 0.0, -1.0])

    elif name == "Remote":
        hx, hy, hz = 0.065, 0.028, 0.013
        for u in np.linspace(-hx + 0.006, hx - 0.006, 22):
            for v in np.linspace(-hy + 0.005, hy - 0.005, 10):
                u_ph = u % (2 * hx / 3)
                v_ph = v % (2 * hy / 5)
                bump = 0.0015 * math.exp(-((u_ph - hx / 3) ** 2 +
                                           (v_ph - hy / 5) ** 2) / 5e-5)
                pts_list.append([u, v, hz + bump])
                nrm_list.append([0.0, 0.0, 1.0])
        for u in np.linspace(-hx, hx, 18):
            for v in np.linspace(-hy, hy, 8):
                pts_list.append([u, v, -hz])
                nrm_list.append([0.0, 0.0, -1.0])
        for sx in [-1, 1]:
            for v in np.linspace(-hy, hy, 8):
                for z in np.linspace(-hz, hz, 6):
                    pts_list.append([sx * hx, v, z])
                    nrm_list.append([float(sx), 0.0, 0.0])
        for sy in [-1, 1]:
            for u in np.linspace(-hx, hx, 18):
                for z in np.linspace(-hz, hz, 6):
                    pts_list.append([u, sy * hy, z])
                    nrm_list.append([0.0, float(sy), 0.0])

    # ─────────────────────────── New 13 (generic + overrides) ────────────────
    elif name == "Cup":
        # Cylinder without handle, slightly thicker wall
        r, h, r_in = 0.030, 0.075, 0.025
        for k in np.linspace(-h / 2, h / 2, 28):
            for t in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                pts_list.append([r * np.cos(t), r * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        # Inner wall (hollow)
        for k in np.linspace(-h / 2 + 0.005, h / 2 - 0.002, 18):
            for t in np.linspace(0, 2 * np.pi, 30, endpoint=False):
                pts_list.append([r_in * np.cos(t), r_in * np.sin(t), k])
                nrm_list.append([-np.cos(t), -np.sin(t), 0.0])
        # Rim annulus
        for rr in np.linspace(r_in, r, 3):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), h / 2])
                nrm_list.append([0.0, 0.0, 1.0])
        # Base
        for rr in np.linspace(0, r, 5):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
                nrm_list.append([0.0, 0.0, -1.0])

    elif name == "Tape":
        # Washer/spool: outer wall + inner wall (hollow) + top/bottom annuli
        r_out, r_in, h = 0.042, 0.018, 0.030
        for k in np.linspace(-h / 2, h / 2, 10):
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts_list.append([r_out * np.cos(t), r_out * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for k in np.linspace(-h / 2, h / 2, 10):
            for t in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                pts_list.append([r_in * np.cos(t), r_in * np.sin(t), k])
                nrm_list.append([-np.cos(t), -np.sin(t), 0.0])
        for sign, nz in [(1, 1.0), (-1, -1.0)]:
            for rr in np.linspace(r_in, r_out, 6):
                for t in np.linspace(0, 2 * np.pi, 44, endpoint=False):
                    pts_list.append([rr * np.cos(t), rr * np.sin(t), sign * h / 2])
                    nrm_list.append([0.0, 0.0, nz])

    elif name == "Penholder":
        # Open-top cylinder (like Mug without handle)
        r, h, r_in = 0.036, 0.115, 0.030
        for k in np.linspace(-h / 2, h / 2, 34):
            for t in np.linspace(0, 2 * np.pi, 42, endpoint=False):
                pts_list.append([r * np.cos(t), r * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for k in np.linspace(-h / 2 + 0.003, h / 2 - 0.001, 26):
            for t in np.linspace(0, 2 * np.pi, 34, endpoint=False):
                pts_list.append([r_in * np.cos(t), r_in * np.sin(t), k])
                nrm_list.append([-np.cos(t), -np.sin(t), 0.0])
        for rr in np.linspace(r_in, r, 4):
            for t in np.linspace(0, 2 * np.pi, 34, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), h / 2])
                nrm_list.append([0.0, 0.0, 1.0])
        for rr in np.linspace(0, r, 6):
            for t in np.linspace(0, 2 * np.pi, 34, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
                nrm_list.append([0.0, 0.0, -1.0])

    elif name == "Bowl":
        # Wide shallow cylinder with paraboloid concave interior
        r_out, h, well_d = 0.082, 0.052, 0.030
        r_in = r_out * 0.88
        for k in np.linspace(-h / 2, h / 2, 8):
            for t in np.linspace(0, 2 * np.pi, 56, endpoint=False):
                pts_list.append([r_out * np.cos(t), r_out * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        # Annular top rim
        for rr in np.linspace(r_in, r_out, 5):
            for t in np.linspace(0, 2 * np.pi, 52, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), h / 2])
                nrm_list.append([0.0, 0.0, 1.0])
        # Concave paraboloid interior
        for rr in np.linspace(0, r_in, 12):
            zz = h / 2 - well_d * (rr / r_in) ** 2
            dz_dr = -2.0 * well_d * rr / (r_in ** 2)
            n_len = math.sqrt(dz_dr ** 2 + 1.0) + 1e-9
            nr, nz = -dz_dr / n_len, 1.0 / n_len
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), zz])
                nrm_list.append([nr * np.cos(t), nr * np.sin(t), nz])
        for rr in np.linspace(0, r_out, 8):
            for t in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h / 2])
                nrm_list.append([0.0, 0.0, -1.0])

    elif name == "Vase":
        # Cylinder with tapered neck at top
        r_body, r_neck = 0.033, 0.018
        h_body, h_neck, taper_h = 0.110, 0.040, 0.015
        for k in np.linspace(-h_body / 2, h_body / 2, 36):
            for t in np.linspace(0, 2 * np.pi, 38, endpoint=False):
                pts_list.append([r_body * np.cos(t), r_body * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        ang_v = math.atan2(r_neck - r_body, taper_h)
        for frac in np.linspace(0, 1, 8):
            r_t = r_body + frac * (r_neck - r_body)
            z_t = h_body / 2 + frac * taper_h
            nr_t, nz_t = math.cos(ang_v), math.sin(abs(ang_v))
            for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
                pts_list.append([r_t * np.cos(t), r_t * np.sin(t), z_t])
                nrm_list.append([nr_t * np.cos(t), nr_t * np.sin(t), nz_t])
        z_nb = h_body / 2 + taper_h
        for k in np.linspace(z_nb, z_nb + h_neck, 10):
            for t in np.linspace(0, 2 * np.pi, 22, endpoint=False):
                pts_list.append([r_neck * np.cos(t), r_neck * np.sin(t), k])
                nrm_list.append([np.cos(t), np.sin(t), 0.0])
        for rr in np.linspace(0, r_body, 6):
            for t in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                pts_list.append([rr * np.cos(t), rr * np.sin(t), -h_body / 2])
                nrm_list.append([0.0, 0.0, -1.0])

    else:
        # ── Generic fallback: cylinder or box ────────────────────────────────
        if obj["type"] == "cylinder":
            pts_arr, nrm_arr = _sample_generic_cylinder(obj, rng)
        else:
            pts_arr, nrm_arr = _sample_generic_box(obj, rng)
        # Resample to n_pts
        n = len(pts_arr)
        idx = rng.choice(n, n_pts, replace=(n < n_pts))
        pts_arr = pts_arr[idx]
        nrm_arr = nrm_arr[idx]
        lens = np.linalg.norm(nrm_arr, axis=1, keepdims=True) + 1e-9
        nrm_arr = nrm_arr / lens
        return pts_arr, nrm_arr

    pts_arr = np.array(pts_list, dtype=np.float32)
    nrm_arr = np.array(nrm_list, dtype=np.float32)

    min_len = min(len(pts_arr), len(nrm_arr))
    pts_arr = pts_arr[:min_len]
    nrm_arr = nrm_arr[:min_len]

    replace = min_len < n_pts
    idx = rng.choice(min_len, n_pts, replace=replace)
    pts_arr = pts_arr[idx]
    nrm_arr = nrm_arr[idx]

    lens = np.linalg.norm(nrm_arr, axis=1, keepdims=True) + 1e-9
    nrm_arr = nrm_arr / lens
    return pts_arr, nrm_arr


def make_object_pcd_world(obj: dict, world_pos,
                           orientation_quat=None,
                           n_pts: int = 2048) -> o3d.geometry.PointCloud:
    """
    Build Open3D PCD for one object in world frame.
    BUG-1 FIX: orientation applied to both points AND normals.
    """
    local_pts, local_nrm = sample_object_surface(obj, n_pts)

    if orientation_quat is not None:
        rot = ScipyR.from_quat(orientation_quat).as_matrix().astype(np.float32)
        world_pts = (rot @ local_pts.T).T + np.array(world_pos, dtype=np.float32)
        world_nrm = (rot @ local_nrm.T).T
    else:
        world_pts = local_pts + np.array(world_pos, dtype=np.float32)
        world_nrm = local_nrm.copy()

    # BUG-4 FIX: float64 for Vector3dVector
    color = np.array(obj["color"][:3], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(world_pts.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(world_nrm.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(
        np.tile(color, (len(world_pts), 1)).astype(np.float64)
    )
    return pcd


# ── PyBullet helpers ─────────────────────────────────────────────────────────

def create_collision_shape(obj: dict) -> int:
    if obj["type"] == "cylinder":
        return pb.createCollisionShape(pb.GEOM_CYLINDER,
                                        radius=obj["radius"], height=obj["height"])
    return pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj["half"])


def create_visual_shape(obj: dict) -> int:
    r, g, b, a = obj["color"]
    if obj["type"] == "cylinder":
        return pb.createVisualShape(pb.GEOM_CYLINDER, radius=obj["radius"],
                                     length=obj["height"], rgbaColor=[r, g, b, a])
    return pb.createVisualShape(pb.GEOM_BOX, halfExtents=obj["half"],
                                 rgbaColor=[r, g, b, a])


def setup_pybullet_scene(positions: list) -> tuple:
    """Build headless PyBullet scene using pre-computed positions."""
    try:
        pb.disconnect()
    except Exception:
        pass
    client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")

    tc = pb.createCollisionShape(pb.GEOM_BOX,
                                  halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2])
    tv = pb.createVisualShape(pb.GEOM_BOX,
                               halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2],
                               rgbaColor=[0.65, 0.45, 0.25, 1.0])
    pb.createMultiBody(0, tc, tv, [0, 0, TABLE_HEIGHT / 2])

    # FIX: Load actual Franka Panda URDF with base mounted ON the table surface.
    # ARM_BASE_POS[2] is already TABLE_HEIGHT; robot base origin sits at that z,
    # so the robot stands upright on the table — not hanging in mid-air.
    pb.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=[ARM_BASE_POS[0], ARM_BASE_POS[1], TABLE_HEIGHT],
        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )

    object_ids = []
    for obj, pos in zip(OBJECTS, positions):
        col = create_collision_shape(obj)
        vis = create_visual_shape(obj)
        oid = pb.createMultiBody(obj["mass"], col, vis, list(pos))
        object_ids.append(oid)
        print(f"  [SPAWN] {obj['name']:<12} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
              f"  d_arm={dist_to_arm(pos[0], pos[1]):.3f}m")

    for _ in range(240):
        pb.stepSimulation()

    return client, object_ids


# ── Camera rendering ─────────────────────────────────────────────────────────

def get_camera_images(cam: dict) -> tuple:
    """Render one camera. Correct non-linear depth linearisation."""
    view_mat = pb.computeViewMatrix(
        cameraEyePosition    = cam["pos"],
        cameraTargetPosition = cam["target"],
        cameraUpVector       = cam["up"],
    )
    proj_mat = pb.computeProjectionMatrixFOV(
        fov     = FOV,
        aspect  = IMG_WIDTH / IMG_HEIGHT,
        nearVal = NEAR_PLANE,
        farVal  = FAR_PLANE,
    )
    _, _, rgb_raw, depth_raw, _ = pb.getCameraImage(
        width            = IMG_WIDTH,
        height           = IMG_HEIGHT,
        viewMatrix       = view_mat,
        projectionMatrix = proj_mat,
        renderer         = pb.ER_TINY_RENDERER,
    )
    rgb = np.array(rgb_raw, dtype=np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH, 4)[:, :, :3]
    depth_buf = np.array(depth_raw, dtype=np.float32).reshape(IMG_HEIGHT, IMG_WIDTH)
    depth_m = (FAR_PLANE * NEAR_PLANE) / (FAR_PLANE - (FAR_PLANE - NEAR_PLANE) * depth_buf)
    return rgb, depth_m, proj_mat, view_mat


def depth_to_pointcloud(depth_m: np.ndarray, rgb: np.ndarray,
                         proj_matrix, view_matrix) -> o3d.geometry.PointCloud:
    """
    Back-project pixel depth to 3-D world points.
    BUG-3 (verified correct): NDC pixel-centre formula uses half-pixel offset.
    """
    P = np.array(proj_matrix).reshape(4, 4).T
    V = np.array(view_matrix).reshape(4, 4).T
    f_x = float(P[0, 0])
    f_y = float(P[1, 1])
    H, W = depth_m.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ndc_x = (2.0 * xs + 1.0) / W - 1.0
    ndc_y = 1.0 - (2.0 * ys + 1.0) / H
    valid = depth_m < (FAR_PLANE - 0.05)
    X_c =  ndc_x * depth_m / f_x
    Y_c =  ndc_y * depth_m / f_y
    Z_c = -depth_m
    cam_pts_h = np.stack([X_c, Y_c, Z_c, np.ones_like(Z_c)], axis=-1)
    inv_view  = np.linalg.inv(V)
    world_pts = (cam_pts_h[valid] @ inv_view.T)[:, :3]
    colors    = rgb[valid].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # BUG-8 FIX: same voxel size for depth and analytic PCDs
    pcd = pcd.voxel_down_sample(voxel_size=0.004)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(str(path), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVE] {path}")


def viz_top_down(positions: list) -> None:
    """BUG-7 FIX: camera arrows use annotation_clip=False."""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.add_patch(plt.Rectangle((-TABLE_HALF_X, -TABLE_HALF_Y),
                                2 * TABLE_HALF_X, 2 * TABLE_HALF_Y,
                                lw=2, edgecolor="#e94560", facecolor="#0f3460", alpha=0.45))
    theta = np.linspace(0, 2 * np.pi, 300)
    for reach, col, lbl in [
        (ARM_MIN_REACH, "#f5a623", f"min {ARM_MIN_REACH:.2f}m"),
        (ARM_MAX_REACH, "#7ed957", f"max {ARM_MAX_REACH:.2f}m"),
    ]:
        ax.plot(ARM_BASE_POS[0] + reach * np.cos(theta),
                ARM_BASE_POS[1] + reach * np.sin(theta),
                "--", color=col, lw=1.2, alpha=0.7, label=lbl)
    ax.add_patch(plt.Circle((ARM_BASE_POS[0], ARM_BASE_POS[1]), 0.065,
                              color="#e94560", zorder=5))
    ax.text(ARM_BASE_POS[0] + 0.09, ARM_BASE_POS[1], "KUKA Base",
            color="#e94560", fontsize=8)
    for i, (obj, pos) in enumerate(zip(OBJECTS, positions)):
        fp = footprint_radius(obj)
        c  = obj["color"][:3]
        d  = dist_to_arm(pos[0], pos[1])
        ok = ARM_MIN_REACH <= d <= ARM_MAX_REACH
        ec = "white" if ok else "red"
        ax.add_patch(plt.Circle((pos[0], pos[1]), fp, color=c, alpha=0.85,
                                  zorder=4, edgecolor=ec, linewidth=1.5))
        ax.text(pos[0], pos[1] + fp + 0.018, obj["name"][:6],
                color="white", fontsize=6.5, ha="center", fontweight="bold")
        ax.text(pos[0], pos[1] - fp - 0.026, f"d={d:.2f}m",
                color="#aaaaaa", fontsize=5.5, ha="center")
    for cam in CAMERAS:
        cx, cy = cam["pos"][0], cam["pos"][1]
        ax.annotate("",
                    xy=(TABLE_CENTER[0], TABLE_CENTER[1]),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#00d4ff", lw=1.2),
                    annotation_clip=False)
        ax.text(cx, cy + 0.04, cam["name"], color="#00d4ff", fontsize=7,
                ha="center", clip_on=False)
    ax.set_xlim(-TABLE_HALF_X - 0.30, TABLE_HALF_X + 0.30)
    ax.set_ylim(-TABLE_HALF_Y - 0.40, TABLE_HALF_Y + 0.40)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title(f"Scene Top-Down (20 Objects) | Arm Reach + Cameras",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#111", labelcolor="white", fontsize=8, loc="upper right")
    _save(fig, "01_scene_top_down.png")


def viz_rgb_grid(cam_data: list) -> None:
    fig, axes = plt.subplots(1, len(cam_data), figsize=(4 * len(cam_data), 4),
                              facecolor="#111111")
    if len(cam_data) == 1:
        axes = [axes]
    for ax, (cname, rgb, _) in zip(axes, cam_data):
        ax.imshow(rgb)
        ax.set_title(cname, color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle("RGB renders — 20-object scene", color="white", fontweight="bold")
    _save(fig, "05_rgb_grid.png")


def viz_pcd_3d(pcd: o3d.geometry.PointCloud, title: str, fname: str) -> None:
    pts = np.asarray(pcd.points)
    col = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(pts), 3)) * 0.5
    fig = plt.figure(figsize=(9, 7), facecolor="#111111")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#111111")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=col, s=0.6, alpha=0.7, rasterized=True)
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    _save(fig, fname)


def viz_per_object_pcds(obj_pcds: list, positions: list) -> None:
    n = len(obj_pcds)
    cols = 5
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                              facecolor="#111111",
                              subplot_kw={"projection": "3d"})
    axes = np.array(axes).flatten()
    for i, (pcd, obj) in enumerate(zip(obj_pcds, OBJECTS)):
        ax = axes[i]
        ax.set_facecolor("#111111")
        pts = np.asarray(pcd.points)
        c   = obj["color"][:3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=[c], s=0.5, alpha=0.8, rasterized=True)
        ax.set_title(obj["name"], color="white", fontsize=7.5, fontweight="bold")
        ax.tick_params(colors="white", labelsize=5)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Per-Object Analytic PCDs (20 objects, 2048 pts each)",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "07_per_object_pcds.png")


def viz_bboxes(fused_pcd: o3d.geometry.PointCloud, positions: list) -> None:
    pts = np.asarray(fused_pcd.points)
    col = np.asarray(fused_pcd.colors) if fused_pcd.has_colors() \
        else np.ones((len(pts), 3)) * 0.5
    fig = plt.figure(figsize=(10, 8), facecolor="#111111")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#111111")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=col, s=0.4, alpha=0.6, rasterized=True)
    for obj, pos in zip(OBJECTS, positions):
        c = obj["color"][:3]
        if obj["type"] == "cylinder":
            r, h = obj["radius"], obj["height"]
            t_ = np.linspace(0, 2 * np.pi, 60)
            for z_off in [-h / 2, h / 2]:
                ax.plot(pos[0] + r * np.cos(t_),
                        pos[1] + r * np.sin(t_),
                        np.full(60, pos[2] + z_off), color=c, lw=0.8, alpha=0.8)
        else:
            hx, hy, hz = obj["half"]
            for xs in [pos[0] - hx, pos[0] + hx]:
                for ys in [pos[1] - hy, pos[1] + hy]:
                    ax.plot([xs, xs], [ys, ys],
                            [pos[2] - hz, pos[2] + hz], color=c, lw=0.8, alpha=0.8)
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.set_title("Fused PCD with Object Bounding Boxes", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    _save(fig, "08_bboxes.png")


def viz_reachability_table(positions: list) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#111111")
    ax.set_facecolor("#111111")
    names = [o["name"] for o in OBJECTS]
    dists = [dist_to_arm(p[0], p[1]) for p in positions]
    colors = ["#7ed957" if ARM_MIN_REACH <= d <= ARM_MAX_REACH else "#e94560"
              for d in dists]
    bars = ax.barh(names, dists, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(ARM_MIN_REACH, color="#f5a623", ls="--", lw=1.5, label=f"min {ARM_MIN_REACH}m")
    ax.axvline(ARM_MAX_REACH, color="#00d4ff", ls="--", lw=1.5, label=f"max {ARM_MAX_REACH}m")
    for bar, d in zip(bars, dists):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{d:.3f}m", va="center", color="white", fontsize=8)
    ax.set_xlabel("Distance to Arm Base (m)", color="white")
    ax.set_title(f"Arm Reachability — All 20 Objects", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
    _save(fig, "09_reachability.png")


def viz_depth_grid(cam_data: list) -> None:
    fig, axes = plt.subplots(1, len(cam_data), figsize=(4 * len(cam_data), 4),
                              facecolor="#111111")
    if len(cam_data) == 1:
        axes = [axes]
    for ax, (cname, _, depth_m) in zip(axes, cam_data):
        im = ax.imshow(depth_m, cmap="plasma", vmin=NEAR_PLANE, vmax=FAR_PLANE)
        plt.colorbar(im, ax=ax, label="m", fraction=0.046)
        ax.set_title(cname, color="white", fontsize=9)
        ax.axis("off")
    fig.suptitle("Linearized Depth Maps", color="white", fontweight="bold")
    fig.patch.set_facecolor("#111111")
    _save(fig, "10_depth_grid.png")


def viz_pcd_stats(pcd: o3d.geometry.PointCloud) -> None:
    pts = np.asarray(pcd.points)
    bb  = pcd.get_axis_aligned_bounding_box()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#111111")
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    colors_hist = ["#e94560", "#7ed957", "#00d4ff"]
    for i, (ax, lbl, col) in enumerate(zip(axes, axis_labels, colors_hist)):
        ax.set_facecolor("#16213e")
        ax.hist(pts[:, i], bins=80, color=col, edgecolor="none", alpha=0.85)
        ax.set_xlabel(lbl, color="white")
        ax.set_ylabel("Count", color="white")
        ax.set_title(f"Point distribution — {lbl}", color="white", fontsize=9)
        ax.tick_params(colors="white")
    fig.suptitle(f"Fused Analytic PCD Statistics ({len(pts):,} pts)",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "11_pcd_stats.png")


def viz_3d_overview(positions: list, fused_pcd: o3d.geometry.PointCloud) -> None:
    pts = np.asarray(fused_pcd.points)
    col = np.asarray(fused_pcd.colors) if fused_pcd.has_colors() \
        else np.ones((len(pts), 3)) * 0.5
    fig = plt.figure(figsize=(12, 9), facecolor="#111111")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")
    ax.scatter(pts[::2, 0], pts[::2, 1], pts[::2, 2],
               c=col[::2], s=0.3, alpha=0.55, rasterized=True)
    ax.scatter(*ARM_BASE_POS, c="#e94560", s=120, zorder=10, marker="^")
    for obj, pos in zip(OBJECTS, positions):
        c = obj["color"][:3]
        ax.text(pos[0], pos[1], pos[2] + 0.04, obj["name"][:4],
                color=c, fontsize=5.5, ha="center")
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.set_title("3-D Scene Overview — 20 Objects", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
    _save(fig, "12_3d_overview.png")


def viz_depth_vs_analytic(fused_depth: o3d.geometry.PointCloud,
                           fused_analytic: o3d.geometry.PointCloud) -> None:
    """
    BUG-5 FIX: vectorised compute_point_cloud_distance instead of KDTree loop.
    BUG-8: both PCDs use 0.004 m voxel size for fair comparison.
    """
    dists = np.asarray(fused_depth.compute_point_cloud_distance(fused_analytic))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#111111")
    for ax in axes:
        ax.set_facecolor("#16213e")
    axes[0].hist(dists, bins=100, color="#7ed957", edgecolor="none", alpha=0.85)
    axes[0].axvline(np.median(dists), color="white", ls="--",
                    label=f"median={np.median(dists)*1000:.2f}mm")
    axes[0].axvline(np.mean(dists), color="#f5a623", ls=":",
                    label=f"mean={np.mean(dists)*1000:.2f}mm")
    axes[0].set_xlabel("Distance depth→analytic (m)", color="white")
    axes[0].set_ylabel("Count", color="white")
    axes[0].set_title("Depth-to-Analytic Distance Histogram", color="white", fontsize=9)
    axes[0].tick_params(colors="white")
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=8)
    axes[1].boxplot(dists, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#00d4ff", alpha=0.6),
                    medianprops=dict(color="white"),
                    whiskerprops=dict(color="white"),
                    capprops=dict(color="white"),
                    flierprops=dict(markerfacecolor="#e94560", marker="o", markersize=2))
    axes[1].set_ylabel("Distance (m)", color="white")
    axes[1].set_title("Distance Boxplot", color="white", fontsize=9)
    axes[1].tick_params(colors="white")
    pct_under5mm = np.mean(dists < 0.005) * 100
    fig.suptitle(f"Depth vs Analytic PCD Accuracy\n"
                 f"{pct_under5mm:.1f}% of depth points within 5 mm of analytic",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "13_depth_vs_analytic.png")


def viz_spawn_verification(positions: list) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), facecolor="#111111")
    axes = axes.flatten()
    for i, (obj, pos) in enumerate(zip(OBJECTS, positions)):
        ax = axes[i]
        ax.set_facecolor("#16213e")
        fp = footprint_radius(obj)
        d  = dist_to_arm(pos[0], pos[1])
        ok = ARM_MIN_REACH <= d <= ARM_MAX_REACH
        c  = obj["color"][:3]
        ec = "#7ed957" if ok else "#e94560"
        ax.add_patch(plt.Circle((0, 0), fp, color=c, alpha=0.7, edgecolor=ec, lw=2))
        ax.set_xlim(-fp * 2, fp * 2)
        ax.set_ylim(-fp * 2, fp * 2)
        ax.set_aspect("equal")
        ax.set_title(f"{obj['name']}\nd={d:.3f}m {'✓' if ok else '✗'}",
                     color="white", fontsize=7.5, fontweight="bold")
        ax.tick_params(colors="white", labelsize=5)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Spawn Verification — 20 Objects (green=in-reach, red=out-of-reach)",
                 color="white", fontsize=10, fontweight="bold")
    _save(fig, "14_spawn_verification.png")


def viz_normal_verification(obj_pcds: list) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(18, 14), facecolor="#111111")
    axes = axes.flatten()
    for i, (pcd, obj) in enumerate(zip(obj_pcds, OBJECTS)):
        ax = axes[i]
        ax.set_facecolor("#0a0a1a")
        pts  = np.asarray(pcd.points)
        nrms = np.asarray(pcd.normals)
        idx  = np.random.choice(len(pts), min(60, len(pts)), replace=False)
        c    = obj["color"][:3]
        ax.scatter(pts[idx, 0], pts[idx, 2], c=[c], s=3, alpha=0.8)
        sc = 0.008
        for ii in idx:
            ax.annotate("", xy=(pts[ii, 0] + sc * nrms[ii, 0],
                                 pts[ii, 2] + sc * nrms[ii, 2]),
                        xytext=(pts[ii, 0], pts[ii, 2]),
                        arrowprops=dict(arrowstyle="-|>", color="#00d4ff",
                                        lw=0.5, mutation_scale=4),
                        annotation_clip=False)
        ax.set_title(obj["name"], color="white", fontsize=7.5, fontweight="bold")
        ax.set_xlabel("X", color="white", fontsize=6)
        ax.set_ylabel("Z", color="white", fontsize=6)
        ax.tick_params(colors="white", labelsize=5)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Surface Normals Verification (X-Z plane, 60 sampled)",
                 color="white", fontsize=10, fontweight="bold")
    _save(fig, "15_normal_verification.png")


def viz_pcd_coverage(fused_depth: o3d.geometry.PointCloud, positions: list) -> None:
    pts = np.asarray(fused_depth.points)
    mask = ((pts[:, 0] >= -TABLE_HALF_X) & (pts[:, 0] <= TABLE_HALF_X) &
            (pts[:, 1] >= -TABLE_HALF_Y) & (pts[:, 1] <= TABLE_HALF_Y))
    pts_t = pts[mask]
    cell = 0.005
    nx = int(2 * TABLE_HALF_X / cell) + 1
    ny = int(2 * TABLE_HALF_Y / cell) + 1
    grid = np.zeros((ny, nx), dtype=np.int32)
    xi = ((pts_t[:, 0] + TABLE_HALF_X) / cell).astype(int).clip(0, nx - 1)
    yi = ((pts_t[:, 1] + TABLE_HALF_Y) / cell).astype(int).clip(0, ny - 1)
    np.add.at(grid, (yi, xi), 1)
    coverage_pct = np.mean(grid > 0) * 100
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#111111")
    ax.set_facecolor("#000000")
    im = ax.imshow(grid, origin="lower", cmap="hot",
                   extent=[-TABLE_HALF_X, TABLE_HALF_X, -TABLE_HALF_Y, TABLE_HALF_Y],
                   aspect="equal")
    plt.colorbar(im, ax=ax, label="Point count")
    for obj, pos in zip(OBJECTS, positions):
        fp = footprint_radius(obj)
        c  = obj["color"][:3]
        ax.add_patch(plt.Circle((pos[0], pos[1]), fp, fill=False,
                                  edgecolor=c, lw=1.5, zorder=5))
        ax.text(pos[0], pos[1], obj["name"][:3], color=c,
                fontsize=5.5, ha="center", va="center", fontweight="bold")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")
    ax.set_title(
        f"Table-Plane PCD Coverage Map (5 mm grid) — 20 Objects\n"
        f"Coverage: {coverage_pct:.1f}% of table cells occupied",
        color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    _save(fig, "16_pcd_coverage.png")


def viz_camera_depth_rgb_comparison(cam_data: list) -> None:
    """
    Per-camera comparison: depth-only (plasma colourmap) vs RGB-coloured depth map.

    For each camera, renders a side-by-side panel:
      Left  — linearised depth map (plasma, no colour info)
      Right — RGB image tinted by depth: each pixel's RGB colour is multiplied
              by a depth-derived intensity so near objects are bright and far
              objects are dark, merging colour + depth in one image.

    Outputs one file per camera (17_depth_rgb_cam_*.png) plus a combined
    summary grid (17_depth_rgb_comparison_all.png).
    """
    for idx, (cname, rgb, depth_m) in enumerate(cam_data):
        # Normalise depth to [0, 1] for overlay (0 = near, 1 = far)
        d_norm = np.clip(
            (depth_m - NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE), 0.0, 1.0
        )
        # Depth-tinted RGB: near = bright, far = dark
        intensity = 1.0 - d_norm            # bright = near
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_depth = np.clip(rgb_f * intensity[:, :, np.newaxis], 0, 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#111111")
        for ax in axes:
            ax.set_facecolor("#111111")
            ax.axis("off")

        # Left: pure depth map
        im0 = axes[0].imshow(depth_m, cmap="plasma",
                             vmin=NEAR_PLANE, vmax=FAR_PLANE)
        axes[0].set_title("Depth Map (plasma)", color="white",
                           fontsize=10, fontweight="bold")
        cb = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb.set_label("metres", color="white")
        cb.ax.tick_params(colors="white")

        # Right: RGB coloured by depth
        axes[1].imshow(rgb_depth)
        axes[1].set_title("RGB × Depth Intensity\n(bright = near, dark = far)",
                           color="white", fontsize=10, fontweight="bold")

        fig.suptitle(f"Depth vs RGB+Depth — {cname}",
                     color="white", fontsize=12, fontweight="bold")
        _save(fig, f"17_depth_rgb_{cname}.png")

    # ── Combined summary grid (all cameras) ──────────────────────────────────
    n = len(cam_data)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), facecolor="#111111")
    if n == 1:
        axes = [axes]

    for row, (cname, rgb, depth_m) in enumerate(cam_data):
        d_norm = np.clip(
            (depth_m - NEAR_PLANE) / (FAR_PLANE - NEAR_PLANE), 0.0, 1.0
        )
        intensity = 1.0 - d_norm
        rgb_f = rgb.astype(np.float32) / 255.0
        rgb_depth = np.clip(rgb_f * intensity[:, :, np.newaxis], 0, 1)

        for ax in axes[row]:
            ax.set_facecolor("#111111")
            ax.axis("off")

        im = axes[row][0].imshow(depth_m, cmap="plasma",
                                 vmin=NEAR_PLANE, vmax=FAR_PLANE)
        axes[row][0].set_title(f"{cname} — Depth", color="white", fontsize=9)
        plt.colorbar(im, ax=axes[row][0], fraction=0.046, pad=0.04
                     ).ax.tick_params(colors="white")

        axes[row][1].imshow(rgb_depth)
        axes[row][1].set_title(f"{cname} — RGB+Depth", color="white", fontsize=9)

    fig.suptitle("All Cameras: Depth Map vs RGB-Coloured Depth",
                 color="white", fontsize=12, fontweight="bold")
    _save(fig, "17_depth_rgb_comparison_all.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sep = "=" * 66
    print(f"{sep}\nQ1 — Tabletop Visualization  [20-Object Extended | Research Grade]\n{sep}")

    # 1. Spawn positions
    print("\n[1] Computing spawn positions for 20 objects...")
    positions = generate_spawn_positions(OBJECTS, seed=42)
    all_ok = True
    for obj, pos in zip(OBJECTS, positions):
        d  = dist_to_arm(pos[0], pos[1])
        ok = ARM_MIN_REACH <= d <= ARM_MAX_REACH
        if not ok:
            all_ok = False
        print(f"    {obj['name']:<12} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
              f"  d={d:.3f}m {'OK' if ok else 'OUT-OF-REACH ← ERROR'}")
    if not all_ok:
        raise RuntimeError("One or more objects are out of arm reach — check spawn logic")
    print(f"    All {len(OBJECTS)} objects within reach. ✓")

    # 2. Top-down overview
    print("\n[2] Top-down map...")
    viz_top_down(positions)

    # 3. PyBullet scene
    print("\n[3] Building PyBullet scene (DIRECT/headless)...")
    client, object_ids = setup_pybullet_scene(positions)

    # 4. Camera rendering
    print("\n[4] Camera rendering + depth linearisation + PCD back-projection...")
    cam_data: list = []
    cam_pcds: list = []
    for idx, cam in enumerate(CAMERAS):
        print(f"    {cam['name']}")
        rgb, depth_m, proj, view = get_camera_images(cam)
        rgb_name = f"02_cam{idx+1}_{cam['name']}_rgb.png"
        Image.fromarray(rgb).save(str(OUTPUT_DIR / rgb_name))
        print(f"      [SAVE] {rgb_name}")
        fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor="#111111")
        ax2.set_facecolor("#111111")
        im2 = ax2.imshow(depth_m, cmap="plasma", vmin=NEAR_PLANE, vmax=FAR_PLANE)
        plt.colorbar(im2, ax=ax2, label="metres")
        ax2.axis("off")
        ax2.set_title(f"Linearized Depth — {cam['name']}", color="white")
        depth_name = f"03_cam{idx+1}_{cam['name']}_depth.png"
        fig2.savefig(str(OUTPUT_DIR / depth_name), dpi=120, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        print(f"      [SAVE] {depth_name}")
        pcd = depth_to_pointcloud(depth_m, rgb, proj, view)
        cam_pcds.append(pcd)
        ply_name = f"cam_{cam['name']}.ply"
        o3d.io.write_point_cloud(str(OUTPUT_DIR / ply_name), pcd)
        print(f"      [SAVE] {ply_name}  ({len(pcd.points)} pts)")
        viz_pcd_3d(pcd, f"PCD — {cam['name']}  ({len(pcd.points)} pts)",
                   f"04_pcd_{cam['name']}.png")
        cam_data.append((cam["name"], rgb, depth_m))

    # 5. RGB grid
    print("\n[5] RGB grid...")
    viz_rgb_grid(cam_data)

    # 6. Multi-camera depth fusion
    print("\n[6] Fusing multi-camera depth PCDs...")
    fused_depth = o3d.geometry.PointCloud()
    for p in cam_pcds:
        fused_depth += p
    # BUG-8 FIX: 0.004 m matches analytic voxel size
    fused_depth = fused_depth.voxel_down_sample(voxel_size=0.004)
    fused_depth, _ = fused_depth.remove_statistical_outlier(nb_neighbors=25, std_ratio=2.0)
    fused_depth.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=30))
    o3d.io.write_point_cloud(str(OUTPUT_DIR / "fused_depth_scene.ply"), fused_depth)
    print(f"    Depth-fused PCD: {len(fused_depth.points)} pts")

    # 7. Per-object analytic PCDs
    print("\n[7] Per-object analytic PCDs (20 objects)...")
    obj_pcds: list = []
    for obj, pos in zip(OBJECTS, positions):
        pcd = make_object_pcd_world(obj, pos, orientation_quat=None, n_pts=2048)
        obj_pcds.append(pcd)
        ply_name = f"obj_{obj['name'].lower()}.ply"
        o3d.io.write_point_cloud(str(OUTPUT_DIR / ply_name), pcd)
        print(f"    {obj['name']:<12} analytic PCD: {len(pcd.points)} pts → {ply_name}")

    fused_analytic = o3d.geometry.PointCloud()
    for p in obj_pcds:
        fused_analytic += p
    fused_analytic = fused_analytic.voxel_down_sample(voxel_size=0.004)
    fused_analytic.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    o3d.io.write_point_cloud(str(OUTPUT_DIR / "fused_scene.ply"), fused_analytic)
    print(f"    Analytic fused PCD: {len(fused_analytic.points)} pts")

    # 8. Depth vs Analytic
    print("\n[8] Depth vs Analytic comparison...")
    viz_depth_vs_analytic(fused_depth, fused_analytic)

    # 9. Core visualisations
    print("\n[9] Core visualisations...")
    viz_per_object_pcds(obj_pcds, positions)
    viz_pcd_3d(fused_analytic,
               f"Fused Analytic PCD ({len(fused_analytic.points)} pts)", "06_fused_pcd.png")
    viz_bboxes(fused_analytic, positions)
    viz_reachability_table(positions)
    viz_depth_grid(cam_data)
    viz_pcd_stats(fused_analytic)
    viz_3d_overview(positions, fused_analytic)

    # 10. Verification visualisations
    print("\n[10] Verification visualisations...")
    viz_spawn_verification(positions)
    viz_normal_verification(obj_pcds)
    viz_pcd_coverage(fused_depth, positions)

    # 11. Per-camera depth vs RGB+depth comparison
    print("\n[11] Per-camera depth vs RGB+depth comparison...")
    viz_camera_depth_rgb_comparison(cam_data)

    # 12. Summary
    with open(str(OUTPUT_DIR / "q1_summary.txt"), "w") as f:
        f.write("Q1 SUMMARY — 20-Object Extended, Research Grade\n")
        f.write("=" * 55 + "\n\n")
        f.write("CHANGES FROM PREVIOUS VERSION:\n")
        f.write("  CHANGE-1: Objects 7 → 20 (13 new added)\n")
        f.write("  CHANGE-2: TABLE_HALF_X/Y enlarged (0.65, 0.55)\n")
        f.write("  CHANGE-3: sample_object_surface generalised\n\n")
        f.write("ALL BUG FIXES PRESERVED:\n")
        f.write("  BUG-1: make_object_pcd_world rotation on pts+normals\n")
        f.write("  BUG-4: float64 for Vector3dVector\n")
        f.write("  BUG-5: vectorised compute_point_cloud_distance\n")
        f.write("  BUG-7: annotation_clip=False for camera arrows\n")
        f.write("  BUG-8: 0.004 m voxel parity (depth & analytic)\n\n")
        f.write("NEW FIXES IN THIS VERSION:\n")
        f.write("  FIX-A: Removed rogue top-level pb.connect(GUI) + loadURDF\n")
        f.write("          that ran at import time outside setup_pybullet_scene.\n")
        f.write("  FIX-B: Robot arm uses actual franka_panda/panda.urdf URDF\n")
        f.write("          (useFixedBase=True) placed at table edge z=TABLE_HEIGHT.\n")
        f.write("          Previous code used a plain cylinder stub.\n")
        f.write("  FIX-C: Added viz_camera_depth_rgb_comparison — per-camera\n")
        f.write("          depth map vs RGB-tinted depth comparison panels.\n\n")
        f.write("SCENE STATISTICS:\n")
        f.write(f"  Objects          : {len(OBJECTS)}\n")
        f.write(f"  Depth-fused PCD  : {len(fused_depth.points)} pts\n")
        f.write(f"  Analytic fused   : {len(fused_analytic.points)} pts\n\n")
        f.write("OBJECT POSITIONS (seed=42):\n")
        for obj, pos in zip(OBJECTS, positions):
            d = dist_to_arm(pos[0], pos[1])
            ok = ARM_MIN_REACH <= d <= ARM_MAX_REACH
            f.write(f"  {obj['name']:<12} pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})"
                    f"  d={d:.3f}m  {'OK' if ok else 'FAIL'}\n")
    print("  [SAVE] q1_summary.txt")

    pb.disconnect()
    print(f"\n{sep}\nQ1 COMPLETE — outputs in: {OUTPUT_DIR.resolve()}\n{sep}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name:<55} {f.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
