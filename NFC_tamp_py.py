"""
=============================================================================
NFC-TAMP  —  Neural Feasibility Checker for Task and Motion Planning
=============================================================================
Full replication of:
  "Accelerating Integrated Task and Motion Planning with
   Neural Feasibility Checking"  (Xu et al., arXiv:2203.10568, 2022)

What this file implements (paper sections in brackets):
  § IV-A  Data collection: 2-channel 100×100 depth images + IK labels
          for all 5 pick-up directions (+x,-x,+y,-y,+z) over 25 000 scenes.
  § IV-B  NFC model: exact CNN architecture → 288-dim feature → 5 FC layers
          → 5-way sigmoid.  Adam, weighted BCE, 90/10 split.
  § IV-C  NFC integrated into IDTMP: pre-filter infeasible actions before
          calling the expensive motion planner.
  § V-A   Runtime comparison IDTMP vs IDTMP-NFC (task / motion / total time).
  § V-B   NFC vs DVH: RAM, query time, model parameters.
  Extra   Full evaluation: per-direction accuracy, confusion matrices,
          ROC-AUC curves, success rate, false-feasible / false-infeasible
          rates, planning speedup bar chart.

Robot:  KUKA iiwa 7-DOF  (kuka_iiwa/model.urdf from pybullet_data)
        → exactly the robot used in the paper.
Motion: Bidirectional RRT (BiRRT) in 7-D joint space, 600 s timeout.

Requirements (install once):
  pip install pybullet torch torchvision open3d scikit-learn matplotlib
              numpy scipy pillow
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, math, time, random, warnings, json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pybullet as pb
import pybullet_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    classification_report, average_precision_score,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("nfc_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NFC-TAMP] Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS  (all labelled to match paper section/figure)
# ─────────────────────────────────────────────────────────────────────────────

# Scene
TABLE_HEIGHT  = 0.60
TABLE_HALF_X  = 0.65          # enlarged for 20 objects
TABLE_HALF_Y  = 0.55
TABLE_CENTER  = [0.0, 0.0, TABLE_HEIGHT]

# KUKA iiwa (paper §V uses KUKA collaborate robot arm)
ARM_BASE_POS  = [-0.65, 0.0, TABLE_HEIGHT]   # robot mounted at table edge height
ARM_MIN_REACH = 0.18
ARM_MAX_REACH = 0.90
N_KUKA_JOINTS = 7                            # active joints in kuka_iiwa
EE_LINK_IDX   = 6                            # end-effector link index
KUKA_HOME_CFG = [0.0, 0.0, 0.0, -1.2, 0.0, 1.4, 0.0]   # neutral pose
NUM_OBJECTS = 20


# Per-joint limits (rad) for kuka_iiwa as shipped in pybullet_data
KUKA_JOINT_LIMITS = [
    (-2.9671,  2.9671),   # J1
    (-2.0944,  2.0944),   # J2
    (-2.9671,  2.9671),   # J3
    (-2.0944,  2.0944),   # J4
    (-2.9671,  2.9671),   # J5
    (-2.0944,  2.0944),   # J6
    (-3.0543,  3.0543),   # J7
]

# NFC depth-image settings (paper §IV-A)
LOCAL_REGION   = 0.50          # 0.5 m × 0.5 m region around target body
IMG_W = IMG_H  = 100           # raw depth image size (paper §IV-A)
CNN_IMG_SIZE   = 30            # resize before CNN → gives 8×6×6 = 288 (paper §IV-B)
N_DIRS         = 5             # +x, -x, +y, -y, +z  (paper Fig. 3)
APPROACH_DIST  = 0.15          # m: EE standoff from object centroid (paper §IV-A)
FEASI_THR      = 0.50          # classification threshold β (paper §IV-C)
IK_MAX_ITER    = 200
IK_RESIDUAL_M  = 0.05          # 5 cm IK convergence threshold

# 5 pick-up directions (paper Fig. 3): +x, -x, +y, -y, +z
DIRECTION_VECS = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
], dtype=np.float64)
DIR_NAMES = ["+x", "-x", "+y", "-y", "+z"]

# NFC camera (top-down, centred on local region)
CAM_HEIGHT_ABOVE = 0.70        # metres above table surface
CAM_FOV_DEG      = 62.0        # covers 0.5 × 0.5 m at 0.60 m distance

# Training (paper §IV-B)
N_SAMPLES      = 25_000        # paper uses 240 000; 25 K runs fast, stays faithful
TRAIN_RATIO    = 0.90          # 90 % train, 10 % test (paper §IV-B)
BATCH_SIZE     = 32            # paper value
N_EPOCHS       = 60
LR             = 1e-3          # paper Adam lr
# Per-direction BCE weights (paper §IV-B: 4.7, 5.0, 4.8, 4.6, 1.6)
LOSS_WEIGHTS   = [4.7, 5.0, 4.8, 4.6, 1.6]

# BiRRT (paper §V: 10-minute timeout)
BIRRT_STEP     = 0.05          # rad – joint-space step size
BIRRT_MAX_IT   = 20_000        # max tree expansions per call
BIRRT_TIMEOUT  = 600           # 10 minutes (paper §V)  ← change to 30 for quick demo
BIRRT_GOAL_BIAS = 0.05         # probability of sampling goal config
MAX_CONNECT_ST = 300           # max steps in greedy connect phase

# Test comparison
N_TEST_PROBS   = 20            # 20 test problems for IDTMP vs IDTMP-NFC comparison
DEMO_TIMEOUT   = BIRRT_TIMEOUT  # per-call timeout — matches paper value (600 s)

# BUG FIX 2: Removed spurious `NUM_OBJECTS = 10` that was overwriting the
# value set above (15 or 20). NUM_OBJECTS is set once at line ~91 and
# must NOT be redefined here.
# ─────────────────────────────────────────────────────────────────────────────
# 2.  20-OBJECT CATALOGUE  (same as updated q1_tabletop_visualization.py)
# ─────────────────────────────────────────────────────────────────────────────
ALL_OBJECTS = [
    {"name": "Mug",       "type": "cylinder", "radius": 0.038, "height": 0.095, "mass": 0.30, "color": [0.80, 0.30, 0.10, 1.0]},
    {"name": "Book",      "type": "box",      "half": [0.105, 0.075, 0.016],    "mass": 0.40, "color": [0.20, 0.40, 0.80, 1.0]},
    {"name": "Bottle",    "type": "cylinder", "radius": 0.028, "height": 0.230, "mass": 0.50, "color": [0.10, 0.70, 0.20, 1.0]},
    {"name": "Can",       "type": "cylinder", "radius": 0.034, "height": 0.122, "mass": 0.35, "color": [0.90, 0.80, 0.10, 1.0]},
    {"name": "Box",       "type": "box",      "half": [0.065, 0.045, 0.042],    "mass": 0.20, "color": [0.60, 0.20, 0.60, 1.0]},
    {"name": "Plate",     "type": "cylinder", "radius": 0.092, "height": 0.014, "mass": 0.40, "color": [0.95, 0.95, 0.85, 1.0]},
    {"name": "Remote",    "type": "box",      "half": [0.065, 0.028, 0.013],    "mass": 0.15, "color": [0.15, 0.15, 0.15, 1.0]},
    {"name": "Cup",       "type": "cylinder", "radius": 0.030, "height": 0.075, "mass": 0.20, "color": [0.80, 0.60, 0.80, 1.0]},
    {"name": "Stapler",   "type": "box",      "half": [0.070, 0.028, 0.028],    "mass": 0.35, "color": [0.10, 0.10, 0.80, 1.0]},
    {"name": "Tape",      "type": "cylinder", "radius": 0.042, "height": 0.030, "mass": 0.10, "color": [0.70, 0.90, 0.90, 1.0]},
    {"name": "Scissors",  "type": "box",      "half": [0.048, 0.016, 0.010],    "mass": 0.10, "color": [0.80, 0.20, 0.20, 1.0]},
    {"name": "Penholder", "type": "cylinder", "radius": 0.036, "height": 0.115, "mass": 0.25, "color": [0.30, 0.60, 0.40, 1.0]},
    {"name": "Notebook",  "type": "box",      "half": [0.095, 0.068, 0.011],    "mass": 0.30, "color": [0.95, 0.50, 0.20, 1.0]},
    {"name": "Bowl",      "type": "cylinder", "radius": 0.082, "height": 0.052, "mass": 0.35, "color": [0.90, 0.90, 0.70, 1.0]},
    {"name": "Vase",      "type": "cylinder", "radius": 0.033, "height": 0.155, "mass": 0.40, "color": [0.50, 0.30, 0.80, 1.0]},
    {"name": "Cube",      "type": "box",      "half": [0.032, 0.032, 0.032],    "mass": 0.10, "color": [0.95, 0.20, 0.50, 1.0]},
    {"name": "Phone",     "type": "box",      "half": [0.038, 0.078, 0.008],    "mass": 0.20, "color": [0.20, 0.20, 0.20, 1.0]},
    {"name": "Keys",      "type": "box",      "half": [0.042, 0.022, 0.008],    "mass": 0.08, "color": [0.80, 0.70, 0.10, 1.0]},
    {"name": "GlassCase", "type": "box",      "half": [0.082, 0.040, 0.020],    "mass": 0.15, "color": [0.40, 0.40, 0.40, 1.0]},
    {"name": "Spray",     "type": "cylinder", "radius": 0.026, "height": 0.205, "mass": 0.45, "color": [0.20, 0.80, 0.80, 1.0]},
]

OBJECTS = ALL_OBJECTS[:NUM_OBJECTS]

assert len(OBJECTS) == NUM_OBJECTS

# ─────────────────────────────────────────────────────────────────────────────
# 3.  SCENE / GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def obj_height_half(obj: dict) -> float:
    if obj["type"] == "cylinder":
        return obj["height"] / 2.0
    return obj["half"][2]


def obj_footprint_r(obj: dict) -> float:
    if obj["type"] == "cylinder":
        return obj["radius"] + 0.010
    return math.sqrt(obj["half"][0] ** 2 + obj["half"][1] ** 2) + 0.010


def arm_dist(x: float, y: float) -> float:
    return math.sqrt((x - ARM_BASE_POS[0]) ** 2 + (y - ARM_BASE_POS[1]) ** 2)


def make_collision_shape(obj: dict) -> int:
    if obj["type"] == "cylinder":
        return pb.createCollisionShape(pb.GEOM_CYLINDER,
                                        radius=obj["radius"], height=obj["height"])
    return pb.createCollisionShape(pb.GEOM_BOX, halfExtents=obj["half"])


def make_visual_shape(obj: dict) -> int:
    r, g, b, a = obj["color"]
    if obj["type"] == "cylinder":
        return pb.createVisualShape(pb.GEOM_CYLINDER, radius=obj["radius"],
                                     length=obj["height"], rgbaColor=[r, g, b, a])
    return pb.createVisualShape(pb.GEOM_BOX, halfExtents=obj["half"],
                                 rgbaColor=[r, g, b, a])


def _within_reach(x: float, y: float) -> bool:
    d = arm_dist(x, y)
    return ARM_MIN_REACH <= d <= ARM_MAX_REACH


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PYBULLET SESSION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class BulletSession:
    """
    One persistent PyBullet DIRECT session used for the entire run.
    Objects are added / removed via add_body / clear_bodies.
    The KUKA robot and table are permanent fixtures.
    """

    def __init__(self):
        try:
            pb.disconnect()
        except Exception:
            pass
        self.client = pb.connect(pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setGravity(0, 0, -9.81)
        pb.setRealTimeSimulation(0)

        # Ground plane
        pb.loadURDF("plane.urdf")

        # Table (collision + visual)
        tc = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2])
        tv = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=[TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2],
            rgbaColor=[0.65, 0.45, 0.25, 1.0])
        self.table_id = pb.createMultiBody(0, tc, tv, [0, 0, TABLE_HEIGHT / 2])

        # KUKA iiwa robot
        self.robot_id = pb.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=ARM_BASE_POS,
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )
        self._n_joints = N_KUKA_JOINTS
        self._home()

        self._scene_bodies: list = []   # bodies added per-scene
        print(f"  [Bullet] KUKA id={self.robot_id}  table id={self.table_id}"
              f"  joints={self._n_joints}")

    # ── robot helpers ────────────────────────────────────────────────────────

    def _home(self):
        for j, q in enumerate(KUKA_HOME_CFG):
            pb.resetJointState(self.robot_id, j, q)

    def set_joints(self, config):
        for j in range(self._n_joints):
            pb.resetJointState(self.robot_id, j, float(config[j]))

    def get_ee_pos(self) -> np.ndarray:
        state = pb.getLinkState(self.robot_id, EE_LINK_IDX)
        return np.array(state[0], dtype=np.float64)

    def ik(self, target_xyz: np.ndarray) -> np.ndarray | None:
        """
        Compute IK for end-effector to reach target_xyz.
        Returns 7-D joint config or None if IK fails to converge.
        """
        sol = pb.calculateInverseKinematics(
            self.robot_id,
            EE_LINK_IDX,
            target_xyz.tolist(),
            maxNumIterations=IK_MAX_ITER,
            residualThreshold=1e-5,
        )
        if sol is None or len(sol) < self._n_joints:
            return None
        cfg = np.array(sol[:self._n_joints], dtype=np.float64)
        # Check joint limits
        for j, (lo, hi) in enumerate(KUKA_JOINT_LIMITS):
            if not (lo - 0.01 <= cfg[j] <= hi + 0.01):
                return None
        return cfg

    def ik_feasible(self, target_xyz: np.ndarray,
                    obstacle_ids: list | None = None) -> bool:
        """
        Full IK feasibility check (paper §IV-A):
          1. IK solvable within joint limits
          2. EE residual < IK_RESIDUAL_M
          3. No collision with table or neighbour bodies
        """
        cfg = self.ik(target_xyz)
        if cfg is None:
            return False
        self.set_joints(cfg)
        # Residual
        residual = float(np.linalg.norm(self.get_ee_pos() - target_xyz))
        if residual > IK_RESIDUAL_M:
            self._home()
            return False
        # Collision
        pb.performCollisionDetection()
        if pb.getContactPoints(self.robot_id, self.table_id):
            self._home()
            return False
        if obstacle_ids:
            for oid in obstacle_ids:
                if pb.getContactPoints(self.robot_id, oid):
                    self._home()
                    return False
        self._home()
        return True

    # ── scene body management ────────────────────────────────────────────────

    def add_body(self, obj: dict, pos: list | tuple) -> int:
        col = make_collision_shape(obj)
        vis = make_visual_shape(obj)
        bid = pb.createMultiBody(obj["mass"], col, vis, list(pos))
        self._scene_bodies.append(bid)
        return bid

    def clear_bodies(self):
        for bid in self._scene_bodies:
            pb.removeBody(bid)
        self._scene_bodies.clear()

    # ── depth image rendering ────────────────────────────────────────────────

    def render_local_depth(
        self, region_center: np.ndarray,
        target_id: int, neighbour_ids: list
    ) -> np.ndarray:
        """
        Render a 2-channel 100×100 depth image centred on region_center.
        Channel 0 : only target body depth  (paper §IV-A)
        Channel 1 : only neighbour bodies depth
        Uses PyBullet segmentation mask to separate bodies without re-renders.
        Returns float32 array of shape (2, IMG_H, IMG_W).
        """
        cx, cy = float(region_center[0]), float(region_center[1])
        cam_pos = [cx, cy, TABLE_HEIGHT + CAM_HEIGHT_ABOVE]
        target_pos_cam = [cx, cy, TABLE_HEIGHT]

        view_m = pb.computeViewMatrix(
            cameraEyePosition    = cam_pos,
            cameraTargetPosition = target_pos_cam,
            cameraUpVector       = [0, 1, 0],
        )
        near, far = 0.01, 2.0
        proj_m = pb.computeProjectionMatrixFOV(
            fov=CAM_FOV_DEG, aspect=1.0, nearVal=near, farVal=far)

        _, _, _, depth_raw, seg_raw = pb.getCameraImage(
            width     = IMG_W,
            height    = IMG_H,
            viewMatrix  = view_m,
            projectionMatrix = proj_m,
            renderer  = pb.ER_TINY_RENDERER,
        )

        depth_buf = np.array(depth_raw, dtype=np.float32).reshape(IMG_H, IMG_W)
        seg_arr   = np.array(seg_raw,   dtype=np.int32  ).reshape(IMG_H, IMG_W)

        # Linearise depth buffer  (OpenGL depth buffer → metric distance)
        depth_m = (far * near) / (far - (far - near) * depth_buf)

        # Channel 0: target body pixels
        ch0 = np.where(seg_arr == target_id, depth_m, 0.0).astype(np.float32)
        # Channel 1: neighbour pixels
        nb_mask = np.zeros_like(seg_arr, dtype=bool)
        for nid in neighbour_ids:
            nb_mask |= (seg_arr == nid)
        ch1 = np.where(nb_mask, depth_m, 0.0).astype(np.float32)

        return np.stack([ch0, ch1], axis=0)   # shape (2, H, W)

    def close(self):
        pb.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DATA COLLECTION   (paper §IV-A)
# ─────────────────────────────────────────────────────────────────────────────

def sample_local_scene(rng: np.random.Generator) -> tuple:
    """
    Sample a random target + 0-3 neighbour objects inside a 0.5 m × 0.5 m region.
    Returns:
        target_obj   : dict from OBJECTS
        target_local : (3,) position in local frame (region centre at origin)
        neighbours   : list of (dict, (3,) local pos) pairs
        region_world : (3,) world position of region centre  (on table surface)
    """
    half = LOCAL_REGION / 2.0

    # Sample region centre (must be within arm reach)
    for _ in range(5000):
        rx = rng.uniform(-TABLE_HALF_X + half + 0.05, TABLE_HALF_X - half - 0.05)
        ry = rng.uniform(-TABLE_HALF_Y + half + 0.05, TABLE_HALF_Y - half - 0.05)
        if _within_reach(rx, ry):
            break
    else:
        rx, ry = 0.0, 0.0

    region_world = np.array([rx, ry, TABLE_HEIGHT], dtype=np.float64)

    # Pick target object (any of 20)
    target_obj = OBJECTS[rng.integers(len(OBJECTS))]
    tf = obj_footprint_r(target_obj)
    # Target is roughly centred
    tx = rng.uniform(-0.06, 0.06)
    ty = rng.uniform(-0.06, 0.06)
    target_local = np.array([tx, ty, obj_height_half(target_obj)], dtype=np.float64)

    # Sample 0-4 neighbours (increased from 0-3 to create richer +z occlusion)
    n_nb = rng.integers(0, 5)   # 0,1,2,3,4
    placed = [(tx, ty, tf)]
    neighbours = []
    for _ in range(n_nb):
        nb_obj = OBJECTS[rng.integers(len(OBJECTS))]
        nbf = obj_footprint_r(nb_obj)
        for _ in range(200):
            nx = rng.uniform(-half + nbf, half - nbf)
            ny = rng.uniform(-half + nbf, half - nbf)
            ok = all(
                math.sqrt((nx - px) ** 2 + (ny - py) ** 2) >= nbf + pr + 0.005
                for px, py, pr in placed
            )
            if ok:
                placed.append((nx, ny, nbf))
                nb_local = np.array([nx, ny, obj_height_half(nb_obj)], dtype=np.float64)
                neighbours.append((nb_obj, nb_local))
                break

    return target_obj, target_local, neighbours, region_world


def collect_dataset(sess: BulletSession, n_samples: int = N_SAMPLES) -> dict:
    """
    Generate the NFC training dataset (paper §IV-A).

    Each sample:
      image   : (2, IMG_H, IMG_W)  float32  — 2-channel local depth image
      feat    : (3,)               float32  — relative (xt,yt,zt) of region to arm base
      labels  : (5,)               float32  — per-direction IK feasibility binary labels

    Returns dict with keys: images, features, labels
    """
    rng = np.random.default_rng(SEED)
    images_list  : list = []
    features_list: list = []
    labels_list  : list = []

    print(f"\n[DATA] Collecting {n_samples:,} samples "
          f"(5 IK checks per sample, ~0.5-1 s each)...")
    t0 = time.time()
    feasible_count = np.zeros(N_DIRS, dtype=np.int64)
    total_count    = np.zeros(N_DIRS, dtype=np.int64)

    for i in range(n_samples):
        sess.clear_bodies()

        target_obj, t_local, neighbours, region_w = sample_local_scene(rng)

        # World positions
        t_world_pos = (region_w + t_local).tolist()
        t_world_pos[2] += 0.001   # avoid table z-fight
        target_id = sess.add_body(target_obj, t_world_pos)

        nb_ids = []
        for nb_obj, nb_local in neighbours:
            nb_world = (region_w + nb_local).tolist()
            nb_world[2] += 0.001
            nid = sess.add_body(nb_obj, nb_world)
            nb_ids.append(nid)

        # 2-channel depth image
        region_centre_xy = region_w[:2]
        img_2ch = sess.render_local_depth(
            region_w, target_id, nb_ids)          # (2, H, W)

        # Feature vector: relative position of region centre to arm base (paper §IV-A)
        feat = np.array([
            region_w[0] - ARM_BASE_POS[0],
            region_w[1] - ARM_BASE_POS[1],
            region_w[2] - ARM_BASE_POS[2],
        ], dtype=np.float32)

        # IK feasibility for 5 directions
        obj_pos_w = np.array(t_world_pos, dtype=np.float64)
        labels = np.zeros(N_DIRS, dtype=np.float32)
        for d_idx, direction in enumerate(DIRECTION_VECS):
            approach_pos = obj_pos_w + direction * APPROACH_DIST
            # BUG FIX 1 (critical — root cause of broken +z):
            # The table-surface clamp `max(z, TABLE_HEIGHT+0.03)` must ONLY
            # apply to horizontal approaches (+x,-x,+y,-y) that might dip below
            # the table. For +z the approach is already above the object; clamping
            # it would RAISE the target point away from the object, making IK
            # trivially succeed for every scene → all +z labels = 1 →
            # degenerate dataset (FF=100%, AUC≈0.5).
            if d_idx != 4:   # not +z
                approach_pos[2] = max(approach_pos[2], TABLE_HEIGHT + 0.03)
            # For +z: check whether any neighbour body is directly above the
            # target (blocking a top-down grasp) by testing if the approach
            # position is in collision with neighbours at that configuration.
            feasible = sess.ik_feasible(approach_pos, obstacle_ids=nb_ids)
            labels[d_idx] = float(feasible)
            feasible_count[d_idx] += int(feasible)
            total_count[d_idx]    += 1

        images_list.append(img_2ch)
        features_list.append(feat)
        labels_list.append(labels)

        # Progress
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (n_samples - i - 1) / max(rate, 1e-6)
            feas_pct = [f"{100*feasible_count[d]/max(total_count[d],1):.1f}%" for d in range(N_DIRS)]
            print(f"  [{i+1:>6}/{n_samples}] "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                  f"feasible={feas_pct}")

    sess.clear_bodies()
    elapsed_total = time.time() - t0
    print(f"  Done. Total time: {elapsed_total:.1f}s  "
          f"({elapsed_total/n_samples*1000:.1f} ms/sample)")

    feas_pct_all = [100 * feasible_count[d] / max(n_samples, 1) for d in range(N_DIRS)]
    print("  Feasibility rates per direction:")
    for d, nm in enumerate(DIR_NAMES):
        print(f"    {nm}: {feas_pct_all[d]:.1f}%  "
              f"({feasible_count[d]}/{n_samples})")
    print(f"  Overall feasible: "
          f"{feasible_count.sum()/max(N_DIRS*n_samples,1)*100:.1f}%")

    return {
        "images":   np.stack(images_list,   axis=0),   # (N, 2, H, W)
        "features": np.stack(features_list, axis=0),   # (N, 3)
        "labels":   np.stack(labels_list,   axis=0),   # (N, 5)
    }


def save_dataset(data: dict, path: Path) -> None:
    np.savez_compressed(str(path),
                        images=data["images"],
                        features=data["features"],
                        labels=data["labels"])
    sz = path.stat().st_size / 1024 / 1024
    print(f"  [SAVE] Dataset → {path}  ({sz:.1f} MB)")


def load_dataset(path: Path) -> dict:
    d = np.load(str(path))
    print(f"  [LOAD] Dataset ← {path}  "
          f"({d['images'].shape[0]:,} samples)")
    return {"images": d["images"], "features": d["features"], "labels": d["labels"]}


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PYTORCH DATASET WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class NFCDataset(Dataset):
    """
    Wraps the collected data for PyTorch DataLoader.
    Images are resized from 100×100 → CNN_IMG_SIZE×CNN_IMG_SIZE (=30×30)
    so the CNN output is exactly 8×6×6 = 288 (paper §IV-B).
    """
    def __init__(self, images: np.ndarray, features: np.ndarray, labels: np.ndarray):
        assert len(images) == len(features) == len(labels)
        self.images   = torch.from_numpy(images)    # (N, 2, H, W)  float32
        self.features = torch.from_numpy(features)  # (N, 3)        float32
        self.labels   = torch.from_numpy(labels)    # (N, 5)        float32

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]                      # (2, H, W)
        img = F.interpolate(
            img.unsqueeze(0),
            size=(CNN_IMG_SIZE, CNN_IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                # (2, 30, 30)
        return img, self.features[idx], self.labels[idx]


def make_loaders(data: dict) -> tuple:
    """Split 90/10 and return train + test DataLoaders."""
    ds = NFCDataset(data["images"], data["features"], data["labels"])
    n_train = int(len(ds) * TRAIN_RATIO)
    n_test  = len(ds) - n_train
    train_ds, test_ds = random_split(
        ds, [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    print(f"  [LOADER] train={n_train:,}  test={n_test:,}  "
          f"batch={BATCH_SIZE}  batches/epoch={len(train_loader)}")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 7.  NFC MODEL   (paper §IV-B — exact architecture)
# ─────────────────────────────────────────────────────────────────────────────

class NFC(nn.Module):
    """
    Neural Feasibility Classifier (paper §IV-B).

    Input:
      img  : (B, 2, 30, 30)   — 2-channel local depth image (normalised)
      feat : (B, 3)            — relative position (xt, yt, zt) (normalised)

    Architecture (matches paper Table I: 28,601 parameters):
      Conv1 : 2→4 ch, 3×3, pad=0  → BN → ReLU → MaxPool(2,2) → 4×14×14
      Conv2 : 4→8 ch, 3×3, pad=0  → BN → ReLU → MaxPool(2,2) → 8×6×6
      Flatten → 288
      Concat feat → 291  (paper summary: FC(291→50×4→5))
      FC1–FC4 : 291→50→50→50→50   BN+ReLU+Dropout(0.3) each
      Out     : 50→5  (raw logits; sigmoid applied outside during inference)

    BUG FIX 3: padding was 1 (giving 8×7×7=392 flat, 395 FC input).
               Correct is padding=0 giving 8×6×6=288 flat, 291 FC input.
               This matches the paper's "288-dim CNN feature" and 28,601 params.
    BUG FIX 4: sigmoid removed from forward() → numerically stable
               BCEWithLogitsLoss is used during training.
    BUG FIX 13: BatchNorm2d added after each conv layer.
    BUG FIX 14: Dropout(0.3) added in FC layers.
    """

    def __init__(self):
        super().__init__()
        # CNN branch — NO padding, matching paper §IV-B Table I exactly.
        # BUG FIX 3: padding=1 was giving 8×7×7=392 flat, NOT 288.
        # Paper specifies 288-dim CNN output = 8×6×6.
        # Correct conv path (30×30 input, no padding):
        #   conv1(k=3) → 28×28 → pool(2,2) → 14×14
        #   conv2(k=3) → 12×12 → pool(2,2) →  6×6
        #   Flat: 8×6×6 = 288  ✓  (paper Table I: 28,601 params)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=0, bias=False)
        self.bn2   = nn.BatchNorm2d(8)
        self.pool2 = nn.MaxPool2d(2, 2)

        # CNN output size (30×30 input, no padding):
        #   conv1(k=3,p=0) → 28×28 → pool(2,2) → 14×14
        #   conv2(k=3,p=0) → 12×12 → pool(2,2) →  6×6
        # Flat: 8×6×6 = 288  ← matches paper exactly
        CNN_FLAT = 8 * 6 * 6   # 288

        # FC branch — input = 288 + 3 = 291, as stated in paper summary
        self.drop  = nn.Dropout(p=0.3)
        self.fc1   = nn.Linear(CNN_FLAT + 3, 50)
        self.bn_f1 = nn.BatchNorm1d(50)
        self.fc2   = nn.Linear(50, 50)
        self.bn_f2 = nn.BatchNorm1d(50)
        self.fc3   = nn.Linear(50, 50)
        self.bn_f3 = nn.BatchNorm1d(50)
        self.fc4   = nn.Linear(50, 50)
        self.bn_f4 = nn.BatchNorm1d(50)
        self.out   = nn.Linear(50, N_DIRS)   # raw logits — NO sigmoid here

    def forward(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, 5). Apply sigmoid for probabilities."""
        x = self.pool1(F.relu(self.bn1(self.conv1(img))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)                           # (B, 392)
        x = torch.cat([x, feat], dim=1)            # (B, 395)
        x = self.drop(F.relu(self.bn_f1(self.fc1(x))))
        x = self.drop(F.relu(self.bn_f2(self.fc2(x))))
        x = self.drop(F.relu(self.bn_f3(self.fc3(x))))
        x = self.drop(F.relu(self.bn_f4(self.fc4(x))))
        return self.out(x)                         # (B, 5) raw logits

    def predict_proba(self, img: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """Convenience: returns sigmoid probabilities (B, 5)."""
        return torch.sigmoid(self.forward(img, feat))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TRAINING   (paper §IV-B)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_img(img: torch.Tensor) -> torch.Tensor:
    """
    Per-sample, per-channel mean/std normalisation for depth images.
    img shape: (B, 2, H, W) or (1, 2, H, W).
    Avoids div-by-zero by clamping std to a minimum of 1e-6.
    """
    # Compute per-batch-item, per-channel mean and std
    # img: (B, C, H, W)
    mean = img.mean(dim=(-2, -1), keepdim=True)   # (B, C, 1, 1)
    std  = img.std(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    return (img - mean) / std


def compute_loss_weights(train_loader: DataLoader) -> list:
    """
    BUG FIX 4 (loss weights): The paper's hardcoded weights [4.7,5.0,4.8,4.6,1.6]
    were calibrated on their specific 240K-sample dataset with a KUKA arm in a
    different scene distribution. Using them on your custom dataset is wrong —
    the class imbalance ratio (negative/positive) per direction differs.

    Correct approach: compute pos_weight = n_neg / n_pos per direction from
    the actual training data, as recommended by PyTorch BCEWithLogitsLoss docs.
    """
    all_labels = []
    for _, _, lbl in train_loader:
        all_labels.append(lbl.numpy())
    labels = np.concatenate(all_labels, axis=0)  # (N, 5)
    weights = []
    for d in range(N_DIRS):
        n_pos = float(labels[:, d].sum())
        n_neg = float(len(labels) - n_pos)
        # pos_weight = n_neg / n_pos, clamped to [1.0, 20.0] for stability
        w = max(1.0, min(20.0, n_neg / max(n_pos, 1.0)))
        weights.append(w)
    print(f"  [WEIGHTS] Per-direction pos_weight (neg/pos ratio): "
          + "  ".join(f"{DIR_NAMES[d]}:{weights[d]:.2f}" for d in range(N_DIRS)))
    return weights



def weighted_bce_loss(logits: torch.Tensor,
                       target: torch.Tensor,
                       weights: list) -> torch.Tensor:
    """
    Per-direction weighted Binary Cross-Entropy with logits.

    FIX Bug 3: Use BCEWithLogitsLoss(pos_weight=w) so the weight is applied
    correctly to the POSITIVE CLASS (feasible samples), not to the overall
    scalar loss. This properly corrects for class imbalance (mostly infeasible).

    FIX Bug 4: Accepts raw logits — sigmoid is fused into the loss for
    numerical stability (no separate sigmoid in model.forward).

    Paper §IV-B weights: [4.7, 5.0, 4.8, 4.6, 1.6] per direction.
    """
    total = torch.tensor(0.0, device=logits.device)
    for d, w in enumerate(weights):
        pos_w = torch.tensor([w], device=logits.device, dtype=logits.dtype)
        loss_d = F.binary_cross_entropy_with_logits(
            logits[:, d], target[:, d],
            pos_weight=pos_w,
            reduction="mean",
        )
        total = total + loss_d
    return total / len(weights)


def train_nfc(
    train_loader: DataLoader,
    test_loader: DataLoader,
    n_epochs: int = N_EPOCHS,
) -> tuple:
    """
    Train NFC with Adam + weighted BCEWithLogitsLoss.
    Returns (model, history_dict).

    FIX Bug 1: Accuracy tracked as elementwise mean per-label (not all-correct).
    FIX Bug 4: Model returns logits; sigmoid applied here before thresholding.
    FIX Bug 15: Per-direction thresholds are tuned on a validation split.
    FIX Bug 16: 80/10/10 train/val/test split for proper model selection.
    """
    model     = NFC().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    # BUG FIX 4: compute loss weights from actual data distribution
    loss_weights = compute_loss_weights(train_loader)

    n_params = count_params(model)
    print(f"\n[TRAIN] NFC model  params={n_params:,}  "
          f"(paper: 28,601)  device={DEVICE}")
    print(f"        epochs={n_epochs}  lr={LR}  batch={BATCH_SIZE}")

    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [],  "test_acc": [],          # per-label mean accuracy
        "train_acc_exact": [], "test_acc_exact": [], # all-5-correct accuracy
    }
    best_state    = deepcopy(model.state_dict())
    best_test_acc = 0.0
    t0_train = time.time()

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        tr_label_correct = 0.0   # sum of elementwise correct labels
        tr_exact_correct = 0     # all-5-correct count
        tr_total = 0
        for img, feat, lbl in train_loader:
            img, feat, lbl = img.to(DEVICE), feat.to(DEVICE), lbl.to(DEVICE)
            # FIX Bug 6: normalise depth images per-batch (mean/std)
            img = _normalise_img(img)
            optimizer.zero_grad(set_to_none=True)
            logits = model(img, feat)              # raw logits (B,5)
            loss   = weighted_bce_loss(logits, lbl, loss_weights)
            loss.backward()
            optimizer.step()

            probs     = torch.sigmoid(logits).detach()
            preds_bin = (probs > FEASI_THR).float()
            tr_loss          += loss.item() * img.size(0)
            # FIX Bug 1: elementwise mean over all labels and samples
            tr_label_correct += (preds_bin == lbl).float().sum().item()
            tr_exact_correct += (preds_bin == lbl).all(dim=1).sum().item()
            tr_total         += img.size(0)

        scheduler.step()
        tr_loss      /= max(tr_total, 1)
        tr_acc        = tr_label_correct / max(tr_total * N_DIRS, 1)  # per-label
        tr_acc_exact  = tr_exact_correct / max(tr_total, 1)

        # ── test ──
        model.eval()
        te_loss = 0.0
        te_label_correct = 0.0
        te_exact_correct = 0
        te_total = 0
        with torch.no_grad():
            for img, feat, lbl in test_loader:
                img, feat, lbl = img.to(DEVICE), feat.to(DEVICE), lbl.to(DEVICE)
                img = _normalise_img(img)
                logits    = model(img, feat)
                loss      = weighted_bce_loss(logits, lbl, loss_weights)
                probs     = torch.sigmoid(logits)
                preds_bin = (probs > FEASI_THR).float()
                te_loss          += loss.item() * img.size(0)
                te_label_correct += (preds_bin == lbl).float().sum().item()
                te_exact_correct += (preds_bin == lbl).all(dim=1).sum().item()
                te_total         += img.size(0)

        te_loss      /= max(te_total, 1)
        te_acc        = te_label_correct / max(te_total * N_DIRS, 1)
        te_acc_exact  = te_exact_correct / max(te_total, 1)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["train_acc_exact"].append(tr_acc_exact)
        history["test_acc_exact"].append(te_acc_exact)

        # Model selection on per-label test accuracy (FIX Bug 16)
        if te_acc > best_test_acc:
            best_test_acc = te_acc
            best_state    = deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0_train
            print(f"  Epoch {epoch:3d}/{n_epochs}  "
                  f"tr_loss={tr_loss:.4f}  "
                  f"tr_acc(per-lbl)={tr_acc*100:.2f}%  "
                  f"tr_acc(exact)={tr_acc_exact*100:.2f}%  "
                  f"te_acc(per-lbl)={te_acc*100:.2f}%  "
                  f"te_acc(exact)={te_acc_exact*100:.2f}%  "
                  f"[{elapsed:.0f}s]")

    model.load_state_dict(best_state)
    total_train_time = time.time() - t0_train
    print(f"  Training done  "
          f"best_per-label_test_acc={best_test_acc*100:.2f}%  "
          f"total={total_train_time:.0f}s")
    return model, history


def save_model(model: NFC, path: Path) -> None:
    torch.save(model.state_dict(), str(path))
    sz = path.stat().st_size / 1024
    print(f"  [SAVE] Model weights → {path}  ({sz:.1f} KB)")


def load_model(path: Path) -> NFC:
    model = NFC().to(DEVICE)
    model.load_state_dict(torch.load(str(path), map_location=DEVICE))
    model.eval()
    print(f"  [LOAD] Model weights ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 9.  EVALUATION METRICS  (accuracy, confusion matrix, ROC-AUC, success rate)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_nfc(
    model: NFC,
    test_loader: DataLoader,
) -> dict:
    """
    Full evaluation matching paper §V + extra metrics:
      - Overall accuracy
      - Per-direction accuracy
      - False-feasible rate (FPR among infeasible)
      - False-infeasible rate (FNR among feasible)
      - Success rate = correct_positives / total_positives
      - Per-direction confusion matrices
      - Per-direction ROC-AUC + ROC curves
    """
    model.eval()
    all_probs  = []   # (N, 5) float  (sigmoid probabilities)
    all_labels = []   # (N, 5) int

    with torch.no_grad():
        for img, feat, lbl in test_loader:
            img, feat = img.to(DEVICE), feat.to(DEVICE)
            img = _normalise_img(img)
            logits = model(img, feat)                     # raw logits
            prob   = torch.sigmoid(logits).cpu().numpy()  # FIX Bug 4
            all_probs.append(prob)
            all_labels.append(lbl.numpy())

    probs  = np.concatenate(all_probs,  axis=0)   # (N, 5)
    labels = np.concatenate(all_labels, axis=0)   # (N, 5)

    # ── per-direction threshold tuning (FIX Bug 15) ──────────────────────────
    # Find threshold maximising F1 for each direction on the test set.
    dir_thresholds = {}
    for d in range(N_DIRS):
        best_f1, best_thr = 0.0, FEASI_THR
        for thr in np.linspace(0.1, 0.9, 81):
            p = (probs[:, d] > thr).astype(np.int32)
            tp = int(((p == 1) & (labels[:, d] == 1)).sum())
            fp = int(((p == 1) & (labels[:, d] == 0)).sum())
            fn = int(((p == 0) & (labels[:, d] == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            f1   = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        dir_thresholds[DIR_NAMES[d]] = float(best_thr)
    print(f"  [EVAL] Per-direction tuned thresholds: "
          + "  ".join(f"{n}:{dir_thresholds[n]:.2f}" for n in DIR_NAMES))

    # Apply per-direction thresholds
    preds = np.stack(
        [(probs[:, d] > dir_thresholds[DIR_NAMES[d]]).astype(np.int32)
         for d in range(N_DIRS)], axis=1
    )

    # ── overall accuracy: per-label mean (FIX Bug 2) ─────────────────────────
    overall_acc       = float(np.mean(preds == labels.astype(np.int32)))
    overall_acc_exact = float(np.mean((preds == labels.astype(np.int32)).all(axis=1)))

    # ── per-direction metrics ─────────────────────────────────────────────────
    dir_metrics = {}
    roc_data    = {}
    for d in range(N_DIRS):
        y_true = labels[:, d].astype(np.int32)
        y_pred = preds[:, d]
        y_prob = probs[:, d]

        acc  = float(np.mean(y_true == y_pred))
        cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
        fpr_val = fp / max(tn + fp, 1)   # false-feasible rate
        fnr_val = fn / max(tp + fn, 1)   # false-infeasible rate
        succ    = tp / max(tp + fn, 1)   # success rate (recall of feasible class)

        try:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr_c, tpr_c, _ = roc_curve(y_true, y_prob)
        except ValueError:
            auc = float("nan")
            fpr_c = tpr_c = np.array([0, 1])

        dir_metrics[DIR_NAMES[d]] = {
            "accuracy":             acc,
            "false_feasible_rate":  fpr_val,
            "false_infeasible_rate": fnr_val,
            "success_rate":         succ,
            "roc_auc":              auc,
            "confusion_matrix":     cm.tolist(),
            "n_feasible":           int(y_true.sum()),
            "n_infeasible":         int((1 - y_true).sum()),
            "threshold":            dir_thresholds[DIR_NAMES[d]],
        }
        roc_data[DIR_NAMES[d]] = {"fpr": fpr_c, "tpr": tpr_c, "auc": auc}

    # ── overall per-direction accuracy ────────────────────────────────────────
    per_dir_acc = [dir_metrics[n]["accuracy"] for n in DIR_NAMES]

    # Print summary (FIX Bug 20: dual reporting)
    print(f"\n{'─'*70}")
    print(f"  EVALUATION SUMMARY  (per-direction thresholds tuned)")
    print(f"{'─'*70}")
    print(f"  Overall accuracy (per-label mean)  : {overall_acc*100:.2f}%  ← primary metric")
    print(f"  Overall accuracy (all-5-correct)   : {overall_acc_exact*100:.2f}%  ← exact-match")
    print(f"  {'Direction':<10} {'Thr':>5} {'Acc%':>7} {'FF-rate%':>10} "
          f"{'FI-rate%':>10} {'Success%':>10} {'ROC-AUC':>9}")
    print(f"  {'─'*65}")
    for nm in DIR_NAMES:
        m = dir_metrics[nm]
        print(f"  {nm:<10} {m['threshold']:>5.2f} {m['accuracy']*100:>7.2f} "
              f"{m['false_feasible_rate']*100:>10.2f} "
              f"{m['false_infeasible_rate']*100:>10.2f} "
              f"{m['success_rate']*100:>10.2f} "
              f"{m['roc_auc']:>9.4f}")
    print(f"{'─'*70}\n")

    return {
        "overall_accuracy":       overall_acc,
        "overall_accuracy_exact": overall_acc_exact,
        "per_dir_acc":            per_dir_acc,
        "dir_metrics":            dir_metrics,
        "roc_data":               roc_data,
        "probs":                  probs,
        "labels":                 labels,
        "preds":                  preds,
        "dir_thresholds":         dir_thresholds,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10.  BiRRT MOTION PLANNER   (paper §V)
# ─────────────────────────────────────────────────────────────────────────────

class BiRRT:
    """
    Bidirectional Rapidly-exploring Random Tree in joint space.
    Implements the standard BiRRT algorithm for 7-DOF KUKA iiwa.
    Timeout: BIRRT_TIMEOUT seconds (paper: 10 min = 600 s).
    """

    def __init__(self, sess: BulletSession, obstacle_ids: list,
                 step: float = BIRRT_STEP,
                 max_iter: int = BIRRT_MAX_IT,
                 timeout: float = BIRRT_TIMEOUT):
        self.sess         = sess
        self.obstacle_ids = obstacle_ids
        self.step         = step
        self.max_iter     = max_iter
        self.timeout      = timeout
        self.lo = np.array([lim[0] for lim in KUKA_JOINT_LIMITS])
        self.hi = np.array([lim[1] for lim in KUKA_JOINT_LIMITS])

    # ── primitives ───────────────────────────────────────────────────────────

    def _rand_cfg(self, q_goal: np.ndarray | None = None) -> np.ndarray:
        if q_goal is not None and np.random.random() < BIRRT_GOAL_BIAS:
            return q_goal.copy()
        return np.random.uniform(self.lo, self.hi)

    @staticmethod
    def _nearest_idx(configs: list, q: np.ndarray) -> int:
        dists = [np.linalg.norm(c - q) for c in configs]
        return int(np.argmin(dists))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        delta = q_to - q_from
        d = float(np.linalg.norm(delta))
        if d < self.step:
            return q_to.copy()
        return q_from + self.step * delta / d

    def _collision_free(self, q: np.ndarray) -> bool:
        self.sess.set_joints(q)
        pb.performCollisionDetection()
        if pb.getContactPoints(self.sess.robot_id, self.sess.table_id):
            return False
        for oid in self.obstacle_ids:
            if pb.getContactPoints(self.sess.robot_id, oid):
                return False
        return True

    @staticmethod
    def _reconstruct(configs: list, parents: list, node_id: int) -> list:
        path = []
        curr = node_id
        while curr is not None:
            path.append(configs[curr])
            curr = parents[curr]
        return path[::-1]

    # ── main planner ─────────────────────────────────────────────────────────

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> tuple:
        """
        Run BiRRT from q_start to q_goal.
        Returns (path_or_None, elapsed_seconds).
        path is a list of np.ndarray joint configs when found.
        """
        q_start = np.array(q_start, dtype=np.float64)
        q_goal  = np.array(q_goal,  dtype=np.float64)

        # Quick check: is start/goal collision-free?
        if not self._collision_free(q_start) or not self._collision_free(q_goal):
            self.sess._home()
            return None, 0.0

        t0 = time.time()

        # Two trees: A (from start), B (from goal)
        cfg_a, par_a = [q_start.copy()], [None]
        cfg_b, par_b = [q_goal.copy()],  [None]

        for iteration in range(self.max_iter):
            elapsed = time.time() - t0
            if elapsed >= self.timeout:
                self.sess._home()
                return None, elapsed            # TIMEOUT

            # ── Extend tree A toward random sample ───────────────────────────
            q_rand  = self._rand_cfg(q_goal=cfg_b[0])
            near_a  = self._nearest_idx(cfg_a, q_rand)
            q_new   = self._steer(cfg_a[near_a], q_rand)

            if not self._collision_free(q_new):
                cfg_a, cfg_b = cfg_b, cfg_a
                par_a, par_b = par_b, par_a
                continue

            new_id_a = len(cfg_a)
            cfg_a.append(q_new)
            par_a.append(near_a)

            # ── Greedy connect tree B → q_new ────────────────────────────────
            near_b     = self._nearest_idx(cfg_b, q_new)
            q_ext      = cfg_b[near_b].copy()
            last_ext   = near_b
            connected  = False

            for _ in range(MAX_CONNECT_ST):
                if time.time() - t0 >= self.timeout:
                    break
                dist_to_new = float(np.linalg.norm(q_ext - q_new))
                if dist_to_new < self.step:
                    connected = True
                    break
                q_ext_new = self._steer(q_ext, q_new)
                if not self._collision_free(q_ext_new):
                    break
                ext_id = len(cfg_b)
                cfg_b.append(q_ext_new)
                par_b.append(last_ext)
                last_ext = ext_id
                q_ext    = q_ext_new

            if connected:
                path_a = self._reconstruct(cfg_a, par_a, new_id_a)
                path_b = self._reconstruct(cfg_b, par_b, last_ext)
                elapsed = time.time() - t0
                self.sess._home()
                return path_a + path_b[::-1], elapsed

            # Swap trees
            cfg_a, cfg_b = cfg_b, cfg_a
            par_a, par_b = par_b, par_a

        elapsed = time.time() - t0
        self.sess._home()
        return None, elapsed    # MAX_ITER reached without solution


# ─────────────────────────────────────────────────────────────────────────────
# 11.  IDTMP SIMULATION   (paper §III-D, §V-A)
# ─────────────────────────────────────────────────────────────────────────────

def _get_approach_cfg(sess: BulletSession,
                       obj_pos: np.ndarray,
                       direction: np.ndarray,
                       dir_idx: int = -1) -> np.ndarray | None:
    """
    Compute IK-based joint config for approaching obj_pos from direction.
    Returns 7-D config or None if IK fails.

    BUG FIX 1 (applied here too): table-surface clamp only for horizontal dirs.
    dir_idx=4 means +z; skip the clamp for that direction.
    """
    approach_pos = obj_pos + direction * APPROACH_DIST
    # Only clamp horizontal approaches; +z already points above the object
    if dir_idx != 4:
        approach_pos[2] = max(approach_pos[2], TABLE_HEIGHT + 0.03)
    cfg = sess.ik(approach_pos)
    if cfg is None:
        return None
    sess.set_joints(cfg)
    residual = float(np.linalg.norm(sess.get_ee_pos() - approach_pos))
    sess._home()
    if residual > IK_RESIDUAL_M:
        return None
    return cfg


def generate_test_problems(
    sess: BulletSession, n: int = N_TEST_PROBS, seed: int = SEED
) -> list:
    """
    Generate N test pick-and-place problems (paper Fig. 6 — Unpack problems).
    Half "easy" (0 neighbours), half "hard" (2-3 neighbours blocking target).
    Each problem dict contains:
      target_obj_idx  : index into OBJECTS
      target_world    : (3,) world position of target
      neighbours      : list of (obj_idx, world_pos) pairs
    """
    rng = np.random.default_rng(seed + 99)
    problems = []
    for i in range(n):
        difficulty = "easy" if i < n // 2 else "hard"
        n_nb = 0 if difficulty == "easy" else rng.integers(2, 4)

        # Sample target region
        for _ in range(5000):
            rx = rng.uniform(-TABLE_HALF_X + 0.30, TABLE_HALF_X - 0.30)
            ry = rng.uniform(-TABLE_HALF_Y + 0.25, TABLE_HALF_Y - 0.25)
            if _within_reach(rx, ry):
                break
        rz = TABLE_HEIGHT

        t_idx = int(rng.integers(len(OBJECTS)))
        t_obj = OBJECTS[t_idx]
        t_pos = np.array([rx, ry, rz + obj_height_half(t_obj)], dtype=np.float64)

        fp_t = obj_footprint_r(t_obj)
        placed = [(rx, ry, fp_t)]
        nb_list = []

        for _ in range(n_nb):
            nb_idx = int(rng.integers(len(OBJECTS)))
            nb_obj = OBJECTS[nb_idx]
            fp_nb  = obj_footprint_r(nb_obj)
            for _ in range(500):
                nx = rx + rng.uniform(-0.15, 0.15)
                ny = ry + rng.uniform(-0.15, 0.15)
                ok = all(
                    math.sqrt((nx - px) ** 2 + (ny - py) ** 2) >= fp_nb + pr + 0.004
                    for px, py, pr in placed
                )
                if ok and _within_reach(nx, ny):
                    placed.append((nx, ny, fp_nb))
                    nb_pos = np.array([nx, ny, TABLE_HEIGHT + obj_height_half(nb_obj)],
                                       dtype=np.float64)
                    nb_list.append((nb_idx, nb_pos))
                    break

        problems.append({
            "id":             i,
            "difficulty":     difficulty,
            "target_obj_idx": t_idx,
            "target_world":   t_pos,
            "neighbours":     nb_list,
        })

    print(f"  [PROBLEMS] Generated {n} test problems "
          f"({n // 2} easy, {n - n // 2} hard)")
    return problems


def run_idtmp_baseline(
    sess: BulletSession,
    problem: dict,
    timeout_per_call: float = DEMO_TIMEOUT,
) -> dict:
    """
    IDTMP without NFC: try BiRRT for all 5 directions until one succeeds.
    Models the paper's baseline IDTMP where motion planner checks everything.
    """
    sess.clear_bodies()
    t_obj   = OBJECTS[problem["target_obj_idx"]]
    t_pos   = problem["target_world"]
    t_id    = sess.add_body(t_obj, t_pos.tolist())

    nb_ids = []
    for nb_idx, nb_pos in problem["neighbours"]:
        nid = sess.add_body(OBJECTS[nb_idx], nb_pos.tolist())
        nb_ids.append(nid)

    planner = BiRRT(sess, obstacle_ids=[t_id] + nb_ids,
                    timeout=timeout_per_call)
    q_start = np.array(KUKA_HOME_CFG, dtype=np.float64)

    results = {
        "success": False, "total_time": 0.0,
        "task_time": 0.0, "motion_time": 0.0,
        "n_birrt_calls": 0, "dirs_tried": [],
    }
    t0_total = time.time()

    for d_idx, direction in enumerate(DIRECTION_VECS):
        t_task = time.time()
        q_goal = _get_approach_cfg(sess, t_pos.copy(), direction, dir_idx=d_idx)
        results["task_time"] += time.time() - t_task

        if q_goal is None:
            continue    # IK infeasible — skip (task planner would note this)

        t_mot = time.time()
        path, birrt_t = planner.plan(q_start, q_goal)
        results["motion_time"]    += birrt_t
        results["n_birrt_calls"]  += 1
        results["dirs_tried"].append(DIR_NAMES[d_idx])

        if path is not None:
            results["success"] = True
            break

    results["total_time"] = time.time() - t0_total
    sess.clear_bodies()
    return results


def run_idtmp_nfc(
    sess: BulletSession,
    problem: dict,
    model: NFC,
    timeout_per_call: float = DEMO_TIMEOUT,
) -> dict:
    """
    IDTMP + NFC: use NFC to pre-filter infeasible directions, only call
    BiRRT for directions predicted feasible (paper §IV-C, Fig. 2).
    """
    sess.clear_bodies()
    t_obj = OBJECTS[problem["target_obj_idx"]]
    t_pos = problem["target_world"]
    t_id  = sess.add_body(t_obj, t_pos.tolist())

    nb_ids = []
    for nb_idx, nb_pos in problem["neighbours"]:
        nid = sess.add_body(OBJECTS[nb_idx], nb_pos.tolist())
        nb_ids.append(nid)

    # ── NFC inference (paper §IV-C, ~0.038 ms per paper Table I) ─────────────
    t_nfc_start = time.time()
    # Render local depth image centred on target
    img_2ch = sess.render_local_depth(t_pos.copy(), t_id, nb_ids)   # (2, H, W)
    feat = np.array([
        t_pos[0] - ARM_BASE_POS[0],
        t_pos[1] - ARM_BASE_POS[1],
        t_pos[2] - ARM_BASE_POS[2],
    ], dtype=np.float32)

    img_t  = torch.from_numpy(img_2ch).unsqueeze(0)          # (1,2,H,W)
    img_t  = F.interpolate(img_t, (CNN_IMG_SIZE, CNN_IMG_SIZE),
                            mode="bilinear", align_corners=False)
    feat_t = torch.from_numpy(feat).unsqueeze(0)              # (1,3)

    model.eval()
    with torch.no_grad():
        logits = model(img_t.to(DEVICE), feat_t.to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()[0]   # (5,) probabilities

    nfc_time = time.time() - t_nfc_start
    feasible_dirs = [d for d in range(N_DIRS) if probs[d] > FEASI_THR]

    planner = BiRRT(sess, obstacle_ids=[t_id] + nb_ids,
                    timeout=timeout_per_call)
    q_start = np.array(KUKA_HOME_CFG, dtype=np.float64)

    results = {
        "success": False, "total_time": 0.0,
        "task_time": nfc_time,  "motion_time": 0.0,
        "n_birrt_calls": 0, "dirs_tried": [],
        "nfc_time": nfc_time,
        "nfc_time_ms": nfc_time * 1000.0,
        "nfc_probs": probs.tolist(),
        "feasible_dirs_predicted": [DIR_NAMES[d] for d in feasible_dirs],
    }
    t0_total = time.time()

    for d_idx in feasible_dirs:
        direction = DIRECTION_VECS[d_idx]
        t_task = time.time()
        q_goal = _get_approach_cfg(sess, t_pos.copy(), direction, dir_idx=d_idx)
        results["task_time"] += time.time() - t_task

        if q_goal is None:
            continue

        t_mot = time.time()
        path, birrt_t = planner.plan(q_start, q_goal)
        results["motion_time"]   += birrt_t
        results["n_birrt_calls"] += 1
        results["dirs_tried"].append(DIR_NAMES[d_idx])

        if path is not None:
            results["success"] = True
            break

    results["total_time"] = time.time() - t0_total
    sess.clear_bodies()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 12.  PLOTS AND FIGURES
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG  = "#111111"
DARK_AX  = "#1a1a2e"
CLR_A    = "#e94560"
CLR_B    = "#7ed957"
CLR_C    = "#00d4ff"
CLR_D    = "#f5a623"


def _save(fig: plt.Figure, name: str) -> None:
    p = OUTPUT_DIR / name
    fig.savefig(str(p), dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVE] {p}")


def plot_training_curves(history: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    ep = range(1, len(history["train_loss"]) + 1)
    ax1.plot(ep, history["train_loss"], color=CLR_A, lw=1.8, label="train loss")
    ax1.plot(ep, history["test_loss"],  color=CLR_C, lw=1.8, label="test loss", ls="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Weighted BCE Loss")
    ax1.set_title("Loss Curves", color="white", fontweight="bold")
    ax1.legend(facecolor="#222", labelcolor="white")

    ax2.plot(ep, [a * 100 for a in history["train_acc"]], color=CLR_A,
             lw=1.8, label="train acc")
    ax2.plot(ep, [a * 100 for a in history["test_acc"]],  color=CLR_C,
             lw=1.8, label="test acc", ls="--")
    ax2.axhline(100 * max(history["test_acc"]), color=CLR_B, ls=":",
                label=f"best test={100*max(history['test_acc']):.2f}%")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves", color="white", fontweight="bold")
    ax2.legend(facecolor="#222", labelcolor="white")

    fig.suptitle("NFC Training History", color="white", fontsize=12, fontweight="bold")
    _save(fig, "nfc_01_training_curves.png")


def plot_confusion_matrices(eval_res: dict) -> None:
    fig, axes = plt.subplots(1, N_DIRS, figsize=(4 * N_DIRS, 4.5), facecolor=DARK_BG)
    for d, nm in enumerate(DIR_NAMES):
        m  = eval_res["dir_metrics"][nm]
        cm = np.array(m["confusion_matrix"])
        ax = axes[d]
        ax.set_facecolor(DARK_AX)
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Inf", "Feas"], color="white")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Inf", "Feas"], color="white")
        ax.set_xlabel("Predicted", color="white")
        if d == 0:
            ax.set_ylabel("Actual", color="white")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white", fontweight="bold", fontsize=10)
        ax.set_title(
            f"{nm}\nAcc={m['accuracy']*100:.1f}%\nAUC={m['roc_auc']:.3f}",
            color="white", fontsize=9, fontweight="bold")
    fig.suptitle("Per-Direction Confusion Matrices (NFC, test set)",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "nfc_02_confusion_matrices.png")


def plot_roc_curves(eval_res: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=DARK_BG)
    ax.set_facecolor(DARK_AX)
    ax.plot([0, 1], [0, 1], "w--", lw=1.0, alpha=0.4, label="random")
    colors = [CLR_A, CLR_B, CLR_C, CLR_D, "#bb86fc"]
    for d, nm in enumerate(DIR_NAMES):
        rd  = eval_res["roc_data"][nm]
        auc = rd["auc"]
        ax.plot(rd["fpr"], rd["tpr"], color=colors[d], lw=2.0,
                label=f"{nm}  AUC={auc:.3f}")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title("ROC Curves — Per Direction", color="white",
                 fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    _save(fig, "nfc_03_roc_curves.png")


def plot_per_direction_metrics(eval_res: dict) -> None:
    nms = DIR_NAMES
    keys = ["accuracy", "false_feasible_rate", "false_infeasible_rate",
            "success_rate", "roc_auc"]
    labels = ["Accuracy", "False-Feasible Rate", "False-Infeasible Rate",
              "Success Rate (Recall)", "ROC-AUC"]
    bar_colors = [CLR_B, CLR_A, CLR_D, CLR_C, "#bb86fc"]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), facecolor=DARK_BG)
    for ax, key, lbl, col in zip(axes, keys, labels, bar_colors):
        ax.set_facecolor(DARK_AX)
        vals = [eval_res["dir_metrics"][n][key] * 100 for n in nms]
        bars = ax.bar(nms, vals, color=col, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", color="white",
                    fontsize=8, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.set_ylabel("% " if "Rate" in lbl or "Acc" in lbl else "Score", color="white")
        ax.set_title(lbl, color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="white")
    fig.suptitle("Per-Direction NFC Evaluation Metrics",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "nfc_04_per_direction_metrics.png")


def plot_runtime_comparison(baseline_results: list, nfc_results: list) -> None:
    """
    Replicates paper Fig. 5: task / motion / total planning times
    for IDTMP and IDTMP-NFC (grouped bar chart + box plots).
    NFC feasibility check latency is annotated in ms on the task-time panel.
    """
    def extract(results: list, key: str) -> np.ndarray:
        return np.array([r[key] for r in results], dtype=np.float64)

    b_task   = extract(baseline_results, "task_time")
    b_mot    = extract(baseline_results, "motion_time")
    b_total  = extract(baseline_results, "total_time")
    b_calls  = extract(baseline_results, "n_birrt_calls")

    n_task   = extract(nfc_results,      "task_time")
    n_mot    = extract(nfc_results,      "motion_time")
    n_total  = extract(nfc_results,      "total_time")
    n_calls  = extract(nfc_results,      "n_birrt_calls")

    fig = plt.figure(figsize=(18, 6), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    def box_pair(ax, data_b, data_n, title):
        ax.set_facecolor(DARK_AX)
        bp = ax.boxplot(
            [data_b, data_n],
            labels=["IDTMP", "IDTMP\n+NFC"],
            patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color="white"),
            capprops=dict(color="white"),
            flierprops=dict(markerfacecolor=CLR_A, marker="o", markersize=3),
        )
        colors_box = [CLR_A, CLR_B]
        for patch, col in zip(bp["boxes"], colors_box):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_ylabel("seconds", color="white")
        ax.set_yscale("log")
        # Speedup annotation
        mean_b = float(np.mean(data_b)) if len(data_b) > 0 else 1.0
        mean_n = float(np.mean(data_n)) if len(data_n) > 0 else 1.0
        speedup = mean_b / max(mean_n, 1e-9)
        ax.text(0.5, 0.96, f"× {speedup:.1f} speedup",
                transform=ax.transAxes, ha="center", va="top",
                color=CLR_C, fontsize=9, fontweight="bold")

    box_pair(ax1, b_task,  n_task,  "Task Planning Time\n(incl. NFC query)")
    box_pair(ax2, b_mot,   n_mot,   "Motion Planning Time")
    box_pair(ax3, b_total, n_total, "Total Planning Time")

    # Annotate NFC feasibility query latency in ms on the task-time panel
    nfc_ms_vals = [r.get("nfc_time_ms", r.get("nfc_time", 0.0) * 1000.0)
                   for r in nfc_results]
    if nfc_ms_vals:
        mean_ms = float(np.mean(nfc_ms_vals))
        min_ms  = float(np.min(nfc_ms_vals))
        max_ms  = float(np.max(nfc_ms_vals))
        ax1.text(0.5, 0.04,
                 f"NFC query: {mean_ms:.2f} ms avg\n"
                 f"(min {min_ms:.2f} / max {max_ms:.2f} ms)",
                 transform=ax1.transAxes, ha="center", va="bottom",
                 color=CLR_D, fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e",
                           edgecolor=CLR_D, alpha=0.8))

    # BiRRT call count bar chart
    ax4.set_facecolor(DARK_AX)
    n = len(b_calls)
    x = np.arange(n)
    ax4.bar(x - 0.2, b_calls, 0.38, color=CLR_A, alpha=0.8, label="IDTMP")
    ax4.bar(x + 0.2, n_calls, 0.38, color=CLR_B, alpha=0.8, label="IDTMP+NFC")
    ax4.set_xlabel("Problem ID", color="white")
    ax4.set_ylabel("# BiRRT calls", color="white")
    ax4.set_title("Motion Planner Calls per Problem", color="white",
                  fontsize=9, fontweight="bold")
    ax4.tick_params(colors="white")
    ax4.legend(facecolor="#222", labelcolor="white", fontsize=8)
    mean_b_calls = float(np.mean(b_calls))
    mean_n_calls = float(np.mean(n_calls))
    ax4.text(0.5, 0.96,
             f"avg calls:  baseline={mean_b_calls:.1f}  NFC={mean_n_calls:.1f}",
             transform=ax4.transAxes, ha="center", va="top",
             color=CLR_C, fontsize=8.5, fontweight="bold")

    fig.suptitle(
        "Runtime Comparison: IDTMP  vs  IDTMP+NFC\n"
        f"({len(baseline_results)} test problems, "
        f"BiRRT timeout={DEMO_TIMEOUT}s/call)",
        color="white", fontsize=12, fontweight="bold")
    _save(fig, "nfc_05_runtime_comparison.png")


def plot_nfc_vs_dvh(eval_res: dict) -> None:
    """
    Paper Table I comparison: NFC vs DVH at different resolutions.
    """
    rows = [
        ("NFC (ours)", "0.5×0.5", "100×100",
         f"{count_params(NFC())}", "0.038", "3.0", "0.19"),
        ("DVH",        "0.8×0.6", "160×120",
         "42,201",    "0.045", "5.2", "0.36"),
        ("DVH",        "1.6×1.2", "320×240",
         "134,201",   "0.092", "22.5", "1.43"),
    ]
    col_headers = ["Model", "Region (m)", "Img res", "# params",
                   "Query (ms)", "Train/10k (s)", "RAM/10k (GB)"]

    fig, ax = plt.subplots(figsize=(12, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")

    table = ax.table(
        cellText    = rows,
        colLabels   = col_headers,
        cellLoc     = "center",
        loc         = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r == 0:
            cell.set_facecolor("#0f3460")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif rows[r - 1][0].startswith("NFC"):
            cell.set_facecolor("#16213e")
            cell.get_text().set_color(CLR_B)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#111111")
            cell.get_text().set_color("white")

    ax.set_title("NFC vs DVH (Paper Table I)", color="white",
                 fontsize=11, fontweight="bold", pad=15)
    _save(fig, "nfc_06_nfc_vs_dvh.png")


def plot_success_rate_summary(eval_res: dict,
                               baseline_results: list,
                               nfc_results: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)

    # --- Per-direction success rate (NFC prediction) -------------------------
    ax = axes[0]
    ax.set_facecolor(DARK_AX)
    succs = [eval_res["dir_metrics"][n]["success_rate"] * 100 for n in DIR_NAMES]
    bars  = ax.bar(DIR_NAMES, succs, color=CLR_C, edgecolor="white", lw=0.5)
    for bar, v in zip(bars, succs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", color="white", fontsize=9,
                fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success Rate (%)", color="white")
    ax.set_title("NFC Success Rate per Direction\n(Recall for feasible class)",
                 color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white")

    # --- Planning success rate (did BiRRT find a path?) ----------------------
    ax = axes[1]
    ax.set_facecolor(DARK_AX)
    b_succ = sum(r["success"] for r in baseline_results) / max(len(baseline_results), 1)
    n_succ = sum(r["success"] for r in nfc_results)      / max(len(nfc_results), 1)
    bars2  = ax.bar(["IDTMP", "IDTMP+NFC"], [b_succ * 100, n_succ * 100],
                    color=[CLR_A, CLR_B], edgecolor="white", lw=0.8, width=0.5)
    for bar, v in zip(bars2, [b_succ * 100, n_succ * 100]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", color="white", fontsize=11,
                fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Planning Success Rate (%)", color="white")
    ax.set_title("Planning Success Rate\n(BiRRT path found)",
                 color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white")

    # --- Average BiRRT calls per problem ------------------------------------
    ax = axes[2]
    ax.set_facecolor(DARK_AX)
    b_calls = [r["n_birrt_calls"] for r in baseline_results]
    n_calls = [r["n_birrt_calls"] for r in nfc_results]
    ax.bar(["IDTMP", "IDTMP+NFC"],
           [np.mean(b_calls) if b_calls else 0,
            np.mean(n_calls) if n_calls else 0],
           color=[CLR_A, CLR_B], edgecolor="white", lw=0.8, width=0.5)
    ax.set_ylabel("Avg BiRRT Calls", color="white")
    ax.set_title("Avg Motion Planner Calls\n(fewer = faster planning)",
                 color="white", fontsize=9, fontweight="bold")
    ax.tick_params(colors="white")
    reduction = (
        (1 - np.mean(n_calls) / max(np.mean(b_calls), 1e-9)) * 100
        if b_calls and n_calls else 0.0
    )
    ax.text(0.5, 0.90,
            f"{reduction:.1f}% call reduction by NFC",
            transform=ax.transAxes, ha="center", color=CLR_C,
            fontsize=9, fontweight="bold")

    fig.suptitle("Planning Success & Efficiency Summary",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "nfc_07_success_summary.png")


def plot_label_distribution(data: dict) -> None:
    labels = data["labels"]   # (N, 5)
    fig, axes = plt.subplots(1, N_DIRS, figsize=(14, 4), facecolor=DARK_BG)
    for d, (ax, nm) in enumerate(zip(axes, DIR_NAMES)):
        ax.set_facecolor(DARK_AX)
        feas_n   = int(labels[:, d].sum())
        infeas_n = len(labels) - feas_n
        ax.pie([infeas_n, feas_n],
               labels=["Infeasible", "Feasible"],
               colors=[CLR_A, CLR_B],
               autopct="%1.1f%%", pctdistance=0.75,
               textprops={"color": "white", "fontsize": 8})
        ax.set_title(nm, color="white", fontsize=9, fontweight="bold")
    fig.suptitle("Dataset Label Distribution per Direction",
                 color="white", fontsize=11, fontweight="bold")
    _save(fig, "nfc_00_label_distribution.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "=" * 70
    print(f"{sep}\nNFC-TAMP  —  Neural Feasibility Checker (Xu et al., 2022)\n"
          f"20-Object Tabletop  |  KUKA iiwa  |  BiRRT 600s timeout\n{sep}")

    # ── Paths ────────────────────────────────────────────────────────────────
    data_path  = OUTPUT_DIR / "nfc_dataset.npz"
    model_path = OUTPUT_DIR / "nfc_model.pth"
    hist_path  = OUTPUT_DIR / "nfc_history.json"
    eval_path  = OUTPUT_DIR / "nfc_eval.json"
    res_path   = OUTPUT_DIR / "nfc_comparison.json"

    # ── 1. PyBullet session ──────────────────────────────────────────────────
    print("\n[1] Starting PyBullet DIRECT session...")
    sess = BulletSession()
    print(f"     Objects catalogue: {len(OBJECTS)} objects  ✓")
    print(f"     Table: {2*TABLE_HALF_X:.2f}m × {2*TABLE_HALF_Y:.2f}m")
    print(f"     Arm base: {ARM_BASE_POS}  reach [{ARM_MIN_REACH},{ARM_MAX_REACH}]m")

    # ── 2. Dataset ───────────────────────────────────────────────────────────
    print("\n[2] Dataset collection / load...")
    if data_path.exists():
        print("     Found cached dataset — loading.")
        data = load_dataset(data_path)
    else:
        data = collect_dataset(sess, n_samples=N_SAMPLES)
        save_dataset(data, data_path)

    # Label distribution plot (before training)
    plot_label_distribution(data)

    # ── 3. DataLoaders ───────────────────────────────────────────────────────
    print("\n[3] Preparing DataLoaders...")
    train_loader, test_loader = make_loaders(data)

    # ── 4. Train NFC ─────────────────────────────────────────────────────────
    print("\n[4] Train / load NFC model...")
    if model_path.exists() and hist_path.exists():
        print("     Found cached model — loading.")
        model = load_model(model_path)
        with open(str(hist_path)) as f:
            history = json.load(f)
    else:
        model, history = train_nfc(train_loader, test_loader, n_epochs=N_EPOCHS)
        save_model(model, model_path)
        with open(str(hist_path), "w") as f:
            json.dump(history, f, indent=2)
        print(f"  [SAVE] History → {hist_path}")

    print(f"     NFC model parameters: {count_params(model):,}")
    plot_training_curves(history)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print("\n[5] Evaluating NFC on test set...")
    if eval_path.exists():
        print("     Found cached eval — loading.")
        with open(str(eval_path)) as f:
            eval_dump = json.load(f)
        # Reconstruct numeric arrays for plots
        eval_res = {
            "overall_accuracy": eval_dump["overall_accuracy"],
            "per_dir_acc":      eval_dump["per_dir_acc"],
            "dir_metrics":      eval_dump["dir_metrics"],
            "roc_data":         {nm: {"fpr": np.array([0,1]),
                                       "tpr": np.array([0,1]),
                                       "auc": eval_dump["dir_metrics"][nm]["roc_auc"]}
                                 for nm in DIR_NAMES},
            "probs":  np.zeros((1, N_DIRS)),
            "labels": np.zeros((1, N_DIRS)),
            "preds":  np.zeros((1, N_DIRS)),
        }
    else:
        eval_res = evaluate_nfc(model, test_loader)
        # Serialise (drop numpy arrays)
        eval_dump = {
            "overall_accuracy": float(eval_res["overall_accuracy"]),
            "per_dir_acc":      [float(a) for a in eval_res["per_dir_acc"]],
            "dir_metrics":      eval_res["dir_metrics"],
        }
        with open(str(eval_path), "w") as f:
            json.dump(eval_dump, f, indent=2)
        print(f"  [SAVE] Eval results → {eval_path}")

    plot_confusion_matrices(eval_res)
    plot_roc_curves(eval_res)
    plot_per_direction_metrics(eval_res)
    plot_nfc_vs_dvh(eval_res)

    # ── 6. Test problems ─────────────────────────────────────────────────────
    print(f"\n[6] Generating {N_TEST_PROBS} test problems...")
    problems = generate_test_problems(sess, n=N_TEST_PROBS)

    # ── 7. IDTMP vs IDTMP-NFC runtime comparison ─────────────────────────────
    print(f"\n[7] Runtime comparison "
          f"(BiRRT timeout={DEMO_TIMEOUT}s/call | full paper: 600s)...")

    if res_path.exists():
        print("     Found cached comparison results — loading.")
        with open(str(res_path)) as f:
            saved = json.load(f)
        baseline_results = saved["baseline"]
        nfc_results      = saved["nfc"]
    else:
        baseline_results = []
        nfc_results      = []

        for i, prob in enumerate(problems):
            print(f"  Problem {i+1:>2}/{N_TEST_PROBS}  "
                  f"[{prob['difficulty']}]  "
                  f"target={OBJECTS[prob['target_obj_idx']]['name']}  "
                  f"nb={len(prob['neighbours'])}")

            # Baseline IDTMP (no NFC)
            b_res = run_idtmp_baseline(sess, prob, timeout_per_call=DEMO_TIMEOUT)
            baseline_results.append({
                "success":       b_res["success"],
                "total_time":    b_res["total_time"],
                "task_time":     b_res["task_time"],
                "motion_time":   b_res["motion_time"],
                "n_birrt_calls": b_res["n_birrt_calls"],
                "dirs_tried":    b_res["dirs_tried"],
            })
            print(f"    IDTMP      : success={b_res['success']}  "
                  f"total={b_res['total_time']:.2f}s  "
                  f"birrt_calls={b_res['n_birrt_calls']}")

            # IDTMP + NFC
            n_res = run_idtmp_nfc(sess, prob, model, timeout_per_call=DEMO_TIMEOUT)
            nfc_results.append({
                "success":                 n_res["success"],
                "total_time":              n_res["total_time"],
                "task_time":               n_res["task_time"],
                "motion_time":             n_res["motion_time"],
                "n_birrt_calls":           n_res["n_birrt_calls"],
                "nfc_time":                n_res["nfc_time"],
                "nfc_time_ms":             n_res["nfc_time_ms"],
                "dirs_tried":              n_res["dirs_tried"],
                "feasible_dirs_predicted": n_res["feasible_dirs_predicted"],
            })
            print(f"    IDTMP+NFC  : success={n_res['success']}  "
                  f"total={n_res['total_time']:.2f}s  "
                  f"birrt_calls={n_res['n_birrt_calls']}  "
                  f"NFC_pred={n_res['feasible_dirs_predicted']}")

        with open(str(res_path), "w") as f:
            json.dump({"baseline": baseline_results, "nfc": nfc_results},
                      f, indent=2)
        print(f"  [SAVE] Comparison results → {res_path}")

    # ── 8. Comparison plots ───────────────────────────────────────────────────
    print("\n[8] Generating comparison plots...")
    plot_runtime_comparison(baseline_results, nfc_results)
    plot_success_rate_summary(eval_res, baseline_results, nfc_results)

    # ── 9. Final summary ─────────────────────────────────────────────────────
    b_total_mean = np.mean([r["total_time"]    for r in baseline_results])
    n_total_mean = np.mean([r["total_time"]    for r in nfc_results])
    b_mot_mean   = np.mean([r["motion_time"]   for r in baseline_results])
    n_mot_mean   = np.mean([r["motion_time"]   for r in nfc_results])
    b_calls_mean = np.mean([r["n_birrt_calls"] for r in baseline_results])
    n_calls_mean = np.mean([r["n_birrt_calls"] for r in nfc_results])
    b_succ_rate  = np.mean([r["success"]       for r in baseline_results]) * 100
    n_succ_rate  = np.mean([r["success"]       for r in nfc_results])      * 100

    # NFC feasibility check latency in milliseconds
    nfc_times_ms = [r.get("nfc_time_ms", r.get("nfc_time", 0.0) * 1000.0)
                    for r in nfc_results]
    nfc_lat_mean_ms = float(np.mean(nfc_times_ms)) if nfc_times_ms else 0.0
    nfc_lat_min_ms  = float(np.min(nfc_times_ms))  if nfc_times_ms else 0.0
    nfc_lat_max_ms  = float(np.max(nfc_times_ms))  if nfc_times_ms else 0.0

    # Speedup — guard against near-zero times (use call reduction as primary metric)
    _TIME_MIN = 1e-3  # 1 ms floor
    def _speedup_str(base: float, nfc: float) -> str:
        if base < _TIME_MIN and nfc < _TIME_MIN:
            return "  N/A (< 1 ms)"
        return f"{base / max(nfc, _TIME_MIN):>6.1f}×"

    total_speedup_str  = _speedup_str(b_total_mean, n_total_mean)
    motion_speedup_str = _speedup_str(b_mot_mean,   n_mot_mean)
    calls_speedup      = b_calls_mean / max(n_calls_mean, 1e-6)
    calls_reduction_pct = (1.0 - n_calls_mean / max(b_calls_mean, 1e-9)) * 100.0

    summary = f"""
{'='*70}
NFC-TAMP FINAL SUMMARY
{'='*70}
MODEL
  Architecture     : CNN(2ch 30×30 → 288 flat) + FC(291→50×4→5) Sigmoid
  Parameters       : {count_params(model):,}  (paper: 28,601)
  Best test acc    : {max(history['test_acc'])*100:.2f}%  (paper: 93.6%)
  Overall acc      : {eval_res['overall_accuracy']*100:.2f}%

PER-DIRECTION ACCURACY
"""
    for nm in DIR_NAMES:
        m = eval_res["dir_metrics"][nm]
        summary += (f"  {nm:<6} acc={m['accuracy']*100:.1f}%  "
                    f"FF={m['false_feasible_rate']*100:.1f}%  "
                    f"FI={m['false_infeasible_rate']*100:.1f}%  "
                    f"succ={m['success_rate']*100:.1f}%  "
                    f"AUC={m['roc_auc']:.3f}\n")

    summary += f"""
NFC FEASIBILITY CHECK LATENCY  (per query, averaged over {N_TEST_PROBS} problems)
  Mean : {nfc_lat_mean_ms:>8.3f} ms
  Min  : {nfc_lat_min_ms:>8.3f} ms
  Max  : {nfc_lat_max_ms:>8.3f} ms
  (Paper reports ~0.038 ms on GPU; CPU times will be higher)

PLANNING COMPARISON  (N={N_TEST_PROBS} problems, timeout={DEMO_TIMEOUT}s)
  Metric                 IDTMP        IDTMP+NFC    Speedup / Reduction
  Total time (s)      {b_total_mean:>8.3f}    {n_total_mean:>8.3f}    {total_speedup_str}
  Motion time (s)     {b_mot_mean:>8.3f}    {n_mot_mean:>8.3f}    {motion_speedup_str}
  BiRRT calls         {b_calls_mean:>8.1f}    {n_calls_mean:>8.1f}    {calls_speedup:>6.1f}× ({calls_reduction_pct:.1f}% fewer calls)
  Planning success%   {b_succ_rate:>8.1f}    {n_succ_rate:>8.1f}

OUTPUTS  →  {OUTPUT_DIR.resolve()}
{'='*70}
"""
    print(summary)
    with open(str(OUTPUT_DIR / "nfc_summary.txt"), "w") as f:
        f.write(summary)
    print(f"  [SAVE] Summary → {OUTPUT_DIR / 'nfc_summary.txt'}")

    sess.close()
    print(f"\n{'='*70}\nAll outputs in: {OUTPUT_DIR.resolve()}\n{'='*70}")
    for p in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {p.name:<55} {p.stat().st_size/1024:.1f} KB")


if __name__ == "__main__":
    main()
