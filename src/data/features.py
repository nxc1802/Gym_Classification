"""
Feature Engineering for Gym Pose Landmarks.
Computes raw coordinates (2D/3D), hip-midpoint-relative coordinates (2D/3D), joint angles (2D/3D), and mix representations.
"""

from itertools import combinations
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
import pandas as pd

from src.constants import RAW_POINTS_33, RAW_POINTS_13, REL_POINTS_12, HIP_MIDPOINT_JOINTS, KINEMATIC_TREE_13

def extract_raw_features(
    df: pd.DataFrame,
    points: List[str],
    dims: List[str] = ("x", "y", "z", "visibility")
) -> np.ndarray:
    """
    Extracts raw coordinate values for specified points and dimensions.
    Shape: (N_frames, len(points) * len(dims))
    """
    cols = []
    for pt in points:
        for d in dims:
            col = f"{pt}_{d}"
            cols.append(col if col in df.columns else None)

    data = []
    for c in cols:
        if c is not None:
            data.append(df[c].fillna(0.0).values)
        else:
            data.append(np.zeros(len(df), dtype=np.float32))

    return np.stack(data, axis=1).astype(np.float32)

def extract_relative_features(
    df: pd.DataFrame,
    points: List[str],
    dims: List[str] = ("x", "y", "z", "visibility"),
    include_origin_vis: bool = True
) -> np.ndarray:
    """
    Re-centers coordinates relative to the hip midpoint (average of LEFT_HIP and RIGHT_HIP).
    Coordinates (x, y, z) become (pt - hip_midpoint).
    Visibility is retained unchanged.
    All input points (including NOSE) are output as hip-midpoint-relative.
    Optionally appends average hip visibility as an additional feature.
    """
    coord_dims = [d for d in dims if d != "visibility"]
    hip_l, hip_r = HIP_MIDPOINT_JOINTS

    # Compute hip midpoint for each coordinate dimension
    origin_coords = {}
    for d in coord_dims:
        col_l = f"{hip_l}_{d}"
        col_r = f"{hip_r}_{d}"
        l_vals = df[col_l].fillna(0.0).values if col_l in df.columns else np.zeros(len(df), dtype=np.float32)
        r_vals = df[col_r].fillna(0.0).values if col_r in df.columns else np.zeros(len(df), dtype=np.float32)
        origin_coords[d] = (l_vals + r_vals) / 2.0

    features = []
    # All points are relative to hip midpoint (no point excluded)
    for pt in points:
        for d in dims:
            col = f"{pt}_{d}"
            arr = df[col].fillna(0.0).values if col in df.columns else np.zeros(len(df), dtype=np.float32)
            if d in coord_dims:
                arr = arr - origin_coords[d]
            features.append(arr)

    # Optionally append average hip visibility
    if include_origin_vis:
        vis_l_col = f"{hip_l}_visibility"
        vis_r_col = f"{hip_r}_visibility"
        vis_l = df[vis_l_col].fillna(0.0).values if vis_l_col in df.columns else np.ones(len(df), dtype=np.float32)
        vis_r = df[vis_r_col].fillna(0.0).values if vis_r_col in df.columns else np.ones(len(df), dtype=np.float32)
        features.append((vis_l + vis_r) / 2.0)

    return np.stack(features, axis=1).astype(np.float32)

def extract_relative_norm_features(
    df: pd.DataFrame,
    points: List[str] = RAW_POINTS_13,
    dims: List[str] = ("x", "y", "z")
) -> np.ndarray:
    """
    Anisotropic Anthropometric Scale Normalization:
    - Re-centers coordinates relative to mid-hip origin.
    - Scales X-axis by bi-iliac hip width (|LEFT_HIP_x - RIGHT_HIP_x|).
    - Scales Y-axis and Z-axis by torso length (|MID_SHOULDER_y - MID_HIP_y|).
    Produces anthropometrically invariant and camera-distance invariant 3D coordinates.
    Shape: (N_frames, len(points) * len(dims)) -> (N_frames, 39)
    """
    n_frames = len(df)
    coords = {}
    needed = set(points) | {"LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER", "NOSE"}
    for pt in needed:
        xs = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        ys = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        zs = df[f"{pt}_z"].fillna(0.0).values if f"{pt}_z" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords[pt] = np.stack([xs, ys, zs], axis=1)

    hip_mid = (coords["LEFT_HIP"] + coords["RIGHT_HIP"]) / 2.0
    sh_mid = (coords["LEFT_SHOULDER"] + coords["RIGHT_SHOULDER"]) / 2.0

    # Scale factors with protective floor to avoid division by zero
    lx = np.maximum(np.abs(coords["RIGHT_HIP"][:, 0] - coords["LEFT_HIP"][:, 0]), 0.05)
    ly = np.maximum(np.abs(sh_mid[:, 1] - hip_mid[:, 1]), 0.10)
    lz = ly

    scale = np.stack([lx, ly, lz], axis=1)  # (N_frames, 3)

    norm_features = []
    for pt in points:
        pt_centered = coords[pt] - hip_mid
        pt_norm = pt_centered / scale
        for d_i in range(len(dims)):
            norm_features.append(pt_norm[:, d_i].astype(np.float32))

    return np.stack(norm_features, axis=1).astype(np.float32)

def extract_bone_features(
    df: pd.DataFrame,
    points: List[str] = RAW_POINTS_13,
    dims: List[str] = ("x", "y", "z")
) -> np.ndarray:
    """
    Extracts skeletal bone vector representations e_u = X_u - X_parent(u).
    For the root joint (NOSE), the bone vector is set to zero ([0, 0, 0]).
    Bone vectors are strictly translation-invariant and explicitly capture limb segment lengths and orientations.
    Shape: (N_frames, len(points) * len(dims))
    """
    n_frames = len(df)
    coords = {}
    for pt in points:
        pt_coords = []
        for d in dims:
            col = f"{pt}_{d}"
            vals = df[col].fillna(0.0).values if col in df.columns else np.zeros(n_frames, dtype=np.float32)
            pt_coords.append(vals)
        coords[pt] = np.stack(pt_coords, axis=1)  # (N_frames, len(dims))

    bone_features = []
    for idx, pt in enumerate(points):
        parent_idx = KINEMATIC_TREE_13.get(idx)
        if parent_idx is not None and parent_idx < len(points):
            parent_pt = points[parent_idx]
            bone_vec = coords[pt] - coords[parent_pt]  # (N_frames, len(dims))
        else:
            bone_vec = np.zeros((n_frames, len(dims)), dtype=np.float32)

        for d_i in range(len(dims)):
            bone_features.append(bone_vec[:, d_i])

    return np.stack(bone_features, axis=1).astype(np.float32)

def compute_triplet_angles_2d(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes joint angles for all C(len(points), 3) triplet combinations on the 2D plane (x, y).
    For 13 points, total triplets = 286 angles in radians [0, pi].
    Shape: (N_frames, 286)
    """
    n_frames = len(df)
    coords_2d = {}
    for pt in points:
        x = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        y = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords_2d[pt] = np.stack([x, y], axis=1)  # (N, 2)

    triplets = list(combinations(points, 3))
    angles = np.zeros((n_frames, len(triplets)), dtype=np.float32)

    for idx, (a, b, c) in enumerate(triplets):
        pt_a = coords_2d[a]
        pt_b = coords_2d[b]  # Vertex
        pt_c = coords_2d[c]

        v1 = pt_a - pt_b
        v2 = pt_c - pt_b

        dot = np.sum(v1 * v2, axis=1)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)

        denom = norm1 * norm2 + 1e-7
        cos_theta = np.clip(dot / denom, -1.0, 1.0)
        angles[:, idx] = np.arccos(cos_theta)

    return angles

def compute_triplet_angles_3d(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes joint angles for all C(len(points), 3) triplet combinations in 3D space (x, y, z).
    Calculates spatial vector angle: arccos( (v1 . v2) / (|v1| * |v2|) ) in radians [0, pi].
    For 13 points, total triplets = 286 angles.
    Shape: (N_frames, 286)
    """
    n_frames = len(df)
    coords_3d = {}
    for pt in points:
        x = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        y = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        z = df[f"{pt}_z"].fillna(0.0).values if f"{pt}_z" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords_3d[pt] = np.stack([x, y, z], axis=1)  # (N, 3)

    triplets = list(combinations(points, 3))
    angles = np.zeros((n_frames, len(triplets)), dtype=np.float32)

    for idx, (a, b, c) in enumerate(triplets):
        pt_a = coords_3d[a]
        pt_b = coords_3d[b]  # Vertex
        pt_c = coords_3d[c]

        v1 = pt_a - pt_b
        v2 = pt_c - pt_b

        dot = np.sum(v1 * v2, axis=1)
        norm1 = np.linalg.norm(v1, axis=1)
        norm2 = np.linalg.norm(v2, axis=1)

        denom = norm1 * norm2 + 1e-7
        cos_theta = np.clip(dot / denom, -1.0, 1.0)
        angles[:, idx] = np.arccos(cos_theta)

    return angles

# Alias for backwards compatibility
compute_triplet_angles = compute_triplet_angles_2d

def compute_pair_angles_2d(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes absolute angle relative to horizontal axis for all C(len(points), 2) pairs in 2D.
    For 13 points, total pairs = 78 angles.
    theta = arctan2(dy, dx) in [-pi, pi].
    Shape: (N_frames, 78)
    """
    n_frames = len(df)
    coords_2d = {}
    for pt in points:
        x = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        y = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords_2d[pt] = np.stack([x, y], axis=1)

    pairs = list(combinations(points, 2))
    angles = np.zeros((n_frames, len(pairs)), dtype=np.float32)

    for idx, (a, b) in enumerate(pairs):
        dx = coords_2d[b][:, 0] - coords_2d[a][:, 0]
        dy = coords_2d[b][:, 1] - coords_2d[a][:, 1]
        angles[:, idx] = np.arctan2(dy, dx)

    return angles

# Alias for backwards compatibility
compute_pair_angles = compute_pair_angles_2d

def compute_pair_angles_3d(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes 3D inclination angle relative to the horizontal ground plane (X-Z) for all C(len(points), 2) pairs.
    In MediaPipe coordinates: Y is vertical, (X, Z) is horizontal ground plane.
    Elevation angle from horizontal ground: theta = arctan2(dy, sqrt(dx^2 + dz^2)) in [-pi/2, pi/2].
    Invariant to camera yaw rotation around the vertical Y-axis.
    For 13 points, total pairs = 78 angles.
    Shape: (N_frames, 78)
    """
    n_frames = len(df)
    coords_3d = {}
    for pt in points:
        x = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        y = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        z = df[f"{pt}_z"].fillna(0.0).values if f"{pt}_z" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords_3d[pt] = np.stack([x, y, z], axis=1)

    pairs = list(combinations(points, 2))
    angles = np.zeros((n_frames, len(pairs)), dtype=np.float32)

    for idx, (a, b) in enumerate(pairs):
        dx = coords_3d[b][:, 0] - coords_3d[a][:, 0]
        dy = coords_3d[b][:, 1] - coords_3d[a][:, 1]
        dz = coords_3d[b][:, 2] - coords_3d[a][:, 2]
        ground_dist = np.sqrt(dx**2 + dz**2)
        angles[:, idx] = np.arctan2(dy, ground_dist)

    return angles

def compute_kinematic_angles_24(df: pd.DataFrame) -> np.ndarray:
    """
    Computes 24 anatomical kinematic angles:
    14 Bone inclination angles (elevation from horizontal ground plane X-Z):
      1. Left Upper Arm (LEFT_SHOULDER -> LEFT_ELBOW)
      2. Right Upper Arm (RIGHT_SHOULDER -> RIGHT_ELBOW)
      3. Left Forearm (LEFT_ELBOW -> LEFT_WRIST)
      4. Right Forearm (RIGHT_ELBOW -> RIGHT_WRIST)
      5. Left Thigh (LEFT_HIP -> LEFT_KNEE)
      6. Right Thigh (RIGHT_HIP -> RIGHT_KNEE)
      7. Left Shin (LEFT_KNEE -> LEFT_ANKLE)
      8. Right Shin (RIGHT_KNEE -> RIGHT_ANKLE)
      9. Shoulder Girdle (LEFT_SHOULDER -> RIGHT_SHOULDER)
      10. Pelvic Girdle (LEFT_HIP -> RIGHT_HIP)
      11. Left Torso Flank (LEFT_SHOULDER -> LEFT_HIP)
      12. Right Torso Flank (RIGHT_SHOULDER -> RIGHT_HIP)
      13. Spine Axis (MID_HIP -> MID_SHOULDER)
      14. Neck Axis (MID_SHOULDER -> NOSE)
    10 3D Joint Articulation Angles (Triplets, 3D dot product in [0, pi]):
      15. Left Elbow Flexion (SHOULDER - ELBOW - WRIST)
      16. Right Elbow Flexion (SHOULDER - ELBOW - WRIST)
      17. Left Shoulder Angle (HIP - SHOULDER - ELBOW)
      18. Right Shoulder Angle (HIP - SHOULDER - ELBOW)
      19. Left Hip Hinge (SHOULDER - HIP - KNEE)
      20. Right Hip Hinge (SHOULDER - HIP - KNEE)
      21. Left Knee Flexion (HIP - KNEE - ANKLE)
      22. Right Knee Flexion (HIP - KNEE - ANKLE)
      23. Torso Posture Angle (NOSE - MID_SHOULDER - MID_HIP)
      24. Arm Splay Angle (LEFT_ELBOW - MID_SHOULDER - RIGHT_ELBOW)
    Shape: (N_frames, 24)
    """
    n_frames = len(df)
    coords = {}
    needed = [
        "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE",
        "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
    ]
    for pt in needed:
        xs = df[f"{pt}_x"].fillna(0.0).values if f"{pt}_x" in df.columns else np.zeros(n_frames, dtype=np.float32)
        ys = df[f"{pt}_y"].fillna(0.0).values if f"{pt}_y" in df.columns else np.zeros(n_frames, dtype=np.float32)
        zs = df[f"{pt}_z"].fillna(0.0).values if f"{pt}_z" in df.columns else np.zeros(n_frames, dtype=np.float32)
        coords[pt] = np.stack([xs, ys, zs], axis=1)

    coords["MID_HIP"] = (coords["LEFT_HIP"] + coords["RIGHT_HIP"]) / 2.0
    coords["MID_SHOULDER"] = (coords["LEFT_SHOULDER"] + coords["RIGHT_SHOULDER"]) / 2.0

    def bone_elev(p1, p2):
        d = p2 - p1
        ground = np.maximum(np.sqrt(d[:, 0]**2 + d[:, 2]**2), 1e-4)
        return np.arctan2(d[:, 1], ground).astype(np.float32)

    def joint_angle(pa, pb, pc):
        # angle at vertex pb
        v1 = pa - pb
        v2 = pc - pb
        v1_norm = np.maximum(np.linalg.norm(v1, axis=1), 1e-4)
        v2_norm = np.maximum(np.linalg.norm(v2, axis=1), 1e-4)
        cos_a = np.sum(v1 * v2, axis=1) / (v1_norm * v2_norm)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        return np.arccos(cos_a).astype(np.float32)

    angles = [
        # 14 Bone elevations
        bone_elev(coords["LEFT_SHOULDER"], coords["LEFT_ELBOW"]),
        bone_elev(coords["RIGHT_SHOULDER"], coords["RIGHT_ELBOW"]),
        bone_elev(coords["LEFT_ELBOW"], coords["LEFT_WRIST"]),
        bone_elev(coords["RIGHT_ELBOW"], coords["RIGHT_WRIST"]),
        bone_elev(coords["LEFT_HIP"], coords["LEFT_KNEE"]),
        bone_elev(coords["RIGHT_HIP"], coords["RIGHT_KNEE"]),
        bone_elev(coords["LEFT_KNEE"], coords["LEFT_ANKLE"]),
        bone_elev(coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"]),
        bone_elev(coords["LEFT_SHOULDER"], coords["RIGHT_SHOULDER"]),
        bone_elev(coords["LEFT_HIP"], coords["RIGHT_HIP"]),
        bone_elev(coords["LEFT_SHOULDER"], coords["LEFT_HIP"]),
        bone_elev(coords["RIGHT_SHOULDER"], coords["RIGHT_HIP"]),
        bone_elev(coords["MID_HIP"], coords["MID_SHOULDER"]),
        bone_elev(coords["MID_SHOULDER"], coords["NOSE"]),
        # 10 Joint articulation angles
        joint_angle(coords["LEFT_SHOULDER"], coords["LEFT_ELBOW"], coords["LEFT_WRIST"]),
        joint_angle(coords["RIGHT_SHOULDER"], coords["RIGHT_ELBOW"], coords["RIGHT_WRIST"]),
        joint_angle(coords["LEFT_HIP"], coords["LEFT_SHOULDER"], coords["LEFT_ELBOW"]),
        joint_angle(coords["RIGHT_HIP"], coords["RIGHT_SHOULDER"], coords["RIGHT_ELBOW"]),
        joint_angle(coords["LEFT_SHOULDER"], coords["LEFT_HIP"], coords["LEFT_KNEE"]),
        joint_angle(coords["RIGHT_SHOULDER"], coords["RIGHT_HIP"], coords["RIGHT_KNEE"]),
        joint_angle(coords["LEFT_HIP"], coords["LEFT_KNEE"], coords["LEFT_ANKLE"]),
        joint_angle(coords["RIGHT_HIP"], coords["RIGHT_KNEE"], coords["RIGHT_ANKLE"]),
        joint_angle(coords["NOSE"], coords["MID_SHOULDER"], coords["MID_HIP"]),
        joint_angle(coords["LEFT_ELBOW"], coords["MID_SHOULDER"], coords["RIGHT_ELBOW"]),
    ]

    return np.stack(angles, axis=1).astype(np.float32)

def extract_mix_features(df: pd.DataFrame, components: Optional[List[str]] = None) -> np.ndarray:
    """
    Extracts unified Mix representation by dynamically concatenating arbitrary feature sets.
    Default components: ["rel_3d", "angle2_3d"] -> 39 + 78 = 117 dimensions.
    """
    if components is None:
        components = ["rel_3d", "angle2_3d"]

    extracted = []
    for comp in components:
        feat = extract_features_by_method(df, comp.strip())
        if isinstance(feat, tuple):
            feat = np.concatenate(feat, axis=1)
        extracted.append(feat)

    return np.concatenate(extracted, axis=1).astype(np.float32)

def extract_joint_motion_features(
    df: pd.DataFrame,
    points: List[str] = RAW_POINTS_13,
    dims: List[str] = ["x", "y", "z"]
) -> np.ndarray:
    """
    Computes Joint Motion (Temporal Velocity) stream:
    M_joint(t) = X_rel(t+1) - X_rel(t)
    Final frame is padded with the last computed motion vector.
    Shape: (N_frames, len(points) * len(dims)) -> (N_frames, 39) for 13 3D joints.
    """
    rel_pos = extract_relative_features(df, points=points, dims=dims, include_origin_vis=False)
    n_frames = rel_pos.shape[0]
    if n_frames <= 1:
        return np.zeros_like(rel_pos, dtype=np.float32)

    motion = np.zeros_like(rel_pos, dtype=np.float32)
    motion[:-1] = rel_pos[1:] - rel_pos[:-1]
    motion[-1] = motion[-2]
    return motion

def extract_bone_motion_features(
    df: pd.DataFrame,
    points: List[str] = RAW_POINTS_13,
    dims: List[str] = ["x", "y", "z"]
) -> np.ndarray:
    """
    Computes Bone Motion (Angular/Deformation Velocity) stream:
    M_bone(t) = B(t+1) - B(t)
    Final frame is padded with the last computed motion vector.
    Shape: (N_frames, len(points) * len(dims)) -> (N_frames, 39) for 13 3D bones.
    """
    bone_vec = extract_bone_features(df, points=points, dims=dims)
    n_frames = bone_vec.shape[0]
    if n_frames <= 1:
        return np.zeros_like(bone_vec, dtype=np.float32)

    motion = np.zeros_like(bone_vec, dtype=np.float32)
    motion[:-1] = bone_vec[1:] - bone_vec[:-1]
    motion[-1] = motion[-2]
    return motion

def extract_features_by_method(df: pd.DataFrame, method: str) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Dispatches feature extraction based on method name.
    Supports single feature streams or dynamic combinations via '+' or 'mix:':
      - raw_2d (26), raw_3d (39), raw_13 (39), raw_13_2d (26), raw_13_3d (39), raw_13_4 (52)
      - rel_2d (26), rel_3d (39), rel_13 (39), rel_13_2d (26), rel_13_3d (39), rel_13_4 (53)
      - bone_2d (26), bone_3d (39)
      - joint_motion_2d (26), joint_motion_3d (39)
      - bone_motion_2d (26), bone_motion_3d (39)
      - angle_2d (286), angle_3d (286)
      - angle2_2d (78), angle2_3d (78)
      - mix (117): Default unified rel_3d (39) + angle2_3d (78)
      - Arbitrary combinations: e.g. 'rel_3d+angle2_3d', 'mix:raw_3d,bone_3d'
    """
    # Dynamic combination check
    if "+" in method:
        parts = [p.strip() for p in method.split("+") if p.strip()]
        return extract_mix_features(df, components=parts)
    elif method.startswith("mix:"):
        inner = method[4:]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return extract_mix_features(df, components=parts)
    elif method == "mix":
        return extract_mix_features(df, components=["rel_3d", "angle2_3d"])  # 117

    if method in ("raw_2d", "raw_13_2d"):
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y"])  # 26
    elif method in ("raw_3d", "raw_13", "raw_13_3d"):
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method in ("rel_2d", "rel_13_2d"):
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y"], include_origin_vis=False)  # 26
    elif method in ("rel_3d", "rel_13", "rel_13_3d"):
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z"], include_origin_vis=False)  # 39
    elif method in ("world_3d", "world_13_3d"):
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39 (already mid-hip centered and metric)
    elif method in ("raw_13_4", "13_4"):
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"])  # 52
    elif method in ("rel_13_4", "12rel_4"):
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 53
    elif method == "bone_2d":
        return extract_bone_features(df, RAW_POINTS_13, ["x", "y"])  # 26
    elif method == "bone_3d":
        return extract_bone_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method == "joint_motion_2d":
        return extract_joint_motion_features(df, RAW_POINTS_13, ["x", "y"])  # 26
    elif method == "joint_motion_3d":
        return extract_joint_motion_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method == "bone_motion_2d":
        return extract_bone_motion_features(df, RAW_POINTS_13, ["x", "y"])  # 26
    elif method == "bone_motion_3d":
        return extract_bone_motion_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method == "angle_2d":
        return compute_triplet_angles_2d(df, RAW_POINTS_13)  # 286
    elif method == "angle_3d":
        return compute_triplet_angles_3d(df, RAW_POINTS_13)  # 286
    elif method in ("angle2_2d", "angle2"):
        return compute_pair_angles_2d(df, RAW_POINTS_13)  # 78
    elif method == "angle2_3d":
        return compute_pair_angles_3d(df, RAW_POINTS_13)  # 78
    elif method in ("rel_3d_norm", "rel_norm", "rel_norm_3d"):
        return extract_relative_norm_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method in ("angle_kinematic_24", "kinematic_24", "angle_24"):
        return compute_kinematic_angles_24(df)  # 24
    elif method in ("mix_v2", "mix_63", "biomechanical_mix_v2"):
        rel_n = extract_relative_norm_features(df, RAW_POINTS_13, ["x", "y", "z"])
        ang_24 = compute_kinematic_angles_24(df)
        return np.concatenate([rel_n, ang_24], axis=1).astype(np.float32)  # 63

    # Legacy support
    elif method == "full_4":
        return extract_raw_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"])  # 132
    elif method == "full_rel_4":
        return extract_relative_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 133
    elif method == "angle3":
        return compute_triplet_angles_2d(df, RAW_POINTS_13)  # 286
    elif method == "direct_concat":
        rel = extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 53
        ang = compute_triplet_angles_2d(df, RAW_POINTS_13)  # 286
        return np.concatenate([rel, ang], axis=1).astype(np.float32)  # 339
    elif method == "branch_concat":
        rel = extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 53
        ang = compute_triplet_angles_2d(df, RAW_POINTS_13)  # 286
        return (rel, ang)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")

