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

def compute_pair_angles(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes absolute angle relative to horizontal axis for all C(len(points), 2) pairs.
    For 13 points, total pairs = 78 angles.
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

def extract_mix_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts the unified Proposed Mix representation:
    Combines hip-midpoint-relative coordinates (rel_3d: 13*3 = 39 dims)
    and joint angles (angle_3d: 286 dims) = 325 dimensions.
    NOTE: Raw concatenation without per-sequence z-score.
    Normalization should be applied at dataset level using train-set statistics.
    """
    rel = extract_relative_features(df, RAW_POINTS_13, dims=["x", "y", "z"], include_origin_vis=False)  # (N, 39)
    ang = compute_triplet_angles_3d(df, RAW_POINTS_13)  # (N, 286)

    return np.concatenate([rel, ang], axis=1).astype(np.float32)

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
    Unified feature extraction dispatch matching the exact paper specification:
      - raw_2d (26), raw_3d (39)
      - rel_2d (26), rel_3d (39)
      - bone_2d (26), bone_3d (39)
      - joint_motion_2d (26), joint_motion_3d (39)
      - bone_motion_2d (26), bone_motion_3d (39)
      - angle_2d (286), angle_3d (286)
      - mix (325): Unified angle_3d + rel_3d + velocity
    """
    if method == "raw_2d":
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y"])  # 26
    elif method == "raw_3d":
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z"])  # 39
    elif method == "rel_2d":
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y"], include_origin_vis=False)  # 26
    elif method == "rel_3d":
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z"], include_origin_vis=False)  # 39
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
    elif method == "mix":
        return extract_mix_features(df)  # 325

    # Legacy support
    elif method == "full_4":
        return extract_raw_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"])  # 132
    elif method == "full_rel_4":
        return extract_relative_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 133
    elif method == "13_4":
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"])  # 52
    elif method == "12rel_4":
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 53
    elif method == "angle3":
        return compute_triplet_angles_2d(df, RAW_POINTS_13)  # 286
    elif method == "angle2":
        return compute_pair_angles(df, RAW_POINTS_13)  # 78
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
