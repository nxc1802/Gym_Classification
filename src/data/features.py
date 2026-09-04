"""
Feature Engineering for Gym Pose Landmarks.
Computes raw coordinates, nose-relative coordinates, joint angles, and concatenations.
"""

from itertools import combinations
from typing import List, Tuple, Dict, Callable
import numpy as np
import pandas as pd

from src.constants import RAW_POINTS_33, RAW_POINTS_13, REL_POINTS_12

def extract_raw_features(df: pd.DataFrame, points: List[str], dims: List[str] = ("x", "y", "z", "visibility")) -> np.ndarray:
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
    Re-centers coordinates relative to NOSE.
    Coordinates (x, y, z) become (pt - nose).
    Visibility is retained unchanged.
    Optionally appends NOSE_visibility as an additional feature.
    """
    # Origin coordinates (NOSE)
    coord_dims = [d for d in dims if d != "visibility"]
    origin_coords = {}
    for d in coord_dims:
        col = f"NOSE_{d}"
        origin_coords[d] = df[col].fillna(0.0).values if col in df.columns else np.zeros(len(df), dtype=np.float32)

    features = []
    # For each non-origin point
    for pt in points:
        if pt == "NOSE":
            continue
        for d in dims:
            col = f"{pt}_{d}"
            arr = df[col].fillna(0.0).values if col in df.columns else np.zeros(len(df), dtype=np.float32)
            if d in coord_dims:
                arr = arr - origin_coords[d]
            features.append(arr)

    if include_origin_vis and "NOSE_visibility" in df.columns:
        features.append(df["NOSE_visibility"].fillna(0.0).values)

    return np.stack(features, axis=1).astype(np.float32)

def compute_triplet_angles(df: pd.DataFrame, points: List[str] = RAW_POINTS_13) -> np.ndarray:
    """
    Computes joint angles for all C(len(points), 3) triplet combinations using vector dot product.
    For 13 points, total triplets = 286 angles.
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

def extract_features_by_method(df: pd.DataFrame, method: str):
    """
    Dispatches feature extraction based on method name.
    Returns:
      - np.ndarray (for single-branch methods)
      - Tuple[np.ndarray, np.ndarray] (for branch_concat)
    """
    if method == "full_4":
        return extract_raw_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"])  # 132
    elif method == "full_rel_4":
        return extract_relative_features(df, RAW_POINTS_33, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 129
    elif method == "13_4":
        return extract_raw_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"])  # 52
    elif method == "12rel_4":
        return extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 49
    elif method == "angle3":
        return compute_triplet_angles(df, RAW_POINTS_13)  # 286
    elif method == "angle2":
        return compute_pair_angles(df, RAW_POINTS_13)  # 78
    elif method == "direct_concat":
        rel = extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 49
        ang = compute_triplet_angles(df, RAW_POINTS_13)  # 286
        return np.concatenate([rel, ang], axis=1).astype(np.float32)  # 335
    elif method == "branch_concat":
        rel = extract_relative_features(df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)  # 49
        ang = compute_triplet_angles(df, RAW_POINTS_13)  # 286
        return (rel, ang)
    else:
        raise ValueError(f"Unsupported feature extraction method: {method}")
