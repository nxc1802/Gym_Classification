"""
Canonical 13-Joint Skeleton Schema and Geometric Normalization.
Ensures identical representation across MediaPipe re-extraction and Native 3D Skeleton pipelines (Fit3D & MM-Fit).
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.constants import RAW_POINTS_13

# Joint index positions in canonical SkelGym 13-joint convention
SKELGYM13_INDICES = {name: i for i, name in enumerate(RAW_POINTS_13)}

# OpenPose COCO-18 Joint Index Table:
# 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
# 5: LShoulder, 6: LElbow, 7: LWrist, 8: RHip, 9: RKnee,
# 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye,
# 15: LEye, 16: REar, 17: LEar
OPENPOSE18_TO_SKELGYM13 = {
    1: 5,   # LEFT_SHOULDER <- OpenPose LShoulder (5)
    2: 2,   # RIGHT_SHOULDER <- OpenPose RShoulder (2)
    3: 6,   # LEFT_ELBOW <- OpenPose LElbow (6)
    4: 3,   # RIGHT_ELBOW <- OpenPose RElbow (3)
    5: 7,   # LEFT_WRIST <- OpenPose LWrist (7)
    6: 4,   # RIGHT_WRIST <- OpenPose RWrist (4)
    7: 11,  # LEFT_HIP <- OpenPose LHip (11)
    8: 8,   # RIGHT_HIP <- OpenPose RHip (8)
    9: 12,  # LEFT_KNEE <- OpenPose LKnee (12)
    10: 9,  # RIGHT_KNEE <- OpenPose RKnee (9)
    11: 13, # LEFT_ANKLE <- OpenPose LAnkle (13)
    12: 10, # RIGHT_ANKLE <- OpenPose RAnkle (10)
}

# MediaPipe 33 landmark indices to SkelGym 13
MEDIAPIPE33_TO_SKELGYM13 = {
    0: 0,   # NOSE
    1: 11,  # LEFT_SHOULDER
    2: 12,  # RIGHT_SHOULDER
    3: 13,  # LEFT_ELBOW
    4: 14,  # RIGHT_ELBOW
    5: 15,  # LEFT_WRIST
    6: 16,  # RIGHT_WRIST
    7: 23,  # LEFT_HIP
    8: 24,  # RIGHT_HIP
    9: 25,  # LEFT_KNEE
    10: 26, # RIGHT_KNEE
    11: 27, # LEFT_ANKLE
    12: 28  # RIGHT_ANKLE
}

# Fit3D / Human3.6M 25-joint indices to SkelGym 13
# 0: Pelvis/Hips, 1: RHip, 2: RKnee, 3: RAnkle, 6: LHip, 7: LKnee, 8: LAnkle,
# 12: Neck/Thorax, 13: Head/Nose proxy, 17: LShoulder, 18: LElbow, 19: LWrist,
# 25: RShoulder, 26: RElbow, 27: RWrist (or 17-joint standard H36M subset)
H36M17_TO_SKELGYM13 = {
    0: 10,  # NOSE <- Head / Cranial proxy (10 in 17-joint H36M)
    1: 11,  # LEFT_SHOULDER
    2: 14,  # RIGHT_SHOULDER
    3: 12,  # LEFT_ELBOW
    4: 15,  # RIGHT_ELBOW
    5: 13,  # LEFT_WRIST
    6: 16,  # RIGHT_WRIST
    7: 4,   # LEFT_HIP
    8: 1,   # RIGHT_HIP
    9: 5,   # LEFT_KNEE
    10: 2,  # RIGHT_KNEE
    11: 6,  # LEFT_ANKLE
    12: 3   # RIGHT_ANKLE
}

def openpose18_to_skelgym13(pose_18: np.ndarray) -> np.ndarray:
    """
    Maps OpenPose-18 3D skeleton to canonical SkelGym 13 joints.
    Input shape: (T, 18, 3) or (3, T, 18).
    Returns shape: (T, 13, 3).
    Handles cranial landmark proxy: if OpenPose Nose (0) is invalid/dummy (>4000),
    constructs cranial proxy from Neck (1) and Hips (8, 11).
    """
    if pose_18.ndim == 3 and pose_18.shape[0] == 3:
        # Permute (3, T, 18) -> (T, 18, 3)
        pose_18 = np.transpose(pose_18, (1, 2, 0))

    T, V, C = pose_18.shape
    out = np.zeros((T, 13, 3), dtype=np.float32)

    # Map the 12 limb and torso joints
    for skel_idx, op_idx in OPENPOSE18_TO_SKELGYM13.items():
        out[:, skel_idx, :] = pose_18[:, op_idx, :3]

    # Handle Cranial Landmark (NOSE proxy, joint 0)
    nose_raw = pose_18[:, 0, :3]
    neck_raw = pose_18[:, 1, :3]
    l_hip = pose_18[:, 11, :3]
    r_hip = pose_18[:, 8, :3]
    mid_hip = (l_hip + r_hip) / 2.0

    # Check for OpenPose dummy / outlier coordinates (often 4050, 8150, or > 4000 mm)
    is_dummy = (np.abs(nose_raw) > 3500).any(axis=-1) | np.isnan(nose_raw).any(axis=-1)
    
    # Cranial proxy: Neck + 0.25 * (Neck - MidHip)
    cranial_proxy = neck_raw + 0.25 * (neck_raw - mid_hip)

    out[:, 0, :] = np.where(is_dummy[:, None], cranial_proxy, nose_raw)
    return out

def mediapipe33_to_skelgym13(pose_33: np.ndarray) -> np.ndarray:
    """
    Extracts canonical 13 joints from MediaPipe 33-landmark array.
    Input shape: (T, 33, 3) or (T, 33, 4).
    Returns shape: (T, 13, 3).
    """
    T = pose_33.shape[0]
    out = np.zeros((T, 13, 3), dtype=np.float32)
    for skel_idx, mp_idx in MEDIAPIPE33_TO_SKELGYM13.items():
        out[:, skel_idx, :] = pose_33[:, mp_idx, :3]
    return out

def fit3d_to_skelgym13(pose_fit3d: np.ndarray) -> np.ndarray:
    """
    Maps Fit3D native joints (17 or 25 joints) to canonical 13 joints.
    Input shape: (T, J, 3).
    Returns shape: (T, 13, 3).
    """
    T, J, _ = pose_fit3d.shape
    out = np.zeros((T, 13, 3), dtype=np.float32)
    if J == 17:
        for skel_idx, f_idx in H36M17_TO_SKELGYM13.items():
            out[:, skel_idx, :] = pose_fit3d[:, f_idx, :3]
    else:
        # 25 joints standard: fallback to closest Human3.6M indexing
        for skel_idx, f_idx in H36M17_TO_SKELGYM13.items():
            idx = min(f_idx, J - 1)
            out[:, skel_idx, :] = pose_fit3d[:, idx, :3]
    return out

def canonical_geometric_normalization(
    skeleton: np.ndarray,
    align_orientation: bool = True,
    canonical_torso_scale: float = 0.255
) -> np.ndarray:
    """
    Applies subject-independent geometric canonicalization:
    1. Translation invariance: Root re-centering relative to hip midpoint.
    2. Scale invariance: Normalization to SkelGym canonical torso scale (~0.255).
    3. Person-centric orientation alignment: Rotates body so that:
       - Up/Vertical axis maps to -Y (SkelGym convention: head has y < 0, hips at 0, ankles > 0).
       - Lateral axis (L_HIP - R_HIP) maps to X.
       - Depth axis maps to Z = cross(X, Y).
    Uses sequence-level robust median vectors to preserve genuine temporal joint kinematics.
    Input shape: (T, 13, 3).
    Returns: normalized skeleton of shape (T, 13, 3).
    """
    T, V, C = skeleton.shape
    norm_skel = skeleton.copy().astype(np.float32)

    # 1. Translation: Hip Midpoint (frame by frame to eliminate camera translation)
    l_hip = norm_skel[:, SKELGYM13_INDICES["LEFT_HIP"], :]
    r_hip = norm_skel[:, SKELGYM13_INDICES["RIGHT_HIP"], :]
    hip_mid = (l_hip + r_hip) / 2.0  # (T, 3)
    norm_skel -= hip_mid[:, None, :]

    # Torso vectors
    l_sh = norm_skel[:, SKELGYM13_INDICES["LEFT_SHOULDER"], :]
    r_sh = norm_skel[:, SKELGYM13_INDICES["RIGHT_SHOULDER"], :]
    sh_mid = (l_sh + r_sh) / 2.0  # (T, 3)

    torso_lengths = np.linalg.norm(sh_mid, axis=-1)  # (T,)
    valid_lens = torso_lengths[torso_lengths > 1e-4]
    med_torso = float(np.median(valid_lens)) if len(valid_lens) > 0 else 1.0

    if align_orientation:
        # Robust sequence-level vertical axis (from mid_hip [0,0,0] to mid_shoulder)
        # In SkelGym, head/shoulder points in -Y direction, so unit_y = - mid_sh
        sh_med = np.median(sh_mid, axis=0)
        norm_sh = np.linalg.norm(sh_med)
        if norm_sh > 1e-4:
            unit_y = - (sh_med / norm_sh)
        else:
            unit_y = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        # Robust lateral axis (L_HIP - R_HIP)
        v_lat = np.median(l_hip - r_hip, axis=0)
        # Project out component along unit_y
        v_lat_ortho = v_lat - np.dot(v_lat, unit_y) * unit_y
        norm_lat = np.linalg.norm(v_lat_ortho)
        if norm_lat > 1e-4:
            unit_x = v_lat_ortho / norm_lat
        else:
            unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Depth axis
        unit_z = np.cross(unit_x, unit_y)
        norm_z = np.linalg.norm(unit_z)
        if norm_z > 1e-4:
            unit_z /= norm_z
        else:
            unit_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        R = np.stack([unit_x, unit_y, unit_z], axis=0)  # (3, 3)

        # Scale factor
        scale = max(med_torso, 1e-4) / canonical_torso_scale

        for t in range(T):
            norm_skel[t] = np.dot(norm_skel[t] / scale, R.T)
    else:
        # Translation & simple torso scale only
        scale = max(med_torso, 1e-4) / canonical_torso_scale
        norm_skel /= scale

    return norm_skel


def resample_temporal_sequence(
    skeleton: np.ndarray,
    source_fps: float,
    target_fps: float = 30.0
) -> np.ndarray:
    """
    Temporal interpolation to resample skeleton sequences from source_fps to target_fps (30 Hz).
    Uses linear interpolation across time.
    Input shape: (T, 13, 3).
    Returns: resampled skeleton of shape (T_new, 13, 3).
    """
    if abs(source_fps - target_fps) < 1e-2 or len(skeleton) <= 1:
        return skeleton

    T, V, C = skeleton.shape
    new_T = int(round(T * (target_fps / source_fps)))
    if new_T <= 0:
        return skeleton

    t_tensor = torch.from_numpy(skeleton).float().permute(1, 2, 0).unsqueeze(0)  # (1, 13, 3, T)
    # Reshape to (1, 39, T)
    t_flat = t_tensor.reshape(1, V * C, T)
    t_resampled = F.interpolate(t_flat, size=new_T, mode="linear", align_corners=False)
    out = t_resampled.reshape(1, V, C, new_T).squeeze(0).permute(2, 0, 1).numpy()
    return out

def skeleton_13_to_dataframe(skeleton: np.ndarray) -> pd.DataFrame:
    """
    Converts (T, 13, 3) numpy array to pandas DataFrame matching SkelGym's landmark CSV format.
    Columns: Frame, NOSE_x, NOSE_y, NOSE_z, ..., RIGHT_ANKLE_z.
    """
    T = skeleton.shape[0]
    cols = ["Frame"]
    for pt in RAW_POINTS_13:
        for d in ["x", "y", "z"]:
            cols.append(f"{pt}_{d}")

    flat_data = np.zeros((T, len(cols)), dtype=np.float32)
    flat_data[:, 0] = np.arange(T)

    col_idx = 1
    for pt_i in range(13):
        for d_i in range(3):
            flat_data[:, col_idx] = skeleton[:, pt_i, d_i]
            col_idx += 1

    return pd.DataFrame(flat_data, columns=cols)
