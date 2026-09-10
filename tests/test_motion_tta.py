"""
Unit tests for FocalLoss, 4-Stream Motion Extractors, Horizontal Mirroring TTA,
and Video-Level Aggregation.
"""

import numpy as np
import pandas as pd
import torch
import pytest

from src.training.trainer import FocalLoss
from src.data.features import (
    extract_joint_motion_features,
    extract_bone_motion_features,
    extract_features_by_method
)
from src.data.dataset import mirror_dataframe_horizontally
from src.models.ensemble import aggregate_video_level_predictions
from src.constants import RAW_POINTS_13, FEATURE_DIMS

def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    # Easy prediction: high probability on target class
    logits_easy = torch.tensor([[10.0, -10.0, -10.0]], dtype=torch.float32)
    targets = torch.tensor([0], dtype=torch.long)
    loss_easy = loss_fn(logits_easy, targets).item()

    # Hard prediction: low probability on target class
    logits_hard = torch.tensor([[-10.0, 10.0, 10.0]], dtype=torch.float32)
    loss_hard = loss_fn(logits_hard, targets).item()

    assert not np.isnan(loss_easy)
    assert not np.isnan(loss_hard)
    assert loss_hard > loss_easy, "Focal loss must penalize hard examples significantly more than easy examples"

def test_motion_feature_extraction():
    n_frames = 10
    cols = ["Frame"]
    for pt in RAW_POINTS_13:
        for d in ["x", "y", "z", "visibility"]:
            cols.append(f"{pt}_{d}")

    data = np.zeros((n_frames, len(cols)), dtype=np.float32)
    df = pd.DataFrame(data, columns=cols)
    # Put a linear motion in NOSE_x
    df["NOSE_x"] = np.linspace(0.0, 1.0, n_frames)

    j_motion = extract_joint_motion_features(df)
    b_motion = extract_bone_motion_features(df)

    assert j_motion.shape == (n_frames, 39)
    assert b_motion.shape == (n_frames, 39)
    assert FEATURE_DIMS["joint_motion_3d"] == 39
    assert FEATURE_DIMS["bone_motion_3d"] == 39

    # Test via dispatch
    feat_j = extract_features_by_method(df, "joint_motion_3d")
    feat_b = extract_features_by_method(df, "bone_motion_3d")
    assert feat_j.shape == (n_frames, 39)
    assert feat_b.shape == (n_frames, 39)

def test_horizontal_mirror_symmetry():
    cols = ["NOSE_x", "LEFT_SHOULDER_x", "RIGHT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_y"]
    df = pd.DataFrame({
        "NOSE_x": [0.5],
        "LEFT_SHOULDER_x": [0.6],
        "RIGHT_SHOULDER_x": [0.4],
        "LEFT_SHOULDER_y": [0.3],
        "RIGHT_SHOULDER_y": [0.3]
    })

    mirrored = mirror_dataframe_horizontally(df)
    assert mirrored["NOSE_x"].iloc[0] == -0.5
    # Swapped left <-> right with negation for x
    assert mirrored["LEFT_SHOULDER_x"].iloc[0] == -0.4
    assert mirrored["RIGHT_SHOULDER_x"].iloc[0] == -0.6
    # Y-coordinates swapped without negation
    assert mirrored["LEFT_SHOULDER_y"].iloc[0] == 0.3
    assert mirrored["RIGHT_SHOULDER_y"].iloc[0] == 0.3

def test_video_level_aggregation():
    # 4 windows from 2 videos
    y_probs = np.array([
        [0.8, 0.2],  # Video A win 1 -> true=0
        [0.6, 0.4],  # Video A win 2 -> true=0
        [0.3, 0.7],  # Video B win 1 -> true=1
        [0.2, 0.8]   # Video B win 2 -> true=1
    ])
    y_trues = np.array([0, 0, 1, 1])
    video_ids = ["vid_A", "vid_A", "vid_B", "vid_B"]

    y_vid_t, y_vid_p, y_vid_pr, metrics = aggregate_video_level_predictions(y_probs, y_trues, video_ids)

    assert len(y_vid_t) == 2
    assert len(y_vid_p) == 2
    assert (y_vid_t == np.array([0, 1])).all()
    assert (y_vid_p == np.array([0, 1])).all()
    assert metrics["accuracy"] == 1.0
