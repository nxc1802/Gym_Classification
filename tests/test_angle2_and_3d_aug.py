"""
Unit tests for 2-point relative angles (angle2_2d, angle2_3d), 13-joint aliases,
and 3D-preserving LandmarkAugmenter integrity.
"""

import numpy as np
import pandas as pd
import pytest

from src.constants import RAW_POINTS_13, FEATURE_DIMS
from src.data.features import (
    compute_pair_angles_2d,
    compute_pair_angles_3d,
    extract_features_by_method
)
from src.data.augmentations import LandmarkAugmenter

def _generate_synthetic_df(n_frames=20):
    cols = ["Frame"]
    for pt in RAW_POINTS_13:
        for d in ["x", "y", "z", "visibility"]:
            cols.append(f"{pt}_{d}")
    
    np.random.seed(42)
    data = np.random.uniform(0.1, 0.9, size=(n_frames, len(cols))).astype(np.float32)
    df = pd.DataFrame(data, columns=cols)
    return df

def test_pair_angles_computation_and_bounds():
    df = _generate_synthetic_df()
    
    # 2D relative angles
    ang2_2d = compute_pair_angles_2d(df)
    assert ang2_2d.shape == (20, 78)
    assert np.all(ang2_2d >= -np.pi) and np.all(ang2_2d <= np.pi)
    
    # 3D elevation relative angles
    ang2_3d = compute_pair_angles_3d(df)
    assert ang2_3d.shape == (20, 78)
    assert np.all(ang2_3d >= -np.pi / 2.0) and np.all(ang2_3d <= np.pi / 2.0)

def test_feature_dispatch_and_dimensions():
    df = _generate_synthetic_df()
    
    expected_dims = {
        "angle2_2d": 78,
        "angle2_3d": 78,
        "angle_2d": 286,
        "angle_3d": 286,
        "mix": 117,
        "raw_13": 39,
        "raw_13_2d": 26,
        "raw_13_3d": 39,
        "raw_13_4": 52,
        "rel_13": 39,
        "rel_13_2d": 26,
        "rel_13_3d": 39,
        "rel_13_4": 53
    }
    
    for feat_name, expected_dim in expected_dims.items():
        assert FEATURE_DIMS[feat_name] == expected_dim
        feat_data = extract_features_by_method(df, feat_name)
        assert feat_data.shape == (20, expected_dim), f"Feature {feat_name} shape mismatch"

def test_augmenter_preserves_3d_and_angle_features():
    import torch
    aug = LandmarkAugmenter()
    
    # 1. Coordinate data with 4 channels (x, y, z, vis)
    raw_4c = torch.empty(16, 52).uniform_(0.1, 0.9)
    # Set visibility to 1.0 explicitly
    raw_4c[:, 3::4] = 1.0
    
    # Test scale
    scaled = aug.scale(raw_4c)
    torch.testing.assert_close(scaled[:, 3::4], torch.ones_like(scaled[:, 3::4]), msg="Scale must preserve visibility channel")
    
    # Test yaw rotation
    yawed = aug.yaw_rotate_3d(raw_4c)
    torch.testing.assert_close(yawed[:, 3::4], torch.ones_like(yawed[:, 3::4]), msg="Yaw rotation must preserve visibility channel")
    torch.testing.assert_close(yawed[:, 1::4], raw_4c[:, 1::4], msg="Yaw rotation around Y must preserve Y coordinates")
    
    # 2. Angle features (286 dims)
    angles_triplet = torch.empty(16, 286).uniform_(0.1, 3.0)
    
    # Mirroring: must permute indices bijectively and keep values in [0, pi]
    mirrored_triplet = aug.mirror(angles_triplet)
    assert mirrored_triplet.shape == (16, 286)
    assert torch.all(mirrored_triplet >= 0.0)
    # Jitter on triplet: must remain bounded in [0, pi]
    jittered_triplet = aug.jitter(angles_triplet)
    assert torch.all(jittered_triplet >= 0.0) and torch.all(jittered_triplet <= np.pi + 1e-4)
    
    # 3. Angle2 features (78 dims)
    angles_pair = torch.empty(16, 78).uniform_(-1.5, 1.5)
    mirrored_pair = aug.mirror(angles_pair)
    assert mirrored_pair.shape == (16, 78)
    jittered_pair = aug.jitter(angles_pair)
    assert torch.all(jittered_pair >= -np.pi - 1e-4) and torch.all(jittered_pair <= np.pi + 1e-4)

    # 4. Mix representation (117 dims = 39 rel_3d + 78 angle2_3d)
    mix_t = torch.cat([raw_4c[:, :39], angles_pair], dim=-1)
    assert mix_t.shape == (16, 117)
    
    # Test mirror on mix
    mirrored_mix = aug.mirror(mix_t)
    assert mirrored_mix.shape == (16, 117)
    assert torch.all(mirrored_mix[:, 39:] >= -np.pi - 1e-4)
    
    # Test yaw_rotate_3d on mix
    yawed_mix = aug.yaw_rotate_3d(mix_t)
    assert yawed_mix.shape == (16, 117)
    torch.testing.assert_close(yawed_mix[:, 39:], mix_t[:, 39:], msg="Angle2 channels must be invariant to yaw rotation")
    
    # Test scale on mix
    scaled_mix = aug.scale(mix_t)
    assert scaled_mix.shape == (16, 117)
    torch.testing.assert_close(scaled_mix[:, 39:], mix_t[:, 39:], msg="Angle2 channels must be invariant to scaling")
    
    # Test full skel_gym_aug on mix
    aug_mix = aug.skel_gym_aug(mix_t)
    assert aug_mix.shape == (16, 117)


