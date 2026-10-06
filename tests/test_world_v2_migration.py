import math
import numpy as np
import pandas as pd
import torch

from src.constants import RAW_POINTS_13, DEFAULT_SEQ_LEN
from src.data.features import (
    extract_features_by_method,
    is_world_feature,
    compute_kinematic_angles_24,
)
from src.data.augmentations import LandmarkAugmenter
from src.data.dataset import GymDataset, build_dataset_from_csvs

def test_feature_semantics_and_shapes():
    # Construct a synthetic DataFrame with 13 joints in RAW_POINTS_13
    n_frames = 32
    cols = []
    data = {}
    for pt in RAW_POINTS_13:
        for d in ["x", "y", "z", "visibility"]:
            col = f"{pt}_{d}"
            # realistic body values: centered around 0 in meters
            data[col] = np.random.uniform(-0.5, 0.5, size=n_frames).astype(np.float32)
            if d == "visibility":
                data[col] = np.ones(n_frames, dtype=np.float32)
    df = pd.DataFrame(data)

    # 1. world_3d -> 39-d
    w3d = extract_features_by_method(df, "world_3d")
    assert w3d.shape == (n_frames, 39), f"Expected (32, 39), got {w3d.shape}"

    # 2. angle_kinematic_24 -> 24-d
    ang24 = compute_kinematic_angles_24(df)
    assert ang24.shape == (n_frames, 24), f"Expected (32, 24), got {ang24.shape}"

    # 3. mix_v2 -> 63-d
    mix = extract_features_by_method(df, "mix_v2")
    assert mix.shape == (n_frames, 63), f"Expected (32, 63), got {mix.shape}"
    np.testing.assert_allclose(mix[:, :39], w3d, atol=1e-6)
    np.testing.assert_allclose(mix[:, 39:], ang24, atol=1e-6)

    # 4. world_joint_motion_3d -> 39-d
    w_motion = extract_features_by_method(df, "world_joint_motion_3d")
    assert w_motion.shape == (n_frames, 39), f"Expected (32, 39), got {w_motion.shape}"

    # 5. raw_3d -> 39-d
    raw3d = extract_features_by_method(df, "raw_3d")
    assert raw3d.shape == (n_frames, 39)

    # 6. bone_3d -> 39-d
    bone3d = extract_features_by_method(df, "bone_3d")
    assert bone3d.shape == (n_frames, 39)

    # 7. is_world_feature check
    assert is_world_feature("world_3d") is True
    assert is_world_feature("mix_v2") is True
    assert is_world_feature("mix_v2_world") is True
    assert is_world_feature("world_joint_motion_3d") is True
    assert is_world_feature("raw_3d") is False
    assert is_world_feature("bone_3d") is False
    assert is_world_feature("rel_3d") is False
    print("✓ test_feature_semantics_and_shapes passed!")

def test_augmenter_63d():
    aug = LandmarkAugmenter(max_yaw_degrees=15.0, jitter_sigma=0.008)
    n_frames = 32
    # Create valid synthetic 63-d sequence
    # 13 joints x 3 coordinates in meters
    coords = torch.randn(n_frames, 13, 3) * 0.3
    w3d = coords.reshape(n_frames, 39)
    ang24 = aug.recompute_kinematic_angles_24(w3d)
    x = torch.cat([w3d, ang24], dim=-1)  # (32, 63)

    # A. Mirror
    mir = aug.mirror(x)
    assert mir.shape == (32, 63)
    # Mirror twice should recover original
    mir2 = aug.mirror(mir)
    # Check coords recovered
    torch.testing.assert_close(mir2[..., :39], x[..., :39], atol=1e-5, rtol=1e-5)
    # Check angles recovered
    torch.testing.assert_close(mir2[..., 39:], x[..., 39:], atol=1e-4, rtol=1e-4)

    # B. Scale
    scaled = aug.scale(x, scale_min=1.1, scale_max=1.1)
    assert scaled.shape == (32, 63)
    # Coordinates scaled by 1.1
    torch.testing.assert_close(scaled[..., :39], x[..., :39] * 1.1, atol=1e-5, rtol=1e-5)
    # Angles unchanged (scale-invariant)
    torch.testing.assert_close(scaled[..., 39:], x[..., 39:], atol=1e-5, rtol=1e-5)

    # C. Jitter
    jit = aug.jitter(x)
    assert jit.shape == (32, 63)
    # Angles are recomputed from jittered coordinates, verify exact consistency
    expected_ang = aug.recompute_kinematic_angles_24(jit[..., :39])
    torch.testing.assert_close(jit[..., 39:], expected_ang, atol=1e-5, rtol=1e-5)

    # D. Yaw rotation
    yaw = aug.yaw_rotate_3d(x, max_yaw_degrees=15.0)
    assert yaw.shape == (32, 63)
    expected_yaw_ang = aug.recompute_kinematic_angles_24(yaw[..., :39])
    torch.testing.assert_close(yaw[..., 39:], expected_yaw_ang, atol=1e-5, rtol=1e-5)

    # E. Time warp
    tw = aug.time_warp(x)
    assert tw.shape == (32, 63)

    # F. Candidate suite dispatch in apply
    cand_methods = [
        "candidate_full_5op",
        "candidate_minus_mirror",
        "candidate_minus_yaw",
        "candidate_minus_scale",
        "candidate_minus_time",
        "candidate_minus_jitter",
        "single_mirror",
        "single_yaw",
        "single_scale",
        "single_time",
        "single_jitter",
        "skel_gym_aug",
    ]
    for m in cand_methods:
        out = aug.apply(x, m)
        assert out.shape == (32, 63), f"Failed for method {m}: got {out.shape}"

    print("✓ test_augmenter_63d passed!")

if __name__ == "__main__":
    test_feature_semantics_and_shapes()
    test_augmenter_63d()
    print("ALL WORLD MIGRATION V2 TESTS PASSED!")
