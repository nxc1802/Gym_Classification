#!/usr/bin/env python3
"""
Feature Parity Verification Unit Test (external_test.md Phase M).
Verifies that the new external dataset canonicalization pipeline produces
100% numerically identical features to the original SkelGym training pipeline
across all 5 constituent feature modalities:
  1. mix (117-d)
  2. rel_3d (39-d)
  3. bone_3d (39-d)
  4. joint_motion_3d (39-d)
  5. bone_motion_3d (39-d)
Tolerance: max absolute error < 1e-5.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RAW_POINTS_13
from src.data.features import extract_features_by_method
from src.external.canonical_pose import skeleton_13_to_dataframe

def test_feature_parity():
    # Use a reference SkelGym test CSV
    csv_candidates = list((PROJECT_ROOT / "data" / "landmarks" / "test").rglob("*.csv"))
    if not csv_candidates:
        print("[Warning] No SkelGym test CSV found. Skipping test.")
        return

    csv_path = csv_candidates[0]
    print(f"Testing feature parity on sample: {csv_path.relative_to(PROJECT_ROOT)}")

    df_orig = pd.read_csv(csv_path)

    # Reconstruct canonical (T, 13, 3) skeleton
    skel13 = np.zeros((len(df_orig), 13, 3), dtype=np.float32)
    for i, pt in enumerate(RAW_POINTS_13):
        skel13[:, i, 0] = df_orig[f"{pt}_x"].values
        skel13[:, i, 1] = df_orig[f"{pt}_y"].values
        skel13[:, i, 2] = df_orig[f"{pt}_z"].values

    df_canon = skeleton_13_to_dataframe(skel13)

    modalities = ["mix", "rel_3d", "bone_3d", "joint_motion_3d", "bone_motion_3d"]
    all_passed = True

    print("\nModality Parity Results:")
    print("-" * 60)
    for mod in modalities:
        f_orig = extract_features_by_method(df_orig, mod)
        f_canon = extract_features_by_method(df_canon, mod)

        assert f_orig.shape == f_canon.shape, f"Shape mismatch in {mod}: {f_orig.shape} vs {f_canon.shape}"
        max_diff = float(np.max(np.abs(f_orig - f_canon)))
        mean_diff = float(np.mean(np.abs(f_orig - f_canon)))

        status = "PASSED" if max_diff < 1e-5 else "FAILED"
        print(f"  {mod:18s} | Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e} | [{status}]")
        if max_diff >= 1e-5:
            all_passed = False

    print("-" * 60)
    if all_passed:
        print("ALL 5 FEATURE MODALITIES PASSED STRICT PARITY CHECK (< 1e-5)!")
    else:
        print("SOME MODALITIES FAILED PARITY CHECK.")
        sys.exit(1)

if __name__ == "__main__":
    test_feature_parity()
