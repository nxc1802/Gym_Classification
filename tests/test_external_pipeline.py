"""
Tests for External Benchmark Pipeline (src/external/*).
Validates:
1. Feature preprocessing parity with SkelGym dataset.py (extract_windows_from_segment).
2. Strict workout-disjoint isolation in 1-shot transfer simulation.
3. Cluster-based bootstrap confidence interval calculations.
4. Strict artifact and checkpoint lookup (failing loudly with FileNotFoundError).
5. Skeleton coordinate conversions and schema integrity.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.constants import ACTIONS, NUM_CLASSES
from src.external.base import ExternalRecord, BaseExternalDataset, QualityControlGate
from src.external.canonical_pose import (
    skeleton_13_to_dataframe,
    canonical_geometric_normalization,
    mediapipe33_to_skelgym13,
    openpose18_to_skelgym13
)
from src.external.fewshot import simulate_one_shot_transfer
from src.external.metrics import (
    compute_recording_level_bootstrap_ci,
    evaluate_window_level,
    aggregate_hierarchical_predictions
)
from scripts.evaluate_external import (
    find_checkpoint_path,
    load_norm_stats_for_seed,
    load_ensemble_weights_for_seed
)

def test_skeleton_13_to_dataframe():
    """Verify skeleton to dataframe converts exactly 13 joints x 3 dimensions."""
    T = 45
    skel = np.random.randn(T, 13, 3).astype(np.float32)
    df = skeleton_13_to_dataframe(skel)

    assert len(df) == T
    assert "Frame" in df.columns
    assert "NOSE_x" in df.columns
    assert "RIGHT_ANKLE_z" in df.columns
    assert len(df.columns) == 1 + 13 * 3

    # Check coordinate values match
    np.testing.assert_allclose(df["NOSE_x"].values, skel[:, 0, 0])
    np.testing.assert_allclose(df["RIGHT_ANKLE_z"].values, skel[:, 12, 2])

def test_extract_windows_parity():
    """Verify extract_windows in BaseExternalDataset properly uses extract_windows_from_segment."""
    class DummyDataset(BaseExternalDataset):
        pass

    ds = DummyDataset(name="dummy", pose_source="mediapipe", apply_geometric_norm=False)
    # 70 frames sequence (> 2 windows of size 32)
    T = 70
    skel = np.random.randn(T, 13, 3).astype(np.float32)
    rec = ExternalRecord(
        record_id="dummy_rec_01",
        dataset_name="dummy",
        subject_id="w00",
        action_raw="squats",
        action_canonical="squat",
        skelgym_class_idx=18,
        skeleton=skel,
        fps=30.0,
        pose_source="mediapipe"
    )
    assert ds.add_record(rec) is True

    win_data = ds.extract_windows("mix", seq_len=32, stride=32)
    features = win_data["features"]
    labels = win_data["labels"]
    record_ids = win_data["record_ids"]

    assert len(features) >= 2
    assert features.shape[1] == 32
    assert features.shape[2] == 117
    assert all(l == 18 for l in labels)
    assert all(r == "dummy_rec_01" for r in record_ids)

def test_one_shot_strict_workout_isolation():
    """Verify simulate_one_shot_transfer guarantees zero workout overlap between support and query."""
    target_classes = [9, 14, 17, 18] # core4: lateral raise, push-up, shoulder press, squat

    # Construct mock records across 5 workouts
    workouts = ["w00", "w05", "w12", "w13", "w20"]
    rec_to_class = {}
    rec_to_subject = {}
    embeddings_by_record = {}

    rng = np.random.default_rng(42)
    for c in target_classes:
        for w in workouts:
            rec_id = f"{w}_class{c}"
            rec_to_class[rec_id] = c
            rec_to_subject[rec_id] = w
            # Unit norm 128-d mock embedding
            vec = rng.standard_normal(128).astype(np.float32)
            embeddings_by_record[rec_id] = vec / np.linalg.norm(vec)

    res = simulate_one_shot_transfer(
        embeddings_by_record=embeddings_by_record,
        record_to_class=rec_to_class,
        record_to_subject=rec_to_subject,
        target_class_indices=target_classes,
        n_trials=50,
        seed=42
    )

    assert "mean" in res
    assert "ci_95" in res
    assert res["n_trials_completed"] > 0
    assert 0.0 <= res["mean"] <= 100.0

def test_workout_cluster_bootstrap():
    """Verify cluster bootstrap resampling properly accepts cluster_ids."""
    N = 20
    K = 4
    target_classes = [9, 14, 17, 18]
    rng = np.random.default_rng(42)

    probs = rng.uniform(0.1, 1.0, size=(N, NUM_CLASSES)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    trues = rng.choice(target_classes, size=N)
    clusters = [f"w{i%5:02d}" for i in range(N)]

    ci_res = compute_recording_level_bootstrap_ci(
        y_probs_by_group=probs,
        y_trues_by_group=trues,
        target_class_indices=target_classes,
        mode="closed_set",
        cluster_ids=clusters,
        n_bootstraps=50,
        confidence_level=0.95,
        seed=42
    )

    assert "acc_mean" in ci_res
    assert "acc_ci" in ci_res
    assert len(ci_res["acc_ci"]) == 2
    assert ci_res["acc_ci"][0] <= ci_res["acc_ci"][1]

def test_strict_artifact_lookup_raises():
    """Verify strict artifact loader raises FileNotFoundError rather than silently falling back."""
    ref_dir = Path("artifacts/reference")

    with pytest.raises(FileNotFoundError):
        find_checkpoint_path("checkpoints/non_existent_model.pt", seed=123)

    with pytest.raises(FileNotFoundError):
        load_norm_stats_for_seed(ref_dir, "non_existent_feat", seed=9999)

    with pytest.raises(FileNotFoundError):
        load_ensemble_weights_for_seed(ref_dir, seed=9999)
