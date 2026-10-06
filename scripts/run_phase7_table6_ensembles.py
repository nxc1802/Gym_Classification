#!/usr/bin/env python3
"""
Phase 7: Table 6 Cross-Paradigm Fusion Protocols (SkelGym-Lite & SkelGym-Full).
Evaluates 3 standard voting methods across 3 random seeds (42, 123, 3407):
  1. Hard Majority Voting (Discrete Baseline)
  2. Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K)
  3. Accuracy-Weighted Soft Voting (Validation-calibrated weights)

Configurations:
  - SkelGym-Lite (2 Streams: Transformer + Bone AAGCN, ~679K params)
  - SkelGym-Full (5 Streams: Transformer + 4 AAGCN Streams, ~1.81M params)

Zero-Leakage:
  - Weights for Accuracy-Weighted Soft Voting are computed exclusively from the validation split.
  - Test predictions are strictly aggregated without test supervision.

Outputs: outputs/table6_cross_paradigm_fusion.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

def send_marimo_toast(msg: str):
    try:
        import marimo as mo
        mo.status.toast(msg)
    except Exception:
        pass

def hard_voting(prob_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Hard majority voting: picks the argmax class with highest count across models.
    """
    preds = [np.argmax(p, axis=1) for p in prob_matrices]
    stacked = np.stack(preds, axis=0)  # (K, N)
    K, N = stacked.shape
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        col = stacked[:, i]
        vals, counts = np.unique(col, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

def evaluate_fusion_method(
    stream_runs_by_seed: Dict[int, List[Dict[str, Any]]],
    method: str
) -> Dict[str, Any]:
    """
    Evaluates a specific fusion method across all 3 seeds.
    method: 'hard', 'uniform_soft', 'accuracy_weighted_soft'
    """
    seed_metrics = []

    for s in SEEDS:
        runs = stream_runs_by_seed[s]
        val_probs = [np.array(r["val_probs"]) for r in runs]
        test_probs = [np.array(r["test_probs"]) for r in runs]
        val_targets = np.array(runs[0]["val_targets"])
        test_targets = np.array(runs[0]["test_targets"])
        val_vids = runs[0]["val_video_ids"]
        test_vids = runs[0]["test_video_ids"]

        if method == "hard":
            val_preds = hard_voting(val_probs)
            test_preds = hard_voting(test_probs)
            val_fused_prob = np.mean(val_probs, axis=0)
            test_fused_prob = np.mean(test_probs, axis=0)

        elif method == "uniform_soft":
            val_fused_prob = np.mean(val_probs, axis=0)
            test_fused_prob = np.mean(test_probs, axis=0)
            val_preds = np.argmax(val_fused_prob, axis=1)
            test_preds = np.argmax(test_fused_prob, axis=1)

        elif method == "accuracy_weighted_soft":
            val_accs = np.array([float(r["val_vid_acc"]) for r in runs])
            weights = val_accs / np.sum(val_accs)
            val_fused_prob = sum(w * p for w, p in zip(weights, val_probs))
            test_fused_prob = sum(w * p for w, p in zip(weights, test_probs))
            val_preds = np.argmax(val_fused_prob, axis=1)
            test_preds = np.argmax(test_fused_prob, axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")

        val_m = compute_metrics(val_targets, val_preds)
        test_m = compute_metrics(test_targets, test_preds)

        _, _, _, val_vid_m = aggregate_video_level_predictions(val_fused_prob, val_targets, val_vids)
        _, _, _, test_vid_m = aggregate_video_level_predictions(test_fused_prob, test_targets, test_vids)

        seed_metrics.append({
            "seed": s,
            "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
            "val_win_f1": round(float(val_m["macro_f1"]), 4),
            "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
            "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
            "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
            "test_win_f1": round(float(test_m["macro_f1"]), 4),
            "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
            "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
        })

    val_w = [r["val_win_acc"] for r in seed_metrics]
    val_v = [r["val_vid_acc"] for r in seed_metrics]
    val_vf1 = [r["val_vid_f1"] for r in seed_metrics]
    test_w = [r["test_win_acc"] for r in seed_metrics]
    test_wf1 = [r["test_win_f1"] for r in seed_metrics]
    test_v = [r["test_vid_acc"] for r in seed_metrics]
    test_vf1 = [r["test_vid_f1"] for r in seed_metrics]

    return {
        "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
        "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
        "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
        "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
        "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
        "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
        "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
        "seeds": seed_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Table 6 Cross-Paradigm Fusion Protocols")
    parser.add_argument("--trans_proposed_dir", type=str, default="checkpoints/ablation_v2")
    parser.add_argument("--graph_streams_dir", type=str, default="checkpoints/graph_streams")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 7: Rebuilding Table 6 (Cross-Paradigm Fusion Protocols)")
    print("=" * 80)

    # Load constituents
    # Constituent 1: Transformer mix_v2 + Proposed Aug
    # Constituent 2: AAGCN Bone + Proposed Aug (T3.4)
    # Constituent 3: AAGCN World Joint + Proposed Aug (T3.5)
    # Constituent 4: AAGCN World Joint Motion + Proposed Aug (T3.6)
    # Constituent 5: AAGCN Bone Motion + Proposed Aug (T3.7)

    # In Table 6 we evaluate:
    # 1. SkelGym-Lite (2 Streams: Transformer + Bone AAGCN, 679K params)
    # 2. SkelGym-Full (5 Streams: Transformer + 4 AAGCN Streams, 1.81M params)
    # With 3 methods:
    # - Hard Majority Voting
    # - Uniform Average Soft Voting
    # - Accuracy-Weighted Soft Voting

    # We will build and save the summary to outputs/table6_cross_paradigm_fusion.json
    print("Orchestrator ready for constituent predictions.")

if __name__ == "__main__":
    main()
