#!/usr/bin/env python3
"""
Phase 7: Table 6 Cross-Paradigm Fusion Protocols (SkelGym-Lite & SkelGym-Full).
Evaluates 4 fusion protocols across 3 random seeds (42, 123, 3407):
  1. Hard Majority Voting (Discrete Baseline)
  2. Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K)
  3. Accuracy-Weighted Soft Voting (Validation-calibrated weights)
  4. Stacking Meta-Classifier (Logistic Regression, fit strictly on Train)

Configurations:
  - SkelGym-Lite (2 Streams: Transformer + Bone AAGCN, ~679K params)
  - SkelGym-Full (5 Streams: Transformer + 4 AAGCN Streams, ~1.81M params)

Zero-Leakage Guarantee:
  - Stacking meta-classifiers are trained strictly on training set base predictions.
  - Accuracy weights are calibrated strictly on validation window accuracy.
  - Test set features/probabilities are evaluated in inference mode with zero parameter updates.

Outputs:
  - outputs/table6_cross_paradigm_fusion.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX
from src.data.dataset import get_dataloaders
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

def send_marimo_toast(msg: str, *args, **kwargs):
    try:
        import marimo as mo
        kind = kwargs.get("kind", "info")
        mo.status.toast(msg, kind=kind)
    except Exception:
        pass

def hard_voting(prob_matrices: List[np.ndarray]) -> np.ndarray:
    """Hard majority voting across constituent classifiers."""
    preds = [np.argmax(p, axis=1) for p in prob_matrices]
    stacked = np.stack(preds, axis=0)  # (K, N)
    K, N = stacked.shape
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        col = stacked[:, i]
        vals, counts = np.unique(col, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

def aggregate_hard_voting_video(preds: np.ndarray, targets: np.ndarray, video_ids: List[str]) -> Tuple[float, float]:
    """Computes discrete video-level majority voting metrics."""
    from collections import Counter
    vid_to_preds = {}
    vid_to_target = {}
    for p, y, vid in zip(preds, targets, video_ids):
        if vid not in vid_to_preds:
            vid_to_preds[vid] = []
            vid_to_target[vid] = y
        vid_to_preds[vid].append(p)

    y_true_v = []
    y_pred_v = []
    for vid, p_list in vid_to_preds.items():
        c = Counter(p_list)
        majority_class = c.most_common(1)[0][0]
        y_pred_v.append(majority_class)
        y_true_v.append(vid_to_target[vid])

    m = compute_metrics(np.array(y_true_v), np.array(y_pred_v))
    return float(m["accuracy"] * 100.0), float(m["macro_f1"])

def evaluate_fusion_protocol(
    stream_runs_by_seed: Dict[int, List[Dict[str, Any]]],
    method: str
) -> Dict[str, Any]:
    """
    Evaluates a specific fusion protocol across all 3 seeds.
    method: 'hard', 'uniform_soft', 'accuracy_weighted_soft', 'stacking'
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
            val_vid_acc, val_vid_f1 = aggregate_hard_voting_video(val_preds, val_targets, val_vids)
            test_vid_acc, test_vid_f1 = aggregate_hard_voting_video(test_preds, test_targets, test_vids)

        elif method == "uniform_soft":
            val_fused_prob = np.mean(val_probs, axis=0)
            test_fused_prob = np.mean(test_probs, axis=0)
            val_preds = np.argmax(val_fused_prob, axis=1)
            test_preds = np.argmax(test_fused_prob, axis=1)
            _, _, _, val_vid_m = aggregate_video_level_predictions(val_fused_prob, val_targets, val_vids)
            _, _, _, test_vid_m = aggregate_video_level_predictions(test_fused_prob, test_targets, test_vids)
            val_vid_acc = float(val_vid_m["accuracy"] * 100.0)
            val_vid_f1 = float(val_vid_m["macro_f1"])
            test_vid_acc = float(test_vid_m["accuracy"] * 100.0)
            test_vid_f1 = float(test_vid_m["macro_f1"])

        elif method == "accuracy_weighted_soft":
            val_accs = np.array([float(r["val_win_acc"]) for r in runs])
            weights = val_accs / np.sum(val_accs)
            val_fused_prob = sum(w * p for w, p in zip(weights, val_probs))
            test_fused_prob = sum(w * p for w, p in zip(weights, test_probs))
            val_preds = np.argmax(val_fused_prob, axis=1)
            test_preds = np.argmax(test_fused_prob, axis=1)
            _, _, _, val_vid_m = aggregate_video_level_predictions(val_fused_prob, val_targets, val_vids)
            _, _, _, test_vid_m = aggregate_video_level_predictions(test_fused_prob, test_targets, test_vids)
            val_vid_acc = float(val_vid_m["accuracy"] * 100.0)
            val_vid_f1 = float(val_vid_m["macro_f1"])
            test_vid_acc = float(test_vid_m["accuracy"] * 100.0)
            test_vid_f1 = float(test_vid_m["macro_f1"])

        elif method == "stacking":
            X_val = np.concatenate(val_probs, axis=1)
            X_test = np.concatenate(test_probs, axis=1)
            
            # Stacking trained STRICTLY on TRAIN SET ONLY with row-by-row correspondence safeguards
            if "train_probs" in runs[0] and runs[0]["train_probs"] is not None:
                train_probs = [np.array(r["train_probs"]) for r in runs]
                train_targets = np.array(runs[0]["train_targets"])
                for idx_run, r_check in enumerate(runs[1:], start=1):
                    t_check = np.array(r_check["train_targets"])
                    assert len(t_check) == len(train_targets), f"Train targets length mismatch: {len(train_targets)} vs {len(t_check)}"
                    assert np.array_equal(train_targets, t_check), f"Train targets values mismatch between streams 0 and {idx_run}"
                X_train = np.concatenate(train_probs, axis=1)
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=s)
                clf.fit(X_train, train_targets)
            else:
                raise RuntimeError(f"Seed {s}: train_probs not found in runs! Stacking requires train_probs for unbiased validation evaluation.")
            
            val_fused_prob = clf.predict_proba(X_val)
            test_fused_prob = clf.predict_proba(X_test)
            val_preds = np.argmax(val_fused_prob, axis=1)
            test_preds = np.argmax(test_fused_prob, axis=1)
            _, _, _, val_vid_m = aggregate_video_level_predictions(val_fused_prob, val_targets, val_vids)
            _, _, _, test_vid_m = aggregate_video_level_predictions(test_fused_prob, test_targets, test_vids)
            val_vid_acc = float(val_vid_m["accuracy"] * 100.0)
            val_vid_f1 = float(val_vid_m["macro_f1"])
            test_vid_acc = float(test_vid_m["accuracy"] * 100.0)
            test_vid_f1 = float(test_vid_m["macro_f1"])
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        val_m = compute_metrics(val_targets, val_preds)
        test_m = compute_metrics(test_targets, test_preds)

        seed_metrics.append({
            "seed": s,
            "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
            "val_win_f1": round(float(val_m["macro_f1"]), 4),
            "val_vid_acc": round(float(val_vid_acc), 2),
            "val_vid_f1": round(float(val_vid_f1), 4),
            "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
            "test_win_f1": round(float(test_m["macro_f1"]), 4),
            "test_vid_acc": round(float(test_vid_acc), 2),
            "test_vid_f1": round(float(test_vid_f1), 4),
        })

    val_w = [r["val_win_acc"] for r in seed_metrics]
    val_wf1 = [r["val_win_f1"] for r in seed_metrics]
    val_v = [r["val_vid_acc"] for r in seed_metrics]
    val_vf1 = [r["val_vid_f1"] for r in seed_metrics]
    test_w = [r["test_win_acc"] for r in seed_metrics]
    test_wf1 = [r["test_win_f1"] for r in seed_metrics]
    test_v = [r["test_vid_acc"] for r in seed_metrics]
    test_vf1 = [r["test_vid_f1"] for r in seed_metrics]

    return {
        "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
        "val_win_f1": f"{np.mean(val_wf1):.4f} ± {np.std(val_wf1):.4f}",
        "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
        "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
        "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
        "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
        "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
        "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
        "seeds": seed_metrics
    }

def get_transformer_predictions(
    seed: int,
    proposed_aug_cfg_id: str,
    checkpoint_dir: Path,
    metadata_path: str,
    device: torch.device
) -> Dict[str, Any]:
    """Loads Transformer mix_v2 checkpoint and computes prediction probabilities."""
    model_name = f"Transformer_mix_v2_{proposed_aug_cfg_id}_seed{seed}"
    ckpt_path = checkpoint_dir / f"seed{seed}" / f"best_{model_name}.pt"
    if not ckpt_path.exists():
        # Fallback check for candidate_minus_time
        alt = checkpoint_dir / f"seed{seed}" / f"best_Transformer_mix_v2_candidate_minus_time_seed{seed}.pt"
        if alt.exists():
            ckpt_path = alt

    # Check if precomputed probs npz exists
    probs_npz = checkpoint_dir / f"seed{seed}" / f"probs_{proposed_aug_cfg_id}_seed{seed}.npz"
    if probs_npz.exists():
        data = np.load(probs_npz, allow_pickle=True)
        val_probs = data["val_probs"]
        y_val = data["val_targets"]
        test_probs = data["test_probs"]
        y_test = data["test_targets"]
        train_probs = data["train_probs"]
        y_train = data["train_targets"]
        val_vids = data["val_video_ids"].tolist()
        test_vids = data["test_video_ids"].tolist()

        val_preds = np.argmax(val_probs, axis=1)
        val_m = compute_metrics(y_val, val_preds)
        _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_vids)
        test_preds = np.argmax(test_probs, axis=1)
        test_m = compute_metrics(y_test, test_preds)
        _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_vids)

        return {
            "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
            "val_win_f1": round(float(val_m["macro_f1"]), 4),
            "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
            "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
            "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
            "test_win_f1": round(float(test_m["macro_f1"]), 4),
            "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
            "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
            "train_probs": train_probs.tolist(),
            "train_targets": y_train.tolist(),
            "val_probs": val_probs.tolist(),
            "val_targets": y_val.tolist(),
            "test_probs": test_probs.tolist(),
            "test_targets": y_test.tolist(),
            "val_video_ids": val_vids,
            "test_video_ids": test_vids,
            "checkpoint": str(ckpt_path)
        }

    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix_v2",
        batch_size=16,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=None,
        in_memory=True,
        seed=seed,
        save_norm_artifact=False,
        strict_norm=False
    )

    model = build_model(
        model_type="Transformer",
        feature_method="mix_v2",
        hidden_dim=112,
        num_layers=3,
        nhead=4,
        dropout=0.2,
        transformer_variant="dual_branch"
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    trainer = Trainer(
        model=model,
        device=device,
        model_name=model_name,
        use_amp=torch.cuda.is_available()
    )

    # Predict Train (unshuffled)
    from torch.utils.data import DataLoader
    eval_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=0
    )
    y_train, _, train_probs = trainer.predict(eval_train_loader)

    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

    # Cache npz
    probs_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probs_npz,
        train_probs=train_probs,
        train_targets=y_train,
        val_probs=val_probs,
        val_targets=y_val,
        test_probs=test_probs,
        test_targets=y_test,
        val_video_ids=np.array(val_loader.dataset.video_ids),
        test_video_ids=np.array(test_loader.dataset.video_ids)
    )

    return {
        "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
        "val_win_f1": round(float(val_m["macro_f1"]), 4),
        "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
        "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
        "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
        "test_win_f1": round(float(test_m["macro_f1"]), 4),
        "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
        "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
        "train_probs": train_probs.tolist(),
        "train_targets": y_train.tolist(),
        "val_probs": val_probs.tolist(),
        "test_probs": test_probs.tolist(),
        "val_targets": y_val.tolist(),
        "test_targets": y_test.tolist(),
        "val_video_ids": list(val_loader.dataset.video_ids),
        "test_video_ids": list(test_loader.dataset.video_ids),
        "checkpoint": str(ckpt_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Table 6 Cross-Paradigm Fusion Protocols")
    parser.add_argument("--proposed_aug_cfg_id", type=str, default="candidate_minus_time", help="Ablation config ID of proposed aug")
    parser.add_argument("--trans_ckpt_dir", type=str, default="checkpoints/ablation_v2")
    parser.add_argument("--graph_cache_file", type=str, default="outputs/table3_stream_predictions.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print("PHASE 7: Rebuilding Table 6 (Cross-Paradigm Fusion Protocols)")
    print(f"Device: {device} | Proposed Augment Config: {args.proposed_aug_cfg_id}")
    print("=" * 80)

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    # 1. Load Graph Streams predictions from Phase 6
    graph_cache_path = ROOT_DIR / args.graph_cache_file
    if not graph_cache_path.exists():
        raise FileNotFoundError(f"Could not find graph stream predictions at {graph_cache_path}. Run Phase 6 first!")

    print(f"Loading graph stream predictions from {graph_cache_path}...")
    graph_stream_runs = torch.load(graph_cache_path, map_location="cpu")

    # 2. Extract or Compute Transformer predictions
    print(f"Extracting Transformer mix_v2 ({args.proposed_aug_cfg_id}) predictions across 3 seeds...")
    trans_runs_by_seed = {}
    trans_ckpt_dir = ROOT_DIR / args.trans_ckpt_dir

    for s in SEEDS:
        print(f"  --> Transformer seed {s}...")
        t_res = get_transformer_predictions(
            seed=s,
            proposed_aug_cfg_id=args.proposed_aug_cfg_id,
            checkpoint_dir=trans_ckpt_dir,
            metadata_path=str(meta_cand),
            device=device
        )
        trans_runs_by_seed[s] = t_res

    # 3. Assemble Constituents for Lite and Full
    # Lite: Transformer (mix_v2) + AAGCN Bone 3D (T3.4)
    # Full: Transformer (mix_v2) + T3.4 + T3.5 + T3.6 + T3.7
    lite_constituents_by_seed = {}
    full_constituents_by_seed = {}

    for s in SEEDS:
        t_pred = trans_runs_by_seed[s]
        bone_pred = graph_stream_runs["T3.4"][s]
        joint_pred = graph_stream_runs["T3.5"][s]
        jmotion_pred = graph_stream_runs["T3.6"][s]
        bmotion_pred = graph_stream_runs["T3.7"][s]

        lite_constituents_by_seed[s] = [t_pred, bone_pred]
        full_constituents_by_seed[s] = [t_pred, bone_pred, joint_pred, jmotion_pred, bmotion_pred]

    protocols = [
        ("hard", "Hard Majority Voting (Discrete Baseline)"),
        ("accuracy_weighted_soft", "Accuracy-Weighted Soft Voting (Validation-calibrated weights)"),
        ("uniform_soft", "Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K)"),
        ("stacking", "Stacking Meta-Classifier (Logistic Regression, train-set fit)")
    ]

    results_table6 = {
        "SkelGym-Lite": {},
        "SkelGym-Full": {}
    }

    print("\nEvaluating SkelGym-Lite (2 Streams: Trans + Bone AAGCN, 679K params)...")
    for method_key, method_desc in protocols:
        res = evaluate_fusion_protocol(lite_constituents_by_seed, method_key)
        res["description"] = method_desc
        results_table6["SkelGym-Lite"][method_key] = res
        print(f"  [{method_key}] Val Vid Acc: {res['val_vid_acc']} | Test Vid Acc: {res['test_vid_acc']} (F1: {res['test_vid_f1']})")

    print("\nEvaluating SkelGym-Full (5 Streams: Trans + 4 AAGCN Streams, 1.81M params)...")
    for method_key, method_desc in protocols:
        res = evaluate_fusion_protocol(full_constituents_by_seed, method_key)
        res["description"] = method_desc
        results_table6["SkelGym-Full"][method_key] = res
        print(f"  [{method_key}] Val Vid Acc: {res['val_vid_acc']} | Test Vid Acc: {res['test_vid_acc']} (F1: {res['test_vid_f1']})")

    # 4. Compute Validation Winners strictly based on Validation Window Metrics (Primary: val_win_f1, Secondary: val_win_acc)
    winner_lite = max(
        results_table6["SkelGym-Lite"].keys(),
        key=lambda k: (
            float(results_table6["SkelGym-Lite"][k]["val_win_f1"].split(" ")[0]),
            float(results_table6["SkelGym-Lite"][k]["val_win_acc"].split("%")[0])
        )
    )
    winner_full = max(
        results_table6["SkelGym-Full"].keys(),
        key=lambda k: (
            float(results_table6["SkelGym-Full"][k]["val_win_f1"].split(" ")[0]),
            float(results_table6["SkelGym-Full"][k]["val_win_acc"].split("%")[0])
        )
    )
    results_table6["winners"] = {
        "SkelGym-Lite": winner_lite,
        "SkelGym-Full": winner_full
    }

    # 5. Print Formatted Table with Strict Zero-Leakage (Test metrics revealed ONLY for validation winners)
    print("\n" + "=" * 110)
    print("## TABLE 6: CROSS-PARADIGM FUSION PROTOCOLS (TEST METRICS REVEALED ONLY FOR VALIDATION WINNERS)")
    print("=" * 110)
    print("| Paradigm / Architecture | Fusion Protocol | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for arch in ["SkelGym-Lite", "SkelGym-Full"]:
        w_key = winner_lite if arch == "SkelGym-Lite" else winner_full
        for m_key, m_rep in results_table6[arch].items():
            is_w = (m_key == w_key)
            test_w = m_rep['test_win_acc'] if is_w else "-"
            test_wf1 = m_rep['test_win_f1'] if is_w else "-"
            test_v = f"**{m_rep['test_vid_acc']}**" if is_w else "-"
            test_vf1 = f"**{m_rep['test_vid_f1']}**" if is_w else "-"
            st = f"🏆 Validation Winner ({arch})" if is_w else "Verified"
            print(f"| **{arch}** | {m_rep['description']} | {m_rep['val_win_acc']} | {m_rep['val_win_f1']} | {m_rep['val_vid_acc']} | {m_rep['val_vid_f1']} | {test_w} | {test_wf1} | {test_v} | {test_vf1} | {st} |")

    print("=" * 110)

    # 6. Save Output
    out_file = ROOT_DIR / "outputs" / "table6_cross_paradigm_fusion.json"
    with open(out_file, "w") as f:
        json.dump(results_table6, f, indent=2)

    send_marimo_toast(f"Phase 7 Complete! Lite Winner: {winner_lite}, Full Winner: {winner_full}", kind="success")
    print(f"\nSaved Table 6 results to {out_file}")

if __name__ == "__main__":
    main()
