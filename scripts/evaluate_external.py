#!/usr/bin/env python3
"""
Unified Cross-Dataset External Evaluation Pipeline.
Evaluates frozen SkelGym architectures (ST-GCN, Transformer, AAGCNs, SkelGym-Lite, SkelGym-Full)
on independent external benchmarks (MM-Fit and Fit3D) under Protocol A (MediaPipe) and Protocol B (Native 3D).
Strictly enforces:
  - Zero retraining / zero domain adaptation
  - Normalization using frozen SkelGym train set statistics only
  - Ensemble soft voting using frozen SkelGym validation weights
  - Hierarchical aggregation: Window -> Segment -> Recording / Workout
  - Subject/Workout-level non-parametric bootstrap 95% confidence intervals (B=1000)
  - Domain-gap diagnostics and error auditing
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ACTIONS, NUM_CLASSES, ACTION_TO_IDX, IDX_TO_ACTION
from src.cli import build_model
from src.training.trainer import Trainer
from src.data.dataset import get_dataloaders
from src.external.mmfit import MMFitExternalDataset
from src.external.fit3d import Fit3DExternalDataset
from src.external.class_mapping import get_target_skelgym_indices, get_target_skelgym_classes
from src.external.metrics import (
    evaluate_window_level,
    aggregate_hierarchical_predictions,
    compute_recording_level_bootstrap_ci,
    compute_paired_bootstrap_difference,
    compute_domain_gap_diagnostics,
    plot_external_confusion_matrix,
    generate_error_analysis_table
)
from src.external.fewshot import extract_penultimate_embeddings

def load_checkpoint(model_type: str, feat_type: str, ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    m = build_model(model_type, feat_type, num_classes=NUM_CLASSES)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    return m

def predict_windows(model: torch.nn.Module, features: np.ndarray, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Inference on numpy array (N, T, D) -> probability matrix (N, 22).
    """
    model.eval()
    probs = []
    N = len(features)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(features[i:i + batch_size]).float().to(device)
            logits = model(batch)
            p = torch.softmax(logits, dim=-1)
            probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, NUM_CLASSES), dtype=np.float32)

def find_checkpoint_path(ckpt_rel: str, seed: int) -> Path:
    p = Path(ckpt_rel)
    stem = p.stem
    suffix = p.suffix
    if seed == 42:
        candidates = [
            PROJECT_ROOT / ckpt_rel,
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}{suffix}",
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}_seed{seed}{suffix}"
        ]
    else:
        candidates = [
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}{suffix}",
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}_seed{seed}{suffix}"
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Strict checkpoint lookup failed for seed {seed}: {ckpt_rel}. "
        f"Checked candidates: {[str(c) for c in candidates]}. No silent fallback permitted."
    )

def load_norm_stats_for_seed(ref_dir: Path, feat: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if seed == 42:
        candidates = [
            ref_dir / f"seed{seed}" / f"normalization_{feat}.npz",
            ref_dir / f"normalization_{feat}.npz"
        ]
    else:
        candidates = [
            ref_dir / f"seed{seed}" / f"normalization_{feat}.npz"
        ]
    for c in candidates:
        if c.exists():
            data = np.load(c)
            return (data["mean"], data["std"])
    raise FileNotFoundError(
        f"Strict normalization artifact lookup failed for feature '{feat}' and seed {seed}. "
        f"Expected: {candidates[0]}."
    )

def load_ensemble_weights_for_seed(ref_dir: Path, seed: int) -> Dict[str, Any]:
    if seed == 42:
        candidates = [
            ref_dir / f"seed{seed}" / "ensemble_weights.json",
            ref_dir / "ensemble_weights.json"
        ]
    else:
        candidates = [
            ref_dir / f"seed{seed}" / "ensemble_weights.json"
        ]
    for c in candidates:
        if c.exists():
            with open(c, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Strict ensemble weights lookup failed for seed {seed}. "
        f"Expected: {candidates[0]}. Run freeze_reference_artifacts.py for seed {seed} first!"
    )

def main():
    parser = argparse.ArgumentParser(description="Unified External Benchmark Evaluation")
    parser.add_argument("--config", type=str, default="configs/external/mmfit.yaml", help="Path to config yaml")
    parser.add_argument("--dataset", type=str, default=None, choices=["mmfit", "fit3d"])
    parser.add_argument("--pose-source", type=str, default=None, choices=["native", "mediapipe"])
    parser.add_argument("--split-group", type=str, default=None, help="Workout split group (e.g. unseen_test)")
    parser.add_argument("--class-set", type=str, default=None, help="Class subset (e.g. core4, core6)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output destination directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="Random seeds to evaluate across (e.g. --seeds 42 123 3407)")
    args = parser.parse_args()

    cfg_p = PROJECT_ROOT / args.config
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f)

    # Command-line overrides
    dataset_name = args.dataset or cfg.get("dataset", "mmfit")
    pose_source = args.pose_source or cfg.get("pose_protocol", {}).get("source", "native")
    class_set = args.class_set or cfg.get("class_set", "core4")
    split_group = args.split_group or cfg.get("workout_split", {}).get("split_group", "unseen_test")
    out_dir = Path(args.out_dir or cfg.get("paths", {}).get("output_dir", f"outputs/external/{dataset_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = PROJECT_ROOT / cfg.get("paths", {}).get("reference_dir", "artifacts/reference")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("=" * 80)
    print(f"GENUINE CROSS-DATASET EXTERNAL EVALUATION BENCHMARK")
    print(f"Dataset: {dataset_name.upper()} | Pose Protocol: {pose_source.upper()} | Class Set: {class_set}")
    print(f"Split Group: {split_group} | Target Device: {device}")
    print("=" * 80)

    # 1. Load External Dataset
    print(f"\n[1/7] Loading and verifying external dataset '{dataset_name}'...")
    if dataset_name == "mmfit":
        ext_ds = MMFitExternalDataset(
            root_dir=cfg.get("paths", {}).get("data_dir", "mm-fit"),
            split_group=split_group,
            class_set=class_set,
            pose_source=pose_source,
            apply_geometric_norm=cfg.get("normalization", {}).get("geometric", False if pose_source == "mediapipe" else True)
        )
    elif dataset_name == "fit3d":
        ext_ds = Fit3DExternalDataset(
            root_dir=cfg.get("paths", {}).get("data_dir", "data_external/fit3d"),
            class_set=class_set,
            pose_source=pose_source,
            source_fps=float(cfg.get("fps", {}).get("source", 50.0)),
            target_fps=float(cfg.get("fps", {}).get("target", 30.0)),
            apply_geometric_norm=cfg.get("normalization", {}).get("geometric", True)
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    audit_csv = out_dir / "dataset_audit.csv"
    ext_ds.export_audit_report(audit_csv)
    print(f"  Accepted action records: {len(ext_ds.records)}")
    print(f"  Audit log saved to: {audit_csv}")

    target_class_indices = get_target_skelgym_indices(dataset_name, class_set)
    target_class_names = get_target_skelgym_classes(dataset_name, class_set)
    print(f"  Target Classes ({len(target_class_indices)}): {target_class_names}")

    # 2. Extract Raw Synchronized Windows for each Feature Representation
    print(f"\n[2/7] Extracting raw synchronized windows for feature representations...")
    seq_len = int(cfg.get("window", {}).get("seq_len", 32))
    stride = int(cfg.get("window", {}).get("stride", 32))

    raw_features = {}
    base_info = None

    for feat in ["mix", "rel_3d", "bone_3d", "joint_motion_3d", "bone_motion_3d"]:
        win_data = ext_ds.extract_windows(feature_method=feat, seq_len=seq_len, stride=stride)
        raw_features[feat] = win_data["features"]  # (N, T, D)

        if base_info is None:
            base_info = {
                "labels": win_data["labels"],
                "record_ids": win_data["record_ids"],
                "subject_ids": win_data["subject_ids"]
            }

    y_win_true = base_info["labels"]
    win_rec_ids = base_info["record_ids"]
    win_subj_ids = base_info["subject_ids"]
    N_windows = len(y_win_true)
    print(f"  Total extracted test windows: {N_windows} across {len(set(win_rec_ids))} unique action records")

    # 3. Multi-Seed Model Evaluation Loop
    seeds = args.seeds if args.seeds else [42]
    eval_models = [
        "ST-GCN (Rel 3D)",
        "LSTM (Mix 117-d)",
        "BiLSTM (Mix 117-d)",
        "Transformer (Mix)",
        "AAGCN (Bone 3D)",
        "SkelGym-Lite",
        "SkelGym-Full"
    ]
    print(f"\n[3/7] Executing evaluation across {len(seeds)} random seed(s): {seeds}...")

    all_seed_results_open = []
    all_seed_results_closed = []
    accum_probs = {m: np.zeros((N_windows, NUM_CLASSES), dtype=np.float64) for m in eval_models}
    eval_counts = {m: 0 for m in eval_models}
    last_models_dict = {}
    last_feature_windows = {}

    model_defs = {
        "ST-GCN (Rel 3D)": ("STGCN", "rel_3d", "checkpoints/best_STGCN_T3.2_rel_3d.pt"),
        "LSTM (Mix 117-d)": ("LSTM", "mix", "checkpoints/best_LSTM_T1.9_mix.pt"),
        "BiLSTM (Mix 117-d)": ("BiLSTM", "mix", "checkpoints/best_BiLSTM_T1.18_mix.pt"),
        "Transformer (Mix)": ("Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt"),
        "AAGCN (Bone 3D)": ("AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt"),
        "AAGCN (Rel 3D)": ("AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt"),
        "AAGCN (Joint Mot)": ("AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt"),
        "AAGCN (Bone Mot)": ("AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt"),
    }

    for s_idx, seed in enumerate(seeds, 1):
        print(f"\n--- [Seed {seed}] ({s_idx}/{len(seeds)}) Loading artifacts & predicting ---")
        norm_stats = {}
        for feat in ["mix", "rel_3d", "bone_3d", "joint_motion_3d", "bone_motion_3d"]:
            norm_stats[feat] = load_norm_stats_for_seed(ref_dir, feat, seed)

        feature_windows = {}
        for feat, raw_feat in raw_features.items():
            mean, std = norm_stats[feat]
            norm_feat = (raw_feat - mean[None, :, :]) / (std[None, :, :] + 1e-7)
            feature_windows[feat] = norm_feat.astype(np.float32)
        last_feature_windows = feature_windows

        test_probs = {}
        for name, (m_type, f_type, ckpt_p) in model_defs.items():
            if name in ["ST-GCN (Rel 3D)", "LSTM (Mix 117-d)", "BiLSTM (Mix 117-d)"] and seed != 42:
                # Baseline models were trained exclusively on canonical seed 42
                continue
            ckpt_full = find_checkpoint_path(ckpt_p, seed)
            model = load_checkpoint(m_type, f_type, str(ckpt_full), device)
            last_models_dict[name] = (model, m_type, f_type)
            feats = feature_windows[f_type]
            p = predict_windows(model, feats, device)
            test_probs[name] = p

        ens_cfg = load_ensemble_weights_for_seed(ref_dir, seed)
        w_lite = ens_cfg.get("lite", {}).get("weights_window", [0.5, 0.5])
        test_probs["SkelGym-Lite"] = (
            float(w_lite[0]) * test_probs["Transformer (Mix)"] +
            float(w_lite[1]) * test_probs["AAGCN (Bone 3D)"]
        )

        w_full = ens_cfg.get("full", {}).get("weights_window", [0.2, 0.2, 0.2, 0.2, 0.2])
        test_probs["SkelGym-Full"] = (
            float(w_full[0]) * test_probs["Transformer (Mix)"] +
            float(w_full[1]) * test_probs["AAGCN (Bone 3D)"] +
            float(w_full[2]) * test_probs["AAGCN (Rel 3D)"] +
            float(w_full[3]) * test_probs["AAGCN (Joint Mot)"] +
            float(w_full[4]) * test_probs["AAGCN (Bone Mot)"]
        )

        for m_name in eval_models:
            if m_name not in test_probs:
                continue
            prob = test_probs[m_name]
            accum_probs[m_name] += prob
            eval_counts[m_name] += 1

            res_open_win = evaluate_window_level(prob, y_win_true, target_class_indices, mode="open_set")
            res_open_rec = aggregate_hierarchical_predictions(prob, y_win_true, win_rec_ids, target_class_indices, mode="open_set")
            res_closed_win = evaluate_window_level(prob, y_win_true, target_class_indices, mode="closed_set")
            res_closed_rec = aggregate_hierarchical_predictions(prob, y_win_true, win_rec_ids, target_class_indices, mode="closed_set")

            all_seed_results_open.append({
                "seed": seed,
                "model": m_name,
                "window_acc": res_open_win["accuracy"],
                "recording_acc": res_open_rec["accuracy"],
                "macro_f1": res_open_rec["macro_f1"]
            })
            all_seed_results_closed.append({
                "seed": seed,
                "model": m_name,
                "window_acc": res_closed_win["accuracy"],
                "recording_acc": res_closed_rec["accuracy"],
                "macro_f1": res_closed_rec["macro_f1"],
                **{f"recall_{k}": v for k, v in res_closed_rec["per_class_recall"].items()}
            })

    # 4. Statistical Aggregation and Bootstrap Confidence Intervals
    print("\n" + "=" * 105)
    print(f"TABLE X: CROSS-DATASET EXTERNAL VALIDATION RESULTS ({dataset_name.upper()} - {split_group.upper()})")
    print(f"Evaluated across {len(seeds)} random seed(s): {seeds}")
    print("=" * 105)
    print(f"{'Model':<22} | {'Open Win (%)':<16} | {'Open Rec (%)':<16} | {'Closed Win (%)':<16} | {'Closed Rec (%)':<16} | {'Macro F1':<16}")
    print("-" * 115)

    summary_open = []
    summary_closed = []
    bootstrap_results = {}

    for m_name in eval_models:
        m_open = [r for r in all_seed_results_open if r["model"] == m_name]
        m_closed = [r for r in all_seed_results_closed if r["model"] == m_name]

        open_win_mean = float(np.mean([r["window_acc"] for r in m_open]))
        open_win_std = float(np.std([r["window_acc"] for r in m_open]))
        open_rec_mean = float(np.mean([r["recording_acc"] for r in m_open]))
        open_rec_std = float(np.std([r["recording_acc"] for r in m_open]))
        open_f1_mean = float(np.mean([r["macro_f1"] for r in m_open]))
        open_f1_std = float(np.std([r["macro_f1"] for r in m_open]))

        closed_win_mean = float(np.mean([r["window_acc"] for r in m_closed]))
        closed_win_std = float(np.std([r["window_acc"] for r in m_closed]))
        closed_rec_mean = float(np.mean([r["recording_acc"] for r in m_closed]))
        closed_rec_std = float(np.std([r["recording_acc"] for r in m_closed]))
        closed_f1_mean = float(np.mean([r["macro_f1"] for r in m_closed]))
        closed_f1_std = float(np.std([r["macro_f1"] for r in m_closed]))

        if len(seeds) > 1:
            print(
                f"{m_name:<22} | {open_win_mean:>5.2f} ± {open_win_std:<6.2f} | "
                f"{open_rec_mean:>5.2f} ± {open_rec_std:<6.2f} | "
                f"{closed_win_mean:>5.2f} ± {closed_win_std:<6.2f} | "
                f"{closed_rec_mean:>5.2f} ± {closed_rec_std:<6.2f} | "
                f"{closed_f1_mean:>5.4f} ± {closed_f1_std:<5.4f}"
            )
        else:
            print(
                f"{m_name:<22} | {open_win_mean:>6.2f}%          | "
                f"{open_rec_mean:>6.2f}%          | "
                f"{closed_win_mean:>6.2f}%          | "
                f"{closed_rec_mean:>6.2f}%          | "
                f"{closed_f1_mean:>6.4f}"
            )

        n_evals = max(1, eval_counts[m_name])
        avg_prob = accum_probs[m_name] / float(n_evals)
        res_open_rec_avg = aggregate_hierarchical_predictions(avg_prob, y_win_true, win_rec_ids, target_class_indices, mode="open_set")
        res_closed_rec_avg = aggregate_hierarchical_predictions(avg_prob, y_win_true, win_rec_ids, target_class_indices, mode="closed_set")

        # Map each action record to its workout cluster for workout-cluster bootstrap
        rec_to_subj = {r: s for r, s in zip(win_rec_ids, win_subj_ids)}
        cluster_ids = [rec_to_subj[g] for g in res_open_rec_avg["group_ids"]]

        ci_open = compute_recording_level_bootstrap_ci(
            res_open_rec_avg["group_probs"], res_open_rec_avg["group_trues"], target_class_indices, mode="open_set", cluster_ids=cluster_ids
        )
        ci_closed = compute_recording_level_bootstrap_ci(
            res_closed_rec_avg["group_probs"], res_closed_rec_avg["group_trues"], target_class_indices, mode="closed_set", cluster_ids=cluster_ids
        )
        bootstrap_results[m_name] = {
            "open_set_recording": ci_open,
            "closed_set_recording": ci_closed
        }

        summary_open.append({
            "model": m_name,
            "window_acc": open_win_mean,
            "recording_acc": open_rec_mean,
            "macro_f1": open_f1_mean,
            "window_acc_mean": open_win_mean,
            "window_acc_std": open_win_std,
            "recording_acc_mean": open_rec_mean,
            "recording_acc_std": open_rec_std,
            "macro_f1_mean": open_f1_mean,
            "macro_f1_std": open_f1_std,
            "ci_95_recording_acc": ci_open["acc_ci"],
            "ci_95_macro_f1": ci_open["f1_ci"]
        })

        recalls_mean = {}
        for k in target_class_indices:
            k_recalls = [r.get(f"recall_{k}", 0.0) for r in m_closed]
            recalls_mean[f"recall_{k}"] = float(np.mean(k_recalls)) if k_recalls else 0.0
            recalls_mean[f"recall_{k}_mean"] = float(np.mean(k_recalls)) if k_recalls else 0.0
            recalls_mean[f"recall_{k}_std"] = float(np.std(k_recalls)) if k_recalls else 0.0

        summary_closed.append({
            "model": m_name,
            "window_acc": closed_win_mean,
            "recording_acc": closed_rec_mean,
            "macro_f1": closed_f1_mean,
            "window_acc_mean": closed_win_mean,
            "window_acc_std": closed_win_std,
            "recording_acc_mean": closed_rec_mean,
            "recording_acc_std": closed_rec_std,
            "macro_f1_mean": closed_f1_mean,
            "macro_f1_std": closed_f1_std,
            "ci_95_recording_acc": ci_closed["acc_ci"],
            "ci_95_macro_f1": ci_closed["f1_ci"],
            **recalls_mean
        })

    # Save summary tables and run traces
    df_metrics_open = pd.DataFrame(summary_open)
    df_metrics_closed = pd.DataFrame(summary_closed)
    df_metrics_open.to_csv(out_dir / "metrics_open.csv", index=False)
    df_metrics_closed.to_csv(out_dir / "metrics_closed.csv", index=False)

    pd.DataFrame(all_seed_results_open).to_csv(out_dir / "metrics_open_runs.csv", index=False)
    pd.DataFrame(all_seed_results_closed).to_csv(out_dir / "metrics_closed_runs.csv", index=False)

    with open(out_dir / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    # 5. Export Predictions & Confusion Matrices
    print(f"\n[5/7] Exporting prediction logs and confusion matrices...")
    best_model_name = "SkelGym-Full"
    best_prob = accum_probs[best_model_name]

    # Save predictions_window.csv
    open_win_preds = np.argmax(best_prob, axis=1)
    df_pred_win = pd.DataFrame({
        "record_id": win_rec_ids,
        "subject_id": win_subj_ids,
        "true_class": [ACTIONS[y] for y in y_win_true],
        "pred_class_open": [ACTIONS[p] for p in open_win_preds],
        "is_correct_open": (open_win_preds == y_win_true)
    })
    df_pred_win.to_csv(out_dir / "predictions_window.csv", index=False)

    # Plot Confusion Matrix for SkelGym-Full
    res_open_rec = aggregate_hierarchical_predictions(best_prob, y_win_true, win_rec_ids, target_class_indices, mode="open_set")
    res_closed_rec = aggregate_hierarchical_predictions(best_prob, y_win_true, win_rec_ids, target_class_indices, mode="closed_set")

    plot_external_confusion_matrix(
        y_true=res_open_rec["group_trues"],
        y_pred=res_open_rec["preds"],
        target_class_indices=target_class_indices,
        dest_path=out_dir / "confusion_open.png",
        title=f"{dataset_name.upper()} Open-Set (SkelGym-Full)"
    )
    plot_external_confusion_matrix(
        y_true=res_closed_rec["group_trues"],
        y_pred=res_closed_rec["preds"],
        target_class_indices=target_class_indices,
        dest_path=out_dir / "confusion_closed.png",
        title=f"{dataset_name.upper()} Closed-Set (SkelGym-Full)"
    )

    # 6. Error Analysis Table
    print(f"\n[6/7] Generating error analysis table...")
    generate_error_analysis_table(
        y_probs=best_prob,
        y_trues=y_win_true,
        record_ids=win_rec_ids,
        subject_ids=win_subj_ids,
        dataset_name=dataset_name,
        pose_source=pose_source,
        dest_path=out_dir / "misclassified_records.csv"
    )

    # 7. Domain-Gap Diagnostics (Class Centroid Cosine Similarity)
    print(f"\n[7/7] Computing Domain-Gap Semantic Alignment Diagnostics...")
    # Extract Transformer embeddings on external dataset
    trans_m = last_models_dict["Transformer (Mix)"][0]
    ext_embs = extract_penultimate_embeddings(trans_m, last_feature_windows["mix"], "Transformer", device)

    # Load SkelGym test set dataloader for centroid comparison
    metadata_path = str(PROJECT_ROOT / "data" / "Final_dataset_metadata.csv")
    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix",
        batch_size=32,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        landmark_dir=str(PROJECT_ROOT / "data" / "landmarks"),
        num_workers=0,
        in_memory=True,
        seed=seeds[0]
    )
    skel_embs = []
    with torch.no_grad():
        for batch, _ in test_loader:
            b = batch.to(device)
            x_norm = trans_m.in_norm(b)
            h = trans_m.input_proj(x_norm)
            h = trans_m.pos_encoder(h)
            enc = trans_m.norm(trans_m.transformer_encoder(h))
            emb = enc.mean(dim=1)
            skel_embs.append(emb.cpu().numpy())
    skel_embs = np.concatenate(skel_embs, axis=0)
    skel_labels = np.array(test_loader.dataset.labels)

    diag = compute_domain_gap_diagnostics(
        skelgym_embs=skel_embs,
        skelgym_labels=skel_labels,
        external_embs=ext_embs,
        external_labels=y_win_true,
        target_class_indices=target_class_indices
    )
    with open(out_dir / "domain_gap_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    print(f"  Mean Cross-Domain Centroid Cosine Similarity: {diag['mean_cross_domain_similarity']:.4f}")
    for c_name in target_class_names:
        if c_name in diag:
            print(f"    {c_name:<25}: sim = {diag[c_name]['cosine_similarity']:.4f}")

    # Run Manifest
    manifest = {
        "dataset": dataset_name,
        "pose_source": pose_source,
        "class_set": class_set,
        "split_group": split_group,
        "num_records": len(ext_ds.records),
        "num_windows": N_windows,
        "evaluated_models": eval_models,
        "timestamp": str(pd.Timestamp.now())
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nExternal evaluation pipeline completed successfully!")
    print(f"All reports and visual figures saved to: {out_dir}\n")

if __name__ == "__main__":
    main()
