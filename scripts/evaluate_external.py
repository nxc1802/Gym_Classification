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
            apply_geometric_norm=cfg.get("normalization", {}).get("geometric", True)
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

    # 2. Load Frozen Train Normalization Statistics
    print(f"\n[2/7] Loading frozen SkelGym train normalization statistics...")
    norm_stats = {}
    for feat in ["mix", "rel_3d", "bone_3d", "joint_motion_3d", "bone_motion_3d"]:
        stat_file = ref_dir / f"normalization_{feat}.npz"
        if not stat_file.exists():
            raise FileNotFoundError(f"Missing frozen normalization artifact: {stat_file}. Run scripts/freeze_reference_artifacts.py first!")
        data = np.load(stat_file)
        norm_stats[feat] = (data["mean"], data["std"])
        print(f"  Loaded {feat} (shape {data['mean'].shape})")

    # 3. Extract and Normalize Windows for each Feature Representation
    print(f"\n[3/7] Extracting synchronized windows and standardizing with train stats...")
    seq_len = int(cfg.get("window", {}).get("seq_len", 32))
    stride = int(cfg.get("window", {}).get("stride", 32))

    feature_windows = {}
    base_info = None

    for feat in ["mix", "rel_3d", "bone_3d", "joint_motion_3d", "bone_motion_3d"]:
        win_data = ext_ds.extract_windows(feature_method=feat, seq_len=seq_len, stride=stride)
        raw_feat = win_data["features"]  # (N, T, D)
        mean, std = norm_stats[feat]
        # Standardize strictly using train set statistics (prevent target data leakage)
        norm_feat = (raw_feat - mean[None, :, :]) / (std[None, :, :] + 1e-7)
        feature_windows[feat] = norm_feat.astype(np.float32)

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

    # 4. Load Models and Execute Forward Pass
    print(f"\n[4/7] Loading frozen neural checkpoints and predicting...")
    model_defs = {
        "ST-GCN (Rel 3D)": ("STGCN", "rel_3d", "checkpoints/best_STGCN_T3.2_rel_3d.pt"),
        "Transformer (Mix)": ("Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt"),
        "AAGCN (Bone 3D)": ("AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt"),
        "AAGCN (Rel 3D)": ("AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt"),
        "AAGCN (Joint Mot)": ("AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt"),
        "AAGCN (Bone Mot)": ("AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt"),
    }

    test_probs = {}
    models_dict = {}

    for name, (m_type, f_type, ckpt_p) in model_defs.items():
        ckpt_full = str(PROJECT_ROOT / ckpt_p)
        model = load_checkpoint(m_type, f_type, ckpt_full, device)
        models_dict[name] = (model, m_type, f_type)
        feats = feature_windows[f_type]
        p = predict_windows(model, feats, device)
        test_probs[name] = p
        print(f"  Predicted {name:<20}: window shape {p.shape}")

    # Load frozen ensemble calibration weights
    ens_weights_file = ref_dir / "ensemble_weights.json"
    with open(ens_weights_file, "r") as f:
        ens_cfg = json.load(f)

    # SkelGym-Lite Blending
    w_lite = ens_cfg["lite"]["weights_window"]
    test_probs["SkelGym-Lite"] = (
        w_lite[0] * test_probs["Transformer (Mix)"] +
        w_lite[1] * test_probs["AAGCN (Bone 3D)"]
    )

    # SkelGym-Full Blending
    w_full = ens_cfg["full"]["weights_window"]
    test_probs["SkelGym-Full"] = (
        w_full[0] * test_probs["Transformer (Mix)"] +
        w_full[1] * test_probs["AAGCN (Bone 3D)"] +
        w_full[2] * test_probs["AAGCN (Rel 3D)"] +
        w_full[3] * test_probs["AAGCN (Joint Mot)"] +
        w_full[4] * test_probs["AAGCN (Bone Mot)"]
    )

    # 5. Open-Set vs. Closed-Set Comprehensive Evaluation
    print(f"\n[5/7] Executing Open-Set and Closed-Set Hierarchical Evaluation...")
    eval_models = ["ST-GCN (Rel 3D)", "Transformer (Mix)", "AAGCN (Bone 3D)", "SkelGym-Lite", "SkelGym-Full"]

    rows_open = []
    rows_closed = []
    bootstrap_results = {}

    print("\n" + "=" * 95)
    print(f"TABLE X: CROSS-DATASET EXTERNAL VALIDATION RESULTS ({dataset_name.upper()} - {split_group.upper()})")
    print("=" * 95)
    print(f"{'Model':<22} | {'Open Win':<9} | {'Open Rec':<9} | {'Closed Win':<10} | {'Closed Rec':<10} | {'Macro F1':<8}")
    print("-" * 80)

    for m_name in eval_models:
        prob = test_probs[m_name]

        # A. Open-Set
        res_open_win = evaluate_window_level(prob, y_win_true, target_class_indices, mode="open_set")
        res_open_rec = aggregate_hierarchical_predictions(prob, y_win_true, win_rec_ids, target_class_indices, mode="open_set")

        # B. Closed-Set
        res_closed_win = evaluate_window_level(prob, y_win_true, target_class_indices, mode="closed_set")
        res_closed_rec = aggregate_hierarchical_predictions(prob, y_win_true, win_rec_ids, target_class_indices, mode="closed_set")

        # C. Recording-Level Bootstrap CI (95%, B=1000)
        ci_open = compute_recording_level_bootstrap_ci(
            res_open_rec["group_probs"], res_open_rec["group_trues"], target_class_indices, mode="open_set"
        )
        ci_closed = compute_recording_level_bootstrap_ci(
            res_closed_rec["group_probs"], res_closed_rec["group_trues"], target_class_indices, mode="closed_set"
        )
        bootstrap_results[m_name] = {
            "open_set_recording": ci_open,
            "closed_set_recording": ci_closed
        }

        print(
            f"{m_name:<22} | {res_open_win['accuracy']:>8.2f}% | {res_open_rec['accuracy']:>8.2f}% | "
            f"{res_closed_win['accuracy']:>9.2f}% | {res_closed_rec['accuracy']:>9.2f}% | {res_closed_rec['macro_f1']:>8.4f}"
        )

        rows_open.append({
            "model": m_name,
            "window_acc": res_open_win["accuracy"],
            "recording_acc": res_open_rec["accuracy"],
            "macro_f1": res_open_rec["macro_f1"],
            "ci_95_recording_acc": ci_open["acc_ci"],
            "ci_95_macro_f1": ci_open["f1_ci"]
        })

        rows_closed.append({
            "model": m_name,
            "window_acc": res_closed_win["accuracy"],
            "recording_acc": res_closed_rec["accuracy"],
            "macro_f1": res_closed_rec["macro_f1"],
            "ci_95_recording_acc": ci_closed["acc_ci"],
            "ci_95_macro_f1": ci_closed["f1_ci"],
            **{f"recall_{k}": v for k, v in res_closed_rec["per_class_recall"].items()}
        })


    # Save summary tables
    df_metrics_open = pd.DataFrame(rows_open)
    df_metrics_closed = pd.DataFrame(rows_closed)
    df_metrics_open.to_csv(out_dir / "metrics_open.csv", index=False)
    df_metrics_closed.to_csv(out_dir / "metrics_closed.csv", index=False)
    with open(out_dir / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    # 6. Export Predictions & Confusion Matrices
    print(f"\n[6/7] Exporting prediction logs and confusion matrices...")
    best_model_name = "SkelGym-Full"
    best_prob = test_probs[best_model_name]

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

    # Error analysis table
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
    trans_m = models_dict["Transformer (Mix)"][0]
    ext_embs = extract_penultimate_embeddings(trans_m, feature_windows["mix"], "Transformer", device)

    # Load SkelGym test set dataloader for centroid comparison
    metadata_path = str(PROJECT_ROOT / "data" / "Final_dataset_metadata.csv")
    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix",
        batch_size=32,
        seq_len=32,
        stride=32,
        landmark_dir=str(PROJECT_ROOT / "data" / "landmarks"),
        num_workers=0,
        in_memory=True
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
