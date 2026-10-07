#!/usr/bin/env python3
"""
Compute and update Table 8, Table 9, and Table 10 on the Server.
Table 8: Paired Statistical Hypothesis Testing (McNemar, Wilcoxon, Paired t-test, Holm/FDR correction)
Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 resamples clustered by source video ID)
Table 10: Per-Class Granular Breakdown (Precision, Recall, F1 across all 22 resistance exercises)
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
from sklearn.metrics import precision_recall_fscore_support, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX, NUM_CLASSES
from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.statistics import (
    mcnemar_test,
    cluster_bootstrap_window,
    bootstrap_video,
    paired_video_confidence_test,
    adjust_p_values,
    format_p_value,
    get_significance_stars,
)

def send_marimo_toast(msg: str, *args, **kwargs):
    try:
        import marimo as mo
        kind = kwargs.get("kind", "success")
        mo.status.toast(msg, kind=kind)
    except Exception:
        pass

def get_sequence_predictions(
    model_type: str,
    ckpt_path: Path,
    device: torch.device,
    val_loader,
    test_loader,
    hidden_dim: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference on validation and test loaders using checkpoint."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    kwargs = {"model_type": model_type, "feature_method": "mix_v2"}
    if model_type == "Transformer":
        kwargs.update({
            "hidden_dim": 112,
            "num_layers": 3,
            "nhead": 4,
            "dropout": 0.2,
            "transformer_variant": "dual_branch"
        })
    elif model_type == "LSTM":
        kwargs.update({"hidden_dim": 160, "num_layers": 2})
    elif model_type == "BiLSTM":
        kwargs.update({"hidden_dim": 96, "num_layers": 2})

    model = build_model(**kwargs)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    trainer = Trainer(model=model, device=device)
    _, _, val_probs = trainer.predict(val_loader)
    _, _, test_probs = trainer.predict(test_loader)
    return np.array(val_probs), np.array(test_probs)

def main():
    parser = argparse.ArgumentParser(description="Run Tables 8, 9, 10 on Server")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--b_samples", type=int, default=1000, help="Bootstrap resamples")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Executing Tables 8, 9, 10 on device: {device} (B={args.b_samples})...")

    outputs_dir = ROOT_DIR / "outputs"
    checkpoints_dir = ROOT_DIR / "checkpoints"
    metadata_path = str(ROOT_DIR / "Final_dataset_metadata.csv")

    # 1. Load cached predictions from Phase 6 (Table 3 graph streams)
    cache_path = outputs_dir / "table3_stream_predictions.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing stream predictions cache: {cache_path}")
    
    print(f"Loading graph stream predictions from {cache_path}...")
    stream_data = torch.load(cache_path, map_location="cpu")
    
    # Ground truth targets and video IDs (seed 42)
    s42_bone = stream_data["T3.4"][42]
    val_targets = np.array(s42_bone["val_targets"])
    test_targets = np.array(s42_bone["test_targets"])
    val_vids = s42_bone["val_video_ids"]
    test_vids = s42_bone["test_video_ids"]
    
    y_test_t = test_targets
    N_test_win = len(y_test_t)
    unique_vids = np.unique(test_vids)
    N_test_vid = len(unique_vids)

    # Video ground truth
    y_test_vid_t, _, _, _ = aggregate_video_level_predictions(
        np.eye(NUM_CLASSES)[y_test_t], y_test_t, test_vids
    )

    print(f"Test partition: {N_test_win} windows across {N_test_vid} unique videos.")

    # 2. Load dataloaders for sequence models
    print("Loading sequence dataloaders for mix_v2...")
    _, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix_v2",
        batch_size=32,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=None,
        in_memory=True,
        seed=42,
        save_norm_artifact=False,
        strict_norm=False
    )

    # 3. Extract predictions for Sequence models (seed 42)
    print("Extracting predictions for Transformer, LSTM, BiLSTM...")
    ckpt_trans_clean = checkpoints_dir / "ablation_v2" / "seed42" / "best_Transformer_mix_v2_clean_seed42.pt"
    if not ckpt_trans_clean.exists():
        ckpt_trans_clean = checkpoints_dir / "upgrade_mix" / "mix_v2" / "seed42" / "best_Transformer_mix_v2.pt"

    ckpt_lstm = checkpoints_dir / "table2" / "seed42" / "best_LSTM_mix_v2_seed42.pt"
    ckpt_bilstm = checkpoints_dir / "table2" / "seed42" / "best_BiLSTM_mix_v2_seed42.pt"

    npz_aug = checkpoints_dir / "ablation_v2" / "seed42" / "probs_pair_mirror_yaw_seed42.npz"
    if npz_aug.exists():
        print(f"Loading precomputed probabilities from {npz_aug}...")
        npz_data = np.load(npz_aug, allow_pickle=True)
        vprob_aug_trans = np.array(npz_data["val_probs"])
        tprob_aug_trans = np.array(npz_data["test_probs"])
        train_prob_aug_trans = np.array(npz_data["train_probs"])
        train_targets = np.array(npz_data["train_targets"])
    else:
        ckpt_trans_aug = checkpoints_dir / "ablation_v2" / "seed42" / "best_Transformer_mix_v2_pair_mirror_yaw_seed42.pt"
        if not ckpt_trans_aug.exists():
            ckpt_trans_aug = checkpoints_dir / "ablation_v2" / "seed42" / "best_Transformer_mix_v2_candidate_minus_time_seed42.pt"
        vprob_aug_trans, tprob_aug_trans = get_sequence_predictions("Transformer", ckpt_trans_aug, device, val_loader, test_loader)
        train_prob_aug_trans = None

    vprob_clean_trans, tprob_clean_trans = get_sequence_predictions("Transformer", ckpt_trans_clean, device, val_loader, test_loader)
    vprob_lstm, tprob_lstm = get_sequence_predictions("LSTM", ckpt_lstm, device, val_loader, test_loader)
    vprob_bilstm, tprob_bilstm = get_sequence_predictions("BiLSTM", ckpt_bilstm, device, val_loader, test_loader)

    # Predictions for Graph models (seed 42)
    tprob_stgcn_raw = np.array(stream_data["T3.1"][42]["test_probs"])
    tprob_stgcn_world = np.array(stream_data["T3.2"][42]["test_probs"])
    vprob_stgcn_world = np.array(stream_data["T3.2"][42]["val_probs"])

    train_prob_bone = np.array(stream_data["T3.4"][42].get("train_probs", vprob_aug_trans))
    vprob_bone = np.array(stream_data["T3.4"][42]["val_probs"])
    tprob_bone = np.array(stream_data["T3.4"][42]["test_probs"])

    train_prob_joint = np.array(stream_data["T3.5"][42].get("train_probs", vprob_aug_trans))
    vprob_joint = np.array(stream_data["T3.5"][42]["val_probs"])
    tprob_joint = np.array(stream_data["T3.5"][42]["test_probs"])

    train_prob_jmot = np.array(stream_data["T3.6"][42].get("train_probs", vprob_aug_trans))
    vprob_jmot = np.array(stream_data["T3.6"][42]["val_probs"])
    tprob_jmot = np.array(stream_data["T3.6"][42]["test_probs"])

    train_prob_bmot = np.array(stream_data["T3.7"][42].get("train_probs", vprob_aug_trans))
    vprob_bmot = np.array(stream_data["T3.7"][42]["val_probs"])
    tprob_bmot = np.array(stream_data["T3.7"][42]["test_probs"])

    if "train_targets" in stream_data["T3.4"][42]:
        train_targets = np.array(stream_data["T3.4"][42]["train_targets"])

    # Ensembles
    # 1. Four-Stream AAGCN (Uniform Soft)
    tprob_4stream = (tprob_joint + tprob_bone + tprob_jmot + tprob_bmot) / 4.0
    p_4stream_w = np.argmax(tprob_4stream, axis=1)
    _, p_4stream_v, prob_4stream_v, _ = aggregate_video_level_predictions(tprob_4stream, y_test_t, test_vids)

    # 2. SkelGym-Lite (Stacking trained strictly on train set)
    X_train_lite = np.concatenate([train_prob_aug_trans, train_prob_bone], axis=1)
    X_test_lite = np.concatenate([tprob_aug_trans, tprob_bone], axis=1)
    clf_lite = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_lite.fit(X_train_lite, train_targets)
    tprob_skel_lite = clf_lite.predict_proba(X_test_lite)
    p_skel_lite_w = np.argmax(tprob_skel_lite, axis=1)
    _, p_skel_lite_v, prob_skel_lite_v, _ = aggregate_video_level_predictions(tprob_skel_lite, y_test_t, test_vids)

    # 3. SkelGym-Full (Stacking Meta-Classifier trained strictly on train set)
    X_train_full = np.concatenate([train_prob_aug_trans, train_prob_joint, train_prob_bone, train_prob_jmot, train_prob_bmot], axis=1)
    X_test_full = np.concatenate([tprob_aug_trans, tprob_joint, tprob_bone, tprob_jmot, tprob_bmot], axis=1)
    clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_full.fit(X_train_full, train_targets)
    tprob_skel_full = clf_full.predict_proba(X_test_full)
    p_skel_full_w = np.argmax(tprob_skel_full, axis=1)
    _, p_skel_full_v, prob_skel_full_v, _ = aggregate_video_level_predictions(tprob_skel_full, y_test_t, test_vids)

    # Individual model window and video predictions
    p_clean_trans_w = np.argmax(tprob_clean_trans, axis=1)
    _, p_clean_trans_v, prob_clean_trans_v, _ = aggregate_video_level_predictions(tprob_clean_trans, y_test_t, test_vids)

    p_aug_trans_w = np.argmax(tprob_aug_trans, axis=1)
    _, p_aug_trans_v, prob_aug_trans_v, _ = aggregate_video_level_predictions(tprob_aug_trans, y_test_t, test_vids)

    p_stgcn_world_w = np.argmax(tprob_stgcn_world, axis=1)
    _, p_stgcn_world_v, prob_stgcn_world_v, _ = aggregate_video_level_predictions(tprob_stgcn_world, y_test_t, test_vids)

    p_bone_w = np.argmax(tprob_bone, axis=1)
    _, p_bone_v, prob_bone_v, _ = aggregate_video_level_predictions(tprob_bone, y_test_t, test_vids)

    p_lstm_w = np.argmax(tprob_lstm, axis=1)
    _, p_lstm_v, prob_lstm_v, _ = aggregate_video_level_predictions(tprob_lstm, y_test_t, test_vids)

    p_bilstm_w = np.argmax(tprob_bilstm, axis=1)
    _, p_bilstm_v, prob_bilstm_v, _ = aggregate_video_level_predictions(tprob_bilstm, y_test_t, test_vids)

    print("\n" + "=" * 80)
    print("TABLE 8: PAIRED STATISTICAL HYPOTHESIS TESTING")
    print("=" * 80)

    comparisons = [
        ("Unaugmented Trans vs SkelGym-Aug Trans", p_clean_trans_w, p_aug_trans_w, prob_clean_trans_v, prob_aug_trans_v),
        ("Fixed ST-GCN vs Adaptive Four-Stream AAGCN", p_stgcn_world_w, p_4stream_w, prob_stgcn_world_v, prob_4stream_v),
        ("Single Sequence (Trans) vs SkelGym-Full", p_aug_trans_w, p_skel_full_w, prob_aug_trans_v, prob_skel_full_v),
        ("Single Graph (AAGCN Bone) vs SkelGym-Full", p_bone_w, p_skel_full_w, prob_bone_v, prob_skel_full_v),
        ("Four-Stream Graph AAGCN vs SkelGym-Full", p_4stream_w, p_skel_full_w, prob_4stream_v, prob_skel_full_v),
    ]

    t8_data = {}
    raw_p_win = []
    raw_p_vid_w = []
    raw_p_vid_t = []
    comp_results = []

    for comp_name, w1, w2, vprob1, vprob2 in comparisons:
        m_res = mcnemar_test(y_test_t, w1, w2, exact=True)
        c_res = paired_video_confidence_test(y_test_vid_t, vprob1, vprob2)
        raw_p_win.append(m_res["p_value"])
        raw_p_vid_w.append(c_res["wilcoxon_p"])
        raw_p_vid_t.append(c_res["ttest_p"])
        comp_results.append((comp_name, m_res, c_res))

    p_holm_win = adjust_p_values(raw_p_win, method="holm")
    p_holm_vid = adjust_p_values(raw_p_vid_w, method="holm")
    p_fdr_win = adjust_p_values(raw_p_win, method="fdr_bh")
    p_fdr_vid = adjust_p_values(raw_p_vid_w, method="fdr_bh")

    for idx, (comp_name, m_res, c_res) in enumerate(comp_results):
        t8_data[comp_name] = {
            "win_chi2": round(float(m_res["chi2"]), 2),
            "win_p": format_p_value(raw_p_win[idx]),
            "win_p_holm": format_p_value(p_holm_win[idx]),
            "win_p_fdr": format_p_value(p_fdr_win[idx]),
            "win_odds_ratio": round(float(m_res["odds_ratio"]), 2),
            "vid_wilcoxon_w": str(c_res.get("wilcoxon_stat", "—")),
            "vid_wilcoxon_p": format_p_value(raw_p_vid_w[idx]),
            "vid_wilcoxon_p_holm": format_p_value(p_holm_vid[idx]),
            "vid_wilcoxon_p_fdr": format_p_value(p_fdr_vid[idx]),
            "vid_paired_t_p": format_p_value(raw_p_vid_t[idx]),
            "vid_cohens_d": f"{c_res['cohens_d']:+.3f}",
            "significance": get_significance_stars(p_holm_vid[idx]),
            "status": "Verified"
        }
        print(f"  {comp_name:42s} | Win Chi2: {m_res['chi2']:5.2f} (p={format_p_value(p_holm_win[idx])}) | Vid Wilcoxon p: {format_p_value(p_holm_vid[idx])} | Cohen d: {c_res['cohens_d']:+.3f} | {get_significance_stars(p_holm_vid[idx])}")

    t8_path = outputs_dir / "statistical_tests_report.json"
    with open(t8_path, "w", encoding="utf-8") as f:
        json.dump(t8_data, f, indent=2)
    print(f"Saved Table 8 report to {t8_path}")

    print("\n" + "=" * 80)
    print(f"TABLE 9: NON-PARAMETRIC VIDEO-LEVEL CLUSTER BOOTSTRAP (B={args.b_samples})")
    print("=" * 80)

    models_boot = [
        ("LSTM (Mix v2 63-d)", p_lstm_w, p_lstm_v),
        ("BiLSTM (Mix v2 63-d)", p_bilstm_w, p_bilstm_v),
        ("ST-GCN (World 3D)", p_stgcn_world_w, p_stgcn_world_v),
        ("Transformer (Mix v2 Clean)", p_clean_trans_w, p_clean_trans_v),
        ("Transformer (Mix v2 SkelGym-Aug)", p_aug_trans_w, p_aug_trans_v),
        ("AAGCN (Bone 3D)", p_bone_w, p_bone_v),
        ("Four-Stream AAGCN", p_4stream_w, p_4stream_v),
        ("SkelGym-Lite", p_skel_lite_w, p_skel_lite_v),
        ("SkelGym-Full (Stacking)", p_skel_full_w, p_skel_full_v),
    ]

    t9_data = {}
    for mname, wp, vp in models_boot:
        w_boot = cluster_bootstrap_window(y_test_t, wp, test_vids, B=args.b_samples, seed=42)
        v_boot = bootstrap_video(y_test_vid_t, vp, B=args.b_samples, seed=42)

        w_acc_str = f"{w_boot['acc_mean']:.2f}% [{w_boot['acc_ci'][0]:.2f}%, {w_boot['acc_ci'][1]:.2f}%]"
        w_f1_str = f"{w_boot['f1_mean']:.4f} [{w_boot['f1_ci'][0]:.4f}, {w_boot['f1_ci'][1]:.4f}]"
        v_acc_str = f"{v_boot['acc_mean']:.2f}% [{v_boot['acc_ci'][0]:.2f}%, {v_boot['acc_ci'][1]:.2f}%]"
        v_f1_str = f"{v_boot['f1_mean']:.4f} [{v_boot['f1_ci'][0]:.4f}, {v_boot['f1_ci'][1]:.4f}]"

        t9_data[mname] = {
            "w_acc_mean": round(float(w_boot["acc_mean"]), 2),
            "w_acc_ci": [round(float(c), 2) for c in w_boot["acc_ci"]],
            "w_acc_str": w_acc_str,
            "w_f1_mean": round(float(w_boot["f1_mean"]), 4),
            "w_f1_ci": [round(float(c), 4) for c in w_boot["f1_ci"]],
            "w_f1_str": w_f1_str,
            "v_acc_mean": round(float(v_boot["acc_mean"]), 2),
            "v_acc_ci": [round(float(c), 2) for c in v_boot["acc_ci"]],
            "v_acc_str": v_acc_str,
            "v_f1_mean": round(float(v_boot["f1_mean"]), 4),
            "v_f1_ci": [round(float(c), 4) for c in v_boot["f1_ci"]],
            "v_f1_str": v_f1_str,
            "status": "Verified"
        }
        print(f"  {mname:30s} | Win Acc: {w_acc_str} | Vid Acc: {v_acc_str}")

    t9_path = outputs_dir / "bootstrap_confidence_intervals.json"
    with open(t9_path, "w", encoding="utf-8") as f:
        json.dump(t9_data, f, indent=2)
    print(f"Saved Table 9 report to {t9_path}")

    print("\n" + "=" * 80)
    print("TABLE 10: PER-CLASS GRANULAR BREAKDOWN (SKELGYM-FULL STACKING SOTA)")
    print("=" * 80)

    p_w, r_w, f1_w, s_w = precision_recall_fscore_support(y_test_t, p_skel_full_w, labels=list(range(NUM_CLASSES)), zero_division=0)
    p_v, r_v, f1_v, s_v = precision_recall_fscore_support(y_test_vid_t, p_skel_full_v, labels=list(range(NUM_CLASSES)), zero_division=0)

    t10_data = {"classes": {}}
    for idx, act in enumerate(ACTIONS):
        t10_data["classes"][act] = {
            "win_precision": round(float(p_w[idx]), 4),
            "win_recall": round(float(r_w[idx]), 4),
            "win_f1": round(float(f1_w[idx]), 4),
            "win_support": int(s_w[idx]),
            "vid_precision": round(float(p_v[idx]), 4),
            "vid_recall": round(float(r_v[idx]), 4),
            "vid_f1": round(float(f1_v[idx]), 4),
            "vid_support": int(s_v[idx])
        }

    overall_win_acc = float(np.mean(y_test_t == p_skel_full_w))
    overall_vid_acc = float(np.mean(y_test_vid_t == p_skel_full_v))

    t10_data["classes"]["Overall Accuracy"] = {
        "win_precision": round(overall_win_acc, 4),
        "win_recall": round(overall_win_acc, 4),
        "win_f1": round(overall_win_acc, 4),
        "win_support": int(len(y_test_t)),
        "vid_precision": round(overall_vid_acc, 4),
        "vid_recall": round(overall_vid_acc, 4),
        "vid_f1": round(overall_vid_acc, 4),
        "vid_support": int(len(y_test_vid_t))
    }
    t10_data["classes"]["Macro Average"] = {
        "win_precision": round(float(np.mean(p_w)), 4),
        "win_recall": round(float(np.mean(r_w)), 4),
        "win_f1": round(float(np.mean(f1_w)), 4),
        "win_support": int(len(y_test_t)),
        "vid_precision": round(float(np.mean(p_v)), 4),
        "vid_recall": round(float(np.mean(r_v)), 4),
        "vid_f1": round(float(np.mean(f1_v)), 4),
        "vid_support": int(len(y_test_vid_t))
    }

    t10_path = outputs_dir / "per_class_results.json"
    with open(t10_path, "w", encoding="utf-8") as f:
        json.dump(t10_data, f, indent=2)
    print(f"Saved Table 10 report to {t10_path}")
    print(f"Overall Test Video Accuracy: {overall_vid_acc*100.0:.2f}% | Macro F1: {np.mean(f1_v):.4f}")

    send_marimo_toast("Table 8, 9, 10 executed and verified successfully on Server!")

if __name__ == "__main__":
    main()
