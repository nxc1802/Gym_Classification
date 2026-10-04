#!/usr/bin/env python3
"""
Rigorous Statistical Evaluation & Hypothesis Testing Engine for SkelGym.
Implements:
  1. Window-Level McNemar Tests (Exact Binomial & Continuity-Corrected Chi-Squared)
  2. Video-Level Paired Tests (Exact McNemar on correctness & Wilcoxon/t-test on continuous prediction probabilities)
  3. Video-Cluster-Aware Bootstrap Resampling (B=1,000 resamples clustered by source video ID)
  4. Multiple Comparison Adjustments (Single-step Bonferroni, Step-down Holm-Bonferroni, Benjamini-Hochberg FDR)
  5. Publication-Ready Markdown and LaTeX Reporting
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import NUM_CLASSES
from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions
from src.utils.reproducibility import load_checkpoint_weights
from src.utils.statistics import (
    mcnemar_test,
    cluster_bootstrap_window,
    bootstrap_video,
    paired_video_confidence_test,
    adjust_p_values,
    format_p_value,
    get_significance_stars,
)

def get_predictions(
    model_type: str,
    feat_type: str,
    ckpt_path: Path,
    device: torch.device,
    val_l,
    test_l
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Safely loads checkpoint weights and computes validation and test predictions.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict, _ = load_checkpoint_weights(ckpt_path, device="cpu")
    m = build_model(model_type, feat_type, num_classes=NUM_CLASSES)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()

    trainer = Trainer(model=m, device=device)
    y_vt, y_vp, y_vprob = trainer.predict(val_l)
    y_tt, y_tp, y_tprob = trainer.predict(test_l)
    return y_vprob, y_tprob, y_vt, y_tt, y_vp, y_tp

def main():
    parser = argparse.ArgumentParser(description="SkelGym Statistical Inference & Bootstrap Runner")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    parser.add_argument("--b_samples", type=int, default=1000, help="Number of bootstrap resamples (default: 1000)")
    parser.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Directory containing landmark CSVs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints base directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for reports")
    parser.add_argument("--smoke_test", action="store_true", help="Run in smoke test mode with synthetic predictions if checkpoints missing")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)
    print(f"Running Statistical Evaluation on device: {device} (B={args.b_samples} resamples)")

    metadata_path = args.metadata if os.path.exists(args.metadata) else ("data/" + args.metadata if os.path.exists("data/" + args.metadata) else args.metadata)
    landmark_dir = args.landmark_dir
    ckpt_base = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Required Checkpoint Paths
    def find_ckpt(name_v2, name_v1):
        p2 = ckpt_base / name_v2
        if p2.exists():
            return p2
        return ckpt_base / name_v1

    ckpt_paths = {
        "LSTM_mix": find_ckpt("best_LSTM_T1.9_mix_v2.pt", "best_LSTM_T1.9_mix.pt"),
        "BiLSTM_mix": find_ckpt("best_BiLSTM_T1.18_mix_v2.pt", "best_BiLSTM_T1.18_mix.pt"),
        "Clean_Trans_mix": find_ckpt("best_Transformer_T1.27_mix_v2.pt", "best_Transformer_T1.27_mix.pt"),
        "Aug_Trans_mix": find_ckpt("best_Transformer_T2.2_mix_v2.pt", "best_Transformer_T2.2_mix.pt"),
        "STGCN_rel_3d": ckpt_base / "best_STGCN_T3.2_rel_3d.pt",
        "AAGCN_bone_3d": ckpt_base / "best_AAGCN_T4.2_bone_3d.pt",
        "AAGCN_rel_3d": ckpt_base / "best_AAGCN_T4.3_rel_3d.pt",
        "AAGCN_joint_motion_3d": ckpt_base / "best_AAGCN_T4.4_joint_motion_3d.pt",
        "AAGCN_bone_motion_3d": ckpt_base / "best_AAGCN_T4.5_bone_motion_3d.pt",
    }

    missing_ckpts = [k for k, p in ckpt_paths.items() if not p.exists()]
    if missing_ckpts and not args.smoke_test:
        print(f"\n[WARNING] Missing {len(missing_ckpts)}/{len(ckpt_paths)} checkpoints:")
        for k in missing_ckpts:
            print(f"  - {k} -> {ckpt_paths[k]}")
        print("To run with synthetic mock predictions for pipeline validation, pass --smoke_test.")
        return

    # Load dataloaders
    print("\nLoading validation and test partitions...")
    loaders = {}
    mix_feat = "mix_v2" if any("mix_v2" in str(p) for p in [ckpt_paths["LSTM_mix"], ckpt_paths["Aug_Trans_mix"]]) else "mix"
    features = [mix_feat, "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    for feat in features:
        _, val_l, test_l = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=feat,
            batch_size=32,
            seq_len=32,
            stride=16,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True,
            smoke_test=args.smoke_test,
            seed=42,
            strict_norm=True
        )
        loaders[feat] = (val_l, test_l)

    val_l_mix, test_l_mix = loaders[mix_feat]
    y_test_true = np.array(test_l_mix.dataset.labels)
    test_vids = test_l_mix.dataset.video_ids
    val_vids = val_l_mix.dataset.video_ids
    y_val_true = np.array(val_l_mix.dataset.labels)
    N_test_win = len(y_test_true)
    unique_vids = np.unique(test_vids)
    N_test_vid = len(unique_vids)

    print(f"Loaded held-out test split: {N_test_win} windows across {N_test_vid} unique videos.")

    if not missing_ckpts:
        print("\nLoading models and extracting predictions...")
        vprob_lstm, tprob_lstm, _, _, _, tp_lstm = get_predictions(
            "LSTM", mix_feat, ckpt_paths["LSTM_mix"], device, loaders[mix_feat][0], loaders[mix_feat][1]
        )
        vprob_bilstm, tprob_bilstm, _, _, _, tp_bilstm = get_predictions(
            "BiLSTM", mix_feat, ckpt_paths["BiLSTM_mix"], device, loaders[mix_feat][0], loaders[mix_feat][1]
        )
        vprob_clean_trans, tprob_clean_trans, _, _, _, tp_clean_trans = get_predictions(
            "Transformer", mix_feat, ckpt_paths["Clean_Trans_mix"], device, loaders[mix_feat][0], loaders[mix_feat][1]
        )
        vprob_aug_trans, tprob_aug_trans, _, _, _, tp_aug_trans = get_predictions(
            "Transformer", mix_feat, ckpt_paths["Aug_Trans_mix"], device, loaders[mix_feat][0], loaders[mix_feat][1]
        )
        vprob_stgcn, tprob_stgcn, _, _, _, tp_stgcn = get_predictions(
            "STGCN", "rel_3d", ckpt_paths["STGCN_rel_3d"], device, loaders["rel_3d"][0], loaders["rel_3d"][1]
        )
        vprob_bone, tprob_bone, _, _, _, tp_bone = get_predictions(
            "AAGCN", "bone_3d", ckpt_paths["AAGCN_bone_3d"], device, loaders["bone_3d"][0], loaders["bone_3d"][1]
        )
        vprob_joint, tprob_joint, _, _, _, tp_joint = get_predictions(
            "AAGCN", "rel_3d", ckpt_paths["AAGCN_rel_3d"], device, loaders["rel_3d"][0], loaders["rel_3d"][1]
        )
        vprob_jmot, tprob_jmot, _, _, _, tp_jmot = get_predictions(
            "AAGCN", "joint_motion_3d", ckpt_paths["AAGCN_joint_motion_3d"], device, loaders["joint_motion_3d"][0], loaders["joint_motion_3d"][1]
        )
        vprob_bmot, tprob_bmot, _, _, _, tp_bmot = get_predictions(
            "AAGCN", "bone_motion_3d", ckpt_paths["AAGCN_bone_motion_3d"], device, loaders["bone_motion_3d"][0], loaders["bone_motion_3d"][1]
        )

        # Ensembles
        four_stream_val = [vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
        four_stream_test = [tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
        four_stream_ens = WeightedSoftVotingEnsemble()
        four_stream_ens.fit_window(four_stream_val, y_val_true)
        tp_4stream_win = four_stream_ens.predict_window(four_stream_test)
        tprob_4stream = np.mean(four_stream_test, axis=0)

        lite_val = [vprob_aug_trans, vprob_bone]
        lite_test = [tprob_aug_trans, tprob_bone]
        skelgym_lite = WeightedSoftVotingEnsemble()
        skelgym_lite.fit_window(lite_val, y_val_true)
        skelgym_lite.fit_video(lite_val, y_val_true, val_vids)
        tp_skelgym_lite_win = skelgym_lite.predict_window(lite_test)
        _, tp_skelgym_lite_vid, tprob_skelgym_lite_vid, _ = skelgym_lite.predict_video(lite_test, y_test_true, test_vids)

        five_stream_val = [vprob_aug_trans, vprob_joint, vprob_bone, vprob_jmot, vprob_bmot]
        five_stream_test = [tprob_aug_trans, tprob_joint, tprob_bone, tprob_jmot, tprob_bmot]
        skelgym_full = WeightedSoftVotingEnsemble()
        skelgym_full.fit_window(five_stream_val, y_val_true)
        skelgym_full.fit_video(five_stream_val, y_val_true, val_vids)
        tp_skelgym_win = skelgym_full.predict_window(five_stream_test)
        y_test_vid_t, tp_skelgym_vid, tprob_skelgym_vid, _ = skelgym_full.predict_video(five_stream_test, y_test_true, test_vids)

        # Video-level aggregation
        _, vp_lstm, tprob_lstm_vid, _ = aggregate_video_level_predictions(tprob_lstm, y_test_true, test_vids)
        _, vp_bilstm, tprob_bilstm_vid, _ = aggregate_video_level_predictions(tprob_bilstm, y_test_true, test_vids)
        _, vp_clean_trans, tprob_clean_trans_vid, _ = aggregate_video_level_predictions(tprob_clean_trans, y_test_true, test_vids)
        _, vp_aug_trans, tprob_aug_trans_vid, _ = aggregate_video_level_predictions(tprob_aug_trans, y_test_true, test_vids)
        _, vp_stgcn, tprob_stgcn_vid, _ = aggregate_video_level_predictions(tprob_stgcn, y_test_true, test_vids)
        _, vp_bone, tprob_bone_vid, _ = aggregate_video_level_predictions(tprob_bone, y_test_true, test_vids)
        _, vp_4stream, tprob_4stream_vid, _ = aggregate_video_level_predictions(tprob_4stream, y_test_true, test_vids)
    else:
        # Smoke test mock generation
        print("\n[SMOKE TEST] Synthesizing mock predictions for testing statistical engine...")
        np.random.seed(42)
        y_test_vid_t, _, _, _ = aggregate_video_level_predictions(
            np.eye(NUM_CLASSES)[y_test_true], y_test_true, test_vids
        )
        K = len(y_test_vid_t)

        def mock_probs(acc_target, N):
            p = np.zeros((N, NUM_CLASSES))
            for i in range(N):
                correct = np.random.rand() < acc_target
                target = y_test_true[i] if N == N_test_win else y_test_vid_t[i]
                if correct:
                    p[i, target] = 0.8
                else:
                    wrong = (target + 1) % NUM_CLASSES
                    p[i, wrong] = 0.8
                p[i] += 0.2 / NUM_CLASSES
                p[i] /= p[i].sum()
            return p

        tprob_lstm = mock_probs(0.50, N_test_win)
        tprob_bilstm = mock_probs(0.53, N_test_win)
        tprob_clean_trans = mock_probs(0.55, N_test_win)
        tprob_aug_trans = mock_probs(0.66, N_test_win)
        tprob_stgcn = mock_probs(0.54, N_test_win)
        tprob_bone = mock_probs(0.65, N_test_win)
        tprob_4stream = mock_probs(0.69, N_test_win)

        tp_lstm = tprob_lstm.argmax(axis=1)
        tp_bilstm = tprob_bilstm.argmax(axis=1)
        tp_clean_trans = tprob_clean_trans.argmax(axis=1)
        tp_aug_trans = tprob_aug_trans.argmax(axis=1)
        tp_stgcn = tprob_stgcn.argmax(axis=1)
        tp_bone = tprob_bone.argmax(axis=1)
        tp_4stream_win = tprob_4stream.argmax(axis=1)
        tp_skelgym_lite_win = mock_probs(0.69, N_test_win).argmax(axis=1)
        tp_skelgym_win = mock_probs(0.72, N_test_win).argmax(axis=1)

        tprob_lstm_vid = mock_probs(0.58, K)
        tprob_bilstm_vid = mock_probs(0.61, K)
        tprob_clean_trans_vid = mock_probs(0.64, K)
        tprob_aug_trans_vid = mock_probs(0.77, K)
        tprob_stgcn_vid = mock_probs(0.62, K)
        tprob_bone_vid = mock_probs(0.76, K)
        tprob_4stream_vid = mock_probs(0.79, K)
        tprob_skelgym_lite_vid = mock_probs(0.80, K)
        tprob_skelgym_vid = mock_probs(0.83, K)

        vp_lstm = tprob_lstm_vid.argmax(axis=1)
        vp_bilstm = tprob_bilstm_vid.argmax(axis=1)
        vp_clean_trans = tprob_clean_trans_vid.argmax(axis=1)
        vp_aug_trans = tprob_aug_trans_vid.argmax(axis=1)
        vp_stgcn = tprob_stgcn_vid.argmax(axis=1)
        vp_bone = tprob_bone_vid.argmax(axis=1)
        vp_4stream = tprob_4stream_vid.argmax(axis=1)
        tp_skelgym_lite_vid = tprob_skelgym_lite_vid.argmax(axis=1)
        tp_skelgym_vid = tprob_skelgym_vid.argmax(axis=1)

    # =========================================================================
    # PART 1: WINDOW-LEVEL MCNEMAR TESTS WITH MULTIPLE TESTING CORRECTION
    # =========================================================================
    print("\n" + "=" * 95)
    print(f"1. MCNEMAR TESTS ON TEMPORAL WINDOWS (N={N_test_win})")
    print("=" * 95)

    comparisons_win = [
        ("Clean Transformer vs Aug Transformer", tp_clean_trans, tp_aug_trans),
        ("ST-GCN vs Four-Stream AAGCN", tp_stgcn, tp_4stream_win),
        ("Transformer Mix (Aug) vs SkelGym-Full", tp_aug_trans, tp_skelgym_win),
        ("AAGCN Bone (Aug) vs SkelGym-Full", tp_bone, tp_skelgym_win),
        ("Four-Stream AAGCN vs SkelGym-Full", tp_4stream_win, tp_skelgym_win)
    ]

    win_results = []
    raw_p_win = []

    for name, p1, p2 in comparisons_win:
        res = mcnemar_test(y_test_true, p1, p2, exact=True)
        win_results.append((name, res))
        raw_p_win.append(res["p_value"])

    p_bonf_win = adjust_p_values(raw_p_win, method="bonferroni")
    p_holm_win = adjust_p_values(raw_p_win, method="holm")
    p_fdr_win = adjust_p_values(raw_p_win, method="fdr_bh")

    print(f"{'Comparison':<40} | {'Acc1 -> Acc2 (Delta)':<22} | {'b / c':<10} | {'p (raw)':<10} | {'p (Holm)':<10} | {'p (FDR)':<10} | {'Sig'}")
    print("-" * 115)
    for idx, (name, res) in enumerate(win_results):
        acc_str = f"{res['acc1']:.1f}% -> {res['acc2']:.1f}% ({res['delta_acc']:+.1f}%)"
        bc_str = f"{res['b']} / {res['c']}"
        raw_str = format_p_value(raw_p_win[idx])
        holm_str = format_p_value(p_holm_win[idx])
        fdr_str = format_p_value(p_fdr_win[idx])
        stars = get_significance_stars(p_holm_win[idx])
        print(f"{name:<40} | {acc_str:<22} | {bc_str:<10} | {raw_str:<10} | {holm_str:<10} | {fdr_str:<10} | {stars}")

    # =========================================================================
    # PART 2: VIDEO-LEVEL STATISTICAL INFERENCE (N=233 Action Videos)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f"2. VIDEO-LEVEL STATISTICAL INFERENCE (N={N_test_vid} Independent Action Videos)")
    print("=" * 95)

    comparisons_vid = [
        ("Clean Transformer vs Aug Transformer", vp_clean_trans, vp_aug_trans, tprob_clean_trans_vid, tprob_aug_trans_vid),
        ("ST-GCN vs Four-Stream AAGCN", vp_stgcn, vp_4stream, tprob_stgcn_vid, tprob_4stream_vid),
        ("Transformer Mix (Aug) vs SkelGym-Full", vp_aug_trans, tp_skelgym_vid, tprob_aug_trans_vid, tprob_skelgym_vid),
        ("AAGCN Bone (Aug) vs SkelGym-Full", vp_bone, tp_skelgym_vid, tprob_bone_vid, tprob_skelgym_vid),
        ("Four-Stream AAGCN vs SkelGym-Full", vp_4stream, tp_skelgym_vid, tprob_4stream_vid, tprob_skelgym_vid)
    ]

    vid_mcnemar_results = []
    raw_p_vid_mcnemar = []
    vid_confidence_results = []
    raw_p_vid_wilcoxon = []

    for name, v1, v2, prob1, prob2 in comparisons_vid:
        # McNemar exact test on video correctness
        m_res = mcnemar_test(y_test_vid_t, v1, v2, exact=True)
        vid_mcnemar_results.append((name, m_res))
        raw_p_vid_mcnemar.append(m_res["p_value"])

        # Continuous probability margin test
        c_res = paired_video_confidence_test(y_test_vid_t, prob1, prob2)
        vid_confidence_results.append((name, c_res))
        raw_p_vid_wilcoxon.append(c_res["wilcoxon_p"])

    p_holm_vid_mcnemar = adjust_p_values(raw_p_vid_mcnemar, method="holm")
    p_fdr_vid_mcnemar = adjust_p_values(raw_p_vid_mcnemar, method="fdr_bh")

    p_holm_vid_wilcoxon = adjust_p_values(raw_p_vid_wilcoxon, method="holm")
    p_fdr_vid_wilcoxon = adjust_p_values(raw_p_vid_wilcoxon, method="fdr_bh")

    print("\n--- A. Video-Level Exact McNemar Test (Binary Correctness) ---")
    print(f"{'Comparison':<40} | {'Vid Acc1 -> Acc2 (Delta)':<25} | {'b / c':<10} | {'p (Exact)':<10} | {'p (Holm)':<10} | {'p (FDR)':<10} | {'Sig'}")
    print("-" * 120)
    for idx, (name, res) in enumerate(vid_mcnemar_results):
        acc_str = f"{res['acc1']:.1f}% -> {res['acc2']:.1f}% ({res['delta_acc']:+.1f}%)"
        bc_str = f"{res['b']} / {res['c']}"
        raw_str = format_p_value(raw_p_vid_mcnemar[idx])
        holm_str = format_p_value(p_holm_vid_mcnemar[idx])
        fdr_str = format_p_value(p_fdr_vid_mcnemar[idx])
        stars = get_significance_stars(p_holm_vid_mcnemar[idx])
        print(f"{name:<40} | {acc_str:<25} | {bc_str:<10} | {raw_str:<10} | {holm_str:<10} | {fdr_str:<10} | {stars}")

    print("\n--- B. Video-Level Confidence Calibration (Wilcoxon on P(True Class)) ---")
    print(f"{'Comparison':<40} | {'Mean Delta P(True)':<20} | {'Cohen d_z':<10} | {'Wilcoxon p':<12} | {'p (Holm)':<10} | {'t-test p':<10} | {'Sig'}")
    print("-" * 120)
    for idx, (name, res) in enumerate(vid_confidence_results):
        delta_p_str = f"{res['mean_delta_prob']:+.4f}"
        d_str = f"{res['cohens_d']:.3f}"
        raw_w_str = format_p_value(res['wilcoxon_p'])
        holm_w_str = format_p_value(p_holm_vid_wilcoxon[idx])
        ttest_str = format_p_value(res['ttest_p'])
        stars = get_significance_stars(p_holm_vid_wilcoxon[idx])
        print(f"{name:<40} | {delta_p_str:<20} | {d_str:<10} | {raw_w_str:<12} | {holm_w_str:<10} | {ttest_str:<10} | {stars}")

    # =========================================================================
    # PART 3: NON-PARAMETRIC CLUSTER BOOTSTRAP (B=1,000 resamples)
    # =========================================================================
    print("\n" + "=" * 95)
    print(f"3. CLUSTER BOOTSTRAP (B={args.b_samples} resamples, Clustered by Video ID)")
    print("=" * 95)

    models_eval = [
        ("LSTM (Mix 117-d)", tp_lstm, vp_lstm),
        ("BiLSTM (Mix 117-d)", tp_bilstm, vp_bilstm),
        ("ST-GCN (Rel 3D)", tp_stgcn, vp_stgcn),
        ("Transformer (Mix 117-d)", tp_aug_trans, vp_aug_trans),
        ("AAGCN (Bone 3D)", tp_bone, vp_bone),
        ("SkelGym-Lite", tp_skelgym_lite_win, tp_skelgym_lite_vid),
        ("SkelGym-Full", tp_skelgym_win, tp_skelgym_vid)
    ]

    bootstrap_table = []
    print(f"{'Architecture':<26} | {'Window Acc (Cluster 95% CI)':<30} | {'Window F1 (Cluster 95% CI)':<28} | {'Video Acc (95% CI)':<26} | {'Video F1 (95% CI)':<24}")
    print("-" * 140)

    for name, p_win, p_vid in models_eval:
        w_boot = cluster_bootstrap_window(y_test_true, p_win, test_vids, B=args.b_samples, seed=42)
        v_boot = bootstrap_video(y_test_vid_t, p_vid, B=args.b_samples, seed=42)

        w_acc_str = f"{w_boot['acc_mean']:.2f}% [{w_boot['acc_ci'][0]:.2f}%, {w_boot['acc_ci'][1]:.2f}%]"
        w_f1_str = f"{w_boot['f1_mean']:.4f} [{w_boot['f1_ci'][0]:.4f}, {w_boot['f1_ci'][1]:.4f}]"
        v_acc_str = f"{v_boot['acc_mean']:.2f}% [{v_boot['acc_ci'][0]:.2f}%, {v_boot['acc_ci'][1]:.2f}%]"
        v_f1_str = f"{v_boot['f1_mean']:.4f} [{v_boot['f1_ci'][0]:.4f}, {v_boot['f1_ci'][1]:.4f}]"

        print(f"{name:<26} | {w_acc_str:<30} | {w_f1_str:<28} | {v_acc_str:<26} | {v_f1_str:<24}")
        bootstrap_table.append({
            "name": name,
            "w_acc_mean": w_boot["acc_mean"],
            "w_acc_ci": w_boot["acc_ci"],
            "w_acc_str": w_acc_str,
            "w_f1_mean": w_boot["f1_mean"],
            "w_f1_ci": w_boot["f1_ci"],
            "w_f1_str": w_f1_str,
            "v_acc_mean": v_boot["acc_mean"],
            "v_acc_ci": v_boot["acc_ci"],
            "v_acc_str": v_acc_str,
            "v_f1_mean": v_boot["f1_mean"],
            "v_f1_ci": v_boot["f1_ci"],
            "v_f1_str": v_f1_str
        })

    # Save to outputs/bootstrap_confidence_intervals.md
    ci_md_path = out_dir / "bootstrap_confidence_intervals.md"
    with open(ci_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Cluster Bootstrap 95% Confidence Intervals (B={args.b_samples} resamples)\n\n")
        f.write(f"Evaluated on held-out test partitions ($N={N_test_win}$ temporal windows across $K={N_test_vid}$ independent action videos).\n")
        f.write("Window-level uncertainty is evaluated via **Cluster Bootstrap by source video ID**, correctly accounting for intra-video temporal autocorrelation.\n\n")
        f.write("| Architecture | Window Accuracy (Cluster 95% CI) | Window Macro F1 (Cluster 95% CI) | Video Accuracy (95% CI) | Video Macro F1 (95% CI) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for row in bootstrap_table:
            bold = "**" if "SkelGym" in row["name"] else ""
            f.write(f"| {bold}{row['name']}{bold} | {row['w_acc_str']} | {row['w_f1_str']} | {row['v_acc_str']} | {row['v_f1_str']} |\n")

    print(f"\nSaved updated Cluster Bootstrap results to: {ci_md_path}")

    # Save to outputs/statistical_tests_report.md
    stat_md_path = out_dir / "statistical_tests_report.md"
    with open(stat_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Statistical Significance & Hypothesis Testing Report\n\n")
        f.write(f"Evaluated on $N={N_test_win}$ temporal sliding windows and $K={N_test_vid}$ independent action videos.\n\n")
        f.write("## 1. Window-Level McNemar Tests with Multiple Testing Correction\n\n")
        f.write("| Comparison | Model 1 Acc | Model 2 Acc | Delta | Discordant (b/c) | McNemar p (Raw) | Holm-Bonferroni p | FDR p (B-H) | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, (name, res) in enumerate(win_results):
            raw_s = format_p_value(raw_p_win[idx])
            holm_s = format_p_value(p_holm_win[idx])
            fdr_s = format_p_value(p_fdr_win[idx])
            stars = get_significance_stars(p_holm_win[idx])
            f.write(f"| {name} | {res['acc1']:.2f}% | {res['acc2']:.2f}% | {res['delta_acc']:+.2f}% | {res['b']}/{res['c']} | {raw_s} | {holm_s} | {fdr_s} | **{stars}** |\n")

        f.write("\n## 2. Video-Level Exact McNemar Tests (Binary Correctness)\n\n")
        f.write("| Comparison | Video Acc 1 | Video Acc 2 | Delta | Discordant (b/c) | Exact p (Raw) | Holm-Bonferroni p | FDR p (B-H) | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, (name, res) in enumerate(vid_mcnemar_results):
            raw_s = format_p_value(raw_p_vid_mcnemar[idx])
            holm_s = format_p_value(p_holm_vid_mcnemar[idx])
            fdr_s = format_p_value(p_fdr_vid_mcnemar[idx])
            stars = get_significance_stars(p_holm_vid_mcnemar[idx])
            f.write(f"| {name} | {res['acc1']:.2f}% | {res['acc2']:.2f}% | {res['delta_acc']:+.2f}% | {res['b']}/{res['c']} | {raw_s} | {holm_s} | {fdr_s} | **{stars}** |\n")

        f.write("\n## 3. Video-Level Confidence Calibration (Continuous Probability Tests)\n\n")
        f.write("| Comparison | Mean Delta P(True) | Cohen's d_z | Wilcoxon W p (Raw) | Holm-Bonferroni p | Paired t p | Significance |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for idx, (name, res) in enumerate(vid_confidence_results):
            raw_s = format_p_value(res['wilcoxon_p'])
            holm_s = format_p_value(p_holm_vid_wilcoxon[idx])
            ttest_s = format_p_value(res['ttest_p'])
            stars = get_significance_stars(p_holm_vid_wilcoxon[idx])
            f.write(f"| {name} | {res['mean_delta_prob']:+.4f} | {res['cohens_d']:.3f} | {raw_s} | {holm_s} | {ttest_s} | **{stars}** |\n")

    print(f"Saved complete Statistical Tests Report to: {stat_md_path}")

if __name__ == "__main__":
    main()
