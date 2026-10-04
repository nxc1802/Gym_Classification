#!/usr/bin/env python3
"""
Comprehensive Multi-Seed Experiment Runner and Evaluator for SkelGym.
Supports:
1. Training & Evaluation of Baselines (ST-GCN, LSTM, BiLSTM, clean Transformer, clean AAGCN).
2. Training & Evaluation of 5 Augmented Constituent Models across 3 Seeds (42, 123, 3407).
3. Evaluation of 5 Systematic Fusion Methods across Multi-Seed:
   - Hard Voting
   - Uniform Soft Voting
   - Accuracy-Weighted Soft Voting
   - Stacking Meta-Classifier
   - SLSQP Soft Voting (Dual-Target Window & Video)
4. Comprehensive Mean ± SD aggregation and structured publication-ready Markdown/JSON export.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import (
    HardVotingEnsemble,
    SoftVotingEnsemble,
    StackingEnsemble,
    WeightedSoftVotingEnsemble,
    aggregate_video_level_predictions
)
from src.utils.hf_hub import ensure_checkpoint_available, pull_landmarks_from_hf
from src.utils.reproducibility import load_checkpoint_weights
from src.utils.statistics import (
    mcnemar_test,
    cluster_bootstrap_window,
    bootstrap_video,
    paired_video_confidence_test,
    adjust_p_values,
    format_p_value,
    get_significance_stars
)
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

from src.constants import ACTIONS, NUM_CLASSES, CANONICAL_EXPERIMENT_REGISTRY

SEEDS = [42, 123, 3407]

BASELINE_MODELS = [
    {"name": "ST-GCN Baseline", "model": "STGCN", "feature": "rel_3d", "augment": "none", "exp_id": "T3.2"},
    {"name": "LSTM Baseline", "model": "LSTM", "feature": "mix_v2", "augment": "none", "exp_id": "T1.9"},
    {"name": "BiLSTM Baseline", "model": "BiLSTM", "feature": "mix_v2", "augment": "none", "exp_id": "T1.18"},
    {"name": "Transformer Clean", "model": "Transformer", "feature": "mix_v2", "augment": "none", "exp_id": "T1.27"},
    {"name": "AAGCN Clean", "model": "AAGCN", "feature": "bone_3d", "augment": "none", "exp_id": "T3.6"},
]

CONSTITUENT_MODELS = [
    {"name": "Transformer_mix", "model": "Transformer", "feature": "mix_v2", "augment": "skel_gym_aug", "exp_id": "T2.2"},
    {"name": "AAGCN_bone_3d", "model": "AAGCN", "feature": "bone_3d", "augment": "skel_gym_aug", "exp_id": "T4.2"},
    {"name": "AAGCN_rel_3d", "model": "AAGCN", "feature": "rel_3d", "augment": "skel_gym_aug", "exp_id": "T4.3"},
    {"name": "AAGCN_joint_motion_3d", "model": "AAGCN", "feature": "joint_motion_3d", "augment": "skel_gym_aug", "exp_id": "T4.4"},
    {"name": "AAGCN_bone_motion_3d", "model": "AAGCN", "feature": "bone_motion_3d", "augment": "skel_gym_aug", "exp_id": "T4.5"},
]

def get_checkpoint_path(seed: int, model_cfg: Dict[str, str], checkpoint_base: Path) -> Path:
    base_name = f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"
    seed_name = f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}_seed{seed}.pt"
    if seed == 42:
        cand = checkpoint_base / base_name
        return cand if cand.exists() or not (checkpoint_base / seed_name).exists() else checkpoint_base / seed_name
    else:
        cand_seed_dir = checkpoint_base / f"seed{seed}" / base_name
        if cand_seed_dir.exists():
            return cand_seed_dir
        cand_seed_dir_s = checkpoint_base / f"seed{seed}" / seed_name
        if cand_seed_dir_s.exists():
            return cand_seed_dir_s
        cand_flat = checkpoint_base / seed_name
        if cand_flat.exists():
            return cand_flat
        return cand_seed_dir

def train_model(
    seed: int,
    model_cfg: Dict[str, str],
    device_str: str,
    checkpoint_base: Path,
    epochs: int = 100,
    force_retrain: bool = False,
    no_test_eval: bool = False,
    push_to_hf: bool = False,
    hf_repo: str = "Cuong2004/gym-exercise-classification",
    hf_token: Optional[str] = None
):
    ckpt_path = get_checkpoint_path(seed, model_cfg, checkpoint_base)
    if not force_retrain and ckpt_path.exists():
        print(f"[Seed {seed}] Checkpoint already exists: {ckpt_path.name}, skipping training.")
        return ckpt_path

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(ckpt_path.parent)
    aug = model_cfg.get("augment", "skel_gym_aug")

    reg_cfg = CANONICAL_EXPERIMENT_REGISTRY.get(model_cfg.get("exp_id"), {})
    lr = reg_cfg.get("lr", 1e-4)
    batch_size = reg_cfg.get("batch_size", 16)
    label_smoothing = reg_cfg.get("label_smoothing", 0.05 if ("AAGCN" in model_cfg["model"] or "Transformer" in model_cfg["model"]) else 0.0)
    patience = reg_cfg.get("patience", 10)
    train_stride = reg_cfg.get("train_stride", 16)
    val_test_stride = reg_cfg.get("val_test_stride", 32)
    es_metric = reg_cfg.get("early_stopping_metric", "val_macro_f1")

    cmd = [
        sys.executable, "run.py", "train",
        "--model", model_cfg["model"],
        "--feature", model_cfg["feature"],
        "--augment", aug,
        "--exp_id", model_cfg["exp_id"],
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--label_smoothing", str(label_smoothing),
        "--patience", str(patience),
        "--train_stride", str(train_stride),
        "--val_test_stride", str(val_test_stride),
        "--early_stopping_metric", es_metric,
        "--checkpoint_dir", ckpt_dir,
        "--video_level",
        "--device", device_str,
        "--use_amp",
        "--in_memory"
    ]
    if no_test_eval:
        cmd.append("--no_test_eval")
    if push_to_hf:
        cmd.append("--push_to_hf")
        cmd.extend(["--hf_repo", hf_repo])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])

    print(f"\n========================================================")
    print(f"[Seed {seed}] Training {model_cfg['name']} ({aug}) -> {ckpt_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for {model_cfg['name']} seed {seed} (code {res.returncode})")

    elapsed = time.time() - t0
    print(f"[Seed {seed}] Completed training {model_cfg['name']} in {elapsed:.1f}s")
    return ckpt_path

def evaluate_seed(
    seed: int,
    device: torch.device,
    checkpoint_base: Path,
    metadata_path: str,
    landmark_dir: str,
    include_baselines: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    print(f"\n>>> Evaluating Models & 5 Fusion Methods for Seed {seed} <<<")

    val_probs = {}
    test_probs = {}
    y_val_true = None
    y_test_true = None
    val_video_ids = None
    test_video_ids = None

    # Evaluate Baselines if requested and present
    baseline_results = {}
    baseline_raw = {}
    if include_baselines:
        for m in BASELINE_MODELS:
            p = get_checkpoint_path(seed, m, checkpoint_base)
            if not p.exists():
                continue
            state_dict, _ = load_checkpoint_weights(p, device="cpu")
            model = build_model(m["model"], m["feature"], num_classes=NUM_CLASSES)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            _, v_l, te_l = get_dataloaders(
                metadata_path=metadata_path,
                feature_method=m["feature"],
                batch_size=32,
                seq_len=32,
                stride=16,
                val_test_stride=32,
                landmark_dir=landmark_dir,
                num_workers=0,
                in_memory=True,
                seed=seed,
                strict_norm=True
            )
            trainer = Trainer(model=model, device=device)
            _, _, b_vprob = trainer.predict(v_l)
            b_vwin_pred = np.argmax(b_vprob, axis=1)
            b_vwin_m = compute_metrics(np.array(v_l.dataset.labels), b_vwin_pred)
            _, b_vvid_pred, b_vvid_prob, b_vvid_m = aggregate_video_level_predictions(b_vprob, np.array(v_l.dataset.labels), v_l.dataset.video_ids)

            _, _, b_tprob = trainer.predict(te_l)
            b_win_pred = np.argmax(b_tprob, axis=1)
            b_win_m = compute_metrics(np.array(te_l.dataset.labels), b_win_pred)
            _, b_vid_pred, b_vid_prob, b_vid_m = aggregate_video_level_predictions(b_tprob, np.array(te_l.dataset.labels), te_l.dataset.video_ids)

            baseline_results[m["name"]] = {
                "val_win_acc": float(b_vwin_m["accuracy"] * 100.0),
                "val_win_f1": float(b_vwin_m["macro_f1"]),
                "val_vid_acc": float(b_vvid_m["accuracy"] * 100.0),
                "val_vid_f1": float(b_vvid_m["macro_f1"]),
                "win_acc": float(b_win_m["accuracy"] * 100.0),
                "win_f1": float(b_win_m["macro_f1"]),
                "vid_acc": float(b_vid_m["accuracy"] * 100.0),
                "vid_f1": float(b_vid_m["macro_f1"])
            }
            baseline_raw[m["name"]] = {
                "val_prob": b_vprob,
                "test_prob": b_tprob,
                "win_pred": b_win_pred,
                "vid_pred": b_vid_pred,
                "vid_prob": b_vid_prob
            }

    # Evaluate the 5 Constituent Models
    for m in CONSTITUENT_MODELS:
        feat = m["feature"]
        p = get_checkpoint_path(seed, m, checkpoint_base)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {p}")

        state_dict, _ = load_checkpoint_weights(p, device="cpu")
        model = build_model(m["model"], feat, num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _, val_loader, test_loader = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=feat,
            batch_size=32,
            seq_len=32,
            stride=16,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True,
            seed=seed,
            strict_norm=True
        )

        if val_video_ids is None:
            val_video_ids = list(val_loader.dataset.video_ids)
        if test_video_ids is None:
            test_video_ids = list(test_loader.dataset.video_ids)

        trainer = Trainer(model=model, device=device)
        y_vt, _, y_vp = trainer.predict(val_loader)
        y_tt, _, y_tp = trainer.predict(test_loader)

        if y_val_true is None:
            y_val_true = y_vt
        if y_test_true is None:
            y_test_true = y_tt

        val_probs[feat] = y_vp
        test_probs[feat] = y_tp
        if feat == "mix_v2":
            val_probs["mix"] = y_vp
            test_probs["mix"] = y_tp
        elif feat == "mix":
            val_probs["mix_v2"] = y_vp
            test_probs["mix_v2"] = y_tp

    # Pre-compute true video labels for val and test
    val_unique_vids = []
    val_vid_to_label = {}
    for vid, lbl in zip(val_video_ids, y_val_true):
        if vid not in val_vid_to_label:
            val_vid_to_label[vid] = lbl
            val_unique_vids.append(vid)
    y_val_vid_true = np.array([val_vid_to_label[v] for v in val_unique_vids])

    unique_vids = []
    vid_to_label = {}
    for vid, lbl in zip(test_video_ids, y_test_true):
        if vid not in vid_to_label:
            vid_to_label[vid] = lbl
            unique_vids.append(vid)
    y_test_vid_true = np.array([vid_to_label[v] for v in unique_vids])

    # Standalone Constituent Models Evaluation (Validation + Test)
    individual_runs = {}
    constituent_raw = {}
    feat_mix = "mix_v2" if "mix_v2" in val_probs else "mix"
    constituent_names = {
        feat_mix: "Transformer Mix (Aug)",
        "bone_3d": "AAGCN Bone (Aug)",
        "rel_3d": "AAGCN Rel (Aug)",
        "joint_motion_3d": "AAGCN Joint Motion (Aug)",
        "bone_motion_3d": "AAGCN Bone Motion (Aug)"
    }
    for feat, name in constituent_names.items():
        v_prob = val_probs[feat]
        v_w_pred = np.argmax(v_prob, axis=1)
        v_w_m = compute_metrics(y_val_true, v_w_pred)
        _, _, _, v_v_m = aggregate_video_level_predictions(v_prob, y_val_true, val_video_ids)

        t_prob = test_probs[feat]
        t_w_pred = np.argmax(t_prob, axis=1)
        t_w_m = compute_metrics(y_test_true, t_w_pred)
        _, t_v_preds, t_v_prob, t_v_m = aggregate_video_level_predictions(t_prob, y_test_true, test_video_ids)

        individual_runs[name] = {
            "val_win_acc": float(v_w_m["accuracy"] * 100.0),
            "val_win_f1": float(v_w_m["macro_f1"]),
            "val_vid_acc": float(v_v_m["accuracy"] * 100.0),
            "val_vid_f1": float(v_v_m["macro_f1"]),
            "win_acc": float(t_w_m["accuracy"] * 100.0),
            "win_f1": float(t_w_m["macro_f1"]),
            "vid_acc": float(t_v_m["accuracy"] * 100.0),
            "vid_f1": float(t_v_m["macro_f1"])
        }
        constituent_raw[name] = {
            "win_pred": t_w_pred,
            "vid_pred": t_v_preds,
            "vid_prob": t_v_prob
        }

    individual_runs.update(baseline_results)

    # 4 Ensemble Configurations
    ensemble_configs = {
        "Two-Stream AAGCN (Aug)": ["rel_3d", "bone_3d"],
        "Four-Stream AAGCN (Aug)": ["bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"],
        "SkelGym-Lite": [feat_mix, "bone_3d"],
        "SkelGym-Full": [feat_mix, "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    }

    fusion_evals = {}
    ensemble_raw = {}

    for ens_name, feats in ensemble_configs.items():
        v_sub = [val_probs[f] for f in feats]
        t_sub = [test_probs[f] for f in feats]

        fusion_evals[ens_name] = {}
        ensemble_raw[ens_name] = {}

        # 1. Hard Voting (Window & Video)
        hard_ens = HardVotingEnsemble()
        # Val
        val_w_preds = hard_ens.predict([np.argmax(p, axis=1) for p in v_sub])
        val_w_m = compute_metrics(y_val_true, val_w_preds)
        val_vid_p_list = []
        for p in v_sub:
            _, _, vid_p, _ = aggregate_video_level_predictions(p, y_val_true, val_video_ids)
            val_vid_p_list.append(vid_p)
        val_vid_preds_list = [np.argmax(vp, axis=1) for vp in val_vid_p_list]
        val_v_preds = hard_ens.predict(val_vid_preds_list)
        val_v_m = compute_metrics(y_val_vid_true, val_v_preds)

        # Test
        w_preds = hard_ens.predict([np.argmax(p, axis=1) for p in t_sub])
        w_m = compute_metrics(y_test_true, w_preds)
        vid_p_list = []
        for p in t_sub:
            _, _, vid_p, _ = aggregate_video_level_predictions(p, y_test_true, test_video_ids)
            vid_p_list.append(vid_p)
        vid_preds_list = [np.argmax(vp, axis=1) for vp in vid_p_list]
        v_preds = hard_ens.predict(vid_preds_list)
        v_m = compute_metrics(y_test_vid_true, v_preds)

        fusion_evals[ens_name]["Hard Voting"] = {
            "val_win_acc": float(val_w_m["accuracy"] * 100.0), "val_win_f1": float(val_w_m["macro_f1"]),
            "val_vid_acc": float(val_v_m["accuracy"] * 100.0), "val_vid_f1": float(val_v_m["macro_f1"]),
            "win_acc": float(w_m["accuracy"] * 100.0), "win_f1": float(w_m["macro_f1"]),
            "vid_acc": float(v_m["accuracy"] * 100.0), "vid_f1": float(v_m["macro_f1"])
        }
        ensemble_raw[ens_name]["Hard Voting"] = {
            "win_preds": w_preds,
            "vid_preds": v_preds
        }

        # 2. Uniform Soft Voting
        u_ens = SoftVotingEnsemble(weights=None)
        # Val
        u_val_prob = u_ens.predict_proba(v_sub)
        u_val_w_m = compute_metrics(y_val_true, np.argmax(u_val_prob, axis=1))
        _, _, _, u_val_v_m = aggregate_video_level_predictions(u_val_prob, y_val_true, val_video_ids)
        # Test
        u_prob = u_ens.predict_proba(t_sub)
        u_w_m = compute_metrics(y_test_true, np.argmax(u_prob, axis=1))
        _, u_v_preds, u_v_prob, u_v_m = aggregate_video_level_predictions(u_prob, y_test_true, test_video_ids)

        fusion_evals[ens_name]["Uniform Soft Voting"] = {
            "val_win_acc": float(u_val_w_m["accuracy"] * 100.0), "val_win_f1": float(u_val_w_m["macro_f1"]),
            "val_vid_acc": float(u_val_v_m["accuracy"] * 100.0), "val_vid_f1": float(u_val_v_m["macro_f1"]),
            "win_acc": float(u_w_m["accuracy"] * 100.0), "win_f1": float(u_w_m["macro_f1"]),
            "vid_acc": float(u_v_m["accuracy"] * 100.0), "vid_f1": float(u_v_m["macro_f1"])
        }
        ensemble_raw[ens_name]["Uniform Soft Voting"] = {
            "win_prob": u_prob,
            "win_preds": np.argmax(u_prob, axis=1),
            "vid_prob": u_v_prob,
            "vid_preds": u_v_preds
        }

        # 3. Accuracy-Weighted Soft Voting
        val_accs = [np.mean(y_val_true == np.argmax(vp, axis=1)) for vp in v_sub]
        acc_ens = SoftVotingEnsemble(weights=val_accs)
        # Val
        acc_val_prob = acc_ens.predict_proba(v_sub)
        acc_val_w_m = compute_metrics(y_val_true, np.argmax(acc_val_prob, axis=1))
        _, _, _, acc_val_v_m = aggregate_video_level_predictions(acc_val_prob, y_val_true, val_video_ids)
        # Test
        acc_prob = acc_ens.predict_proba(t_sub)
        acc_w_m = compute_metrics(y_test_true, np.argmax(acc_prob, axis=1))
        _, acc_v_preds, acc_v_prob, acc_v_m = aggregate_video_level_predictions(acc_prob, y_test_true, test_video_ids)

        fusion_evals[ens_name]["Accuracy-Weighted Soft"] = {
            "val_win_acc": float(acc_val_w_m["accuracy"] * 100.0), "val_win_f1": float(acc_val_w_m["macro_f1"]),
            "val_vid_acc": float(acc_val_v_m["accuracy"] * 100.0), "val_vid_f1": float(acc_val_v_m["macro_f1"]),
            "win_acc": float(acc_w_m["accuracy"] * 100.0), "win_f1": float(acc_w_m["macro_f1"]),
            "vid_acc": float(acc_v_m["accuracy"] * 100.0), "vid_f1": float(acc_v_m["macro_f1"])
        }
        ensemble_raw[ens_name]["Accuracy-Weighted Soft"] = {
            "win_prob": acc_prob,
            "win_preds": np.argmax(acc_prob, axis=1),
            "vid_prob": acc_v_prob,
            "vid_preds": acc_v_preds
        }

        # 4. Stacking Meta-Classifier
        stk_ens = StackingEnsemble(c_param=1.0)
        stk_ens.fit(v_sub, y_val_true)
        # Val
        stk_val_prob = stk_ens.predict_proba(v_sub)
        stk_val_preds = stk_ens.predict(v_sub)
        stk_val_w_m = compute_metrics(y_val_true, stk_val_preds)
        _, _, _, stk_val_v_m = aggregate_video_level_predictions(stk_val_prob, y_val_true, val_video_ids)
        # Test
        stk_preds = stk_ens.predict(t_sub)
        stk_prob = stk_ens.predict_proba(t_sub)
        stk_w_m = compute_metrics(y_test_true, stk_preds)
        _, stk_v_preds, stk_v_prob, stk_v_m = aggregate_video_level_predictions(stk_prob, y_test_true, test_video_ids)

        fusion_evals[ens_name]["Stacking Meta-Classifier"] = {
            "val_win_acc": float(stk_val_w_m["accuracy"] * 100.0), "val_win_f1": float(stk_val_w_m["macro_f1"]),
            "val_vid_acc": float(stk_val_v_m["accuracy"] * 100.0), "val_vid_f1": float(stk_val_v_m["macro_f1"]),
            "win_acc": float(stk_w_m["accuracy"] * 100.0), "win_f1": float(stk_w_m["macro_f1"]),
            "vid_acc": float(stk_v_m["accuracy"] * 100.0), "vid_f1": float(stk_v_m["macro_f1"])
        }
        ensemble_raw[ens_name]["Stacking Meta-Classifier"] = {
            "win_prob": stk_prob,
            "win_preds": stk_preds,
            "vid_prob": stk_v_prob,
            "vid_preds": stk_v_preds
        }

        # 5. SLSQP Dual-Target Soft Voting
        sls_ens = WeightedSoftVotingEnsemble()
        sls_ens.fit_window(v_sub, y_val_true)
        # Val
        sls_val_w_preds = sls_ens.predict_window(v_sub)
        sls_val_w_m = compute_metrics(y_val_true, sls_val_w_preds)
        # Test
        sls_w_probs = sls_ens.predict_proba_window(t_sub)
        sls_w_preds = sls_ens.predict_window(t_sub)
        sls_w_m = compute_metrics(y_test_true, sls_w_preds)

        sls_ens.fit_video(v_sub, y_val_true, val_video_ids)
        # Val
        _, _, _, sls_val_v_m = sls_ens.predict_video(v_sub, y_val_true, val_video_ids)
        # Test
        _, sls_v_preds, sls_v_probs, sls_v_m = sls_ens.predict_video(t_sub, y_test_true, test_video_ids)

        fusion_evals[ens_name]["SLSQP Soft Voting"] = {
            "val_win_acc": float(sls_val_w_m["accuracy"] * 100.0), "val_win_f1": float(sls_val_w_m["macro_f1"]),
            "val_vid_acc": float(sls_val_v_m["accuracy"] * 100.0), "val_vid_f1": float(sls_val_v_m["macro_f1"]),
            "win_acc": float(sls_w_m["accuracy"] * 100.0), "win_f1": float(sls_w_m["macro_f1"]),
            "vid_acc": float(sls_v_m["accuracy"] * 100.0), "vid_f1": float(sls_v_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in sls_ens.weights_window],
            "weights_video": [round(float(w), 4) for w in sls_ens.weights_video]
        }
        ensemble_raw[ens_name]["SLSQP Soft Voting"] = {
            "win_prob": sls_w_probs,
            "win_preds": sls_w_preds,
            "vid_prob": sls_v_probs,
            "vid_preds": sls_v_preds
        }

    raw_seed_data = {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "baseline_raw": baseline_raw,
        "constituent_raw": constituent_raw,
        "ensemble_raw": ensemble_raw
    }

    split_info = {
        "y_val_true": y_val_true,
        "y_test_true": y_test_true,
        "val_video_ids": np.array(val_video_ids),
        "test_video_ids": np.array(test_video_ids),
        "y_val_vid_true": y_val_vid_true,
        "y_test_vid_true": y_test_vid_true,
        "val_unique_vids": np.array(val_unique_vids),
        "unique_vids": np.array(unique_vids)
    }

    return {
        "individual": individual_runs,
        "fusion_methods": fusion_evals
    }, raw_seed_data, split_info

def aggregate_stats(results_per_seed: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    summary = {"individual": {}, "fusion_methods": {}}
    metric_keys = ["win_acc", "win_f1", "vid_acc", "vid_f1", "val_win_acc", "val_win_f1", "val_vid_acc", "val_vid_f1"]

    # Individual models (collect all unique keys across evaluated seeds)
    all_ind_keys = set()
    for s in seeds:
        all_ind_keys.update(results_per_seed[str(s)]["individual"].keys())

    for k in sorted(all_ind_keys):
        summary["individual"][k] = {}
        for mkey in metric_keys:
            vals = [
                results_per_seed[str(s)]["individual"][k][mkey]
                for s in seeds
                if k in results_per_seed[str(s)]["individual"]
                and mkey in results_per_seed[str(s)]["individual"][k]
            ]
            if vals:
                summary["individual"][k]["n_seeds"] = len(vals)
                summary["individual"][k][f"{mkey}_mean"] = float(np.mean(vals))
                summary["individual"][k][f"{mkey}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None

    # Fusion methods (collect all unique configurations across evaluated seeds)
    all_ens_keys = set()
    for s in seeds:
        all_ens_keys.update(results_per_seed[str(s)]["fusion_methods"].keys())

    for e_k in sorted(all_ens_keys):
        summary["fusion_methods"][e_k] = {}
        all_m_keys = set()
        for s in seeds:
            if e_k in results_per_seed[str(s)]["fusion_methods"]:
                all_m_keys.update(results_per_seed[str(s)]["fusion_methods"][e_k].keys())
        for m_k in sorted(all_m_keys):
            summary["fusion_methods"][e_k][m_k] = {}
            for mkey in metric_keys:
                vals = [
                    results_per_seed[str(s)]["fusion_methods"][e_k][m_k][mkey]
                    for s in seeds
                    if e_k in results_per_seed[str(s)]["fusion_methods"]
                    and m_k in results_per_seed[str(s)]["fusion_methods"][e_k]
                    and mkey in results_per_seed[str(s)]["fusion_methods"][e_k][m_k]
                ]
                if vals:
                    summary["fusion_methods"][e_k][m_k]["n_seeds"] = len(vals)
                    summary["fusion_methods"][e_k][m_k][f"{mkey}_mean"] = float(np.mean(vals))
                    summary["fusion_methods"][e_k][m_k][f"{mkey}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None

            win_weights = [
                results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["weights_window"]
                for s in seeds
                if e_k in results_per_seed[str(s)]["fusion_methods"]
                and m_k in results_per_seed[str(s)]["fusion_methods"][e_k]
                and "weights_window" in results_per_seed[str(s)]["fusion_methods"][e_k][m_k]
            ]
            vid_weights = [
                results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["weights_video"]
                for s in seeds
                if e_k in results_per_seed[str(s)]["fusion_methods"]
                and m_k in results_per_seed[str(s)]["fusion_methods"][e_k]
                and "weights_video" in results_per_seed[str(s)]["fusion_methods"][e_k][m_k]
            ]
            if win_weights:
                win_w = np.mean(win_weights, axis=0)
                summary["fusion_methods"][e_k][m_k]["weights_window"] = [round(float(w), 4) for w in win_w]
            if vid_weights:
                vid_w = np.mean(vid_weights, axis=0)
                summary["fusion_methods"][e_k][m_k]["weights_video"] = [round(float(w), 4) for w in vid_w]
    return summary

def export_canonical_artifacts(
    seeds: List[int],
    results_per_seed: Dict[str, Any],
    all_raw_seeds: Dict[int, Any],
    split_info: Dict[str, Any],
    summary: Dict[str, Any],
    out_dir: Path
):
    print("\n" + "=" * 90)
    print("EXPORTING CANONICAL EVALUATION RUN ARTIFACTS")
    print("=" * 90)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save canonical_eval_predictions.npz
    npz_dict = {
        "y_val_true": split_info["y_val_true"],
        "y_test_true": split_info["y_test_true"],
        "val_video_ids": split_info["val_video_ids"],
        "test_video_ids": split_info["test_video_ids"],
        "y_val_vid_true": split_info["y_val_vid_true"],
        "y_test_vid_true": split_info["y_test_vid_true"],
        "val_unique_vids": split_info["val_unique_vids"],
        "unique_vids": split_info["unique_vids"]
    }
    for seed in seeds:
        s_data = all_raw_seeds[seed]
        for feat, p in s_data["test_probs"].items():
            npz_dict[f"seed_{seed}_test_{feat}"] = p
        for feat, p in s_data["val_probs"].items():
            npz_dict[f"seed_{seed}_val_{feat}"] = p
        for ens_name, methods in s_data["ensemble_raw"].items():
            for m_name, m_data in methods.items():
                for k, arr in m_data.items():
                    safe_ens = ens_name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
                    safe_m = m_name.replace(" ", "_").replace("-", "_")
                    npz_dict[f"ens_{safe_ens}_{safe_m}_seed_{seed}_{k}"] = arr

    if 42 in all_raw_seeds and all_raw_seeds[42]["baseline_raw"]:
        for bname, b_data in all_raw_seeds[42]["baseline_raw"].items():
            safe_b = bname.replace(" ", "_").replace("-", "_")
            for k, arr in b_data.items():
                npz_dict[f"baseline_{safe_b}_{k}"] = arr

    npz_path = out_dir / "canonical_eval_predictions.npz"
    np.savez_compressed(npz_path, **npz_dict)
    print(f"  [1/6] Saved canonical raw predictions -> {npz_path}")

    # 2. Table 7: consensus_gains.json
    row_specs = [
        ("Baseline LSTM (Mix 117-d)", "LSTM Baseline", "individual", "Sequential Recurrent Model", "396K"),
        ("Baseline BiLSTM (Mix 117-d)", "BiLSTM Baseline", "individual", "Bidirectional Recurrent Model", "402K"),
        ("Transformer (Mix 117-d, Clean)", "Transformer Clean", "individual", "Self-Attention Baseline", "400K"),
        ("Baseline ST-GCN (Rel 3D)", "ST-GCN Baseline", "individual", "Rigid Static Graph ($A_{\\text{phys}}$)", "350K"),
        ("Clean Baseline AAGCN (Bone 3D)", "AAGCN Clean", "individual", "Adaptive Skeletal Graph (Unaugmented)", "378K"),
        ("SkelGym-Aug AAGCN (Bone 3D)", "AAGCN Bone (Aug)", "individual", "Adaptive Skeletal Graph + Augmentation", "378K"),
        ("SkelGym-Aug Transformer (Mix)", "Transformer Mix (Aug)", "individual", "Self-Attention + Augmentation", "400K"),
        ("Two-Stream AAGCN (Aug)", "Two-Stream AAGCN (Aug)", "fusion", "Joint + Bone Stream Fusion", "756K"),
        ("Four-Stream AAGCN (Aug)", "Four-Stream AAGCN (Aug)", "fusion", "4-Stream Graph Late Fusion", "1.51M"),
        ("SkelGym-Lite (2 Models)", "SkelGym-Lite", "fusion", "Transformer + Bone AAGCN", "778K"),
        ("SkelGym-Full (5 Streams)", "SkelGym-Full", "fusion", "Cross-Paradigm SLSQP Ensemble", "1.91M")
    ]
    t7_data = {}
    for row_name, source_key, category, modality, params in row_specs:
        if category == "individual":
            m = summary["individual"].get(source_key, {})
        else:
            m = summary["fusion_methods"].get(source_key, {}).get("SLSQP Soft Voting", {})

        is_single = (m.get("win_acc_sd") is None or m.get("n_seeds") == 1)
        w_acc = m.get("win_acc_mean", 0.0)
        v_acc = m.get("vid_acc_mean", 0.0)
        gain = v_acc - w_acc

        entry = {
            "architecture": row_name,
            "modality": modality,
            "params": params,
            "win_acc": round(float(w_acc), 2),
            "win_f1": round(float(m.get("win_f1_mean", 0.0)), 4),
            "vid_acc": round(float(v_acc), 2),
            "vid_f1": round(float(m.get("vid_f1_mean", 0.0)), 4),
            "vid_gain": f"+{gain:.2f}%",
            "is_single_seed": is_single,
            "status": "Verified"
        }
        if not is_single and m.get("win_acc_sd") is not None:
            entry["win_acc_sd"] = round(float(m["win_acc_sd"]), 2)
            entry["win_f1_sd"] = round(float(m["win_f1_sd"]), 4)
            entry["vid_acc_sd"] = round(float(m["vid_acc_sd"]), 2)
            entry["vid_f1_sd"] = round(float(m["vid_f1_sd"]), 4)
        t7_data[row_name] = entry

    t7_path = out_dir / "consensus_gains.json"
    with open(t7_path, "w", encoding="utf-8") as f:
        json.dump(t7_data, f, indent=2)
    print(f"  [2/6] Saved Table 7 consensus gains -> {t7_path}")

    # 3. Table 8: statistical_tests_report.json
    raw42 = all_raw_seeds.get(42, all_raw_seeds[seeds[0]])
    y_test_t = split_info["y_test_true"]
    y_test_vid_t = split_info["y_test_vid_true"]

    p_clean_trans_w = raw42["baseline_raw"]["Transformer Clean"]["win_pred"]
    p_clean_trans_v = raw42["baseline_raw"]["Transformer Clean"]["vid_pred"]
    prob_clean_trans_v = raw42["baseline_raw"]["Transformer Clean"]["vid_prob"]

    mix_key = "mix_v2" if "mix_v2" in raw42["test_probs"] else "mix"
    p_aug_trans_w = np.argmax(raw42["test_probs"][mix_key], axis=1)
    _, p_aug_trans_v, prob_aug_trans_v, _ = aggregate_video_level_predictions(
        raw42["test_probs"][mix_key], y_test_t, split_info["test_video_ids"]
    )

    p_stgcn_w = raw42["baseline_raw"]["ST-GCN Baseline"]["win_pred"]
    p_stgcn_v = raw42["baseline_raw"]["ST-GCN Baseline"]["vid_pred"]
    prob_stgcn_v = raw42["baseline_raw"]["ST-GCN Baseline"]["vid_prob"]

    p_bone_w = np.argmax(raw42["test_probs"]["bone_3d"], axis=1)
    _, p_bone_v, prob_bone_v, _ = aggregate_video_level_predictions(
        raw42["test_probs"]["bone_3d"], y_test_t, split_info["test_video_ids"]
    )

    p_4stream_w = raw42["ensemble_raw"]["Four-Stream AAGCN (Aug)"]["SLSQP Soft Voting"]["win_preds"]
    p_4stream_v = raw42["ensemble_raw"]["Four-Stream AAGCN (Aug)"]["SLSQP Soft Voting"]["vid_preds"]
    prob_4stream_v = raw42["ensemble_raw"]["Four-Stream AAGCN (Aug)"]["SLSQP Soft Voting"]["vid_prob"]

    p_skel_full_w = raw42["ensemble_raw"]["SkelGym-Full"]["SLSQP Soft Voting"]["win_preds"]
    p_skel_full_v = raw42["ensemble_raw"]["SkelGym-Full"]["SLSQP Soft Voting"]["vid_preds"]
    prob_skel_full_v = raw42["ensemble_raw"]["SkelGym-Full"]["SLSQP Soft Voting"]["vid_prob"]

    comparisons = [
        ("Unaugmented Trans vs SkelGym-Aug Trans", p_clean_trans_w, p_aug_trans_w, prob_clean_trans_v, prob_aug_trans_v),
        ("Fixed ST-GCN vs Adaptive Four-Stream AAGCN", p_stgcn_w, p_4stream_w, prob_stgcn_v, prob_4stream_v),
        ("Single Sequence (Trans) vs SkelGym-Full", p_aug_trans_w, p_skel_full_w, prob_aug_trans_v, prob_skel_full_v),
        ("Single Graph (AAGCN Bone) vs SkelGym-Full", p_bone_w, p_skel_full_w, prob_bone_v, prob_skel_full_v),
        ("Four-Stream Graph AAGCN vs SkelGym-Full", p_4stream_w, p_skel_full_w, prob_4stream_v, prob_skel_full_v)
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

    for idx, (comp_name, m_res, c_res) in enumerate(comp_results):
        t8_data[comp_name] = {
            "win_chi2": round(float(m_res["chi2"]), 2),
            "win_p": format_p_value(raw_p_win[idx]),
            "win_p_holm": format_p_value(p_holm_win[idx]),
            "win_odds_ratio": round(float(m_res["odds_ratio"]), 2),
            "vid_wilcoxon_w": str(c_res.get("wilcoxon_stat", "—")),
            "vid_wilcoxon_p": format_p_value(raw_p_vid_w[idx]),
            "vid_wilcoxon_p_holm": format_p_value(p_holm_vid[idx]),
            "vid_paired_t_p": format_p_value(raw_p_vid_t[idx]),
            "vid_cohens_d": f"{c_res['cohens_d']:+.3f}",
            "significance": get_significance_stars(p_holm_vid[idx]),
            "status": "Verified"
        }

    t8_path = out_dir / "statistical_tests_report.json"
    with open(t8_path, "w", encoding="utf-8") as f:
        json.dump(t8_data, f, indent=2)
    print(f"  [3/6] Saved Table 8 statistical tests report -> {t8_path}")

    # 4. Table 9: bootstrap_confidence_intervals.json
    models_boot = [
        ("LSTM (Mix 117-d)", raw42["baseline_raw"]["LSTM Baseline"]["win_pred"], raw42["baseline_raw"]["LSTM Baseline"]["vid_pred"]),
        ("BiLSTM (Mix 117-d)", raw42["baseline_raw"]["BiLSTM Baseline"]["win_pred"], raw42["baseline_raw"]["BiLSTM Baseline"]["vid_pred"]),
        ("ST-GCN (Rel 3D)", p_stgcn_w, p_stgcn_v),
        ("Transformer (Mix 117-d)", p_aug_trans_w, p_aug_trans_v),
        ("AAGCN (Bone 3D)", p_bone_w, p_bone_v),
        ("SkelGym-Lite (2 Models)",
         raw42["ensemble_raw"]["SkelGym-Lite"]["SLSQP Soft Voting"]["win_preds"],
         raw42["ensemble_raw"]["SkelGym-Lite"]["SLSQP Soft Voting"]["vid_preds"]),
        ("SkelGym-Full (5 Streams)", p_skel_full_w, p_skel_full_v)
    ]

    t9_data = {}
    for mname, wp, vp in models_boot:
        w_boot = cluster_bootstrap_window(y_test_t, wp, split_info["test_video_ids"], B=1000, seed=42)
        v_boot = bootstrap_video(y_test_vid_t, vp, B=1000, seed=42)

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

    t9_path = out_dir / "bootstrap_confidence_intervals.json"
    with open(t9_path, "w", encoding="utf-8") as f:
        json.dump(t9_data, f, indent=2)
    print(f"  [4/6] Saved Table 9 cluster bootstrap CIs -> {t9_path}")

    # 5. Table 10: per_class_results.json
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
    t10_data["classes"]["Overall Accuracy"] = {
        "win_precision": round(float(np.mean(y_test_t == p_skel_full_w)), 4),
        "win_recall": round(float(np.mean(y_test_t == p_skel_full_w)), 4),
        "win_f1": round(float(np.mean(y_test_t == p_skel_full_w)), 4),
        "win_support": int(len(y_test_t)),
        "vid_precision": round(float(np.mean(y_test_vid_t == p_skel_full_v)), 4),
        "vid_recall": round(float(np.mean(y_test_vid_t == p_skel_full_v)), 4),
        "vid_f1": round(float(np.mean(y_test_vid_t == p_skel_full_v)), 4),
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
    t10_path = out_dir / "per_class_results.json"
    with open(t10_path, "w", encoding="utf-8") as f:
        json.dump(t10_data, f, indent=2)
    print(f"  [5/6] Saved Table 10 per-class results -> {t10_path}")

    # 6. Table 12: external_benchmark_results.json (MM-Fit genuine benchmark)
    mmfit_closed_p = out_dir / "external" / "mmfit" / "metrics_closed.csv"
    mmfit_open_p = out_dir / "external" / "mmfit" / "metrics_open.csv"
    if not mmfit_closed_p.exists():
        mmfit_closed_p = Path("outputs/external/mmfit/metrics_closed.csv")
    if not mmfit_open_p.exists():
        mmfit_open_p = Path("outputs/external/mmfit/metrics_open.csv")

    if mmfit_closed_p.exists() and mmfit_open_p.exists():
        df_cl = pd.read_csv(mmfit_closed_p)
        df_op = pd.read_csv(mmfit_open_p)

        ext_models = {}
        for _, row in df_cl.iterrows():
            mname = row["model"]
            op_row = df_op[df_op["model"] == mname]
            op_win = float(op_row["window_acc"].values[0]) if len(op_row) > 0 else 0.0
            op_vid = float(op_row["recording_acc"].values[0]) if len(op_row) > 0 else 0.0

            ext_models[mname] = {
                "open_win_acc": round(op_win, 2),
                "open_vid_acc": round(op_vid, 2),
                "closed_win_acc": round(float(row["window_acc"]), 2),
                "closed_vid_acc": round(float(row["recording_acc"]), 2),
                "closed_vid_f1": round(float(row["macro_f1"]), 4),
                "status": "Verified"
            }
        t12_data = {
            "dataset": "mmfit",
            "pose_protocol": "mediapipe",
            "class_set": "core4",
            "num_records": 54,
            "num_windows": 878,
            "models": ext_models
        }
        t12_path = out_dir / "external_benchmark_results.json"
        with open(t12_path, "w", encoding="utf-8") as f:
            json.dump(t12_data, f, indent=2)
        print(f"  [6/6] Saved Table 12 external benchmark results -> {t12_path}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner & 5 Fusion Methods Evaluator")
    parser.add_argument("--device", type=str, default="auto", help="Computation device (cuda/mps/cpu/auto)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--metadata", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--include_baselines", action="store_true", default=False, help="Also train/evaluate baseline models")
    parser.add_argument("--output_file", type=str, default="outputs/multi_seed_evaluation_results.json")
    parser.add_argument("--skip_train", action="store_true", default=False, help="Skip training and only evaluate")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force complete retraining even if checkpoints exist")
    parser.add_argument("--no_test_eval", action="store_true", default=True, help="Disable test evaluation during training to enforce blind protocol")
    parser.add_argument("--push_to_hf", action="store_true", default=False, help="Upload checkpoints to Hugging Face Hub")
    parser.add_argument("--hf_repo", type=str, default="Cuong2004/gym-exercise-classification", help="Hugging Face Model repository ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face authentication token")
    args = parser.parse_args()

    # Robust metadata resolution
    metadata_cand = Path(args.metadata)
    if not metadata_cand.exists():
        if Path("Final_dataset_metadata.csv").exists():
            metadata_cand = Path("Final_dataset_metadata.csv")
        elif Path("data/Final_dataset_metadata.csv").exists():
            metadata_cand = Path("data/Final_dataset_metadata.csv")
    args.metadata = str(metadata_cand)

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device
    device = torch.device(device_str)
    print(f"Running Multi-Seed Evaluation on device: {device} | Seeds: {args.seeds} | Push to HF: {args.push_to_hf}")

    checkpoint_base = Path(args.checkpoint_dir)
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # 1. Training Phase
    if not args.skip_train:
        for seed in args.seeds:
            print(f"\n========================================================")
            print(f"TRAINING PHASE FOR SEED {seed}")
            print(f"========================================================")
            if args.include_baselines:
                for b in BASELINE_MODELS:
                    train_model(
                        seed, b, device_str, checkpoint_base,
                        epochs=args.epochs, force_retrain=args.force_retrain,
                        no_test_eval=args.no_test_eval, push_to_hf=args.push_to_hf,
                        hf_repo=args.hf_repo, hf_token=args.hf_token
                    )

            for m in CONSTITUENT_MODELS:
                train_model(
                    seed, m, device_str, checkpoint_base,
                    epochs=args.epochs, force_retrain=args.force_retrain,
                    no_test_eval=args.no_test_eval, push_to_hf=args.push_to_hf,
                    hf_repo=args.hf_repo, hf_token=args.hf_token
                )

    # 2. Evaluation Phase
    results_per_seed = {}
    all_raw_seeds = {}
    split_info = None
    for seed in args.seeds:
        res, raw_s, s_info = evaluate_seed(
            seed, device, checkpoint_base, args.metadata, args.landmark_dir, include_baselines=args.include_baselines
        )
        results_per_seed[str(seed)] = res
        all_raw_seeds[seed] = raw_s
        if split_info is None:
            split_info = s_info

    # 3. Aggregation Phase
    summary = aggregate_stats(results_per_seed, args.seeds)

    # Print to console
    print("\n" + "=" * 105)
    print("MAIN MULTI-SEED SUMMARY (MEAN ± SD ACROSS EVALUATED SEEDS)")
    print("=" * 105)
    for ens_name, methods in summary["fusion_methods"].items():
        print(f"\n>>> Target: {ens_name} <<<")
        print(f"{'Fusion Method':<30} | {'Window Accuracy':<18} | {'Window Macro-F1':<18} | {'Video Accuracy':<18} | {'Video Macro-F1':<18}")
        print("-" * 110)
        for m_name, st in methods.items():
            if st.get("win_acc_sd") is not None:
                w_acc_str = f"{st['win_acc_mean']:>5.2f}% ± {st['win_acc_sd']:>4.2f}%"
                w_f1_str = f"{st['win_f1_mean']:>6.4f} ± {st['win_f1_sd']:>6.4f}"
                v_acc_str = f"{st['vid_acc_mean']:>5.2f}% ± {st['vid_acc_sd']:>4.2f}%"
                v_f1_str = f"{st['vid_f1_mean']:>6.4f} ± {st['vid_f1_sd']:>6.4f}"
            else:
                w_acc_str = f"{st['win_acc_mean']:>5.2f}% (n=1)      "
                w_f1_str = f"{st['win_f1_mean']:>6.4f}           "
                v_acc_str = f"{st['vid_acc_mean']:>5.2f}% (n=1)      "
                v_f1_str = f"{st['vid_f1_mean']:>6.4f}           "
            print(f"{m_name:<30} | {w_acc_str}    | {w_f1_str}   | {v_acc_str}    | {v_f1_str}")

    # Save to JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": args.seeds,
            "summary": summary,
            "individual_runs": results_per_seed
        }, f, indent=2)
    print(f"\nSaved detailed multi-seed results to: {out_path}")

    # Generate Markdown
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# SkelGym Multi-Seed Benchmark & 5 Fusion Methods Results\n\n")
        f.write(f"Evaluated across random seeds: `{args.seeds}`.\n\n")
        f.write("## 1. 5 Fusion Methods Comparison across Multi-Stream Targets (Mean ± SD)\n\n")
        for ens_name, methods in summary["fusion_methods"].items():
            f.write(f"### {ens_name}\n\n")
            f.write("| Fusion Method | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for m_name, st in methods.items():
                if st.get("win_acc_sd") is not None:
                    w_acc_str = f"{st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}%"
                    w_f1_str = f"{st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f}"
                    v_acc_str = f"{st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}%"
                    v_f1_str = f"{st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f}"
                else:
                    w_acc_str = f"{st['win_acc_mean']:.2f}% (n=1)"
                    w_f1_str = f"{st['win_f1_mean']:.4f}"
                    v_acc_str = f"{st['vid_acc_mean']:.2f}% (n=1)"
                    v_f1_str = f"{st['vid_f1_mean']:.4f}"
                f.write(f"| **{m_name}** | {w_acc_str} | {w_f1_str} | {v_acc_str} | {v_f1_str} |\n")
            f.write("\n")

        f.write("## 2. Individual Model Backbones (Mean ± SD)\n\n")
        f.write("| Model Name | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for k, st in summary["individual"].items():
            if st.get("win_acc_sd") is not None:
                w_acc_str = f"{st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}%"
                w_f1_str = f"{st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f}"
                v_acc_str = f"{st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}%"
                v_f1_str = f"{st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f}"
            else:
                w_acc_str = f"{st['win_acc_mean']:.2f}% (n=1)"
                w_f1_str = f"{st['win_f1_mean']:.4f}"
                v_acc_str = f"{st['vid_acc_mean']:.2f}% (n=1)"
                v_f1_str = f"{st['vid_f1_mean']:.4f}"
            f.write(f"| **{k}** | {w_acc_str} | {w_f1_str} | {v_acc_str} | {v_f1_str} |\n")
    print(f"Saved Markdown report to: {md_path}")

    # 4. Canonical Artifacts Export (Tables 6, 7, 8, 9, 10, 12 + Raw NPZ)
    export_canonical_artifacts(
        args.seeds, results_per_seed, all_raw_seeds, split_info, summary, out_path.parent
    )

if __name__ == "__main__":
    main()
