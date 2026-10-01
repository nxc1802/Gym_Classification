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
from typing import Dict, List, Any, Optional
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

from src.constants import ACTIONS, NUM_CLASSES, CANONICAL_EXPERIMENT_REGISTRY

SEEDS = [42, 123, 3407]

BASELINE_MODELS = [
    {"name": "ST-GCN Baseline", "model": "STGCN", "feature": "rel_3d", "augment": "none", "exp_id": "T3.2"},
    {"name": "LSTM Baseline", "model": "LSTM", "feature": "mix", "augment": "none", "exp_id": "T1.9"},
    {"name": "BiLSTM Baseline", "model": "BiLSTM", "feature": "mix", "augment": "none", "exp_id": "T1.18"},
    {"name": "Transformer Clean", "model": "Transformer", "feature": "mix", "augment": "none", "exp_id": "T1.27"},
    {"name": "AAGCN Clean", "model": "AAGCN", "feature": "bone_3d", "augment": "none", "exp_id": "T3.6"},
]

CONSTITUENT_MODELS = [
    {"name": "Transformer_mix", "model": "Transformer", "feature": "mix", "augment": "skel_gym_aug", "exp_id": "T2.2"},
    {"name": "AAGCN_bone_3d", "model": "AAGCN", "feature": "bone_3d", "augment": "skel_gym_aug", "exp_id": "T4.2"},
    {"name": "AAGCN_rel_3d", "model": "AAGCN", "feature": "rel_3d", "augment": "skel_gym_aug", "exp_id": "T4.3"},
    {"name": "AAGCN_joint_motion_3d", "model": "AAGCN", "feature": "joint_motion_3d", "augment": "skel_gym_aug", "exp_id": "T4.4"},
    {"name": "AAGCN_bone_motion_3d", "model": "AAGCN", "feature": "bone_motion_3d", "augment": "skel_gym_aug", "exp_id": "T4.5"},
]

def get_checkpoint_path(seed: int, model_cfg: Dict[str, str], checkpoint_base: Path) -> Path:
    base_name = f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"
    if seed == 42:
        return checkpoint_base / base_name
    else:
        return checkpoint_base / f"seed{seed}" / base_name

def train_model(seed: int, model_cfg: Dict[str, str], device_str: str, checkpoint_base: Path, epochs: int = 100, force_retrain: bool = False, no_test_eval: bool = False):
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
) -> Dict[str, Any]:
    print(f"\n>>> Evaluating Models & 5 Fusion Methods for Seed {seed} <<<")

    val_probs = {}
    test_probs = {}
    y_val_true = None
    y_test_true = None
    val_video_ids = None
    test_video_ids = None

    # Evaluate Baselines if requested and present
    baseline_results = {}
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
                seed=seed
            )
            trainer = Trainer(model=model, device=device)
            _, _, b_tprob = trainer.predict(te_l)
            b_win_pred = np.argmax(b_tprob, axis=1)
            b_win_m = compute_metrics(np.array(te_l.dataset.labels), b_win_pred)
            _, _, _, b_vid_m = aggregate_video_level_predictions(b_tprob, np.array(te_l.dataset.labels), te_l.dataset.video_ids)

            baseline_results[m["name"]] = {
                "win_acc": float(b_win_m["accuracy"] * 100.0),
                "win_f1": float(b_win_m["macro_f1"]),
                "vid_acc": float(b_vid_m["accuracy"] * 100.0),
                "vid_f1": float(b_vid_m["macro_f1"])
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
            seed=seed
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

    # Pre-compute true video labels
    unique_vids = []
    vid_to_label = {}
    for vid, lbl in zip(test_video_ids, y_test_true):
        if vid not in vid_to_label:
            vid_to_label[vid] = lbl
            unique_vids.append(vid)
    y_test_vid_true = np.array([vid_to_label[v] for v in unique_vids])

    # Standalone Augmented Transformer
    t_test_prob = test_probs["mix"]
    t_win_pred = np.argmax(t_test_prob, axis=1)
    t_win_metrics = compute_metrics(y_test_true, t_win_pred)
    _, _, _, t_vid_metrics = aggregate_video_level_predictions(t_test_prob, y_test_true, test_video_ids)

    individual_runs = {
        "Transformer Mix (Aug)": {
            "win_acc": float(t_win_metrics["accuracy"] * 100.0),
            "win_f1": float(t_win_metrics["macro_f1"]),
            "vid_acc": float(t_vid_metrics["accuracy"] * 100.0),
            "vid_f1": float(t_vid_metrics["macro_f1"])
        }
    }
    individual_runs.update(baseline_results)

    # 4 Ensemble Configurations
    ensemble_configs = {
        "Two-Stream AAGCN (Aug)": ["rel_3d", "bone_3d"],
        "Four-Stream AAGCN (Aug)": ["bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"],
        "SkelGym-Lite": ["mix", "bone_3d"],
        "SkelGym-Full": ["mix", "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    }

    fusion_evals = {}

    for ens_name, feats in ensemble_configs.items():
        v_sub = [val_probs[f] for f in feats]
        t_sub = [test_probs[f] for f in feats]

        fusion_evals[ens_name] = {}

        # 1. Hard Voting
        hard_ens = HardVotingEnsemble()
        w_preds = hard_ens.predict([np.argmax(p, axis=1) for p in t_sub])
        w_m = compute_metrics(y_test_true, w_preds)
        vid_p_list = []
        for p in t_sub:
            _, _, vid_p, _ = aggregate_video_level_predictions(p, y_test_true, test_video_ids)
            vid_p_list.append(vid_p)
        v_preds = hard_ens.predict(vid_p_list)
        v_m = compute_metrics(y_test_vid_true, v_preds)
        fusion_evals[ens_name]["Hard Voting"] = {
            "win_acc": float(w_m["accuracy"] * 100.0), "win_f1": float(w_m["macro_f1"]),
            "vid_acc": float(v_m["accuracy"] * 100.0), "vid_f1": float(v_m["macro_f1"])
        }

        # 2. Uniform Soft Voting
        u_ens = SoftVotingEnsemble(weights=None)
        u_prob = u_ens.predict_proba(t_sub)
        u_w_m = compute_metrics(y_test_true, np.argmax(u_prob, axis=1))
        _, _, _, u_v_m = aggregate_video_level_predictions(u_prob, y_test_true, test_video_ids)
        fusion_evals[ens_name]["Uniform Soft Voting"] = {
            "win_acc": float(u_w_m["accuracy"] * 100.0), "win_f1": float(u_w_m["macro_f1"]),
            "vid_acc": float(u_v_m["accuracy"] * 100.0), "vid_f1": float(u_v_m["macro_f1"])
        }

        # 3. Accuracy-Weighted Soft Voting
        val_accs = [np.mean(y_val_true == np.argmax(vp, axis=1)) for vp in v_sub]
        acc_ens = SoftVotingEnsemble(weights=val_accs)
        acc_prob = acc_ens.predict_proba(t_sub)
        acc_w_m = compute_metrics(y_test_true, np.argmax(acc_prob, axis=1))
        _, _, _, acc_v_m = aggregate_video_level_predictions(acc_prob, y_test_true, test_video_ids)
        fusion_evals[ens_name]["Accuracy-Weighted Soft"] = {
            "win_acc": float(acc_w_m["accuracy"] * 100.0), "win_f1": float(acc_w_m["macro_f1"]),
            "vid_acc": float(acc_v_m["accuracy"] * 100.0), "vid_f1": float(acc_v_m["macro_f1"])
        }

        # 4. Stacking Meta-Classifier
        stk_ens = StackingEnsemble(c_param=1.0)
        stk_ens.fit(v_sub, y_val_true)
        stk_preds = stk_ens.predict(t_sub)
        stk_prob = stk_ens.predict_proba(t_sub)
        stk_w_m = compute_metrics(y_test_true, stk_preds)
        _, _, _, stk_v_m = aggregate_video_level_predictions(stk_prob, y_test_true, test_video_ids)
        fusion_evals[ens_name]["Stacking Meta-Classifier"] = {
            "win_acc": float(stk_w_m["accuracy"] * 100.0), "win_f1": float(stk_w_m["macro_f1"]),
            "vid_acc": float(stk_v_m["accuracy"] * 100.0), "vid_f1": float(stk_v_m["macro_f1"])
        }

        # 5. SLSQP Dual-Target Soft Voting
        sls_ens = WeightedSoftVotingEnsemble()
        sls_ens.fit_window(v_sub, y_val_true)
        sls_w_preds = sls_ens.predict_window(t_sub)
        sls_w_m = compute_metrics(y_test_true, sls_w_preds)

        sls_ens.fit_video(v_sub, y_val_true, val_video_ids)
        _, _, _, sls_v_m = sls_ens.predict_video(t_sub, y_test_true, test_video_ids)
        fusion_evals[ens_name]["SLSQP Soft Voting"] = {
            "win_acc": float(sls_w_m["accuracy"] * 100.0), "win_f1": float(sls_w_m["macro_f1"]),
            "vid_acc": float(sls_v_m["accuracy"] * 100.0), "vid_f1": float(sls_v_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in sls_ens.weights_window],
            "weights_video": [round(float(w), 4) for w in sls_ens.weights_video]
        }

    return {
        "individual": individual_runs,
        "fusion_methods": fusion_evals
    }

def aggregate_stats(results_per_seed: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    summary = {"individual": {}, "fusion_methods": {}}

    # Individual models
    sample_seed = str(seeds[0])
    ind_keys = list(results_per_seed[sample_seed]["individual"].keys())
    for k in ind_keys:
        w_accs = [results_per_seed[str(s)]["individual"][k]["win_acc"] for s in seeds]
        w_f1s = [results_per_seed[str(s)]["individual"][k]["win_f1"] for s in seeds]
        v_accs = [results_per_seed[str(s)]["individual"][k]["vid_acc"] for s in seeds]
        v_f1s = [results_per_seed[str(s)]["individual"][k]["vid_f1"] for s in seeds]

        summary["individual"][k] = {
            "win_acc_mean": float(np.mean(w_accs)), "win_acc_sd": float(np.std(w_accs, ddof=1)),
            "win_f1_mean": float(np.mean(w_f1s)), "win_f1_sd": float(np.std(w_f1s, ddof=1)),
            "vid_acc_mean": float(np.mean(v_accs)), "vid_acc_sd": float(np.std(v_accs, ddof=1)),
            "vid_f1_mean": float(np.mean(v_f1s)), "vid_f1_sd": float(np.std(v_f1s, ddof=1)),
        }

    # Fusion methods
    ens_keys = list(results_per_seed[sample_seed]["fusion_methods"].keys())
    for e_k in ens_keys:
        summary["fusion_methods"][e_k] = {}
        m_keys = list(results_per_seed[sample_seed]["fusion_methods"][e_k].keys())
        for m_k in m_keys:
            w_accs = [results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["win_acc"] for s in seeds]
            w_f1s = [results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["win_f1"] for s in seeds]
            v_accs = [results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["vid_acc"] for s in seeds]
            v_f1s = [results_per_seed[str(s)]["fusion_methods"][e_k][m_k]["vid_f1"] for s in seeds]

            summary["fusion_methods"][e_k][m_k] = {
                "win_acc_mean": float(np.mean(w_accs)), "win_acc_sd": float(np.std(w_accs, ddof=1)),
                "win_f1_mean": float(np.mean(w_f1s)), "win_f1_sd": float(np.std(w_f1s, ddof=1)),
                "vid_acc_mean": float(np.mean(v_accs)), "vid_acc_sd": float(np.std(v_accs, ddof=1)),
                "vid_f1_mean": float(np.mean(v_f1s)), "vid_f1_sd": float(np.std(v_f1s, ddof=1)),
            }
    return summary

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner & 5 Fusion Methods Evaluator")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/mps/cpu)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--metadata", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--include_baselines", action="store_true", default=False, help="Also train/evaluate baseline models")
    parser.add_argument("--output_file", type=str, default="outputs/multi_seed_evaluation_results.json")
    parser.add_argument("--skip_train", action="store_true", default=False, help="Skip training and only evaluate")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force complete retraining even if checkpoints exist")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running Multi-Seed Evaluation on device: {device} | Seeds: {args.seeds}")

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
                    train_model(seed, b, args.device, checkpoint_base, epochs=args.epochs, force_retrain=args.force_retrain)

            for m in CONSTITUENT_MODELS:
                train_model(seed, m, args.device, checkpoint_base, epochs=args.epochs, force_retrain=args.force_retrain)

    # 2. Evaluation Phase
    results_per_seed = {}
    for seed in args.seeds:
        results_per_seed[str(seed)] = evaluate_seed(
            seed, device, checkpoint_base, args.metadata, args.landmark_dir, include_baselines=args.include_baselines
        )

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
            print(f"{m_name:<30} | {st['win_acc_mean']:>5.2f}% ± {st['win_acc_sd']:>4.2f}%    | {st['win_f1_mean']:>6.4f} ± {st['win_f1_sd']:>6.4f}   | {st['vid_acc_mean']:>5.2f}% ± {st['vid_acc_sd']:>4.2f}%    | {st['vid_f1_mean']:>6.4f} ± {st['vid_f1_sd']:>6.4f}")

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
                f.write(f"| **{m_name}** | {st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}% | {st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f} | {st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}% | {st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f} |\n")
            f.write("\n")

        f.write("## 2. Individual Model Backbones (Mean ± SD)\n\n")
        f.write("| Model Name | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for k, st in summary["individual"].items():
            f.write(f"| **{k}** | {st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}% | {st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f} | {st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}% | {st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f} |\n")
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()
