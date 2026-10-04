#!/usr/bin/env python3
"""
Comprehensive Local Evaluation & Multi-Method Ensemble Benchmarking.
Evaluates:
1. Baseline Models:
   - ST-GCN Baseline (Rel 3D)
   - LSTM Baseline (Mix 117-d)
   - BiLSTM Baseline (Mix 117-d)
   - Transformer Clean (Mix 117-d, no aug)
   - AAGCN Clean (Bone 3D, no aug)
2. Augmented Constituent Models:
   - Transformer Mix + Aug (T2.2)
   - AAGCN Joint (Rel 3D) + Aug (T4.3)
   - AAGCN Bone (Bone 3D) + Aug (T4.2)
   - AAGCN Joint Motion (J-Motion) + Aug (T4.4)
   - AAGCN Bone Motion (B-Motion) + Aug (T4.5)
3. 5 Systematic Late Fusion Methods:
   - Method 1: Hard Voting (Majority Rule)
   - Method 2: Uniform Soft Voting (Arithmetic Mean)
   - Method 3: Accuracy-Weighted Soft Voting (Validation Accuracy Weights)
   - Method 4: Stacking Meta-Classifier (Logistic Regression on Probabilities)
   - Method 5: SLSQP Dual-Target Soft Voting (Validation NLL-Minimization)
4. Multi-Seed Support across Seeds [42, 123, 3407]:
   - Computes individual seed metrics and multi-seed Mean ± SD.
   - Generates comprehensive markdown report and LaTeX tables.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, plot_confusion_matrix
from src.models.ensemble import (
    HardVotingEnsemble,
    SoftVotingEnsemble,
    StackingEnsemble,
    WeightedSoftVotingEnsemble,
    aggregate_video_level_predictions
)
from src.utils.reproducibility import load_checkpoint_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LocalEval")

BASELINE_MODELS = [
    {"name": "ST-GCN Baseline (Rel 3D)", "model": "STGCN", "feat": "rel_3d", "exp_id": "T3.2"},
    {"name": "LSTM Baseline (Mix)", "model": "LSTM", "feat": "mix_v2", "exp_id": "T1.9"},
    {"name": "BiLSTM Baseline (Mix)", "model": "BiLSTM", "feat": "mix_v2", "exp_id": "T1.18"},
    {"name": "Transformer Clean (Mix)", "model": "Transformer", "feat": "mix_v2", "exp_id": "T1.27"},
    {"name": "AAGCN Clean (Bone 3D)", "model": "AAGCN", "feat": "bone_3d", "exp_id": "T3.6"},
]

CONSTITUENT_MODELS = [
    {"name": "Transformer Mix (Aug)", "model": "Transformer", "feat": "mix_v2", "exp_id": "T2.2"},
    {"name": "AAGCN Joint (Aug)", "model": "AAGCN", "feat": "rel_3d", "exp_id": "T4.3"},
    {"name": "AAGCN Bone (Aug)", "model": "AAGCN", "feat": "bone_3d", "exp_id": "T4.2"},
    {"name": "AAGCN J-Motion (Aug)", "model": "AAGCN", "feat": "joint_motion_3d", "exp_id": "T4.4"},
    {"name": "AAGCN B-Motion (Aug)", "model": "AAGCN", "feat": "bone_motion_3d", "exp_id": "T4.5"},
]

def resolve_checkpoint_path(seed: int, model_cfg: Dict[str, str], checkpoint_dir: Path) -> Path:
    base_name = f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feat']}.pt"
    if seed == 42:
        return checkpoint_dir / base_name
    else:
        return checkpoint_dir / f"seed{seed}" / base_name

def evaluate_models_for_seed(
    seed: int,
    device: torch.device,
    checkpoint_dir: Path,
    metadata_path: str,
    landmark_dir: str
) -> Dict[str, Any]:
    logger.info(f"\n========================================================")
    logger.info(f"EVALUATING MODELS FOR SEED {seed}")
    logger.info(f"========================================================")

    # Cache dataloaders by feature method
    dataloaders_cache = {}
    val_video_ids = None
    test_video_ids = None
    y_val_true = None
    y_test_true = None

    def get_cached_loaders(feat: str):
        nonlocal val_video_ids, test_video_ids, y_val_true, y_test_true
        if feat not in dataloaders_cache:
            tr_l, v_l, te_l = get_dataloaders(
                metadata_path=metadata_path,
                feature_method=feat,
                batch_size=32,
                seq_len=32,
                stride=16,
                val_test_stride=32,
                landmark_dir=landmark_dir,
                num_workers=0,
                in_memory=True,
                seed=42,
                strict_norm=True
            )
            dataloaders_cache[feat] = (tr_l, v_l, te_l)
            if val_video_ids is None and hasattr(v_l.dataset, "video_ids"):
                val_video_ids = list(v_l.dataset.video_ids)
            if test_video_ids is None and hasattr(te_l.dataset, "video_ids"):
                test_video_ids = list(te_l.dataset.video_ids)
            if y_val_true is None:
                y_val_true = np.array(v_l.dataset.labels)
            if y_test_true is None:
                y_test_true = np.array(te_l.dataset.labels)
        return dataloaders_cache[feat]

    # Pre-load all required loaders
    for m in CONSTITUENT_MODELS + BASELINE_MODELS:
        get_cached_loaders(m["feat"])

    individual_results = {}
    val_probs_dict = {}
    test_probs_dict = {}

    all_models_to_check = [("baseline", m) for m in BASELINE_MODELS] + [("constituent", m) for m in CONSTITUENT_MODELS]

    for category, cfg in all_models_to_check:
        ckpt_path = resolve_checkpoint_path(seed, cfg, checkpoint_dir)
        if not ckpt_path.exists():
            if category == "baseline":
                logger.info(f"Baseline checkpoint not found (skipping): {ckpt_path.name}")
                continue
            else:
                logger.warning(f"Constituent checkpoint missing: {ckpt_path}. Skipping.")
                continue

        logger.info(f"Loading checkpoint: {ckpt_path.name}...")
        state_dict, _ = load_checkpoint_weights(ckpt_path, device="cpu")
        model = build_model(cfg["model"], cfg["feat"], num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _, val_l, test_l = get_cached_loaders(cfg["feat"])
        trainer = Trainer(model=model, device=device)

        t0 = time.time()
        _, _, v_prob = trainer.predict(val_l)
        _, _, t_prob = trainer.predict(test_l)
        inference_time = time.time() - t0

        val_probs_dict[cfg["name"]] = v_prob
        test_probs_dict[cfg["name"]] = t_prob

        # Window metrics
        t_win_pred = np.argmax(t_prob, axis=1)
        win_metrics = compute_metrics(y_test_true, t_win_pred)

        # Video metrics
        _, _, _, vid_metrics = aggregate_video_level_predictions(t_prob, y_test_true, test_video_ids)

        latency_ms = (inference_time / len(y_test_true)) * 1000

        individual_results[cfg["name"]] = {
            "category": category,
            "win_acc": win_metrics["accuracy"] * 100.0,
            "win_f1": win_metrics["macro_f1"],
            "vid_acc": vid_metrics["accuracy"] * 100.0,
            "vid_f1": vid_metrics["macro_f1"],
            "latency_ms": latency_ms
        }
        logger.info(f"[{cfg['name']}] Win Acc: {win_metrics['accuracy']*100:.2f}% | Vid Acc: {vid_metrics['accuracy']*100:.2f}% | Latency: {latency_ms:.2f}ms")

    # ----------------------------------------------------
    # Evaluate 5 Fusion Methods across Ensemble Targets
    # ----------------------------------------------------
    # Check if the 5 constituent models are present
    constituent_names = [m["name"] for m in CONSTITUENT_MODELS]
    missing_constituents = [n for n in constituent_names if n not in test_probs_dict]

    if missing_constituents:
        logger.warning(f"Cannot run 5 fusion ensemble comparison. Missing: {missing_constituents}")
        return {"seed": seed, "individual": individual_results, "ensembles": {}}

    ensemble_targets = {
        "Two-Stream AAGCN (Aug)": ["AAGCN Joint (Aug)", "AAGCN Bone (Aug)"],
        "Four-Stream AAGCN (Aug)": ["AAGCN Joint (Aug)", "AAGCN Bone (Aug)", "AAGCN J-Motion (Aug)", "AAGCN B-Motion (Aug)"],
        "SkelGym-Lite": ["Transformer Mix (Aug)", "AAGCN Bone (Aug)"],
        "SkelGym-Full": ["Transformer Mix (Aug)", "AAGCN Joint (Aug)", "AAGCN Bone (Aug)", "AAGCN J-Motion (Aug)", "AAGCN B-Motion (Aug)"]
    }

    # Pre-compute true video labels
    unique_vids = []
    vid_to_label = {}
    for vid, lbl in zip(test_video_ids, y_test_true):
        if vid not in vid_to_label:
            vid_to_label[vid] = lbl
            unique_vids.append(vid)
    y_test_vid_true = np.array([vid_to_label[v] for v in unique_vids])

    fusion_results = {}

    for target_name, model_keys in ensemble_targets.items():
        v_probs_sub = [val_probs_dict[k] for k in model_keys]
        t_probs_sub = [test_probs_dict[k] for k in model_keys]

        fusion_results[target_name] = {}

        # 1. Hard Voting
        hard_ens = HardVotingEnsemble()
        win_preds_each = [np.argmax(p, axis=1) for p in t_probs_sub]
        hard_win_preds = hard_ens.predict(win_preds_each)
        hard_win_metrics = compute_metrics(y_test_true, hard_win_preds)

        vid_preds_each = []
        for p in t_probs_sub:
            _, _, p_vid, _ = aggregate_video_level_predictions(p, y_test_true, test_video_ids)
            vid_preds_each.append(p_vid)
        hard_vid_preds = hard_ens.predict(vid_preds_each)
        hard_vid_metrics = compute_metrics(y_test_vid_true, hard_vid_preds)

        fusion_results[target_name]["Hard Voting"] = {
            "win_acc": hard_win_metrics["accuracy"] * 100.0,
            "win_f1": hard_win_metrics["macro_f1"],
            "vid_acc": hard_vid_metrics["accuracy"] * 100.0,
            "vid_f1": hard_vid_metrics["macro_f1"]
        }

        # 2. Uniform Soft Voting (Arithmetic Mean)
        uniform_ens = SoftVotingEnsemble(weights=None)
        u_win_probs = uniform_ens.predict_proba(t_probs_sub)
        u_win_preds = np.argmax(u_win_probs, axis=1)
        u_win_metrics = compute_metrics(y_test_true, u_win_preds)
        _, _, _, u_vid_metrics = aggregate_video_level_predictions(u_win_probs, y_test_true, test_video_ids)

        fusion_results[target_name]["Uniform Soft Voting"] = {
            "win_acc": u_win_metrics["accuracy"] * 100.0,
            "win_f1": u_win_metrics["macro_f1"],
            "vid_acc": u_vid_metrics["accuracy"] * 100.0,
            "vid_f1": u_vid_metrics["macro_f1"]
        }

        # 3. Accuracy-Weighted Soft Voting
        val_accs = [np.mean(y_val_true == np.argmax(vp, axis=1)) for vp in v_probs_sub]
        acc_ens = SoftVotingEnsemble(weights=val_accs)
        acc_win_probs = acc_ens.predict_proba(t_probs_sub)
        acc_win_preds = np.argmax(acc_win_probs, axis=1)
        acc_win_metrics = compute_metrics(y_test_true, acc_win_preds)
        _, _, _, acc_vid_metrics = aggregate_video_level_predictions(acc_win_probs, y_test_true, test_video_ids)

        fusion_results[target_name]["Accuracy-Weighted Soft"] = {
            "win_acc": acc_win_metrics["accuracy"] * 100.0,
            "win_f1": acc_win_metrics["macro_f1"],
            "vid_acc": acc_vid_metrics["accuracy"] * 100.0,
            "vid_f1": acc_vid_metrics["macro_f1"],
            "weights": [round(float(w), 4) for w in (np.array(val_accs) / np.sum(val_accs))]
        }

        # 4. Stacking Meta-Classifier (Logistic Regression)
        stack_ens = StackingEnsemble(c_param=1.0)
        stack_ens.fit(v_probs_sub, y_val_true)
        stk_win_preds = stack_ens.predict(t_probs_sub)
        stk_win_probs = stack_ens.predict_proba(t_probs_sub)
        stk_win_metrics = compute_metrics(y_test_true, stk_win_preds)
        _, _, _, stk_vid_metrics = aggregate_video_level_predictions(stk_win_probs, y_test_true, test_video_ids)

        fusion_results[target_name]["Stacking Meta-Classifier"] = {
            "win_acc": stk_win_metrics["accuracy"] * 100.0,
            "win_f1": stk_win_metrics["macro_f1"],
            "vid_acc": stk_vid_metrics["accuracy"] * 100.0,
            "vid_f1": stk_vid_metrics["macro_f1"]
        }

        # 5. SLSQP Dual-Target Soft Voting (Proposed)
        slsqp_ens = WeightedSoftVotingEnsemble()
        slsqp_ens.fit_window(v_probs_sub, y_val_true)
        sls_win_preds = slsqp_ens.predict_window(t_probs_sub)
        sls_win_metrics = compute_metrics(y_test_true, sls_win_preds)

        slsqp_ens.fit_video(v_probs_sub, y_val_true, val_video_ids)
        _, _, _, sls_vid_metrics = slsqp_ens.predict_video(t_probs_sub, y_test_true, test_video_ids)

        fusion_results[target_name]["SLSQP Soft Voting"] = {
            "win_acc": sls_win_metrics["accuracy"] * 100.0,
            "win_f1": sls_win_metrics["macro_f1"],
            "vid_acc": sls_vid_metrics["accuracy"] * 100.0,
            "vid_f1": sls_vid_metrics["macro_f1"],
            "w_win": [round(float(w), 4) for w in slsqp_ens.weights_window],
            "w_vid": [round(float(w), 4) for w in slsqp_ens.weights_video]
        }

        # Save confusion matrix for Seed 42 SkelGym-Full
        if seed == 42 and target_name == "SkelGym-Full":
            cm_out = PROJECT_ROOT / "outputs" / "ensemble" / "cm_ensemble_T5.1_weighted_soft.png"
            cm_out.parent.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(y_test_true, sls_win_preds, str(cm_out), normalize=True, title="SkelGym-Full (SLSQP Soft Voting)")
            # Sync to paper images
            paper_cm = PROJECT_ROOT / "paper" / "images" / "cm_ensemble_T5.1_weighted_soft.png"
            paper_cm.parent.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(y_test_true, sls_win_preds, str(paper_cm), normalize=True, title="SkelGym-Full (SLSQP Soft Voting)")

    return {
        "seed": seed,
        "individual": individual_results,
        "ensembles": fusion_results
    }

def aggregate_multiseed_results(all_seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes Mean ± SD across all available seeds.
    """
    aggregated = {"individual": {}, "ensembles": {}}

    # Aggregate individuals
    ind_keys = set()
    for s_res in all_seed_results:
        ind_keys.update(s_res["individual"].keys())

    for k in sorted(ind_keys):
        win_accs = [s["individual"][k]["win_acc"] for s in all_seed_results if k in s["individual"]]
        win_f1s = [s["individual"][k]["win_f1"] for s in all_seed_results if k in s["individual"]]
        vid_accs = [s["individual"][k]["vid_acc"] for s in all_seed_results if k in s["individual"]]
        vid_f1s = [s["individual"][k]["vid_f1"] for s in all_seed_results if k in s["individual"]]

        aggregated["individual"][k] = {
            "win_acc_mean": float(np.mean(win_accs)),
            "win_acc_sd": float(np.std(win_accs, ddof=1)) if len(win_accs) > 1 else 0.0,
            "win_f1_mean": float(np.mean(win_f1s)),
            "win_f1_sd": float(np.std(win_f1s, ddof=1)) if len(win_f1s) > 1 else 0.0,
            "vid_acc_mean": float(np.mean(vid_accs)),
            "vid_acc_sd": float(np.std(vid_accs, ddof=1)) if len(vid_accs) > 1 else 0.0,
            "vid_f1_mean": float(np.mean(vid_f1s)),
            "vid_f1_sd": float(np.std(vid_f1s, ddof=1)) if len(vid_f1s) > 1 else 0.0,
        }

    # Aggregate ensembles
    target_keys = set()
    for s_res in all_seed_results:
        target_keys.update(s_res["ensembles"].keys())

    for t_key in sorted(target_keys):
        aggregated["ensembles"][t_key] = {}
        fusion_keys = set()
        for s_res in all_seed_results:
            if t_key in s_res["ensembles"]:
                fusion_keys.update(s_res["ensembles"][t_key].keys())

        for f_key in sorted(fusion_keys):
            win_accs = [s["ensembles"][t_key][f_key]["win_acc"] for s in all_seed_results if t_key in s["ensembles"] and f_key in s["ensembles"][t_key]]
            win_f1s = [s["ensembles"][t_key][f_key]["win_f1"] for s in all_seed_results if t_key in s["ensembles"] and f_key in s["ensembles"][t_key]]
            vid_accs = [s["ensembles"][t_key][f_key]["vid_acc"] for s in all_seed_results if t_key in s["ensembles"] and f_key in s["ensembles"][t_key]]
            vid_f1s = [s["ensembles"][t_key][f_key]["vid_f1"] for s in all_seed_results if t_key in s["ensembles"] and f_key in s["ensembles"][t_key]]

            aggregated["ensembles"][t_key][f_key] = {
                "win_acc_mean": float(np.mean(win_accs)),
                "win_acc_sd": float(np.std(win_accs, ddof=1)) if len(win_accs) > 1 else 0.0,
                "win_f1_mean": float(np.mean(win_f1s)),
                "win_f1_sd": float(np.std(win_f1s, ddof=1)) if len(win_f1s) > 1 else 0.0,
                "vid_acc_mean": float(np.mean(vid_accs)),
                "vid_acc_sd": float(np.std(vid_accs, ddof=1)) if len(vid_accs) > 1 else 0.0,
                "vid_f1_mean": float(np.mean(vid_f1s)),
                "vid_f1_sd": float(np.std(vid_f1s, ddof=1)) if len(vid_f1s) > 1 else 0.0,
            }

    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Evaluate Local Checkpoints & 5 Late Fusion Methods across Multi-Seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407], help="List of random seeds to evaluate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--metadata", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks")
    parser.add_argument("--output_md", type=str, default="outputs/multi_seed_evaluation_results.md")
    args = parser.parse_args()

    dev_str = args.device
    if dev_str == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(dev_str)

    ckpt_dir = PROJECT_ROOT / args.checkpoint_dir

    all_seed_results = []
    for s in args.seeds:
        res = evaluate_models_for_seed(s, device, ckpt_dir, args.metadata, args.landmark_dir)
        all_seed_results.append(res)

    agg = aggregate_multiseed_results(all_seed_results)

    # Print summary tables
    print("\n" + "=" * 105)
    print("MULTI-SEED EVALUATION SUMMARY (MEAN ± SD ACROSS EVALUATED SEEDS)")
    print("=" * 105)
    print(f"{'Model / Architecture':<32} | {'Window Acc (%)':<20} | {'Window Macro F1':<18} | {'Video Acc (%)':<20} | {'Video Macro F1':<18}")
    print("-" * 115)
    for name, m in agg["individual"].items():
        print(f"{name:<32} | {m['win_acc_mean']:>6.2f} ± {m['win_acc_sd']:<6.2f}% | {m['win_f1_mean']:>6.4f} ± {m['win_f1_sd']:<6.4f} | {m['vid_acc_mean']:>6.2f} ± {m['vid_acc_sd']:<6.2f}% | {m['vid_f1_mean']:>6.4f} ± {m['vid_f1_sd']:<6.4f}")

    print("\n" + "=" * 115)
    print("5 FUSION METHODS COMPARISON (MEAN ± SD ACROSS SEEDS)")
    print("=" * 115)
    for target_name, f_dict in agg["ensembles"].items():
        print(f"\n--- Target: {target_name} ---")
        print(f"{'Fusion Method':<32} | {'Window Acc (%)':<20} | {'Window Macro F1':<18} | {'Video Acc (%)':<20} | {'Video Macro F1':<18}")
        print("-" * 115)
        for f_name, m in f_dict.items():
            print(f"{f_name:<32} | {m['win_acc_mean']:>6.2f} ± {m['win_acc_sd']:<6.2f}% | {m['win_f1_mean']:>6.4f} ± {m['win_f1_sd']:<6.4f} | {m['vid_acc_mean']:>6.2f} ± {m['vid_acc_sd']:<6.2f}% | {m['vid_f1_mean']:>6.4f} ± {m['vid_f1_sd']:<6.4f}")

    # Export to markdown
    out_md = PROJECT_ROOT / args.output_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SkelGym Multi-Seed & 5 Fusion Methods Benchmark Results\n\n")
        f.write(f"Evaluated across random seeds: `{args.seeds}`.\n\n")
        f.write("## 1. Individual Backbone Models (Baselines & Augmented Streams)\n\n")
        f.write("| Model / Architecture | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for name, m in agg["individual"].items():
            f.write(f"| **{name}** | {m['win_acc_mean']:.2f}% ± {m['win_acc_sd']:.2f}% | {m['win_f1_mean']:.4f} ± {m['win_f1_sd']:.4f} | {m['vid_acc_mean']:.2f}% ± {m['vid_acc_sd']:.2f}% | {m['vid_f1_mean']:.4f} ± {m['vid_f1_sd']:.4f} |\n")

        f.write("\n## 2. Late Fusion Comparison (5 Methods across Ensemble Targets)\n\n")
        for target_name, f_dict in agg["ensembles"].items():
            f.write(f"### {target_name}\n\n")
            f.write("| Fusion Method | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for f_name, m in f_dict.items():
                f.write(f"| **{f_name}** | {m['win_acc_mean']:.2f}% ± {m['win_acc_sd']:.2f}% | {m['win_f1_mean']:.4f} ± {m['win_f1_sd']:.4f} | {m['vid_acc_mean']:.2f}% ± {m['vid_acc_sd']:.2f}% | {m['vid_f1_mean']:.4f} ± {m['vid_f1_sd']:.4f} |\n")
            f.write("\n")

    logger.info(f"\nSaved full benchmark report to: {out_md}")

if __name__ == '__main__':
    main()
