#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner and Evaluator for SkelGym.
Evaluates 3 independent random seeds (42, 123, 3407) across the 4 Final Experiments:
1. Transformer Mix + Aug (T2.2)
2. Four-Stream AAGCN (T4.7)
3. SkelGym-Lite (T5.2)
4. SkelGym-Full (T5.1)

Maintains strictly identical dataset splits, preprocessing, augmentation, architectures,
and training/evaluation protocols across all seeds.
Reports Mean ± SD for Window Accuracy, Window Macro-F1, Video Accuracy, and Video Macro-F1.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions
from src.utils.hf_hub import ensure_checkpoint_available, pull_landmarks_from_hf

SEEDS = [42, 123, 3407]

CONSTITUENT_MODELS = [
    {"name": "Transformer_mix", "model": "Transformer", "feature": "mix", "exp_id": "T2.2"},
    {"name": "AAGCN_bone_3d", "model": "AAGCN", "feature": "bone_3d", "exp_id": "T4.2"},
    {"name": "AAGCN_rel_3d", "model": "AAGCN", "feature": "rel_3d", "exp_id": "T4.3"},
    {"name": "AAGCN_joint_motion_3d", "model": "AAGCN", "feature": "joint_motion_3d", "exp_id": "T4.4"},
    {"name": "AAGCN_bone_motion_3d", "model": "AAGCN", "feature": "bone_motion_3d", "exp_id": "T4.5"},
]

def get_checkpoint_path(seed: int, model_cfg: Dict[str, str], checkpoint_base: Path) -> Path:
    if seed == 42:
        return checkpoint_base / f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"
    else:
        return checkpoint_base / f"seed{seed}" / f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"

def train_model(seed: int, model_cfg: Dict[str, str], device_str: str, checkpoint_base: Path):
    ckpt_path = get_checkpoint_path(seed, model_cfg, checkpoint_base)
    if ckpt_path.exists():
        print(f"[Seed {seed}] Checkpoint already exists: {ckpt_path}, skipping training.")
        return ckpt_path

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = str(ckpt_path.parent)

    cmd = [
        sys.executable, "run.py", "train",
        "--model", model_cfg["model"],
        "--feature", model_cfg["feature"],
        "--augment", "skel_gym_aug",
        "--exp_id", model_cfg["exp_id"],
        "--seed", str(seed),
        "--epochs", "100",
        "--patience", "10",
        "--checkpoint_dir", ckpt_dir,
        "--video_level",
        "--device", device_str,
        "--use_amp",
        "--in_memory"
    ]

    print(f"\n========================================================")
    print(f"[Seed {seed}] Training {model_cfg['name']} -> {ckpt_path.name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"========================================================")

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Training failed for {model_cfg['name']} seed {seed} (code {res.returncode})")
    
    elapsed = time.time() - t0
    print(f"[Seed {seed}] Completed training {model_cfg['name']} in {elapsed:.1f}s")
    return ckpt_path

def evaluate_seed(seed: int, device: torch.device, checkpoint_base: Path, metadata_path: str, landmark_dir: str):
    print(f"\n>>> Evaluating 4 Final Configurations for Seed {seed} <<<")
    
    # Checkpoints for this seed
    ckpts = {
        m["feature"]: get_checkpoint_path(seed, m, checkpoint_base)
        for m in CONSTITUENT_MODELS
    }
    
    # Load loaders and models
    val_probs = {}
    test_probs = {}
    y_val_true = None
    y_test_true = None
    val_video_ids = None
    test_video_ids = None

    for m in CONSTITUENT_MODELS:
        feat = m["feature"]
        p = ckpts[feat]
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {p}")

        state_dict = torch.load(p, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model = build_model(m["model"], feat, num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        _, val_loader, test_loader = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=feat,
            batch_size=32,
            seq_len=32,
            stride=32,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True
        )

        if val_video_ids is None:
            val_video_ids = val_loader.dataset.video_ids
        if test_video_ids is None:
            test_video_ids = test_loader.dataset.video_ids

        trainer = Trainer(model=model, device=device)
        y_vt, _, y_vp = trainer.predict(val_loader)
        y_tt, _, y_tp = trainer.predict(test_loader)

        if y_val_true is None:
            y_val_true = y_vt
        if y_test_true is None:
            y_test_true = y_tt

        val_probs[feat] = y_vp
        test_probs[feat] = y_tp

    seed_results = {}

    # ----------------------------------------------------
    # Config 1: Transformer Mix + Aug (T2.2)
    # ----------------------------------------------------
    t_test_prob = test_probs["mix"]
    t_win_pred = np.argmax(t_test_prob, axis=1)
    t_win_metrics = compute_metrics(y_test_true, t_win_pred)
    _, _, _, t_vid_metrics = aggregate_video_level_predictions(t_test_prob, y_test_true, test_video_ids)

    seed_results["Transformer Mix + Aug"] = {
        "win_acc": float(t_win_metrics["accuracy"] * 100.0),
        "win_f1": float(t_win_metrics["macro_f1"]),
        "vid_acc": float(t_vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(t_vid_metrics["macro_f1"])
    }

    # ----------------------------------------------------
    # Config 2: Four-Stream AAGCN (T4.7)
    # ----------------------------------------------------
    four_feats = ["bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    four_val = [val_probs[f] for f in four_feats]
    four_test = [test_probs[f] for f in four_feats]

    ens_four = WeightedSoftVotingEnsemble()
    ens_four.fit_window(four_val, y_val_true)
    ens_four_win_pred = ens_four.predict_window(four_test)
    four_win_metrics = compute_metrics(y_test_true, ens_four_win_pred)

    ens_four.fit_video(four_val, y_val_true, val_video_ids)
    _, _, _, four_vid_metrics = ens_four.predict_video(four_test, y_test_true, test_video_ids)

    seed_results["Four-Stream AAGCN"] = {
        "win_acc": float(four_win_metrics["accuracy"] * 100.0),
        "win_f1": float(four_win_metrics["macro_f1"]),
        "vid_acc": float(four_vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(four_vid_metrics["macro_f1"]),
        "weights_window": [round(float(w), 4) for w in ens_four.weights_window],
        "weights_video": [round(float(w), 4) for w in ens_four.weights_video]
    }

    # ----------------------------------------------------
    # Config 3: SkelGym-Lite (T5.2: Transformer Mix + AAGCN Bone)
    # ----------------------------------------------------
    lite_feats = ["mix", "bone_3d"]
    lite_val = [val_probs[f] for f in lite_feats]
    lite_test = [test_probs[f] for f in lite_feats]

    ens_lite = WeightedSoftVotingEnsemble()
    ens_lite.fit_window(lite_val, y_val_true)
    ens_lite_win_pred = ens_lite.predict_window(lite_test)
    lite_win_metrics = compute_metrics(y_test_true, ens_lite_win_pred)

    ens_lite.fit_video(lite_val, y_val_true, val_video_ids)
    _, _, _, lite_vid_metrics = ens_lite.predict_video(lite_test, y_test_true, test_video_ids)

    seed_results["SkelGym-Lite"] = {
        "win_acc": float(lite_win_metrics["accuracy"] * 100.0),
        "win_f1": float(lite_win_metrics["macro_f1"]),
        "vid_acc": float(lite_vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(lite_vid_metrics["macro_f1"]),
        "weights_window": [round(float(w), 4) for w in ens_lite.weights_window],
        "weights_video": [round(float(w), 4) for w in ens_lite.weights_video]
    }

    # ----------------------------------------------------
    # Config 4: SkelGym-Full (T5.1: Transformer Mix + 4-Stream AAGCN)
    # ----------------------------------------------------
    full_feats = ["mix", "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    full_val = [val_probs[f] for f in full_feats]
    full_test = [test_probs[f] for f in full_feats]

    ens_full = WeightedSoftVotingEnsemble()
    ens_full.fit_window(full_val, y_val_true)
    ens_full_win_pred = ens_full.predict_window(full_test)
    full_win_metrics = compute_metrics(y_test_true, ens_full_win_pred)

    ens_full.fit_video(full_val, y_val_true, val_video_ids)
    _, _, _, full_vid_metrics = ens_full.predict_video(full_test, y_test_true, test_video_ids)

    seed_results["SkelGym-Full"] = {
        "win_acc": float(full_win_metrics["accuracy"] * 100.0),
        "win_f1": float(full_win_metrics["macro_f1"]),
        "vid_acc": float(full_vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(full_vid_metrics["macro_f1"]),
        "weights_window": [round(float(w), 4) for w in ens_full.weights_window],
        "weights_video": [round(float(w), 4) for w in ens_full.weights_video]
    }

    print(f"\nResults for Seed {seed}:")
    for cfg_name, res in seed_results.items():
        print(f"  {cfg_name:<25}: Win Acc: {res['win_acc']:.2f}% | Win F1: {res['win_f1']:.4f} | Vid Acc: {res['vid_acc']:.2f}% | Vid F1: {res['vid_f1']:.4f}")

    return seed_results

def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Experiment Runner (Seeds 42, 123, 3407)")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/mps/cpu)")
    parser.add_argument("--metadata", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_file", type=str, default="outputs/multi_seed_evaluation_results.json")
    parser.add_argument("--skip_train", "--evaluate_only", dest="skip_train", action="store_true", help="Skip training and only evaluate")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Running Multi-Seed Evaluation on device: {device}")

    # Ensure landmarks exist
    if not os.path.exists(args.landmark_dir) or not os.path.exists(args.metadata):
        print("Dataset not found locally, pulling from HF...")
        pull_landmarks_from_hf(dest_dir=args.landmark_dir)

    checkpoint_base = Path(args.checkpoint_dir)
    checkpoint_base.mkdir(parents=True, exist_ok=True)

    # Ensure Seed 42 checkpoints are available
    print("\nEnsuring Seed 42 checkpoints are available...")
    for m in CONSTITUENT_MODELS:
        ckpt = get_checkpoint_path(42, m, checkpoint_base)
        if not ckpt.exists():
            print(f"Downloading {ckpt} from HF...")
            ensure_checkpoint_available(str(ckpt))

    # Train Seed 123 and Seed 3407 if needed
    if not args.skip_train:
        for seed in [123, 3407]:
            print(f"\n========================================================")
            print(f"TRAINING PHASE FOR SEED {seed}")
            print(f"========================================================")
            for m in CONSTITUENT_MODELS:
                train_model(seed, m, args.device, checkpoint_base)

    # Evaluate all 3 seeds
    all_seed_results = {}
    for seed in SEEDS:
        all_seed_results[str(seed)] = evaluate_seed(seed, device, checkpoint_base, args.metadata, args.landmark_dir)

    # Compute Mean ± SD across seeds
    configs = ["Transformer Mix + Aug", "Four-Stream AAGCN", "SkelGym-Lite", "SkelGym-Full"]
    summary_stats = {}

    print("\n" + "=" * 95)
    print("MAIN TABLE: MULTI-SEED SUMMARY (MEAN ± SD ACROSS SEEDS 42, 123, 3407)")
    print("=" * 95)
    print(f"{'Model Architecture':<30} | {'Window Accuracy':<18} | {'Window Macro-F1':<18} | {'Video Accuracy':<18} | {'Video Macro-F1':<18}")
    print("-" * 110)

    for cfg in configs:
        win_accs = [all_seed_results[str(s)][cfg]["win_acc"] for s in SEEDS]
        win_f1s = [all_seed_results[str(s)][cfg]["win_f1"] for s in SEEDS]
        vid_accs = [all_seed_results[str(s)][cfg]["vid_acc"] for s in SEEDS]
        vid_f1s = [all_seed_results[str(s)][cfg]["vid_f1"] for s in SEEDS]

        summary_stats[cfg] = {
            "win_acc_mean": float(np.mean(win_accs)),
            "win_acc_sd": float(np.std(win_accs, ddof=1)),
            "win_f1_mean": float(np.mean(win_f1s)),
            "win_f1_sd": float(np.std(win_f1s, ddof=1)),
            "vid_acc_mean": float(np.mean(vid_accs)),
            "vid_acc_sd": float(np.std(vid_accs, ddof=1)),
            "vid_f1_mean": float(np.mean(vid_f1s)),
            "vid_f1_sd": float(np.std(vid_f1s, ddof=1)),
            "raw_seeds": {
                str(s): all_seed_results[str(s)][cfg] for s in SEEDS
            }
        }

        s_str = (
            f"{cfg:<30} | "
            f"{summary_stats[cfg]['win_acc_mean']:>5.2f}% ± {summary_stats[cfg]['win_acc_sd']:>4.2f}%    | "
            f"{summary_stats[cfg]['win_f1_mean']:>6.4f} ± {summary_stats[cfg]['win_f1_sd']:>6.4f}   | "
            f"{summary_stats[cfg]['vid_acc_mean']:>5.2f}% ± {summary_stats[cfg]['vid_acc_sd']:>4.2f}%    | "
            f"{summary_stats[cfg]['vid_f1_mean']:>6.4f} ± {summary_stats[cfg]['vid_f1_sd']:>6.4f}"
        )
        print(s_str)
    print("=" * 110)

    # Save to JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "seeds": SEEDS,
            "models": configs,
            "summary": summary_stats,
            "individual_runs": all_seed_results
        }, f, indent=2)
    print(f"\nSaved detailed multi-seed results to: {out_path}")

    # Generate Markdown
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# SkelGym Multi-Seed Evaluation Benchmark\n\n")
        f.write("Evaluation across 3 independent random seeds (`42`, `123`, `3407`) under strictly fixed splits and hyperparameter settings.\n\n")
        f.write("## Main Results Table (Mean ± SD)\n\n")
        f.write("| Model Architecture | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cfg in configs:
            st = summary_stats[cfg]
            f.write(f"| **{cfg}** | {st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}% | {st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f} | {st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}% | {st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f} |\n")
        
        f.write("\n## Per-Seed Detailed Breakdown\n\n")
        for cfg in configs:
            f.write(f"### {cfg}\n\n")
            f.write("| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for s in SEEDS:
                r = all_seed_results[str(s)][cfg]
                f.write(f"| {s} | {r['win_acc']:.2f}% | {r['win_f1']:.4f} | {r['vid_acc']:.2f}% | {r['vid_f1']:.4f} |\n")
            f.write("\n")
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()
