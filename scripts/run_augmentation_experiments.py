#!/usr/bin/env python3
"""
Orchestrator for Systematic Augmentation Experiments & Leave-One-Out Ablation.
Evaluates:
  Part A: Leave-One-Out Ablation on Representative Sequence Transformer (mix 117d) across 3 seeds (42, 123, 3407).
          - Full SkelGym-Aug
          - w/o Bilateral Mirror (skel_gym_aug_no_mirror)
          - w/o 3D Yaw Rotation (skel_gym_aug_no_yaw)
          - w/o Spatial Scale (skel_gym_aug_no_scale)
          - w/o Temporal TimeWarp (skel_gym_aug_no_timewarp)
          - w/o Sensor Jitter (skel_gym_aug_no_jitter)
  Part B: Cross-Backbone Generalization (No-Aug vs SkelGym-Aug) across 3 seeds (42, 123, 3407).
          - Transformer (mix 117d)
          - AAGCN (bone 3d)
          - BiLSTM (mix 117d)
          - ST-GCN (rel 3d)

Features:
  - Concurrent parallel training (5-8 workers) leveraging server-grade GPU VRAM.
  - Automatically re-uses pre-existing checkpoints (skips retraining).
  - Evaluates both Window-level and Video-level Accuracy and Macro-F1.
  - Generates comprehensive Mean ± Std summary tables.
  - Periodic progress logging with JSON checkpoints.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import pandas as pd

from src.constants import NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

def find_existing_checkpoint(task: Dict[str, Any], checkpoint_base: Path) -> Optional[Path]:
    """
    Checks if a checkpoint already exists in the filesystem.
    """
    seed = task["seed"]
    model = task["model"]
    feature = task["feature"]
    aug = task["augment"]
    task_id = task["id"]

    candidates = []

    # 1. Custom task-specific checkpoint
    candidates.append(checkpoint_base / f"{task_id}.pt")
    candidates.append(checkpoint_base / f"seed{seed}" / f"{task_id}.pt")

    # 2. Known seed 42 checkpoints from base benchmark
    if seed == 42:
        if model == "Transformer" and feature == "mix" and aug == "none":
            candidates.append(checkpoint_base / "best_Transformer_T1.27_mix.pt")
        elif model == "Transformer" and feature == "mix" and aug == "skel_gym_aug":
            candidates.append(checkpoint_base / "best_Transformer_T2.2_mix.pt")
        elif model == "AAGCN" and feature == "bone_3d" and aug == "none":
            candidates.append(checkpoint_base / "best_AAGCN_T4.1_bone_3d.pt")
        elif model == "AAGCN" and feature == "bone_3d" and aug == "skel_gym_aug":
            candidates.append(checkpoint_base / "best_AAGCN_T4.2_bone_3d.pt")
        elif model == "BiLSTM" and feature == "mix" and aug == "none":
            candidates.append(checkpoint_base / "best_BiLSTM_T1.18_mix.pt")
        elif model == "STGCN" and feature == "rel_3d" and aug == "none":
            candidates.append(checkpoint_base / "best_STGCN_T3.2_rel_3d.pt")
    else:
        # Check seed directory
        candidates.append(checkpoint_base / f"seed{seed}" / f"best_{model}_{feature}.pt")
        candidates.append(checkpoint_base / f"seed{seed}" / f"best_{model}_{aug}_{feature}.pt")

    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 1000:
            return cand

    return None

def train_task_subprocess(task: Dict[str, Any], ckpt_path: Path, device: str = "cuda") -> bool:
    """
    Executes training via a separate python subprocess for memory isolation.
    """
    cmd = [
        sys.executable, "-m", "src.cli", "train",
        "--model", task["model"],
        "--feature", task["feature"],
        "--augment", task["augment"],
        "--seed", str(task["seed"]),
        "--epochs", "100",
        "--patience", "10",
        "--batch_size", "16",
        "--device", device,
        "--checkpoint_dir", str(ckpt_path.parent),
        "--output_dir", "outputs/ablation_runs",
        "--metadata", task.get("metadata", "Final_dataset_metadata.csv"),
        "--landmark_dir", task.get("landmark_dir", "data/landmarks")
    ]

    log_dir = Path("outputs/ablation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task['id']}.log"

    print(f"[START] Training {task['id']} on device {device} -> {ckpt_path.name}")
    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as lf:
        res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)

    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[DONE] {task['id']} finished in {elapsed:.1f}s")
        # Rename checkpoint if default name was used
        default_ckpt = ckpt_path.parent / f"best_{task['model']}_{task['feature']}.pt"
        if default_ckpt.exists() and default_ckpt != ckpt_path:
            default_ckpt.rename(ckpt_path)
        return True
    else:
        print(f"[ERROR] {task['id']} failed with code {res.returncode}. See {log_file}")
        return False

def evaluate_checkpoint(
    ckpt_path: Path,
    model_type: str,
    feature_method: str,
    device: torch.device,
    metadata_path: str = "Final_dataset_metadata.csv",
    landmark_dir: str = "data/landmarks"
) -> Dict[str, float]:
    """
    Loads checkpoint and performs rigorous held-out test evaluation.
    Returns: window_acc, window_f1, video_acc, video_f1.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model = build_model(model_type, feature_method, num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method=feature_method,
        batch_size=32,
        seq_len=32,
        stride=32,
        val_test_stride=32,
        landmark_dir=landmark_dir,
        num_workers=0,
        in_memory=True
    )

    test_video_ids = test_loader.dataset.video_ids
    trainer = Trainer(model=model, device=device)
    y_test_true, _, y_test_probs = trainer.predict(test_loader)

    # Window metrics
    win_preds = np.argmax(y_test_probs, axis=1)
    win_metrics = compute_metrics(y_test_true, win_preds)

    # Video metrics
    _, _, _, vid_metrics = aggregate_video_level_predictions(y_test_probs, y_test_true, test_video_ids)

    return {
        "win_acc": float(win_metrics["accuracy"] * 100.0),
        "win_f1": float(win_metrics["macro_f1"]),
        "vid_acc": float(vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(vid_metrics["macro_f1"])
    }

def build_task_list(metadata_path: str, landmark_dir: str) -> List[Dict[str, Any]]:
    """
    Defines the complete list of tasks for Leave-One-Out and Backbone Ablation.
    """
    tasks = []

    # ---------------------------------------------------------
    # PART A: Leave-One-Out Ablation on Transformer (mix 117d)
    # ---------------------------------------------------------
    loo_variants = [
        ("Full_SkelGym_Aug", "skel_gym_aug"),
        ("Minus_Mirror", "skel_gym_aug_no_mirror"),
        ("Minus_Yaw", "skel_gym_aug_no_yaw"),
        ("Minus_Scale", "skel_gym_aug_no_scale"),
        ("Minus_TimeWarp", "skel_gym_aug_no_timewarp"),
        ("Minus_Jitter", "skel_gym_aug_no_jitter"),
        ("Clean_Baseline_NoAug", "none")
    ]

    for variant_name, aug_method in loo_variants:
        for seed in SEEDS:
            task_id = f"LOO_Trans_{variant_name}_seed{seed}"
            tasks.append({
                "group": "Leave_One_Out",
                "variant": variant_name,
                "id": task_id,
                "model": "Transformer",
                "feature": "mix",
                "augment": aug_method,
                "seed": seed,
                "metadata": metadata_path,
                "landmark_dir": landmark_dir
            })

    # ---------------------------------------------------------
    # PART B: Cross-Backbone Generalization (No-Aug vs SkelGym-Aug)
    # ---------------------------------------------------------
    backbones = [
        ("Transformer", "mix"),
        ("AAGCN", "bone_3d"),
        ("BiLSTM", "mix"),
        ("STGCN", "rel_3d")
    ]

    for model, feat in backbones:
        for aug in ["none", "skel_gym_aug"]:
            aug_tag = "Aug" if aug == "skel_gym_aug" else "NoAug"
            for seed in SEEDS:
                # Skip if already in Part A
                if model == "Transformer" and feat == "mix":
                    continue
                task_id = f"Backbone_{model}_{feat}_{aug_tag}_seed{seed}"
                tasks.append({
                    "group": "Backbone_Ablation",
                    "variant": f"{model}_{feat}_{aug_tag}",
                    "id": task_id,
                    "model": model,
                    "feature": feat,
                    "augment": aug,
                    "seed": seed,
                    "metadata": metadata_path,
                    "landmark_dir": landmark_dir
                })

    return tasks

def main():
    parser = argparse.ArgumentParser(description="Systematic Augmentation Ablation Runner")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent worker count for parallel training")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/mps/cpu)")
    parser.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Landmarks directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoints directory")
    parser.add_argument("--output_file", type=str, default="outputs/augmentation_ablation_results.json", help="Results JSON")
    args = parser.parse_args()

    ckpt_base = Path(args.checkpoint_dir)
    ckpt_base.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_task_list(args.metadata, args.landmark_dir)
    print(f"Total ablation tasks defined: {len(tasks)}")

    # 1. Discover existing vs pending tasks
    tasks_to_train = []
    task_ckpts = {}

    for t in tasks:
        existing = find_existing_checkpoint(t, ckpt_base)
        if existing:
            print(f"[REUSE] Task {t['id']} -> {existing.name}")
            task_ckpts[t["id"]] = existing
        else:
            target_ckpt = ckpt_base / f"ablation_aug" / f"{t['id']}.pt"
            target_ckpt.parent.mkdir(parents=True, exist_ok=True)
            task_ckpts[t["id"]] = target_ckpt
            tasks_to_train.append((t, target_ckpt))

    print(f"Tasks to train: {len(tasks_to_train)}, Tasks reusing checkpoints: {len(task_ckpts) - len(tasks_to_train)}")

    # 2. Parallel Training Execution
    if tasks_to_train:
        print(f"\n>>> Launching {len(tasks_to_train)} training tasks with {args.workers} parallel workers on {args.device} <<<\n")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(train_task_subprocess, t, ckpt, args.device): t["id"]
                for t, ckpt in tasks_to_train
            }
            for fut in as_completed(futures):
                t_id = futures[fut]
                try:
                    success = fut.result()
                    if not success:
                        print(f"[WARNING] Task {t_id} did not finish successfully.")
                except Exception as e:
                    print(f"[EXCEPTION] Task {t_id} raised: {e}")

    # 3. Comprehensive Evaluation Phase
    print("\n>>> Evaluating all checkpoints on held-out test partition <<<\n")
    eval_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_results = {}

    for t in tasks:
        t_id = t["id"]
        ckpt = task_ckpts.get(t_id)
        if not ckpt or not ckpt.exists():
            print(f"[SKIP] Checkpoint missing for {t_id}: {ckpt}")
            continue

        print(f"[EVAL] Evaluating {t_id} ({ckpt.name})...")
        metrics = evaluate_checkpoint(
            ckpt_path=ckpt,
            model_type=t["model"],
            feature_method=t["feature"],
            device=eval_device,
            metadata_path=args.metadata,
            landmark_dir=args.landmark_dir
        )
        t_record = dict(t)
        t_record["metrics"] = metrics
        t_record["checkpoint"] = str(ckpt)
        all_results[t_id] = t_record

    # 4. Aggregate Mean ± Std across seeds
    summary_loo = {}
    summary_backbones = {}

    # Aggregate Leave-One-Out
    loo_variants = [
        "Full_SkelGym_Aug",
        "Minus_Mirror",
        "Minus_Yaw",
        "Minus_Scale",
        "Minus_TimeWarp",
        "Minus_Jitter",
        "Clean_Baseline_NoAug"
    ]

    for v in loo_variants:
        v_tasks = [r for r in all_results.values() if r["group"] == "Leave_One_Out" and r["variant"] == v]
        if v_tasks:
            w_accs = [r["metrics"]["win_acc"] for r in v_tasks]
            w_f1s = [r["metrics"]["win_f1"] for r in v_tasks]
            v_accs = [r["metrics"]["vid_acc"] for r in v_tasks]
            v_f1s = [r["metrics"]["vid_f1"] for r in v_tasks]
            summary_loo[v] = {
                "count": len(v_tasks),
                "win_acc_mean": float(np.mean(w_accs)),
                "win_acc_std": float(np.std(w_accs)),
                "win_f1_mean": float(np.mean(w_f1s)),
                "win_f1_std": float(np.std(w_f1s)),
                "vid_acc_mean": float(np.mean(v_accs)),
                "vid_acc_std": float(np.std(v_accs)),
                "vid_f1_mean": float(np.mean(v_f1s)),
                "vid_f1_std": float(np.std(v_f1s))
            }

    # Aggregate Backbones
    bb_variants = set(r["variant"] for r in all_results.values() if r["group"] == "Backbone_Ablation")
    for v in sorted(bb_variants):
        v_tasks = [r for r in all_results.values() if r["group"] == "Backbone_Ablation" and r["variant"] == v]
        if v_tasks:
            w_accs = [r["metrics"]["win_acc"] for r in v_tasks]
            w_f1s = [r["metrics"]["win_f1"] for r in v_tasks]
            v_accs = [r["metrics"]["vid_acc"] for r in v_tasks]
            v_f1s = [r["metrics"]["vid_f1"] for r in v_tasks]
            summary_backbones[v] = {
                "count": len(v_tasks),
                "win_acc_mean": float(np.mean(w_accs)),
                "win_acc_std": float(np.std(w_accs)),
                "win_f1_mean": float(np.mean(w_f1s)),
                "win_f1_std": float(np.std(w_f1s)),
                "vid_acc_mean": float(np.mean(v_accs)),
                "vid_acc_std": float(np.std(v_accs)),
                "vid_f1_mean": float(np.mean(v_f1s)),
                "vid_f1_std": float(np.std(v_f1s))
            }

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": SEEDS,
        "summary_leave_one_out": summary_loo,
        "summary_backbones": summary_backbones,
        "raw_tasks": all_results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n[SAVED] All ablation results written to {out_path}")

    # Generate Markdown Summary
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("# Systematic SkelGym-Augmentation Ablation Benchmark\n\n")
        mf.write(f"Evaluated across {len(SEEDS)} independent random seeds: `{SEEDS}`.\n\n")

        mf.write("## 1. Leave-One-Out Ablation on Representative Sequence Transformer (Mix 117-d)\n\n")
        mf.write("| Configuration | Window Acc (%) | Window Macro-F1 | Video Acc (%) | Video Macro-F1 |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for v in loo_variants:
            if v in summary_loo:
                s = summary_loo[v]
                mf.write(f"| **{v}** | {s['win_acc_mean']:.2f}% ± {s['win_acc_std']:.2f}% | {s['win_f1_mean']:.4f} ± {s['win_f1_std']:.4f} | {s['vid_acc_mean']:.2f}% ± {s['vid_acc_std']:.2f}% | {s['vid_f1_mean']:.4f} ± {s['vid_f1_std']:.4f} |\n")

        mf.write("\n## 2. Cross-Backbone Generalization (No-Aug vs SkelGym-Aug)\n\n")
        mf.write("| Architecture / Stream | Setting | Window Acc (%) | Window Macro-F1 | Video Acc (%) | Video Macro-F1 |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for v in sorted(bb_variants):
            if v in summary_backbones:
                s = summary_backbones[v]
                mf.write(f"| **{v}** | 3-seed Mean | {s['win_acc_mean']:.2f}% ± {s['win_acc_std']:.2f}% | {s['win_f1_mean']:.4f} ± {s['win_f1_std']:.4f} | {s['vid_acc_mean']:.2f}% ± {s['vid_acc_std']:.2f}% | {s['vid_f1_mean']:.4f} ± {s['vid_f1_std']:.4f} |\n")

    print(f"[SAVED] Markdown report written to {md_path}")

if __name__ == "__main__":
    main()
