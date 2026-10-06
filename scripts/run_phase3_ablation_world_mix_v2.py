#!/usr/bin/env python3
"""
Phase 3: Table 4 (Leave-One-Out) and Table 5 (Single-Operator) Ablation on WORLD mix_v2.
Model: Dual-Branch Transformer (~301K params: d_model=112, nhead=4, layers=3, d_ff=168).
Feature: mix_v2 (63-d: 39-d metric world landmarks + 24-d kinematic angles).
Seeds: 42, 123, 3407.

Unique Configurations (12 total):
  1. clean (none)
  2. candidate_full_5op (Mirror + Yaw + Scale + Time + Jitter)
  3. candidate_minus_mirror
  4. candidate_minus_yaw
  5. candidate_minus_scale
  6. candidate_minus_time (identical to proposed 4-op)
  7. candidate_minus_jitter
  8. single_mirror
  9. single_yaw
  10. single_scale
  11. single_time
  12. single_jitter

Outputs:
  - outputs/table4_loo_world_mix_v2.json
  - outputs/table5_single_world_mix_v2.json
  - outputs/augmentation_ablation_results.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX
from src.data.dataset import get_dataloaders
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

ABLATION_CONFIGS = [
    # Table 4 (LOO) + Table 5 (Single)
    {"id": "clean", "aug": "none", "desc": "Clean Baseline (No Aug)", "in_table4": True, "in_table5": True},
    {"id": "candidate_full_5op", "aug": "candidate_full_5op", "desc": "Candidate Full 5-op", "in_table4": True, "in_table5": False},
    {"id": "candidate_minus_mirror", "aug": "candidate_minus_mirror", "desc": "Minus Mirror", "in_table4": True, "in_table5": False},
    {"id": "candidate_minus_yaw", "aug": "candidate_minus_yaw", "desc": "Minus Yaw", "in_table4": True, "in_table5": False},
    {"id": "candidate_minus_scale", "aug": "candidate_minus_scale", "desc": "Minus Scale", "in_table4": True, "in_table5": False},
    {"id": "candidate_minus_time", "aug": "candidate_minus_time", "desc": "Minus TimeWarp (Proposed 4-op)", "in_table4": True, "in_table5": False},
    {"id": "candidate_minus_jitter", "aug": "candidate_minus_jitter", "desc": "Minus Jitter", "in_table4": True, "in_table5": False},
    {"id": "single_mirror", "aug": "single_mirror", "desc": "Only Mirror", "in_table4": False, "in_table5": True},
    {"id": "single_yaw", "aug": "single_yaw", "desc": "Only Yaw", "in_table4": False, "in_table5": True},
    {"id": "single_scale", "aug": "single_scale", "desc": "Only Scale", "in_table4": False, "in_table5": True},
    {"id": "single_time", "aug": "single_time", "desc": "Only TimeWarp", "in_table4": False, "in_table5": True},
    {"id": "single_jitter", "aug": "single_jitter", "desc": "Only Jitter", "in_table4": False, "in_table5": True},
]

def send_marimo_toast(msg: str):
    try:
        import marimo as mo
        mo.status.toast(msg)
    except Exception:
        pass

def train_ablation_run(
    cfg_id: str,
    aug_method: str,
    seed: int,
    metadata_path: str,
    device: torch.device,
    checkpoint_dir: Path
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n---> Training Transformer mix_v2 on {cfg_id} (aug: {aug_method}) [Seed {seed}]...")
    t_start = time.time()

    # 1. DataLoaders
    # Note: augment_method is passed directly. dataset.py handles on-the-fly augmentation dynamically
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix_v2",
        batch_size=16,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=aug_method if aug_method != "none" else None,
        zero_frame_handling="interpolate",
        landmark_dir=None,  # auto-detected data/world_landmarks
        in_memory=True,
        seed=seed,
        save_norm_artifact=True,
        strict_norm=False
    )

    # 2. Model: Dual-Branch Transformer (301K params)
    model = build_model(
        model_type="Transformer",
        feature_method="mix_v2",
        hidden_dim=112,
        num_layers=3,
        nhead=4,
        dropout=0.2,
        transformer_variant="dual_branch"
    )

    # 3. Setup Trainer
    seed_ckpt_dir = checkpoint_dir / f"seed{seed}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"Transformer_mix_v2_{cfg_id}_seed{seed}"
    best_ckpt_path = seed_ckpt_dir / f"best_{model_name}.pt"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir=str(seed_ckpt_dir),
        model_name=model_name,
        label_smoothing=0.05,
        early_stopping_metric="val_macro_f1",
        use_amp=torch.cuda.is_available(),
        feature_method="mix_v2",
        augment_method=aug_method,
        seed=seed
    )

    # 4. Fit Model
    history = trainer.fit(train_loader, val_loader, epochs=100)
    train_time = time.time() - t_start

    # 5. Evaluate on Validation
    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    # 6. Evaluate on Held-Out Test
    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

    prov_file = best_ckpt_path.with_suffix(".provenance.json")
    val_loss = None
    best_epoch = None
    if prov_file.exists():
        try:
            with open(prov_file) as pf:
                pdata = json.load(pf)
                val_loss = pdata.get("val_loss")
                best_epoch = pdata.get("epoch")
        except Exception:
            pass

    res = {
        "cfg_id": cfg_id,
        "aug_method": aug_method,
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "best_epoch": best_epoch or 0,
        "val_loss": round(float(val_loss), 4) if val_loss is not None else 0.0,
        "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
        "val_win_f1": round(float(val_m["macro_f1"]), 4),
        "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
        "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
        "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
        "test_win_f1": round(float(test_m["macro_f1"]), 4),
        "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
        "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
        "checkpoint": str(best_ckpt_path)
    }

    print(f"[{cfg_id} | Seed {seed}] Val Vid Acc: {res['val_vid_acc']}%, Val Vid F1: {res['val_vid_f1']} | Test Vid Acc: {res['test_vid_acc']}%, Test Vid F1: {res['test_vid_f1']}")
    return res

def summarize_group(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    val_w = [r["val_win_acc"] for r in runs]
    val_v = [r["val_vid_acc"] for r in runs]
    val_vf1 = [r["val_vid_f1"] for r in runs]
    test_w = [r["test_win_acc"] for r in runs]
    test_wf1 = [r["test_win_f1"] for r in runs]
    test_v = [r["test_vid_acc"] for r in runs]
    test_vf1 = [r["test_vid_f1"] for r in runs]
    return {
        "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
        "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
        "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
        "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
        "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
        "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
        "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
        "runs": runs
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Table 4 & Table 5 Ablations on WORLD mix_v2")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(f"PHASE 3: Running Table 4 & 5 Ablations on WORLD mix_v2 ({device})")
    print("=" * 80)

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    checkpoint_dir = ROOT_DIR / "checkpoints" / "ablation_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_by_config = {}
    total_runs = len(ABLATION_CONFIGS) * len(SEEDS)
    current_run = 0

    for cfg in ABLATION_CONFIGS:
        cfg_id = cfg["id"]
        aug = cfg["aug"]
        results_by_config[cfg_id] = []

        for s in SEEDS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running {cfg_id} (Seed {s})...")
            send_marimo_toast(f"Phase 3 [{current_run}/{total_runs}]: Training {cfg_id} (Seed {s})")
            res = train_ablation_run(
                cfg_id=cfg_id,
                aug_method=aug,
                seed=s,
                metadata_path=str(meta_cand),
                device=device,
                checkpoint_dir=checkpoint_dir
            )
            results_by_config[cfg_id].append(res)

    # 1. Compile Table 4: Leave-One-Out Ablation
    table4_results = {}
    print("\n" + "=" * 90)
    print("## TABLE 4: LEAVE-ONE-OUT ABLATION ON BIOMECHANICAL MIX V2 (63-D WORLD)")
    print("=" * 90)
    print("| Configuration | Operators | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cfg in ABLATION_CONFIGS:
        if cfg["in_table4"]:
            cfg_id = cfg["id"]
            runs = results_by_config[cfg_id]
            summary = summarize_group(runs)
            summary["description"] = cfg["desc"]
            table4_results[cfg_id] = summary
            print(f"| **{cfg['desc']}** | `{cfg['aug']}` | {summary['val_vid_acc']} | {summary['val_vid_f1']} | {summary['test_win_acc']} | {summary['test_win_f1']} | **{summary['test_vid_acc']}** | **{summary['test_vid_f1']}** |")

    # 2. Compile Table 5: Single-Operator Ablation
    table5_results = {}
    print("\n" + "=" * 90)
    print("## TABLE 5: SINGLE-OPERATOR ABLATION ON BIOMECHANICAL MIX V2 (63-D WORLD)")
    print("=" * 90)
    print("| Configuration | Operator | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for cfg in ABLATION_CONFIGS:
        if cfg["in_table5"]:
            cfg_id = cfg["id"]
            runs = results_by_config[cfg_id]
            summary = summarize_group(runs)
            summary["description"] = cfg["desc"]
            table5_results[cfg_id] = summary
            print(f"| **{cfg['desc']}** | `{cfg['aug']}` | {summary['val_vid_acc']} | {summary['val_vid_f1']} | {summary['test_win_acc']} | {summary['test_win_f1']} | **{summary['test_vid_acc']}** | **{summary['test_vid_f1']}** |")

    # 3. Save Unified Results
    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    with open(outputs_dir / "table4_loo_world_mix_v2.json", "w") as f:
        json.dump(table4_results, f, indent=2)

    with open(outputs_dir / "table5_single_world_mix_v2.json", "w") as f:
        json.dump(table5_results, f, indent=2)

    unified = {
        "table4_leave_one_out": table4_results,
        "table5_single_operator": table5_results,
        "all_runs": results_by_config
    }
    with open(outputs_dir / "augmentation_ablation_results.json", "w") as f:
        json.dump(unified, f, indent=2)

    send_marimo_toast("Phase 3 Complete: Table 4 & 5 Ablations finished successfully!")
    print(f"\nSaved all results to {outputs_dir / 'augmentation_ablation_results.json'}")

if __name__ == "__main__":
    main()
