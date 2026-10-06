#!/usr/bin/env python3
"""
Phase 2: Refresh Clean Sequence Experiments for Table 2 (Feature Screening)
Trains 12 clean runs (4 configs x 3 seeds: 42, 123, 3407):
  1. LSTM + world_3d (39-d)
  2. LSTM + mix_v2 (63-d)
  3. BiLSTM + world_3d (39-d)
  4. BiLSTM + mix_v2 (63-d)
And compiles with Transformer + world_3d (already completed in Phase 0/1).
Evaluates window-level and video-level metrics on Validation and Held-out Test sets.
Outputs results to outputs/table2_clean_sequences.json.
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
CONFIGS = [
    ("LSTM", "world_3d"),
    ("LSTM", "mix_v2"),
    ("BiLSTM", "world_3d"),
    ("BiLSTM", "mix_v2"),
]

def send_marimo_toast(msg: str):
    try:
        import marimo as mo
        mo.status.toast(msg)
    except Exception:
        pass

def train_and_eval_run(
    model_type: str,
    feature_method: str,
    seed: int,
    metadata_path: str,
    device: torch.device,
    checkpoint_dir: Path
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n---> Training {model_type} on {feature_method} (Seed {seed}) on {device}...")
    t_start = time.time()

    # 1. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method=feature_method,
        batch_size=16,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=None,
        zero_frame_handling="interpolate",
        landmark_dir=None,  # will auto-detect data/world_landmarks
        in_memory=True,
        seed=seed,
        save_norm_artifact=True,
        strict_norm=False
    )

    # 2. Build Model
    model = build_model(model_type=model_type, feature_method=feature_method)

    # 3. Setup Trainer
    seed_ckpt_dir = checkpoint_dir / f"seed{seed}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{model_type}_{feature_method}_seed{seed}"
    best_ckpt_path = seed_ckpt_dir / f"best_{model_name}.pt"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir=str(seed_ckpt_dir),
        model_name=model_name,
        label_smoothing=0.0,
        early_stopping_metric="val_macro_f1",
        use_amp=torch.cuda.is_available(),
        feature_method=feature_method,
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
        "model": model_type,
        "feature": feature_method,
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

    print(f"[{model_type} + {feature_method} | Seed {seed}] Completed in {res['train_time_s']}s:")
    print(f"  Val Win Acc: {res['val_win_acc']}% | Val Vid Acc: {res['val_vid_acc']}%")
    print(f"  Test Win Acc: {res['test_win_acc']}% | Test Vid Acc: {res['test_vid_acc']}% (F1: {res['test_vid_f1']})")

    return res

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Refresh Table 2 Clean Sequence Experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print(f"PHASE 2: Refreshing Clean Sequence Experiments on {device}")
    print("=" * 80)

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    checkpoint_dir = ROOT_DIR / "checkpoints" / "table2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    total_runs = len(CONFIGS) * len(SEEDS)
    current_run = 0

    for model_type, feat in CONFIGS:
        key = f"{model_type}_{feat}"
        all_results[key] = []
        for s in SEEDS:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Running {model_type} + {feat} (Seed {s})...")
            send_marimo_toast(f"Phase 2 [{current_run}/{total_runs}]: Training {model_type} + {feat} (Seed {s})")
            res = train_and_eval_run(
                model_type=model_type,
                feature_method=feat,
                seed=s,
                metadata_path=str(meta_cand),
                device=device,
                checkpoint_dir=checkpoint_dir
            )
            all_results[key].append(res)

    # Load existing Transformer + world_3d benchmark results if available
    trans_world_path = ROOT_DIR / "outputs" / "world_3d_benchmark_results.json"
    if trans_world_path.exists():
        try:
            with open(trans_world_path) as f:
                tw_data = json.load(f)
                trans_seeds = []
                for s_info in tw_data.get("seeds", []):
                    trans_seeds.append({
                        "model": "Transformer",
                        "feature": "world_3d",
                        "seed": s_info["seed"],
                        "train_time_s": s_info.get("train_time_s", 0.0),
                        "best_epoch": s_info.get("best_epoch", 0),
                        "val_loss": s_info.get("val_loss", 0.0),
                        "val_win_acc": s_info.get("val_win_acc", 0.0),
                        "val_win_f1": s_info.get("val_win_f1", 0.0),
                        "val_vid_acc": s_info.get("val_vid_acc", 0.0),
                        "val_vid_f1": s_info.get("val_vid_f1", 0.0),
                        "test_win_acc": s_info.get("test_win_acc", 0.0),
                        "test_win_f1": s_info.get("test_win_f1", 0.0),
                        "test_vid_acc": s_info.get("test_vid_acc", 0.0),
                        "test_vid_f1": s_info.get("test_vid_f1", 0.0),
                    })
                all_results["Transformer_world_3d"] = trans_seeds
        except Exception as e:
            print(f"Warning: Failed to load Transformer world_3d results: {e}")

    # Compute Summary Statistics across seeds
    summary_report = {}
    for key, seed_runs in all_results.items():
        m_type, f_type = key.split("_", 1)
        val_w = [r["val_win_acc"] for r in seed_runs]
        val_v = [r["val_vid_acc"] for r in seed_runs]
        test_w = [r["test_win_acc"] for r in seed_runs]
        test_wf1 = [r["test_win_f1"] for r in seed_runs]
        test_v = [r["test_vid_acc"] for r in seed_runs]
        test_vf1 = [r["test_vid_f1"] for r in seed_runs]

        summary_report[key] = {
            "model": m_type,
            "feature": f_type,
            "dim": 39 if f_type == "world_3d" else 63,
            "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
            "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
            "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
            "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
            "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
            "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
            "runs": seed_runs
        }

    out_file = ROOT_DIR / "outputs" / "table2_clean_sequences.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 90)
    print("## PHASE 2 REFRESHED TABLE 2 CLEAN SEQUENCE BENCHMARK RESULTS")
    print("=" * 90)
    print("| Model | Feature | Dim | Val Win Acc (%) | Val Vid Acc (%) | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for key, rep in summary_report.items():
        print(f"| **{rep['model']}** | `{rep['feature']}` | {rep['dim']} | {rep['val_win_acc']} | {rep['val_vid_acc']} | {rep['test_win_acc']} | {rep['test_win_f1']} | **{rep['test_vid_acc']}** | **{rep['test_vid_f1']}** |")
    print("=" * 90)

    send_marimo_toast("Phase 2 Complete: Table 2 Clean Sequences refreshed successfully!")
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    main()
