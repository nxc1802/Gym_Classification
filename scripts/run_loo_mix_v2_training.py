#!/usr/bin/env python3
"""
Dedicated Leave-One-Out (Table 4) Training & Evaluation Runner on Biomechanical Mix v2 (mix_v2, 63-d).

Executes the 12 missing LOO runs across 3 seeds (42, 123, 3407):
1. Minus Mirror (skel_gym_aug_no_mirror) - 3 seeds
2. Minus Yaw (skel_gym_aug_no_yaw) - 3 seeds
3. Minus Scale (skel_gym_aug_no_scale) - 3 seeds
4. Minus Jitter (skel_gym_aug_no_jitter) - 3 seeds

Reuses existing Table 5 checkpoints for:
- Candidate Full (5-op)
- Minus TimeWarp / SkelGym-Aug 4-op
- Clean Baseline (No Aug)

Merges all results into outputs/augmentation_ablation_results.json under summary_leave_one_out.
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import NUM_CLASSES
from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.reproducibility import load_checkpoint_weights

SEEDS = [42, 123, 3407]

LOO_TARGET_VARIANTS = [
    ("Minus_Mirror", "skel_gym_aug_no_mirror", "w/o Sagittal Reflection (-Mirror)", "Geometric"),
    ("Minus_Yaw", "skel_gym_aug_no_yaw", "w/o Gravitational Yaw (-Yaw)", "Geometric"),
    ("Minus_Scale", "skel_gym_aug_no_scale", "w/o Proportional Scaling (-Scale)", "Geometric"),
    ("Minus_Jitter", "skel_gym_aug_no_jitter", "w/o Sensor Jitter (-Jitter)", "Noise-based"),
]

def parse_val_from_log(log_path: Path) -> Dict[str, Any]:
    """Extracts validation metrics from training log."""
    import re
    if not log_path.exists():
        return {}
    best_val_acc = 0.0
    best_val_loss = 999.0
    best_train_loss = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/\d+ .* - loss: ([\d\.]+) - acc: ([\d\.]+) - val_loss: ([\d\.]+) - val_acc: ([\d\.]+)", line)
            if m:
                ep = int(m.group(1))
                t_loss = float(m.group(2))
                t_acc = float(m.group(3))
                v_loss = float(m.group(4))
                v_acc = float(m.group(5))
            if "--> Best checkpoint saved" in line:
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_train_loss = t_loss
                best_train_acc = t_acc
                best_epoch = ep
    return {
        "val_acc": round(best_val_acc * 100.0, 2),
        "val_loss": round(best_val_loss, 4),
        "train_loss": round(best_train_loss, 4),
        "train_acc": round(best_train_acc * 100.0, 2),
        "best_epoch": best_epoch,
        "loss_gap": round(best_val_loss - best_train_loss, 4)
    }

def train_one_task(
    var_name: str,
    aug_method: str,
    seed: int,
    metadata_path: str,
    landmark_dir: str,
    device: str = "cuda",
    force_retrain: bool = False
) -> Path:
    task_id = f"LOO_Trans_{var_name}_seed{seed}"
    ckpt_dir = ROOT_DIR / "checkpoints" / "ablation_aug" / task_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / f"best_{task_id}.pt"
    log_dir = ROOT_DIR / "outputs" / "ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_id}.log"

    if not force_retrain and best_ckpt.exists() and best_ckpt.stat().st_size > 1000:
        print(f"[SKIP] {task_id} already exists ({best_ckpt.stat().st_size/1e6:.1f} MB)", flush=True)
        return best_ckpt

    print(f"\n[TRAIN START] {task_id} (Aug: {aug_method}, Seed: {seed}) on {device}...", flush=True)
    t0 = time.time()
    cmd = [
        sys.executable, "-u", "-m", "src.cli", "train",
        "--model", "Transformer",
        "--feature", "mix_v2",
        "--augment", aug_method,
        "--seed", str(seed),
        "--checkpoint_name", task_id,
        "--epochs", "100",
        "--lr", "0.0001",
        "--patience", "10",
        "--batch_size", "16",
        "--label_smoothing", "0.05",
        "--early_stopping_metric", "val_macro_f1",
        "--train_stride", "16",
        "--val_test_stride", "32",
        "--device", device,
        "--checkpoint_dir", str(ckpt_dir),
        "--output_dir", str(ROOT_DIR / "outputs" / "ablation_runs" / task_id),
        "--metadata", metadata_path,
        "--landmark_dir", landmark_dir,
        "--no_test_eval"
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_file, "w", encoding="utf-8") as lf:
        res = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)

    elapsed = time.time() - t0
    if res.returncode != 0:
        raise RuntimeError(f"Training {task_id} failed with return code {res.returncode}. Log: {log_file}")
    print(f"[TRAIN DONE] {task_id} completed in {elapsed:.1f}s", flush=True)
    return best_ckpt

def evaluate_task(
    ckpt_path: Path,
    metadata_path: str,
    landmark_dir: str,
    seed: int,
    device: torch.device
) -> Dict[str, Any]:
    state_dict, _ = load_checkpoint_weights(ckpt_path, device="cpu")
    model = build_model("Transformer", "mix_v2", num_classes=NUM_CLASSES, transformer_variant="dual_branch")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load dataloaders for this seed
    _, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="mix_v2",
        batch_size=32,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        landmark_dir=landmark_dir,
        num_workers=0,
        in_memory=True,
        seed=seed,
        strict_norm=False
    )
    trainer = Trainer(model=model, device=device)

    # 1. Validation evaluation
    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    # 2. Test evaluation (Held-out)
    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

    # Provenance metrics if available
    prov_file = ckpt_path.with_suffix(".provenance.json")
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

    return {
        "val_acc": round(float(val_m["accuracy"] * 100.0), 2),
        "val_loss": round(float(val_loss), 4) if val_loss is not None else 0.0,
        "val_macro_f1": round(float(val_m["macro_f1"]), 4),
        "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
        "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
        "win_acc": float(test_m["accuracy"] * 100.0),
        "win_f1": float(test_m["macro_f1"]),
        "vid_acc": float(test_vid_m["accuracy"] * 100.0),
        "vid_f1": float(test_vid_m["macro_f1"]),
        "best_epoch": best_epoch or 0
    }

def main():
    parser = argparse.ArgumentParser(description="Run 12 LOO training runs on mix_v2")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain existing checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"
    metadata_path = str(meta_cand)
    landmark_dir = str(ROOT_DIR / "data" / "landmarks")

    device = torch.device(args.device)
    print(f"============================================================")
    print(f"=== SKELGYM LOO mix_v2 (63-d) BENCHMARK RUNNER           ===")
    print(f"=== Device: {device} | Metadata: {metadata_path} ===")
    print(f"============================================================")

    results_json = ROOT_DIR / "outputs" / "augmentation_ablation_results.json"
    existing_data = {}
    if results_json.exists():
        with open(results_json, "r") as f:
            existing_data = json.load(f)

    raw_tasks = existing_data.get("raw_tasks", {})
    summary_single = existing_data.get("summary_single_component", {})

    loo_eval_results = {}

    # Run the 12 tasks
    total_start = time.time()
    for var_name, aug_method, disp_name, domain in LOO_TARGET_VARIANTS:
        loo_eval_results[var_name] = []
        print(f"\n>>> Running Variant: {var_name} ({disp_name}) <<<")
        for seed in SEEDS:
            task_id = f"LOO_Trans_{var_name}_seed{seed}"
            # 1. Train
            ckpt_path = train_one_task(
                var_name=var_name,
                aug_method=aug_method,
                seed=seed,
                metadata_path=metadata_path,
                landmark_dir=landmark_dir,
                device=args.device,
                force_retrain=args.force_retrain
            )
            # 2. Evaluate
            metrics = evaluate_task(
                ckpt_path=ckpt_path,
                metadata_path=metadata_path,
                landmark_dir=landmark_dir,
                seed=seed,
                device=device
            )
            loo_eval_results[var_name].append(metrics)
            raw_tasks[task_id] = {
                "group": "Leave_One_Out",
                "variant": var_name,
                "display_name": disp_name,
                "domain": domain,
                "model": "Transformer",
                "feature": "mix_v2",
                "augment": aug_method,
                "seed": seed,
                "checkpoint_path": str(ckpt_path.relative_to(ROOT_DIR)),
                "metrics": metrics
            }
            print(f"  [EVAL] {task_id} -> ValWin: {metrics['val_acc']}%, TestWin: {metrics['win_acc']:.2f}%, TestVid: {metrics['vid_acc']:.2f}%", flush=True)

    # Now compute aggregated summary_leave_one_out
    summary_loo = {}

    # 1. Candidate Full (5-op): Reuse from Table 5
    if "Candidate_Full_5op" in summary_single:
        ref_cand = dict(summary_single["Candidate_Full_5op"])
        ref_cand["display_name"] = "Candidate Full (All 5 Operators)"
        ref_cand["domain"] = "None (Reference Suite)"
        summary_loo["Candidate_Full_5op"] = ref_cand
        ref_test_acc = ref_cand["win_acc_mean"]
        print(f"\n[REUSED] Candidate Full (5-op): Test Win Acc = {ref_test_acc:.2f}% ± {ref_cand['win_acc_std']:.2f}%")
    else:
        ref_test_acc = 69.24

    # 2-5: The 4 evaluated LOO variants
    for var_name, aug_method, disp_name, domain in LOO_TARGET_VARIANTS:
        eval_list = loo_eval_results[var_name]
        w_accs = [m["win_acc"] for m in eval_list]
        w_f1s = [m["win_f1"] for m in eval_list]
        v_accs = [m["vid_acc"] for m in eval_list]
        v_f1s = [m["vid_f1"] for m in eval_list]
        val_accs = [m["val_acc"] for m in eval_list]
        val_losses = [m["val_loss"] for m in eval_list]
        val_f1s = [m["val_macro_f1"] for m in eval_list]

        summary_loo[var_name] = {
            "count": len(eval_list),
            "display_name": disp_name,
            "domain": domain,
            "win_acc_mean": float(np.mean(w_accs)),
            "win_acc_std": float(np.std(w_accs)),
            "win_f1_mean": float(np.mean(w_f1s)),
            "win_f1_std": float(np.std(w_f1s)),
            "vid_acc_mean": float(np.mean(v_accs)),
            "vid_acc_std": float(np.std(v_accs)),
            "vid_f1_mean": float(np.mean(v_f1s)),
            "vid_f1_std": float(np.std(v_f1s)),
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_std": float(np.std(val_accs)),
            "val_loss_mean": float(np.mean(val_losses)),
            "val_loss_std": float(np.std(val_losses)),
            "val_macro_f1_mean": float(np.mean(val_f1s)),
            "val_macro_f1_std": float(np.std(val_f1s))
        }

    # 6. Minus TimeWarp / SkelGym-Aug 4-op: Reuse from Table 5
    if "SkelGym_Aug_4op" in summary_single:
        ref_tw = dict(summary_single["SkelGym_Aug_4op"])
        ref_tw["display_name"] = "w/o Temporal TimeWarp (-TimeWarp / SkelGym-Aug)"
        ref_tw["domain"] = "Temporal"
        summary_loo["Minus_TimeWarp"] = ref_tw
        print(f"[REUSED] Minus TimeWarp (4-op): Test Win Acc = {ref_tw['win_acc_mean']:.2f}% ± {ref_tw['win_acc_std']:.2f}%")

    # 7. Clean Baseline (No Aug): Reuse from Table 5
    if "Clean_Baseline_NoAug" in summary_single:
        ref_clean = dict(summary_single["Clean_Baseline_NoAug"])
        ref_clean["display_name"] = "Clean Baseline (No Augmentation)"
        ref_clean["domain"] = "All Operators Excluded"
        summary_loo["Clean_Baseline_NoAug"] = ref_clean
        print(f"[REUSED] Clean Baseline (No Aug): Test Win Acc = {ref_clean['win_acc_mean']:.2f}% ± {ref_clean['win_acc_std']:.2f}%")

    # Save to JSON
    existing_data["summary_leave_one_out"] = summary_loo
    existing_data["raw_tasks"] = raw_tasks
    existing_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(results_json, "w") as f:
        json.dump(existing_data, f, indent=2)
    print(f"\n[SAVED] Successfully updated {results_json}")

    # Print Table 4 in Markdown
    print("\n" + "=" * 80)
    print("## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix v2")
    print("=" * 80)
    print("| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Win Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Macro F1 | Delta vs Full (Test Win) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    display_order = [
        ("Candidate_Full_5op", "None (Reference Suite)"),
        ("Minus_Jitter", "Gaussian Coordinate Jitter (sigma=0.008)"),
        ("Minus_Mirror", "Sagittal Horizontal Flip (p=0.5)"),
        ("Minus_Yaw", "Gravitational Yaw Rotation (+/- 15 deg)"),
        ("Minus_Scale", "Proportional Scale Variation (+/- 10%)"),
        ("Minus_TimeWarp", "Temporal Resampling (0.8x - 1.2x)"),
        ("Clean_Baseline_NoAug", "All Operators Excluded"),
    ]

    ref_acc = summary_loo.get("Candidate_Full_5op", {}).get("win_acc_mean", 69.24)
    for var_k, domain_desc in display_order:
        row = summary_loo.get(var_k)
        if not row:
            continue
        v_loss = f"{row.get('val_loss_mean', 0.0):.4f}"
        v_acc = f"{row.get('val_acc_mean', 0.0):.2f}% ± {row.get('val_acc_std', 0.0):.2f}%"
        v_f1 = f"{row.get('val_macro_f1_mean', 0.0):.4f} ± {row.get('val_macro_f1_std', 0.0):.4f}"
        t_acc = f"{row.get('win_acc_mean', 0.0):.2f}% ± {row.get('win_acc_std', 0.0):.2f}%"
        t_f1 = f"{row.get('win_f1_mean', 0.0):.4f} ± {row.get('win_f1_std', 0.0):.4f}"
        cur_acc = row.get("win_acc_mean", 0.0)
        diff = cur_acc - ref_acc
        delta = f"{diff:+.2f}%" if var_k != "Candidate_Full_5op" else "0.00% (Ref)"
        print(f"| **{row.get('display_name')}** | {domain_desc} | {v_loss} | {v_acc} | {v_f1} | {t_acc} | {t_f1} | {delta} |")

    print(f"\nAll 12 runs completed in {time.time() - total_start:.1f}s!")

if __name__ == "__main__":
    main()
