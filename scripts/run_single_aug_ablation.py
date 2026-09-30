#!/usr/bin/env python3
"""
Single Augmentation Ablation Runner.
Trains each candidate augmentation operator in complete isolation against the Clean Baseline:
1. Baseline (No Aug) - reference
2. + Mirror only (single_mirror)
3. + Yaw only (single_yaw)
4. + Scale only (single_scale)
5. + TimeWarp only (single_timewarp)
6. + Jitter only (single_jitter)

Evaluates on both Validation and Test sets across seeds [42, 123, 3407].
Runs 5 workers concurrently on GPU (1 worker per variant), seed-by-seed.
"""

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SINGLE_AUG_VARIANTS = [
    ("Mirror", "single_mirror"),
    ("Yaw", "single_yaw"),
    ("Scale", "single_scale"),
    ("TimeWarp", "single_timewarp"),
    ("Jitter", "single_jitter"),
]

def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Parses training log to extract best val metrics, min val loss, and test metrics."""
    import re
    if not log_path.exists():
        return {}
    
    best_val_acc = 0.0
    best_val_loss = 999.0
    best_train_loss = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    min_val_loss = 999.0
    min_loss_epoch = 0
    test_acc = None
    test_f1 = None
    
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/\d+ .* - loss: ([\d\.]+) - acc: ([\d\.]+) - val_loss: ([\d\.]+) - val_acc: ([\d\.]+)", line)
            if m:
                ep = int(m.group(1))
                t_loss = float(m.group(2))
                t_acc = float(m.group(3))
                v_loss = float(m.group(4))
                v_acc = float(m.group(5))
                if v_loss < min_val_loss:
                    min_val_loss = v_loss
                    min_loss_epoch = ep
            if "--> Best checkpoint saved (val_acc:" in line:
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_train_loss = t_loss
                best_train_acc = t_acc
                best_epoch = ep
            m_test = re.search(r"Test Accuracy: ([\d\.]+)% \| Macro F1: ([\d\.]+)", line)
            if m_test:
                test_acc = float(m_test.group(1))
                test_f1 = float(m_test.group(2))
                
    return {
        "best_epoch": best_epoch,
        "val_acc": round(best_val_acc * 100.0, 2),
        "val_loss_at_best": round(best_val_loss, 4),
        "min_val_loss": round(min_val_loss, 4),
        "min_loss_epoch": min_loss_epoch,
        "train_loss_at_best": round(best_train_loss, 4),
        "loss_gap": round(best_val_loss - best_train_loss, 4),
        "test_acc": test_acc,
        "test_f1": test_f1
    }

def train_single_task(task: Dict[str, Any], device: str = "cuda", force_retrain: bool = False) -> Dict[str, Any]:
    task_id = task["id"]
    model = task["model"]
    feature = task["feature"]
    aug = task["augment"]
    seed = task["seed"]
    
    # Isolated checkpoint folder per task to avoid concurrent write collisions
    base_ckpt_dir = ROOT_DIR / "checkpoints" / "ablation_aug"
    task_ckpt_dir = base_ckpt_dir / task_id
    task_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    final_ckpt = base_ckpt_dir / f"best_{model}_{feature}_aug_{aug}_seed{seed}.pt"
    
    log_dir = ROOT_DIR / "outputs" / "ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_id}.log"
    
    # Check if already completed (unless force_retrain is requested)
    if not force_retrain and final_ckpt.exists() and log_file.exists():
        parsed = parse_training_log(log_file)
        if parsed.get("test_acc") is not None:
            print(f"[SKIP] {task_id} already completed (Val Acc: {parsed.get('val_acc')}%, Test Acc: {parsed.get('test_acc')}%)", flush=True)
            task["metrics"] = parsed
            return task

    print(f"[START] Training {task_id} on {device} (PID {os.getpid()})...", flush=True)
    t0 = time.time()
    
    cmd = [
        sys.executable, "-u", "-m", "src.cli", "train",
        "--model", model,
        "--feature", feature,
        "--augment", aug,
        "--seed", str(seed),
        "--checkpoint_name", f"{model}_{feature}_aug_{aug}_seed{seed}",
        "--epochs", "100",
        "--patience", "10",
        "--batch_size", "16",
        "--device", device,
        "--checkpoint_dir", str(task_ckpt_dir),
        "--output_dir", str(ROOT_DIR / "outputs" / "ablation_runs" / task_id),
        "--metadata", task.get("metadata", "Final_dataset_metadata.csv"),
        "--landmark_dir", task.get("landmark_dir", "data/landmarks")
    ]
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=str(ROOT_DIR), stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
        
    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"[DONE] {task_id} completed in {elapsed:.1f}s", flush=True)
        # Move best checkpoint to final location if needed
        produced_ckpt = task_ckpt_dir / f"best_{model}_{feature}_aug_{aug}_seed{seed}.pt"
        if not produced_ckpt.exists():
            produced_ckpt = task_ckpt_dir / f"best_{model}_{feature}_aug_{aug}.pt"
        if produced_ckpt.exists() and produced_ckpt != final_ckpt:
            shutil.copy2(produced_ckpt, final_ckpt)
            
        metrics = parse_training_log(log_file)
        task["metrics"] = metrics
        print(f"[RESULT] {task_id} -> Val Acc: {metrics.get('val_acc')}%, Val Loss: {metrics.get('val_loss_at_best')}, Test Acc: {metrics.get('test_acc')}%, Test F1: {metrics.get('test_f1')}", flush=True)
    else:
        print(f"[FAIL] {task_id} failed with code {proc.returncode}. Log: {log_file}", flush=True)
        task["metrics"] = {"error": f"Failed with code {proc.returncode}"}
        
    return task

def run_all(seeds: List[int], workers: int = 5, device: str = "cuda", force_retrain: bool = False):
    all_results = []
    out_json = ROOT_DIR / "outputs" / "single_aug_ablation_results.json"
    
    # Load existing results if any and not force_retrain
    if not force_retrain and out_json.exists():
        try:
            with open(out_json, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = []
            
    for s in seeds:
        print(f"\n==========================================", flush=True)
        print(f"  LAUNCHING BATCH: SEED {s} (5 Variants in Parallel)", flush=True)
        print(f"==========================================", flush=True)
        
        batch_tasks = []
        for var_name, aug_code in SINGLE_AUG_VARIANTS:
            t_id = f"Single_Trans_{var_name}_seed{s}"
            task = {
                "group": "Single_Augmentation",
                "variant": var_name,
                "augment": aug_code,
                "model": "Transformer",
                "feature": "mix",
                "seed": s,
                "id": t_id,
                "metadata": "Final_dataset_metadata.csv",
                "landmark_dir": "data/landmarks"
            }
            batch_tasks.append(task)
            
        with ThreadPoolExecutor(max_workers=min(workers, len(batch_tasks))) as executor:
            futures = {executor.submit(train_single_task, t, device, force_retrain): t for t in batch_tasks}
            for future in as_completed(futures):
                res = future.result()
                # Update all_results
                all_results = [r for r in all_results if r["id"] != res["id"]] + [res]
                # Save incremental JSON
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                    
        print(f"Batch for seed {s} finished! Current saved tasks: {len(all_results)}", flush=True)
        
    print(f"\nAll single augmentation ablation tasks finished! Results saved to {out_json}", flush=True)
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 3407])
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force complete retraining from scratch")
    args = parser.parse_args()
    
    run_all(seeds=args.seeds, workers=args.workers, device=args.device, force_retrain=args.force_retrain)
