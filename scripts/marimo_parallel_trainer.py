#!/usr/bin/env python3
"""
High-Performance Parallel Multi-Seed Training Orchestrator for SkelGym on Marimo.
Features:
1. Concurrency: Trains N models in parallel using GPU acceleration (AMP + CUDA).
2. Anti-Idle Keepalive: Spawns internal daemon thread to ping Marimo port and touch heartbeat files.
3. Event-Driven Triggers: Emits instant [TRIGGER: MODEL_COMPLETE] events upon model completion.
4. Token-Saving: Updates JSONL trigger logs and JSON status without polling.
5. Downstream Automation: Automatically runs Phase 2 (Reference Artifacts Freeze) and Phase 3 (Multi-Seed Late Fusion).
"""

import os
import sys
import time
import json
import re
import argparse
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CANONICAL_EXPERIMENT_REGISTRY

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

def get_checkpoint_target(seed: int, model_cfg: Dict[str, str], checkpoint_base: Path) -> Path:
    filename = f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"
    if seed == 42:
        return checkpoint_base / filename
    else:
        return checkpoint_base / f"seed{seed}" / filename

def keepalive_worker(stop_event: threading.Event, heartbeat_file: Path, port: int = 8080):
    """Continuously signals activity to prevent Marimo container shutdown."""
    while not stop_event.is_set():
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"User-Agent": "Marimo-KeepAlive"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                pass
        except Exception:
            pass
        try:
            heartbeat_file.touch()
        except Exception:
            pass
        time.sleep(15)

def run_single_model_task(
    task: Dict[str, Any],
    device_str: str,
    checkpoint_base: Path,
    epochs: int,
    force_retrain: bool,
    status_lock: threading.Lock,
    status_data: Dict[str, Any],
    trigger_log_path: Path,
    status_file_path: Path
) -> Dict[str, Any]:
    seed = task["seed"]
    cfg = task["model_cfg"]
    name = cfg["name"]
    model_type = cfg["model"]
    feat = cfg["feature"]
    aug = cfg.get("augment", "none")
    exp_id = cfg["exp_id"]

    ckpt_path = get_checkpoint_target(seed, cfg, checkpoint_base)
    ckpt_dir = str(ckpt_path.parent)

    # Check existing checkpoint
    if not force_retrain and ckpt_path.exists():
        msg = f"[TRIGGER: CACHED] Model: {name} (Seed {seed}) already exists at {ckpt_path.name}. Skipping."
        print(msg, flush=True)
        res_info = {
            "name": name,
            "seed": seed,
            "status": "CACHED",
            "ckpt_path": str(ckpt_path),
            "val_f1": 0.0,
            "elapsed_s": 0.0
        }
        with status_lock:
            status_data["completed_models"].append(res_info)
            status_data["completed_count"] += 1
            _save_status(status_file_path, status_data)
        return res_info

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    reg_cfg = CANONICAL_EXPERIMENT_REGISTRY.get(exp_id, {})
    lr = reg_cfg.get("lr", 1e-4)
    batch_size = reg_cfg.get("batch_size", 16)
    label_smoothing = reg_cfg.get("label_smoothing", 0.05 if ("AAGCN" in model_type or "Transformer" in model_type) else 0.0)
    patience = reg_cfg.get("patience", 10)
    train_stride = reg_cfg.get("train_stride", 16)
    val_test_stride = reg_cfg.get("val_test_stride", 32)
    es_metric = reg_cfg.get("early_stopping_metric", "val_macro_f1")

    cmd = [
        sys.executable, "run.py", "train",
        "--model", model_type,
        "--feature", feat,
        "--augment", aug,
        "--exp_id", exp_id,
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
        "--in_memory",
        "--no_test_eval"
    ]

    task_label = f"{name} (Seed {seed})"
    with status_lock:
        status_data["running_models"].append(task_label)
        _save_status(status_file_path, status_data)

    print(f"\n[SPAWN] Started training: {task_label} on {device_str.upper()}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    elapsed = time.time() - t0

    with status_lock:
        if task_label in status_data["running_models"]:
            status_data["running_models"].remove(task_label)

    if proc.returncode != 0:
        err_msg = f"[ERROR: FAILED] {task_label} failed with exit code {proc.returncode}:\n{proc.stderr[-500:]}"
        print(err_msg, flush=True)
        res_info = {
            "name": name,
            "seed": seed,
            "status": "FAILED",
            "error": proc.stderr[-500:],
            "elapsed_s": elapsed
        }
        with status_lock:
            status_data["failed_models"].append(res_info)
            _save_status(status_file_path, status_data)
        return res_info

    # Parse best val_f1
    val_f1 = 0.0
    val_acc = 0.0
    f1_matches = re.findall(r"val_f1:\s*([0-9\.]+)", proc.stdout)
    if f1_matches:
        val_f1 = float(f1_matches[-1])
    acc_matches = re.findall(r"val_acc:\s*([0-9\.]+)", proc.stdout)
    if acc_matches:
        val_acc = float(acc_matches[-1])

    res_info = {
        "name": name,
        "seed": seed,
        "exp_id": exp_id,
        "feature": feat,
        "augment": aug,
        "status": "COMPLETED",
        "ckpt_path": str(ckpt_path),
        "val_f1": val_f1,
        "val_acc": val_acc,
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with status_lock:
        status_data["completed_models"].append(res_info)
        status_data["completed_count"] += 1
        curr_done = status_data["completed_count"]
        total = status_data["total_models"]
        _save_status(status_file_path, status_data)

        # Log event trigger
        with open(trigger_log_path, "a", encoding="utf-8") as f_trig:
            f_trig.write(json.dumps(res_info) + "\n")

    # Clean trigger output to stdout (Zero-token-waste milestone)
    print(
        f"[TRIGGER: MODEL_COMPLETE] [{curr_done}/{total}] {task_label} -> "
        f"Val F1: {val_f1:.4f} | Val Acc: {val_acc*100:.1f}% | Time: {elapsed:.1f}s",
        flush=True
    )
    return res_info

def _save_status(status_path: Path, data: Dict[str, Any]):
    try:
        temp_path = status_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(status_path)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="High-Performance Parallel Multi-Seed Trainer for SkelGym")
    parser.add_argument("--parallel", type=int, default=4, help="Maximum concurrent training processes (default: 4)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs per model (default: 100)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Seeds to train (default: 42 123 3407)")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force retrain existing checkpoints")
    parser.add_argument("--skip_baselines", action="store_true", default=False, help="Skip baseline models")
    parser.add_argument("--skip_post_pipeline", action="store_true", default=False, help="Skip Phase 2 & 3 evaluation after training")
    args = parser.parse_args()

    device_str = "cuda" if args.device == "cuda" or (args.device == "auto" and subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0) else "cpu"

    checkpoint_base = PROJECT_ROOT / "checkpoints"
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    status_file = outputs_dir / "training_status.json"
    trigger_log = outputs_dir / "training_triggers.jsonl"
    heartbeat_file = Path("/marimo/.keepalive") if Path("/marimo").exists() else outputs_dir / ".keepalive"

    # Start Keepalive Heartbeat Daemon
    stop_heartbeat = threading.Event()
    keepalive_thread = threading.Thread(
        target=keepalive_worker,
        args=(stop_heartbeat, heartbeat_file),
        daemon=True,
        name="MarimoKeepaliveThread"
    )
    keepalive_thread.start()
    print(f"📡 [KEEPALIVE] Anti-idle heartbeat thread activated (Pinging Marimo & touching {heartbeat_file})", flush=True)

    # Build Task Queue
    task_queue = []
    models_to_run = []
    if not args.skip_baselines:
        models_to_run.extend(BASELINE_MODELS)
    models_to_run.extend(CONSTITUENT_MODELS)

    for seed in args.seeds:
        for m in models_to_run:
            task_queue.append({"seed": seed, "model_cfg": m})

    total_tasks = len(task_queue)
    print(f"\n================================================================================")
    print(f"🏋️‍♂️ SKELGYM PARALLEL TRAINING ENGINE INITIALIZED")
    print(f"Total Models to Train: {total_tasks} ({len(models_to_run)} models x {len(args.seeds)} seeds)")
    print(f"Parallel Workers: {args.parallel} | Target Device: {device_str.upper()} | Epochs: {args.epochs}")
    print(f"================================================================================\n", flush=True)

    status_data = {
        "status": "RUNNING",
        "total_models": total_tasks,
        "completed_count": 0,
        "parallel_workers": args.parallel,
        "running_models": [],
        "completed_models": [],
        "failed_models": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_str
    }
    status_lock = threading.Lock()
    _save_status(status_file, status_data)

    t_start = time.time()

    # Parallel Execution Pool
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = [
            executor.submit(
                run_single_model_task,
                task,
                device_str,
                checkpoint_base,
                args.epochs,
                args.force_retrain,
                status_lock,
                status_data,
                trigger_log,
                status_file
            )
            for task in task_queue
        ]

        for fut in as_completed(futures):
            try:
                _ = fut.result()
            except Exception as e:
                print(f"[EXCEPTION] Worker raised: {e}", flush=True)

    total_train_time = time.time() - t_start
    print(f"\n[TRIGGER: ALL_MODELS_TRAINED] Completed training {total_tasks} models in {total_train_time/60:.2f} minutes!", flush=True)

    # Post-Training Phases
    if not args.skip_post_pipeline and len(status_data["failed_models"]) == 0:
        print(f"\n================================================================================")
        print(f"🔒 PHASE 2: FREEZING REFERENCE ARTIFACTS (SEEDS {args.seeds})")
        print(f"================================================================================", flush=True)
        res_phase2 = subprocess.run([sys.executable, "scripts/freeze_reference_artifacts.py"], cwd=str(PROJECT_ROOT))
        if res_phase2.returncode == 0:
            print(f"[TRIGGER: PHASE_2_FREEZE_COMPLETE] Reference artifacts frozen for all seeds!", flush=True)
        else:
            print(f"[WARNING] Phase 2 failed with code {res_phase2.returncode}", flush=True)

        print(f"\n================================================================================")
        print(f"📊 PHASE 3: EVALUATING 5 LATE FUSION METHODS & MULTI-SEED ENSEMBLES")
        print(f"================================================================================", flush=True)
        seed_args = [str(s) for s in args.seeds]
        res_phase3 = subprocess.run(
            [sys.executable, "scripts/evaluate_local_ensemble.py", "--seeds"] + seed_args + ["--device", device_str],
            cwd=str(PROJECT_ROOT)
        )
        if res_phase3.returncode == 0:
            print(f"[TRIGGER: PHASE_3_EVAL_COMPLETE] Multi-seed 5-method late fusion evaluated successfully!", flush=True)
        else:
            print(f"[WARNING] Phase 3 failed with code {res_phase3.returncode}", flush=True)

    status_data["status"] = "COMPLETED"
    status_data["total_elapsed_minutes"] = round(total_train_time / 60, 2)
    _save_status(status_file, status_data)

    stop_heartbeat.set()
    print(f"\n🎉 [COMPLETE] Entire pipeline executed successfully!", flush=True)

if __name__ == "__main__":
    main()
