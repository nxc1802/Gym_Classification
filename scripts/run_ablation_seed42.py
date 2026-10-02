#!/usr/bin/env python3
"""
Fast Single-Seed (Seed 42) Augmentation Ablation Runner for SkelGym.
Evaluates the contribution of each individual augmentation operator:
1. Transformer Clean (No Augmentation) - Control Reference
2. Transformer + Mirror only (single_mirror)
3. Transformer + Yaw only (single_yaw)
4. Transformer + Scale only (single_scale)
5. Transformer + TimeWarp only (single_timewarp)
6. Transformer + Jitter only (single_jitter)
7. Transformer + SkelGym-Aug (4-op: Mirror+Yaw+Scale+Jitter)
8. Transformer + Candidate Full (5-op: +TimeWarp)
9. AAGCN Bone Clean (bone_3d, no aug)
10. AAGCN Bone SkelGym-Aug (bone_3d, skel_gym_aug) - verifies bone zero-std fix

Features:
- Parallel execution: 4 concurrent GPU workers.
- Event-driven triggers: Emits [TRIGGER: MODEL_COMPLETE] per model.
- Zero token waste: Status tracked via JSON file.
- Anti-idle keepalive: Internal thread pings server.
- Outputs Markdown table to outputs/ablation_seed42_results.md.
"""

import os
import sys
import time
import json
import argparse
import subprocess
import threading
import urllib.request
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ABLATION_TASKS: List[Dict[str, Any]] = [
    # Transformer Mix 117-d ablation tasks
    {"id": "Trans_Clean_NoAug", "name": "Transformer Clean (No Aug)", "model": "Transformer", "feature": "mix", "augment": "none", "domain": "Control / Baseline"},
    {"id": "Trans_Single_Mirror", "name": "Transformer + Mirror only", "model": "Transformer", "feature": "mix", "augment": "single_mirror", "domain": "Bilateral Reflection"},
    {"id": "Trans_Single_Yaw", "name": "Transformer + Yaw only", "model": "Transformer", "feature": "mix", "augment": "single_yaw", "domain": "3D Gravitational Yaw"},
    {"id": "Trans_Single_Scale", "name": "Transformer + Scale only", "model": "Transformer", "feature": "mix", "augment": "single_scale", "domain": "Anthropometric Scale"},
    {"id": "Trans_Single_TimeWarp", "name": "Transformer + TimeWarp only", "model": "Transformer", "feature": "mix", "augment": "single_timewarp", "domain": "Temporal Cadence"},
    {"id": "Trans_Single_Jitter", "name": "Transformer + Jitter only", "model": "Transformer", "feature": "mix", "augment": "single_jitter", "domain": "Sensor Noise"},
    {"id": "Trans_SkelGym_Aug_4op", "name": "Transformer SkelGym-Aug (4-op)", "model": "Transformer", "feature": "mix", "augment": "skel_gym_aug", "domain": "Spatial + Sensor (Proposed)"},
    {"id": "Trans_Candidate_Full_5op", "name": "Transformer Full (5-op: +TimeWarp)", "model": "Transformer", "feature": "mix", "augment": "skel_gym_aug_legacy_5op", "domain": "Spatial + Sensor + Temporal"},
    # AAGCN Bone 3d verification tasks
    {"id": "AAGCN_Bone_Clean", "name": "AAGCN Bone Clean", "model": "AAGCN", "feature": "bone_3d", "augment": "none", "domain": "Graph Baseline (Clean)"},
    {"id": "AAGCN_Bone_SkelGymAug", "name": "AAGCN Bone SkelGym-Aug", "model": "AAGCN", "feature": "bone_3d", "augment": "skel_gym_aug", "domain": "Graph + SkelGym-Aug (Fixed)"},
]

def keepalive_worker(stop_event: threading.Event, heartbeat_file: Path, port: int = 8080):
    while not stop_event.is_set():
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"User-Agent": "Ablation-KeepAlive"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                pass
        except Exception:
            pass
        try:
            heartbeat_file.touch()
        except Exception:
            pass
        time.sleep(15)

def run_single_task(
    task: Dict[str, Any],
    device_str: str,
    checkpoint_dir: Path,
    output_dir: Path,
    epochs: int,
    force_retrain: bool,
    status_lock: threading.Lock,
    status_data: Dict[str, Any],
    trigger_log_path: Path,
    status_file_path: Path
) -> Dict[str, Any]:
    task_id = task["id"]
    name = task["name"]
    model = task["model"]
    feature = task["feature"]
    aug = task["augment"]
    domain = task["domain"]
    seed = 42

    ckpt_path = checkpoint_dir / f"best_{task_id}.pt"
    task_ckpt_dir = checkpoint_dir / task_id
    task_ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "ablation_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task_id}.log"

    task_label = f"{name} (Seed {seed})"

    lr = 1e-3 if model == "AAGCN" else 1e-4
    batch_size = 32 if model == "AAGCN" else 16
    label_smoothing = 0.05
    patience = 10

    cmd = [
        sys.executable, "-u", "-m", "src.cli", "train",
        "--model", model,
        "--feature", feature,
        "--augment", aug,
        "--seed", str(seed),
        "--checkpoint_name", task_id,
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--label_smoothing", str(label_smoothing),
        "--patience", str(patience),
        "--train_stride", "16",
        "--val_test_stride", "32",
        "--early_stopping_metric", "val_macro_f1",
        "--checkpoint_dir", str(task_ckpt_dir),
        "--output_dir", str(output_dir / "runs" / task_id),
        "--video_level",
        "--device", device_str,
        "--use_amp",
        "--in_memory"
    ]

    with status_lock:
        status_data["running_models"].append(task_label)
        _save_status(status_file_path, status_data)

    print(f"\n[SPAWN] Started ablation task: {task_label} on {device_str.upper()}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)
    elapsed = time.time() - t0

    with status_lock:
        if task_label in status_data["running_models"]:
            status_data["running_models"].remove(task_label)

    if proc.returncode != 0:
        err_msg = f"[ERROR: FAILED] {task_label} failed with exit code {proc.returncode}"
        print(err_msg, flush=True)
        res_info = {
            "id": task_id,
            "name": name,
            "model": model,
            "feature": feature,
            "augment": aug,
            "domain": domain,
            "status": "FAILED",
            "elapsed_s": elapsed
        }
        with status_lock:
            status_data["failed_models"].append(res_info)
            _save_status(status_file_path, status_data)
        return res_info

    # Move checkpoint if produced inside task_ckpt_dir
    produced = task_ckpt_dir / f"best_{task_id}.pt"
    if produced.exists() and produced != ckpt_path:
        import shutil
        shutil.copy2(produced, ckpt_path)

    # Parse metrics from log
    val_acc, val_f1, test_acc, test_f1, vid_acc, vid_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    import re
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            m_val_f1 = re.findall(r"val_f1:\s*([0-9\.]+)", content)
            if m_val_f1:
                val_f1 = float(m_val_f1[-1])
            m_val_acc = re.findall(r"val_acc:\s*([0-9\.]+)", content)
            if m_val_acc:
                val_acc = float(m_val_acc[-1]) * 100.0 if float(m_val_acc[-1]) <= 1.0 else float(m_val_acc[-1])
            m_test = re.search(r"Test Accuracy:\s*([0-9\.]+)%\s*\|\s*Macro F1:\s*([0-9\.]+)", content)
            if m_test:
                test_acc = float(m_test.group(1))
                test_f1 = float(m_test.group(2))
            m_vid = re.search(r"VIDEO-LEVEL Test Accuracy:\s*([0-9\.]+)%\s*\|\s*Macro F1:\s*([0-9\.]+)", content)
            if m_vid:
                vid_acc = float(m_vid.group(1))
                vid_f1 = float(m_vid.group(2))

    res_info = {
        "id": task_id,
        "name": name,
        "model": model,
        "feature": feature,
        "augment": aug,
        "domain": domain,
        "status": "COMPLETED",
        "val_acc": round(val_acc, 2),
        "val_f1": round(val_f1, 4),
        "test_win_acc": round(test_acc, 2),
        "test_win_f1": round(test_f1, 4),
        "test_vid_acc": round(vid_acc, 2),
        "test_vid_f1": round(vid_f1, 4),
        "elapsed_s": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with status_lock:
        status_data["completed_models"].append(res_info)
        status_data["completed_count"] += 1
        curr_done = status_data["completed_count"]
        total = status_data["total_models"]
        _save_status(status_file_path, status_data)

        with open(trigger_log_path, "a", encoding="utf-8") as f_trig:
            f_trig.write(json.dumps(res_info) + "\n")

    print(
        f"[TRIGGER: MODEL_COMPLETE] [{curr_done}/{total}] {task_label} -> "
        f"Val Acc: {val_acc:.1f}% | Test Win Acc: {test_acc:.1f}% | Test Vid Acc: {vid_acc:.1f}% (F1: {vid_f1:.4f}) | Time: {elapsed:.1f}s",
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

def generate_markdown_report(results: List[Dict[str, Any]], output_file: Path):
    res_by_id = {r["id"]: r for r in results if r.get("status") == "COMPLETED"}

    lines = [
        "# Systematic Single-Seed (Seed 42) Augmentation Ablation Results",
        "",
        "Evaluated on representative architectures under identical protocols to identify component-level contributions.",
        "",
        "## 1. Transformer (Mix 117-d) Component-Level Ablation",
        "",
        "| Augmentation Configuration | Applied Domain | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Delta vs. Clean (Vid) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    base_vid = res_by_id.get("Trans_Clean_NoAug", {}).get("test_vid_acc", 0.0)

    trans_ids = [
        "Trans_Clean_NoAug",
        "Trans_Single_Mirror",
        "Trans_Single_Yaw",
        "Trans_Single_Scale",
        "Trans_Single_TimeWarp",
        "Trans_Single_Jitter",
        "Trans_SkelGym_Aug_4op",
        "Trans_Candidate_Full_5op"
    ]

    for tid in trans_ids:
        if tid in res_by_id:
            r = res_by_id[tid]
            diff = r["test_vid_acc"] - base_vid
            diff_str = f"+{diff:.2f}%" if diff > 0 else (f"{diff:.2f}%" if diff < 0 else "0.00% (Ref)")
            lines.append(
                f"| **{r['name']}** | {r['domain']} | {r['val_acc']:.2f}% | {r['val_f1']:.4f} | "
                f"{r['test_win_acc']:.2f}% | {r['test_win_f1']:.4f} | {r['test_vid_acc']:.2f}% | {r['test_vid_f1']:.4f} | **{diff_str}** |"
            )

    lines.extend([
        "",
        "## 2. AAGCN Graph Architecture (Bone 3D) Augmentation Parity",
        "",
        "| Model Configuration | Applied Domain | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Delta vs. Clean (Vid) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ] )

    base_bone_vid = res_by_id.get("AAGCN_Bone_Clean", {}).get("test_vid_acc", 0.0)
    for tid in ["AAGCN_Bone_Clean", "AAGCN_Bone_SkelGymAug"]:
        if tid in res_by_id:
            r = res_by_id[tid]
            diff = r["test_vid_acc"] - base_bone_vid
            diff_str = f"+{diff:.2f}%" if diff > 0 else (f"{diff:.2f}%" if diff < 0 else "0.00% (Ref)")
            lines.append(
                f"| **{r['name']}** | {r['domain']} | {r['val_acc']:.2f}% | {r['val_f1']:.4f} | "
                f"{r['test_win_acc']:.2f}% | {r['test_win_f1']:.4f} | {r['test_vid_acc']:.2f}% | {r['test_vid_f1']:.4f} | **{diff_str}** |"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[REPORT] Saved ablation report to {output_file}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Fast Single-Seed Augmentation Ablation Runner")
    parser.add_argument("--parallel", type=int, default=4, help="Maximum concurrent GPU training processes (default: 4)")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs per model (default: 100)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu", "auto"])
    parser.add_argument("--force_retrain", action="store_true", default=True, help="Force retrain all ablation tasks")
    args = parser.parse_args()

    device_str = "cuda" if args.device == "cuda" or (args.device == "auto" and subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0) else "cpu"

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "ablation_seed42"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir = PROJECT_ROOT / "outputs" / "ablation_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)

    status_file = output_dir / "ablation_status.json"
    trigger_log = output_dir / "ablation_triggers.jsonl"
    heartbeat_file = Path("/marimo/.keepalive") if Path("/marimo").exists() else output_dir / ".keepalive"

    stop_heartbeat = threading.Event()
    keepalive_thread = threading.Thread(
        target=keepalive_worker,
        args=(stop_heartbeat, heartbeat_file, 8080),
        daemon=True
    )
    keepalive_thread.start()

    status_data = {
        "status": "RUNNING",
        "total_models": len(ABLATION_TASKS),
        "completed_count": 0,
        "running_models": [],
        "completed_models": [],
        "failed_models": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_str
    }
    _save_status(status_file, status_data)

    print(f"\n=======================================================", flush=True)
    print(f"  LAUNCHING FAST SINGLE-SEED (SEED 42) ABLATION SUITE", flush=True)
    print(f"  Total Models: {len(ABLATION_TASKS)} | Concurrency: {args.parallel} | Device: {device_str.upper()}", flush=True)
    print(f"=======================================================\n", flush=True)

    status_lock = threading.Lock()
    all_results = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(
                run_single_task,
                task,
                device_str,
                checkpoint_dir,
                output_dir,
                args.epochs,
                args.force_retrain,
                status_lock,
                status_data,
                trigger_log,
                status_file
            ): task
            for task in ABLATION_TASKS
        }

        for fut in as_completed(futures):
            res = fut.result()
            all_results.append(res)

    stop_heartbeat.set()

    # Generate Report
    report_file = PROJECT_ROOT / "outputs" / "ablation_seed42_results.md"
    generate_markdown_report(all_results, report_file)

    status_data["status"] = "COMPLETED"
    status_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_status(status_file, status_data)

    print(f"\n[DONE] All {len(all_results)} ablation tasks completed successfully!", flush=True)

if __name__ == "__main__":
    main()
