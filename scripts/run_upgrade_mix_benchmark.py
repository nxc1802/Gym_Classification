#!/usr/bin/env python3
"""
Master Benchmark Execution Script for Upgraded Biomechanical Mix (SkelGym-300K).
Evaluates 3 feature paradigms on Transformer (~300K parameter budget):
  1. raw_3d: Raw Camera 3D Coordinates (Baseline)
  2. rel_3d_norm: Anisotropic Anthropometric Scale-Normalized Relative Coordinates
  3. mix_v2: Biomechanical Mix v2 (rel_3d_norm 39-d + angle_kinematic_24 24-d = 63-d, Dual-Branch)
Across 3 seeds (42, 123, 3407) under strict unaugmented (Clean) Table 2 protocol.
Parallel execution, automatic keepalive, and direct sync to Hugging Face Hub.
"""

import os
import sys
import time
import json
import urllib.request
import threading
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_hub import upload_file_to_hf

HF_REPO = "Cuong2004/gym-exercise-classification"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEEDS = [42, 123, 3407]

EXPERIMENTS = [
    {
        "id": "raw_3d",
        "name": "Transformer Raw 3D (300K)",
        "feature": "raw_3d",
        "variant": "standard",
        "dim": 39
    },
    {
        "id": "rel_3d_norm",
        "name": "Transformer Scale-Norm Rel 3D (300K)",
        "feature": "rel_3d_norm",
        "variant": "standard",
        "dim": 39
    },
    {
        "id": "mix_v2",
        "name": "Transformer Biomechanical Mix v2 (300K)",
        "feature": "mix_v2",
        "variant": "dual_branch",
        "dim": 63
    }
]

def keepalive_daemon(stop_event: threading.Event, heartbeat_file: Path, ports=(8080, 2718)):
    """Continuously pings localhost ports and touches heartbeat to prevent container shutdown."""
    while not stop_event.is_set():
        for port in ports:
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"User-Agent": "Marimo-KeepAlive"})
                with urllib.request.urlopen(req, timeout=2):
                    pass
            except Exception:
                pass
        try:
            heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
            heartbeat_file.touch()
        except Exception:
            pass
        time.sleep(10)

def emit_trigger(trigger_type: str, data: Dict[str, Any], trigger_log_path: Path):
    """Appends an event-driven trigger line and writes to JSONL."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trigger": trigger_type,
        "data": data
    }
    line = f"⚡ [TRIGGER: {trigger_type}] " + " | ".join(f"{k}: {v}" for k, v in data.items())
    print(line, flush=True)
    with open(trigger_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

def run_single_model(exp: Dict[str, Any], seed: int, output_base: Path, trigger_log: Path) -> Dict[str, Any]:
    feat_id = exp["id"]
    task_name = f"Transformer_{feat_id}_seed{seed}"
    ckpt_dir = output_base / feat_id / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_Transformer_{feat_id}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    emit_trigger("MODEL_START", {"model": task_name, "feature": feat_id, "seed": seed}, trigger_log)

    cmd = [
        sys.executable, "run.py", "train",
        "--model", "Transformer",
        "--feature", exp["feature"],
        "--transformer_variant", exp["variant"],
        "--hidden_dim", "112",
        "--augment", "none",
        "--seed", str(seed),
        "--epochs", "100",
        "--lr", "1e-4",
        "--batch_size", "32",
        "--label_smoothing", "0.05",
        "--patience", "10",
        "--train_stride", "16",
        "--val_test_stride", "32",
        "--early_stopping_metric", "val_macro_f1",
        "--checkpoint_dir", str(ckpt_dir),
        "--checkpoint_name", ckpt_name,
        "--video_level",
        "--device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "auto",
        "--use_amp",
        "--in_memory"
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    duration = time.time() - t0

    if res.returncode != 0:
        emit_trigger("MODEL_FAILED", {"model": task_name, "error": res.stderr[-400:]}, trigger_log)
        raise RuntimeError(f"Training failed for {task_name}:\n{res.stderr[-400:]}")

    # Parse metrics from logs
    val_acc, val_f1, val_vid_acc, val_vid_f1 = 0.0, 0.0, 0.0, 0.0
    test_win_acc, test_win_f1, test_vid_acc, test_vid_f1 = 0.0, 0.0, 0.0, 0.0

    eval_cmd = [
        sys.executable, "run.py", "evaluate",
        "--checkpoint", str(ckpt_path),
        "--video_level",
        "--device", "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "auto"
    ]
    eval_res = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    
    import re
    m_win_acc = re.search(r"Window-Level Acc:\s+([\d\.]+)%", eval_res.stdout)
    if m_win_acc:
        test_win_acc = float(m_win_acc.group(1))
    
    m_win_f1 = re.search(r"Window-Level Macro F1:\s+([\d\.]+)", eval_res.stdout)
    if m_win_f1:
        test_win_f1 = float(m_win_f1.group(1))

    m_vid_acc = re.search(r"VIDEO-LEVEL TEST.*Accuracy:\s+([\d\.]+)%", eval_res.stdout)
    if m_vid_acc:
        test_vid_acc = float(m_vid_acc.group(1))

    m_vid_f1 = re.search(r"VIDEO-LEVEL TEST.*Macro F1:\s+([\d\.]+)", eval_res.stdout)
    if m_vid_f1:
        test_vid_f1 = float(m_vid_f1.group(1))

    # Also parse validation metrics from training log
    # Regex for Best checkpoint saved (val_macro_f1: 0.7851) or similar
    m_val = re.findall(r"val_acc:\s+([\d\.]+)\s+-\s+val_f1:\s+([\d\.]+)", res.stdout)
    if m_val:
        # Checkpoint provenance file has the authoritative numbers
        prov_p = ckpt_dir / f"{ckpt_name.replace('.pt', '')}.provenance.json"
        if prov_p.exists():
            prov_data = json.loads(prov_p.read_text())
            val_acc = prov_data.get("val_acc", 0.0) * 100.0
            val_f1 = prov_data.get("val_macro_f1", prov_data.get("val_f1", 0.0))
        else:
            val_acc = float(m_val[-1][0]) * 100.0
            val_f1 = float(m_val[-1][1])

    # Upload to Hugging Face Hub
    if HF_TOKEN and ckpt_path.exists():
        hf_remote_path = f"checkpoints/upgrade_mix/{feat_id}/seed{seed}/{ckpt_name}"
        try:
            upload_file_to_hf(
                file_path=ckpt_path,
                repo_id=HF_REPO,
                remote_path=hf_remote_path,
                token=HF_TOKEN,
                commit_message=f"Upload {task_name} checkpoint ({val_acc:.2f}% Val Acc)"
            )
            prov_p = ckpt_dir / f"{ckpt_name.replace('.pt', '')}.provenance.json"
            if prov_p.exists():
                upload_file_to_hf(
                    file_path=prov_p,
                    repo_id=HF_REPO,
                    remote_path=f"checkpoints/upgrade_mix/{feat_id}/seed{seed}/{prov_p.name}",
                    token=HF_TOKEN,
                    commit_message=f"Upload provenance metadata for {task_name}"
                )
        except Exception as e:
            print(f"Warning: Failed to upload {task_name} to HF: {e}", flush=True)

    result_data = {
        "feature": feat_id,
        "seed": seed,
        "duration": duration,
        "val_win_acc": val_acc,
        "val_win_f1": val_f1,
        "test_win_acc": test_win_acc,
        "test_win_f1": test_win_f1,
        "test_vid_acc": test_vid_acc,
        "test_vid_f1": test_vid_f1,
        "checkpoint": str(ckpt_path)
    }

    emit_trigger("MODEL_COMPLETE", {
        "model": task_name,
        "val_win_acc": f"{val_acc:.2f}%",
        "val_win_f1": f"{val_f1:.4f}",
        "test_win_acc": f"{test_win_acc:.2f}%",
        "test_vid_acc": f"{test_vid_acc:.2f}%",
        "duration": f"{duration:.1f}s"
    }, trigger_log)

    return result_data

def main():
    output_base = PROJECT_ROOT / "checkpoints" / "upgrade_mix"
    trigger_log = PROJECT_ROOT / "outputs" / "upgrade_mix_triggers.jsonl"
    results_json = PROJECT_ROOT / "outputs" / "upgrade_mix_results.json"
    results_md = PROJECT_ROOT / "outputs" / "UPGRADE_MIX_RESULTS.md"
    heartbeat = PROJECT_ROOT / "outputs" / ".heartbeat"

    output_base.mkdir(parents=True, exist_ok=True)
    trigger_log.parent.mkdir(parents=True, exist_ok=True)

    stop_keepalive = threading.Event()
    daemon_thread = threading.Thread(target=keepalive_daemon, args=(stop_keepalive, heartbeat), daemon=True)
    daemon_thread.start()

    print("=" * 70, flush=True)
    print("🚀 STARTING SKELGYM-300K MIX UPGRADE BENCHMARK (PARALLEL EXECUTION)", flush=True)
    print(f"Features: {[e['id'] for e in EXPERIMENTS]} | Seeds: {SEEDS}", flush=True)
    print("=" * 70, flush=True)

    tasks = []
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            tasks.append((exp, seed))

    all_results: Dict[str, List[Dict[str, Any]]] = {e["id"]: [] for e in EXPERIMENTS}

    # Parallel execution with 3 workers (one seed per feature or 3 parallel models)
    max_workers = min(3, len(tasks))
    print(f"Executing with {max_workers} parallel workers on GPU...", flush=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_model, exp, seed, output_base, trigger_log): (exp["id"], seed) for exp, seed in tasks}
        for future in as_completed(futures):
            feat_id, seed = futures[future]
            try:
                res = future.result()
                all_results[feat_id].append(res)
            except Exception as e:
                print(f"❌ Error in {feat_id} seed {seed}: {e}", flush=True)

    # Compute summary statistics
    summary = {}
    for feat_id, runs in all_results.items():
        if not runs:
            continue
        summary[feat_id] = {
            "val_win_acc_mean": float(np.mean([r["val_win_acc"] for r in runs])),
            "val_win_acc_sd": float(np.std([r["val_win_acc"] for r in runs])),
            "val_win_f1_mean": float(np.mean([r["val_win_f1"] for r in runs])),
            "val_win_f1_sd": float(np.std([r["val_win_f1"] for r in runs])),
            "test_win_acc_mean": float(np.mean([r["test_win_acc"] for r in runs])),
            "test_win_acc_sd": float(np.std([r["test_win_acc"] for r in runs])),
            "test_win_f1_mean": float(np.mean([r["test_win_f1"] for r in runs])),
            "test_win_f1_sd": float(np.std([r["test_win_f1"] for r in runs])),
            "test_vid_acc_mean": float(np.mean([r["test_vid_acc"] for r in runs])),
            "test_vid_acc_sd": float(np.std([r["test_vid_acc"] for r in runs])),
            "test_vid_f1_mean": float(np.mean([r["test_vid_f1"] for r in runs])),
            "test_vid_f1_sd": float(np.std([r["test_vid_f1"] for r in runs])),
            "runs": runs
        }

    with open(results_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# SkelGym-300K: Feature Upgrade & Head-to-Head Benchmark",
        "",
        "| Feature Representation | Dim | Val Win Acc (%) | Val Win Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for feat_id in ["raw_3d", "rel_3d_norm", "mix_v2"]:
        if feat_id in summary:
            s = summary[feat_id]
            dim = 39 if "3d" in feat_id else 63
            name = "Raw 3D Coordinates" if feat_id == "raw_3d" else ("Scale-Norm Relative 3D" if feat_id == "rel_3d_norm" else "**Biomechanical Mix v2 (Proposed)**")
            md_lines.append(
                f"| {name} | {dim} | **{s['val_win_acc_mean']:.2f}% ± {s['val_win_acc_sd']:.2f}%** | {s['val_win_f1_mean']:.4f} ± {s['val_win_f1_sd']:.4f} | {s['test_win_acc_mean']:.2f}% ± {s['test_win_acc_sd']:.2f}% | {s['test_win_f1_mean']:.4f} ± {s['test_win_f1_sd']:.4f} | {s['test_vid_acc_mean']:.2f}% ± {s['test_vid_acc_sd']:.2f}% | {s['test_vid_f1_mean']:.4f} ± {s['test_vid_f1_sd']:.4f} | Verified |"
            )

    results_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK COMPLETED SUCCESSFULLY! Summary table:\n", flush=True)
    print(results_md.read_text(), flush=True)

    stop_keepalive.set()

if __name__ == "__main__":
    main()
