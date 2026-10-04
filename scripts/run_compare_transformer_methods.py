#!/usr/bin/env python3
"""
Benchmark comparison script for Transformer Mix Feature Enhancement:
Evaluates:
  1. Method 1: No InNorm (self.in_norm = nn.Identity())
  2. Method 2: Dual-Branch Embedding (coords 39 -> 64, angles 78 -> 64 => 128)
Across 3 seeds (42, 123, 3407) under strict unaugmented (Clean) Table 2 protocol.
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
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CANONICAL_EXPERIMENT_REGISTRY
from src.utils.hf_hub import upload_file_to_hf

HF_REPO = "Cuong2004/gym-exercise-classification"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SEEDS = [42, 123, 3407]
METHODS = ["no_innorm", "dual_branch"]

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

def run_task(variant: str, seed: int, output_base: Path, trigger_log: Path) -> Dict[str, Any]:
    task_name = f"Transformer_{variant}_mix_seed{seed}"
    ckpt_dir = output_base / variant / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"best_Transformer_{variant}_mix.pt"
    ckpt_path = ckpt_dir / ckpt_name

    emit_trigger("MODEL_START", {"model": task_name, "variant": variant, "seed": seed}, trigger_log)

    cmd = [
        sys.executable, "run.py", "train",
        "--model", "Transformer",
        "--feature", "mix",
        "--transformer_variant", variant,
        "--augment", "none",
        "--seed", str(seed),
        "--epochs", "100",
        "--lr", "1e-4",
        "--batch_size", "16",
        "--label_smoothing", "0.05",
        "--patience", "10",
        "--train_stride", "16",
        "--val_test_stride", "32",
        "--early_stopping_metric", "val_macro_f1",
        "--checkpoint_dir", str(ckpt_dir),
        "--checkpoint_name", ckpt_name,
        "--video_level",
        "--device", "cuda",
        "--use_amp",
        "--in_memory"
    ]

    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    duration = time.time() - t0

    if res.returncode != 0:
        emit_trigger("MODEL_FAILED", {"model": task_name, "error": res.stderr[-400:]}, trigger_log)
        raise RuntimeError(f"Training failed for {task_name}:\n{res.stderr[-400:]}")

    # Extract metrics from checkpoint provenance
    val_acc = 0.0
    val_loss = 0.0
    val_f1 = 0.0
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        prov = sd.get("provenance", {})
        val_acc = prov.get("val_acc", 0.0) * 100.0 if prov.get("val_acc", 0.0) <= 1.0 else prov.get("val_acc", 0.0)
        val_loss = float(prov.get("val_loss", 0.0))
        val_f1 = float(prov.get("val_macro_f1", 0.0))
    except Exception as e:
        print(f"Warning reading provenance: {e}")

    # Evaluate on test set
    eval_cmd = [
        sys.executable, "run.py", "evaluate",
        "--checkpoint", str(ckpt_path),
        "--model", "Transformer",
        "--feature", "mix",
        "--transformer_variant", variant,
        "--device", "cuda",
        "--video_level",
        "--in_memory"
    ]
    eval_res = subprocess.run(eval_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    test_win_acc = 0.0
    test_vid_acc = 0.0
    test_f1 = 0.0

    if eval_res.returncode == 0:
        import re
        m_win = re.search(r"Window-Level Acc:\s+([\d\.]+)%", eval_res.stdout)
        if not m_win:
            m_win = re.search(r"(?:Window\s+)?Accuracy:\s+([\d\.]+)%", eval_res.stdout)
        if m_win:
            test_win_acc = float(m_win.group(1))

        m_f1 = re.search(r"Macro F1:\s+([\d\.]+)", eval_res.stdout)
        if m_f1:
            test_f1 = float(m_f1.group(1))

        m_vid = re.search(r"VIDEO-LEVEL[^\n]*Accuracy:\s+([\d\.]+)%", eval_res.stdout)
        if not m_vid:
            m_vid = re.search(r"Video-Level Consensus Accuracy:\s+([\d\.]+)%", eval_res.stdout)
        if m_vid:
            test_vid_acc = float(m_vid.group(1))

    # Upload to HF
    try:
        rel_in_repo = str(ckpt_path.relative_to(PROJECT_ROOT))
        upload_file_to_hf(str(ckpt_path), path_in_repo=rel_in_repo, repo_id=HF_REPO, token=HF_TOKEN)
        prov_file = ckpt_path.with_suffix(".provenance.json")
        if prov_file.exists():
            upload_file_to_hf(str(prov_file), path_in_repo=str(prov_file.relative_to(PROJECT_ROOT)), repo_id=HF_REPO, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Upload Warning] Failed to upload {ckpt_path.name}: {e}")

    emit_trigger("MODEL_COMPLETE", {
        "model": task_name,
        "variant": variant,
        "seed": seed,
        "duration": f"{duration:.1f}s",
        "val_acc": f"{val_acc:.2f}%",
        "test_win_acc": f"{test_win_acc:.2f}%",
        "macro_f1": f"{test_f1:.4f}",
        "video_acc": f"{test_vid_acc:.2f}%"
    }, trigger_log)

    return {
        "variant": variant,
        "seed": seed,
        "duration": duration,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "val_f1": val_f1,
        "test_win_acc": test_win_acc,
        "test_vid_acc": test_vid_acc,
        "macro_f1": test_f1,
        "checkpoint": str(ckpt_path)
    }

def main():
    trigger_log = PROJECT_ROOT / "outputs" / "triggers_mix_methods.jsonl"
    trigger_log.parent.mkdir(parents=True, exist_ok=True)
    if trigger_log.exists():
        trigger_log.unlink()

    heartbeat_file = PROJECT_ROOT / "outputs" / "keepalive_mix_methods.heartbeat"
    stop_event = threading.Event()
    daemon = threading.Thread(target=keepalive_daemon, args=(stop_event, heartbeat_file), daemon=True)
    daemon.start()

    output_base = PROJECT_ROOT / "checkpoints" / "mix_methods"
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in METHODS}

    print("=" * 80)
    print("STARTING EXPERIMENT: Transformer Mix Enhancement Comparison across 3 Seeds")
    print("Method 1: No InNorm (self.in_norm = nn.Identity())")
    print("Method 2: Dual-Branch Embedding (Coords 39->64, Angles 78->64 => 128)")
    print("=" * 80)

    try:
        for variant in METHODS:
            for seed in SEEDS:
                res = run_task(variant, seed, output_base, trigger_log)
                results[variant].append(res)
            
            # Print intermediate method summary
            val_accs = [r["val_acc"] for r in results[variant]]
            test_accs = [r["test_win_acc"] for r in results[variant]]
            f1s = [r["macro_f1"] for r in results[variant]]
            vids = [r["test_vid_acc"] for r in results[variant]]
            emit_trigger("METHOD_COMPLETE", {
                "variant": variant,
                "val_acc_mean": f"{np.mean(val_accs):.2f} ± {np.std(val_accs):.2f}%",
                "test_win_acc_mean": f"{np.mean(test_accs):.2f} ± {np.std(test_accs):.2f}%",
                "macro_f1_mean": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                "video_acc_mean": f"{np.mean(vids):.2f} ± {np.std(vids):.2f}%"
            }, trigger_log)

    finally:
        stop_event.set()

    # Compile final comparison summary
    summary = {}
    for variant in METHODS:
        v_list = results[variant]
        val_accs = [r["val_acc"] for r in v_list]
        test_accs = [r["test_win_acc"] for r in v_list]
        f1s = [r["macro_f1"] for r in v_list]
        vids = [r["test_vid_acc"] for r in v_list]
        summary[variant] = {
            "variant": variant,
            "seeds": SEEDS,
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_sd": float(np.std(val_accs)),
            "test_win_acc_mean": float(np.mean(test_accs)),
            "test_win_acc_sd": float(np.std(test_accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_sd": float(np.std(f1s)),
            "test_vid_acc_mean": float(np.mean(vids)),
            "test_vid_acc_sd": float(np.std(vids)),
            "runs": v_list
        }

    out_json = PROJECT_ROOT / "artifacts" / "results" / "test_mix_methods_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    try:
        upload_file_to_hf(str(out_json), path_in_repo=str(out_json.relative_to(PROJECT_ROOT)), repo_id=HF_REPO, token=HF_TOKEN)
    except Exception as e:
        print(f"[HF Upload Warning] Failed to upload comparison JSON: {e}")

    emit_trigger("PHASE_COMPLETE", {"total_runs": len(METHODS) * len(SEEDS), "saved_json": str(out_json)}, trigger_log)

if __name__ == "__main__":
    main()
