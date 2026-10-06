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

Parallel Execution:
  Executes all 3 seeds for each configuration simultaneously using independent worker processes,
  optimizing the multi-core CPU (20 cores) and 96GB VRAM NVIDIA Blackwell GPU.
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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

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

def run_config_3seeds_parallel(
    cfg_id: str,
    aug: str,
    checkpoint_dir: Path,
    metadata_path: str,
    device: str,
    resume: bool = True
) -> List[Dict[str, Any]]:
    """
    Executes all 3 seeds concurrently using subprocess workers.
    """
    print(f"\n================================================================================")
    print(f"Executing Configuration: {cfg_id} (aug: {aug}) across 3 seeds concurrently...")
    print(f"================================================================================")
    send_marimo_toast(f"⚡ Phase 3 Parallel: Running {cfg_id} across 3 seeds concurrently!")

    worker_script = ROOT_DIR / "scripts" / "train_single_ablation.py"
    procs = []
    logs = {}

    for s in SEEDS:
        out_json = checkpoint_dir / f"seed{s}" / f"result_{cfg_id}_seed{s}.json"
        log_file = checkpoint_dir / f"seed{s}" / f"train_{cfg_id}_seed{s}.log"
        logs[s] = (log_file, out_json)

        cmd = [
            sys.executable, str(worker_script),
            "--cfg_id", cfg_id,
            "--aug", aug,
            "--seed", str(s),
            "--device", device,
            "--checkpoint_dir", str(checkpoint_dir),
            "--metadata_path", metadata_path,
            "--out_json", str(out_json)
        ]
        if resume:
            cmd.append("--resume")

        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        procs.append((s, p, lf, out_json))

    # Monitor parallel execution
    t_start = time.time()
    while True:
        all_done = all(p.poll() is not None for _, p, _, _ in procs)
        if all_done:
            break
        time.sleep(5)
        elapsed = time.time() - t_start
        status_str = " | ".join([f"Seed {s}: {'DONE' if p.poll() is not None else 'RUNNING'}" for s, p, _, _ in procs])
        print(f"  [{elapsed:.0f}s elapsed] {cfg_id} -> {status_str}", end="\r", flush=True)

    print(f"\nAll 3 seeds completed in {time.time() - t_start:.1f}s.")

    # Collect and verify results
    runs = []
    for s, p, lf, out_json in procs:
        lf.close()
        if p.returncode != 0:
            print(f"Warning: Worker for seed {s} exited with returncode {p.returncode}! Inspect {logs[s][0]}")
        if out_json.exists():
            with open(out_json) as f:
                res = json.load(f)
                runs.append(res)
                print(f"  -> Seed {s}: Val Vid {res['val_vid_acc']}%, Test Vid {res['test_vid_acc']}% (F1: {res['test_vid_f1']})")
        else:
            raise RuntimeError(f"Output JSON missing for {cfg_id} seed {s}: {out_json}")

    return runs

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Table 4 & Table 5 Ablations on WORLD mix_v2 (Parallel)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints if available (default: True)")
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retrain even if checkpoints exist")
    args = parser.parse_args()

    resume = args.resume and not args.force_retrain

    print("=" * 80)
    print(f"PHASE 3: Parallel Table 4 & 5 Ablations on WORLD mix_v2 ({args.device}, resume={resume})")
    print("=" * 80)

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    checkpoint_dir = ROOT_DIR / "checkpoints" / "ablation_v2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results_by_config = {}
    total_configs = len(ABLATION_CONFIGS)
    cfg_idx = 0

    for cfg in ABLATION_CONFIGS:
        cfg_idx += 1
        cfg_id = cfg["id"]
        aug = cfg["aug"]
        print(f"\n[{cfg_idx}/{total_configs}] Processing Config: {cfg_id} ({cfg['desc']})...")

        runs = run_config_3seeds_parallel(
            cfg_id=cfg_id,
            aug=aug,
            checkpoint_dir=checkpoint_dir,
            metadata_path=str(meta_cand),
            device=args.device,
            resume=resume
        )
        results_by_config[cfg_id] = runs

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

    send_marimo_toast("🎉 Phase 3 Complete: Table 4 & 5 Ablations finished successfully!")
    print(f"\nSaved all results to {outputs_dir / 'augmentation_ablation_results.json'}")

if __name__ == "__main__":
    main()
