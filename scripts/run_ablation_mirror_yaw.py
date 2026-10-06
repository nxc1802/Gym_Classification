#!/usr/bin/env python3
"""
Ablation Run: Mirror + Yaw (Spatial Pair) on WORLD mix_v2.
Backbone: Dual-Branch Transformer (~301K params).
Input: mix_v2 (63-d: 39-d world coordinates + 24-d kinematic angles).
Seeds: 42, 123, 3407 (Concurrently on GPU).

Evaluates:
  - Validation metrics: Val Loss, Val Win Acc, Val Win F1, Val Vid Acc, Val Vid F1
  - Test metrics: Test Win Acc, Test Win F1, Test Vid Acc, Test Vid F1
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

DEFAULT_SEEDS = [42, 123, 3407]

def send_marimo_toast(msg: str, kind: str = "info"):
    try:
        import marimo as mo
        mo.status.toast(msg, kind=kind)
    except Exception:
        pass

def summarize_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    val_l = [r["val_loss"] for r in runs if "val_loss" in r]
    val_w = [r["val_win_acc"] for r in runs]
    val_wf1 = [r["val_win_f1"] for r in runs]
    val_v = [r["val_vid_acc"] for r in runs]
    val_vf1 = [r["val_vid_f1"] for r in runs]

    test_w = [r["test_win_acc"] for r in runs]
    test_wf1 = [r["test_win_f1"] for r in runs]
    test_v = [r["test_vid_acc"] for r in runs]
    test_vf1 = [r["test_vid_f1"] for r in runs]

    return {
        "val_loss": f"{np.mean(val_l):.4f} ± {np.std(val_l):.4f}" if val_l else "N/A",
        "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
        "val_win_f1": f"{np.mean(val_wf1):.4f} ± {np.std(val_wf1):.4f}",
        "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
        "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
        "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
        "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
        "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
        "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
        "val_loss_mean": float(np.mean(val_l)) if val_l else 0.0,
        "val_win_acc_mean": float(np.mean(val_w)),
        "val_win_f1_mean": float(np.mean(val_wf1)),
        "val_vid_acc_mean": float(np.mean(val_v)),
        "val_vid_f1_mean": float(np.mean(val_vf1)),
        "test_win_acc_mean": float(np.mean(test_w)),
        "test_win_f1_mean": float(np.mean(test_wf1)),
        "test_vid_acc_mean": float(np.mean(test_v)),
        "test_vid_f1_mean": float(np.mean(test_vf1)),
        "runs": runs
    }

def main():
    parser = argparse.ArgumentParser(description="Mirror + Yaw 3-Seed Parallel Ablation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ablation_v2")
    parser.add_argument("--metadata_path", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    args = parser.parse_args()

    cfg_id = "pair_mirror_yaw"
    aug = "mirror_yaw"
    seeds = args.seeds
    device = args.device

    print("=" * 80)
    print(f"🚀 SKELGYM-AUG ABLATION: Mirror + Yaw (Spatial Pair) on WORLD mix_v2")
    print(f"Backbone: Dual-Branch Transformer (~301K params)")
    print(f"Seeds: {seeds} concurrently on {device}")
    print("=" * 80)

    send_marimo_toast(f"🚀 Bắt đầu huấn luyện song song Mirror + Yaw trên 3 seeds {seeds}...", kind="info")

    ckpt_dir = ROOT_DIR / args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta_cand = ROOT_DIR / args.metadata_path
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    worker_script = ROOT_DIR / "scripts" / "train_single_ablation.py"
    procs = []
    logs = {}

    for s in seeds:
        seed_dir = ckpt_dir / f"seed{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        out_json = seed_dir / f"result_{cfg_id}_seed{s}.json"
        log_file = seed_dir / f"train_{cfg_id}_seed{s}.log"
        logs[s] = (log_file, out_json)

        cmd = [
            sys.executable, "-u", str(worker_script),
            "--cfg_id", cfg_id,
            "--aug", aug,
            "--seed", str(s),
            "--device", device,
            "--checkpoint_dir", str(ckpt_dir),
            "--metadata_path", str(meta_cand),
            "--out_json", str(out_json)
        ]
        if not args.force_retrain:
            cmd.append("--resume")
        else:
            cmd.append("--force-retrain")

        lf = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
        procs.append((s, p, lf, out_json))

    t_start = time.time()
    print(f"Spawning 3 workers in parallel...")
    while True:
        all_done = all(p.poll() is not None for _, p, _, _ in procs)
        elapsed = time.time() - t_start
        status_strs = []
        for s, p, _, out_json in procs:
            if p.poll() is not None:
                status_strs.append(f"Seed {s}: DONE")
            else:
                status_strs.append(f"Seed {s}: RUNNING")
        status_line = " | ".join(status_strs)
        print(f"  [{elapsed:6.1f}s] {status_line}", end="\r", flush=True)
        try:
            with open(ckpt_dir / "mirror_yaw_progress.txt", "w") as pf:
                pf.write(f"[{elapsed:6.1f}s] {status_line}\n")
        except Exception:
            pass

        if all_done:
            break
        time.sleep(5)

    total_time = time.time() - t_start
    print(f"\n\nAll {len(seeds)} seeds finished in {total_time:.1f}s.")

    runs = []
    for s, p, lf, out_json in procs:
        lf.close()
        if p.returncode != 0:
            print(f"⚠️ Worker seed {s} returned error code {p.returncode}. Log: {logs[s][0]}")
            if logs[s][0].exists():
                with open(logs[s][0]) as f:
                    print("Tail of log:")
                    lines = f.readlines()
                    print("".join(lines[-20:]))
        if out_json.exists():
            with open(out_json) as f:
                res = json.load(f)
                runs.append(res)
                print(f"  ✅ Seed {s}: Val Loss={res.get('val_loss')}, Val Vid Acc={res['val_vid_acc']}%, Val Vid F1={res['val_vid_f1']} | Test Vid Acc={res['test_vid_acc']}%")
        else:
            raise RuntimeError(f"Output JSON missing for seed {s}: {out_json}")

    summary = summarize_metrics(runs)
    print("\n" + "=" * 80)
    print("🎯 AGGREGATED 3-SEED RESULTS: Mirror + Yaw (Spatial Pair)")
    print("=" * 80)
    print(f"Validation Loss:         {summary['val_loss']}")
    print(f"Validation Window Acc:   {summary['val_win_acc']}")
    print(f"Validation Window F1:    {summary['val_win_f1']}")
    print(f"Validation Video Acc:    {summary['val_vid_acc']}")
    print(f"Validation Video F1:     {summary['val_vid_f1']}")
    print("-" * 50)
    print(f"Held-out Test Window Acc:{summary['test_win_acc']}")
    print(f"Held-out Test Window F1: {summary['test_win_f1']}")
    print(f"Held-out Test Video Acc: {summary['test_vid_acc']}")
    print(f"Held-out Test Video F1:  {summary['test_vid_f1']}")
    print("=" * 80)

    # Save summary artifact
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "ablation_mirror_yaw_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Update table5_single_world_mix_v2.json
    t5_path = out_dir / "table5_single_world_mix_v2.json"
    if t5_path.exists():
        try:
            with open(t5_path) as f:
                t5_data = json.load(f)
            t5_data["pair_mirror_yaw"] = summary
            with open(t5_path, "w") as f:
                json.dump(t5_data, f, indent=2)
            print(f"Updated {t5_path} with pair_mirror_yaw entry.")
        except Exception as e:
            print(f"Error updating table5: {e}")

    # Toast inside Marimo
    send_marimo_toast(
        f"🎉 Mirror + Yaw (3 seeds) xong! Val Vid: {summary['val_vid_acc']}, F1: {summary['val_vid_f1']} (Test: {summary['test_vid_acc']})",
        kind="success"
    )

if __name__ == "__main__":
    main()
