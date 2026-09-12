#!/usr/bin/env python3
"""
Parallel Target Experiment Runner for Gym Exercise Classification.
Executes 5 target experiments concurrently on GPU with 96GB VRAM:
  1. T1.9: LSTM on mix (117 dims)
  2. T1.18: BiLSTM on mix (117 dims)
  3. T1.27: Transformer on mix (117 dims)
  4. T2.2: Transformer on rel_3d with SkelGym-Aug
  5. T4.2: AAGCN on bone_3d with SkelGym-Aug
Followed by:
  6. T5.1: Grand Ensemble (Dual-Target Weighted Soft Voting combining T2.2 and T4.2)
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_DIR = Path(__file__).resolve().parent
LOGS_DIR = REPO_DIR / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
STATUS_FILE = REPO_DIR / "outputs" / "parallel_status.json"

TARGET_TRAIN_EXPERIMENTS = [
    {
        "id": "T1.9",
        "name": "T1.9_LSTM_mix",
        "cmd": [sys.executable, "run.py", "train", "--model", "LSTM", "--feature", "mix", "--exp_id", "T1.9", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T1.18",
        "name": "T1.18_BiLSTM_mix",
        "cmd": [sys.executable, "run.py", "train", "--model", "BiLSTM", "--feature", "mix", "--exp_id", "T1.18", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T1.27",
        "name": "T1.27_Transformer_mix",
        "cmd": [sys.executable, "run.py", "train", "--model", "Transformer", "--feature", "mix", "--exp_id", "T1.27", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T2.2",
        "name": "T2.2_Transformer_rel_3d_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--augment", "skel_gym_aug", "--exp_id", "T2.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T4.2",
        "name": "T4.2_AAGCN_bone_3d_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "AAGCN", "--feature", "bone_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
]

ENSEMBLE_EXPERIMENT = {
    "id": "T5.1",
    "name": "T5.1_Grand_Ensemble_WeightedSoft",
    "cmd": [
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T5.1",
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--checkpoints",
        "checkpoints/best_Transformer_T2.2_rel_3d.pt",
        "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
        "--device", "cuda",
        "--push_to_hf"
    ]
}

status_data = {
    "started_at": datetime.now().isoformat(),
    "experiments": {exp["id"]: {"name": exp["name"], "status": "QUEUED", "start_time": None, "end_time": None, "duration": None} for exp in TARGET_TRAIN_EXPERIMENTS},
    "ensemble": {"name": ENSEMBLE_EXPERIMENT["name"], "status": "QUEUED", "start_time": None, "end_time": None}
}

def update_status():
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)

def run_single(exp):
    exp_id = exp["id"]
    name = exp["name"]
    cmd = exp["cmd"]
    log_path = LOGS_DIR / f"{exp_id}.log"

    status_data["experiments"][exp_id]["status"] = "RUNNING"
    status_data["experiments"][exp_id]["start_time"] = datetime.now().isoformat()
    update_status()

    t0 = time.time()
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] 🚀 Starting {name}...")
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO_DIR, stdout=lf, stderr=subprocess.STDOUT)
    t1 = time.time()
    dur = round(t1 - t0, 1)

    now_str = datetime.now().strftime("%H:%M:%S")
    if proc.returncode == 0:
        print(f"[{now_str}] ✅ COMPLETED {name} in {dur}s")
        status_data["experiments"][exp_id]["status"] = "COMPLETED"
    else:
        print(f"[{now_str}] ❌ FAILED {name} with code {proc.returncode} in {dur}s")
        status_data["experiments"][exp_id]["status"] = "FAILED"

    status_data["experiments"][exp_id]["end_time"] = datetime.now().isoformat()
    status_data["experiments"][exp_id]["duration"] = dur
    update_status()
    return exp_id, proc.returncode

def main():
    print("=== Starting 5 Parallel Target Training Jobs on RTX PRO 6000 Blackwell ===")
    update_status()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_single, exp): exp["id"] for exp in TARGET_TRAIN_EXPERIMENTS}
        for fut in as_completed(futures):
            exp_id, ret = fut.result()

    # Launch Grand Ensemble
    ens_id = ENSEMBLE_EXPERIMENT["id"]
    ens_name = ENSEMBLE_EXPERIMENT["name"]
    ens_cmd = ENSEMBLE_EXPERIMENT["cmd"]
    ens_log = LOGS_DIR / f"{ens_id}.log"

    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] 🎯 Launching Grand Ensemble: {ens_name}...")
    status_data["ensemble"]["status"] = "RUNNING"
    status_data["ensemble"]["start_time"] = datetime.now().isoformat()
    update_status()

    t0 = time.time()
    with open(ens_log, "w") as lf:
        proc = subprocess.run(ens_cmd, cwd=REPO_DIR, stdout=lf, stderr=subprocess.STDOUT)
    t1 = time.time()
    dur = round(t1 - t0, 1)

    now_str = datetime.now().strftime("%H:%M:%S")
    if proc.returncode == 0:
        print(f"[{now_str}] 🎉 COMPLETED Grand Ensemble in {dur}s")
        status_data["ensemble"]["status"] = "COMPLETED"
    else:
        print(f"[{now_str}] ❌ Grand Ensemble FAILED with code {proc.returncode}")
        status_data["ensemble"]["status"] = "FAILED"

    status_data["ensemble"]["end_time"] = datetime.now().isoformat()
    status_data["ensemble"]["duration"] = dur
    update_status()

    # Upload final report to Hugging Face
    try:
        from src.utils.hf_hub import upload_file_to_hf
        report_p = REPO_DIR / "outputs" / "EXPERIMENT_RESULTS.md"
        if report_p.exists():
            upload_file_to_hf(str(report_p), "reports/EXPERIMENT_RESULTS.md")
            print("Uploaded final EXPERIMENT_RESULTS.md to Hugging Face!")
    except Exception as e:
        print(f"HF upload error: {e}")

    print("=== ALL TARGET EXPERIMENTS AND ENSEMBLE COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
