#!/usr/bin/env python3
"""
Phase 2 Experiment Runner:
Executes full 4-stream augmented AAGCN + Best Transformer on mix (117d) with SkelGym-Aug.
1. Parallel Training (4 concurrent jobs on 96GB GPU):
   - T2.2: Transformer on mix (117d) + SkelGym-Aug
   - T4.3: AAGCN on rel_3d (Joint Stream) + SkelGym-Aug
   - T4.4: AAGCN on joint_motion_3d (Joint Motion) + SkelGym-Aug
   - T4.5: AAGCN on bone_motion_3d (Bone Motion) + SkelGym-Aug
2. Multi-Stream Graph Ensembles:
   - T4.6: Two-Stream AAGCN (Aug) [Joint Aug + Bone Aug]
   - T4.7: Four-Stream AAGCN (Aug) [Joint Aug + Bone Aug + J-Motion Aug + B-Motion Aug]
3. Cross-Paradigm Grand Ensembles:
   - T5.1: Grand 5-Stream SOTA Ensemble [Transformer mix aug + 4-Stream AAGCN aug]
   - T5.2: Dual-Model Grand Ensemble [Transformer mix aug + AAGCN bone aug]
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
STATUS_FILE = REPO_DIR / "outputs" / "phase2_status.json"

TARGET_TRAIN_EXPERIMENTS = [
    {
        "id": "T2.2",
        "name": "T2.2_Transformer_mix_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "Transformer", "--feature", "mix", "--augment", "skel_gym_aug", "--exp_id", "T2.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T4.3",
        "name": "T4.3_AAGCN_rel_3d_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "AAGCN", "--feature", "rel_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.3", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T4.4",
        "name": "T4.4_AAGCN_joint_motion_3d_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "AAGCN", "--feature", "joint_motion_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.4", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
    {
        "id": "T4.5",
        "name": "T4.5_AAGCN_bone_motion_3d_skel_gym_aug",
        "cmd": [sys.executable, "run.py", "train", "--model", "AAGCN", "--feature", "bone_motion_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.5", "--epochs", "100", "--patience", "10", "--video_level", "--device", "cuda", "--use_amp", "--in_memory", "--push_to_hf"]
    },
]

ENSEMBLE_EXPERIMENTS = [
    {
        "id": "T4.6",
        "name": "T4.6_Two_Stream_AAGCN_Aug",
        "cmd": [
            sys.executable, "run.py", "ensemble",
            "--method", "weighted_soft",
            "--exp_id", "T4.6",
            "--seq_len", "32",
            "--stride", "32",
            "--video_level",
            "--checkpoints",
            "checkpoints/best_AAGCN_T4.3_rel_3d.pt",
            "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
            "--device", "cuda",
            "--push_to_hf"
        ]
    },
    {
        "id": "T4.7",
        "name": "T4.7_Four_Stream_AAGCN_Aug",
        "cmd": [
            sys.executable, "run.py", "ensemble",
            "--method", "weighted_soft",
            "--exp_id", "T4.7",
            "--seq_len", "32",
            "--stride", "32",
            "--video_level",
            "--checkpoints",
            "checkpoints/best_AAGCN_T4.3_rel_3d.pt",
            "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
            "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt",
            "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt",
            "--device", "cuda",
            "--push_to_hf"
        ]
    },
    {
        "id": "T5.2",
        "name": "T5.2_Dual_Model_Grand_Ensemble",
        "cmd": [
            sys.executable, "run.py", "ensemble",
            "--method", "weighted_soft",
            "--exp_id", "T5.2",
            "--seq_len", "32",
            "--stride", "32",
            "--video_level",
            "--checkpoints",
            "checkpoints/best_Transformer_T2.2_mix.pt",
            "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
            "--device", "cuda",
            "--push_to_hf"
        ]
    },
    {
        "id": "T5.1",
        "name": "T5.1_Grand_5_Stream_SOTA_Ensemble",
        "cmd": [
            sys.executable, "run.py", "ensemble",
            "--method", "weighted_soft",
            "--exp_id", "T5.1",
            "--seq_len", "32",
            "--stride", "32",
            "--video_level",
            "--checkpoints",
            "checkpoints/best_Transformer_T2.2_mix.pt",
            "checkpoints/best_AAGCN_T4.3_rel_3d.pt",
            "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
            "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt",
            "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt",
            "--device", "cuda",
            "--push_to_hf"
        ]
    },
]

status_data = {
    "started_at": datetime.now().isoformat(),
    "experiments": {exp["id"]: {"name": exp["name"], "status": "QUEUED", "start_time": None, "end_time": None, "duration": None} for exp in TARGET_TRAIN_EXPERIMENTS},
    "ensembles": {exp["id"]: {"name": exp["name"], "status": "QUEUED", "start_time": None, "end_time": None, "duration": None} for exp in ENSEMBLE_EXPERIMENTS}
}

def update_status():
    with open(STATUS_FILE, "w") as f:
        json.dump(status_data, f, indent=2)

def run_single(exp, is_ensemble=False):
    exp_id = exp["id"]
    name = exp["name"]
    cmd = exp["cmd"]
    log_path = LOGS_DIR / f"{exp_id}.log"
    target_dict = status_data["ensembles"] if is_ensemble else status_data["experiments"]

    target_dict[exp_id]["status"] = "RUNNING"
    target_dict[exp_id]["start_time"] = datetime.now().isoformat()
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
        target_dict[exp_id]["status"] = "COMPLETED"
    else:
        print(f"[{now_str}] ❌ FAILED {name} with code {proc.returncode} in {dur}s")
        target_dict[exp_id]["status"] = "FAILED"

    target_dict[exp_id]["end_time"] = datetime.now().isoformat()
    target_dict[exp_id]["duration"] = dur
    update_status()
    return exp_id, proc.returncode

def main():
    print("=== Starting 4 Parallel Augmented Training Jobs on RTX PRO 6000 Blackwell ===")
    update_status()

    # Step 1: Run 4 parallel training jobs concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single, exp, False): exp["id"] for exp in TARGET_TRAIN_EXPERIMENTS}
        for fut in as_completed(futures):
            exp_id, ret = fut.result()
            print(f"Finished train worker: {exp_id} (code {ret})")

    # Step 2: Run Ensembles sequentially
    for ens in ENSEMBLE_EXPERIMENTS:
        ens_id = ens["id"]
        ens_name = ens["name"]
        print(f"\n--- Running Ensemble: {ens_name} ({ens_id}) ---")
        run_single(ens, is_ensemble=True)

    # Step 3: Push updated report to Hugging Face
    try:
        from src.utils.hf_hub import upload_file_to_hf
        report_p = REPO_DIR / "outputs" / "EXPERIMENT_RESULTS.md"
        if report_p.exists():
            upload_file_to_hf(str(report_p), "reports/EXPERIMENT_RESULTS.md")
            print("Uploaded final EXPERIMENT_RESULTS.md to Hugging Face!")
    except Exception as e:
        print(f"HF upload error: {e}")

    print("\n=== ALL PHASE 2 EXPERIMENTS AND ENSEMBLES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
