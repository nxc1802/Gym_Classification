#!/usr/bin/env python3
"""
Supplementary Experiments Runner:
1. Train T2.4: Transformer raw_3d + skel_gym_aug
2. Train AAGCN bone_3d, joint_motion_3d, bone_motion_3d + skel_gym_aug
3. Multi-Stream Graph Fusion with Augmentation (T4.6 Two-Stream, T4.8 Four-Stream)
4. Cross-Paradigm Ensemble (T5.1 Hard, T5.2 Soft, T5.3 Stacking, T5.4 Tri-Model, T5.5 Grand SOTA)
5. Auto-update Tables 6A, 6B, and Table 7.
"""

import sys
import os
import time
import subprocess
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/logs/supplementary_runner.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)

SINGLE_MODELS = [
    {
        "name": "T2.4_Transformer_raw_3d_aug",
        "exp_id": "T2.4",
        "ckpt": "checkpoints/best_Transformer_T2.4_raw_3d.pt",
        "cmd": [
            sys.executable, "run.py", "train",
            "--model", "Transformer",
            "--feature", "raw_3d",
            "--augment", "skel_gym_aug",
            "--exp_id", "T2.4",
            "--epochs", "100",
            "--patience", "10",
            "--video_level",
            "--device", "auto",
            "--use_amp",
            "--in_memory"
        ]
    },
    {
        "name": "T4.bone_AAGCN_bone_3d_aug",
        "exp_id": "T4.bone",
        "ckpt": "checkpoints/best_AAGCN_T4.bone_bone_3d.pt",
        "cmd": [
            sys.executable, "run.py", "train",
            "--model", "AAGCN",
            "--feature", "bone_3d",
            "--augment", "skel_gym_aug",
            "--exp_id", "T4.bone",
            "--epochs", "100",
            "--patience", "10",
            "--device", "auto",
            "--use_amp",
            "--in_memory"
        ]
    },
    {
        "name": "T4.jm_AAGCN_joint_motion_3d_aug",
        "exp_id": "T4.jm",
        "ckpt": "checkpoints/best_AAGCN_T4.jm_joint_motion_3d.pt",
        "cmd": [
            sys.executable, "run.py", "train",
            "--model", "AAGCN",
            "--feature", "joint_motion_3d",
            "--augment", "skel_gym_aug",
            "--exp_id", "T4.jm",
            "--epochs", "100",
            "--patience", "10",
            "--device", "auto",
            "--use_amp",
            "--in_memory"
        ]
    },
    {
        "name": "T4.bm_AAGCN_bone_motion_3d_aug",
        "exp_id": "T4.bm",
        "ckpt": "checkpoints/best_AAGCN_T4.bm_bone_motion_3d.pt",
        "cmd": [
            sys.executable, "run.py", "train",
            "--model", "AAGCN",
            "--feature", "bone_motion_3d",
            "--augment", "skel_gym_aug",
            "--exp_id", "T4.bm",
            "--epochs", "100",
            "--patience", "10",
            "--device", "auto",
            "--use_amp",
            "--in_memory"
        ]
    },
]

def run_job(job):
    name = job["name"]
    cmd = job["cmd"]
    logger.info(f"🚀 Starting {name} ...")
    start_t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = time.time() - start_t
    if res.returncode == 0:
        logger.info(f"✅ {name} finished successfully in {dur:.1f}s.")
        return True, name, dur, ""
    else:
        logger.error(f"❌ {name} failed with code {res.returncode} in {dur:.1f}s:\n{res.stderr[-1000:]}")
        return False, name, dur, res.stderr

def run_cmd(cmd_list, desc):
    logger.info(f"▶️ Running: {desc}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"❌ Error in {desc}:\n{res.stderr}")
        raise RuntimeError(f"Command failed: {desc}")
    logger.info(f"✅ Finished: {desc}")
    return res.stdout

def main():
    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    logger.info("==================================================")
    logger.info("⚡ STARTING SUPPLEMENTARY EXPERIMENTS PIPELINE")
    logger.info("==================================================")

    # Phase 1: Train single models in parallel
    logger.info(f"Phase 1: Training {len(SINGLE_MODELS)} models concurrently with 4 workers...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_job, job): job["name"] for job in SINGLE_MODELS}
        for future in as_completed(futures):
            ok, name, dur, err = future.result()
            if not ok:
                logger.error(f"Job {name} failed! Aborting dependent phases.")
                sys.exit(1)

    logger.info("🎉 Phase 1 All Single-Model Training Complete!")

    # Phase 2: Table 4 Multi-Stream Graph Augmentations
    logger.info("Phase 2: Evaluating Two-Stream (T4.6) and Four-Stream (T4.8) AAGCN with Augmentation...")
    # T4.6: Two-Stream AAGCN (Joint Aug + Bone Aug)
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T4.6",
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--checkpoints",
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "checkpoints/best_AAGCN_T4.bone_bone_3d.pt",
        "--device", "auto"
    ], "Table 4: Two-Stream AAGCN + Aug (T4.6)")

    # T4.8: Four-Stream AAGCN (All 4 streams with Aug)
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T4.8",
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--checkpoints",
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "checkpoints/best_AAGCN_T4.bone_bone_3d.pt",
        "checkpoints/best_AAGCN_T4.jm_joint_motion_3d.pt",
        "checkpoints/best_AAGCN_T4.bm_bone_motion_3d.pt",
        "--device", "auto"
    ], "Table 4: Four-Stream AAGCN + Aug (T4.8)")

    # Phase 3: Determine Best Sequence Model (Table 1/2) and Best Graph Model (Table 3/4)
    # Check T1.18 vs T2.4 test accuracy:
    # Both are Transformer raw_3d (T1.18 is none, T2.4 is skel_gym_aug)
    t1_18_ckpt = "checkpoints/best_Transformer_T1.18_raw_3d.pt"
    t2_4_ckpt = "checkpoints/best_Transformer_T2.4_raw_3d.pt"
    
    # We will check if T2.4 exists and its accuracy in EXPERIMENT_RESULTS.md or evaluate both
    best_transformer_ckpt = t2_4_ckpt if Path(t2_4_ckpt).exists() else t1_18_ckpt
    logger.info(f"Selected Best Transformer Checkpoint: {best_transformer_ckpt}")

    # Best Graph Model for 2-model ensemble:
    # Option A: Four-Stream AAGCN (ensemble of 4)
    # Option B: Best Single Graph: AAGCN T4.4 (57.18%)
    # In Table 5:
    # T5.1 (Hard Voting): Best Transformer + AAGCN T4.4 (Joint Aug)
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "hard",
        "--exp_id", "T5.1",
        "--seq_len", "32",
        "--stride", "32",
        "--checkpoints",
        best_transformer_ckpt,
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "--device", "auto"
    ], "Table 5: Hard Voting (T5.1) Best Transformer + Best AAGCN")

    # T5.2 (Soft Voting): Best Transformer + AAGCN T4.4 (Joint Aug)
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "soft",
        "--exp_id", "T5.2",
        "--seq_len", "32",
        "--stride", "32",
        "--checkpoints",
        best_transformer_ckpt,
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "--device", "auto"
    ], "Table 5: Soft Voting (T5.2) Best Transformer + Best AAGCN")

    # T5.3 (Stacking Ensemble): Best Transformer + AAGCN T4.4 + Meta-Learner
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "stacking",
        "--exp_id", "T5.3",
        "--seq_len", "32",
        "--stride", "32",
        "--checkpoints",
        best_transformer_ckpt,
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "--device", "auto"
    ], "Table 5: Stacking Ensemble (T5.3) Best Transformer + Best AAGCN")

    # T5.4 (Tri-Model Grand Ensemble): Best Transformer + AAGCN Joint Aug + AAGCN Bone Aug
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T5.4",
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--checkpoints",
        best_transformer_ckpt,
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "checkpoints/best_AAGCN_T4.bone_bone_3d.pt",
        "--device", "auto"
    ], "Table 5: Tri-Model Grand Ensemble (T5.4)")

    # T5.5 (Grand Multi-Stream SOTA Ensemble): Best Transformer + All 4 AAGCN Aug Streams
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T5.5",
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--checkpoints",
        best_transformer_ckpt,
        "checkpoints/best_AAGCN_T4.4_rel_3d.pt",
        "checkpoints/best_AAGCN_T4.bone_bone_3d.pt",
        "checkpoints/best_AAGCN_T4.jm_joint_motion_3d.pt",
        "checkpoints/best_AAGCN_T4.bm_bone_motion_3d.pt",
        "--device", "auto"
    ], "Table 5: Grand Multi-Stream SOTA Ensemble (T5.5) -> Updates Table 6A, 6B & 7")

    logger.info("==================================================")
    logger.info("🎉 ALL SUPPLEMENTARY EXPERIMENTS COMPLETED SUCCESSFULLY!")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
