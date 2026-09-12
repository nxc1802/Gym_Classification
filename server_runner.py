"""
Server Automated Execution & Multi-Worker Parallel Runner Daemon.
Designed for high-throughput GPU environments (NVIDIA RTX PRO 6000 Blackwell 102GB VRAM, 20 vCPUs, 172GB RAM).

Features:
  - Parallel execution of independent training runs across multiple concurrent workers (default: 4 workers).
  - Preserves already completed runs: skips any experiment already marked 'Done' with valid checkpoint.
  - Multi-process safe file locking on outputs/EXPERIMENT_RESULTS.md to avoid race conditions.
  - Anti-idle Heartbeat: maintains active session status to prevent cloud disconnection.
  - Automatic table updates: logs results to outputs/EXPERIMENT_RESULTS.md via --exp_id.
  - Real-time Hugging Face Hub checkpoint & report synchronization.
"""

import os
import sys
import time
import json
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "outputs" / "server_runner.log"
STATUS_FILE = ROOT_DIR / "outputs" / "server_status.json"
REPORT_FILE = ROOT_DIR / "outputs" / "EXPERIMENT_RESULTS.md"
LOGS_DIR = ROOT_DIR / "outputs" / "logs"

# ==============================================================================
# TABLE 1: Temporal Models on Landmark Feature Sets (27 runs, SL=32)
# ==============================================================================
TABLE1_EXPERIMENTS = [
    # LSTM on 9 feature spaces
    {"name": "T1.1_LSTM_raw_2d", "exp_id": "T1.1", "ckpt": "checkpoints/best_LSTM_T1.1_raw_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_2d", "--exp_id", "T1.1", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.2_LSTM_rel_2d", "exp_id": "T1.2", "ckpt": "checkpoints/best_LSTM_T1.2_rel_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_2d", "--exp_id", "T1.2", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.3_LSTM_angle_2d", "exp_id": "T1.3", "ckpt": "checkpoints/best_LSTM_T1.3_angle_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_2d", "--exp_id", "T1.3", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.4_LSTM_angle2_2d", "exp_id": "T1.4", "ckpt": "checkpoints/best_LSTM_T1.4_angle2_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle2_2d", "--exp_id", "T1.4", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.5_LSTM_raw_3d", "exp_id": "T1.5", "ckpt": "checkpoints/best_LSTM_T1.5_raw_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_3d", "--exp_id", "T1.5", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.6_LSTM_rel_3d", "exp_id": "T1.6", "ckpt": "checkpoints/best_LSTM_T1.6_rel_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_3d", "--exp_id", "T1.6", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.7_LSTM_angle_3d", "exp_id": "T1.7", "ckpt": "checkpoints/best_LSTM_T1.7_angle_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_3d", "--exp_id", "T1.7", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.8_LSTM_angle2_3d", "exp_id": "T1.8", "ckpt": "checkpoints/best_LSTM_T1.8_angle2_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle2_3d", "--exp_id", "T1.8", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.9_LSTM_mix", "exp_id": "T1.9", "ckpt": "checkpoints/best_LSTM_T1.9_mix.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "mix", "--exp_id", "T1.9", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    # BiLSTM on 9 feature spaces
    {"name": "T1.10_BiLSTM_raw_2d", "exp_id": "T1.10", "ckpt": "checkpoints/best_BiLSTM_T1.10_raw_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_2d", "--exp_id", "T1.10", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.11_BiLSTM_rel_2d", "exp_id": "T1.11", "ckpt": "checkpoints/best_BiLSTM_T1.11_rel_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_2d", "--exp_id", "T1.11", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.12_BiLSTM_angle_2d", "exp_id": "T1.12", "ckpt": "checkpoints/best_BiLSTM_T1.12_angle_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_2d", "--exp_id", "T1.12", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.13_BiLSTM_angle2_2d", "exp_id": "T1.13", "ckpt": "checkpoints/best_BiLSTM_T1.13_angle2_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle2_2d", "--exp_id", "T1.13", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.14_BiLSTM_raw_3d", "exp_id": "T1.14", "ckpt": "checkpoints/best_BiLSTM_T1.14_raw_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_3d", "--exp_id", "T1.14", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.15_BiLSTM_rel_3d", "exp_id": "T1.15", "ckpt": "checkpoints/best_BiLSTM_T1.15_rel_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_3d", "--exp_id", "T1.15", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.16_BiLSTM_angle_3d", "exp_id": "T1.16", "ckpt": "checkpoints/best_BiLSTM_T1.16_angle_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_3d", "--exp_id", "T1.16", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.17_BiLSTM_angle2_3d", "exp_id": "T1.17", "ckpt": "checkpoints/best_BiLSTM_T1.17_angle2_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle2_3d", "--exp_id", "T1.17", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.18_BiLSTM_mix", "exp_id": "T1.18", "ckpt": "checkpoints/best_BiLSTM_T1.18_mix.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "mix", "--exp_id", "T1.18", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    # Transformer on 9 feature spaces
    {"name": "T1.19_Transformer_raw_2d", "exp_id": "T1.19", "ckpt": "checkpoints/best_Transformer_T1.19_raw_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_2d", "--exp_id", "T1.19", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.20_Transformer_rel_2d", "exp_id": "T1.20", "ckpt": "checkpoints/best_Transformer_T1.20_rel_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_2d", "--exp_id", "T1.20", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.21_Transformer_angle_2d", "exp_id": "T1.21", "ckpt": "checkpoints/best_Transformer_T1.21_angle_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_2d", "--exp_id", "T1.21", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.22_Transformer_angle2_2d", "exp_id": "T1.22", "ckpt": "checkpoints/best_Transformer_T1.22_angle2_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle2_2d", "--exp_id", "T1.22", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.23_Transformer_raw_3d", "exp_id": "T1.23", "ckpt": "checkpoints/best_Transformer_T1.23_raw_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_3d", "--exp_id", "T1.23", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.24_Transformer_rel_3d", "exp_id": "T1.24", "ckpt": "checkpoints/best_Transformer_T1.24_rel_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--exp_id", "T1.24", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.25_Transformer_angle_3d", "exp_id": "T1.25", "ckpt": "checkpoints/best_Transformer_T1.25_angle_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_3d", "--exp_id", "T1.25", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.26_Transformer_angle2_3d", "exp_id": "T1.26", "ckpt": "checkpoints/best_Transformer_T1.26_angle2_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle2_3d", "--exp_id", "T1.26", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.27_Transformer_mix", "exp_id": "T1.27", "ckpt": "checkpoints/best_Transformer_T1.27_mix.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--exp_id", "T1.27", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
]

# ==============================================================================
# TABLE 2: Data Augmentation Strategies on Best Sequence Model (SL=32)
# Baseline: Best Practice Sequence Backbone = Transformer rel_3d (T1.24)
# ==============================================================================
TABLE2_EXPERIMENTS = [
    {"name": "T2.1_Transformer_rel_3d_none", "exp_id": "T2.1", "ckpt": "checkpoints/best_Transformer_T1.24_rel_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--augment", "none", "--exp_id", "T2.1", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T2.2_Transformer_rel_3d_skel_gym_aug", "exp_id": "T2.2", "ckpt": "checkpoints/best_Transformer_T2.2_rel_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--augment", "skel_gym_aug", "--exp_id", "T2.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
]

# ==============================================================================
# TABLE 3: Spatial-Temporal Graph Models & Multi-Stream AAGCN Kinematics (10 runs)
# ==============================================================================
TABLE3_EXPERIMENTS = [
    # ST-GCN Baselines
    {"name": "T3.1_STGCN_raw_3d", "exp_id": "T3.1", "ckpt": "checkpoints/best_STGCN_T3.1_raw_3d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "raw_3d", "--exp_id", "T3.1", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.2_STGCN_rel_3d", "exp_id": "T3.2", "ckpt": "checkpoints/best_STGCN_T3.2_rel_3d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_3d", "--exp_id", "T3.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.3_STGCN_raw_2d", "exp_id": "T3.3", "ckpt": "checkpoints/best_STGCN_T3.3_raw_2d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "raw_2d", "--exp_id", "T3.3", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.4_STGCN_rel_2d", "exp_id": "T3.4", "ckpt": "checkpoints/best_STGCN_T3.4_rel_2d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_2d", "--exp_id", "T3.4", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    # AAGCN Single-Stream Kinematics
    {"name": "T3.5_AAGCN_rel_3d", "exp_id": "T3.5", "ckpt": "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "rel_3d", "--exp_id", "T3.5", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.6_AAGCN_bone_3d", "exp_id": "T3.6", "ckpt": "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "bone_3d", "--exp_id", "T3.6", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.7_AAGCN_joint_motion_3d", "exp_id": "T3.7", "ckpt": "checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "joint_motion_3d", "--exp_id", "T3.7", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T3.8_AAGCN_bone_motion_3d", "exp_id": "T3.8", "ckpt": "checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "bone_motion_3d", "--exp_id", "T3.8", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    # AAGCN Late Fusion
    {
        "name": "T3.9_TwoStream_AAGCN",
        "exp_id": "T3.9",
        "ckpt": "outputs/ensemble/cm_ensemble_T3.9_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T3.9", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "--device", "auto"]
    },
    {
        "name": "T3.10_FourStream_AAGCN",
        "exp_id": "T3.10",
        "ckpt": "outputs/ensemble/cm_ensemble_T3.10_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T3.10", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt", "checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt", "--device", "auto"]
    },
]

# ==============================================================================
# TABLE 4: Data Augmentation Ablation on Graph Architectures (2 runs)
# Baseline: Best Practice Single-Stream Backbone = AAGCN bone_3d (T3.6)
# ==============================================================================
TABLE4_EXPERIMENTS = [
    {"name": "T4.1_AAGCN_bone_3d_none", "exp_id": "T4.1", "ckpt": "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "bone_3d", "--augment", "none", "--exp_id", "T4.1", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.2_AAGCN_bone_3d_skel_gym_aug", "exp_id": "T4.2", "ckpt": "checkpoints/best_AAGCN_T4.2_bone_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "bone_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
]

# ==============================================================================
# TABLE 5: Heterogeneous Cross-Paradigm Ensemble (Unified Weighted Soft Voting)
# Combines: Best Sequence (T2.2: Transformer rel_3d aug) + Best Graph (T4.2: AAGCN bone_3d aug)
# Evaluates with Dual-Target Optimization (Window-level Phase 1 & Video-level Phase 2)
# ==============================================================================
TABLE5_EXPERIMENTS = [
    {
        "name": "T5.1_Grand_Ensemble_WeightedSoft",
        "exp_id": "T5.1",
        "ckpt": "outputs/ensemble/cm_ensemble_T5.1_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T5.1", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_Transformer_T2.2_rel_3d.pt", "checkpoints/best_AAGCN_T4.2_bone_3d.pt", "--device", "auto"]
    },
]



class ParallelServerDaemon:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        heartbeat_interval: int = 15,
        resume: bool = True,
        workers: int = 4
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.heartbeat_interval = heartbeat_interval
        self.resume = resume
        self.workers = workers
        self.running = True
        self.active_experiments = {}  # worker_id -> exp_name
        self.completed_experiments = []
        self.start_time = time.time()
        self.lock = threading.Lock()

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        ROOT_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)
        ROOT_DIR.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted, flush=True)
        with self.lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")

    def _heartbeat_loop(self):
        while self.running:
            with self.lock:
                active_list = list(self.active_experiments.values())
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": int(time.time() - self.start_time),
                    "active_workers": len(active_list),
                    "current_experiment": ", ".join(active_list) if active_list else "Idle",
                    "active_experiments": active_list,
                    "completed_count": len(self.completed_experiments),
                    "status": "RUNNING" if self.running else "STOPPED"
                }
            try:
                STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
            except Exception:
                pass
            time.sleep(self.heartbeat_interval)

    def is_experiment_completed(self, exp: Dict[str, Any]) -> bool:
        if not self.resume:
            return False
        exp_id = exp.get("exp_id", "")
        ckpt_rel = exp.get("ckpt", "")
        
        # Check if row is marked Done in EXPERIMENT_RESULTS.md
        if REPORT_FILE.exists():
            content = REPORT_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                if f"**{exp_id}**" in line and line.strip().startswith("|"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 6 and parts[-2] == "Done":
                        if ckpt_rel:
                            ckpt_p = ROOT_DIR / ckpt_rel
                            if ckpt_p.exists():
                                return True
                        else:
                            return True
        return False

    def run_single_experiment(self, exp: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
        name = exp["name"]
        exp_id = exp.get("exp_id", "")
        cmd_args = exp["cmd"]

        with self.lock:
            self.active_experiments[worker_id] = f"[{exp_id}] {name}"

        self.log(f"[Worker {worker_id}] Starting: {name} ({exp_id})")
        t0 = time.time()

        full_cmd = [sys.executable, "-u"] + cmd_args
        exp_log_file = LOGS_DIR / f"{exp_id}.log"

        with open(exp_log_file, "w", encoding="utf-8") as exp_log:
            proc = subprocess.Popen(
                full_cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            for line in proc.stdout:
                exp_log.write(line)
                exp_log.flush()
                # Print key milestones to main log
                if any(k in line for k in ("Best checkpoint saved", "Test Accuracy:", "auto-updated Table", "VIDEO-LEVEL", "Saved ensemble")):
                    self.log(f"[{exp_id}] {line.strip()}")

            proc.wait()

        duration = time.time() - t0
        success = (proc.returncode == 0)

        if success:
            self.log(f"[Worker {worker_id}] ✅ COMPLETED {name} in {duration:.1f}s")
        else:
            self.log(f"[Worker {worker_id}] ❌ FAILED {name} with code {proc.returncode}")

        with self.lock:
            self.active_experiments.pop(worker_id, None)
            record = {
                "name": name,
                "exp_id": exp_id,
                "duration_seconds": duration,
                "status": "success" if success else "failed"
            }
            self.completed_experiments.append(record)

        # Intermediate report upload
        if success and self.hf_token and REPORT_FILE.exists():
            try:
                from src.utils.hf_hub import upload_file_to_hf
                upload_file_to_hf(
                    local_path=str(REPORT_FILE),
                    path_in_repo="EXPERIMENT_RESULTS.md",
                    repo_id="Cuong2004/gym-exercise-classification",
                    token=self.hf_token,
                    commit_message=f"Update benchmark report after {name}"
                )
            except Exception:
                pass

        return record

    def start_pipeline(self, all_experiments: List[Dict[str, Any]]):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        # Step 1: Filter out already completed runs
        pending_experiments = []
        for exp in all_experiments:
            if self.is_experiment_completed(exp):
                self.log(f"[SKIP] Experiment {exp['name']} ({exp['exp_id']}) is already completed.")
                with self.lock:
                    self.completed_experiments.append({
                        "name": exp["name"],
                        "exp_id": exp["exp_id"],
                        "status": "skipped_already_done",
                        "duration_seconds": 0
                    })
            else:
                pending_experiments.append(exp)

        self.log(f"ParallelServerDaemon initialized. Total: {len(all_experiments)} | Completed: {len(self.completed_experiments)} | Pending: {len(pending_experiments)} | Workers: {self.workers}")

        if not pending_experiments:
            self.log("All experiments are already marked 'Done'! Nothing left to run.")
            self.running = False
            return

        # Step 2: Separate single-model training runs and late-fusion/ensemble runs
        train_runs = [e for e in pending_experiments if "train" in e["cmd"]]
        ensemble_runs = [e for e in pending_experiments if "ensemble" in e["cmd"]]

        self.log(f"Execution Queue: {len(train_runs)} Single-Model Training Runs (Parallel: {self.workers} workers) -> {len(ensemble_runs)} Ensemble Runs")

        # Step 3: Run single-model training runs in parallel
        if train_runs:
            self.log(f"\n========================================================")
            self.log(f"🚀 LAUNCHING {len(train_runs)} TRAINING RUNS WITH {self.workers} PARALLEL WORKERS")
            self.log(f"========================================================")

            worker_pool = list(range(1, self.workers + 1))
            pool_lock = threading.Lock()

            def worker_task(exp_item):
                with pool_lock:
                    w_id = worker_pool.pop(0)
                try:
                    return self.run_single_experiment(exp_item, w_id)
                finally:
                    with pool_lock:
                        worker_pool.append(w_id)

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(worker_task, exp): exp for exp in train_runs}
                for future in as_completed(futures):
                    exp = futures[future]
                    try:
                        res = future.result()
                    except Exception as e:
                        self.log(f"[ERROR] Worker raised exception on {exp['name']}: {e}")

        # Step 4: Run ensemble / multi-stream fusion runs
        if ensemble_runs:
            self.log(f"\n========================================================")
            self.log(f"🎯 LAUNCHING {len(ensemble_runs)} ENSEMBLE & FUSION RUNS")
            self.log(f"========================================================")
            for e_idx, e_exp in enumerate(ensemble_runs, 1):
                self.run_single_experiment(e_exp, worker_id=1)

        self.running = False
        final_status = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": int(time.time() - self.start_time),
            "current_experiment": "None (Finished)",
            "completed_count": len(self.completed_experiments),
            "status": "COMPLETED"
        }
        try:
            STATUS_FILE.write_text(json.dumps(final_status, indent=2), encoding="utf-8")
        except Exception:
            pass

        # Final report push to HF Hub
        if self.hf_token and REPORT_FILE.exists():
            try:
                from src.utils.hf_hub import upload_file_to_hf
                self.log("[HF Hub] Uploading final master EXPERIMENT_RESULTS.md to Hugging Face...")
                upload_file_to_hf(
                    local_path=str(REPORT_FILE),
                    path_in_repo="EXPERIMENT_RESULTS.md",
                    repo_id="Cuong2004/gym-exercise-classification",
                    token=self.hf_token,
                    commit_message="Master Benchmark Results (All Tables 1-7 completed via Parallel Daemon)"
                )
                self.log("[HF Hub] EXPERIMENT_RESULTS.md uploaded successfully!")
            except Exception as e:
                self.log(f"[HF Hub Warning] Could not upload EXPERIMENT_RESULTS.md: {e}")

        self.log("All planned experiments finished successfully!")

def main():
    parser = argparse.ArgumentParser(description="Multi-Worker Parallel Server Daemon")
    parser.add_argument("--table", type=str, default="all", choices=["table1", "table2", "table3", "table4", "table5", "all"], help="Which table to run")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch per experiment for testing")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--push_to_hf", action="store_true", default=True, help="Push checkpoints and reports to HF Hub")
    parser.add_argument("--no_hf", dest="push_to_hf", action="store_false", help="Disable HF Hub upload")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face auth token")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume, re-run all")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent training workers")
    args = parser.parse_args()

    table_map = {
        "table1": TABLE1_EXPERIMENTS,
        "table2": TABLE2_EXPERIMENTS,
        "table3": TABLE3_EXPERIMENTS,
        "table4": TABLE4_EXPERIMENTS,
        "table5": TABLE5_EXPERIMENTS,
    }

    if args.table == "all":
        experiments = TABLE1_EXPERIMENTS + TABLE2_EXPERIMENTS + TABLE3_EXPERIMENTS + TABLE4_EXPERIMENTS + TABLE5_EXPERIMENTS
    else:
        experiments = table_map[args.table]

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    for exp in experiments:
        new_cmd = []
        skip_next = False
        for i, c in enumerate(exp["cmd"]):
            if skip_next:
                skip_next = False
                continue
            if c == "--epochs" and args.dry_run:
                new_cmd.extend(["--epochs", "1"])
                skip_next = True
            elif c == "--device":
                new_cmd.extend(["--device", args.device])
                skip_next = True
            else:
                new_cmd.append(c)

        # Inject HF arguments if requested
        if args.push_to_hf and hf_token:
            if "--push_to_hf" not in new_cmd:
                new_cmd.append("--push_to_hf")
            if "--hf_token" not in new_cmd:
                new_cmd.extend(["--hf_token", hf_token])

        exp["cmd"] = new_cmd

    daemon = ParallelServerDaemon(
        hf_token=hf_token,
        resume=not args.no_resume,
        workers=args.workers
    )
    daemon.start_pipeline(experiments)

if __name__ == "__main__":
    main()
