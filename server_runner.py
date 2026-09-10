"""
Server Automated Execution & Keep-Alive Daemon.
Designed for high-throughput GPU environments (e.g. NVIDIA RTX PRO 6000 Blackwell 102GB VRAM).

Features:
  - Full sequential execution of Tables 1 -> 2 -> 3 -> 4 -> 5 -> 6/7 matching task.md.
  - Resume capability: skips already completed runs (Status == 'Done' in EXPERIMENT_RESULTS.md).
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
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "outputs" / "server_runner.log"
STATUS_FILE = ROOT_DIR / "outputs" / "server_status.json"
REPORT_FILE = ROOT_DIR / "outputs" / "EXPERIMENT_RESULTS.md"

# ==============================================================================
# TABLE 1: Temporal Models on Landmark Feature Sets (21 runs, SL=32)
# ==============================================================================
TABLE1_EXPERIMENTS = [
    # LSTM on 7 feature spaces
    {"name": "T1.1_LSTM_raw_2d", "exp_id": "T1.1", "ckpt": "checkpoints/best_LSTM_T1.1_raw_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_2d", "--exp_id", "T1.1", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.2_LSTM_rel_2d", "exp_id": "T1.2", "ckpt": "checkpoints/best_LSTM_T1.2_rel_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_2d", "--exp_id", "T1.2", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.3_LSTM_angle_2d", "exp_id": "T1.3", "ckpt": "checkpoints/best_LSTM_T1.3_angle_2d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_2d", "--exp_id", "T1.3", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.4_LSTM_raw_3d", "exp_id": "T1.4", "ckpt": "checkpoints/best_LSTM_T1.4_raw_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_3d", "--exp_id", "T1.4", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.5_LSTM_rel_3d", "exp_id": "T1.5", "ckpt": "checkpoints/best_LSTM_T1.5_rel_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_3d", "--exp_id", "T1.5", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.6_LSTM_angle_3d", "exp_id": "T1.6", "ckpt": "checkpoints/best_LSTM_T1.6_angle_3d.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_3d", "--exp_id", "T1.6", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.7_LSTM_mix", "exp_id": "T1.7", "ckpt": "checkpoints/best_LSTM_T1.7_mix.pt", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "mix", "--exp_id", "T1.7", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    # BiLSTM on 7 feature spaces
    {"name": "T1.8_BiLSTM_raw_2d", "exp_id": "T1.8", "ckpt": "checkpoints/best_BiLSTM_T1.8_raw_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_2d", "--exp_id", "T1.8", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.9_BiLSTM_rel_2d", "exp_id": "T1.9", "ckpt": "checkpoints/best_BiLSTM_T1.9_rel_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_2d", "--exp_id", "T1.9", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.10_BiLSTM_angle_2d", "exp_id": "T1.10", "ckpt": "checkpoints/best_BiLSTM_T1.10_angle_2d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_2d", "--exp_id", "T1.10", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.11_BiLSTM_raw_3d", "exp_id": "T1.11", "ckpt": "checkpoints/best_BiLSTM_T1.11_raw_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_3d", "--exp_id", "T1.11", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.12_BiLSTM_rel_3d", "exp_id": "T1.12", "ckpt": "checkpoints/best_BiLSTM_T1.12_rel_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_3d", "--exp_id", "T1.12", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.13_BiLSTM_angle_3d", "exp_id": "T1.13", "ckpt": "checkpoints/best_BiLSTM_T1.13_angle_3d.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_3d", "--exp_id", "T1.13", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.14_BiLSTM_mix", "exp_id": "T1.14", "ckpt": "checkpoints/best_BiLSTM_T1.14_mix.pt", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "mix", "--exp_id", "T1.14", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    # Transformer on 7 feature spaces
    {"name": "T1.15_Transformer_raw_2d", "exp_id": "T1.15", "ckpt": "checkpoints/best_Transformer_T1.15_raw_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_2d", "--exp_id", "T1.15", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.16_Transformer_rel_2d", "exp_id": "T1.16", "ckpt": "checkpoints/best_Transformer_T1.16_rel_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_2d", "--exp_id", "T1.16", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.17_Transformer_angle_2d", "exp_id": "T1.17", "ckpt": "checkpoints/best_Transformer_T1.17_angle_2d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_2d", "--exp_id", "T1.17", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.18_Transformer_raw_3d", "exp_id": "T1.18", "ckpt": "checkpoints/best_Transformer_T1.18_raw_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_3d", "--exp_id", "T1.18", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.19_Transformer_rel_3d", "exp_id": "T1.19", "ckpt": "checkpoints/best_Transformer_T1.19_rel_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--exp_id", "T1.19", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.20_Transformer_angle_3d", "exp_id": "T1.20", "ckpt": "checkpoints/best_Transformer_T1.20_angle_3d.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_3d", "--exp_id", "T1.20", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.21_Transformer_mix", "exp_id": "T1.21", "ckpt": "checkpoints/best_Transformer_T1.21_mix.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--exp_id", "T1.21", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
]

# ==============================================================================
# TABLE 2: Data Augmentation Strategies on Best Transformer (2 runs, SL=32)
# ==============================================================================
TABLE2_EXPERIMENTS = [
    {"name": "T2.1_Transformer_mix_none", "exp_id": "T2.1", "ckpt": "checkpoints/best_Transformer_T2.1_mix.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--augment", "none", "--exp_id", "T2.1", "--epochs", "100", "--patience", "10", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T2.2_Transformer_mix_skel_gym_aug", "exp_id": "T2.2", "ckpt": "checkpoints/best_Transformer_T2.2_mix.pt", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--augment", "skel_gym_aug", "--exp_id", "T2.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
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
        "ckpt": "outputs/ensemble/cm_ensemble_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T3.9", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "--device", "auto"]
    },
    {
        "name": "T3.10_FourStream_AAGCN",
        "exp_id": "T3.10",
        "ckpt": "outputs/ensemble/cm_ensemble_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T3.10", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt", "checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt", "--device", "auto"]
    },
]

# ==============================================================================
# TABLE 4: Data Augmentation Ablation on Graph Architectures (4 runs)
# ==============================================================================
TABLE4_EXPERIMENTS = [
    {"name": "T4.1_STGCN_rel_3d_none", "exp_id": "T4.1", "ckpt": "checkpoints/best_STGCN_T4.1_rel_3d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_3d", "--augment", "none", "--exp_id", "T4.1", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.2_STGCN_rel_3d_skel_gym_aug", "exp_id": "T4.2", "ckpt": "checkpoints/best_STGCN_T4.2_rel_3d.pt", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.2", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.3_AAGCN_rel_3d_none", "exp_id": "T4.3", "ckpt": "checkpoints/best_AAGCN_T4.3_rel_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "rel_3d", "--augment", "none", "--exp_id", "T4.3", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.4_AAGCN_rel_3d_skel_gym_aug", "exp_id": "T4.4", "ckpt": "checkpoints/best_AAGCN_T4.4_rel_3d.pt", "cmd": ["run.py", "train", "--model", "AAGCN", "--feature", "rel_3d", "--augment", "skel_gym_aug", "--exp_id", "T4.4", "--epochs", "100", "--patience", "10", "--video_level", "--device", "auto", "--use_amp", "--in_memory"]},
]

# ==============================================================================
# TABLE 5: Heterogeneous Cross-Paradigm Ensemble (5 runs)
# ==============================================================================
TABLE5_EXPERIMENTS = [
    {
        "name": "T5.1_Ensemble_HardVoting",
        "exp_id": "T5.1",
        "ckpt": "outputs/ensemble/cm_ensemble_hard.png",
        "cmd": ["run.py", "ensemble", "--method", "hard", "--exp_id", "T5.1", "--seq_len", "32", "--stride", "32", "--checkpoints", "checkpoints/best_Transformer_T1.21_mix.pt", "checkpoints/best_STGCN_T3.2_rel_3d.pt", "--device", "auto"]
    },
    {
        "name": "T5.2_Ensemble_SoftVoting",
        "exp_id": "T5.2",
        "ckpt": "outputs/ensemble/cm_ensemble_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "soft", "--exp_id", "T5.2", "--seq_len", "32", "--stride", "32", "--checkpoints", "checkpoints/best_Transformer_T1.21_mix.pt", "checkpoints/best_STGCN_T3.2_rel_3d.pt", "--device", "auto"]
    },
    {
        "name": "T5.3_Ensemble_Stacking",
        "exp_id": "T5.3",
        "ckpt": "outputs/ensemble/cm_ensemble_stacking.png",
        "cmd": ["run.py", "ensemble", "--method", "stacking", "--exp_id", "T5.3", "--seq_len", "32", "--stride", "32", "--checkpoints", "checkpoints/best_Transformer_T1.21_mix.pt", "checkpoints/best_STGCN_T3.2_rel_3d.pt", "--device", "auto"]
    },
    {
        "name": "T5.4_TriModel_GrandEnsemble",
        "exp_id": "T5.4",
        "ckpt": "outputs/ensemble/cm_ensemble_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T5.4", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_Transformer_T1.21_mix.pt", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "--device", "auto"]
    },
    {
        "name": "T5.5_Grand_5Stream_SOTA",
        "exp_id": "T5.5",
        "ckpt": "outputs/ensemble/cm_ensemble_weighted_soft.png",
        "cmd": ["run.py", "ensemble", "--method", "weighted_soft", "--exp_id", "T5.5", "--seq_len", "32", "--stride", "32", "--video_level", "--checkpoints", "checkpoints/best_Transformer_T1.21_mix.pt", "checkpoints/best_AAGCN_T3.5_rel_3d.pt", "checkpoints/best_AAGCN_T3.6_bone_3d.pt", "checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt", "checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt", "--device", "auto"]
    },
]

class ServerDaemon:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        heartbeat_interval: int = 15,
        resume: bool = True
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.heartbeat_interval = heartbeat_interval
        self.resume = resume
        self.running = True
        self.current_experiment = "Idle"
        self.completed_experiments = []
        self.start_time = time.time()

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        ROOT_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)
        ROOT_DIR.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{ts}] {msg}"
        print(formatted, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")

    def _heartbeat_loop(self):
        while self.running:
            status = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": int(time.time() - self.start_time),
                "current_experiment": self.current_experiment,
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
        
        # Check if row is Done in EXPERIMENT_RESULTS.md
        if REPORT_FILE.exists():
            content = REPORT_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                if f"**{exp_id}**" in line and line.strip().startswith("|"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 6 and parts[-2] == "Done":
                        # If checkpoint is specified, verify it exists
                        if ckpt_rel:
                            ckpt_p = ROOT_DIR / ckpt_rel
                            if ckpt_p.exists():
                                return True
                        else:
                            return True
        return False

    def run_cmd(self, cmd_args: list) -> subprocess.CompletedProcess:
        full_cmd = [sys.executable, "-u"] + cmd_args
        self.log(f"[RUNNING] {' '.join(full_cmd)}")
        proc = subprocess.Popen(
            full_cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        output_lines = []
        if proc.stdout:
            for line in proc.stdout:
                line_clean = line.rstrip()
                if line_clean:
                    self.log(line_clean)
                output_lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            self.log(f"[ERROR] Process failed with exit code {proc.returncode}")
        else:
            self.log("[SUCCESS] Command completed successfully.")
        return subprocess.CompletedProcess(full_cmd, proc.returncode, "".join(output_lines), "")

    def start_pipeline(self, experiments: list):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        self.log(f"ServerDaemon started for {len(experiments)} total experiments (Resume={self.resume}).")
        
        for idx, exp in enumerate(experiments, 1):
            name = exp["name"]
            exp_id = exp.get("exp_id", "")
            
            # Check if experiment is already done
            if self.is_experiment_completed(exp):
                self.log(f"[SKIP] Experiment [{idx}/{len(experiments)}] {name} ({exp_id}) is already completed.")
                self.completed_experiments.append({
                    "name": name,
                    "exp_id": exp_id,
                    "status": "skipped_already_done",
                    "duration_seconds": 0
                })
                continue

            self.current_experiment = f"[{idx}/{len(experiments)}] {name}"
            self.log(f"\n========================================================")
            self.log(f"Starting Experiment: {self.current_experiment}")
            self.log(f"========================================================")

            t0 = time.time()
            res = self.run_cmd(exp["cmd"])
            duration = time.time() - t0

            exp_record = {
                "name": name,
                "exp_id": exp_id,
                "duration_seconds": duration,
                "status": "success" if res.returncode == 0 else "failed"
            }
            self.completed_experiments.append(exp_record)
            
            # Push master report after each successful experiment if token present
            if res.returncode == 0 and self.hf_token and REPORT_FILE.exists():
                try:
                    from src.utils.hf_hub import upload_file_to_hf
                    upload_file_to_hf(
                        local_path=str(REPORT_FILE),
                        path_in_repo="EXPERIMENT_RESULTS.md",
                        repo_id="Cuong2004/gym-exercise-classification",
                        token=self.hf_token,
                        commit_message=f"Update benchmark report after {name}"
                    )
                except Exception as e:
                    self.log(f"[HF Hub Warning] Could not push intermediate report: {e}")

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
                    commit_message="Master Benchmark Results (All Tables 1-7 completed)"
                )
                self.log("[HF Hub] EXPERIMENT_RESULTS.md uploaded successfully!")
            except Exception as e:
                self.log(f"[HF Hub Warning] Could not upload EXPERIMENT_RESULTS.md: {e}")

        self.log("All planned experiments finished!")

def main():
    parser = argparse.ArgumentParser(description="Server Daemon & Automated Batch Runner")
    parser.add_argument("--table", type=str, default="all", choices=["table1", "table2", "table3", "table4", "table5", "all"], help="Which table to run")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch per experiment for testing")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--push_to_hf", action="store_true", default=True, help="Push checkpoints and reports to HF Hub")
    parser.add_argument("--no_hf", dest="push_to_hf", action="store_false", help="Disable HF Hub upload")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face auth token")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume, re-run all")
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

    daemon = ServerDaemon(hf_token=hf_token, resume=not args.no_resume)
    daemon.start_pipeline(experiments)

if __name__ == "__main__":
    main()
