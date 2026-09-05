"""
Server Automated Execution & Keep-Alive Daemon.
Designed for high-throughput GPU environments (e.g. 96GB VRAM GPUs).
Features:
  - Anti-idle Heartbeat: keeps the remote session actively stimulated to prevent disconnection.
  - Table-based Execution: run specific tables (table1, table2, table2b, table3, table4) or all.
  - Automatic updates to outputs/EXPERIMENT_RESULTS.md via --exp_id.
  - Real-time checkpoint & report uploading to Hugging Face Hub.
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

TABLE1_EXPERIMENTS = [
    # LSTM on 7 feature sets
    {"name": "T1.1_LSTM_raw_2d", "exp_id": "T1.1", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_2d", "--exp_id", "T1.1", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.2_LSTM_rel_2d", "exp_id": "T1.2", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_2d", "--exp_id", "T1.2", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.3_LSTM_angle_2d", "exp_id": "T1.3", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_2d", "--exp_id", "T1.3", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.4_LSTM_raw_3d", "exp_id": "T1.4", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "raw_3d", "--exp_id", "T1.4", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.5_LSTM_rel_3d", "exp_id": "T1.5", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "rel_3d", "--exp_id", "T1.5", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.6_LSTM_angle_3d", "exp_id": "T1.6", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "angle_3d", "--exp_id", "T1.6", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.7_LSTM_mix", "exp_id": "T1.7", "cmd": ["run.py", "train", "--model", "LSTM", "--feature", "mix", "--exp_id", "T1.7", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    # BiLSTM on 7 feature sets
    {"name": "T1.8_BiLSTM_raw_2d", "exp_id": "T1.8", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_2d", "--exp_id", "T1.8", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.9_BiLSTM_rel_2d", "exp_id": "T1.9", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_2d", "--exp_id", "T1.9", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.10_BiLSTM_angle_2d", "exp_id": "T1.10", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_2d", "--exp_id", "T1.10", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.11_BiLSTM_raw_3d", "exp_id": "T1.11", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "raw_3d", "--exp_id", "T1.11", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.12_BiLSTM_rel_3d", "exp_id": "T1.12", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "rel_3d", "--exp_id", "T1.12", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.13_BiLSTM_angle_3d", "exp_id": "T1.13", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "angle_3d", "--exp_id", "T1.13", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.14_BiLSTM_mix", "exp_id": "T1.14", "cmd": ["run.py", "train", "--model", "BiLSTM", "--feature", "mix", "--exp_id", "T1.14", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    # Transformer on 7 feature sets
    {"name": "T1.15_Transformer_raw_2d", "exp_id": "T1.15", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_2d", "--exp_id", "T1.15", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.16_Transformer_rel_2d", "exp_id": "T1.16", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_2d", "--exp_id", "T1.16", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.17_Transformer_angle_2d", "exp_id": "T1.17", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_2d", "--exp_id", "T1.17", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.18_Transformer_raw_3d", "exp_id": "T1.18", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "raw_3d", "--exp_id", "T1.18", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.19_Transformer_rel_3d", "exp_id": "T1.19", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "rel_3d", "--exp_id", "T1.19", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.20_Transformer_angle_3d", "exp_id": "T1.20", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "angle_3d", "--exp_id", "T1.20", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T1.21_Transformer_mix", "exp_id": "T1.21", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--exp_id", "T1.21", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
]

TABLE2_EXPERIMENTS = [
    {"name": "T2.1_Transformer_mix_no_aug", "exp_id": "T2.1", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--augment", "none", "--exp_id", "T2.1", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T2.2_Transformer_mix_combined_aug", "exp_id": "T2.2", "cmd": ["run.py", "train", "--model", "Transformer", "--feature", "mix", "--augment", "combined", "--exp_id", "T2.2", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
]

TABLE4_EXPERIMENTS = [
    {"name": "T4.1_STGCN_raw_3d", "exp_id": "T4.1", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "raw_3d", "--exp_id", "T4.1", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.2_STGCN_rel_3d", "exp_id": "T4.2", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_3d", "--exp_id", "T4.2", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.3_STGCN_raw_2d", "exp_id": "T4.3", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "raw_2d", "--exp_id", "T4.3", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
    {"name": "T4.4_STGCN_rel_2d", "exp_id": "T4.4", "cmd": ["run.py", "train", "--model", "STGCN", "--feature", "rel_2d", "--exp_id", "T4.4", "--epochs", "100", "--batch_size", "16", "--device", "auto", "--use_amp", "--in_memory"]},
]

TABLE5_EXPERIMENTS = [
    {
        "name": "T5.1_Ensemble_HardVoting",
        "exp_id": "T5.1",
        "cmd": [
            "run.py", "ensemble",
            "--checkpoints", "checkpoints/best_Transformer_T2.2_mix.pt", "checkpoints/best_STGCN_T4.1_raw_3d.pt",
            "--method", "hard",
            "--exp_id", "T5.1",
            "--batch_size", "16",
            "--device", "auto"
        ]
    },
    {
        "name": "T5.2_Ensemble_SoftVoting",
        "exp_id": "T5.2",
        "cmd": [
            "run.py", "ensemble",
            "--checkpoints", "checkpoints/best_Transformer_T2.2_mix.pt", "checkpoints/best_STGCN_T4.1_raw_3d.pt",
            "--method", "soft",
            "--exp_id", "T5.2",
            "--batch_size", "16",
            "--device", "auto"
        ]
    },
    {
        "name": "T5.3_Ensemble_Stacking_SOTA",
        "exp_id": "T5.3",
        "cmd": [
            "run.py", "ensemble",
            "--checkpoints", "checkpoints/best_Transformer_T2.2_mix.pt", "checkpoints/best_STGCN_T4.1_raw_3d.pt",
            "--method", "stacking",
            "--exp_id", "T5.3",
            "--batch_size", "16",
            "--device", "auto"
        ]
    }
]

class ServerDaemon:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        heartbeat_interval: int = 30
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.heartbeat_interval = heartbeat_interval
        self.running = True
        self.current_experiment = "Idle"
        self.completed_experiments = []
        self.start_time = time.time()

        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
        ROOT_DIR.joinpath("outputs").mkdir(parents=True, exist_ok=True)

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

    def run_cmd(self, cmd_args: list) -> subprocess.CompletedProcess:
        full_cmd = [sys.executable] + cmd_args
        self.log(f"[RUNNING] {' '.join(full_cmd)}")
        res = subprocess.run(full_cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
        if res.returncode != 0:
            self.log(f"[ERROR] Process failed with exit code {res.returncode}:\n{res.stderr[-2000:]}")
        else:
            self.log("[SUCCESS] Command completed successfully.")
        return res

    def start_pipeline(self, experiments: list):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        self.log(f"ServerDaemon started for {len(experiments)} experiments.")
        for idx, exp in enumerate(experiments, 1):
            name = exp["name"]
            self.current_experiment = f"[{idx}/{len(experiments)}] {name}"
            self.log(f"\n========================================================")
            self.log(f"Starting Experiment: {self.current_experiment}")
            self.log(f"========================================================")

            t0 = time.time()
            res = self.run_cmd(exp["cmd"])
            duration = time.time() - t0

            exp_record = {
                "name": name,
                "exp_id": exp.get("exp_id", ""),
                "duration_seconds": duration,
                "status": "success" if res.returncode == 0 else "failed"
            }
            self.completed_experiments.append(exp_record)

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
        self.log("All planned experiments in this run finished!")

def main():
    parser = argparse.ArgumentParser(description="Server Daemon & Automated Batch Runner")
    parser.add_argument("--table", type=str, default="all", choices=["table1", "table2", "table4", "table5", "all"], help="Which table to run")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch per experiment for testing")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "mps", "auto"])
    args = parser.parse_args()

    table_map = {
        "table1": TABLE1_EXPERIMENTS,
        "table2": TABLE2_EXPERIMENTS,
        "table4": TABLE4_EXPERIMENTS,
        "table5": TABLE5_EXPERIMENTS,
    }

    if args.table == "all":
        experiments = TABLE1_EXPERIMENTS + TABLE2_EXPERIMENTS + TABLE4_EXPERIMENTS + TABLE5_EXPERIMENTS
    else:
        experiments = table_map[args.table]

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
        # For ensemble commands, auto-resolve existing best checkpoints if present
        if len(exp["cmd"]) > 1 and exp["cmd"][1] == "ensemble":
            ckpt_dir = ROOT_DIR / "checkpoints"
            trans_candidates = sorted(ckpt_dir.glob("best_Transformer_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            stgcn_candidates = sorted(ckpt_dir.glob("best_STGCN_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if trans_candidates and stgcn_candidates:
                best_t = str(trans_candidates[0].relative_to(ROOT_DIR))
                best_s = str(stgcn_candidates[0].relative_to(ROOT_DIR))
                try:
                    c_idx = new_cmd.index("--checkpoints")
                    new_cmd[c_idx + 1] = best_t
                    new_cmd[c_idx + 2] = best_s
                except (ValueError, IndexError):
                    pass
        exp["cmd"] = new_cmd

    daemon = ServerDaemon()
    daemon.start_pipeline(experiments)

if __name__ == "__main__":
    main()
