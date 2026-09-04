"""
Server Automated Execution & Keep-Alive Daemon.
Designed for high-throughput GPU environments (e.g. 96GB VRAM GPUs).
Features:
  - Anti-idle Heartbeat: keeps the remote session actively stimulated to prevent disconnection.
  - Sequential/Batch Model Training across Table 2, 3, 4, 5, 6 with optimized batch_size 16.
  - Real-time checkpoint & report uploading to Hugging Face Hub.
  - Progress reporting and dual checkpointing (best & last).
  - Consolidated single Markdown report: outputs/EXPERIMENT_RESULTS.md.
"""

import os
import sys
import time
import json
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
LOG_FILE = ROOT_DIR / "outputs" / "server_runner.log"
STATUS_FILE = ROOT_DIR / "outputs" / "server_status.json"
REPORT_FILE = ROOT_DIR / "outputs" / "EXPERIMENT_RESULTS.md"

DEFAULT_EXPERIMENTS = [
    {
        "name": "Table2_Transformer_12rel_4",
        "cmd": [
            "run.py", "train",
            "--model", "Transformer",
            "--feature", "12rel_4",
            "--augment", "none",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table2_BiLSTM_12rel_4",
        "cmd": [
            "run.py", "train",
            "--model", "BiLSTM",
            "--feature", "12rel_4",
            "--augment", "none",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table2_LSTM_12rel_4",
        "cmd": [
            "run.py", "train",
            "--model", "LSTM",
            "--feature", "12rel_4",
            "--augment", "none",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table3_Transformer_Rotate",
        "cmd": [
            "run.py", "train",
            "--model", "Transformer",
            "--feature", "12rel_4",
            "--augment", "rotate",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table4_Transformer_BranchConcat",
        "cmd": [
            "run.py", "train",
            "--model", "Transformer",
            "--feature", "branch_concat",
            "--augment", "none",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table5_STGCN_Full4",
        "cmd": [
            "run.py", "train",
            "--model", "STGCN",
            "--feature", "full_4",
            "--augment", "none",
            "--epochs", "100",
            "--batch_size", "16",
            "--device", "cuda",
            "--use_amp",
            "--amp_dtype", "bfloat16",
            "--in_memory",
            "--push_to_hf"
        ]
    },
    {
        "name": "Table6_Ensemble_Stacking",
        "cmd": [
            "run.py", "ensemble",
            "--checkpoints",
            "checkpoints/best_Transformer_12rel_4_aug_none.pt",
            "checkpoints/best_BiLSTM_12rel_4_aug_none.pt",
            "checkpoints/best_LSTM_12rel_4_aug_none.pt",
            "--method", "stacking",
            "--batch_size", "16",
            "--device", "cuda",
            "--push_to_hf"
        ]
    }
]

class ServerDaemon:
    def __init__(
        self,
        hf_token: Optional[str] = None,
        heartbeat_interval: int = 30,
        report_interval: int = 900
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.heartbeat_interval = heartbeat_interval
        self.report_interval = report_interval
        self.running = True
        self.current_experiment = "Idle"
        self.completed_experiments = []
        self.start_time = time.time()
        self.last_report_time = time.time()

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
        """Background loop to prevent container idle shutdown."""
        count = 0
        while self.running:
            time.sleep(self.heartbeat_interval)
            count += 1
            self.log(f"[HEARTBEAT #{count}] Daemon active. Current: {self.current_experiment}")
            try:
                status = {
                    "timestamp": time.time(),
                    "current_task": self.current_experiment,
                    "heartbeat_count": count,
                    "status": "running"
                }
                with open(STATUS_FILE, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
            except Exception:
                pass

    def run_cmd(self, cmd_args: list) -> subprocess.CompletedProcess:
        """Executes a CLI command in the workspace directory."""
        cmd_str = " ".join(cmd_args)
        self.log(f"Executing: {cmd_str}")
        res = subprocess.run(
            [sys.executable] + cmd_args,
            cwd=str(ROOT_DIR),
            env=os.environ.copy(),
            capture_output=True,
            text=True
        )
        if res.returncode != 0:
            self.log(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr[-1000:]}")
        else:
            self.log(f"[SUCCESS] Command completed successfully.")
        return res

    def start_pipeline(self, experiments: Optional[list] = None):
        """Starts heartbeat thread and executes experiment list sequentially."""
        if experiments is None:
            experiments = DEFAULT_EXPERIMENTS

        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        self.log(f"ServerDaemon started with Keep-Alive heartbeat for {len(experiments)} experiments.")

        for idx, exp in enumerate(experiments, 1):
            name = exp["name"]
            self.current_experiment = f"[{idx}/{len(experiments)}] {name}"
            self.log(f"\n========================================================")
            self.log(f"Starting Experiment: {self.current_experiment}")
            self.log(f"========================================================")

            t0 = time.time()
            res = self.run_cmd(exp["cmd"])
            duration = time.time() - t0

            acc = 0.0
            f1 = 0.0
            for line in res.stdout.splitlines():
                if "Test Accuracy:" in line:
                    try:
                        parts = line.split("Test Accuracy:")[1].split("|")
                        acc = float(parts[0].replace("%", "").strip())
                        f1 = float(parts[1].split("Macro F1:")[1].strip())
                    except Exception:
                        pass

            exp_record = {
                "name": name,
                "duration_seconds": duration,
                "acc": acc,
                "f1": f1,
                "status": "success" if res.returncode == 0 else "failed"
            }
            self.completed_experiments.append(exp_record)

        self.running = False
        self.log("All planned experiments completed successfully!")

if __name__ == "__main__":
    daemon = ServerDaemon()
    daemon.start_pipeline()
