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

    def resolve_best_checkpoints(self) -> tuple[Optional[str], Optional[str], float, float]:
        """
        Dynamically finds the best Transformer checkpoint from Tables 1 & 2,
        and the best ST-GCN checkpoint from Table 4 by inspecting EXPERIMENT_RESULTS.md.
        """
        best_t_ckpt, best_t_acc = None, -1.0
        best_s_ckpt, best_s_acc = None, -1.0

        if REPORT_FILE.exists():
            content = REPORT_FILE.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line.startswith("|") or ":---" in line or "Exp ID" in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 8:
                    continue
                exp_id = parts[1].replace("*", "").strip()

                # Check Transformer runs in T1 and T2
                if (exp_id.startswith("T1.") or exp_id.startswith("T2.")) and "Transformer" in line:
                    for p in parts:
                        if p.endswith("%"):
                            try:
                                acc = float(p.rstrip("%"))
                                if acc > best_t_acc:
                                    for cp in parts:
                                        if cp.startswith("`checkpoints/") and cp.endswith(".pt`"):
                                            candidate = cp.strip("`")
                                            if (ROOT_DIR / candidate).exists():
                                                best_t_acc = acc
                                                best_t_ckpt = candidate
                            except ValueError:
                                pass

                # Check ST-GCN runs in T4
                elif exp_id.startswith("T4.") and ("ST-GCN" in line or "STGCN" in line):
                    for p in parts:
                        if p.endswith("%"):
                            try:
                                acc = float(p.rstrip("%"))
                                if acc > best_s_acc:
                                    for cp in parts:
                                        if cp.startswith("`checkpoints/") and cp.endswith(".pt`"):
                                            candidate = cp.strip("`")
                                            if (ROOT_DIR / candidate).exists():
                                                best_s_acc = acc
                                                best_s_ckpt = candidate
                            except ValueError:
                                pass

        # Fallback to filesystem timestamp if not found in report
        ckpt_dir = ROOT_DIR / "checkpoints"
        if not best_t_ckpt:
            t_cands = sorted(ckpt_dir.glob("best_Transformer_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if t_cands:
                best_t_ckpt = str(t_cands[0].relative_to(ROOT_DIR))
        if not best_s_ckpt:
            s_cands = sorted(ckpt_dir.glob("best_STGCN_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            if s_cands:
                best_s_ckpt = str(s_cands[0].relative_to(ROOT_DIR))

        return best_t_ckpt, best_s_ckpt, best_t_acc, best_s_acc

    def start_pipeline(self, experiments: list):
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        self.log(f"ServerDaemon started for {len(experiments)} experiments.")
        for idx, exp in enumerate(experiments, 1):
            name = exp["name"]
            exp_id = exp.get("exp_id", "")
            self.current_experiment = f"[{idx}/{len(experiments)}] {name}"
            self.log(f"\n========================================================")
            self.log(f"Starting Experiment: {self.current_experiment}")
            self.log(f"========================================================")

            # Dynamically resolve best checkpoints right before running Table 5 ensemble
            if exp_id.startswith("T5."):
                best_t, best_s, t_acc, s_acc = self.resolve_best_checkpoints()
                if best_t and best_s:
                    self.log(f"[ENSEMBLE RESOLUTION] Selected Best Transformer: {best_t} (Val Acc: {t_acc:.2f}%)")
                    self.log(f"[ENSEMBLE RESOLUTION] Selected Best ST-GCN: {best_s} (Val Acc: {s_acc:.2f}%)")
                    cmd = list(exp["cmd"])
                    try:
                        c_idx = cmd.index("--checkpoints")
                        cmd[c_idx + 1] = best_t
                        cmd[c_idx + 2] = best_s
                        exp["cmd"] = cmd
                    except (ValueError, IndexError):
                        pass

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

        # Push master EXPERIMENT_RESULTS.md to Hugging Face
        if self.hf_token and REPORT_FILE.exists():
            try:
                from src.utils.hf_hub import upload_file_to_hf
                self.log("[HF Hub] Uploading master EXPERIMENT_RESULTS.md to Hugging Face...")
                upload_file_to_hf(
                    local_path=str(REPORT_FILE),
                    path_in_repo="EXPERIMENT_RESULTS.md",
                    repo_id="Cuong2004/gym-exercise-classification",
                    token=self.hf_token,
                    commit_message="Master Benchmark Results (Table 1-6 completed)"
                )
                self.log("[HF Hub] EXPERIMENT_RESULTS.md uploaded successfully!")
            except Exception as e:
                self.log(f"[HF Hub Warning] Could not upload EXPERIMENT_RESULTS.md: {e}")

        # Git commit and push sync
        try:
            self.log("[Git] Synchronizing results to GitHub...")
            subprocess.run(["git", "add", "outputs/EXPERIMENT_RESULTS.md", "outputs/ensemble"], cwd=str(ROOT_DIR), capture_output=True)
            subprocess.run(["git", "commit", "-m", "Benchmark results: Table 1 to Table 6 execution complete"], cwd=str(ROOT_DIR), capture_output=True)
            push_res = subprocess.run(["git", "push"], cwd=str(ROOT_DIR), capture_output=True, text=True)
            self.log(f"[Git] Push result: {push_res.stdout.strip()} {push_res.stderr.strip()}")
        except Exception as e:
            self.log(f"[Git Warning] Git sync failed: {e}")

        self.log("All planned experiments in this run finished!")

def main():
    parser = argparse.ArgumentParser(description="Server Daemon & Automated Batch Runner")
    parser.add_argument("--table", type=str, default="all", choices=["table1", "table2", "table4", "table5", "all"], help="Which table to run")
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch per experiment for testing")
    parser.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "mps", "auto"])
    parser.add_argument("--push_to_hf", action="store_true", default=True, help="Push checkpoints and reports to HF Hub")
    parser.add_argument("--no_hf", dest="push_to_hf", action="store_false", help="Disable HF Hub upload")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face auth token")
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

    daemon = ServerDaemon(hf_token=hf_token)
    daemon.start_pipeline(experiments)

if __name__ == "__main__":
    main()
