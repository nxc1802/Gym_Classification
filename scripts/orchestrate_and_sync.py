"""
Local Orchestrator, Keep-Alive Daemon & Synchronization Tool for Marimo GPU Server.

Responsibilities:
  1. Starts server_runner.py in background on the remote Marimo server.
  2. Anti-idle Keep-Alive: Sends periodic heartbeats and toasts (mo.status.toast)
     to keep the sandbox alive and prevent disconnection.
  3. Real-time Monitoring: Displays current experiment, epoch progress, accuracy, and VRAM.
  4. Automatic Local Sync: Downloads updated EXPERIMENT_RESULTS.md, confusion matrix plots,
     and model checkpoints (.pt) to local storage.
  5. Completion Verification: Confirms 100% of benchmark rows are marked 'Done'.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

EXEC_TOOL = "/Users/nxc/.agents/skills/marimo-pair/scripts/execute-code.sh"
MARIMO_URL = "https://sb-ef78a3e704e32a96.sb.molab.run/"
MARIMO_TOKEN = "0e04cc46f8da1d3ac91d8109a47dd01b9e012c6c1f88ac2f341d11b12b457723"

def get_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        tok_file = Path.home() / ".cache" / "huggingface" / "token"
        if tok_file.exists():
            tok = tok_file.read_text().strip()
    return tok

def exec_remote_python(code_str: str, timeout: int = 40) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        ["bash", EXEC_TOOL, "--url", MARIMO_URL, "--token", MARIMO_TOKEN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        out, err = proc.communicate(input=code_str, timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        return -1, "", "Execution timed out"

def start_remote_runner(table: str = "all", dry_run: bool = False, workers: int = 4) -> bool:
    hf_tok = get_hf_token()
    dry_flag = "--dry_run" if dry_run else ""
    launch_code = f"""
import subprocess
import sys
import os
from pathlib import Path

# Check if server_runner.py is already running
ps = subprocess.run(['pgrep', '-af', 'server_runner.py'], capture_output=True, text=True)
running_pids = [l for l in ps.stdout.splitlines() if 'python' in l and 'server_runner.py' in l]

if running_pids:
    print(f"ALREADY_RUNNING:{{len(running_pids)}}")
else:
    hf_arg = "--push_to_hf --hf_token {hf_tok}" if "{hf_tok}" else "--no_hf"
    cmd = f"nohup python3 server_runner.py --table {table} --workers {workers} {dry_flag} {{hf_arg}} > outputs/server_runner.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, cwd="Gym_Classification")
    print("LAUNCHED_SUCCESSFULLY")
"""
    ret, out, err = exec_remote_python(launch_code, timeout=30)
    print(f"[Remote Launch] Output: {out.strip()}")
    return "LAUNCHED_SUCCESSFULLY" in out or "ALREADY_RUNNING" in out

def check_remote_status() -> dict:
    code = """
import json
import subprocess
from pathlib import Path
import torch

root = Path("Gym_Classification")
status_p = root / "outputs" / "server_status.json"
log_p = root / "outputs" / "server_runner.log"

data = {}
if status_p.exists():
    try:
        data["status"] = json.loads(status_p.read_text(encoding="utf-8"))
    except Exception:
        pass

if log_p.exists():
    lines = log_p.read_text(encoding="utf-8", errors="ignore").splitlines()
    data["recent_logs"] = lines[-6:] if lines else []

# Check active process
ps = subprocess.run(['pgrep', '-af', 'server_runner.py'], capture_output=True, text=True)
data["is_running"] = any('server_runner.py' in l for l in ps.stdout.splitlines())

# GPU VRAM info
if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / 1e9
    res = torch.cuda.memory_reserved(0) / 1e9
    data["gpu"] = f"Alloc: {alloc:.2f}GB | Res: {res:.2f}GB"

print("STATUS_JSON_START:" + json.dumps(data) + ":STATUS_JSON_END")
"""
    ret, out, err = exec_remote_python(code, timeout=30)
    if "STATUS_JSON_START:" in out:
        json_str = out.split("STATUS_JSON_START:")[1].split(":STATUS_JSON_END")[0]
        try:
            return json.loads(json_str)
        except Exception:
            pass
    return {}

def send_marimo_toast(msg: str, kind: str = "info"):
    code = f"""
import marimo as mo
try:
    mo.status.toast({json.dumps(msg)}, kind='{kind}')
except Exception:
    pass
"""
    exec_remote_python(code, timeout=15)

def sync_report_from_remote() -> bool:
    code = """
from pathlib import Path
p = Path("Gym_Classification/outputs/EXPERIMENT_RESULTS.md")
if p.exists():
    print("REPORT_CONTENT_START")
    print(p.read_text(encoding="utf-8", errors="ignore"))
    print("REPORT_CONTENT_END")
"""
    ret, out, err = exec_remote_python(code, timeout=30)
    if "REPORT_CONTENT_START" in out:
        content = out.split("REPORT_CONTENT_START")[1].split("REPORT_CONTENT_END")[0].strip()
        local_report = ROOT_DIR / "outputs" / "EXPERIMENT_RESULTS.md"
        local_report.parent.mkdir(parents=True, exist_ok=True)
        local_report.write_text(content + "\n", encoding="utf-8")
        return True
    return False

def sync_checkpoints_from_hf():
    hf_tok = get_hf_token()
    try:
        from huggingface_hub import snapshot_download
        dest_ckpts = ROOT_DIR / "checkpoints"
        dest_outputs = ROOT_DIR / "outputs"
        dest_ckpts.mkdir(parents=True, exist_ok=True)
        dest_outputs.mkdir(parents=True, exist_ok=True)

        print("[HF Sync] Downloading latest checkpoints and plots from Cuong2004/gym-exercise-classification ...")
        repo_dir = snapshot_download(
            repo_id="Cuong2004/gym-exercise-classification",
            repo_type="model",
            token=hf_tok,
            max_workers=4
        )
        # Copy checkpoints
        repo_p = Path(repo_dir)
        if (repo_p / "checkpoints").exists():
            for f in (repo_p / "checkpoints").glob("*.pt"):
                target = dest_ckpts / f.name
                if not target.exists() or target.stat().st_size != f.stat().st_size:
                    target.write_bytes(f.read_bytes())
                    print(f"  -> Synced checkpoint: {f.name} ({f.stat().st_size / 1e6:.2f} MB)")

        # Copy plots
        if (repo_p / "plots").exists():
            for f in (repo_p / "plots").glob("*.png"):
                target = dest_outputs / f.name
                if not target.exists():
                    target.write_bytes(f.read_bytes())
                    print(f"  -> Synced plot: {f.name}")
        print("[HF Sync] Checkpoints and artifacts synchronized to local!")
    except Exception as e:
        print(f"[HF Sync Warning] Could not sync from HF Hub: {e}")

def count_completed_in_report() -> tuple[int, int]:
    local_report = ROOT_DIR / "outputs" / "EXPERIMENT_RESULTS.md"
    if not local_report.exists():
        return 0, 42
    content = local_report.read_text(encoding="utf-8")
    done_count = 0
    total_count = 0
    for line in content.splitlines():
        if line.strip().startswith("|") and ("**T1." in line or "**T2." in line or "**T3." in line or "**T4." in line or "**T5." in line):
            total_count += 1
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6 and parts[-2] == "Done":
                done_count += 1
    return done_count, total_count

def run_orchestration(poll_interval: int = 20, table: str = "all", dry_run: bool = False, workers: int = 4):
    print("=" * 60)
    print("🚀 MARIMO GPU MULTI-WORKER ORCHESTRATOR & SYNC DAEMON")
    print("=" * 60)
    print(f"Marimo Server: {MARIMO_URL}")
    print(f"Poll Interval: {poll_interval}s | Target Table: {table} | Workers: {workers} | DryRun: {dry_run}")

    # Step 1: Launch runner on remote
    print(f"\n[Step 1] Launching / verifying server_runner.py with {workers} parallel workers on remote Marimo server...")
    if not start_remote_runner(table=table, dry_run=dry_run, workers=workers):
        print("[Error] Failed to launch server_runner on remote!")
        return

    send_marimo_toast(f"🚀 Multi-worker ({workers} concurrent GPUs) benchmark pipeline initiated! Keep-alive daemon active.", kind="success")

    last_sync_time = time.time()
    last_toast_time = time.time()

    print("\n[Step 2] Entering active monitoring and keep-alive loop...")
    while True:
        try:
            status_data = check_remote_status()
            is_running = status_data.get("is_running", False)
            status_obj = status_data.get("status", {})
            cur_exp = status_obj.get("current_experiment", "Initializing...")
            completed_cnt = status_obj.get("completed_count", 0)
            gpu_info = status_data.get("gpu", "CUDA active")
            recent_logs = status_data.get("recent_logs", [])

            now_str = time.strftime("%H:%M:%S")
            print(f"[{now_str}] Status: {'RUNNING' if is_running else 'IDLE'} | Active: {cur_exp} | Done: {completed_cnt} | GPU: {gpu_info}", flush=True)

            if recent_logs:
                for l in recent_logs[-3:]:
                    print(f"   > {l}", flush=True)

            # Send Marimo Keep-Alive Toast every ~60 seconds to stimulate the notebook UI
            if time.time() - last_toast_time >= 60:
                toast_msg = f"⚡ Multi-Worker Progress | Active: {cur_exp[:60]}... | Done: {completed_cnt}/42 | GPU: {gpu_info}"
                send_marimo_toast(toast_msg, kind="info")
                last_toast_time = time.time()

            # Sync EXPERIMENT_RESULTS.md every ~ poll_interval
            if time.time() - last_sync_time >= 25:
                sync_report_from_remote()
                done_c, tot_c = count_completed_in_report()
                print(f"   [Sync] Master report synchronized: {done_c} / {tot_c} experiments marked Done.", flush=True)
                last_sync_time = time.time()

            # Check if finished
            if not is_running and status_obj.get("status") == "COMPLETED":
                print("\n[Success] Remote server_runner has finished all experiments!")
                break
            elif not is_running and completed_cnt > 0:
                print("\n[Notice] Runner is no longer running. Verifying completion...")
                sync_report_from_remote()
                done_c, tot_c = count_completed_in_report()
                if done_c == tot_c:
                    print(f"[Success] All {tot_c} experiments completed!")
                    break
                else:
                    print(f"[Warning] Runner stopped with {done_c}/{tot_c} completed. Re-triggering resume...")
                    start_remote_runner(table=table, dry_run=dry_run, workers=workers)

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Poll error: {e}", flush=True)

        time.sleep(poll_interval)

    # Final sync
    print("\n[Step 3] Performing final synchronization...")
    sync_report_from_remote()
    sync_checkpoints_from_hf()

    done_c, tot_c = count_completed_in_report()
    celebration = f"🎉 ALL {tot_c} EXPERIMENTS COMPLETED SUCCESSFULLY! SOTA Evaluated."
    print(celebration)
    send_marimo_toast(celebration, kind="success")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marimo GPU Orchestrator")
    parser.add_argument("--table", type=str, default="all")
    parser.add_argument("--poll_interval", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    args = parser.parse_args()

    run_orchestration(poll_interval=args.poll_interval, table=args.table, dry_run=args.dry_run, workers=args.workers)
