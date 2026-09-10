import sys
import time
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.hf_hub import pull_landmarks_from_hf

exec_tool = "/Users/nxc/.agents/skills/marimo-pair/scripts/execute-code.sh"
url = "https://sb-ef78a3e704e32a96.sb.molab.run/"
marimo_tok = "0e04cc46f8da1d3ac91d8109a47dd01b9e012c6c1f88ac2f341d11b12b457723"

check_code = """
import subprocess
from pathlib import Path

ps = subprocess.run(['pgrep', '-af', 'marimo_extract_landmarks_complexity2'], capture_output=True, text=True)
pids = [l.split()[0] for l in ps.stdout.strip().splitlines() if l.strip()]

p = Path('/marimo/Gym_Classification/data/landmarks')
csv_count = len(list(p.glob('**/*.csv'))) if p.exists() else 0

log_p = Path('/marimo/extract_landmarks.log')
last_lines = []
if log_p.exists():
    last_lines = [l for l in log_p.read_text().splitlines() if 'Progress:' in l or 'Extract' in l or 'HF' in l or 'Archive' in l][-6:]

print(f"STATUS:{len(pids)}|COUNT:{csv_count}|LINES:" + "///".join(last_lines))
"""

print("[Monitor] Starting monitoring of Marimo background extraction...")

while True:
    try:
        p = subprocess.Popen(["bash", exec_tool, "--url", url, "--token", marimo_tok],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(input=check_code, timeout=40)
        
        status_line = None
        for l in out.splitlines():
            if l.startswith("STATUS:"):
                status_line = l
                break
        
        if status_line:
            parts = status_line.split("|")
            active_pids = int(parts[0].split(":")[1])
            csv_count = int(parts[1].split(":")[1])
            recent_logs = parts[2].split("LINES:")[1].split("///") if "LINES:" in parts[2] else []
            
            print(f"[{time.strftime('%H:%M:%S')}] Active workers: {active_pids} | Extracted CSVs: {csv_count} / 1024", flush=True)
            for rl in recent_logs:
                if rl.strip():
                    print(f"   > {rl.strip()}", flush=True)
            
            if active_pids == 0:
                print("\n[Monitor] Background extraction process has completed!", flush=True)
                break
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Awaiting status response...", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Poll error: {e}", flush=True)
        
    time.sleep(25)

print("\n[Monitor] Fetching final server extraction log...")
fetch_log_code = """
from pathlib import Path
log_p = Path('/marimo/extract_landmarks.log')
if log_p.exists():
    print(log_p.read_text()[-2500:])
"""
p = subprocess.Popen(["bash", exec_tool, "--url", url, "--token", marimo_tok],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
out, err = p.communicate(input=fetch_log_code, timeout=40)
print(out)

# Now download to local data/landmarks
dest_dir = ROOT_DIR / "data" / "landmarks"
print(f"\n[Local] Pulling landmarks from Hugging Face into {dest_dir} ...")
pull_landmarks_from_hf(
    dest_dir=str(dest_dir),
    repo_id="Cuong2004/gym-exercise-landmarks",
    filename="landmarks_dataset.zip"
)

# Verification
csvs = list(dest_dir.glob("**/*.csv"))
print(f"\n[Local Verification] Successfully downloaded {len(csvs)} CSV files into {dest_dir}!")
for sp in ["train", "val", "test"]:
    sp_p = dest_dir / sp
    sp_count = len(list(sp_p.glob("**/*.csv"))) if sp_p.exists() else 0
    print(f"  - Split '{sp}': {sp_count} files")

# Send celebration toast to marimo
toast_code = f"""
import marimo as mo
mo.status.toast("🎉 MediaPipe Pose (model_complexity=2 Heavy) extraction & local sync 100% complete! Total: {len(csvs)} files.")
"""
subprocess.run(["bash", exec_tool, "--url", url, "--token", marimo_tok],
               input=toast_code, text=True, capture_output=True)
print("\n[Finished] All tasks completed successfully!")
