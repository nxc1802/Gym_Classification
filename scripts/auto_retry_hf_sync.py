#!/usr/bin/env python3
"""
Background auto-retry synchronizer for Hugging Face Hub.
Polls commit rate limit and commits all pending reports atomically as soon as rate limit opens.
"""

import os
import time
import datetime
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        # Check huggingface CLI cached token
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    
    if not token:
        print("[Error] No Hugging Face token provided.")
        sys.exit(1)

    print(f"[{datetime.datetime.now()}] HF Auto-Sync Daemon started. Waiting for commit rate limit window to slide...")
    
    max_wait = 1800  # 30 mins max
    start_t = time.time()
    
    while time.time() - start_t < max_wait:
        print(f"[{datetime.datetime.now()}] Attempting atomic batch sync...")
        cmd = [sys.executable, str(ROOT_DIR / "scripts" / "sync_to_hf_hub.py"), "--token", token]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"[{datetime.datetime.now()}] SUCCESS! All reports and outputs synced to Hugging Face Hub!")
            print(res.stdout)
            sys.exit(0)
        else:
            if "429" in res.stderr or "rate limit" in res.stderr:
                print(f"[{datetime.datetime.now()}] Rate limit still active. Sleeping 120s...")
                time.sleep(120)
            else:
                print(f"[{datetime.datetime.now()}] Error: {res.stderr}")
                time.sleep(60)

if __name__ == "__main__":
    main()
