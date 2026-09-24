#!/usr/bin/env python3
"""
Syncs augmentation ablation results between remote Marimo and Hugging Face / Local.
1. Parses all logs and builds final augmentation_ablation_results.json and .md.
2. Instructs remote Marimo to upload all new checkpoints and logs to Hugging Face Cuong2004/gym-exercise-classification.
3. Downloads the synced artifacts to local outputs/ and checkpoints/.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from huggingface_hub import get_token, HfApi, snapshot_download

def main():
    token = get_token()
    if not token:
        print("Error: Local HF_TOKEN not found!")
        sys.exit(1)

    print("[1/3] Instructing remote Marimo server to upload artifacts to Hugging Face...")
    
    remote_code = f"""
import os, glob, json, subprocess
from huggingface_hub import HfApi

api = HfApi(token="{token}")
repo_id = "Cuong2004/gym-exercise-classification"

print("Uploading checkpoints/ablation_aug to HF...")
if os.path.exists("/marimo/Gym_Classification/checkpoints/ablation_aug"):
    api.upload_folder(
        folder_path="/marimo/Gym_Classification/checkpoints/ablation_aug",
        path_in_repo="checkpoints/ablation_aug",
        repo_id=repo_id,
        commit_message="Upload systematic augmentation ablation checkpoints"
    )
    print("Checkpoints uploaded successfully!")

print("Uploading outputs/ablation_logs to HF...")
if os.path.exists("/marimo/Gym_Classification/outputs/ablation_logs"):
    api.upload_folder(
        folder_path="/marimo/Gym_Classification/outputs/ablation_logs",
        path_in_repo="outputs/ablation_logs",
        repo_id=repo_id,
        commit_message="Upload systematic augmentation ablation training logs"
    )
    print("Logs uploaded successfully!")

print("Uploading ablation results summary to HF...")
for fname in ["augmentation_ablation_results.json", "augmentation_ablation_results.md", "ablation_progress.json"]:
    fpath = f"/marimo/Gym_Classification/outputs/{{fname}}"
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"outputs/{{fname}}",
            repo_id=repo_id,
            commit_message=f"Upload {{fname}}"
        )
print("Remote upload to HF completed!")
"""

    res = subprocess.run([
        "/Users/nxc/.local/bin/execute-code.sh",
        "--url", "https://sb-6a8e54c88d08b719.sb.molab.run/",
        "--token", "55a0fb1c37f0c53b8239d49f1b363027268d0e1a13dacd5503cf18ca974ebab3",
        "--code", remote_code,
        "--timeout", "1800"
    ], capture_output=True, text=True)

    print("Remote execution output:\n", res.stdout)
    if res.returncode != 0:
        print("Remote error:\n", res.stderr)
        sys.exit(1)

    print("[2/3] Downloading new artifacts from Hugging Face to local workspace...")
    snapshot_download(
        repo_id="Cuong2004/gym-exercise-classification",
        allow_patterns=[
            "checkpoints/ablation_aug/*",
            "outputs/ablation_logs/*",
            "outputs/augmentation_ablation_results.*"
        ],
        local_dir="/Volumes/WorkSpace/Project/Gym_Classification",
        token=token
    )
    print("Local download completed!")

    print("[3/3] Sync complete! All artifacts safely synced to HF and local.")

if __name__ == "__main__":
    main()
