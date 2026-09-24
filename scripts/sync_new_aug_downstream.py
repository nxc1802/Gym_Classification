#!/usr/bin/env python3
"""
Syncs New SkelGym-Aug downstream training and evaluation results from remote Marimo
to Hugging Face Hub (Cuong2004/gym-exercise-classification) and downloads to local workspace.
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import get_token, HfApi, snapshot_download

def main():
    token = get_token()
    if not token:
        print("Error: Local HF_TOKEN not found!")
        sys.exit(1)

    print("[1/3] Instructing remote Marimo server to upload downstream artifacts to Hugging Face...")
    
    remote_code = f"""
import os, glob
from huggingface_hub import HfApi

api = HfApi(token="{token}")
repo_id = "Cuong2004/gym-exercise-classification"

os.chdir('/marimo/Gym_Classification')

print("Uploading checkpoints to HF...")
# Root checkpoints (Seed 42)
for ckpt in glob.glob('checkpoints/*.pt'):
    api.upload_file(
        path_or_fileobj=ckpt,
        path_in_repo=f"checkpoints/{{os.path.basename(ckpt)}}",
        repo_id=repo_id,
        commit_message=f"Update checkpoint {{os.path.basename(ckpt)}} with new SkelGym-Aug"
    )

# Seed 123 checkpoints
for ckpt in glob.glob('checkpoints/seed123/*.pt'):
    api.upload_file(
        path_or_fileobj=ckpt,
        path_in_repo=f"checkpoints/seed123/{{os.path.basename(ckpt)}}",
        repo_id=repo_id,
        commit_message=f"Update seed123 checkpoint {{os.path.basename(ckpt)}} with new SkelGym-Aug"
    )

# Seed 3407 checkpoints
for ckpt in glob.glob('checkpoints/seed3407/*.pt'):
    api.upload_file(
        path_or_fileobj=ckpt,
        path_in_repo=f"checkpoints/seed3407/{{os.path.basename(ckpt)}}",
        repo_id=repo_id,
        commit_message=f"Update seed3407 checkpoint {{os.path.basename(ckpt)}} with new SkelGym-Aug"
    )

print("Uploading downstream logs to HF...")
if os.path.exists("outputs/downstream_logs"):
    api.upload_folder(
        folder_path="outputs/downstream_logs",
        path_in_repo="outputs/downstream_logs",
        repo_id=repo_id,
        commit_message="Upload downstream training logs with new SkelGym-Aug"
    )

print("Uploading results reports and confusion matrix to HF...")
for fname in ["new_aug_downstream_results.json", "new_aug_downstream_results.md", "downstream_progress.json"]:
    fpath = f"outputs/{{fname}}"
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"outputs/{{fname}}",
            repo_id=repo_id,
            commit_message=f"Upload {{fname}}"
        )

if os.path.exists("outputs/ensemble"):
    api.upload_folder(
        folder_path="outputs/ensemble",
        path_in_repo="outputs/ensemble",
        repo_id=repo_id,
        commit_message="Upload final ensemble predictions and confusion matrices"
    )

print("Remote upload to HF completed!")
"""

    res = subprocess.run([
        "/Users/nxc/.local/bin/execute-code.sh",
        "--url", "https://sb-9a72f8ecf300bb7e.sb.molab.run/",
        "--token", "8c9c51c9d3c1407a9261d721764d9a2a11e09a782b6d3d4f94027a1535e6ebe6",
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
            "checkpoints/best_*.pt",
            "checkpoints/seed123/best_*.pt",
            "checkpoints/seed3407/best_*.pt",
            "outputs/downstream_logs/*",
            "outputs/new_aug_downstream_results.*",
            "outputs/ensemble/*"
        ],
        local_dir="/Volumes/WorkSpace/Project/Gym_Classification",
        token=token
    )
    print("Local download completed!")

    print("[3/3] Sync complete! All artifacts safely synced to HF and local.")

if __name__ == "__main__":
    main()
