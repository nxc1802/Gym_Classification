#!/usr/bin/env python3
"""
Hugging Face Synchronization Tool for MM-Fit External Benchmark.
Handles:
  1. Uploading local MM-Fit RGB videos (10 GB) to Hugging Face Hub dataset repository.
  2. Generating the remote Marimo server execution payload.
  3. Pulling extracted MediaPipe landmarks ZIP from Hugging Face back to local storage.
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_hub import get_hf_token, ensure_hf_repo

DEFAULT_MMFIT_VIDEO_REPO = "Cuong2004/mmfit-unseen-rgb"
DEFAULT_LANDMARK_REPO = "Cuong2004/gym-exercise-landmarks"

def upload_videos_to_hf(video_dir: Path, repo_id: str, token: str):
    """Uploads the 5 RGB videos to a Hugging Face Dataset repository."""
    api = HfApi(token=token)
    ensure_hf_repo(repo_id, repo_type="dataset", token=token, private=True)

    videos = list(video_dir.glob("w*_rgb.mp4"))
    if not videos:
        print(f"[Error] No videos found in {video_dir}")
        return False

    print(f"\n[HF Upload] Found {len(videos)} videos to upload to {repo_id}:")
    for v in videos:
        print(f"  - {v.name} ({v.stat().st_size / (1024*1024):.1f} MB)")

    for v in videos:
        print(f"\nUploading {v.name} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=str(v),
            path_in_repo=f"rgb/{v.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload MM-Fit RGB video {v.name}"
        )
        print(f"  [Done] {v.name} uploaded.")

    print(f"\n[Success] All {len(videos)} videos uploaded to https://huggingface.co/datasets/{repo_id}")
    return True

def generate_remote_extraction_code(video_repo_id: str, landmark_repo_id: str, hf_token: str) -> str:
    """Generates the Python code string to be executed on the remote Marimo GPU server."""
    code = f"""
import os, sys, glob, zipfile, time
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

TOKEN = "{hf_token}"
VIDEO_REPO = "{video_repo_id}"
LANDMARK_REPO = "{landmark_repo_id}"

print("[Remote Server] 1/4: Downloading MM-Fit RGB videos from Hugging Face...")
work_dir = Path("/marimo/Gym_Classification")
if not work_dir.exists():
    work_dir = Path("./Gym_Classification")
work_dir.mkdir(parents=True, exist_ok=True)
os.chdir(str(work_dir))

raw_dir = work_dir / "data_external" / "mmfit" / "raw" / "rgb"
landmarks_dir = work_dir / "data_external" / "mmfit" / "landmarks"
raw_dir.mkdir(parents=True, exist_ok=True)
landmarks_dir.mkdir(parents=True, exist_ok=True)

# Fast snapshot download of videos from HF Hub
snapshot_download(
    repo_id=VIDEO_REPO,
    repo_type="dataset",
    local_dir=str(raw_dir.parent),
    token=TOKEN
)
print("[Remote Server] Videos ready at:", raw_dir)

# 2. Run High-Throughput MediaPipe Extraction
print("[Remote Server] 2/4: Running MediaPipe Pose Heavy extraction...")
cmd = f"python3 scripts/extract_external_mediapipe.py --raw-dir data_external/mmfit/raw/rgb --out-dir data_external/mmfit/landmarks --workers 8 --complexity 2"
ret = os.system(cmd)
print(f"[Remote Server] Extraction finished with exit code {{ret}}.")

# 3. Zip Landmarks
print("[Remote Server] 3/4: Compressing landmark CSVs into ZIP...")
zip_path = work_dir / "mmfit_mediapipe_landmarks.zip"
with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
    for csv_file in landmarks_dir.glob("*.csv"):
        zf.write(csv_file, arcname=f"landmarks/{{csv_file.name}}")
print(f"[Remote Server] Landmark ZIP created: {{zip_path}} ({{zip_path.stat().st_size / 1024:.1f}} KB)")

# 4. Upload Landmark ZIP back to HF
print("[Remote Server] 4/4: Uploading landmark ZIP to Hugging Face...")
api = HfApi(token=TOKEN)
api.upload_file(
    path_or_fileobj=str(zip_path),
    path_in_repo="external/mmfit_mediapipe_landmarks.zip",
    repo_id=LANDMARK_REPO,
    repo_type="dataset",
    commit_message="Upload extracted MM-Fit MediaPipe Heavy landmarks"
)
print("[Remote Server] All tasks completed successfully!")
"""
    return code

def download_landmarks_from_hf(repo_id: str, dest_dir: Path, token: str):
    """Downloads extracted landmark ZIP from Hugging Face and unzips locally."""
    print(f"\n[HF Download] Fetching landmark ZIP from https://huggingface.co/datasets/{repo_id}...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename="external/mmfit_mediapipe_landmarks.zip",
        repo_type="dataset",
        token=token
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir.parent)
    print(f"[Success] Extracted landmarks to {dest_dir}:")
    for f in dest_dir.glob("*.csv"):
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

def main():
    parser = argparse.ArgumentParser(description="MM-Fit Hugging Face Sync Pipeline")
    parser.add_argument("--action", choices=["upload_videos", "download_landmarks", "gen_remote_code"], required=True)
    parser.add_argument("--token", type=str, default=None, help="HF Access Token")
    parser.add_argument("--video-repo", type=str, default=DEFAULT_MMFIT_VIDEO_REPO, help="HF dataset repo for RGB videos")
    parser.add_argument("--landmark-repo", type=str, default=DEFAULT_LANDMARK_REPO, help="HF dataset repo for landmark results")
    args = parser.parse_args()

    tok = get_hf_token(args.token)
    if not tok and args.action != "gen_remote_code":
        print("[Error] Hugging Face token is required. Pass --token <HF_TOKEN> or set HF_TOKEN environment variable.")
        sys.exit(1)

    video_dir = PROJECT_ROOT / "data_external" / "mmfit" / "raw" / "rgb"
    landmark_dir = PROJECT_ROOT / "data_external" / "mmfit" / "landmarks"

    if args.action == "upload_videos":
        upload_videos_to_hf(video_dir, args.video_repo, tok)
    elif args.action == "download_landmarks":
        download_landmarks_from_hf(args.landmark_repo, landmark_dir, tok)
    elif args.action == "gen_remote_code":
        code = generate_remote_extraction_code(args.video_repo, args.landmark_repo, tok or "YOUR_HF_TOKEN")
        print(code)

if __name__ == "__main__":
    main()
