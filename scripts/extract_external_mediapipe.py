#!/usr/bin/env python3
"""
High-Throughput MediaPipe Pose Heavy Landmark Extraction for External Datasets (MM-Fit & Fit3D).
Strictly adheres to external_test.md Phase E protocol:
  - Model: MediaPipe Pose Heavy (model_complexity = 2)
  - Mode: Continuous video stream (static_image_mode = False)
  - Coordinates: result.pose_landmarks (normalized image coordinates x, y in [0, 1], relative z, visibility)
  - Target format: 133-column CSV matching SkelGym format (Frame, NOSE_x, NOSE_y, NOSE_z, NOSE_visibility, ...)
  - Whole-workout extraction in a single continuous pass to preserve temporal tracking continuity.
"""

import os
import sys
import time
import argparse
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RAW_POINTS_33
from src.data.extractor import extract_landmarks_from_video

# Official Zenodo download URLs for MM-Fit Unseen-Test RGB videos
MMFIT_ZENODO_RECORD_ID = "7672767"
MMFIT_ZENODO_BASE_URL = f"https://zenodo.org/records/{MMFIT_ZENODO_RECORD_ID}/files"

ZENODO_VIDEO_URLS = {
    # Unseen Test (Primary External Benchmark)
    "w00": f"{MMFIT_ZENODO_BASE_URL}/w00_rgb.mp4",
    "w05": f"{MMFIT_ZENODO_BASE_URL}/w05_rgb.mp4",
    "w12": f"{MMFIT_ZENODO_BASE_URL}/w12_rgb.mp4",
    "w13": f"{MMFIT_ZENODO_BASE_URL}/w13_rgb.mp4",
    "w20": f"{MMFIT_ZENODO_BASE_URL}/w20_rgb.mp4",
    # Seen Test (Secondary Benchmark)
    "w09": f"{MMFIT_ZENODO_BASE_URL}/w09_rgb.mp4",
    "w10": f"{MMFIT_ZENODO_BASE_URL}/w10_rgb.mp4",
    "w11": f"{MMFIT_ZENODO_BASE_URL}/w11_rgb.mp4",
}

def download_video_with_progress(url: str, dest_path: Path) -> bool:
    """Downloads a video file with terminal progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    
    print(f"\n[Download] Fetching: {dest_path.name}")
    print(f"           Source: {url}")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f_out:
            total_size = int(resp.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 MB
            downloaded = 0
            start_time = time.time()
            
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f_out.write(chunk)
                downloaded += len(chunk)
                elapsed = max(1e-4, time.time() - start_time)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s", end="", flush=True)
                else:
                    print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB | Speed: {speed_mb:.2f} MB/s", end="", flush=True)
            
            print()
            
        temp_path.rename(dest_path)
        print(f"[Done] Saved: {dest_path}")
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\n[Error] Failed to download {url}: {e}")
        return False

def _extract_worker(task: Tuple[str, str, int, float, float]) -> Tuple[bool, str, str]:
    video_path, out_csv, complexity, det_conf, track_conf = task
    v_name = Path(video_path).name
    try:
        t0 = time.time()
        extract_landmarks_from_video(
            video_path=video_path,
            output_csv_path=out_csv,
            model_complexity=complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf
        )
        elapsed = time.time() - t0
        return True, v_name, f"Finished in {elapsed:.1f}s"
    except Exception as e:
        return False, v_name, str(e)

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose Heavy Extractor for External Video Benchmarks")
    parser.add_argument("--workouts", nargs="+", default=["w00", "w05", "w12", "w13", "w20"],
                        help="Workout IDs to process (default: unseen-test: w00 w05 w12 w13 w20)")
    parser.add_argument("--raw-dir", type=str, default="data_external/mmfit/raw/rgb",
                        help="Directory where raw RGB videos reside or will be downloaded")
    parser.add_argument("--out-dir", type=str, default="data_external/mmfit/landmarks",
                        help="Directory to save extracted MediaPipe landmark CSVs")
    parser.add_argument("--download-missing", action="store_true", default=True,
                        help="Automatically download missing videos from Zenodo (default: True)")
    parser.add_argument("--download-only", action="store_true", default=False,
                        help="Only download the videos without running extraction")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes for video extraction")
    parser.add_argument("--complexity", type=int, default=2,
                        help="MediaPipe model complexity: 2 (Heavy, required for primary protocol)")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / args.raw_dir
    out_dir = PROJECT_ROOT / args.out_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEDIAPIPE POSE HEAVY EXTERNAL LANDMARK EXTRACTOR")
    print(f"Target Workouts: {args.workouts}")
    print(f"Raw Videos Dir: {raw_dir}")
    print(f"Landmarks Dir:  {out_dir}")
    print(f"Model Complexity: {args.complexity} (Heavy) | Workers: {args.workers}")
    print("=" * 80)

    # 1. Check or Download Videos
    tasks_to_extract = []
    for w in args.workouts:
        v_filename = f"{w}_rgb.mp4"
        v_path = raw_dir / v_filename
        out_csv = out_dir / f"{w}_mediapipe.csv"

        # Check alternate video paths if already downloaded elsewhere
        alt_paths = [
            v_path,
            PROJECT_ROOT / "mm-fit" / "raw" / "rgb" / v_filename,
            PROJECT_ROOT / "mm-fit" / w / v_filename,
            PROJECT_ROOT / "data_external" / "mmfit" / "raw" / "rgb" / v_filename
        ]
        found_video = None
        for ap in alt_paths:
            if ap.exists() and ap.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                found_video = ap
                break

        if found_video is None:
            if args.download_missing:
                if w in ZENODO_VIDEO_URLS:
                    url = ZENODO_VIDEO_URLS[w]
                    success = download_video_with_progress(url, v_path)
                    if success:
                        found_video = v_path
                    else:
                        print(f"[Warning] Failed downloading {w}. Skipping.")
                        continue
                else:
                    print(f"[Warning] No Zenodo URL configured for {w}. Skipping.")
                    continue
            else:
                print(f"[Skip] Video not found for {w}: {v_path}")
                continue

        if args.download_only:
            print(f"[Downloaded] {w} video ready at {found_video}")
            continue

        # Check if already extracted
        if out_csv.exists() and out_csv.stat().st_size > 1000:
            print(f"[Cached] Landmark CSV already exists for {w} ({out_csv.stat().st_size / 1024:.1f} KB). Skipping.")
            continue

        tasks_to_extract.append((str(found_video), str(out_csv), args.complexity, 0.5, 0.5))

    if args.download_only:
        print("\nDownload-only mode completed successfully.")
        return

    # 2. Parallel Landmark Extraction
    if not tasks_to_extract:
        print("\nAll target landmark CSVs already exist and are valid. Nothing to extract.")
        return

    print(f"\nLaunching landmark extraction for {len(tasks_to_extract)} videos using {args.workers} workers...")
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(_extract_worker, t): t[0] for t in tasks_to_extract}
        for future in as_completed(future_to_task):
            v_src = future_to_task[future]
            ok, v_name, msg = future.result()
            if ok:
                print(f"  [DONE] {v_name}: {msg}")
            else:
                print(f"  [FAIL] {v_name}: {msg}")

    total_time = time.time() - t_start
    print(f"\nExtraction complete in {total_time:.1f}s.")
    print(f"Landmarks stored in: {out_dir}")

if __name__ == "__main__":
    main()
