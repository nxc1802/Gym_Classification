#!/usr/bin/env python3
"""
Marimo Notebook / Standalone Pipeline:
Extract MediaPipe Pose Landmarks (model_complexity = 2 Heavy) directly from Kaggle Dataset.

Source of Truth: https://www.kaggle.com/datasets/nguyenxuancuongk18dn/gym-exercise-classification-dataset
NO SHUFFLE. NO SPLIT REORGANIZATION. ZERO RENAMING.
Preserves 100% original directory layout (train / val / test) and video filenames.

Can be run:
  1. As standalone CLI: python3 scripts/marimo_extract_landmarks_complexity2.py --workers 16
  2. Inside Marimo UI:  marimo edit scripts/marimo_extract_landmarks_complexity2.py
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.extractor import extract_landmarks_from_video

KAGGLE_DATASET_ID = "nguyenxuancuongk18dn/gym-exercise-classification-dataset"

def download_kaggle_dataset(dataset_id: str = KAGGLE_DATASET_ID) -> str:
    """Downloads dataset from Kaggle via kagglehub without modifying any files."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is required. Install via: pip install kagglehub")

    print(f"[1/4] Downloading / Verifying Kaggle Dataset: {dataset_id} ...")
    raw_path = kagglehub.dataset_download(dataset_id)
    print(f"      Source dataset ready at: {raw_path}")
    return raw_path

def collect_video_tasks(raw_base_dir: str, output_landmark_dir: str):
    """
    Scans the downloaded Kaggle dataset directory and prepares 1-to-1 extraction tasks.
    Preserves exact split, class, and filename.
    """
    raw_path = Path(raw_base_dir)
    out_path = Path(output_landmark_dir)

    video_extensions = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    all_videos = [
        f for f in raw_path.rglob("*")
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    tasks = []
    skipped = 0

    for vid in all_videos:
        rel = vid.relative_to(raw_path)
        parts = rel.parts

        # Handle nested wrappers if Kaggle unzips into a subfolder
        if len(parts) >= 3 and parts[0].lower() in {"train", "val", "test"}:
            split, act, filename = parts[0], parts[1], parts[-1]
        elif len(parts) >= 4 and parts[1].lower() in {"train", "val", "test"}:
            split, act, filename = parts[1], parts[2], parts[-1]
        elif len(parts) >= 4 and parts[0] == "Final_dataset" and parts[1].lower() in {"train", "val", "test"}:
            split, act, filename = parts[1], parts[2], parts[-1]
        else:
            # Fallback: scan for train/val/test anywhere in parts
            split_idx = -1
            for idx_p, p in enumerate(parts):
                if p.lower() in {"train", "val", "test"}:
                    split_idx = idx_p
                    break
            if split_idx != -1 and len(parts) > split_idx + 2:
                split = parts[split_idx]
                act = parts[split_idx + 1]
                filename = parts[-1]
            else:
                continue

        csv_name = f"{Path(filename).stem}.csv"
        target_csv = out_path / split / act / csv_name

        # Skip if already fully extracted (> 100 bytes)
        if target_csv.exists() and target_csv.stat().st_size > 100:
            skipped += 1
        else:
            tasks.append((str(vid), str(target_csv)))

    return all_videos, tasks, skipped

def _worker_extract(task_tuple):
    video_path, output_csv = task_tuple
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        extract_landmarks_from_video(
            video_path=video_path,
            output_csv_path=output_csv,
            model_complexity=2,  # Heavy model (SOTA research accuracy)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return True, Path(video_path).name, None
    except Exception as e:
        return False, Path(video_path).name, str(e)

def run_extraction_pipeline(
    raw_dir: str = None,
    output_dir: str = str(ROOT_DIR / "data" / "landmarks"),
    num_workers: int = 16
):
    """Executes the full pipeline with model_complexity = 2."""
    print("=" * 70)
    print(" MediaPipe Pose Landmark Extraction: model_complexity = 2 (Heavy)")
    print(f" Target Output: {output_dir}")
    print(f" Parallel Workers: {num_workers}")
    print("=" * 70)

    # 1. Resolve raw dataset
    if not raw_dir or not os.path.exists(raw_dir):
        raw_dir = download_kaggle_dataset(KAGGLE_DATASET_ID)

    # 2. Collect 1-to-1 tasks
    print("\n[2/4] Scanning video hierarchy ...")
    all_videos, tasks, skipped = collect_video_tasks(raw_dir, output_dir)
    print(f"      Total videos discovered : {len(all_videos)}")
    print(f"      Already extracted (skip): {skipped}")
    print(f"      Tasks to process        : {len(tasks)}")

    if not tasks:
        print("\nAll videos have already been extracted! Output directory is up-to-date.")
        _summarize_splits(output_dir)
        return

    # 3. Parallel extraction
    print(f"\n[3/4] Launching parallel extraction with model_complexity=2 ({num_workers} workers) ...")
    success_cnt = 0
    fail_cnt = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_worker_extract, t): t for t in tasks}
        total_tasks = len(tasks)
        for idx, fut in enumerate(as_completed(futures), 1):
            ok, name, err = fut.result()
            if ok:
                success_cnt += 1
            else:
                fail_cnt += 1
                print(f"  [FAIL] {name}: {err}")

            if idx % 25 == 0 or idx == total_tasks:
                print(f"  Progress: [{idx}/{total_tasks}] (Success: {success_cnt}, Failed: {fail_cnt})")

    # 4. Final summary
    print("\n[4/4] Extraction completed!")
    _summarize_splits(output_dir)

def _summarize_splits(output_dir: str):
    base = Path(output_dir)
    print("-" * 50)
    print("Landmark Dataset Summary:")
    for sp in ["train", "val", "test"]:
        p = base / sp
        if p.exists():
            csv_cnt = len(list(p.glob("**/*.csv")))
            class_cnt = len([d for d in p.iterdir() if d.is_dir()])
            print(f"  - {sp:6s}: {csv_cnt:4d} CSV files across {class_cnt:2d} classes")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Extract landmarks with model_complexity=2 directly from Kaggle dataset.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw videos (downloads from Kaggle if not provided)")
    parser.add_argument("--output_dir", type=str, default=str(ROOT_DIR / "data" / "landmarks"), help="Output directory for landmark CSVs")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker processes")
    args = parser.parse_args()

    run_extraction_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()
