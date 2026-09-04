import os
import sys
import glob
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure workspace root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.extractor import extract_landmarks_from_video

def worker(task):
    video_path, csv_path = task
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            extract_landmarks_from_video(video_path, csv_path)
        return True, os.path.basename(video_path), None
    except Exception as e:
        return False, os.path.basename(video_path), str(e)

def main():
    parser = argparse.ArgumentParser(description="High-throughput parallel MediaPipe landmarks extractor.")
    parser.add_argument("--raw_dir", type=str, default=str(ROOT_DIR / "data" / "raw"), help="Directory containing raw videos.")
    parser.add_argument("--landmark_dir", type=str, default=str(ROOT_DIR / "data" / "landmarks"), help="Output directory for landmark CSVs.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel worker processes.")
    args = parser.parse_args()

    raw_files = [f for f in glob.glob(os.path.join(args.raw_dir, "**", "*"), recursive=True) if os.path.isfile(f) and f.lower().endswith(('.mp4', '.avi', '.mov'))]

    tasks = []
    for rf in raw_files:
        rel = os.path.relpath(rf, args.raw_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            split, act, filename = parts[0], parts[1], parts[2]
            stem = os.path.splitext(filename)[0]
            csv_path = os.path.join(args.landmark_dir, split, act, f"{stem}.csv")
            if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
                tasks.append((rf, csv_path))

    print(f"Found {len(raw_files)} total raw videos. Need extraction for {len(tasks)} videos using {args.workers} workers...")

    if tasks:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker, t): t[0] for t in tasks}
            for f in as_completed(futs):
                ok, name, err = f.result()
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"Progress: [{done}/{len(tasks)}] finished.")
                if not ok:
                    print(f"Error extracting {name}: {err}")

    csvs = glob.glob(os.path.join(args.landmark_dir, "**", "*.csv"), recursive=True)
    print(f"Extraction completed! Total landmark CSVs now on disk: {len(csvs)}")

if __name__ == "__main__":
    main()
