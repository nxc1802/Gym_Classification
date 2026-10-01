#!/usr/bin/env python3
"""
Remote Server Landmark Extractor & Keep-Alive Daemon for Marimo GPU Sandbox.
Executed directly inside /marimo/Gym_Classification on the remote server.
"""

import os
import sys
import time
import json
import zipfile
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HF_TOKEN = os.environ.get("HF_TOKEN", "")
VIDEO_REPO = os.environ.get("HF_VIDEO_REPO", "Cuong2004/mmfit-unseen-rgb")
LANDMARK_REPO = os.environ.get("HF_LANDMARK_REPO", "Cuong2004/gym-exercise-landmarks")

TARGET_WORKOUTS = ["w20", "w05", "w00", "w13", "w12"]  # Ordered by size ascending for fastest feedback

def send_marimo_toast(message: str, kind: str = "info"):
    """Sends visual keep-alive toast to Marimo UI if available."""
    try:
        import marimo as mo
        mo.status.toast(message, kind=kind)
    except Exception:
        pass

def keep_alive_worker(stop_event, progress_file: Path):
    """Background keep-alive thread: sends periodic heartbeats to prevent idle disconnect."""
    while not stop_event.is_set():
        try:
            # Update heartbeat timestamp
            if progress_file.exists():
                with open(progress_file, "r") as f:
                    data = json.load(f)
                data["last_heartbeat"] = time.time()
                data["heartbeat_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                with open(progress_file, "w") as f:
                    json.dump(data, f, indent=2)

            send_marimo_toast(f"💓 MM-Fit Extractor Keep-Alive: Running... [{time.strftime('%H:%M:%S')}]", kind="info")
        except Exception:
            pass
        stop_event.wait(60)  # Heartbeat every 60 seconds

def main():
    from huggingface_hub import HfApi, hf_hub_download
    from src.data.extractor import extract_landmarks_from_video

    work_dir = PROJECT_ROOT
    raw_dir = work_dir / "data_external" / "mmfit" / "raw" / "rgb"
    out_dir = work_dir / "data_external" / "mmfit" / "landmarks"
    progress_file = work_dir / "outputs" / "extraction_progress.json"

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("REMOTE SERVER MEDIAPIPE HEAVY EXTRACTION DAEMON")
    print(f"Target Workouts: {TARGET_WORKOUTS}")
    print(f"Video Source Repo: {VIDEO_REPO}")
    print(f"Landmark Dest Repo: {LANDMARK_REPO}")
    print("=" * 80)

    # Initialize progress record
    progress_data = {
        "status": "INITIALIZING",
        "start_time": time.time(),
        "start_time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "target_workouts": TARGET_WORKOUTS,
        "completed_workouts": [],
        "current_workout": None,
        "stats": {}
    }
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

    # Start background keep-alive thread
    stop_event = threading.Event()
    keep_thread = threading.Thread(target=keep_alive_worker, args=(stop_event, progress_file), daemon=True)
    keep_thread.start()

    api = HfApi(token=HF_TOKEN)

    for w in TARGET_WORKOUTS:
        progress_data["status"] = "IN_PROGRESS"
        progress_data["current_workout"] = w
        with open(progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

        v_name = f"{w}_rgb.mp4"
        v_local = raw_dir / v_name
        out_csv = out_dir / f"{w}_mediapipe.csv"

        # 1. Download video from HF if not present locally
        if not v_local.exists() or v_local.stat().st_size < 10 * 1024 * 1024:
            print(f"\n[Server] Waiting/Downloading {v_name} from {VIDEO_REPO}...")
            send_marimo_toast(f"📥 Downloading {v_name} from Hugging Face...", kind="info")
            downloaded = False
            for retry in range(60):  # Wait up to 30 mins if still uploading from local
                try:
                    download_path = hf_hub_download(
                        repo_id=VIDEO_REPO,
                        filename=f"rgb/{v_name}",
                        repo_type="dataset",
                        token=HF_TOKEN,
                        local_dir=str(raw_dir.parent)
                    )
                    if os.path.exists(download_path) and os.path.getsize(download_path) > 10 * 1024 * 1024:
                        v_local = Path(download_path)
                        downloaded = True
                        print(f"  Downloaded {v_name} ({v_local.stat().st_size / (1024*1024):.1f} MB)")
                        break
                except Exception as e:
                    print(f"  [Waiting {retry*30}s] {v_name} not yet ready on HF: {e}")
                    time.sleep(30)

            if not downloaded:
                print(f"[Error] Failed to acquire {v_name} on server. Skipping.")
                continue

        # 2. Extract Landmarks with MediaPipe Pose Heavy (complexity=2)
        if out_csv.exists() and out_csv.stat().st_size > 1000:
            print(f"[Server] Cached landmarks already exist for {w}: {out_csv.stat().st_size / 1024:.1f} KB")
            progress_data["completed_workouts"].append(w)
            continue

        print(f"\n[Server] Extracting MediaPipe Pose Heavy landmarks for {v_name}...")
        send_marimo_toast(f"🏃 Extracting MediaPipe Heavy on {v_name}...", kind="info")
        t_start = time.time()
        try:
            df_lm = extract_landmarks_from_video(
                video_path=str(v_local),
                output_csv_path=str(out_csv),
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            elapsed = time.time() - t_start
            n_frames = len(df_lm)
            fps = n_frames / elapsed if elapsed > 0 else 0
            print(f"  [Done] {w}: {n_frames} frames in {elapsed:.1f}s ({fps:.1f} fps). CSV: {out_csv.stat().st_size / 1024:.1f} KB")
            
            progress_data["completed_workouts"].append(w)
            progress_data["stats"][w] = {
                "frames": n_frames,
                "elapsed_s": round(elapsed, 1),
                "fps": round(fps, 1),
                "csv_size_kb": round(out_csv.stat().st_size / 1024, 1)
            }
            with open(progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)

            send_marimo_toast(f"✅ Extracted {w}: {n_frames} frames ({fps:.1f} fps)!", kind="success")
        except Exception as e:
            print(f"  [Error] Failed extracting {w}: {e}")
            progress_data["stats"][w] = {"status": "FAILED", "error": str(e)}

    # 3. Zip all extracted landmarks
    print("\n[Server] All extractions complete. Packaging landmark CSVs into ZIP...")
    zip_path = work_dir / "mmfit_mediapipe_landmarks.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_p in out_dir.glob("*.csv"):
            zf.write(csv_p, arcname=f"landmarks/{csv_p.name}")
    print(f"  ZIP created: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")

    # 4. Upload ZIP to Hugging Face
    print(f"\n[Server] Uploading landmark ZIP to {LANDMARK_REPO}...")
    send_marimo_toast("📦 Uploading final Landmark ZIP to Hugging Face...", kind="info")
    try:
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="external/mmfit_mediapipe_landmarks.zip",
            repo_id=LANDMARK_REPO,
            repo_type="dataset",
            commit_message="Upload extracted MM-Fit MediaPipe Heavy landmarks (unseen-test)"
        )
        print(f"  [Success] Landmark ZIP uploaded to https://huggingface.co/datasets/{LANDMARK_REPO}")
    except Exception as e:
        print(f"  [Warning] Failed uploading to {LANDMARK_REPO}: {e}")

    try:
        # Also upload to video repo for convenience
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo="landmarks/mmfit_mediapipe_landmarks.zip",
            repo_id=VIDEO_REPO,
            repo_type="dataset",
            commit_message="Upload extracted MM-Fit MediaPipe Heavy landmarks (unseen-test)"
        )
    except Exception:
        pass

    # Stop keep-alive and finalize
    stop_event.set()
    progress_data["status"] = "COMPLETED"
    progress_data["end_time"] = time.time()
    progress_data["total_elapsed_s"] = round(time.time() - progress_data["start_time"], 1)
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

    send_marimo_toast("🎉 MM-Fit Landmark Extraction 100% COMPLETE!", kind="success")
    print("\n========================================================================")
    print("EXTRACTION AND SYNC FULLY COMPLETED!")
    print(f"Total Elapsed: {progress_data['total_elapsed_s']}s")
    print("========================================================================")

if __name__ == "__main__":
    main()
