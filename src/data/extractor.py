"""
MediaPipe Pose Landmark Extractor.
Extracts 33 3D landmarks (x, y, z, visibility) per frame from raw gym exercise videos.
Supports both modern MediaPipe Tasks (mediapipe >= 0.10.x) and legacy mp.solutions.pose.
"""

import os
import urllib.request
from pathlib import Path
from typing import Optional, List
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.constants import RAW_POINTS_33

MODEL_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_LOCAL_PATH = Path(__file__).resolve().parent.parent.parent / "models_cache" / "pose_landmarker_full.task"

def _ensure_task_model() -> str:
    """
    Ensures that the pose_landmarker model bundle exists locally.
    """
    if not MODEL_LOCAL_PATH.exists():
        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose landmarker model to {MODEL_LOCAL_PATH} ...")
        urllib.request.urlretrieve(MODEL_TASK_URL, str(MODEL_LOCAL_PATH))
        print("Model downloaded successfully.")
    return str(MODEL_LOCAL_PATH)

def extract_landmarks_from_video(
    video_path: str,
    output_csv_path: Optional[str] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5
) -> pd.DataFrame:
    """
    Extracts 33 pose landmarks for each frame of a video using MediaPipe Pose.
    Returns a DataFrame with columns: ['Frame', '{LANDMARK}_x', '{LANDMARK}_y', '{LANDMARK}_z', '{LANDMARK}_visibility']
    Total columns = 1 + 33*4 = 133 columns.
    """
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    records = []

    # Check which MediaPipe API is available
    use_tasks_api = False
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        use_tasks_api = True
    except (ImportError, AttributeError):
        use_tasks_api = False

    if use_tasks_api and not hasattr(mp, "solutions"):
        # Modern Tasks API
        model_path = _ensure_task_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(frame_idx * 1000.0 / fps)
            result = detector.detect_for_video(mp_image, ts_ms)

            row = [frame_idx]
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                first_person = result.pose_landmarks[0]
                for lm in first_person:
                    vis = getattr(lm, "visibility", 1.0)
                    vis = 1.0 if vis is None else float(vis)
                    row.extend([float(lm.x), float(lm.y), float(lm.z), vis])
            else:
                row.extend([0.0] * (len(RAW_POINTS_33) * 4))

            records.append(row)

        detector.close()

    else:
        # Legacy Solutions API
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            row = [frame_idx]
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    row.extend([float(lm.x), float(lm.y), float(lm.z), float(lm.visibility)])
            else:
                row.extend([0.0] * (len(RAW_POINTS_33) * 4))

            records.append(row)

        pose.close()

    cap.release()

    # Column names
    col_names = ["Frame"]
    for pt in RAW_POINTS_33:
        col_names.extend([f"{pt}_x", f"{pt}_y", f"{pt}_z", f"{pt}_visibility"])

    df = pd.DataFrame(records, columns=col_names)

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df

def batch_extract_landmarks(
    video_paths: List[str],
    output_dir: str,
    num_workers: int = 4
) -> None:
    """
    Parallel batch extraction of pose landmarks from a list of video files.
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for vp in video_paths:
            vpath = Path(vp)
            out_file = out_dir_path / f"{vpath.stem}.csv"
            f = executor.submit(extract_landmarks_from_video, str(vpath), str(out_file))
            futures[f] = vpath.name

        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                print(f"[Done] Extracted landmarks for {name}")
            except Exception as e:
                print(f"[Error] Failed extracting {name}: {e}")
