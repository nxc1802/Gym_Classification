#!/usr/bin/env python3
"""
MM-Fit Data Audit, Alignment Verification, and Metadata Builder.
Strictly implements external_test.md Phase B, Phase D, and Phase F:
  - Audits raw video metadata (FPS, duration, total frames)
  - Audits frame alignment between official labels and MediaPipe landmark CSVs
  - Quality Control Gate on landmarks: detection rate, zero ratio (<20% rule), mean visibility
  - Builds master table: mmfit_external_metadata.csv documenting all Core-4 and Extended-5 segments.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RAW_POINTS_33, RAW_POINTS_13
from src.external.class_mapping import get_class_mapping

MMFIT_UNSEEN_TEST = ["w00", "w05", "w12", "w13", "w20"]
MMFIT_SEEN_TEST = ["w09", "w10", "w11"]

def audit_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extracts duration, FPS, and frame count from video using OpenCV."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"exists": False, "frames": 0, "fps": 0.0, "duration_s": 0.0, "width": 0, "height": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = frames / fps if fps > 0 else 0.0
    cap.release()
    return {
        "exists": True,
        "frames": frames,
        "fps": round(fps, 2),
        "duration_s": round(dur, 2),
        "width": w,
        "height": h
    }

def audit_landmark_csv(csv_path: Path) -> Dict[str, Any]:
    """Audits extracted MediaPipe landmark CSV file for quality metrics."""
    if not csv_path.exists():
        return {"exists": False, "total_frames": 0, "zero_ratio": 1.0, "mean_vis": 0.0}
    
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    if total_frames == 0:
        return {"exists": True, "total_frames": 0, "zero_ratio": 1.0, "mean_vis": 0.0}
    
    # Check coordinate zero ratio across 13 canonical joints
    coord_cols = [f"{pt}_{d}" for pt in RAW_POINTS_13 for d in ["x", "y", "z"]]
    valid_coord_cols = [c for c in coord_cols if c in df.columns]
    
    if valid_coord_cols:
        zeros_mask = (df[valid_coord_cols].abs() < 1e-6) | df[valid_coord_cols].isna()
        zero_frames = zeros_mask.all(axis=1).sum()
        zero_ratio = float(zero_frames) / total_frames
    else:
        zero_ratio = 1.0
        
    vis_cols = [f"{pt}_visibility" for pt in RAW_POINTS_13 if f"{pt}_visibility" in df.columns]
    mean_vis = float(df[vis_cols].mean().mean()) if vis_cols else 1.0

    return {
        "exists": True,
        "total_frames": total_frames,
        "zero_ratio": round(zero_ratio, 4),
        "mean_vis": round(mean_vis, 4)
    }

def main():
    parser = argparse.ArgumentParser(description="MM-Fit Data Audit and Master Metadata Builder")
    parser.add_argument("--mmfit-dir", type=str, default="mm-fit", help="Path to official MM-Fit dataset dir")
    parser.add_argument("--rgb-dir", type=str, default="data_external/mmfit/raw/rgb", help="Path to RGB videos")
    parser.add_argument("--landmarks-dir", type=str, default="data_external/mmfit/landmarks", help="Path to landmark CSVs")
    parser.add_argument("--out-dir", type=str, default="outputs/external/mmfit", help="Destination output directory")
    args = parser.parse_args()

    mmfit_dir = PROJECT_ROOT / args.mmfit_dir
    rgb_dir = PROJECT_ROOT / args.rgb_dir
    landmarks_dir = PROJECT_ROOT / args.landmarks_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    core4_map = get_class_mapping("mmfit", "core4")
    ext5_map = get_class_mapping("mmfit", "extended5")

    target_workouts = MMFIT_UNSEEN_TEST + MMFIT_SEEN_TEST

    print("=" * 80)
    print("MM-FIT PROTOCOL AUDIT AND MASTER METADATA GENERATOR")
    print(f"MM-Fit Root:  {mmfit_dir}")
    print(f"RGB Videos:   {rgb_dir}")
    print(f"Landmarks:    {landmarks_dir}")
    print("=" * 80)

    workout_audit_records = []
    segment_metadata_records = []
    landmark_segment_qc_records = []

    for w in target_workouts:
        split = "unseen_test" if w in MMFIT_UNSEEN_TEST else "seen_test"
        w_dir = mmfit_dir / w
        labels_csv = w_dir / f"{w}_labels.csv"
        pose_3d_file = w_dir / f"{w}_pose_3d.npy"
        v_file = rgb_dir / f"{w}_rgb.mp4"
        if not v_file.exists():
            # Check fallback locations
            alt_v = mmfit_dir / w / f"{w}_rgb.mp4"
            if alt_v.exists():
                v_file = alt_v

        lm_file = landmarks_dir / f"{w}_mediapipe.csv"
        if not lm_file.exists():
            alt_lm = mmfit_dir / w / f"{w}_mediapipe.csv"
            if alt_lm.exists():
                lm_file = alt_lm

        # 1. Audit Video
        v_meta = audit_video_metadata(v_file) if v_file.exists() else {"exists": False, "frames": 0, "fps": 30.0, "duration_s": 0.0}
        
        # 2. Audit Labels
        num_labels = 0
        first_lbl_frame = -1
        last_lbl_frame = -1
        df_labels = None
        if labels_csv.exists():
            try:
                df_labels = pd.read_csv(labels_csv, header=None)
                num_labels = len(df_labels)
                first_lbl_frame = int(df_labels.iloc[0, 0])
                last_lbl_frame = int(df_labels.iloc[-1, 1])
            except Exception:
                pass

        # 3. Audit Native Pose 3D
        pose3d_frames = 0
        if pose_3d_file.exists():
            try:
                p3d_shape = np.load(pose_3d_file, mmap_mode="r").shape
                pose3d_frames = p3d_shape[1] if len(p3d_shape) == 3 else p3d_shape[0]
            except Exception:
                pass

        # 4. Audit MediaPipe Landmark CSV
        lm_meta = audit_landmark_csv(lm_file)

        workout_audit_records.append({
            "workout_id": w,
            "split": split,
            "rgb_video_exists": v_meta["exists"],
            "rgb_frames": v_meta["frames"],
            "rgb_fps": v_meta["fps"],
            "rgb_duration_s": v_meta["duration_s"],
            "native_pose3d_frames": pose3d_frames,
            "mediapipe_csv_exists": lm_meta["exists"],
            "mediapipe_frames": lm_meta["total_frames"],
            "mediapipe_zero_ratio": lm_meta["zero_ratio"],
            "label_count": num_labels,
            "first_label_frame": first_lbl_frame,
            "last_label_frame": last_lbl_frame,
            "frame_alignment_status": "ALIGNED" if (lm_meta["total_frames"] >= last_lbl_frame or pose3d_frames >= last_lbl_frame) else "CHECK_BOUNDS"
        })

        # 5. Parse Segments for Master Table
        if df_labels is not None:
            # Load landmark dataframe once if available for segment QC
            df_lm = pd.read_csv(lm_file) if lm_file.exists() else None

            for seg_idx, row in df_labels.iterrows():
                try:
                    s_frame = int(row[0])
                    e_frame = int(row[1])
                    reps = int(row[2])
                    raw_act = str(row[3]).strip()
                except Exception:
                    continue

                is_core4 = raw_act in core4_map
                is_ext5 = raw_act in ext5_map
                if not (is_core4 or is_ext5):
                    continue

                target_act = core4_map[raw_act] if is_core4 else ext5_map[raw_act]
                mapping_type = "exact" if raw_act in ("squats", "pushups") else ("variant" if is_core4 else "supplementary")
                seg_id = f"{w}_set_{seg_idx+1:03d}_{raw_act}"
                num_frames = max(0, e_frame - s_frame)

                # Segment QC Gate
                seg_zero_ratio = 0.0
                seg_mean_vis = 1.0
                seg_qc_status = "PENDING_LANDMARKS"

                if df_lm is not None:
                    max_f = len(df_lm)
                    s_clamped = max(0, min(s_frame, max_f - 1))
                    e_clamped = min(max_f, max(s_clamped + 1, e_frame))
                    df_sub = df_lm.iloc[s_clamped:e_clamped]

                    coord_cols = [f"{pt}_{d}" for pt in RAW_POINTS_13 for d in ["x", "y", "z"]]
                    valid_c = [c for c in coord_cols if c in df_sub.columns]
                    if valid_c and len(df_sub) > 0:
                        z_mask = (df_sub[valid_c].abs() < 1e-6) | df_sub[valid_c].isna()
                        seg_zero_ratio = float(z_mask.all(axis=1).mean())
                    vis_cols = [f"{pt}_visibility" for pt in RAW_POINTS_13 if f"{pt}_visibility" in df_sub.columns]
                    if vis_cols and len(df_sub) > 0:
                        seg_mean_vis = float(df_sub[vis_cols].mean().mean())

                    if seg_zero_ratio > 0.20:
                        seg_qc_status = "REJECTED_HIGH_ZERO_RATIO"
                    elif num_frames < 16:
                        seg_qc_status = "REJECTED_TOO_SHORT"
                    else:
                        seg_qc_status = "ACCEPTED"

                segment_metadata_records.append({
                    "segment_id": seg_id,
                    "workout_id": w,
                    "split": split,
                    "source_class": raw_act,
                    "target_class": target_act,
                    "start_frame": s_frame,
                    "end_frame": e_frame,
                    "reps": reps,
                    "num_frames": num_frames,
                    "rgb_path": str(v_file.relative_to(PROJECT_ROOT)) if v_file.exists() else "N/A",
                    "landmark_path": str(lm_file.relative_to(PROJECT_ROOT)) if lm_file.exists() else "N/A",
                    "mapping_type": mapping_type,
                    "is_core4": is_core4,
                    "is_extended5": is_ext5,
                    "qc_status": seg_qc_status,
                    "zero_ratio": round(seg_zero_ratio, 4),
                    "mean_visibility": round(seg_mean_vis, 4)
                })

    df_workout_audit = pd.DataFrame(workout_audit_records)
    workout_audit_csv = out_dir / "data_audit.csv"
    df_workout_audit.to_csv(workout_audit_csv, index=False)
    print(f"\n[1/2] Workout audit table saved to: {workout_audit_csv}")
    print(df_workout_audit[["workout_id", "split", "rgb_video_exists", "mediapipe_csv_exists", "label_count", "frame_alignment_status"]].to_string(index=False))

    df_seg_meta = pd.DataFrame(segment_metadata_records)
    seg_meta_csv = out_dir / "segment_metadata.csv"
    df_seg_meta.to_csv(seg_meta_csv, index=False)
    print(f"\n[2/2] Segment master metadata saved to: {seg_meta_csv}")
    print(f"      Total Core/Ext segments indexed: {len(df_seg_meta)}")
    if "is_core4" in df_seg_meta.columns and len(df_seg_meta) > 0:
        c4_counts = df_seg_meta[df_seg_meta["is_core4"] & (df_seg_meta["split"] == "unseen_test")]["source_class"].value_counts().to_dict()
        print(f"      Unseen-Test Core-4 breakdown: {c4_counts}")

if __name__ == "__main__":
    main()
