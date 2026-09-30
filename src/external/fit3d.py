"""
Fit3D External Benchmark Dataset Adapter.
Loads Fit3D multi-view and Vicon 3D ground-truth sequences, performs 50 Hz -> 30 Hz temporal resampling,
remaps joints to canonical 13-joint schema, and organizes hierarchical repetition segmentation.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from src.constants import ACTION_TO_IDX
from src.external.base import BaseExternalDataset, ExternalRecord, QualityControlGate
from src.external.canonical_pose import fit3d_to_skelgym13, mediapipe33_to_skelgym13, resample_temporal_sequence
from src.external.class_mapping import get_class_mapping

FIT3D_DEFAULT_SUBJECTS: List[str] = [
    "s02", "s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"
]

class Fit3DExternalDataset(BaseExternalDataset):
    """
    Fit3D Benchmark Adapter supporting 50->30 Hz resampling,
    Protocol A (MediaPipe Pose) and Protocol B (Native Vicon 3D).
    """
    def __init__(
        self,
        root_dir: Union[str, Path] = "data_external/fit3d",
        subjects: Optional[List[str]] = None,
        class_set: str = "core6",           # 'core6' or 'extended7'
        pose_source: str = "native",        # 'native' or 'mediapipe'
        source_fps: float = 50.0,
        target_fps: float = 30.0,
        qc_gate: Optional[QualityControlGate] = None,
        apply_geometric_norm: bool = True
    ):
        super().__init__(
            name="fit3d",
            pose_source=pose_source,
            qc_gate=qc_gate or QualityControlGate(),
            apply_geometric_norm=apply_geometric_norm
        )
        self.root_dir = Path(root_dir)
        self.subjects = subjects or FIT3D_DEFAULT_SUBJECTS
        self.class_set = class_set
        self.source_fps = source_fps
        self.target_fps = target_fps
        self.class_mapping = get_class_mapping("fit3d", class_set)

        self._load_dataset()

    def _load_dataset(self) -> None:
        if not self.root_dir.exists():
            # Fit3D not downloaded or path points to instruction placeholder
            return

        for subj in self.subjects:
            subj_dir = self.root_dir / subj
            if not subj_dir.exists():
                continue

            # Scan exercise subdirectories or recording files
            # Expected structure: subj_dir / {exercise_name} / joints3d_25.json or .npy
            for ex_dir in subj_dir.iterdir():
                if not ex_dir.is_dir():
                    continue

                raw_act = ex_dir.name
                if raw_act not in self.class_mapping:
                    continue

                canonical_act = self.class_mapping[raw_act]
                skel_class_idx = ACTION_TO_IDX[canonical_act]

                # Look for native or mediapipe pose file
                pose_file = None
                if self.pose_source == "native":
                    for cand in ["joints3d_25.npy", "joints3d_17.npy", "vicon_3d.npy"]:
                        if (ex_dir / cand).exists():
                            pose_file = ex_dir / cand
                            break
                elif self.pose_source == "mediapipe":
                    for cand in ["mediapipe_33.npy", "mediapipe.npy", "pose_landmarks.npy"]:
                        if (ex_dir / cand).exists():
                            pose_file = ex_dir / cand
                            break

                if pose_file is None or not pose_file.exists():
                    continue

                try:
                    raw_pose = np.load(pose_file)
                except Exception:
                    continue

                # Map to canonical 13 joints
                if self.pose_source == "native":
                    skel_13 = fit3d_to_skelgym13(raw_pose)
                else:
                    if raw_pose.shape[1] == 33:
                        skel_13 = mediapipe33_to_skelgym13(raw_pose)
                    else:
                        skel_13 = raw_pose[:, :13, :3]

                # Resample 50 Hz -> 30 Hz
                skel_13_resampled = resample_temporal_sequence(
                    skel_13,
                    source_fps=self.source_fps,
                    target_fps=self.target_fps
                )

                # Parse repetitions if rep_annotations.json exists
                rep_file = ex_dir / "rep_annotations.json"
                reps_meta = []
                if rep_file.exists():
                    try:
                        import json
                        with open(rep_file, "r") as f:
                            reps_meta = json.load(f)
                    except Exception:
                        pass

                record = ExternalRecord(
                    record_id=f"fit3d_{subj}_{raw_act}",
                    dataset_name="fit3d",
                    subject_id=subj,
                    action_raw=raw_act,
                    action_canonical=canonical_act,
                    skelgym_class_idx=skel_class_idx,
                    skeleton=skel_13_resampled,
                    fps=self.target_fps,
                    pose_source=self.pose_source,
                    metadata={
                        "subject": subj,
                        "repetitions": reps_meta,
                        "original_fps": self.source_fps,
                        "num_frames": len(skel_13_resampled)
                    }
                )

                self.add_record(record)
