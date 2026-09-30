"""
MM-Fit External Benchmark Dataset Adapter.
Loads synchronized workout sessions from MM-Fit (Strömbäck et al.), parses activity segments,
converts OpenPose 18 / MediaPipe poses to canonical 13-joint schema, and partitions into
official Unseen-Subject and Seen-Subject splits.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from src.constants import ACTION_TO_IDX
from src.external.base import BaseExternalDataset, ExternalRecord, QualityControlGate
from src.external.canonical_pose import openpose18_to_skelgym13, mediapipe33_to_skelgym13
from src.external.class_mapping import get_class_mapping

# Official MM-Fit Workout Partitions
MMFIT_UNSEEN_TEST: List[str] = ["w00", "w05", "w12", "w13", "w20"]
MMFIT_SEEN_TEST: List[str] = ["w09", "w10", "w11"]
MMFIT_VAL: List[str] = ["w14", "w15", "w19"]
MMFIT_TRAIN: List[str] = ["w01", "w02", "w03", "w04", "w06", "w07", "w08", "w16", "w17", "w18"]

class MMFitExternalDataset(BaseExternalDataset):
    """
    MM-Fit Dataset Adapter supporting both Protocol A (MediaPipe from RGB)
    and Protocol B (Native OpenPose 3D).
    """
    def __init__(
        self,
        root_dir: Union[str, Path] = "mm-fit",
        split_group: str = "unseen_test",  # 'unseen_test', 'seen_test', 'all_test', 'all'
        class_set: str = "core4",          # 'core4' or 'extended5'
        pose_source: str = "native",       # 'native' or 'mediapipe'
        qc_gate: Optional[QualityControlGate] = None,
        apply_geometric_norm: bool = True
    ):
        super().__init__(
            name="mmfit",
            pose_source=pose_source,
            qc_gate=qc_gate or QualityControlGate(),
            apply_geometric_norm=apply_geometric_norm
        )
        self.root_dir = Path(root_dir)
        self.split_group = split_group
        self.class_set = class_set
        self.class_mapping = get_class_mapping("mmfit", class_set)

        self._load_dataset()

    def _determine_workouts(self) -> List[str]:
        sg = self.split_group.lower()
        if sg in ("unseen", "unseen_test", "primary"):
            return MMFIT_UNSEEN_TEST
        elif sg in ("seen", "seen_test", "secondary"):
            return MMFIT_SEEN_TEST
        elif sg in ("all_test", "test"):
            return MMFIT_UNSEEN_TEST + MMFIT_SEEN_TEST
        elif sg == "val":
            return MMFIT_VAL
        elif sg == "train":
            return MMFIT_TRAIN
        elif sg == "all":
            all_w = MMFIT_UNSEEN_TEST + MMFIT_SEEN_TEST + MMFIT_VAL + MMFIT_TRAIN
            return sorted(all_w)
        else:
            raise ValueError(f"Unknown MM-Fit split_group '{self.split_group}'.")

    def _load_dataset(self) -> None:
        target_workouts = self._determine_workouts()
        fps = 30.0  # MM-Fit RGB-D capture rate is native 30 Hz

        for w in target_workouts:
            w_dir = self.root_dir / w
            if not w_dir.exists():
                # Workout directory not found locally; log in audit
                self.audit_log.append({
                    "record_id": f"mmfit_{w}_missing",
                    "dataset": "mmfit",
                    "subject": w,
                    "action_raw": "N/A",
                    "action_canonical": "N/A",
                    "status": "EXCLUDED",
                    "reason": f"Directory '{w_dir}' does not exist",
                    "num_frames": 0
                })
                continue

            labels_csv = w_dir / f"{w}_labels.csv"
            if not labels_csv.exists():
                continue

            # MM-Fit labels format (headerless): start_frame, end_frame, rep_count, action_name
            try:
                df_labels = pd.read_csv(labels_csv, header=None)
            except Exception as e:
                continue

            # Load full workout pose data
            full_pose_3d = None
            if self.pose_source == "native":
                pose_file = w_dir / f"{w}_pose_3d.npy"
                if pose_file.exists():
                    full_pose_3d = np.load(pose_file)  # shape (3, T, 18)
            elif self.pose_source == "mediapipe":
                # Check for extracted MediaPipe landmarks file if present
                mp_file = w_dir / f"{w}_mediapipe.npy"
                if mp_file.exists():
                    full_pose_3d = np.load(mp_file)

            if full_pose_3d is None:
                continue

            # Parse each action segment
            for seg_idx, row in df_labels.iterrows():
                try:
                    s_frame = int(row[0])
                    e_frame = int(row[1])
                    reps = int(row[2])
                    raw_act = str(row[3]).strip()
                except Exception:
                    continue

                if raw_act not in self.class_mapping:
                    # Excluded class (e.g. lunges, situps, non_activity)
                    continue

                canonical_act = self.class_mapping[raw_act]
                skel_class_idx = ACTION_TO_IDX[canonical_act]

                # Extract frames slice
                T_total = full_pose_3d.shape[1] if full_pose_3d.ndim == 3 and full_pose_3d.shape[0] == 3 else len(full_pose_3d)
                s = max(0, min(s_frame, T_total - 1))
                e = min(T_total, max(s + 1, e_frame))

                if e - s < 16:
                    continue

                if self.pose_source == "native":
                    # full_pose_3d shape is (3, T_total, 18)
                    seg_pose = full_pose_3d[:, s:e, :]
                    skel_13 = openpose18_to_skelgym13(seg_pose)  # (T_seg, 13, 3)
                else:
                    seg_pose = full_pose_3d[s:e]
                    if seg_pose.shape[1] == 33:
                        skel_13 = mediapipe33_to_skelgym13(seg_pose)
                    else:
                        skel_13 = seg_pose[:, :13, :3]

                record = ExternalRecord(
                    record_id=f"mmfit_{w}_seg{seg_idx:02d}_{raw_act}",
                    dataset_name="mmfit",
                    subject_id=w,
                    action_raw=raw_act,
                    action_canonical=canonical_act,
                    skelgym_class_idx=skel_class_idx,
                    skeleton=skel_13,
                    fps=fps,
                    pose_source=self.pose_source,
                    metadata={
                        "workout": w,
                        "rep_count": reps,
                        "start_frame": s,
                        "end_frame": e,
                        "num_frames": e - s
                    }
                )

                self.add_record(record)
