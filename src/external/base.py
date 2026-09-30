"""
Base Interfaces and Quality-Control Gate for External Datasets.
Provides ExternalRecord, BaseExternalDataset, and QualityControlGate for reproducible audit.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterator
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.external.canonical_pose import skeleton_13_to_dataframe, canonical_geometric_normalization
from src.data.features import extract_features_by_method
from src.constants import DEFAULT_SEQ_LEN

@dataclass
class ExternalRecord:
    """
    Unified record representation for an action execution segment.
    """
    record_id: str                   # Unique identifier (e.g. 'mmfit_w00_act01_squats')
    dataset_name: str                # 'mmfit' or 'fit3d'
    subject_id: str                  # Subject identifier (e.g. 'w00' or 's01')
    action_raw: str                  # Original dataset label (e.g. 'squats')
    action_canonical: str            # Mapped canonical SkelGym label (e.g. 'squat')
    skelgym_class_idx: int           # 0-21 integer index into ACTIONS
    skeleton: np.ndarray             # (T, 13, 3) canonical 13-joint Cartesian coordinates
    fps: float                       # Canonical frame rate (30.0 Hz)
    pose_source: str                 # 'mediapipe' or 'native'
    metadata: Dict[str, Any] = field(default_factory=dict) # repetitions, original bounds, etc.

@dataclass
class QualityControlGate:
    """
    Rigorous pre-inference audit gate to reject corrupted or severely incomplete sequences.
    Logs every exclusion with quantitative rationale to ensure audit transparency.
    """
    max_nan_ratio: float = 0.20
    max_zero_ratio: float = 0.20
    min_frames: int = 16
    min_pose_success: float = 0.80

    def audit_record(self, record: ExternalRecord) -> Tuple[bool, Optional[str]]:
        """
        Evaluates a single ExternalRecord against QC thresholds.
        Returns (is_passed, exclusion_reason).
        """
        skel = record.skeleton
        if skel is None or len(skel) == 0:
            return False, "Empty or None skeleton array"

        T = skel.shape[0]
        if T < self.min_frames:
            return False, f"Sequence length T={T} < min_frames={self.min_frames}"

        # 1. NaN check
        nan_mask = np.isnan(skel).any(axis=(1, 2))
        nan_ratio = np.mean(nan_mask)
        if nan_ratio > self.max_nan_ratio:
            return False, f"NaN frame ratio {nan_ratio:.2%} > {self.max_nan_ratio:.2%}"

        # 2. Zero-frame / missing landmark check
        flat_coords = np.nan_to_num(skel, nan=0.0)
        zeros_mask = (np.abs(flat_coords) < 1e-4).all(axis=(1, 2))
        zero_ratio = np.mean(zeros_mask)
        if zero_ratio > self.max_zero_ratio:
            return False, f"Zero frame ratio {zero_ratio:.2%} > {self.max_zero_ratio:.2%}"

        # 3. Pose success rate (valid non-NaN, non-zero frames)
        valid_frames = (~nan_mask) & (~zeros_mask)
        pose_success = np.mean(valid_frames)
        if pose_success < self.min_pose_success:
            return False, f"Pose detection success rate {pose_success:.2%} < {self.min_pose_success:.2%}"

        return True, None

class BaseExternalDataset:
    """
    Abstract Base Class for external benchmarks.
    """
    def __init__(
        self,
        name: str,
        pose_source: str = "native",
        qc_gate: Optional[QualityControlGate] = None,
        apply_geometric_norm: bool = True
    ):
        self.name = name
        self.pose_source = pose_source
        self.qc_gate = qc_gate or QualityControlGate()
        self.apply_geometric_norm = apply_geometric_norm
        self.records: List[ExternalRecord] = []
        self.audit_log: List[Dict[str, Any]] = []

    def add_record(self, record: ExternalRecord) -> bool:
        """
        Validates record through QC gate before admission.
        Logs to audit table if rejected.
        """
        is_valid, reason = self.qc_gate.audit_record(record)
        if not is_valid:
            self.audit_log.append({
                "record_id": record.record_id,
                "dataset": self.name,
                "subject": record.subject_id,
                "action_raw": record.action_raw,
                "action_canonical": record.action_canonical,
                "status": "EXCLUDED",
                "reason": reason,
                "num_frames": len(record.skeleton) if record.skeleton is not None else 0
            })
            return False

        if self.apply_geometric_norm:
            record.skeleton = canonical_geometric_normalization(record.skeleton)

        self.records.append(record)
        self.audit_log.append({
            "record_id": record.record_id,
            "dataset": self.name,
            "subject": record.subject_id,
            "action_raw": record.action_raw,
            "action_canonical": record.action_canonical,
            "status": "ACCEPTED",
            "reason": "Passed all QC gates",
            "num_frames": len(record.skeleton)
        })
        return True

    def export_audit_report(self, dest_path: Union[str, Path]) -> pd.DataFrame:
        """
        Exports external_data_audit.csv documenting dataset retention metrics.
        """
        dest_p = Path(dest_path)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        df_audit = pd.DataFrame(self.audit_log)
        df_audit.to_csv(dest_p, index=False)
        return df_audit

    def extract_windows(
        self,
        feature_method: str,
        seq_len: int = DEFAULT_SEQ_LEN,
        stride: int = DEFAULT_SEQ_LEN
    ) -> Dict[str, Any]:
        """
        Extracts fixed-length feature windows segmented strictly within record boundaries.
        Returns:
          - 'features': np.ndarray of shape (N_windows, seq_len, D)
          - 'labels': np.ndarray of shape (N_windows,) - canonical SkelGym class indices
          - 'record_ids': list of length N_windows
          - 'subject_ids': list of length N_windows
        """
        all_features = []
        all_labels = []
        all_rec_ids = []
        all_subj_ids = []

        half_seq = seq_len // 2

        for rec in self.records:
            skel = rec.skeleton  # (T, 13, 3)
            T = skel.shape[0]
            if T < half_seq:
                continue

            # Convert to DataFrame matching SkelGym feature format
            df_rec = skeleton_13_to_dataframe(skel)
            feat = extract_features_by_method(df_rec, feature_method)
            # feat shape: (T, D)

            def _slice(s, e):
                sub_feat = feat[s:e]
                L = len(sub_feat)
                if L < half_seq:
                    return None
                if L < seq_len:
                    t = torch.from_numpy(sub_feat).float().unsqueeze(0).permute(0, 2, 1)
                    t_stretched = F.interpolate(t, size=seq_len, mode="linear", align_corners=False)
                    return t_stretched.squeeze(0).permute(1, 0).numpy()
                return sub_feat

            if T < seq_len:
                w = _slice(0, T)
                if w is not None:
                    all_features.append(w)
                    all_labels.append(rec.skelgym_class_idx)
                    all_rec_ids.append(rec.record_id)
                    all_subj_ids.append(rec.subject_id)
            else:
                for start in range(0, T, stride):
                    end = start + seq_len
                    if end <= T:
                        w = _slice(start, end)
                        if w is not None:
                            all_features.append(w)
                            all_labels.append(rec.skelgym_class_idx)
                            all_rec_ids.append(rec.record_id)
                            all_subj_ids.append(rec.subject_id)
                    else:
                        w = _slice(start, T)
                        if w is not None:
                            all_features.append(w)
                            all_labels.append(rec.skelgym_class_idx)
                            all_rec_ids.append(rec.record_id)
                            all_subj_ids.append(rec.subject_id)
                        break

        if len(all_features) == 0:
            feat_arr = np.zeros((0, seq_len, 39), dtype=np.float32)
        else:
            feat_arr = np.stack(all_features, axis=0).astype(np.float32)

        return {
            "features": feat_arr,
            "labels": np.array(all_labels, dtype=np.int64),
            "record_ids": all_rec_ids,
            "subject_ids": all_subj_ids
        }
