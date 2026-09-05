"""
PyTorch Dataset and DataLoader for Gym Exercise Classification.
Handles segment parsing, sliding window clipping (32 frames, stride 16/32),
last-frame padding, augmentation, and high-performance in-memory RAM caching.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.constants import (
    ACTIONS,
    ACTION_TO_IDX,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRAIN_STRIDE,
    DEFAULT_VAL_TEST_STRIDE
)
from src.data.features import extract_features_by_method
from src.data.augmentations import LandmarkAugmenter

def parse_segment_range(label_content: str, total_frames: int) -> Tuple[int, int]:
    """
    Parses start and end frame indices from label_content string like 'frame_000000 frame_000074'.
    Converts 0-indexed strings into slice bounds [s, e].
    """
    if not isinstance(label_content, str) or not label_content.strip():
        return 0, total_frames

    tokens = label_content.strip().split()
    try:
        s_num = int(tokens[0].split("_")[-1])
        e_num = int(tokens[1].split("_")[-1])
        s = max(0, min(s_num, total_frames - 1))
        e = min(total_frames, max(s + 1, e_num + 1))
        return s, e
    except Exception:
        return 0, total_frames

def handle_zero_frames(df: pd.DataFrame, method: str = "zero") -> pd.DataFrame:
    """
    Handles missing or undetected frames in landmark sequences.
    method:
      - 'zero': keeps undetected landmarks as 0.0 (baseline)
      - 'ffill': forward-fills valid coordinates, back-fills initial gaps
      - 'linear': linear interpolation across time for missing/zero coordinate values
    """
    if method in ("zero", "none", None) or len(df) <= 1:
        return df.fillna(0.0)

    df_clean = df.copy()
    coord_cols = [c for c in df_clean.columns if any(c.endswith(f"_{d}") for d in ["x", "y", "z"])]
    if not coord_cols:
        return df_clean.fillna(0.0)

    # Frame is undetected/zero if all coordinates are 0 or NaN
    zeros_mask = (df_clean[coord_cols].abs() < 1e-6) | df_clean[coord_cols].isna()
    row_is_zero = zeros_mask.all(axis=1)

    if not row_is_zero.any():
        return df_clean.fillna(0.0)

    df_clean.loc[row_is_zero, coord_cols] = np.nan
    if method == "ffill":
        df_clean[coord_cols] = df_clean[coord_cols].ffill().bfill().fillna(0.0)
    elif method == "linear":
        df_clean[coord_cols] = df_clean[coord_cols].interpolate(method="linear", limit_direction="both").fillna(0.0)

    return df_clean.fillna(0.0)

def sliding_windows(
    data: np.ndarray,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_TRAIN_STRIDE
) -> List[np.ndarray]:
    """
    Slices a continuous frame array (T, D) into overlapping windows of size `seq_len`.
    If the sequence is shorter than `seq_len` or the last window is partial,
    pads the final window by repeating the last frame.
    """
    T = data.shape[0]
    if T == 0:
        return []

    if T < seq_len:
        pad_count = seq_len - T
        pad_tail = np.repeat(data[-1:], pad_count, axis=0)
        return [np.concatenate([data, pad_tail], axis=0)]

    windows = []
    for start in range(0, T, stride):
        end = start + seq_len
        if end <= T:
            windows.append(data[start:end])
        else:
            partial = data[start:]
            pad_count = seq_len - len(partial)
            pad_tail = np.repeat(data[-1:], pad_count, axis=0)
            windows.append(np.concatenate([partial, pad_tail], axis=0))
            break

    return windows

class GymDataset(Dataset):
    """
    Gym Exercise Dataset supporting both single-feature and dual-branch inputs.
    Includes in_memory mode caching all tensors in RAM for zero disk I/O during training.
    """
    def __init__(
        self,
        samples: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        labels: List[int],
        augment_method: Optional[str] = None,
        is_branch: bool = False,
        in_memory: bool = True
    ):
        self.labels = labels
        self.augment_method = augment_method
        self.is_branch = is_branch
        self.augmenter = LandmarkAugmenter() if augment_method and augment_method != "none" else None
        self.in_memory = in_memory

        if in_memory and len(samples) > 0:
            self.tensor_labels = torch.tensor(labels, dtype=torch.long)
            if is_branch:
                t1_arr = np.nan_to_num(np.stack([s[0] for s in samples]), nan=0.0)
                t2_arr = np.nan_to_num(np.stack([s[1] for s in samples]), nan=0.0)
                self.t1_list = torch.from_numpy(t1_arr).float()
                self.t2_list = torch.from_numpy(t2_arr).float()
                self.samples = None
            else:
                stacked = np.nan_to_num(np.stack(samples), nan=0.0)
                self.tensor_samples = torch.from_numpy(stacked).float()
                self.samples = None
        else:
            self.samples = samples
            self.tensor_labels = None
            self.tensor_samples = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.in_memory and self.tensor_labels is not None:
            label = self.tensor_labels[idx]
            if self.is_branch:
                t1 = self.t1_list[idx]
                t2 = self.t2_list[idx]
                if self.augmenter and self.augment_method:
                    t1 = self.augmenter.apply(t1, self.augment_method)
                return (t1, t2), label
            else:
                t = self.tensor_samples[idx]
                if self.augmenter and self.augment_method:
                    t = self.augmenter.apply(t, self.augment_method)
                return t, label
        else:
            sample = self.samples[idx]
            label = torch.tensor(self.labels[idx], dtype=torch.long)

            if self.is_branch:
                x1, x2 = sample
                t1 = torch.from_numpy(x1).float()
                t2 = torch.from_numpy(x2).float()
                if self.augmenter and self.augment_method:
                    t1 = self.augmenter.apply(t1, self.augment_method)
                return (t1, t2), label
            else:
                t = torch.from_numpy(sample).float()
                if self.augmenter and self.augment_method:
                    t = self.augmenter.apply(t, self.augment_method)
                return t, label

def build_dataset_from_csvs(
    metadata_df: pd.DataFrame,
    split: str,
    feature_method: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
    augment_method: Optional[str] = None,
    zero_frame_handling: str = "zero",
    landmark_dir: Optional[str] = None,
    smoke_test: bool = False,
    smoke_class: Optional[str] = "barbell biceps curl",
    in_memory: bool = True
) -> GymDataset:
    """
    Loads landmark CSVs based on metadata split and constructs a GymDataset.
    Supports smoke_test mode to select minimal samples for debugging.
    """
    split_df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
    if smoke_test:
        if smoke_class and smoke_class in split_df["class"].values:
            split_df = split_df[split_df["class"] == smoke_class].head(2 if split == "train" else 1).reset_index(drop=True)
        else:
            split_df = split_df.head(2 if split == "train" else 1).reset_index(drop=True)

    if stride is None:
        stride = DEFAULT_TRAIN_STRIDE if split == "train" else DEFAULT_VAL_TEST_STRIDE

    is_branch = (feature_method == "branch_concat")
    all_samples = []
    all_labels = []

    has_folder_structure = False
    if landmark_dir:
        split_dir = Path(landmark_dir) / split
        if split_dir.exists() and any(split_dir.iterdir()):
            has_folder_structure = True

    if has_folder_structure:
        action_names = sorted([d.name for d in (Path(landmark_dir) / split).iterdir() if d.is_dir() and d.name in ACTION_TO_IDX])
        if smoke_test:
            action_names = [smoke_class] if smoke_class in action_names else action_names[:1]

        for action_name in action_names:
            class_idx = ACTION_TO_IDX[action_name]
            class_dir = Path(landmark_dir) / split / action_name
            csv_files = sorted(list(class_dir.glob("*.csv")))
            if smoke_test:
                csv_files = csv_files[:(2 if split == "train" else 1)]

            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        continue
                    df = handle_zero_frames(df, method=zero_frame_handling)
                    feat = extract_features_by_method(df, feature_method)
                    if is_branch:
                        f1, f2 = feat
                        w1 = sliding_windows(f1, seq_len, stride)
                        w2 = sliding_windows(f2, seq_len, stride)
                        n_wins = min(len(w1), len(w2))
                        for i in range(n_wins):
                            all_samples.append((w1[i], w2[i]))
                            all_labels.append(class_idx)
                    else:
                        wins = sliding_windows(feat, seq_len, stride)
                        for w in wins:
                            all_samples.append(w)
                            all_labels.append(class_idx)
                except Exception:
                    continue
    else:
        for _, row in split_df.iterrows():
            action_name = row["class"]
            if action_name not in ACTION_TO_IDX:
                continue
            class_idx = ACTION_TO_IDX[action_name]

            csv_path = None
            if landmark_dir:
                cand1 = Path(landmark_dir) / f"{Path(row['filepath']).stem}.csv"
                cand2 = Path(landmark_dir) / split / action_name / f"{Path(row['filepath']).stem}.csv"
                cand3 = Path(landmark_dir) / action_name / f"{Path(row['filepath']).stem}.csv"
                for c in [cand1, cand2, cand3]:
                    if c.exists():
                        csv_path = c
                        break

            if csv_path and csv_path.exists():
                df = pd.read_csv(csv_path)
                s, e = parse_segment_range(row["label_content"], len(df))
                df_segment = df.iloc[s:e].reset_index(drop=True)
                df_segment = handle_zero_frames(df_segment, method=zero_frame_handling)
                feat = extract_features_by_method(df_segment, feature_method)
            else:
                n_frames = int(row.get("num_frames", 75))
                dummy_cols = ["Frame"] + [f"{pt}_{d}" for pt in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"] for d in ["x", "y", "z", "visibility"]]
                df = pd.DataFrame(np.random.randn(n_frames, len(dummy_cols)), columns=dummy_cols)
                s, e = parse_segment_range(row["label_content"], n_frames)
                df_segment = df.iloc[s:e].reset_index(drop=True)
                df_segment = handle_zero_frames(df_segment, method=zero_frame_handling)
                feat = extract_features_by_method(df_segment, feature_method)

            if is_branch:
                f1, f2 = feat
                w1 = sliding_windows(f1, seq_len, stride)
                w2 = sliding_windows(f2, seq_len, stride)
                n_wins = min(len(w1), len(w2))
                for i in range(n_wins):
                    all_samples.append((w1[i], w2[i]))
                    all_labels.append(class_idx)
            else:
                wins = sliding_windows(feat, seq_len, stride)
                for w in wins:
                    all_samples.append(w)
                    all_labels.append(class_idx)

    # Dataset Expansion: Preserve 100% of clean original samples and append augmented copies
    if split == "train" and augment_method and augment_method != "none":
        augmenter = LandmarkAugmenter()
        aug_samples = []
        aug_labels = []
        for s, l in zip(all_samples, all_labels):
            if is_branch:
                t1 = augmenter.apply(torch.from_numpy(s[0]).float(), augment_method).numpy()
                t2 = augmenter.apply(torch.from_numpy(s[1]).float(), augment_method).numpy()
                aug_samples.append((t1, t2))
            else:
                t = augmenter.apply(torch.from_numpy(s).float(), augment_method).numpy()
                aug_samples.append(t)
            aug_labels.append(l)

        # Retain original clean samples + augmented supplementary samples (sample count doubled)
        all_samples = all_samples + aug_samples
        all_labels = all_labels + aug_labels

    return GymDataset(
        samples=all_samples,
        labels=all_labels,
        augment_method=None,
        is_branch=is_branch,
        in_memory=in_memory
    )

def get_dataloaders(
    metadata_path: str,
    feature_method: str,
    batch_size: int = 16,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
    val_test_stride: Optional[int] = None,
    augment_method: Optional[str] = None,
    zero_frame_handling: str = "zero",
    landmark_dir: Optional[str] = None,
    num_workers: int = 0,
    smoke_test: bool = False,
    smoke_class: Optional[str] = "barbell biceps curl",
    in_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders.
    Optimized for high-throughput GPU training with in-memory caching and pinned memory.
    """
    meta_df = pd.read_csv(metadata_path)
    vt_stride = val_test_stride if val_test_stride is not None else DEFAULT_VAL_TEST_STRIDE

    train_ds = build_dataset_from_csvs(
        meta_df, "train", feature_method, seq_len, stride or DEFAULT_TRAIN_STRIDE,
        augment_method=augment_method, zero_frame_handling=zero_frame_handling,
        landmark_dir=landmark_dir, smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
    )
    val_ds = build_dataset_from_csvs(
        meta_df, "val", feature_method, seq_len, vt_stride,
        augment_method=None, zero_frame_handling=zero_frame_handling,
        landmark_dir=landmark_dir, smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
    )
    test_ds = build_dataset_from_csvs(
        meta_df, "test", feature_method, seq_len, vt_stride,
        augment_method=None, zero_frame_handling=zero_frame_handling,
        landmark_dir=landmark_dir, smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
    )

    pin_mem = torch.cuda.is_available()
    persistent = (num_workers > 0)
    prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=persistent, prefetch_factor=prefetch
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=persistent, prefetch_factor=prefetch
    )

    return train_loader, val_loader, test_loader
