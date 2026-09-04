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
                self.t1_list = torch.from_numpy(np.stack([s[0] for s in samples])).float()
                self.t2_list = torch.from_numpy(np.stack([s[1] for s in samples])).float()
                self.samples = None
            else:
                self.tensor_samples = torch.from_numpy(np.stack(samples)).float()
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
            feat = extract_features_by_method(df_segment, feature_method)
        else:
            n_frames = int(row.get("num_frames", 75))
            dummy_cols = ["Frame"] + [f"{pt}_{d}" for pt in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"] for d in ["x", "y", "z", "visibility"]]
            df = pd.DataFrame(np.random.randn(n_frames, len(dummy_cols)), columns=dummy_cols)
            s, e = parse_segment_range(row["label_content"], n_frames)
            df_segment = df.iloc[s:e].reset_index(drop=True)
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

    return GymDataset(
        samples=all_samples,
        labels=all_labels,
        augment_method=augment_method if split == "train" else None,
        is_branch=is_branch,
        in_memory=in_memory
    )

def get_dataloaders(
    metadata_path: str,
    feature_method: str,
    batch_size: int = 16,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
    augment_method: Optional[str] = None,
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

    train_ds = build_dataset_from_csvs(
        meta_df, "train", feature_method, seq_len, stride or DEFAULT_TRAIN_STRIDE,
        augment_method=augment_method, landmark_dir=landmark_dir,
        smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
    )
    val_ds = build_dataset_from_csvs(
        meta_df, "val", feature_method, seq_len, DEFAULT_VAL_TEST_STRIDE,
        augment_method=None, landmark_dir=landmark_dir,
        smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
    )
    test_ds = build_dataset_from_csvs(
        meta_df, "test", feature_method, seq_len, DEFAULT_VAL_TEST_STRIDE,
        augment_method=None, landmark_dir=landmark_dir,
        smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory
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
