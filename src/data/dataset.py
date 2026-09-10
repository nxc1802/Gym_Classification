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
import torch.nn.functional as F
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

def handle_zero_frames(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    """
    Handles missing or undetected frames (zero frames from MediaPipe failures) in landmark sequences.
    method:
      - 'zero': keeps undetected landmarks as 0.0 (baseline)
      - 'ffill': forward-fills valid coordinates, back-fills initial gaps
      - 'linear': pandas linear interpolation across time for missing/zero coordinate values
      - 'interpolate': intelligent interpolation using torch.nn.functional.interpolate.
            For gaps < 50% of segment: fills via linear interpolation between neighbors.
            For segments >= 50% zero: stretches valid frames via 1D interpolation to fill.
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

    if method == "ffill":
        df_clean.loc[row_is_zero, coord_cols] = np.nan
        df_clean[coord_cols] = df_clean[coord_cols].ffill().bfill().fillna(0.0)
    elif method == "linear":
        df_clean.loc[row_is_zero, coord_cols] = np.nan
        df_clean[coord_cols] = df_clean[coord_cols].interpolate(method="linear", limit_direction="both").fillna(0.0)
    elif method == "interpolate":
        n_total = len(df_clean)
        n_zero = row_is_zero.sum()
        zero_ratio = n_zero / n_total

        valid_mask = ~row_is_zero
        if not valid_mask.any():
            # All frames are zero — nothing to interpolate from
            return df_clean.fillna(0.0)

        valid_data = df_clean.loc[valid_mask, coord_cols].values.astype(np.float32)  # (N_valid, C)
        n_valid = valid_data.shape[0]

        if zero_ratio >= 0.5:
            # High zero ratio: stretch valid frames to fill entire segment via torch interpolate
            t_valid = torch.from_numpy(valid_data).unsqueeze(0).permute(0, 2, 1)  # (1, C, N_valid)
            t_stretched = F.interpolate(t_valid, size=n_total, mode="linear", align_corners=False)
            filled = t_stretched.squeeze(0).permute(1, 0).numpy()  # (N_total, C)
            df_clean[coord_cols] = filled
        else:
            # Low zero ratio: use linear interpolation between valid neighbors
            df_clean.loc[row_is_zero, coord_cols] = np.nan
            df_clean[coord_cols] = df_clean[coord_cols].interpolate(
                method="linear", limit_direction="both"
            ).fillna(0.0)

    return df_clean.fillna(0.0)

def sliding_windows(
    data: np.ndarray,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_TRAIN_STRIDE
) -> List[np.ndarray]:
    """
    Slices a continuous frame array (T, D) into overlapping windows of size `seq_len`.
    Last-sample handling:
      - If total frames < seq_len and >= 50% of seq_len: interpolate-stretch to seq_len.
      - If total frames < seq_len and < 50%: discard (return empty).
      - For trailing partial windows:
        - >= 50% of seq_len: stretch via torch.nn.functional.interpolate.
        - < 50% of seq_len: discard the partial window.
    """
    T = data.shape[0]
    if T == 0:
        return []

    half_seq = seq_len // 2  # 50% threshold

    if T < seq_len:
        if T >= half_seq:
            # Stretch via interpolation
            t_data = torch.from_numpy(data).float().unsqueeze(0).permute(0, 2, 1)  # (1, D, T)
            t_stretched = F.interpolate(t_data, size=seq_len, mode="linear", align_corners=False)
            return [t_stretched.squeeze(0).permute(1, 0).numpy()]  # (seq_len, D)
        else:
            # Too short — discard
            return []

    windows = []
    for start in range(0, T, stride):
        end = start + seq_len
        if end <= T:
            windows.append(data[start:end])
        else:
            partial = data[start:]
            partial_len = len(partial)
            if partial_len >= half_seq:
                # Stretch partial to full seq_len via interpolation
                t_partial = torch.from_numpy(partial).float().unsqueeze(0).permute(0, 2, 1)  # (1, D, partial_len)
                t_stretched = F.interpolate(t_partial, size=seq_len, mode="linear", align_corners=False)
                windows.append(t_stretched.squeeze(0).permute(1, 0).numpy())
            # else: discard — too few frames
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
        in_memory: bool = True,
        video_ids: Optional[List[str]] = None
    ):
        self.labels = labels
        self.augment_method = augment_method
        self.is_branch = is_branch
        self.augmenter = LandmarkAugmenter() if augment_method and augment_method != "none" else None
        self.in_memory = in_memory
        self.video_ids = video_ids

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

def mirror_dataframe_horizontally(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies bilateral horizontal mirror symmetry:
    1. Inverts x coordinates (x -> -x).
    2. Swaps corresponding left and right landmark columns.
    """
    df_flipped = df.copy()
    pairs = [
        ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        ("LEFT_ELBOW", "RIGHT_ELBOW"),
        ("LEFT_WRIST", "RIGHT_WRIST"),
        ("LEFT_HIP", "RIGHT_HIP"),
        ("LEFT_KNEE", "RIGHT_KNEE"),
        ("LEFT_ANKLE", "RIGHT_ANKLE"),
    ]
    if "NOSE_x" in df_flipped.columns:
        df_flipped["NOSE_x"] = -df_flipped["NOSE_x"]

    for l_pt, r_pt in pairs:
        for dim in ["x", "y", "z", "visibility"]:
            l_col = f"{l_pt}_{dim}"
            r_col = f"{r_pt}_{dim}"
            if l_col in df.columns and r_col in df.columns:
                if dim == "x":
                    df_flipped[l_col] = -df[r_col]
                    df_flipped[r_col] = -df[l_col]
                else:
                    df_flipped[l_col] = df[r_col]
                    df_flipped[r_col] = df[l_col]
    return df_flipped

def build_dataset_from_csvs(
    metadata_df: pd.DataFrame,
    split: str,
    feature_method: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
    augment_method: Optional[str] = None,
    zero_frame_handling: str = "interpolate",
    landmark_dir: Optional[str] = None,
    smoke_test: bool = False,
    smoke_class: Optional[str] = "barbell biceps curl",
    in_memory: bool = True,
    is_horizontal_flip: bool = False
) -> GymDataset:
    """
    Loads landmark CSVs based on metadata split and constructs a GymDataset.
    Supports smoke_test mode to select minimal samples for debugging.
    Supports is_horizontal_flip for Test-Time Augmentation (TTA).
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
    all_video_ids = []

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
                    vid_name = csv_path.stem
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        continue
                    df = handle_zero_frames(df, method=zero_frame_handling)
                    if is_horizontal_flip:
                        df = mirror_dataframe_horizontally(df)
                    feat = extract_features_by_method(df, feature_method)
                    if is_branch:
                        f1, f2 = feat
                        w1 = sliding_windows(f1, seq_len, stride)
                        w2 = sliding_windows(f2, seq_len, stride)
                        n_wins = min(len(w1), len(w2))
                        for i in range(n_wins):
                            all_samples.append((w1[i], w2[i]))
                            all_labels.append(class_idx)
                            all_video_ids.append(vid_name)
                    else:
                        wins = sliding_windows(feat, seq_len, stride)
                        for w in wins:
                            all_samples.append(w)
                            all_labels.append(class_idx)
                            all_video_ids.append(vid_name)
                except Exception:
                    continue
    else:
        for _, row in split_df.iterrows():
            action_name = row["class"]
            if action_name not in ACTION_TO_IDX:
                continue
            class_idx = ACTION_TO_IDX[action_name]

            vid_name = Path(row.get("filepath", f"video_{class_idx}")).stem
            csv_path = None
            if landmark_dir:
                # Priority 1: split-specific directory structure
                cand1 = Path(landmark_dir) / split / action_name / f"{Path(row['filepath']).stem}.csv"
                # Priority 2: flat landmark directory (legacy, no split separation)
                cand2 = Path(landmark_dir) / f"{Path(row['filepath']).stem}.csv"
                # NOTE: Removed cand3 (landmark_dir/action/) to prevent cross-split data leakage
                for c in [cand1, cand2]:
                    if c.exists():
                        csv_path = c
                        break

            if csv_path and csv_path.exists():
                df = pd.read_csv(csv_path)
                s, e = parse_segment_range(row["label_content"], len(df))
                df_segment = df.iloc[s:e].reset_index(drop=True)
                df_segment = handle_zero_frames(df_segment, method=zero_frame_handling)
                if is_horizontal_flip:
                    df_segment = mirror_dataframe_horizontally(df_segment)
                feat = extract_features_by_method(df_segment, feature_method)
            else:
                n_frames = int(row.get("num_frames", 75))
                dummy_cols = ["Frame"] + [f"{pt}_{d}" for pt in ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"] for d in ["x", "y", "z", "visibility"]]
                df = pd.DataFrame(np.random.randn(n_frames, len(dummy_cols)), columns=dummy_cols)
                s, e = parse_segment_range(row["label_content"], n_frames)
                df_segment = df.iloc[s:e].reset_index(drop=True)
                df_segment = handle_zero_frames(df_segment, method=zero_frame_handling)
                if is_horizontal_flip:
                    df_segment = mirror_dataframe_horizontally(df_segment)
                feat = extract_features_by_method(df_segment, feature_method)

            if is_branch:
                f1, f2 = feat
                w1 = sliding_windows(f1, seq_len, stride)
                w2 = sliding_windows(f2, seq_len, stride)
                n_wins = min(len(w1), len(w2))
                for i in range(n_wins):
                    all_samples.append((w1[i], w2[i]))
                    all_labels.append(class_idx)
                    all_video_ids.append(vid_name)
            else:
                wins = sliding_windows(feat, seq_len, stride)
                for w in wins:
                    all_samples.append(w)
                    all_labels.append(class_idx)
                    all_video_ids.append(vid_name)

    # Dataset Expansion: Preserve 100% of clean original samples and append 3 augmented variants (1→4 total)
    if split == "train" and augment_method and augment_method != "none":
        augmenter = LandmarkAugmenter()
        aug_samples = []
        aug_labels = []
        aug_video_ids = []
        for s, l, v_id in zip(all_samples, all_labels, all_video_ids):
            if is_branch:
                # For branch: augment first branch, keep second unchanged
                t1 = torch.from_numpy(s[0]).float()
                variants = augmenter.generate_augmented_variants(t1, augment_method)
                for idx_var, v in enumerate(variants):
                    aug_samples.append((v.numpy(), s[1].copy()))
                    aug_labels.append(l)
                    aug_video_ids.append(f"{v_id}_aug_{idx_var}")
            else:
                t = torch.from_numpy(s).float()
                variants = augmenter.generate_augmented_variants(t, augment_method)
                for idx_var, v in enumerate(variants):
                    aug_samples.append(v.numpy())
                    aug_labels.append(l)
                    aug_video_ids.append(f"{v_id}_aug_{idx_var}")

        # Retain original clean samples + 3 augmented variants per sample (4× total)
        all_samples = all_samples + aug_samples
        all_labels = all_labels + aug_labels
        all_video_ids = all_video_ids + aug_video_ids

    return GymDataset(
        samples=all_samples,
        labels=all_labels,
        augment_method=None,
        is_branch=is_branch,
        in_memory=in_memory,
        video_ids=all_video_ids
    )

def _compute_train_stats(train_ds: 'GymDataset') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes global mean and std across all training samples for z-score normalization.
    Returns (mean, std) tensors of shape (1, D) where D = feature dimension.
    """
    if train_ds.in_memory and train_ds.tensor_samples is not None:
        # Efficient: work directly with the stacked tensor (N, T, D)
        all_data = train_ds.tensor_samples  # (N, T, D)
        # Compute stats across samples and time steps
        flat = all_data.reshape(-1, all_data.shape[-1])  # (N*T, D)
    else:
        # Collect from samples list
        arrays = []
        for s in train_ds.samples:
            if isinstance(s, tuple):
                arrays.append(s[0])  # Only first branch for stats
            elif isinstance(s, np.ndarray):
                arrays.append(s)
            else:
                arrays.append(s.numpy() if isinstance(s, torch.Tensor) else s)
        stacked = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        flat = torch.from_numpy(stacked).float()

    mean = flat.mean(dim=0, keepdim=True)  # (1, D)
    std = flat.std(dim=0, keepdim=True) + 1e-7  # (1, D)
    return mean, std

def _apply_normalization(ds: 'GymDataset', mean: torch.Tensor, std: torch.Tensor) -> None:
    """
    Applies z-score normalization in-place to a GymDataset using provided mean/std.
    """
    if ds.in_memory and ds.tensor_samples is not None:
        # In-memory tensor: normalize directly
        # mean/std shape (1, D), tensor_samples shape (N, T, D)
        ds.tensor_samples = (ds.tensor_samples - mean.unsqueeze(0)) / std.unsqueeze(0)
    elif ds.samples is not None:
        for i, s in enumerate(ds.samples):
            if isinstance(s, tuple):
                continue  # Skip branch samples for now
            if isinstance(s, np.ndarray):
                m = mean.squeeze(0).numpy()
                s_np = std.squeeze(0).numpy()
                ds.samples[i] = ((s - m) / s_np).astype(np.float32)

    # Store stats as attributes for inference
    ds.train_mean = mean
    ds.train_std = std

def get_dataloaders(
    metadata_path: str,
    feature_method: str,
    batch_size: int = 16,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: Optional[int] = None,
    val_test_stride: Optional[int] = None,
    augment_method: Optional[str] = None,
    zero_frame_handling: str = "interpolate",
    landmark_dir: Optional[str] = None,
    num_workers: int = 0,
    smoke_test: bool = False,
    smoke_class: Optional[str] = "barbell biceps curl",
    in_memory: bool = True,
    is_horizontal_flip: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders.
    Applies global z-score normalization using train-set statistics to prevent data leakage.
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
        landmark_dir=landmark_dir, smoke_test=smoke_test, smoke_class=smoke_class, in_memory=in_memory,
        is_horizontal_flip=is_horizontal_flip
    )

    # Global z-score normalization using TRAIN-SET statistics only (prevents data leakage)
    is_branch = (feature_method == "branch_concat")
    if not is_branch and len(train_ds) > 0:
        train_mean, train_std = _compute_train_stats(train_ds)
        _apply_normalization(train_ds, train_mean, train_std)
        _apply_normalization(val_ds, train_mean, train_std)
        _apply_normalization(test_ds, train_mean, train_std)

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
