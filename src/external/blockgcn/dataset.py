"""
Specialized 33-Joint 5D Skeleton Dataset and DataLoader for BlockGCN.
Extracts 33 raw MediaPipe joints (XYZ), centers on hip midpoint, applies sliding windows,
and formats into canonical [C=3, T=32, V=33, M=1] tensors with RAM caching.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.constants import RAW_POINTS_33, ACTIONS, ACTION_TO_IDX

# MediaPipe landmark column names for 33 joints (X, Y, Z)
RAW_33_XYZ_COLS = []
for pt in RAW_POINTS_33:
    for dim in ("x", "y", "z"):
        RAW_33_XYZ_COLS.append(f"{pt}_{dim}")

HIP_LEFT_COLS = ["LEFT_HIP_x", "LEFT_HIP_y", "LEFT_HIP_z"]
HIP_RIGHT_COLS = ["RIGHT_HIP_x", "RIGHT_HIP_y", "RIGHT_HIP_z"]


def parse_segment_ranges(label_content: str, total_frames: int) -> List[Tuple[int, int]]:
    """
    Parses start and end frame indices from segment label string.
    """
    if not isinstance(label_content, str) or not label_content.strip():
        return [(0, total_frames)]

    tokens = label_content.strip().split()
    segments = []
    for i in range(0, len(tokens), 2):
        if i + 1 < len(tokens):
            try:
                s_num = int(tokens[i].split("_")[-1])
                e_num = int(tokens[i + 1].split("_")[-1])
                s = max(0, min(s_num, total_frames - 1))
                e = min(total_frames, max(s + 1, e_num + 1))
                if e > s:
                    segments.append((s, e))
            except Exception:
                continue
    return segments if segments else [(0, total_frames)]


def extract_33j_windows(
    df: pd.DataFrame,
    seq_len: int = 32,
    stride: int = 16,
    center_hip: bool = True
) -> List[np.ndarray]:
    """
    Extracts raw 33-joint XYZ coordinates, centers on hip midpoint,
    and returns list of windows with shape (C=3, T=32, V=33, M=1).
    """
    total_frames = len(df)
    if total_frames == 0:
        return []

    # Verify column presence
    available_cols = [c for c in RAW_33_XYZ_COLS if c in df.columns]
    if len(available_cols) < len(RAW_33_XYZ_COLS):
        coords_raw = np.zeros((total_frames, 33 * 3), dtype=np.float32)
        for idx, col in enumerate(RAW_33_XYZ_COLS):
            if col in df.columns:
                coords_raw[:, idx] = df[col].fillna(0.0).values
    else:
        coords_raw = df[RAW_33_XYZ_COLS].fillna(0.0).values.astype(np.float32)

    # Reshape to (T, 33, 3)
    coords = coords_raw.reshape(total_frames, 33, 3)

    if center_hip:
        # LEFT_HIP is index 23, RIGHT_HIP is index 24 in MediaPipe 33
        hip_mid = (coords[:, 23:24, :] + coords[:, 24:25, :]) / 2.0  # (T, 1, 3)
        coords = coords - hip_mid

    half_seq = seq_len // 2
    windows = []

    if total_frames < seq_len:
        if total_frames >= half_seq:
            t_data = torch.from_numpy(coords).permute(1, 2, 0).unsqueeze(0)  # (1, 33, 3, T)
            t_data = t_data.view(1, 99, total_frames)
            t_stretched = F.interpolate(t_data, size=seq_len, mode="linear", align_corners=False)
            w = t_stretched.view(33, 3, seq_len).permute(2, 0, 1).numpy()  # (T=32, 33, 3)
            # Permute to (C=3, T=32, V=33, M=1)
            w_5d = np.transpose(w, (2, 0, 1))[:, :, :, np.newaxis]
            return [w_5d.astype(np.float32)]
        else:
            return []

    for start in range(0, total_frames, stride):
        end = start + seq_len
        if end <= total_frames:
            w = coords[start:end]  # (T, 33, 3)
            w_5d = np.transpose(w, (2, 0, 1))[:, :, :, np.newaxis]  # (3, T, 33, 1)
            windows.append(w_5d.astype(np.float32))
        else:
            partial = coords[start:]
            p_len = len(partial)
            if p_len >= half_seq:
                t_p = torch.from_numpy(partial).permute(1, 2, 0).unsqueeze(0).view(1, 99, p_len)
                t_str = F.interpolate(t_p, size=seq_len, mode="linear", align_corners=False)
                w = t_str.view(33, 3, seq_len).permute(2, 0, 1).numpy()
                w_5d = np.transpose(w, (2, 0, 1))[:, :, :, np.newaxis]
                windows.append(w_5d.astype(np.float32))
            break

    return windows


class BlockGCNDataset(Dataset):
    """
    In-memory Dataset for BlockGCN.
    Tensors stored as (N, C=3, T=32, V=33, M=1).
    """
    def __init__(
        self,
        samples: List[np.ndarray],
        labels: List[int],
        video_ids: List[str],
        in_memory: bool = True
    ):
        self.labels = labels
        self.video_ids = video_ids
        self.in_memory = in_memory

        if in_memory and len(samples) > 0:
            stacked = np.stack(samples, axis=0)  # (N, 3, 32, 33, 1)
            self.tensor_samples = torch.from_numpy(stacked).float()
            self.tensor_labels = torch.tensor(labels, dtype=torch.long)
            self.samples = None
        else:
            self.samples = samples
            self.tensor_samples = None
            self.tensor_labels = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.in_memory and self.tensor_samples is not None:
            return self.tensor_samples[idx], self.tensor_labels[idx]
        else:
            x = torch.from_numpy(self.samples[idx]).float()
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y


def get_blockgcn_dataloaders(
    metadata_path: str = "Final_dataset_metadata.csv",
    landmark_dir: str = "data/landmarks",
    batch_size: int = 64,
    seq_len: int = 32,
    train_stride: int = 16,
    val_test_stride: int = 32,
    smoke_test: bool = False,
    smoke_class: str = "barbell biceps curl",
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds Train, Val, Test DataLoaders for BlockGCN.
    """
    meta_df = pd.read_csv(metadata_path)
    landmark_base = Path(landmark_dir)

    loaders = {}
    for split in ["train", "val", "test"]:
        stride = train_stride if split == "train" else val_test_stride
        split_meta = meta_df[meta_df["split"] == split]

        if smoke_test:
            # Minimal subset for fast smoke test verification
            split_meta = split_meta[split_meta["class"] == smoke_class].head(2)

        all_samples = []
        all_labels = []
        all_vids = []

        for _, row in split_meta.iterrows():
            c_name = str(row["class"]).strip()
            c_idx = ACTION_TO_IDX.get(c_name, 0)
            f_path = str(row["filepath"])
            vid_name = Path(f_path).stem
            csv_path = landmark_base / split / c_name / f"{vid_name}.csv"

            if not csv_path.exists():
                # Search recursively in landmark_base / split
                matches = list((landmark_base / split).glob(f"**/{vid_name}.csv"))
                if matches:
                    csv_path = matches[0]
                else:
                    continue

            try:
                df_vid = pd.read_csv(csv_path)
            except Exception:
                continue

            seg_content = str(row.get("label", ""))
            segments = parse_segment_ranges(seg_content, len(df_vid))

            for s_start, s_end in segments:
                df_seg = df_vid.iloc[s_start:s_end].reset_index(drop=True)
                wins = extract_33j_windows(df_seg, seq_len=seq_len, stride=stride, center_hip=True)
                for w in wins:
                    all_samples.append(w)
                    all_labels.append(c_idx)
                    all_vids.append(vid_name)

        if len(all_samples) == 0 and not smoke_test:
            raise RuntimeError(f"No samples extracted for split '{split}'. Check landmark paths.")

        ds = BlockGCNDataset(
            samples=all_samples,
            labels=all_labels,
            video_ids=all_vids,
            in_memory=True
        )

        shuffle = (split == "train")
        loader_bs = min(batch_size, max(2, len(ds))) if smoke_test else batch_size
        loaders[split] = DataLoader(
            ds,
            batch_size=loader_bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train" and len(ds) > loader_bs)
        )

    return loaders["train"], loaders["val"], loaders["test"]
