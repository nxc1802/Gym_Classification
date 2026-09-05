"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Gym Exercise Skeleton Sequences.
Faithfully follows the canonical architecture by Yan et al. (AAAI 2018):
- Input BatchNorm2d to standardize raw spatial coordinates across batches and time.
- Spatial Graph Convolution: A_norm * M (learnable edge importance mask) + Conv2d + BatchNorm2d + ReLU.
- Temporal Convolution (TCN): Conv2d along time dimension + BatchNorm2d + Dropout.
- Residual shortcut: Conv2d + BatchNorm2d when channels/strides change.
- Calibrated ~350K parameter budget.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import POSE_CONNECTIONS_33, EDGES_13

class STGCNBlock(nn.Module):
    """
    Standard ST-GCN Block (Yan et al. 2018):
    Spatial Graph Conv (with learned edge mask) -> BN -> ReLU -> Temporal Conv -> BN -> Dropout -> Residual Sum -> ReLU.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_norm: torch.Tensor,
        stride: int = 1,
        temporal_kernel: int = 9,
        dropout: float = 0.1
    ):
        super().__init__()
        V = A_norm.size(0)
        self.register_buffer("A_norm", A_norm)
        # Learnable edge importance weighting (initialized to 1.0 to preserve graph structure)
        self.edge_mask = nn.Parameter(torch.ones(V, V))

        # Spatial Graph Convolution
        self.gcn_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.gcn_bn = nn.BatchNorm2d(out_channels)
        self.gcn_act = nn.ReLU(inplace=True)

        # Temporal Convolution (TCN)
        padding = (temporal_kernel - 1) // 2
        self.tcn_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias=False
        )
        self.tcn_bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual shortcut
        if in_channels != out_channels or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, V)
        res = self.res(x)

        # 1. Spatial GCN with edge importance weighting
        A = self.A_norm * self.edge_mask
        x_g = torch.einsum("vw,bctw->bctv", A, x)
        x_g = self.gcn_act(self.gcn_bn(self.gcn_conv(x_g)))

        # 2. Temporal TCN
        x_t = self.dropout(self.tcn_bn(self.tcn_conv(x_g)))

        # 3. Residual connection + activation
        return self.act(x_t + res)

class STGCNModel(nn.Module):
    """
    Calibrated ST-GCN Classifier for Gym Exercise Recognition (~350K parameters).
    Accepts (B, T, D) and reshapes to canonical (B, C, T, V).
    Applies data BatchNorm2d, passes through 3 ST-GCN blocks, and classifies via GAP + Linear.
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_joints: Optional[int] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        if num_joints is None:
            if feat_dim in (24, 36, 48):
                num_joints = 12
            elif feat_dim in (26, 39, 52):
                num_joints = 13
            elif feat_dim in (64, 96, 128):
                num_joints = 32
            elif feat_dim in (66, 99, 132):
                num_joints = 33
            elif feat_dim % 13 == 0:
                num_joints = 13
            elif feat_dim % 12 == 0:
                num_joints = 12
            else:
                num_joints = 13

        self.num_joints = num_joints
        self.c_per_joint = max(2, feat_dim // num_joints)
        C = self.c_per_joint

        # Build symmetrically normalized adjacency matrix D^(-1/2) (A + I) D^(-1/2)
        A_norm = self._build_adjacency_matrix(self.num_joints)

        # Data BatchNorm to normalize coordinate scales across all joints and time
        self.data_bn = nn.BatchNorm2d(C)

        # Calibrated 3-stage architecture: channels [48, 96, 150] yields ~350,000 parameters
        channels = [48, 96, 150]
        strides = [1, 1, 2]
        self.blocks = nn.ModuleList()
        in_c = C
        for out_c, s in zip(channels, strides):
            self.blocks.append(
                STGCNBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    A_norm=A_norm,
                    stride=s,
                    temporal_kernel=9,
                    dropout=dropout
                )
            )
            in_c = out_c

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    @classmethod
    def _build_adjacency_matrix(cls, V: int) -> torch.Tensor:
        """
        Builds symmetric normalized adjacency matrix: A_norm = D^(-1/2) (A + I) D^(-1/2)
        """
        if V >= 33:
            edges = POSE_CONNECTIONS_33
        elif V == 32:
            edges = [(i - 1, j - 1) for i, j in POSE_CONNECTIONS_33 if i > 0 and j > 0 and i - 1 < 32 and j - 1 < 32]
        elif V == 12:
            edges = [
                (0, 1), (0, 2), (2, 4), (1, 3), (3, 5),
                (0, 6), (1, 7), (6, 7), (6, 8), (8, 10),
                (7, 9), (9, 11)
            ]
        else:
            edges = EDGES_13

        A = np.eye(V, dtype=np.float32)  # Include self-connections (loops)
        for i, j in edges:
            if i < V and j < V:
                A[i, j] = 1.0
                A[j, i] = 1.0

        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-7))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return torch.from_numpy(A_norm).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        B, T, D = x.shape
        V = self.num_joints
        C = self.c_per_joint
        expected_len = V * C

        if D >= expected_len:
            x_joints = x[:, :, :expected_len].reshape(B, T, V, C)
        else:
            pad = torch.zeros(B, T, expected_len - D, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=-1)
            x_joints = x_padded.reshape(B, T, V, C)

        # Permute to canonical ST-GCN format (B, C, T, V)
        x_in = x_joints.permute(0, 3, 1, 2).contiguous()
        h = self.data_bn(x_in)

        for block in self.blocks:
            h = block(h)

        pooled = self.gap(h).flatten(1)  # (B, 150)
        return self.fc(pooled)
