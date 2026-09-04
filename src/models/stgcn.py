"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Gym Exercise Skeleton Sequences.
Constructs spatial adjacency from MediaPipe skeleton topology with symmetric normalization:
A_norm = D^(-1/2) A D^(-1/2).
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import POSE_CONNECTIONS_33, EDGES_13

class STGCNBlock(nn.Module):
    """
    ST-GCN unit consisting of Spatial Graph Convolution + Multi-scale Temporal Convolution
    with residual connection and BatchNorm.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_norm: torch.Tensor,
        stride: int = 1,
        kernel_sizes: Tuple[int, ...] = (9, 5, 3),
        dropout: float = 0.1
    ):
        super().__init__()
        V = A_norm.size(0)
        self.register_buffer("A_norm", A_norm)
        # Learnable edge weight matrix B
        self.B = nn.Parameter(torch.zeros(V, V))
        nn.init.uniform_(self.B, -1e-4, 1e-4)

        # 1x1 Conv for feature dimension transition
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # Multi-scale temporal convolution branches along time dimension T
        self.temporal_branches = nn.ModuleList([
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(k, 1),
                stride=(stride, 1),
                padding=(k // 2, 0),
                bias=False
            )
            for k in kernel_sizes
        ])

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

        if in_channels != out_channels or stride != 1:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False)
            self.res_bn = nn.BatchNorm2d(out_channels)
        else:
            self.res_conv = None
            self.res_bn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, V)
        A_adj = self.A_norm + torch.softmax(self.B, dim=1)
        # Spatial graph propagation: sum_j A_ij * x_j
        x_graph = torch.einsum("ij,bctj->bcti", A_adj, x)
        x_graph = self.conv1x1(x_graph)

        # Multi-scale temporal aggregation
        x_temp = sum(branch(x_graph) for branch in self.temporal_branches) / len(self.temporal_branches)
        x_temp = self.bn(x_temp)

        # Residual connection
        res = self.res_bn(self.res_conv(x)) if self.res_conv else x
        return self.dropout(self.act(x_temp + res))

class STGCNModel(nn.Module):
    """
    ST-GCN model for skeletal action recognition.
    Accepts input sequence (B, T, Feat_Dim), automatically decomposes it into (B, C, T, V),
    passes through 4 ST-GCN stages, and applies Global Average Pooling.
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        num_joints: Optional[int] = None,
        dropout: float = 0.3
    ):
        super().__init__()
        if num_joints is None:
            if feat_dim % 33 == 0 or (feat_dim - 1) % 32 == 0:
                num_joints = 33
            elif feat_dim % 13 == 0 or (feat_dim - 1) % 12 == 0:
                num_joints = 13
            else:
                num_joints = 13

        self.num_joints = num_joints
        # Calculate channels per joint C
        if feat_dim % num_joints == 0:
            self.c_per_joint = feat_dim // num_joints
        elif (feat_dim - 1) % (num_joints - 1) == 0:
            # relative coordinates case e.g. 129 -> 32*4 + 1
            self.c_per_joint = (feat_dim - 1) // (num_joints - 1)
        else:
            self.c_per_joint = max(2, feat_dim // num_joints)

        # Build symmetrically normalized adjacency matrix D^(-1/2) A D^(-1/2)
        A_norm = self._build_adjacency_matrix(self.num_joints)
        C = self.c_per_joint

        self.block1 = STGCNBlock(C, 64, A_norm, stride=1)
        self.block2 = STGCNBlock(64, 64, A_norm, stride=1)
        self.block3 = STGCNBlock(64, 128, A_norm, stride=2)
        self.block4 = STGCNBlock(128, 256, A_norm, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    @classmethod
    def _build_adjacency_matrix(cls, V: int) -> torch.Tensor:
        """
        Builds symmetric normalized adjacency matrix: A_norm = D^(-1/2) (A + I) D^(-1/2)
        """
        edges = POSE_CONNECTIONS_33 if V >= 33 else EDGES_13
        A = np.eye(V, dtype=np.float32)  # Include self-connections
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
            # Pad if dimension is slightly lower
            pad = torch.zeros(B, T, expected_len - D, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, pad], dim=-1)
            x_joints = x_padded.reshape(B, T, V, C)

        # Permute to (B, C, T, V)
        x_in = x_joints.permute(0, 3, 1, 2).contiguous()

        h = self.block1(x_in)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)

        pooled = self.gap(h).flatten(1)  # (B, 256)
        return self.fc(self.drop(pooled))
