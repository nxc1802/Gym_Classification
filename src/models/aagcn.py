"""
Two-Stream Adaptive Spatial-Temporal Graph Convolutional Network (2s-AAGCN).
Faithfully follows Shi et al. (CVPR 2019):
- Adaptive Graph Convolution with 3 components:
    A_adaptive = (A_topo * M) + B + C(X)
    1. A_topo: Anatomical skeletal topology from MediaPipe.
    2. M: Learnable edge importance weight mask.
    3. B: Global learnable dependency matrix between all joint pairs.
    4. C(X): Sample-dependent dynamic self-attention graph matrix.
- Temporal Convolution (TCN) with multi-head spatial-temporal feature capture.
- Calibrated to fair ~350K parameter budget.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import POSE_CONNECTIONS_33, EDGES_13

class AdaptiveGraphConv(nn.Module):
    """
    Adaptive Graph Convolution layer:
    A_adaptive = (A_topo * M) + B + C(X)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_norm: torch.Tensor,
        embed_channels: int = 16
    ):
        super().__init__()
        V = A_norm.size(0)
        self.register_buffer("A_norm", A_norm)
        
        # 1. Learnable edge importance mask
        self.edge_mask = nn.Parameter(torch.ones(V, V))
        
        # 2. Global learnable adjacency matrix
        self.B = nn.Parameter(torch.zeros(V, V))
        nn.init.normal_(self.B, std=1e-4)
        
        # 3. Sample-dependent attention projections
        self.embed_channels = embed_channels
        self.conv_theta = nn.Conv2d(in_channels, embed_channels, 1, bias=False)
        self.conv_phi = nn.Conv2d(in_channels, embed_channels, 1, bias=False)
        
        # 4. Feature projection convolution
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, V)
        B, C, T, V = x.shape
        
        # Dynamic sample-dependent matrix C(X)
        # Pool across temporal dimension to obtain joint representations
        theta = self.conv_theta(x).mean(dim=2)  # (B, Ce, V)
        phi = self.conv_phi(x).mean(dim=2)      # (B, Ce, V)
        
        # Self-attention dot product across joints
        # (B, V, Ce) @ (B, Ce, V) -> (B, V, V)
        theta_t = theta.permute(0, 2, 1)
        sim = torch.bmm(theta_t, phi) / np.sqrt(self.embed_channels)
        C_dyn = torch.softmax(sim, dim=-1)  # (B, V, V)
        
        # Combine topology, global learnable, and sample-dependent graphs
        # A_static: (V, V), broadcasted to (B, V, V)
        A_static = (self.A_norm * self.edge_mask) + self.B
        A_total = A_static.unsqueeze(0) + C_dyn  # (B, V, V)
        
        # Spatial graph message passing: (B, V, V) x (B, C, T, V) -> (B, C, T, V)
        x_g = torch.einsum("bvw,bctw->bctv", A_total, x)
        return self.act(self.bn(self.conv(x_g)))

class AAGCNBlock(nn.Module):
    """
    AAGCN Spatial-Temporal Block:
    AdaptiveGraphConv -> TCN -> Residual Connection -> GELU
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A_norm: torch.Tensor,
        stride: int = 1,
        temporal_kernel: int = 9,
        dropout: float = 0.2
    ):
        super().__init__()
        # Spatial GCN
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A_norm)
        
        # Temporal TCN
        padding = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
        # Residual shortcut
        if in_channels != out_channels or stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()
            
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res(x)
        x_g = self.gcn(x)
        x_t = self.tcn(x_g)
        return self.act(x_t + res)

class AAGCNModel(nn.Module):
    """
    Calibrated Adaptive Graph Convolutional Network (~350K parameters).
    Compatible with both Joint coordinates (rel_3d, raw_3d) and Bone vectors (bone_3d).
    Input shape: (B, T, D) where D = num_joints * channels_per_joint.
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

        # Data BatchNorm to normalize coordinate/bone scales across all joints and time
        self.data_bn = nn.BatchNorm2d(C)

        # Calibrated 3-stage architecture: channels [48, 96, 150] yields ~355,000 parameters
        channels = [48, 96, 150]
        strides = [1, 1, 2]
        self.blocks = nn.ModuleList()
        in_c = C
        for out_c, s in zip(channels, strides):
            self.blocks.append(
                AAGCNBlock(
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
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.GELU(),
            nn.LayerNorm(channels[-1]),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes)
        )

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

        A = np.eye(V, dtype=np.float32)  # Include self-loops
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
