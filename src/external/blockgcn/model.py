"""
BlockGCN: Redefine Topology Awareness for Skeleton-Based Action Recognition (CVPR 2024).
Faithfully adapted for SkelGym:
  - V = 33 MediaPipe joints (instead of NTU 25)
  - M = 1 person (removed hardcoded repeat(2, 1))
  - C = 22 gym exercise classes
  - T = 32 temporal window
  - Retains all core architectural components: BlockGC, Multi-scale TCN, TopoTrans, Hop-distance RPE.
"""

import math
import warnings
from typing import Optional, Tuple, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.external.blockgcn.graph import MediaPipe33Graph

# Optional upstream dependency: torch_topological
try:
    from torch_topological.nn.data import make_tensor
    from torch_topological.nn import VietorisRipsComplex
    from torch_topological.nn.layers import StructureElementLayer
    HAS_TORCH_TOPOLOGICAL = True
except ImportError:
    HAS_TORCH_TOPOLOGICAL = False


def conv_init(conv: nn.Module):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None and isinstance(conv.bias, torch.Tensor):
        nn.init.constant_(conv.bias, 0)


def bn_init(bn: nn.Module, scale: float):
    if bn.weight is not None:
        nn.init.constant_(bn.weight, scale)
    if bn.bias is not None:
        nn.init.constant_(bn.bias, 0)


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if hasattr(m, "bias") and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: Optional[List[int]] = None,
        residual: bool = False,
        residual_kernel_size: int = 1
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2]
        assert out_channels % (len(dilations) + 2) == 0, "# out channels should be multiples of # branches"

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
            )
            for dilation in dilations
        ])

        # MaxPool branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        # 1x1 conv branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1), bias=False),
            nn.BatchNorm2d(branch_channels)
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        branch_outs = [b(x) for b in self.branches]
        out = torch.cat(branch_outs, dim=1) + res
        return out


class unit_gcn(nn.Module):
    """
    BlockGC (Block Graph Convolution):
    Partitions channels into groups (heads), computes group-wise spatial projection,
    and incorporates hop-distance relative positional encoding (RPE).
    """
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, hops: np.ndarray):
        super().__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1
        num_node = A.shape[-1]

        # Learnable topology refinement weights for 3 subsets
        self.fc1 = nn.Parameter(
            torch.stack([
                torch.stack([torch.eye(num_node, dtype=torch.float32) for _ in range(self.num_heads)], dim=0)
                for _ in range(3)
            ], dim=0),
            requires_grad=True
        )

        self.fc2 = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads, bias=False)
            for _ in range(3)
        ])

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Register hop distance matrix
        self.register_buffer("hops", torch.from_numpy(hops).long())
        max_hop = int(hops.max())
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, max_hop + 1), dtype=torch.float32))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def L2_norm(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        y = None

        # Hop distance positional encoding
        pos_emb = self.rpe[:, :, self.hops]  # (3, num_heads, V, V)

        for i in range(3):
            w1 = self.fc1[i]
            weight_norm = self.L2_norm(w1)
            w1_norm = w1 / weight_norm

            pos_norm = pos_emb[i] / self.L2_norm(pos_emb[i])
            w_total = w1_norm + pos_norm

            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z = torch.einsum("nhctv, hvw -> nhctw", x_in, w_total).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)

            y = z + y if y is not None else z

        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: np.ndarray, hops: np.ndarray, stride: int = 1):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, hops)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, stride=stride, residual=False)
        self.relu = nn.ReLU(inplace=True)

        if (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))


class TopoTrans(nn.Module):
    """
    Topology Feature Projection Module:
    Adapts single-person (M=1) representation without the hard-coded repeat(2, 1).
    """
    def __init__(self, out_dim: int, in_dim: int = 64):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, 64)
        h = self.mlp(x)
        h = self.bn(h)
        h = self.relu(h)
        return h.unsqueeze(2).unsqueeze(3)  # (N, out_dim, 1, 1)


class Topo(nn.Module):
    """
    Persistent Homology Topological Feature Extractor.
    Uses torch_topological when available (e.g. on Linux server with Python <= 3.12).
    Falls back gracefully to pure-PyTorch structural distance layer for local testing / Python 3.13.
    """
    def __init__(self, num_nodes: int = 33, out_dim: int = 64):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim

        if HAS_TORCH_TOPOLOGICAL:
            self.vr = VietorisRipsComplex(dim=0)
            self.pl = StructureElementLayer(n_elements=out_dim)
        else:
            warnings.warn(
                "torch_topological is NOT installed (requires Python <=3.12 + giotto-ph). "
                "The Topo branch is using a DEGRADED nn.Linear proxy instead of real persistent homology. "
                "This is acceptable for local smoke testing ONLY. For production training on the server, "
                "ensure torch_topological is installed to retain the paper's full architecture.",
                RuntimeWarning,
                stacklevel=2
            )
            # Degraded proxy: learns a projection from pairwise distance summary instead of
            # VietorisRipsComplex + StructureElementLayer. NOT equivalent to persistent homology.
            self.proj = nn.Sequential(
                nn.Linear(num_nodes, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim)
            )

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        # joint shape: (N, M, C, T, V)
        N, M, C, T, V = joint.size()
        x = joint.mean(dim=1)  # (N, C, T, V)
        # Pairwise distance across joints
        x_diff = x.unsqueeze(-1) - x.unsqueeze(-2)  # (N, C, T, V, V)
        dist = torch.norm(x_diff, p=2, dim=1).mean(dim=1)  # (N, V, V)

        # Min-max scaling
        d_min = dist.amin(dim=(1, 2), keepdim=True)
        d_max = dist.amax(dim=(1, 2), keepdim=True) + 1e-6
        dist_norm = (dist - d_min) / (d_max - d_min)

        if HAS_TORCH_TOPOLOGICAL:
            vr_res = self.vr(dist_norm)
            tensor_res = make_tensor(vr_res)
            return self.pl(tensor_res)
        else:
            # Pool across one dimension and project
            feat = dist_norm.mean(dim=-1)  # (N, V)
            return self.proj(feat)  # (N, 64)


class BlockGCNModel(nn.Module):
    """
    Full BlockGCN Model (CVPR 2024) adapted for SkelGym:
      - 33 MediaPipe Joints
      - 1 Person
      - 22 Action Classes
      - 10 GCN-TCN blocks
    """
    def __init__(
        self,
        num_class: int = 22,
        num_point: int = 33,
        num_person: int = 1,
        in_channels: int = 3,
        drop_out: float = 0.0,
        window_size: int = 32,
        graph: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person
        self.window_size = window_size

        # Graph initialization
        self.graph_wrapper = MediaPipe33Graph()
        A = self.graph_wrapper.A        # (3, 33, 33)
        hops = self.graph_wrapper.hops  # (33, 33)

        self.to_joint_embedding = nn.Linear(in_channels, 128)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, 128) * 0.02)
        self.data_bn = nn.BatchNorm1d(num_person * 128 * num_point)

        # 10 GCN-TCN Blocks (Channels: 128 -> 128 -> 128 -> 128 -> 256 -> 256 -> 256 -> 256 -> 256 -> 256)
        self.l1 = TCN_GCN_unit(128, 128, A, hops)
        self.l2 = TCN_GCN_unit(128, 128, A, hops)
        self.l3 = TCN_GCN_unit(128, 128, A, hops)
        self.l4 = TCN_GCN_unit(128, 128, A, hops)
        self.l5 = TCN_GCN_unit(128, 256, A, hops, stride=2)
        self.l6 = TCN_GCN_unit(256, 256, A, hops)
        self.l7 = TCN_GCN_unit(256, 256, A, hops)
        self.l8 = TCN_GCN_unit(256, 256, A, hops, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, hops)
        self.l10 = TCN_GCN_unit(256, 256, A, hops)

        # 10 TopoTrans Modules
        self.t0 = TopoTrans(out_dim=128)
        self.t1 = TopoTrans(out_dim=128)
        self.t2 = TopoTrans(out_dim=128)
        self.t3 = TopoTrans(out_dim=128)
        self.t4 = TopoTrans(out_dim=128)
        self.t5 = TopoTrans(out_dim=256)
        self.t6 = TopoTrans(out_dim=256)
        self.t7 = TopoTrans(out_dim=256)
        self.t8 = TopoTrans(out_dim=256)
        self.t9 = TopoTrans(out_dim=256)

        self.topo = Topo(num_nodes=num_point)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1.0)

        self.drop_out = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        joint: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass accepting:
          - x: (B, C=3, T=32, V=33, M=1) or (B, T=32, D=99)
        """
        if x.ndim == 3:
            # Auto-reshape if passed as (B, T, V*C)
            B, T, D = x.shape
            V = self.num_point
            C = D // V
            x = x.view(B, T, V, C).permute(0, 3, 1, 2).unsqueeze(-1)  # (B, C, T, V, 1)

        N, C, T, V, M = x.size()
        if joint is None:
            joint = x

        # Topological branch
        # a: (N, 64)
        j_rearr = rearrange(joint, "n c t v m -> n m c t v", m=M, v=V).contiguous()
        a = self.topo(j_rearr)

        # Coordinate embedding + Positional Encoding
        h = rearrange(x, "n c t v m -> (n m t) v c", m=M, v=V).contiguous()
        h = self.to_joint_embedding(h)
        h = h + self.pos_embedding[:, :self.num_point]
        h = rearrange(h, "(n m t) v c -> n (m v c) t", m=M, t=T).contiguous()

        h = self.data_bn(h)
        h = h.view(N, M, V, 128, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, 128, T, V)

        # 10 Stages with topological injection
        h = self.l1(h + self.t0(a))
        h = self.l2(h + self.t1(a))
        h = self.l3(h + self.t2(a))
        h = self.l4(h + self.t3(a))
        h = self.l5(h + self.t4(a))
        h = self.l6(h + self.t5(a))
        h = self.l7(h + self.t6(a))
        h = self.l8(h + self.t7(a))
        h = self.l9(h + self.t8(a))
        h = self.l10(h + self.t9(a))

        # Global average pooling
        c_new = h.size(1)
        h = h.view(N, M, c_new, -1).mean(3).mean(1)
        h = self.drop_out(h)
        out = self.fc(h)

        if y is not None:
            return out, y
        return out
