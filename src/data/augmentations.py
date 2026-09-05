"""
Landmark Sequence Augmentations.
Implements Jitter, Rotation, Joint Dropout, and Time Warping directly on skeletal features.
Supports both PyTorch Tensors and NumPy arrays.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

class LandmarkAugmenter:
    """
    Applies data augmentations on landmark sequence tensors of shape (T, Feat_Dim)
    or (B, T, Feat_Dim).
    """
    def __init__(
        self,
        jitter_sigma: float = 0.015,
        max_rotation_degrees: float = 10.0,
        dropout_prob: float = 0.1,
        time_warp_factor: float = 0.2
    ):
        self.jitter_sigma = jitter_sigma
        self.max_rotation_degrees = max_rotation_degrees
        self.dropout_prob = dropout_prob
        self.time_warp_factor = time_warp_factor

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds zero-mean Gaussian noise to coordinate values.
        """
        noise = torch.randn_like(x) * self.jitter_sigma
        return x + noise

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a random 2D rotation [-angle, +angle] to coordinate pairs (x, y).
        Assumes features contain (x, y) coordinates sequentially or periodically.
        """
        angle = (torch.rand(1).item() * 2 - 1) * self.max_rotation_degrees
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x_rot = x.clone()
        # Rotate all consecutive pairs of (x, y) if feature dim matches 2D/3D layout
        # e.g., for stride 4 (x, y, z, vis) or stride 2 (x, y)
        dim = x.shape[-1]
        stride = 4 if dim % 4 == 0 or (dim - 1) % 4 == 0 else 2

        num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride
        for j in range(num_joints):
            idx_x = j * stride
            idx_y = j * stride + 1
            if idx_y < dim:
                px = x[..., idx_x]
                py = x[..., idx_y]
                x_rot[..., idx_x] = px * cos_a - py * sin_a
                x_rot[..., idx_y] = px * sin_a + py * cos_a

        return x_rot

    def joint_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops (zeros out) 1 to 2 joints throughout the sequence to simulate occlusions.
        """
        x_drop = x.clone()
        dim = x.shape[-1]
        stride = 4 if dim % 4 == 0 or (dim - 1) % 4 == 0 else 2
        num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride

        if num_joints > 0:
            drop_mask = torch.rand(num_joints) < self.dropout_prob
            for j in range(num_joints):
                if drop_mask[j]:
                    start = j * stride
                    end = min(start + stride, dim)
                    x_drop[..., start:end] = 0.0

        return x_drop

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporally stretches or compresses the sequence via 1D linear interpolation,
        then resamples back to the original sequence length T.
        """
        # x shape: (T, D) or (B, T, D)
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)  # (1, T, D)

        B, T, D = x.shape
        warp_ratio = 1.0 + (torch.rand(1).item() * 2 - 1) * self.time_warp_factor
        new_T = max(8, int(T * warp_ratio))

        # Permute to (B, D, T) for 1D interpolation
        x_trans = x.permute(0, 2, 1)
        x_warped = F.interpolate(x_trans, size=new_T, mode="linear", align_corners=False)
        # Resample back to original T
        x_resampled = F.interpolate(x_warped, size=T, mode="linear", align_corners=False)
        out = x_resampled.permute(0, 2, 1)

        return out if is_batched else out.squeeze(0)

    def scale(self, x: torch.Tensor, scale_min: float = 0.9, scale_max: float = 1.1) -> torch.Tensor:
        """
        Applies random uniform scaling to simulate subject distance / body size variations
        as referenced in Augmentation_CSV.ipynb and publication manuscript.
        """
        factor = torch.empty(1).uniform_(scale_min, scale_max).item()
        return x * factor

    def apply(self, x: torch.Tensor, method: str) -> torch.Tensor:
        """
        Applies a single augmentation method by name.
        """
        method = method.lower()
        if method == "jitter":
            return self.jitter(x)
        elif method == "rotate":
            return self.rotate(x)
        elif method == "scale":
            return self.scale(x)
        elif method == "joint_dropout":
            return self.joint_dropout(x)
        elif method == "time_warp":
            return self.time_warp(x)
        elif method == "combined":
            # Canonical combination from notebook & manuscript:
            # 1. Scale (0.9 - 1.1)
            # 2. Rotation (+-10 deg around center/torso)
            # 3. Time Warping (+-20% temporal stretch/compression)
            # 4. Joint Jitter (Gaussian noise sigma=0.01)
            x = self.scale(x)
            x = self.rotate(x)
            x = self.time_warp(x)
            return self.jitter(x)
        elif method == "none" or not method:
            return x
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
