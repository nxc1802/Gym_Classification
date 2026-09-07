"""
Landmark Sequence Augmentations.
Implements Jitter, Rotation, Joint Dropout, Time Warping, Mirror (L/R flip), and Speed Perturbation
directly on skeletal features. Supports both PyTorch Tensors and NumPy arrays.
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

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        """
        Horizontal mirror (left/right flip) by swapping corresponding L/R joint features.
        For coordinate-based features, also negates the x-axis to create a true mirror.
        Works for feature layouts with stride 2 (x,y), 3 (x,y,z), or 4 (x,y,z,vis).

        Joint swap pairs for 13-joint layout (RAW_POINTS_13 order):
          idx 0: NOSE (no swap)
          idx 1 <-> 2: LEFT_SHOULDER <-> RIGHT_SHOULDER
          idx 3 <-> 4: LEFT_ELBOW <-> RIGHT_ELBOW
          idx 5 <-> 6: LEFT_WRIST <-> RIGHT_WRIST
          idx 7 <-> 8: LEFT_HIP <-> RIGHT_HIP
          idx 9 <-> 10: LEFT_KNEE <-> RIGHT_KNEE
          idx 11 <-> 12: LEFT_ANKLE <-> RIGHT_ANKLE
        """
        x_mir = x.clone()
        dim = x.shape[-1]

        # Determine stride based on dimension
        if dim % 4 == 0 or (dim - 1) % 4 == 0:
            stride = 4
        elif dim % 3 == 0:
            stride = 3
        else:
            stride = 2

        num_joints = dim // stride

        # Define swap pairs by joint index (L/R pairs in 13-joint layout)
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]

        for j_left, j_right in swap_pairs:
            if j_left >= num_joints or j_right >= num_joints:
                continue
            l_start = j_left * stride
            l_end = l_start + stride
            r_start = j_right * stride
            r_end = r_start + stride
            # Swap features of left and right joints
            x_mir[..., l_start:l_end], x_mir[..., r_start:r_end] = (
                x[..., r_start:r_end].clone(), x[..., l_start:l_end].clone()
            )

        # Negate x-coordinates (first in each stride block) to create true horizontal mirror
        for j in range(num_joints):
            idx_x = j * stride
            if idx_x < dim:
                x_mir[..., idx_x] = -x_mir[..., idx_x]

        return x_mir

    def speed_perturb(self, x: torch.Tensor, speed_min: float = 0.8, speed_max: float = 1.2) -> torch.Tensor:
        """
        Resamples the temporal axis at a random speed factor, then crops or pads
        back to the original sequence length T. Unlike time_warp which stretches
        and resamples back, this simulates actual speed variation.
        """
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)  # (1, T, D)

        B, T, D = x.shape
        speed = speed_min + torch.rand(1).item() * (speed_max - speed_min)
        new_T = max(8, int(T / speed))  # Slower speed = more frames, faster = fewer

        # Resample to new_T
        x_trans = x.permute(0, 2, 1)  # (B, D, T)
        x_resampled = F.interpolate(x_trans, size=new_T, mode="linear", align_corners=False)

        # Crop or pad back to original T
        if new_T >= T:
            # Crop: take center portion
            start = (new_T - T) // 2
            x_out = x_resampled[:, :, start:start + T]
        else:
            # Pad: repeat-pad the end
            pad_size = T - new_T
            x_out = F.pad(x_resampled, (0, pad_size), mode="replicate")

        out = x_out.permute(0, 2, 1)  # (B, T, D)
        return out if is_batched else out.squeeze(0)

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
        elif method == "mirror":
            return self.mirror(x)
        elif method == "speed_perturb":
            return self.speed_perturb(x)
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

    def generate_augmented_variants(self, x: torch.Tensor, method: str = "combined") -> list:
        """
        Generates 3 augmented variants from a single sample for dataset expansion (1→4 total).
        Variant 1: Combined augmentation (scale + rotate + time_warp + jitter)
        Variant 2: Mirror (horizontal flip)
        Variant 3: Mirror + combined augmentation
        """
        variants = []
        # Variant 1: combined augmentation pipeline
        variants.append(self.apply(x.clone(), method if method != "none" else "combined"))
        # Variant 2: mirror only
        variants.append(self.mirror(x.clone()))
        # Variant 3: mirror + combined
        mirrored = self.mirror(x.clone())
        variants.append(self.apply(mirrored, method if method != "none" else "combined"))
        return variants
