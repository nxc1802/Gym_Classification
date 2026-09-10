"""
Landmark Sequence Augmentations.
Implements Jitter, Rotation, 3D Yaw Rotation, Joint Dropout, Time Warping, 
Bilateral Mirror (L/R flip), Speed Perturbation, and SkelGym-Aug directly 
on skeletal features. Supports both PyTorch Tensors and NumPy arrays.
"""

import math
from itertools import combinations
from typing import List
import numpy as np
import torch
import torch.nn.functional as F

class LandmarkAugmenter:
    """
    Applies biomechanically grounded data augmentations on landmark sequence tensors 
    of shape (T, Feat_Dim) or (B, T, Feat_Dim).
    """
    def __init__(
        self,
        jitter_sigma: float = 0.008,
        max_rotation_degrees: float = 10.0,
        max_yaw_degrees: float = 15.0,
        dropout_prob: float = 0.1,
        time_warp_factor: float = 0.15
    ):
        self.jitter_sigma = jitter_sigma
        self.max_rotation_degrees = max_rotation_degrees
        self.max_yaw_degrees = max_yaw_degrees
        self.dropout_prob = dropout_prob
        self.time_warp_factor = time_warp_factor

        # Precompute triplet indices for 13 MediaPipe joints for fast vectorized 3D angle recomputation
        triplets = list(combinations(range(13), 3))
        self.triplet_a = torch.tensor([t[0] for t in triplets], dtype=torch.long)
        self.triplet_b = torch.tensor([t[1] for t in triplets], dtype=torch.long)
        self.triplet_c = torch.tensor([t[2] for t in triplets], dtype=torch.long)

    def recompute_mix_angles(self, rel_3d: torch.Tensor) -> torch.Tensor:
        """
        Vectorized recomputation of 286 triplet 3D angles in [0, pi] from rel_3d coordinates.
        rel_3d shape: (..., 39) representing 13 joints x 3 (x, y, z).
        Returns: (..., 286)
        """
        orig_shape = rel_3d.shape
        flat_rel = rel_3d.reshape(-1, 13, 3)
        device = rel_3d.device
        
        a_idx = self.triplet_a.to(device)
        b_idx = self.triplet_b.to(device)
        c_idx = self.triplet_c.to(device)

        va = flat_rel[:, a_idx, :] - flat_rel[:, b_idx, :]  # (N, 286, 3)
        vb = flat_rel[:, c_idx, :] - flat_rel[:, b_idx, :]  # (N, 286, 3)

        dot = (va * vb).sum(dim=-1)
        norm = torch.norm(va, dim=-1) * torch.norm(vb, dim=-1) + 1e-7
        cos_theta = torch.clamp(dot / norm, -1.0, 1.0)
        angles = torch.acos(cos_theta)

        return angles.reshape(*orig_shape[:-1], 286)

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds zero-mean Gaussian noise to feature values.
        For mix features (325 dims), applies lighter noise to angles.
        """
        dim = x.shape[-1]
        if dim == 325:
            noise_rel = torch.randn_like(x[..., :39]) * self.jitter_sigma
            noise_ang = torch.randn_like(x[..., 39:]) * (self.jitter_sigma * 0.5)
            return torch.cat([x[..., :39] + noise_rel, x[..., 39:] + noise_ang], dim=-1)
        else:
            noise = torch.randn_like(x) * self.jitter_sigma
            return x + noise

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a random 2D in-plane rotation [-angle, +angle] to coordinate pairs (x, y).
        For mix representation (325 dims), rotates only the rel_3d coordinates and preserves angles.
        """
        angle = (torch.rand(1).item() * 2 - 1) * self.max_rotation_degrees
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x_rot = x.clone()
        dim = x.shape[-1]

        if dim == 325:
            # First 39 dims are rel_3d (13 joints * 3)
            for j in range(13):
                idx_x = j * 3
                idx_y = j * 3 + 1
                px = x[..., idx_x]
                py = x[..., idx_y]
                x_rot[..., idx_x] = px * cos_a - py * sin_a
                x_rot[..., idx_y] = px * sin_a + py * cos_a
            # Angles remain invariant under rigid rotation
            return x_rot

        stride = 4 if (dim % 4 == 0 or (dim - 1) % 4 == 0) else (3 if dim % 3 == 0 else 2)
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

    def yaw_rotate_3d(self, x: torch.Tensor, max_yaw_degrees: float = 15.0) -> torch.Tensor:
        """
        Applies a realistic 3D yaw rotation around the vertical Y-axis:
        x' = x * cos(theta) + z * sin(theta)
        y' = y
        z' = -x * sin(theta) + z * cos(theta)
        Simulates subject turning relative to the camera viewpoint.
        """
        angle = (torch.rand(1).item() * 2 - 1) * max_yaw_degrees
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x_rot = x.clone()
        dim = x.shape[-1]

        if dim == 325:
            # First 39 dims are rel_3d (stride 3: x, y, z)
            for j in range(13):
                idx_x = j * 3
                idx_z = j * 3 + 2
                px = x[..., idx_x]
                pz = x[..., idx_z]
                x_rot[..., idx_x] = px * cos_a + pz * sin_a
                x_rot[..., idx_z] = -px * sin_a + pz * cos_a
            return x_rot

        elif dim == 39 or dim % 3 == 0:
            stride = 3
            num_joints = dim // stride
            for j in range(num_joints):
                idx_x = j * stride
                idx_z = j * stride + 2
                px = x[..., idx_x]
                pz = x[..., idx_z]
                x_rot[..., idx_x] = px * cos_a + pz * sin_a
                x_rot[..., idx_z] = -px * sin_a + pz * cos_a
            return x_rot

        elif dim % 4 == 0:
            stride = 4
            num_joints = dim // stride
            for j in range(num_joints):
                idx_x = j * stride
                idx_z = j * stride + 2
                px = x[..., idx_x]
                pz = x[..., idx_z]
                x_rot[..., idx_x] = px * cos_a + pz * sin_a
                x_rot[..., idx_z] = -px * sin_a + pz * cos_a
            return x_rot

        else:
            # Fallback for 2D representations
            return self.rotate(x)

    def joint_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops (zeros out) 1 to 2 joints throughout the sequence to simulate occlusions.
        """
        x_drop = x.clone()
        dim = x.shape[-1]
        stride = 3 if dim == 39 else (4 if dim % 4 == 0 or (dim - 1) % 4 == 0 else 2)
        num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride

        if num_joints > 0 and dim != 325:
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
        Applies random uniform scaling to simulate subject distance / body size variations.
        For mix representation (325 dims), scales only coordinate features and preserves angles.
        """
        factor = torch.empty(1).uniform_(scale_min, scale_max).item()
        dim = x.shape[-1]
        if dim == 325:
            scaled_rel = x[..., :39] * factor
            return torch.cat([scaled_rel, x[..., 39:]], dim=-1)
        return x * factor

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        """
        Horizontal bilateral mirror (left/right flip) by swapping corresponding L/R joint features.
        Negates x coordinates and preserves anatomical consistency.
        For mix representation (325 dims), flips rel_3d and recomputes the 286 3D angles dynamically.

        Joint swap pairs for 13-joint layout (RAW_POINTS_13 order):
          idx 0: NOSE (no swap, x -> -x)
          idx 1 <-> 2: LEFT_SHOULDER <-> RIGHT_SHOULDER
          idx 3 <-> 4: LEFT_ELBOW <-> RIGHT_ELBOW
          idx 5 <-> 6: LEFT_WRIST <-> RIGHT_WRIST
          idx 7 <-> 8: LEFT_HIP <-> RIGHT_HIP
          idx 9 <-> 10: LEFT_KNEE <-> RIGHT_KNEE
          idx 11 <-> 12: LEFT_ANKLE <-> RIGHT_ANKLE
        """
        dim = x.shape[-1]
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]

        if dim == 325:
            # Extract and mirror rel_3d (first 39 dims, stride 3)
            rel = x[..., :39].clone()
            stride = 3
            num_joints = 13

            for j_left, j_right in swap_pairs:
                l_start = j_left * stride
                l_end = l_start + stride
                r_start = j_right * stride
                r_end = r_start + stride
                rel[..., l_start:l_end], rel[..., r_start:r_end] = (
                    x[..., r_start:r_end].clone(), x[..., l_start:l_end].clone()
                )

            for j in range(num_joints):
                rel[..., j * stride] = -rel[..., j * stride]

            # Recompute 286 triplet angles dynamically from mirrored 3D coordinates
            mirrored_angles = self.recompute_mix_angles(rel)
            return torch.cat([rel, mirrored_angles], dim=-1)

        x_mir = x.clone()
        if dim == 39 or dim % 3 == 0:
            stride = 3
        elif dim % 4 == 0 or (dim - 1) % 4 == 0:
            stride = 4
        else:
            stride = 2

        num_joints = dim // stride

        for j_left, j_right in swap_pairs:
            if j_left >= num_joints or j_right >= num_joints:
                continue
            l_start = j_left * stride
            l_end = l_start + stride
            r_start = j_right * stride
            r_end = r_start + stride
            x_mir[..., l_start:l_end], x_mir[..., r_start:r_end] = (
                x[..., r_start:r_end].clone(), x[..., l_start:l_end].clone()
            )

        for j in range(num_joints):
            idx_x = j * stride
            if idx_x < dim:
                x_mir[..., idx_x] = -x_mir[..., idx_x]

        return x_mir

    def speed_perturb(self, x: torch.Tensor, speed_min: float = 0.8, speed_max: float = 1.2) -> torch.Tensor:
        """
        Resamples the temporal axis at a random speed factor, then crops or pads
        back to the original sequence length T.
        """
        is_batched = (x.ndim == 3)
        if not is_batched:
            x = x.unsqueeze(0)

        B, T, D = x.shape
        speed = speed_min + torch.rand(1).item() * (speed_max - speed_min)
        new_T = max(8, int(T / speed))

        x_trans = x.permute(0, 2, 1)
        x_resampled = F.interpolate(x_trans, size=new_T, mode="linear", align_corners=False)

        if new_T >= T:
            start = (new_T - T) // 2
            x_out = x_resampled[:, :, start:start + T]
        else:
            pad_size = T - new_T
            x_out = F.pad(x_resampled, (0, pad_size), mode="replicate")

        out = x_out.permute(0, 2, 1)
        return out if is_batched else out.squeeze(0)

    def skel_gym_aug(self, x: torch.Tensor) -> torch.Tensor:
        """
        Proposed SOTA SkelGym-Aug Pipeline:
        1. Bilateral Mirroring (p=0.5)
        2. 3D Yaw Rotation (+-15 deg)
        3. Body / Scale adjustment (0.9 - 1.1)
        4. Temporal Time Warping (+-15%)
        5. Mild Sensor Jitter (sigma=0.008)
        """
        out = x.clone()
        if torch.rand(1).item() < 0.5:
            out = self.mirror(out)
        out = self.yaw_rotate_3d(out, max_yaw_degrees=self.max_yaw_degrees)
        out = self.scale(out, scale_min=0.9, scale_max=1.1)
        out = self.time_warp(out)
        out = self.jitter(out)
        return out

    def apply(self, x: torch.Tensor, method: str) -> torch.Tensor:
        """
        Applies a single augmentation method by name.
        """
        method = method.lower()
        if method == "jitter":
            return self.jitter(x)
        elif method == "rotate":
            return self.rotate(x)
        elif method == "yaw_rotate_3d":
            return self.yaw_rotate_3d(x)
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
        elif method == "skel_gym_aug":
            return self.skel_gym_aug(x)
        elif method == "none" or not method:
            return x
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

    def generate_augmented_variants(self, x: torch.Tensor, method: str = "skel_gym_aug") -> List[torch.Tensor]:
        """
        Generates 3 augmented variants from a single sample for dataset expansion (1->4 total).
        For 'skel_gym_aug':
          Variant 1: Bilateral Mirror (100% physically valid left-right symmetry)
          Variant 2: Spatial 3D Yaw Rotation + Body Scale + Mild Jitter
          Variant 3: Bilateral Mirror + Temporal Time-Warp + 3D Yaw Rotation
        """
        variants = []
        if method.lower() == "skel_gym_aug":
            # Variant 1: Bilateral mirror (true ergonomic flip)
            variants.append(self.mirror(x.clone()))

            # Variant 2: 3D spatial transformation (viewpoint yaw + scale + jitter)
            v2 = self.yaw_rotate_3d(x.clone(), max_yaw_degrees=self.max_yaw_degrees)
            v2 = self.scale(v2, scale_min=0.9, scale_max=1.1)
            v2 = self.jitter(v2)
            variants.append(v2)

            # Variant 3: Bilateral mirror combined with temporal time-warp and yaw
            v3 = self.mirror(x.clone())
            v3 = self.time_warp(v3)
            v3 = self.yaw_rotate_3d(v3, max_yaw_degrees=self.max_yaw_degrees)
            variants.append(v3)
        else:
            # Fallback single method variants
            variants.append(self.apply(x.clone(), method if method != "none" else "skel_gym_aug"))
            variants.append(self.mirror(x.clone()))
            mirrored = self.mirror(x.clone())
            variants.append(self.apply(mirrored, method if method != "none" else "skel_gym_aug"))

        return variants
