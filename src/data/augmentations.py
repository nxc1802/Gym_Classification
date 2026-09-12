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

        # Precompute bijective symmetric permutations for left-right skeletal mirroring
        SWAP_MAP = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11}
        pairs = list(combinations(range(13), 2))
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        pair_sym = [pair_to_idx[(min(SWAP_MAP[a], SWAP_MAP[b]), max(SWAP_MAP[a], SWAP_MAP[b]))] for a, b in pairs]
        self.pair_sym_indices = torch.tensor(pair_sym, dtype=torch.long)

        triplet_to_idx = {t: i for i, t in enumerate(triplets)}
        triplet_sym = [triplet_to_idx[tuple(sorted([SWAP_MAP[a], SWAP_MAP[b], SWAP_MAP[c]]))] for a, b, c in triplets]
        self.triplet_sym_indices = torch.tensor(triplet_sym, dtype=torch.long)

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
        Adds zero-mean Gaussian noise to feature values while strictly preserving structure.
        - mix (325 dims): coordinates receive jitter_sigma, angles receive jitter_sigma * 0.5.
        - triplet angles (286 dims) / pair angles (78 dims): light noise bounded in valid range.
        - 4-channel coordinates (52, 53, 132, 133): noise applied only to (x, y, z), visibility preserved.
        """
        dim = x.shape[-1]
        if dim == 325:
            noise_rel = torch.randn_like(x[..., :39]) * self.jitter_sigma
            noise_ang = torch.randn_like(x[..., 39:]) * (self.jitter_sigma * 0.5)
            ang_jittered = torch.clamp(x[..., 39:] + noise_ang, 0.0, math.pi)
            return torch.cat([x[..., :39] + noise_rel, ang_jittered], dim=-1)
        elif dim == 286:
            noise = torch.randn_like(x) * (self.jitter_sigma * 0.5)
            return torch.clamp(x + noise, 0.0, math.pi)
        elif dim == 78:
            noise = torch.randn_like(x) * (self.jitter_sigma * 0.5)
            return torch.clamp(x + noise, -math.pi, math.pi)
        elif dim in (52, 53, 132, 133) or (dim % 4 == 0 or (dim - 1) % 4 == 0):
            # Coordinates with visibility: apply jitter only to (x, y, z), skip visibility
            x_jit = x.clone()
            stride = 4
            num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride
            for j in range(num_joints):
                x_jit[..., j * stride: j * stride + 3] += torch.randn_like(x[..., j * stride: j * stride + 3]) * self.jitter_sigma
            return x_jit
        else:
            noise = torch.randn_like(x) * self.jitter_sigma
            return x + noise

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a random 2D in-plane rotation [-angle, +angle] to coordinate pairs (x, y).
        - Preserves z-coordinate and visibility untouched for 3D coordinates.
        - Triplet angles (286 dims) are rigid-rotation invariant -> untouched.
        - 2D pair angles (78 dims) are shifted by the rotation angle.
        - mix representation (325 dims): rotates only rel_3d (x, y) and preserves angles.
        """
        dim = x.shape[-1]
        if dim == 286:
            # Triplet angles are strictly invariant under in-plane rigid rotation
            return x.clone()
        elif dim == 78:
            # 2D pair inclination angles shift by the rotation angle
            angle = (torch.rand(1).item() * 2 - 1) * self.max_rotation_degrees
            rad = math.radians(angle)
            return (x + rad + math.pi) % (2 * math.pi) - math.pi

        angle = (torch.rand(1).item() * 2 - 1) * self.max_rotation_degrees
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x_rot = x.clone()

        if dim == 325:
            # First 39 dims are rel_3d (13 joints * 3)
            for j in range(13):
                idx_x = j * 3
                idx_y = j * 3 + 1
                px = x[..., idx_x]
                py = x[..., idx_y]
                x_rot[..., idx_x] = px * cos_a - py * sin_a
                x_rot[..., idx_y] = px * sin_a + py * cos_a
            # Z coordinates (idx_z = j * 3 + 2) and angles remain untouched
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
            # Notice: z coordinate (idx_x + 2) and visibility (idx_x + 3) are strictly untouched!

        return x_rot

    def yaw_rotate_3d(self, x: torch.Tensor, max_yaw_degrees: float = 15.0) -> torch.Tensor:
        """
        Applies a realistic 3D yaw rotation around the vertical Y-axis:
        x' = x * cos(theta) + z * sin(theta)
        y' = y
        z' = -x * sin(theta) + z * cos(theta)
        Strictly preserves y-coordinate and visibility.
        Angle features (286, 78) are invariant to yaw rotation around vertical axis -> untouched.
        """
        dim = x.shape[-1]
        if dim in (286, 78):
            # Spatial triplet angles and elevation angles from horizontal ground are invariant to yaw rotation
            return x.clone()

        angle = (torch.rand(1).item() * 2 - 1) * max_yaw_degrees
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        x_rot = x.clone()

        if dim == 325:
            # First 39 dims are rel_3d (stride 3: x, y, z)
            for j in range(13):
                idx_x = j * 3
                idx_z = j * 3 + 2
                px = x[..., idx_x]
                pz = x[..., idx_z]
                x_rot[..., idx_x] = px * cos_a + pz * sin_a
                x_rot[..., idx_z] = -px * sin_a + pz * cos_a
            # Dynamically recompute angles from the yaw-rotated coordinates
            mirrored_angles = self.recompute_mix_angles(x_rot[..., :39])
            return torch.cat([x_rot[..., :39], mirrored_angles], dim=-1)

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

        elif dim % 4 == 0 or (dim - 1) % 4 == 0:
            # Covers 4-channel formats (13_4: 52, 12rel_4: 53, full_4: 132, full_rel_4: 133)
            stride = 4
            num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride
            for j in range(num_joints):
                idx_x = j * stride
                idx_z = j * stride + 2
                px = x[..., idx_x]
                pz = x[..., idx_z]
                x_rot[..., idx_x] = px * cos_a + pz * sin_a
                x_rot[..., idx_z] = -px * sin_a + pz * cos_a
            return x_rot

        else:
            # Fallback for 2D representations (in-plane rotation)
            return self.rotate(x)

    def joint_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """
        Randomly drops (zeros out) 1 to 2 joints throughout the sequence to simulate occlusions.
        Only applied to coordinate representations (not applied to angle representations).
        """
        dim = x.shape[-1]
        if dim in (325, 286, 78):
            # Angles are derived from multiple joints; dropping individual angle channels corrupts topological validity
            return x.clone()

        x_drop = x.clone()
        stride = 3 if dim == 39 else (4 if dim % 4 == 0 or (dim - 1) % 4 == 0 else 2)
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
        - mix (325 dims): scales only 3D coordinates, preserves angles.
        - angle features (286, 78): scale invariant -> untouched.
        - 4-channel coordinates (52, 53, 132, 133): scales only (x, y, z), preserves visibility channel.
        """
        dim = x.shape[-1]
        if dim in (286, 78):
            # Angles are scale-invariant
            return x.clone()

        factor = torch.empty(1).uniform_(scale_min, scale_max).item()

        if dim == 325:
            scaled_rel = x[..., :39] * factor
            return torch.cat([scaled_rel, x[..., 39:]], dim=-1)
        elif dim in (52, 53, 132, 133) or (dim % 4 == 0 or (dim - 1) % 4 == 0):
            x_scaled = x.clone()
            stride = 4
            num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride
            for j in range(num_joints):
                # Scale only (x, y, z), skip visibility
                x_scaled[..., j * stride: j * stride + 3] *= factor
            return x_scaled
        else:
            return x * factor

    def mirror(self, x: torch.Tensor) -> torch.Tensor:
        """
        Horizontal bilateral mirror (left/right flip) by swapping corresponding L/R joint features.
        - Coordinates: Negates x coordinates (x -> -x), preserves y, z and visibility.
        - Triplet angles (286 dims): Swaps symmetric triplet channels without negating values.
        - Pair angles (78 dims): Swaps symmetric pair channels.
        - mix representation (325 dims): flips rel_3d and recomputes the 286 3D angles dynamically.

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
        device = x.device

        # Dedicated handling for Angle representations
        if dim == 286:
            # Triplet angles: swap symmetric triplet indices, angles remain positive in [0, pi]
            sym_idx = self.triplet_sym_indices.to(device)
            return x[..., sym_idx]
        elif dim == 78:
            # Pair angles: swap symmetric pair indices
            sym_idx = self.pair_sym_indices.to(device)
            return x[..., sym_idx]

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

        num_joints = (dim - 1) // stride if (dim - 1) % stride == 0 else dim // stride

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
        # Notice: y, z, and visibility channels are fully preserved!

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
        elif method in ("skel_gym_aug", "combined"):
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
