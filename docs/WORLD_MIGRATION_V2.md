# SkelGym: World Migration & Upgraded Research Specification (v2)

> **Document Status:** Authoritative Migration Specification  
> **Reference Document:** [RESEARCH_UPGRADE_ROADMAP.md](file:///Volumes/WorkSpace/Project/Gym_Classification/docs/RESEARCH_UPGRADE_ROADMAP.md)  
> **Canonical Seeds:** `42, 123, 3407`  
> **Zero-Leakage Assurance:** Strict train-only normalization, pristine validation/test partitions, no synthetic augmentations at evaluation time.

---

## 1. Feature Representations & Semantics

| Feature Name | Dimension | Landmark Source | Coordinate Space & Normalization | Description |
| :--- | :---: | :--- | :--- | :--- |
| `raw_3d` | 39 | `data/{train,val,test}` (`pose_landmarks`) | Camera perspective image-normalized $[0, 1]$ | 13 anatomical keypoints $(x, y, z)$. |
| `world_3d` | 39 | `data/world_landmarks` (`pose_world_landmarks`) | Metric physical coordinates (meters), origin at pelvic mid-hip | 13 anatomical keypoints $(x, y, z)$. Preserves true Euclidean segment lengths. |
| `mix_v2` | 63 | `data/world_landmarks` (`pose_world_landmarks`) | Dual-branch: 39-d Metric World + 24-d World Kinematic Angles | Branch A (39-d): `world_3d`; Branch B (24-d): 10 3D joint articulation flexion/extension triplets + 14 spatial limb segment orientation angles calculated from the same world landmarks. |
| `world_joint_motion_3d` (or `joint_motion_3d` under WORLD) | 39 | `data/world_landmarks` (`pose_world_landmarks`) | First-order backward temporal difference: $\Delta X(t) = X(t) - X(t-1)$ | Velocity vector for 13 joints in physical metric space. |
| `bone_3d` | 39 | `data/{train,val,test}` | Directed bone vector from parent to child in kinematic tree | 13 directed anatomical bone vectors. |
| `bone_motion_3d` | 39 | `data/{train,val,test}` | First-order backward temporal difference: $\Delta B(t) = B(t) - B(t-1)$ | Bone velocity vector for 13 directed segments. |

---

## 2. Canonical Fusion Protocols (Table 6)

The ensemble evaluation is strictly restricted to three transparent, zero-leakage, calibration-free / validation-calibrated voting methods:

1. **Hard Majority Voting:** Discrete consensus across constituent stream argmax predictions.
2. **Uniform Average Soft Voting ($w_i = 1/K$):** Zero-parameter heuristic SOTA averaging softmax probability distributions across all streams.
3. **Accuracy-Weighted Soft Voting:** Soft voting weighted by validation video accuracy normalized to sum to 1 ($\sum w_i = 1$).

*Note on Deprecation:* SLSQP constrained optimization and Stacking Meta-Classifier (Ridge/Logistic) are permanently removed from the master tables due to validation cross-entropy overfitting and risk of calibration variance.

---

## 3. Dependency Map & Execution Order

```text
Phase 0: Khóa Specification (Tài liệu này)
    ↓
Phase 1: Code Migration
    - Cập nhật extract_features cho mix_v2 và world_joint_motion_3d
    - Sửa LandmarkAugmenter cho mix_v2 (augment trên world skeleton, recompute angles 24-d)
    - Xây dựng 12 candidate augmentation runners (Clean, Full 5-op, 5 LOO, 5 Single)
    ↓
Phase 2: Table 2 Clean Sequence Experiments
    - LSTM + world_3d (3 seeds)
    - LSTM + mix_v2 (3 seeds)
    - BiLSTM + world_3d (3 seeds)
    - BiLSTM + mix_v2 (3 seeds)
    - Transformer + world_3d: Tái sử dụng kết quả 3 seeds đã hoàn thành (75.68% Vid Acc)
    ↓
Phase 3: Table 4 (LOO) & Table 5 (Single) Ablation trên WORLD mix_v2
    - 12 configurations × 3 seeds = 36 training runs trên Validation selection set
    ↓
Phase 4: Phân tích & Chốt Proposed SkelGym-Aug
    - Đánh giá Val Macro-F1, Val Acc, Val Loss, seed variance
    - Freeze config: configs/augmentation/skelgym_aug_v2.yaml
    ↓
Phase 5: Finalize Table 4-5 & Transformer Aug Checkpoints
    - Đánh giá trên Held-Out Test split
    - Chốt checkpoint Transformer Mix v2 Proposed Aug
    ↓
Phase 6: Table 3 Graph Kinematic Streams (AAGCN)
    - T3.1: ST-GCN Raw 3D Clean (Reuse)
    - T3.2: ST-GCN World 3D Clean (3 seeds)
    - T3.3: AAGCN Bone Clean (Reuse)
    - T3.4: AAGCN Bone + Proposed Aug (3 seeds)
    - T3.5: AAGCN World Joint + Proposed Aug (3 seeds)
    - T3.6: AAGCN World Joint Motion + Proposed Aug (3 seeds)
    - T3.7: AAGCN Bone Motion + Proposed Aug (3 seeds)
    - T3.8: Two-Stream (World Joint + Bone, Uniform Soft)
    - T3.9: Four-Stream (World Joint + Bone + World Joint Motion + Bone Motion, Uniform Soft)
    ↓
Phase 7: Table 6 Rebuild (SkelGym-Lite & SkelGym-Full)
    - 3 Fusion protocols: Hard, Uniform Soft, Accuracy-Weighted Soft
    ↓
Phase 8: Tables 7-10 Generation (Consensus, Statistical Tests, Bootstrap CI, Per-class)
    ↓
Phase 9: Tables 11 & 13 (Computational Complexity & External Baselines)
    ↓
Phase 10: Canonical Source of Truth (canonical_results_v2.json)
    ↓
Phase 11: Paper Synchronous Update
    ↓
Phase 12: Audit & Release
```
