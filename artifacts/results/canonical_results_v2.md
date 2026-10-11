# SkelGym: Cross-Paradigm Biomechanical Skeleton Action Recognition

- **Feature Pipeline Version:** `mix_v2_world_63d`
- **Random Seeds (Multi-seed downstream consistency):** `[42, 123, 3407]`
- **Target Dataset Metadata:** `Final_dataset_metadata.csv` (22 Exercise Classes)
- **Status:** 🏆 **100% Comprehensive Paper-Freeze Canonical Benchmark Single Source of Truth**

### Trainable Parameter Footprints

| Architecture Family | Model Name / Variant | Trainable Parameters | Paradigm / Modality |
| :--- | :--- | :---: | :--- |
| **LSTM** | LSTM (World 3D / Mix v2) | **362K** | Recurrent Sequential |
| **BiLSTM** | BiLSTM (World 3D / Mix v2) | **360K** | Bidirectional Recurrent |
| **Transformer** | Transformer Backbone (mix_v2) | **301K** | Self-Attention Sequential |
| **ST-GCN** | ST-GCN (Raw 3D / World 3D) | **350K** | Fixed Graph Convolutional |
| **AAGCN** | AAGCN (Per Kinematic Stream) | **378K** | Adaptive Graph Convolutional |
| **SkelGym-Lite** | SkelGym-Lite (Trans + Bone AAGCN) | **679K** | Compact Cross-Paradigm (2 Streams) |
| **Four-Stream AAGCN** | Four-Stream AAGCN (Joint+Bone+2 Motions) | **1.51M** | Unified Graph Ensemble (4 Streams) |
| **SkelGym-Full** | SkelGym-Full (Trans + 4 AAGCN Streams) | **1.81M** | Full Cross-Paradigm Ensemble (5 Streams) |

---

## Table 1: Systematic Exploration of Kinematic Encodings (Feature Screening Benchmark — 27 Experiments)

*Description:* Comprehensive evaluation of 3 temporal sequence models across 9 kinematic representations on held-out validation and test sets. Mean ± SD across 3 random seeds (42, 123, 3407).

| Exp ID | Model Backbone | Feature Representation | Dim | Val Win Acc (%) | Val Win Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **T1.1** | LSTM | Raw 2D Coordinates | 26 | 67.49% ± 0.75% | 0.6556 ± 0.0075 | 52.80% ± 1.04% | 0.5143 ± 0.0138 | Verified |
| **T1.2** | LSTM | Relative 2D (Mid-Hip) | 26 | 72.40% ± 1.07% | 0.7131 ± 0.0110 | 56.13% ± 0.66% | 0.5501 ± 0.0082 | Verified |
| **T1.3** | LSTM | Angle 2D (Triplets) | 286 | 64.74% ± 2.62% | 0.6310 ± 0.0278 | 50.45% ± 1.59% | 0.4896 ± 0.0166 | Verified |
| **T1.4** | LSTM | Angle2 2D (Pairs) | 78 | 65.88% ± 1.61% | 0.6468 ± 0.0142 | 52.25% ± 0.70% | 0.5132 ± 0.0028 | Verified |
| **T1.5** | LSTM | Raw 3D Coordinates | 39 | 65.32% ± 2.25% | 0.6323 ± 0.0293 | 52.77% ± 1.77% | 0.5041 ± 0.0250 | Verified |
| **T1.6** | LSTM | Relative 3D (Mid-Hip) | 39 | 72.24% ± 0.99% | 0.7096 ± 0.0097 | 59.28% ± 0.77% | 0.5786 ± 0.0079 | Verified |
| **T1.7** | LSTM | Angle 3D (Triplets) | 286 | 65.73% ± 1.02% | 0.6398 ± 0.0083 | 50.45% ± 0.82% | 0.4886 ± 0.0086 | Verified |
| **T1.8** | LSTM | Angle2 3D (Pair Elevation) | 78 | 70.31% ± 3.35% | 0.6901 ± 0.0408 | 55.29% ± 1.49% | 0.5495 ± 0.0190 | Verified |
| **T1.9** | LSTM | **Biomechanical Mix (Ours)** | 117 | **74.28% ± 0.50%** | **0.7348 ± 0.0063** | **59.35% ± 0.19%** | **0.5827 ± 0.0015** | Verified |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | 65.12% ± 3.22% | 0.6272 ± 0.0331 | 51.48% ± 0.71% | 0.4901 ± 0.0184 | Verified |
| **T1.11** | BiLSTM | Relative 2D (Mid-Hip) | 26 | 64.98% ± 6.30% | 0.6411 ± 0.0516 | 54.51% ± 2.33% | 0.5247 ± 0.0290 | Verified |
| **T1.12** | BiLSTM | Angle 2D (Triplets) | 286 | 65.82% ± 0.63% | 0.6523 ± 0.0046 | 52.73% ± 1.58% | 0.5068 ± 0.0175 | Verified |
| **T1.13** | BiLSTM | Angle2 2D (Pairs) | 78 | 69.59% ± 2.28% | 0.6929 ± 0.0223 | 55.35% ± 1.80% | 0.5379 ± 0.0188 | Verified |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | 67.63% ± 0.11% | 0.6659 ± 0.0039 | 53.25% ± 0.88% | 0.5120 ± 0.0087 | Verified |
| **T1.15** | BiLSTM | Relative 3D (Mid-Hip) | 39 | 72.47% ± 1.39% | 0.7172 ± 0.0103 | 58.09% ± 1.46% | 0.5595 ± 0.0089 | Verified |
| **T1.16** | BiLSTM | Angle 3D (Triplets) | 286 | 64.37% ± 0.83% | 0.6293 ± 0.0075 | 50.51% ± 1.57% | 0.4846 ± 0.0201 | Verified |
| **T1.17** | BiLSTM | Angle2 3D (Pair Elevation) | 78 | 72.58% ± 1.06% | 0.7204 ± 0.0087 | 58.56% ± 0.67% | 0.5773 ± 0.0046 | Verified |
| **T1.18** | BiLSTM | **Biomechanical Mix (Ours)** | 117 | **74.39% ± 0.26%** | **0.7371 ± 0.0043** | **60.19% ± 1.14%** | **0.5896 ± 0.0126** | Verified |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | 77.82% ± 0.22% | 0.7700 ± 0.0008 | 63.51% ± 0.80% | 0.6323 ± 0.0094 | Verified |
| **T1.20** | Transformer | Relative 2D (Mid-Hip) | 26 | 76.45% ± 0.41% | 0.7539 ± 0.0033 | 63.06% ± 2.61% | 0.6159 ± 0.0244 | Verified |
| **T1.21** | Transformer | Angle 2D (Triplets) | 286 | 71.78% ± 0.55% | 0.7107 ± 0.0077 | 57.37% ± 0.20% | 0.5437 ± 0.0060 | Verified |
| **T1.22** | Transformer | Angle2 2D (Pairs) | 78 | 69.65% ± 0.68% | 0.6877 ± 0.0111 | 55.27% ± 1.47% | 0.5402 ± 0.0155 | Verified |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | 79.65% ± 0.80% | 0.7888 ± 0.0080 | 65.01% ± 0.27% | 0.6375 ± 0.0027 | Verified |
| **T1.24** | Transformer | Relative 3D (Mid-Hip) | 39 | 77.69% ± 0.69% | 0.7692 ± 0.0107 | 63.52% ± 1.18% | 0.6151 ± 0.0126 | Verified |
| **T1.25** | Transformer | Angle 3D (Triplets) | 286 | 70.33% ± 1.10% | 0.6842 ± 0.0096 | 56.86% ± 1.09% | 0.5567 ± 0.0081 | Verified |
| **T1.26** | Transformer | Angle2 3D (Pair Elevation) | 78 | 73.61% ± 0.66% | 0.7325 ± 0.0083 | 59.27% ± 1.32% | 0.5826 ± 0.0132 | Verified |
| **T1.27** | Transformer | **Biomechanical Mix (Dual-Branch)** | 117 | **78.14% ± 0.61%** | **0.7774 ± 0.0071** | **66.14% ± 0.11%** | **0.6516 ± 0.0028** | Verified |

---

## Table 2: Clean Sequences Temporal Architecture Comparison (Standardized World 3D & Mix v2)

*Description:* Direct comparison between metric physical space World 3D (39-d) and the upgraded Biomechanical Mix v2 (63-d). Mean ± std across seeds 42, 123, 3407.

| Model Key | Architecture | Modality | Dim | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LSTM_world_3d` | **LSTM** | `world_3d` | 39 | 73.64% ± 1.26% | 0.7244 ± 0.0128 | 76.97% ± 2.63% | 61.41% ± 0.80% | 0.6002 ± 0.0063 | **71.10% ± 2.33%** | **0.6941 ± 0.0294** |
| `LSTM_mix_v2` | **LSTM** | `mix_v2` | 63 | 75.90% ± 1.60% | 0.7541 ± 0.0173 | 80.19% ± 2.09% | 62.80% ± 0.83% | 0.6171 ± 0.0089 | **72.39% ± 1.76%** | **0.7077 ± 0.0172** |
| `BiLSTM_world_3d` | **BiLSTM** | `world_3d` | 39 | 74.72% ± 0.19% | 0.7341 ± 0.0020 | 79.23% ± 0.79% | 62.45% ± 1.52% | 0.6119 ± 0.0128 | **73.11% ± 2.28%** | **0.7136 ± 0.0204** |
| `BiLSTM_mix_v2` | **BiLSTM** | `mix_v2` | 63 | 76.90% ± 0.30% | 0.7599 ± 0.0050 | 80.68% ± 1.97% | 64.82% ± 0.37% | 0.6316 ± 0.0027 | **75.39% ± 1.46%** | **0.7342 ± 0.0144** |
| `Transformer_world_3d` | **Transformer** | `world_3d` | 39 | 78.76% ± 0.46% | 0.7781 ± 0.0053 | 81.97% ± 0.23% | 65.33% ± 0.71% | 0.6406 ± 0.0035 | **75.68% ± 0.53%** | **0.7343 ± 0.0085** |

---

## Table 3: Graph Kinematic Streams Benchmark on AAGCN (9 Streams / Fusions)

*Description:* Spatial and dynamic kinematic graph representations evaluated on Adaptive Graph Convolutional Networks (AAGCN) and ST-GCN. Mean ± std across seeds 42, 123, 3407.

| Stream ID | Description | Model | Modality | Augmentation | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T3.1** | Raw 3D Joint (ST-GCN Clean) | STGCN | `raw_3d` | `none` | 63.01% ± 1.86% | 0.6255 ± 0.0224 | 69.08% ± 1.42% | 0.6932 ± 0.0109 | 48.35% ± 0.94% | 0.4701 ± 0.0141 | **58.65% ± 1.76%** | **0.5432 ± 0.0208** |
| **T3.2** | World 3D Joint (ST-GCN Clean) | STGCN | `world_3d` | `none` | 73.75% ± 1.86% | 0.7332 ± 0.0154 | 79.71% ± 1.98% | 0.8092 ± 0.0083 | 62.59% ± 2.31% | 0.6034 ± 0.0219 | **74.68% ± 1.26%** | **0.7060 ± 0.0161** |
| **T3.3** | Bone 3D Stream (AAGCN Clean) | AAGCN | `bone_3d` | `none` | 76.00% ± 0.35% | 0.7542 ± 0.0037 | 79.06% ± 1.60% | 0.8026 ± 0.0161 | 62.10% ± 2.11% | 0.6101 ± 0.0158 | **71.67% ± 2.80%** | **0.7025 ± 0.0307** |
| **T3.4** | Bone 3D Stream (AAGCN + Aug) | AAGCN | `bone_3d` | `mirror_yaw` | 79.30% ± 0.22% | 0.7858 ± 0.0032 | 82.77% ± 1.49% | 0.8408 ± 0.0088 | 66.36% ± 1.41% | 0.6548 ± 0.0173 | **75.39% ± 2.33%** | **0.7440 ± 0.0335** |
| **T3.5** | World Joint Stream (AAGCN + Aug) | AAGCN | `world_3d` | `mirror_yaw` | 78.86% ± 0.31% | 0.7856 ± 0.0024 | 80.68% ± 0.40% | 0.8209 ± 0.0028 | 70.99% ± 0.38% | 0.6962 ± 0.0012 | **80.98% ± 0.41%** | **0.8086 ± 0.0084** |
| **T3.6** | World Joint Motion (Delta X) | AAGCN | `world_joint_motion_3d` | `mirror_yaw` | 64.19% ± 1.61% | 0.6464 ± 0.0083 | 75.36% ± 0.40% | 0.7683 ± 0.0115 | 57.70% ± 0.65% | 0.5583 ± 0.0082 | **73.25% ± 1.46%** | **0.6997 ± 0.0314** |
| **T3.7** | Bone Motion (Delta B) | AAGCN | `bone_motion_3d` | `mirror_yaw` | 61.93% ± 1.56% | 0.6362 ± 0.0107 | 73.91% ± 1.81% | 0.7396 ± 0.0316 | 57.83% ± 0.83% | 0.5727 ± 0.0019 | **74.25% ± 1.85%** | **0.7235 ± 0.0163** |
| **T3.8** | Two-Stream AAGCN (World Joint + Bone) (T3.5+T3.4) | AAGCN | `multi-stream` | `mirror_yaw` | 82.03% ± 0.41% | 0.8179 ± 0.0036 | 84.06% ± 0.68% | 0.8533 ± 0.0069 | 72.06% ± 0.54% | 0.7105 ± 0.0051 | **82.69% ± 0.73%** | **0.8225 ± 0.0077** |
| **T3.9** | Four-Stream AAGCN (Full) (T3.5+T3.4+T3.6+T3.7) | AAGCN | `multi-stream` | `mirror_yaw` | 84.34% ± 0.26% | 0.8414 ± 0.0030 | 87.92% ± 1.42% | 0.8882 ± 0.0115 | 74.50% ± 0.65% | 0.7341 ± 0.0048 | **84.41% ± 1.01%** | **0.8308 ± 0.0179** |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation (7 Configurations)

*Description:* Evaluating necessity of each augmentation operator on Transformer Mix (63-d) by leaving one operator out of Candidate Full (5-op). Mean ± std across seeds 42, 123, 3407.

| Configuration Key | Strategy Description | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | **Clean Baseline (No Aug)** | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 84.38% ± 1.27% | 0.8505 ± 0.0109 | 69.24% ± 0.20% | 0.6815 ± 0.0017 | **79.83% ± 1.27%** | **0.7806 ± 0.0214** |
| `candidate_full_5op` | **Candidate Full 5-op** | 79.19% ± 1.26% | 0.7882 ± 0.0152 | 83.41% ± 1.94% | 0.8383 ± 0.0156 | 72.40% ± 1.19% | 0.7145 ± 0.0083 | **81.97% ± 0.70%** | **0.8146 ± 0.0081** |
| `candidate_minus_mirror` | **Minus Mirror** | 78.78% ± 1.44% | 0.7757 ± 0.0197 | 81.32% ± 2.85% | 0.8226 ± 0.0247 | 68.27% ± 2.32% | 0.6729 ± 0.0231 | **78.26% ± 2.68%** | **0.7679 ± 0.0288** |
| `candidate_minus_yaw` | **Minus Yaw** | 80.06% ± 0.32% | 0.7964 ± 0.0026 | 84.22% ± 1.21% | 0.8468 ± 0.0120 | 72.99% ± 0.60% | 0.7220 ± 0.0043 | **82.69% ± 0.41%** | **0.8210 ± 0.0086** |
| `candidate_minus_scale` | **Minus Scale** | 80.13% ± 0.13% | 0.7985 ± 0.0022 | 84.54% ± 0.68% | 0.8508 ± 0.0070 | 73.02% ± 1.24% | 0.7222 ± 0.0109 | **82.69% ± 1.62%** | **0.8211 ± 0.0168** |
| `candidate_minus_time` | **Minus TimeWarp (Proposed 4-op)** | 79.54% ± 0.57% | 0.7926 ± 0.0060 | 83.90% ± 0.60% | 0.8450 ± 0.0113 | 73.59% ± 0.48% | 0.7257 ± 0.0072 | **83.55% ± 0.54%** | **0.8300 ± 0.0054** |
| `candidate_minus_jitter` | **Minus Jitter** | 79.68% ± 0.22% | 0.7937 ± 0.0048 | 84.22% ± 0.82% | 0.8488 ± 0.0109 | 73.17% ± 0.90% | 0.7233 ± 0.0102 | **82.12% ± 0.73%** | **0.8181 ± 0.0134** |

---

## Table 5: Single & Paired Component Augmentation Ablation (7 Configurations)

*Description:* Evaluating standalone and paired contributions of individual operators relative to clean control. Mean ± std across seeds 42, 123, 3407.

| Configuration Key | Strategy Description | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean` | **Clean Baseline (No Aug)** | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 84.38% ± 1.27% | 0.8505 ± 0.0109 | 69.24% ± 0.20% | 0.6815 ± 0.0017 | **79.83% ± 1.27%** | **0.7806 ± 0.0214** |
| `single_mirror` | **Only Mirror** | 80.45% ± 0.45% | 0.8001 ± 0.0067 | 85.18% ± 0.99% | 0.8510 ± 0.0157 | 73.66% ± 0.56% | 0.7261 ± 0.0046 | **82.69% ± 0.73%** | **0.8192 ± 0.0075** |
| `single_yaw` | **Only Yaw** | 80.29% ± 0.34% | 0.7977 ± 0.0031 | 85.02% ± 0.40% | 0.8581 ± 0.0012 | 69.27% ± 1.19% | 0.6853 ± 0.0124 | **79.26% ± 0.54%** | **0.7838 ± 0.0173** |
| `single_scale` | **Only Scale** | 79.68% ± 0.22% | 0.7904 ± 0.0066 | 83.90% ± 1.21% | 0.8426 ± 0.0064 | 68.90% ± 1.13% | 0.6807 ± 0.0106 | **78.54% ± 0.70%** | **0.7708 ± 0.0091** |
| `single_time` | **Only TimeWarp** | 79.74% ± 0.24% | 0.7920 ± 0.0039 | 83.25% ± 0.23% | 0.8361 ± 0.0095 | 68.90% ± 0.75% | 0.6825 ± 0.0083 | **79.40% ± 0.93%** | **0.7865 ± 0.0184** |
| `single_jitter` | **Only Jitter** | 79.55% ± 0.82% | 0.7884 ± 0.0093 | 83.25% ± 0.45% | 0.8378 ± 0.0035 | 68.75% ± 0.87% | 0.6768 ± 0.0083 | **78.54% ± 1.53%** | **0.7739 ± 0.0173** |
| `pair_mirror_yaw` | **🥇 Proposed SkelGym-Aug (Mirror + Yaw)** | 81.20% ± 0.72% | 0.8119 ± 0.0072 | 84.70% ± 0.82% | 0.8469 ± 0.0105 | 73.14% ± 0.90% | 0.7253 ± 0.0106 | **82.69% ± 1.66%** | **0.8187 ± 0.0196** |

---

## Table 6: Multi-Stream Cross-Paradigm Ensemble Comparison

*Description:* Benchmarking 3 standardized fusion methods across SkelGym-Lite and SkelGym-Full. Mean ± std across seeds 42, 123, 3407.

| Ensemble Architecture | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Validation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SkelGym-Lite** | Hard Majority Voting (Discrete Baseline) | 79.47% ± 0.81% | 0.7883 ± 0.0084 | 82.93% ± 0.60% | 0.8277 ± 0.0065 | 67.31% ± 0.46% | 0.6799 ± 0.0030 | 76.68% ± 0.88% | 0.7615 ± 0.0064 | Verified |
| **SkelGym-Lite** | Accuracy-Weighted Soft Voting (Validation-calibrated weights) | **83.47% ± 0.14%** | **0.8351 ± 0.0021** | 85.67% ± 1.14% | 0.8634 ± 0.0066 | 73.74% ± 0.63% | 0.7310 ± 0.0058 | **82.12% ± 0.40%** | **0.8198 ± 0.0030** | 🏆 Validation Winner (SkelGym-Lite) |
| **SkelGym-Lite** | Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K) | 83.26% ± 0.06% | 0.8327 ± 0.0008 | 85.67% ± 1.14% | 0.8634 ± 0.0066 | 73.44% ± 0.65% | 0.7284 ± 0.0058 | 82.12% ± 0.40% | 0.8198 ± 0.0030 | Verified |
| **SkelGym-Full** | Hard Majority Voting (Discrete Baseline) | 84.11% ± 0.30% | 0.8411 ± 0.0041 | 86.15% ± 1.21% | 0.8660 ± 0.0090 | 74.07% ± 0.11% | 0.7371 ± 0.0011 | 84.12% ± 1.53% | 0.8316 ± 0.0141 | Verified |
| **SkelGym-Full** | Accuracy-Weighted Soft Voting (Validation-calibrated weights) | 84.85% ± 0.29% | 0.8475 ± 0.0024 | 87.28% ± 1.27% | 0.8821 ± 0.0113 | 76.24% ± 0.02% | 0.7537 ± 0.0020 | 85.55% ± 0.54% | 0.8440 ± 0.0086 | Verified |
| **SkelGym-Full** | Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K) | **85.05% ± 0.47%** | **0.8487 ± 0.0046** | 87.92% ± 1.37% | 0.8886 ± 0.0126 | 76.19% ± 0.16% | 0.7537 ± 0.0014 | **85.27% ± 0.41%** | **0.8386 ± 0.0075** | 🏆 Validation Winner (SkelGym-Full) |

---

## Table 7: Master Unified Benchmark Matrix (All 10 Models)

*Description:* Comprehensive performance summary of all 10 audited architectures across window and video consensus regimes on held-out test data. Mean ± std across seeds 42, 123, 3407.

| Model Architecture | Params | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LSTM (World 3D Clean) | 362K | 73.64% ± 1.26% | 0.7244 ± 0.0128 | 76.97% ± 2.63% | 61.41% ± 0.80% | 0.6002 ± 0.0063 | 71.10% ± 2.33% | 0.6941 ± 0.0294 |
| LSTM (Biomechanical Mix v2) | 362K | 75.90% ± 1.60% | 0.7541 ± 0.0173 | 80.19% ± 2.09% | 62.80% ± 0.83% | 0.6171 ± 0.0089 | 72.39% ± 1.76% | 0.7077 ± 0.0172 |
| BiLSTM (Biomechanical Mix v2) | 360K | 76.90% ± 0.30% | 0.7599 ± 0.0050 | 80.68% ± 1.97% | 64.82% ± 0.37% | 0.6316 ± 0.0027 | 75.39% ± 1.46% | 0.7342 ± 0.0144 |
| Transformer (Biomechanical Mix v2 Clean) | 301K | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 84.38% ± 1.27% | 69.24% ± 0.20% | 0.6815 ± 0.0017 | 79.83% ± 1.27% | 0.7806 ± 0.0214 |
| Transformer (Biomechanical Mix v2 + SkelGym-Aug) | 301K | 81.20% ± 0.72% | 0.8119 ± 0.0072 | 84.70% ± 0.82% | 73.14% ± 0.90% | 0.7253 ± 0.0106 | 82.69% ± 1.66% | 0.8187 ± 0.0196 |
| ST-GCN (Raw 3D Clean) | 350K | 63.01% ± 1.86% | 0.6255 ± 0.0224 | 69.08% ± 1.42% | 48.35% ± 0.94% | 0.4701 ± 0.0141 | 58.65% ± 1.76% | 0.5432 ± 0.0208 |
| AAGCN (Bone 3D + SkelGym-Aug) | 378K | 79.30% ± 0.22% | 0.7858 ± 0.0032 | 82.77% ± 1.49% | 66.36% ± 1.41% | 0.6548 ± 0.0173 | 75.39% ± 2.33% | 0.7440 ± 0.0335 |
| Four-Stream AAGCN (Uniform Soft Voting) | 1.51M | 84.34% ± 0.26% | 0.8414 ± 0.0030 | 87.92% ± 1.42% | 74.50% ± 0.65% | 0.7341 ± 0.0048 | 84.41% ± 1.01% | 0.8308 ± 0.0179 |
| SkelGym-Lite (2 Streams, Accuracy-Weighted Soft Voting (Validation-calibrated weights)) | 679K | 83.47% ± 0.14% | 0.8351 ± 0.0021 | 85.67% ± 1.14% | 73.74% ± 0.63% | 0.7310 ± 0.0058 | 82.12% ± 0.40% | 0.8198 ± 0.0030 |
| **SkelGym-Full (5 Streams, Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K))** | 1.81M | 85.05% ± 0.47% | 0.8487 ± 0.0046 | 87.92% ± 1.37% | 76.19% ± 0.16% | 0.7537 ± 0.0014 | **85.27% ± 0.41%** | **0.8386 ± 0.0075** |

---

## Table 8: Paired Statistical Hypothesis Testing (Statistical Significance)

*Description:* McNemar test on test windows (N=2,743) and Wilcoxon signed-rank + paired t-test on video clusters (N=233).

| Pairwise Comparison (M_A vs M_B) | Window McNemar χ² | Window p-value | Window Odds Ratio | Video Wilcoxon W | Video p-value | Video Paired t p-value | Video Cohen's d | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 16.65 | 4.21e-05 | 1.57 | 10432.5 | 0.0019 | 0.0022 | +0.203 | Verified (**) |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 118.62 | 1.46e-28 | 2.63 | 12666.0 | 0.3491 | 0.1026 | +0.107 | Verified (n.s.) |
| **Single Sequence (Trans) vs SkelGym-Full** | 42.44 | 3.84e-11 | 2.42 | 6086.0 | 2.39e-13 | 2.12e-11 | -0.461 | Verified (***) |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 137.47 | 5.60e-34 | 4.21 | 11026.0 | 0.0115 | 0.2098 | -0.082 | Verified (*) |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 15.82 | 5.88e-05 | 2.12 | 6086.0 | 2.39e-13 | 2.12e-11 | +0.461 | Verified (***) |

---

## Table 9: Non-Parametric Bootstrap 95% Confidence Intervals (B=1,000 Resamples)

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM (Mix v2 63-d)** | 61.48% [55.12%, 67.40%] | 0.5962 [0.5375, 0.6507] | 74.31% [69.09%, 79.83%] | 0.7209 [0.6606, 0.7828] |
| **BiLSTM (Mix v2 63-d)** | 64.57% [58.54%, 70.37%] | 0.6112 [0.5514, 0.6765] | 76.93% [71.24%, 81.97%] | 0.7358 [0.6688, 0.7969] |
| **ST-GCN (World 3D)** | 65.07% [58.49%, 71.17%] | 0.6115 [0.5610, 0.6657] | 76.63% [70.82%, 81.98%] | 0.7267 [0.6703, 0.7861] |
| **Transformer (Mix v2 Clean)** | 69.91% [63.40%, 75.76%] | 0.6723 [0.6093, 0.7365] | 79.88% [74.68%, 84.98%] | 0.7788 [0.7219, 0.8325] |
| **Transformer (Mix v2 SkelGym-Aug)** | 72.60% [66.60%, 78.19%] | 0.6996 [0.6390, 0.7548] | 82.03% [76.81%, 86.70%] | 0.7985 [0.7349, 0.8521] |
| **AAGCN (Bone 3D)** | 68.20% [62.11%, 74.12%] | 0.6621 [0.6017, 0.7195] | 78.51% [72.96%, 83.69%] | 0.7792 [0.7274, 0.8297] |
| **Four-Stream AAGCN** | 74.67% [68.69%, 80.33%] | 0.7222 [0.6647, 0.7769] | 85.75% [81.12%, 90.13%] | 0.8478 [0.7998, 0.8927] |
| **SkelGym-Lite** | 73.63% [67.68%, 79.13%] | 0.7143 [0.6543, 0.7677] | 82.42% [77.25%, 87.12%] | 0.8093 [0.7513, 0.8584] |
| **SkelGym-Full (Uniform Soft)** | **76.34% [70.03%, 81.90%]** | **0.7412 [0.6785, 0.7969]** | **85.83% [81.12%, 90.13%]** | **0.8409 [0.7848, 0.8937]** |

---

## Table 10: Per-Class Granular Breakdown (All 22 Exercise Classes, N=233, 200 Correct)

| Exercise Class | Win Prec | Win Rec | Win F1 | Win Supp | Vid Prec | Vid Rec | Vid F1 | Vid Supp |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.4069 | 0.8551 | 0.5514 | 69 | 0.6667 | 1.0000 | 0.8000 | 14 |
| bench press | 0.4091 | 0.6562 | 0.5040 | 96 | 0.7500 | 0.6429 | 0.6923 | 14 |
| chest fly machine | 0.9213 | 1.0000 | 0.9591 | 82 | 0.8889 | 1.0000 | 0.9412 | 8 |
| deadlift | 0.4046 | 0.7910 | 0.5354 | 67 | 0.6923 | 0.9000 | 0.7826 | 10 |
| decline bench press | 0.6270 | 0.5896 | 0.6077 | 134 | 0.7500 | 0.7500 | 0.7500 | 8 |
| hammer curl | 0.5932 | 0.4348 | 0.5018 | 161 | 0.8750 | 0.5000 | 0.6364 | 14 |
| hip thrust | 0.9245 | 0.6229 | 0.7443 | 236 | 1.0000 | 0.8889 | 0.9412 | 9 |
| incline bench press | 0.7397 | 0.7105 | 0.7248 | 76 | 0.7778 | 0.7778 | 0.7778 | 9 |
| lat pulldown | 0.6691 | 0.9300 | 0.7782 | 100 | 0.8125 | 1.0000 | 0.8966 | 13 |
| lateral raise | 1.0000 | 0.9484 | 0.9735 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 |
| leg extension | 0.9542 | 1.0000 | 0.9766 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 |
| leg raises | 0.9898 | 0.8509 | 0.9151 | 114 | 1.0000 | 1.0000 | 1.0000 | 11 |
| plank | 0.7925 | 0.7500 | 0.7706 | 56 | 0.6667 | 1.0000 | 0.8000 | 2 |
| pull Up | 0.7971 | 0.6707 | 0.7285 | 82 | 1.0000 | 0.7000 | 0.8235 | 10 |
| push-up | 0.9062 | 1.0000 | 0.9508 | 87 | 0.9231 | 1.0000 | 0.9600 | 12 |
| romanian deadlift | 0.6792 | 0.4932 | 0.5714 | 146 | 0.7500 | 0.5000 | 0.6000 | 6 |
| russian twist | 0.9844 | 0.9197 | 0.9509 | 137 | 1.0000 | 1.0000 | 1.0000 | 6 |
| shoulder press | 0.7101 | 0.6447 | 0.6759 | 152 | 0.7500 | 0.6923 | 0.7200 | 13 |
| squat | 0.9384 | 0.8319 | 0.8820 | 238 | 1.0000 | 1.0000 | 1.0000 | 15 |
| t bar row | 0.8400 | 0.5385 | 0.6562 | 117 | 1.0000 | 0.7000 | 0.8235 | 10 |
| tricep Pushdown | 0.6083 | 0.7849 | 0.6854 | 93 | 0.7857 | 0.9167 | 0.8462 | 12 |
| tricep dips | 0.8987 | 0.9682 | 0.9322 | 220 | 0.8889 | 0.8889 | 0.8889 | 9 |
| **Overall Accuracy** | 0.7641 | 0.7641 | 0.7641 | **2743** | 0.8584 | 0.8584 | 0.8584 | **233** |
| **Macro Average** | 0.7634 | 0.7723 | 0.7534 | **2743** | 0.8626 | 0.8572 | 0.8491 | **233** |

---

## Table 11: Hardware Complexity & Inference Latency Benchmark

| Architecture / Stream | Parameters | MFLOPs / Win | Server GPU Latency (ms) | GPU FPS | Apple M4 MPS Latency (ms) | M4 FPS | Edge CPU Latency (ms) | CPU FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (63-d)** | 301K | 18.87 | 0.59 ms | 1685 | 1.36 ms | 738 | 0.49 ms | 2041 |
| **AAGCN Joint Stream** | 378K | 202.86 | 1.01 ms | 987 | 1.89 ms | 530 | 0.9 ms | 1115 |
| **AAGCN Bone Stream** | 378K | 202.86 | 1.02 ms | 976 | 1.72 ms | 581 | 0.88 ms | 1138 |
| **AAGCN Joint-Motion Stream** | 378K | 202.86 | 1.01 ms | 990 | 1.9 ms | 527 | 1.02 ms | 978 |
| **AAGCN Bone-Motion Stream** | 378K | 202.86 | 1.02 ms | 980 | 1.9 ms | 527 | 1.03 ms | 975 |
| **SkelGym-Lite (Transformer + Bone)** | 679K | 221.72 | 1.66 ms | 603 | 2.55 ms | 393 | 1.49 ms | 671 |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 1.81M | 830.29 | 4.65 ms | 215 | 5.86 ms | 171 | 3.96 ms | 253 |

---

## Table 12: External S&C Comparative Evaluation (Deyzel et al. 4-Exercise Benchmark)

| Model Architecture | Open-Set Win Acc (%) | Open-Set Vid Acc (%) | Closed-Set Win Acc (%) | Closed-Set Vid Acc (%) | Closed-Set Vid F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| ST-GCN (Rel 3D) | 59.74% | 68.52% | 82.61% | 85.19% | 0.8563 | Verified |
| LSTM (Mix 63-d) | 64.08% | 85.19% | 78.64% | 90.74% | 0.9071 | Verified |
| BiLSTM (Mix 63-d) | 71.64% | 85.19% | 84.12% | 90.74% | 0.9126 | Verified |
| AAGCN (Bone 3D) | 72.02% | 85.19% | 87.90% | 92.59% | 0.9281 | Verified |
| Transformer (Mix 63-d) | 77.88% | 85.19% | 96.03% | 98.15% | 0.9788 | Verified |
| **SkelGym-Lite** | 84.12% | 94.44% | **95.84%** | **98.15%** | **0.9782** | Verified |
| **SkelGym-Full** | 80.72% | 96.30% | **93.01%** | **96.30%** | **0.9626** | Verified |
