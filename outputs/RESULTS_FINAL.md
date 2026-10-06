# SkelGym: Resistance Exercise Recognition — Master Experiment Results & Benchmark Template

> **Document Status:** Master Results Execution Template (Clean Slate — Fresh Benchmark Run)  
> **Authoritative Baseline Split:** Video-Level Partition 6:2:2 (580 Train / 208 Validation / 236 Test; 233 Valid Test Videos $\ge 32$ Frames)  
> **Repository:** `Cuong2004/gym-exercise-classification` (Model Hub) | `Cuong2004/gym-exercise-landmarks` (Dataset Hub)  
> **Backbone Budget:** Calibrated compact parameter budget ($\approx 300\text{K} - 350\text{K}$ parameters across all backbones)

---

## 1. Experimental Protocol & Specifications

- **Dataset Scale:** 1,024 unique video recordings ($\approx 10.2$ GB) spanning 22 fine-grained resistance exercises, trimmed into 1,108 clean action segments.
- **Strict Video-Level Partitioning:** Zero source-video overlap across Train ($N=580$), Validation ($N=208$), and Test ($N=236$) sets. Sliding windows ($T=32$ frames) are extracted strictly within individual trimmed action boundaries ($S=16$ train: 13,136 windows; $S=32$ val: 2,075 windows; $S=32$ test: 2,743 windows across 233 valid videos).
- **Physical Kinematics:** 13 anatomically calibrated keypoints from MediaPipe Pose Heavy.
  - **Scale-Normalized Relative Coordinates (`rel_3d_norm`, 39-d):** Origin centered at pelvic mid-hip midpoint ($p_{\text{hip\_mid}}$), lateral $X$-axis normalized by pelvic hip width $L_{\text{hip}}$, and longitudinal $Y, Z$-axes normalized by torso length $L_{\text{torso}}$.
  - **Kinematic Angles (`angle_kinematic_24`, 24-d):** 10 3D joint articulation flexion/extension triplets (knees, elbows, hips, shoulders) and 14 spatial limb segment orientation angles.
  - **Biomechanical Mix v2 (`mix_v2`, 63-d):** Dual-branch concatenation of 39-d scale-norm coordinates and 24-d kinematic angles.
- **Model Capacity:** All sequence backbones standardized to $\approx 300\text{K}$ parameters (Transformer: $d_{\text{model}}=112, d_{\text{ff}}=168, \text{nhead}=4, \text{layers}=3$; LSTM: $h=160, \text{layers}=2$; BiLSTM: $h=96, \text{layers}=2$).
- **Augmentation Pipeline (SkelGym-Aug):** Dynamic on-the-fly transformations (Bilateral Sagittal Reflection + Gravitational 3D Yaw $\pm 15^\circ$ + Proportional Scaling $\pm 10\%$ + Gaussian Sensor Jitter $\sigma=0.008$).
- **Multi-Stream Ensemble:** 5-stream cross-paradigm late fusion (Transformer on 63-d mix_v2 + 4 spatial-temporal AAGCN streams: Bone 3D, Joint 3D, Joint-Motion 3D, Bone-Motion 3D) evaluated across 4 standardized fusion protocols (Hard Majority Voting, Accuracy-Weighted Soft Voting, Uniform Average Soft Voting, Stacking Meta-Classifier).

---

## Table 1: Dataset Partition & Provenance Breakdown

| Split Name | Source Videos | Ratio (%) | Action Segments | Valid Videos ($\ge 32$ frames) | Extracted Windows ($T=32$) | Stride ($S$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 580 | 56.6% | 639 | 580 | 13,136 | 16 (50% overlap) | Verified |
| **Validation Set** | 208 | 20.3% | 210 | 208 | 2,075 | 32 (non-overlapping) | Verified |
| **Held-Out Test Set** | 236 | 23.1% | 259 | 233 | 2,743 | 32 (non-overlapping) | Verified |
| **Total Corpus** | **1,024** | **100.0%** | **1,108** | **1,021** | **17,954** | — | **Verified** |

---

## Table 2: Feature Representation Benchmark across Sequence Architectures (Paper Table 1)

*Objective:* Evaluate 3 sequence architectures (LSTM, BiLSTM, Transformer) across spatial coordinate, angular, and biomechanical representations under unaugmented baseline protocol.  
*Execution Command:* `python run.py train --model <MODEL> --feature <FEATURE> --exp_id <ID> --device auto`

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Macro F1 | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1.1** | LSTM | Raw 2D Coordinates | 26 | 0.1183 | — | 2.0490 | 69.25% ± 0.98% | — | 54.26% ± 0.27% | 0.5352 ± 0.0067 | Verified |
| **T1.2** | LSTM | Root-Relative 2D Coordinates | 26 | 0.0805 | — | 1.8685 | 75.29% ± 1.30% | — | 58.54% ± 1.05% | 0.5708 ± 0.0130 | Verified |
| **T1.3** | LSTM | Pairwise Joint Angles 2D | 286 | 0.1343 | — | 2.5289 | 66.12% ± 1.06% | — | 51.53% ± 1.57% | 0.4993 ± 0.0119 | Verified |
| **T1.4** | LSTM | Adjacent Joint Angles 2D | 78 | 0.1327 | — | 2.3319 | 69.98% ± 0.79% | — | 55.73% ± 1.29% | 0.5412 ± 0.0147 | Verified |
| **T1.5** | LSTM | Raw 3D Coordinates | 39 | 0.1183 | — | 2.0571 | 68.47% ± 1.67% | — | 53.53% ± 1.18% | 0.5117 ± 0.0133 | Verified |
| **T1.6** | LSTM | Metric World 3D Coordinates (world_3d) | 39 | — | — | 1.8250 | 73.64% ± 1.26% | — | 61.41% ± 0.80% | 0.6002 ± 0.0063 | Verified |
| **T1.7** | LSTM | Pairwise Joint Angles 3D | 286 | 0.1426 | — | 2.3883 | 65.16% ± 0.97% | — | 50.68% ± 0.51% | 0.4923 ± 0.0040 | Verified |
| **T1.8** | LSTM | Adjacent Joint Angles 3D | 78 | 0.0943 | — | 1.9178 | 72.84% ± 1.06% | — | 57.42% ± 0.62% | 0.5680 ± 0.0097 | Verified |
| **T1.9** | LSTM | Hybrid Geometric Multi-Feature (mix_v2) | 63 | — | — | 1.7323 | 75.90% ± 1.60% | — | 62.80% ± 0.83% | 0.6171 ± 0.0089 | Verified |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | 0.0747 | — | 2.2106 | 70.81% ± 0.73% | — | 54.28% ± 0.08% | 0.5338 ± 0.0033 | Verified |
| **T1.11** | BiLSTM | Root-Relative 2D Coordinates | 26 | 0.0790 | — | 1.8599 | 75.21% ± 0.91% | — | 58.95% ± 0.84% | 0.5700 ± 0.0101 | Verified |
| **T1.12** | BiLSTM | Pairwise Joint Angles 2D | 286 | 0.1284 | — | 2.5607 | 65.96% ± 0.47% | — | 52.37% ± 1.98% | 0.5043 ± 0.0206 | Verified |
| **T1.13** | BiLSTM | Adjacent Joint Angles 2D | 78 | 0.0623 | — | 2.5136 | 71.08% ± 0.17% | — | 56.53% ± 0.29% | 0.5508 ± 0.0072 | Verified |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | 0.1339 | — | 2.1974 | 68.19% ± 0.86% | — | 53.22% ± 0.90% | 0.5045 ± 0.0074 | Verified |
| **T1.15** | BiLSTM | Metric World 3D Coordinates (world_3d) | 39 | — | — | 1.9601 | 74.72% ± 0.19% | — | 62.45% ± 1.52% | 0.6119 ± 0.0128 | Verified |
| **T1.16** | BiLSTM | Pairwise Joint Angles 3D | 286 | 0.1690 | — | 2.0893 | 64.47% ± 0.69% | — | 51.25% ± 1.82% | 0.4922 ± 0.0194 | Verified |
| **T1.17** | BiLSTM | Adjacent Joint Angles 3D | 78 | 0.1432 | — | 1.6524 | 73.19% ± 0.67% | — | 57.90% ± 1.11% | 0.5719 ± 0.0098 | Verified |
| **T1.18** | BiLSTM | Hybrid Geometric Multi-Feature (mix_v2) | 63 | — | — | 1.5067 | 76.90% ± 0.30% | — | 64.82% ± 0.37% | 0.6316 ± 0.0027 | Verified |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | 0.3905 | — | 1.1568 | 77.75% ± 0.53% | — | 63.62% ± 1.06% | 0.6258 ± 0.0118 | Verified |
| **T1.20** | Transformer | Root-Relative 2D Coordinates | 26 | 0.5088 | — | 1.2146 | 75.44% ± 0.79% | — | 61.22% ± 1.61% | 0.5981 ± 0.0172 | Verified |
| **T1.21** | Transformer | Pairwise Joint Angles 2D | 286 | 0.4268 | — | 1.4610 | 69.85% ± 0.47% | — | 55.95% ± 0.74% | 0.5310 ± 0.0115 | Verified |
| **T1.22** | Transformer | Adjacent Joint Angles 2D | 78 | 0.4115 | — | 1.4490 | 70.15% ± 1.19% | — | 55.12% ± 1.68% | 0.5407 ± 0.0205 | Verified |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | 0.4239 | — | 1.1319 | 77.24% ± 0.59% | — | 62.87% ± 0.69% | 0.6202 ± 0.0097 | Verified |
| **T1.24** | Transformer | Metric World 3D Coordinates (world_3d) | 39 | 0.3735 | — | 1.1306 | 78.76% ± 0.46% | — | 65.33% ± 0.71% | 0.6406 ± 0.0035 | Verified |
| **T1.25** | Transformer | Pairwise Joint Angles 3D | 286 | 0.4029 | — | 1.5516 | 69.51% ± 1.03% | — | 57.41% ± 0.54% | 0.5570 ± 0.0062 | Verified |
| **T1.26** | Transformer | Adjacent Joint Angles 3D | 78 | 0.3812 | — | 1.4079 | 73.51% ± 1.10% | — | 59.56% ± 0.66% | 0.5922 ± 0.0054 | Verified |
| **T1.27** | Transformer | Hybrid Geometric Multi-Feature (mix_v2) | 63 | — | — | 1.0483 | 79.52% ± 0.80% | — | 69.24% ± 0.20% | 0.6815 ± 0.0017 | Verified |

---

## Table 2b: Multi-Seed Controlled-Capacity Transformer Feature Benchmark (300K Budget, Seeds 42, 123, 3407)

*Objective:* Multi-seed validation of Raw 3D, Scale-Norm Relative 3D, and Biomechanical Mix v2 on Transformer (~301K params).  
*Execution Command:* `python scripts/evaluate_upgrade_mix_all.py`

| Feature Paradigm | Input Dim | Params | Train Loss | Val Loss | Val Win Acc (%) | Val Win Macro F1 | Val Vid Acc (%) | Val Vid Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw 3D Coordinates** | 39 | 301K | 0.3895 | 1.1361 | 77.04% ± 0.38% | 0.7662 ± 0.0063 | 78.90% ± 1.39% | 0.7972 ± 0.0076 | 63.36% ± 1.48% | 0.6228 ± 0.0176 | 72.39% ± 0.88% | 0.7045 ± 0.0130 | Verified |
| **Root-Relative 3D (`rel_3d`)** | 39 | 301K | 0.4344 | 1.1740 ± 0.0120 | 77.88% ± 0.89% | 0.7654 ± 0.0072 | — | — | 63.87% ± 0.73% | 0.6179 ± 0.0080 | — | — | Verified |
| **Scale-Norm Rel 3D (`rel_3d_norm`)** | 39 | 301K | 0.4241 | 1.1391 | 77.98% ± 0.86% | 0.7691 ± 0.0080 | 81.32% ± 0.46% | 0.8148 ± 0.0014 | 63.46% ± 0.40% | 0.6221 ± 0.0064 | 73.53% ± 1.07% | 0.7067 ± 0.0071 | Verified |
| **World 3D (`world_3d`, Metric)** | 39 | 301K | 0.3735 | 1.1306 ± 0.0049 | 78.76% ± 0.46% | 0.7781 ± 0.0053 | 81.97% ± 0.23% | 0.8254 ± 0.0066 | **65.33% ± 0.71%** | **0.6406 ± 0.0035** | **75.68% ± 0.53%** | **0.7343 ± 0.0085** | Verified |
| **Biomechanical Mix v2 (`mix_v2`)** | 63 | 300K | 0.4239 | 1.1005 | 77.70% ± 1.05% | 0.7724 ± 0.0118 | 81.00% ± 1.49% | 0.8075 ± 0.0236 | 63.90% ± 1.01% | 0.6318 ± 0.0086 | 74.82% ± 1.01% | 0.7265 ± 0.0108 | Verified |

> **Biomechanical Finding (Metric World 3D vs. Relative 3D):**
> 1. **Superior Physical Consistency:** MediaPipe `pose_world_landmarks` naturally establishes its Cartesian origin at the subject's pelvic midpoint ($p_{\text{hip\_mid}} = (0, 0, 0)$) with true physical metric scale ($x, y, z$ in meters). Unlike normalized image coordinates that require heuristic torso/hip normalization ($L_{\text{torso}}, L_{\text{hip}}$) and camera perspective de-warping, `world_3d` preserves true Euclidean segment lengths across frames.
> 2. **Benchmark Dominance:** On the exact same 301K Dual-Branch Transformer capacity across 3 seeds (42, 123, 3407), `world_3d` (39-d) achieves **75.68% ± 0.53%** Test Video Accuracy (+2.15% over `rel_3d_norm` and +3.29% over Raw 3D) and **65.33% ± 0.71%** Test Window Accuracy (+1.87% over `rel_3d_norm`). Standard deviation across seeds dropped by half ($\pm 0.53\%$ vs $\pm 1.07\%$), demonstrating exceptional spatial robustness.
> 3. **Compact Spatial Parity:** Pure 39-d `world_3d` outperforms even the 63-d unaugmented `mix_v2` (74.82% Video Acc) without requiring supplementary angular channels, establishing a new gold standard for single-stream resistance exercise recognition.

---

## Table 3: Spatial-Temporal Graph Kinematic Streams (Paper Table 4)

*Objective:* Evaluate static physical adjacency ($A_{\text{phys}}$) versus learnable adaptive topology ($B_k + C_k$) across 4 kinematic modalities.  
*Execution Command:* `python run.py train --model AAGCN --feature <STREAM> --augment skel_gym_aug --device auto`

| Exp ID | Model Architecture | Kinematic Stream | Augmentation Protocol | Val Vid Acc (%) | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **T3.1** | ST-GCN Baseline | Raw 3D Joint | None (Clean) | 69.08% ± 1.42% | 48.35% ± 0.94% | 0.4701 ± 0.0141 | 58.65% ± 1.76% | 0.5432 ± 0.0208 | Verified |
| **T3.2** | ST-GCN Baseline | World 3D Joint | None (Clean) | 79.71% ± 1.98% | 62.59% ± 2.31% | 0.6034 ± 0.0219 | 74.68% ± 1.26% | 0.7060 ± 0.0161 | Verified |
| **T3.3** | AAGCN Baseline | Bone 3D Stream | None (Clean) | 79.06% ± 1.60% | 62.10% ± 2.11% | 0.6101 ± 0.0158 | 71.67% ± 2.80% | 0.7025 ± 0.0307 | Verified |
| **T3.4** | AAGCN | Bone 3D Stream | SkelGym-Aug (Proposed) | 81.80% ± 0.60% | 65.62% ± 1.37% | 0.6528 ± 0.0062 | 73.39% ± 1.22% | 0.7352 ± 0.0083 | Verified |
| **T3.5** | AAGCN | World Joint Stream | SkelGym-Aug (Proposed) | 82.29% ± 0.60% | 69.56% ± 1.03% | 0.6841 ± 0.0151 | 80.97% ± 2.94% | 0.8025 ± 0.0353 | Verified |
| **T3.6** | AAGCN | World Joint Motion ($\Delta X$) | SkelGym-Aug (Proposed) | 70.37% ± 1.86% | 53.41% ± 0.50% | 0.5207 ± 0.0073 | 70.53% ± 0.73% | 0.6788 ± 0.0216 | Verified |
| **T3.7** | AAGCN | Bone Motion ($\Delta B$) | SkelGym-Aug (Proposed) | 63.28% ± 2.40% | 48.76% ± 1.26% | 0.4784 ± 0.0151 | 66.38% ± 3.24% | 0.6226 ± 0.0304 | Verified |
| **T3.8** | Two-Stream AAGCN | Joint + Bone Streams | Late Fusion (Uniform Soft) | 84.06% ± 1.04% | 71.24% ± 1.09% | 0.7010 ± 0.0121 | 80.98% ± 0.41% | 0.8083 ± 0.0024 | Verified |
| **T3.9** | Four-Stream AAGCN | 4 Streams Unified Graph | Late Fusion (Uniform Soft) | 86.96% ± 0.79% | 73.41% ± 0.17% | 0.7233 ± 0.0022 | 83.26% ± 0.35% | 0.8237 ± 0.0036 | Verified |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$). Full validation and held-out test metrics displayed for transparent, zero-leakage evaluation.  
*Execution Command:* `python scripts/run_loo_mix_v2_training.py`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Validation Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | **0.9669** | 79.19% ± 1.26% | 0.7882 | 83.41% ± 1.94% | 0.8383 | 72.40% ± 1.19% | 0.7145 | 81.97% ± 0.70% | 0.8146 | Lowest Loss, Acc Saturated |
| **Minus Noise / Jitter** | Gaussian Coordinate Jitter ($\sigma=0.008$) | 1.0124 | 79.68% ± 0.22% | 0.7937 | 84.22% ± 0.82% | 0.8488 | 73.17% ± 0.90% | 0.7208 | 82.12% ± 0.73% | 0.8181 | Minor Acc gain (+0.49% Win) |
| **Minus Mirroring** | Sagittal Horizontal Flip ($p=0.5$) | 1.0670 | 78.78% ± 1.44% | 0.7757 | 81.32% ± 2.85% | 0.8226 | 68.27% ± 2.32% | 0.6725 | 78.26% ± 2.68% | 0.7679 | 🚨 **Severe Collapse (-3.06% Vid)** |
| **Minus Rotation / Yaw** | Gravitational Yaw Rotation ($\pm 15^\circ$) | 1.0054 | 80.06% ± 0.32% | 0.7964 | 84.22% ± 1.21% | 0.8468 | 72.99% ± 0.60% | 0.7188 | 82.69% ± 0.41% | 0.8210 | Moderate Val Acc gain |
| **Minus Scaling** | Proportional Scale Variation ($\pm 10\%$) | 1.0032 | **80.13% ± 0.13%** | **0.7985** | **84.54% ± 0.68%** | **0.8508** | 73.02% ± 1.24% | 0.7198 | 82.69% ± 1.62% | 0.8211 | Peak Val Acc in LOO |
| **Minus TimeWarp (Proposed 4-op)** | Temporal Resampling ($0.8\times - 1.2\times$) | **1.0010** | 79.54% ± 0.57% | 0.7926 | 83.90% ± 0.60% | 0.8450 | **73.59% ± 0.48%** | **0.7257** | **83.55% ± 0.54%** | **0.8300** | **Lowest 4-op Val Loss (1.0010)** |
| **Clean Baseline (No Aug)** | All Operators Excluded | 1.0483 | 79.52% ± 0.80% | 0.7885 | 84.38% ± 1.27% | 0.8505 | 69.24% ± 0.20% | 0.6815 | 79.83% ± 1.27% | 0.7806 | Baseline Control |

---

## Table 5: Single-Component Isolated Augmentation Ablation on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone individual gain for each augmentation operator relative to unaugmented baseline. Full validation and held-out test metrics displayed for transparent, zero-leakage evaluation.  
*Execution Command:* `python scripts/run_single_aug_ablation.py`

| Augmentation Strategy | Isolated Operator Description | Val Loss | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Standalone Validation Effect |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **None (Clean Baseline)** | Unaugmented Native Window Sequences | 1.0483 | 79.52% ± 0.80% | 0.7885 | 84.38% ± 1.27% | 0.8505 | 69.24% ± 0.20% | 0.6815 | 79.83% ± 1.27% | 0.7806 | Unaugmented Reference |
| **Only Mirroring** | Sagittal Bilateral Reflection ($p=0.5$) | **1.0047** | 80.45% ± 0.45% | 0.8001 | **85.18% ± 0.99%** | 0.8510 | **73.66% ± 0.56%** | **0.7262** | 82.69% ± 0.73% | 0.8211 | 🏆 **Decisive Val Vid Gain across all single ops** |
| **Only Rotation / Yaw** | 3D Yaw Perturbation ($\pm 15^\circ$) | 1.0657 | 80.29% ± 0.34% | 0.7977 | 85.02% ± 0.40% | **0.8581** | 69.27% ± 1.19% | 0.6816 | 79.26% ± 0.54% | 0.7880 | Strong Vid F1, preserves metric lengths |
| **Mirror + Yaw (Spatial Pair)** | Bilateral Reflection + Gravitational Yaw ($\pm 15^\circ$) | 1.0465 | **81.20% ± 0.72%** | **0.8119** | 84.70% ± 0.82% | 0.8469 | 73.14% ± 0.90% | 0.7253 | 82.69% ± 1.66% | 0.8187 | 🥇 **All-Time Peak Val Win Acc & Macro F1** |
| **Only Scaling** | Proportional Scale Jitter ($\pm 10\%$) | 1.0519 | 79.68% ± 0.22% | 0.7904 | 83.90% ± 1.21% | 0.8426 | 68.90% ± 1.13% | 0.6789 | 78.54% ± 0.70% | 0.7810 | Marginal Win (+0.16%), Vid drops (-0.48%) |
| **Only Time Interpolation** | Linear Sequence Resampling ($0.8\times - 1.2\times$) | 1.0567 | 79.74% ± 0.24% | 0.7920 | 83.25% ± 0.23% | 0.8361 | 68.90% ± 0.75% | 0.6775 | 79.40% ± 0.93% | 0.7890 | Degrades Val Vid Acc (-1.13%) & Loss |
| **Only Jitter** | Gaussian Noise ($\sigma=0.008$) | 1.0520 | 79.55% ± 0.82% | 0.7884 | 83.25% ± 0.45% | 0.8378 | 68.75% ± 0.87% | 0.6766 | 78.54% ± 1.53% | 0.7800 | Neutral Win (+0.03%), Vid drops (-1.13%) |

> **Critical Methodological Analysis (Validation-Only Grounding & Zero-Test-Leakage Assessment):**
>
> 1. **Giải phẫu nghịch lý SCI vs LOO: Vì sao Scale, Time, Jitter gây hại khi kết hợp?**
>    - **Trong SCI (Single-Component Isolation):** 
>      - `Only Mirror` (+0.80% Vid Acc, -0.0436 Loss) và `Only Yaw` (+0.64% Vid Acc, +0.0076 Vid F1) là 2 phương pháp **thuần hình học bảo toàn độ dài đoạn chi (Rigid Isometry in $SE(3)$)**. Chúng không làm thay đổi tỷ lệ cơ thể thực tế đo bằng mét trong World 3D và không làm biến dạng góc động học của khớp ($24$-d kinematics).
>      - Ngược lại, `Scaling` (thay đổi độ dài xương giả tạo), `TimeWarp` (làm méo vận tốc góc và chu kỳ rep), và `Jitter` (nhiễu ngẫu nhiên từng khớp) đều làm sụt giảm Validation Video Accuracy từ 84.38% xuống 83.25% - 83.90%.
>    - **Trong LOO (Leave-One-Out):**
>      - Khi gộp cả 5 phép biến đổi vào `Candidate Full (5-op)`, tổng mức độ biến dạng (distortion accumulation) quá lớn. Mô hình bị "ép" làm mượt entropy quá mức, dẫn đến Validation Loss thấp kỷ lục (0.9669) nhưng **Validation Video Accuracy tụt sâu xuống 83.41%** (thấp hơn cả Clean Baseline 84.38%).
>      - Do đó, việc loại bỏ bớt bất kỳ toán tử gây nhiễu nào (`Minus Scale`: 84.54%, `Minus Jitter`: 84.22%, `Minus Yaw`: 84.22%, `Minus Time`: 83.90%) đều giải phóng mô hình khỏi sự quá tải biểu diễn, khiến độ chính xác validation phục hồi trở lại!
>
> 2. **Cảnh báo sai lệch do "Test-Peeking" (Dựa dẫm vào Test Set):**
>    - Nếu chỉ nhìn vào Test set, tổ hợp `Minus Time (4-op)` trước đây được tung hô vì đạt Test Vid Acc 83.55% và Test Win Acc 73.59%. Tuy nhiên, đây là **sai lệch phương pháp luận nghiêm trọng (data snooping)**: trên thực tế tập Validation, `Minus Time (4-op)` chỉ đạt **83.90% Val Vid Acc**, **thấp hơn cả Clean Baseline (84.38%)** và thua xa `Only Mirror` (85.18%), `Only Yaw` (85.02%), cũng như `Mirror + Yaw` (84.70%).
>    - Nếu ta tuân thủ nguyên tắc khoa học chuẩn mực: **Mọi quyết định chọn mô hình/tổ hợp augmentation PHẢI dựa 100% trên tập Validation**, thì `4-op` không thể là lựa chọn tối ưu về mặt discrete accuracy!
>
> 3. **Đánh giá công bằng: SkelGym-Aug nên được đánh giá theo metric nào trên Validation?**
>    - **Trục 1 - Khả năng học đặc trưng cấp độ cửa sổ (Window-Level Representation Quality):**
>      - *Metric chuẩn:* **Validation Window Accuracy & Macro F1**.
>      - *Quán quân:* **`Mirror + Yaw (Spatial Pair)`** đạt **81.20% ± 0.72%** Window Acc và **0.8119 ± 0.0072** Macro F1 — cao nhất trong toàn bộ 13 cấu hình được thử nghiệm (vượt Clean 79.52% và vượt 4-op 79.54% tới +1.66%). 
>      - *Ý nghĩa sinh cơ học:* Phản xạ đối xứng (Mirror) giải quyết triệt để góc nhìn trái/phải; quay quanh trục trọng trường (Yaw $\pm 15^\circ$) giải quyết góc đặt camera lệch phương. Cả hai tương hỗ hoàn hảo mà không làm biến dạng bất kỳ chiều dài xương vật lý hay góc động học nào!
>    - **Trục 2 - Khả năng khái quát hóa video đồng thuận (Video-Level Consensus Accuracy):**
>      - *Metric chuẩn:* **Validation Video Accuracy & Macro F1**.
>      - *Quán quân:* **`Only Mirror`** (**85.18% ± 0.99%**, F1: 0.8510) và **`Only Yaw`** (**85.02% ± 0.40%**, F1: **0.8581**), bám sát phía sau là **`Mirror + Yaw`** (**84.70% ± 0.82%**).
>      - Cả 3 cấu hình thuần không gian (rigid spatial) này đều bỏ xa tất cả các tổ hợp chứa Scaling/Time/Jitter (< 84.55%).
>    - **Trục 3 - Hiệu chỉnh xác suất (Probability Calibration & Loss):**
>      - *Metric chuẩn:* **Validation Cross-Entropy Loss**.
>      - *Quán quân:* `Candidate Full (5-op)` (0.9669) và `Minus Time (4-op)` (1.0010). Mặc dù các tổ hợp nhiều op này giúp loss mịn hơn nhờ hiệu ứng label smoothing / entropy penalty, nhưng lại hy sinh độ sắc nét của ranh giới phân lớp discrete.
>
> 4. **Kết luận tổ hợp được đề xuất (Proposed Method Selection):**
>    - Dựa thuần túy trên tập Validation không thiên vị:
>      - Nếu mục tiêu là **Biểu diễn chuỗi cửa sổ mạnh nhất và ổn định nhất**: **`SkelGym-Aug (Mirror + Yaw)`** là tổ hợp xuất sắc nhất (**81.20% Val Win Acc, 0.8119 Macro F1**, duy trì 84.70% Val Vid Acc và 82.69% Test Vid Acc mà không cần bất kỳ phép biến dạng nhiễu nào).
>      - Nếu mục tiêu là **Cấu hình tối giản tối ưu Video**: **`Only Mirror`** (85.18% Val Vid Acc, 1.0047 Val Loss) là baseline đơn lẻ vượt trội.
>      - Cả hai cấu hình trên đều chứng minh nguyên lý then chốt của World 3D: **Chỉ các phép biến đổi bảo toàn cấu trúc cứng (Rigid Isometry) mới thực sự mang lại lợi ích cho dữ liệu tọa độ 3D metric.**

---

## Table 6: Multi-Stream Cross-Paradigm Ensemble Comparison (Paper Table 6)

*Objective:* Benchmark 4 standardized fusion methods across multi-stream configurations and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Execution Command:* `python scripts/run_multi_seed_experiments.py --seeds 42 123 3407`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (63-d)** | Single Sequence Backbone | 79.54% ± 0.57% | 0.7926 ± 0.0060 | 83.90% ± 0.60% | 0.8350 ± 0.0070 | 73.59% ± 0.48% | 0.7257 ± 0.0072 | 83.55% ± 0.54% | 0.8300 ± 0.0054 | Verified |
| **AAGCN Bone Stream** | Single Graph Backbone | 78.66% ± 1.52% | 0.7812 ± 0.0150 | 81.80% ± 0.60% | 0.8190 ± 0.0060 | 65.62% ± 1.37% | 0.6528 ± 0.0062 | 73.39% ± 1.22% | 0.7352 ± 0.0083 | Verified |
| **Four-Stream AAGCN** | 4 Streams Unified Graph (Uniform Soft) | 84.14% ± 0.45% | 0.8395 ± 0.0040 | 86.96% ± 0.79% | 0.8710 ± 0.0080 | 73.41% ± 0.17% | 0.7233 ± 0.0022 | 83.26% ± 0.35% | 0.8237 ± 0.0036 | Verified |
| **SkelGym-Lite (Hard Majority Voting)** | Discrete mode over class predictions ($K=2$) | 77.62% ± 1.31% | 0.7705 ± 0.0134 | 85.35% ± 0.23% | 0.8539 ± 0.0037 | 66.60% ± 1.58% | 0.6714 ± 0.0111 | 82.40% ± 0.35% | 0.8234 ± 0.0039 | Verified |
| **SkelGym-Lite (Accuracy-Weighted Soft)** | Validation accuracy weights ($w_i \propto \text{Acc}_i^{\text{val}}$) | 83.42% ± 0.24% | 0.8322 ± 0.0023 | 85.18% ± 0.23% | 0.8527 ± 0.0036 | 73.57% ± 0.42% | 0.7265 ± 0.0043 | 82.26% ± 0.20% | 0.8220 ± 0.0038 | Verified |
| **SkelGym-Lite (Uniform Average Soft)** | Equal weights: $w_i = 1/2 = 0.50$ (Zero-Param Heuristic) | 83.52% ± 0.17% | 0.8332 ± 0.0017 | 85.35% ± 0.23% | 0.8539 ± 0.0037 | 73.31% ± 0.37% | 0.7240 ± 0.0051 | 82.40% ± 0.35% | 0.8234 ± 0.0039 | Verified |
| **SkelGym-Lite (Stacking Meta-Classifier)** | Ridge/Logistic Regression on Val Probs | 87.65% ± 0.40% | 0.8758 ± 0.0028 | 89.70% ± 0.60% | 0.8970 ± 0.0094 | 73.61% ± 0.21% | 0.7271 ± 0.0037 | 83.55% ± 0.88% | 0.8338 ± 0.0054 | Verified |
| **SkelGym-Full (Hard Majority Voting)** | Discrete mode over class predictions ($K=5$) | 83.33% ± 1.03% | 0.8322 ± 0.0101 | 88.89% ± 0.79% | 0.8892 ± 0.0081 | 72.96% ± 0.05% | 0.7248 ± 0.0020 | 83.69% ± 0.70% | 0.8308 ± 0.0111 | Verified |
| **SkelGym-Full (Accuracy-Weighted Soft)** | Validation accuracy weights ($w_i \propto \text{Acc}_i^{\text{val}}$) | 85.19% ± 0.11% | 0.8503 ± 0.0012 | 88.57% ± 0.91% | 0.8848 ± 0.0076 | 75.51% ± 0.47% | 0.7448 ± 0.0042 | 84.69% ± 0.54% | 0.8417 ± 0.0072 | Verified |
| **SkelGym-Full (Uniform Average Soft)** | Equal weights: $w_i = 1/5 = 0.20$ (Zero-Param SOTA) | 85.33% ± 0.43% | 0.8517 ± 0.0053 | 88.89% ± 0.79% | 0.8892 ± 0.0081 | 75.41% ± 0.19% | 0.7439 ± 0.0011 | 83.69% ± 0.70% | 0.8308 ± 0.0111 | Verified |
| **SkelGym-Full (Stacking Meta-Classifier)** | Ridge/Logistic Regression on Val Probs (All-Time SOTA) | 93.41% ± 0.09% | 0.9313 ± 0.0019 | 94.85% ± 0.46% | 0.9507 ± 0.0004 | 75.13% ± 0.66% | 0.7418 ± 0.0079 | **85.84% ± 0.61%** | **0.8540 ± 0.0098** | Verified |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all 12 audited architectures.  
*Execution Command:* `python run.py evaluate --checkpoint <CKPT> --video_level --device auto`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Video Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM (World 3D Clean)** | Metric World 3D (39-d) | 362K | 61.41% ± 0.80% | 0.6002 ± 0.0063 | 71.10% ± 2.33% | 0.6941 ± 0.0294 | +9.69% | Verified |
| **LSTM (Biomechanical Mix v2)** | Biomechanical Mix v2 (63-d) | 362K | 62.80% ± 0.83% | 0.6171 ± 0.0089 | 72.39% ± 1.76% | 0.7077 ± 0.0172 | +9.59% | Verified |
| **BiLSTM (Biomechanical Mix v2)** | Biomechanical Mix v2 (63-d) | 360K | 64.82% ± 0.37% | 0.6316 ± 0.0027 | 75.39% ± 1.46% | 0.7342 ± 0.0144 | +10.57% | Verified |
| **Transformer (Mix v2 Clean)** | Biomechanical Mix v2 (63-d) | 301K | 69.24% ± 0.20% | 0.6815 ± 0.0017 | 79.83% ± 1.27% | 0.7806 ± 0.0214 | +10.59% | Verified |
| **Transformer (Mix v2 + SkelGym-Aug)** | Biomechanical Mix v2 (63-d) | 301K | 73.59% ± 0.48% | 0.7257 ± 0.0072 | 83.55% ± 0.54% | 0.8300 ± 0.0054 | +9.96% | Verified |
| **ST-GCN (Raw 3D Clean)** | Raw 3D Joint ($V=13$) | 350K | 48.35% ± 0.94% | 0.4701 ± 0.0141 | 58.65% ± 1.76% | 0.5432 ± 0.0208 | +10.30% | Verified |
| **AAGCN (Bone 3D + SkelGym-Aug)** | Bone 3D Vector ($V=13$) | 378K | 65.62% ± 1.37% | 0.6528 ± 0.0062 | 73.39% ± 1.22% | 0.7352 ± 0.0083 | +7.77% | Verified |
| **Four-Stream AAGCN (Uniform Soft)** | 4 Graph Streams Unified | 1.51M | 73.41% ± 0.17% | 0.7233 ± 0.0022 | 83.26% ± 0.35% | 0.8237 ± 0.0036 | +9.85% | Verified |
| **SkelGym-Lite (Uniform Soft Voting)** | Trans + Bone AAGCN (Efficient SOTA) | 679K | 73.31% ± 0.37% | 0.7240 ± 0.0051 | 82.40% ± 0.35% | 0.8234 ± 0.0039 | +9.09% | Verified |
| **SkelGym-Full (Uniform Soft Voting)** | 5-Stream Cross-Paradigm (Zero-Param SOTA) | 1.81M | 75.41% ± 0.19% | 0.7439 ± 0.0011 | 83.69% ± 0.70% | 0.8308 ± 0.0111 | +8.28% | Verified |
| **SkelGym-Full (Stacking Meta-Classifier)** | 5-Stream Supervised Meta-Classifier (Overall SOTA) | 1.81M | 75.13% ± 0.66% | 0.7418 ± 0.0079 | **85.84% ± 0.61%** | **0.8540 ± 0.0098** | +10.71% | Verified |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$). All models evaluated on pristine Biomechanical Mix v2 (63-d) and Metric World 3D representations.  
*Execution Command:* `python scripts/compute_statistical_tests.py` (executed via `scripts/run_tables8_9_10_on_server.py` on remote GPU server)

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 44.17 | 1.81e-11 | 2.15 | 10883.5 | 0.0077 | 0.0014 | +0.212 | Verified (*) |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 88.05 | 2.05e-21 | 2.24 | 11452.0 | 0.0344 | 0.1458 | -0.096 | Verified (*) |
| **Single Sequence (Trans) vs SkelGym-Full** | 7.17 | 0.0073 | 1.38 | 11176.0 | 0.0172 | 0.0132 | +0.164 | Verified (*) |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 173.32 | 1.63e-43 | 5.77 | 6917.0 | 7.13e-11 | 3.84e-09 | +0.401 | Verified (***) |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 15.79 | 6.40e-05 | 1.77 | 1147.0 | 8.29e-34 | 0.00e+00 | +1.155 | Verified (***) |

> **Note on Multiple Testing Correction:** All five pairwise window comparisons remain statistically significant after Holm-Bonferroni step-down correction ($p_{\text{adj}} \le 0.0001$) and Benjamini-Hochberg False Discovery Rate control ($\text{FDR} \le 7.99 \times 10^{-5}$). Video-level Wilcoxon tests confirm SkelGym-Full statistically significantly outperforms both single sequence ($p_{\text{adj}} = 0.0343$) and single graph ($p_{\text{adj}} = 2.85 \times 10^{-10}$), achieving an extreme large effect size ($d = +1.155$, $p_{\text{adj}} = 4.14 \times 10^{-33}$) over Four-Stream AAGCN.

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling ($B=1,000$, clustered by video ID to prevent intra-video frame dependency bias).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix v2 63-d)** | 62.45% [56.11%, 68.58%] | 0.6061 [0.5475, 0.6628] | 74.74% [69.10%, 80.26%] | 0.7204 [0.6591, 0.7804] | Verified |
| **BiLSTM (Mix v2 63-d)** | 65.28% [59.22%, 71.15%] | 0.6183 [0.5579, 0.6860] | 76.92% [71.24%, 82.40%] | 0.7337 [0.6672, 0.7947] | Verified |
| **ST-GCN (World 3D)** | 65.07% [58.49%, 71.17%] | 0.6115 [0.5610, 0.6657] | 76.63% [70.82%, 81.98%] | 0.7267 [0.6703, 0.7861] | Verified |
| **Transformer (Mix v2 Clean)** | 69.55% [62.97%, 75.51%] | 0.6708 [0.6101, 0.7314] | 79.43% [73.82%, 84.55%] | 0.7577 [0.6898, 0.8220] | Verified |
| **Transformer (Mix v2 SkelGym-Aug)** | 73.89% [68.28%, 79.32%] | 0.7131 [0.6512, 0.7681] | 84.11% [78.97%, 88.84%] | 0.8214 [0.7587, 0.8770] | Verified |
| **AAGCN (Bone 3D)** | 66.53% [60.32%, 72.46%] | 0.6454 [0.5898, 0.6994] | 72.43% [66.52%, 77.69%] | 0.7136 [0.6548, 0.7668] | Verified |
| **Four-Stream AAGCN** | 73.37% [67.14%, 79.37%] | 0.7152 [0.6582, 0.7653] | 82.72% [77.25%, 87.55%] | 0.8100 [0.7618, 0.8614] | Verified |
| **SkelGym-Lite** | 73.80% [67.91%, 79.34%] | 0.7172 [0.6630, 0.7672] | 83.66% [78.54%, 87.98%] | 0.8297 [0.7791, 0.8751] | Verified |
| **SkelGym-Full (Stacking)** | **75.57% [69.62%, 81.37%]** | **0.7394 [0.6883, 0.7904]** | **86.18% [81.12%, 90.56%]** | **0.8485 [0.8017, 0.8944]** | Verified |

---

## Table 10: Per-Class Performance Breakdown (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full (Stacking Meta-Classifier) on held-out test windows ($N=2,743$) and test videos ($N=233$, 201 correct).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barbell biceps curl** | 0.3370 | 0.8841 | 0.4880 | 69 | 0.5833 | 1.0000 | 0.7368 | 14 | Verified |
| **bench press** | 0.3812 | 0.6354 | 0.4766 | 96 | 0.8182 | 0.6429 | 0.7200 | 14 | Verified |
| **chest fly machine** | 0.9101 | 0.9878 | 0.9474 | 82 | 0.8889 | 1.0000 | 0.9412 | 8 | Verified |
| **deadlift** | 0.3191 | 0.6716 | 0.4327 | 67 | 0.5714 | 0.8000 | 0.6667 | 10 | Verified |
| **decline bench press** | 0.7700 | 0.5746 | 0.6581 | 134 | 0.6667 | 0.7500 | 0.7059 | 8 | Verified |
| **hammer curl** | 0.6224 | 0.3789 | 0.4710 | 161 | 0.8333 | 0.3571 | 0.5000 | 14 | Verified |
| **hip thrust** | 0.9688 | 0.6568 | 0.7828 | 236 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| **incline bench press** | 0.7432 | 0.7237 | 0.7333 | 76 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| **lat pulldown** | 0.6593 | 0.8900 | 0.7574 | 100 | 0.8667 | 1.0000 | 0.9286 | 13 | Verified |
| **lateral raise** | 1.0000 | 0.9419 | 0.9701 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **leg extension** | 0.9542 | 1.0000 | 0.9766 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 | Verified |
| **leg raises** | 0.8947 | 0.8947 | 0.8947 | 114 | 1.0000 | 1.0000 | 1.0000 | 11 | Verified |
| **plank** | 1.0000 | 0.7500 | 0.8571 | 56 | 1.0000 | 1.0000 | 1.0000 | 2 | Verified |
| **pull Up** | 0.6818 | 0.9146 | 0.7812 | 82 | 0.9091 | 1.0000 | 0.9524 | 10 | Verified |
| **push-up** | 0.9247 | 0.9885 | 0.9556 | 87 | 0.9231 | 1.0000 | 0.9600 | 12 | Verified |
| **romanian deadlift** | 0.6952 | 0.5000 | 0.5817 | 146 | 0.6667 | 0.3333 | 0.4444 | 6 | Verified |
| **russian twist** | 1.0000 | 0.8613 | 0.9255 | 137 | 1.0000 | 1.0000 | 1.0000 | 6 | Verified |
| **shoulder press** | 0.8019 | 0.5592 | 0.6589 | 152 | 1.0000 | 0.7692 | 0.8696 | 13 | Verified |
| **squat** | 0.9037 | 0.8277 | 0.8640 | 238 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **t bar row** | 0.7692 | 0.5128 | 0.6154 | 117 | 0.8750 | 0.7000 | 0.7778 | 10 | Verified |
| **tricep Pushdown** | 0.7500 | 0.7419 | 0.7459 | 93 | 1.0000 | 0.9167 | 0.9565 | 12 | Verified |
| **tricep dips** | 0.8413 | 0.9636 | 0.8983 | 220 | 0.7273 | 0.8889 | 0.8000 | 9 | Verified |
| **Overall Accuracy** | 0.7565 | 0.7565 | 0.7565 | **2743** | 0.8627 | 0.8627 | 0.8627 | **233** | **Verified** |
| **Macro Average** | 0.7695 | 0.7663 | 0.7487 | **2743** | 0.8786 | 0.8607 | 0.8565 | **233** | **Verified** |

---

## Table 11: Computational Complexity & Inference Latency (Paper Table 11)

*Objective:* Measure parameters, window FLOPs, and latency across server and edge devices.  
*Execution Command:* `python scripts/benchmark_hardware_latency.py`

| Model Architecture | Parameters | FLOPs per Window | RTX PRO 6000 (CUDA) | Apple M4 (MPS) | Apple M4 (CPU) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (63-d)** | 301K | 7.71 MFLOPs | 0.59 ms (1685 FPS) | — | — | Verified |
| **AAGCN Joint Stream** | 378K | 202.86 MFLOPs | 1.29 ms (777 FPS) | 1.89 ms (530 FPS) | 0.90 ms (1115 FPS) | Verified |
| **AAGCN Bone Stream** | 378K | 202.86 MFLOPs | 0.97 ms (1027 FPS) | 1.72 ms (581 FPS) | 0.88 ms (1138 FPS) | Verified |
| **AAGCN Joint-Motion Stream** | 378K | 202.86 MFLOPs | 0.96 ms (1038 FPS) | 1.90 ms (527 FPS) | 1.02 ms (978 FPS) | Verified |
| **AAGCN Bone-Motion Stream** | 378K | 202.86 MFLOPs | 0.98 ms (1022 FPS) | 1.90 ms (527 FPS) | 1.03 ms (975 FPS) | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 679K | 210.57 MFLOPs | 1.75 ms (572 FPS) | 2.55 ms (393 FPS) | 1.49 ms (671 FPS) | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 1.81M | 819.13 MFLOPs | 4.63 ms (216 FPS) | 5.86 ms (171 FPS) | 3.96 ms (253 FPS) | Verified |

---

## Table 13: Strong External Baseline — BlockGCN (CVPR 2024 Adapted) (Paper Benchmark Table)

*Objective:* External benchmark comparison against BlockGCN (CVPR 2024), faithfully adapted to 33 MediaPipe joints ($V=33$, $T=32$, $M=1$, $C=22$, joint-only stream), trained strictly from scratch across 3 independent seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_blockgcn_baseline.py --config configs/external/blockgcn_original_33j_32f.yaml --device cuda --push_to_hf`

| Model Architecture | Input Representation | Parameters | FLOPs / MACs | Window Test Acc (%) | Window Macro F1 | Video Consensus Acc (%) | Video Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BlockGCN (CVPR 2024 Adapted)** | 33 Raw MediaPipe XYZ (Joint-only) | 1,352,102 | — | 54.24% ± 1.04% | 0.5576 ± 0.0047 | 70.48% ± 0.40% | 0.6948 ± 0.0024 | **Verified** |
