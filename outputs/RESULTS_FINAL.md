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
- **Multi-Stream Ensemble:** 5-stream cross-paradigm late fusion (Transformer on 63-d mix_v2 + 4 spatial-temporal AAGCN streams: Bone 3D, Relative 3D, Joint-Motion 3D, Bone-Motion 3D) calibrated via SLSQP on the validation partition.

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
| **T1.6** | LSTM | Root-Relative 3D Coordinates | 39 | 0.1548 | — | 1.6984 | 73.09% ± 1.02% | — | 58.77% ± 1.38% | 0.5738 ± 0.0127 | Verified |
| **T1.7** | LSTM | Pairwise Joint Angles 3D | 286 | 0.1426 | — | 2.3883 | 65.16% ± 0.97% | — | 50.68% ± 0.51% | 0.4923 ± 0.0040 | Verified |
| **T1.8** | LSTM | Adjacent Joint Angles 3D | 78 | 0.0943 | — | 1.9178 | 72.84% ± 1.06% | — | 57.42% ± 0.62% | 0.5680 ± 0.0097 | Verified |
| **T1.9** | LSTM | Hybrid Geometric Multi-Feature (mix_v2) | 63 | 0.0417 | — | 1.9940 | 77.78% ± 1.15% | — | 51.17% ± 13.08% | 0.5014 ± 0.1232 | Verified |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | 0.0747 | — | 2.2106 | 70.81% ± 0.73% | — | 54.28% ± 0.08% | 0.5338 ± 0.0033 | Verified |
| **T1.11** | BiLSTM | Root-Relative 2D Coordinates | 26 | 0.0790 | — | 1.8599 | 75.21% ± 0.91% | — | 58.95% ± 0.84% | 0.5700 ± 0.0101 | Verified |
| **T1.12** | BiLSTM | Pairwise Joint Angles 2D | 286 | 0.1284 | — | 2.5607 | 65.96% ± 0.47% | — | 52.37% ± 1.98% | 0.5043 ± 0.0206 | Verified |
| **T1.13** | BiLSTM | Adjacent Joint Angles 2D | 78 | 0.0623 | — | 2.5136 | 71.08% ± 0.17% | — | 56.53% ± 0.29% | 0.5508 ± 0.0072 | Verified |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | 0.1339 | — | 2.1974 | 68.19% ± 0.86% | — | 53.22% ± 0.90% | 0.5045 ± 0.0074 | Verified |
| **T1.15** | BiLSTM | Root-Relative 3D Coordinates | 39 | 0.1059 | — | 1.9566 | 73.24% ± 0.83% | — | 58.43% ± 0.99% | 0.5634 ± 0.0034 | Verified |
| **T1.16** | BiLSTM | Pairwise Joint Angles 3D | 286 | 0.1690 | — | 2.0893 | 64.47% ± 0.69% | — | 51.25% ± 1.82% | 0.4922 ± 0.0194 | Verified |
| **T1.17** | BiLSTM | Adjacent Joint Angles 3D | 78 | 0.1432 | — | 1.6524 | 73.19% ± 0.67% | — | 57.90% ± 1.11% | 0.5719 ± 0.0098 | Verified |
| **T1.18** | BiLSTM | Hybrid Geometric Multi-Feature (mix_v2) | 63 | 0.0828 | — | 1.6630 | 76.16% ± 2.19% | — | 49.75% ± 13.47% | 0.4921 ± 0.1297 | Verified |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | 0.3905 | — | 1.1568 | 77.75% ± 0.53% | — | 63.62% ± 1.06% | 0.6258 ± 0.0118 | Verified |
| **T1.20** | Transformer | Root-Relative 2D Coordinates | 26 | 0.5088 | — | 1.2146 | 75.44% ± 0.79% | — | 61.22% ± 1.61% | 0.5981 ± 0.0172 | Verified |
| **T1.21** | Transformer | Pairwise Joint Angles 2D | 286 | 0.4268 | — | 1.4610 | 69.85% ± 0.47% | — | 55.95% ± 0.74% | 0.5310 ± 0.0115 | Verified |
| **T1.22** | Transformer | Adjacent Joint Angles 2D | 78 | 0.4115 | — | 1.4490 | 70.15% ± 1.19% | — | 55.12% ± 1.68% | 0.5407 ± 0.0205 | Verified |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | 0.4239 | — | 1.1319 | 77.24% ± 0.59% | — | 62.87% ± 0.69% | 0.6202 ± 0.0097 | Verified |
| **T1.24** | Transformer | Root-Relative 3D Coordinates | 39 | 0.4344 | — | 1.1740 | 77.88% ± 0.89% | — | 63.87% ± 0.73% | 0.6179 ± 0.0080 | Verified |
| **T1.25** | Transformer | Pairwise Joint Angles 3D | 286 | 0.4029 | — | 1.5516 | 69.51% ± 1.03% | — | 57.41% ± 0.54% | 0.5570 ± 0.0062 | Verified |
| **T1.26** | Transformer | Adjacent Joint Angles 3D | 78 | 0.3812 | — | 1.4079 | 73.51% ± 1.10% | — | 59.56% ± 0.66% | 0.5922 ± 0.0054 | Verified |
| **T1.27** | Transformer | Hybrid Geometric Multi-Feature (mix_v2) | 63 | 0.3777 | — | 1.0535 | 80.98% ± 0.80% | — | 58.57% ± 12.11% | 0.5830 ± 0.1145 | Verified |

---

## Table 2b: Multi-Seed Controlled-Capacity Transformer Feature Benchmark (300K Budget, Seeds 42, 123, 3407)

*Objective:* Multi-seed validation of Raw 3D, Scale-Norm Relative 3D, and Biomechanical Mix v2 on Transformer (~301K params).  
*Execution Command:* `python scripts/evaluate_upgrade_mix_all.py`

| Feature Paradigm | Input Dim | Params | Train Loss | Val Loss | Val Win Acc (%) | Val Win Macro F1 | Val Vid Acc (%) | Val Vid Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw 3D Coordinates** | 39 | 301K | 0.3895 | 1.1361 | 77.04% ± 0.38% | 0.7662 ± 0.0063 | 78.90% ± 1.39% | 0.7972 ± 0.0076 | 63.36% ± 1.48% | 0.6228 ± 0.0176 | 72.39% ± 0.88% | 0.7045 ± 0.0130 | Verified |
| **Scale-Norm Rel 3D (`rel_3d_norm`)** | 39 | 301K | 0.4241 | 1.1391 | 77.98% ± 0.86% | 0.7691 ± 0.0080 | 81.32% ± 0.46% | 0.8148 ± 0.0014 | 63.46% ± 0.40% | 0.6221 ± 0.0064 | 73.53% ± 1.07% | 0.7067 ± 0.0071 | Verified |
| **Biomechanical Mix v2 (`mix_v2`)** | 63 | 300K | 0.4239 | 1.1005 | 77.70% ± 1.05% | 0.7724 ± 0.0118 | 81.00% ± 1.49% | 0.8075 ± 0.0236 | 63.90% ± 1.01% | 0.6318 ± 0.0086 | 74.82% ± 1.01% | 0.7265 ± 0.0108 | Verified |

---

## Table 3: Spatial-Temporal Graph Kinematic Streams (Paper Table 4)

*Objective:* Evaluate static physical adjacency ($A_{\text{phys}}$) versus learnable adaptive topology ($B_k + C_k$) across 4 kinematic modalities.  
*Execution Command:* `python run.py train --model AAGCN --feature <STREAM> --augment skel_gym_aug --device auto`

| Exp ID | Model Architecture | Kinematic Stream | Augmentation Protocol | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Vid Acc (%) | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T3.1** | ST-GCN Baseline | Raw 3D Joint | None (Clean) | — | — | — | 65.35% ± 1.16% | — | 53.03% ± 0.33% | 64.95% ± 1.33% | Verified |
| **T3.2** | ST-GCN Baseline | Relative 3D Joint | None (Clean) | — | — | — | 70.46% ± 0.45% | — | 57.58% ± 0.87% | 66.24% ± 1.32% | Verified |
| **T3.3** | AAGCN Baseline | Bone 3D Stream | None (Clean) | — | — | — | 76.21% ± 0.49% | — | 63.02% ± 1.85% | 71.67% ± 1.60% | Verified |
| **T3.4** | AAGCN | Bone 3D Stream | SkelGym-Aug (Proposed) | — | — | — | 78.78% ± 1.66% | — | 65.34% ± 1.26% | 73.10% ± 1.07% | Verified |
| **T3.5** | AAGCN | Joint Stream (Rel 3D) | SkelGym-Aug (Proposed) | — | — | — | 76.51% ± 0.56% | — | 67.46% ± 0.22% | 76.39% ± 1.75% | Verified |
| **T3.6** | AAGCN | Joint Motion 3D ($\Delta X$) | SkelGym-Aug (Proposed) | — | — | — | 54.76% ± 3.73% | — | 48.81% ± 0.27% | 65.66% ± 1.26% | Verified |
| **T3.7** | AAGCN | Bone Motion 3D ($\Delta B$) | SkelGym-Aug (Proposed) | — | — | — | 48.71% ± 0.92% | — | 49.17% ± 1.09% | 67.24% ± 2.63% | Verified |
| **T3.8** | Two-Stream AAGCN | Joint + Bone Streams | Late Fusion (Uniform Soft Voting) | — | — | — | 78.80% ± 0.50% | — | 69.57% ± 0.55% | 78.68% ± 0.41% | Verified |
| **T3.9** | Four-Stream AAGCN | 4 Streams Unified | Late Fusion (SLSQP Calibrated) | — | — | — | 79.40% ± 0.60% | — | 70.71% ± 0.93% | 80.40% ± 1.23% | Verified |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Full (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | 1.0923 | 78.06% ± 1.17% | 0.7741 ± 0.0134 | 66.80% ± 0.63% | 0.6574 ± 0.0066 | 0.00% (Ref) | Verified |
| **Minus Noise / Jitter** | Gaussian Coordinate Jitter ($\sigma=0.008$) | 1.1101 | 77.35% ± 1.17% | 0.7683 ± 0.0088 | 68.49% ± 1.57% | 0.6701 ± 0.0152 | +1.69% | Verified |
| **Minus Mirroring** | Sagittal Horizontal Flip ($p=0.5$) | 1.2502 | 76.31% ± 0.63% | 0.7588 ± 0.0099 | 62.80% ± 1.35% | 0.6164 ± 0.0115 | -4.00% | Verified |
| **Minus Rotation / Yaw** | Gravitational Yaw Rotation ($\pm 15^\circ$) | 1.1288 | 78.84% ± 0.60% | 0.7836 ± 0.0079 | 68.81% ± 2.10% | 0.6796 ± 0.0213 | +2.01% | Verified |
| **Minus Scaling** | Proportional Scale Variation ($\pm 10\%$) | 1.1061 | 78.38% ± 0.59% | 0.7776 ± 0.0088 | 69.23% ± 1.39% | 0.6810 ± 0.0127 | +2.43% | Verified |
| **Minus Time Interpolation** | Temporal Resampling ($0.8\times - 1.2\times$) | 1.0948 | 78.25% ± 0.04% | 0.7754 ± 0.0029 | 68.68% ± 2.04% | 0.6763 ± 0.0133 | +1.88% | Verified |

---

## Table 5: Single-Component Isolated Augmentation Ablation on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone individual gain for each augmentation operator relative to unaugmented baseline.  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain`

| Augmentation Strategy | Isolated Operator Description | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Clean Baseline)** | Unaugmented Native Window Sequences | 1.0535 | 80.98% ± 0.80% | 0.8040 ± 0.0090 | 66.11% ± 1.46% | 0.6536 ± 0.0147 | 0.00% (Ref) | Verified |
| **Only Jitter** | Gaussian Noise ($\sigma=0.008$) | 1.0988 | 79.94% ± 1.03% | 0.7940 ± 0.0112 | 65.10% ± 0.85% | 0.6418 ± 0.0061 | -1.01% | Verified |
| **Only Mirroring** | Sagittal Bilateral Reflection | 1.0059 | 81.11% ± 0.65% | 0.8054 ± 0.0089 | 69.29% ± 1.00% | 0.6906 ± 0.0143 | +3.18% | Verified |
| **Only Rotation** | 3D Yaw Perturbation ($\pm 15^\circ$) | 1.0790 | 80.16% ± 0.82% | 0.7973 ± 0.0105 | 66.20% ± 1.16% | 0.6537 ± 0.0134 | +0.10% | Verified |
| **Only Scaling** | Proportional Scale Jitter ($\pm 10\%$) | 1.0770 | 79.86% ± 1.54% | 0.7936 ± 0.0138 | 65.82% ± 0.74% | 0.6512 ± 0.0071 | -0.29% | Verified |
| **Only Time Interpolation** | Linear Sequence Resampling | 1.0739 | 79.69% ± 1.04% | 0.7900 ± 0.0115 | 65.79% ± 0.82% | 0.6521 ± 0.0102 | -0.32% | Verified |

---

## Table 6: Multi-Stream Cross-Paradigm Ensemble Comparison (Paper Table 6)

*Objective:* Benchmark 5 systematic fusion methods and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Execution Command:* `python scripts/run_multi_seed_experiments.py --seeds 42 123 3407`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (63-d)** | Single Sequence Backbone | 81.90% ± 0.81% | 0.8155 ± 0.0096 | 84.06% ± 2.56% | 0.8409 ± 0.0208 | 69.05% ± 2.22% | 0.6877 ± 0.0261 | 79.26% ± 2.12% | 0.7892 ± 0.0252 | Verified |
| **AAGCN Bone Stream** | Single Graph Backbone | 78.78% ± 2.03% | 0.7805 ± 0.0218 | 81.48% ± 1.12% | 0.8236 ± 0.0100 | 65.34% ± 1.55% | 0.6507 ± 0.0055 | 73.10% ± 1.31% | 0.7292 ± 0.0054 | Verified |
| **Four-Stream AAGCN** | 4 Streams Unified Graph | 80.74% ± 1.00% | 0.8005 ± 0.0113 | 82.13% ± 0.97% | 0.8299 ± 0.0078 | 68.67% ± 1.42% | 0.6799 ± 0.0112 | 78.40% ± 0.99% | 0.7777 ± 0.0121 | Verified |
| **Hard Majority Voting** | Discrete mode over class predictions | 82.52% ± 0.69% | 0.8224 ± 0.0098 | 84.38% ± 1.83% | 0.8465 ± 0.0187 | 70.57% ± 1.37% | 0.7023 ± 0.0157 | 81.40% ± 0.89% | 0.8043 ± 0.0114 | Verified |
| **Uniform Average Soft Voting** | Equal weights: $w_i = 1/5 = 0.20$ | 84.37% ± 0.46% | 0.8401 ± 0.0063 | 86.63% ± 1.01% | 0.8653 ± 0.0074 | 73.01% ± 1.41% | 0.7220 ± 0.0161 | 83.12% ± 0.99% | 0.8205 ± 0.0207 | Verified |
| **Accuracy-Weighted Soft Voting** | Validation accuracy weights ($w_i \propto \text{Acc}_i^{\text{val}}$) | 84.39% ± 0.80% | 0.8405 ± 0.0101 | 86.31% ± 1.55% | 0.8635 ± 0.0134 | 73.41% ± 1.59% | 0.7254 ± 0.0175 | 82.83% ± 1.72% | 0.8172 ± 0.0258 | Verified |
| **SLSQP Soft Voting** | SLSQP Constrained Calibration ($\sum w_i = 1$) | 83.63% ± 0.75% | 0.8334 ± 0.0092 | 85.51% ± 0.48% | 0.8547 ± 0.0063 | 72.45% ± 1.68% | 0.7164 ± 0.0181 | 81.40% ± 1.94% | 0.8103 ± 0.0262 | Verified |
| **Stacking Meta-Classifier** | Ridge Classifier on Val Probs | 93.62% ± 0.36% | 0.9336 ± 0.0038 | 95.65% ± 1.28% | 0.9571 ± 0.0135 | 73.37% ± 1.31% | 0.7242 ± 0.0153 | 83.83% ± 0.50% | 0.8336 ± 0.0109 | Verified |
| **SkelGym-Lite (2 Models)** | Trans + Bone AAGCN (SLSQP Calibrated) | 83.58% ± 0.82% | 0.8332 ± 0.0090 | 85.35% ± 0.56% | 0.8516 ± 0.0040 | 71.49% ± 1.66% | 0.7101 ± 0.0169 | 80.26% ± 2.27% | 0.8027 ± 0.0261 | Verified |
| **SkelGym-Full (5 Streams)** | Trans + 4 AAGCN (SLSQP Calibrated) | 83.63% ± 0.75% | 0.8334 ± 0.0092 | 85.51% ± 0.48% | 0.8547 ± 0.0063 | 72.45% ± 1.68% | 0.7164 ± 0.0181 | 81.40% ± 1.94% | 0.8103 ± 0.0262 | Verified |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all 11 architectures.  
*Execution Command:* `python run.py evaluate --checkpoint <CKPT> --video_level --device auto`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Video Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 63-d)** | Sequential Recurrent Model | 396K | 61.27% | 0.5943 | 71.67% | 0.6932 | +10.40% | Verified |
| **Baseline BiLSTM (Mix 63-d)** | Bidirectional Recurrent Model | 402K | 59.63% | 0.5831 | 69.10% | 0.6650 | +9.47% | Verified |
| **Transformer (Mix 63-d, Clean)** | Self-Attention Baseline | 400K | 66.11% | 0.6536 | 76.25% | 0.7536 | +10.14% | Verified |
| **Baseline ST-GCN (Rel 3D)** | Rigid Static Graph ($A_{\text{phys}}$) | 350K | 57.58% | 0.5574 | 66.24% | 0.6350 | +8.66% | Verified |
| **Clean Baseline AAGCN (Bone 3D)** | Adaptive Skeletal Graph (Unaugmented) | 378K | 63.02% | 0.6140 | 71.67% | 0.6985 | +8.65% | Verified |
| **SkelGym-Aug AAGCN (Bone 3D)** | Adaptive Skeletal Graph + Augmentation | 378K | 65.34% | 0.6507 | 73.10% | 0.7292 | +7.76% | Verified |
| **SkelGym-Aug Transformer (Mix)** | Self-Attention + Augmentation | 400K | 69.05% | 0.6877 | 79.26% | 0.7892 | +10.21% | Verified |
| **Two-Stream AAGCN (Aug)** | Joint + Bone Stream Fusion | 756K | 68.42% | 0.6770 | 78.40% | 0.7777 | +9.98% | Verified |
| **Four-Stream AAGCN (Aug)** | 4-Stream Graph Late Fusion | 1.51M | 68.67% | 0.6799 | 78.40% | 0.7777 | +9.73% | Verified |
| **SkelGym-Lite (2 Models)** | Transformer + Bone AAGCN | 778K | 71.49% | 0.7101 | 80.26% | 0.8027 | +8.77% | Verified |
| **SkelGym-Full (5 Streams)** | Cross-Paradigm SLSQP Ensemble | 1.91M | 72.45% | 0.7164 | 81.40% | 0.8103 | +8.95% | Verified |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 18.33 | 1.73e-05 | 1.57 | 9496.5 | 6.02e-05 | 7.60e-06 | +0.300 | Verified |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 88.64 | 1.60e-21 | 2.19 | 12775.0 | 0.4062 | 0.5831 | +0.036 | Verified |
| **Single Sequence (Trans) vs SkelGym-Full** | 43.69 | 2.11e-11 | 2.32 | 11218.0 | 0.0192 | 0.0820 | -0.114 | Verified |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 89.49 | 1.63e-22 | 3.82 | 12932.0 | 0.4977 | 0.2073 | +0.083 | Verified |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 44.35 | 1.27e-11 | 2.59 | 10516.0 | 0.0025 | 0.0366 | +0.138 | Verified |

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling.  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix 63-d)** | 62.98% [56.99%, 68.81%] | 0.5908 [0.5336, 0.6519] | 73.87% [67.80%, 79.40%] | 0.7172 [0.6560, 0.7782] | Verified |
| **BiLSTM (Mix 63-d)** | 60.36% [53.64%, 67.17%] | 0.5680 [0.5091, 0.6266] | 71.32% [65.24%, 76.82%] | 0.6677 [0.6070, 0.7329] | Verified |
| **ST-GCN (Rel 3D)** | 58.79% [52.74%, 64.58%] | 0.5509 [0.4901, 0.6095] | 67.01% [60.94%, 72.97%] | 0.6349 [0.5638, 0.7011] | Verified |
| **Transformer (Mix 63-d)** | 67.17% [60.65%, 73.36%] | 0.6510 [0.5938, 0.7103] | 80.21% [74.68%, 85.41%] | 0.7845 [0.7235, 0.8397] | Verified |
| **AAGCN (Bone 3D)** | 65.51% [59.10%, 71.70%] | 0.6382 [0.5795, 0.6926] | 73.32% [67.81%, 78.97%] | 0.7199 [0.6672, 0.7755] | Verified |
| **SkelGym-Lite (2 Models)** | 69.97% [63.50%, 76.07%] | 0.6810 [0.6220, 0.7356] | 79.32% [73.82%, 84.55%] | 0.7862 [0.7334, 0.8410] | Verified |
| **SkelGym-Full (5 Streams)** | 71.17% [64.82%, 77.28%] | 0.6909 [0.6321, 0.7466] | 79.31% [73.82%, 84.55%] | 0.7743 [0.7165, 0.8317] | Verified |

---

## Table 10: Per-Class Performance Breakdown (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full on held-out test windows ($N=2,743$) and test videos ($N=233$).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barbell biceps curl** | 0.3226 | 0.8696 | 0.4706 | 69 | 0.5385 | 1.0000 | 0.7000 | 14 | Verified |
| **bench press** | 0.3571 | 0.6771 | 0.4676 | 96 | 0.6429 | 0.6429 | 0.6429 | 14 | Verified |
| **chest fly machine** | 0.9186 | 0.9634 | 0.9405 | 82 | 0.8750 | 0.8750 | 0.8750 | 8 | Verified |
| **deadlift** | 0.2980 | 0.6716 | 0.4128 | 67 | 0.6429 | 0.9000 | 0.7500 | 10 | Verified |
| **decline bench press** | 0.5725 | 0.5597 | 0.5660 | 134 | 0.4167 | 0.6250 | 0.5000 | 8 | Verified |
| **hammer curl** | 0.5888 | 0.3913 | 0.4701 | 161 | 0.8000 | 0.2857 | 0.4211 | 14 | Verified |
| **hip thrust** | 0.9038 | 0.5975 | 0.7194 | 236 | 0.8750 | 0.7778 | 0.8235 | 9 | Verified |
| **incline bench press** | 0.7759 | 0.5921 | 0.6716 | 76 | 1.0000 | 0.6667 | 0.8000 | 9 | Verified |
| **lat pulldown** | 0.6115 | 0.9600 | 0.7471 | 100 | 0.6842 | 1.0000 | 0.8125 | 13 | Verified |
| **lateral raise** | 0.9012 | 0.9419 | 0.9211 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **leg extension** | 0.9843 | 1.0000 | 0.9921 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 | Verified |
| **leg raises** | 0.9518 | 0.6930 | 0.8020 | 114 | 1.0000 | 0.7273 | 0.8421 | 11 | Verified |
| **plank** | 0.7400 | 0.6607 | 0.6981 | 56 | 0.6667 | 1.0000 | 0.8000 | 2 | Verified |
| **pull Up** | 0.7059 | 0.7317 | 0.7186 | 82 | 0.8750 | 0.7000 | 0.7778 | 10 | Verified |
| **push-up** | 0.7748 | 0.9885 | 0.8687 | 87 | 0.9231 | 1.0000 | 0.9600 | 12 | Verified |
| **romanian deadlift** | 0.5567 | 0.3699 | 0.4444 | 146 | 0.5000 | 0.3333 | 0.4000 | 6 | Verified |
| **russian twist** | 0.9760 | 0.8905 | 0.9313 | 137 | 1.0000 | 1.0000 | 1.0000 | 6 | Verified |
| **shoulder press** | 0.6832 | 0.4539 | 0.5455 | 152 | 0.7500 | 0.4615 | 0.5714 | 13 | Verified |
| **squat** | 0.9026 | 0.7395 | 0.8129 | 238 | 1.0000 | 0.9333 | 0.9655 | 15 | Verified |
| **t bar row** | 0.8286 | 0.4957 | 0.6203 | 117 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| **tricep Pushdown** | 0.6952 | 0.7849 | 0.7374 | 93 | 0.7857 | 0.9167 | 0.8462 | 12 | Verified |
| **tricep dips** | 0.9174 | 0.9091 | 0.9132 | 220 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| **Overall Accuracy** | 0.7124 | 0.7124 | 0.7124 | **2743** | 0.7940 | 0.7940 | 0.7940 | **233** | **Verified** |
| **Macro Average** | 0.7257 | 0.7246 | 0.7032 | **2743** | 0.8171 | 0.7925 | 0.7842 | **233** | **Verified** |

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
