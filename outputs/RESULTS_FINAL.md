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

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Full (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | 0.9669 | 79.19% ± 1.26% | 0.7882 ± 0.0152 | 72.40% ± 1.19% | 0.7145 ± 0.0083 | 0.00% (Ref) | Verified |
| **Minus Noise / Jitter** | Gaussian Coordinate Jitter ($\sigma=0.008$) | 1.0124 | 79.68% ± 0.22% | 0.7937 ± 0.0048 | 73.17% ± 0.90% | 0.7208 ± 0.0125 | +0.77% | Verified |
| **Minus Mirroring** | Sagittal Horizontal Flip ($p=0.5$) | 1.0670 | 78.78% ± 1.44% | 0.7757 ± 0.0197 | 68.27% ± 2.32% | 0.6725 ± 0.0242 | -4.13% | Verified |
| **Minus Rotation / Yaw** | Gravitational Yaw Rotation ($\pm 15^\circ$) | 1.0054 | 80.06% ± 0.32% | 0.7964 ± 0.0026 | 72.99% ± 0.60% | 0.7188 ± 0.0085 | +0.59% | Verified |
| **Minus Scaling** | Proportional Scale Variation ($\pm 10\%$) | 1.0032 | 80.13% ± 0.13% | 0.7985 ± 0.0022 | 73.02% ± 1.24% | 0.7198 ± 0.0123 | +0.62% | Verified |
| **Minus Time Interpolation** | Temporal Resampling ($0.8\times - 1.2\times$, **SkelGym-Aug 4-op**) | 1.0010 | 79.54% ± 0.57% | 0.7926 ± 0.0060 | 73.59% ± 0.48% | 0.7257 ± 0.0072 | +1.19% | Verified |
| **Clean Baseline (No Augmentation)** | All Operators Excluded | 1.0483 | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 69.24% ± 0.20% | 0.6815 ± 0.0017 | -3.16% | Verified |

---

## Table 5: Single-Component Isolated Augmentation Ablation on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone individual gain for each augmentation operator relative to unaugmented baseline.  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain`

| Augmentation Strategy | Isolated Operator Description | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Clean Baseline)** | Unaugmented Native Window Sequences | 1.0483 | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 69.24% ± 0.20% | 0.6815 ± 0.0017 | 0.00% (Ref) | Verified |
| **Only Jitter** | Gaussian Noise ($\sigma=0.008$) | 1.0520 | 79.55% ± 0.82% | 0.7884 ± 0.0093 | 68.75% ± 0.87% | 0.6766 ± 0.0097 | -0.49% | Verified |
| **Only Mirroring** | Sagittal Bilateral Reflection | 1.0047 | 80.45% ± 0.45% | 0.8001 ± 0.0067 | 73.66% ± 0.56% | 0.7262 ± 0.0078 | +4.42% | Verified |
| **Only Rotation** | 3D Yaw Perturbation ($\pm 15^\circ$) | 1.0657 | 80.29% ± 0.34% | 0.7977 ± 0.0031 | 69.27% ± 1.19% | 0.6816 ± 0.0135 | +0.03% | Verified |
| **Only Scaling** | Proportional Scale Jitter ($\pm 10\%$) | 1.0519 | 79.68% ± 0.22% | 0.7904 ± 0.0066 | 68.90% ± 1.13% | 0.6789 ± 0.0125 | -0.34% | Verified |
| **Only Time Interpolation** | Linear Sequence Resampling | 1.0567 | 79.74% ± 0.24% | 0.7920 ± 0.0039 | 68.90% ± 0.75% | 0.6775 ± 0.0075 | -0.34% | Verified |

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
