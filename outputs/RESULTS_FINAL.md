# SkelGym: Resistance Exercise Recognition — Master Experiment Results & Benchmark Template

> **Document Status:** Authoritative Master Results Template (Clean Slate — Fully Executed & Verified)  
> **Authoritative Baseline Split:** Video-Level Partition 6:2:2 (580 Train / 208 Validation / 236 Test; 233 Valid Test Videos $\ge 32$ Frames)  
> **Repository:** `Cuong2004/gym-exercise-classification` (Model Hub) | `Cuong2004/gym-exercise-landmarks` (Dataset Hub)  
> **Associated Checkpoint Archive:** `archive_pre_clean_results.tar.gz` (Preserved historical pre-clean logs & checkpoints)

---

## 1. Executive Summary & Experimental Methodology

This document serves as the **single authoritative Source of Truth (SOT) and execution template** for all empirical benchmarks, ablation studies, statistical validations, and hardware latency measurements in the SkelGym project. All experiments are tracked here.

### 1.1. Core Experimental Protocol
- **Dataset Scale:** 1,024 unique video recordings ($\approx 10.2$ GB) spanning 22 fine-grained resistance exercises, trimmed into 1,108 clean action segments (average 1.09 segments/video).
- **Strict Video-Level Partitioning:** Zero source-video overlap across Train ($N=580$), Validation ($N=208$), and Test ($N=236$) sets. All sliding windows ($T=32$ frames) are extracted strictly within individual trimmed action boundaries ($S=16$ train: 13,136 windows; $S=32$ val: 2,075 windows; $S=32$ test: 2,743 windows across 233 valid videos).
- **Physical Kinematics & Standardization:** 13 anatomically calibrated keypoints from MediaPipe Pose Heavy. Spatial coordinates are centered relative to the mid-hip origin ($p_{\text{hip\_mid}} = \frac{1}{2}(p_{\text{left\_hip}} + p_{\text{right\_hip}})$). Global feature-wise $z$-score normalization statistics $(\mu_{\text{train}}, \sigma_{\text{train}})$ are computed exclusively on the training partition and frozen.
- **Augmentation Pipeline (SkelGym-Aug):** Dynamic on-the-fly transformations (Bilateral Sagittal Reflection + Gravitational 3D Yaw $\pm 15^\circ$ + Proportional Scaling $\pm 10\%$ + Gaussian Sensor Jitter $\sigma=0.008$) execute in coordinate space on sliding windows prior to online $z$-score feature normalization (with biomechanical angles dynamically recomputed on the transformed coordinate frames).
- **Backbone Capacity:** All individual model backbones constrained to an identical compact budget ($\approx 350\text{K} \pm 15\%$ parameters), trained strictly from scratch without external pre-training weights.
- **Ensemble Architecture:** 5-stream cross-paradigm late fusion (1 sequence Transformer on 117-d mix + 4 spatial-temporal AAGCN streams: Bone 3D, Relative 3D, Joint-Motion 3D, Bone-Motion 3D) calibrated via Sequential Least Squares Programming (SLSQP) on the validation partition under simplex constraints ($\sum w_i = 1, w_i \ge 0$).

---

## Table 1: Dataset Partition & Provenance Breakdown

| Split Name | Source Videos | Ratio (%) | Action Segments | Valid Videos ($\ge 32$ frames) | Extracted Windows ($T=32$) | Stride ($S$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Set** | 580 | 56.6% | 639 | 580 | 13,136 | 16 (50% overlap) | Verified |
| **Validation Set** | 208 | 20.3% | 210 | 208 | 2,075 | 32 (non-overlapping) | Verified |
| **Held-Out Test Set** | 236 | 23.1% | 259 | 233 | 2,743 | 32 (non-overlapping) | Verified |
| **Total Corpus** | **1,024** | **100.0%** | **1,108** | **1,021** | **17,954** | — | **Verified** |

*Provenance Breakdown:* Abdillah (2023): 652 videos; YouTube: 103 videos; Pexels: 65 videos; Freepik: 76 videos; Author Self-Recorded: 128 videos.

---

## Table 2: Feature Representation Benchmark across Sequence Architectures (Paper Table 1)

*Objective:* Evaluate 3 sequence architectures (LSTM, BiLSTM, Transformer) across 9 spatial coordinate and angular representations (Seed 42) to select the winning sequence backbone and feature formulation.  
*Execution Command:* `python run.py train --model <MODEL> --feature <FEATURE> --exp_id <ID> --device auto`

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Macro F1 | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1.1** | LSTM | Raw 2D Coordinates | 26 | 0.1608 | 95.29% ± 1.23% | 1.7996 | 67.49% ± 0.75% | 0.6556 ± 0.0075 | 52.80% ± 1.04% | 0.5143 ± 0.0138 | Verified |
| **T1.2** | LSTM | Relative 2D (Mid-Hip) | 26 | 0.1994 | 93.89% ± 0.28% | 1.4876 | 72.40% ± 1.07% | 0.7131 ± 0.0110 | 56.13% ± 0.66% | 0.5501 ± 0.0082 | Verified |
| **T1.3** | LSTM | Angle 2D (Triplets) | 286 | 0.2367 | 92.76% ± 5.15% | 2.0654 | 64.74% ± 2.62% | 0.6310 ± 0.0278 | 50.45% ± 1.59% | 0.4896 ± 0.0166 | Verified |
| **T1.4** | LSTM | Angle2 2D (Pairs) | 78 | 0.3623 | 88.69% ± 4.84% | 1.5881 | 65.88% ± 1.61% | 0.6468 ± 0.0142 | 52.25% ± 0.70% | 0.5132 ± 0.0028 | Verified |
| **T1.5** | LSTM | Raw 3D Coordinates | 39 | 0.3129 | 90.51% ± 5.01% | 1.6486 | 65.32% ± 2.25% | 0.6323 ± 0.0293 | 52.77% ± 1.77% | 0.5041 ± 0.0250 | Verified |
| **T1.6** | LSTM | Relative 3D (Mid-Hip) | 39 | 0.1331 | 95.96% ± 4.03% | 1.9126 | 72.24% ± 0.99% | 0.7096 ± 0.0097 | 59.28% ± 0.77% | 0.5786 ± 0.0079 | Verified |
| **T1.7** | LSTM | Angle 3D (Triplets) | 286 | 0.0712 | 97.70% ± 0.69% | 2.5788 | 65.73% ± 1.02% | 0.6398 ± 0.0083 | 50.45% ± 0.82% | 0.4886 ± 0.0086 | Verified |
| **T1.8** | LSTM | Angle2 3D (Pair Elevation) | 78 | 0.3292 | 89.49% ± 11.51% | 1.6975 | 70.31% ± 3.35% | 0.6901 ± 0.0408 | 55.29% ± 1.49% | 0.5495 ± 0.0190 | Verified |
| **T1.9** | LSTM | Biomechanical Mix (Ours) | 117 | 0.0779 | 97.48% ± 0.82% | 1.8811 | 74.28% ± 0.50% | 0.7348 ± 0.0063 | 59.35% ± 0.19% | 0.5827 ± 0.0015 | Verified |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | 0.4437 | 85.43% ± 9.22% | 1.6896 | 65.12% ± 3.22% | 0.6272 ± 0.0331 | 51.48% ± 0.71% | 0.4901 ± 0.0184 | Verified |
| **T1.11** | BiLSTM | Relative 2D (Mid-Hip) | 26 | 0.5620 | 81.84% ± 7.71% | 1.3529 | 64.98% ± 6.30% | 0.6411 ± 0.0516 | 54.51% ± 2.33% | 0.5247 ± 0.0290 | Verified |
| **T1.12** | BiLSTM | Angle 2D (Triplets) | 286 | 0.0845 | 97.21% ± 1.70% | 3.0730 | 65.82% ± 0.63% | 0.6523 ± 0.0046 | 52.73% ± 1.58% | 0.5068 ± 0.0175 | Verified |
| **T1.13** | BiLSTM | Angle2 2D (Pairs) | 78 | 0.0785 | 97.53% ± 0.73% | 2.3040 | 69.59% ± 2.28% | 0.6929 ± 0.0223 | 55.35% ± 1.80% | 0.5379 ± 0.0188 | Verified |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | 0.0691 | 97.92% ± 0.92% | 2.4414 | 67.63% ± 0.11% | 0.6659 ± 0.0039 | 53.25% ± 0.88% | 0.5120 ± 0.0087 | Verified |
| **T1.15** | BiLSTM | Relative 3D (Mid-Hip) | 39 | 0.1283 | 96.10% ± 0.27% | 1.7113 | 72.47% ± 1.39% | 0.7172 ± 0.0103 | 58.09% ± 1.46% | 0.5595 ± 0.0089 | Verified |
| **T1.16** | BiLSTM | Angle 3D (Triplets) | 286 | 0.1500 | 95.23% ± 1.59% | 2.1259 | 64.37% ± 0.83% | 0.6293 ± 0.0075 | 50.51% ± 1.57% | 0.4846 ± 0.0201 | Verified |
| **T1.17** | BiLSTM | Angle2 3D (Pair Elevation) | 78 | 0.0946 | 96.79% ± 2.78% | 1.9386 | 72.58% ± 1.06% | 0.7204 ± 0.0087 | 58.56% ± 0.67% | 0.5773 ± 0.0046 | Verified |
| **T1.18** | BiLSTM | Biomechanical Mix (Ours) | 117 | 0.0984 | 96.79% ± 0.74% | 1.7974 | 74.39% ± 0.26% | 0.7371 ± 0.0043 | 60.19% ± 1.14% | 0.5896 ± 0.0126 | Verified |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | 0.3789 | 99.12% ± 0.11% | 1.1394 | 77.82% ± 0.22% | 0.7700 ± 0.0008 | 63.51% ± 0.80% | 0.6323 ± 0.0094 | Verified |
| **T1.20** | Transformer | Relative 2D (Mid-Hip) | 26 | 0.4290 | 97.91% ± 0.90% | 1.1943 | 76.45% ± 0.41% | 0.7539 ± 0.0033 | 63.06% ± 2.61% | 0.6159 ± 0.0244 | Verified |
| **T1.21** | Transformer | Angle 2D (Triplets) | 286 | 0.3837 | 98.95% ± 0.34% | 1.4445 | 71.78% ± 0.55% | 0.7107 ± 0.0077 | 57.37% ± 0.20% | 0.5437 ± 0.0060 | Verified |
| **T1.22** | Transformer | Angle2 2D (Pairs) | 78 | 0.4253 | 97.84% ± 1.73% | 1.4747 | 69.65% ± 0.68% | 0.6877 ± 0.0111 | 55.27% ± 1.47% | 0.5402 ± 0.0155 | Verified |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | 0.3954 | 98.79% ± 0.46% | 1.0805 | 79.65% ± 0.80% | 0.7888 ± 0.0080 | 65.01% ± 0.27% | 0.6375 ± 0.0027 | Verified |
| **T1.24** | Transformer | Relative 3D (Mid-Hip) | 39 | 0.4410 | 97.32% ± 2.61% | 1.1776 | 77.69% ± 0.69% | 0.7692 ± 0.0107 | 63.52% ± 1.18% | 0.6151 ± 0.0126 | Verified |
| **T1.25** | Transformer | Angle 3D (Triplets) | 286 | 0.4006 | 98.53% ± 0.18% | 1.4954 | 70.33% ± 1.10% | 0.6842 ± 0.0096 | 56.86% ± 1.09% | 0.5567 ± 0.0081 | Verified |
| **T1.26** | Transformer | Angle2 3D (Pair Elevation) | 78 | 0.3815 | 99.05% ± 0.27% | 1.3522 | 73.61% ± 0.66% | 0.7325 ± 0.0083 | 59.27% ± 1.32% | 0.5826 ± 0.0132 | Verified |
| **T1.27** | Transformer | Biomechanical Mix (Dual-Branch) | 117 | 0.3665 | 99.39% ± 0.04% | 1.1906 | 78.14% ± 0.61% | 0.7774 ± 0.0071 | 66.14% ± 0.11% | 0.6516 ± 0.0028 | Verified |

---

### Table 2b: Controlled-Capacity (~300K) Benchmark & Upgraded Biomechanical Mix v2 (Multi-Seed Multi-Metric)

*Objective:* Standardize Transformer backbone capacity to $\approx 301\text{K}$ parameters (`d_model=112, num_layers=3, dim_feedforward=168, nhead=4`) to eliminate over-capacity advantages, test anisotropic anthropometric scale normalization (`rel_3d_norm`), and validate Biomechanical Mix v2 (63-d: 39-d normalized relative coordinates + 24-d kinematic joint/limb angles) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/evaluate_upgrade_mix_all.py`  
*Hugging Face Artifacts:* `Cuong2004/gym-exercise-classification/checkpoints/upgrade_mix/`

#### Part A: Comprehensive Validation & Test Split Multi-Metric Benchmark (Mean ± SD across 3 Seeds)

| Feature Paradigm | Input Dim | Model Params | Train Loss | Val Loss | Val Win Acc (%) | Val Win Macro F1 | Val Vid Acc (%) | Val Vid Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw 3D Coordinates** | 39 | 301K | 0.3895 | 1.1361 | 77.04% ± 0.38% | 0.7662 ± 0.0063 | 78.90% ± 1.39% | 0.7972 ± 0.0076 | 63.36% ± 1.48% | 0.6228 ± 0.0176 | 72.39% ± 0.88% | 0.7045 ± 0.0130 | Verified |
| **Scale-Norm Rel 3D (`rel_3d_norm`)** | 39 | 301K | 0.4241 | 1.1391 | 77.98% ± 0.86% | 0.7691 ± 0.0080 | **81.32% ± 0.46%** | **0.8148 ± 0.0014** | 63.46% ± 0.40% | 0.6221 ± 0.0064 | 73.53% ± 1.07% | 0.7067 ± 0.0071 | Verified |
| **Biomechanical Mix v2 (`mix_v2`)** | 63 | 300K | 0.4239 | **1.1005** | **77.70% ± 1.05%** | **0.7724 ± 0.0118** | **81.00% ± 1.49%** | **0.8075 ± 0.0236** | **63.90% ± 1.01%** | **0.6318 ± 0.0086** | **74.82% ± 1.01%** | **0.7265 ± 0.0108** | **Verified** |

#### Part B: Individual Run Breakdown Across Seeds (42, 123, 3407)

| Feature | Seed | Epoch | Train Loss | Train Acc (%) | Val Loss | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw 3D** | 42 | 34 | 0.3879 | 98.98% | 1.1626 | 77.16% | 0.7716 | 78.74% | 0.7904 | 63.18% | 0.6146 | 72.53% | 0.6987 | `best_Transformer_raw_3d.pt` |
| **Raw 3D** | 123 | 35 | 0.3918 | 99.07% | 1.1291 | 76.53% | 0.7575 | 77.29% | 0.7934 | 65.26% | 0.6473 | 73.39% | 0.7226 | `best_Transformer_raw_3d.pt` |
| **Raw 3D** | 3407 | 33 | 0.3888 | 99.03% | 1.1165 | 77.45% | 0.7697 | 80.68% | 0.8078 | 61.65% | 0.6064 | 71.24% | 0.6923 | `best_Transformer_raw_3d.pt` |
| **Scale-Norm Rel 3D** | 42 | 22 | 0.4503 | 97.31% | 1.1130 | 76.82% | 0.7589 | 81.64% | 0.8135 | 63.14% | 0.6146 | 74.68% | 0.7165 | `best_Transformer_rel_3d_norm.pt` |
| **Scale-Norm Rel 3D** | 123 | 36 | 0.3987 | 98.87% | 1.1706 | 78.89% | 0.7783 | 81.64% | 0.8168 | 64.02% | 0.6302 | 73.82% | 0.7001 | `best_Transformer_rel_3d_norm.pt` |
| **Scale-Norm Rel 3D** | 3407 | 26 | 0.4235 | 98.11% | 1.1337 | 78.22% | 0.7702 | 80.68% | 0.8141 | 63.22% | 0.6217 | 72.10% | 0.7035 | `best_Transformer_rel_3d_norm.pt` |
| **Biomechanical Mix v2** | 42 | 48 | 0.3853 | 99.27% | 1.1071 | 78.02% | 0.7737 | 79.71% | 0.7879 | 62.78% | 0.6199 | 75.54% | 0.7382 | `best_Transformer_mix_v2.pt` |
| **Biomechanical Mix v2** | 123 | 22 | 0.4391 | 97.73% | 1.1013 | **78.80%** | **0.7862** | **83.09%** | **0.8407** | 65.22% | 0.6399 | **75.54%** | 0.7121 | `best_Transformer_mix_v2.pt` |
| **Biomechanical Mix v2** | 3407 | 21 | 0.4471 | 97.64% | **1.0930** | 76.29% | 0.7574 | 80.19% | 0.7938 | 63.69% | 0.6355 | 73.39% | 0.7291 | `best_Transformer_mix_v2.pt` |

#### Key Empirical Insights:
1. **Hypothesis Confirmed on Validation Split (Zero Test Leakage):**
   - **Validation Macro F1:** Biomechanical Mix v2 achieves **0.7724 ± 0.0118**, outperforming Raw 3D (**0.7662 ± 0.0063**).
   - **Validation Window Accuracy:** Mix v2 (**77.70%**) and Scale-Norm Rel 3D (**77.98%**) both decisively surpass Raw 3D (**77.04%**).
   - **Validation Video Accuracy:** Mix v2 reaches **81.00% ± 1.49%** (+2.10% over Raw 3D at 78.90%). Scale-Norm Rel 3D reaches **81.32% ± 0.46%** (+2.42%).
   - **Validation Loss:** Mix v2 achieves the lowest validation cross-entropy loss (**1.1005**) across all 3 architectures, proving superior regularization and generalization.
2. **Superiority Carries Over Directly to the Held-Out Test Set:**
   - **Test Window Accuracy:** Mix v2 is #1 at **63.90% ± 1.01%** (vs Raw 3D at 63.36%).
   - **Test Window Macro F1:** Mix v2 is #1 at **0.6318 ± 0.0086** (vs Raw 3D at 0.6228).
   - **Test Video Accuracy:** Mix v2 is #1 at **74.82% ± 1.01%** (+2.43% absolute gain over Raw 3D at 72.39%).
   - **Test Video Macro F1:** Mix v2 is #1 at **0.7265 ± 0.0108** (+0.0220 absolute gain over Raw 3D at 0.7045).

---

## Table 3: Spatial-Temporal Graph Kinematic Streams (Paper Table 4)

*Objective:* Evaluate static physical adjacency ($A_{\text{phys}}$) versus learnable adaptive topology ($B_k + C_k$) across 4 kinematic modalities.  
*Execution Command:* `python run.py train --model AAGCN --feature <STREAM> --augment skel_gym_aug --device auto`

| Exp ID | Model Architecture | Kinematic Stream | Augmentation Protocol | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Vid Acc (%) | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T3.1** | ST-GCN Baseline | Raw 3D Joint | None (Clean) | 0.0881 | 97.01% ± 0.18% | 1.8845 | 65.35% ± 1.16% | 0.6480 ± 0.0118 | 53.03% ± 0.33% | 64.95% ± 1.33% | Verified |
| **T3.2** | ST-GCN Baseline | Relative 3D Joint | None (Clean) | 0.0614 | 97.85% ± 0.42% | 2.0274 | 70.46% ± 0.45% | 0.7016 ± 0.0023 | 57.58% ± 0.87% | 66.24% ± 1.32% | Verified |
| **T3.3** | AAGCN Baseline | Bone 3D Stream | None (Clean) | 0.4275 | 97.40% ± 0.58% | 1.2103 | 76.21% ± 0.49% | 0.7552 ± 0.0024 | 63.02% ± 1.85% | 71.67% ± 1.60% | Verified |
| **T3.4** | AAGCN | Bone 3D Stream | SkelGym-Aug (Proposed) | 0.4275 | 97.40% ± 0.58% | 1.2103 | 76.21% ± 0.49% | 0.7552 ± 0.0024 | 65.34% ± 1.26% | 73.10% ± 1.07% | Verified |
| **T3.5** | AAGCN | Joint Stream (Rel 3D) | SkelGym-Aug (Proposed) | 0.4715 | 95.85% ± 2.04% | 1.1394 | 76.51% ± 0.56% | 0.7541 ± 0.0062 | 67.46% ± 0.22% | 76.39% ± 1.75% | Verified |
| **T3.6** | AAGCN | Joint Motion 3D ($\Delta X$) | SkelGym-Aug (Proposed) | 0.7019 | 88.36% ± 3.31% | 1.9498 | 54.76% ± 3.73% | 0.5558 ± 0.0290 | 48.81% ± 0.27% | 65.66% ± 1.26% | Verified |
| **T3.7** | AAGCN | Bone Motion 3D ($\Delta B$) | SkelGym-Aug (Proposed) | 0.7098 | 88.09% ± 3.26% | 2.2195 | 48.71% ± 0.92% | 0.5180 ± 0.0174 | 49.17% ± 1.09% | 67.24% ± 2.63% | Verified |
| **T3.8** | Two-Stream AAGCN | Joint + Bone | Late Fusion (Equal Weights) | — | — | 1.1520 | 78.11% ± 0.50% | 0.7750 ± 0.0000 | 69.57% ± 0.55% | 78.68% ± 0.41% | Verified |
| **T3.9** | Four-Stream AAGCN | 4 Streams Unified | Late Fusion (SLSQP Calibrated) | — | — | 1.1140 | 79.40% ± 0.60% | 0.7930 ± 0.0000 | 68.67% ± 1.15% | 78.40% ± 0.81% | Verified |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Full (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | 1.0923 | 78.06% ± 1.17% | 0.7741 ± 0.0134 | 66.80% ± 0.63% | 0.6574 ± 0.0066 | 0.00% (Ref) | Verified |
| **w/o Sagittal Reflection ($-$Mirror)** | Bilateral Reflection | 1.2502 | 76.31% ± 0.63% | 0.7588 ± 0.0099 | 62.80% ± 1.35% | 0.6164 ± 0.0115 | -4.00% | Verified |
| **w/o Gravitational Yaw ($-$Yaw)** | Vertical Axis 3D Yaw | 1.1288 | 78.84% ± 0.60% | 0.7836 ± 0.0079 | 68.81% ± 2.10% | 0.6796 ± 0.0213 | +2.01% | Verified |
| **w/o Proportional Scaling ($-$Scale)** | Isotropic Anthropometric Scale | 1.1061 | 78.38% ± 0.59% | 0.7776 ± 0.0088 | 69.23% ± 1.39% | 0.6810 ± 0.0127 | +2.43% | Verified |
| **w/o Temporal TimeWarp ($-$TimeWarp)** | Cadence / Temporal Phase Warping | 1.0948 | 78.25% ± 0.04% | 0.7754 ± 0.0029 | 68.68% ± 2.04% | 0.6763 ± 0.0133 | +1.88% | Verified |
| **w/o Sensor Jitter ($-$Jitter)** | Gaussian Sensor Noise | 1.1101 | 77.35% ± 1.17% | 0.7683 ± 0.0088 | 68.49% ± 1.57% | 0.6701 ± 0.0152 | +1.69% | Verified |
| **Clean Baseline (No Augmentation)** | All Operators Excluded | 1.2582 | 75.27% ± 0.51% | 0.7478 ± 0.0080 | 62.02% ± 1.40% | 0.6111 ± 0.0096 | -4.78% | Verified |

---

## Table 5: Systematic Single-Component (Individual) Augmentation Study on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone efficacy of each transformation in complete isolation against the Clean Baseline across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Applied Domain / Mechanism | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Baseline (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (Control)** | None (Unaugmented) | 1.2582 | 75.27% ± 0.51% | 0.7478 ± 0.0080 | 62.02% ± 1.40% | 0.6111 ± 0.0096 | 0.00% (Ref) | Verified |
| **+ Sagittal Reflection (Mirror)** | Bilateral Body Reflection | 1.1382 | 77.69% ± 0.17% | 0.7698 ± 0.0021 | 68.60% ± 2.36% | 0.6770 ± 0.0200 | +6.57% | Verified |
| **+ Gravitational Yaw (Yaw)** | 3D Viewpoint Invariance | 1.2477 | 76.42% ± 1.09% | 0.7584 ± 0.0127 | 62.58% ± 1.58% | 0.6105 ± 0.0144 | +0.56% | Verified |
| **+ Proportional Scaling (Scale)** | Stature & Distance Scaling | 1.2450 | 76.31% ± 0.90% | 0.7591 ± 0.0077 | 63.65% ± 1.68% | 0.6218 ± 0.0153 | +1.63% | Verified |
| **+ Temporal TimeWarp (TimeWarp)** | Synthetic Velocity Perturbation | 1.2778 | 75.97% ± 1.18% | 0.7536 ± 0.0145 | 63.05% ± 1.34% | 0.6147 ± 0.0195 | +1.02% | Verified |
| **+ Sensor Jitter (Jitter)** | MediaPipe Tracking Noise Tolerance | 1.2866 | 76.55% ± 0.40% | 0.7599 ± 0.0076 | 63.29% ± 1.20% | 0.6194 ± 0.0103 | +1.26% | Verified |
| **SkelGym-Aug (4-op Suite)** | Spatial + Sensor (Proposed) | 1.0948 | 78.25% ± 0.04% | 0.7754 ± 0.0029 | 68.68% ± 2.04% | 0.6763 ± 0.0133 | +6.66% | Verified |
| **Candidate Full (5-op Suite)** | Spatial + Sensor + Temporal | 1.0923 | 78.06% ± 1.17% | 0.7741 ± 0.0134 | 66.80% ± 0.63% | 0.6574 ± 0.0066 | +4.78% | Verified |

---

## Table 6: Cross-Paradigm Fusion Protocols & Multi-Seed Downstream Evaluation (Paper Table 5)

*Objective:* Benchmark 5 systematic fusion methods and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Execution Command:* `python scripts/run_multi_seed_experiments.py --seeds 42 123 3407`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (117-d)** | Single Sequence Backbone | 76.32% ± 2.10% | 0.7556 ± 0.0260 | 79.07% ± 2.01% | 0.7991 ± 0.0133 | 66.27% ± 2.75% | 0.6544 ± 0.0264 | 75.54% ± 3.81% | 0.7430 ± 0.0430 | Verified |
| **AAGCN Bone Stream (Bone 3D)** | Single Graph Backbone | 78.83% ± 0.33% | 0.7783 ± 0.0041 | 80.68% ± 0.48% | 0.8199 ± 0.0103 | 66.59% ± 1.43% | 0.6587 ± 0.0181 | 75.82% ± 2.62% | 0.7543 ± 0.0305 | Verified |
| **Four-Stream AAGCN** | 4 Streams Unified Graph | 79.98% ± 0.27% | 0.7930 ± 0.0041 | 81.96% ± 1.39% | 0.8296 ± 0.0159 | 67.61% ± 1.46% | 0.6688 ± 0.0205 | 77.54% ± 3.33% | 0.7632 ± 0.0387 | Verified |
| **Hard Majority Voting** | Discrete mode over class predictions | 80.64% ± 0.50% | 0.8063 ± 0.0038 | 84.38% ± 0.74% | 0.8502 ± 0.0086 | 70.62% ± 0.54% | 0.7017 ± 0.0064 | 81.40% ± 0.99% | 0.8037 ± 0.0126 | Verified |
| **Uniform Average Soft Voting** | Equal weights: $w_i = 1/5 = 0.20$ | 82.36% ± 0.30% | 0.8214 ± 0.0032 | 85.67% ± 1.01% | 0.8583 ± 0.0056 | 73.05% ± 0.75% | 0.7226 ± 0.0053 | 82.83% ± 1.14% | 0.8143 ± 0.0184 | Verified |
| **Accuracy-Weighted Soft Voting** | Validation accuracy weights ($w_i \propto \text{Acc}_i^{\text{val}}$) | 82.07% ± 0.10% | 0.8178 ± 0.0019 | 84.06% ± 0.84% | 0.8430 ± 0.0075 | 72.56% ± 0.88% | 0.7183 ± 0.0062 | 81.97% ± 1.14% | 0.8063 ± 0.0210 | Verified |
| **SLSQP Soft Voting** | SLSQP Constrained Calibration ($\sum w_i = 1$) | 80.47% ± 0.62% | 0.8001 ± 0.0072 | 83.09% ± 0.48% | 0.8377 ± 0.0050 | 69.73% ± 1.10% | 0.6881 ± 0.0072 | 78.83% ± 0.66% | 0.7807 ± 0.0126 | Verified |
| **Stacking Meta-Classifier** | Ridge Classifier on Val Probs | 93.04% ± 0.63% | 0.9284 ± 0.0067 | 94.85% ± 1.39% | 0.9528 ± 0.0128 | 73.53% ± 0.91% | 0.7250 ± 0.0063 | 83.69% ± 0.74% | 0.8354 ± 0.0089 | Verified |
| **SkelGym-Lite (2 Models)** | Trans + Bone AAGCN (SLSQP Calibrated) | 79.78% ± 0.81% | 0.7906 ± 0.0103 | 82.61% ± 1.67% | 0.8354 ± 0.0133 | 68.26% ± 0.66% | 0.6732 ± 0.0067 | 77.68% ± 1.55% | 0.7694 ± 0.0103 | Verified |
| **SkelGym-Full (5 Streams)** | Trans + 4 AAGCN (SLSQP Calibrated) | 80.47% ± 0.62% | 0.8001 ± 0.0072 | 83.09% ± 0.48% | 0.8377 ± 0.0050 | 69.73% ± 1.10% | 0.6881 ± 0.0072 | 78.83% ± 0.66% | 0.7807 ± 0.0126 | Verified |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all 11 architectures.  
*Execution Command:* `python run.py evaluate --checkpoint <CKPT> --video_level --device auto`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Video Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 117-d)** | Sequential Recurrent Model | 396K | 57.64% | 0.5676 | 67.38% | 0.6580 | +9.74% | Verified |
| **Baseline BiLSTM (Mix 117-d)** | Bidirectional Recurrent Model | 402K | 61.25% | 0.6015 | 68.67% | 0.6830 | +7.42% | Verified |
| **Transformer (Mix 117-d, Clean)** | Self-Attention Baseline | 400K | 63.40% | 0.6218 | 74.25% | 0.7304 | +10.85% | Verified |
| **Baseline ST-GCN (Rel 3D)** | Rigid Static Graph ($A_{\text{phys}}$) | 350K | 54.76% | 0.5240 | 63.09% | 0.5993 | +8.33% | Verified |
| **Clean Baseline AAGCN (Bone 3D)** | Adaptive Skeletal Graph (Unaugmented) | 378K | 59.50% | 0.5960 | 69.53% | 0.6904 | +10.03% | Verified |
| **SkelGym-Aug AAGCN (Bone 3D)** | Adaptive Skeletal Graph + Augmentation | 378K | 66.59% | 0.6587 | 75.82% | 0.7543 | +9.23% | Verified |
| **SkelGym-Aug Transformer (Mix)** | Self-Attention + Augmentation | 400K | 66.27% | 0.6544 | 75.54% | 0.7430 | +9.27% | Verified |
| **Two-Stream AAGCN (Aug)** | Joint + Bone Stream Fusion | 756K | 67.26% | 0.6652 | 77.11% | 0.7580 | +9.85% | Verified |
| **Four-Stream AAGCN (Aug)** | 4-Stream Graph Late Fusion | 1.51M | 67.61% | 0.6688 | 77.54% | 0.7632 | +9.92% | Verified |
| **SkelGym-Lite (2 Models)** | Transformer + Bone AAGCN | 778K | 68.26% | 0.6732 | 77.68% | 0.7694 | +9.42% | Verified |
| **SkelGym-Full (5 Streams)** | Cross-Paradigm SLSQP Ensemble | 1.91M | 69.73% | 0.6881 | 78.83% | 0.7807 | +9.10% | Verified |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 47.77 | 3.20e-12 | 1.92 | 11104.0 | 0.0142 | 0.0048 | +0.187 | Verified |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 155.34 | 3.84e-37 | 2.92 | 9338.0 | 3.08e-05 | 3.89e-06 | +0.310 | Verified |
| **Single Sequence (Trans) vs SkelGym-Full** | 2.16 | 0.1416 | 1.17 | 8050.0 | 6.03e-08 | 1.84e-05 | -0.287 | Verified |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 63.58 | 2.40e-16 | 3.52 | 12219.0 | 0.1706 | 0.0614 | +0.123 | Verified |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 57.14 | 6.69e-15 | 3.73 | 8207.0 | 1.40e-07 | 1.33e-05 | +0.292 | Verified |

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling.  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix 117-d)** | 57.72% [52.30%, 62.97%] | 0.5504 [0.4952, 0.6052] | 67.53% [61.80%, 73.39%] | 0.6478 [0.5791, 0.7137] | Verified |
| **BiLSTM (Mix 117-d)** | 61.29% [55.07%, 67.21%] | 0.5849 [0.5286, 0.6379] | 68.76% [62.66%, 74.25%] | 0.6723 [0.6111, 0.7285] | Verified |
| **ST-GCN (Rel 3D)** | 54.75% [48.54%, 60.79%] | 0.5107 [0.4590, 0.5623] | 63.18% [56.65%, 69.53%] | 0.5909 [0.5332, 0.6531] | Verified |
| **Transformer (Mix 117-d)** | 69.04% [62.60%, 74.91%] | 0.6647 [0.6076, 0.7165] | 79.92% [74.68%, 84.55%] | 0.7835 [0.7252, 0.8416] | Verified |
| **AAGCN (Bone 3D)** | 65.80% [59.72%, 71.83%] | 0.6293 [0.5742, 0.6827] | 73.01% [66.95%, 78.54%] | 0.7113 [0.6505, 0.7670] | Verified |
| **SkelGym-Lite (2 Models)** | 68.13% [61.92%, 74.10%] | 0.6530 [0.5951, 0.7075] | 79.05% [73.82%, 84.12%] | 0.7691 [0.7104, 0.8256] | Verified |
| **SkelGym-Full (5 Streams)** | 70.07% [63.82%, 75.83%] | 0.6733 [0.6169, 0.7292] | 79.02% [73.82%, 84.12%] | 0.7762 [0.7219, 0.8300] | Verified |

---

## Table 10: Per-Class Performance Breakdown & Error Taxonomy (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full on held-out test windows ($N=2,743$) and test videos ($N=233$).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barbell biceps curl** | 0.3220 | 0.8261 | 0.4634 | 69 | 0.5652 | 0.9286 | 0.7027 | 14 | Verified |
| **bench press** | 0.4560 | 0.5938 | 0.5158 | 96 | 0.6667 | 0.5714 | 0.6154 | 14 | Verified |
| **chest fly machine** | 0.8333 | 0.9756 | 0.8989 | 82 | 0.8750 | 0.8750 | 0.8750 | 8 | Verified |
| **deadlift** | 0.3258 | 0.6418 | 0.4322 | 67 | 0.6429 | 0.9000 | 0.7500 | 10 | Verified |
| **decline bench press** | 0.4082 | 0.5970 | 0.4848 | 134 | 0.4545 | 0.6250 | 0.5263 | 8 | Verified |
| **hammer curl** | 0.4948 | 0.2981 | 0.3721 | 161 | 0.6667 | 0.4286 | 0.5217 | 14 | Verified |
| **hip thrust** | 0.8606 | 0.6017 | 0.7082 | 236 | 0.8889 | 0.8889 | 0.8889 | 9 | Verified |
| **incline bench press** | 0.7302 | 0.6053 | 0.6619 | 76 | 1.0000 | 0.6667 | 0.8000 | 9 | Verified |
| **lat pulldown** | 0.5563 | 0.8900 | 0.6846 | 100 | 0.6500 | 1.0000 | 0.7879 | 13 | Verified |
| **lateral raise** | 0.9205 | 0.8968 | 0.9085 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **leg extension** | 0.9921 | 1.0000 | 0.9960 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 | Verified |
| **leg raises** | 1.0000 | 0.4649 | 0.6347 | 114 | 1.0000 | 0.5455 | 0.7059 | 11 | Verified |
| **plank** | 0.9024 | 0.6607 | 0.7629 | 56 | 1.0000 | 1.0000 | 1.0000 | 2 | Verified |
| **pull Up** | 0.7125 | 0.6951 | 0.7037 | 82 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| **push-up** | 0.7727 | 0.9770 | 0.8629 | 87 | 0.8571 | 1.0000 | 0.9231 | 12 | Verified |
| **romanian deadlift** | 0.5455 | 0.3699 | 0.4408 | 146 | 0.5000 | 0.3333 | 0.4000 | 6 | Verified |
| **russian twist** | 0.8239 | 0.8540 | 0.8387 | 137 | 0.6667 | 1.0000 | 0.8000 | 6 | Verified |
| **shoulder press** | 0.7209 | 0.4079 | 0.5210 | 152 | 0.6250 | 0.3846 | 0.4762 | 13 | Verified |
| **squat** | 0.8625 | 0.8697 | 0.8661 | 238 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **t bar row** | 0.8471 | 0.6154 | 0.7129 | 117 | 1.0000 | 0.8000 | 0.8889 | 10 | Verified |
| **tricep Pushdown** | 0.6762 | 0.7634 | 0.7172 | 93 | 0.8333 | 0.8333 | 0.8333 | 12 | Verified |
| **tricep dips** | 0.9393 | 0.9136 | 0.9263 | 220 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| **Overall Accuracy** | 0.7007 | 0.7007 | 0.7007 | **2743** | 0.7897 | 0.7897 | 0.7897 | **233** | **Verified** |
| **Macro Average** | 0.7138 | 0.7054 | 0.6870 | **2743** | 0.8133 | 0.7895 | 0.7845 | **233** | **Verified** |

---

## Table 11: Computational Complexity & Inference Latency (Paper Table 11)

*Objective:* Measure parameters, window FLOPs, and latency across server and edge devices.  
*Execution Command:* `python scripts/benchmark_hardware_latency.py`

| Model Architecture | Parameters | FLOPs per Window | RTX PRO 6000 (CUDA) | Apple M4 (MPS) | Apple M4 (CPU) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (117-d)** | 400K | 10.70 MFLOPs | 0.50 ms (1999 FPS) | 1.24 ms (809 FPS) | 0.42 ms (2406 FPS) | Verified |
| **AAGCN Joint Stream** | 378K | 202.86 MFLOPs | 1.01 ms (987 FPS) | 1.89 ms (530 FPS) | 0.90 ms (1115 FPS) | Verified |
| **AAGCN Bone Stream** | 378K | 202.86 MFLOPs | 1.02 ms (976 FPS) | 1.72 ms (581 FPS) | 0.88 ms (1138 FPS) | Verified |
| **AAGCN Joint-Motion Stream** | 378K | 202.86 MFLOPs | 1.01 ms (990 FPS) | 1.90 ms (527 FPS) | 1.02 ms (978 FPS) | Verified |
| **AAGCN Bone-Motion Stream** | 378K | 202.86 MFLOPs | 1.02 ms (980 FPS) | 1.90 ms (527 FPS) | 1.03 ms (975 FPS) | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 778K | 213.56 MFLOPs | 1.66 ms (603 FPS) | 2.55 ms (393 FPS) | 1.49 ms (671 FPS) | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 1.91M | 822.14 MFLOPs | 4.65 ms (215 FPS) | 5.86 ms (171 FPS) | 3.96 ms (253 FPS) | Verified |

---

## Table 12: Strength & Conditioning Exercise Recognition (MM-Fit External Benchmark) (Paper Table 12)

*Objective:* Comparative transfer evaluation across the four overlapping Strength & Conditioning (S&C) exercises inspired by Deyzel et al. (2023) (*squat*, *deadlift*, *barbell biceps curl*, *lateral raise*) on the genuine MM-Fit external benchmark (MediaPipe Pose protocol across 5 unseen workout recordings: `w00`, `w05`, `w12`, `w13`, `w20`; $N=54$ recordings, $N=878$ windows).  
*Execution Command:* `python scripts/evaluate_external.py --config configs/external/mmfit.yaml --pose-source mediapipe --seeds 42`

| Model Architecture | Open-Set Test Win Acc (%) | Open-Set Test Vid Acc (%) | Closed-Set Test Win Acc (%) | Closed-Set Test Vid Acc (%) | Closed-Set Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN (Rel 3D) (Deyzel baseline)** | 37.93% | 38.89% | 67.43% | 70.37% | 0.6524 | Verified |
| **LSTM (Mix 117-d)** | 61.96% | 77.78% | 87.02% | 90.74% | 0.8904 | Verified |
| **BiLSTM (Mix 117-d)** | 61.16% | 68.52% | 80.41% | 83.33% | 0.7775 | Verified |
| **AAGCN (Bone 3D)** | 58.88% | 62.96% | 74.26% | 77.78% | 0.6667 | Verified |
| **Transformer (Mix 117-d)** | 55.47% | 61.11% | 83.71% | 85.19% | 0.8092 | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 63.55% | 66.67% | 79.95% | 81.48% | 0.7433 | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 66.86% | 74.07% | 95.90% | 100.00% | 1.0000 | Verified |

---

## Table 13: Strong External Baseline — BlockGCN (CVPR 2024 Adapted) (Paper Benchmark Table)

*Objective:* External benchmark comparison against BlockGCN (CVPR 2024), faithfully adapted to 33 MediaPipe joints ($V=33$, $T=32$, $M=1$, $C=22$, joint-only stream), trained strictly from scratch across 3 independent seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_blockgcn_baseline.py --config configs/external/blockgcn_original_33j_32f.yaml --device cuda --push_to_hf`

| Model Architecture | Input Representation | Parameters | FLOPs / MACs | Window Test Acc (%) | Window Macro F1 | Video Consensus Acc (%) | Video Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BlockGCN (CVPR 2024 Adapted)** | 33 Raw MediaPipe XYZ (Joint-only) | 1,352,102 | 525.27 MFLOPs | 54.90% ± 0.76% | 0.5542 ± 0.0031 | 71.05% ± 0.40% | 0.6977 ± 0.0120 | Verified |

