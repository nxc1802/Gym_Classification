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

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Val Loss | Val Acc (%) | Test Win Acc (%) | Macro F1 | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T1.1** | LSTM | Raw 2D Coordinates | 26 | 0.1083 | 2.0594 | 67.13% | 52.31% | 0.5179 | Verified |
| **T1.2** | LSTM | Relative 2D (Mid-Hip) | 26 | 0.1897 | 1.4853 | 71.04% | 56.91% | 0.5616 | Verified |
| **T1.3** | LSTM | Angle 2D (Triplets) | 286 | 0.0802 | 2.3990 | 66.99% | 50.42% | 0.4954 | Verified |
| **T1.4** | LSTM | Angle2 2D (Pairs) | 78 | 0.1486 | 1.9007 | 68.00% | 52.31% | 0.5171 | Verified |
| **T1.5** | LSTM | Raw 3D Coordinates | 39 | 0.1527 | 1.7012 | 68.29% | 54.87% | 0.5259 | Verified |
| **T1.6** | LSTM | Relative 3D (Mid-Hip) | 39 | 0.0360 | 2.2855 | 71.42% | 58.51% | 0.5734 | Verified |
| **T1.7** | LSTM | Angle 3D (Triplets) | 286 | 0.0447 | 2.3879 | 66.41% | 49.33% | 0.4770 | Verified |
| **T1.8** | LSTM | Angle2 3D (Pair Elevation) | 78 | 0.0889 | 1.7770 | 71.23% | 55.81% | 0.5631 | Verified |
| **T1.9** | LSTM | Biomechanical Mix (Ours) | 117 | 0.0772 | 1.9226 | 73.93% | 59.28% | 0.5843 | Verified |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | 0.0538 | 2.2651 | 69.40% | 52.46% | 0.5161 | Verified |
| **T1.11** | BiLSTM | Relative 2D (Mid-Hip) | 26 | 0.2316 | 1.3567 | 72.10% | 55.56% | 0.5455 | Verified |
| **T1.12** | BiLSTM | Angle 2D (Triplets) | 286 | 0.0093 | 3.8922 | 65.06% | 51.11% | 0.4855 | Verified |
| **T1.13** | BiLSTM | Angle2 2D (Pairs) | 78 | 0.1037 | 2.0297 | 66.36% | 52.83% | 0.5133 | Verified |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | 0.0496 | 2.2351 | 67.71% | 52.61% | 0.5208 | Verified |
| **T1.15** | BiLSTM | Relative 3D (Mid-Hip) | 39 | 0.1182 | 1.7121 | 70.89% | 56.07% | 0.5470 | Verified |
| **T1.16** | BiLSTM | Angle 3D (Triplets) | 286 | 0.0893 | 2.3198 | 63.23% | 50.35% | 0.4753 | Verified |
| **T1.17** | BiLSTM | Angle2 3D (Pair Elevation) | 78 | 0.0263 | 2.2670 | 71.42% | 58.62% | 0.5759 | Verified |
| **T1.18** | BiLSTM | Biomechanical Mix (Ours) | 117 | 0.0794 | 1.7916 | 74.41% | 58.66% | 0.5735 | Verified |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | 0.3871 | 1.1225 | 77.69% | 62.41% | 0.6190 | Verified |
| **T1.20** | Transformer | Relative 2D (Mid-Hip) | 26 | 0.4396 | 1.2003 | 76.43% | 64.16% | 0.6281 | Verified |
| **T1.21** | Transformer | Angle 2D (Triplets) | 286 | 0.3663 | 1.4473 | 71.95% | 57.49% | 0.5408 | Verified |
| **T1.22** | Transformer | Angle2 2D (Pairs) | 78 | 0.3863 | 1.4641 | 70.60% | 57.09% | 0.5598 | Verified |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | 0.4260 | 1.0587 | 79.13% | 65.11% | 0.6339 | Verified |
| **T1.24** | Transformer | Relative 3D (Mid-Hip) | 39 | 0.3763 | 1.1425 | 78.31% | 64.86% | 0.6301 | Verified |
| **T1.25** | Transformer | Angle 3D (Triplets) | 286 | 0.4132 | 1.5141 | 68.77% | 56.36% | 0.5545 | Verified |
| **T1.26** | Transformer | Angle2 3D (Pair Elevation) | 78 | 0.3973 | 1.3702 | 72.67% | 60.41% | 0.5860 | Verified |
| **T1.27** | **Transformer** | **Biomechanical Mix (Ours)** | **117** | 0.4274 | 1.2714 | 74.55% | 63.91% | 0.6240 | **Verified** |

---

## Table 3: Spatial-Temporal Graph Kinematic Streams (Paper Table 4)

*Objective:* Evaluate static physical adjacency ($A_{\text{phys}}$) versus learnable adaptive topology ($B_k + C_k$) across 4 kinematic modalities.  
*Execution Command:* `python run.py train --model AAGCN --feature <STREAM> --augment skel_gym_aug --device auto`

| Exp ID | Model Architecture | Kinematic Stream | Augmentation Protocol | Val Acc (%) | Test Win Acc (%) | Test Vid Acc (%) | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **T3.1** | ST-GCN Baseline | Raw 3D Joint | None (Clean) | 63.71% | 53.26% | 65.67% | Verified |
| **T3.2** | ST-GCN Baseline | Relative 3D Joint | None (Clean) | 72.39% | 58.80% | 66.95% | Verified |
| **T3.3** | AAGCN Baseline | Bone 3D Stream | None (Clean) | 76.53% | 62.67% | 72.10% | Verified |
| **T3.4** | AAGCN | Bone 3D Stream | SkelGym-Aug (Proposed) | 80.92% | 65.59% | 73.39% | Verified |
| **T3.5** | AAGCN | Joint Stream (Rel 3D) | SkelGym-Aug (Proposed) | 79.52% | 67.15% | 74.25% | Verified |
| **T3.6** | AAGCN | Joint Motion 3D ($\Delta X$) | SkelGym-Aug (Proposed) | 63.66% | 48.74% | 63.95% | Verified |
| **T3.7** | AAGCN | Bone Motion 3D ($\Delta B$) | SkelGym-Aug (Proposed) | 64.72% | 49.43% | 64.81% | Verified |
| **T3.8** | Two-Stream AAGCN | Joint + Bone | Late Fusion (Equal Weights) | 78.11% | 69.63% | 78.54% | Verified |
| **T3.9** | **Four-Stream AAGCN** | **4 Streams Unified** | Late Fusion (SLSQP Calibrated) | 79.40% | 67.34% | 77.25% | **Verified** |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Excluded Operator / Domain | Val Window Acc (%) | Val Loss | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Full (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | 78.06% ± 1.17% | 1.0923 | 66.80% ± 0.63% | 0.6574 ± 0.0066 | 0.00% (Ref) | Verified |
| **w/o Sagittal Reflection ($-$Mirror)** | Bilateral Reflection | 76.31% ± 0.63% | 1.2502 | 62.80% ± 1.35% | 0.6164 ± 0.0115 | -4.00% | Verified |
| **w/o Gravitational Yaw ($-$Yaw)** | Vertical Axis 3D Yaw | 78.84% ± 0.60% | 1.1288 | 68.81% ± 2.10% | 0.6796 ± 0.0213 | +2.01% | Verified |
| **w/o Proportional Scaling ($-$Scale)** | Isotropic Anthropometric Scale | 78.38% ± 0.59% | 1.1061 | 69.23% ± 1.39% | 0.6810 ± 0.0127 | +2.43% | Verified |
| **w/o Temporal TimeWarp ($-$TimeWarp)** | Cadence / Temporal Phase Warping | 78.25% ± 0.04% | 1.0948 | 68.68% ± 2.04% | 0.6763 ± 0.0133 | +1.88% | **Verified** |
| **w/o Sensor Jitter ($-$Jitter)** | Gaussian Sensor Noise | 77.35% ± 1.17% | 1.1101 | 68.49% ± 1.57% | 0.6701 ± 0.0152 | +1.69% | Verified |
| **Clean Baseline (No Augmentation)** | All Operators Excluded | 75.27% ± 0.51% | 1.2582 | 62.02% ± 1.40% | 0.6111 ± 0.0096 | -4.78% | Verified |

---

## Table 5: Systematic Single-Component (Individual) Augmentation Study on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone efficacy of each transformation in complete isolation against the Clean Baseline across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Applied Domain / Mechanism | Val Window Acc (%) | Val Loss | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Baseline (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline (Control)** | None (Unaugmented) | 75.27% ± 0.51% | 1.2582 | 62.02% ± 1.40% | 0.6111 ± 0.0096 | 0.00% (Ref) | Verified |
| **+ Sagittal Reflection (Mirror)** | Bilateral Body Reflection | 77.69% ± 0.17% | 1.1382 | 68.60% ± 2.36% | 0.6770 ± 0.0200 | +6.57% | Verified |
| **+ Gravitational Yaw (Yaw)** | 3D Viewpoint Invariance | 76.42% ± 1.09% | 1.2477 | 62.58% ± 1.58% | 0.6105 ± 0.0144 | +0.56% | Verified |
| **+ Proportional Scaling (Scale)** | Stature & Distance Scaling | 76.31% ± 0.90% | 1.2450 | 63.65% ± 1.68% | 0.6218 ± 0.0153 | +1.63% | Verified |
| **+ Temporal TimeWarp (TimeWarp)** | Synthetic Velocity Perturbation | 75.97% ± 1.18% | 1.2778 | 63.05% ± 1.34% | 0.6147 ± 0.0195 | +1.02% | Verified |
| **+ Sensor Jitter (Jitter)** | MediaPipe Tracking Noise Tolerance | 76.55% ± 0.40% | 1.2866 | 63.29% ± 1.20% | 0.6194 ± 0.0103 | +1.26% | Verified |
| **SkelGym-Aug (4-op Suite)** | **Spatial + Sensor (Proposed)** | 78.25% ± 0.04% | 1.0948 | 68.68% ± 2.04% | 0.6763 ± 0.0133 | +6.66% | **Verified** |
| **Candidate Full (5-op Suite)** | Spatial + Sensor + Temporal | 78.06% ± 1.17% | 1.0923 | 66.80% ± 0.63% | 0.6574 ± 0.0066 | +4.78% | Verified |

---

## Table 6: Cross-Paradigm Fusion Protocols & Multi-Seed Downstream Evaluation (Paper Table 5)

*Objective:* Benchmark 5 systematic fusion methods and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Execution Command:* `python scripts/run_multi_seed_experiments.py --seeds 42 123 3407`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Vid Acc (%) | Test Win Acc (%) | Test Macro F1 | Test Vid Acc (%) | Video Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (117-d)** | Single Sequence Backbone | 76.32% ± 2.10% | 79.07% ± 2.01% | 66.27% ± 2.75% | 0.6544 ± 0.0264 | 75.54% ± 3.81% | 0.7430 ± 0.0430 | Verified |
| **AAGCN Bone Stream (Bone 3D)** | Single Graph Backbone | 78.83% ± 0.33% | 80.68% ± 0.48% | 66.59% ± 1.43% | 0.6587 ± 0.0181 | 75.82% ± 2.62% | 0.7543 ± 0.0305 | Verified |
| **Four-Stream AAGCN** | 4 Streams Unified Graph | 79.98% ± 0.27% | 81.96% ± 1.39% | 67.61% ± 1.46% | 0.6688 ± 0.0205 | 77.54% ± 3.33% | 0.7632 ± 0.0387 | Verified |
| **Hard Majority Voting** | Discrete mode over class predictions | — | — | 70.62% ± 0.54% | 0.7017 ± 0.0064 | 81.40% ± 0.99% | 0.8037 ± 0.0126 | Verified |
| **Uniform Average Soft Voting** | Equal weights: $w_i = 1/5 = 0.20$ | — | — | 73.05% ± 0.75% | 0.7226 ± 0.0053 | 82.83% ± 1.14% | 0.8143 ± 0.0184 | Verified |
| **Accuracy-Weighted Soft Voting** | Validation accuracy weights ($w_i \propto \text{Acc}_i^{\text{val}}$) | — | — | 72.56% ± 0.88% | 0.7183 ± 0.0062 | 81.97% ± 1.14% | 0.8063 ± 0.0210 | Verified |
| **SkelGym-Lite (2 Models)** | Trans + Bone AAGCN (SLSQP Calibrated) | — | — | 68.26% ± 0.66% | 0.6732 ± 0.0067 | 77.68% ± 1.55% | 0.7694 ± 0.0103 | Verified |
| **SkelGym-Full (5 Streams)** | **Trans + 4 AAGCN (SLSQP Calibrated)** | — | — | 69.73% ± 1.10% | 0.6881 ± 0.0072 | 78.83% ± 0.66% | 0.7807 ± 0.0126 | **Verified** |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all 11 architectures.  
*Execution Command:* `python run.py evaluate --checkpoint <CKPT> --video_level --device auto`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Video Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 117-d)** | Sequential Recurrent Model | 396K | 59.21% | 0.5838 | 68.67% | 0.6757 | +9.46% | Verified |
| **Baseline BiLSTM (Mix 117-d)** | Bidirectional Recurrent Model | 402K | 60.41% | 0.5898 | 67.81% | 0.6540 | +7.40% | Verified |
| **Transformer (Mix 117-d, Clean)** | Self-Attention Baseline | 400K | 61.61% | 0.6088 | 74.25% | 0.7272 | +12.64% | Verified |
| **Baseline ST-GCN (Rel 3D)** | Rigid Static Graph ($A_{\text{phys}}$) | 350K | 58.84% | 0.5658 | 66.95% | 0.6480 | +8.11% | Verified |
| **Clean Baseline AAGCN (Bone 3D)** | Adaptive Skeletal Graph (Unaugmented) | 378K | 62.56% | 0.6006 | 72.10% | 0.7002 | +9.54% | Verified |
| **SkelGym-Aug AAGCN (Bone 3D)** | Adaptive Skeletal Graph + Augmentation | 378K | 65.48% | 0.6514 | 73.39% | 0.7304 | +7.91% | Verified |
| **SkelGym-Aug Transformer (Mix)** | Self-Attention + Augmentation | 400K | 71.53% | 0.6945 | 81.97% | 0.7920 | +10.44% | Verified |
| **Two-Stream AAGCN (Aug)** | Joint + Bone Stream Fusion | 756K | 67.85% | 0.6720 | 77.68% | 0.7775 | +9.83% | Verified |
| **Four-Stream AAGCN (Aug)** | 4-Stream Graph Late Fusion | 1.51M | 67.34% | 0.6685 | 77.25% | 0.7740 | +9.91% | Verified |
| **SkelGym-Lite (2 Models)** | Transformer + Bone AAGCN | 778K | 69.60% | 0.6822 | 77.68% | 0.7626 | +8.08% | Verified |
| **SkelGym-Full (5 Streams)** | **Cross-Paradigm SLSQP Ensemble** | **1.91M** | 71.05% | 0.6983 | 79.40% | 0.7640 | +8.35% | **Verified** |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 146.21 | $3.89 \times 10^{-35}$ | 3.32 | — | $8.14 \times 10^{-9}$ | $2.75 \times 10^{-8}$ | +0.377 | Verified |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 86.04 | $9.28 \times 10^{-21}$ | 2.17 | — | $1.61 \times 10^{-5}$ | 0.0009 | -0.221 | Verified |
| **Single Sequence (Trans) vs SkelGym-Full** | 0.66 | 0.4524 (n.s.) | 0.90 | — | $6.05 \times 10^{-6}$ | $5.34 \times 10^{-6}$ | -0.305 | Verified |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 117.63 | $2.20 \times 10^{-30}$ | 7.65 | — | 0.0309 | 0.0071 | +0.178 | Verified |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 62.67 | $4.79 \times 10^{-16}$ | 4.19 | — | $2.81 \times 10^{-21}$ | $1.93 \times 10^{-22}$ | +0.711 | **Verified** |

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling.  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix 117-d)** | 59.29% [52.85%, 65.51%] | 0.5670 [0.5078, 0.6281] | 68.81% [63.09%, 74.68%] | 0.6656 [0.6054, 0.7275] | Verified |
| **BiLSTM (Mix 117-d)** | 60.44% [53.62%, 66.61%] | 0.5733 [0.5109, 0.6324] | 67.93% [61.79%, 73.39%] | 0.6429 [0.5763, 0.7049] | Verified |
| **ST-GCN (Rel 3D)** | 58.83% [52.77%, 64.61%] | 0.5505 [0.4895, 0.6085] | 67.01% [60.94%, 72.97%] | 0.6349 [0.5638, 0.7011] | Verified |
| **Transformer (Mix 117-d)** | 71.55% [64.67%, 77.99%] | 0.6822 [0.6239, 0.7438] | 81.98% [76.81%, 86.70%] | 0.7835 [0.7254, 0.8429] | Verified |
| **AAGCN (Bone 3D)** | 65.40% [58.93%, 71.64%] | 0.6369 [0.5780, 0.6907] | 73.32% [67.81%, 78.97%] | 0.7199 [0.6672, 0.7755] | Verified |
| **SkelGym-Lite (2 Models)** | 69.57% [62.88%, 75.83%] | 0.6693 [0.6118, 0.7263] | 77.63% [72.10%, 82.83%] | 0.7530 [0.6946, 0.8096] | Verified |
| **SkelGym-Full (5 Streams)** | 71.02% [64.30%, 77.31%] | 0.6871 [0.6278, 0.7441] | 79.35% [73.82%, 84.55%] | 0.7546 [0.6941, 0.8151] | **Verified** |

---

## Table 10: Per-Class Performance Breakdown & Error Taxonomy (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full on held-out test windows ($N=2,743$) and test videos ($N=233$).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barbell biceps curl** | 0.3464 | 0.8986 | 0.5000 | 69 | 0.5833 | 1.0000 | 0.7368 | 14 | Verified |
| **bench press** | 0.3523 | 0.6458 | 0.4559 | 96 | 0.5625 | 0.6429 | 0.6000 | 14 | Verified |
| **chest fly machine** | 0.9419 | 0.9878 | 0.9643 | 82 | 0.8750 | 0.8750 | 0.8750 | 8 | Verified |
| **deadlift** | 0.2410 | 0.5970 | 0.3433 | 67 | 0.6000 | 0.9000 | 0.7200 | 10 | Verified |
| **decline bench press** | 0.5347 | 0.5746 | 0.5540 | 134 | 0.5000 | 0.6250 | 0.5556 | 8 | Verified |
| **hammer curl** | 0.5745 | 0.3354 | 0.4235 | 161 | 0.8000 | 0.2857 | 0.4211 | 14 | Verified |
| **hip thrust** | 0.8782 | 0.5805 | 0.6990 | 236 | 0.8889 | 0.8889 | 0.8889 | 9 | Verified |
| **incline bench press** | 0.7778 | 0.6447 | 0.7050 | 76 | 0.7500 | 0.3333 | 0.4615 | 9 | Verified |
| **lat pulldown** | 0.6013 | 0.9500 | 0.7364 | 100 | 0.7222 | 1.0000 | 0.8387 | 13 | Verified |
| **lateral raise** | 0.9091 | 0.9032 | 0.9061 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **leg extension** | 0.9921 | 1.0000 | 0.9960 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 | Verified |
| **leg raises** | 1.0000 | 0.6140 | 0.7609 | 114 | 1.0000 | 0.8182 | 0.9000 | 11 | Verified |
| **plank** | 0.7442 | 0.5714 | 0.6465 | 56 | 0.3333 | 0.5000 | 0.4000 | 2 | Verified |
| **pull Up** | 0.9104 | 0.7439 | 0.8188 | 82 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| **push-up** | 0.7611 | 0.9885 | 0.8600 | 87 | 0.8571 | 1.0000 | 0.9231 | 12 | Verified |
| **romanian deadlift** | 0.5109 | 0.3219 | 0.3950 | 146 | 0.6667 | 0.3333 | 0.4444 | 6 | Verified |
| **russian twist** | 0.9173 | 0.8905 | 0.9037 | 137 | 1.0000 | 1.0000 | 1.0000 | 6 | Verified |
| **shoulder press** | 0.6900 | 0.4539 | 0.5476 | 152 | 0.7000 | 0.5385 | 0.6087 | 13 | Verified |
| **squat** | 0.9193 | 0.8613 | 0.8894 | 238 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| **t bar row** | 0.8286 | 0.4957 | 0.6203 | 117 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| **tricep Pushdown** | 0.6455 | 0.7634 | 0.6995 | 93 | 0.7857 | 0.9167 | 0.8462 | 12 | Verified |
| **tricep dips** | 0.9364 | 0.9364 | 0.9364 | 220 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| **Overall Accuracy** | — | — | 0.7105 | **2743** | — | — | 0.7940 | **233** | **Verified** |
| **Macro Average** | 0.7278 | 0.7163 | 0.6983 | **2743** | 0.8011 | 0.7703 | 0.7640 | **233** | **Verified** |

---

## Table 11: Computational Complexity & Inference Latency (Paper Table 11)

*Objective:* Measure parameters, window FLOPs, and latency across server and edge devices.  
*Execution Command:* `python scripts/benchmark_hardware_latency.py`

| Model Architecture | Parameters | FLOPs per Window | RTX PRO 6000 (CUDA) | Apple M4 (MPS) | Apple M4 (CPU) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (117-d)** | 185K | 10.70 MFLOPs | 0.50 ms (1999 FPS) | 1.24 ms (809 FPS) | 0.42 ms (2406 FPS) | Verified |
| **AAGCN Joint Stream** | 377K | 202.86 MFLOPs | 1.01 ms (987 FPS) | 1.89 ms (530 FPS) | 0.90 ms (1115 FPS) | Verified |
| **AAGCN Bone Stream** | 377K | 202.86 MFLOPs | 1.02 ms (976 FPS) | 1.72 ms (581 FPS) | 0.88 ms (1138 FPS) | Verified |
| **AAGCN Joint-Motion Stream** | 377K | 202.86 MFLOPs | 1.01 ms (990 FPS) | 1.90 ms (527 FPS) | 1.02 ms (978 FPS) | Verified |
| **AAGCN Bone-Motion Stream** | 377K | 202.86 MFLOPs | 1.02 ms (980 FPS) | 1.90 ms (527 FPS) | 1.03 ms (975 FPS) | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 562K | 213.56 MFLOPs | 1.66 ms (603 FPS) | 2.55 ms (393 FPS) | 1.49 ms (671 FPS) | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | **1.69M** | 822.14 MFLOPs | 4.65 ms (215 FPS) | 5.86 ms (171 FPS) | 3.96 ms (253 FPS) | **Verified** |

---

## Table 12: Strength & Conditioning Exercise Recognition (Deyzel et al. Subset) (Paper Table 12)

*Objective:* Comparative evaluation across the four overlapping Strength & Conditioning (S&C) exercises inspired by Deyzel et al. (2023) (*squat*, *deadlift*, *barbell biceps curl*, *lateral raise*) over 54 held-out test videos ($N=529$ windows).  
*Execution Command:* `python scripts/evaluate_external_benchmark.py`

| Model Architecture | Open-Set Test Win Acc (%) | Open-Set Test Vid Acc (%) | Closed-Set Test Win Acc (%) | Closed-Set Test Vid Acc (%) | Closed-Set Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN (Rel 3D) (Deyzel baseline)** | 67.49% | 77.78% | 82.42% | 88.89% | 0.8901 | Verified |
| **LSTM (Mix 117-d)** | 63.89% | 77.78% | 78.83% | 87.04% | 0.8746 | Verified |
| **BiLSTM (Mix 117-d)** | 62.38% | 79.63% | 76.75% | 90.74% | 0.9119 | Verified |
| **AAGCN (Bone 3D)** | 74.67% | 94.44% | 87.15% | 94.44% | 0.9443 | Verified |
| **Transformer (Mix 117-d)** | 85.82% | 96.30% | 95.27% | 96.30% | 0.9615 | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 82.80% | 96.30% | 94.14% | 96.30% | 0.9615 | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 84.50% | 98.15% | 93.95% | 98.15% | 0.9782 | **Verified** |
