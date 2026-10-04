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
| **T1.1** | LSTM | Raw 2D Coordinates | 26 | - | - | - | - | - | - | - | Pending |
| **T1.2** | LSTM | Relative 2D (Mid-Hip) | 26 | - | - | - | - | - | - | - | Pending |
| **T1.3** | LSTM | Angle 2D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.4** | LSTM | Angle2 2D (Pairs) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.5** | LSTM | Raw 3D Coordinates | 39 | - | - | - | - | - | - | - | Pending |
| **T1.6** | LSTM | Relative 3D (Mid-Hip) | 39 | - | - | - | - | - | - | - | Pending |
| **T1.7** | LSTM | Angle 3D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.8** | LSTM | Angle2 3D (Pair Elevation) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.9** | LSTM | Biomechanical Mix v2 (Ours) | 63 | - | - | - | - | - | - | - | Pending |
| **T1.10** | BiLSTM | Raw 2D Coordinates | 26 | - | - | - | - | - | - | - | Pending |
| **T1.11** | BiLSTM | Relative 2D (Mid-Hip) | 26 | - | - | - | - | - | - | - | Pending |
| **T1.12** | BiLSTM | Angle 2D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.13** | BiLSTM | Angle2 2D (Pairs) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.14** | BiLSTM | Raw 3D Coordinates | 39 | - | - | - | - | - | - | - | Pending |
| **T1.15** | BiLSTM | Relative 3D (Mid-Hip) | 39 | - | - | - | - | - | - | - | Pending |
| **T1.16** | BiLSTM | Angle 3D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.17** | BiLSTM | Angle2 3D (Pair Elevation) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.18** | BiLSTM | Biomechanical Mix v2 (Ours) | 63 | - | - | - | - | - | - | - | Pending |
| **T1.19** | Transformer | Raw 2D Coordinates | 26 | - | - | - | - | - | - | - | Pending |
| **T1.20** | Transformer | Relative 2D (Mid-Hip) | 26 | - | - | - | - | - | - | - | Pending |
| **T1.21** | Transformer | Angle 2D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.22** | Transformer | Angle2 2D (Pairs) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.23** | Transformer | Raw 3D Coordinates | 39 | - | - | - | - | - | - | - | Pending |
| **T1.24** | Transformer | Relative 3D (Mid-Hip) | 39 | - | - | - | - | - | - | - | Pending |
| **T1.25** | Transformer | Angle 3D (Triplets) | 286 | - | - | - | - | - | - | - | Pending |
| **T1.26** | Transformer | Angle2 3D (Pair Elevation) | 78 | - | - | - | - | - | - | - | Pending |
| **T1.27** | Transformer | Biomechanical Mix v2 (Ours) | 63 | - | - | - | - | - | - | - | Pending |

---

## Table 2b: Multi-Seed Controlled-Capacity Transformer Feature Benchmark (300K Budget, Seeds 42, 123, 3407)

*Objective:* Multi-seed validation of Raw 3D, Scale-Norm Relative 3D, and Biomechanical Mix v2 on Transformer (~301K params).  
*Execution Command:* `python scripts/evaluate_upgrade_mix_all.py`

| Feature Paradigm | Input Dim | Params | Train Loss | Val Loss | Val Win Acc (%) | Val Win Macro F1 | Val Vid Acc (%) | Val Vid Macro F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw 3D Coordinates** | 39 | 301K | - | - | - | - | - | - | - | - | - | - | Pending |
| **Scale-Norm Rel 3D (`rel_3d_norm`)** | 39 | 301K | - | - | - | - | - | - | - | - | - | - | Pending |
| **Biomechanical Mix v2 (`mix_v2`)** | 63 | 300K | - | - | - | - | - | - | - | - | - | - | Pending |

---

## Table 3: Spatial-Temporal Graph Kinematic Streams (Paper Table 4)

*Objective:* Evaluate static physical adjacency ($A_{\text{phys}}$) versus learnable adaptive topology ($B_k + C_k$) across 4 kinematic modalities.  
*Execution Command:* `python run.py train --model AAGCN --feature <STREAM> --augment skel_gym_aug --device auto`

| Exp ID | Model Architecture | Kinematic Stream | Augmentation Protocol | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Val Macro F1 | Test Win Acc (%) | Test Vid Acc (%) | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **T3.1** | ST-GCN Baseline | Raw 3D Joint | None (Clean) | - | - | - | - | - | - | - | Pending |
| **T3.2** | ST-GCN Baseline | Relative 3D Joint | None (Clean) | - | - | - | - | - | - | - | Pending |
| **T3.3** | AAGCN Baseline | Bone 3D Stream | None (Clean) | - | - | - | - | - | - | - | Pending |
| **T3.4** | AAGCN | Bone 3D Stream | SkelGym-Aug (Proposed) | - | - | - | - | - | - | - | Pending |
| **T3.5** | AAGCN | Joint Stream (Rel 3D) | SkelGym-Aug (Proposed) | - | - | - | - | - | - | - | Pending |
| **T3.6** | AAGCN | Joint Motion 3D ($\Delta X$) | SkelGym-Aug (Proposed) | - | - | - | - | - | - | - | Pending |
| **T3.7** | AAGCN | Bone Motion 3D ($\Delta B$) | SkelGym-Aug (Proposed) | - | - | - | - | - | - | - | Pending |
| **T3.8** | Two-Stream AAGCN | Joint + Bone | Late Fusion (Equal Weights) | - | - | - | - | - | - | - | Pending |
| **T3.9** | Four-Stream AAGCN | 4 Streams Unified | Late Fusion (SLSQP Calibrated) | - | - | - | - | - | - | - | Pending |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | $\Delta$ vs Full (Test Win) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | - | - | - | - | - | 0.00% (Ref) | Pending |
| **Minus Noise / Jitter** | Gaussian Coordinate Jitter ($\sigma=0.008$) | - | - | - | - | - | - | Pending |
| **Minus Mirroring** | Sagittal Horizontal Flip ($p=0.5$) | - | - | - | - | - | - | Pending |
| **Minus Rotation / Yaw** | Gravitational Yaw Rotation ($\pm 15^\circ$) | - | - | - | - | - | - | Pending |
| **Minus Scaling** | Proportional Scale Variation ($\pm 10\%$) | - | - | - | - | - | - | Pending |
| **Minus Time Interpolation** | Temporal Resampling ($0.8\times - 1.2\times$) | - | - | - | - | - | - | Pending |

---

## Table 5: Single-Component Isolated Augmentation Ablation on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone individual gain for each augmentation operator relative to unaugmented baseline.  
*Execution Command:* `python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain`

| Augmentation Strategy | Isolated Operator Description | Val Loss | Val Window Acc (%) | Val Macro F1 | Test Window Acc (%) | Test Macro F1 | Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **None (Clean Baseline)** | Unaugmented Native Window Sequences | - | - | - | - | - | 0.00% (Ref) | Pending |
| **Only Jitter** | Gaussian Noise ($\sigma=0.008$) | - | - | - | - | - | - | Pending |
| **Only Mirroring** | Sagittal Bilateral Reflection | - | - | - | - | - | - | Pending |
| **Only Rotation** | 3D Yaw Perturbation ($\pm 15^\circ$) | - | - | - | - | - | - | Pending |
| **Only Scaling** | Proportional Scale Jitter ($\pm 10\%$) | - | - | - | - | - | - | Pending |
| **Only Time Interpolation** | Linear Sequence Resampling | - | - | - | - | - | - | Pending |

---

## Table 6: Multi-Stream Cross-Paradigm Ensemble Comparison (Paper Table 6)

*Objective:* Benchmark 5 systematic fusion methods and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Execution Command:* `python scripts/run_multi_seed_experiments.py --seeds 42 123 3407`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix (63-d)** | Single Sequence Backbone | - | - | - | - | - | - | - | - | Pending |
| **AAGCN Bone Stream** | Single Graph Backbone | - | - | - | - | - | - | - | - | Pending |
| **Four-Stream AAGCN** | 4 Streams Unified Graph | - | - | - | - | - | - | - | - | Pending |
| **Hard Majority Voting** | Discrete mode over class predictions | - | - | - | - | - | - | - | - | Pending |
| **Uniform Average Soft Voting** | Equal weights ($w_i = 1/5 = 0.20$) | - | - | - | - | - | - | - | - | Pending |
| **Accuracy-Weighted Soft Voting** | Validation accuracy weights | - | - | - | - | - | - | - | - | Pending |
| **SLSQP Soft Voting** | SLSQP Constrained Calibration ($\sum w_i = 1$) | - | - | - | - | - | - | - | - | Pending |
| **Stacking Meta-Classifier** | Ridge Classifier on Val Probs | - | - | - | - | - | - | - | - | Pending |
| **SkelGym-Lite (2 Models)** | Trans + Bone AAGCN (SLSQP Calibrated) | - | - | - | - | - | - | - | - | Pending |
| **SkelGym-Full (5 Streams)** | Trans + 4 AAGCN (SLSQP Calibrated) | - | - | - | - | - | - | - | - | Pending |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all 11 architectures.  
*Execution Command:* `python run.py evaluate --checkpoint <CKPT> --video_level --device auto`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Video Gain (+$\Delta$%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 63-d)** | Sequential Recurrent Model | ~378K | - | - | - | - | - | Pending |
| **Baseline BiLSTM (Mix 63-d)** | Bidirectional Recurrent Model | ~367K | - | - | - | - | - | Pending |
| **Transformer (Mix 63-d, Clean)** | Self-Attention Baseline | ~300K | - | - | - | - | - | Pending |
| **Baseline ST-GCN (Rel 3D)** | Rigid Static Graph ($A_{\text{phys}}$) | ~350K | - | - | - | - | - | Pending |
| **Clean Baseline AAGCN (Bone 3D)** | Adaptive Skeletal Graph (Unaugmented) | ~378K | - | - | - | - | - | Pending |
| **SkelGym-Aug AAGCN (Bone 3D)** | Adaptive Skeletal Graph + Augmentation | ~378K | - | - | - | - | - | Pending |
| **SkelGym-Aug Transformer (Mix)** | Self-Attention + Augmentation | ~300K | - | - | - | - | - | Pending |
| **Two-Stream AAGCN (Aug)** | Joint + Bone Stream Fusion | ~756K | - | - | - | - | - | Pending |
| **Four-Stream AAGCN (Aug)** | 4-Stream Graph Late Fusion | ~1.51M | - | - | - | - | - | Pending |
| **SkelGym-Lite (2 Models)** | Transformer + Bone AAGCN | ~678K | - | - | - | - | - | Pending |
| **SkelGym-Full (5 Streams)** | Cross-Paradigm SLSQP Ensemble | ~1.81M | - | - | - | - | - | Pending |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | - | - | - | - | - | - | - | Pending |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | - | - | - | - | - | - | - | Pending |
| **Single Sequence (Trans) vs SkelGym-Full** | - | - | - | - | - | - | - | Pending |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | - | - | - | - | - | - | - | Pending |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | - | - | - | - | - | - | - | Pending |

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling.  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix 63-d)** | - | - | - | - | Pending |
| **BiLSTM (Mix 63-d)** | - | - | - | - | Pending |
| **ST-GCN (Rel 3D)** | - | - | - | - | Pending |
| **Transformer (Mix 63-d)** | - | - | - | - | Pending |
| **AAGCN (Bone 3D)** | - | - | - | - | Pending |
| **SkelGym-Lite (2 Models)** | - | - | - | - | Pending |
| **SkelGym-Full (5 Streams)** | - | - | - | - | Pending |

---

## Table 10: Per-Class Performance Breakdown (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full on held-out test windows ($N=2,743$) and test videos ($N=233$).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **barbell biceps curl** | - | - | - | 69 | - | - | - | 14 | Pending |
| **bench press** | - | - | - | 96 | - | - | - | 14 | Pending |
| **chest fly machine** | - | - | - | 82 | - | - | - | 8 | Pending |
| **deadlift** | - | - | - | 67 | - | - | - | 10 | Pending |
| **decline bench press** | - | - | - | 134 | - | - | - | 8 | Pending |
| **hammer curl** | - | - | - | 161 | - | - | - | 14 | Pending |
| **hip thrust** | - | - | - | 236 | - | - | - | 9 | Pending |
| **incline bench press** | - | - | - | 76 | - | - | - | 9 | Pending |
| **lat pulldown** | - | - | - | 100 | - | - | - | 13 | Pending |
| **lateral raise** | - | - | - | 155 | - | - | - | 15 | Pending |
| **leg extension** | - | - | - | 125 | - | - | - | 13 | Pending |
| **leg raises** | - | - | - | 114 | - | - | - | 11 | Pending |
| **plank** | - | - | - | 56 | - | - | - | 2 | Pending |
| **pull Up** | - | - | - | 82 | - | - | - | 10 | Pending |
| **push-up** | - | - | - | 87 | - | - | - | 12 | Pending |
| **romanian deadlift** | - | - | - | 146 | - | - | - | 6 | Pending |
| **russian twist** | - | - | - | 137 | - | - | - | 6 | Pending |
| **shoulder press** | - | - | - | 152 | - | - | - | 13 | Pending |
| **squat** | - | - | - | 238 | - | - | - | 15 | Pending |
| **t bar row** | - | - | - | 117 | - | - | - | 10 | Pending |
| **tricep Pushdown** | - | - | - | 93 | - | - | - | 12 | Pending |
| **tricep dips** | - | - | - | 220 | - | - | - | 9 | Pending |
| **Overall Accuracy** | - | - | - | **2743** | - | - | - | **233** | **Pending** |
| **Macro Average** | - | - | - | **2743** | - | - | - | **233** | **Pending** |

---

## Table 11: Computational Complexity & Inference Latency (Paper Table 11)

*Objective:* Measure parameters, window FLOPs, and latency across server and edge devices.  
*Execution Command:* `python scripts/benchmark_hardware_latency.py`

| Model Architecture | Parameters | FLOPs per Window | RTX PRO 6000 (CUDA) | Apple M4 (MPS) | Apple M4 (CPU) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (63-d)** | ~300K | - | - | - | - | Pending |
| **AAGCN Joint Stream** | ~378K | - | - | - | - | Pending |
| **AAGCN Bone Stream** | ~378K | - | - | - | - | Pending |
| **AAGCN Joint-Motion Stream** | ~378K | - | - | - | - | Pending |
| **AAGCN Bone-Motion Stream** | ~378K | - | - | - | - | Pending |
| **SkelGym-Lite (Transformer + Bone)** | ~678K | - | - | - | - | Pending |
| **SkelGym-Full (Transformer + 4 AAGCN)** | ~1.81M | - | - | - | - | Pending |
