# Master Experiment Results: Deep Learning for Gym Exercise Classification

This document serves as the primary tracking log and benchmark sheet for the research paper. All experimental results are systematically categorized into structured tables corresponding directly to the paper's narrative and evaluation phases.

### Experimental Protocol & Standards (MediaPipe Pose Heavy SOTA Dataset)
- **Dataset Source of Truth:** 1,024 clean, un-shuffled video recordings across 22 classes from Kaggle (`nguyenxuancuongk18dn/gym-exercise-classification-dataset`).
  - **Train Set:** 580 videos (266,717 frames $\rightarrow$ 15,820 windows at $T=32$, stride 16).
  - **Validation Set:** 208 videos (72,105 frames $\rightarrow$ 2,147 windows at $T=32$, stride 32).
  - **Held-out Test Set:** 236 videos (110,577 frames $\rightarrow$ 3,337 windows at $T=32$, stride 32).
- **Landmark Model:** MediaPipe Pose `model_complexity = 2` (Heavy) with 33 full-body landmarks (13 calibrated keypoints used for spatial body kinematics).
- **Controlled Parameter Budget:** All individual backbones calibrated to $\approx 350\text{K} \pm 15\%$ parameters for strict fairness.
- **Optimization:** Adam / AdamW optimizer, Cosine Annealing with Warmup or ReduceLROnPlateau, fixed 100 epochs, early stopping patience = 10, Automatic Mixed Precision (AMP) enabled.

---

## Table 1: Temporal Models on Landmark Feature Sets (Controlled Budget $\approx 350\text{K}$)
*Objective:* Benchmark 3 temporal architectures (LSTM, BiLSTM, Transformer) across 7 coordinate & angular representations (2D/3D raw, relative, angles, and unified mix).

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T1.1** | **LSTM** | raw_2d | 26 | 0.2236 | 2.1459 | 60.63% | 40.79% | 0.4232 | `checkpoints/best_LSTM_T1.1_raw_2d.pt` | Done |
| **T1.2** | **LSTM** | rel_2d | 26 | 0.2302 | 1.8470 | 65.68% | 49.20% | 0.4969 | `checkpoints/best_LSTM_T1.2_rel_2d.pt` | Done |
| **T1.3** | **LSTM** | angle_2d | 286 | 0.1130 | 2.6723 | 62.53% | 40.59% | 0.4215 | `checkpoints/best_LSTM_T1.3_angle_2d.pt` | Done |
| **T1.4** | **LSTM** | raw_3d | 39 | 0.1184 | 2.7792 | 61.03% | 46.08% | 0.4631 | `checkpoints/best_LSTM_T1.4_raw_3d.pt` | Done |
| **T1.5** | **LSTM** | rel_3d | 39 | 0.1391 | 2.3413 | 65.23% | 51.84% | 0.5070 | `checkpoints/best_LSTM_T1.5_rel_3d.pt` | Done |
| **T1.6** | **LSTM** | angle_3d | 429 | 0.1719 | 2.3113 | 59.92% | 41.46% | 0.4133 | `checkpoints/best_LSTM_T1.6_angle_3d.pt` | Done |
| **T1.7** | **LSTM** | mix | 754 | 0.0109 | 3.5754 | 64.26% | 48.37% | 0.4931 | `checkpoints/best_LSTM_T1.7_mix.pt` | Done |
| **T1.8** | **BiLSTM** | raw_2d | 26 | 0.2606 | 1.9333 | 63.29% | 44.32% | 0.4515 | `checkpoints/best_BiLSTM_T1.8_raw_2d.pt` | Done |
| **T1.9** | **BiLSTM** | rel_2d | 26 | 0.2950 | 1.9242 | 65.10% | 50.19% | 0.5113 | `checkpoints/best_BiLSTM_T1.9_rel_2d.pt` | Done |
| **T1.10** | **BiLSTM** | angle_2d | 286 | 0.0443 | 3.7970 | 58.37% | 42.82% | 0.4287 | `checkpoints/best_BiLSTM_T1.10_angle_2d.pt` | Done |
| **T1.11** | **BiLSTM** | raw_3d | 39 | 0.1428 | 2.1928 | 60.01% | 44.23% | 0.4497 | `checkpoints/best_BiLSTM_T1.11_raw_3d.pt` | Done |
| **T1.12** | **BiLSTM** | rel_3d | 39 | 0.0998 | 2.2450 | 66.39% | 50.80% | 0.5125 | `checkpoints/best_BiLSTM_T1.12_rel_3d.pt` | Done |
| **T1.13** | **BiLSTM** | angle_3d | 429 | 0.0978 | 2.9793 | 58.59% | 42.79% | 0.4271 | `checkpoints/best_BiLSTM_T1.13_angle_3d.pt` | Done |
| **T1.14** | **BiLSTM** | mix | 754 | 0.0760 | 2.2194 | 64.48% | 47.44% | 0.4853 | `checkpoints/best_BiLSTM_T1.14_mix.pt` | Done |
| **T1.15** | **Transformer** | raw_2d | 26 | 0.0254 | 1.6511 | 72.63% | 53.98% | 0.5548 | `checkpoints/best_Transformer_T1.15_raw_2d.pt` | Done |
| **T1.16** | **Transformer** | rel_2d | 26 | 0.0138 | 2.0363 | 71.97% | 56.61% | 0.5657 | `checkpoints/best_Transformer_T1.16_rel_2d.pt` | Done |
| **T1.17** | **Transformer** | angle_2d | 286 | 0.0660 | 2.0764 | 65.94% | 48.68% | 0.4961 | `checkpoints/best_Transformer_T1.17_angle_2d.pt` | Done |
| **T1.18** | **Transformer** | raw_3d | 39 | 0.1300 | 1.1634 | 72.76% | 57.21% | 0.5864 | `checkpoints/best_Transformer_T1.18_raw_3d.pt` | Done |
| **T1.19** | **Transformer** | rel_3d | 39 | 0.0436 | 1.5060 | 71.83% | 56.87% | 0.5772 | `checkpoints/best_Transformer_T1.19_rel_3d.pt` | Done |
| **T1.20** | **Transformer** | angle_3d | 429 | 0.0440 | 2.2274 | 64.70% | 43.39% | 0.4483 | `checkpoints/best_Transformer_T1.20_angle_3d.pt` | Done |
| **T1.21** | **Transformer** | mix | 754 | 0.0793 | 1.8965 | 66.39% | 47.09% | 0.4879 | `checkpoints/best_Transformer_T1.21_mix.pt` | Done |

---

## Table 2: Data Augmentation Strategies on Best Transformer
*Objective:* Assess whether dynamic on-the-fly augmentation (SkelGym-Aug with bilateral symmetry mirroring, 3D yaw rotation, and temporal warping) outperforms the unaugmented clean baseline.

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline)** | `augment: none` (Clean original samples) | 0.0793 | 1.8965 | 66.39% | 47.09% | 0.4879 | `checkpoints/best_Transformer_T2.1_mix.pt` | Done |
| **T2.2** | **SkelGym-Aug (Dynamic On-the-Fly)** | `augment: skel_gym_aug` (Bilateral Flip $p=0.5$, 3D Yaw $\pm 15^\circ$, Scale, Time-Warp, Jitter) | 0.0985 | 1.6043 | 68.60% | 51.92% | 0.5305 | `checkpoints/best_Transformer_T2.2_mix.pt` | Done |

---

## Table 3: Spatial-Temporal Graph Models & Multi-Stream AAGCN Kinematics
*Objective:* Evaluate Graph Neural Networks under spatial-temporal skeletal graph topology $(B, C, T, V)$, systematically benchmarking baseline ST-GCN against dynamic Adaptive Graph Convolutional Networks (AAGCN) and their multi-stream fusion ablations.

| Exp ID | Model Architecture | Graph Stream | Tensor Shape $(C, T, V)$ | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T3.1** | **ST-GCN** | raw_3d | $(3, 32, 13)$ | 0.2922 | 2.1152 | 54.69% | 42.21% | 0.4266 | `checkpoints/best_STGCN_T3.1_raw_3d.pt` | Done |
| **T3.2** | **ST-GCN** | rel_3d | $(3, 32, 13)$ | 0.2504 | 1.8976 | 62.49% | 43.57% | 0.4595 | `checkpoints/best_STGCN_T3.2_rel_3d.pt` | Done |
| **T3.3** | **ST-GCN** | raw_2d | $(2, 32, 13)$ | 0.3915 | 2.4482 | 48.10% | 43.89% | 0.4339 | `checkpoints/best_STGCN_T3.3_raw_2d.pt` | Done |
| **T3.4** | **ST-GCN** | rel_2d | $(2, 32, 13)$ | 0.7506 | 1.9097 | 50.18% | 43.08% | 0.4235 | `checkpoints/best_STGCN_T3.4_rel_2d.pt` | Done |
| **T3.5** | **AAGCN (Adaptive GCN)** | rel_3d (Joint Stream) | $(3, 32, 13)$ | 0.0778 | 2.0461 | 68.73% | 54.00% | 0.5362 | `checkpoints/best_AAGCN_T3.5_rel_3d.pt` | Done |
| **T3.6** | **AAGCN (Adaptive GCN)** | bone_3d (Bone Stream) | $(3, 32, 13)$ | 0.2273 | 1.5015 | 69.57% | 52.27% | 0.5338 | `checkpoints/best_AAGCN_T3.6_bone_3d.pt` | Done |
| **T3.7** | **AAGCN (Adaptive GCN)** | joint_motion_3d ($\Delta X$) | $(3, 32, 13)$ | 0.2709 | 2.6010 | 46.99% | 40.13% | 0.4377 | `checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt` | Done |
| **T3.8** | **AAGCN (Adaptive GCN)** | bone_motion_3d ($\Delta B$) | $(3, 32, 13)$ | 0.5334 | 2.1556 | 47.25% | 42.35% | 0.4596 | `checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt` | Done |
| **T3.9** | **Two-Stream AAGCN** | Joint + Bone Stream Fusion | Late Fusion ($T=32$) | - | - | - | 54.38% | 0.5496 | `outputs/ensemble/cm_ensemble_T3.9_weighted_soft.png` | Done |
| **T3.10** | **Four-Stream AAGCN** | Joint + Bone + J-Motion + B-Motion | Late Fusion ($T=32$) | - | - | - | 56.03% | 0.5668 | `outputs/ensemble/cm_ensemble_T3.10_weighted_soft.png` | Done |

---

## Table 4: Data Augmentation Ablation on Graph Architectures
*Objective:* Empirical ablation assessing the effectiveness of dynamic skeletal data augmentation (SkelGym-Aug: bilateral symmetry mirroring, 3D yaw rotation $\pm 15^\circ$, time-warping, jitter) on spatial-temporal graph models (ST-GCN and AAGCN).

| Exp ID | Model Architecture | Graph Stream | Augmentation Strategy | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **ST-GCN** | rel_3d (Joint) | `augment: none` (Baseline) | 0.2504 | 1.8976 | 62.49% | 43.57% | 0.4595 | `checkpoints/best_STGCN_T4.1_rel_3d.pt` | Done |
| **T4.2** | **ST-GCN** | rel_3d (Joint) | `augment: skel_gym_aug` (Dynamic) | 0.5803 | 1.5091 | 60.19% | 52.24% | 0.5159 | `checkpoints/best_STGCN_T4.2_rel_3d.pt` | Done |
| **T4.3** | **AAGCN** | rel_3d (Joint) | `augment: none` (Baseline) | 0.0778 | 2.0461 | 68.73% | 54.00% | 0.5362 | `checkpoints/best_AAGCN_T4.3_rel_3d.pt` | Done |
| **T4.4** | **AAGCN** | rel_3d (Joint) | `augment: skel_gym_aug` (Dynamic) | 0.3145 | 1.2315 | 68.87% | 57.18% | 0.5710 | `checkpoints/best_AAGCN_T4.4_rel_3d.pt` | Done |

---

## Table 5: Heterogeneous Cross-Paradigm Ensemble (Transformer + Graph Models)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Table 1/2) and spatial-temporal graph models (Table 3/4) to establish SOTA accuracy.

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Hard Voting** | Best Transformer + Best ST-GCN | 42.12% | 0.4572 | 0.4414 | `outputs/ensemble/cm_ensemble_hard.png` | Done |
| **T5.2** | **Soft Voting** | Best Transformer + Best ST-GCN | 49.47% | 0.5165 | 0.5055 | `outputs/ensemble/cm_ensemble_soft.png` | Done |
| **T5.3** | **Stacking Ensemble** | Best Transformer + Best ST-GCN + Meta-Learner | 50.16% | 0.5171 | 0.4978 | `outputs/ensemble/cm_ensemble_stacking.png` | Done |
| **T5.4** | **Tri-Model Grand Ensemble** | Transformer Mix + AAGCN Joint + AAGCN Bone | 54.78% | 0.5584 | 0.5612 | `outputs/ensemble/cm_ensemble_T5.4_weighted_soft.png` | Done |
| **T5.5** | **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | 56.11% | 0.5715 | 0.5771 | `outputs/ensemble/cm_ensemble_T5.5_weighted_soft.png` | Done |

---

## Table 6A: Detailed Classification Report (Window-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble across all 22 gym exercise categories on the held-out test windows ($N=3,337$ windows at $T=32$, stride 32).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.2321 | 0.7536 | 0.3549 | 69 |
| bench press | 0.2736 | 0.5918 | 0.3742 | 98 |
| chest fly machine | 0.7115 | 0.9024 | 0.7957 | 82 |
| deadlift | 0.3818 | 0.6269 | 0.4746 | 67 |
| decline bench press | 0.1584 | 0.2656 | 0.1984 | 192 |
| hammer curl | 0.2407 | 0.3077 | 0.2701 | 169 |
| hip thrust | 0.7843 | 0.5303 | 0.6328 | 528 |
| incline bench press | 0.6731 | 0.4430 | 0.5344 | 79 |
| lat pulldown | 0.6716 | 0.8571 | 0.7531 | 105 |
| lateral raise | 0.8176 | 0.8688 | 0.8424 | 160 |
| leg extension | 0.8960 | 0.7778 | 0.8327 | 144 |
| leg raises | 0.9722 | 0.3017 | 0.4605 | 116 |
| plank | 0.8889 | 0.4286 | 0.5783 | 56 |
| pull Up | 0.5842 | 0.6705 | 0.6243 | 88 |
| push-up | 0.6164 | 0.9890 | 0.7595 | 91 |
| romanian deadlift | 0.4153 | 0.2952 | 0.3451 | 166 |
| russian twist | 0.8496 | 0.7584 | 0.8014 | 149 |
| shoulder press | 0.6016 | 0.3500 | 0.4425 | 220 |
| squat | 0.8222 | 0.5781 | 0.6789 | 256 |
| t bar row | 0.3641 | 0.5194 | 0.4281 | 129 |
| tricep Pushdown | 0.7059 | 0.7742 | 0.7385 | 93 |
| tricep dips | 0.7986 | 0.5522 | 0.6529 | 402 |
| **Accuracy** | | | **56.11%** | **3459** |
| **Macro avg** | **0.6118** | **0.5974** | **0.5715** | **3459** |
| **Weighted avg** | **0.6469** | **0.5611** | **0.5771** | **3459** |

---

## Table 6B: Detailed Classification Report (Video-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble aggregated at the full exercise clip level across all 236 independent held-out test videos ($N=236$ videos).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.5000 | 0.9286 | 0.6500 | 14 |
| bench press | 0.4737 | 0.6000 | 0.5294 | 15 |
| chest fly machine | 0.8750 | 0.8750 | 0.8750 | 8 |
| deadlift | 0.7500 | 0.9000 | 0.8182 | 10 |
| decline bench press | 0.0714 | 0.1111 | 0.0870 | 9 |
| hammer curl | 0.3333 | 0.1429 | 0.2000 | 14 |
| hip thrust | 0.7778 | 0.7778 | 0.7778 | 9 |
| incline bench press | 1.0000 | 0.6667 | 0.8000 | 9 |
| lat pulldown | 0.8125 | 0.9286 | 0.8667 | 14 |
| lateral raise | 1.0000 | 1.0000 | 1.0000 | 15 |
| leg extension | 1.0000 | 0.8462 | 0.9167 | 13 |
| leg raises | 1.0000 | 0.4545 | 0.6250 | 11 |
| plank | 1.0000 | 0.5000 | 0.6667 | 2 |
| pull Up | 0.8750 | 0.7000 | 0.7778 | 10 |
| push-up | 0.8571 | 1.0000 | 0.9231 | 12 |
| romanian deadlift | 0.6000 | 0.5000 | 0.5455 | 6 |
| russian twist | 0.8333 | 0.8333 | 0.8333 | 6 |
| shoulder press | 0.6667 | 0.6154 | 0.6400 | 13 |
| squat | 1.0000 | 0.6667 | 0.8000 | 15 |
| t bar row | 0.5833 | 0.7000 | 0.6364 | 10 |
| tricep Pushdown | 1.0000 | 0.9167 | 0.9565 | 12 |
| tricep dips | 0.9000 | 1.0000 | 0.9474 | 9 |
| **Accuracy** | | | **72.46%** | **236** |
| **Macro avg** | **0.7686** | **0.7120** | **0.7215** | **236** |
| **Weighted avg** | **0.7607** | **0.7246** | **0.7249** | **236** |

---

## Table 7: Video-Level Aggregation Summary Benchmark
*Objective:* Empirical comparison between temporal window-level predictions and complete clip/video-level predictions ($N=236$ complete test videos), analyzing the accuracy gains achievable through temporal consensus pooling.

| Model / Ensemble Architecture | Input Modality / Paradigm | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 | Video Gain (+Δ%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix)** | Sequential Recurrent Model | 48.37% | 0.4931 | 67.37% | 0.6575 | +19.01% |
| **Baseline BiLSTM (Mix)** | Bidirectional Recurrent Model | 47.44% | 0.4853 | 63.98% | 0.6060 | +16.54% |
| **Baseline ST-GCN (Rel 3D)** | Static Graph Convolution | 43.57% | 0.4595 | 58.47% | 0.5817 | +14.91% |
| **Best Transformer (Mix)** | Self-Attention on Kinematic Vectors | 47.09% | 0.4879 | 65.25% | 0.6312 | +18.16% |
| **Best Transformer + SkelGym-Aug** | Self-Attention with Dynamic Aug | 51.92% | 0.5305 | 66.53% | 0.6716 | +14.60% |
| **Two-Stream AAGCN** | Joint Stream + Bone Stream | 54.38% | 0.5496 | 71.61% | 0.7146 | +17.23% |
| **Four-Stream AAGCN** | Joint + Bone + Joint Motion + Bone Motion | 56.03% | 0.5668 | 73.31% | 0.7313 | +17.28% |
| **Tri-Model Grand Ensemble** | Transformer Mix + 2-Stream AAGCN | 54.78% | 0.5584 | 71.19% | 0.7033 | +16.40% |
| **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | 56.11% | 0.5715 | 72.46% | 0.7215 | +16.34% |
