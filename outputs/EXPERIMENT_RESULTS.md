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

### Table 1: Temporal Models on Landmark Feature Sets (Controlled Budget $\approx 350\text{K}$)
*Objective:* Benchmark 3 temporal architectures (LSTM, BiLSTM, Transformer) across 9 coordinate & angular representations (2D/3D raw, relative, 3-point angles, 2-point relative angles, and unified mix).

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T1.1** | **LSTM** | raw_2d | 26 | 0.2236 | 2.1459 | 60.63% | 40.79% | 0.4232 | `checkpoints/best_LSTM_T1.1_raw_2d.pt` | Done |
| **T1.2** | **LSTM** | rel_2d | 26 | 0.2302 | 1.8470 | 65.68% | 49.20% | 0.4969 | `checkpoints/best_LSTM_T1.2_rel_2d.pt` | Done |
| **T1.3** | **LSTM** | angle_2d | 286 | 0.1130 | 2.6723 | 62.53% | 40.59% | 0.4215 | `checkpoints/best_LSTM_T1.3_angle_2d.pt` | Done |
| **T1.4** | **LSTM** | angle2_2d | 78 | 0.1984 | 2.4102 | 61.12% | 42.15% | 0.4320 | `checkpoints/best_LSTM_T1.4_angle2_2d.pt` | Done |
| **T1.5** | **LSTM** | raw_3d | 39 | 0.1184 | 2.7792 | 61.03% | 46.08% | 0.4631 | `checkpoints/best_LSTM_T1.5_raw_3d.pt` | Done |
| **T1.6** | **LSTM** | rel_3d | 39 | 0.1391 | 2.3413 | 65.23% | 51.84% | 0.5070 | `checkpoints/best_LSTM_T1.6_rel_3d.pt` | Done |
| **T1.7** | **LSTM** | angle_3d | 286 | 0.1719 | 2.3113 | 59.92% | 41.46% | 0.4133 | `checkpoints/best_LSTM_T1.7_angle_3d.pt` | Done |
| **T1.8** | **LSTM** | angle2_3d | 78 | 0.1452 | 2.1205 | 63.40% | 46.12% | 0.4710 | `checkpoints/best_LSTM_T1.8_angle2_3d.pt` | Done |
| **T1.9** | **LSTM** | mix (rel_3d + angle2_3d) | 117 | 0.1820 | 1.3346 | 73.40% | 57.67% | 0.5681 | `checkpoints/best_LSTM_T1.9_mix.pt` | Done |
| **T1.10** | **BiLSTM** | raw_2d | 26 | 0.2606 | 1.9333 | 63.29% | 44.32% | 0.4515 | `checkpoints/best_BiLSTM_T1.10_raw_2d.pt` | Done |
| **T1.11** | **BiLSTM** | rel_2d | 26 | 0.2950 | 1.9242 | 65.10% | 50.19% | 0.5113 | `checkpoints/best_BiLSTM_T1.11_rel_2d.pt` | Done |
| **T1.12** | **BiLSTM** | angle_2d | 286 | 0.0443 | 3.7970 | 58.37% | 42.82% | 0.4287 | `checkpoints/best_BiLSTM_T1.12_angle_2d.pt` | Done |
| **T1.13** | **BiLSTM** | angle2_2d | 78 | 0.1821 | 2.1540 | 62.15% | 45.30% | 0.4612 | `checkpoints/best_BiLSTM_T1.13_angle2_2d.pt` | Done |
| **T1.14** | **BiLSTM** | raw_3d | 39 | 0.1428 | 2.1928 | 60.01% | 44.23% | 0.4497 | `checkpoints/best_BiLSTM_T1.14_raw_3d.pt` | Done |
| **T1.15** | **BiLSTM** | rel_3d | 39 | 0.0998 | 2.2450 | 66.39% | 50.80% | 0.5125 | `checkpoints/best_BiLSTM_T1.15_rel_3d.pt` | Done |
| **T1.16** | **BiLSTM** | angle_3d | 286 | 0.0978 | 2.9793 | 58.59% | 42.79% | 0.4271 | `checkpoints/best_BiLSTM_T1.16_angle_3d.pt` | Done |
| **T1.17** | **BiLSTM** | angle2_3d | 78 | 0.1245 | 2.0512 | 64.10% | 47.90% | 0.4855 | `checkpoints/best_BiLSTM_T1.17_angle2_3d.pt` | Done |
| **T1.18** | **BiLSTM** | mix (rel_3d + angle2_3d) | 117 | 0.0897 | 1.4966 | 73.30% | 61.17% | 0.6008 | `checkpoints/best_BiLSTM_T1.18_mix.pt` | Done |
| **T1.19** | **Transformer** | raw_2d | 26 | 0.0254 | 1.6511 | 72.63% | 53.98% | 0.5548 | `checkpoints/best_Transformer_T1.19_raw_2d.pt` | Done |
| **T1.20** | **Transformer** | rel_2d | 26 | 0.0138 | 2.0363 | 71.97% | 56.61% | 0.5657 | `checkpoints/best_Transformer_T1.20_rel_2d.pt` | Done |
| **T1.21** | **Transformer** | angle_2d | 286 | 0.0660 | 2.0764 | 65.94% | 48.68% | 0.4961 | `checkpoints/best_Transformer_T1.21_angle_2d.pt` | Done |
| **T1.22** | **Transformer** | angle2_2d | 78 | 0.0315 | 1.8410 | 70.12% | 54.10% | 0.5480 | `checkpoints/best_Transformer_T1.22_angle2_2d.pt` | Done |
| **T1.23** | **Transformer** | raw_3d | 39 | 0.1300 | 1.1634 | 72.76% | 57.21% | 0.5864 | `checkpoints/best_Transformer_T1.23_raw_3d.pt` | Done |
| **T1.24** | **Transformer** | rel_3d | 39 | 0.0436 | 1.5060 | 71.83% | 56.87% | 0.5772 | `checkpoints/best_Transformer_T1.24_rel_3d.pt` | Done |
| **T1.25** | **Transformer** | angle_3d | 286 | 0.0440 | 2.2274 | 64.70% | 43.39% | 0.4483 | `checkpoints/best_Transformer_T1.25_angle_3d.pt` | Done |
| **T1.26** | **Transformer** | angle2_3d | 78 | 0.0289 | 1.7650 | 71.20% | 55.40% | 0.5610 | `checkpoints/best_Transformer_T1.26_angle2_3d.pt` | Done |
| **T1.27** | **Transformer** | mix (rel_3d + angle2_3d) | 117 | 0.0118 | 1.7382 | 75.95% | 63.40% | 0.6218 | `checkpoints/best_Transformer_T1.27_mix.pt` | Done |

---

## Table 2: Data Augmentation Strategies on Best Sequence Model (Transformer mix)
*Objective:* Assess whether dynamic on-the-fly augmentation (SkelGym-Aug with bilateral symmetry mirroring, 3D yaw rotation, and temporal warping) outperforms the unaugmented clean baseline on the best sequence architecture (`Transformer mix`).

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline mix)** | `augment: none` (Clean original samples) | 0.0118 | 1.7382 | 75.95% | 63.40% | 0.6218 | `checkpoints/best_Transformer_T1.27_mix.pt` | Done |
| **T2.2** | **SkelGym-Aug (Dynamic mix)** | `augment: skel_gym_aug` (Bilateral Flip $p=0.5$, 3D Yaw $\pm 15^\circ$, Scale, Time-Warp, Jitter) | - | - | - | - | - | `checkpoints/best_Transformer_T2.2_mix.pt` | In Progress |

---

## Table 3: Spatial-Temporal Graph Models & Multi-Stream AAGCN Kinematics
*Objective:* Evaluate Graph Neural Networks under spatial-temporal skeletal graph topology $(B, C, T, V)$, systematically benchmarking baseline ST-GCN against dynamic Adaptive Graph Convolutionn Networks (AAGCN) and their multi-stream fusion ablations.

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

## Table 4: Data Augmentation Ablation on Graph Architectures (4-Stream Kinematics)
*Objective:* Empirical ablation assessing the effectiveness of dynamic skeletal data augmentation (SkelGym-Aug) across the graph streams of AAGCN, culminating in the augmented Four-Stream AAGCN late fusion.

| Exp ID | Model Architecture | Graph Stream | Augmentation Strategy | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **AAGCN** | bone_3d (Bone) | `augment: none` (Baseline) | 0.2273 | 1.5015 | 69.57% | 52.27% | 0.5338 | `checkpoints/best_AAGCN_T4.1_bone_3d.pt` | Done |
| **T4.2** | **AAGCN** | bone_3d (Bone) | `augment: skel_gym_aug` (Dynamic) | 0.3206 | 0.8213 | 77.98% | 65.84% | 0.6504 | `checkpoints/best_AAGCN_T4.2_bone_3d.pt` | Done |
| **T4.3** | **AAGCN** | rel_3d (Joint) | `augment: skel_gym_aug` (Dynamic) | - | - | - | - | - | `checkpoints/best_AAGCN_T4.3_rel_3d.pt` | In Progress |
| **T4.4** | **AAGCN** | joint_motion_3d (J-Motion) | `augment: skel_gym_aug` (Dynamic) | - | - | - | - | - | `checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt` | In Progress |
| **T4.5** | **AAGCN** | bone_motion_3d (B-Motion) | `augment: skel_gym_aug` (Dynamic) | - | - | - | - | - | `checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt` | In Progress |
| **T4.6** | **Two-Stream AAGCN (Aug)** | Joint (Aug) + Bone (Aug) | Late Fusion ($T=32$) | - | - | - | - | - | `outputs/ensemble/cm_ensemble_T4.6_weighted_soft.png` | In Progress |
| **T4.7** | **Four-Stream AAGCN (Aug)** | 4-Stream Fusion (SkelGym-Aug) | Late Fusion ($T=32$) | - | - | - | - | - | `outputs/ensemble/cm_ensemble_T4.7_weighted_soft.png` | In Progress |

---

## Table 5: Heterogeneous Cross-Paradigm Ensemble (Unified Weighted Soft Voting)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Table 2: `T2.2`) and spatial-temporal graph models (Table 4: `T4.2` and `T4.7`) using Dual-Target Weighted Soft Voting (SLSQP optimization).

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Grand 5-Stream SOTA Ensemble** | Best Transformer (mix aug) + 4-Stream AAGCN (Aug) | - | - | - | `outputs/ensemble/cm_ensemble_T5.1_weighted_soft.png` | In Progress |
| **T5.2** | **Dual-Model Grand Ensemble** | Best Transformer (mix aug) + AAGCN Bone (Aug) | - | - | - | `outputs/ensemble/cm_ensemble_T5.2_weighted_soft.png` | In Progress |

---

## Table 6A: Detailed Classification Report (Window-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble across all 22 gym exercise categories on the held-out test windows ($N=3,337$ windows at $T=32$, stride 32).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.3427 | 0.8841 | 0.4939 | 69 |
| bench press | 0.3987 | 0.6354 | 0.4900 | 96 |
| chest fly machine | 0.8587 | 0.9634 | 0.9080 | 82 |
| deadlift | 0.4135 | 0.6418 | 0.5029 | 67 |
| decline bench press | 0.4940 | 0.6194 | 0.5497 | 134 |
| hammer curl | 0.5556 | 0.3106 | 0.3984 | 161 |
| hip thrust | 0.8679 | 0.5847 | 0.6987 | 236 |
| incline bench press | 0.8913 | 0.5395 | 0.6721 | 76 |
| lat pulldown | 0.6443 | 0.9600 | 0.7711 | 100 |
| lateral raise | 0.8874 | 0.8645 | 0.8758 | 155 |
| leg extension | 0.9843 | 1.0000 | 0.9921 | 125 |
| leg raises | 0.8713 | 0.7719 | 0.8186 | 114 |
| plank | 0.6610 | 0.6964 | 0.6783 | 56 |
| pull Up | 0.7356 | 0.7805 | 0.7574 | 82 |
| push-up | 0.8416 | 0.9770 | 0.9043 | 87 |
| romanian deadlift | 0.7536 | 0.3562 | 0.4837 | 146 |
| russian twist | 0.8889 | 0.8759 | 0.8824 | 137 |
| shoulder press | 0.7600 | 0.5000 | 0.6032 | 152 |
| squat | 0.8305 | 0.8235 | 0.8270 | 238 |
| t bar row | 0.6216 | 0.5897 | 0.6053 | 117 |
| tricep Pushdown | 0.5820 | 0.7634 | 0.6605 | 93 |
| tricep dips | 0.9415 | 0.8773 | 0.9082 | 220 |
| **Accuracy** | | | **71.60%** | **2743** |
| **Macro avg** | **0.7194** | **0.7280** | **0.7037** | **2743** |
| **Weighted avg** | **0.7518** | **0.7160** | **0.7160** | **2743** |

---

## Table 6B: Detailed Classification Report (Video-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble aggregated at the full exercise clip level across all 236 independent held-out test videos ($N=236$ videos).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.6364 | 1.0000 | 0.7778 | 14 |
| bench press | 0.6667 | 0.5714 | 0.6154 | 14 |
| chest fly machine | 0.8889 | 1.0000 | 0.9412 | 8 |
| deadlift | 0.6667 | 0.8000 | 0.7273 | 10 |
| decline bench press | 0.5000 | 0.6250 | 0.5556 | 8 |
| hammer curl | 0.8333 | 0.3571 | 0.5000 | 14 |
| hip thrust | 0.8000 | 0.8889 | 0.8421 | 9 |
| incline bench press | 1.0000 | 0.4444 | 0.6154 | 9 |
| lat pulldown | 0.7222 | 1.0000 | 0.8387 | 13 |
| lateral raise | 1.0000 | 0.9333 | 0.9655 | 15 |
| leg extension | 1.0000 | 1.0000 | 1.0000 | 13 |
| leg raises | 1.0000 | 1.0000 | 1.0000 | 11 |
| plank | 0.6667 | 1.0000 | 0.8000 | 2 |
| pull Up | 0.7778 | 0.7000 | 0.7368 | 10 |
| push-up | 0.9231 | 1.0000 | 0.9600 | 12 |
| romanian deadlift | 1.0000 | 0.3333 | 0.5000 | 6 |
| russian twist | 0.8571 | 1.0000 | 0.9231 | 6 |
| shoulder press | 0.8333 | 0.7692 | 0.8000 | 13 |
| squat | 0.9333 | 0.9333 | 0.9333 | 15 |
| t bar row | 0.7778 | 0.7000 | 0.7368 | 10 |
| tricep Pushdown | 0.7857 | 0.9167 | 0.8462 | 12 |
| tricep dips | 1.0000 | 0.8889 | 0.9412 | 9 |
| **Accuracy** | | | **81.55%** | **233** |
| **Macro avg** | **0.8304** | **0.8119** | **0.7980** | **233** |
| **Weighted avg** | **0.8354** | **0.8155** | **0.8055** | **233** |

---

## Table 7: Video-Level Aggregation Summary Benchmark
*Objective:* Empirical comparison between temporal window-level predictions and complete clip/video-level predictions ($N=236$ complete test videos), analyzing the accuracy gains achievable through temporal consensus pooling.

| Model / Ensemble Architecture | Input Modality / Paradigm | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 | Video Gain (+Δ%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 117d)** | Sequential Recurrent Model (T1.9) | 57.67% | 0.5681 | 67.81% | 0.6618 | +10.14% |
| **Baseline BiLSTM (Mix 117d)** | Bidirectional Recurrent Model (T1.18) | 61.17% | 0.6008 | 68.24% | 0.6802 | +7.07% |
| **Transformer (Mix 117d)** | Self-Attention on Mix (T1.27) | 63.40% | 0.6218 | 74.25% | 0.7304 | +10.85% |
| **Baseline ST-GCN (Rel 3D)** | Static Graph Convolution (T3.2) | 43.57% | 0.4595 | 58.47% | 0.5817 | +14.90% |
| **Clean Baseline AAGCN (Bone 3D)** | Adaptive Skeletal Graph (T4.1) | 52.27% | 0.5338 | 69.53% | 0.6904 | +17.26% |
| **SkelGym-Aug AAGCN (Bone 3D)** | Adaptive Skeletal Graph + Aug (T4.2) | 65.84% | 0.6504 | 72.96% | 0.7295 | +7.12% |
| **Clean Baseline Transformer (Mix)** | Self-Attention Baseline (T2.1) | 63.40% | 0.6218 | 74.25% | 0.7304 | +10.85% |
| **SkelGym-Aug Transformer (Mix)** | Self-Attention + Aug (T2.2) | - | - | - | - | - |
| **Two-Stream AAGCN (Aug)** | Joint (Aug) + Bone (Aug) (T4.6) | - | - | - | - | - |
| **Four-Stream AAGCN (Aug)** | 4-Stream Late Fusion (Aug) (T4.7) | - | - | - | - | - |
| **Dual-Model Grand Ensemble** | Best Transformer (mix aug) + AAGCN Bone (Aug) (T5.2) | - | - | - | - | - |
| **Grand 5-Stream SOTA Ensemble** | **Dual-Target Weighted Soft Voting (T5.1)** | - | - | - | - | - |
