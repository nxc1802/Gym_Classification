# Master Experiment Results: Deep Learning for Gym Exercise Classification

This document serves as the primary tracking log and benchmark sheet for the research paper. All experimental results are systematically categorized into structured tables corresponding directly to the paper's narrative and evaluation phases.

### Experimental Protocol & Standards (MediaPipe Pose Heavy SOTA Dataset)
- **Dataset Source of Truth:** 1,024 clean, un-shuffled video recordings across 22 classes from Kaggle (`nguyenxuancuongk18dn/gym-exercise-classification-dataset`).
  - **Train Set:** 580 videos (266,717 frames $\rightarrow$ 15,820 windows at $T=32$, stride 16; 25,852 windows at $T=20$, stride 10).
  - **Validation Set:** 208 videos (72,105 frames $\rightarrow$ 2,147 windows at $T=32$, stride 32; 3,513 windows at $T=20$, stride 20).
  - **Held-out Test Set:** 236 videos (110,577 frames $\rightarrow$ 3,337 windows at $T=32$, stride 32; 5,416 windows at $T=20$, stride 20).
- **Landmark Model:** MediaPipe Pose `model_complexity = 2` (Heavy) with 33 full-body landmarks (13 calibrated keypoints used for spatial body kinematics).
- **Controlled Parameter Budget:** All individual backbones calibrated to $\approx 350\text{K} \pm 15\%$ parameters for strict fairness.
- **Optimization:** Adam / AdamW optimizer, Cosine Annealing with Warmup or ReduceLROnPlateau, up to 100 epochs, early stopping patience = 20, Automatic Mixed Precision (AMP) enabled.

---

## Table 1: Temporal Models on Landmark Feature Sets (Controlled Budget $\approx 350\text{K}$)
*Objective:* Benchmark 3 temporal architectures (LSTM, BiLSTM, Transformer) across 7 coordinate & angular representations (2D/3D raw, relative, angles, and unified mix).

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T1.1** | **LSTM** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_LSTM_T1.1_raw_2d.pt` | Pending |
| **T1.2** | **LSTM** | rel_2d | 26 | - | - | - | - | - | `checkpoints/best_LSTM_T1.2_rel_2d.pt` | Pending |
| **T1.3** | **LSTM** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_LSTM_T1.3_angle_2d.pt` | Pending |
| **T1.4** | **LSTM** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_LSTM_T1.4_raw_3d.pt` | Pending |
| **T1.5** | **LSTM** | rel_3d | 39 | - | - | - | - | - | `checkpoints/best_LSTM_T1.5_rel_3d.pt` | Pending |
| **T1.6** | **LSTM** | angle_3d | 429 | - | - | - | - | - | `checkpoints/best_LSTM_T1.6_angle_3d.pt` | Pending |
| **T1.7** | **LSTM** | mix | 754 | - | - | - | - | - | `checkpoints/best_LSTM_T1.7_mix.pt` | Pending |
| **T1.8** | **BiLSTM** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.8_raw_2d.pt` | Pending |
| **T1.9** | **BiLSTM** | rel_2d | 26 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.9_rel_2d.pt` | Pending |
| **T1.10** | **BiLSTM** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.10_angle_2d.pt` | Pending |
| **T1.11** | **BiLSTM** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.11_raw_3d.pt` | Pending |
| **T1.12** | **BiLSTM** | rel_3d | 39 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.12_rel_3d.pt` | Pending |
| **T1.13** | **BiLSTM** | angle_3d | 429 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.13_angle_3d.pt` | Pending |
| **T1.14** | **BiLSTM** | mix | 754 | - | - | - | - | - | `checkpoints/best_BiLSTM_T1.14_mix.pt` | Pending |
| **T1.15** | **Transformer** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_Transformer_T1.15_raw_2d.pt` | Pending |
| **T1.16** | **Transformer** | rel_2d | 26 | - | - | - | - | - | `checkpoints/best_Transformer_T1.16_rel_2d.pt` | Pending |
| **T1.17** | **Transformer** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_Transformer_T1.17_angle_2d.pt` | Pending |
| **T1.18** | **Transformer** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_Transformer_T1.18_raw_3d.pt` | Pending |
| **T1.19** | **Transformer** | rel_3d | 39 | - | - | - | - | - | `checkpoints/best_Transformer_T1.19_rel_3d.pt` | Pending |
| **T1.20** | **Transformer** | angle_3d | 429 | - | - | - | - | - | `checkpoints/best_Transformer_T1.20_angle_3d.pt` | Pending |
| **T1.21** | **Transformer** | mix | 754 | - | - | - | - | - | `checkpoints/best_Transformer_T1.21_mix.pt` | Pending |

---

## Table 2: Data Augmentation Strategies on Best Transformer
*Objective:* Assess whether dynamic on-the-fly augmentation (SkelGym-Aug with bilateral symmetry mirroring, 3D yaw rotation, and temporal warping) outperforms the unaugmented clean baseline.

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline)** | `augment: none` (Clean original samples) | - | - | - | - | - | `checkpoints/best_Transformer_T2.1_mix.pt` | Pending |
| **T2.2** | **SkelGym-Aug (Dynamic On-the-Fly)** | `augment: skel_gym_aug` (Bilateral Flip $p=0.5$, 3D Yaw $\pm 15^\circ$, Scale, Time-Warp, Jitter) | - | - | - | - | - | `checkpoints/best_Transformer_T2.2_mix.pt` | Pending |

---

## Table 4: Spatial-Temporal Graph Models & Multi-Stream AAGCN Ablation
*Objective:* Evaluate Graph Neural Networks under spatial-temporal skeletal graph topology $(B, C, T, V)$, systematically benchmarking baseline ST-GCN against dynamic Adaptive Graph Convolutional Networks (AAGCN) and their multi-stream fusion ablations.

| Exp ID | Model Architecture | Graph Stream | Tensor Shape $(C, T, V)$ | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **ST-GCN** | raw_3d | $(3, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_T4.1_raw_3d.pt` | Pending |
| **T4.2** | **ST-GCN** | rel_3d | $(3, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_T4.2_rel_3d.pt` | Pending |
| **T4.3** | **ST-GCN** | raw_2d | $(2, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_T4.3_raw_2d.pt` | Pending |
| **T4.4** | **ST-GCN** | rel_2d | $(2, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_T4.4_rel_2d.pt` | Pending |
| **T4.5** | **AAGCN (Adaptive GCN)** | rel_3d (Joint Stream) | $(3, 20, 13)$ | - | - | - | - | - | `checkpoints/best_AAGCN_T4.5_rel_3d.pt` | Pending |
| **T4.6** | **AAGCN (Adaptive GCN)** | bone_3d (Bone Stream) | $(3, 20, 13)$ | - | - | - | - | - | `checkpoints/best_AAGCN_T4.6_bone_3d.pt` | Pending |
| **T4.7** | **AAGCN (Adaptive GCN)** | joint_motion_3d ($\Delta X$) | $(3, 20, 13)$ | - | - | - | - | - | `checkpoints/best_AAGCN_T4.7_joint_motion_3d.pt` | Pending |
| **T4.8** | **AAGCN (Adaptive GCN)** | bone_motion_3d ($\Delta B$) | $(3, 20, 13)$ | - | - | - | - | - | `checkpoints/best_AAGCN_T4.8_bone_motion_3d.pt` | Pending |
| **T4.9** | **Two-Stream AAGCN** | Joint + Bone Stream Fusion | Late Fusion | - | - | - | - | - | `outputs/ensemble/cm_2s_aagcn.png` | Pending |
| **T4.10** | **Four-Stream AAGCN** | Joint + Bone + J-Motion + B-Motion | Late Fusion | - | - | - | - | - | `outputs/ensemble/cm_4s_aagcn.png` | Pending |

---

## Table 5: Heterogeneous Cross-Paradigm Ensemble (Transformer + AAGCN)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Bảng 1/2) and spatial-temporal graph models (Bảng 4) to establish SOTA accuracy.

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Hard Voting** | Best Transformer + Best ST-GCN | - | - | - | `outputs/ensemble/cm_ensemble_hard.png` | Pending |
| **T5.2** | **Soft Voting** | Best Transformer + Best ST-GCN | - | - | - | `outputs/ensemble/cm_ensemble_soft.png` | Pending |
| **T5.3** | **Stacking Ensemble** | Best Transformer + Best ST-GCN + Meta-Learner | - | - | - | `outputs/ensemble/cm_ensemble_stacking.png` | Pending |
| **T5.4** | **Tri-Model Grand Ensemble** | Transformer Mix + AAGCN Joint + AAGCN Bone | - | - | - | `outputs/ensemble/cm_tri_model.png` | Pending |
| **T5.5** | **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | - | - | - | `outputs/ensemble/cm_grand_5s_sota.png` | Pending |

---

## Table 6A: Detailed Classification Report (Window-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble across all 22 gym exercise categories on the held-out test windows ($N=3,337$ at $T=32$, or $N=5,416$ at $T=20$).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | - | - | - | - |
| bench press | - | - | - | - |
| chest fly machine | - | - | - | - |
| deadlift | - | - | - | - |
| decline bench press | - | - | - | - |
| hammer curl | - | - | - | - |
| hip thrust | - | - | - | - |
| incline bench press | - | - | - | - |
| lat pulldown | - | - | - | - |
| lateral raise | - | - | - | - |
| leg extension | - | - | - | - |
| leg raises | - | - | - | - |
| plank | - | - | - | - |
| pull Up | - | - | - | - |
| push-up | - | - | - | - |
| romanian deadlift | - | - | - | - |
| russian twist | - | - | - | - |
| shoulder press | - | - | - | - |
| squat | - | - | - | - |
| t bar row | - | - | - | - |
| tricep Pushdown | - | - | - | - |
| tricep dips | - | - | - | - |
| **Accuracy** | | | **-** | **-** |
| **Macro avg** | **-** | **-** | **-** | **-** |
| **Weighted avg** | **-** | **-** | **-** | **-** |

---

## Table 6B: Detailed Classification Report (Video-Level Benchmark)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble aggregated at the full exercise clip level across all 236 independent held-out test videos ($N=236$ videos).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | - | - | - | - |
| bench press | - | - | - | - |
| chest fly machine | - | - | - | - |
| deadlift | - | - | - | - |
| decline bench press | - | - | - | - |
| hammer curl | - | - | - | - |
| hip thrust | - | - | - | - |
| incline bench press | - | - | - | - |
| lat pulldown | - | - | - | - |
| lateral raise | - | - | - | - |
| leg extension | - | - | - | - |
| leg raises | - | - | - | - |
| plank | - | - | - | - |
| pull Up | - | - | - | - |
| push-up | - | - | - | - |
| romanian deadlift | - | - | - | - |
| russian twist | - | - | - | - |
| shoulder press | - | - | - | - |
| squat | - | - | - | - |
| t bar row | - | - | - | - |
| tricep Pushdown | - | - | - | - |
| tricep dips | - | - | - | - |
| **Accuracy** | | | **-** | **-** |
| **Macro avg** | **-** | **-** | **-** | **-** |
| **Weighted avg** | **-** | **-** | **-** | **-** |

---

## Table 7: Video-Level Aggregation Summary Benchmark
*Objective:* Empirical comparison between temporal window-level predictions and complete clip/video-level predictions ($N=236$ complete test videos), analyzing the accuracy gains achievable through temporal consensus pooling.

| Model / Ensemble Architecture | Input Modality / Paradigm | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 | Video Gain (+Δ%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix)** | Sequential Recurrent Model | - | - | - | - | - |
| **Baseline BiLSTM (Mix)** | Bidirectional Recurrent Model | - | - | - | - | - |
| **Baseline ST-GCN (Rel 3D)** | Static Graph Convolution | - | - | - | - | - |
| **Best Transformer (Mix)** | Self-Attention on Kinematic Vectors | - | - | - | - | - |
| **Best Transformer + SkelGym-Aug** | Self-Attention with Dynamic Aug | - | - | - | - | - |
| **Two-Stream AAGCN** | Joint Stream + Bone Stream | - | - | - | - | - |
| **Four-Stream AAGCN** | Joint + Bone + Joint Motion + Bone Motion | - | - | - | - | - |
| **Tri-Model Grand Ensemble** | Transformer Mix + 2-Stream AAGCN | - | - | - | - | - |
| **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | - | - | - | - | - |
