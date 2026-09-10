# Master Experiment Results: Deep Learning for Gym Exercise Classification

This document serves as the primary tracking log and benchmark sheet for the research paper. All experimental results are systematically categorized into structured tables corresponding directly to the paper's narrative and evaluation phases.

### Experimental Protocol & Standards
- **Dataset Support:** 1,108 segments across 22 classes (639 Train, 210 Validation, 259 Held-out Test).
- **Temporal Sliding Window:** Sequence length $T = 32$ frames. Train stride = 16 (50% overlap), Val/Test stride = 32 (non-overlapping).
- **Controlled Parameter Budget:** All backbones are calibrated to $\approx 350\text{K} \pm 15\%$ parameters for strict fairness.
- **Optimization:** Adam optimizer ($lr = 10^{-4}$), ReduceLROnPlateau scheduler, up to 100 epochs, early stopping patience = 20, AMP enabled, `use_class_weights = False`.

---

## Table 1: Temporal Models on Landmark Feature Sets (Controlled Budget $\approx 350\text{K}$)
*Objective:* Benchmark 3 temporal models (LSTM, BiLSTM, Transformer) across 7 coordinate & angular representations (2D/3D raw, relative, angles, and unified mix).

| Exp ID | Model Architecture | Feature Representation | Dimension | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T1.1** | **LSTM** | raw_2d | 26 | 0.3074 | 2.2601 | 47.56% | 47.90% | 0.4712 | `checkpoints/best_LSTM_T1.1_raw_2d.pt` | Done |
| **T1.2** | **LSTM** | rel_2d | 26 | 0.4024 | 2.3667 | 50.12% | 45.70% | 0.4622 | `checkpoints/best_LSTM_T1.2_rel_2d.pt` | Done |
| **T1.3** | **LSTM** | angle_2d | 286 | 0.1309 | 3.0266 | 48.04% | 43.80% | 0.4342 | `checkpoints/best_LSTM_T1.3_angle_2d.pt` | Done |
| **T1.4** | **LSTM** | raw_3d | 39 | 0.1985 | 2.6284 | 49.13% | 48.27% | 0.4754 | `checkpoints/best_LSTM_T1.4_raw_3d.pt` | Done |
| **T1.5** | **LSTM** | rel_3d | 39 | 0.2856 | 2.5757 | 52.85% | 48.85% | 0.4814 | `checkpoints/best_LSTM_T1.5_rel_3d.pt` | Done |
| **T1.6** | **LSTM** | angle_3d | 286 | 0.1232 | 3.4244 | 45.20% | 44.17% | 0.4485 | `checkpoints/best_LSTM_T1.6_angle_3d.pt` | Done |
| **T1.7** | **LSTM** | mix | 325 | 0.0559 | 3.5758 | 49.33% | 54.91% | 0.5439 | `checkpoints/best_LSTM_T1.7_mix.pt` | Done |
| **T1.8** | **BiLSTM** | raw_2d | 26 | 0.2915 | 2.0462 | 54.56% | 44.78% | 0.4474 | `checkpoints/best_BiLSTM_T1.8_raw_2d.pt` | Done |
| **T1.9** | **BiLSTM** | rel_2d | 26 | 0.3273 | 2.3025 | 50.26% | 50.20% | 0.4902 | `checkpoints/best_BiLSTM_T1.9_rel_2d.pt` | Done |
| **T1.10** | **BiLSTM** | angle_2d | 286 | 0.1235 | 2.9666 | 51.52% | 47.49% | 0.4535 | `checkpoints/best_BiLSTM_T1.10_angle_2d.pt` | Done |
| **T1.11** | **BiLSTM** | raw_3d | 39 | 0.2631 | 2.7134 | 47.32% | 44.82% | 0.4326 | `checkpoints/best_BiLSTM_T1.11_raw_3d.pt` | Done |
| **T1.12** | **BiLSTM** | rel_3d | 39 | 0.3466 | 2.4109 | 49.40% | 44.44% | 0.4028 | `checkpoints/best_BiLSTM_T1.12_rel_3d.pt` | Done |
| **T1.13** | **BiLSTM** | angle_3d | 286 | 0.0984 | 3.2998 | 48.96% | 46.04% | 0.4730 | `checkpoints/best_BiLSTM_T1.13_angle_3d.pt` | Done |
| **T1.14** | **BiLSTM** | mix | 325 | 0.0654 | 3.3699 | 50.70% | 52.95% | 0.5282 | `checkpoints/best_BiLSTM_T1.14_mix.pt` | Done |
| **T1.15** | **Transformer** | raw_2d | 26 | 0.1623 | 2.5206 | 55.48% | 52.91% | 0.5451 | `checkpoints/best_Transformer_T1.15_raw_2d.pt` | Done |
| **T1.16** | **Transformer** | rel_2d | 26 | 0.1928 | 2.3190 | 58.87% | 54.51% | 0.5360 | `checkpoints/best_Transformer_T1.16_rel_2d.pt` | Done |
| **T1.17** | **Transformer** | angle_2d | 286 | 0.1049 | 2.6590 | 56.58% | 49.42% | 0.4788 | `checkpoints/best_Transformer_T1.17_angle_2d.pt` | Done |
| **T1.18** | **Transformer** | raw_3d | 39 | 0.1174 | 2.3503 | 55.59% | 51.90% | 0.5404 | `checkpoints/best_Transformer_T1.18_raw_3d.pt` | Done |
| **T1.19** | **Transformer** | rel_3d | 39 | 0.1802 | 2.3504 | 55.55% | 52.81% | 0.5213 | `checkpoints/best_Transformer_T1.19_rel_3d.pt` | Done |
| **T1.20** | **Transformer** | angle_3d | 286 | 0.1003 | 3.2583 | 49.95% | 51.12% | 0.5245 | `checkpoints/best_Transformer_T1.20_angle_3d.pt` | Done |
| **T1.21** | **Transformer** | mix | 325 | 0.0843 | 2.7207 | 51.86% | 53.49% | 0.5411 | `checkpoints/best_Transformer_T1.21_mix.pt` | Done |

---

## Table 2: Data Augmentation on Best Transformer
*Objective:* Assess whether dataset expansion (combining 100% clean original samples + augmented supplementary samples via Scale, Rotate, Time-Warp, and Jitter) outperforms the unaugmented baseline.

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline)** | `augment: none` (Clean original samples: $N$) | 0.0843 | 2.7207 | 51.86% | 53.49% | 0.5411 | `checkpoints/best_Transformer_T2.1_mix.pt` | Done |
| **T2.2** | **Augmented Expansion (2x Samples)** | `augment: combined` (Original $N$ + Augmented $N$ via Scale, Rotate $\pm 10^\circ$, Time-Warp, Jitter) | 0.0054 | 4.2401 | 54.36% | 52.34% | 0.5480 | `checkpoints/best_Transformer_T2.2_mix.pt` | Done |

---

## Table 3: Feature Fusion (Lược bỏ / Replaced by Unified Mix Representation)
*Ghi chú:* Trong paper gốc, Bảng 3 khảo sát Branch-Concat vs Direct-Concat. Phương pháp `mix` (kết hợp chuẩn hóa z-score giữa tọa độ tương đối và góc tam giác 3D) đã được tích hợp trực tiếp vào Bảng 1 (T1.7, T1.14, T1.21), thay thế hoàn toàn cấu trúc đa nhánh cồng kềnh. Do đó Bảng 3 được lược bỏ theo đúng chỉ đạo.

---

## Table 4: ST-GCN on Raw vs. Relative Graph Streams (Controlled Budget $\approx 350\text{K}$)
*Objective:* Evaluate ST-GCN under spatial-temporal graph topology $(B, C, T, V)$ comparing raw coordinates and relative translation frames.

| Exp ID | Model Architecture | Graph Stream | Tensor Shape $(C, T, V)$ | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **ST-GCN** | raw_3d | $(3, 32, 13)$ | 0.6066 | 2.5094 | 37.99% | 35.87% | 0.3471 | `checkpoints/best_STGCN_T4.1_raw_3d.pt` | Done |
| **T4.2** | **ST-GCN** | rel_3d | $(3, 32, 13)$ | 0.5516 | 2.2020 | 41.99% | 41.94% | 0.4128 | `checkpoints/best_STGCN_T4.2_rel_3d.pt` | Done |
| **T4.3** | **ST-GCN** | raw_2d | $(2, 32, 13)$ | 0.7527 | 2.9184 | 34.40% | 39.63% | 0.3346 | `checkpoints/best_STGCN_T4.3_raw_2d.pt` | Done |
| **T4.4** | **ST-GCN** | rel_2d | $(2, 32, 13)$ | 0.7432 | 2.5220 | 35.19% | 35.03% | 0.3397 | `checkpoints/best_STGCN_T4.4_rel_2d.pt` | Done |
| **AAGCN_JOINT_SL20** | **AAGCN (Adaptive GCN)** | rel_3d (Joint) | $(3, 20, 13)$ | 0.6341 | 2.0259 | 60.79% | 58.87% | 0.6281 | `checkpoints/best_AAGCN_AAGCN_JOINT_SL20_rel_3d.pt` | Done |
| **AAGCN_BONE_SL20** | **AAGCN (Adaptive GCN)** | bone_3d (Bone) | $(3, 20, 13)$ | 0.6284 | 1.9725 | 62.00% | 59.29% | 0.6181 | `checkpoints/best_AAGCN_AAGCN_BONE_SL20_bone_3d.pt` | Done |
| **AAGCN_J_MOTION_SL20** | **AAGCN (Adaptive GCN)** | joint_motion_3d ($\Delta X$) | $(3, 20, 13)$ | 0.8124 | 2.1805 | 49.70% | 41.96% | 0.4489 | `checkpoints/best_AAGCN_AAGCN_J_MOTION_SL20_joint_motion_3d.pt` | Done |
| **AAGCN_B_MOTION_SL20** | **AAGCN (Adaptive GCN)** | bone_motion_3d ($\Delta B$) | $(3, 20, 13)$ | 0.8032 | 2.1246 | 51.04% | 44.57% | 0.4771 | `checkpoints/best_AAGCN_AAGCN_B_MOTION_SL20_bone_motion_3d.pt` | Done |
| **2S_AAGCN_SL20** | **Two-Stream AAGCN** | Joint + Bone Fusion | Late Fusion | - | - | - | **60.40%** | **0.6349** | `outputs/ensemble/cm_ensemble_weighted_soft.png` | Done |
| **4S_AAGCN_SL20** | **Four-Stream AAGCN** | Joint + Bone + J-Motion + B-Motion | Late Fusion | - | - | - | **60.88%** | **0.6414** | `outputs/ensemble/cm_ensemble_4S_AAGCN_SL20_weighted_soft.png` | Done |
| **4S_AAGCN_SL20_TTA** | **Four-Stream AAGCN + TTA** | 4-Stream with Bilateral Mirroring | Late Fusion | - | - | - | **61.50%** | **0.6500** | `outputs/ensemble/cm_ensemble_4S_AAGCN_SL20_TTA_weighted_soft.png` | Done |

---

## Table 5: Heterogeneous Ensemble (Best Transformer + Best ST-GCN $\rightarrow$ SOTA)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Bảng 1/2) and the best skeletal graph ST-GCN (Bảng 4) to achieve SOTA accuracy.

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Hard Voting** | Best Transformer + Best ST-GCN | 49.73% | 0.5103 | 0.5031 | `outputs/ensemble/cm_ensemble_hard.png` | Done |
| **T5.2** | **Soft Voting** | Best Transformer + Best ST-GCN | 54.30% | 0.5343 | 0.5422 | `outputs/ensemble/cm_ensemble_soft.png` | Done |
| **T5.3** | **Stacking Ensemble** | Best Transformer + Best ST-GCN + Meta-Classifier | 53.05% | 0.5224 | 0.5420 | `outputs/ensemble/cm_ensemble_stacking.png` | Done |
| **T5.4** | **Tri-Model Soft Voting** | Transformer Mix + AAGCN Joint + AAGCN Bone (Equal Weights) | 63.32% | 0.6586 | 0.6308 | `outputs/ensemble/cm_ensemble_soft.png` | Done |
| **SOTA_GRAND_ENSEMBLE** | **Grand Weighted Soft Voting** | Transformer Mix (0.36) + AAGCN Joint (0.24) + AAGCN Bone (0.40) via SLSQP | **63.31%** | **0.6544** | **0.6307** | `outputs/ensemble/cm_ensemble_SOTA_GRAND_ENSEMBLE_weighted_soft.png` | Done |
| **4S_AAGCN_SL20** | **Four-Stream Weighted Soft Voting** | Joint (0.23) + Bone (0.41) + J-Motion (0.21) + B-Motion (0.15) via SLSQP | 60.88% | 0.6414 | 0.6080 | `outputs/ensemble/cm_ensemble_4S_AAGCN_SL20_weighted_soft.png` | Done |
| **4S_AAGCN_SL20_TTA** | **Four-Stream AAGCN + TTA** | 4-Stream Late Fusion with Bilateral Mirroring TTA | 61.50% | 0.6500 | 0.6135 | `outputs/ensemble/cm_ensemble_4S_AAGCN_SL20_TTA_weighted_soft.png` | Done |
| **GRAND_5S_SOTA_TTA** | **Grand 5-Stream SOTA + TTA** | Transformer (0.25) + Joint (0.19) + Bone (0.35) + JM (0.09) + BM (0.12) | **62.84%** | **0.6588** | **0.6245** | `outputs/ensemble/cm_ensemble_GRAND_5S_SOTA_TTA_weighted_soft.png` | Done |

---

## Table 6: Detailed Classification Report of Proposed Grand SOTA Ensemble (Weighted Soft Voting)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble across all 22 gym exercise categories on the clean leak-free test set ($N=9,364$).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.5947 | 0.4982 | 0.5422 | 542 |
| bench press | 0.2997 | 0.6418 | 0.4086 | 268 |
| chest fly machine | 0.7310 | 0.6754 | 0.7021 | 499 |
| deadlift | 0.5016 | 0.7574 | 0.6036 | 202 |
| decline bench press | 0.4579 | 0.4871 | 0.4721 | 815 |
| hammer curl | 0.4520 | 0.3288 | 0.3807 | 444 |
| hip thrust | 0.9462 | 0.3770 | 0.5392 | 1634 |
| incline bench press | 0.7538 | 0.5948 | 0.6649 | 422 |
| lat pulldown | 0.5436 | 0.6952 | 0.6101 | 269 |
| lateral raise | 0.6548 | 0.7746 | 0.7097 | 426 |
| leg extension | 0.7078 | 0.8559 | 0.7749 | 569 |
| leg raises | 0.8992 | 0.8479 | 0.8728 | 263 |
| plank | 1.0000 | 0.6989 | 0.8227 | 176 |
| pull Up | 0.6056 | 0.8958 | 0.7227 | 240 |
| push-up | 0.9101 | 0.9918 | 0.9492 | 245 |
| romanian deadlift | 0.4518 | 0.6817 | 0.5434 | 289 |
| russian twist | 0.6471 | 0.9416 | 0.7670 | 257 |
| shoulder press | 0.5599 | 0.7378 | 0.6366 | 431 |
| squat | 0.7429 | 0.6213 | 0.6767 | 544 |
| t bar row | 0.6565 | 0.7818 | 0.7137 | 220 |
| tricep Pushdown | 0.7706 | 0.6942 | 0.7304 | 242 |
| tricep dips | 0.5397 | 0.8147 | 0.6493 | 367 |
| **Accuracy** | | | **62.84%** | **9364** |
| **Macro avg** | **0.6558** | **0.6997** | **0.6588** | **9364** |
| **Weighted avg** | **0.6821** | **0.6284** | **0.6245** | **9364** |

---

## Table 7: Video-Level Aggregation & Test-Time Augmentation (TTA) Benchmark
*Objective:* Empirical comparison between frame/window-level predictions ($N=9,364$ temporal windows) and clip/video-level predictions ($N=214$ complete exercise videos), investigating the synergistic impact of bilateral horizontal mirroring Test-Time Augmentation (TTA) and temporal soft average pooling.

| Model / Ensemble System | Input Modality / Architecture | Window Acc (%) | Window Macro F1 | Window Acc (TTA) | Window F1 (TTA) | Video Acc (Standard) | Video Macro F1 | Video Acc (TTA) | Video Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SOTA Transformer (SL=20, CE)** | Mix Representation (Angles + Rel 3D + Vel) | 56.80% | 0.5648 | 59.03% | 0.5997 | 71.96% | 0.7291 | **73.36%** | **0.7452** |
| **SOTA Transformer (SL=20, Focal)** | Mix Representation (Focal Loss $\gamma=2.0$) | 56.23% | 0.5741 | 55.33% | 0.5882 | 72.43% | 0.7380 | **74.77%** | **0.7555** |
| **Tri-Model Grand Ensemble** | Transformer Mix + AAGCN Joint + Bone | 63.31% | 0.6544 | 63.32% | 0.6586 | 79.44% | 0.7994 | **79.91%** | **0.8055** |
| **Four-Stream AAGCN** | Joint + Bone + Joint Motion + Bone Motion | 60.88% | 0.6414 | 61.50% | 0.6500 | 80.37% | 0.8137 | **80.84%** | **0.8198** |
| **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | 62.84% | 0.6588 | 62.84% | 0.6588 | 80.37% | 0.8142 | **80.84%** | **0.8166** |

---

## Proposed SOTA: Upgraded Skeletal Transformer (Mix Representation & Loss Ablation)
*Objective:* Comprehensive empirical evaluation of the proposed SOTA architecture incorporating all Roadmap upgrades: Learnable Positional Embeddings, Pre-LN, GeLU, AdamW, Cosine Annealing with Warmup, exploring sequence lengths ($SL=16$, $SL=20$, $SL=32$) and Loss function alternatives (Label Smoothing vs. Focal Loss $\gamma=2.0$).

| Exp ID | Architecture | Feature | Seq Len | Stride | Augment | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **SOTA_TRANSFORMER_MIX** | **Transformer (Pre-LN + GeLU + Learnable PE)** | mix | 16 | 8 | combined (1->4) | 0.6206 | 2.4735 | 53.54% | 53.05% | 0.5442 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_mix.pt` | Done |
| **SOTA_TRANSFORMER_MIX_SL20** | **Transformer (Pre-LN + GeLU + Label Smoothing 0.1)** | mix | 20 | 10 | combined (1->4) | 0.9008 | 2.1202 | 53.73% | 56.80% | 0.5648 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_SL20_mix.pt` | Done |
| **SOTA_TRANSFORMER_MIX_SL20_FOCAL** | **Transformer (Pre-LN + GeLU + Focal Loss $\gamma=2.0$)** | mix | 20 | 10 | combined (1->4) | 0.5841 | 1.8329 | 52.48% | 56.23% | 0.5741 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_SL20_FOCAL_mix.pt` | Done |
| **SOTA_TRANSFORMER_MIX_SL32** | **Transformer (Pre-LN + GeLU + Learnable PE)** | mix | 32 | 16 | combined (1->4) | 0.6421 | 2.2641 | 56.31% | 55.16% | 0.5690 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_SL32_mix.pt` | Done |

---

## Proposed Ultimate SOTA: Multi-Stream AAGCN + Skeletal Transformer Ensemble
*Objective:* Synergy between sequential self-attention (Skeletal Transformer with Mix representation) and spatial-temporal graph reasoning (4-Stream Adaptive Graph Convolutional Network on Joint, Bone, and Motion streams).

| Method / Architecture | Input Stream / Modality | Parameters | Window Test Acc (%) | Window Macro F1 | Video Acc (TTA) | Video Macro F1 | Primary Benefit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline ST-GCN** | Raw 3D Joint $(C=3, V=13)$ | 358K | 35.87% | 0.3471 | 48.13% | 0.4612 | Static physical topology baseline |
| **Upgraded Transformer** | Mix (Angles + Rel 3D + Velocity) | 352K | 56.80% | 0.5648 | 73.36% | 0.7452 | Long-range temporal kinematics |
| **Upgraded Transformer (Focal)** | Mix + Focal Loss $\gamma=2.0$ | 352K | 56.23% | 0.5741 | 74.77% | 0.7555 | Hard class discrimination (`hammer curl` F1 +5.7%) |
| **AAGCN (Joint Stream)** | Rel 3D Joint $(C=3, V=13)$ | 378K | 58.87% | 0.6281 | 75.70% | 0.7688 | Dynamic inter-joint attention graph |
| **AAGCN (Bone Stream)** | Bone 3D Vector $(C=3, V=13)$ | 378K | 59.29% | 0.6181 | 76.64% | 0.7715 | Bone orientation & limb segment physics |
| **AAGCN (Joint Motion Stream)** | Velocity Vector $\Delta X$ $(C=3, V=13)$ | 378K | 41.96% | 0.4489 | 59.35% | 0.6120 | First-order joint motion kinematics |
| **AAGCN (Bone Motion Stream)** | Bone Velocity $\Delta B$ $(C=3, V=13)$ | 378K | 44.57% | 0.4771 | 62.15% | 0.6384 | Limb segment angular velocity dynamics |
| **Two-Stream AAGCN** | Joint Stream + Bone Stream | 756K | **60.40%** | **0.6349** | 78.50% | 0.7891 | Orthogonal skeletal & limb dynamics |
| **Four-Stream AAGCN** | Joint + Bone + J-Motion + B-Motion | 1.51M | **60.88%** | **0.6414** | **80.37%** | **0.8137** | Complete spatial-temporal 4-stream topology |
| **Four-Stream AAGCN + TTA** | 4-Stream AAGCN + Bilateral Mirroring | 1.51M | **61.50%** | **0.6500** | **80.84%** | **0.8198** | Invariant geometric TTA + Video pooling |
| **Tri-Model Grand Ensemble** | Transformer Mix + 2s-AAGCN | 1.11M | **63.31%** | **0.6544** | **79.91%** | **0.8055** | Optimal multi-modal synergy (SLSQP) |
| **Grand 5-Stream SOTA Ensemble** | Transformer Mix + 4-Stream AAGCN | 1.86M | **62.84%** | **0.6588** | **80.84%** | **0.8166** | Peak Macro F1 across all modalities |

