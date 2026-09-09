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
| **2S_AAGCN_SL20** | **Two-Stream AAGCN** | Joint + Bone Fusion | Late Fusion | - | - | - | **60.40%** | **0.6349** | `outputs/ensemble/cm_ensemble_weighted_soft.png` | Done |

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

---

## Table 6: Detailed Classification Report of Proposed Grand SOTA Ensemble (Weighted Soft Voting)
*Objective:* Comprehensive per-class evaluation of the proposed Grand SOTA ensemble (Transformer Mix + AAGCN Joint + AAGCN Bone via SLSQP Weighted Soft Voting) across all 22 gym exercise categories on the clean leak-free test set ($N=9,364$).

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.5769 | 0.4982 | 0.5347 | 542 |
| bench press | 0.3046 | 0.6455 | 0.4139 | 268 |
| chest fly machine | 0.6972 | 0.6413 | 0.6681 | 499 |
| deadlift | 0.5151 | 0.7624 | 0.6148 | 202 |
| decline bench press | 0.5108 | 0.5534 | 0.5312 | 815 |
| hammer curl | 0.4257 | 0.2838 | 0.3405 | 444 |
| hip thrust | 0.9278 | 0.4247 | 0.5827 | 1634 |
| incline bench press | 0.7747 | 0.5948 | 0.6729 | 422 |
| lat pulldown | 0.5126 | 0.6803 | 0.5847 | 269 |
| lateral raise | 0.6309 | 0.7582 | 0.6887 | 426 |
| leg extension | 0.7320 | 0.8594 | 0.7906 | 569 |
| leg raises | 0.8740 | 0.8441 | 0.8588 | 263 |
| plank | 1.0000 | 0.6875 | 0.8148 | 176 |
| pull Up | 0.6182 | 0.9042 | 0.7343 | 240 |
| push-up | 0.9101 | 0.9918 | 0.9492 | 245 |
| romanian deadlift | 0.4798 | 0.6574 | 0.5547 | 289 |
| russian twist | 0.6576 | 0.9416 | 0.7744 | 257 |
| shoulder press | 0.5714 | 0.7239 | 0.6387 | 431 |
| squat | 0.7467 | 0.6232 | 0.6794 | 544 |
| t bar row | 0.5794 | 0.6636 | 0.6186 | 220 |
| tricep Pushdown | 0.6983 | 0.6983 | 0.6983 | 242 |
| tricep dips | 0.5518 | 0.7984 | 0.6526 | 367 |
| **Accuracy** | | | **63.31%** | **9364** |
| **Macro avg** | **0.6498** | **0.6925** | **0.6544** | **9364** |
| **Weighted avg** | **0.6786** | **0.6331** | **0.6307** | **9364** |

---

## Proposed SOTA: Upgraded Skeletal Transformer (Mix Representation)
*Objective:* Comprehensive empirical evaluation of the proposed SOTA architecture incorporating all Roadmap upgrades: Learnable Positional Embeddings, Pre-LN, GeLU, AdamW, Cosine Annealing with Warmup, Label Smoothing 0.1, exploring temporal sequence lengths ($SL=16$, $SL=20$, $SL=32$) with 50% overlap and BS=16, on the verified anti-leakage MediaPipe landmark dataset.

| Exp ID | Architecture | Feature | Seq Len | Stride | Augment | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **SOTA_TRANSFORMER_MIX** | **Transformer (Pre-LN + GeLU + Learnable PE)** | mix | 16 | 8 | combined (1->4) | 0.6206 | 2.4735 | 53.54% | 53.05% | 0.5442 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_mix.pt` | Done |
| **SOTA_TRANSFORMER_MIX_SL20** | **Transformer (Pre-LN + GeLU + Learnable PE)** | mix | 20 | 10 | combined (1->4) | 0.9008 | 2.1202 | 53.73% | 56.80% | 0.5648 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_SL20_mix.pt` | Done |
| **SOTA_TRANSFORMER_MIX_SL32** | **Transformer (Pre-LN + GeLU + Learnable PE)** | mix | 32 | 16 | combined (1->4) | 0.6421 | 2.2641 | 56.31% | 55.16% | 0.5690 | `checkpoints/best_Transformer_SOTA_TRANSFORMER_MIX_SL32_mix.pt` | Done |

---

## Proposed Ultimate SOTA: Two-Stream AAGCN + Skeletal Transformer Ensemble
*Objective:* Synergy between sequential self-attention (Skeletal Transformer with Mix representation) and spatial-temporal graph reasoning (2-Stream Adaptive Graph Convolutional Network on Joint and Bone streams).

| Method / Architecture | Input Stream / Modality | Parameters | Test Acc (%) | Macro F1 | Weighted F1 | Primary Benefit |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline ST-GCN** | Raw 3D Joint $(C=3, V=13)$ | 358K | 35.87% | 0.3471 | 0.3346 | Static physical topology baseline |
| **Upgraded Transformer** | Mix (Angles + Rel 3D + Velocity) | 352K | 56.80% | 0.5648 | 0.5512 | Long-range temporal kinematics |
| **AAGCN (Joint Stream)** | Rel 3D Joint $(C=3, V=13)$ | 378K | 58.87% | 0.6281 | 0.5837 | Dynamic inter-joint attention graph |
| **AAGCN (Bone Stream)** | Bone 3D Vector $(C=3, V=13)$ | 378K | 59.29% | 0.6181 | 0.5897 | Bone orientation & limb segment physics |
| **Two-Stream AAGCN** | Joint Stream + Bone Stream | 756K | **60.40%** | **0.6349** | **0.6027** | Orthogonal skeletal & limb dynamics |
| **Grand SOTA Ensemble** | Transformer Mix + 2s-AAGCN | 1.11M | **63.31%** | **0.6544** | **0.6307** | **Optimal multi-modal synergy (SLSQP)** |
| **Tri-Model Soft Voting** | Transformer Mix + 2s-AAGCN | 1.11M | **63.32%** | **0.6586** | **0.6308** | **Peak Macro F1 (+9.38% over Transformer)** |

