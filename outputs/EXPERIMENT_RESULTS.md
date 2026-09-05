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
| **T1.1** | **LSTM** | raw_2d | 26 | 0.3328 | 0.7408 | 82.80% | 55.24% | 0.5273 | `checkpoints/best_LSTM_T1.1_raw_2d.pt` | Done |
| **T1.2** | **LSTM** | rel_2d | 24 | 0.4362 | 0.7774 | 81.07% | 54.97% | 0.5235 | `checkpoints/best_LSTM_T1.2_rel_2d.pt` | Done |
| **T1.3** | **LSTM** | angle_2d | 286 | 0.1405 | 0.5446 | 87.86% | 58.86% | 0.5587 | `checkpoints/best_LSTM_T1.3_angle_2d.pt` | Done |
| **T1.4** | **LSTM** | raw_3d | 39 | 0.2000 | 0.7636 | 86.27% | 57.63% | 0.5432 | `checkpoints/best_LSTM_T1.4_raw_3d.pt` | Done |
| **T1.5** | **LSTM** | rel_3d | 36 | 0.2381 | 0.6846 | 86.56% | 59.67% | 0.5774 | `checkpoints/best_LSTM_T1.5_rel_3d.pt` | Done |
| **T1.6** | **LSTM** | angle_3d | 286 | 0.2205 | 0.5940 | 85.55% | 56.89% | 0.5327 | `checkpoints/best_LSTM_T1.6_angle_3d.pt` | Done |
| **T1.7** | **LSTM** | mix | 322 | 0.0206 | 0.7323 | 92.77% | 52.49% | 0.4759 | `checkpoints/best_LSTM_T1.7_mix.pt` | Done |
| **T1.8** | **BiLSTM** | raw_2d | 26 | 0.1960 | 0.6544 | 87.72% | 57.51% | 0.5584 | `checkpoints/best_BiLSTM_T1.8_raw_2d.pt` | Done |
| **T1.9** | **BiLSTM** | rel_2d | 24 | 0.3695 | 0.6475 | 81.94% | 58.17% | 0.5542 | `checkpoints/best_BiLSTM_T1.9_rel_2d.pt` | Done |
| **T1.10** | **BiLSTM** | angle_2d | 286 | 0.1315 | 0.5625 | 88.44% | 58.74% | 0.5583 | `checkpoints/best_BiLSTM_T1.10_angle_2d.pt` | Done |
| **T1.11** | **BiLSTM** | raw_3d | 39 | 0.1908 | 0.6540 | 87.28% | 58.62% | 0.5604 | `checkpoints/best_BiLSTM_T1.11_raw_3d.pt` | Done |
| **T1.12** | **BiLSTM** | rel_3d | 36 | 0.3674 | 0.7463 | 82.66% | 56.41% | 0.5404 | `checkpoints/best_BiLSTM_T1.12_rel_3d.pt` | Done |
| **T1.13** | **BiLSTM** | angle_3d | 286 | 0.2591 | 0.7354 | 84.97% | 57.54% | 0.5435 | `checkpoints/best_BiLSTM_T1.13_angle_3d.pt` | Done |
| **T1.14** | **BiLSTM** | mix | 322 | 0.0149 | 0.6321 | 91.91% | 54.10% | 0.5055 | `checkpoints/best_BiLSTM_T1.14_mix.pt` | Done |
| **T1.15** | **Transformer** | raw_2d | 26 | 0.1486 | 0.5698 | 89.31% | 60.72% | 0.5801 | `checkpoints/best_Transformer_T1.15_raw_2d.pt` | Done |
| **T1.16** | **Transformer** | rel_2d | 24 | 0.1813 | 0.6779 | 87.72% | 63.32% | 0.6069 | `checkpoints/best_Transformer_T1.16_rel_2d.pt` | Done |
| **T1.17** | **Transformer** | angle_2d | 286 | 0.0543 | 0.5028 | 90.46% | 62.19% | 0.5927 | `checkpoints/best_Transformer_T1.17_angle_2d.pt` | Done |
| **T1.18** | **Transformer** | raw_3d | 39 | 0.0665 | 0.5389 | 91.62% | 63.71% | 0.6167 | `checkpoints/best_Transformer_T1.18_raw_3d.pt` | Done |
| **T1.19** | **Transformer** | rel_3d | 36 | 0.0899 | 0.6060 | 90.32% | 60.96% | 0.5838 | `checkpoints/best_Transformer_T1.19_rel_3d.pt` | Done |
| **T1.20** | **Transformer** | angle_3d | 286 | 0.0421 | 0.6084 | 90.61% | 62.10% | 0.5952 | `checkpoints/best_Transformer_T1.20_angle_3d.pt` | Done |
| **T1.21** | **Transformer** | mix | 322 | 0.0085 | 0.4870 | 90.90% | 60.81% | 0.5720 | `checkpoints/best_Transformer_T1.21_mix.pt` | Done |

---

## Table 2: Data Augmentation on Best Transformer
*Objective:* Assess whether dataset expansion (combining 100% clean original samples + augmented supplementary samples via Scale, Rotate, Time-Warp, and Jitter) outperforms the unaugmented baseline.

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline)** | `augment: none` (Clean original samples: $N$) | 0.0085 | 0.4870 | 90.90% | 60.81% | 0.5720 | `checkpoints/best_Transformer_T2.1_mix.pt` | Done |
| **T2.2** | **Augmented Expansion (2x Samples)** | `augment: combined` (Original $N$ + Augmented $N$ via Scale, Rotate $\pm 10^\circ$, Time-Warp, Jitter) | 0.0050 | 0.6842 | 91.18% | 61.71% | 0.5710 | `checkpoints/best_Transformer_T2.2_mix.pt` | Done |

---

## Table 3: Feature Fusion (Lược bỏ / Replaced by Unified Mix Representation)
*Ghi chú:* Trong paper gốc, Bảng 3 khảo sát Branch-Concat vs Direct-Concat. Phương pháp `mix` (kết hợp chuẩn hóa z-score giữa tọa độ tương đối và góc tam giác 3D) đã được tích hợp trực tiếp vào Bảng 1 (T1.7, T1.14, T1.21), thay thế hoàn toàn cấu trúc đa nhánh cồng kềnh. Do đó Bảng 3 được lược bỏ theo đúng chỉ đạo.

---

## Table 4: ST-GCN on Raw vs. Relative Graph Streams (Controlled Budget $\approx 350\text{K}$)
*Objective:* Evaluate ST-GCN under spatial-temporal graph topology $(B, C, T, V)$ comparing raw coordinates and relative translation frames.

| Exp ID | Model Architecture | Graph Stream | Tensor Shape $(C, T, V)$ | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **ST-GCN** | raw_3d | $(3, 32, 13)$ | 0.2932 | 1.0418 | 75.87% | 51.05% | 0.4829 | `checkpoints/best_STGCN_T4.1_raw_3d.pt` | Done |
| **T4.2** | **ST-GCN** | rel_3d | $(3, 32, 12)$ | 0.4585 | 0.9909 | 74.13% | 51.80% | 0.4939 | `checkpoints/best_STGCN_T4.2_rel_3d.pt` | Done |
| **T4.3** | **ST-GCN** | raw_2d | $(2, 32, 13)$ | 0.5006 | 1.7527 | 59.25% | 36.74% | 0.3373 | `checkpoints/best_STGCN_T4.3_raw_2d.pt` | Done |
| **T4.4** | **ST-GCN** | rel_2d | $(2, 32, 12)$ | 0.5125 | 1.4366 | 68.64% | 43.35% | 0.4161 | `checkpoints/best_STGCN_T4.4_rel_2d.pt` | Done |

---

## Table 5: Heterogeneous Ensemble (Best Transformer + Best ST-GCN $\rightarrow$ SOTA)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Bảng 1/2) and the best skeletal graph ST-GCN (Bảng 4) to achieve SOTA accuracy.

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Hard Voting** | Best Transformer + Best ST-GCN | 58.83% | 0.5723 | 0.5746 | `outputs/ensemble/cm_ensemble_hard.png` | Done |
| **T5.2** | **Soft Voting** | Best Transformer + Best ST-GCN | 62.40% | 0.6020 | 0.6047 | `outputs/ensemble/cm_ensemble_soft.png` | Done |
| **T5.3** | **Stacking Ensemble** | Best Transformer + Best ST-GCN + Meta-Classifier | 53.62% | 0.4760 | 0.4820 | `outputs/ensemble/cm_ensemble_stacking.png` | Done |

---

## Table 6: Detailed Classification Report of Best Ensemble Method (Stacking)
*Objective:* Comprehensive per-class evaluation of the proposed SOTA ensemble across all 22 gym exercise categories.

| Exercise Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.3986 | 0.7626 | 0.5235 | 219 |
| bench press | 0.2700 | 0.8504 | 0.4099 | 127 |
| chest fly machine | 0.5590 | 0.6632 | 0.6066 | 193 |
| deadlift | 0.4182 | 0.9055 | 0.5721 | 127 |
| decline bench press | 0.0000 | 0.0000 | 0.0000 | 259 |
| hammer curl | 0.0000 | 0.0000 | 0.0000 | 187 |
| hip thrust | 0.5854 | 0.8108 | 0.6799 | 148 |
| incline bench press | 0.8921 | 0.7607 | 0.8212 | 163 |
| lat pulldown | 0.4517 | 0.8603 | 0.5924 | 136 |
| lateral raise | 0.8551 | 0.6782 | 0.7564 | 174 |
| leg extension | 0.8418 | 0.7340 | 0.7842 | 203 |
| leg raises | 0.5455 | 0.1983 | 0.2909 | 121 |
| plank | 1.0000 | 0.5522 | 0.7115 | 67 |
| pull Up | 0.0000 | 0.0000 | 0.0000 | 80 |
| push-up | 0.8544 | 1.0000 | 0.9215 | 135 |
| romanian deadlift | 0.4380 | 0.5048 | 0.4690 | 105 |
| russian twist | 0.9070 | 0.9689 | 0.9369 | 161 |
| shoulder press | 0.7949 | 0.3647 | 0.5000 | 170 |
| squat | 0.2865 | 0.5464 | 0.3759 | 194 |
| t bar row | 0.0000 | 0.0000 | 0.0000 | 128 |
| tricep Pushdown | 0.0000 | 0.0000 | 0.0000 | 85 |
| tricep dips | 0.6050 | 0.4557 | 0.5199 | 158 |
| **Accuracy** | | | **53.62%** | **3340** |
| **Macro avg** | **0.4865** | **0.5280** | **0.4760** | **3340** |
| **Weighted avg** | **0.4857** | **0.5362** | **0.4820** | **3340** |
