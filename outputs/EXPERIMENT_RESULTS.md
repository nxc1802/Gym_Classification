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
| **T1.1** | **LSTM** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_LSTM_raw_2d.pt` | Pending |
| **T1.2** | **LSTM** | rel_2d | 24 | - | - | - | - | - | `checkpoints/best_LSTM_rel_2d.pt` | Pending |
| **T1.3** | **LSTM** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_LSTM_angle_2d.pt` | Pending |
| **T1.4** | **LSTM** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_LSTM_raw_3d.pt` | Pending |
| **T1.5** | **LSTM** | rel_3d | 36 | - | - | - | - | - | `checkpoints/best_LSTM_rel_3d.pt` | Pending |
| **T1.6** | **LSTM** | angle_3d | 286 | - | - | - | - | - | `checkpoints/best_LSTM_angle_3d.pt` | Pending |
| **T1.7** | **LSTM** | mix | 322 | - | - | - | - | - | `checkpoints/best_LSTM_mix.pt` | Pending |
| **T1.8** | **BiLSTM** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_BiLSTM_raw_2d.pt` | Pending |
| **T1.9** | **BiLSTM** | rel_2d | 24 | - | - | - | - | - | `checkpoints/best_BiLSTM_rel_2d.pt` | Pending |
| **T1.10** | **BiLSTM** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_BiLSTM_angle_2d.pt` | Pending |
| **T1.11** | **BiLSTM** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_BiLSTM_raw_3d.pt` | Pending |
| **T1.12** | **BiLSTM** | rel_3d | 36 | - | - | - | - | - | `checkpoints/best_BiLSTM_rel_3d.pt` | Pending |
| **T1.13** | **BiLSTM** | angle_3d | 286 | - | - | - | - | - | `checkpoints/best_BiLSTM_angle_3d.pt` | Pending |
| **T1.14** | **BiLSTM** | mix | 322 | - | - | - | - | - | `checkpoints/best_BiLSTM_mix.pt` | Pending |
| **T1.15** | **Transformer** | raw_2d | 26 | - | - | - | - | - | `checkpoints/best_Transformer_raw_2d.pt` | Pending |
| **T1.16** | **Transformer** | rel_2d | 24 | - | - | - | - | - | `checkpoints/best_Transformer_rel_2d.pt` | Pending |
| **T1.17** | **Transformer** | angle_2d | 286 | - | - | - | - | - | `checkpoints/best_Transformer_angle_2d.pt` | Pending |
| **T1.18** | **Transformer** | raw_3d | 39 | - | - | - | - | - | `checkpoints/best_Transformer_raw_3d.pt` | Pending |
| **T1.19** | **Transformer** | rel_3d | 36 | - | - | - | - | - | `checkpoints/best_Transformer_rel_3d.pt` | Pending |
| **T1.20** | **Transformer** | angle_3d | 286 | - | - | - | - | - | `checkpoints/best_Transformer_angle_3d.pt` | Pending |
| **T1.21** | **Transformer** | mix | 322 | - | - | - | - | - | `checkpoints/best_Transformer_mix.pt` | Pending |

---

## Table 2: Data Augmentation on Best Transformer
*Objective:* Assess whether dataset expansion (combining 100% clean original samples + augmented supplementary samples via Scale, Rotate, Time-Warp, and Jitter) outperforms the unaugmented baseline.

| Exp ID | Augmentation Strategy | Configuration | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T2.1** | **None (Baseline)** | `augment: none` (Clean original samples: $N$) | - | - | - | - | - | `checkpoints/best_Transformer_mix.pt` | Pending |
| **T2.2** | **Augmented Expansion (2x Samples)** | `augment: combined` (Original $N$ + Augmented $N$ via Scale, Rotate $\pm 10^\circ$, Time-Warp, Jitter) | - | - | - | - | - | `checkpoints/best_Transformer_mix_aug.pt` | Pending |

---

## Table 3: Feature Fusion (Lược bỏ / Replaced by Unified Mix Representation)
*Ghi chú:* Trong paper gốc, Bảng 3 khảo sát Branch-Concat vs Direct-Concat. Phương pháp `mix` (kết hợp chuẩn hóa z-score giữa tọa độ tương đối và góc tam giác 3D) đã được tích hợp trực tiếp vào Bảng 1 (T1.7, T1.14, T1.21), thay thế hoàn toàn cấu trúc đa nhánh cồng kềnh. Do đó Bảng 3 được lược bỏ theo đúng chỉ đạo.

---

## Table 4: ST-GCN on Raw vs. Relative Graph Streams (Controlled Budget $\approx 350\text{K}$)
*Objective:* Evaluate ST-GCN under spatial-temporal graph topology $(B, C, T, V)$ comparing raw coordinates and relative translation frames.

| Exp ID | Model Architecture | Graph Stream | Tensor Shape $(C, T, V)$ | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T4.1** | **ST-GCN** | raw_3d | $(3, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_raw_3d.pt` | Pending |
| **T4.2** | **ST-GCN** | rel_3d | $(3, 32, 12)$ | - | - | - | - | - | `checkpoints/best_STGCN_rel_3d.pt` | Pending |
| **T4.3** | **ST-GCN** | raw_2d | $(2, 32, 13)$ | - | - | - | - | - | `checkpoints/best_STGCN_raw_2d.pt` | Pending |
| **T4.4** | **ST-GCN** | rel_2d | $(2, 32, 12)$ | - | - | - | - | - | `checkpoints/best_STGCN_rel_2d.pt` | Pending |

---

## Table 5: Heterogeneous Ensemble (Best Transformer + Best ST-GCN $\rightarrow$ SOTA)
*Objective:* Fuse complementary dynamics from the best sequence Transformer (Bảng 1/2) and the best skeletal graph ST-GCN (Bảng 4) to achieve SOTA accuracy.

| Exp ID | Ensemble Strategy | Component Models | Test Acc (%) | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **T5.1** | **Hard Voting** | Best Transformer + Best ST-GCN | - | - | - | `outputs/ensemble/cm_ensemble_hard.png` | Pending |
| **T5.2** | **Soft Voting** | Best Transformer + Best ST-GCN | - | - | - | `outputs/ensemble/cm_ensemble_soft.png` | Pending |
| **T5.3** | **Stacking Ensemble** | Best Transformer + Best ST-GCN + Meta-Classifier | - | - | - | `outputs/ensemble/cm_ensemble_stacking.png` | Pending |

---

## Table 6: Detailed Classification Report of Best Ensemble Method (Stacking)
*Objective:* Comprehensive per-class evaluation of the proposed SOTA ensemble across all 22 gym exercise categories.

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
| **Accuracy** | | | - | - |
| **Macro avg** | - | - | - | - |
| **Weighted avg** | - | - | - | - |
