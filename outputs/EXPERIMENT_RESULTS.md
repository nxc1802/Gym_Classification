# Master Experiment Results: Deep Learning for Gym Exercise Classification

This file aggregates all experimental evaluation results, tracking accuracy, Macro F1, and checkpoints.

| Timestamp | Model | Feature | Augment | Epochs | Batch | Train Loss | Val Loss | Test Acc (%) | Macro F1 | Best Checkpoint | HF Synced |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 2026-09-04 13:45:57 | Transformer | 12rel_4 | none | 2 | 4 | 3.0234 | 2.9414 | 0.00% | 0.0000 | `best_Transformer_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:49:04 | Transformer | 12rel_4 | none | 2 | 4 | 3.0508 | 2.9492 | 0.00% | 0.0000 | `best_Transformer_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:56:47 | Transformer | 12rel_4 | none | 50 | 128 | 3.0263 | 3.1304 | 3.67% | 0.0098 | `best_Transformer_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:57:12 | BiLSTM | 12rel_4 | none | 50 | 128 | 3.0801 | 3.1018 | 3.38% | 0.0047 | `best_BiLSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:57:34 | LSTM | 12rel_4 | none | 50 | 128 | 3.0853 | 3.0967 | 5.87% | 0.0084 | `best_LSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:59:43 | Transformer | 12rel_4 | rotate | 50 | 128 | 2.8635 | 3.3664 | 3.79% | 0.0243 | `best_Transformer_12rel_4_aug_rotate.pt` | Yes |
| 2026-09-04 07:00:40 | Transformer | branch_concat | none | 50 | 128 | 3.0719 | 3.0988 | 1.30% | 0.0042 | `best_Transformer_branch_concat_aug_none.pt` | Yes |
| 2026-09-04 07:01:20 | STGCN | full_4 | none | 50 | 128 | 2.8057 | 3.2348 | 3.02% | 0.0269 | `best_STGCN_full_4_aug_none.pt` | Yes |
| 2026-09-04 07:08:12 | Ensemble_STACKING | multi | none | - | - | - | - | 1.02% | 0.0009 | `ensemble_stacking` | Yes |
| 2026-09-04 07:09:19 | Ensemble_STACKING | multi | none | - | - | - | - | 1.02% | 0.0009 | `ensemble_stacking` | Yes |
| 2026-09-04 09:33:12 | Transformer | 12rel_4 | none | 100 | 16 | 0.8340 | 2.3472 | 33.97% | 0.3922 | `best_Transformer_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 09:35:01 | BiLSTM | 12rel_4 | none | 100 | 16 | 0.9352 | 2.1845 | 37.39% | 0.4491 | `best_BiLSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 09:36:10 | LSTM | 12rel_4 | none | 100 | 16 | 0.9353 | 2.2530 | 37.55% | 0.4454 | `best_LSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 09:38:41 | Transformer | 12rel_4 | rotate | 100 | 16 | 0.9100 | 2.2156 | 34.78% | 0.4164 | `best_Transformer_12rel_4_aug_rotate.pt` | Yes |
| 2026-09-04 09:41:31 | Transformer | branch_concat | none | 100 | 16 | 0.5546 | 2.4651 | 38.51% | 0.4809 | `best_Transformer_branch_concat_aug_none.pt` | Yes |
| 2026-09-04 09:44:08 | STGCN | full_4 | none | 100 | 16 | 1.6435 | 2.4650 | 28.47% | 0.3421 | `best_STGCN_full_4_aug_none.pt` | Yes |
| 2026-09-04 09:44:40 | Ensemble_STACKING | multi | none | - | - | - | - | 35.15% | 0.4468 | `ensemble_stacking` | Yes |
