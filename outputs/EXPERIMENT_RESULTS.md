# Master Experiment Results: Deep Learning for Gym Exercise Classification

This file aggregates all experimental evaluation results, tracking accuracy, Macro F1, and checkpoints.

| Timestamp | Model | Feature | Augment | Epochs | Batch | Train Loss | Val Loss | Test Acc (%) | Macro F1 | Best Checkpoint | HF Synced |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 2026-09-04 06:56:47 | Transformer | 12rel_4 | none | 50 | 128 | 3.0263 | 3.1304 | 3.67% | 0.0098 | `best_Transformer_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:57:12 | BiLSTM | 12rel_4 | none | 50 | 128 | 3.0801 | 3.1018 | 3.38% | 0.0047 | `best_BiLSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:57:34 | LSTM | 12rel_4 | none | 50 | 128 | 3.0853 | 3.0967 | 5.87% | 0.0084 | `best_LSTM_12rel_4_aug_none.pt` | Yes |
| 2026-09-04 06:59:43 | Transformer | 12rel_4 | rotate | 50 | 128 | 2.8635 | 3.3664 | 3.79% | 0.0243 | `best_Transformer_12rel_4_aug_rotate.pt` | Yes |
| 2026-09-04 07:00:40 | Transformer | branch_concat | none | 50 | 128 | 3.0719 | 3.0988 | 1.30% | 0.0042 | `best_Transformer_branch_concat_aug_none.pt` | Yes |
| 2026-09-04 07:01:20 | STGCN | full_4 | none | 50 | 128 | 2.8057 | 3.2348 | 3.02% | 0.0269 | `best_STGCN_full_4_aug_none.pt` | Yes |
| 2026-09-04 07:09:19 | Ensemble_STACKING | multi | none | - | - | - | - | 1.02% | 0.0009 | `ensemble_stacking` | Yes |
