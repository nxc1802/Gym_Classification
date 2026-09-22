# SkelGym Multi-Seed Evaluation Benchmark

Evaluation across 3 independent random seeds (`42`, `123`, `3407`) under strictly fixed splits and hyperparameter settings.

## Main Results Table (Mean ± SD)

| Model Architecture | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Mix + Aug** | 66.39% ± 1.94% | 0.6492 ± 0.0137 | 72.39% ± 2.59% | 0.7089 ± 0.0211 |
| **Four-Stream AAGCN** | 68.78% ± 1.35% | 0.6782 ± 0.0106 | 77.25% ± 1.55% | 0.7678 ± 0.0155 |
| **SkelGym-Lite** | 69.66% ± 0.71% | 0.6852 ± 0.0087 | 76.68% ± 0.25% | 0.7582 ± 0.0036 |
| **SkelGym-Full** | 70.03% ± 0.70% | 0.6877 ± 0.0051 | 77.68% ± 0.86% | 0.7689 ± 0.0090 |

## Per-Seed Detailed Breakdown

### Transformer Mix + Aug

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 68.61% | 0.6650 | 72.10% | 0.6983 |
| 123 | 65.04% | 0.6412 | 69.96% | 0.6952 |
| 3407 | 65.51% | 0.6415 | 75.11% | 0.7332 |

### Four-Stream AAGCN

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 67.34% | 0.6662 | 75.54% | 0.7500 |
| 123 | 69.01% | 0.6820 | 77.68% | 0.7752 |
| 3407 | 70.00% | 0.6863 | 78.54% | 0.7782 |

### SkelGym-Lite

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 69.27% | 0.6771 | 76.82% | 0.7550 |
| 123 | 69.23% | 0.6842 | 76.39% | 0.7575 |
| 3407 | 70.47% | 0.6944 | 76.82% | 0.7620 |

### SkelGym-Full

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 70.11% | 0.6858 | 77.68% | 0.7649 |
| 123 | 69.30% | 0.6839 | 76.82% | 0.7627 |
| 3407 | 70.69% | 0.6935 | 78.54% | 0.7792 |