# SkelGym Multi-Seed Evaluation Benchmark

Evaluation across 3 independent random seeds (`42`, `123`, `3407`) under strictly fixed splits and hyperparameter settings.

## Main Results Table (Mean ± SD)

| Model Architecture | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Mix + Aug** | 66.25% ± 2.86% | 0.6548 ± 0.0279 | 75.68% ± 4.06% | 0.7455 ± 0.0472 |
| **Four-Stream AAGCN** | 67.69% ± 1.37% | 0.6690 ± 0.0195 | 77.54% ± 2.92% | 0.7636 ± 0.0346 |
| **SkelGym-Lite** | 68.26% ± 0.69% | 0.6733 ± 0.0072 | 77.83% ± 1.31% | 0.7708 ± 0.0080 |
| **SkelGym-Full** | **69.74% ± 1.04%** | **0.6882 ± 0.0068** | **79.11% ± 0.25%** | **0.7834 ± 0.0082** |

## Per-Seed Detailed Breakdown

### Transformer Mix + Aug

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 69.12% | 0.6838 | 80.26% | 0.7990 |
| 123 | 63.40% | 0.6281 | 72.53% | 0.7096 |
| 3407 | 66.24% | 0.6524 | 74.25% | 0.7279 |

### Four-Stream AAGCN

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 66.53% | 0.6489 | 74.25% | 0.7282 |
| 123 | 69.19% | 0.6879 | 78.54% | 0.7655 |
| 3407 | 67.34% | 0.6702 | 79.83% | 0.7973 |

### SkelGym-Lite

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 68.10% | 0.6672 | 78.97% | 0.7776 |
| 123 | 69.01% | 0.6812 | 78.11% | 0.7728 |
| 3407 | 67.66% | 0.6716 | 76.39% | 0.7619 |

### SkelGym-Full

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 70.07% | 0.6872 | 78.97% | 0.7845 |
| 123 | 70.58% | 0.6954 | 78.97% | 0.7747 |
| 3407 | 68.57% | 0.6819 | 79.40% | 0.7909 |