# SkelGym-Aug New Formulation Downstream Benchmark Results

Evaluation across 3 independent random seeds (`42`, `123`, `3407`) using the new 4-operator SkelGym-Aug protocol (Bilateral Mirroring + 3D Yaw Rotation + Scaling + Jitter; No TimeWarp).

## Main Summary Table (Mean ± SD)

| Model Configuration | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer Mix (Aug)** | 66.25% ± 2.86% | 0.6548 ± 0.0279 | 75.68% ± 4.06% | 0.7455 ± 0.0472 |
| **AAGCN Bone (Aug)** | 66.68% ± 1.33% | 0.6597 ± 0.0170 | 76.39% ± 2.39% | 0.7596 ± 0.0283 |
| **AAGCN Joint (Aug)** | 65.27% ± 1.09% | 0.6372 ± 0.0106 | 74.68% ± 1.97% | 0.7268 ± 0.0212 |
| **AAGCN Joint-Motion (Aug)** | 49.90% ± 0.14% | 0.4923 ± 0.0044 | 68.53% ± 1.73% | 0.6513 ± 0.0159 |
| **AAGCN Bone-Motion (Aug)** | 55.33% ± 0.74% | 0.5462 ± 0.0024 | 72.39% ± 2.16% | 0.6847 ± 0.0322 |
| **Two-Stream AAGCN (Aug)** | 67.36% ± 1.50% | 0.6660 ± 0.0205 | 77.25% ± 3.09% | 0.7593 ± 0.0412 |
| **Four-Stream AAGCN (Aug)** | 67.69% ± 1.37% | 0.6690 ± 0.0195 | 77.54% ± 2.92% | 0.7636 ± 0.0346 |
| **SkelGym-Lite** | 68.26% ± 0.69% | 0.6733 ± 0.0072 | 77.83% ± 1.31% | 0.7708 ± 0.0080 |
| **SkelGym-Full** | 69.74% ± 1.04% | 0.6882 ± 0.0068 | 79.11% ± 0.25% | 0.7834 ± 0.0082 |

## Per-Seed Breakdown

### Transformer Mix (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 69.12% | 0.6838 | 80.26% | 0.7990 |
| 123 | 63.40% | 0.6281 | 72.53% | 0.7096 |
| 3407 | 66.24% | 0.6524 | 74.25% | 0.7279 |

### AAGCN Bone (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 66.02% | 0.6458 | 73.82% | 0.7290 |
| 123 | 68.21% | 0.6786 | 78.54% | 0.7847 |
| 3407 | 65.80% | 0.6546 | 76.82% | 0.7651 |

### AAGCN Joint (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 65.59% | 0.6339 | 72.53% | 0.7060 |
| 123 | 64.05% | 0.6287 | 75.11% | 0.7261 |
| 3407 | 66.17% | 0.6491 | 76.39% | 0.7484 |

### AAGCN Joint-Motion (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 50.05% | 0.4961 | 66.95% | 0.6405 |
| 123 | 49.84% | 0.4875 | 68.24% | 0.6440 |
| 3407 | 49.80% | 0.4934 | 70.39% | 0.6695 |

### AAGCN Bone-Motion (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 54.50% | 0.5490 | 72.10% | 0.7073 |
| 123 | 55.56% | 0.5450 | 74.68% | 0.6991 |
| 3407 | 55.92% | 0.5447 | 70.39% | 0.6478 |

### Two-Stream AAGCN (Aug)

| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 42 | 66.10% | 0.6445 | 73.82% | 0.7166 |
| 123 | 69.01% | 0.6854 | 78.11% | 0.7626 |
| 3407 | 66.97% | 0.6680 | 79.83% | 0.7988 |

### Four-Stream AAGCN (Aug)

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

