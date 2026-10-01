# SkelGym Multi-Seed & 5 Fusion Methods Benchmark Results

Evaluated across random seeds: `[42, 123, 3407]`.

## 1. Individual Backbone Models (Baselines & Augmented Streams)

| Model / Architecture | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **AAGCN B-Motion (Aug)** | 3.93% ± 1.74% | 0.0034 ± 0.0015 | 5.72% ± 0.50% | 0.0049 ± 0.0004 |
| **AAGCN Bone (Aug)** | 3.93% ± 1.74% | 0.0034 ± 0.0015 | 5.72% ± 0.50% | 0.0049 ± 0.0004 |
| **AAGCN Clean (Bone 3D)** | 61.62% ± 2.72% | 0.5970 ± 0.0223 | 71.67% ± 3.51% | 0.6886 ± 0.0326 |
| **AAGCN J-Motion (Aug)** | 6.25% ± 1.01% | 0.0354 ± 0.0072 | 7.44% ± 1.51% | 0.0270 ± 0.0147 |
| **AAGCN Joint (Aug)** | 6.81% ± 2.21% | 0.0418 ± 0.0039 | 7.30% ± 1.55% | 0.0319 ± 0.0056 |
| **BiLSTM Baseline (Mix 117-d)** | 55.15% ± 2.55% | 0.5337 ± 0.0313 | 62.37% ± 4.59% | 0.5939 ± 0.0581 |
| **LSTM Baseline (Mix 117-d)** | 54.87% ± 2.82% | 0.5290 ± 0.0413 | 65.52% ± 3.25% | 0.6193 ± 0.0562 |
| **ST-GCN Baseline (Rel 3D)** | 53.97% ± 1.63% | 0.5145 ± 0.0151 | 62.23% ± 2.61% | 0.5812 ± 0.0292 |
| **Transformer Clean (Mix 117-d)** | 60.55% ± 4.34% | 0.5846 ± 0.0514 | 69.10% ± 5.68% | 0.6582 ± 0.0637 |
| **Transformer Mix (Aug)** | 31.13% ± 2.85% | 0.2878 ± 0.0355 | 33.33% ± 3.87% | 0.3191 ± 0.0478 |

## 2. Late Fusion Comparison (5 Methods across Ensemble Targets)

### Four-Stream AAGCN (Aug)

| Fusion Method | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 6.48% ± 2.67% | 0.0401 ± 0.0137 | 8.01% ± 1.51% | 0.0344 ± 0.0117 |
| **Hard Voting** | 3.99% ± 1.71% | 0.0060 ± 0.0044 | 6.01% ± 0.00% | 0.0052 ± 0.0000 |
| **SLSQP Soft Voting** | 6.25% ± 2.91% | 0.0354 ± 0.0111 | 7.73% ± 1.14% | 0.0277 ± 0.0043 |
| **Stacking Meta-Classifier** | 11.42% ± 2.19% | 0.0838 ± 0.0170 | 14.16% ± 3.35% | 0.1043 ± 0.0265 |
| **Uniform Soft Voting** | 6.45% ± 2.59% | 0.0419 ± 0.0110 | 7.87% ± 1.51% | 0.0370 ± 0.0103 |

### SkelGym-Full

| Fusion Method | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 26.75% ± 6.22% | 0.2494 ± 0.0670 | 29.61% ± 8.38% | 0.2759 ± 0.0779 |
| **Hard Voting** | 4.81% ± 1.95% | 0.0241 ± 0.0264 | 6.01% ± 0.00% | 0.0052 ± 0.0000 |
| **SLSQP Soft Voting** | 31.13% ± 2.92% | 0.2881 ± 0.0359 | 33.33% ± 4.12% | 0.3180 ± 0.0459 |
| **Stacking Meta-Classifier** | 29.89% ± 4.25% | 0.2622 ± 0.0447 | 36.77% ± 5.12% | 0.3165 ± 0.0575 |
| **Uniform Soft Voting** | 13.61% ± 6.22% | 0.1170 ± 0.0461 | 16.17% ± 6.03% | 0.1445 ± 0.0612 |

### SkelGym-Lite

| Fusion Method | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 31.11% ± 2.91% | 0.2867 ± 0.0362 | 33.62% ± 3.87% | 0.3209 ± 0.0500 |
| **Hard Voting** | 11.25% ± 9.75% | 0.0948 ± 0.1062 | 6.01% ± 0.00% | 0.0052 ± 0.0000 |
| **SLSQP Soft Voting** | 31.13% ± 2.94% | 0.2870 ± 0.0356 | 33.33% ± 3.87% | 0.3176 ± 0.0460 |
| **Stacking Meta-Classifier** | 29.21% ± 2.67% | 0.2434 ± 0.0282 | 32.19% ± 3.81% | 0.2647 ± 0.0410 |
| **Uniform Soft Voting** | 30.76% ± 2.85% | 0.2811 ± 0.0388 | 33.05% ± 4.72% | 0.3159 ± 0.0536 |

### Two-Stream AAGCN (Aug)

| Fusion Method | Window Acc (%) | Window Macro F1 | Video Acc (%) | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 6.77% ± 2.27% | 0.0417 ± 0.0037 | 7.15% ± 1.62% | 0.0305 ± 0.0054 |
| **Hard Voting** | 4.07% ± 1.36% | 0.0146 ± 0.0183 | 6.01% ± 0.00% | 0.0052 ± 0.0000 |
| **SLSQP Soft Voting** | 5.25% ± 3.55% | 0.0256 ± 0.0209 | 6.15% ± 0.89% | 0.0236 ± 0.0104 |
| **Stacking Meta-Classifier** | 9.07% ± 1.63% | 0.0580 ± 0.0124 | 8.15% ± 2.58% | 0.0553 ± 0.0211 |
| **Uniform Soft Voting** | 6.74% ± 2.34% | 0.0411 ± 0.0045 | 7.15% ± 1.62% | 0.0300 ± 0.0048 |
