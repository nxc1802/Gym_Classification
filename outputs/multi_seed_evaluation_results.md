# SkelGym Multi-Seed Benchmark & 5 Fusion Methods Results

Evaluated across random seeds: `[42, 123, 3407]`.

## 1. 5 Fusion Methods Comparison across Multi-Stream Targets (Mean ± SD)

### Four-Stream AAGCN (Aug)

| Fusion Method | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 71.33% ± 0.18% | 0.7042 ± 0.0042 | 81.97% ± 1.29% | 0.8008 ± 0.0130 |
| **Hard Voting** | 66.97% ± 0.24% | 0.6676 ± 0.0027 | 79.26% ± 0.89% | 0.7831 ± 0.0156 |
| **SLSQP Soft Voting** | 67.61% ± 1.46% | 0.6688 ± 0.0205 | 77.54% ± 3.33% | 0.7632 ± 0.0387 |
| **Stacking Meta-Classifier** | 72.17% ± 0.39% | 0.7116 ± 0.0037 | 81.97% ± 1.72% | 0.8211 ± 0.0147 |
| **Uniform Soft Voting** | 71.79% ± 0.21% | 0.7086 ± 0.0073 | 82.12% ± 1.94% | 0.8029 ± 0.0191 |

### SkelGym-Full

| Fusion Method | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 72.56% ± 0.88% | 0.7183 ± 0.0062 | 81.97% ± 1.14% | 0.8063 ± 0.0210 |
| **Hard Voting** | 70.62% ± 0.54% | 0.7017 ± 0.0064 | 81.40% ± 0.99% | 0.8037 ± 0.0126 |
| **SLSQP Soft Voting** | 69.73% ± 1.10% | 0.6881 ± 0.0072 | 78.83% ± 0.66% | 0.7807 ± 0.0126 |
| **Stacking Meta-Classifier** | 73.53% ± 0.91% | 0.7250 ± 0.0063 | 83.69% ± 0.74% | 0.8354 ± 0.0089 |
| **Uniform Soft Voting** | 73.05% ± 0.75% | 0.7226 ± 0.0053 | 82.83% ± 1.14% | 0.8143 ± 0.0184 |

### SkelGym-Lite

| Fusion Method | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 69.69% ± 1.60% | 0.6874 ± 0.0111 | 77.54% ± 1.38% | 0.7649 ± 0.0072 |
| **Hard Voting** | 64.33% ± 0.38% | 0.6476 ± 0.0036 | 73.25% ± 0.50% | 0.7258 ± 0.0041 |
| **SLSQP Soft Voting** | 68.26% ± 0.66% | 0.6732 ± 0.0067 | 77.68% ± 1.55% | 0.7694 ± 0.0103 |
| **Stacking Meta-Classifier** | 69.39% ± 0.83% | 0.6805 ± 0.0049 | 78.68% ± 1.24% | 0.7791 ± 0.0110 |
| **Uniform Soft Voting** | 69.70% ± 1.52% | 0.6878 ± 0.0103 | 77.54% ± 1.38% | 0.7648 ± 0.0073 |

### Two-Stream AAGCN (Aug)

| Fusion Method | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy-Weighted Soft** | 68.26% ± 0.53% | 0.6739 ± 0.0120 | 78.25% ± 2.16% | 0.7626 ± 0.0252 |
| **Hard Voting** | 63.75% ± 0.29% | 0.6424 ± 0.0088 | 73.68% ± 0.99% | 0.7394 ± 0.0074 |
| **SLSQP Soft Voting** | 67.26% ± 1.41% | 0.6652 ± 0.0198 | 77.11% ± 3.33% | 0.7580 ± 0.0432 |
| **Stacking Meta-Classifier** | 68.50% ± 0.26% | 0.6744 ± 0.0103 | 78.11% ± 3.24% | 0.7727 ± 0.0463 |
| **Uniform Soft Voting** | 68.43% ± 0.35% | 0.6740 ± 0.0101 | 78.25% ± 2.16% | 0.7627 ± 0.0253 |

## 2. Individual Model Backbones (Mean ± SD)

| Model Name | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **AAGCN Bone (Aug)** | 66.59% ± 1.43% | 0.6587 ± 0.0181 | 75.82% ± 2.62% | 0.7543 ± 0.0305 |
| **AAGCN Bone Motion (Aug)** | 55.28% ± 0.69% | 0.5462 ± 0.0027 | 72.25% ± 2.36% | 0.6829 ± 0.0354 |
| **AAGCN Clean** | 59.50% ± 0.00% | 0.5960 ± 0.0000 | 69.53% ± 0.00% | 0.6904 ± 0.0000 |
| **AAGCN Joint Motion (Aug)** | 49.92% ± 0.23% | 0.4924 ± 0.0054 | 68.67% ± 1.55% | 0.6521 ± 0.0151 |
| **AAGCN Rel (Aug)** | 65.24% ± 1.19% | 0.6365 ± 0.0115 | 74.54% ± 1.73% | 0.7258 ± 0.0198 |
| **BiLSTM Baseline** | 61.25% ± 0.00% | 0.6015 ± 0.0000 | 68.67% ± 0.00% | 0.6830 ± 0.0000 |
| **LSTM Baseline** | 57.64% ± 0.00% | 0.5676 ± 0.0000 | 67.38% ± 0.00% | 0.6580 ± 0.0000 |
| **ST-GCN Baseline** | 54.76% ± 0.00% | 0.5240 ± 0.0000 | 63.09% ± 0.00% | 0.5993 ± 0.0000 |
| **Transformer Clean** | 63.40% ± 0.00% | 0.6218 ± 0.0000 | 74.25% ± 0.00% | 0.7304 ± 0.0000 |
| **Transformer Mix (Aug)** | 66.27% ± 2.75% | 0.6544 ± 0.0264 | 75.54% ± 3.81% | 0.7430 ± 0.0430 |
