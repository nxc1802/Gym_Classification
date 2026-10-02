# Statistical Significance & Hypothesis Testing Report

Evaluated on $N=2743$ temporal sliding windows and $K=233$ independent action videos.

## 1. Window-Level McNemar Tests with Multiple Testing Correction

| Comparison | Model 1 Acc | Model 2 Acc | Delta | Discordant (b/c) | McNemar p (Raw) | Holm-Bonferroni p | FDR p (B-H) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Transformer vs Aug Transformer | 61.61% | 71.53% | +9.92% | 117/389 | 3.89e-35 | 1.94e-34 | 1.94e-34 | ******* |
| ST-GCN vs Four-Stream AAGCN | 58.84% | 67.34% | +8.49% | 199/432 | 9.28e-21 | 2.78e-20 | 1.55e-20 | ******* |
| Transformer Mix (Aug) vs SkelGym-Full | 71.53% | 71.05% | -0.47% | 134/121 | 0.4524 | 0.4524 | 0.4524 | **n.s.** |
| AAGCN Bone (Aug) vs SkelGym-Full | 65.48% | 71.05% | +5.58% | 23/176 | 2.20e-30 | 8.81e-30 | 5.51e-30 | ******* |
| Four-Stream AAGCN vs SkelGym-Full | 67.34% | 71.05% | +3.72% | 32/134 | 4.79e-16 | 9.59e-16 | 5.99e-16 | ******* |

## 2. Video-Level Exact McNemar Tests (Binary Correctness)

| Comparison | Video Acc 1 | Video Acc 2 | Delta | Discordant (b/c) | Exact p (Raw) | Holm-Bonferroni p | FDR p (B-H) | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Transformer vs Aug Transformer | 74.25% | 81.97% | +7.73% | 8/26 | 0.0029 | 0.0103 | 0.0049 | ***** |
| ST-GCN vs Four-Stream AAGCN | 66.95% | 79.40% | +12.45% | 11/40 | 5.70e-05 | 0.0003 | 0.0003 | ******* |
| Transformer Mix (Aug) vs SkelGym-Full | 81.97% | 79.40% | -2.58% | 10/4 | 0.1796 | 0.3591 | 0.2245 | **n.s.** |
| AAGCN Bone (Aug) vs SkelGym-Full | 73.39% | 79.40% | +6.01% | 3/17 | 0.0026 | 0.0103 | 0.0049 | ***** |
| Four-Stream AAGCN vs SkelGym-Full | 79.40% | 79.40% | +0.00% | 8/8 | 1.0000 | 1.0000 | 1.0000 | **n.s.** |

## 3. Video-Level Confidence Calibration (Continuous Probability Tests)

| Comparison | Mean Delta P(True) | Cohen's d_z | Wilcoxon W p (Raw) | Holm-Bonferroni p | Paired t p | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean Transformer vs Aug Transformer | +0.0906 | 0.377 | 8.14e-09 | 3.26e-08 | 2.75e-08 | ******* |
| ST-GCN vs Four-Stream AAGCN | -0.0724 | -0.221 | 1.61e-05 | 3.22e-05 | 0.0009 | ******* |
| Transformer Mix (Aug) vs SkelGym-Full | -0.0344 | -0.305 | 6.05e-06 | 1.81e-05 | 5.34e-06 | ******* |
| AAGCN Bone (Aug) vs SkelGym-Full | +0.0181 | 0.178 | 0.0309 | 0.0309 | 0.0071 | ***** |
| Four-Stream AAGCN vs SkelGym-Full | +0.1054 | 0.711 | 2.81e-21 | 1.41e-20 | 1.93e-22 | ******* |
