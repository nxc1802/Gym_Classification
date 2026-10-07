# Master Audit and Final Results: SkelGym Benchmark Suite

## Table 2: Clean Sequences Temporal Architecture Comparison (Paper Table 1)

*Objective:* Evaluate sequential backbones without augmentations across metric World 3D (39-d) and Biomechanical Mix v2 (63-d). Test metrics and video-level consensus are intentionally hidden to enforce zero-leakage validation screening.  
*Execution Command:* `python scripts/run_phase2_table2_refresh.py`

| Architecture / Model | Input Modality | Dim | Params | Val Loss | Val Win Acc (%) | Val Win F1 | Validation Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM (world_3d)** | world_3d | 39-d | 362K | N/A | 73.64% ± 1.26% | 0.7244 ± 0.0128 | Verified |
| **LSTM (mix_v2)** | mix_v2 | 63-d | 362K | N/A | 75.90% ± 1.60% | 0.7541 ± 0.0173 | Verified |
| **BiLSTM (world_3d)** | world_3d | 39-d | 362K | N/A | 74.72% ± 0.19% | 0.7341 ± 0.0020 | Verified |
| **BiLSTM (mix_v2)** | mix_v2 | 63-d | 362K | N/A | 76.90% ± 0.30% | 0.7599 ± 0.0050 | Verified |
| **Transformer (world_3d)** | world_3d | 39-d | 301K | N/A | 78.76% ± 0.46% | 0.7781 ± 0.0053 | Verified |

---

## Table 3: Graph Kinematic Streams Benchmark (Paper Table 5)

*Objective:* Evaluate multi-stream geometric and dynamic representations on Adaptive GCN (AAGCN). Test metrics and video-level consensus are hidden to enforce zero-leakage validation screening.  
*Execution Command:* `python scripts/run_phase6_graph_streams.py --aug_method mirror_yaw`

| Stream ID | Stream / Configuration | Model Backbone | Input Modality | Val Win Acc (%) | Val Win F1 | Validation Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **T3.1** | Raw 3D Joint (ST-GCN Clean) | STGCN | raw_3d | 63.01% ± 1.86% | 0.6255 ± 0.0224 | Verified |
| **T3.2** | World 3D Joint (ST-GCN Clean) | STGCN | world_3d | 73.75% ± 1.86% | 0.7332 ± 0.0154 | Verified |
| **T3.3** | Bone 3D Stream (AAGCN Clean) | AAGCN | bone_3d | 76.00% ± 0.35% | 0.7542 ± 0.0037 | Verified |
| **T3.4** | Bone 3D Stream (AAGCN + Aug) | AAGCN | bone_3d | 79.30% ± 0.22% | 0.7858 ± 0.0032 | Verified |
| **T3.5** | World Joint Stream (AAGCN + Aug) | AAGCN | world_3d | 78.86% ± 0.31% | 0.7856 ± 0.0024 | Verified |
| **T3.6** | World Joint Motion (Delta X) | AAGCN | world_joint_motion_3d | 64.19% ± 1.61% | 0.6464 ± 0.0083 | Verified |
| **T3.7** | Bone Motion (Delta B) | AAGCN | bone_motion_3d | 61.93% ± 1.56% | 0.6362 ± 0.0107 | Verified |
| **T3.8** | Two-Stream AAGCN (World Joint + Bone) | AAGCN | Graph | 82.03% ± 0.41% | 0.8179 ± 0.0036 | Verified |
| **T3.9** | Four-Stream AAGCN (Full) | AAGCN | Graph | 84.34% ± 0.26% | 0.8414 ± 0.0030 | Verified |

---

## Table 4: Systematic Leave-One-Out (LOO) Augmentation Ablation on Transformer Mix (Paper Table 2)

*Objective:* Evaluate necessity of individual operators by excluding one at a time from Candidate Full (5-op) across 3 seeds ($42, 123, 3407$). Displayed strictly on validation window metrics for fair, zero-test-leakage assessment.  
*Execution Command:* `python scripts/run_phase3_ablation_world_mix_v2.py`

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Win Acc (%) | Val Win F1 | Validation Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clean** | All Operators Excluded | N/A | 79.52% ± 0.80% | 0.7885 ± 0.0101 | Unaugmented Baseline Control |
| **Candidate Full 5Op** | None (Reference 5-Op Suite) | N/A | 79.19% ± 1.26% | 0.7882 ± 0.0152 | Reference 5-Operator Suite (Val Win F1: 0.7882) |
| **Candidate Minus Mirror** | Sagittal Bilateral Reflection (p=0.5) | N/A | 78.78% ± 1.44% | 0.7757 ± 0.0197 | 🚨 Weakest LOO window performance (-1.25% Win F1) |
| **Candidate Minus Yaw** | Gravitational Yaw Rotation (±15°) | N/A | 80.06% ± 0.32% | 0.7964 ± 0.0026 | Moderate window gain over 5-op (+0.82% Win F1) |
| **Candidate Minus Scale** | Proportional Scale Variation (±10%) | N/A | 80.13% ± 0.13% | 0.7985 ± 0.0022 | Peak window gain in LOO (+1.03% Win F1) |
| **Candidate Minus Time** | Temporal Resampling (0.8x - 1.2x) | N/A | 79.54% ± 0.57% | 0.7926 ± 0.0060 | Lowest 4-op Val Loss (1.0010, +0.44% Win F1) |
| **Candidate Minus Jitter** | Gaussian Coordinate Jitter (σ=0.008) | N/A | 79.68% ± 0.22% | 0.7937 ± 0.0048 | Minor window gain (+0.55% Win F1) |

---

## Table 5: Single & Paired Component Isolated Augmentation Ablation on Transformer Mix (Paper Table 3)

*Objective:* Evaluate standalone and paired individual gains relative to the unaugmented baseline. Displayed strictly on validation window metrics for fair, zero-test-leakage assessment.  
*Execution Command:* `python scripts/run_ablation_mirror_yaw.py`

| Augmentation Strategy | Isolated Operator Description | Val Loss | Val Win Acc (%) | Val Win F1 | Standalone Validation Effect |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Clean** | Unaugmented Native Window Sequences | N/A | 79.52% ± 0.80% | 0.7885 ± 0.0101 | Unaugmented Baseline Control |
| **Single Mirror** | Sagittal Bilateral Reflection (p=0.5) | N/A | 80.45% ± 0.45% | 0.8001 ± 0.0067 | 🏆 Decisive window gain across single ops (+1.16% Win F1) |
| **Single Yaw** | 3D Yaw Perturbation (±15°) | N/A | 80.29% ± 0.34% | 0.7977 ± 0.0031 | Strong standalone window gain (+0.92% Win F1) |
| **Pair Mirror Yaw** | Bilateral Reflection + Gravitational Yaw (±15°) | 1.0465 ± 0.0327 | 81.20% ± 0.72% | 0.8119 ± 0.0072 | 🥇 **All-Time Peak Val Win Acc & Macro F1 (+2.34% Win F1)** |
| **Single Scale** | Proportional Scale Jitter (±10%) | N/A | 79.68% ± 0.22% | 0.7904 ± 0.0066 | Marginal window gain (+0.19% Win F1) |
| **Single Time** | Linear Sequence Resampling (0.8x - 1.2x) | N/A | 79.74% ± 0.24% | 0.7920 ± 0.0039 | Marginal window gain over clean (+0.35% Win F1) |
| **Single Jitter** | Gaussian Noise (σ=0.008) | N/A | 79.55% ± 0.82% | 0.7884 ± 0.0093 | Neutral window performance (-0.01% Win F1) |

> **Critical Methodological Rationale (Validation-Driven Grounding):**
> 1. **Distance-Preserving Euclidean Transformations:** Bilateral Reflection and Gravitational Yaw ($\pm 15^\circ$) strictly preserve physical metric dimensions (measured in meters in World 3D) and kinematic joint angles. While Yaw rotation belongs to the proper rotation group $SO(3) \subset SE(3)$, bilateral reflection acts as an improper Euclidean isometry ($\det = -1$) that exploits anatomical bilateral symmetry across the sagittal plane in bilateral resistance training exercises.
> 2. **Peak Representation Learning:** The paired configuration **Mirror + Yaw** achieves the highest Validation Window Accuracy (**81.20% ± 0.72%**) and Macro F1 (**0.8119 ± 0.0072**) across all 13 experimental configurations tested (+1.68% Win Acc, +0.0234 Win F1 over Clean Baseline).
> 3. **Non-Rigid Dynamics & Metric Distortion Elimination:** Jitter introduces high-frequency sensor noise, Scale alters anatomical limb proportions, and TimeWarp disrupts exercise tempo and velocity profiles, explaining why combining all five operators in a monolithic suite degrades validation window representations.

---

## Table 6: Multi-Stream Cross-Paradigm Ensemble Comparison (Paper Table 6)

*Objective:* Benchmark 3 standardized fusion methods across multi-stream configurations and multi-seed downstream consistency across seeds $42, 123, 3407$.  
*Methodological Guarantee:* Test metrics are **revealed ONLY for the Validation Winners** of SkelGym-Lite and SkelGym-Full; all other rows remain hidden to preserve strict post-validation test separation.  
*Execution Command:* `python scripts/run_phase7_table6_ensembles.py --proposed_aug_cfg_id pair_mirror_yaw`

| Architecture / Configuration | Fusion Protocol & Weighting | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Transformer Mix (63-d)** | Single Sequence Backbone (Mirror+Yaw) | 81.20% ± 0.72% | 0.8119 ± 0.0072 | 84.70% ± 0.82% | 0.8469 ± 0.0105 | - | - | - | - | Reference Backbone |
| **SkelGym-Lite** | Hard Majority Voting (Discrete Baseline) | 79.47% ± 0.81% | 0.7883 ± 0.0084 | 82.93% ± 0.60% | 0.8277 ± 0.0065 | - | - | - | - | Verified |
| **SkelGym-Lite** | Accuracy-Weighted Soft Voting (Validation-calibrated weights) | 83.47% ± 0.14% | 0.8351 ± 0.0021 | 85.67% ± 1.14% | 0.8634 ± 0.0066 | 73.74% ± 0.63% | 0.7310 ± 0.0058 | **82.12% ± 0.40%** | **0.8198 ± 0.0030** | 🏆 Validation Winner (SkelGym-Lite) |
| **SkelGym-Lite** | Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K) | 83.26% ± 0.06% | 0.8327 ± 0.0008 | 85.67% ± 1.14% | 0.8634 ± 0.0066 | - | - | - | - | Verified |
| **SkelGym-Full** | Hard Majority Voting (Discrete Baseline) | 84.11% ± 0.30% | 0.8411 ± 0.0041 | 86.15% ± 1.21% | 0.8660 ± 0.0090 | - | - | - | - | Verified |
| **SkelGym-Full** | Accuracy-Weighted Soft Voting (Validation-calibrated weights) | 84.85% ± 0.29% | 0.8475 ± 0.0024 | 87.28% ± 1.27% | 0.8821 ± 0.0113 | - | - | - | - | Verified |
| **SkelGym-Full** | Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K) | 85.05% ± 0.47% | 0.8487 ± 0.0046 | 87.92% ± 1.37% | 0.8886 ± 0.0126 | 76.19% ± 0.16% | 0.7537 ± 0.0014 | **85.27% ± 0.41%** | **0.8386 ± 0.0075** | 🏆 Validation Winner (SkelGym-Full) |

---

## Table 7: Window-Level vs Video Consensus Predictions & Parameter Footprints (Paper Table 7)

*Objective:* Quantify consensus pooling accuracy gains and compare total trainable parameters across all audited architectures.  
*Execution Command:* `python scripts/run_phase8_10_evaluation_and_reports.py`

| Model Architecture | Input Modality / Paradigm | Trainable Params | Val Win Acc (%) | Val Win F1 | Test Win Acc (%) | Test Win Macro F1 | Test Vid Acc (%) | Test Vid Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM (World 3D Clean)** | Skeletal Motion | 362K | 73.64% ± 1.26% | 0.7244 ± 0.0128 | 61.41% ± 0.80% | 0.6002 ± 0.0063 | **71.10% ± 2.33%** | **0.6941 ± 0.0294** | Verified |
| **LSTM (Biomechanical Mix v2)** | Skeletal Motion | 362K | 75.90% ± 1.60% | 0.7541 ± 0.0173 | 62.80% ± 0.83% | 0.6171 ± 0.0089 | **72.39% ± 1.76%** | **0.7077 ± 0.0172** | Verified |
| **BiLSTM (Biomechanical Mix v2)** | Skeletal Motion | 360K | 76.90% ± 0.30% | 0.7599 ± 0.0050 | 64.82% ± 0.37% | 0.6316 ± 0.0027 | **75.39% ± 1.46%** | **0.7342 ± 0.0144** | Verified |
| **Transformer (Biomechanical Mix v2 Clean)** | Skeletal Motion | 301K | 79.52% ± 0.80% | 0.7885 ± 0.0101 | 69.24% ± 0.20% | 0.6815 ± 0.0017 | **79.83% ± 1.27%** | **0.7806 ± 0.0214** | Verified |
| **Transformer (Biomechanical Mix v2 + SkelGym-Aug)** | Skeletal Motion | 301K | 81.20% ± 0.72% | 0.8119 ± 0.0072 | 73.14% ± 0.90% | 0.7253 ± 0.0106 | **82.69% ± 1.66%** | **0.8187 ± 0.0196** | Verified |
| **ST-GCN (Raw 3D Clean)** | Skeletal Motion | 350K | 63.01% ± 1.86% | 0.6255 ± 0.0224 | 48.35% ± 0.94% | 0.4701 ± 0.0141 | **58.65% ± 1.76%** | **0.5432 ± 0.0208** | Verified |
| **AAGCN (Bone 3D + SkelGym-Aug)** | Skeletal Motion | 378K | 79.30% ± 0.22% | 0.7858 ± 0.0032 | 66.36% ± 1.41% | 0.6548 ± 0.0173 | **75.39% ± 2.33%** | **0.7440 ± 0.0335** | Verified |
| **Four-Stream AAGCN (Uniform Soft Voting)** | Skeletal Motion | 1.51M | 84.34% ± 0.26% | 0.8414 ± 0.0030 | 74.50% ± 0.65% | 0.7341 ± 0.0048 | **84.41% ± 1.01%** | **0.8308 ± 0.0179** | Verified |
| **SkelGym-Lite (2 Streams, Accuracy-Weighted Soft Voting (Validation-calibrated weights))** | Skeletal Motion | 679K | 83.47% ± 0.14% | 0.8351 ± 0.0021 | 73.74% ± 0.63% | 0.7310 ± 0.0058 | **82.12% ± 0.40%** | **0.8198 ± 0.0030** | Verified |
| **SkelGym-Full (5 Streams, Uniform Average Soft Voting (Zero-parameter heuristic SOTA, w_i = 1/K))** | Skeletal Motion | 1.81M | 85.05% ± 0.47% | 0.8487 ± 0.0046 | 76.19% ± 0.16% | 0.7537 ± 0.0014 | **85.27% ± 0.41%** | **0.8386 ± 0.0075** | Verified |

---

## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unaugmented Trans vs SkelGym-Aug Trans** | 16.65 | 4.21e-05 | 1.57 | 10432.5 | 0.0019 | 0.0022 | +0.203 | Verified (**) |
| **Fixed ST-GCN vs Adaptive Four-Stream AAGCN** | 118.62 | 1.46e-28 | 2.63 | 12666.0 | 0.3491 | 0.1026 | +0.107 | Verified (n.s.) |
| **Single Sequence (Trans) vs SkelGym-Full** | 42.44 | 3.84e-11 | 2.42 | 6086.0 | 2.39e-13 | 2.12e-11 | -0.461 | Verified (***) |
| **Single Graph (AAGCN Bone) vs SkelGym-Full** | 137.47 | 5.60e-34 | 4.21 | 11026.0 | 0.0115 | 0.2098 | -0.082 | Verified (*) |
| **Four-Stream Graph AAGCN vs SkelGym-Full** | 15.82 | 5.88e-05 | 2.12 | 6086.0 | 2.39e-13 | 2.12e-11 | +0.461 | Verified (***) |

> **Note on Multiple Testing Correction & Statistical Semantics:** All five pairwise window comparisons remain statistically significant after Holm-Bonferroni step-down correction ($p_{\text{adj}} \le 0.0001$) and Benjamini-Hochberg False Discovery Rate control ($\text{FDR} \le 5.88 \times 10^{-5}$). Video-level Wilcoxon signed-rank and paired $t$-tests evaluate paired differences in ground-truth class posterior probabilities (confidence) across video clusters ($N=233$), quantifying confidence calibration rather than discrete video classification differences.

---

## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling ($B=1,000$, clustered by video ID to prevent intra-video frame dependency bias).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM (Mix v2 63-d)** | 61.48% [55.12%, 67.40%] | 0.5962 [0.5375, 0.6507] | 74.31% [69.09%, 79.83%] | 0.7209 [0.6606, 0.7828] | Verified |
| **BiLSTM (Mix v2 63-d)** | 64.57% [58.54%, 70.37%] | 0.6112 [0.5514, 0.6765] | 76.93% [71.24%, 81.97%] | 0.7358 [0.6688, 0.7969] | Verified |
| **ST-GCN (World 3D)** | 65.07% [58.49%, 71.17%] | 0.6115 [0.5610, 0.6657] | 76.63% [70.82%, 81.98%] | 0.7267 [0.6703, 0.7861] | Verified |
| **Transformer (Mix v2 Clean)** | 69.91% [63.40%, 75.76%] | 0.6723 [0.6093, 0.7365] | 79.88% [74.68%, 84.98%] | 0.7788 [0.7219, 0.8325] | Verified |
| **Transformer (Mix v2 SkelGym-Aug)** | 72.60% [66.60%, 78.19%] | 0.6996 [0.6390, 0.7548] | 82.03% [76.81%, 86.70%] | 0.7985 [0.7349, 0.8521] | Verified |
| **AAGCN (Bone 3D)** | 68.20% [62.11%, 74.12%] | 0.6621 [0.6017, 0.7195] | 78.51% [72.96%, 83.69%] | 0.7792 [0.7274, 0.8297] | Verified |
| **Four-Stream AAGCN** | 74.67% [68.69%, 80.33%] | 0.7222 [0.6647, 0.7769] | 85.75% [81.12%, 90.13%] | 0.8478 [0.7998, 0.8927] | Verified |
| **SkelGym-Lite** | 73.63% [67.68%, 79.13%] | 0.7143 [0.6543, 0.7677] | 82.42% [77.25%, 87.12%] | 0.8093 [0.7513, 0.8584] | Verified |
| **SkelGym-Full (Uniform Soft)** | **76.34% [70.03%, 81.90%]** | **0.7412 [0.6785, 0.7969]** | **85.83% [81.12%, 90.13%]** | **0.8409 [0.7848, 0.8937]** | Verified |

---

## Table 10: Per-Class Performance Breakdown — SkelGym-Full (Uniform Soft SOTA) (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full (Uniform Soft SOTA) on held-out test windows ($N=2,743$) and test videos ($N=233$, 200 correct, 85.84% accuracy).

| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| barbell biceps curl | 0.4069 | 0.8551 | 0.5514 | 69 | 0.6667 | 1.0000 | 0.8000 | 14 | Verified |
| bench press | 0.4091 | 0.6562 | 0.5040 | 96 | 0.7500 | 0.6429 | 0.6923 | 14 | Verified |
| chest fly machine | 0.9213 | 1.0000 | 0.9591 | 82 | 0.8889 | 1.0000 | 0.9412 | 8 | Verified |
| deadlift | 0.4046 | 0.7910 | 0.5354 | 67 | 0.6923 | 0.9000 | 0.7826 | 10 | Verified |
| decline bench press | 0.6270 | 0.5896 | 0.6077 | 134 | 0.7500 | 0.7500 | 0.7500 | 8 | Verified |
| hammer curl | 0.5932 | 0.4348 | 0.5018 | 161 | 0.8750 | 0.5000 | 0.6364 | 14 | Verified |
| hip thrust | 0.9245 | 0.6229 | 0.7443 | 236 | 1.0000 | 0.8889 | 0.9412 | 9 | Verified |
| incline bench press | 0.7397 | 0.7105 | 0.7248 | 76 | 0.7778 | 0.7778 | 0.7778 | 9 | Verified |
| lat pulldown | 0.6691 | 0.9300 | 0.7782 | 100 | 0.8125 | 1.0000 | 0.8966 | 13 | Verified |
| lateral raise | 1.0000 | 0.9484 | 0.9735 | 155 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| leg extension | 0.9542 | 1.0000 | 0.9766 | 125 | 1.0000 | 1.0000 | 1.0000 | 13 | Verified |
| leg raises | 0.9898 | 0.8509 | 0.9151 | 114 | 1.0000 | 1.0000 | 1.0000 | 11 | Verified |
| plank | 0.7925 | 0.7500 | 0.7706 | 56 | 0.6667 | 1.0000 | 0.8000 | 2 | Verified |
| pull Up | 0.7971 | 0.6707 | 0.7285 | 82 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| push-up | 0.9062 | 1.0000 | 0.9508 | 87 | 0.9231 | 1.0000 | 0.9600 | 12 | Verified |
| romanian deadlift | 0.6792 | 0.4932 | 0.5714 | 146 | 0.7500 | 0.5000 | 0.6000 | 6 | Verified |
| russian twist | 0.9844 | 0.9197 | 0.9509 | 137 | 1.0000 | 1.0000 | 1.0000 | 6 | Verified |
| shoulder press | 0.7101 | 0.6447 | 0.6759 | 152 | 0.7500 | 0.6923 | 0.7200 | 13 | Verified |
| squat | 0.9384 | 0.8319 | 0.8820 | 238 | 1.0000 | 1.0000 | 1.0000 | 15 | Verified |
| t bar row | 0.8400 | 0.5385 | 0.6562 | 117 | 1.0000 | 0.7000 | 0.8235 | 10 | Verified |
| tricep Pushdown | 0.6083 | 0.7849 | 0.6854 | 93 | 0.7857 | 0.9167 | 0.8462 | 12 | Verified |
| tricep dips | 0.8987 | 0.9682 | 0.9322 | 220 | 0.8889 | 0.8889 | 0.8889 | 9 | Verified |
| **Overall Accuracy** | 0.7641 | 0.7641 | 0.7641 | **2743** | 0.8584 | 0.8584 | 0.8584 | **233** | **Verified** |
| **Macro Average** | 0.7634 | 0.7723 | 0.7534 | **2743** | 0.8626 | 0.8572 | 0.8491 | **233** | **Verified** |

---

## Table 11: Computational Complexity & Inference Latency (Paper Table 11)

*Objective:* Measure parameters, window FLOPs, and latency across server and edge devices.  
*Execution Command:* `python scripts/benchmark_hardware_latency.py`

| Model Architecture | Parameters | FLOPs per Window | RTX PRO 6000 (CUDA) | Apple M4 (MPS) | Apple M4 (CPU) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (63-d)** | 301K | 18.87 MFLOPs | 0.59 ms (1685 FPS) | 1.36 ms (738 FPS) | 0.49 ms (2041 FPS) | Verified |
| **AAGCN Joint Stream** | 378K | 202.86 MFLOPs | 1.01 ms (987 FPS) | 1.89 ms (530 FPS) | 0.90 ms (1115 FPS) | Verified |
| **AAGCN Bone Stream** | 378K | 202.86 MFLOPs | 1.02 ms (976 FPS) | 1.72 ms (581 FPS) | 0.88 ms (1138 FPS) | Verified |
| **AAGCN Joint-Motion Stream** | 378K | 202.86 MFLOPs | 1.01 ms (990 FPS) | 1.90 ms (527 FPS) | 1.02 ms (978 FPS) | Verified |
| **AAGCN Bone-Motion Stream** | 378K | 202.86 MFLOPs | 1.02 ms (980 FPS) | 1.90 ms (527 FPS) | 1.03 ms (975 FPS) | Verified |
| **SkelGym-Lite (Transformer + Bone)** | 679K | 221.72 MFLOPs | 1.66 ms (603 FPS) | 2.55 ms (393 FPS) | 1.49 ms (671 FPS) | Verified |
| **SkelGym-Full (Transformer + 4 AAGCN)** | 1.81M | 830.29 MFLOPs | 4.65 ms (215 FPS) | 5.86 ms (171 FPS) | 3.96 ms (253 FPS) | Verified |

---

## Table 13: Strong External Baseline — BlockGCN (CVPR 2024 Adapted) (Paper Benchmark Table)

*Objective:* External benchmark comparison against BlockGCN (CVPR 2024), faithfully adapted to 33 MediaPipe joints ($V=33$, $T=32$, $M=1$, $C=22$, joint-only stream), trained strictly from scratch across 3 independent seeds ($42, 123, 3407$).  
*Execution Command:* `python scripts/run_blockgcn_baseline.py --config configs/external/blockgcn_original_33j_32f.yaml --device cuda --push_to_hf`

| Model Architecture | Input Representation | Parameters | FLOPs / MACs | Window Test Acc (%) | Window Macro F1 | Video Consensus Acc (%) | Video Macro F1 | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BlockGCN (CVPR 2024 Adapted)** | 33 Raw MediaPipe XYZ (Joint-only) | 1,352,102 | — | 54.24% ± 1.04% | 0.5576 ± 0.0047 | 70.48% ± 0.40% | 0.6948 ± 0.0024 | **Verified** |
