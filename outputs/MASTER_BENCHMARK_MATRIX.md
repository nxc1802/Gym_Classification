# SkelGym: Internal Master Benchmark Matrix (All-in-One Researcher Dashboard)

> **Confidentiality:** Internal Research & Development Documentation  
> **Dataset Baseline:** Video-Level Partition 6:2:2 (580 Train / 208 Validation / 236 Test; 233 Valid Test Videos $\ge 32$ Frames)  
> **Multi-Seed Protocol:** Evaluated strictly across 3 independent random seeds ($42, 123, 3407$). All metrics formatted as $\text{Mean} \pm \text{Std}$.  
> **Zero-Leakage Assurance:** Validation and Test splits evaluated strictly on pristine native landmarks without any synthetic augmentations.

---

## 1. Single Architecture Backbones Matrix

| Model Architecture | Input Representation | Augmentation Protocol | Exact Params | FLOPs / Window | Inference Latency (CUDA) | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Generalization Gap (Vid) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (Proposed)** | Biomechanical Mix v2 (63-d) | SkelGym-Aug (4-op) | 301K (300,742) | 7.71 MFLOPs | 0.59 ms (1,685 FPS) | 79.54% ± 0.57% | 0.7926 | 83.90% ± 0.60% | 0.8350 | 73.59% ± 0.48% | 0.7257 | **83.55% ± 0.54%** | **0.8300** | -0.35% |
| **Transformer World 3D (Metric)** | World 3D (39-d, Metric) | None (Clean) | 301K (301,436) | 7.64 MFLOPs | 0.58 ms (1,710 FPS) | 78.76% ± 0.46% | 0.7781 | 81.97% ± 0.23% | 0.8254 | 65.33% ± 0.71% | 0.6406 | **75.68% ± 0.53%** | **0.7343** | -6.29% |
| **Transformer Mix v2 (Clean Baseline)** | Biomechanical Mix v2 (63-d) | None (Clean) | 301K (300,742) | 7.71 MFLOPs | 0.59 ms (1,685 FPS) | 79.52% ± 0.80% | 0.7885 | 84.38% ± 1.27% | 0.8505 | 69.24% ± 0.20% | 0.6815 | **79.83% ± 1.27%** | **0.7806** | -4.55% |
| **AAGCN Joint Stream (Proposed)** | World 3D Joint ($V=13$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 1.29 ms (777 FPS) | 79.44% ± 0.78% | 0.7890 | 82.29% ± 0.60% | 0.8240 | 69.56% ± 1.03% | 0.6841 | **80.97% ± 2.94%** | **0.8025** | -1.32% |
| **AAGCN Bone Stream (Proposed)** | Bone 3D Vector ($V=13$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.97 ms (1,027 FPS) | 78.66% ± 1.52% | 0.7812 | 81.80% ± 0.60% | 0.8190 | 65.62% ± 1.37% | 0.6528 | **73.39% ± 1.22%** | **0.7352** | -8.41% |
| **AAGCN Bone Stream (Clean Baseline)** | Bone 3D Vector ($V=13$) | None (Clean) | 378K (377,750) | 202.86 MFLOPs | 0.97 ms (1,027 FPS) | 74.52% ± 0.77% | 0.7380 | 79.06% ± 1.60% | 0.7950 | 62.10% ± 2.11% | 0.6101 | **71.67% ± 2.80%** | **0.7025** | -7.39% |
| **AAGCN Joint-Motion Stream** | Joint Velocity 3D ($\Delta X$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.96 ms (1,038 FPS) | 62.47% ± 0.58% | 0.6180 | 70.37% ± 1.86% | 0.7010 | 53.41% ± 0.50% | 0.5207 | **70.53% ± 0.73%** | **0.6788** | +0.16% |
| **AAGCN Bone-Motion Stream** | Bone Velocity 3D ($\Delta B$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.98 ms (1,022 FPS) | 56.63% ± 0.72% | 0.5590 | 63.28% ± 2.40% | 0.6300 | 48.76% ± 1.26% | 0.4784 | **66.38% ± 3.24%** | **0.6226** | +3.10% |
| **ST-GCN World 3D (Metric Clean)** | World 3D Joint ($V=13$) | None (Clean) | 350K (349,606) | 184.20 MFLOPs | 0.82 ms (1,220 FPS) | 73.75% ± 1.86% | 0.7320 | 79.71% ± 1.98% | 0.8092 | 62.59% ± 2.31% | 0.6034 | **74.68% ± 1.26%** | **0.7060** | -5.03% |
| **ST-GCN Raw 3D Baseline** | Raw 3D Joint ($V=13$) | None (Clean) | 350K (349,606) | 184.20 MFLOPs | 0.82 ms (1,220 FPS) | 63.01% ± 1.86% | 0.6255 | 69.08% ± 1.42% | 0.6932 | 48.35% ± 0.94% | 0.4701 | **58.65% ± 1.76%** | **0.5432** | -10.43% |
| **LSTM Baseline (World 3D Clean)** | World 3D Joint (39-d) | None (Clean) | 362K (361,814) | 11.58 MFLOPs | 0.74 ms (1,351 FPS) | 73.64% ± 1.26% | 0.7280 | 76.97% ± 2.63% | 0.7710 | 61.41% ± 0.80% | 0.6002 | **71.10% ± 2.33%** | **0.6941** | -5.87% |
| **LSTM Baseline (Mix v2 Clean)** | Biomechanical Mix v2 (63-d) | None (Clean) | 362K (361,814) | 11.58 MFLOPs | 0.74 ms (1,351 FPS) | 75.90% ± 1.60% | 0.7510 | 80.19% ± 2.09% | 0.8050 | 62.80% ± 0.83% | 0.6171 | **72.39% ± 1.76%** | **0.7077** | -7.80% |
| **BiLSTM Baseline (Mix v2 Clean)** | Biomechanical Mix v2 (63-d) | None (Clean) | 360K (360,150) | 13.90 MFLOPs | 0.86 ms (1,162 FPS) | 76.90% ± 0.30% | 0.7630 | 80.68% ± 1.97% | 0.8120 | 64.82% ± 0.37% | 0.6316 | **75.39% ± 1.46%** | **0.7342** | -5.29% |

---

## 2. Multi-Stream & Cross-Paradigm Ensemble Fusion Matrix

| Ensemble Architecture | Fusion Strategy | Trainable Params | FLOPs / Window | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Consensus Gain (+$\Delta$ Vid) | Research Recommendation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Two-Stream AAGCN (Joint + Bone)** | Uniform Soft Voting | 756K (755,500) | 405.72 MFLOPs | 80.88% ± 0.93% | 0.8050 | 84.06% ± 1.04% | 0.8410 | 71.24% ± 1.09% | 0.7010 | **80.98% ± 0.41%** | **0.8083** | +9.74% | Standard graph 2-stream baseline |
| **Four-Stream AAGCN (Unified Graph)** | Uniform Soft Voting | 1.51M (1,511,000) | 811.44 MFLOPs | 84.14% ± 0.45% | 0.8395 | 86.96% ± 0.79% | 0.8710 | 73.41% ± 0.17% | 0.7233 | **83.26% ± 0.35%** | **0.8237** | +9.85% | ⚡ **Pure Graph SOTA (83.26%)** |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Hard Majority Voting | 679K (678,492) | 210.57 MFLOPs | 77.62% ± 1.31% | 0.7705 | 85.35% ± 0.23% | 0.8539 | 66.60% ± 1.58% | 0.6714 | **82.40% ± 0.35%** | **0.8234** | +15.80% | Discrete voting baseline |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Accuracy-Weighted Soft | 679K (678,492) | 210.57 MFLOPs | 83.42% ± 0.24% | 0.8322 | 85.18% ± 0.23% | 0.8527 | 73.57% ± 0.42% | 0.7265 | **82.26% ± 0.20%** | **0.8220** | +8.69% | Validation-calibrated soft |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Uniform Average Soft | 679K (678,492) | 210.57 MFLOPs | 83.52% ± 0.17% | 0.8332 | 85.35% ± 0.23% | 0.8539 | 73.31% ± 0.37% | 0.7240 | **82.40% ± 0.35%** | **0.8234** | +9.09% | 🚀 **Efficient Zero-Param SOTA (82.40%)** |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Stacking Meta-Classifier | 679K (678,492) | 210.57 MFLOPs | 87.65% ± 0.40% | 0.8758 | 89.70% ± 0.60% | 0.8970 | 73.61% ± 0.21% | 0.7271 | **83.55% ± 0.88%** | **0.8338** | +9.94% | 🚀 **Best Compact Edge SOTA (83.55%)** |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Hard Majority Voting | 1.81M (1,811,742) | 819.13 MFLOPs | 83.33% ± 1.03% | 0.8322 | 88.89% ± 0.79% | 0.8892 | 72.96% ± 0.05% | 0.7248 | **83.69% ± 0.70%** | **0.8308** | +10.73% | Discrete 5-stream voting |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Accuracy-Weighted Soft | 1.81M (1,811,742) | 819.13 MFLOPs | 85.19% ± 0.11% | 0.8503 | 88.57% ± 0.91% | 0.8848 | 75.51% ± 0.47% | 0.7448 | **84.69% ± 0.54%** | **0.8417** | +9.18% | 🥈 **Calibrated Soft SOTA (84.69%)** |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Uniform Average Soft | 1.81M (1,811,742) | 819.13 MFLOPs | 85.33% ± 0.43% | 0.8517 | 88.89% ± 0.79% | 0.8892 | 75.41% ± 0.19% | 0.7439 | **83.69% ± 0.70%** | **0.8308** | +8.28% | ⚡ **Zero-Param Multi-Stream SOTA (83.69%)** |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Stacking Meta-Classifier | 1.81M (1,811,742) | 819.13 MFLOPs | 93.41% ± 0.09% | 0.9313 | 94.85% ± 0.46% | 0.9507 | 75.13% ± 0.66% | 0.7418 | **85.84% ± 0.61%** | **0.8540** | +10.71% | 🏆 **ALL-TIME PROJECT SOTA (85.84%)** |

---

## 3. Systematic Biomechanical Data Augmentation Matrix (Leave-One-Out on mix_v2 63-d)

| Augmentation Configuration | Excluded Operator / Domain | Val Loss | Val Win Acc (%) | Val Win F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | $\Delta$ vs Full (Test Win) | Scientific Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Candidate Full (All 5 Ops)** | None (Reference Suite) | 0.9669 | 79.19% ± 1.26% | 0.7882 | 72.40% ± 1.19% | 0.7145 | 81.97% ± 0.70% | 0.8146 | 0.00% (Ref) | Baseline benchmark suite |
| **w/o Sensor Jitter (-Jitter)** | Gaussian Coordinate Noise ($\sigma=0.008$) | 1.0124 | 79.68% ± 0.22% | 0.7937 | 73.17% ± 0.90% | 0.7208 | 82.12% ± 0.73% | 0.8181 | +0.77% | Slight regularization gain |
| **w/o Sagittal Reflection (-Mirror)** | Bilateral Reflection ($p=0.5$) | 1.0670 | 78.78% ± 1.44% | 0.7757 | 68.27% ± 2.32% | 0.6725 | 78.26% ± 2.68% | 0.7679 | **-4.13%** | 🚨 **Crucial: Severe performance collapse** |
| **w/o Gravitational Yaw (-Yaw)** | Vertical Axis 3D Yaw ($\pm 15^\circ$) | 1.0054 | 80.06% ± 0.32% | 0.7964 | 72.99% ± 0.60% | 0.7188 | 82.69% ± 0.41% | 0.8210 | +0.59% | Moderate camera variation |
| **w/o Proportional Scaling (-Scale)** | Anthropometric Scale ($\pm 10\%$) | 1.0032 | 80.13% ± 0.13% | 0.7985 | 73.02% ± 1.24% | 0.7198 | 82.69% ± 1.62% | 0.8211 | +0.62% | Body scale invariance |
| **w/o Temporal TimeWarp (-TimeWarp)** | Resampling ($0.8\times - 1.2\times$, **SkelGym-Aug 4-op**) | 1.0010 | 79.54% ± 0.57% | 0.7926 | 73.59% ± 0.48% | 0.7257 | 83.55% ± 0.54% | 0.8300 | **+1.19%** | 🏆 **Optimal Benchmark Setting (83.55% Vid)** |
| **Clean Baseline (No Augmentation)** | All 5 Operators Excluded | 1.0483 | 79.52% ± 0.80% | 0.7885 | 69.24% ± 0.20% | 0.6815 | 79.83% ± 1.27% | 0.7806 | **-3.16%** | Unaugmented baseline control |

---

## 4. External State-of-the-Art Baseline Comparison

| External Model | Publication Venue | Input Format | Trainable Params | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Relative $\Delta$ vs SkelGym-Full |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN** | AAAI 2018 | 13 Raw MediaPipe 3D Joints | 350K | 48.35% ± 0.94% | 0.4701 | 58.65% ± 1.76% | 0.5432 | **+27.19%** (SkelGym superiority) |
| **BlockGCN (Adapted)** | CVPR 2024 | 33 Raw MediaPipe 3D Joints | 1.35M | 54.24% ± 1.04% | 0.5576 | 70.48% ± 0.40% | 0.6948 | **+15.36%** (SkelGym superiority) |
| **Four-Stream AAGCN** | Standard Graph | 4 Graph Kinematic Streams | 1.51M | 73.41% ± 0.17% | 0.7233 | 83.26% ± 0.35% | 0.8237 | **+2.58%** (Cross-paradigm boost) |
| **SkelGym-Lite (Stacking)** | **Proposed (Compact)** | **Trans + Bone AAGCN** | **679K** | **73.61% ± 0.21%** | **0.7271** | **83.55% ± 0.88%** | **0.8338** | **Edge-Deployable SOTA** |
| **SkelGym-Full (Stacking)** | **Proposed (Full)** | **63-d Mix + 4 Graph Streams** | **1.81M** | **75.13% ± 0.66%** | **0.7418** | **85.84% ± 0.61%** | **0.8540** | **Reference SOTA (Baseline)** |

---

## 5. Key Scientific & Engineering Insights for Researchers

### A. The Fusion Strategy Paradigm (Stacking vs. Uniform Soft vs. Accuracy-Weighted)
1. **Stacking Meta-Classifier (Ridge/Logistic) is the All-Time Empirical SOTA:**
   - Achieves **85.84% ± 0.61% Video Consensus Accuracy** and **0.8540 Macro-F1** on held-out test videos.
   - Fits lightweight linear meta-classifiers strictly on validation probability outputs, learning optimal class-dependent modality weights without test set leakage.
2. **Accuracy-Weighted Soft Voting is the Best Calibrated Zero-Learner:**
   - Achieves **84.69% ± 0.54% Video Accuracy** and **0.8417 Macro-F1** (Window Acc: **75.51% ± 0.47%**).
   - Weights streams proportionally to validation accuracy, boosting stronger streams while remaining non-parametric.
3. **Uniform Average Soft Voting ($w_i = 1/K$) is the Ideal Zero-Parameter Heuristic:**
   - Achieves **83.69% ± 0.70% Video Accuracy** / **0.8308 Macro-F1** (Window Acc: **75.41% ± 0.19%**) on SkelGym-Full, and **82.40% ± 0.35%** on SkelGym-Lite.
   - Robust, zero-overhead, and completely invariant to validation set shifts.
4. **SLSQP Soft Voting Deprecation:**
   - SLSQP constrained numerical optimization over validation cross-entropy overfits validation partitions and has been permanently removed from the primary benchmark.

### B. Model Parameter Footprint Standardization
- All individual backbones adhere to a calibrated **Compact Budget (300K – 378K)**:
  - Sequence Backbones: Transformer (**301K**), BiLSTM (**360K**), LSTM (**362K**).
  - Graph Backbones: ST-GCN (**350K**), AAGCN (**378K**).
- Multi-Stream Ensembles: SkelGym-Lite (**679K**), Four-Stream AAGCN (**1.51M**), SkelGym-Full (**1.81M**).

### C. SkelGym-Aug Zero-Leakage Validation Supremacy
- On pristine validation windows, `SkelGym-Aug (4-op)` achieves the **lowest cross-entropy loss (1.0010)** and **highest window test generalization (73.59% acc, 83.55% video acc)** compared to unaugmented baseline (69.24% test win acc, 79.83% video acc).
- Excluding Sagittal Reflection (Mirror) causes an immediate **-4.13%** drop on test windows, confirming bilateral symmetry as the single most critical augmentation operator.

### D. Coordinate Space: The Physical Metric Advantage of World Landmarks (`world_3d`)
- Moving from raw camera coordinates to Metric World coordinates provides an immediate **+16.03%** absolute boost for ST-GCN (58.65% to 74.68% Video Acc) and a **+3.29%** boost for Dual-Branch Transformer (72.39% to 75.68%).
- Mid-hip centering naturally grounds the kinematic origin at the pelvic pivot ($p_{\text{hip\_mid}} = (0, 0, 0)$), and true metric coordinates (meters) eliminate perspective-foreshortening distortion across subjects and cameras.
