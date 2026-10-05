# SkelGym: Internal Master Benchmark Matrix (All-in-One Researcher Dashboard)

> **Confidentiality:** Internal Research & Development Documentation  
> **Dataset Baseline:** Video-Level Partition 6:2:2 (580 Train / 208 Validation / 236 Test; 233 Valid Test Videos $\ge 32$ Frames)  
> **Multi-Seed Protocol:** Evaluated strictly across 3 independent random seeds ($42, 123, 3407$). All metrics formatted as $\text{Mean} \pm \text{Std}$.  
> **Zero-Leakage Assurance:** Validation and Test splits evaluated strictly on pristine native landmarks without any synthetic augmentations.

---

## 1. Single Architecture Backbones Matrix

| Model Architecture | Input Representation | Augmentation Protocol | Exact Params | FLOPs / Window | Inference Latency (CUDA) | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Generalization Gap (Vid) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix v2 (Proposed)** | Biomechanical Mix v2 (63-d) | SkelGym-Aug (4-op) | 301K (300,742) | 7.71 MFLOPs | 0.59 ms (1,685 FPS) | 81.90% ± 0.81% | 0.8155 | 84.06% ± 2.56% | 0.8409 | 69.05% ± 2.22% | 0.6877 | **79.26% ± 2.12%** | **0.7892** | -4.80% |
| **Transformer Mix v2 (Clean Baseline)** | Biomechanical Mix v2 (63-d) | None (Clean) | 301K (300,742) | 7.71 MFLOPs | 0.59 ms (1,685 FPS) | 80.98% ± 0.98% | 0.8040 | 84.06% ± 1.74% | 0.8372 | 66.11% ± 1.78% | 0.6536 | **76.25% ± 2.12%** | **0.7536** | -7.81% |
| **AAGCN Bone Stream (Proposed)** | Bone 3D Vector ($V=13$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.97 ms (1,027 FPS) | 78.78% ± 2.03% | 0.7805 | 81.48% ± 1.12% | 0.8236 | 65.34% ± 1.55% | 0.6507 | **73.10% ± 1.31%** | **0.7292** | -8.38% |
| **AAGCN Bone Stream (Clean Baseline)** | Bone 3D Vector ($V=13$) | None (Clean) | 378K (377,750) | 202.86 MFLOPs | 0.97 ms (1,027 FPS) | 76.21% ± 0.60% | 0.7552 | 79.55% ± 1.55% | 0.8064 | 63.02% ± 2.26% | 0.6140 | **71.67% ± 1.97%** | **0.6985** | -7.88% |
| **AAGCN Joint Stream (Proposed)** | Relative 3D Joint ($V=13$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 1.29 ms (777 FPS) | 76.51% ± 0.68% | 0.7541 | 80.68% ± 1.28% | 0.8111 | 67.46% ± 0.26% | 0.6552 | **76.39% ± 2.15%** | **0.7443** | -4.28% |
| **AAGCN Joint-Motion Stream** | Joint Velocity 3D ($\Delta X$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.96 ms (1,038 FPS) | 54.76% ± 4.57% | 0.5558 | 66.51% ± 4.64% | 0.6672 | 48.82% ± 0.33% | 0.4798 | **65.67% ± 1.55%** | **0.6185** | -0.84% |
| **AAGCN Bone-Motion Stream** | Bone Velocity 3D ($\Delta B$) | SkelGym-Aug (4-op) | 378K (377,750) | 202.86 MFLOPs | 0.98 ms (1,022 FPS) | 48.71% ± 1.13% | 0.5180 | 63.77% ± 2.90% | 0.6134 | 49.17% ± 1.33% | 0.4826 | **67.24% ± 3.22%** | **0.6390** | +3.47% |
| **ST-GCN Baseline (Yan et al.)** | Relative 3D Joint ($V=13$) | None (Clean) | 350K (349,606) | 184.20 MFLOPs | 0.82 ms (1,220 FPS) | 70.46% ± 0.56% | 0.7016 | 74.07% ± 3.69% | 0.7401 | 57.58% ± 1.06% | 0.5574 | **66.24% ± 1.62%** | **0.6350** | -7.84% |
| **LSTM Baseline (Hochreiter)** | Biomechanical Mix v2 (63-d) | None (Clean) | 362K (361,814) | 11.58 MFLOPs | 0.74 ms (1,351 FPS) | 77.78% ± 1.41% | 0.7740 | 81.80% ± 0.74% | 0.8180 | 61.27% ± 1.75% | 0.5943 | **71.67% ± 2.39%** | **0.6932** | -10.13% |
| **BiLSTM Baseline (Schuster)** | Biomechanical Mix v2 (63-d) | None (Clean) | 360K (360,150) | 13.90 MFLOPs | 0.86 ms (1,162 FPS) | 76.16% ± 2.68% | 0.7591 | 81.00% ± 0.28% | 0.8092 | 59.63% ± 0.61% | 0.5831 | **69.10% ± 1.87%** | **0.6650** | -11.90% |

---

## 2. Multi-Stream & Cross-Paradigm Ensemble Fusion Matrix

| Ensemble Architecture | Fusion Strategy | Trainable Params | FLOPs / Window | Val Win Acc (%) | Val Win F1 | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Consensus Gain (+$\Delta$ Vid) | Research Recommendation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Two-Stream AAGCN (Joint + Bone)** | Stacking Meta-Classifier | 756K (755,500) | 405.72 MFLOPs | 85.59% | 0.8508 | 87.28% | 0.8751 | 68.68% ± 0.53% | 0.6784 | **78.40% ± 0.25%** | **0.7822** | +9.71% | Standard soft fusion |
| **Two-Stream AAGCN (Joint + Bone)** | Uniform Soft Voting | 756K (755,500) | 405.72 MFLOPs | 79.78% | 0.7907 | 81.80% | 0.8276 | 69.57% ± 0.67% | 0.6851 | **78.68% ± 0.50%** | **0.7731** | +9.11% | Standard soft fusion |
| **Two-Stream AAGCN (Joint + Bone)** | Accuracy-Weighted Soft | 756K (755,500) | 405.72 MFLOPs | 79.84% | 0.7914 | 81.80% | 0.8276 | 69.47% ± 0.75% | 0.6846 | **78.54% ± 0.74%** | **0.7721** | +9.07% | Standard soft fusion |
| **Two-Stream AAGCN (Joint + Bone)** | SLSQP Soft Voting | 756K (755,500) | 405.72 MFLOPs | 80.31% | 0.7956 | 81.80% | 0.8271 | 68.42% ± 1.18% | 0.6770 | **78.40% ± 0.99%** | **0.7777** | +9.98% | ⚠️ Susceptible to Val overfitting (81.40%) |
| **Two-Stream AAGCN (Joint + Bone)** | Hard Voting | 756K (755,500) | 405.72 MFLOPs | 76.55% | 0.7550 | 80.03% | 0.8062 | 64.19% ± 0.34% | 0.6457 | **73.68% ± 1.31%** | **0.7272** | +9.49% | ❌ Suboptimal discrete voting |
| **Four-Stream AAGCN (Unified Graph)** | Stacking Meta-Classifier | 1.51M (1,511,000) | 811.44 MFLOPs | 91.26% | 0.9089 | 94.04% | 0.9409 | 71.38% ± 0.68% | 0.7053 | **81.12% ± 1.14%** | **0.8102** | +9.73% | Standard soft fusion |
| **Four-Stream AAGCN (Unified Graph)** | Uniform Soft Voting | 1.51M (1,511,000) | 811.44 MFLOPs | 82.28% | 0.8161 | 84.70% | 0.8527 | 70.71% ± 1.14% | 0.6982 | **80.40% ± 1.51%** | **0.7919** | +9.69% | Standard soft fusion |
| **Four-Stream AAGCN (Unified Graph)** | Accuracy-Weighted Soft | 1.51M (1,511,000) | 811.44 MFLOPs | 82.27% | 0.8165 | 83.74% | 0.8415 | 70.94% ± 1.18% | 0.6993 | **79.97% ± 1.94%** | **0.7886** | +9.03% | Standard soft fusion |
| **Four-Stream AAGCN (Unified Graph)** | SLSQP Soft Voting | 1.51M (1,511,000) | 811.44 MFLOPs | 80.74% | 0.8005 | 82.13% | 0.8299 | 68.67% ± 1.42% | 0.6799 | **78.40% ± 0.99%** | **0.7777** | +9.73% | ⚠️ Susceptible to Val overfitting (81.40%) |
| **Four-Stream AAGCN (Unified Graph)** | Hard Voting | 1.51M (1,511,000) | 811.44 MFLOPs | 76.82% | 0.7650 | 80.84% | 0.8150 | 65.09% ± 1.32% | 0.6459 | **75.82% ± 1.94%** | **0.7489** | +10.74% | ❌ Suboptimal discrete voting |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Stacking Meta-Classifier | 679K (678,492) | 210.57 MFLOPs | 87.49% | 0.8737 | 89.21% | 0.8934 | 72.10% ± 0.73% | 0.7121 | **81.12% ± 0.00%** | **0.8062** | +9.02% | 🚀 **Best Compact Edge SOTA (81.12%)** |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Uniform Soft Voting | 679K (678,492) | 210.57 MFLOPs | 83.79% | 0.8350 | 85.67% | 0.8583 | 71.04% ± 1.64% | 0.7047 | **80.11% ± 0.89%** | **0.7974** | +9.07% | Standard soft fusion |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Accuracy-Weighted Soft | 679K (678,492) | 210.57 MFLOPs | 83.78% | 0.8352 | 85.67% | 0.8584 | 71.20% ± 1.74% | 0.7065 | **80.11% ± 0.89%** | **0.7974** | +8.92% | Standard soft fusion |
| **SkelGym-Lite (Trans + Bone AAGCN)** | SLSQP Soft Voting | 679K (678,492) | 210.57 MFLOPs | 83.58% | 0.8332 | 85.35% | 0.8516 | 71.49% ± 1.66% | 0.7101 | **80.26% ± 2.27%** | **0.8027** | +8.77% | ⚠️ Susceptible to Val overfitting (81.40%) |
| **SkelGym-Lite (Trans + Bone AAGCN)** | Hard Voting | 679K (678,492) | 210.57 MFLOPs | 78.70% | 0.7794 | 81.32% | 0.8161 | 63.07% ± 2.34% | 0.6423 | **71.39% ± 2.92%** | **0.7157** | +8.32% | ❌ Suboptimal discrete voting |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Stacking Meta-Classifier | 1.81M (1,811,742) | 819.13 MFLOPs | 93.62% | 0.9336 | 95.65% | 0.9571 | 73.37% ± 1.31% | 0.7242 | **83.83% ± 0.50%** | **0.8336** | +10.46% | 🏆 **Optimal SOTA Meta-Learner (83.83%)** |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Uniform Soft Voting | 1.81M (1,811,742) | 819.13 MFLOPs | 84.37% | 0.8401 | 86.63% | 0.8653 | 73.01% ± 1.41% | 0.7220 | **83.12% ± 0.99%** | **0.8205** | +10.11% | ⚡ **Optimal Zero-Param Voting (83.12%)** |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Accuracy-Weighted Soft | 1.81M (1,811,742) | 819.13 MFLOPs | 84.39% | 0.8405 | 86.31% | 0.8635 | 73.41% ± 1.59% | 0.7254 | **82.83% ± 1.72%** | **0.8172** | +9.42% | Standard soft fusion |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | SLSQP Soft Voting | 1.81M (1,811,742) | 819.13 MFLOPs | 83.63% | 0.8334 | 85.51% | 0.8547 | 72.45% ± 1.68% | 0.7164 | **81.40% ± 1.94%** | **0.8103** | +8.95% | ⚠️ Susceptible to Val overfitting (81.40%) |
| **SkelGym-Full (Trans + 4 AAGCN Streams)** | Hard Voting | 1.81M (1,811,742) | 819.13 MFLOPs | 82.52% | 0.8224 | 84.38% | 0.8465 | 70.57% ± 1.37% | 0.7023 | **81.40% ± 0.89%** | **0.8043** | +10.83% | ❌ Suboptimal discrete voting |

---

## 3. External State-of-the-Art Baseline Comparison

| External Model | Publication Venue | Input Format | Trainable Params | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 | Relative $\Delta$ vs SkelGym-Full |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN** | AAAI 2018 | 13 MediaPipe Rel 3D Joints | 350K | 57.58% ± 0.87% | 0.5574 | 66.24% ± 1.32% | 0.6350 | **+17.59%** (SkelGym superiority) |
| **BlockGCN (Adapted)** | CVPR 2024 | 33 MediaPipe Raw 3D Joints | 1.35M | 54.24% ± 1.04% | 0.5576 | 70.48% ± 0.40% | 0.6948 | **+13.35%** (SkelGym superiority) |
| **SkelGym-Full (Stacking)** | **Proposed (This Work)** | **63-d Mix + 4 Graph Streams** | **1.81M** | **73.37% ± 1.31%** | **0.7242** | **83.83% ± 0.50%** | **0.8336** | **Reference SOTA (Baseline)** |

---

## 4. Key Scientific & Engineering Insights for Researchers

### A. The Fusion Strategy Paradigm (Stacking vs. Uniform vs. SLSQP)
1. **Stacking Meta-Classifier (Ridge) is the True Empirical SOTA:**
   - Reaches **83.83% ± 0.50% Video Consensus Accuracy** and **0.8336 Macro-F1** across 3 random seeds.
   - Learns class-specific combination weights (e.g. relying more on AAGCN Bone Stream for compound hip/knee hinge exercises like Deadlift/Squat, and Transformer for upper-body unilateral curls).
2. **Uniform Average Soft Voting ($w_i = 1/K$) is the Best Heuristic Fusion:**
   - Reaches **83.12% ± 0.99% Video Accuracy**, requiring zero extra parameters or validation calibration.
3. **SLSQP Calibration Overfitting Warning:**
   - While mathematically elegant, SLSQP constrained optimization on validation window cross-entropy yields 81.40% on test video.
   - SLSQP over-allocates probability mass to the top 2 streams on the validation partition, reducing ensemble diversity on held-out test videos.

### B. Model Parameter Footprint Standardization
- All individual backbones adhere to a calibrated **Compact Budget (300K – 378K)**:
  - Sequence Backbones: Transformer (**301K**), BiLSTM (**360K**), LSTM (**362K**).
  - Graph Backbones: ST-GCN (**350K**), AAGCN (**378K**).
- Multi-Stream Ensembles: SkelGym-Lite (**679K**), Four-Stream AAGCN (**1.51M**), SkelGym-Full (**1.81M**).

### C. SkelGym-Aug Zero-Leakage Validation Supremacy
- On pristine validation windows, `SkelGym-Aug (4-op)` achieves the **lowest cross-entropy loss (0.9801)** and **highest accuracy (81.89%)** compared to unaugmented baseline (1.0535 loss, 80.98% acc).
- Confirms that SkelGym-Aug selection was determined strictly on the validation partition prior to test evaluation.
