# SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification

Official PyTorch implementation, pre-trained model checkpoints, and dataset artifacts for the paper:
> **"SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification"**  
> *Target Journal: Expert Systems with Applications (Elsevier)*  
> *Open Science Hub: [huggingface.co/Cuong2004/gym-exercise-classification](https://huggingface.co/Cuong2004/gym-exercise-classification)*

---

## 📌 Executive Summary

**SkelGym** is an end-to-end, privacy-preserving, and computationally lightweight intelligent exercise recognition framework operating exclusively on 3D skeletal coordinates extracted from monocular video via Google MediaPipe Pose Heavy. 

By abstracting raw video frames into sparse 3D joint trajectories on edge hardware, SkelGym eliminates the heavy computational footprint of 3D-CNNs/Video Transformers, prevents background/apparel memorization, and mitigates privacy risks by processing video frames ephemerally without persistent video streaming or storage.

```
                                  SKELGYM END-TO-END PIPELINE
                                         [Video Input]
                                               │
                                 [MediaPipe Pose Heavy (3D)]
                                               │
                                  [13 Primary Joints Pruned]
                                               │
             ┌─────────────────────────────────┴─────────────────────────────────┐
             │ (Sequence Track)                                                  │ (Graph Track)
             ▼                                                                   ▼
   [3D Relative (39-d)] + [Pairwise Angles (78-d)]                     [Graph G(V=13, E=12)]
             │                                                                   │
   [117-d Biomechanical Mix]                                           [4 Streams: Joint / Bone /]
             │                                                         [J-Motion / B-Motion      ]
             ▼                                                                   ▼
   [SkelGym-Aug: Sagittal Flip + Yaw]                                  [SkelGym-Aug: Symmetry + Yaw]
             │                                                                   │
             ▼                                                                   ▼
   [Temporal Transformer (4L, 8H, 399K)]                               [4-Stream AAGCN (Adaptive GCN, 1.5M)]
             │                                                                   │
             └─────────────────────────────────┬─────────────────────────────────┘
                                               ▼
                            [Validation-Calibrated SLSQP Soft Voting]
                                               │
                             ┌─────────────────┴─────────────────┐
                             ▼                                   ▼
                  [Window-Level: 70.11%]               [Video Consensus: 77.68%]
                  (Macro F1: 0.6858)                   (Macro F1: 0.7649)
```

### 🔬 Core Innovations

1. **117-dimensional Biomechanical Compound Representation:** Fuses 3D relative joint coordinates ($39$-d) with pairwise directional elevation angles ($78$-d), reducing dimensionality by $59.1\%$ compared to combinatorial $3$-joint triplets ($\binom{13}{3} = 286$-d) while resolving multicollinearity and retaining scale-invariant directional orientation.
2. **Physiologically Grounded Augmentation (SkelGym-Aug):** Restricts data transformations to physically valid human postures through bilateral sagittal reflection ($\mathbb{Z}_2$ symmetry with exact joint permutations) and gravitational yaw rotation ($\mathrm{SO}(2)$ invariance about the vertical axis). Applied strictly on-the-fly during training forward passes (zero test-time corruption).
3. **Ultra-Lightweight Scratch-Trained Backbones:** All models ($\approx 350\text{K} \pm 15\%$ parameters per stream) are trained entirely from scratch without external pre-training weights, achieving an edge classifier latency of **$1.01\text{ ms}$** on Apple Silicon M4 MPS ($0.54\text{ ms}$ on CUDA, $4.74\text{ ms}$ on CPU).
4. **Validation-Calibrated Ensemble (SLSQP):** Formulates soft voting stream weights via Sequential Least Squares Programming minimizing Negative Log-Likelihood strictly on validation partitions (leakage-free), combined with temporal video consensus aggregation.

---

## 📊 Key Benchmark Results

Evaluated on **$2,743$ held-out test windows** across **$233$ out-of-sample test videos** under a strict **Video-Level Partition (6:2:2)** with zero temporal leakage:

| Model / Ensemble Architecture | Parameter Footprint | Window Accuracy | Window Macro F1 | Video Consensus Acc | Video Macro F1 | Apple M4 Latency (MPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline LSTM (Mix 117-d)** | 378K | 57.67% | 0.5681 | 67.81% | 0.6618 | 0.40 ms |
| **Baseline BiLSTM (Mix 117-d)** | 367K | 61.17% | 0.6008 | 68.24% | 0.6802 | 0.42 ms |
| **Baseline ST-GCN (Rel 3D)** | 365K | 43.57% | 0.4595 | 58.47% | 0.5817 | 0.22 ms |
| **Transformer Mix (SkelGym-Aug)** | 399K | 68.65% | 0.6654 | 72.10% | 0.6983 | 0.14 ms |
| **AAGCN Bone Stream (SkelGym-Aug)** | 378K | 65.84% | 0.6504 | 72.96% | 0.7295 | 0.20 ms |
| **Four-Stream AAGCN (Unified)** | 1.51M | 67.34% | 0.6662 | 75.54% | 0.7500 | 0.87 ms |
| **SkelGym-Lite (Transformer + Bone)** | **777K** | **69.27%** | **0.6771** | **76.82%** | **0.7550** | **0.34 ms (2,941 FPS)** |
| **SkelGym-Full (Transformer + 4S-AAGCN)** | **1.91M** | **70.11%** | **0.6858** | **77.68%** | **0.7649** | **1.01 ms (990 FPS)** |

* **Statistical Significance:** Confirmed via McNemar's test ($p < 10^{-11}$) and Wilcoxon signed-rank test ($p < 0.005$).
* **Bootstrap Reliability:** Non-parametric bootstrap ($B=1,000$) establishes a 95% Confidence Interval for Video Accuracy of **$[74.68\%, 84.98\%]$** (mean: $79.83\% \pm 2.65\%$).
* **External Benchmark:** Resolves the "Deyzel Dilemma" (Squat vs. Deadlift ambiguity from Deyzel et al., CVPRW 2023), elevating Deadlift recall from $30.0\%$ to **$90.0\%$**, and achieving **$97.32\%$** 1-shot transfer accuracy.

---

## 📁 Repository Structure

```
Gym_Classification/
├── paper/                             # Manuscript source files
│   ├── paper_eswa.tex                 # Elsevier ESWA version (elsarticle format)
│   ├── paper_eswa_overleaf.zip        # Ready-to-upload Overleaf package (ESWA)
│   ├── paper_llncs.tex                # Springer LNCS version
│   ├── paper_llncs.pdf                # Compiled PDF (LNCS)
│   ├── paper_llncs_overleaf.zip       # Ready-to-upload Overleaf package (LNCS)
│   ├── cover_letter.tex / .pdf        # Submission cover letter to ESWA Editor-in-Chief
│   ├── elsarticle.cls / .bst          # Elsevier class and bibliography styles
│   └── images/                        # High-resolution architectural figures & confusion matrix
├── configs/                           # Experiment YAML configuration files
│   ├── default.yaml
│   └── experiments/                   # Table-specific reproducibility presets
├── src/                               # Core Python library
│   ├── constants.py                   # 22 classes, 13 joints, skeleton graph, feature mappings
│   ├── cli.py                         # CLI controller
│   ├── data/
│   │   ├── dataset.py                 # Sliding window loader (T=32, S=16 train, S=32 val/test)
│   │   ├── features.py                # 117-d Biomechanical Mix, Relative 3D, Elevation Angles
│   │   ├── augmentations.py           # SkelGym-Aug: Sagittal reflection, Yaw rotation, Warping
│   │   └── extractor.py               # MediaPipe Pose Heavy landmark extraction pipeline
│   ├── models/
│   │   ├── transformer.py             # Temporal Transformer encoder with sinusoidal PE
│   │   ├── stgcn.py                   # Spatial Temporal Graph Convolutional Network
│   │   ├── aagcn.py                   # Multi-Stream Adaptive Graph Convolutional Network
│   │   └── ensemble.py                # SLSQP validation-calibrated soft voting & video consensus
│   ├── training/
│   │   ├── trainer.py                 # Training loops, Cosine Annealing, AMP, Early Stopping
│   │   └── metrics.py                 # Macro F1, Accuracy, Confusion Matrix, LaTeX export
│   └── utils/
│       ├── reproducibility.py         # Deterministic random seeding (seed = 42)
│       └── config.py                  # YAML config reader
├── scripts/
│   ├── evaluate_local_ensemble.py     # Re-evaluates local checkpoints (Tables 1-7)
│   └── compute_statistical_tests.py   # Computes McNemar, Wilcoxon, and Bootstrap CI (Tables 8-9)
├── tests/
│   └── test_pipeline.py               # Automated pipeline unit tests (7/7 passing)
├── Final_dataset_metadata.csv         # Complete video metadata (1,024 videos, 1,108 segments)
├── submission_checklist.md            # Comprehensive publication checklist
└── run.py                             # Main CLI execution entrypoint
```

---

## 🛠️ Quickstart & Reproducibility

### 1. Installation
```bash
git clone https://huggingface.co/Cuong2004/gym-exercise-classification
cd gym-exercise-classification
pip install -r requirements.txt
```

### 2. Deterministic Result Verification
Pre-trained model weights for all backbones (Transformer Mix, ST-GCN, 4-Stream AAGCN) are available on Hugging Face Model Hub. You can verify the reported tables without re-training:

```bash
# Evaluate local or downloaded ensemble checkpoints (Tables 1-7)
python scripts/evaluate_local_ensemble.py

# Run statistical hypothesis testing & bootstrap confidence intervals (Tables 8-9)
python scripts/compute_statistical_tests.py
```

### 3. Training Backbones from Scratch
```bash
# Train Transformer with 117-d Biomechanical Mix + SkelGym-Aug
python run.py train --model Transformer --feature mix_117 --augment skelgym_aug --epochs 100 --batch_size 16 --lr 1e-4

# Train 4-Stream AAGCN Bone Model
python run.py train --model AAGCN --stream bone --augment skelgym_aug --epochs 100 --batch_size 32 --lr 1e-3

# Run SLSQP validation calibration across trained checkpoints
python run.py ensemble --method slsqp --val_calibration
```

---

## 📜 Dataset & Ethical Compliance

* **Corpus Scale:** 1,024 unique video recordings ($\approx 10.2\text{ GB}$), trimmed into 1,108 active exercise segments across 22 resistance training classes.
* **Provenance:** 652 videos from Abdillah (2023), 244 videos curated from open stock media (YouTube, Pexels, Freepik), and 128 multi-angle gym videos recorded by the authors with informed consent under the Declaration of Helsinki.
* **Licensing:**
  * Landmark coordinate tensors (NPY, JSON, HDF5), action boundaries, and metadata manifests: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
  * Source code, training scripts, and configurations: **MIT License**.

---

## 📖 Citation

```bibtex
@article{nguyen2026skelgym,
  title={SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification},
  author={Nguyen, Xuan-Cuong and Truong, Nhat-Quang and Vo, Quoc-Trinh},
  journal={Expert Systems with Applications},
  year={2026},
  publisher={Elsevier}
}
```
