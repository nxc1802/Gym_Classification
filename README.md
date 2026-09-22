# SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification

Official PyTorch implementation, pre-trained model checkpoints, and dataset artifacts for the paper:
> **"SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification"**  
> *Target Journal: Expert Systems with Applications (Elsevier)*  
> *Open Science Hub: [huggingface.co/Cuong2004/gym-exercise-classification](https://huggingface.co/Cuong2004/gym-exercise-classification)*

---

## 📌 Executive Summary

**SkelGym** is an end-to-end, privacy-aware, and computationally lightweight intelligent exercise recognition framework operating exclusively on 3D skeletal coordinates extracted from monocular video via Google MediaPipe Pose Heavy. 

By abstracting raw video frames into sparse 3D joint trajectories on edge hardware, SkelGym eliminates the heavy computational footprint of 3D-CNNs/Video Transformers, reduces reliance on background and athletic apparel appearance cues, and mitigates privacy exposure by processing video frames ephemerally in volatile memory without persistent video streaming or disk storage.

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
             [Window-Level: 70.03% ± 0.70%]      [Video Consensus: 77.68% ± 0.86%]
               (Macro F1: 0.6877 ± 0.0051)         (Macro F1: 0.7689 ± 0.0090)
```

### 🔬 Core Innovations

1. **117-dimensional Biomechanical Compound Representation:** Fuses 3D relative joint coordinates ($39$-d) with pairwise directional elevation angles ($78$-d), reducing dimensionality by $59.1\%$ compared to combinatorial $3$-joint triplets ($\binom{13}{3} = 286$-d) while resolving multicollinearity and retaining scale-invariant directional orientation.
2. **Physiologically Grounded Augmentation (SkelGym-Aug):** Restricts data transformations to physically valid human postures through bilateral sagittal reflection ($\mathbb{Z}_2$ symmetry with exact joint permutations) and gravitational yaw rotation ($\mathrm{SO}(2)$ invariance about the vertical axis). Applied strictly on-the-fly during training forward passes (zero test-time corruption).
3. **Ultra-Lightweight Scratch-Trained Backbones:** All models ($\approx 350\text{K} \pm 15\%$ parameters per stream) are trained entirely from scratch without external pre-training weights, achieving an edge classifier latency of **$0.42\text{--}4.33\text{ ms}$** on host CPU ($0.08\text{--}0.54\text{ ms}$ on CUDA, $1.11\text{--}8.77\text{ ms}$ on MPS; where individual backbones require $0.42\text{--}1.16\text{ ms}$ on CPU and the full 5-stream ensemble executes in $4.33\text{ ms}$).
4. **Validation-Calibrated Ensemble (SLSQP):** Formulates soft voting stream weights via Sequential Least Squares Programming minimizing Negative Log-Likelihood strictly on validation partitions (leakage-free), combined with temporal video consensus aggregation.

---

## 📊 Key Benchmark Results

Evaluated on **$2,743$ held-out test windows** across **$233$ out-of-sample test videos** under a strict **Video-Level Partition (6:2:2)** with no source-video overlap across splits:

| Model / Ensemble Architecture | Parameter Footprint | FLOPs / MACs | Window Accuracy | Window Macro F1 | Video Consensus Acc | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN Baseline** (Rel 3D) | 378K | 101.43 MFLOPs | 43.57% | 0.4595 | 58.47% | 0.5817 |
| **LSTM Baseline** (Mix 117-d) | 362K | 7.42 MFLOPs | 57.67% | 0.5681 | 67.81% | 0.6618 |
| **BiLSTM Baseline** (Mix 117-d) | 384K | 14.84 MFLOPs | 61.17% | 0.6008 | 68.24% | 0.6802 |
| **AAGCN Baseline** (Bone 3D) | 378K | 101.43 MFLOPs | 52.27% | 0.5338 | 69.53% | 0.6904 |
| **SkelGym-Aug AAGCN** (Bone 3D) | 378K | 101.43 MFLOPs | 65.84% | 0.6504 | 72.96% | 0.7295 |
| **Transformer Mix** (Clean) | 399K | 12.50 MFLOPs | 63.40% | 0.6218 | 74.25% | 0.7304 |
| **SkelGym-Aug Transformer** (Mix) | 399K | 12.50 MFLOPs | 66.39% $\pm$ 1.94% | 0.6492 $\pm$ 0.0137 | 72.39% $\pm$ 2.59% | 0.7089 $\pm$ 0.0211 |
| **Two-Stream AAGCN** (Aug) | 756K | 202.86 MFLOPs | 66.61% | 0.6586 | 73.82% | 0.7348 |
| **Four-Stream AAGCN** (Aug) | 1.51M | 405.71 MFLOPs | 68.78% $\pm$ 1.35% | 0.6782 $\pm$ 0.0106 | 77.25% $\pm$ 1.55% | 0.7678 $\pm$ 0.0155 |
| **SkelGym-Lite** (Transformer + Bone) | **777K** | **113.93 MFLOPs** | **69.66% $\pm$ 0.71%** | **0.6852 $\pm$ 0.0087** | **76.68% $\pm$ 0.25%** | **0.7582 $\pm$ 0.0036** |
| **SkelGym-Full** (Cross-Paradigm Ensemble) | **1.91M** | **418.21 MFLOPs** | **70.03% $\pm$ 0.70%** | **0.6877 $\pm$ 0.0051** | **77.68% $\pm$ 0.86%** | **0.7689 $\pm$ 0.0090** |

*Note:* For the four final configurations, metrics are reported as $\text{Mean} \pm \text{SD}$ across 3 independent random seeds (42, 123, 3407) under fixed splits and identical protocols.

* **Training Stability Across 3 Seeds:** Across three independent random seeds (42, 123, 3407), SkelGym-Full and SkelGym-Lite exhibited relatively low variation across the three runs (SkelGym-Full: Video Acc $77.68\% \pm 0.86\%$, Macro F1 $0.7689 \pm 0.0090$; SkelGym-Lite: $76.68\% \pm 0.25\%$, Macro F1 $0.7582 \pm 0.0036$), while standalone backbones showed higher run-to-run sensitivity (Transformer Mix: $\pm 2.59\%$), illustrating the variance-dampening advantage of cross-paradigm late fusion.
* **Statistical Significance:** Selected model comparisons were evaluated using paired statistical tests: McNemar's test at window level ($N=2,743, p < 10^{-11}$) and Wilcoxon signed-rank test at video level ($N=233, p < 0.005$) against a Bonferroni-adjusted threshold $\alpha_{\text{adj}} = 0.01$.
* **Bootstrap Reliability:** Non-parametric bootstrap ($B=1,000$, computed on baseline seed-42 test predictions) provides 95% confidence intervals: Window Accuracy **$[68.46\%, 71.75\%]$** (mean: $70.14\% \pm 0.84\%$) and Video Accuracy **$[72.09\%, 83.26\%]$** (mean: $77.65\% \pm 2.85\%$). Both point estimates lie centrally inside their respective 95% CIs.
* **External Benchmark:** Mitigates the Squat--Deadlift confusion observed in the Deyzel et al. (CVPRW 2023) benchmark, elevating Deadlift video recall from **$40.0\%$** on baseline ST-GCN to **$80.0\%$** on Transformer Mix and **$90.0\%$** on SkelGym-Full (under closed-set consensus; 80.0% under open-set), and achieving a mean of **$90.36\% \pm 7.35\%$** (peak trial: **$98.00\%$**, 95% CI: $[68.95\%, 98.00\%]$) in 1-shot classification simulations across 100 trials.

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
│   ├── evaluate_external_benchmark.py # External Deyzel S&C benchmark & 1-shot simulation (Table 8)
│   ├── benchmark_hardware_latency.py  # Standardized latency (Mean/Median/p95) & FLOPs complexity
│   └── compute_statistical_tests.py   # Computes McNemar, Wilcoxon, and Bootstrap CI (Tables 11-12)
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
# 1. Evaluate local or downloaded ensemble checkpoints (Tables 1-7)
python scripts/evaluate_local_ensemble.py

# 2. Evaluate external S&C benchmark and 1-shot classification simulation (Table 8)
python scripts/evaluate_external_benchmark.py

# 3. Standardized hardware latency and theoretical FLOPs/MACs benchmark (Table 12)
python scripts/benchmark_hardware_latency.py --device auto

# 4. Run statistical hypothesis testing & bootstrap confidence intervals (Table 11)
python scripts/compute_statistical_tests.py

# 5. Run multi-seed evaluation across seeds 42, 123, 3407 (Training Stability)
python scripts/run_multi_seed_experiments.py --skip_train
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

## 📜 Dataset, Licensing & Ethical Compliance

* **Corpus Scale:** 1,024 unique video recordings ($\approx 10.2\text{ GB}$), trimmed into 1,108 active exercise segments across 22 resistance training classes.
* **Provenance:** 652 videos from Abdillah (2023), 244 videos curated from open stock media (YouTube, Pexels, Freepik), and 128 multi-angle gym videos recorded by the authors with informed consent under the Declaration of Helsinki.
* **Data Licensing & Artifact Availability:**
  * **Raw Videos:** Raw third-party videos are not redistributed. Their original hosting locations, attribution information, and segmentation boundaries are retained in the metadata manifests where applicable.
  * **Extracted Skeletal Representations (NPY, CSV), Segment Boundaries, and Metadata:** Publicly available under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
  * **Source Code, Training Scripts, and Checkpoints:** Fully open-sourced under the **MIT License**.
  * All artifacts are hosted at: [huggingface.co/Cuong2004/gym-exercise-classification](https://huggingface.co/Cuong2004/gym-exercise-classification).

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
