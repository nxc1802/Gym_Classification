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
   [Temporal Transformer (3L, 8H, 399K)]                               [4-Stream AAGCN (Adaptive GCN, 1.5M)]
             │                                                                   │
             └─────────────────────────────────┬─────────────────────────────────┘
                                               ▼
                            [Validation-Calibrated SLSQP Soft Voting]
                                               │
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
              [Window-Level: 69.73% ± 1.10%]      [Video Consensus: 78.83% ± 0.66%]
                (Macro F1: 0.6881 ± 0.0072)         (Macro F1: 0.7807 ± 0.0126)
```

### 🔬 Core Innovations

1. **Curated Multi-Source Landmark Benchmark:** Curates and releases 1,024 multi-camera in-the-wild videos spanning 22 resistance exercises across three provenance streams with strict 6:2:2 video-level partitioning (zero source-video overlap), eliminating temporal data leakage prevalent in prior benchmarks.
2. **117-dimensional Biomechanical Compound Representation:** Fuses 3D relative joint coordinates ($39$-d) with pairwise directional elevation angles ($78$-d), reducing dimensionality by $59.1\%$ compared to combinatorial $3$-joint triplets ($\binom{13}{3} = 286$-d) while resolving multicollinearity, eliminating gradient singularities at joint lockouts, and retaining scale-invariant directional orientation.
3. **Empirically Selected Augmentation Configuration (SkelGym-Aug):** Systematically investigates a candidate pool of five established skeletal augmentation operators (bilateral sagittal reflection, gravitational yaw rotation, proportional scaling, temporal time warping, and sensor jitter). Through multi-seed Leave-One-Out (LOO) ablation, we demonstrate that temporal warping disrupts exercise cadence and degrades accuracy, leading to an empirically selected 4-operator configuration that boosts generalization while preserving exercise biomechanics. Applied strictly on-the-fly during training forward passes (zero test-time corruption).
4. **Cross-Paradigm Fusion, Rigorous Benchmarking, and Real-Time Edge Deployment:** Couples custom lightweight backbones ($\approx 350\text{K}$ parameters per stream, trained from scratch) spanning self-attention Transformers and 4-Stream AAGCNs via validation-calibrated SLSQP soft voting, achieving **$69.73\% \pm 1.10\%$** window accuracy, **$78.83\% \pm 0.66\%$** video consensus accuracy (Video Macro F1: **$0.7807 \pm 0.0126$**, Window Macro F1: **$0.6881 \pm 0.0072$**), and ultra-low edge latency (**$0.42\text{--}4.33\text{ ms}$** on host CPU, **$0.08\text{--}0.54\text{ ms}$** on CUDA) for privacy-aware deployment.

---

## 📊 Key Benchmark Results

Evaluated on **$2,743$ held-out test windows** across **$233$ out-of-sample test videos** under a strict **Video-Level Partition (6:2:2)** with no source-video overlap across splits.

> **Authoritative Benchmark Source of Truth (SOT):** All empirical metrics, ablation sweeps, multi-seed downstream benchmarks, statistical hypothesis tests, and hardware latencies are centralized and actively maintained in [`outputs/RESULTS_FINAL.md`](outputs/RESULTS_FINAL.md).

| Model / Ensemble Architecture | Parameter Footprint | FLOPs / MACs | Window Accuracy | Window Macro F1 | Video Consensus Acc | Video Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ST-GCN Baseline** (Rel 3D) | 350K | 101.43 MFLOPs | 54.76% (n=1) | 0.5240 | 63.09% (n=1) | 0.5993 |
| **LSTM Baseline** (Mix 117-d) | 396K | 7.42 MFLOPs | 57.64% (n=1) | 0.5676 | 67.38% (n=1) | 0.6580 |
| **BiLSTM Baseline** (Mix 117-d) | 402K | 14.84 MFLOPs | 61.25% (n=1) | 0.6015 | 68.67% (n=1) | 0.6830 |
| **AAGCN Baseline** (Bone 3D) | 378K | 101.43 MFLOPs | 59.50% (n=1) | 0.5960 | 69.53% (n=1) | 0.6904 |
| **SkelGym-Aug AAGCN** (Bone 3D) | 378K | 101.43 MFLOPs | 66.59% $\pm$ 1.43% | 0.6587 $\pm$ 0.0181 | 75.82% $\pm$ 2.62% | 0.7543 $\pm$ 0.0305 |
| **Transformer Mix** (Clean) | 400K | 12.50 MFLOPs | 63.40% (n=1) | 0.6218 | 74.25% (n=1) | 0.7304 |
| **SkelGym-Aug Transformer** (Mix) | 400K | 12.50 MFLOPs | 66.27% $\pm$ 2.75% | 0.6544 $\pm$ 0.0264 | 75.54% $\pm$ 3.81% | 0.7430 ± 0.0430 |
| **Two-Stream AAGCN** (Aug) | 756K | 202.86 MFLOPs | 67.26% $\pm$ 1.41% | 0.6652 $\pm$ 0.0198 | 77.11% $\pm$ 3.33% | 0.7580 $\pm$ 0.0432 |
| **Four-Stream AAGCN** (Aug) | 1.51M | 405.71 MFLOPs | 67.61% $\pm$ 1.46% | 0.6688 $\pm$ 0.0205 | 77.54% $\pm$ 3.33% | 0.7632 $\pm$ 0.0387 |
| **SkelGym-Lite** (Transformer + Bone) | **778K** | **113.93 MFLOPs** | **68.26% $\pm$ 0.66%** | **0.6732 $\pm$ 0.0067** | **77.68% $\pm$ 1.55%** | **0.7694 $\pm$ 0.0103** |
| **SkelGym-Full** (Cross-Paradigm Ensemble) | **1.91M** | **418.21 MFLOPs** | **69.73% $\pm$ 1.10%** | **0.6881 $\pm$ 0.0072** | **78.83% $\pm$ 0.66%** | **0.7807 $\pm$ 0.0126** |

*Note:* For multi-seed configurations, metrics are reported as $\text{Mean} \pm \text{SD}$ across 3 independent random seeds (42, 123, 3407) under fixed splits and identical protocols. Single-seed baselines are denoted as $(n=1)$.

* **Training Stability Across 3 Seeds:** Across three independent random seeds (42, 123, 3407), SkelGym-Full exhibited low variation across the three runs (Video Acc $78.83\% \pm 0.66\%$, constituent runs: 78.97%, 78.11%, 79.40%; Macro F1 $0.7807 \pm 0.0126$; SkelGym-Lite: $77.68\% \pm 1.55\%$, Macro F1 $0.7694 \pm 0.0103$), while standalone backbones showed higher run-to-run sensitivity (Transformer Mix: $\pm 3.81\%$), illustrating the variance-dampening advantage of cross-paradigm late fusion.
* **Statistical Significance:** Selected model comparisons were evaluated using paired statistical tests: McNemar's test at window level ($N=2,743, p < 10^{-14}$) and Wilcoxon signed-rank test at video level ($N=233, p < 0.005$) with Holm-Bonferroni correction.
* **Bootstrap Reliability:** Non-parametric cluster bootstrap ($B=1,000$, computed on baseline seed-42 test predictions) provides 95% confidence intervals: Window Accuracy **$[63.82\%, 75.83\%]$** (mean: $70.07\%$) and Video Accuracy **$[73.82\%, 84.12\%]$** (mean: $79.02\%$). Both point estimates lie centrally inside their respective 95% CIs.
* **Comparative Transfer on External MM-Fit Benchmark:** Evaluated against the genuine MM-Fit external benchmark (MediaPipe Pose protocol across 5 unseen workout recordings: `w00`, `w05`, `w12`, `w13`, `w20`; $N=54$ recordings, $N=878$ windows) across the 4 overlapping exercises (*squat*, *deadlift*, *barbell biceps curl*, *lateral raise*), SkelGym-Full achieves **95.90%** closed-set window accuracy, **100.00%** video consensus accuracy (Macro F1: **1.0000**), and **74.07%** open-set video consensus accuracy.

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
│   │   ├── transformer.py             # Temporal Transformer encoder with learnable PE
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
│   ├── evaluate_external_benchmark.py # S&C subset benchmark (Deyzel et al. protocol) & 1-shot simulation (Table 8)
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

### 1. Installation & Artifact Bootstrap
```bash
# 1. Clone GitHub Source Repository
git clone https://github.com/nxc1802/Gym_Classification.git
cd Gym_Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Bootstrap datasets, metadata, and checkpoints
python scripts/bootstrap_artifacts.py
```

### 2. Deterministic Result Verification
Pre-trained model weights for all backbones (Transformer Mix, ST-GCN, 4-Stream AAGCN) are available on Hugging Face Model Hub (or locally via `python scripts/bootstrap_artifacts.py --from-archive`). You can verify the reported tables without re-training:

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

# 6. Run augmentation ablation studies (LOO & Single Component: Tables 2 & 3)
python scripts/run_augmentation_experiments.py --mode all --device mps
```

### 3. Training Backbones from Scratch
```bash
# Train Transformer with 117-d Biomechanical Mix + SkelGym-Aug
python run.py train --model Transformer --feature mix --augment skel_gym_aug --epochs 100 --batch_size 16 --lr 1e-4

# Train 4-Stream AAGCN Bone Model
python run.py train --model AAGCN --feature bone_3d --augment skel_gym_aug --epochs 100 --batch_size 32 --lr 1e-3

# Run SLSQP validation-calibrated soft voting ensemble across trained checkpoints
python run.py ensemble --method weighted_soft --video_level \
  --checkpoints checkpoints/best_Transformer_T2.2_mix.pt \
                checkpoints/best_AAGCN_T4.2_bone_3d.pt \
                checkpoints/best_AAGCN_T4.3_rel_3d.pt \
                checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt \
                checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt
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
