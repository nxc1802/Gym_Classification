# Systematic SkelGym-Augmentation Ablation Benchmark

Evaluated across 3 independent random seeds: `[42, 123, 3407]`.

## 1. Leave-One-Out Ablation on Representative Sequence Transformer (Mix 117-d)

| Configuration | Seed 42 | Seed 123 | Seed 3407 | Mean ± Std (Window Acc) | Mean ± Std (Window Macro-F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full SkelGym-Aug** | 68.65% | 65.29% | 65.29% | **66.41% ± 1.58%** | **0.6531 ± 0.0087** |
| **w/o Bilateral Mirror (-Mirror)** | 64.71% | 64.71% | 64.71% | **64.71% ± 0.00%** | **0.6387 ± 0.0000** |
| **w/o 3D Yaw Rotation (-Yaw)** | 59.24% | 64.13% | 65.44% | **62.94% ± 2.67%** | **0.6249 ± 0.0296** |
| **w/o Spatial Scale (-Scale)** | 64.60% | 65.26% | 65.26% | **65.04% ± 0.31%** | **0.6475 ± 0.0043** |
| **w/o Temporal TimeWarp (-TimeWarp)** | 69.05% | 67.70% | 66.72% | **67.82% ± 0.96%** | **0.6696 ± 0.0098** |
| **w/o Sensor Jitter (-Jitter)** | 63.40% | 65.62% | 65.62% | **64.88% ± 1.05%** | **0.6482 ± 0.0158** |
| **Clean Baseline (No Augmentation)** | 63.40% | 63.32% | 63.32% | **63.35% ± 0.04%** | **0.6121 ± 0.0069** |

## 2. Cross-Backbone Generalization (No-Aug vs SkelGym-Aug)

| Architecture / Stream | Unaugmented (Mean ± Std) | SkelGym-Aug (Mean ± Std) | Absolute Gain (Δ Acc) | Δ Macro F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer (Mix 117d)** | 63.35% ± 0.04% (F1: 0.6121) | 66.41% ± 1.58% (F1: 0.6531) | **+3.06%** | **+0.0410** |
| **AAGCN (Bone 3D)** | 58.86% ± 4.66% (F1: 0.5814) | 66.81% ± 0.69% (F1: 0.6605) | **+7.95%** | **+0.0791** |
| **BiLSTM (Mix 117d)** | 59.00% ± 0.60% (F1: 0.5779) | 64.03% ± 0.37% (F1: 0.6278) | **+5.03%** | **+0.0499** |
| **ST-GCN (Rel 3D)** | 48.01% ± 3.96% (F1: 0.4677) | 58.72% ± 0.19% (F1: 0.5599) | **+10.71%** | **+0.0923** |
