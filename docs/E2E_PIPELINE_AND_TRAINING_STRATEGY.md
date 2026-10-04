# SkelGym: End-to-End Pipeline Execution & Training Strategy Documentation
**Authoritative Protocol, Architectural Calibration, and Scientific Execution Guide**

> **SELF-CONTAINED EXECUTION SPEC:** A successor AI agent can open this file as the *sole* context and execute the entire benchmark end-to-end without any additional information from the user. Every command, credential, strategy, and monitoring rule is documented below.

---

## 0. Remote Server Access & Credentials

### 0.1 Server Endpoint

| Field | Value |
|---|---|
| **Marimo Server URL** | `https://sb-3be79594b2b9bd02.sb.molab.run/` |
| **API Token** | `fc65dba73c0c4b07bf79df345eee08db64262ff1fb6e8fc5810f6c95103ce312` |
| **Working Directory** | `/marimo/Gym_Classification` |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition (102 GB VRAM) |
| **HF Repository** | `Cuong2004/gym-exercise-classification` |
| **HF Token env var** | `HF_TOKEN` — must be set in environment before running |

### 0.2 Execute-Code Helper

All code is executed remotely via the `execute-code.sh` wrapper (in PATH on the server):

```bash
execute-code.sh \
  --url "https://sb-3be79594b2b9bd02.sb.molab.run/" \
  --token "fc65dba73c0c4b07bf79df345eee08db64262ff1fb6e8fc5810f6c95103ce312" \
  --code "<PYTHON_CODE_STRING>"
```

### 0.3 Health Check (run before starting)

```python
# Verify GPU and project root are ready
import subprocess, torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
result = subprocess.run(["git", "log", "--oneline", "-3"], cwd="/marimo/Gym_Classification", capture_output=True, text=True)
print("Recent commits:\n", result.stdout)
result2 = subprocess.run(["ls", "checkpoints/"], cwd="/marimo/Gym_Classification", capture_output=True, text=True)
print("Checkpoints:", result2.stdout or "(empty — clean slate)")
```

### 0.4 Before Running — Pull Latest Code

```python
import subprocess
r = subprocess.run(["git", "pull", "origin", "main"], cwd="/marimo/Gym_Classification", capture_output=True, text=True)
print(r.stdout); print(r.stderr)
```

---

## 1. Executive Summary & Scientific Principles

This document defines the complete End-to-End (E2E) execution pipeline, training strategy, and methodological guardrails for the **SkelGym** resistance exercise classification benchmark. It serves as the primary technical specification for replicating all empirical findings, from raw landmark processing to publication-ready LaTeX tables and Hugging Face Hub artifacts.

### Core Scientific Principles:
1. **Zero Data Leakage (Strict Validation Anchoring):**
   - All feature normalization parameters (μ, σ) are computed exclusively on the **Train** partition.
   - All hyperparameter selections, early stopping checkpoints, and model comparison decisions are anchored strictly on the **Validation** partition (`val_macro_f1` and `val_loss`).
   - The **Held-Out Test** partition is strictly quarantined and evaluated only once per model to assess true generalization.
2. **Controlled-Capacity Architectural Parity:**
   - All sequence backbones are calibrated to a compact, edge-deployable budget (~300K–365K parameters):
     - **Transformer (`mix_v2`, 63-d):** **300,742** parameters (~301K). `d_model=112`, `nhead=4`, `d_ff=168`, 3 encoder layers, dual-branch projection (39→56 coords, 24→56 angles → fused 112-d).
     - **Transformer (`raw_3d`/`rel_3d_norm`, 39-d):** **301,436** parameters (~301K).
     - **ST-GCN (`rel_3d`, 39-d):** **350,083** parameters (~350K).
     - **BiLSTM (`mix_v2`, 63-d):** **360,150** parameters (~360K). `hidden=96`, `layers=2`.
     - **LSTM (`mix_v2`, 63-d):** **361,814** parameters (~361K). `hidden=160`, `layers=2`.
     - **AAGCN (per stream, 39-d):** **378,244** parameters (~378K).
     - **SkelGym-Lite (Transformer + 1 AAGCN stream):** **678,986** parameters (~679K).
     - **SkelGym-Full (Transformer + 4 AAGCN streams):** **1,813,718** parameters (~1.81M).
   - Table 2b compares `raw_3d` vs. `rel_3d_norm` vs. `mix_v2` under **exact parameter parity** (Δ < 0.2%).
3. **Multi-Seed Statistical Replication:**
   - **Seeds:** {42, 123, 3407} — all 3 seeds are run for every core experiment.
   - All final metrics report Mean ± Standard Deviation alongside paired hypothesis tests (McNemar, Wilcoxon, and 1,000 video-clustered bootstrap resamples).
   - Running all 3 seeds eliminates any reviewer accusation of seed cherry-picking.
4. **Reproducibility & Open Science:**
   - Model weights and provenance metadata are uploaded to: [`Cuong2004/gym-exercise-classification`](https://huggingface.co/Cuong2004/gym-exercise-classification)
   - Pre-extracted 3D pose landmarks: [`Cuong2004/gym-exercise-landmarks`](https://huggingface.co/datasets/Cuong2004/gym-exercise-landmarks)

---

## 2. Dataset & Feature Engineering Specifications

### 2.1 Dataset Partitioning

The SkelGym corpus: **1,024 raw video recordings** spanning **22 resistance exercise classes**, trimmed into **1,108 clean action execution segments**.

- **Split Ratio:** Strict 6:2:2 video-level partition with zero source-video overlap:
  - **Train:** 580 videos (56.6%), 639 segments, 13,136 sliding windows (stride S=16, ~50% overlap).
  - **Validation:** 208 videos (20.3%), 210 segments, 2,075 sliding windows (stride S=32, non-overlapping).
  - **Held-Out Test:** 236 videos (23.1%), 259 segments, 2,743 windows (stride S=32, non-overlapping, ≥32 frames).
- **Temporal Windowing:** Sliding window of T=32 frames (~1.07 s at 30 FPS).

### 2.2 63-Dimensional Biomechanical Mix v2 (`mix_v2`)

The `mix_v2` feature set explicitly separates spatial coordinates from anatomical articulation:

1. **Scale-Normalized Relative Coordinates (`rel_3d_norm`, 39-d):** 13 key joints, origin at pelvic midpoint, anisotropic anthropometric scale normalization.
2. **Kinematic Articulation Angles (`angle_kinematic_24`, 24-d):** 10 3D joint articulation angles (flexion/extension) + 14 spatial limb segment elevation angles.

3. **Dual-Branch Transformer Architecture:**
   - Coordinates and Angles have fundamentally disparate physical scales. Independent projection heads are used:
     - Coords: Linear(39→56) + LayerNorm
     - Angles: Linear(24→56) + LayerNorm
     - Fused: concat → 112-d, then 3-layer Transformer Encoder → GlobalAvgPool → Classifier → **300,742 params**

---

## 3. Comprehensive Training Strategy & Hyperparameter Protocols

All training jobs strictly adhere to canonical hyperparameters in `src/constants.py:CANONICAL_EXPERIMENT_REGISTRY`.

### 3.1 Optimizer & Learning Rate Schedule
- **Optimizer:** AdamW — weight decay λ=1e-4, β₁=0.9, β₂=0.999, ε=1e-8.
- **Schedules:**
  - **Transformer:** Peak LR=1e-4. Linear warmup 5 epochs → Cosine Annealing to 1e-6.
  - **AAGCN / ST-GCN / LSTM / BiLSTM:** Initial LR=1e-3 → Cosine Annealing.
- **Gradient Clipping:** Max norm=1.0 across all architectures.
- **Precision:** AMP with `torch.bfloat16` on Blackwell GPU.

### 3.2 Regularization & Loss Calibration
- **Label Smoothing:** Transformer & AAGCN: α=0.05. LSTM, BiLSTM, ST-GCN: α=0.0.
- **Dropout:** Transformer: p=0.20. LSTM/BiLSTM: p=0.30. AAGCN: p=0.20. ST-GCN: p=0.30.
- **Early Stopping:** Metric: `val_macro_f1`. Max epochs=100. Patience=10. Best checkpoint saved as `best_*.pt` with `.provenance.json` sidecar.

### 3.3 Data Augmentation Strategy (SkelGym-Aug)

Stochastic on-the-fly 3D skeletal transformations on training batches:

1. **Bilateral Sagittal Reflection (p=0.50):** Mirror x-coordinates with anatomical keypoint index swapping (left ↔ right).
2. **Gravitational 3D Yaw Rotation (p=0.70):** θ ~ U(-15°, +15°) around vertical Y-axis.
3. **Proportional Spatial Scaling (p=0.70):** Scale factor s ~ U(0.90, 1.10).
4. **Sensor Noise Jitter (p=0.50):** Gaussian N(0, 0.008²) noise added to coordinates.

> **TimeWarp Omitted:** LOO ablation confirmed time-warping degrades performance because execution tempo and concentric/eccentric duration ratios are class-discriminative physiological features.

### 3.4 Parallel Training Strategy

#### Cơ chế hoạt động

Training song song được thực hiện qua `concurrent.futures.ThreadPoolExecutor` kết hợp với `subprocess`. Mỗi "worker" là một Python thread trong main process, nhưng thực sự nó spawn một **process con hoàn toàn độc lập** (`subprocess.run`) để chạy `run.py train`. Đây là thiết kế quan trọng:

```
Main Process (marimo_master_e2e_runner.py)
│
├── Thread 1 (worker 1) ──► subprocess: python run.py train --model LSTM --feature raw_2d --seed 42
├── Thread 2 (worker 2) ──► subprocess: python run.py train --model LSTM --feature rel_2d --seed 42
├── Thread 3 (worker 3) ──► subprocess: python run.py train --model LSTM --feature raw_3d --seed 123
└── Thread 4 (worker 4) ──► subprocess: python run.py train --model BiLSTM --feature mix_v2 --seed 42
         ...
         Khi một subprocess kết thúc, thread đó nhận task mới từ queue
```

#### Tại sao không xung đột GPU?

- Mỗi subprocess là **process Python riêng biệt**, có PyTorch CUDA context riêng.
- NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM) đủ chứa 4 model nhỏ (~300–380K params) cùng lúc với AMP (`bfloat16`). Ước tính VRAM mỗi model: ~200–400 MB ⟹ tổng 4 workers: ~1–2 GB, chỉ ~1–2% VRAM.
- CUDA MPS (Multi-Process Service) không cần cấu hình thêm — PyTorch tự xử lý time-slicing GPU ở mức kernel.

#### Checkpoint deduplication (cache-hit)

Nếu `best_*.pt` **đã tồn tại** và `--force_retrain` **không được đặt**, worker sẽ phát hiện checkpoint hiện có và **bỏ qua** training task đó (emit `MODEL_CACHED` trigger). Điều này cho phép resume sau khi crash:

```python
# Logic trong execute_training_task():
if not force_retrain and ckpt_path.exists():
    emit_trigger("MODEL_CACHED", {...})
    return {"status": "cached"}   # Skip — không train lại
```

> **Quan trọng:** Khi chạy clean-slate toàn bộ, **phải dùng `--force_retrain`** để bắt buộc train lại từ đầu, bất kể checkpoint có tồn tại không.

#### Error Handling khi Worker Thất Bại

Nếu một subprocess fail (returncode != 0), worker đó raise `RuntimeError`. `ThreadPoolExecutor` sẽ:
1. Ghi trigger `MODEL_FAILED` vào `outputs/triggers.jsonl`.
2. Raise exception lên main thread khi `fut.result()` được gọi.
3. **Crash toàn bộ pipeline** (để bảo đảm không có kết quả nào được tính từ partial failure).

```python
# Cách phát hiện worker thất bại trong triggers.jsonl:
# {"trigger": "MODEL_FAILED", "model": "LSTM_T1.3_angle_2d (Seed 42)", "error": "..."}
```

Nếu pipeline crash, có thể restart với `--skip_phase1a` / `--skip_phase1b` tùy theo stage đã hoàn thành, kết hợp **bỏ `--force_retrain`** để tận dụng checkpoint đã lưu.

#### Theo dõi tiến độ từng Worker

```python
import json, subprocess
from pathlib import Path

# Xem 20 trigger events gần nhất (bao gồm MODEL_COMPLETE, MODEL_CACHED, MODEL_FAILED)
log = Path("/marimo/Gym_Classification/outputs/triggers.jsonl")
if log.exists():
    lines = [l for l in log.read_text().strip().split("\n") if l]
    for ev in [json.loads(l) for l in lines[-20:]]:
        icon = {"MODEL_COMPLETE": "✅", "MODEL_CACHED": "⚡", "MODEL_FAILED": "❌",
                "PHASE_COMPLETE": "🏁"}.get(ev["trigger"], "ℹ️")
        print(f"{icon} [{ev['timestamp']}] {ev['trigger']} — {ev.get('model', ev.get('phase', ''))}")

# Đếm checkpoints hiện có
r = subprocess.run(["find", "checkpoints/", "-name", "*.pt"],
                   cwd="/marimo/Gym_Classification", capture_output=True, text=True)
ckpts = [f for f in r.stdout.strip().split("\n") if f]
print(f"\n📦 Checkpoints saved: {len(ckpts)} / 150 expected total")
```

#### Thứ tự Execution thực tế

Tasks được submit theo thứ tự định nghĩa trong `build_phase1a_tasks(seeds)`, nhưng do parallel execution, thứ tự **hoàn thành** là không xác định. Điều này hoàn toàn bình thường — aggregation sau đó sẽ collect đủ từ checkpoint files, không phụ thuộc thứ tự hoàn thành.

#### Cấu hình khuyến nghị

| Scenario | `--workers` | Ghi chú |
|---|---|---|
| **Full pipeline (default)** | `4` | Optimal trên Blackwell 102GB |
| Debug / troubleshoot một model | `1` | Sequential, dễ đọc log |
| Nếu VRAM đầy (unlikely) | `2` | Giảm concurrency |
| Muốn tăng throughput tối đa | `6–8` | Thận trọng — monitor VRAM |


### 3.5 Anti-Idle Keepalive Daemon

The master runner automatically spawns a background daemon thread to prevent sandbox container timeout:

```python
# Runs every 15 seconds in background — DO NOT stop this manually
def keepalive_daemon(stop_event, heartbeat_file, ports=(8080, 2718)):
    while not stop_event.is_set():
        for port in ports:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            except:
                pass
        heartbeat_file.touch()   # outputs/keepalive.heartbeat
        time.sleep(15)
```

- Pings localhost Marimo ports (8080, 2718) every 15 seconds.
- Touches `outputs/keepalive.heartbeat` to signal liveness.
- This is automatically activated when `marimo_master_e2e_runner.py` is launched — no manual action needed.

### 3.6 Monitoring Protocol (Strict 15-Minute Cadence)

> **Rule:** The AI executor (successor agent) **MUST check progress exactly once every 15 minutes**, not more frequently. Do not poll every few minutes.

**How to check progress:**

```python
import json
from pathlib import Path

# 1. Read trigger events
log_path = Path("/marimo/Gym_Classification/outputs/triggers.jsonl")
if log_path.exists():
    lines = log_path.read_text().strip().split("\n")
    last_10 = [json.loads(l) for l in lines[-10:] if l]
    for ev in last_10:
        print(f"[{ev['timestamp']}] {ev['trigger']} — {ev}")

# 2. Count checkpoints
import subprocess
r = subprocess.run(["find", "checkpoints/", "-name", "*.pt", "-type", "f"],
                   cwd="/marimo/Gym_Classification", capture_output=True, text=True)
files = [f for f in r.stdout.strip().split("\n") if f]
print(f"Checkpoints saved: {len(files)}")

# 3. Read heartbeat age
import time, os
hb = "/marimo/Gym_Classification/outputs/keepalive.heartbeat"
if os.path.exists(hb):
    age = time.time() - os.path.getmtime(hb)
    print(f"Last heartbeat: {age:.0f}s ago ({'OK' if age < 30 else 'WARNING — may be stuck'})")
```

**Report format to user every 15 minutes:**
```
⏱ [HH:MM] Progress Report
  ✅ Completed triggers: N events
  📊 Checkpoints saved: K/TOTAL
  💓 Server keepalive: OK (Xs ago)
  ▶ Currently running: <phase name>
  ⏳ ETA: ~X minutes
```

---

## 4. Multi-Stream Late Fusion Strategy

### 4.1 Constituent Models
- **Stream 1 (Temporal Sequence):** Transformer on 63-d `mix_v2` + SkelGym-Aug (~301K params).
- **Stream 2 (Structural Geometry):** AAGCN on 39-d `bone_3d` + SkelGym-Aug (~378K params).
- **Stream 3 (Joint Topology):** AAGCN on 39-d `rel_3d` + SkelGym-Aug (~378K params).
- **Stream 4 (Joint Dynamics):** AAGCN on 39-d `joint_motion_3d` (ΔX_t = X_t − X_{t-1}) + SkelGym-Aug (~378K params).
- **Stream 5 (Bone Dynamics):** AAGCN on 39-d `bone_motion_3d` (ΔB_t = B_t − B_{t-1}) + SkelGym-Aug (~378K params).

### 4.2 Ensembles
- **SkelGym-Lite (2 streams):** Transformer + AAGCN bone → **678,986** params (~679K).
- **SkelGym-Full (5 streams):** All 5 streams → **1,813,718** params (~1.81M).
- **Fusion Calibration:** Stream weights **w** optimized via SLSQP strictly on the validation partition by minimizing NLL under non-negativity and unit-sum constraints.

---

## 5. End-to-End Pipeline Execution Roadmap

The complete empirical suite is automated via `scripts/marimo_master_e2e_runner.py` across 5 sequential stages.

```
Stage 1 → Phase 1A: Feature Screening
           81 models (27 configs × 3 seeds: 42, 123, 3407)
           4 parallel workers, ~10–12 minutes total
           → Aggregate: table1_feature_screening_multiseed.json
           → Subt: Table 2b (9 runs via run_upgrade_mix_benchmark.py)

Stage 2 → Phase 1B: Core Multi-Seed Backbones
           33 models (11 configs × 3 seeds)
           [6 baselines: STGCN raw, STGCN rel, LSTM, BiLSTM, Transformer, AAGCN clean]
           [5 augmented constituents: Transformer+aug, AAGCN×4+aug]
           4 parallel workers, ~4–5 minutes
           → Phase 2: Freeze reference artifacts & SLSQP weights
           → Phase 3: 5 fusion methods evaluation
           → Aggregate: graph_streams_multiseed.json

Stage 3 → Phase 1C: Augmentation Ablation Studies
           LOO suite (4 ops × 3 seeds = 12 runs)
           Single-component suite (4 ops × 3 seeds = 12 runs)
           → Tables 4 & 5

Stage 4 → Downstream Statistical Tests & Strong Baselines
           BlockGCN strong baseline (3 seeds, ~140 epochs each, ~3.5 min)
           Phase 4: Statistical tests + 1,000 bootstrap resamples
           Phase 5: Per-class breakdown (Table 10)
           Phase 6: Hardware latency profiling (Table 11)
           Phase 7: External MM-Fit transfer benchmark (Table 12)

Stage 5 → Final Synchronization
           update_results_final.py --phase all (all 12 tables)
           Push RESULTS_FINAL.md + all JSON artifacts to HF Hub
```

### Total Estimated Runtime

| Stage | Models | Time |
|---|---|---|
| Stage 1 (81 Phase 1A + 9 Table 2b) | 90 | ~12–14 min |
| Stage 2 (33 Phase 1B + fusion) | 33 | ~5–7 min |
| Stage 3 (24 ablation) | 24 | ~4–5 min |
| Stage 4 (BlockGCN + tests) | 3+eval | ~5–7 min |
| Stage 5 (sync) | — | <1 min |
| **Total** | **~150+** | **~30–35 min** |

---

## 6. Single-Command Execution (Recommended)

### 6.1 Full Pipeline Launch Command

```python
# Run this ONCE to launch the complete pipeline
# All 5 stages execute sequentially; keepalive daemon starts automatically
import subprocess, os

env = dict(os.environ)
env["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"   # Set your HF token

proc = subprocess.Popen(
    [
        "python3", "scripts/marimo_master_e2e_runner.py",
        "--device", "cuda",
        "--workers", "4",
        "--seeds", "42", "123", "3407",
        "--epochs", "100",
        "--force_retrain"    # Clean slate — re-train everything from scratch
    ],
    cwd="/marimo/Gym_Classification",
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env
)
print(f"Pipeline PID: {proc.pid}")
```

> **Note:** `--force_retrain` ensures all models are retrained from scratch on a clean slate. Omit this flag if resuming from a partial run.

### 6.2 Available CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--device cuda` | `cuda` | Target GPU device |
| `--workers 4` | `4` | Parallel training workers |
| `--seeds 42 123 3407` | `[42, 123, 3407]` | Seeds to run |
| `--epochs 100` | `100` | Max training epochs |
| `--force_retrain` | `False` | Ignore cached checkpoints, retrain all |
| `--skip_phase1a` | `False` | Skip Stage 1 (Feature Screening) |
| `--skip_phase1b` | `False` | Skip Stage 2 (Core Backbones) |
| `--skip_phase1c` | `False` | Skip Stage 3 (Ablation) |
| `--skip_blockgcn` | `False` | Skip BlockGCN strong baseline |

---

## 7. Stage-by-Stage Execution Details

### Stage 1: Phase 1A — Feature Screening (Table 2)

**Objective:** Evaluate 3 sequence models × 9 feature spaces × 3 seeds = **81 training runs** to generate mean±SD metrics for Table 2.

**Models:** LSTM (361K), BiLSTM (360K), Transformer (301K).

**Feature Spaces (9 configs, T1.1 → T1.27):**

| Exp IDs | Feature | Dim | Description |
|---|---|---|---|
| T1.1, T1.10, T1.19 | `raw_2d` | 26 | Raw 2D pixel coordinates |
| T1.2, T1.11, T1.20 | `rel_2d` | 26 | Root-relative 2D coordinates |
| T1.3, T1.12, T1.21 | `angle_2d` | 286 | Pairwise joint angles 2D |
| T1.4, T1.13, T1.22 | `angle2_2d` | 78 | Adjacent joint angles 2D |
| T1.5, T1.14, T1.23 | `raw_3d` | 39 | Raw 3D coordinates |
| T1.6, T1.15, T1.24 | `rel_3d` | 39 | Root-relative 3D coordinates |
| T1.7, T1.16, T1.25 | `angle_3d` | 286 | Pairwise joint angles 3D |
| T1.8, T1.17, T1.26 | `angle2_3d` | 78 | Adjacent joint angles 3D |
| **T1.9, T1.18, T1.27** | **`mix_v2`** | **63** | **Biomechanical Mix v2 (Ours)** |

**Checkpoint naming convention:**
- Seed 42: `checkpoints/best_{MODEL}_{EID}_{FEATURE}.pt`
- Seed 123/3407: `checkpoints/seed{SEED}/best_{MODEL}_{EID}_{FEATURE}.pt`

**Post-Stage 1A aggregation** (automatic inside runner):
```python
# Automatically called after training:
aggregate_table2_feature_screening(seeds=[42, 123, 3407], ckpt_base, out_dir, device)
# → outputs/table1_feature_screening_multiseed.json
# → artifacts/results/table1_feature_screening_multiseed.json
```

**Table 2b Sub-stage (immediately after 1A):**
```bash
python scripts/run_upgrade_mix_benchmark.py --workers 4 --device cuda
python scripts/evaluate_upgrade_mix_all.py
# → outputs/upgrade_mix_evaluation_report.json → Table 2b
```

---

### Stage 2: Phase 1B — Core Multi-Seed Backbones

**Models × Seeds:** 11 configs × 3 seeds = **33 training runs**.

| Exp ID | Model | Feature | Augment | Note |
|---|---|---|---|---|
| T3.1 | ST-GCN | `raw_3d` | none | ST-GCN raw baseline |
| T3.2 | ST-GCN | `rel_3d` | none | ST-GCN rel baseline |
| T1.9 | LSTM | `mix_v2` | none | LSTM clean baseline |
| T1.18 | BiLSTM | `mix_v2` | none | BiLSTM clean baseline |
| T1.27 | Transformer | `mix_v2` | none | Transformer clean baseline |
| T3.6 | AAGCN | `bone_3d` | none | AAGCN clean baseline |
| T2.2 | Transformer | `mix_v2` | skel_gym_aug | SkelGym-Aug constituent |
| T4.2 | AAGCN | `bone_3d` | skel_gym_aug | SkelGym-Aug constituent |
| T4.3 | AAGCN | `rel_3d` | skel_gym_aug | SkelGym-Aug constituent |
| T4.4 | AAGCN | `joint_motion_3d` | skel_gym_aug | SkelGym-Aug constituent |
| T4.5 | AAGCN | `bone_motion_3d` | skel_gym_aug | SkelGym-Aug constituent |

**Phase 2 (immediately after 1B training):**
```bash
python scripts/freeze_reference_artifacts.py
# → Extracts normalization stats, calibrates SLSQP weights per seed
```

**Phase 3 (immediately after Phase 2):**
```bash
python scripts/run_multi_seed_experiments.py \
  --skip_train --seeds 42 123 3407 --device cuda --include_baselines
# → Evaluates 5 fusion methods: Hard Vote, Uniform Soft, Acc-Weighted, Stacking, SLSQP
# → outputs/multi_seed_evaluation_results.json → Tables 6 & 7
```

**Post-Stage 2 aggregation** (automatic inside runner):
```python
aggregate_table3_graph_streams(seeds=[42, 123, 3407], ckpt_base, out_dir, device)
# → outputs/graph_streams_multiseed.json
# → artifacts/results/graph_streams_multiseed.json
# → Includes T3.8 (Two-Stream) and T3.9 (Four-Stream) ensemble fusion
```

---

### Stage 3: Phase 1C — Augmentation Ablation Studies

**LOO Suite (Table 4):** Leave one augmentation operation out at a time across 3 seeds.

```bash
python scripts/run_augmentation_experiments.py \
  --mode loo --seeds 42 123 3407 --device cuda --workers 4 --push_to_hf
```

**Single-Component Suite (Table 5):** Evaluate each augmentation alone across 3 seeds.

```bash
python scripts/run_augmentation_experiments.py \
  --mode single --seeds 42 123 3407 --device cuda --workers 4 --push_to_hf
```

---

### Stage 4: Downstream Tests & Strong Baselines

#### 4a. BlockGCN Strong Baseline (3 Seeds)

BlockGCN is an external state-of-the-art graph convolutional network serving as the strongest external comparison point in Table 13.

```bash
python scripts/run_blockgcn_baseline.py \
  --seeds 42 123 3407 \
  --device cuda \
  --push_to_hf \
  --hf_token $HF_TOKEN
```

- **Architecture:** BlockGCN (original 33-joint, 32-frame STGCN variant), adapted to SkelGym 13-joint graph topology.
- **Training:** 140 epochs × 3 seeds (automatic inside the script).
- **ETA:** ~3–4 minutes total.

#### 4b. Statistical Tests (Phase 4)

```bash
python scripts/compute_statistical_tests.py --device cuda --b_samples 1000
# → McNemar's chi-square + Wilcoxon paired tests (Table 8)
# → 1,000 video-clustered bootstrap 95% CI (Table 9)
```

#### 4c. Per-Class Breakdown (Phase 5)

```bash
python scripts/evaluate_local_ensemble.py --seeds 42 --device cuda
# → Per-class precision, recall, F1 for all 22 exercise classes (Table 10)
```

#### 4d. Hardware Latency Profiling (Phase 6)

```bash
python scripts/benchmark_hardware_latency.py --device cuda
# → MACs, FLOPs, parameters, inference latency (ms/window), throughput (FPS)
# → outputs/hardware_latency.json → Table 11
```

#### 4e. External Transfer Benchmark (Phase 7, MM-Fit)

```bash
python scripts/evaluate_external_benchmark.py --device cuda
# → Zero-shot cross-dataset generalization on MM-Fit / Deyzel S&C benchmark
# → outputs/external_benchmark_results.json → Table 12
```

---

### Stage 5: Final Synchronization

```bash
python scripts/update_results_final.py --phase all
```

This performs:
1. Reads all generated JSON artifacts.
2. Populates all 12 result tables in `outputs/RESULTS_FINAL.md`.
3. Runs automated numerical verification (cross-checks JSON vs. markdown tables).
4. Uploads `RESULTS_FINAL.md` and all JSON artifacts to the HF Hub.

---

## 8. Checkpoint Organization

```
checkpoints/
├── best_LSTM_T1.1_raw_2d.pt              # Seed 42 Phase 1A
├── best_LSTM_T1.9_mix_v2.pt              # Seed 42 LSTM baseline
├── best_Transformer_T1.27_mix_v2.pt       # Seed 42 Transformer clean
├── best_AAGCN_T4.2_bone_3d.pt            # Seed 42 constituent
├── best_STGCN_T3.2_rel_3d.pt             # Seed 42 ST-GCN baseline
├── seed123/
│   ├── best_LSTM_T1.9_mix_v2.pt
│   ├── best_Transformer_T1.27_mix_v2.pt
│   └── ...
├── seed3407/
│   ├── best_LSTM_T1.9_mix_v2.pt
│   └── ...
└── external/
    └── blockgcn_seed42.pt                # BlockGCN baseline checkpoints
```

---

## 9. Output Artifacts

| File | Generated by | Purpose |
|---|---|---|
| `outputs/RESULTS_FINAL.md` | `update_results_final.py` | Master results document (12 tables) |
| `outputs/table1_feature_screening_multiseed.json` | `aggregate_table2_feature_screening()` | Table 2 source data |
| `artifacts/results/table1_feature_screening_multiseed.json` | same | Canonical artifact copy |
| `outputs/graph_streams_multiseed.json` | `aggregate_table3_graph_streams()` | Table 3 source data |
| `outputs/upgrade_mix_evaluation_report.json` | `evaluate_upgrade_mix_all.py` | Table 2b source data |
| `outputs/multi_seed_evaluation_results.json` | `run_multi_seed_experiments.py` | Tables 6 & 7 source data |
| `outputs/augmentation_ablation_results.json` | `run_augmentation_experiments.py` | Tables 4 & 5 source data |
| `outputs/statistical_tests_report.json` | `compute_statistical_tests.py` | Table 8 source data |
| `outputs/bootstrap_confidence_intervals.json` | `compute_statistical_tests.py` | Table 9 source data |
| `outputs/per_class_results.json` | `evaluate_local_ensemble.py` | Table 10 source data |
| `outputs/hardware_latency.json` | `benchmark_hardware_latency.py` | Table 11 source data |
| `outputs/external_benchmark_results.json` | `evaluate_external_benchmark.py` | Table 12 source data |
| `outputs/triggers.jsonl` | Master runner (event log) | Real-time progress tracking |
| `outputs/keepalive.heartbeat` | Keepalive daemon | Server liveness signal |

---

## 10. Operational Guidelines for AI Executor

The following rules **must** be followed by any AI agent executing this pipeline:

### Rule 1 — Read-Before-Execute
Before executing ANY command, read this document (`docs/E2E_PIPELINE_AND_TRAINING_STRATEGY.md`) in full. No assumptions about the pipeline structure should be made from memory.

### Rule 2 — Pull Latest Code First
Always run `git pull origin main` in `/marimo/Gym_Classification` before starting. This ensures all latest script changes (especially `marimo_master_e2e_runner.py` and `run_tables2_3_multi_seed.py`) are present.

### Rule 3 — Single Command, No Manual Stages
**Never** run individual stage scripts manually unless resuming from a specific failure point. Launch the master runner with `--force_retrain` for a clean-slate full run.

### Rule 4 — Strict 15-Minute Check Cadence
After launching the master runner, **check progress exactly once every 15 minutes** by reading `outputs/triggers.jsonl` and counting checkpoint files. Do NOT poll every 2–3 minutes. Do NOT interrupt the runner.

### Rule 5 — Never Kill the Keepalive
The keepalive daemon is internal to the runner process. Killing or restarting the process unnecessarily will stop the keepalive and risk sandbox timeout.

### Rule 6 — Monitor Heartbeat
If `outputs/keepalive.heartbeat` is older than 60 seconds, the runner may have crashed. Check process status before declaring failure:
```python
import subprocess
r = subprocess.run(["pgrep", "-f", "marimo_master_e2e_runner"], capture_output=True, text=True)
print("Runner PID:", r.stdout.strip() or "NOT RUNNING")
```

### Rule 7 — HF_TOKEN
Always ensure `HF_TOKEN` environment variable is set before running. Verify:
```python
import os
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
```

### Rule 8 — On Completion
When `[TRIGGER: MASTER_PIPELINE_COMPLETE]` appears in `outputs/triggers.jsonl`, the pipeline is done. Report final metrics from `outputs/RESULTS_FINAL.md` to the user.

---

## 11. Expected Model Count Summary

| Stage | Config | Seeds | Total Models |
|---|---|---|---|
| Phase 1A (Feature Screening) | 27 (3 models × 9 features) | 3 | **81** |
| Table 2b (Transformer Upgrade) | 3 feature configs | 3 | **9** |
| Phase 1B (Core Backbones) | 11 (6 baselines + 5 aug) | 3 | **33** |
| Phase 1C LOO Ablation | ~4 ops | 3 | **~12** |
| Phase 1C Single-Op Ablation | ~4 ops | 3 | **~12** |
| BlockGCN Baseline | 1 | 3 | **3** |
| **Grand Total** | | | **~150** |

---

## 12. Git Commit History Reference

| Commit | Description |
|---|---|
| `7b24412` | Created E2E docs and runner integration |
| `86d1dec` | Updated papers to mix_v2 (63-d) and 301K budget |
| Latest | Multi-seed Phase 1A, TABLE3_SPECS, BlockGCN in master runner |
