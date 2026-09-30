# SkelGym End-to-End Pipeline Execution Plan (Master Plan E2E — Multi-Seed Edition)

> **Mục tiêu**: Hướng dẫn vận hành chi tiết từ đầu (from scratch) toàn bộ pipeline nghiên cứu **SkelGym**: từ thẩm định tính toàn vẹn dữ liệu, huấn luyện 5 mô hình baseline và 5 mô hình thành phần tăng cường trên **3 random seeds (42, 123, 3407)**, so sánh có hệ thống **5 phương pháp Late Fusion**, nghiên cứu ablation augmentation, kiểm định thống kê bootstrap, đo đạc latency phần cứng, cho đến benchmark tổng quát hóa trên tập dữ liệu external (MM-Fit / Fit3D) hoàn toàn trên **multi-seed**.

---

## 📌 0. Tổng Quan & Trạng Thái Khởi Tạo (Clean Slate)

- **Trạng thái dọn dẹp:** Toàn bộ checkpoints, outputs, và artifacts từ các đợt chạy trước đã được đóng gói an toàn tại `archive_previous_run_backup_20261001.tar.gz`.
- **Cấu trúc thư mục sạch:**
  - `checkpoints/`: Sạch sẽ, sẵn sàng ghi nhận trọng số mới theo từng seed (`checkpoints/best_*.pt` cho seed 42; `checkpoints/seed123/`, `checkpoints/seed3407/`).
  - `outputs/`: Sạch sẽ (sẵn sàng `outputs/logs/` và `outputs/external/`).
  - `artifacts/reference/`: Sẵn sàng cho việc đóng băng thống kê chuẩn hóa.
- **Quy chuẩn khoa học bắt buộc xuyên suốt:**
  1. **Strict Video-Level Partition:** Phân chia 6:2:2 ở cấp độ source-video (1,024 videos; train: 580, val: 208, test: 236) — tuyệt đối không overlap video giữa các split.
  2. **Zero Test-Data Leakage:** Thống kê chuẩn hóa (z-score mean/std) tính độc quyền trên tập **Train**. Trọng số ensemble soft-voting (SLSQP) và Stacking meta-classifier tối ưu hóa độc quyền trên tập **Validation**. Tập **Test** chỉ dùng một lần duy nhất để đánh giá.
  3. **Multi-Seed Rigor (Bắt buộc Phase 1, 2, 3, 4, 5, 7):** Mọi kết quả từ baseline, constituent models, 5 fusion methods, ablation, bootstrap, đến external test đều phải được thực thi và báo cáo theo định dạng **$\text{Mean} \pm \text{SD}$** trên 3 seed cố định: `42`, `123`, `3407`.

---

## 🗺️ Quy Trình Vận Hành 8 Bước Multi-Seed (Workflow Diagram)

```
[Phase 0: Pre-flight Verification & Data Report]
                      │
                      ▼
[Phase 1: Multi-Seed Training (Seeds 42, 123, 3407)]
  ├── 5 Baseline Models: ST-GCN, LSTM, BiLSTM, Transformer Clean, AAGCN Clean
  └── 5 Augmented Constituent Models: Transformer Mix, AAGCN Bone, Rel, J-Mot, B-Mot
                      │
                      ▼
[Phase 2: Multi-Seed Reference Artifacts Freeze (Train Stats & Val Weights)]
                      │
                      ▼
[Phase 3: Multi-Seed Late Fusion Comparison (5 Methods x 4 Ensembles)]
  ├── 1. Hard Voting (Majority Rule)
  ├── 2. Uniform Soft Voting (Arithmetic Mean)
  ├── 3. Accuracy-Weighted Soft Voting (Validation Accuracy Weights)
  ├── 4. Stacking Meta-Classifier (Logistic Regression on Probabilities)
  └── 5. SLSQP Dual-Target Soft Voting (Proposed: NLL-Minimization)
                      │
                      ▼
[Phase 4: Multi-Seed Augmentation Ablation Studies (LOO & Single Component)]
                      │
                      ▼
[Phase 5: Multi-Seed Statistical Hypothesis Testing & Video-Cluster Bootstrap]
                      │
                      ▼
[Phase 6: Hardware Latency & Complexity Profiling (CPU / MPS / CUDA)]
                      │
                      ▼
[Phase 7: Multi-Seed External Cross-Dataset Benchmark (MM-Fit & Fit3D)]
  ├── Protocol A (MediaPipe Re-extraction) & Protocol B (Native 3D Pose)
  └── Zero-Shot Open/Closed Set + 100-Trial Few-Shot Simulation
                      │
                      ▼
[Phase 8: Manuscript LaTeX Synchronization & Publication Compilation]
```

---

## 🚀 Hướng Dẫn Thực Thi Chi Tiết Từng Phase

### Phase 0: Kiểm Tra Môi Trường & Chạy Test Suite

Chạy kiểm tra toàn bộ 43 unit và reproducibility test cases trước khi train:

```bash
# 1. Chạy toàn bộ test suite
pytest tests/ -v

# 2. Sinh báo cáo thống kê dataset (bảo đảm 1,024 videos, 22 classes, 13 joints)
python run.py data-report --output_dir outputs/test_report
```

*Tiêu chí đạt (Quality Gate):* 43/43 tests PASSED; file `outputs/test_report/dataset_report.md` và `table1_counts.tex` được tạo thành công.

---

### Phase 1: Huấn Luyện Multi-Seed (Seeds 42, 123, 3407)

Thực hiện huấn luyện trên cả 3 seed cho **10 mô hình** (5 Baselines + 5 Augmented Constituent models):

#### Danh mục 10 Mô hình:
1. **5 Baseline Models (Không Augmentation):**
   - `ST-GCN Baseline` (T3.2, `rel_3d`, `--augment none`)
   - `LSTM Baseline` (T1.9, `mix`, `--augment none`)
   - `BiLSTM Baseline` (T1.18, `mix`, `--augment none`)
   - `Transformer Clean` (T1.27, `mix`, `--augment none`)
   - `AAGCN Clean` (T3.6, `bone_3d`, `--augment none`)
2. **5 Augmented Constituent Models (SkelGym-Aug):**
   - `Transformer Mix` (T2.2, `mix`, `--augment skel_gym_aug`)
   - `AAGCN Bone 3D` (T4.2, `bone_3d`, `--augment skel_gym_aug`)
   - `AAGCN Rel 3D` (T4.3, `rel_3d`, `--augment skel_gym_aug`)
   - `AAGCN Joint Motion 3D` (T4.4, `joint_motion_3d`, `--augment skel_gym_aug`)
   - `AAGCN Bone Motion 3D` (T4.5, `bone_motion_3d`, `--augment skel_gym_aug`)

#### Lệnh Huấn Luyện Trọn Gói Multi-Seed (Khuyên Dùng):

```bash
# Huấn luyện toàn bộ 5 Baselines + 5 Augmented models trên cả 3 seeds 42, 123, 3407
python scripts/run_multi_seed_experiments.py --device auto --epochs 100 --include_baselines
```

#### Hoặc Huấn Luyện Thủ Công Từng Mô Hình (Cú Pháp Mẫu):

```bash
# Ví dụ chạy với Seed 42, 123, 3407 cho Baselines
for SEED in 42 123 3407; do
  CKPT_DIR="checkpoints"
  [ "$SEED" -ne 42 ] && CKPT_DIR="checkpoints/seed${SEED}"

  # 1. ST-GCN Baseline
  python run.py train --model STGCN --feature rel_3d --augment none --exp_id T3.2 \
    --seed $SEED --epochs 100 --patience 10 --checkpoint_dir $CKPT_DIR --device auto --use_amp --in_memory

  # 2. LSTM Baseline
  python run.py train --model LSTM --feature mix --augment none --exp_id T1.9 \
    --seed $SEED --epochs 100 --patience 10 --checkpoint_dir $CKPT_DIR --device auto --use_amp --in_memory

  # 3. BiLSTM Baseline
  python run.py train --model BiLSTM --feature mix --augment none --exp_id T1.18 \
    --seed $SEED --epochs 100 --patience 10 --checkpoint_dir $CKPT_DIR --device auto --use_amp --in_memory

  # 4. Transformer Clean Baseline
  python run.py train --model Transformer --feature mix --augment none --exp_id T1.27 \
    --seed $SEED --epochs 100 --patience 10 --checkpoint_dir $CKPT_DIR --device auto --use_amp --in_memory

  # 5. AAGCN Clean Baseline
  python run.py train --model AAGCN --feature bone_3d --augment none --exp_id T3.6 \
    --seed $SEED --epochs 100 --patience 10 --checkpoint_dir $CKPT_DIR --device auto --use_amp --in_memory
done
```

*Sản phẩm đầu ra:*
- `checkpoints/best_*.pt` (Seed 42)
- `checkpoints/seed123/best_*.pt` (Seed 123)
- `checkpoints/seed3407/best_*.pt` (Seed 3407)

---

### Phase 2: Đóng Băng Reference Artifacts (Multi-Seed Reference Freeze)

Trích xuất và cố định vĩnh viễn thống kê chuẩn hóa từ tập Train và trọng số SLSQP validation cho seed 42 (tham chiếu chính) và lưu trữ manifests cho các seeds:

```bash
python scripts/freeze_reference_artifacts.py
```

*Sản phẩm đầu ra tại `artifacts/reference/`:*
- `normalization_mix.npz`, `normalization_rel_3d.npz`, `normalization_bone_3d.npz`, `normalization_joint_motion_3d.npz`, `normalization_bone_motion_3d.npz`
- `ensemble_weights.json` (Trọng số SLSQP Validation)
- `checkpoint_manifest.json` (Mã băm SHA-256 các model)
- `class_names.json` (22 nhãn lớp canonical)

---

### Phase 3: So Sánh 5 Phương Pháp Late Fusion trên Multi-Seed

Đánh giá và so sánh toàn diện **5 phương pháp Fusion** trên **4 cấu hình ensemble** xuyên suốt cả 3 random seeds:

#### 5 Phương Pháp Late Fusion:
1. **Hard Voting:** Biểu quyết đa số (Majority Voting) giữa các dự đoán nhãn rời rạc của các stream.
2. **Uniform Soft Voting:** Trung bình cộng số học xác suất dự đoán (Unweighted Average Soft Voting): $P(y) = \frac{1}{M}\sum_{m=1}^M P_m(y)$.
3. **Accuracy-Weighted Soft Voting:** Trọng số xác suất tỉ lệ thuận với độ chính xác trên tập Validation: $w_m = \frac{\text{val\_acc}_m}{\sum \text{val\_acc}_j}$.
4. **Stacking Meta-Classifier:** Huấn luyện bộ phân loại meta Logistic Regression trên vector xác suất validation ghép nối.
5. **SLSQP Soft Voting (Đề xuất SkelGym):** Tối ưu hóa ràng buộc simplex ($\sum w_i = 1, w_i \ge 0$) cực tiểu hóa hàm NLL độc lập trên Validation Windows và Validation Video Consensus.

#### 4 Cấu hình Ensemble:
- **Two-Stream AAGCN (Aug):** Joint + Bone
- **Four-Stream AAGCN (Aug):** Joint + Bone + Joint Motion + Bone Motion
- **SkelGym-Lite:** Transformer Mix + AAGCN Bone
- **SkelGym-Full:** Transformer Mix + 4-Stream AAGCN

#### Lệnh Chạy:

```bash
# Đánh giá đồng thời 5 phương pháp Fusion và 10 mô hình đơn lẻ trên cả 3 seeds
python scripts/evaluate_local_ensemble.py --seeds 42 123 3407 --device auto
```

*Sản phẩm đầu ra:* `outputs/multi_seed_evaluation_results.md` chứa bảng đối chiếu đầy đủ $\text{Mean} \pm \text{SD}$ cho cả 5 phương pháp Fusion ở cấp độ Window và Video Consensus.

---

### Phase 4: Nghiên Cứu Ablation Augmentation trên Multi-Seed

Chạy 2 nghiên cứu ablation độc lập xuyên suốt cả 3 random seeds:
1. **Leave-One-Out (LOO) Ablation:** Loại bỏ lần lượt từng toán tử khỏi Candidate Full (5-op):
   - Full Candidate (5-op)
   - w/o Sagittal Reflection (-Mirror)
   - w/o Gravitational Yaw (-Yaw)
   - w/o Proportional Scaling (-Scale)
   - w/o Temporal TimeWarp (-TimeWarp $\to$ Đề xuất SkelGym-Aug 4-op)
   - w/o Sensor Jitter (-Jitter)
   - Clean Baseline (No Augmentation)
2. **Single Component Ablation:** Thêm đơn lẻ từng toán tử vào Clean Baseline.

```bash
# Chạy cả 2 nghiên cứu trên cả 3 seeds
python scripts/run_augmentation_experiments.py --mode all --device auto --seeds 42 123 3407
```

*Sản phẩm đầu ra:*
- `outputs/augmentation_ablation_results.json`
- `outputs/augmentation_ablation_results.md`
- `outputs/table_leave_one_out.tex`
- `outputs/table_single_component.tex`
- `outputs/per_class_augmentation_ablation.csv`

---

### Phase 5: Kiểm Định Thống Kê & Video-Cluster Bootstrap CI trên Multi-Seed

Chạy kiểm định giả thuyết thống kê và ước lượng khoảng tin cậy phi tham số cho các mô hình trên multi-seed:
- **McNemar's Test** ở Window-level.
- **Wilcoxon Signed-Rank Test** ở Video-level kèm hiệu chỉnh Bonferroni.
- **Video-Cluster Non-parametric Bootstrap** ($B = 1,000$ iterations) ước tính 95% Confidence Intervals cho Window Accuracy và Video Consensus Accuracy trên từng seed và pooled across seeds.

```bash
python scripts/compute_statistical_tests.py
```

*Sản phẩm đầu ra:* `outputs/bootstrap_confidence_intervals.md`.

---

### Phase 6: Đo Đạc Latency Phần Cứng & Phân Tích Độ Phức Tạp

Đo đạc tốc độ suy luận classifier và độ phức tạp tính toán:
- Thời gian trễ (Mean, Median, p95 Latency) trên CPU, Apple MPS, hoặc NVIDIA CUDA.
- Kích thước tham số (Parameter Count) và FLOPs/MACs lý thuyết qua thư viện THOP.

```bash
python scripts/benchmark_hardware_latency.py --device auto
```

---

### Phase 7: Benchmark Độc Lập Bên Ngoài trên Multi-Seed (MM-Fit & Fit3D)

Đánh giá khả năng tổng quát hóa trên tập dữ liệu external chưa từng tham gia huấn luyện trên **cả 3 seeds (42, 123, 3407)**:

#### 1. MM-Fit Benchmark (Dataset có sẵn tại thư mục `mm-fit/`)

```bash
# a. Trích xuất MediaPipe canonical landmarks (Protocol A)
python scripts/prepare_mmfit_external.py --root mm-fit --pose-source mediapipe

# b. Đánh giá Zero-shot Cross-Dataset trên cả 3 seeds (Open-set & Closed-set)
python scripts/evaluate_external.py --config configs/external/mmfit.yaml --seeds 42 123 3407

# c. Đánh giá One-shot Classification Simulation (100 trials) trên cả 3 seeds
python scripts/evaluate_external_fewshot.py --config configs/external/mmfit.yaml --trials 100 --seeds 42 123 3407
```

#### 2. Fit3D Benchmark (Nếu có dataset Fit3D)

```bash
python scripts/prepare_fit3d_external.py --root /path/to/Fit3D --pose-source mediapipe
python scripts/evaluate_external.py --config configs/external/fit3d.yaml --seeds 42 123 3407
```

*Sản phẩm đầu ra tại `outputs/external/mmfit/`:*
- `dataset_audit.csv`
- `metrics_closed.csv` & `metrics_open.csv` (ghi nhận $\text{Mean} \pm \text{SD}$ qua 3 seeds)
- `fewshot_summary.csv` & `fewshot_100_trials.csv`
- `confusion_closed.png` & `confusion_open.png`
- `bootstrap_ci.json`
- `run_manifest.json`

---

### Phase 8: Đồng Bộ Hóa Bảng Biểu & Biên Dịch Manuscript Paper

Sau khi thu thập đầy đủ số liệu thực nghiệm:
1. Đồng bộ các bảng LaTeX vào manuscript Elsevier ESWA (`paper/paper_eswa.tex`) và Springer LNCS (`paper/paper_llncs.tex`).
2. Biên dịch hình vẽ kiến trúc TikZ:
   ```bash
   python scripts/compile_horizontal_tikz.py
   ```
3. Biên dịch bản thảo PDF:
   ```bash
   cd paper && pdflatex paper_eswa.tex && bibtex paper_eswa && pdflatex paper_eswa.tex && pdflatex paper_eswa.tex
   ```

---

## ⚡ Lệnh Chạy Tự Động Toàn Trình (Multi-Seed One-Liner)

Bạn có thể chạy toàn bộ quy trình tuần tự bằng 1 dòng lệnh bash:

```bash
pytest tests/ && \
python run.py data-report && \
python scripts/run_multi_seed_experiments.py --device auto --include_baselines && \
python scripts/freeze_reference_artifacts.py && \
python scripts/evaluate_local_ensemble.py --seeds 42 123 3407 --device auto && \
python scripts/run_augmentation_experiments.py --mode all --device auto --seeds 42 123 3407 && \
python scripts/compute_statistical_tests.py && \
python scripts/benchmark_hardware_latency.py --device auto && \
python scripts/prepare_mmfit_external.py --root mm-fit --pose-source mediapipe && \
python scripts/evaluate_external.py --config configs/external/mmfit.yaml --seeds 42 123 3407 && \
python scripts/evaluate_external_fewshot.py --config configs/external/mmfit.yaml --trials 100
```
