# SkelGym End-to-End Pipeline Execution Plan (Master Plan E2E — Full Paper Edition)

> **Mục tiêu**: Hướng dẫn vận hành chi tiết từ đầu (from scratch) **100% toàn bộ hệ thống thí nghiệm của bài báo SkelGym**: từ thẩm định tính toàn vẹn dữ liệu, sàng lọc 27 mô hình Feature Engineering (Table 1), huấn luyện 10 mô hình cốt lõi đa hạt giống trên 3 random seeds (42, 123, 3407), nghiên cứu Leave-One-Out và Single-Component Augmentation Ablation (Table 2 & Table 3), so sánh có hệ thống 5 phương pháp Late Fusion (Table 5 & Table 7), kiểm định thống kê bootstrap, đo đạc latency phần cứng (Table 11), cho đến benchmark tổng quát hóa trên tập dữ liệu external (Table 12).

---

## 📌 0. Tổng Quan & Trạng Thái Khởi Tạo (Clean Slate)

- **Trạng thái dọn dẹp:** Toàn bộ checkpoints, outputs, và artifacts từ các đợt chạy trước đã được đóng gói an toàn tại `archive_pre_clean_results.tar.gz`. Toàn bộ file kết quả rác, logs cũ đã được dọn sạch.
- **Tập tin kết quả duy nhất (Source of Truth Template):**
  - 👉 [`outputs/RESULTS_FINAL.md`](file:///Volumes/WorkSpace/Project/Gym_Classification/outputs/RESULTS_FINAL.md): Chứa đầy đủ cấu trúc của toàn bộ 12 bảng biểu trong bài báo, đang ở trạng thái `Pending` chờ điền kết quả tự động sau mỗi phase.
- **Cơ chế Tự Động Đẩy Checkpoint lên Hugging Face Hub (Zero Local Disk Pressure):**
  - **Model Hub Repository:** [`Cuong2004/gym-exercise-classification`](https://huggingface.co/Cuong2004/gym-exercise-classification)
  - **Cấu hình Token xác thực:** Thiết lập biến môi trường trước khi chạy trên server/Marimo:
    ```bash
    export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```
    *(hoặc truyền cờ `--hf_token <TOKEN>` trong câu lệnh)*.
  - **Cờ kích hoạt:** Bổ sung cờ `--push_to_hf` vào tất cả các lệnh huấn luyện.
  - **Cơ chế Stream Trực Tiếp:** Ngay khi một mô hình tìm thấy epoch tối ưu, Trainer tự động upload `best_*.pt` kèm sidecar `best_*.provenance.json` (chứa commit SHA, cấu hình seed, siêu tham số) lên HF Hub. Trọng số được lưu trữ vĩnh viễn trên Cloud ngay cả khi container Marimo/GPU bị tắt ngang, hoàn toàn không cần tải checkpoint nặng về máy local.
- **Dữ Liệu Sẵn Sàng Trên Hugging Face Hub (TUYỆT ĐỐI KHÔNG CẦN Trích Xuất Lại Bằng MediaPipe):**
  - **Dataset Hub Repository:** [`Cuong2004/gym-exercise-landmarks`](https://huggingface.co/datasets/Cuong2004/gym-exercise-landmarks)
  - **Lưu ý tối quan trọng:** **KHÔNG CẦN tải video MP4 gốc (10 GB) hay chạy lại MediaPipe từ đầu**. Toàn bộ 1,024 video SkelGym và tập External Benchmark (MM-Fit / Deyzel et al.) đã được trích xuất sẵn tọa độ không gian 3D keypoints chuẩn (`complexity=2`), thẩm định chất lượng và đóng gói sẵn trên Hugging Face.
  - **Tải nhanh Dataset SkelGym (Landmarks & Metadata):**
    ```bash
    # Cách 1: Lệnh CLI chính thức (tự động tải và giải nén vào data/landmarks/)
    python run.py pull-landmarks-hf --dest_dir data/landmarks

    # Cách 2: Bootstrap tự động kiểm tra và tải các file còn thiếu
    python scripts/bootstrap_artifacts.py
    ```
  - **Tải nhanh External Dataset (MM-Fit / Deyzel Benchmark):**
    ```bash
    # Tự động tải landmarks MediaPipe của tập external MM-Fit vào data_external/mmfit/landmarks/
    python scripts/sync_mmfit_hf.py --action download_landmarks
    ```
- **Quy chuẩn khoa học bắt buộc xuyên suốt:**
  1. **Strict Video-Level Partition:** Phân chia 6:2:2 ở cấp độ source-video (1,024 videos; train: 580, val: 208, test: 236; 233 valid test videos $\ge 32$ frames) — tuyệt đối không overlap video giữa các split.
  2. **Zero Test-Data Leakage:** Thống kê chuẩn hóa (z-score mean/std) tính độc quyền trên tập **Train**. Trọng số ensemble soft-voting (SLSQP) và Stacking meta-classifier tối ưu hóa độc quyền trên tập **Validation**. Tập **Test** chỉ dùng một lần duy nhất để đánh giá cuối cùng.
  3. **Data Pipeline Bug-Free Guarantee:** Đã sửa triệt để lỗi double-normalization trong `src/data/dataset.py` và lỗi zero-variance root-bone trong `src/data/augmentations.py`.
  4. **Multi-Seed Rigor:** Mọi bảng kết quả chính (Phase 1B, 1C, 3, 4, 7) bắt buộc phải thực thi và báo cáo theo định dạng **$\text{Mean} \pm \text{SD}$** trên 3 seed cố định: `42`, `123`, `3407`.

---

## 🗺️ Bản Đồ Vận Hành 9 Pha Toàn Diện (Full Workflow Diagram)

```
[Phase 0: Data Pull from HF, Verification & Test Suite] (51/51 tests)
                      │
                      ▼
[Phase 1A: Feature Representation Screening (Paper Table 1)]
  └── 27 Thí nghiệm: 3 Kiến trúc (LSTM, BiLSTM, Transformer) x 9 Feature Spaces (Seed 42)
      * Hỗ trợ --push_to_hf stream checkpoint thẳng lên Hugging Face Hub
                      │
                      ▼
[Phase 1B: Multi-Seed Core Backbone Training (Paper Table 4, 5, 7)]
  ├── 5 Baseline Models (Seed 42, 123, 3407): ST-GCN, LSTM, BiLSTM, Trans Clean, AAGCN Clean
  └── 5 Augmented Streams (Seed 42, 123, 3407): Trans Mix, AAGCN Bone, Rel, J-Mot, B-Mot
      * Hỗ trợ --push_to_hf stream checkpoint thẳng lên Hugging Face Hub
                      │
                      ▼
[Phase 1C: Multi-Seed Augmentation Ablation Studies (Paper Table 2 & Table 3)]
  ├── Table 2: Leave-One-Out (LOO) Suite (7 configs x 3 seeds)
  └── Table 3: Single-Component Suite (8 configs x 3 seeds)
      * Hỗ trợ --push_to_hf stream checkpoint thẳng lên Hugging Face Hub
                      │
                      ▼
[Phase 2: Multi-Seed Reference Artifacts Freeze]
  └── Trích xuất z-score stats & Tối ưu hóa trọng số SLSQP trên Validation Set (3 seeds)
                      │
                      ▼
[Phase 3: Multi-Seed Late Fusion Comparison (Paper Table 5 & Table 7)]
  └── 5 Phương pháp Fusion x 4 Cấu hình Ensemble (Two-Stream, Four-Stream, Lite, Full)
                      │
                      ▼
[Phase 4: Statistical Testing & Video-Cluster Bootstrap (Paper Table 8 & Table 9)]
  ├── McNemar Test, Wilcoxon Signed-Rank Test, Paired t-Test
  └── 1,000 Iterations Video-Cluster Bootstrap with 95% Confidence Intervals
                      │
                      ▼
[Phase 5: Per-Class Breakdown & Error Taxonomy (Paper Table 10 & Table 13)]
  └── 22 Classes Evaluation & Phân tích 4 cơ chế lỗi sinh cơ học
                      │
                      ▼
[Phase 6: Hardware Complexity & Latency Profiling (Paper Table 11)]
  └── Đo FLOPs, Tham số, và Latency thực tế trên RTX PRO 6000 (CUDA), Apple M4 (MPS & CPU)
                      │
                      ▼
[Phase 7: S&C Subset & External Cross-Dataset Benchmark (Paper Table 12)]
  └── 4 Bài tập S&C (Deyzel et al. CVPRW 2023): Open-Set & Closed-Set, Zero/Few-shot
                      │
                      ▼
[Phase 8: LaTeX Synchronization & Manuscript Compilation]
  └── Đồng bộ toàn bộ số liệu vào paper/paper.tex và biên dịch PDF
```

---

## 🚀 Hướng Dẫn Thực Thi Chi Tiết Từng Pha

### Phase 0: Chuẩn Bị Dữ Liệu, Thẩm Định Môi Trường & Chạy Test Suite

```bash
# 1. Tải nhanh toàn bộ landmarks SkelGym từ Hugging Face (Bỏ qua nếu data/landmarks đã có sẵn)
python run.py pull-landmarks-hf --dest_dir data/landmarks

# 2. Tải landmarks external MM-Fit từ Hugging Face (Dành cho Phase 7)
python scripts/sync_mmfit_hf.py --action download_landmarks

# 3. Chạy toàn bộ test suite (Bảo đảm 51/51 tests PASSED)
pytest tests/ -v

# 4. Sinh báo cáo thống kê dataset
python run.py data-report --output_dir outputs/test_report
```

---

### Phase 1A: Feature Representation Screening (Paper Table 1 — 27 Thí Nghiệm)

*Mục tiêu:* Đánh giá 3 kiến trúc chuỗi trên 9 dạng đặc trưng trên **Seed 42** để sàng lọc và chứng minh tính ưu việt của đặc trưng đề xuất **Biomechanical Mix 117-d**.

#### Danh Mục 27 Thí Nghiệm (T1.1 $\to$ T1.27):
1. **LSTM (9 feature spaces):**
   - `T1.1`: LSTM on `raw_2d` (dim 26)
   - `T1.2`: LSTM on `rel_2d` (dim 26)
   - `T1.3`: LSTM on `angle_2d` (dim 286)
   - `T1.4`: LSTM on `angle2_2d` (dim 78)
   - `T1.5`: LSTM on `raw_3d` (dim 39)
   - `T1.6`: LSTM on `rel_3d` (dim 39)
   - `T1.7`: LSTM on `angle_3d` (dim 286)
   - `T1.8`: LSTM on `angle2_3d` (dim 78)
   - `T1.9`: LSTM on `mix` (dim 117)
2. **BiLSTM (9 feature spaces):**
   - `T1.10` $\to$ `T1.18`: Tương tự cho BiLSTM trên 9 không gian đặc trưng.
3. **Transformer (9 feature spaces):**
   - `T1.19` $\to$ `T1.27`: Tương tự cho Transformer trên 9 không gian đặc trưng (`T1.27` là cấu hình chiến thắng).

#### Lệnh Thực Thi (Tự động stream checkpoints lên HF Hub):
```bash
# Chạy toàn bộ 27 thí nghiệm Table 1 (Hỗ trợ 4 workers song song, tự động upload lên HF):
python server_runner.py --table table1 --workers 4 --push_to_hf
```
*Ghi chú:* 27 checkpoint của đợt huấn luyện này cũng có thể được phục hồi tức thì từ file `archive_pre_clean_results.tar.gz` nếu cần đối chiếu kết quả nhanh.

---

### Phase 1B: Huấn Luyện Multi-Seed Core Backbones (Paper Table 4, 5, 7 — 10 Mô Hình $\times$ 3 Seeds)

*Mục tiêu:* Huấn luyện 10 mô hình chủ lực trên **3 seeds (42, 123, 3407)** làm nền tảng cho các bộ ghép Ensemble, tự động stream weights lên HF Hub.

#### Danh Mục 10 Mô Hình:
1. **5 Baselines (Không Augmentation):**
   - `ST-GCN Baseline` (T3.2, feature `rel_3d`)
   - `LSTM Baseline` (T1.9, feature `mix`)
   - `BiLSTM Baseline` (T1.18, feature `mix`)
   - `Transformer Clean` (T1.27, feature `mix`)
   - `AAGCN Clean` (T3.6, feature `bone_3d`)
2. **5 Augmented Constituent Streams (SkelGym-Aug):**
   - `Transformer Mix` (T2.2, feature `mix`)
   - `AAGCN Bone 3D` (T4.2, feature `bone_3d`)
   - `AAGCN Rel 3D` (T4.3, feature `rel_3d`)
   - `AAGCN Joint Motion 3D` (T4.4, feature `joint_motion_3d`)
   - `AAGCN Bone Motion 3D` (T4.5, feature `bone_motion_3d`)

#### Lệnh Thực Thi:
```bash
# Cách 1: Chạy tuần tự / đa luồng chuẩn (Kèm cờ --push_to_hf)
python scripts/run_multi_seed_experiments.py --device auto --epochs 100 --include_baselines --seeds 42 123 3407 --push_to_hf

# Cách 2: Chạy song song tốc độ cao trên Marimo Server (4 workers song song, tự động keepalive & push HF)
python scripts/marimo_parallel_trainer.py --parallel 4 --epochs 100 --push_to_hf
```

---

### Phase 1C: Multi-Seed Augmentation Ablation Studies (Paper Table 2 & Table 3)

*Mục tiêu:* Phân tích định lượng sự cần thiết và đóng góp độc lập của từng phép tăng cường trên `Transformer mix` qua 3 seeds (42, 123, 3407), tự động upload lên HF Hub.

#### 1. Leave-One-Out (LOO) Suite (Table 2 — 7 Cấu hình $\times$ 3 Seeds):
- Full 5-op, -Mirror, -Yaw, -Scale, -TimeWarp (SkelGym-Aug 4-op), -Jitter, Clean Baseline.
```bash
python scripts/run_augmentation_experiments.py --mode loo --seeds 42 123 3407 --force_retrain --push_to_hf
```

#### 2. Single-Component Isolation Suite (Table 3 — 8 Cấu hình $\times$ 3 Seeds):
- Clean Baseline, +Mirror, +Yaw, +Scale, +TimeWarp, +Jitter, SkelGym-Aug 4-op, Full 5-op.
```bash
python scripts/run_augmentation_experiments.py --mode single --seeds 42 123 3407 --force_retrain --push_to_hf
```

---

### Phase 2: Đóng Băng Thống Kê & Tối Ưu Hóa Trọng Số SLSQP

```bash
python scripts/freeze_reference_artifacts.py --seeds 42 123 3407
```
- Đóng băng z-score chuẩn hóa từ tập Train sạch.
- Tối ưu hóa trọng số SLSQP độc lập trên tập Validation cho cả 3 seeds (lưu trữ tại `artifacts/reference/`).

---

### Phase 3: So Sánh 5 Phương Pháp Late Fusion (Paper Table 5 & Table 7)

Đánh giá 5 phương pháp Late Fusion (Hard Voting, Uniform Soft, Accuracy-Weighted Soft, Stacking Meta-Classifier, SLSQP Soft Voting) trên 4 tổ hợp Ensemble:
1. Two-Stream AAGCN (Joint + Bone)
2. Four-Stream AAGCN (4 GCN streams)
3. SkelGym-Lite (Transformer Mix + AAGCN Bone)
4. SkelGym-Full (Transformer Mix + 4 AAGCN streams)

```bash
python scripts/evaluate_local_ensemble.py --seeds 42 123 3407 --all_methods
```

---

### Phase 4: Kiểm Định Thống Kê & Video-Cluster Bootstrap (Paper Table 8 & Table 9)

```bash
python scripts/compute_statistical_tests.py --bootstrap_resamples 1000 --seeds 42 123 3407
```
- Xuất kiểm định McNemar (Window-level) và Wilcoxon / Paired t-test (Video-level).
- Tính toán khoảng tin cậy 95% Confidence Interval từ 1,000 lượt bootstrap.

---

### Phase 5: Phân Tích Chi Tiết 22 Class & Cơ Chế Lỗi (Paper Table 10 & Table 13)

```bash
python scripts/evaluate_local_ensemble.py --per_class --error_taxonomy
```

---

### Phase 6: Đo Đạc Độ Phức Tạp & Latency Phần Cứng (Paper Table 11)

```bash
python scripts/benchmark_hardware_latency.py --device auto --iterations 1000
```
- Đo FLOPs, tổng tham số, thời gian inference trên CPU, MPS và CUDA.

---

### Phase 7: Benchmark Mở Rộng Tập Dữ Liệu S&C (Paper Table 12 & External Benchmark)

```bash
python scripts/evaluate_external_benchmark.py --seeds 42 123 3407
```
- Đánh giá trên 4 bài tập Strength & Conditioning (Squat, Deadlift, Biceps Curl, Lateral Raise) theo Deyzel et al. (CVPRW 2023).

---

### Phase 8: Đồng Bộ Hóa & Biên Dịch Bài Báo (LaTeX Compilation)

```bash
python scripts/sync_ablation_results.py
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## 📊 Ma Trận Tiến Độ Tổng Thể (Master Tracking Matrix)

| Giai Đoạn | Nội Dung | Số Mô Hình / Thí Nghiệm | Tệp Script Chính & Cờ HF Hub | Trạng Thái |
| :--- | :--- | :---: | :--- | :---: |
| **Phase 0** | Unit Test Suite & Data Report | 51 tests | `pytest tests/` | **PASSED (51/51)** |
| **Phase 1A** | Feature Representation Screening | 27 models (Seed 42) | `server_runner.py --table table1 --push_to_hf` | **Pending** |
| **Phase 1B** | Multi-Seed Core Backbone Training | 10 models x 3 seeds | `run_multi_seed_experiments.py --push_to_hf` | **Pending** |
| **Phase 1C** | Multi-Seed Augmentation Ablation | 15 configs x 3 seeds | `run_augmentation_experiments.py --push_to_hf` | **Pending** |
| **Phase 2** | Freeze Reference & SLSQP Weights | 3 seeds | `freeze_reference_artifacts.py` | **Pending** |
| **Phase 3** | Late Fusion Comparison (5 methods) | 4 ensembles x 3 seeds | `evaluate_local_ensemble.py` | **Pending** |
| **Phase 4** | Statistical Testing & Bootstrap CI | 1,000 resamples | `compute_statistical_tests.py` | **Pending** |
| **Phase 5** | Per-Class Breakdown & Taxonomy | 22 classes | `evaluate_local_ensemble.py` | **Pending** |
| **Phase 6** | Hardware Latency & Complexity | 7 architectures | `benchmark_hardware_latency.py` | **Pending** |
| **Phase 7** | S&C Benchmark (Deyzel et al.) | 7 architectures | `evaluate_external_benchmark.py` | **Pending** |
| **Phase 8** | LaTeX Sync & PDF Compilation | Paper manuscript | `paper/paper.tex` | **Pending** |
