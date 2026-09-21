# SkelGym — Final Submission Checklist (Version 3 — Unfinished Items Only)

> **Mục tiêu**: Checklist tập trung 100% vào **những việc còn chưa hoàn thành** sau khi kiểm tra commit mới nhất. Loại bỏ hoàn toàn SkateFormer/CTR-GCN. Giữ lại đầy đủ từng chi tiết kỹ thuật, protocol và vị trí file cần xử lý theo phân cấp **🔴 P0 (Bắt buộc)** $\rightarrow$ **🟠 P1 (Cleanup quan trọng)** $\rightarrow$ **🟡 P2 (Audit số liệu toàn diện)**.

---

## 🔴 P0 — Cần Hoàn Thành Trước Khi Submit

### 1. External Benchmark / One-Shot Protocol

#### 1.1. Audit Lại Claim “1-shot transfer”
- **File cần kiểm tra**:
  - `scripts/evaluate_external_benchmark.py`
  - `paper/paper_eswa.tex`
  - `paper/paper_llncs.tex`
  - `README.md`
- **Vấn đề hiện tại**:
  Script đang lấy embedding từ các test video của chính SkelGym, sau đó:
  - Chọn 1 support video/class;
  - Dùng cosine similarity;
  - Classify các video còn lại.
  
  *Quy trình này phù hợp hơn với:* **1-shot classification simulation** (hoặc 1-shot recognition simulation) chứ chưa nên gọi mạnh là **1-shot transfer learning**.

- [x] Xác định chính xác experiment đang đo cái gì: Đo metric space generalizability của frozen representation mà không cần finetuning.
- [x] Ghi rõ source domain (22-class Gym dataset) và target domain (4 S&C classes).
- [x] Ghi rõ 4 classes được sử dụng (`squat`, `deadlift`, `barbell biceps curl`, `lateral raise`).
- [x] Ghi rõ số support videos/class: 1 exemplar per class (4 support videos total per trial).
- [x] Ghi rõ số query videos/trial: 50 query videos per trial (tổng 54 test videos).
- [x] Ghi rõ $K=1$.
- [x] Ghi rõ số trial = 100.
- [x] Ghi rõ random seed = 42.
- [x] Ghi rõ classifier = cosine nearest prototype/support trên $L_2$-normalized mean temporal embeddings.
- [x] Xác nhận không fine-tuning trong experiment này (zero-adaptation / frozen backbone).
- [x] Nếu không có adaptation/fine-tuning, không gọi là transfer learning.
- [x] Đổi terminology thành **1-shot classification simulation** (hoặc 1-shot recognition simulation) trong cả 2 bản TeX, README và checklist.
- [x] Giải thích rõ ràng metric matching qua cosine similarity.

#### 1.2. Đồng Bộ Số Liệu 1-Shot
Đã đồng bộ hóa 100% về một bộ số liệu duy nhất:
- Mean accuracy chính thức: **90.36%**
- Standard deviation: **7.35%**
- 95% Confidence Interval: **[68.95%, 98.00%]**
- Maximum single trial peak: **98.00%**

- [x] Chạy lại experiment từ commit mới nhất (`scripts/evaluate_external_benchmark.py`).
- [x] Xác định mean accuracy chính thức: **90.36%**.
- [x] Xác định standard deviation: **7.35%**.
- [x] Xác định 95% CI: **[68.95%, 98.00%]**.
- [x] Xác định maximum trial: **98.00%**.
- [x] Chọn một bộ số liệu duy nhất.
- [x] Update Table 8.
- [x] Update Section External Evaluation.
- [x] Update Abstract (nếu có claim liên quan).
- [x] Update `README.md`.
- [x] Xóa hoàn toàn số `97.32%` khỏi toàn bộ codebase và bài báo.

---

### 2. External Benchmark — Deadlift/Squat Claim

Script mới đã kiểm tra khá kỹ, nhưng paper cần biến kết quả thành protocol rõ ràng trong bản thảo.

- [x] Ghi rõ số lượng external videos: 54 held-out test videos.
- [x] Ghi rõ số lượng windows: 529 windows.
- [x] Ghi rõ số class được overlap giữa hai dataset: 4 classes (`squat`, `deadlift`, `barbell biceps curl`, `lateral raise`).
- [x] Ghi rõ định nghĩa Open-Set evaluation (suy luận trực tiếp từ không gian 22 lớp gốc).
- [x] Ghi rõ định nghĩa Closed-Set evaluation (tái chuẩn hóa phân phối xác suất hạn chế trong 4 lớp S&C).
- [x] Giải thích tại sao cần cả Open-Set và Closed-Set: Closed-Set đối sánh trực tiếp với Deyzel et al., Open-Set chứng minh không rò rỉ xác suất sang 18 lớp khác.
- [x] Ghi rõ Video-level aggregation (mean softmax across windows).
- [x] Ghi rõ Window-level aggregation.
- [x] Xác minh lại claim Deadlift recall $30\% \to 90\%$.
- [x] Xác minh chính xác:
  - [x] Window recall trước (ST-GCN baseline: 53.7% deadlift, 0.0% squat);
  - [x] Window recall sau (Transformer Mix: 53.7% deadlift, 80.7% squat; SkelGym-Full: 67.2% deadlift, 81.5% squat);
  - [x] Video recall trước (ST-GCN baseline: 40.0% deadlift, 0.0% squat; y văn Deyzel báo cáo baseline ~30.0%);
  - [x] Video recall sau (Transformer Mix: 80.0% deadlift, 93.3% squat; SkelGym-Full: 80.0% deadlift, 86.7% squat, lên tới 90.0% trong closed-set consensus).
- [x] Không dùng một con số $30\% \to 90\%$ nếu nó đến từ hai metric khác nhau (đã tách bạch rõ ràng metric video recall và window recall).
- [x] Đưa protocol vào Methods/Experimental Setup (Section 4.3).

> [!NOTE]
> **Đã nghiệm thu**: Đã giải thích tường minh trong Section 4.3 của cả hai bản `paper_eswa.tex` và `paper_llncs.tex`.

---

## 🟠 P1 — Cleanup Quan Trọng Trước Submission

### 3. Chuẩn Hóa Latency Benchmark

- **File liên quan**:
  - `scripts/benchmark_hardware_latency.py`
  - `paper/paper_eswa.tex`
  - `paper/paper_llncs.tex`
  - `README.md`

- [x] Chốt số warm-up runs chính thức: 50 runs.
- [x] Chốt số timing iterations chính thức: 500 timed runs (với per-pass timestamping).
- [x] Đồng bộ checklist với code: 50 warm-up, 500 timed runs.
- [x] Chốt device chính thức báo cáo trong paper: Host CPU (0.42--4.33 ms), CUDA (0.08--0.54 ms), Apple Silicon MPS (1.11--9.91 ms).
- [x] Chốt batch size: batch size = 1.
- [x] Chốt input shape chuẩn $T=32$ frames.
- [x] Chốt precision/dtype: `torch.float32`.
- [x] Có synchronization trước khi bấm giờ timing (`torch.cuda.synchronize()` hoặc `torch.mps.synchronize()`).
- [x] Có synchronization sau khi bấm giờ timing.
- [x] Loại bỏ hoàn toàn giai đoạn warm-up khỏi thống kê.
- [x] Báo cáo đầy đủ: Mean, Median, P95 percentile latency.
- [x] Phân biệt rõ ràng 3 cấp độ trễ:
  - [x] Model latency (chỉ riêng forward pass của classifier: CPU 0.42--4.33 ms, CUDA 0.08--0.54 ms)
  - [x] Feature extraction latency (~0.08 ms)
  - [x] End-to-end latency (MediaPipe Pose ~8--15 ms + Normalization + Classifier -> ~10--18 ms)
- [x] Không gọi estimated latency là measured latency.
- [x] Ghi rõ MediaPipe latency là edge reference / literature benchmark.
- [x] Đồng bộ số liệu latency giữa Paper, README, Tables, Figures, Cover Letter.
- [x] Kiểm tra tính hợp lệ của claim real-time (tất cả đều dưới 33.3 ms budget, đạt 71--2,000+ FPS).
- [x] Làm rõ con số `1.01 ms` (đo trên Apple Silicon MPS trong cấu hình un-synchronized/early baseline; đã bổ sung bảng đo đạc chính xác chi tiết cho từng thiết bị).

**Output chuẩn đã cập nhật vào Table 12 của bài báo:**

| Model | Params | FLOPs / MACs | CPU Mean (ms) | MPS Mean (ms) | CUDA Mean (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Transformer Mix** | 399K | **12.50 MFLOPs** | 0.42 ms | 1.11 ms | 0.08 ms |
| **AAGCN Single Stream** | 378K | **101.43 MFLOPs** | 0.98 ms | 2.11--2.31 ms | 0.11 ms |
| **SkelGym-Lite (2 Models)** | 777K | **113.93 MFLOPs** | 1.40 ms | 3.42 ms | 0.19 ms |
| **SkelGym-Full (5 Streams)** | 1.91M | **418.21 MFLOPs** | 4.33 ms | 9.91 ms | 0.54 ms |

---

### 4. Reproducibility

- **File liên quan**:
  - `README.md`
  - `run.py`
  - `scripts/evaluate_local_ensemble.py`
  - `scripts/evaluate_external_benchmark.py`

#### Dataset / Path
- [x] Kiểm tra đường dẫn `data/Final_dataset_metadata.csv`.
- [x] Kiểm tra fallback tự động tải từ Hugging Face.
- [x] Đảm bảo local path và HF path dùng cùng version metadata.
- [x] Đảm bảo checkpoint version khớp với paper.
- [x] Đảm bảo skeleton data version khớp với metadata.
- [x] Không để README hướng dẫn một path nhưng code sử dụng path khác.

#### Repository Structure
README phải phân biệt rõ:
```text
GitHub
├── source code
├── scripts
├── configuration
└── paper

Hugging Face
├── skeleton coordinates
├── metadata/data artifacts
└── model checkpoints
```

#### Reproduction Commands
- [x] Người mới clone repo có thể biết phải chạy command nào.
- [x] Có command reproduce final result: `python scripts/evaluate_local_ensemble.py`.
- [x] Có command evaluate local test set.
- [x] Có command evaluate external benchmark: `python scripts/evaluate_external_benchmark.py`.
- [x] Có command benchmark latency: `python scripts/benchmark_hardware_latency.py --device auto`.
- [x] Có thông tin dependency / version (`requirements.txt`).
- [x] Có fixed random seed ($42$).
- [x] Có khai báo hardware / software environment.

> [!NOTE]
> **Đã nghiệm thu**:
> - README không tuyên bố raw video được public bừa bãi; ghi rõ academic fair-use và không tái phân phối file raw video.
> - Tất cả số liệu trong `README.md` khớp hoàn toàn 100% với paper.

---

### 5. Statistical Significance — Đồng Bộ Toàn Bộ Tài Liệu

- **File liên quan**:
  - `paper/paper_eswa.tex`
  - `paper/paper_llncs.tex`
  - `paper/cover_letter.tex`
  - `README.md`

- [x] Abstract chỉ claim những comparison thực sự được kiểm định.
- [x] Results ghi rõ test nào được sử dụng.
- [x] Ghi rõ:
  - [x] McNemar’s test — window level
  - [x] Wilcoxon signed-rank test — video level
- [x] Ghi rõ sample unit:
  - [x] 2,743 windows
  - [x] 233 videos
- [x] Ghi rõ Bonferroni correction nếu đang sử dụng.
- [x] Kiểm tra ngưỡng $\alpha_{\text{adj}} = 0.01$.
- [x] Kiểm tra tất cả $p$-values ($p < 10^{-11}$, $p < 0.005$).
- [x] **Không dùng câu**: *"All improvements are statistically significant."*
- [x] Cover Letter phải dùng cùng wording chuẩn với paper.
- [x] Kiểm tra effect size nếu đã báo cáo.
- [x] Đảm bảo mỗi claim significance đều có corresponding statistical test.

**Cách diễn đạt chuẩn đã áp dụng đồng bộ:**
> *"Key architectural comparisons were statistically evaluated using McNemar’s test at the window level ($N=2,743, p < 10^{-11}$) and Wilcoxon signed-rank tests at the video level ($N=233, p < 0.005$) against a Bonferroni-adjusted threshold of $\alpha_{\text{adj}} = 0.01$."*

---

### 6. Dataset Licensing

- **File liên quan**:
  - `paper/paper_eswa.tex`
  - `paper/paper_llncs.tex`
  - `README.md`
  - `paper/cover_letter.tex`

- [x] Phân biệt rạch ròi giữa Raw videos và Derived skeleton data.
- [x] Không nói toàn bộ raw videos được public nếu không redistribute.
- [x] Kiểm tra quyền sử dụng từng nguồn:
  - [x] YouTube
  - [x] Pexels
  - [x] Freepik
  - [x] Author-recorded (Informed consent under Helsinki Declaration)
  - [x] Public dataset (Abdillah 2023)
- [x] Ghi rõ raw videos không được redistribute nếu đúng (academic fair-use, withheld from public redistribution).
- [x] Ghi rõ các thành phần được release công khai:
  - [x] Skeleton coordinates (13 landmarks, 32 frames)
  - [x] Metadata manifests
  - [x] Segment information (`.txt`)
  - [x] Checkpoints
  - [x] Source code
- [x] Kiểm tra claim CC BY 4.0 áp dụng cho toàn bộ derived skeleton package.
- [x] Kiểm tra claim MIT License cho code / checkpoints.

> [!NOTE]
> **Đã cập nhật câu chuẩn**:
> *"All source code, extracted skeletal representations, metadata, and pretrained checkpoints used for reproducibility are publicly available."*

---

### 7. Privacy Wording

- **File liên quan**:
  - `paper/paper_eswa.tex`
  - `paper/paper_llncs.tex`
  - `README.md`
  - `paper/cover_letter.tex`

- [x] Thay `privacy-preserving` bằng `privacy-aware`.
- [x] Không tuyên bố skeleton hoàn toàn không chứa thông tin cá nhân.
- [x] Nêu rõ skeleton vẫn có thể chứa:
  - [x] Body proportions
  - [x] Movement patterns
  - [x] Coarse gait / motion characteristics
- [x] Claim chính: **giảm exposure của raw RGB information bằng cách xử lý ephemerally trong RAM và hủy ngay lập tức**.
- [x] Đồng bộ wording giữa Abstract / Introduction / Discussion / Conclusion / README / Cover Letter.

---

## 🟡 P2 — Audit Số Liệu Toàn Diện

### 8. README Final Consistency Audit — 🟢 Đã Hoàn Thành

Đã tìm kiếm toàn repository và xác nhận:
- [x] Không còn bootstrap CI cũ (đã cập nhật $[68.46\%, 71.75\%]$ và $[72.09\%, 83.26\%]$).
- [x] Không còn `97.32%` (đã cập nhật chính thức Mean $90.36\% \pm 7.35\%$, peak $98.00\%$).
- [x] Không còn số liệu latency cũ hay mâu thuẫn (đã phân tách rõ ràng classifier CPU 0.42--4.33 ms, CUDA 0.08--0.54 ms, MPS 1.11--9.91 ms).
- [x] Không còn statistical claim cũ overclaim (chỉ claim các cặp kiểm định đạt $p < 10^{-11}$ và $p < 0.005$ với Bonferroni correction).
- [x] Không còn thuật ngữ `SOTA` tự xưng.
- [x] Không còn `Grand 5-Stream SOTA`.
- [x] Không còn `zero temporal leakage` (thay bằng `strict video-level partitioning with no source-video overlap across splits`).
- [x] Không còn claim raw dataset public không chính xác (ghi rõ academic fair-use và không tái phân phối raw videos).
- [x] Không còn claim privacy tuyệt đối (dùng `privacy-aware`, thừa nhận giới hạn về tỷ lệ thân hình và dáng đi thô).

---

### 9. Final Paper Numerical Audit — 🟢 Đã Nghiệm Thu Hoàn Toàn

Toàn bộ các con số đã được đối soát chéo và khớp 100% giữa Code, Scripts, Tables, TeX, README và Cover Letter:

#### Core Metrics
- [x] Window Accuracy = **70.11%**
- [x] Window Macro-F1 = **0.6858**
- [x] Video Accuracy = **77.68%**
- [x] Video Macro-F1 = **0.7649**
- [x] Test windows = **2,743**
- [x] Test videos = **233**
- [x] Params = **1.91M** (Full Ensemble) / **777K** (Lite) / **399K** (Transformer Mix)

#### Statistical Metrics
- [x] Window CI = $[68.46\%, 71.75\%]$ (Bootstrap Mean: $70.14\%$)
- [x] Video CI = $[72.09\%, 83.26\%]$ (Bootstrap Mean: $77.65\%$)
- [x] Khoảng CI bắt buộc chứa point estimate (Cả $70.11\%$ và $77.68\%$ đều nằm ở trung tâm của khoảng tin cậy).

#### Architecture
- [x] 1 Transformer (Mix 117-d)
- [x] 4 AAGCN streams (Bone 3D, Relative 3D, Joint Motion 3D, Bone Motion 3D)
- [x] 5 independent models trained separately
- [x] SLSQP late fusion
- [x] Validation-calibrated weights under simplex constraints ($\sum w_i = 1, w_i \ge 0$)

#### Efficiency
- [x] FLOPs thống nhất: Transformer 12.50 MFLOPs, AAGCN Stream 101.43 MFLOPs, SkelGym-Lite 113.93 MFLOPs, SkelGym-Full 418.21 MFLOPs.
- [x] Params thống nhất: Transformer 399K, AAGCN Stream 378K, Lite 777K, Full 1.91M.
- [x] Latency thống nhất: CPU 0.42--4.33 ms, CUDA 0.08--0.54 ms, MPS 1.11--9.91 ms.
- [x] Hardware thống nhất: Apple Silicon M4 GPU (MPS) / CPU, NVIDIA RTX PRO 6000 (CUDA).

#### Dataset Statistics
- [x] 1,024 source videos (10.2 GB).
- [x] 22 classes.
- [x] 6:2:2 video-level split (580 train, 208 val, 236 test / 233 valid $\ge 32$ frames).
- [x] Window generation thực hiện nghiêm ngặt sau video-level split.
- [x] 32-frame window ($T=32$).
- [x] Train stride 16 ($S=16$, 13,136 windows).
- [x] Val/Test stride 32 ($S=32$, Val: 2,075 windows, Test: 2,743 windows).

---

## 🟢 Những Task Đã Hoàn Thành (Đóng Hoàn Toàn)

Toàn bộ các hạng mục sau đã được nghiệm thu và đánh dấu **DONE**:
- [x] Bootstrap CI methodology (khắc phục hoàn toàn anomaly điểm ước lượng nằm ngoài CI).
- [x] Xóa self-claimed SOTA trong code và văn bản.
- [x] Làm rõ kiến trúc 5-stream late fusion.
- [x] Sửa leakage wording sang strict video-level partitioning.
- [x] FLOPs/MACs lý thuyết tính toán qua `thop`.
- [x] Core 22-class experiment chuẩn hóa.
- [x] 117-d biomechanical compound representation.
- [x] SkelGym-Augmentation pipeline.
- [x] SLSQP validation calibration under simplex constraints.
- [x] Video consensus aggregation (soft voting).
- [x] Class-level biomechanical error analysis.
- [x] External S&C benchmark và 1-shot classification simulation.
- [x] Tách bạch rõ rệt và chính xác Deadlift/Squat recall ($40\% \to 80\%-90\%$ video recall).
- [x] Đồng bộ hóa Licensing (Raw video fair-use vs CC BY 4.0 skeletons vs MIT codebase).
- [x] Đồng bộ hóa Privacy-aware wording toàn diện.

---

## 🎯 Checklist Cuối Cùng (Action Plan 12 Bước) — 🟢 100% Complete

```text
[x] 1. Chốt lại 1-shot experiment
      └─ Đổi terminology thành 1-shot classification simulation (zero-adaptation)
      └─ Chốt 1 bộ số liệu duy nhất: Mean 90.36% ± 7.35%, CI [68.95%, 98.00%], Peak 98.00%
[x] 2. Chốt External Benchmark
      └─ Deadlift/Squat: ST-GCN 40% -> Transformer 80%, SkelGym 80%-90% video recall
      └─ Open-set / Closed-set định nghĩa rõ ràng
      └─ Video/Window protocol chi tiết
[x] 3. Chốt Latency
      └─ 50 warm-up, 500 timing iterations
      └─ Đồng bộ sync trước/sau timing
      └─ Báo cáo mean / median / p95
      └─ Phân định model vs feature extraction vs end-to-end (MediaPipe)
[x] 4. Đồng bộ Reproducibility
      └─ GitHub (code) vs Hugging Face (data/checkpoints)
      └─ Metadata & Checkpoints versioning đồng nhất
      └─ Lệnh reproduce hoàn chỉnh
[x] 5. Sửa Statistical wording
      └─ Paper & Cover Letter: McNemar (p < 10^-11), Wilcoxon (p < 0.005), Bonferroni alpha = 0.01
[x] 6. Sửa Licensing wording (Raw video fair-use vs CC BY 4.0 Skeletons vs MIT Code)
[x] 7. Sửa Privacy wording (Privacy-aware thay cho claim bảo mật tuyệt đối)
[x] 8. README numerical audit (Khớp 100% với paper)
[x] 9. Full paper numerical audit (Khớp 100% chéo các file)
[x] 10. Compile final packages (Overleaf ZIPs: paper_eswa_overleaf.zip & paper_llncs_overleaf.zip)
[x] 11. Search toàn repo lần cuối (Đã sạch toàn bộ legacy terms)
[x] 12. Final submission package (manuscript ESWA/LNCS, cover letter, overleaf packages, checklist)
```

> [!NOTE]
> Tất cả 12 bước trong kế hoạch hành động đã được hoàn tất và nghiệm thu toàn diện. Toàn bộ repository và bản thảo bài báo hiện đã đạt trạng thái **100% Submission-Ready**!
