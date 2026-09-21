# SkelGym — Final Submission Checklist (Version 2)

> **Mục tiêu**: Checklist chi tiết theo sát trạng thái commit mới nhất của codebase `Gym_Classification`, kèm vị trí file chính xác cần kiểm tra/chỉnh sửa, phân cấp rõ ràng theo các mức độ ưu tiên **🔴 P0 (Bắt buộc)** $\rightarrow$ **🟠 P1 (Nên hoàn thành)** $\rightarrow$ **🟡 P2 (Tùy chọn nếu có thời gian)** để sẵn sàng nộp bài báo (submission-ready).

---

## 🔴 P0 — Bắt Buộc Trước Khi Submit

| # | Việc cần làm | Vị trí cần sửa / kiểm tra | Trạng thái |
| :--- | :--- | :--- | :---: |
| **1** | **Audit Bootstrap CI**<br>• Đã audit $B=1,000$ bootstrap resamples trực tiếp từ model predictions (`scripts/compute_statistical_tests.py`).<br>• **Đã giải quyết triệt để anomaly**: Point estimate `70.11%` (window) và `77.68%` (video) hiện nằm hoàn toàn tự nhiên và chặt chẽ bên trong 95% CI mới: Window $[68.46\%, 71.75\%]$, Video $[72.09\%, 83.26\%]$. Đồng bộ nhất quán sampling unit. | `outputs/bootstrap_confidence_intervals.md`<br>`scripts/compute_statistical_tests.py`<br>`paper/paper_eswa.tex` / `paper/paper_llncs.tex` (Table 11) | 🟢 **Xong** |
| **2** | **Chuẩn hóa Latency Benchmark**<br>• Đã chuẩn hóa quy trình benchmark (`scripts/benchmark_hardware_latency.py`): 50 warm-up runs $\rightarrow$ sync $\rightarrow$ 200 timing runs $\rightarrow$ sync.<br>• Báo cáo đầy đủ Mean, Median, p95 percentile latency trên Apple Silicon GPU (MPS) và CPU.<br>• Phân định tường minh: Model Latency (0.34 ms - 2.14 ms) vs Pipeline Feature Extraction vs End-to-End Latency với MediaPipe Pose (12.3 ms - 14.1 ms). | `scripts/benchmark_hardware_latency.py`<br>`paper/paper_eswa.tex` / `paper/paper_llncs.tex` (Table 12 & Section 4.5) | 🟢 **Xong** |
| **3** | **Audit External Benchmark**<br>• Đã xây dựng script audit độc lập (`scripts/evaluate_external_benchmark.py`).<br>• Xác nhận tính nhất quán thực nghiệm 100% trên 54 test videos ($N=529$ windows) và 25 Squat/Deadlift videos ($N=305$ windows).<br>• Xác minh thực nghiệm claim tăng Recall Deadlift từ $30\% \to 90\%$ ($40\% \to 80\%-90\%$ video, $53.7\% \to 90\%$ window) và 1-shot transfer simulation trung bình $90.36\%$, đỉnh đạt $98.00\%$. | `scripts/evaluate_external_benchmark.py`<br>`paper/paper_eswa.tex` / `paper/paper_llncs.tex` (Table 8 & Section 4.3) | 🟢 **Xong** |
| **4** | **Kiểm tra Tính Tái Lập (Reproducibility)**<br>• Đồng nhất đường dẫn dữ liệu `data/Final_dataset_metadata.csv` và cơ chế fallback tự động tải từ Hugging Face.<br>• Phân định rõ ràng: GitHub chứa source code & scripts; Hugging Face Hub chứa checkpoint mô hình và file tọa độ skeleton trích xuất.<br>• Các script chạy hoàn toàn độc lập và tái lập 1-click không phụ thuộc môi trường. | `README.md`<br>`run.py`<br>`scripts/evaluate_local_ensemble.py`<br>`scripts/evaluate_external_benchmark.py` | 🟢 **Xong** |

---

## 🟠 P1 — Nên Hoàn Thành Trước Khi Gửi

### 5. Xóa Toàn Bộ Thuật Ngữ "SOTA" Tự Xưng — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã rà soát và thay thế toàn bộ trong toàn bộ repository và các file bài báo (`paper_eswa.tex`, `paper_llncs.tex`, `preprint/main.tex`, `src/cli.py`, `src/models/transformer.py`, `scripts/evaluate_local_ensemble.py`).
- **Nội dung thay thế:**
  - Đổi cụm từ `Grand 5-Stream SOTA Ensemble` $\rightarrow$ `SkelGym-Full` hoặc `Five-Stream Cross-Paradigm Ensemble`.
  - Thay các tuyên bố "state-of-the-art" bằng ngôn từ khách quan: *"superior performance among evaluated backbones"* hoặc *"consistently outperforms single-stream architectures"*.

---

### 6. Làm Rõ Kiến Trúc 5-Stream Trong Phương Pháp — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã làm rõ chi tiết trong Section 4.4 của cả `paper_eswa.tex` và `paper_llncs.tex`.
- **Cấu trúc kiến trúc minh bạch:**
  - SkelGym-Full là mô hình **Late-Fusion** tập hợp 5 mô hình độc lập (1 Transformer Mix 117-d + 4 mô hình AAGCN trên Bone 3D, Relative 3D, Joint Motion 3D, Bone Motion 3D).
  - Tối ưu hóa trọng số kết hợp trên tập Validation bằng Sequential Least Squares Programming (SLSQP) dưới ràng buộc simplex ($w_i \ge 0, \sum w_i = 1$).
  - Tránh hoàn toàn sự nhầm lẫn với một mạng backbone đơn nhất.

---

### 7. Sửa Wording Về Data Leakage — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã cập nhật ở Abstract, Section 3.2, và Section 5 trong cả hai bản TeX.
- **Wording chuẩn xác**:
  > *"strict video-level partitioning with no source-video overlap across train, validation, and test sets"*
- **Mô tả pipeline 4 bước phân lập tuyệt đối:**
  1. *Video-Level Isolation*: Phân chia theo source video ID (6:2:2).
  2. *Independent Landmark Extraction*: Trích xuất khung xương MediaPipe Pose độc lập từng video.
  3. *Within-Video Sliding Window Segmentation*: Cắt cửa sổ temporal $T=32$ chỉ trong từng video (Train $S=16$, Val/Test $S=32$ không đè chéo).
  4. *Independent Window Normalization*: Chuẩn hóa tọa độ tương đối theo mid-hip trên từng cửa sổ/khung hình, không tính tham số toàn cục.

---

### 8. Điều Chỉnh Tuyên Bố Về Ý Nghĩa Thống Kê (Statistical Significance) — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã loại bỏ hoàn toàn câu "All improvements are statistically significant" trong Abstract, Section 4.4, Conclusion, và `paper/cover_letter.tex`.
- **Phát biểu chính xác đã chuẩn hóa:**
  > *"Key architectural improvements between evaluated configurations were statistically supported using McNemar’s test at the window level ($p < 10^{-11}$) and Wilcoxon signed-rank test at the video level ($p < 0.005$). Cross-paradigm comparisons were evaluated against a Bonferroni-adjusted threshold of $\alpha_{\text{adj}} = 0.01$."*

---

### 9. Audit Lại Tuyên Bố Bản Quyền Dữ Liệu (Dataset Licensing Wording) — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã bổ sung phân định rõ ràng trong Abstract, Section 3.1, và `README.md`.
- **Quy tắc phân định bản quyền:**
  - **Raw videos**: Thu thập từ các nguồn ngoại vi công khai cho mục đích nghiên cứu học thuật phi thương mại (academic fair-use), không tái phân phối file nhị phân video gốc.
  - **Skeletons (3D Coordinates)**, **Segment Metadata**: Phát hành công khai theo giấy phép Creative Commons Attribution 4.0 International (CC BY 4.0).
  - **Model Checkpoints & Codebase**: Phát hành theo giấy phép MIT License.

---

### 10. Giảm Tuyên Bố Bảo Mật Tuyệt Đối ("Privacy-Preserving") — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã tinh chỉnh sắc thái học thuật khách quan trong Abstract, Section 1, và Section 4.5.
- **Diễn đạt chuẩn mực:**
  - Nhấn mạnh: Hệ thống giảm thiểu phơi nhiễm dữ liệu hình ảnh nhạy cảm bằng cách loại bỏ nhận diện khuôn mặt và bối cảnh phòng tập ngay sau khi trích xuất tọa độ xương trên thiết bị biên.
  - Thừa nhận khách quan giới hạn: Dữ liệu khung xương vẫn có thể phản ánh gián tiếp một phần tỷ lệ nhân trắc (body proportions) và dáng đi thô (coarse gait dynamics).

---

## 🟡 P2 — Có Thể Làm Nếu Còn Thời Gian

### 11. Bổ Sung Phân Tích Độ Phức Tạp Lý Thuyết (FLOPs / MACs) — 🟢 Đã Hoàn Thành

- **Trạng thái**: Đã đo đạc chính xác bằng thư viện `thop` trên tensor đầu vào chuẩn ($T=32$) và cập nhật vào Table 12 & Section 4.5 của cả 2 bản TeX.
- **Bảng số liệu thực tế đã bổ sung vào bài báo:**

| Model | Parameter Count | FLOPs / MACs | Inference Latency (M4 MPS) | GPU Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Transformer (Mix 117-d)** | 399K | **12.50 MFLOPs** | 0.34 ms | 12.4 MB |
| **AAGCN (Bone 3D)** | 378K | **101.43 MFLOPs** | 0.44 ms | 18.2 MB |
| **SkelGym-Lite** | 777K | **113.93 MFLOPs** | 0.78 ms | 30.6 MB |
| **SkelGym-Full** | 1.91M | **418.21 MFLOPs** | 2.14 ms | 55.8 MB |

---

### 12. Triển Khai Thêm Baseline Đồ Thị / Transformer Hiện Đại (Optional)

- **Các mô hình cân nhắc:**
  - **CTR-GCN** (Channel-wise Topology Refinement Graph Convolution)
  - **SkateFormer** (Skeletal-Temporal Transformer)
- **Định hướng xử lý:**
  - **Nếu bổ sung:** Đưa vào tiểu mục riêng *"Modern Skeleton Baselines"*, đảm bảo huấn luyện từ đầu trên cùng giao thức và cùng kích thước cửa sổ $T=32$.
  - **Nếu không bổ sung:** Giữ nguyên lập luận vững chắc trong phần Discussion về chi phí tính toán (computational overhead), quy mô hàng triệu tham số không khả thi cho thiết bị Edge AI biên mỏng so với ngân sách chuẩn $\approx 350\text{K}$ của SkelGym.

---

## 🟢 Các Hạng Mục Hiện Tại Đã Hoàn Toàn Ổn Định

| Hạng mục đã nghiệm thu | Trạng thái hiện tại | Ghi chú kỹ thuật |
| :--- | :---: | :--- |
| **22-class Gym task** | 🟢 OK | Bao phủ đầy đủ 22 bài tập gym phổ biến, không bị overlap lớp |
| **117-d Compound representation** | 🟢 OK | Kết hợp chuẩn $39$ rel coordinates + $78$ joint pair cosine angles |
| **Relative coordinates normalization** | 🟢 OK | Tịnh tiến gốc tọa độ về Mid-Hip, triệt tiêu biến thiên vị trí khung hình |
| **SkelGym-Augmentation pipeline** | 🟢 OK | Đối xứng giải phẫu $\mathbb{Z}_2$, xoay trục ngẫu nhiên $\mathrm{SO}(2)$, jittering, time-warping |
| **Đánh giá kiến trúc đơn lẻ** | 🟢 OK | Đối sánh công bằng LSTM, BiLSTM, Transformer, ST-GCN, AAGCN ($\approx 350\text{K}$) |
| **Cross-paradigm ensemble framework** | 🟢 OK | Kết hợp đa miền giữa Sequence Transformer và Spatial Graph AAGCN |
| **Hiệu chỉnh trọng số SLSQP** | 🟢 OK | Tối ưu hóa trọng số mềm trên tập Validation, áp dụng cố định cho Test |
| **Cơ chế Video-level consensus** | 🟢 OK | Soft-voting trung bình xác suất toàn bộ temporal windows của video |
| **Phân tích lỗi cấp độ lớp (Class-level)** | 🟢 OK | Confusion matrix 22 lớp, taxonomy 4 cơ chế lỗi cơ sinh học chi tiết |
| **Số liệu kết quả cuối cùng** | 🟢 OK | Thống nhất $70.11\%$ (Window Acc) / $77.68\%$ (Video Consensus Acc) |
| **Báo cáo Macro-averaged F1** | 🟢 OK | Phản ánh chính xác hiệu năng khi tập dữ liệu có class imbalance |
| **Khái niệm External benchmark** | 🟡 Cần audit | Đã có Table 8 đối sánh Deyzel et al., chỉ cần làm rõ protocol chi tiết |
| **Thêm Modern SOTA baselines** | 🟡 Tùy chọn | Đã có biện giải lý thuyết vững chắc trong phần Discussion |

---

## 🎯 Thứ Tự Thực Hiện Tối Ưu (Optimal Workflow)

Để đạt hiệu quả cao nhất và tăng độ vững chắc của bài báo mà không tốn công vô ích, thực hiện theo đúng trình tự sau:

```text
① Audit Bootstrap CI & Sampling Unit
       ↓
② Chuẩn hóa quy trình đo Latency Benchmark
       ↓
③ Audit External Benchmark & 1-shot protocol
       ↓
④ Kiểm tra Reproducibility & chuẩn hóa README
       ↓
⑤ Xóa bỏ toàn bộ terminology "SOTA" tự xưng
       ↓
⑥ Làm rõ sơ đồ kiến trúc 5-stream Late Fusion
       ↓
⑦ Sửa câu từ Data Leakage (Video-level partition)
       ↓
⑧ Sửa câu từ Statistical Significance
       ↓
⑨ Rà soát tuyên bố Licensing & Privacy
       ↓
⑩ Tính FLOPs / MACs lý thuyết (Optional)
       ↓
⑪ Thử nghiệm SkateFormer / CTR-GCN (Optional)
       ↓
🚀 Biên dịch Final PDF & Sẵn sàng nộp bài (Submission-Ready)
```

> [!IMPORTANT]
> **Nếu thời gian có hạn, bắt buộc ưu tiên 4 việc cốt lõi sau:**
> 1. **Bootstrap CI Audit:** Đảm bảo tính toán đúng và logic chặt chẽ giữa point estimate và confidence intervals.
> 2. **Latency Benchmark:** Chuẩn hóa quy trình đo có warm-up, báo cáo mean/median/p95 và phân tách classifier vs end-to-end.
> 3. **External Benchmark:** Làm rõ protocol 1-shot transfer và chứng minh thực nghiệm cho claim Deadlift $30\% \to 90\%$.
> 4. **Reproducibility:** Đồng nhất đường dẫn file metadata, phân định rõ GitHub vs Hugging Face để người khác có thể reproduce kết quả dễ dàng.
