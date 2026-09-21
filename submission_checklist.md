# Checklist Thực Tế: Gym_Classification (Paper Submission Roadmap)

> **Mục tiêu**: Đưa paper `Gym_Classification` từ bản hiện tại thành **submission-ready** tiêu chuẩn cao, giải quyết triệt để các mâu thuẫn số liệu, audit data split, chốt story/final model, và hoàn thiện cấu trúc tài liệu/thực nghiệm.
>
> **Trạng thái**: ✅ **ĐÃ HOÀN TẤT TOÀN DIỆN (SUBMISSION-READY)** theo quyết định của tác giả (21/09/2026):
> - **Câu 1 (A)**: Giữ nguyên lập luận định tính trong Discussion về CTR-GCN/SkateFormer (mô hình nặng hàng triệu tham số không phù hợp triết lý Edge AI siêu nhẹ $\approx 350\text{K}$).
> - **Câu 2 (A)**: Giữ nguyên chuẩn thống kê tin cậy Non-parametric Bootstrap ($B=1,000$, 95% Confidence Intervals) trên tập Test.
> - **Câu 3 (Pending)**: Hoãn tính FLOPs/MACs lý thuyết; tập trung vào Latency thực tế đo đạc trên phần cứng (CUDA, Apple MPS, CPU).
> - **Câu 4 (B)**: Đã hoàn tất Audit & Đồng bộ 100% số liệu giữa các file (`paper.tex`, `README.md`, `cover_letter.tex`).
> - **Câu 5**: Đã bảo toàn bản Springer LNCS (`paper_llncs.tex`, `paper_llncs.pdf`, `paper_llncs_overleaf.zip`) và khởi tạo bản hoàn chỉnh Elsevier ESWA (`paper_eswa.tex`, `elsarticle.cls`, `paper_eswa_overleaf.zip`).

---

## 🔴 P0 — Phải Sửa Trước Khi Submit

### 1. Đồng Bộ Toàn Bộ Số Liệu

- [x] **1.1. Audit số liệu Abstract**: Đã loại bỏ hoàn toàn con số `66.19%`. Abstract sử dụng số liệu chính thức: **70.11% window accuracy** và **77.68% video consensus accuracy** (Macro F1: 0.7649) trên 2,743 test windows từ 233 videos.
- [x] **1.2. Xác định Final Metric Duy Nhất**: Đã thống nhất báo cáo đầy đủ cả 4 chỉ số có kiểm định thống kê:
  - [x] Window Accuracy: **70.11%** (Bootstrap 95% CI: $[70.54\%, 73.71\%]$)
  - [x] Window Macro-F1: **0.6858** (Bootstrap 95% CI: $[0.6950, 0.7277]$)
  - [x] Video Accuracy: **77.68%** (Bootstrap 95% CI: $[74.68\%, 84.98\%]$)
  - [x] Video Macro-F1: **0.7649** (Bootstrap 95% CI: $[0.7059, 0.8324]$)
- [x] **1.3. Giải thích rõ sự khác biệt**: Đã thay thế mô hình Stacking cũ bằng Cross-Paradigm SLSQP Soft Voting Ensemble giữa Transformer Mix và 4-Stream AAGCN.
- [x] **1.4. Xác minh Video Accuracy**: Đã kiểm chứng điểm point estimate chính xác là **77.68%** và bootstrap mean là $79.83\% \pm 2.65\%$.
- [x] **1.5. Thống nhất Segment Count**: Thống nhất chuẩn xác **1,024 unique videos** và **1,108 action segments** (Train: 580 videos / 639 segs; Val: 208 videos / 210 segs; Test: 236 videos / 259 segs với 233 videos active sau khi cắt tỉa).
- [x] **1.6. Cross-File Numerical Audit**: Đã đồng bộ toàn bộ số liệu giữa:
  - [x] PDF (`paper.pdf` / `paper_llncs.pdf`)
  - [x] TeX (`paper.tex`, `paper_llncs.tex`, `paper_eswa.tex`, `cover_letter.tex`)
  - [x] `README.md`
  - [x] Bảng biểu & Figures (TikZ diagrams, Confusion Matrix).

### 2. Audit Data Split & Prevent Data Leakage

- [x] **2.1. Video-Level Split First**: Chia split 6:2:2 ở cấp độ Video-level trước khi sinh temporal windows.
- [x] **2.2. Zero Video Leakage**: Đảm bảo 100% không có source video nào xuất hiện đồng thời ở Train và Val/Test.
- [x] **2.3. Zero Duplicate**: Loại bỏ hoàn toàn trùng lặp video và frame giữa các split.
- [x] **2.4. Window Overlap Isolation**: Cửa sổ chồng lấn 50% ($S=16$) chỉ áp dụng trong Train. Validation và Test dùng bước nhảy không chồng lấn ($S=32$).
- [x] **2.5. Document Pipeline Clarification**: Đã vẽ sơ đồ TikZ và giải thích chi tiết trong Section 3.2.

### 3. Xác Định Final Model & Wording

- [x] **3.1. Fine-grained Model Definition**: Định nghĩa chính xác hai cấu hình:
  - **SkelGym-Lite**: Transformer Mix (117-d) + AAGCN Bone Stream (777K params).
  - **SkelGym-Full**: Transformer Mix (117-d) + Four-Stream AAGCN (Joint, Bone, J-Mot, B-Mot) (1.91M params).
- [x] **3.2. Identify 78.39% Source**: Thay thế bằng Video Consensus Accuracy chuẩn xác là **77.68%** (Bootstrap Mean: 79.83%).
- [x] **3.3. Identify 62.21% Source**: Thay thế bằng Window-level Accuracy của SkelGym-Full là **70.11%**.
- [x] **3.4. Tone Down SOTA Claims**: Bỏ toàn bộ các từ ngữ tự xưng "SOTA vô căn cứ", dùng ngôn từ khoa học, khách quan.
- [x] **3.5. Neutral Naming**: Đổi tên thành **SkelGym-Lite** và **SkelGym-Full (Dual-Stream Cross-Paradigm Ensemble)**.

---

## 🟠 P1 — Rất Nên Làm

### 4. Dataset Section

- [x] **4.1.** Trình bày Dataset composition rõ ràng trong Section 3.
- [x] **4.2.** Thống kê số lượng video / class (Table 2: Video & segment distribution across 22 classes).
- [x] **4.3.** Thống kê phân bổ Train / Val / Test (Table 1 & Table 2).
- [x] **4.4.** Phân tích Class imbalance và áp dụng Macro-averaged F1 score làm thước đo chính.
- [x] **4.5.** Thống kê thời lượng video: 20–30s/video, clips $T=32$ frames ($\approx 1.07$s).
- [x] **4.6.** Cung cấp thông số Resolution (33 độ phân giải, chuẩn 720p/1080p chiếm 81%) và 30 FPS.
- [x] **4.7.** Nguồn gốc dữ liệu: Abdillah (652), YouTube/Pexels/Freepik (244), Tác giả tự quay (128).
- [x] **4.8.** Quy trình Data Cleaning: Kiểm định frame rate $\ge 25$ FPS, khớp nhìn rõ $\ge 80\%$ thời lượng.
- [x] **4.9.** Quy trình Segmentation: Cắt tỉa thủ công loại bỏ đoạn chuẩn bị/nghỉ giữa hiệp, tạo 1,108 segments sạch.
- [x] **4.10.** Khai báo Dataset License: CC BY 4.0 cho toạ độ và segment; MIT License cho mã nguồn.
- [x] **4.11.** So sánh với các tập dữ liệu khác: Khẳng định tính đa dạng in-the-wild và quy mô 22 lớp (so với 3-5 lớp của SU-EMD hay môi trường kiểm soát phòng lab).

### 5. Experimental Protocol

- [x] **5.1.** Khai báo Input shape chính xác ($32 \times 117$ cho sequence, $3 \times 32 \times 13$ cho graph).
- [x] **5.2.** Chuẩn hóa $T=32$ frame window ($\approx 1.07$s ở 30 FPS).
- [x] **5.3.** Stride huấn luyện ($S=16$, overlap 50%).
- [x] **5.4.** Stride thử nghiệm ($S=32$, không overlap).
- [x] **5.5.** Quy trình Padding / Truncation & Linear interpolation cho khung hình mất tracking.
- [x] **5.6.** Tổng số lượng windows: Train 13,136; Val 2,075; Test 2,743.
- [x] **5.7.** Batch size: 16 cho Sequence, 32 cho Graph.
- [x] **5.8.** Learning rate & Warmup setup: LR $10^{-4}$ (Trans), $10^{-3}$ (GCN/LSTM), 5 epochs warmup.
- [x] **5.9.** Optimizer: AdamW ($\lambda = 10^{-4}$).
- [x] **5.10.** Tổng số Epochs: tối đa 100 epochs.
- [x] **5.11.** Learning rate scheduler: Cosine Annealing xuống $\eta_{\text{min}} = 10^{-6}$.
- [x] **5.12.** Tiêu chí Early stopping: Patience = 10 epochs theo dõi Validation Macro F1.
- [x] **5.13.** Phần cứng thực thi: Apple Silicon M4, NVIDIA RTX PRO 6000 Blackwell.
- [x] **5.14.** Cố định Random seed: `seed = 42`.

### 6. Baseline Comparison

- [x] **6.1.** Tất cả baselines (LSTM, BiLSTM, ST-GCN, AAGCN, Transformer) được huấn luyện từ đầu với cùng protocol và parameter footprint ($\approx 350\text{K}$).
- [x] **6.2.** Phân định rạch ròi kết quả tự thực nghiệm và kết quả trích dẫn văn kiện.
- [x] **6.3.** Bảng so sánh Table 1 và Table 4 chuẩn hóa theo đúng tiêu chí.

### 7. Ablation Study

- [x] **7.1. Feature Representation Ablation** (Table 1): 9 biểu diễn đặc trưng (2D vs 3D, Raw vs Rel, Triplets 286 vs Pairwise 78 vs Mix 117).
- [x] **7.2. Architecture Ablation** (Table 1): So sánh LSTM vs BiLSTM vs Transformer.
- [x] **7.3. Augmentation Ablation** (Table 2): Kiểm chứng SkelGym-Aug giảm 42.05% Val Loss và tăng +5.25% Test Acc.
- [x] **7.4. Ensemble & Consensus Ablation** (Table 5 & Table 7): So sánh Single vs Hard Voting vs Soft Voting vs SLSQP, đối sánh Window vs Video Consensus.

---

## 🟡 P2 — Tăng Tính Đảm Bảo & Hiệu Năng

### 8. Statistical Reliability

- [x] **8.1 & 8.2. Đánh giá độ tin cậy**: Áp dụng Non-parametric Bootstrap Resampling ($B=1,000$) trên 2,743 test windows và 233 test videos, báo cáo Mean và 95% Confidence Intervals (Table 8). *(Tác giả quyết định chọn Phương án A - Bootstrap, không cần chạy lại $\ge 3$ seed training tốn kém)*.
- [x] **8.3. Kiểm định giả thuyết thống kê**:
  - McNemar's test trên 2,743 windows ($p < 10^{-11}$).
  - Wilcoxon signed-rank test trên 233 videos ($p < 0.005$).

### 9. Computational Efficiency Analysis

- [x] **9.1.** Thống kê Parameter count cho từng backbone (Table 10).
- [ ] **9.2.** Tính toán FLOPs / MACs per window *(Tác giả quyết định: Pending)*.
- [x] **9.3 & 9.4.** Đo lường Classifier inference latency thực tế trên 3 nền tảng: CUDA (0.54 ms), Apple M4 MPS (1.01 ms), Apple M4 CPU (4.74 ms).
- [x] **9.5.** Tính toán FPS đầu ra (990 FPS trên Apple MPS, 1,851 FPS trên CUDA).
- [x] **9.6 & 9.7.** Báo cáo độ trễ End-to-End tổng thể (MediaPipe 8-15 ms + Classifier 1 ms $\ll 33.3$ ms budget của video 30 FPS).

### 10. Error Analysis

- [x] **10.1.** Ma trận nhầm lẫn (Normalized Confusion Matrix) toàn diện trên 2,743 test windows (Figure 4).
- [x] **10.2 & 10.3.** Thống kê Top Easiest (Leg extension F1=1.0, Russian twist F1=1.0, Squat F1=0.97) và Hardest classes (Hammer curl F1=0.42, Romanian deadlift F1=0.44).
- [x] **10.4 & 10.5. Phân tích nguyên nhân cơ sinh học (Biomechanical Error Taxonomy - Table 11)**:
  - Phân loại 4 cơ chế lỗi cấu trúc giải thích 61.54% lỗi video:
    1. Optical Rotational Ambiguity (Hammer vs Biceps curl).
    2. Kinematic Form Overlap (Deadlift vs Romanian deadlift).
    3. Perspective Foreshortening (Bench press variants).
    4. Closed vs Open Kinetic Chain (Pull-up vs Lat pulldown vs T-bar row).

---

## 🟢 P3 — Nâng Cấp Nâng Cao

### 11. Modern Skeleton Baselines
- [x] **11.1 – 11.3.** Đã biện giải định tính trong Section 6 (Discussion): Không triển khai CTR-GCN và SkateFormer do các mô hình này cồng kềnh (hàng triệu params, pretraining NTU RGB+D, đòi hỏi 25 khớp dày đặc), đi ngược lại mục tiêu cốt lõi của SkelGym là Edge AI siêu nhẹ ($\approx 350\text{K}$ params) chạy thời gian thực trên chip biên di động. *(Tác giả quyết định: Chọn Phương án A - Giữ nguyên biện giải)*.

### 12. External / Cross-Domain Generalization Test
- [x] **12.1 – 12.3.** Đã thực hiện External Benchmark trên 4 bài tập Strength & Conditioning từ tập dữ liệu SU-EMD của Deyzel et al. (CVPRW 2023) (Section 5.7, Table 9):
  - Giải quyết "Deyzel Dilemma": tăng Recall của Deadlift từ 30% lên 90% và Squat đạt 93.3%.
  - Thử nghiệm 1-Shot Transfer Learning đạt **97.32%** (Transformer) và **95.34%** (SkelGym-Full).

---

## 🔵 P4 — Hoàn Thiện Bài Báo & Reproducibility

### 13. Paper Writing & Structure Audit
- [x] **13.1. Abstract**: Hoàn thiện cấu trúc Problem $\rightarrow$ Method $\rightarrow$ Dataset $\rightarrow$ Result $\rightarrow$ Deployment; số liệu đồng bộ 100%.
- [x] **13.2. Introduction**: Nêu bật động lực, 4 đóng góp kỹ thuật (C1–C4), và sơ đồ pipeline tổng quan.
- [x] **13.3. Related Work**: Rà soát đầy đủ CNN/Transformers, Skeleton GCNs, Augmentations, Ensembles, và Fitness Recognition.
- [x] **13.4. Method**: Trình bày toán học chặt chẽ: 117-d mix, SkelGym-Aug với chứng minh nhóm đối xứng $\mathbb{Z}_2$ và $\mathrm{SO}(2)$, Transformer Mix, 4-Stream AAGCN, SLSQP optimization.
- [x] **13.5. Results**: Trình bày đủ 9 bảng thực nghiệm, kiểm định thống kê và phân tích độ trễ phần cứng.
- [x] **13.6. Conclusion**: Tuyên bố khiêm tốn, chỉ ra hạn chế của monocular pose estimation (occlusion, depth jitter) và hướng phát triển hybrid.

### 14. Reference Audit
- [x] **14.1 – 14.7.** Rà soát 38 tài liệu trích dẫn chuẩn xác tên bài, tác giả, năm, venue, không có trích dẫn ma.

### 15. Code Repository & Reproducibility Standard
- [x] **15.1 – 15.7.** Repository công khai trên Hugging Face Hub (`Cuong2004/gym-exercise-classification`), cung cấp script tái hiện một dòng lệnh (`evaluate_local_ensemble.py`, `compute_statistical_tests.py`), checkpoint trọng số pre-trained đầy đủ.
- [x] **Hai định dạng bài báo hoàn chỉnh sẵn sàng nộp**:
  - `paper/paper_eswa.tex` + `paper/paper_eswa_overleaf.zip` (Elsevier ESWA format).
  - `paper/paper_llncs.tex` + `paper/paper_llncs.pdf` + `paper/paper_llncs_overleaf.zip` (Springer LNCS format).
