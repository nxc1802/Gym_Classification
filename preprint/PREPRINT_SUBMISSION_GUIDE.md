# arXiv / Preprint Submission Package & Guide

Tài liệu này hướng dẫn nộp bản Preprint của công trình **SkelGym** lên **arXiv** (hoặc Zenodo / ResearchGate).

---

## 📦 1. Danh Sách Tệp Trong Thư Mục `preprint/`

| Tên tệp / thư mục | Mô tả |
| :--- | :--- |
| **`main.pdf`** (hoặc `SkelGym_Preprint.pdf`) | Bản PDF hoàn chỉnh (34 trang, đã biên dịch sạch 100%, 0 warning, 0 error). |
| **`main.tex`** | Mã nguồn LaTeX chính thức cho preprint (đã sửa toàn bộ lỗi vi mô). |
| **`arxiv_submission.zip`** | **Gói nộp hoàn chỉnh cho arXiv** (chứa `main.tex`, `llncs.cls`, `loadpkg_lncs.sty`, `splncs04.bst`, và thư mục `images/`). Sẵn sàng upload 1 chạm. |
| **`images/`** | Toàn bộ ảnh vector / raster chất lượng cao (`Dataset.png`, `Mediapipe.png`, `Mediapipe_2.png`, `cm_ensemble_T5.1_weighted_soft.pdf` & `.png`). |

---

## 📝 2. Thông Tin Metadata Nộp arXiv (Copy & Paste Sẵn)

### **Title:**
```text
SkelGym: Cross-Paradigm Kinematic Fusion and Dynamic Augmentation for 22-Class Gym Exercise Classification
```

### **Authors:**
```text
Xuan-Cuong Nguyen, Nhat-Quang Truong, Quoc-Trinh Vo
```

### **Primary Subject Category:**
* **`cs.CV` (Computer Vision and Pattern Recognition)** *(Bắt buộc chọn cái này làm Primary)*

### **Cross-Lists / Secondary Categories:**
* **`cs.AI` (Artificial Intelligence)**
* **`cs.LG` (Machine Learning)**
* **`cs.HC` (Human-Computer Interaction)**

### **Comments:**
```text
34 pages, 5 figures, 13 tables. Complete benchmark dataset, models, and code available at https://huggingface.co/Cuong2004/gym-exercise-classification
```

### **Abstract:**
```text
Automated gym exercise recognition is critical for developing intelligent virtual personal trainer systems capable of monitoring workout execution, tracking training volume, and mitigating musculoskeletal injury risks. However, traditional RGB video approaches incur prohibitive computational costs, suffer from environmental background overfitting, and raise severe privacy concerns when streaming from home environments. We evaluate on a curated corpus of 1,024 in-the-wild videos spanning 22 fine-grained resistance exercises under strict video-level partitioning with no source-video overlap across train, validation, and test sets. We present SkelGym, a lightweight, privacy-aware intelligent exercise recognition framework operating exclusively on 3D skeletal coordinates extracted via monocular pose estimation. SkelGym introduces three core innovations: (1) a 117-dimensional Biomechanical Compound Representation fusing relative 3D joint displacements with pairwise elevation and azimuth angles; (2) SkelGym-Aug, an anatomically valid skeletal augmentation protocol preserving musculoskeletal constraints through bilateral sagittal reflection and gravitational yaw rotation; and (3) a cross-paradigm ensemble coupling a self-attention Transformer with a four-stream Adaptive Graph Convolutional Network via validation-calibrated SLSQP soft voting. SkelGym achieves 69.74% ± 1.04% window-level accuracy and 79.11% ± 0.25% video-level consensus accuracy (Video Macro F1: 0.7834 ± 0.0082, Window Macro F1: 0.6882 ± 0.0068) across three independent random seeds under fixed splits (2,743 test windows from 233 held-out videos), with selected model comparisons evaluated via paired statistical tests (McNemar p < 10^-11, Wilcoxon p < 0.005 against \alpha_adj = 0.01). With an ultra-low single-window classifier inference latency of merely 0.42--4.33 ms on host CPU (0.08--0.54 ms on CUDA, 1.11--8.77 ms on MPS; where single-model backbones execute in 0.42--1.16 ms and the full five-stream ensemble executes in 4.33 ms) per 32-frame window, SkelGym enables real-time, privacy-aware exercise recognition suitable for smart mirrors, mobile devices, and embedded gym displays. All source code, extracted skeletal representations, metadata, and pretrained checkpoints used for reproducibility are publicly available at https://huggingface.co/Cuong2004/gym-exercise-classification.
```

---

## 🚀 3. Các Bước Nộp Lên arXiv (Chỉ Mất ~5 Phút)

1. Đăng nhập vào https://arxiv.org/submit
2. Chọn **"Start New Submission"**.
3. Tại bước **"Files"**: Tải lên duy nhất tệp zip:
   👉 `preprint/arxiv_submission.zip`
4. arXiv sẽ tự động giải nén và biên dịch `main.tex`.
5. Điền các trường thông tin **Title**, **Authors**, **Abstract**, **Category (`cs.CV`)** và **Comments** theo phần 2 ở trên.
6. Xem trước bản PDF (Preview PDF) được sinh tự động bởi arXiv, sau đó bấm **Submit**.
