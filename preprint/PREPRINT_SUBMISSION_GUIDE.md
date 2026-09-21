# arXiv / Preprint Submission Package & Guide

Tài liệu này hướng dẫn nộp bản Preprint của công trình **SkelGym** lên **arXiv** (hoặc Zenodo / ResearchGate).

---

## 📦 1. Danh Sách Tệp Trong Thư Mục `preprint/`

| Tên tệp / thư mục | Mô tả |
| :--- | :--- |
| **`main.pdf`** (hoặc `SkelGym_Preprint.pdf`) | Bản PDF hoàn chỉnh (34 trang, đã biên dịch sạch 100%, 0 warning, 0 error). |
| **`main.tex`** | Mã nguồn LaTeX chính thức cho preprint (đã sửa toàn bộ lỗi vi mô). |
| **`arxiv_submission.zip`** | **Gói nộp hoàn chỉnh cho arXiv** (chứa `main.tex`, `llncs.cls`, `loadpkg_lncs.sty`, `splncs04.bst`, và thư mục `images/`). Sẵn sàng upload 1 chạm. |
| **`images/`** | Toàn bộ 4 ảnh vector / raster chất lượng cao (`Dataset.png`, `Mediapipe.png`, `Mediapipe_2.png`, `cm_ensemble_T5.1_weighted_soft.png`). |

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
Video-based gym exercise classification is fundamental for developing automated virtual personal trainer systems capable of tracking workout regimens, providing form feedback, and mitigating acute musculoskeletal injury risks. However, traditional RGB video models incur excessive computational complexity and suffer from environmental background overfitting, clothing variability, and severe privacy intrusion concerns in home environments. In this work, we present SkelGym, a lightweight and biomechanically grounded framework for fine-grained 22-class gym exercise recognition. SkelGym introduces three core technical innovations: (1) the proposed 117-dimensional Biomechanical Compound Representation combining 3D relative joint offsets with pairwise elevation angles (	heta) and azimuth angles (\phi), resolving the curse of dimensionality inherent in conventional 3-joint angle triplets; (2) SkelGym-Aug, an on-the-fly skeletal data augmentation protocol that preserves anatomical validity through bilateral body symmetry reflection with exact angular permutation and elevation-invariant 3D yaw rotations; and (3) a cross-paradigm ensemble architecture coupling a self-attention Transformer encoder with a 4-stream Adaptive Graph Convolutional Network (AAGCN) via Validation-Calibrated Weighted Soft Voting optimized using Sequential Least Squares Programming (SLSQP) across window-level and video-level consensus aggregations. Under a strictly controlled parameter budget (365K-399K parameters, pprox 350K \pm 15% per backbone), SkelGym achieves the strongest performance among evaluated methods in our benchmark: 70.11% window-level accuracy and 77.68% video-level consensus accuracy (Macro F1: 0.7649) across 2,743 test windows and 233 held-out test videos. Crucially, all model selection and ensemble optimization were established strictly on validation partitions, ensuring strictly held-out out-of-sample evaluation. With an ultra-low classifier inference latency of 1.01 ms per 32-frame window on edge devices (excluding monocular pose extraction, which requires pprox 8-15 ms/frame, yielding an estimated end-to-end latency of 9-16 ms well within the 33.3 ms real-time threshold), SkelGym is suited for real-time mobile and embedded deployment. All code, datasets, and pre-trained model checkpoints are publicly available at https://huggingface.co/Cuong2004/gym-exercise-classification.
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
