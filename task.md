# Kế Hoạch & Danh Sách CLI Thực Nghiệm (Gym Classification)

> **Môi trường, Tiêu Chuẩn & Cấu Hình Cố Định:**
> - **Dữ liệu chuẩn:** 1,024 videos gốc across 22 classes từ Kaggle (`nguyenxuancuongk18dn/gym-exercise-classification-dataset`) trích xuất bằng **MediaPipe Pose Heavy (`model_complexity = 2`)**.
> - **Cấu hình Windowing cố định ($SL = 32$):** Áp dụng duy nhất một cấu hình chuẩn $T = 32$ (không chạy nhiều config $SL$ phân tán):
>   - **Train Set (580 videos):** 15,820 windows (stride = 16).
>   - **Validation Set (208 videos):** 2,147 windows (stride = 32).
>   - **Held-out Test Set (236 videos):** 3,337 windows (stride = 32).
> - **Tham số huấn luyện cố định:** **`--epochs 100 --patience 10`** trên toàn bộ các mô hình đơn lẻ.
> - **Đồng bộ dữ liệu:** Đã lưu trữ trên Hugging Face Hub: [`Cuong2004/gym-exercise-landmarks`](https://huggingface.co/datasets/Cuong2004/gym-exercise-landmarks). Tải toàn bộ bằng: `python3 run.py pull-landmarks-hf --dest_dir data/landmarks` (tự động giải nén landmarks và tải đồng bộ file metadata phân đoạn `data/Final_dataset_metadata.csv`).

---

## I. Danh Sách Nhiệm Vụ Cốt Lõi (Core Tasks)
- [x] 1. Setup & kết nối Marimo server, clone repository, trích xuất landmarks MediaPipe Pose Heavy (complexity=2).
- [x] 2. Push landmarks dataset và `Final_dataset_metadata.csv` lên Hugging Face và đồng bộ về local (`data/landmarks/` & `data/Final_dataset_metadata.csv`).
- [x] 3. Chạy Smoke Test (cách ly tại `outputs/smoke_test/` và `checkpoints/smoke_test/`, kiểm tra $SL=32$, train, eval, ensemble).
- [x] 4. Thực thi tuần tự các Bảng thực nghiệm với $SL=32$, epochs=100, patience=10 (Bảng 1 $\rightarrow$ Bảng 2 $\rightarrow$ Bảng 3 $\rightarrow$ Bảng 4 $\rightarrow$ Bảng 5 $\rightarrow$ Bảng 6A, 6B & Bảng 7).
- [x] 5. Tự động đồng bộ checkpoint, kết quả và confusion matrix vào `outputs/EXPERIMENT_RESULTS.md` và Hugging Face Hub.
- [x] 6. Nghiệm thu SOTA, cập nhật LaTeX table cho paper và đồng bộ code lên GitHub.

---

## II. Hướng Dẫn Quy Trình Cho Researcher & Developer

0. **Chuẩn bị Dữ Liệu & Metadata từ Hugging Face Hub:**
   - Để đồng bộ môi trường dữ liệu trên server mới hoặc máy cá nhân, chạy duy nhất 1 lệnh CLI:
     ```bash
     python3 run.py pull-landmarks-hf --dest_dir data/landmarks
     ```
   - **Lưu ý quan trọng:** Lệnh trên tự động tải đồng thời cả tập tin landmarks nén (`landmarks_dataset.zip` giải nén vào `data/landmarks/`) và file metadata phân đoạn [`data/Final_dataset_metadata.csv`](data/Final_dataset_metadata.csv).
   - Tệp [`data/Final_dataset_metadata.csv`](data/Final_dataset_metadata.csv) là nguồn chân lý (Ground Truth) chứa danh sách 1,024 video phân chia Train/Val/Test và các mốc thời gian động tác (`label_content`) để trích xuất Action Segments trước khi tạo sliding windows $T=32$.

1. **Thứ tự thực nghiệm chuẩn (Bảng 1 $\rightarrow$ Bảng 7 không khoảng trống):**
   $$\text{Bảng 1 (Temporal Baselines)} \longrightarrow \text{Bảng 2 (Temporal Aug)} \longrightarrow \text{Bảng 3 (GNN \& AAGCN Multi-Stream)} \longrightarrow \text{Bảng 4 (Graph Aug Ablation)} \longrightarrow \text{Bảng 5 (Ensemble)} \longrightarrow \text{Bảng 6A, 6B \& 7}$$

2. **Cơ chế Auto-Logging:**
   - Mỗi lệnh `run.py train` hoặc `run.py ensemble` đều có tham số `--exp_id <ID>` (ví dụ: `--exp_id T1.1`, `--exp_id T3.9`, `--exp_id T4.2`).
   - Khi hoàn thành, script sẽ **tự động cập nhật trực tiếp** các chỉ số Train Loss, Val Loss, Val Acc, Test Acc, Macro F1 và đường dẫn Checkpoint vào đúng dòng trong [`outputs/EXPERIMENT_RESULTS.md`](outputs/EXPERIMENT_RESULTS.md).

3. **Nguyên tắc Cách Ly Tuyệt Đối Cho Smoke Test:**
   - Khi chạy với cờ `--smoke_test`, hệ thống tự động định tuyến:
     - Checkpoints: `checkpoints/smoke_test/`
     - Plots / Confusion Matrices: `outputs/smoke_test/`
     - Kết quả bảng: `outputs/smoke_test/SMOKE_RESULTS.md`
     - Tự động tắt upload: `push_to_hf = False`.
   - Tệp chính thức [`outputs/EXPERIMENT_RESULTS.md`](outputs/EXPERIMENT_RESULTS.md) và thư mục `checkpoints/` chính thức **tuyệt đối không bị ảnh hưởng (0 diff)**.

4. **Tối ưu tốc độ:**
   - Luôn kèm cờ `--use_amp --in_memory --device auto` để kích hoạt Automatic Mixed Precision và nạp toàn bộ dataset vào RAM.
   - Trên GPU server (CUDA / RTX PRO 6000), mỗi run với 100 epochs (patience 10) chỉ mất khoảng 2-4 phút do early stopping dừng tối ưu.

5. **Checkpoints phục vụ Multi-Stream & Ensemble (Bảng 3, 4, 5, 6, 7):**
   - **Bảng 3 & 4 Graph Backbones ($SL=32$):**
     - ST-GCN Baseline: `checkpoints/best_STGCN_T3.2_rel_3d.pt`
     - Joint: `checkpoints/best_AAGCN_T3.5_rel_3d.pt`
     - Bone: `checkpoints/best_AAGCN_T3.6_bone_3d.pt`
     - Joint Motion: `checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt`
     - Bone Motion: `checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt`
   - **Bảng 5 Cross-Paradigm Ensemble:** Kết hợp Best Sequence Model (`checkpoints/best_Transformer_T2.2_raw_3d.pt`) với mô hình đồ thị tốt nhất (`checkpoints/best_AAGCN_T4.2_rel_3d.pt`).

---

## III. Danh Sách CLI Thực Nghiệm Chi Tiết

### 0. Tải Dữ Liệu & Smoke Test Pipeline

#### Tải đồng bộ Landmarks và Metadata từ HF Hub:
```bash
python3 run.py pull-landmarks-hf --dest_dir data/landmarks
```

#### Smoke Test tự động (Train $\rightarrow$ Eval $\rightarrow$ Ensemble $\rightarrow$ Video-Level $\rightarrow$ Verify Isolation):
```bash
python3 scripts/smoke_test.py
```

---

### Bảng 1: Temporal Models on Landmark Feature Sets (27 runs, $SL=32$)
*Benchmark 3 kiến trúc chuỗi thời gian trên 9 không gian biểu diễn (Controlled Budget $\approx 350\text{K}$, Epochs 100, Patience 10).*

#### 1. LSTM (9 runs)
```bash
python3 run.py train --model LSTM --feature raw_2d --exp_id T1.1 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_2d --exp_id T1.2 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_2d --exp_id T1.3 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle2_2d --exp_id T1.4 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature raw_3d --exp_id T1.5 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_3d --exp_id T1.6 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_3d --exp_id T1.7 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle2_3d --exp_id T1.8 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature mix --exp_id T1.9 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

#### 2. BiLSTM (9 runs)
```bash
python3 run.py train --model BiLSTM --feature raw_2d --exp_id T1.10 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_2d --exp_id T1.11 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_2d --exp_id T1.12 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle2_2d --exp_id T1.13 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature raw_3d --exp_id T1.14 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_3d --exp_id T1.15 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_3d --exp_id T1.16 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle2_3d --exp_id T1.17 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature mix --exp_id T1.18 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

#### 3. Transformer (9 runs)
```bash
python3 run.py train --model Transformer --feature raw_2d --exp_id T1.19 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_2d --exp_id T1.20 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_2d --exp_id T1.21 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle2_2d --exp_id T1.22 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature raw_3d --exp_id T1.23 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_3d --exp_id T1.24 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_3d --exp_id T1.25 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle2_3d --exp_id T1.26 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --exp_id T1.27 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

---

### Bảng 2: Data Augmentation Strategies on Best Sequence Model (SL=32)
*Khảo sát tăng cường dữ liệu: Baseline sạch (None) vs. SkelGym-Aug động trên mô hình chuỗi thời gian tốt nhất (Transformer rel_3d).*

```bash
# T2.1: Baseline sạch không augmentation (kế thừa từ T1.24)
python3 run.py train --model Transformer --feature rel_3d --augment none --exp_id T2.1 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory

# T2.2: SkelGym-Aug động (Lật gương song ánh, Xoay yaw 3D +-15 deg, Co dãn không hỏng vis, Jitter kẹp biên)
python3 run.py train --model Transformer --feature rel_3d --augment skel_gym_aug --exp_id T2.2 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

---

### Bảng 3: Spatial-Temporal Graph Models & Multi-Stream AAGCN Kinematics (10 runs, $SL=32$)
*Khảo sát mạng đồ thị không-thời gian: ST-GCN baselines và AAGCN 4 luồng động học chuẩn hóa trên $SL=32$.*

#### 1. ST-GCN Baselines (4 runs, $SL=32$)
```bash
python3 run.py train --model STGCN --feature raw_3d --exp_id T3.1 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_3d --exp_id T3.2 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature raw_2d --exp_id T3.3 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_2d --exp_id T3.4 --epochs 100 --patience 10 --device auto --use_amp --in_memory
```

#### 2. Adaptive GCN (AAGCN) Single-Stream Kinematics (4 runs, $SL=32$)
```bash
python3 run.py train --model AAGCN --feature rel_3d --exp_id T3.5 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature bone_3d --exp_id T3.6 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature joint_motion_3d --exp_id T3.7 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature bone_motion_3d --exp_id T3.8 --epochs 100 --patience 10 --device auto --use_amp --in_memory
```

#### 3. AAGCN Multi-Stream Late Fusion Ablations (2 runs, $SL=32$, Stride=32)
```bash
# Two-Stream AAGCN (Joint + Bone) - Tự động cập nhật Bảng 3 (T3.9) và Bảng 7
python3 run.py ensemble --method weighted_soft --exp_id T3.9 --seq_len 32 --stride 32 --video_level --checkpoints checkpoints/best_AAGCN_T3.5_rel_3d.pt checkpoints/best_AAGCN_T3.6_bone_3d.pt --device auto

# Four-Stream AAGCN (Joint + Bone + Joint Motion + Bone Motion) - Tự động cập nhật Bảng 3 (T3.10) và Bảng 7
python3 run.py ensemble --method weighted_soft --exp_id T3.10 --seq_len 32 --stride 32 --video_level --checkpoints checkpoints/best_AAGCN_T3.5_rel_3d.pt checkpoints/best_AAGCN_T3.6_bone_3d.pt checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt --device auto
```

---

### Bảng 4: Data Augmentation Ablation on Graph Architectures (2 runs, $SL=32$)
*Ablation study đánh giá định lượng tác động của SkelGym-Aug (None vs Dynamic Augmentation) trên mô hình đồ thị tốt nhất (AAGCN bone_3d).*

```bash
# T4.1: AAGCN (Bone 3D) - Baseline sạch (None, kế thừa từ T3.6)
python3 run.py train --model AAGCN --feature bone_3d --augment none --exp_id T4.1 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory

# T4.2: AAGCN (Bone 3D) - SkelGym-Aug (Dynamic on-the-fly)
python3 run.py train --model AAGCN --feature bone_3d --augment skel_gym_aug --exp_id T4.2 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

---

### Bảng 5: Heterogeneous Cross-Paradigm Ensemble (Thống nhất Weighted Soft Voting)
*Hợp nhất đa mô hình giữa chuỗi thời gian (Best Sequence: Transformer rel_3d aug) và đồ thị thích ứng (Best Graph: AAGCN bone_3d aug) qua Dual-Target Optimization.*

```bash
# Grand Heterogeneous Ensemble (Weighted Soft Voting với Tối ưu hóa kép: Window Phase 1 & Video Phase 2)
# Tự động cập nhật Bảng 5, Bảng 6A, 6B và dòng Grand SOTA trong Bảng 7
python3 run.py ensemble --method weighted_soft --exp_id T5.1 --seq_len 32 --stride 32 --video_level --checkpoints checkpoints/best_Transformer_T2.2_rel_3d.pt checkpoints/best_AAGCN_T4.2_bone_3d.pt --device auto
```

---

### Bảng 6A, 6B & 7: Video-Level Aggregation Benchmark
*Đánh giá toàn diện cấp độ Video trên toàn bộ 236 video bài tập độc lập (tự động cập nhật bởi lệnh ensemble trên).*