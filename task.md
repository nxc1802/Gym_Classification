# Kế Hoạch & Danh Sách CLI Thực Nghiệm (Gym Classification)

> **Môi trường, Tiêu Chuẩn & Cấu Hình Cố Định:**
> - **Dữ liệu chuẩn:** 1,024 videos gốc across 22 classes từ Kaggle (`nguyenxuancuongk18dn/gym-exercise-classification-dataset`) trích xuất bằng **MediaPipe Pose Heavy (`model_complexity = 2`)**.
> - **Cấu hình Windowing cố định ($SL = 32$):** Áp dụng duy nhất một cấu hình chuẩn $T = 32$ (không chạy nhiều config $SL$ phân tán):
>   - **Train Set (580 videos):** 15,820 windows (stride = 16).
>   - **Validation Set (208 videos):** 2,147 windows (stride = 32).
>   - **Held-out Test Set (236 videos):** 3,337 windows (stride = 32).
> - **Tham số huấn luyện cố định:** **`--epochs 100 --patience 10`** trên toàn bộ các mô hình đơn lẻ.
> - **Đồng bộ dữ liệu:** Đã lưu trữ trên Hugging Face Hub: [`Cuong2004/gym-exercise-landmarks`](https://huggingface.co/datasets/Cuong2004/gym-exercise-landmarks) và đồng bộ tại `data/landmarks/`.

---

## I. Danh Sách Nhiệm Vụ Cốt Lõi (Core Tasks)
- [x] 1. Setup & kết nối Marimo server, clone repository, trích xuất landmarks MediaPipe Pose Heavy (complexity=2).
- [x] 2. Push landmarks dataset lên Hugging Face và đồng bộ về local `data/landmarks/`.
- [x] 3. Chạy Smoke Test (cách ly tại `outputs/smoke_test/` và `checkpoints/smoke_test/`, kiểm tra $SL=32$, train, eval, ensemble).
- [ ] 4. Thực thi tuần tự các Bảng thực nghiệm với $SL=32$, epochs=100, patience=10 (Bảng 1 $\rightarrow$ Bảng 2 $\rightarrow$ Bảng 3 $\rightarrow$ Bảng 4 $\rightarrow$ Bảng 5 $\rightarrow$ Bảng 6A, 6B & Bảng 7).
- [ ] 5. Tự động đồng bộ checkpoint, kết quả và confusion matrix vào `outputs/EXPERIMENT_RESULTS.md` và Hugging Face Hub.
- [ ] 6. Nghiệm thu SOTA, cập nhật LaTeX table cho paper và đồng bộ code lên GitHub.

---

## II. Hướng Dẫn Quy Trình Cho Researcher & Developer

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
   - **Bảng 5 Cross-Paradigm Ensemble:** Kết hợp Best Transformer (`checkpoints/best_Transformer_T1.21_mix.pt` hoặc `checkpoints/best_Transformer_T2.2_mix.pt`) với các mô hình đồ thị từ Bảng 3/4.

---

## III. Danh Sách CLI Thực Nghiệm Chi Tiết

### 0. Smoke Test Pipeline (Kiểm tra nhanh toàn diện)

#### Cách 1: Chạy toàn bộ test suite tự động (Train $\rightarrow$ Eval $\rightarrow$ Ensemble $\rightarrow$ Video-Level $\rightarrow$ Verify Isolation)
```bash
python3 scripts/smoke_test.py
```

#### Cách 2: Chạy lẻ từng thành phần Smoke Test
```bash
# 1. Smoke test Transformer (Mix, SL=32)
python3 run.py train --model Transformer --feature mix --smoke_test --epochs 2 --device auto --use_amp --in_memory

# 2. Smoke test Graph AAGCN (Rel 3D, SL=32)
python3 run.py train --model AAGCN --feature rel_3d --smoke_test --epochs 2 --device auto --use_amp --in_memory

# 3. Smoke test Single-Model Video-Level Evaluation
python3 run.py evaluate --checkpoint checkpoints/smoke_test/best_Transformer_T1.21_mix.pt --model Transformer --feature mix --video_level --smoke_test --device auto

# 4. Smoke test Ensemble Video-Level Evaluation (SL=32, Stride=32)
python3 run.py ensemble --method weighted_soft --checkpoints checkpoints/smoke_test/best_Transformer_T1.21_mix.pt checkpoints/smoke_test/best_AAGCN_T3.5_rel_3d.pt --seq_len 32 --stride 32 --video_level --smoke_test --device auto
```

---

### Bảng 1: Temporal Models on Landmark Feature Sets (21 runs, $SL=32$)
*Benchmark 3 kiến trúc chuỗi thời gian trên 7 không gian biểu diễn (Controlled Budget $\approx 350\text{K}$, Epochs 100, Patience 10).*

#### 1. LSTM (7 runs)
```bash
python3 run.py train --model LSTM --feature raw_2d --exp_id T1.1 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_2d --exp_id T1.2 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_2d --exp_id T1.3 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature raw_3d --exp_id T1.4 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_3d --exp_id T1.5 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_3d --exp_id T1.6 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature mix --exp_id T1.7 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

#### 2. BiLSTM (7 runs)
```bash
python3 run.py train --model BiLSTM --feature raw_2d --exp_id T1.8 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_2d --exp_id T1.9 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_2d --exp_id T1.10 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature raw_3d --exp_id T1.11 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_3d --exp_id T1.12 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_3d --exp_id T1.13 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature mix --exp_id T1.14 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

#### 3. Transformer (7 runs)
```bash
python3 run.py train --model Transformer --feature raw_2d --exp_id T1.15 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_2d --exp_id T1.16 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_2d --exp_id T1.17 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature raw_3d --exp_id T1.18 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_3d --exp_id T1.19 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_3d --exp_id T1.20 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --exp_id T1.21 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

---

### Bảng 2: Data Augmentation Strategies on Best Transformer (2 runs, $SL=32$)
*Khảo sát tăng cường dữ liệu: Baseline sạch (None) vs. SkelGym-Aug động trên Transformer Mix.*

```bash
python3 run.py train --model Transformer --feature mix --augment none --exp_id T2.1 --epochs 100 --patience 10 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --augment skel_gym_aug --exp_id T2.2 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
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

### Bảng 4: Data Augmentation Ablation on Graph Architectures (4 runs, $SL=32$)
*Ablation study đánh giá định lượng tác động của SkelGym-Aug (None vs Dynamic Augmentation) trên mô hình đồ thị không-thời gian (ST-GCN và AAGCN).*

#### 1. ST-GCN Augmentation Ablation (2 runs)
```bash
# ST-GCN (Rel 3D) - Baseline sạch (None)
python3 run.py train --model STGCN --feature rel_3d --augment none --exp_id T4.1 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory

# ST-GCN (Rel 3D) - SkelGym-Aug (Dynamic on-the-fly)
python3 run.py train --model STGCN --feature rel_3d --augment skel_gym_aug --exp_id T4.2 --epochs 100 --patience 10 --video_level --device auto --use_amp --in_memory
```

#### 2. Adaptive GCN (AAGCN) Augmentation Ablation (2 runs)
```bash
# AAGCN (Rel 3D) - Baseline sạch (None)
python3 run.py train --model AAGCN --feature rel_3d --augment none --exp_id T4.3 --epochs 100 --patience 10 --device auto --use_amp --in_memory

# AAGCN (Rel 3D) - SkelGym-Aug (Dynamic on-the-fly)
python3 run.py train --model AAGCN --feature rel_3d --augment skel_gym_aug --exp_id T4.4 --epochs 100 --patience 10 --device auto --use_amp --in_memory
```

---

### Bảng 5: Heterogeneous Cross-Paradigm Ensemble (5 runs, $SL=32$)
*Hợp nhất đa mô hình giữa chuỗi thời gian (Transformer) và đồ thị thích ứng (AAGCN) trên cùng độ dài chuỗi $SL=32$.*

```bash
# 1. Hard Voting (Best Transformer + Best ST-GCN)
python3 run.py ensemble --method hard --exp_id T5.1 --seq_len 32 --stride 32 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T3.2_rel_3d.pt --device auto

# 2. Soft Voting (Best Transformer + Best ST-GCN)
python3 run.py ensemble --method soft --exp_id T5.2 --seq_len 32 --stride 32 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T3.2_rel_3d.pt --device auto

# 3. Stacking Ensemble (Best Transformer + Best ST-GCN)
python3 run.py ensemble --method stacking --exp_id T5.3 --seq_len 32 --stride 32 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T3.2_rel_3d.pt --device auto

# 4. Tri-Model Grand Ensemble (Transformer Mix + AAGCN Joint + AAGCN Bone) - Tự động cập nhật Bảng 5 (T5.4) và Bảng 7
python3 run.py ensemble --method weighted_soft --exp_id T5.4 --seq_len 32 --stride 32 --video_level --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T3.5_rel_3d.pt checkpoints/best_AAGCN_T3.6_bone_3d.pt --device auto

# 5. Grand 5-Stream SOTA Ensemble (Transformer Mix + Four-Stream AAGCN) - Tự động cập nhật Bảng 5 (T5.5), Bảng 6A, Bảng 6B và Bảng 7
python3 run.py ensemble --method weighted_soft --exp_id T5.5 --seq_len 32 --stride 32 --video_level --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T3.5_rel_3d.pt checkpoints/best_AAGCN_T3.6_bone_3d.pt checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt --device auto
```

---

### Bảng 6A, 6B & 7: Video-Level Aggregation Benchmark
*Đánh giá toàn diện cấp độ Video trên toàn bộ 236 video bài tập độc lập.*

#### 1. Đánh giá Grand SOTA Ensemble (Tự động cập nhật Bảng 6A, 6B và dòng Grand SOTA trong Bảng 7)
```bash
python3 run.py ensemble --method weighted_soft --exp_id T5.5 --video_level --seq_len 32 --stride 32 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T3.5_rel_3d.pt checkpoints/best_AAGCN_T3.6_bone_3d.pt checkpoints/best_AAGCN_T3.7_joint_motion_3d.pt checkpoints/best_AAGCN_T3.8_bone_motion_3d.pt --device auto
```

#### 2. Đánh giá Video-Level cho các Baseline chuỗi thời gian & Đồ thị đơn lẻ (Cập nhật Bảng 7)
*(Chỉ cần chạy nếu trong lúc train chưa kèm cờ `--video_level`)*
```bash
# Baseline LSTM (Mix)
python3 run.py evaluate --checkpoint checkpoints/best_LSTM_T1.7_mix.pt --model LSTM --feature mix --video_level --device auto

# Baseline BiLSTM (Mix)
python3 run.py evaluate --checkpoint checkpoints/best_BiLSTM_T1.14_mix.pt --model BiLSTM --feature mix --video_level --device auto

# Baseline ST-GCN (Rel 3D)
python3 run.py evaluate --checkpoint checkpoints/best_STGCN_T3.2_rel_3d.pt --model STGCN --feature rel_3d --video_level --device auto

# Best Transformer (Mix)
python3 run.py evaluate --checkpoint checkpoints/best_Transformer_T1.21_mix.pt --model Transformer --feature mix --video_level --device auto

# Best Transformer + SkelGym-Aug
python3 run.py evaluate --checkpoint checkpoints/best_Transformer_T2.2_mix.pt --model Transformer --feature mix --video_level --device auto
```