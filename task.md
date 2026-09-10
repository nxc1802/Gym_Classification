# Kế Hoạch & Danh Sách CLI Thực Nghiệm (Gym Classification)

> **Môi trường & Dữ liệu:**
> - Toàn bộ dữ liệu được trích xuất từ **Kaggle Source of Truth** (`nguyenxuancuongk18dn/gym-exercise-classification-dataset`) bằng **MediaPipe Pose Heavy (`model_complexity = 2`)**.
> - Cấu trúc: 1,024 videos gốc across 22 classes (580 Train, 208 Validation, 236 Test).
> - Đã lưu trữ trên Hugging Face Hub: [`Cuong2004/gym-exercise-landmarks`](https://huggingface.co/datasets/Cuong2004/gym-exercise-landmarks) và đồng bộ tại `data/landmarks/`.

---

## I. Danh Sách Nhiệm Vụ Cốt Lõi (Core Tasks)
- [x] 1. Setup & kết nối Marimo server, trích xuất landmarks MediaPipe Pose Heavy (complexity=2).
- [x] 2. Push landmarks dataset lên Hugging Face và đồng bộ về local `data/landmarks/`.
- [ ] 3. Chạy Smoke Test (1-2 epochs) kiểm tra toàn bộ pipeline training và auto-logging.
- [ ] 4. Thực thi tuần tự các Bảng thực nghiệm (Bảng 1 $\rightarrow$ Bảng 2 $\rightarrow$ Bảng 3 $\rightarrow$ Bảng 4 $\rightarrow$ Bảng 5 $\rightarrow$ Bảng 6 & 7).
- [ ] 5. Tự động đồng bộ checkpoint, kết quả và confusion matrix vào `outputs/EXPERIMENT_RESULTS.md` và Hugging Face Hub.
- [ ] 6. Nghiệm thu SOTA, cập nhật LaTeX table cho paper và đồng bộ code lên GitHub.

---

## II. Hướng Dẫn Quy Trình Cho Researcher & Developer

1. **Thứ tự thực nghiệm chuẩn:**
   $$\text{Bảng 1 (Temporal Baselines)} \longrightarrow \text{Bảng 2 (Augmentation)} \longrightarrow \text{Bảng 3 (Feature Fusion / SOTA Loss)} \longrightarrow \text{Bảng 4 (GNN / AAGCN Streams)} \longrightarrow \text{Bảng 5, 6, 7 (Ensemble / Video-Level / TTA)}$$
2. **Cơ chế Auto-Logging:**
   - Mỗi lệnh `run.py train` hoặc `run.py ensemble` đều có tham số `--exp_id <ID>` (ví dụ: `--exp_id T1.1`).
   - Khi hoàn thành, script sẽ **tự động cập nhật trực tiếp** chỉ số Train Loss, Val Loss, Val Acc, Test Acc, Macro F1 và đường dẫn Checkpoint vào đúng dòng trong [`outputs/EXPERIMENT_RESULTS.md`](outputs/EXPERIMENT_RESULTS.md).
3. **Tối ưu tốc độ:**
   - Luôn kèm cờ `--use_amp --in_memory --device auto` để kích hoạt Automatic Mixed Precision (bfloat16/fp16) và nạp toàn bộ dataset vào RAM để đạt thông lượng tối đa.
   - Trên server GPU (RTX PRO 6000 / CUDA), mỗi run chỉ mất khoảng 2-4 phút.
4. **Chọn Best Checkpoints cho Ensemble (Bảng 5, 6, 7):**
   - Sau khi chạy xong Bảng 1, 2, 3: Chọn checkpoint Transformer có Val Acc cao nhất $\rightarrow$ `<BEST_TRANSFORMER_CKPT>`.
   - Sau khi chạy xong Bảng 4: Chọn các checkpoint AAGCN (Joint, Bone, J-Motion, B-Motion) $\rightarrow$ `<AAGCN_JOINT>`, `<AAGCN_BONE>`, `<AAGCN_JM>`, `<AAGCN_BM>`.

---

## III. Danh Sách CLI Thực Nghiệm Chi Tiết

### 0. Smoke Test Pipeline (Kiểm tra nhanh 2 epochs)
```bash
python3 run.py train --model Transformer --feature mix --smoke_test --epochs 2 --device auto --use_amp --in_memory
```

---

### Bảng 1: Temporal Models on Landmark Feature Sets (21 runs)
*Benchmark 3 kiến trúc chuỗi thời gian trên 7 không gian biểu diễn (Controlled Budget $\approx 350\text{K}$).*

#### 1. LSTM (7 runs)
```bash
python3 run.py train --model LSTM --feature raw_2d --exp_id T1.1 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_2d --exp_id T1.2 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_2d --exp_id T1.3 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature raw_3d --exp_id T1.4 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature rel_3d --exp_id T1.5 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature angle_3d --exp_id T1.6 --device auto --use_amp --in_memory
python3 run.py train --model LSTM --feature mix --exp_id T1.7 --device auto --use_amp --in_memory
```

#### 2. BiLSTM (7 runs)
```bash
python3 run.py train --model BiLSTM --feature raw_2d --exp_id T1.8 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_2d --exp_id T1.9 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_2d --exp_id T1.10 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature raw_3d --exp_id T1.11 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature rel_3d --exp_id T1.12 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature angle_3d --exp_id T1.13 --device auto --use_amp --in_memory
python3 run.py train --model BiLSTM --feature mix --exp_id T1.14 --device auto --use_amp --in_memory
```

#### 3. Transformer (7 runs)
```bash
python3 run.py train --model Transformer --feature raw_2d --exp_id T1.15 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_2d --exp_id T1.16 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_2d --exp_id T1.17 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature raw_3d --exp_id T1.18 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature rel_3d --exp_id T1.19 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature angle_3d --exp_id T1.20 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --exp_id T1.21 --device auto --use_amp --in_memory
```

---

### Bảng 2: Data Augmentation Strategies on Best Transformer (3 runs)
*Khảo sát các chiến lược tăng cường dữ liệu trên mô hình Transformer tốt nhất (mặc định: `mix`).*

```bash
python3 run.py train --model Transformer --feature mix --augment none --exp_id T2.1 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --augment combined --exp_id T2.2 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --augment skel_gym_aug --exp_id T2.3 --device auto --use_amp --in_memory
```

---

### Bảng 3: Feature Fusion & SOTA Architecture Ablations (4 runs)
*Đánh giá phương thức hợp nhất đa đặc trưng và kỹ thuật tối ưu hàm Loss (Label Smoothing, Focal Loss).*

```bash
python3 run.py train --model Transformer --feature direct_concat --exp_id T3.1 --device auto --use_amp --in_memory
python3 run.py train --model BranchConcat --feature branch_concat --exp_id T3.2 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --label_smoothing 0.1 --exp_id T3.3 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --loss focal --focal_gamma 2.0 --exp_id T3.4 --device auto --use_amp --in_memory
```

---

### Bảng 4: Spatial-Temporal Graph Models (ST-GCN & AAGCN Multi-Stream) (8 runs)
*Khảo sát mạng đồ thị không-thời gian: ST-GCN trên toạ độ tĩnh và AAGCN trên 4 luồng động học.*

#### 1. ST-GCN Baselines (4 runs, $T=32$)
```bash
python3 run.py train --model STGCN --feature raw_3d --exp_id T4.1 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_3d --exp_id T4.2 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature raw_2d --exp_id T4.3 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_2d --exp_id T4.4 --device auto --use_amp --in_memory
```

#### 2. Adaptive GCN (AAGCN 4-Stream, $T=20$)
```bash
python3 run.py train --model AAGCN --feature rel_3d --seq_len 20 --train_stride 10 --val_test_stride 20 --exp_id T4.5 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature bone_3d --seq_len 20 --train_stride 10 --val_test_stride 20 --exp_id T4.6 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature joint_motion_3d --seq_len 20 --train_stride 10 --val_test_stride 20 --exp_id T4.7 --device auto --use_amp --in_memory
python3 run.py train --model AAGCN --feature bone_motion_3d --seq_len 20 --train_stride 10 --val_test_stride 20 --exp_id T4.8 --device auto --use_amp --in_memory
```

---

### Bảng 5: Heterogeneous Ensemble & Multi-Modal Fusion (7 runs)
*Hợp nhất các mô hình chuỗi thời gian (Transformer) và mô hình đồ thị (AAGCN) để thiết lập SOTA.*

```bash
# 1. Hard Voting (Best Transformer + Best ST-GCN)
python3 run.py ensemble --method hard --exp_id T5.1 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T4.2_rel_3d.pt --device auto

# 2. Soft Voting (Best Transformer + Best ST-GCN)
python3 run.py ensemble --method soft --exp_id T5.2 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T4.2_rel_3d.pt --device auto

# 3. Stacking Ensemble
python3 run.py ensemble --method stacking --exp_id T5.3 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_STGCN_T4.2_rel_3d.pt --device auto

# 4. Two-Stream AAGCN (Joint + Bone)
python3 run.py ensemble --method weighted_soft --exp_id T5.4 --seq_len 20 --stride 20 --checkpoints checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt --device auto

# 5. Four-Stream AAGCN (Joint + Bone + J-Motion + B-Motion)
python3 run.py ensemble --method weighted_soft --exp_id T5.5 --seq_len 20 --stride 20 --checkpoints checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt checkpoints/best_AAGCN_T4.7_joint_motion_3d.pt checkpoints/best_AAGCN_T4.8_bone_motion_3d.pt --device auto

# 6. Tri-Model Grand Ensemble (Transformer Mix + AAGCN Joint + AAGCN Bone)
python3 run.py ensemble --method weighted_soft --exp_id T5.6 --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt --device auto

# 7. Grand 5-Stream SOTA + TTA (Transformer Mix + 4-Stream AAGCN + Bilateral Mirroring)
python3 run.py ensemble --method weighted_soft --exp_id T5.7 --tta --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt checkpoints/best_AAGCN_T4.7_joint_motion_3d.pt checkpoints/best_AAGCN_T4.8_bone_motion_3d.pt --device auto
```

---

### Bảng 6 & 7: Video-Level Aggregation & Test-Time Augmentation (TTA)
*Đánh giá độ chính xác thực tế trên toàn bộ 236 video bài tập độc lập (Clip/Video-level).*

```bash
# Đánh giá Video-Level trên Grand SOTA Ensemble
python3 run.py ensemble --method weighted_soft --video_level --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt checkpoints/best_AAGCN_T4.7_joint_motion_3d.pt checkpoints/best_AAGCN_T4.8_bone_motion_3d.pt --device auto

# Đánh giá Video-Level kết hợp TTA (Test-Time Augmentation)
python3 run.py ensemble --method weighted_soft --video_level --tta --checkpoints checkpoints/best_Transformer_T1.21_mix.pt checkpoints/best_AAGCN_T4.5_rel_3d.pt checkpoints/best_AAGCN_T4.6_bone_3d.pt checkpoints/best_AAGCN_T4.7_joint_motion_3d.pt checkpoints/best_AAGCN_T4.8_bone_motion_3d.pt --device auto
```