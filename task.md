# Kế Hoạch & Danh Sách CLI Thực Nghiệm (Gym Classification)

## I. Danh Sách Nhiệm Vụ Cốt Lõi (Core Tasks)
- [ ] 1. Setup & kết nối Marimo server, clone project.
- [ ] 2. Chạy dataset preparation + MediaPipe một lần và lưu/push output lên HF.
- [ ] 3. Chạy smoke test MediaPipe → smoke training để kiểm tra toàn bộ pipeline.
- [ ] 4. Chạy training experiments, đảm bảo server không disconnect, theo dõi tiến độ mỗi 15 phút và chạy đồng thời nhiều model nếu có thể.
- [ ] 5. Upload checkpoint (best/last), test results và report lên HF.
- [ ] 6. Ghi toàn bộ experiment results vào outputs/EXPERIMENT_RESULTS.md
- [ ] 7. Đồng bộ code lên GitHub sau khi test thành công.

---

## II. Hướng Dẫn Quy Trình Cho Researcher
1. **Thứ tự thực hiện:** Chạy lần lượt **Bảng 1** $\rightarrow$ **Bảng 2** $\rightarrow$ **Bảng 4**.
2. **Chọn Best Checkpoints:** 
   - Sau khi hoàn thành Bảng 1 & 2, kiểm tra `outputs/EXPERIMENT_RESULTS.md` chọn checkpoint Transformer có Val Acc cao nhất $\rightarrow$ `<BEST_TRANSFORMER_CKPT>` (ví dụ: `checkpoints/best_Transformer_T2.2_mix.pt`).
   - Sau khi hoàn thành Bảng 4, chọn checkpoint ST-GCN có Val Acc cao nhất $\rightarrow$ `<BEST_STGCN_CKPT>` (ví dụ: `checkpoints/best_STGCN_T4.1_raw_3d.pt`).
3. **Chạy Ensemble (Bảng 5 & 6):** Điền 2 checkpoint tốt nhất vào lệnh Bảng 5. Khi chạy Stacking (`--exp_id T5.3`), hệ thống sẽ tự động cập nhật kết quả Bảng 5 và sinh báo cáo chi tiết cho Bảng 6.

---

## III. Danh Sách CLI Thực Nghiệm

### Bảng 1: Temporal Models on Feature Sets

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

### Bảng 2: Data Augmentation on Best Transformer (2 runs)
*Chạy trên biến thể biểu diễn tốt nhất của Transformer từ Bảng 1 (mặc định: `mix`).*
```bash
python3 run.py train --model Transformer --feature mix --augment none --exp_id T2.1 --device auto --use_amp --in_memory
python3 run.py train --model Transformer --feature mix --augment combined --exp_id T2.2 --device auto --use_amp --in_memory
```

---

### Bảng 3: Feature Fusion
*Đã bỏ vì phương pháp `mix` ở Bảng 1 đã thay thế hoàn toàn.*

---

### Bảng 4: ST-GCN on Raw vs. Relative Graph Streams (4 runs)
```bash
python3 run.py train --model STGCN --feature raw_3d --exp_id T4.1 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_3d --exp_id T4.2 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature raw_2d --exp_id T4.3 --device auto --use_amp --in_memory
python3 run.py train --model STGCN --feature rel_2d --exp_id T4.4 --device auto --use_amp --in_memory
```

---

### Bảng 5: Heterogeneous Ensemble (Chỉ chạy sau khi có Best Transformer & Best ST-GCN)
*Thay `<BEST_TRANSFORMER_CKPT>` và `<BEST_STGCN_CKPT>` bằng đường dẫn checkpoint thực tế tốt nhất:*
```bash
python3 run.py ensemble --method hard --exp_id T5.1 --checkpoints <BEST_TRANSFORMER_CKPT> <BEST_STGCN_CKPT> --device auto
python3 run.py ensemble --method soft --exp_id T5.2 --checkpoints <BEST_TRANSFORMER_CKPT> <BEST_STGCN_CKPT> --device auto
python3 run.py ensemble --method stacking --exp_id T5.3 --checkpoints <BEST_TRANSFORMER_CKPT> <BEST_STGCN_CKPT> --device auto
```

---

### Bảng 6: Classification Report của Best Ensemble (Stacking)
*Tự động sinh và cập nhật vào `outputs/EXPERIMENT_RESULTS.md` khi chạy lệnh Stacking ở Bảng 5 (`--exp_id T5.3`).*