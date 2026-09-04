# Deep Learning Framework for Video-based Gym Exercise Classification

Hệ thống mã nguồn tái hiện và phát triển nghiên cứu **"Deep Learning Approaches for Video-based Gym Exercise Classification"** (xuất bản theo định dạng Springer LNCS).

Toàn bộ pipeline được chuẩn hóa 100% trên **PyTorch** với hỗ trợ tăng tốc phần cứng GPU (NVIDIA CUDA), kiểm thử tự động, cấu hình linh hoạt qua file YAML và điều khiển toàn diện qua giao diện dòng lệnh (**CLI**).

---

## 1. Cấu Trúc Dự Án

```
Final_Gym/
├── Final_dataset_metadata.csv         # Metadata chuẩn 1026 videos, 22 classes, 3 splits
├── run.py                             # Entrypoint chính của CLI
├── configs/                           # File cấu hình YAML
│   ├── default.yaml                   # Cấu hình siêu tham số mặc định
│   └── experiments/                   # Presets tái hiện từng bảng trong bài báo
│       ├── table2_spatial_models.yaml # Table 2: Temporal models on landmark features
│       ├── table3_augmentations.yaml  # Table 3: Transformer with augmentations
│       ├── table4_concatenations.yaml # Table 4: Branch-Concat vs Direct-Concat
│       ├── table5_stgcn.yaml          # Table 5: ST-GCN on Raw vs Relative
│       └── table6_ensembles.yaml      # Table 6 & 7: Ensemble voting & Stacking
├── src/                               # Package mã nguồn chính
│   ├── constants.py                   # 22 classes, 33/13 joints, skeleton graph, feature dims
│   ├── cli.py                         # Trình điều khiển CLI hoàn chỉnh
│   ├── data/
│   │   ├── download.py                # Tải tự động dataset từ Kaggle qua kagglehub
│   │   ├── report.py                  # Tạo báo cáo mô tả tập dữ liệu, đối soát Bảng 1, nén MediaPipe
│   │   ├── extractor.py               # Trích xuất MediaPipe 33 landmarks từ video (hỗ trợ Tasks API & Solutions)
│   │   ├── features.py                # Tính Relative coordinates, Angle-2 (78), Angle-3 (286), Concat
│   │   ├── augmentations.py           # Jitter, Rotation, Joint Dropout, Time Warping
│   │   └── dataset.py                 # PyTorch Dataset, Sliding Window (32 frames, stride 16/32), Smoke test
│   ├── models/
│   │   ├── lstm.py                    # LSTM, BiLSTM, BranchConcatModel
│   │   ├── transformer.py             # Positional Encoding, TransformerModel, BranchConcatTransformer
│   │   ├── stgcn.py                   # Spatial-Temporal Graph Convolutional Network
│   │   └── ensemble.py                # Hard Voting, Soft Voting, Stacking Meta-Classifier
│   ├── training/
│   │   ├── trainer.py                 # Training loop, Early Stopping, Checkpointing, Class Weights
│   │   └── metrics.py                 # Accuracy, Precision, Recall, F1, Confusion Matrix, LaTeX Table Export
│   └── utils/
│       ├── logger.py                  # Hệ thống log màu và file
│       ├── config.py                  # Đọc ghi cấu hình YAML
│       └── reproducibility.py         # Cố định seed cho PyTorch, CUDA, NumPy, Python
├── tests/
│   └── test_pipeline.py               # Bộ test tự động kiểm thử toàn bộ module (7/7 passed)
├── GYM_Proposal/                      # Bản đề cương gốc (IEEE format)
├── GYM_Publication/                   # Bản bài báo chỉnh sửa (Springer LNCS format)
└── Source Code/                       # Notebooks thử nghiệm ban đầu
```

---

## 2. Hướng Dẫn Sử Dụng Toàn Diện CLI

### A. Tải Dataset từ Kaggle (`download-dataset`)
Tải trực tiếp tập dữ liệu `truongnhatquangk18dn/the-gym-exercise-classification-dataset` từ Kaggle:
```bash
# Tải dataset về thư mục mặc định data/raw:
py run.py download-dataset --output_dir data/raw
```

---

### B. Kiểm chứng thống kê bài báo & Trích xuất MediaPipe (`data-report`)
Lệnh này tính toán các chỉ số thống kê từ metadata, đối soát chính xác với **Table 1** trong bài báo (`manuscript.tex`), đồng thời hỗ trợ trích xuất toàn bộ MediaPipe landmarks ra file CSV và đóng gói thành ZIP (`outputs/landmarks_smoketest.zip` hoặc `outputs/landmarks_dataset.zip`) để thuận tiện upload lên server mà **không cần upload video nặng**:
```bash
# 1. Chạy thống kê & kiểm chứng Bảng 1 (sinh file Markdown và mã nguồn LaTeX):
py run.py data-report --output_dir outputs/dataset_report

# 2. Smoke test: trích xuất MediaPipe cho 1 class (barbell biceps curl) với số mẫu tối thiểu:
py run.py data-report --raw_dir data/raw --extract_mediapipe --smoke_test --smoke_class "barbell biceps curl"

# 3. Trích xuất MediaPipe cho toàn bộ 1026 video của tập dữ liệu:
py run.py data-report --raw_dir data/raw --extract_mediapipe
```

---

### C. Huấn luyện thử nghiệm nhanh (`train --smoke_test`)
Chế độ `--smoke_test` giúp kiểm tra toàn bộ luồng huấn luyện, nạp tensor, forward/backward pass, EarlyStopping và tính toán ma trận nhầm lẫn trong thời gian ngắn nhất (2 epochs, minimal batch size, 1 class):
```bash
# Smoke test huấn luyện Transformer trên GPU:
py run.py train --model Transformer --feature 12rel_4 --smoke_test --smoke_class "barbell biceps curl" --landmark_dir data/landmarks --device cuda

# Smoke test huấn luyện BiLSTM:
py run.py train --model BiLSTM --feature 12rel_4 --smoke_test --smoke_class "barbell biceps curl" --landmark_dir data/landmarks --device cuda
```

---

### D. Huấn luyện mô hình đầy đủ (`train`)
Hỗ trợ các mô hình: `Transformer`, `STGCN`, `LSTM`, `BiLSTM`, `BranchConcat`.
Hỗ trợ các bộ đặc trưng: `12rel_4`, `full_rel_4`, `13_4`, `full_4`, `angle3`, `angle2`, `direct_concat`, `branch_concat`.
Hỗ trợ tăng cường dữ liệu: `none`, `jitter`, `rotate`, `joint_dropout`, `time_warp`.

```bash
# Huấn luyện Transformer với đặc trưng Rel. 12x4+1 (Table 2):
py run.py train --model Transformer --feature 12rel_4 --epochs 100 --batch_size 16 --lr 1e-4

# Huấn luyện Transformer với Data Augmentation Rotate (Table 3):
py run.py train --model Transformer --feature 12rel_4 --augment rotate --epochs 100

# Huấn luyện ST-GCN trên toàn bộ 33 khớp xương (Table 5):
py run.py train --model STGCN --feature full_4 --epochs 100

# Huấn luyện mô hình kết hợp 2 nhánh Branch-Concat (Table 4):
py run.py train --model BranchConcat --feature branch_concat --epochs 100
```

---

### E. Đánh giá mô hình & Xuất bảng báo cáo (`evaluate`)
```bash
py run.py evaluate \
  --checkpoint checkpoints/best_Transformer_12rel_4_aug_none.pt \
  --model Transformer \
  --feature 12rel_4 \
  --split test \
  --save_cm outputs/cm_transformer.png \
  --save_table7 outputs/table7_report.tex
```

---

### F. Kết hợp mô hình (`ensemble` - Table 6 & 7)
```bash
py run.py ensemble \
  --checkpoints \
      checkpoints/best_Transformer_branch_concat.pt \
      checkpoints/best_STGCN_full_4.pt \
      checkpoints/best_BiLSTM_branch_concat.pt \
  --method stacking \
  --output_dir outputs/ensemble
```

---

### G. Chạy chuỗi tái hiện kết quả tự động (`reproduce`)
```bash
# Chạy kiểm thử toàn bộ luồng nhanh (1 epoch dry-run):
py run.py reproduce --table all --dry_run

# Chạy tái hiện đầy đủ Bảng 2:
py run.py reproduce --table table2

# Chạy tái hiện đầy đủ Bảng 3 (Augmentations):
py run.py reproduce --table table3

# Chạy tái hiện đầy đủ Bảng 6 (Ensembles):
py run.py reproduce --table table6
```

---

## 3. Chạy Kiểm Thử Tự Động (Automated Testing)

Chạy bộ unit & integration tests để xác minh tính đúng đắn của toàn bộ module:
```bash
py tests/test_pipeline.py
```
Kết quả kiểm thử: **7/7 tests passed** (Feature Engineering, Augmentations, Windowing, Models Forward Pass, Ensembling, Metrics & LaTeX Export, Dataset Report Generation).
