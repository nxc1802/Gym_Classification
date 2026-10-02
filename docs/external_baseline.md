# Kế hoạch tối giản: BlockGCN Single-Stream, 33 MediaPipe Landmarks, 3 Seeds

Mục tiêu: bổ sung một Strong External Baseline BlockGCN vào SkelGym, chỉ dùng một cấu hình huấn luyện cố định, không tuning hyperparameter, không ensemble và không giảm kích thước backbone để khớp ngân sách tham số của SkelGym.

| Architecture          | BlockGCN gốc, thích nghi graph đầu vào      |
| --------------------- | ------------------------------------------- |
| Input                 | 33 raw MediaPipe landmarks, tọa độ XYZ      |
| Tensor shape          | `[B, 3, 32, 33, 1]`                         |
| Stream                | Joint-only, không dùng Bone/Motion ensemble |
| Classes               | 22                                          |
| Pretrained weights    | Không, train from scratch                   |
| Seeds                 | 42, 123, 3407                               |
| Hyperparameter tuning | Không                                       |

Một lưu ý về tính trung thành với paper: BlockGCN gốc dùng skeleton NTU 25 joints, trong khi yêu cầu này dùng 33 MediaPipe joints và 32 frames. Vì vậy, kết quả cần được gọi là BlockGCN adapted to SkelGym, không phải tái lập nguyên xi thí nghiệm NTU. Toàn bộ cơ chế kiến trúc chính sẽ được giữ; chỉ sửa những thành phần phụ thuộc trực tiếp vào định dạng dữ liệu.

Ngoài ra, file `default.yaml` trong repo tác giả hiện khai báo `model.ctrgcn.Model`. Script `train.sh` mới ghi đè bằng `model.BlockGCN.Model`. Kế hoạch này sẽ chỉ định BlockGCN một cách tường minh để tránh vô tình huấn luyện CTR-GCN.

## 1. Chuẩn bị dữ liệu và graph 33 joints

Sử dụng toàn bộ 33 landmarks theo thứ tự `RAW_POINTS_33` của MediaPipe. Mỗi landmark chỉ lấy `x, y, z`, không đưa visibility vào backbone vì BlockGCN gốc sử dụng `in_channels=3`.

Pipeline dữ liệu cố định:

```
MediaPipe landmark CSV
       ↓
33 joints × XYZ
       ↓
Xử lý missing frames theo quy tắc SkelGym hiện có
       ↓
Sliding windows: 32 frames
  Train stride = 16
  Val/Test stride = 32
       ↓
BlockGCN joint preprocessing
       ↓
[B, 3, 32, 33, 1]
```

Cần tạo loader riêng cho 33 joints, vì `raw_3d` hiện tại của SkelGym chỉ lấy 13 joints, tương đương 39 features.

Graph được xây từ `POSE_CONNECTIONS_33`, gồm các cạnh self-link, inward và outward như cách xây adjacency của BlockGCN gốc. Cần xử lý các thành phần rời rạc trong graph MediaPipe, đặc biệt là face, mouth và torso, bằng một bộ cạnh kết nối cố định được công bố cùng implementation. Sau đó tính lại hop-distance encoding trên graph 33 joints.

Không bổ sung joint thứ 34, không loại bỏ landmarks và không tuning graph bằng validation accuracy.

## 2. Giữ nguyên BlockGCN architecture

Sử dụng trực tiếp logic từ [`model/BlockGCN.py`](https://github.com/ZhouYuxuanYX/BlockGCN/blob/main/model/BlockGCN.py) của tác giả, với các thành phần sau được bảo toàn:

- 10 khối GCN–TCN, channels 128→256 và các vị trí temporal downsampling gốc.
- BlockGC với grouped feature projection, adaptive graph và hop-distance relative positional encoding.
- Multi-scale temporal convolution, residual connections và topology branch sử dụng persistent homology.
- Topological feature injection vào các GCN–TCN stages.

Đây là các cơ chế cốt lõi tạo nên BlockGCN trong paper.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

openaccess.thecvf.com

+1



Chỉ thay đổi các thành phần bắt buộc:

| Tham số       | BlockGCN gốc | SkelGym adaptation |
| ------------- | ------------ | ------------------ |
| `num_point`   | 25           | 33                 |
| `num_person`  | 2            | 1                  |
| `window_size` | 64           | 32                 |
| `num_class`   | 60/120       | 22                 |
| `in_channels` | 3            | 3                  |
| Graph         | NTU          | MediaPipe 33       |

Trong source hiện tại, `TopoTrans.forward()` có `x.repeat(2, 1)` được hard-code cho hai người. Phải sửa thành xử lý theo `num_person=1` thực tế; đồng thời cập nhật BatchNorm, positional embeddings và topology tensors theo 33 joints.

Không được xóa topology branch, thay BlockGC bằng GCN thường hoặc giảm channels để đạt một số params mục tiêu.

## 3. Một file config duy nhất

Lấy training recipe NTU60/NTU120 của paper và official implementation làm cơ sở: SGD với Nesterov momentum 0.9, learning rate 0.05, weight decay `4e-4`, batch 64, warm-up 5 epochs, LR decay ở epochs 110/120 và tổng 140 epochs. Loss là cross-entropy.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.researchgate.net\&sz=32)

researchgate.net

+1



File đề xuất: `configs/external/blockgcn_original_33j_32f.yaml`.

```
model: BlockGCN
upstream_model: model.BlockGCN.Model
pretrained: false

model_args:
  num_class: 22
  num_point: 33
  num_person: 1
  in_channels: 3
  window_size: 32
  graph: graph.mediapipe33.Graph
  drop_out: 0

data:
  feature: raw_33_xyz
  stream: joint
  train_stride: 16
  val_test_stride: 32
  normalization: false

sampling:
  train_p_interval: [0.5, 1.0]
  eval_p_interval: [0.95]

augmentation:
  random_choose: false
  random_shift: false
  random_move: false
  random_rot: true

training:
  optimizer: SGD
  base_lr: 0.05
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0004
  loss: cross_entropy
  batch_size: 64
  warm_up_epoch: 5
  step: [110, 120]
  lr_decay_rate: 0.1
  num_epoch: 140
  early_stopping: false

evaluation:
  checkpoint_selection: best_val_accuracy
  seeds: [42, 123, 3407]
```

Đây là config cần tạo, chưa phải một file có sẵn trong repo.

Ba điều cần ghi rõ trong tài liệu:

- `T=32`, `V=33`, `M=1` và `num_class=22` là adaptation theo yêu cầu, không phải thông số paper gốc.
- Native joint preprocessing cần được chuyển từ spine-center của NTU sang hip midpoint của MediaPipe. Đầu vào vẫn xuất phát từ đầy đủ 33 raw XYZ landmarks, không dùng feature engineering `mix`, Bone hoặc Motion.
- `p_interval` được áp dụng trên các cửa sổ 32 frames rồi resample về T=32. Đây là cách thích nghi sampling gốc với window pipeline của SkelGym, không phải tái lập nguyên xi sampling trên các sequence NTU dài.

Không áp dụng SkelGym-Aug, z-score mặc định của `get_dataloaders()` hay bất kỳ hyperparameter search nào.

## 4. Kiểm thử và chạy ba seeds

Trước khi train thật, cần hoàn thành một lần kiểm thử tích hợp với các điều kiện: input `[B,3,32,33,1]` cho output `[B,22]`; graph adjacency có shape `[3,33,33]`; topology branch có gradient; loss và gradients hữu hạn; checkpoint lưu và nạp lại cho kết quả nhất quán.

Sau đó chạy ba lần độc lập, chỉ thay seed:

| Run            | Seed | Config  | Epochs |
| -------------- | ---- | ------- | ------ |
| BlockGCN-S42   | 42   | Cố định | 140    |
| BlockGCN-S123  | 123  | Cố định | 140    |
| BlockGCN-S3407 | 3407 | Cố định | 140    |

Mỗi run dùng đúng video-level train/validation/test split của SkelGym. Huấn luyện đủ 140 epochs; lưu checkpoint cuối và checkpoint tốt nhất theo validation accuracy. Tuyệt đối không dùng test accuracy để chọn checkpoint hoặc thay đổi config giữa các seeds.

## 5. Đánh giá và bàn giao

Với từng seed, báo cáo Window Accuracy, Window Macro-F1, Video Accuracy và Video Macro-F1. Để tính video-level prediction, lấy trung bình xác suất các cửa sổ thuộc cùng một video có nhãn nhất quán. Báo cáo kết quả tổng hợp dưới dạng mean ± standard deviation qua ba seeds.

Đo thêm tổng số trainable parameters, FLOPs/MACs thực tế trên đầu vào `[1,3,32,33,1]`, latency và peak GPU memory. Không sử dụng số params của paper NTU thay cho số đo của graph 33 joints.

Bộ artifact cuối cùng gồm một file YAML, ba checkpoints, ba training logs, prediction files, một bảng kết quả tổng hợp và một `ADAPTATION.md` mô tả khác biệt so với upstream. Giữ license và attribution của BlockGCN.

Phạm vi chốt: một BlockGCN Joint-only, một config cố định, ba seeds, ba lần train đầy đủ. Không có hai protocol, ablation, tuning, Bone/Motion streams hay ensemble.

Khi đưa vào paper SkelGym, gọi đây là “BlockGCN (CVPR 2024), adapted to 33-joint MediaPipe skeletons, trained from scratch”. Vì model sử dụng 33 raw joints trong khi một số baseline SkelGym chỉ sử dụng 13 joints, bảng so sánh phản ánh hiệu quả của hệ thống model + input representation, không phải một thí nghiệm cô lập tuyệt đối ảnh hưởng của architecture.