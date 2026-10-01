Với MM-Fit, mình đề xuất khóa protocol ngay từ đầu theo hướng **external validation thật sự**, tức toàn bộ model/checkpoint/normalization/ensemble weights đều lấy từ SkelGym hiện tại :chatgpt-content-reference{index="0"}; MM-Fit chỉ đóng vai trò dữ liệu ngoài, tuyệt đối không dùng để tune model trước khi mở kết quả.

MM-Fit chính thức cung cấp RGB-D/pose ở 30 Hz và 10 exercise classes; starter code cũng định nghĩa `unseen_test = {w00,w05,w12,w13,w20}`. :chatgpt-content-reference{index="1"} RGB từng workout có thể tải riêng từ Zenodo, nên giai đoạn đầu chỉ cần khoảng 5 video unseen-test thay vì toàn bộ 39.1 GB. :chatgpt-content-reference{index="2"}

# 1. Protocol tổng thể cần khóa trước

Pipeline chính nên là:

```text
MM-Fit RGB
    │
    ├── official labels: start_frame, end_frame, reps, activity
    │
    ▼
MediaPipe Pose Heavy
    │
    ▼
33 normalized MediaPipe landmarks
    │
    ▼
13 SkelGym joints
    │
    ▼
exercise-set segmentation
    │
    ▼
same SkelGym feature engineering
    │
    ├── Mix 117-d
    ├── Rel-3D
    ├── Bone-3D
    ├── Joint Motion
    └── Bone Motion
    │
    ▼
SkelGym TRAIN normalization
    │
    ▼
Frozen SkelGym checkpoints
    │
    ├── Open-set 22-class
    ├── Closed-set MM-Fit subset
    └── Frozen-embedding 1-shot
    │
    ▼
Window → Exercise-set → Workout/Class consensus
```

Điểm quan trọng: với MM-Fit, **không gọi cả `w00_rgb.mp4` là một video sample**, bởi một MP4 chứa nhiều exercise khác nhau. Đơn vị tương đương “video-level” của SkelGym nên là **exercise-set-level**, tức một dòng annotation `(start_frame, end_frame, reps, activity)`.

---

# 2. Phase A — Download dữ liệu

## A1. Download MM-Fit core archive

Từ project page chính thức, tải archive MM-Fit chính. Nó chứa các modality đã đồng bộ, đặc biệt thứ ta cần là:

```text
wXX/
├── ...labels....csv
├── ...pose_2d....npy
├── ...pose_3d....npy
└── sensor files...
```

Starter code chính thức đọc label theo format:

```text
Start Frame
End Frame
Repetition Count
Activity
```

Đây là nguồn ground-truth segmentation chính; **không tự detect start/end từ RGB**.

Project page chính thức cung cấp download archive và xác nhận camera RGB/Depth chạy nominal 30 Hz. :chatgpt-content-reference{index="3"}

## A2. Chỉ download 5 unseen-test RGB trước

Primary external benchmark:

```text
w00_rgb.mp4    ~2.2 GB
w05_rgb.mp4    ~1.9 GB
w12_rgb.mp4    ~3.1 GB
w13_rgb.mp4    ~2.5 GB
w20_rgb.mp4    ~0.96 GB
```

Tổng khoảng **10.7 GB**, thay vì download toàn bộ 39.1 GB. Zenodo cung cấp từng MP4 và MD5 riêng. :chatgpt-content-reference{index="4"}

Sau khi pipeline hoàn thiện mới tải secondary seen-test:

```text
w09
w10
w11
```

Primary publication result vẫn phải là `unseen_test`.

## A3. Directory

Mình khuyên tổ chức:

```text
data_external/
└── mmfit/
    ├── raw/
    │   ├── official/
    │   │   ├── w00/
    │   │   ├── w05/
    │   │   ├── w12/
    │   │   ├── w13/
    │   │   └── w20/
    │   │
    │   └── rgb/
    │       ├── w00_rgb.mp4
    │       ├── w05_rgb.mp4
    │       ├── w12_rgb.mp4
    │       ├── w13_rgb.mp4
    │       └── w20_rgb.mp4
    │
    ├── landmarks/
    ├── metadata/
    └── cache/
```

Toàn bộ `data_external/mmfit/raw/` phải nằm trong `.gitignore`.

---

# 3. Phase B — Data audit trước khi chạy model

Chưa extract toàn bộ ngay. Đầu tiên tạo:

```text
scripts/audit_mmfit.py
```

Mỗi workout ghi:

```text
workout_id
rgb_frames
rgb_fps
rgb_duration
resolution
label_count
first_label_frame
last_label_frame
pose3d_first_frame
pose3d_last_frame
```

Quan trọng nhất là kiểm tra **frame indexing**.

Existing SkelGym extractor hiện đếm:

```python
frame_idx = 1
```

trong khi MM-Fit labels có frame IDs riêng.

Không được đoán MM-Fit là 0-based hay 1-based.

Cần so:

```text
MM-Fit label start/end
        ↕
official pose_3d[:, :, frame_id]
        ↕
decoded RGB frame index
```

Rồi tạo duy nhất một conversion:

```python
mmfit_frame_to_rgb_frame()
```

Ví dụ nếu audit xác nhận:

```text
MM-Fit 0 → RGB decoded frame 0
```

thì giữ nguyên.

Nếu lệch một frame:

```text
MM-Fit frame n → RGB frame n+1
```

thì ghi offset vào manifest.

Không được sửa thủ công từng workout.

---

# 4. Phase C — Class mapping

Official MM-Fit có:

```text
squats
lunges
bicep_curls
situps
pushups
tricep_extensions
dumbbell_rows
jumping_jacks
dumbbell_shoulder_press
lateral_shoulder_raises
non_activity
```

Official project page mô tả từng exercise. :chatgpt-content-reference{index="5"}

Mình sẽ định nghĩa hai benchmark.

## Core-4 — PRIMARY

| MM-Fit | SkelGym | Matching |
|---|---|---|
| `squats` | `squat` | rất gần exact |
| `pushups` | `push-up` | exact |
| `dumbbell_shoulder_press` | `shoulder press` | subtype/equipment variation |
| `lateral_shoulder_raises` | `lateral raise` | seated variation |

Đây là benchmark chính.

## Extended-5 — SUPPLEMENTARY

Thêm:

```text
bicep_curls
    ↓
barbell biceps curl
```

Nhưng MM-Fit thực hiện dumbbell alternating curl, còn SkelGym class là barbell curl, nên không coi đây là exact class mapping.

Không map:

```text
lunges
situps
tricep_extensions
dumbbell_rows
jumping_jacks
```

Ví dụ:

```text
dumbbell_rows ≠ t bar row
tricep_extensions ≠ tricep pushdown
```

và `non_activity` không dùng trong recognition benchmark chính.

Tạo file:

```text
configs/external/mmfit_class_mapping.yaml
```

```yaml
core4:
  squats: squat
  pushups: push-up
  dumbbell_shoulder_press: shoulder press
  lateral_shoulder_raises: lateral raise

extended5:
  bicep_curls: barbell biceps curl

excluded:
  - lunges
  - situps
  - tricep_extensions
  - dumbbell_rows
  - jumping_jacks
  - non_activity
```

---

# 5. Phase D — Build MM-Fit segment metadata

Từ labels CSV tạo một master table:

```text
mmfit_external_metadata.csv
```

Một row = **một exercise set**.

Ví dụ:

```csv
segment_id,workout_id,source_class,target_class,start_frame,end_frame,reps,split
w00_set_001,w00,squats,squat,1520,1845,10,unseen_test
w00_set_002,w00,pushups,push-up,2932,3238,10,unseen_test
...
```

Thêm:

```text
num_frames
rgb_path
landmark_path
mapping_type
```

Trong đó:

```text
mapping_type =
exact
near_exact
variant
```

Không đưa những class excluded vào inference Core-4.

---

# 6. Phase E — MediaPipe landmark extraction

Đây là phần cần giữ **giống SkelGym nhất có thể**.

SkelGym hiện dùng:

```text
MediaPipe Pose Heavy
model_complexity = 2
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
```

Và quan trọng nhất:

```python
result.pose_landmarks
```

chứ **không phải**:

```python
result.pose_world_landmarks
```

Do vậy MM-Fit primary protocol cũng phải dùng `pose_landmarks`.

Tức coordinates:

```text
x : normalized image coordinate
y : normalized image coordinate
z : MediaPipe relative depth
visibility
```

Không lấy MM-Fit native 3D skeleton trong primary experiment.

## Extract cả workout một lần

Không làm:

```text
segment → MediaPipe
segment → MediaPipe
segment → MediaPipe
```

Mà:

```text
w00_rgb.mp4
     ↓
MediaPipe entire video
     ↓
w00_mediapipe.csv
```

rồi mới slice segments.

Lý do:

- nhanh hơn;
- giữ tracking continuity;
- tránh restart detector tại đầu mỗi set;
- giống real-time deployment hơn.

Output:

```text
landmarks/
├── w00_mediapipe.csv
├── w05_mediapipe.csv
├── w12_mediapipe.csv
├── w13_mediapipe.csv
└── w20_mediapipe.csv
```

Columns:

```text
Frame
NOSE_x
NOSE_y
NOSE_z
NOSE_visibility
...
RIGHT_FOOT_INDEX_visibility
```

Giống hệt SkelGym.

---

# 7. Phase F — Landmark QC

Không chạy classifier ngay sau extraction.

Tạo:

```text
mmfit_landmark_audit.csv
```

cho mỗi exercise set:

```text
segment_id
frames
detected_frames
missing_frames
detection_rate
max_missing_run
zero_ratio
mean_visibility
```

Current SkelGym pipeline loại window nếu zero-frame ratio:

```text
> 20%
```

External phải giữ nguyên rule này.

Không đổi thành 10%, 30% hay 50% vì thấy accuracy.

## Visual audit

Random cố định, ví dụ seed 42:

```text
5 workouts
× 4 classes
× 1 segment
= 20 segments
```

render overlay MediaPipe skeleton lên RGB.

Chỉ kiểm tra:

- frame alignment;
- left/right đúng;
- landmark không lệch người;
- label segment đúng exercise.

Không nhìn classifier accuracy ở bước này.

---

# 8. Phase G — Temporal handling

MM-Fit nominal là 30 Hz, nên thông thường:

```text
source FPS = target FPS = 30
```

và không resample. :chatgpt-content-reference{index="6"}

Nhưng audit vẫn cần đọc video metadata.

Rule nên được khóa trước:

```text
if actual encoded FPS ∈ [29, 31]:
    preserve original frame sequence
else:
    resample landmarks to 30 Hz
```

Không resample chỉ vì kết quả classifier thấp.

Boundary segmentation vẫn dựa vào MM-Fit official frame IDs.

---

# 9. Phase H — 33 → 13 joints

Sử dụng trực tiếp SkelGym:

```text
NOSE
LEFT_SHOULDER
RIGHT_SHOULDER
LEFT_ELBOW
RIGHT_ELBOW
LEFT_WRIST
RIGHT_WRIST
LEFT_HIP
RIGHT_HIP
LEFT_KNEE
RIGHT_KNEE
LEFT_ANKLE
RIGHT_ANKLE
```

Không cần joint mapping phức tạp vì MediaPipe RGB extraction đã cho cùng 33-joint definition.

Đây chính là lý do MediaPipe-RGB protocol mạnh hơn native MM-Fit skeleton.

---

# 10. Phase I — Segmentation

Sau khi có full-workout landmark CSV:

```text
w00_mediapipe.csv
        +
w00_labels.csv
        ↓
exercise sets
```

Ví dụ:

```text
frames 1500–1830 → squats
frames 2900–3240 → pushups
```

Không cho window vượt:

```text
end squat → rest → start push-up
```

Mỗi segment được xử lý độc lập.

---

# 11. Phase J — Feature engineering

Không viết feature extractor riêng cho MM-Fit.

Phải gọi **chính code SkelGym hiện tại**.

### Transformer

```text
Mix 117-d
=
Relative 3D: 13 × 3 = 39
+
Pair elevation angles C(13,2) = 78
```

Tức:

```text
39 + 78 = 117
```

### AAGCN streams

```text
Rel 3D          = 39
Bone 3D         = 39
Joint Motion 3D = 39
Bone Motion 3D  = 39
```

### ST-GCN

```text
Rel 3D = 39
```

Không sửa coordinate sign, không invert y, không canonicalize body orientation trong primary experiment.

Đặc biệt: dù MediaPipe image `y` về mặt hình học tăng từ trên xuống dưới, model SkelGym đã train với representation đó. External test phải giữ cùng convention.

---

# 12. Phase K — Window extraction

Primary external protocol:

```text
T = 32
stride = 32
```

Tức non-overlapping windows.

Ở 30 Hz:

```text
32 frames ≈ 1.07 s
```

Giữ đúng val/test setup của SkelGym.

Một exercise set:

```text
300 frames
    ↓
32-frame windows
    ↓
~9 windows
```

Current logic được giữ:

```text
segment/window < 16 frames
→ discard

16 ≤ last chunk < 32
→ temporal interpolate → 32

zero ratio > 20%
→ discard
```

Không dùng augmentation ở external test.

---

# 13. Phase L — Normalization: phần quan trọng nhất

Ở Primary MM-Fit protocol **không tính normalization từ MM-Fit**.

Không được:

```python
mean = mmfit.mean()
std = mmfit.std()
```

Ngay cả không dùng labels thì đây vẫn là target-domain adaptation.

Phải dùng:

```text
SkelGym TRAIN mean/std
```

riêng cho:

```text
mix_117
rel_3d
bone_3d
joint_motion_3d
bone_motion_3d
```

Ví dụ:

```text
artifacts/reference/
├── mix_117_stats.npz
├── rel3d_stats.npz
├── bone3d_stats.npz
├── joint_motion_stats.npz
└── bone_motion_stats.npz
```

Mỗi file:

```text
mean: (D,)
std:  (D,)
```

External:

```python
x_mmfit = (x_mmfit - skelgym_train_mean) / skelgym_train_std
```

## Nếu stats chưa được save

Recompute từ:

```text
SkelGym original TRAIN split
```

bằng đúng commit/code/version dùng train checkpoint.

Không được recompute từ cả SkelGym train+val+test.

---

# 14. Phase M — Feature parity test

Trước MM-Fit inference, chọn một SkelGym test CSV cũ.

Chạy qua:

```text
old evaluation pipeline
vs
new external/common feature pipeline
```

Phải đảm bảo:

```text
max absolute feature difference < tolerance
same number of windows
same predictions
```

Mục tiêu là chứng minh adapter mới không âm thầm thay đổi feature extraction.

Đây là unit test rất quan trọng.

---

# 15. Phase N — Freeze model configuration

Một manifest:

```yaml
source_dataset: SkelGym
target_dataset: MM-Fit

target_split:
  - w00
  - w05
  - w12
  - w13
  - w20

classes:
  - squat
  - push-up
  - shoulder press
  - lateral raise

seq_len: 32
stride: 32

normalization: skelgym_train_only

mediapipe:
  model: pose_heavy
  detection_confidence: 0.5
  tracking_confidence: 0.5

ensemble:
  calibration: skelgym_validation_only
```

Sau khi file này khóa, mới chạy model metrics.

---

# 16. Evaluation 1 — Open-set zero-adaptation

Đây nên là benchmark quan trọng nhất.

Input chỉ thuộc Core-4, nhưng model vẫn prediction:

```text
22 SkelGym classes
```

Không restrict logits.

Ví dụ MM-Fit squat có thể bị model đoán:

```text
squat
deadlift
romanian deadlift
hip thrust
...
```

Và đó là lỗi hợp lệ.

Report:

```text
Window Accuracy
Window Macro F1
Balanced Accuracy
Per-class Recall
Outside-Core Prediction Rate
```

`Outside-Core Prediction Rate` rất hữu ích:

```text
# prediction thuộc 18 non-MMFit classes
---------------------------------------
# total MMFit samples
```

Nó cho biết cross-domain probability leakage.

---

# 17. Evaluation 2 — Closed-set zero-adaptation

Lấy xác suất 4 classes:

```text
squat
push-up
shoulder press
lateral raise
```

rồi:

```python
p4 = p22[:, core_indices]
p4 = p4 / p4.sum(axis=1, keepdims=True)
```

Không retrain classifier.

Report cùng metrics.

Ý nghĩa:

> Nếu model được biết trước observation thuộc một trong 4 exercise chung, representation có phân biệt đúng kinematics không?

---

# 18. Window-level evaluation

Một prediction / 32 frames.

Output:

```text
predictions_window.csv
```

Ví dụ:

```csv
workout,segment,window,true,pred,confidence
w00,w00_set01,0,squat,squat,0.91
w00,w00_set01,1,squat,deadlift,0.56
...
```

Đây là lowest-level metric.

Nhưng không nên coi hàng nghìn windows là hàng nghìn independent samples.

---

# 19. Exercise-set-level — primary consensus metric

Đây là **MM-Fit equivalent của video-level accuracy**.

Một official annotation row:

```text
start_frame
end_frame
reps
activity
```

được xem là một set.

Với windows:

```text
p1
p2
p3
...
pn
```

aggregate:

```python
p_set = mean(p_windows)
pred_set = argmax(p_set)
```

Report:

```text
Set Accuracy
Set Macro F1
Set Per-Class Recall
```

Đây nên là metric chính trong paper.

Không gọi là `Video Accuracy`.

Mình sẽ đặt tên:

> **Exercise-Set Consensus Accuracy**

và giải thích đây là analogue của SkelGym video consensus.

---

# 20. Workout-class-level

Có thể thêm level thứ ba.

Ví dụ `w00` có nhiều squat sets:

```text
w00 squat set 1
w00 squat set 2
w00 squat set 3
```

Aggregate tất cả:

```text
w00 + squat
```

thành một prediction.

Khi đó:

```text
5 unseen workouts × 4 classes
≈ 20 workout-class samples
```

Metric này đo liệu model có nhận đúng exercise của một participant/session khi nhìn nhiều sets.

Nó là supplementary, không thay set-level.

---

# 21. Không dùng whole-workout-level classification

Không làm:

```text
w00_rgb.mp4 → squat
```

vì trong một workout có:

```text
squat
pushup
lunge
curl
...
```

Nó không phải single-label video classification.

---

# 22. Evaluation 3 — 1-shot target transfer

Đây là experiment riêng với zero-shot classifier.

Không dùng classifier head.

Dùng:

```text
frozen penultimate embeddings
```

## Unit support

Một **exercise set**, không phải một window.

Ví dụ:

```text
support:
1 squat set
1 push-up set
1 shoulder-press set
1 lateral-raise set
```

## Set embedding

Nếu một set có embeddings:

```text
z1 ... zn
```

thì:

```python
z_set = mean(z1, ..., zn)
z_set = L2_normalize(z_set)
```

---

# 23. Strict support/query isolation

Ideal:

```text
support subject != query subjects
```

Nếu subject mapping từ MM-Fit metadata được xác nhận, bắt buộc dùng subject-disjoint.

Nếu chỉ xác nhận được workout identity thì:

```text
support workout != query workouts
```

và paper phải gọi đúng:

> workout-disjoint one-shot

không được gọi subject-disjoint.

Không được:

```text
set 1 from w00 → support
set 2 from w00 → query
```

vì background/camera/person giống hệt.

---

# 24. 1-shot trial

Một trial:

```text
Core classes C = 4

Choose support group
        ↓
randomly select 1 set/class
        ↓
4 support embeddings
        ↓
all valid sets from independent query groups
        ↓
cosine similarity
        ↓
nearest support
```

Classifier:

```python
similarity = query @ support.T
prediction = argmax(similarity)
```

Primary:

```text
100 trials
seed = 42
```

để consistent với External Test hiện tại.

Supplementary có thể chạy:

```text
1,000 trials
```

để ổn định CI hơn.

Report:

```text
Mean Accuracy
SD
Macro F1
95% percentile interval
per-class Recall
```

---

# 25. One-shot cho từng backbone

Nên chạy:

```text
Transformer Mix
AAGCN Bone
SkelGym-Lite
SkelGym-Full embedding
```

Transformer:

```text
mean pooled transformer encoder output
```

AAGCN:

```text
global average pooled graph feature
```

## Full embedding

Không nên lặp lại cách script cũ gọi concat Transformer + Bone là `SkelGym-Full`, vì Full thực tế có 5 constituent streams.

Có thể định nghĩa chính xác:

```text
z1 = normalized Transformer embedding
z2 = normalized Bone embedding
z3 = normalized Rel embedding
z4 = normalized Joint-Motion embedding
z5 = normalized Bone-Motion embedding
```

Với SLSQP weights từ **SkelGym validation**:

```text
w1 ... w5
```

xây:

\[
z_\text{Full}
=
[
\sqrt{w_1}z_1,
\sqrt{w_2}z_2,
...,
\sqrt{w_5}z_5
]
\]

Nếu:

\[
\sum_i w_i = 1
\]

thì cosine similarity giữa hai Full embeddings tương ứng với weighted combination của constituent similarities.

Không cần tune trên MM-Fit.

---

# 26. Models cho zero-shot benchmark

Main table:

```text
ST-GCN Rel3D
Transformer Mix
AAGCN Bone
SkelGym-Lite
SkelGym-Full
```

Không nhất thiết nhét mọi baseline vào main table.

LSTM/BiLSTM và từng AAGCN stream có thể đưa supplementary.

---

# 27. Multi-seed

Implementation phase đầu:

```text
seed 42 checkpoint
```

để xác minh pipeline.

Publication run:

```text
42
123
3407
```

Mỗi seed sử dụng:

```text
checkpoint riêng
source normalization tương ứng nếu preprocessing giống nhau thì stats chung
source-validation SLSQP weights tương ứng
```

Report:

```text
mean ± SD across training seeds
```

Điều này giúp external result nhất quán với claim multi-seed hiện tại của SkelGym.

---

# 28. Confidence interval

Không bootstrap windows độc lập làm primary CI.

Windows cùng set/workout correlated.

Ưu tiên:

```text
cluster bootstrap by workout
B = 1,000 or 2,000
```

Set-level có thể bootstrap với workout làm cluster.

Vì unseen split chỉ có 5 workout IDs, CI chắc chắn khá rộng; đây là đặc tính dataset, không nên che đi.

---

# 29. Error analysis

Open-set đặc biệt cần file:

```text
mmfit_open_set_errors.csv
```

Fields:

```text
workout
segment_id
source_class
target_class
predicted_class
confidence
runner_up
margin
num_windows
pose_detection_rate
```

Các confusion cần kiểm tra:

```text
squat
→ deadlift / Romanian deadlift / hip thrust

lateral raise
→ shoulder press

shoulder press
→ lateral raise

bicep curl Extended-5
→ hammer curl
```

Đây sẽ là phần analysis rất giá trị trong paper.

---

# 30. Required outputs

```text
outputs/external/mmfit/
├── run_manifest.yaml
├── data_audit.csv
├── frame_alignment_audit.csv
├── landmark_audit.csv
├── segment_metadata.csv
│
├── zero_shot/
│   ├── window_predictions.csv
│   ├── set_predictions.csv
│   ├── workout_class_predictions.csv
│   ├── metrics_open.json
│   ├── metrics_closed.json
│   ├── confusion_open.png
│   └── confusion_closed.png
│
├── one_shot/
│   ├── trial_results.csv
│   ├── support_query_manifest.csv
│   └── summary.json
│
└── bootstrap/
    └── confidence_intervals.json
```

`support_query_manifest.csv` đặc biệt quan trọng để chứng minh không leakage.

---

# 31. Code architecture nên thêm vào repo

```text
src/
└── external/
    ├── __init__.py
    ├── mmfit.py
    ├── common.py
    ├── evaluator.py
    └── fewshot.py

scripts/
├── download_mmfit_manifest.py
├── audit_mmfit.py
├── extract_mmfit_landmarks.py
├── build_mmfit_metadata.py
├── evaluate_mmfit_external.py
└── evaluate_mmfit_oneshot.py

configs/
└── external/
    └── mmfit.yaml
```

`src/data/features.py` **không duplicate**.

MM-Fit adapter phải gọi chính feature functions hiện có.

---

# 32. Thứ tự thực hiện thực tế

Mình sẽ triển khai theo thứ tự sau:

1. **Download** core MM-Fit archive + 5 unseen RGB videos.
2. **Audit** RGB FPS/frame count, label frame ranges và official pose frame IDs.
3. **Freeze class mapping Core-4**.
4. **Build segment metadata** từ official label CSV.
5. **Extract full-workout MediaPipe Heavy landmarks**.
6. **Audit frame/label alignment + pose quality** trước khi nhìn accuracy.
7. **Export/reconstruct SkelGym train normalization statistics**.
8. **Run feature parity test** trên SkelGym để xác nhận pipeline mới giống pipeline cũ.
9. **Generate MM-Fit windows/features** với `T=32`, stride 32.
10. **Freeze run manifest**, rồi mới chạy open-set/closed-set.
11. **Aggregate window → exercise-set → workout-class**.
12. **Run strict group-disjoint 1-shot**.
13. **Run 3 source-training seeds** cho final publication result.
14. **Bootstrap + error analysis + tables/figures**.

Điểm mình xem là quan trọng nhất trong toàn plan này là bước **6–10**: phải hoàn thành và khóa preprocessing **trước khi xem accuracy MM-Fit**. Nếu sau khi nhìn kết quả rồi mới đổi FPS handling, class mapping, normalization, window stride hoặc pose filtering, External Test sẽ rất dễ biến thành target-set tuning.

Bước triển khai hợp lý tiếp theo là **download + audit 5 unseen-test workouts và viết `mmfit_external_metadata.csv`**, vì từ đó ta sẽ biết chính xác số set Core-4, số frames, class balance và frame alignment trước khi chạy MediaPipe.