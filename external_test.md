Được. Với mục tiêu publication, mình đề xuất **không gộp Fit3D và MM-Fit thành một “External Test” duy nhất**. Nên thiết kế thành **hai external benchmarks độc lập**, nhưng chạy qua cùng một evaluation framework để kết quả so sánh được và reviewer thấy protocol nhất quán.

Repo hiện tại là `Gym_Classification` bạn đã cung cấp :chatgpt-content-reference{index="0"}. Kiến trúc hiện tại dùng 13 joints, window `T=32`, các representation `rel_3d`, `bone_3d`, `joint_motion_3d`, `bone_motion_3d`, `mix=117`, rồi Transformer/AAGCN/SkelGym-Lite/Full. Plan dưới đây giữ nguyên các model đã train, **không retrain trên external dataset trong experiment chính**.

# 1. Mục tiêu khoa học

Hai External Test sẽ trả lời hai câu hỏi khác nhau:

**Fit3D: Cross-dataset / cross-sensor generalization.** Đây là benchmark mạnh nhất vì Fit3D có ground-truth Vicon 3D, 25 joints, 47 exercises, 50 fps và repetition annotations. Trang chính thức hiện mô tả 611 recordings, 8 train subjects + 3 test subjects. :chatgpt-content-reference{index="1"}

**MM-Fit: Cross-dataset / cross-subject / in-home generalization.** MM-Fit có 10 exercise classes, RGB-D ở 30 Hz, 2D/3D pose và 21 workout sessions. Các class chính thức gồm squat, push-up, shoulder press, lunge, row, sit-up, triceps extension, biceps curl, lateral raise, jumping jack. :chatgpt-content-reference{index="2"}

Cả hai benchmark nên có **hai input protocols**:

1. **Protocol A — Same Pose Pipeline, PRIMARY:** lấy RGB external → chạy lại **MediaPipe Pose Heavy giống hệt SkelGym** → 13 joints → feature SkelGym → frozen model.
2. **Protocol B — Native Skeleton Transfer, SECONDARY:** dùng trực tiếp 3D skeleton do external dataset cung cấp → joint remapping + coordinate normalization → frozen model.

Protocol A nên là kết quả chính trong paper. Protocol B là robustness analysis.

Lý do: SkelGym hiện extract `pose_landmarks` bằng MediaPipe, trong khi Fit3D dùng Vicon và MM-Fit cung cấp skeleton 3D từ pipeline khác. Nếu dùng native skeleton làm kết quả chính, reviewer có thể hỏi kết quả kém/tốt là do classifier hay do khác hệ tọa độ/pose estimator.

---

# 2. Nguyên tắc bắt buộc để External Test thực sự hợp lệ

**Không được:**

- retrain SkelGym bằng Fit3D/MM-Fit trước external evaluation;
- chọn checkpoint dựa trên external test accuracy;
- tính z-score mean/std từ external test;
- tune ensemble weight trên external dataset;
- thử nhiều preprocessing rồi lấy preprocessing cho kết quả tốt nhất trên test;
- dùng một repetition của cùng recording làm support và repetition khác của recording đó làm query trong one-shot.

**Phải:**

- freeze checkpoint;
- freeze SLSQP ensemble weights được học từ **SkelGym validation**;
- dùng normalization statistics của **SkelGym train**;
- mapping class/joint được khai báo trước;
- aggregate window → recording/video;
- bootstrap CI ở recording/subject level;
- lưu prediction CSV để experiment audit được.

Đây là phần quyết định External Test có thuyết phục reviewer hay không.

---

# 3. Kiến trúc code chung

Không nên tạo hai script hoàn toàn độc lập. Tạo một framework external:

```text
src/
└── external/
    ├── __init__.py
    ├── base.py
    ├── canonical_pose.py
    ├── class_mapping.py
    ├── fit3d.py
    ├── mmfit.py
    ├── metrics.py
    └── fewshot.py

configs/
└── external/
    ├── fit3d.yaml
    └── mmfit.yaml

scripts/
├── prepare_fit3d_external.py
├── prepare_mmfit_external.py
├── evaluate_external.py
└── evaluate_external_fewshot.py

outputs/
└── external/
    ├── fit3d/
    └── mmfit/
```

Interface chung:

```python
ExternalDataset
    .records
    .get_pose(record_id)
    .get_label(record_id)
    .get_subject(record_id)
    .get_group_id(record_id)
```

Mỗi sample cuối cùng phải được convert về format canonical:

```text
Frame
NOSE_x/y/z
LEFT_SHOULDER_x/y/z
RIGHT_SHOULDER_x/y/z
...
RIGHT_ANKLE_x/y/z
```

Tức đúng **13-joint SkelGym schema**.

---

# 4. Phase 0 — Freeze SkelGym reference model

Trước khi đụng external data, tạo:

```text
artifacts/reference/
├── class_names.json
├── normalization_mix.npz
├── normalization_rel3d.npz
├── normalization_bone3d.npz
├── ensemble_weights.json
└── checkpoint_manifest.json
```

Manifest phải ghi:

```yaml
model:
  transformer: ...
  aagcn_bone: ...
  aagcn_rel: ...
  aagcn_joint_motion: ...
  aagcn_bone_motion: ...

normalization_source: skelgym_train_only
ensemble_calibration_source: skelgym_validation_only

seq_len: 32
test_stride: 32
seed: 42
```

Điểm rất quan trọng: hiện `get_dataloaders()` tính train mean/std rồi normalize val/test. External loader không được tự tính external mean/std.

Nên refactor thành:

```python
stats = load_normalization_stats(...)
external_ds = build_external_dataset(...)
apply_normalization(external_ds, stats.mean, stats.std)
```

---

# 5. External Test 1 — Fit3D

## 5.1 Data acquisition

Fit3D hiện yêu cầu login; download page liệt kê khoảng **18 GB training + 1.4 GB testing**. :chatgpt-content-reference{index="3"}

License cho phép non-commercial scientific research nhưng cấm redistribution. Vì vậy repo SkelGym chỉ commit:

```text
data_external/fit3d/README.md
```

và **không commit Fit3D data**. :chatgpt-content-reference{index="4"}

README local hướng dẫn:

```text
FIT3D_ROOT=/path/to/Fit3D
```

---

# 6. Fit3D class mapping

Không map toàn bộ 47 classes. Chỉ dùng classes có semantic overlap hợp lý với taxonomy 22-class SkelGym.

Candidate core set sau khi verify chính xác bằng annotations tải về:

| Fit3D | SkelGym | Status |
|---|---|---|
| `deadlift` | `deadlift` | exact |
| `squat` | `squat` | exact |
| `pushup` | `push-up` | exact |
| `side_lateral_raise` | `lateral raise` | strong |
| `dumbbell_overhead_shoulder_press` | `shoulder press` | subtype |
| `neutral_overhead_shoulder_press` | `shoulder press` | subtype |
| `dumbbell_hammer_curls` | `hammer curl` | strong |
| `dumbbell_biceps_curls` | `barbell biceps curl` | variant |

Không nên merge `dumbbell_biceps_curls → barbell biceps curl` vào experiment chính ngay lập tức vì equipment và arm execution khác.

Do đó nên có:

**Fit3D-Core-6**

```text
deadlift
squat
push-up
lateral raise
shoulder press
hammer curl
```

và supplementary:

**Fit3D-Extended-7**

```text
Core-6
+ biceps curl → barbell biceps curl
```

Việc xác nhận tên label cuối cùng phải lấy trực tiếp từ downloaded Fit3D metadata trước khi chạy.

---

# 7. Fit3D Protocol A — MediaPipe re-extraction

Đây là experiment chính.

Pipeline:

```text
Fit3D RGB
   ↓
MediaPipe Pose Heavy
   ↓
33 landmarks
   ↓
same 13 SkelGym joints
   ↓
segment using Fit3D repetition annotation
   ↓
32-frame windows
   ↓
SkelGym feature extraction
   ↓
SkelGym train normalization
   ↓
Frozen checkpoints
   ↓
Open-set / Closed-set / Video consensus
```

Fit3D RGB là 50 fps, trong khi SkelGym video thường khác fps. Không nên trực tiếp lấy 32 Fit3D frames nếu muốn temporal duration tương đương.

Có hai lựa chọn.

### Recommended

Resample external skeleton về **30 fps canonical** trước:

```text
50 fps Fit3D
→ temporal interpolation
→ 30 fps
→ T=32
```

Một window ≈ 1.07 s.

Điều này gần với pipeline MediaPipe/video phổ biến hơn và tránh Fit3D window chỉ ≈0.64 s.

Cấu hình:

```yaml
external:
  dataset: fit3d
  protocol: mediapipe
  target_fps: 30
  seq_len: 32
  stride: 32
  segmentation: repetition
```

---

# 8. Fit3D Protocol B — Native Vicon 3D

Fit3D có **25-joint ground-truth 3D skeleton**, bao gồm 17 Human3.6M joints. :chatgpt-content-reference{index="5"}

Mapping:

```text
Fit3D LEFT_SHOULDER → SkelGym LEFT_SHOULDER
LEFT_ELBOW          → LEFT_ELBOW
LEFT_WRIST          → LEFT_WRIST
LEFT_HIP            → LEFT_HIP
LEFT_KNEE           → LEFT_KNEE
LEFT_ANKLE          → LEFT_ANKLE
... right side ...
```

Vấn đề duy nhất là `NOSE`.

Human3.6M-style skeleton có head/neck chứ không thực sự có MediaPipe nose. Không nên giả vờ chúng giống nhau.

Nên tạo:

```text
NOSE_EXTERNAL = Neck/Nose or Head proxy
```

và khai báo rõ đây là **proxy cranial landmark**.

Sau đó canonical normalization:

### Translation

```python
hip_mid = (L_HIP + R_HIP) / 2
X = X - hip_mid
```

### Scale

Native Vicon tính bằng metric units, MediaPipe không cùng scale.

Normalize bằng shoulder–hip body scale:

```text
scale =
0.5 * (
 distance(mid_shoulder, mid_hip)
 + mean femur length
)
```

hoặc đơn giản hơn:

```text
torso_length = ||mid_shoulder - mid_hip||
X /= torso_length
```

Mình ưu tiên torso normalization.

### Orientation

Canonical vertical axis trước.

Sau đó tạo body coordinate system:

```text
left-right axis = R_HIP - L_HIP
vertical axis   = mid_shoulder - mid_hip
forward axis    = cross(left-right, vertical)
```

Rotate skeleton về person-centric frame.

Lúc đó yaw-camera variation được loại bỏ một cách deterministic.

---

# 9. Fit3D evaluation modes

Chạy ba task.

### F1 — Open-set Zero-shot

Input Fit3D thuộc 6 classes nhưng model vẫn được phép prediction toàn bộ 22 SkelGym classes:

```python
pred = argmax(p_22)
```

Đây là test khó nhất.

Metrics:

```text
Window Accuracy
Window Macro F1
Recording Accuracy
Recording Macro F1
Per-class Recall
```

### F2 — Closed-set Zero-shot

Chỉ giữ xác suất 6 classes:

```python
p6 = p22[fit3d_indices]
p6 /= p6.sum()
```

Không retrain.

Cho biết model có nhận ra kinematics nếu biết trước hypothesis space.

### F3 — 1-shot external metric transfer

Frozen embeddings.

Mỗi trial:

```text
1 recording / class = support
all remaining recordings = query
```

Classifier:

```text
cosine similarity
nearest support/prototype
```

Không dùng windows của cùng recording ở cả support/query.

Run:

```text
100 trials
seed = 42
```

Báo:

```text
mean
SD
2.5–97.5 percentile CI
```

---

# 10. Fit3D aggregation

Fit3D có repetition segmentation cho từng recording; mỗi recording chứa >5 repetitions. :chatgpt-content-reference{index="6"}

Nên báo 3 levels:

```text
window
repetition
recording
```

Aggregation:

```python
repetition_prob = mean(window_probs_in_rep)
recording_prob = mean(repetition_probs)
```

Điều này tốt hơn chỉ video consensus vì Fit3D có ground-truth repetition boundary.

---

# 11. External Test 2 — MM-Fit

MM-Fit khác Fit3D khá nhiều nên không copy nguyên protocol Fit3D.

Official dataset có RGB-D 30 Hz cùng 2D/3D pose estimates. :chatgpt-content-reference{index="7"}

RGB videos cũng hiện có trên Zenodo; record hiện khoảng 39 GB cho RGB version. :chatgpt-content-reference{index="8"}

---

# 12. MM-Fit class mapping

SkelGym không có đủ 10 class MM-Fit.

Mình đề xuất hai sets.

## Core-4

```text
MM-Fit                       → SkelGym
------------------------------------------------
squats                       → squat
pushups                      → push-up
dumbbell_shoulder_press      → shoulder press
lateral_shoulder_raises      → lateral raise
```

Đây là set sạch nhất.

## Extended-5

Thêm:

```text
bicep_curls → barbell biceps curl
```

Nhưng phải ghi rõ đây là **cross-equipment variant**:

```text
alternating dumbbell curl
vs
barbell curl
```

Không map:

```text
lunges
situps
dumbbell_rows
tricep_extensions
jumping_jacks
```

vì SkelGym không có exact equivalent.

Đặc biệt:

```text
dumbbell_rows != t bar row
tricep_extensions != tricep pushdown
```

Không nên ép mapping để tăng N.

---

# 13. MM-Fit split

MM-Fit paper/code có split chính thức:

```text
Train:
01 02 03 04 06 07 08 16 17 18

Validation:
14 15 19

Seen-subject test:
09 10 11

Unseen-subject test:
00 05 12 13 20
``` :chatgpt-content-reference{index="9"}


Đối với SkelGym external validation:

### Primary

Chỉ dùng:

```text
MM-Fit unseen-subject test
w00, w05, w12, w13, w20
```

Đây là lựa chọn mạnh nhất.

### Secondary

Report thêm:

```text
seen-subject test
w09, w10, w11
```

Nhưng không trộn hai nhóm.

Paper sẽ có:

```text
MM-Fit External — Unseen Subjects
MM-Fit External — Seen Subjects
```

---

# 14. MM-Fit Protocol A — RGB → MediaPipe

Đây cũng là primary.

```text
MM-Fit RGB
30 fps
↓
MediaPipe Heavy
↓
33 joints
↓
13 SkelGym joints
↓
MM-Fit action timestamps
↓
T=32
↓
same feature functions
↓
SkelGym train normalization
↓
frozen model
```

MM-Fit camera 30 Hz nên **không cần temporal resampling**. :chatgpt-content-reference{index="10"}

Đây là một lợi thế lớn.

---

# 15. MM-Fit segmentation

Không dùng toàn workout video thành một sample.

MM-Fit labels định nghĩa activity ranges.

Chỉ lấy:

```text
[start_frame, end_frame, action]
```

Exclude hoàn toàn:

```text
non_activity
```

Sau đó:

```text
segment
→ windows T=32
→ stride=32
```

Không tạo window vượt qua boundary hai exercises.

---

# 16. MM-Fit Protocol B — native 3D pose

MM-Fit paper mô tả 17-joint 3D pose, person-relative với hip-center làm reference; representation sau loại reference còn 16 spatial joints. :chatgpt-content-reference{index="11"}

Vì thế adapter:

```python
mmfit17_to_skelgym13()
```

Mapping 12 limb joints khá trực tiếp.

`NOSE` lại phải dùng head/neck proxy tương tự Fit3D.

Sau mapping:

```text
root center
orientation normalization
scale normalization
30 fps
T=32
```

Không dùng MM-Fit's cylindrical representation. Ta cần raw Cartesian joints rồi tự chạy feature engineering SkelGym.

---

# 17. MM-Fit evaluation modes

Giống Fit3D:

### M1 Open-set Core-4

```text
argmax over 22 classes
```

Đây sẽ là metric quan trọng nhất.

### M2 Closed-set Core-4

```text
condition logits/probs on:
squat
push-up
shoulder press
lateral raise
```

### M3 Extended-5

Thêm biceps curl, report riêng.

### M4 One-shot Core-4

Mỗi trial:

```text
1 full action segment / class → support
remaining independent segments/workouts → query
```

Quan trọng:

support/query phải khác **workout**.

Tốt hơn nữa:

support/query khác **subject** nếu metadata cho phép.

100 trials, fixed seed.

---

# 18. Một test rất giá trị: Pose-source ablation

Sau khi có A và B cho cả hai datasets, ta sẽ có bảng:

| Dataset | External pose source | Purpose |
|---|---|---|
| Fit3D | MediaPipe from RGB | end-to-end domain transfer |
| Fit3D | native Vicon | representation robustness |
| MM-Fit | MediaPipe from RGB | end-to-end domain transfer |
| MM-Fit | native lifted 3D | representation robustness |

Điều này tạo một experiment khá mạnh.

Nếu:

```text
MediaPipe > native
```

thì vấn đề chủ yếu là pose-domain mismatch.

Nếu:

```text
native > MediaPipe
```

thì pose quality là bottleneck.

Nếu cả hai tốt:

```text
representation thực sự generalize
```

Đây là một analysis reviewer-friendly hơn nhiều so với chỉ báo một accuracy.

---

# 19. Models phải chạy

Không cần chạy mọi baseline trong external test.

Primary table chỉ nên có:

```text
ST-GCN Rel3D
Transformer Mix
AAGCN Bone
SkelGym-Lite
SkelGym-Full
```

Tức giống spirit của Table 8 cũ.

Nếu chi phí thấp, supplementary thêm:

```text
LSTM
BiLSTM
4-stream components
```

---

# 20. Ensemble handling

Một lỗi cần tránh:

```python
external_validation → SLSQP.fit(...)
```

Tuyệt đối không.

Phải:

```text
weights learned on SkelGym validation
↓ freeze
↓ external test
```

Ví dụ:

```python
external_prob =
    w_transformer * p_transformer
  + w_bone * p_bone
  + w_rel * p_rel
  + w_joint_motion * p_joint_motion
  + w_bone_motion * p_bone_motion
```

Không recalibrate.

---

# 21. Normalization experiment

Mình đề xuất primary:

```text
SkelGym train z-score only
```

Secondary ablation:

```text
A. source normalization
B. per-sequence geometric normalization
```

Nhưng **không report external-dataset z-score adaptation như zero-shot**, vì nó dùng target distribution.

Canonical geometric normalization thì được vì không dùng labels/statistical population:

```text
hip centering
torso scaling
person-centric orientation
```

---

# 22. Confidence intervals

Không bootstrap individual windows vì chúng highly correlated.

Primary CI:

### Fit3D

bootstrap **recordings/subjects**.

### MM-Fit

bootstrap **workout sessions** hoặc subjects.

```text
B = 1,000
95% percentile CI
```

Sau đó window CI chỉ supplementary.

---

# 23. Statistical comparisons

Nếu cần so:

```text
Transformer vs SkelGym-Full
ST-GCN vs SkelGym-Full
Native pose vs MediaPipe pose
```

Video/recording classification:

**paired bootstrap difference** là lựa chọn an toàn.

Ví dụ:

```text
Δ Accuracy
95% CI of Δ
```

Không cần quá nhiều p-values.

External dataset N thường nhỏ, effect size + CI có ý nghĩa hơn.

---

# 24. Error analysis

Cho mỗi dataset tự động tạo:

```text
confusion_matrix_open.png
confusion_matrix_closed.png
per_class_recall.csv
misclassified_records.csv
```

Mỗi error record:

```csv
dataset,
subject,
recording,
true_class,
pred_class,
confidence,
second_class,
margin,
pose_source
```

Sau đó đặc biệt phân tích:

### Fit3D

```text
deadlift ↔ squat
hammer curl ↔ biceps curl
lateral raise ↔ shoulder press
```

### MM-Fit

```text
lateral raise ↔ shoulder press
squat ↔ non-corresponding SkelGym lower-body classes
biceps curl ↔ hammer curl
```

Open-set sẽ đặc biệt hữu ích vì cho biết probability leak sang 22-class distractors nào.

---

# 25. Domain-gap diagnostics

Ngoài accuracy, thêm một experiment nhỏ nhưng rất hữu ích.

Lấy frozen Transformer embeddings:

```text
SkelGym test
Fit3D
MM-Fit
```

Sau đó tính:

```text
class centroid cosine distance
intra-class distance
inter-class distance
```

Không dùng t-SNE làm evidence chính.

Có thể report:

```text
mean cross-domain centroid similarity
```

Ví dụ:

```text
SkelGym squat ↔ Fit3D squat
SkelGym squat ↔ MM-Fit squat
```

Điều này giúp giải thích tại sao một dataset transfer tốt hơn dataset kia.

---

# 26. Quality-control gate

Trước inference phải có automatic checks:

```text
NaN ratio
zero-frame ratio
joint completeness
pose detection success
segment length
fps correctness
left/right consistency
scale distribution
```

Tạo:

```text
external_data_audit.csv
```

Các record fail:

```text
pose_success < 80%
segment < 16 frames
>20% zero frames
```

không được silently drop.

Phải log:

```text
Excluded: n / total
reason
class distribution before/after
```

---

# 27. Cấu hình Fit3D

```yaml
dataset: fit3d
mode: external

class_set: core6

pose_protocol:
  primary: mediapipe
  secondary: native

fps:
  source: 50
  target: 30

window:
  seq_len: 32
  stride: 32

normalization:
  geometric: true
  feature_stats: skelgym_train

evaluation:
  open_set: true
  closed_set: true
  one_shot: true
  trials: 100
  seed: 42

aggregation:
  window: true
  repetition: true
  recording: true
```

---

# 28. Cấu hình MM-Fit

```yaml
dataset: mmfit
mode: external

class_set: core4

workout_split:
  primary:
    - "00"
    - "05"
    - "12"
    - "13"
    - "20"

pose_protocol:
  primary: mediapipe
  secondary: native

fps:
  source: 30
  target: 30

window:
  seq_len: 32
  stride: 32

normalization:
  geometric: true
  feature_stats: skelgym_train

evaluation:
  open_set: true
  closed_set: true
  one_shot: true
  trials: 100
  seed: 42
```

---

# 29. CLI cuối cùng

Mục tiêu nên chạy được:

```bash
python scripts/prepare_fit3d_external.py \
    --root /data/Fit3D \
    --pose-source mediapipe
```

```bash
python scripts/evaluate_external.py \
    --config configs/external/fit3d.yaml
```

và:

```bash
python scripts/prepare_mmfit_external.py \
    --root /data/MMFit \
    --pose-source mediapipe
```

```bash
python scripts/evaluate_external.py \
    --config configs/external/mmfit.yaml
```

Native:

```bash
python scripts/evaluate_external.py \
    --config configs/external/fit3d.yaml \
    --pose-source native
```

---

# 30. Output bắt buộc

Mỗi benchmark sinh:

```text
outputs/external/fit3d/
├── dataset_audit.json
├── class_mapping.json
├── joint_mapping.json
├── predictions_window.csv
├── predictions_recording.csv
├── metrics_open.csv
├── metrics_closed.csv
├── fewshot_100_trials.csv
├── bootstrap_ci.json
├── confusion_open.png
├── confusion_closed.png
└── run_manifest.json
```

MM-Fit tương tự.

`run_manifest.json` phải ghi SHA checkpoint và Git commit để reproducible.

---

# 31. Tables mình đề xuất cho paper

### Table X — Genuine Cross-Dataset External Validation

| External Dataset | Model | Classes | Open-set Window | Open-set Recording | Closed-set Recording | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| Fit3D | ST-GCN | 6 | | | | |
| Fit3D | Transformer | 6 | | | | |
| Fit3D | SkelGym-Lite | 6 | | | | |
| Fit3D | SkelGym-Full | 6 | | | | |
| MM-Fit unseen | ST-GCN | 4 | | | | |
| MM-Fit unseen | Transformer | 4 | | | | |
| MM-Fit unseen | SkelGym-Lite | 4 | | | | |
| MM-Fit unseen | SkelGym-Full | 4 | | | | |

Không so trực tiếp Fit3D 6-class accuracy với MM-Fit 4-class accuracy như thể task giống nhau.

### Table Y — Pose-source Robustness

| Dataset | Pose | Transformer | SkelGym-Full |
|---|---|---:|---:|
| Fit3D | MediaPipe | | |
| Fit3D | Native Vicon | | |
| MM-Fit | MediaPipe | | |
| MM-Fit | Native 3D | | |

### Table Z — One-shot Cross-Dataset Transfer

| Dataset | Classes | Model | Mean ± SD | 95% CI |
|---|---:|---|---:|---:|
| Fit3D | 6 | Transformer | | |
| Fit3D | 6 | SkelGym-Full | | |
| MM-Fit | 4 | Transformer | | |
| MM-Fit | 4 | SkelGym-Full | | |

---

# 32. Thứ tự triển khai

Mình sẽ chia implementation thành **8 bước khóa tuần tự**:

1. **Refactor normalization + frozen evaluation** — tách train statistics khỏi `get_dataloaders`.
2. **Canonical external pose schema** — 13 joints + coordinate convention.
3. **Fit3D adapter** — metadata, repetition boundaries, MediaPipe/native converters.
4. **MM-Fit adapter** — workout labels, unseen split, MediaPipe/native converters.
5. **Unified external evaluator** — open/closed set + recording aggregation.
6. **Few-shot evaluator** — frozen embeddings, support/query group isolation.
7. **Audit/statistics** — bootstrap, QC, prediction manifests.
8. **Publication sync** — thay Table 8 cũ bằng genuine Fit3D + MM-Fit results và giữ Deyzel-inspired experiment thành supplementary/protocol-aligned benchmark.

## Ưu tiên triển khai

Mình sẽ làm **MM-Fit trước, Fit3D sau**.

MM-Fit hợp pipeline hiện tại hơn vì RGB chạy **30 Hz đúng với canonical rate**, labels và official unseen-subject split đã rõ, 4 classes mapping khá sạch, và RGB có thể tải trực tiếp từ Zenodo. :chatgpt-content-reference{index="12"}

Sau khi MM-Fit framework chạy ổn, Fit3D chỉ cần thêm adapter + resampling 50→30 Hz + repetition hierarchy. Fit3D sau đó sẽ trở thành benchmark mạnh hơn nhờ Vicon ground truth, multi-view capture và exercise taxonomy lớn. :chatgpt-content-reference{index="13"}

**Target cuối cùng:** paper có thể tuyên bố một cách chính xác rằng SkelGym được đánh giá trên **hai genuinely independent external datasets**, với **zero-shot open-set**, **closed-set**, **one-shot transfer**, và **pose-source robustness**, thay vì “external benchmark” chỉ dùng subset của chính SkelGym.