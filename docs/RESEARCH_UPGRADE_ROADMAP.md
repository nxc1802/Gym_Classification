Mình đã đối chiếu lại `outputs/RESULTS_FINAL.md`. Với trạng thái hiện tại, thứ tự **thực thi experiment** không nên đi theo thứ tự hiển thị bảng. Đặc biệt, Table 3 hiện chứa nhiều experiment dùng `SkelGym-Aug (Proposed)`, trong khi thành phần Proposed lại phải được quyết định từ Table 4–5. Vì vậy execution order mới phải là:

**Table 1 → Table 2/2b → Table 4–5 → chốt Proposed Aug → Table 3 → Table 6 → Table 7–10 → Table 11/13 → SOT → Paper.**

`RESULTS_FINAL.md` hiện tại chỉ nên dùng để xác định phạm vi/thứ tự experiment legacy; nó vẫn còn các dòng SLSQP/Stacking nên không còn là nguồn sự thật cuối cùng.

---

# Phase 0 — Khóa migration specification và dependency

### Mục tiêu
Xác định chính xác project sau migration sẽ hiểu từng representation/model/experiment như thế nào, trước khi sửa code.

### Trạng thái đã được xem là hoàn thành

Không làm lại:

- WORLD vs REL validation.
- Transformer WORLD 3-seed experiment.
- Việc chứng minh `pose_world_landmarks` tốt hơn REL.
- Việc tạo feature `world_3d` cơ bản.

Kết quả WORLD Transformer hiện có trong Table 2b được coi là **decision evidence đã hoàn tất**.

### Việc cần làm

Khóa feature semantics mới:

```text
raw_3d
→ pose_landmarks
→ giữ nguyên

world_3d
→ pose_world_landmarks
→ 39-d
→ thay thế vai trò rel_3d trong pipeline mới

mix_v2
→ world_3d 39-d
 + world-based kinematic angles 24-d
→ 63-d

joint_motion_3d
→ temporal difference của WORLD joint coordinates

bone_3d
→ giữ semantics hiện tại nếu không phụ thuộc REL

bone_motion_3d
→ giữ semantics hiện tại nếu không phụ thuộc REL
```

Điểm quan trọng là không nên tiếp tục gọi WORLD stream là `rel_3d` trong paper nếu code thực tế dùng metric world coordinates. Có thể giữ alias cũ để compatibility, nhưng canonical config/result nên dùng tên rõ như:

```text
world_3d
world_joint_motion_3d
mix_v2_world
```

Lập dependency map của experiment:

```text
WORLD migration
├── Table 2 affected sequence rows
├── mix_v2
│   ├── Table 4
│   ├── Table 5
│   ├── Transformer Aug
│   ├── Table 6
│   └── Table 7+
├── STGCN World Joint
├── AAGCN World Joint
├── Joint Motion World
└── downstream ensembles
```

Khóa canonical fusion còn:

```text
Hard Voting
Uniform Soft Voting
Accuracy-Weighted Soft Voting
```

SLSQP và Stacking không còn tồn tại trong experiment matrix mới.

### Output của phase

Một migration specification ngắn, ví dụ:

```text
docs/WORLD_MIGRATION_V2.md
```

trong đó ghi:

- feature definitions;
- affected experiments;
- reusable experiments;
- experiments phải rerun;
- canonical seeds `42, 123, 3407`;
- naming/version mới cho checkpoint/artifact.

### Gate

Sang Phase 1 khi không còn ambiguity về feature nào dùng WORLD và feature nào giữ raw.

---

# Phase 1 — Hoàn thiện code WORLD migration + sửa augmentation infrastructure

Đây là phase code chính. Chưa chạy benchmark lớn.

## 1. WORLD feature routing

Dùng `world_3d` hiện đã tồn tại và đưa nó thành representation canonical cho nhánh trước đây dùng REL.

Cập nhật:

```text
dataset
feature dispatcher
experiment registry
training configs
evaluation scripts
checkpoint metadata
```

để các experiment WORLD load đúng landmark source.

`raw_3d` vẫn đọc dữ liệu normalized cũ.

---

## 2. Chuyển `mix_v2` sang WORLD

Hiện tại source vẫn đang:

```python
rel_n = extract_relative_norm_features(...)
ang_24 = compute_kinematic_angles_24(...)
return concat(rel_n, ang_24)
```

Canonical `mix_v2` mới phải là:

```text
39-d pose_world_landmarks
+
24-d angles tính từ cùng world landmarks
=
63-d
```

Transformer dual-branch vẫn có thể giữ:

```text
Branch A: 39
Branch B: 24
```

Không thay architecture nếu chưa có lý do khác.

---

## 3. Chuyển Joint Motion sang WORLD

Code hiện tại:

```text
joint_motion
→ extract_relative_features(...)
→ temporal difference
```

phải đổi thành:

```text
world_xyz(t+1) - world_xyz(t)
```

vì stream Joint Motion phụ thuộc trực tiếp vào representation joint.

Đây là lý do T3.6 sau này bắt buộc rerun.

---

## 4. Giữ Bone/Bone Motion đúng dependency

Code hiện tại cho:

```text
bone_3d
bone_motion_3d
```

được xây trực tiếp từ coordinate DataFrame chứ không gọi `extract_relative_features()`.

Theo quyết định “raw giữ nguyên, chỉ REL chuyển WORLD”, không tự động chuyển hai stream này sang WORLD.

---

## 5. Sửa SkelGym-Aug cho `mix_v2`

Bug hiện tại của 63-d phải được loại bỏ hoàn toàn.

Không còn:

```text
mix_v2 63-d
→ LandmarkAugmenter
→ coi 63 như 21 XYZ
```

Pipeline phải là:

```text
pose_world_landmarks
        ↓
candidate augmentation
        ↓
world_xyz 39-d
        +
world angle24
        ↓
mix_v2 63-d
```

---

## 6. Implement đủ candidate operators cho ablation

Chưa gọi bất kỳ combination nào là `Proposed`.

Giữ candidate universe tương ứng Table 4–5 hiện tại:

```text
Mirror
Yaw
Scale
Time Interpolation
Jitter
```

Từng operator phải hoạt động đúng trên WORLD landmark sequence.

**Mirror**
- reflect WORLD skeleton;
- swap left/right;
- sau đó recompute `mix_v2`.

**Yaw**
- rotate XYZ world coordinates;
- sau đó recompute feature.

**Scale**
- proportional world-coordinate scaling;
- sau đó recompute angles/features.

**Time Interpolation**
- resample landmark sequence theo temporal axis;
- sau đó extract feature.

**Jitter**
- perturb world XYZ;
- sau đó recompute features.

Parameters của candidate suite phải được freeze trước ablation, ví dụ giữ protocol hiện tại:

```text
Mirror p = 0.5
Yaw ±15°
Scale ±10%
Time interpolation 0.8×–1.2×
Jitter σ = 0.008
```

Nếu WORLD dùng meter, unit của jitter phải được ghi rõ trong config/provenance.

---

## 7. Refactor augmentation naming

Trước khi Phase 4 quyết định, config chỉ nên có tên trung tính:

```text
candidate_full_5op
candidate_minus_mirror
candidate_minus_yaw
candidate_minus_scale
candidate_minus_time
candidate_minus_jitter

single_mirror
single_yaw
single_scale
single_time
single_jitter
```

Không hard-code:

```text
skel_gym_aug = Mirror + Yaw + Scale + Jitter
```

như runner hiện tại đang làm.

Tên `skel_gym_aug` chỉ được bind vào một composition sau Phase 4.

---

## 8. Khóa Transformer configuration

Current `RESULTS_FINAL.md` định nghĩa:

```text
d_model = 112
d_ff = 168
nhead = 4
layers = 3
```

Canonical config phải truyền explicit toàn bộ các giá trị này.

Không để `nhead` hoặc architecture phụ thuộc default trong CLI.

---

## 9. Tests

Tối thiểu phải có:

```text
world_3d → 39-d
mix_v2 → 63-d
mix_v2 first branch → WORLD
angle24 → WORLD
joint_motion → ΔWORLD
raw_3d → unchanged
bone_3d → unchanged

Mirror → valid WORLD skeleton
Mirror twice → approximately original
Yaw(+θ) + Yaw(-θ) → approximately original
Time interpolation → correct T
Augmented mix_v2 → recomputed, not channel-mutated

Train normalization → train only
Val/Test → deterministic
```

### Output

Một codebase có thể chạy tất cả candidate augmentation nhưng **chưa có Proposed SkelGym-Aug**.

### Gate

Toàn bộ unit/integration tests pass.

---

# Phase 2 — Refresh các clean sequence experiments của Table 2

Table 2 là unaugmented benchmark nên có thể chạy trước ablation.

Không chạy lại toàn bộ 27 rows.

Chỉ refresh những rows đã thay semantics.

## LSTM

Rerun:

```text
LSTM + World 3D
LSTM + new WORLD mix_v2
```

3 seeds mỗi experiment.

## BiLSTM

Rerun:

```text
BiLSTM + World 3D
BiLSTM + new WORLD mix_v2
```

3 seeds mỗi experiment.

## Transformer

`Transformer + World 3D`:

- dùng lại experiment WORLD 3-seed đã hoàn thành;
- không train lại.

`Transformer + new mix_v2 clean`:

- chưa cần chạy riêng;
- Clean configuration của Table 4–5 ở Phase 3 sẽ chính là checkpoint/result này.

Các Raw/2D/angle rows không bị WORLD migration ảnh hưởng nên giữ nếu provenance xác nhận config không đổi.

### Tổng số training mới ở phase này

```text
4 configurations × 3 seeds = 12 runs
```

### Output

Các clean sequence checkpoints/results mới cho LSTM/BiLSTM và canonical record cho WORLD Transformer đã có.

---

# Phase 3 — Rerun Table 4 & Table 5 ablation trên WORLD `mix_v2`

Đây là phase quan trọng nhất trước toàn bộ experiment sử dụng augmentation.

Không chạy Table 3 augmented experiments trước phase này.

## Table 4 — Leave-One-Out

Unique configurations:

```text
Clean

Candidate Full:
Mirror + Yaw + Scale + Time + Jitter

Minus Mirror
Minus Yaw
Minus Scale
Minus Time
Minus Jitter
```

Tổng:

```text
7 configurations
```

---

## Table 5 — Single Component

```text
Clean
Mirror only
Yaw only
Scale only
Time only
Jitter only
```

Clean đã có từ Table 4 nên reuse.

Thêm:

```text
5 configurations
```

---

## Tổng workload

Unique configs:

```text
Clean
Full
5 × LOO
5 × Single
= 12 configurations
```

Với 3 seeds:

```text
12 × 3 = 36 training runs
```

Không duplicate Clean/Full giữa hai tables.

---

## Training protocol

Dùng:

```text
Transformer
new WORLD mix_v2 63-d
same canonical architecture
seeds 42 / 123 / 3407
same optimizer/scheduler/early stopping
```

Mỗi run lưu:

```text
config
seed
commit
best epoch
train metrics
validation metrics
checkpoint SHA
normalization metadata
```

### Quan trọng về test set

Nên tách selection khỏi test.

Ở phase này:

```text
train
→ validation evaluation
→ aggregate 3 seeds
```

Đủ dữ liệu để ra quyết định augmentation.

Test predictions có thể được tạo sau khi composition Proposed đã được freeze.

Điều này tránh việc nhìn test score rồi quyết định operator nào trở thành Proposed.

### Output

Hai dataset ablation ở validation level:

```text
table4_loo_world_mix_v2.json
table5_single_world_mix_v2.json
```

cùng toàn bộ checkpoints.

Clean Transformer run ở đây đồng thời trở thành:

```text
Transformer Mix v2 Clean
```

cho Table 2/Table 7.

---

# Phase 4 — Discuss và chốt Proposed SkelGym-Aug

Phase này **không train downstream model**.

Đây là phase người dùng + agent phân tích kết quả Phase 3.

## Input

Table 4:

```text
Full
-Mirror
-Yaw
-Scale
-Time
-Jitter
Clean
```

Table 5:

```text
Clean
Mirror
Yaw
Scale
Time
Jitter
```

---

## Phân tích từng operator

Đối với mỗi operator, xem đồng thời:

```text
Single-component gain
LOO necessity
Val Macro-F1
Val Accuracy
Val Loss
variance giữa 3 seeds
semantic validity trên WORLD skeleton
```

Ví dụ logic:

```text
Single tốt + remove khỏi Full làm giảm mạnh
→ strong evidence giữ

Single yếu nhưng remove khỏi Full làm giảm
→ interaction effect, cần cân nhắc giữ

Single xấu + remove khỏi Full cải thiện
→ strong evidence bỏ

Single tốt nhưng remove khỏi Full cũng cải thiện
→ có interaction/conflict, cần thảo luận
```

### Selection metric

Primary:

```text
Validation Macro-F1
```

Secondary:

```text
Validation Accuracy
stability across seeds
```

Test score không dùng để lựa chọn composition.

---

## Output bắt buộc

Một quyết định rõ ràng:

```text
SkelGym-Aug v2 =
{operator A, operator B, ...}
```

và config immutable, ví dụ:

```text
configs/augmentation/skelgym_aug_v2.yaml
```

Config phải ghi:

```text
operators
probabilities
ranges
units
application order
feature schema
```

Từ thời điểm này:

```text
skel_gym_aug
```

mới được alias tới composition vừa chốt.

### Gate

Không sang bất kỳ experiment augmented nào cho tới khi config này được freeze.

---

# Phase 5 — Finalize Table 4–5 và canonical Transformer Aug

Sau khi Proposed đã được chốt.

## Trường hợp A — Proposed đã xuất hiện trong ablation

Ví dụ composition được chọn trùng với một LOO config.

Reuse ba checkpoint tương ứng.

Không train lại.

## Trường hợp B — Proposed là combination chưa được chạy

Ví dụ quyết định cuối là:

```text
Mirror + Scale
```

mà Table 4/5 chưa có exact combination này.

Chạy:

```text
Transformer WORLD mix_v2
+ Proposed SkelGym-Aug
× 3 seeds
```

để có canonical Proposed checkpoints.

---

## Sau khi composition freeze

Evaluate **toàn bộ Table 4–5 configs** trên held-out test set.

Generate final Table 4:

```text
Val Loss
Val Acc
Val F1
Test Acc
Test F1
Δ vs Full
```

Generate final Table 5 tương ứng.

Như vậy test metrics được dùng để báo cáo, không dùng để chọn augmentation.

### Output

```text
augmentation_ablation_results_v2.json
Transformer Mix Clean checkpoints
Transformer Mix Proposed checkpoints
```

Từ đây Proposed SkelGym-Aug chính thức tồn tại.

---

# Phase 6 — Rerun Table 3 Graph Kinematic Streams

Chỉ bắt đầu sau Phase 5.

Theo `RESULTS_FINAL.md` hiện tại:

### T3.1 — ST-GCN Raw 3D Clean

Raw không thay đổi.

Có thể reuse sau provenance verification.

---

### T3.2 — ST-GCN Relative 3D Clean

REL đã bị WORLD thay thế.

Rerun thành:

```text
ST-GCN + World 3D
Clean
× 3 seeds
```

Tên table cũng phải đổi từ Relative 3D sang World 3D.

---

### T3.3 — AAGCN Bone Clean

Bone semantics không đổi và không có augmentation.

Có thể reuse nếu code/config/checkpoint hash xác nhận cùng architecture.

---

### T3.4 — AAGCN Bone + Proposed

Bắt buộc rerun:

```text
AAGCN Bone
+ SkelGym-Aug v2
× 3
```

vì augmentation composition vừa được xác định lại.

---

### T3.5 — AAGCN Joint Stream

REL Joint cũ bị thay bằng WORLD.

Rerun:

```text
AAGCN World Joint
+ SkelGym-Aug v2
× 3
```

---

### T3.6 — Joint Motion

Code hiện tại dùng ΔREL nên bị WORLD migration ảnh hưởng.

Rerun:

```text
AAGCN World Joint Motion
+ SkelGym-Aug v2
× 3
```

---

### T3.7 — Bone Motion

Feature có thể giữ nguyên, nhưng augmentation đã thay đổi.

Rerun:

```text
AAGCN Bone Motion
+ SkelGym-Aug v2
× 3
```

---

### T3.8 — Two-Stream

Không train model mới.

Dùng prediction của:

```text
World Joint
+
Bone
```

Fusion:

```text
Uniform Soft
```

regenerate cả val/test window + video metrics.

---

### T3.9 — Four-Stream

Dùng:

```text
World Joint
Bone
World Joint Motion
Bone Motion
```

Canonical fusion:

```text
Uniform Soft
```

SLSQP không còn.

---

## Training workload chính

Nếu reuse T3.1 và T3.3:

```text
T3.2  3 runs
T3.4  3
T3.5  3
T3.6  3
T3.7  3
--------------
15 training runs
```

T3.8–T3.9 chỉ fusion/evaluation.

---

# Phase 7 — Rebuild Table 6: SkelGym-Lite / SkelGym-Full

Sau Phase 6, tất cả constituent checkpoints đã ổn định.

## Base models

Canonical Full:

```text
Transformer WORLD Mix v2 + Proposed Aug
AAGCN Bone + Proposed Aug
AAGCN World Joint + Proposed Aug
AAGCN World Joint Motion + Proposed Aug
AAGCN Bone Motion + Proposed Aug
```

Lite:

```text
Transformer
+
Bone AAGCN
```

---

## Generate prediction artifacts

Cho từng seed, lưu một lần:

```text
validation probabilities
test probabilities
video IDs
labels
```

Sau đó mọi fusion dùng cùng prediction artifact.

---

## Fusion methods

Chỉ evaluate:

```text
Hard Voting
Uniform Soft Voting
Accuracy-Weighted Soft Voting
```

Cho:

```text
Lite
Full
```

Table 6 mới không còn:

```text
Stacking
SLSQP
```

---

## Metrics

Cho mỗi phương pháp:

```text
Val Window Accuracy
Val Window Macro-F1
Val Video Accuracy
Val Video Macro-F1

Test Window Accuracy
Test Window Macro-F1
Test Video Accuracy
Test Video Macro-F1
```

tất cả mean ± std trên 3 seeds.

---

## Freeze final Full method

Vì Table 8–10 cần một `SkelGym-Full` cụ thể, phase này phải ghi lại primary fusion rule.

Chọn bằng validation criterion đã quy định trước, không chọn vì test score cao nhất.

Output ví dụ:

```text
configs/system/skelgym_full_v2.yaml
configs/system/skelgym_lite_v2.yaml
```

---

# Phase 8 — Regenerate Table 7 đến Table 10

Không còn training model ở đây.

## Table 7 — Window vs Video + Parameters

Regenerate từ canonical checkpoints/results.

Các rows bị thay đổi gồm tối thiểu:

```text
LSTM Mix 63-d
BiLSTM Mix 63-d
Transformer Mix Clean
ST-GCN World
Transformer Mix Proposed Aug
AAGCN Proposed rows
Two-Stream
Four-Stream
Lite
Full
```

Xóa Stacking row.

Parameter counts phải lấy trực tiếp từ model instances/config mới.

---

## Table 8 — Statistical tests

Rerun bằng final prediction artifacts.

Các comparison có thể giữ cùng logic hiện tại:

```text
Transformer Clean
vs
Transformer Proposed Aug

ST-GCN World
vs
Four-Stream AAGCN

Single Transformer
vs
SkelGym-Full

Single Bone
vs
SkelGym-Full

Four-Stream
vs
SkelGym-Full
```

Regenerate:

```text
McNemar
Wilcoxon
paired t-test
effect size
```

từ predictions mới.

---

## Table 9 — Bootstrap

Regenerate cluster bootstrap trên video IDs mới/final predictions.

Không reuse CI cũ.

Các rows nên sử dụng đúng model versions mới:

```text
LSTM Mix
BiLSTM Mix
ST-GCN World
Transformer Proposed
AAGCN Bone Proposed
Lite
Full
```

---

## Table 10 — Per-class breakdown

Regenerate từ primary `SkelGym-Full v2`.

Bao gồm:

```text
window precision/recall/F1/support
video precision/recall/F1/support
macro averages
overall accuracy
confusion matrix artifact
```

---

# Phase 9 — Runtime và external benchmark

## Table 11 — Computational complexity

Dù architecture dimensions có thể không đổi, regenerate từ final model definitions để SOT nhất quán.

Benchmark:

```text
Transformer Mix v2
World Joint AAGCN
Bone AAGCN
World Joint Motion
Bone Motion
Lite
Full
```

Lưu:

```text
parameter count
FLOPs/MACs
latency
FPS
device
batch/window settings
warmup/repeat metadata
```

---

## Table 13 — BlockGCN

Current BlockGCN uses:

```text
33 raw MediaPipe XYZ
```

nên WORLD migration không làm thay đổi baseline này.

Không cần retrain nếu artifact/checkpoints/provenance vẫn hợp lệ.

Chỉ import nó vào SOT mới.

Nếu paper vẫn chứa external benchmark khác mà sử dụng:

```text
mix_v2
WORLD stream
Lite
Full
```

thì benchmark đó phải rerun với models v2 trong phase này.

---

# Phase 10 — Xây canonical Source of Truth

Đây là lúc `RESULTS_FINAL.md` cũ chính thức bị thay thế làm nguồn dữ liệu.

## Nguồn đầu vào

Chỉ lấy từ:

```text
experiment configs
checkpoint metadata
raw experiment result JSON
prediction artifacts
statistical artifacts
benchmark artifacts
```

Không đọc metric ngược từ paper.

---

## Tạo

```text
artifacts/results/canonical_results_v2.json
```

Schema nên chứa:

```text
repository commit

dataset
feature schema
augmentation schema

models
training configs
seeds

Table 1 data
Table 2 results
Table 2b WORLD validation
Table 3 graph
Table 4 LOO
Table 5 single
Table 6 systems
Table 7 consensus
Table 8 statistics
Table 9 bootstrap
Table 10 per-class
Table 11 latency
Table 13 baseline

checkpoint hashes
prediction hashes
artifact hashes
```

---

## Provenance

Mỗi result phải trace được:

```text
metric
→ seed/run
→ prediction
→ checkpoint
→ config
→ git commit
```

Generate:

```text
provenance_manifest_v2.json
```

---

## Regenerate `RESULTS_FINAL.md`

`RESULTS_FINAL.md` mới phải được generate từ SOT.

Các stale items hiện tại phải biến mất:

```text
Root-Relative stream được thay thế
old mix_v2 numbers
old augmentation tables
old graph Aug numbers
SLSQP
Stacking
old Full/Lite numbers
old stats/bootstrap/per-class
```

### Publication table order

Có thể vẫn giữ numbering paper hiện tại.

Nhưng execution dependency đã là:

```text
Table 1
  ↓
Table 2 / 2b
  ↓
Table 4 / 5
  ↓
Augmentation Decision
  ↓
Table 3
  ↓
Table 6
  ↓
Table 7
  ↓
Table 8 / 9 / 10
  ↓
Table 11 / 13
```

Đây là điểm cần phân biệt giữa **thứ tự xuất hiện trong paper** và **thứ tự chạy experiment**.

---

# Phase 11 — Rewrite paper từ SOT

Chỉ bắt đầu khi Phase 10 đã freeze.

## Methods

Cập nhật representation:

```text
pose_landmarks
→ raw baselines

pose_world_landmarks
→ WORLD representation
→ mix_v2
→ world joint motion
```

Mô tả `mix_v2` đúng implementation mới:

```text
39-d metric world coordinates
+
24-d world-space kinematic angles
```

Mô tả SkelGym-Aug đúng composition được quyết định ở Phase 4.

Không viết trước số lượng operator.

---

## Graph methodology

Đổi:

```text
Relative Joint stream
```

thành WORLD Joint stream.

Joint Motion cũng phải mô tả là temporal derivative của WORLD coordinates.

---

## Ensemble methodology

Chỉ còn:

```text
Hard
Uniform Soft
Accuracy-Weighted Soft
```

Không còn bất kỳ wording nào về:

```text
SLSQP
Stacking
Ridge
meta-classifier
```

---

## Results

Mọi con số trong:

```text
Abstract
Results
Discussion
Conclusion
Tables
captions
```

đều lấy từ:

```text
canonical_results_v2.json
```

hoặc generated LaTeX tables.

Không nhập metric thủ công.

---

## Dataset wording

Dùng số thật:

```text
580 Train
208 Validation
236 Test
```

và nếu nói ratio thì:

```text
approximately 6:2:2
```

---

# Phase 12 — Final reproducibility & consistency audit

Phase cuối không tạo model mới.

Kiểm tra toàn project theo chuỗi:

```text
CODE
 ↓
CONFIG
 ↓
CHECKPOINT
 ↓
PREDICTIONS
 ↓
RESULT JSON
 ↓
CANONICAL SOT
 ↓
PAPER
```

Audit:

- raw vẫn dùng normalized landmarks;
- WORLD thực sự thay REL ở đúng streams;
- `mix_v2` thực sự 39 WORLD + 24 WORLD angles;
- augmentation không còn trực tiếp corrupt tensor 63-d;
- Proposed config đúng kết quả Phase 4;
- mọi augmented experiment đều được chạy **sau khi** Proposed được freeze;
- Transformer config thống nhất;
- 3 seeds đầy đủ;
- Table 3 dùng augmentation v2;
- Table 6 chỉ còn 3 voting methods;
- không còn SLSQP;
- không còn Stacking;
- không còn checkpoint legacy được vô tình load;
- paper metric = SOT metric;
- README metric = SOT metric;
- provenance hashes đầy đủ.

Sau khi pass thì mới tạo final tag/release.

---

## Tóm tắt dependency toàn dự án

```text
PHASE 0
Freeze migration specification
        ↓
PHASE 1
Code migration:
WORLD + mix_v2 + Aug infrastructure
        ↓
PHASE 2
Clean Table 2 refresh
        ↓
PHASE 3
Table 4 + Table 5 ablation
(validation selection evidence)
        ↓
PHASE 4
DISCUSS + CHOOSE
Proposed SkelGym-Aug
        ↓
PHASE 5
Freeze Proposed
Finalize Table 4/5 + Transformer Aug
        ↓
PHASE 6
Table 3 graph experiments
        ↓
PHASE 7
Table 6 Lite / Full / Voting
        ↓
PHASE 8
Table 7–10
Consensus + Stats + Bootstrap + Per-Class
        ↓
PHASE 9
Table 11 + External/Baseline
        ↓
PHASE 10
Canonical SOT
        ↓
PHASE 11
Paper
        ↓
PHASE 12
Final Audit / Release
```
