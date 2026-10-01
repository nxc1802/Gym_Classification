Mình đã audit lại HEAD mới nhất của `nxc1802/Gym_Classification` tại commit **`463a5c569906298687a4ca9e81d2054725a2e20f`** (30/09/2026). Đây là một overhaul lớn: pipeline dữ liệu, augmentation, provenance, thống kê, external benchmark, CI, dependency manifests và manuscript đều đã được thay đổi đáng kể.

## Kết luận

**Dự án đã sẵn sàng để chạy pre-flight / smoke test, nhưng chưa nên chạy full canonical rerun để lấy số liệu cuối cho paper.**

Nếu mục tiêu là rerun 3 seeds × toàn bộ baselines × constituents × ensembles × ablations × external benchmarks, mình đánh giá trạng thái hiện tại khoảng **70–75% ready**. Các lỗi implementation nghiêm trọng lần trước phần lớn đã được sửa, nhưng hiện vẫn còn một nhóm blocker liên quan đến **protocol reproducibility và code–paper synchronization**. Nếu chạy ngay, có khả năng bạn tiêu tốn rất nhiều GPU time nhưng cuối cùng phải rerun lần nữa.

### Trạng thái theo từng khối

| Khối | Trạng thái | Đánh giá |
|---|---|---|
| Dataset split / leakage | ✅ Tốt | 1,024 videos, 580/208/236, không overlap filepath |
| Missing-data handling | ✅ Đã sửa | Không còn silent random fallback ngoài smoke mode |
| Augmentation → normalization order | ✅ Đã sửa | Augment trước z-score |
| Mirror signed angles | ✅ Đã sửa | Có parity/sign handling và recompute cho mix |
| Checkpoint provenance | ✅ Tốt hơn nhiều | SHA, git SHA, seed, aug, sidecar provenance |
| Statistical utilities | ✅ Tốt hơn | Cluster bootstrap + multiple-comparison correction |
| Packaging | ✅ Tốt | requirements, pyproject, environment, LICENSE |
| GitHub CI | 🟡 Khá | Green 3.10/3.11 nhưng chưa chạy toàn bộ pytest-style tests |
| Training protocol | 🔴 Chưa khóa | Runner ≠ paper/config |
| Normalization provenance | 🔴 Chưa khóa | Evaluation recompute stats khác training |
| Model architecture vs paper | 🔴 Lệch đáng kể | Transformer + AAGCN không khớp manuscript |
| Ablation provenance | 🔴 Còn contamination case | Seed-42 5-op có thể reuse 4-op checkpoint |
| External E2E | 🔴 Chưa chạy được như plan | CLI `--seeds` không tồn tại; MediaPipe preparation chưa thật sự extract |
| Final reported metrics | 🔴 Chưa hợp lệ với HEAD | 69.74/79.11 là kết quả pre-fix |

---

# Những điểm đã sửa tốt

So với lần audit trước, commit này tiến bộ rõ rệt.

**Augmentation order đã đúng.** `GymDataset.__getitem__()` hiện thực hiện augmentation trong physical feature space rồi mới z-score. Val/test được standardize bằng train statistics mà không augment. Đây là sửa chữa quan trọng nhất.

**Missing landmarks đã fail-fast.** Normal training/evaluation sẽ `raise FileNotFoundError` hoặc `RuntimeError`, thay vì âm thầm tạo `np.random.randn()` như trước. Synthetic data chỉ còn dành cho smoke testing.

**Mirror 78-d signed elevation đã được sửa.** Có `pair_sym_signs`; với `mix=117`, mirror còn recompute elevation từ tọa độ mirrored. Đây đúng hơn nhiều về consistency hình học.

**Checkpoint provenance đã được nâng cấp.** `Trainer` ghi provenance, git SHA, seed, feature, augmentation, checkpoint SHA-256 và tạo sidecar JSON.

**Statistics đã tiến bộ.** `src/utils/statistics.py` có video-cluster bootstrap, exact/asymptotic McNemar, video-level paired tests và Bonferroni/Holm/FDR adjustment.

**Latency benchmark cũng được sửa đúng.** Ensemble hiện được benchmark bằng actual sequential invocation, không còn cộng median/p95 của từng model. Script cũng tách Observation Horizon / Pose Tracking / Classifier Post-Window — tốt hơn lần trước.

**CI của chính commit `463a5c5` đang green trên Python 3.10 và 3.11.**

Nhưng CI green chưa đủ để bật full rerun.

---

# Các blocker cần sửa trước full rerun

| Priority | Vấn đề | Vì sao quan trọng | Cách sửa |
|---|---|---|---|
| **P0** | Train normalization dùng stride 16, evaluation/freeze dùng stride 32 | Model train và model eval có thể nhận z-score theo hai bộ μ/σ khác nhau | Không recompute stats khi eval; load chính xác stats từ checkpoint/reference artifact |
| **P0** | Multi-seed runner không truyền hyperparameter protocol của paper | Rerun hiện tại sẽ là experiment khác | Tạo canonical experiment registry và dùng nó cho mọi runner |
| **P0** | Transformer code ≠ paper | Paper 4L/sinusoidal; code 3L/learnable PE | Chọn implementation làm source of truth rồi đồng bộ paper |
| **P0** | AAGCN code ≠ paper | Paper mô tả 3 partitions + MS-TCN; code là custom 3-stage adaptive GCN | Rewrite Method hoặc implement đúng architecture paper |
| **P0** | Ablation seed42 5-op có thể reuse checkpoint 4-op | Làm invalid LOO | Xóa legacy mapping 5-op→T2.2 + bắt buộc provenance-match |
| **P0** | Augmentation recipe code ≠ paper | Contribution chính chưa được định nghĩa duy nhất | Freeze chính xác probabilities/ranges/sigma |
| **P1** | `cmd_train` tự evaluate test sau mỗi model | Không khớp “test only once” protocol | Thêm train-only mode; final test sau khi configs frozen |
| **P1** | External E2E commands không chạy như tài liệu | `--seeds` chưa tồn tại | Implement multi-seed external CLI hoặc sửa plan |
| **P1** | MediaPipe MM-Fit preparation chưa thực sự extract | Script chỉ tìm `wXX_mediapipe.npy` đã tồn tại | Thêm RGB→MediaPipe extraction |
| **P1** | CI dùng unittest thay vì pytest | Một số geometry tests bị bỏ qua | Đổi workflow thành `pytest tests/ -v` |
| **P1** | `configs/default.yaml` không khớp API | `skelgym_aug` ≠ `skel_gym_aug` | Fix tên và biến config thành source of truth |
| **P1** | Reference freeze chỉ thực chất seed42 | External multi-seed cần weights theo seed | Freeze `seed42/123/3407` riêng |
| **P2** | Dependency chỉ có lower bounds | Khó exact reproduction | Thêm lockfile / exact benchmark environment |
| **P2** | Historical metrics vẫn nằm trong README/paper | Có thể bị nhầm là HEAD results | Mark legacy hoặc blank cho tới rerun |

## 1. Normalization mismatch là blocker kỹ thuật số 1

Training mặc định:

```text
train stride = 16
val/test stride = 32
```

`cmd_train()` dùng train windows stride 16 để tính `train_mean` / `train_std`.

Nhưng những script sau lại gọi:

```python
get_dataloaders(... stride=32, val_test_stride=32)
```

bao gồm ít nhất:

`run_multi_seed_experiments.py`, `evaluate_local_ensemble.py`, `compute_statistical_tests.py`, `run_augmentation_experiments.py`, `freeze_reference_artifacts.py`.

Do train windows overlap khác nhau, distribution được dùng để tính mean/std không nhất thiết giống nhau.

Đặc biệt `freeze_reference_artifacts.py` còn ghi:

```json
"stride": 32
```

trong manifest, trong khi training protocol là stride 16.

**Không nên chỉ sửa tất cả thành stride=16 rồi dừng ở đó.**

Kiến trúc tốt hơn là:

```text
Training
 ├─ compute train stats ONCE
 ├─ save stats/hash into checkpoint provenance
 └─ train model

Evaluation
 ├─ load checkpoint
 ├─ load its exact normalization artifact
 └─ NEVER recompute μ/σ
```

Nếu làm vậy, evaluator không còn phụ thuộc vào cách build lại train loader.

---

# 2. Hyperparameter protocol hiện chưa có một source of truth

Paper hiện mô tả:

```text
Transformer LR       1e-4
RNN / GNN LR         1e-3
Batch sequence       16
Batch graph          32
Label smoothing      0.05
Early stopping       Val Macro-F1
Patience             10
```

Nhưng `run_multi_seed_experiments.py` chỉ truyền những thứ như:

```text
--model
--feature
--augment
--seed
--epochs
--patience
--device
```

Các tham số còn lại rơi về CLI defaults:

```text
batch_size          16 cho tất cả
lr                  1e-4 cho tất cả
label_smoothing     0.0
early stopping      val_acc
```

Thậm chí `Trainer` hiện chỉ hỗ trợ:

```text
val_acc
val_loss
```

chứ **chưa hỗ trợ `val_macro_f1`**, trong khi paper nói monitor Macro-F1.

Vì vậy chạy Phase 1 ngay bây giờ sẽ không thực thi protocol đang viết trong paper.

Mình khuyến nghị tạo một registry như:

```yaml
T2.2:
  model: Transformer
  feature: mix
  augment: skel_gym_aug
  lr: 0.0001
  batch_size: 16
  layers: 3
  d_model: 128
  d_ff: 192
  dropout: 0.2
  label_smoothing: 0.05
  early_stopping: val_macro_f1
  train_stride: 16
  eval_stride: 32
```

và mọi runner phải đọc registry này thay vì hardcode riêng.

---

# 3. Transformer trong paper không phải Transformer trong code

Đây là mismatch rất rõ.

Implementation hiện tại thực chất là:

```text
d_model = 128
heads = 8
layers = 3
FFN = 192
dropout = 0.2
learnable positional embedding
```

Cấu hình này cho khoảng **399.7K parameters**, tức rất phù hợp với con số “399K” mà paper báo cáo.

Nhưng paper lại ghi:

```text
4 layers
8 heads
d_model = 128
d_ff = 512
dropout = 0.1
sinusoidal positional encoding
399K params
```

Cấu hình `4L + d_ff=512` không thể đồng thời chỉ khoảng 399K; riêng encoder đã lớn hơn đáng kể.

Paper còn có phần text khác ghi `d_ff=192`, nên manuscript hiện tự mâu thuẫn.

**Mình nghiêng về việc giữ implementation hiện tại** — 3L, FF192, learnable PE — vì nó phù hợp parameter-budget 399K. Sau đó sửa toàn bộ manuscript, diagrams, config và README theo implementation.

---

# 4. AAGCN mismatch còn lớn hơn Transformer

Paper mô tả gần với full multi-partition 2S-AAGCN:

```text
K_v = 3 topology partitions
A_k + B_k + C_k
root / centripetal / centrifugal
MS-TCN
dilation {1,2}
9 spatial-temporal blocks
64 channels ...
```

Implementation hiện tại lại là custom lightweight architecture:

```text
single normalized anatomical adjacency
+ edge mask
+ global learnable B
+ one dynamic attention matrix C(X)

3 stages:
[48, 96, 150]

single temporal Conv2d
kernel = 9
no multi-scale dilation architecture
```

Nói cách khác đây là **lightweight adaptive GCN inspired by AAGCN**, không phải implementation chính xác kiến trúc paper đang mô tả.

Mình không cho rằng cần rewrite model thành full AAGCN. Với mục tiêu ~350K parameters, implementation hiện tại khá hợp lý.

Nhưng paper phải mô tả model thực sự được chạy.

---

# 5. SkelGym-Aug vẫn chưa có một định nghĩa thống nhất

Code final 4-op hiện tại:

```text
Mirror: p = 0.5
Yaw: luôn áp dụng, α ∈ [-15°, 15°]
Scale: luôn áp dụng, s ∈ [0.9, 1.1]
Jitter: luôn áp dụng, σ = 0.008
```

Paper lại có:

```text
Mirror: p = 0.5
Yaw: p = 0.5
Scale: [0.95, 1.05]
Jitter: σ = 0.005
```

Paper thậm chí nói jitter được thêm vào “normalized coordinates”, trong khi code mới đã cố ý sửa thành augmentation **trước** standardization.

Đây là contribution C3 của project nên không thể để mơ hồ.

Trước rerun phải freeze một specification duy nhất.

---

# 6. Mix jitter vẫn làm mất consistency coordinate ↔ angle

Với `mix=117`:

```text
39 dims coordinates
+
78 dims derived elevation angles
```

Mirror hiện đúng vì recompute 78 angles từ mirrored coordinates.

Nhưng `jitter()` hiện làm:

```text
coordinate noise ~ sigma
angle noise ~ sigma * 0.5
```

hai phần độc lập.

Sau jitter, elevation angle có thể không còn là elevation của 39-d coordinates trong chính sample đó.

Nếu mục tiêu là mô phỏng MediaPipe sensor noise, mình khuyến nghị:

```text
jitter 39-d coordinates
→ recompute 78-d elevation
→ concatenate
```

Đây sẽ làm representation mathematically coherent và dễ defend hơn trong review.

---

# 7. Ablation checkpoint contamination vẫn chưa được loại bỏ hoàn toàn

Phần generic cross-seed contamination đã được sửa rất tốt.

Nhưng vẫn còn mapping legacy:

```python
if seed == 42:
    if aug in ("skel_gym_aug", "skel_gym_aug_legacy_5op"):
        use best_Transformer_T2.2_mix.pt
```

`T2.2` là checkpoint của final 4-op.

Điều nguy hiểm là E2E hiện chạy:

```text
Phase 1 → train T2.2 final 4-op
Phase 4 → run LOO
```

Khi đến Candidate Full 5-op, runner có thể lấy luôn checkpoint T2.2 4-op làm checkpoint 5-op.

Và E2E plan hiện không có `--force_retrain` cho Phase 4.

**Cần xóa mapping này cho `skel_gym_aug_legacy_5op`.**

Tốt hơn nữa: `find_existing_checkpoint()` chỉ được reuse checkpoint nếu provenance sidecar khớp tuyệt đối:

```text
seed
model
feature
augmentation
model config
data config
git SHA / code version
```

Path/name chỉ nên là convenience, không phải evidence.

---

# 8. Ablation table đang hardcode kết luận “Optimal”

Generator có logic:

```python
if var_name == "Minus_TimeWarp":
    ... "(Optimal)"
```

và single-component table cũng hardcode final recipe.

Nghĩa là dù rerun mới cho kết quả khác, bảng vẫn bold `Minus_TimeWarp` là “Optimal”.

Việc này nên loại bỏ trước rerun.

Nên chọn recipe theo một rule được freeze **trước khi nhìn test**:

```text
Primary selection metric:
mean validation Macro-F1 across 3 seeds

Tie-break:
lower validation variance
then simpler recipe
```

Sau khi recipe được khóa mới evaluate test.

---

# 9. “Test only once” hiện chưa được enforce

`E2E_PIPELINE_RUNNING_PLAN.md` nói:

> Test chỉ dùng một lần duy nhất để đánh giá.

Nhưng `cmd_train()` tự evaluate test ngay sau mỗi training.

Và ablation runner cũng evaluate từng candidate trên test.

Điều này chưa phải leakage vào gradient/early stopping, vì training không dùng test để update weights. Nhưng nó **không đảm bảo blind-held-out protocol** như plan tuyên bố, vì người làm experiment sẽ nhìn test liên tục.

Nên có:

```bash
run.py train ... --no_test_eval
```

Phase training/ablation chỉ tạo:

```text
train metrics
validation metrics
checkpoints
```

Sau khi model/augmentation/fusion được frozen mới mở Phase Final Test.

Điều này sẽ làm methodology mạnh hơn đáng kể.

---

# 10. External benchmark mới tốt hơn nhiều, nhưng E2E hiện chưa runnable

Đây là một cải tiến rất đáng kể so với “Deyzel overlap” cũ: project hiện có adapters cho MM-Fit/Fit3D, class mapping, hierarchical metrics, bootstrap, few-shot.

Nhưng plan và implementation chưa khớp.

E2E ghi:

```bash
python scripts/evaluate_external.py \
  --config configs/external/mmfit.yaml \
  --seeds 42 123 3407
```

nhưng `evaluate_external.py` **không có `--seeds`**.

`evaluate_external_fewshot.py` cũng có `--seed` singular, không có `--seeds`.

Hơn nữa evaluator hiện load thẳng:

```text
checkpoints/best_...
```

tức seed-42/base paths, không iterate seed123/3407.

`freeze_reference_artifacts.py` cũng freeze ensemble weights từ base seed42 בלבד.

Do đó tuy E2E document nói “multi-seed external”, implementation hiện là single-seed.

---

# 11. Protocol A MediaPipe cho MM-Fit chưa được implement end-to-end

E2E nói:

> `prepare_mmfit_external.py --pose-source mediapipe` → trích xuất MediaPipe.

Nhưng script đó thực tế chỉ instantiate dataset adapter/audit.

Trong `MMFitExternalDataset`, MediaPipe mode chỉ tìm:

```text
wXX/wXX_mediapipe.npy
```

Nếu file không tồn tại thì bỏ qua workout.

Mình không thấy bước trong script này tạo các file đó từ RGB video.

Vì vậy cần một extraction stage thật sự:

```text
MM-Fit RGB
→ MediaPipe Pose Heavy
→ canonical 33 landmarks
→ wXX_mediapipe.npy
→ audit
→ external evaluation
```

Hoặc gọi preparation là “audit prepared MediaPipe poses” thay vì “extract”.

Ngoài ra `mmfit.yaml` mặc định:

```yaml
pose_protocol:
  source: native
```

nên E2E hiện prepare “mediapipe”, nhưng evaluate config lại quay về “native” nếu không truyền `--pose-source mediapipe`.

---

# 12. CI cần đổi sang pytest

GitHub Actions đang green, nhưng workflow thực thi:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Trong repo vẫn có các test top-level pytest-style:

```text
test_angle2_and_3d_aug.py  → 3 tests
test_motion_tta.py         → 4 tests
```

`unittest discover` import các module này nhưng không chạy 7 function tests đó.

Đáng chú ý, geometry/mirror augmentation checks nằm trong nhóm này.

E2E plan chính nó lại yêu cầu:

```bash
pytest tests/ -v
```

CI nên dùng đúng command đó.

Có thể thêm tiếp:

```bash
ruff check src scripts tests
python -m compileall src scripts
```

và một smoke CLI job 1 epoch.

---

# 13. `configs/default.yaml` hiện không phải default đáng tin cậy

Nó chứa:

```yaml
augmentation: "skelgym_aug"
```

nhưng code nhận:

```text
skel_gym_aug
```

`skelgym_aug` hiện sẽ đi vào:

```python
raise ValueError("Unknown augmentation method")
```

Ngoài ra config nói:

```text
batch_size 32
patience 20
num_layers 4
dropout 0.1
```

trong khi runtime/paper/runner đang dùng các combination khác.

Các file `configs/experiments/*.yaml` cũng chủ yếu là legacy experiments 12rel_4/full_4/branch_concat, chưa phản ánh benchmark final SkelGym.

Mình sẽ không gọi chúng là “verified configs” ở trạng thái này.

---

# 14. Normalization provenance hiện chưa thực sự được ghi từ CLI

`Trainer` đã hỗ trợ:

```python
normalization_stats
```

trong provenance — tốt.

Nhưng `cmd_train()` đang kiểm tra attribute dạng:

```python
train_loader.dataset.mean
train_loader.dataset.std
```

trong khi `GymDataset` thực tế lưu:

```text
train_mean
train_std
norm_mean
norm_std
```

Vì vậy provenance thực tế có nguy cơ nhận `normalization_stats=None`.

Nên sửa thành đọc `train_mean/train_std`, hoặc tốt hơn:

```text
normalization artifact path
SHA-256
feature
train_stride
metadata SHA
landmark dataset SHA/version
```

Không cần nhét toàn bộ float arrays vào JSON.

---

# 15. Reference artifacts cần theo seed

Normalization stats có thể share giữa seeds vì dataset giống nhau.

Nhưng SLSQP ensemble weights phụ thuộc predictions của từng checkpoint, nên:

```text
seed42 weights
seed123 weights
seed3407 weights
```

không nên dùng chung.

Cấu trúc tốt hơn:

```text
artifacts/reference/
  data/
    normalization_mix.npz
    ...
    metadata_manifest.json

  seed42/
    ensemble_weights.json
    checkpoint_manifest.json

  seed123/
    ensemble_weights.json
    checkpoint_manifest.json

  seed3407/
    ensemble_weights.json
    checkpoint_manifest.json
```

External benchmark multi-seed sau đó có thể dùng đúng seed-specific models và weights.

---

# 16. Paper hiện chưa được phép giữ các số final cũ như kết quả của HEAD

README/paper vẫn ghi:

```text
69.74% ± 1.04% window
79.11% ± 0.25% video
```

Nhưng pipeline đã thay đổi ở các điểm ảnh hưởng trực tiếp training:

```text
augmentation order
mirror angle semantics
normalization implementation
provenance
possibly experiment orchestration
```

Do đó các số đó hiện nên được hiểu là **historical/pre-fix results**, không phải kết quả xác nhận của commit `463a5c5`.

Sau clean rerun, chúng có thể cao hơn, thấp hơn hoặc gần giống. Cả ba trường hợp đều bình thường.

Mình khuyến nghị tạm đánh dấu:

> Previous benchmark result; pending clean rerun under corrected pipeline.

cho đến Phase Final Test.

---

# Thứ tự mình đề xuất trước khi bật full GPU rerun

1. **Freeze canonical protocol** trong một config/registry duy nhất: architecture, LR, batch, augmentation probabilities/ranges, loss, label smoothing, early stopping, stride.
2. **Fix normalization lifecycle**: training computes once → checkpoint/reference stores hash → all evaluation loads exact artifact.
3. **Sync model definitions với paper**: đặc biệt Transformer và AAGCN; mình đề xuất sửa paper theo actual lightweight implementation thay vì tăng model lên architecture mô tả cũ.
4. **Fix SkelGym-Aug**: thống nhất yaw probability, scale range, jitter sigma; recompute angle after jitter cho `mix=117`.
5. **Fix ablation provenance**: bỏ legacy 5-op→T2.2 reuse, provenance-match tuyệt đối, bỏ hardcoded `(Optimal)`.
6. **Enforce blind test protocol**: Phase 1–4 không evaluate test; chọn config bằng validation; final test ở phase riêng.
7. **Fix runners/E2E**: metadata path, `--device auto`, external `--seeds`, per-seed freeze, MediaPipe external extraction.
8. **Change CI to pytest + smoke integration**, rồi chạy toàn bộ preflight. Chỉ khi tất cả pass mới bật 3-seed full rerun.

## Quality gate mình sẽ dùng

Sau các sửa trên, mình chỉ gọi project **READY TO RERUN** khi các điều kiện sau đồng thời đúng:

| Gate | Điều kiện |
|---|---|
| Tests | `pytest tests/ -v` 100% pass |
| CI | Python 3.10 + 3.11 green với pytest |
| Data | 1024 videos, 580/208/236, zero overlap |
| Normalization | train/eval SHA của normalization artifact giống tuyệt đối |
| Config | mỗi experiment có frozen config + hash |
| Models | parameter count + topology khớp manuscript |
| Augmentation | code/spec/tests khớp từng operator |
| Provenance | mỗi checkpoint có seed/config/data/code hashes |
| Ablation | không reuse sai variant; selection validation-only |
| Test protocol | training không tự mở test |
| Multi-seed | 42/123/3407 dùng cùng protocol |
| External | CLI multi-seed thật sự runnable |
| E2E | one-liner hoặc orchestrator chạy qua một smoke E2E hoàn chỉnh |

### Đánh giá cuối

Commit mới đã **giải quyết phần lớn lỗi code correctness nguy hiểm của lần audit trước**. Pipeline dữ liệu bây giờ đáng tin hơn đáng kể, augmentation order và missing-data policy đã đi đúng hướng, provenance/statistics/reproducibility infrastructure cũng tốt hơn nhiều.

Nhưng **chưa nên full rerun ngay**. Vấn đề hiện tại không còn chủ yếu là “bug model” mà là **không có một protocol duy nhất mà code, runner, config và paper cùng tuân theo**. Đây chính xác là loại vấn đề dễ gây ra một rerun tốn hàng chục GPU-hours nhưng sau đó không thể defend vì không biết đâu là configuration chính thức.

Sau khi xử lý các **P0** ở trên, mình sẽ coi internal 3-seed benchmark là ready. Sau khi xử lý thêm external multi-seed/E2E, toàn bộ project mới thực sự **publication-grade rerun ready**.