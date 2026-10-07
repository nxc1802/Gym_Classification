Chưa nên **freeze paper**, nhưng mình nghĩ bạn đã đủ gần để bắt đầu chỉnh **Methods / Experimental Protocol / Architecture**. Riêng **Results + Discussion + claims cuối** thì nên chờ thêm một vòng sửa pipeline/reporting.

Commit mới nhất mình kiểm tra là `40553a2e2ed5b7cc59c627f64910cb2f5f3baaee` — `complete full rebuild with Mirror+Yaw, train-set stacking, and zero-leakage display rules`. Hướng tổng thể đã đúng hơn rất nhiều: WORLD `mix_v2`, SkelGym-Aug = Mirror+Yaw, graph streams đã rerun, Stacking chuyển sang train-set, có canonical SOT và downstream statistics.

## Những phần mình đánh giá đã ổn

**SkelGym-Aug hiện có thể freeze về mặt method.** `mix_v2` hiện thực sự dùng 39-d WORLD + 24 kinematic angles. Augmenter có xử lý riêng 63-d; Mirror/Yaw biến đổi 39 tọa độ WORLD rồi recompute 24 angles, và augmentation xảy ra trước z-score. Với proposed method chỉ gồm Mirror+Yaw, lỗi làm hỏng angle channels trước đây về cơ bản đã được giải quyết.

Kết quả downstream cũng cho thấy Mirror+Yaw không chỉ là một experiment riêng lẻ. Ví dụ AAGCN Bone tăng lên `66.36%` test-window, World Joint `70.99%`, Joint Motion `57.70%`, Bone Motion `57.83%`; tức augmentation mới đã thực sự được propagate sang graph pipeline.

**Stacking mới hợp lý hơn bản cũ.** Code hiện train Logistic Regression trên `train_probs + train_targets`, rồi evaluate trên Val/Test. Vì vậy vấn đề “train Stacking trên Val rồi tự chấm Val” đã biến mất. Đây là một comparison hợp lệ với Uniform/Hard trên validation. Việc train_probs là in-sample của base models vẫn là một limitation so với OOF stacking, nhưng không phải leakage Val/Test và không phải blocker.

**SOT direction đã đúng.** `artifacts/results/canonical_results_v2.json` đã trở thành nơi gom Table 2–7 và downstream artifacts được sinh lại. Đây chính là kiến trúc code → artifacts → paper mà ta muốn.

---

# Nhưng hiện còn 4 blocker lớn

## 1. Table 6 đang có bug `Val Win F1`

Trong `run_phase7_table6_ensembles.py`:

```python
val_vf1 = [r["val_vid_f1"] for r in seed_metrics]

...

"val_win_f1": f"{np.mean(val_vf1):.4f} ..."
```

Tức là **Val Window F1 đang lấy nhầm Video F1**.

Ví dụ JSON hiện ghi cho SkelGym-Full Uniform:

```text
Val Win F1 = 0.8886
```

nhưng từ ba seed thực tế:

```text
0.8435
0.8481
0.8546
```

mean đúng khoảng:

> **0.8487**, không phải `0.8886`.

Các giá trị đúng từ per-seed hiện tại:

| Fusion | Lite Val Win F1 | Full Val Win F1 |
|---|---:|---:|
| Hard | .7883 | .8411 |
| Accuracy-weighted | **.8341** | **.8487** |
| Uniform Soft | .8327 | **.8487** |
| Stacking | .8294 | .8442 |

Đây là lỗi phải sửa trước paper.

---

## 2. Fusion winner hiện đang chọn bằng **Video metric**, trái protocol mà ta vừa thống nhất

Code hiện:

```python
winner_full = max(...,
    key=lambda k: (
        val_vid_acc,
        val_vid_f1
    )
)
```

Trong khi hướng methodology chúng ta vừa thống nhất là:

> intermediate method selection → Validation Window Macro-F1 primary.

Do đó Full hiện bị chọn thành:

> **Hard Voting**

nhưng đó không phải winner theo Window-level criterion.

Với số hiện tại:

- Accuracy Weighted: Val Win F1 ≈ `.8487`, Win Acc `84.99`
- Uniform: Val Win F1 ≈ `.8487`, Win Acc **85.05**
- Hard: Val Win F1 `.8411`, Win Acc `84.11`

Nếu rule là:

> Primary Val Win Macro-F1 → Secondary Val Win Acc

thì **SkelGym-Full phải nghiêng về Uniform Soft**, không phải Hard.

Lite thì Accuracy-Weighted vẫn có vẻ là winner.

Còn một chi tiết nữa: Accuracy-Weighted hiện tính weight từ:

```python
r["val_vid_acc"]
```

Nếu bạn thực sự muốn protocol **window-only cho design selection**, nó cũng nên chuyển thành `val_win_acc` hoặc một criterion window-level đã freeze.

---

## 3. Tables 8–10 đang đánh giá **Stacking**, không phải hệ thống winner ở Table 6

Đây là blocker nghiêm trọng nhất về consistency.

`run_tables8_9_10_on_server.py` hard-code:

```python
# SkelGym-Lite
LogisticRegression(...)

# SkelGym-Full
LogisticRegression(...)
```

rồi Table 8, 9, 10 đều dùng các predictions này.

Thành ra hiện tại:

```text
Table 6/7:
SkelGym-Full = Hard Voting

Table 8/9/10:
SkelGym-Full = Stacking
```

Đây là hai hệ thống khác nhau.

Table 9 thậm chí ghi rõ:

> `SkelGym-Full (Stacking)`

và Table 10 cũng được generate từ `p_skel_full_*` của Stacking.

Nếu sau khi sửa criterion Full winner thành Uniform thì downstream lại càng lệch.

**Tables 8–10 phải đọc `t6["winners"]` và dựng đúng predictions của system đã freeze.**

Không cần retrain model; chủ yếu regenerate fusion predictions + statistics.

---

## 4. Table 9 có một kết quả rõ ràng sai/stale

Table 7:

> Transformer Mix v2 Clean = **69.24% Test Win Acc**

nhưng Table 9 bootstrap:

> Transformer Mix v2 Clean = **36.45%**

Hai số này không thể cùng đúng cho cùng một model/test set.

Script Table 9 đang tự load lại clean checkpoint và inference lại, nên khả năng cao có mismatch ở:

- checkpoint;
- normalization artifact;
- feature-version;
- hoặc fallback checkpoint.

Đây là dấu hiệu provenance chưa khóa hoàn toàn.

Trước khi viết Results, cần bắt buộc đạt invariant:

```text
point estimate từ bootstrap
≈ direct metric của chính prediction artifact
≈ Table 7 metric
```

Chênh lệch do bootstrap mean có thể nhỏ, nhưng không thể `69% → 36%`.

---

# Một số lỗi nhỏ nhưng nên sửa trước paper

`RESULTS_FINAL.md` vẫn chứa vài wording cũ/mâu thuẫn:

- Table 4 nói “hide video metrics” nhưng verdict vẫn ghi `Severe Collapse (-3.06% Vid)`.
- Table 5 vẫn ghi `Strong Vid F1`, `Vid drops ...`.
- nhiều `Val Win F1 = N/A` dù run-level JSON có sẵn F1.
- `candidate_minus_time` trong Phase 3 script vẫn còn comment `identical to proposed 4-op`.
- `run_phase7...` docstring vẫn nói Stacking “fit strictly on Val”, trong khi code đã chuyển sang Train.
- mô tả `Ridge/Logistic Regression` trong khi thực tế hiện chỉ dùng `LogisticRegression`.

Ngoài ra câu:

> “Mirror and Yaw are rigid isometries in SE(3)”

không đúng toán học. **Yaw rotation** thuộc rotation/rigid-motion group, nhưng **reflection/mirroring không thuộc SE(3)** vì reflection có determinant `-1`. Trong paper nên dùng kiểu:

> “distance-preserving Euclidean transformations”

hoặc phân biệt:

> Yaw rotation + bilateral reflection.

Và câu “Scale/Jitter/TimeWarp corrupt metric proportions and velocity profiles” cũng quá gộp. Uniform Scale không làm thay đổi joint angles hay tỷ lệ hình học nội tại; TimeWarp mới trực tiếp làm thay đổi temporal dynamics.

---

# Có một bug khác ở Hard Voting video metrics

Trong branch Hard:

```python
val_preds = hard_voting(val_probs)
val_fused_prob = np.mean(val_probs, axis=0)
```

Sau đó Video metric lại được tính từ:

```python
aggregate_video_level_predictions(val_fused_prob, ...)
```

Tức:

> Window metric = Hard Voting  
> Video metric = thực chất Uniform Soft probabilities

Đó là lý do Hard và Uniform có Video metrics giống nhau trong Table 6.

Nếu vẫn report Video metric cho từng fusion method, phải định nghĩa lại semantics. Nếu mục tiêu của bạn là window-only selection thì đơn giản nhất là **không dùng Video columns ở Table 6 selection**, rồi final video evaluation chỉ chạy trên winner.

---

# Stacking còn một safeguard nên thêm

Train-set Stacking hiện concatenate:

```python
Transformer train_probs
Bone train_probs
Joint train_probs
...
```

nhưng chưa thấy assertion kiểu:

```python
assert np.array_equal(train_targets_stream1,
                      train_targets_stream2)
```

Các dataset có vẻ được tạo deterministic và eval-train đều `shuffle=False`, nên khả năng alignment đúng khá cao.

Nhưng vì Stacking phụ thuộc **row-by-row correspondence**, nên trước paper nên thêm assertion cho:

- same number of train samples;
- identical train_targets;
- nếu có thể, identical sample/video/window IDs.

Đây là một guard nhỏ nhưng rất đáng có.

---

# Verdict của mình

### Methods / System Design
**Sẵn sàng khoảng 90%.**

Bạn có thể bắt đầu chỉnh:
- WORLD representation;
- mix_v2;
- SkelGym-Aug Mirror+Yaw;
- Transformer/AAGCN architectures;
- augmentation rationale;
- train-set Stacking methodology;
- dataset/protocol.

### Results / Discussion
**Chưa sẵn sàng freeze.**

Cần xử lý bốn blocker:

1. sửa `val_win_f1`;
2. freeze fusion selection criterion theo Window metric;
3. regenerate Tables 7–10 từ đúng validation winner;
4. xử lý mismatch Transformer Clean `69.24 ↔ 36.45`.

### Experiment training
Khả năng cao **không cần một vòng retrain lớn nữa**.

Phần lớn việc còn lại là:

> fix metric aggregation → refusion → regenerate downstream statistics → consistency audit.

Chỉ retrain nếu điều tra `36.45%` cho thấy checkpoint Clean thật sự stale/incompatible.

Sau khi bốn điểm đó pass, mình sẽ xem repository **đủ ổn để chuyển sang phase chỉnh paper toàn diện và dùng canonical SOT làm nguồn duy nhất**.