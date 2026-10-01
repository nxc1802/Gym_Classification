Mình đã kiểm tra HEAD mới nhất của repo. Commit mới nhất hiện là **`6760317f918321096ffebdc48ad7437270cedafb`**, message:

`feat(external): add MMFit data audit, mediapipe extraction scripts, and sync pipeline tests`

Commit lúc **17:13 ngày 01/10/2026 (UTC+7)**.

Kết luận: **chưa nên chạy final External Test để lấy số liệu publication**. Phần **download + MediaPipe extraction đã gần như ready**, nhưng evaluation còn vài lỗi có thể làm kết quả sai về mặt protocol, đặc biệt là frame alignment, geometric normalization và 1-shot leakage.

## Trạng thái hiện tại

| Thành phần | Trạng thái | Nhận xét |
|---|---|---|
| Download 5 MM-Fit RGB | 🟢 Ready | `extract_external_mediapipe.py` có URL Zenodo và download tự động |
| MediaPipe Heavy extraction | 🟢 Gần ready | whole-workout extraction đúng hướng |
| Audit raw data | 🟡 Có thể chạy | chưa kiểm tra frame-ID alignment đúng nghĩa |
| Core-4 class mapping | 🟢 Ready | mapping hợp lý |
| Feature engineering | 🟡 Gần ready | dùng lại SkelGym feature code, nhưng missing-frame handling chưa parity |
| SkelGym train normalization | 🟡 Logic đúng | cần frozen artifacts thực tế |
| Open/Closed-set evaluation | 🔴 Chưa ready | geometric norm + frame alignment |
| Set-level consensus | 🟢 Logic cơ bản đúng | hiện gọi `recording_acc`, nên đổi tên |
| Workout-level aggregation | 🟠 Chưa thực hiện thật | docstring nói có nhưng code chỉ window→set |
| Bootstrap CI | 🔴 Chưa đúng protocol | đang resample set, không cluster theo workout |
| 1-shot | 🔴 Chưa ready | leakage + “SkelGym-Full” chưa thật sự Full |
| Multi-seed | 🔴 Có silent fallback nguy hiểm | có thể vô tình dùng seed-42 checkpoint/weights |

---

## Các blocker cần sửa trước final run

1. **P0 — MediaPipe đang bị geometric-normalize dù không nên.** Đây là lỗi quan trọng nhất sau frame alignment. Trong `MMFitExternalDataset`, commit mới đã sửa đúng logic mặc định: `pose_source=="mediapipe"` thì không geometric normalization. Nhưng các caller lại override nó. `configs/external/mmfit.yaml` hiện có `source: native` và `normalization.geometric: true`; `evaluate_external.py` truyền `apply_geometric_norm=True` từ config; `prepare_mmfit_external.py` hardcode `apply_geometric_norm=True`; `evaluate_external_fewshot.py` cũng hardcode `True`. Kết quả là MediaPipe coordinates `[x,y,z]` vốn cùng representation với SkelGym lại bị hip-center + torso scaling + rotation thêm một lần. Với Protocol A hiện tại, config nên khóa thành:

```yaml
pose_protocol:
  source: mediapipe

normalization:
  geometric: false
  feature_stats: skelgym_train
```

và tốt hơn là caller truyền `None` để adapter tự quyết định theo `pose_source`.

2. **P0 — `start_frame/end_frame` của MM-Fit đang bị dùng như array index.** Starter code MM-Fit chính thức không làm vậy. Nó đọc frame ID chứa trong pose data rồi đối chiếu với label start/end. Trong adapter hiện tại lại có:

```python
s = s_frame
e = e_frame
skel_13 = full_pose_3d[s:e]
```

Điều này đặc biệt sai với native pose. Chính `data_audit.csv` commit mới cho thấy ví dụ `w00`: native pose chỉ có `63,918` samples nhưng label cuối tới frame `67,650`; `w13`: `68,561` samples nhưng label tới `73,024`. Nghĩa là **frame ID ≠ array position**.

Với MediaPipe cũng chưa nên dùng `df.iloc[s:e]`, vì extractor SkelGym hiện ghi cột:

```text
Frame = 1, 2, 3, ...
```

trong khi row index là:

```text
0, 1, 2, ...
```

Ít nhất có nguy cơ lệch một frame. Adapter nên giữ cột `Frame` và slice bằng frame ID:

```python
mask = (df_mp["Frame"] >= start_frame) & (df_mp["Frame"] < end_frame)
df_segment = df_mp.loc[mask]
```

Sau đó audit phải kiểm tra actual IDs, không phải chỉ:

```python
len(df_mp) >= last_label_frame
```

Hiện `frame_alignment_status` chưa thực sự xác nhận alignment.

3. **P0 — 1-shot hiện bị support/query leakage giữa workouts.** Trong `simulate_one_shot_transfer()`, code có tạo:

```python
support_subjects = set()
```

nhưng không dùng nó để filter query toàn cục. Nó chỉ loại query nếu query workout giống support workout **của cùng class**. Ví dụ support `squat` lấy từ `w00`; query `push-up` từ `w00` vẫn có thể lọt vào nếu support push-up lấy từ workout khác. Đây vẫn là same-camera/same-person/same-session leakage.

Phải đổi thành logic kiểu:

```python
support_groups = {
    record_to_subject[r]
    for r in support_records.values()
}

query_records = [
    r for r in all_records
    if record_to_subject[r] not in support_groups
]
```

Tức **bất kỳ workout nào xuất hiện trong support đều bị loại khỏi toàn bộ query set**.

Ngoài ra hiện `subject_id = w00/w05/...`, thực chất là **workout ID**, không phải participant ID. Vì vậy nếu chưa build mapping workout→participant, paper chỉ nên gọi đây là **workout-disjoint 1-shot**, không gọi subject-disjoint.

4. **P0 — “SkelGym-Full” trong 1-shot không phải SkelGym-Full.** `evaluate_external_fewshot.py` hiện chỉ tạo:

```python
concat(
    Transformer embedding,
    AAGCN Bone embedding
)
```

rồi đặt tên:

```text
SkelGym-Full
```

Nhưng SkelGym-Full classification có 5 components:

```text
Transformer Mix
AAGCN Bone
AAGCN Rel
AAGCN Joint Motion
AAGCN Bone Motion
```

Hiện embedding này gần với **Lite representation**, không phải Full. Có hai lựa chọn hợp lệ: đổi tên nó thành `Transformer+Bone embedding`, hoặc implement đủ 5 streams với frozen validation weights. Cho publication mình chọn phương án thứ hai.

5. **P0/P1 — bootstrap hiện không phải workout-cluster bootstrap.** Code hiện window→exercise-set rồi `compute_recording_level_bootstrap_ci()` resample từng set. Nhưng các set trong cùng `w00` dùng cùng participant/camera/session và correlated mạnh. Docstring nói “Subject/Workout-level bootstrap”, implementation chưa làm vậy. Với MM-Fit unseen benchmark chỉ có 5 workouts, primary CI nên cluster theo:

```text
w00
w05
w12
w13
w20
```

Một bootstrap draw phải lấy cả cluster tất cả sets thuộc workout đó.

6. **P0 — multi-seed có silent fallback làm sai provenance.** `find_checkpoint_path()` với seed `123` hoặc `3407` nếu không tìm thấy checkpoint seed-specific sẽ fallback sang:

```text
checkpoints/best_XXX.pt
```

tức rất có thể seed 42. Sau đó bảng vẫn ghi kết quả là seed 123/3407. Tương tự ensemble weights: nếu `seed123/ensemble_weights.json` thiếu, nó fallback sang base `ensemble_weights.json`; nếu không có gì thì còn fallback uniform weights. Điều này trái với chính claim “frozen SkelGym validation weights”. Final external evaluator nên **fail loudly**, không fallback:

```python
if seed_specific_checkpoint_missing:
    raise FileNotFoundError
```

và ensemble weights cũng vậy.

7. **P1 — external window preprocessing chưa hoàn toàn giống SkelGym.** SkelGym training/test pipeline hiện làm zero-frame interpolation trước feature extraction và reject **từng window** nếu zero ratio >20%. `BaseExternalDataset.extract_windows()` hiện không làm bước đó. Nó chỉ QC cả exercise set trước đó. Có thể một set có zero ratio 10% nhưng toàn bộ missing frames tập trung vào một window 32-frame; window đó vẫn đi vào inference. Cách sạch nhất là reuse chính `extract_windows_from_segment()` của SkelGym thay vì duy trì một phiên bản external riêng.

---

## Dữ liệu hiện tại cũng chưa sẵn sàng để evaluation

Commit đã commit `outputs/external/mmfit/data_audit.csv`. File đó hiện ghi cho cả 8 workouts:

```text
rgb_video_exists = False
mediapipe_csv_exists = False
frame_alignment_status = CHECK_BOUNDS
```

Tức commit này mới chứng minh **official labels/native pose đã được index**, chưa có RGB hoặc MediaPipe output ở thời điểm audit được commit.

Điểm tốt là segment metadata đã build khá sạch. Với unseen Core-4, các exercise-set đã được index cho squat, push-up, dumbbell shoulder press và lateral raise, đồng thời biceps curl được tách thành supplementary Extended-5. Phần này mình xem là ổn.

---

## Một vấn đề về test coverage

Commit mới thêm test cho augmentation pipeline, nhưng **không có pytest nào cho `src/external/*`**. `scripts/test_feature_parity.py` là standalone script và còn có logic:

```python
if no SkelGym test CSV:
    print("Skipping test")
    return
```

nên nó không đảm bảo CI fail nếu external parity chưa được kiểm chứng.

Ngoài ra commit `6760317` hiện không có GitHub combined status hoặc workflow run gắn với commit mà mình có thể thấy. Vì vậy mình không coi “latest external pipeline has passed CI” là đã được xác nhận.

---

# Đánh giá cuối cùng

**Download/extract phase: 8/10 — có thể bắt đầu chạy ngay.**

Script:

```bash
python scripts/extract_external_mediapipe.py \
  --workouts w00 w05 w12 w13 w20 \
  --raw-dir data_external/mmfit/raw/rgb \
  --out-dir data_external/mmfit/landmarks \
  --complexity 2 \
  --workers 4
```

là hợp lý để bắt đầu. Mình sẽ dùng **2–4 workers**, chưa dùng mặc định server 8 workers ngay từ đầu vì MediaPipe Heavy trên 5 video lớn có thể tạo RAM/CPU pressure.

Sau extraction có thể chạy audit, **nhưng chưa chạy final classifier metrics** cho tới khi frame-ID alignment được sửa.

**Open/Closed-set final evaluation: 5/10 — chưa ready.**

**1-shot final evaluation: 3/10 — chưa ready**, chủ yếu vì support/query leakage và `SkelGym-Full` chưa đúng định nghĩa.

**Publication-ready External Test: chưa.**

Nếu sửa 6 điểm chính là **MediaPipe geometric norm, frame-ID slicing, one-shot global workout isolation, true Full embedding, workout-cluster bootstrap, strict seed artifact validation**, thì architecture tổng thể hiện tại đã đủ tốt để tiến tới final run. Sau đó nên sửa thêm zero-frame/window parity trước khi đóng số liệu.