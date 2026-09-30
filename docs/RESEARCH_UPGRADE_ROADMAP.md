Mình đã audit repository `nxc1802/Gym_Classification` trực tiếp từ GitHub Source Project, bao gồm cấu trúc repo, pipeline dữ liệu, feature engineering, augmentation, LSTM/BiLSTM/Transformer/ST-GCN/AAGCN, ensemble, training, tests, scripts benchmark/thống kê, outputs và manuscript/preprint. Repo nguồn đúng là repo được khai báo trong project. :chatgpt-content-reference{index="0"}

Đây là **static audit theo code hiện tại trên `main`**, không phải việc chạy lại toàn bộ 100-epoch experiments trên GPU. Vì repo hiện thiếu dependency manifest, mình cũng không coi câu “tests passing” trong README là đã được xác nhận độc lập.

## 1. Tổng quan kiến trúc

Dự án hiện đã vượt khá xa dạng notebook nghiên cứu ban đầu. Có thể nhìn thành pipeline:

**Video → MediaPipe Pose → landmark CSV → segment/window → feature engineering → augmentation → normalization → backbone → ensemble → window/video metrics → paper/results.**

Phần chính nằm trong `src/`; `Source Code/` là thế hệ notebook cũ. `scripts/` chứa orchestration, multi-seed, ablation, thống kê, latency, external benchmark; `outputs/` giữ kết quả; `preprint/`, `paper/`, `GYM_Publication/` chứa manuscript.

Dataset metadata thực tế có **1.024 rows, 22 lớp**, split **580 train / 208 val / 236 test**. Mình kiểm tra `filepath` và không thấy filepath nào nằm ở hơn một split. Vì vậy claim **source-video-level separation** có cơ sở. Tuy nhiên không thể suy rộng thành “subject-independent split”, vì các nguồn web không có identity ID để chứng minh cùng một người không xuất hiện ở hai video khác nhau.

Phần feature hiện tại xoay quanh 13 joints. Representation chính `mix=117` gồm **39-d relative 3D + 78-d pairwise elevation angles**. Transformer và AAGCN là hai paradigm chính, sau đó fusion bằng validation-calibrated SLSQP weighted soft voting.

Kết quả final hiện được report qua 3 seed `42, 123, 3407`. `outputs/new_aug_downstream_results.md` ghi SkelGym-Full:

| Metric | Mean ± SD |
|---|---:|
| Window accuracy | **69.74% ± 1.04%** |
| Window Macro-F1 | **0.6882 ± 0.0068** |
| Video accuracy | **79.11% ± 0.25%** |
| Video Macro-F1 | **0.7834 ± 0.0082** |

Các con số này nhất quán giữa file downstream mới và `multi_seed_evaluation_results.md`. Nhưng có các vấn đề implementation bên dưới đủ lớn để mình khuyến nghị **chưa freeze paper/final benchmark trước khi re-run một vòng sạch**.

---

# 2. Các phát hiện quan trọng nhất

| Mức | Phát hiện | Ảnh hưởng |
|---|---|---|
| **Critical** | Dynamic SkelGym-Aug chạy **sau z-score normalization** | Phép mirror/yaw/scale không còn là phép biến đổi hình học trên skeleton vật lý như paper mô tả |
| **Critical** | Thiếu landmark CSV thì pipeline bình thường có thể âm thầm tạo **random dummy data** | Có thể train/evaluate dữ liệu giả thay vì fail |
| **Critical/High** | Runner ablation cho phép reuse checkpoint generic giữa các augmentation variant | Có nguy cơ một số LOO result không đến từ model riêng của đúng variant |
| **High** | Mirror của 78 signed elevation angles chỉ permutation mà không xử lý orientation/sign | `rel_3d` và `angle2_3d` có thể không còn mô tả cùng skeleton |
| **High** | Method section và code không thống nhất về coordinate origin, normalization và azimuth | Reproducibility của paper không phản ánh implementation |
| **High** | Window bootstrap/McNemar coi hàng nghìn window cùng video là độc lập | CI có thể quá hẹp và p-value quá nhỏ |
| **High** | “External benchmark” không chạy trên dataset external Deyzel | Đang là internal 4-class overlap analysis, không phải external validation |
| **Medium/High** | Latency end-to-end được suy ra chưa đúng protocol | 4.33 ms chỉ là classifier; không phải video→prediction end-to-end |
| **Medium** | Repo không có dependency manifest, LICENSE và CI | Fresh clone chưa reproducible như README mô tả |
| **Medium** | Một số output/docs là phiên bản lịch sử, chứa local filesystem links | Người đọc dễ nhầm preliminary với final benchmark |

## 3. Lỗi lớn nhất: augmentation đang áp dụng sai không gian

Trong `get_dataloaders()` của `src/data/dataset.py`, flow hiện tại là:

```text
build train dataset
→ compute train mean/std
→ z-score train/val/test
→ tạo DataLoader
→ __getitem__()
→ LandmarkAugmenter.apply(...)
```

Với `skel_gym_aug`, augmentation được thực hiện động trong `GymDataset.__getitem__()`. Như vậy skeleton đã được standardize theo từng feature dimension trước khi `mirror`, `yaw_rotate_3d`, `scale`, `jitter`.

Đây là vấn đề bản chất.

Ví dụ yaw đúng trong tọa độ vật lý cần:

```text
x' = x cos θ + z sin θ
z' = -x sin θ + z cos θ
```

Nhưng nếu `x` và `z` đã lần lượt biến thành `(x-μx)/σx` và `(z-μz)/σz`, phép rotation trên hai đại lượng này không tương đương rotation skeleton ban đầu, đặc biệt khi `σx != σz`.

Scale cũng tương tự: nhân giá trị standardized với 0.9–1.1 không còn có nghĩa “body size / subject distance scaling”.

Mirror cũng bị ảnh hưởng bởi mean không zero chính xác trên từng coordinate channel.

**Flow đúng nên là:**

```text
raw landmarks
→ physical augmentation
→ feature extraction
→ normalize bằng train statistics
→ model
```

Hoặc nếu muốn augmentation trực tiếp trên feature representation thì phải chứng minh transform trong feature space tương đương transform trên skeleton vật lý.

Điểm này quan trọng vì contribution C3 của paper lập luận SkelGym-Aug “strictly enforces musculoskeletal and gravitational constraints”. Implementation hiện tại không đáp ứng chính xác câu đó.

---

# 4. Silent random-data fallback là lỗi reproducibility rất nguy hiểm

Trong `build_dataset_from_csvs()`, khi không tìm thấy landmark CSV, code tạo DataFrame bằng:

```python
np.random.randn(...)
```

rồi tiếp tục extract windows/features.

Điều này hữu ích cho smoke test, nhưng **không được phép xảy ra trong normal train/eval**.

Một người clone repo mà chưa tải `data/landmarks` có thể không nhận một `FileNotFoundError` rõ ràng. Thay vào đó họ có nguy cơ chạy model trên dữ liệu ngẫu nhiên.

Nên đổi logic thành:

```text
if smoke_test:
    synthesize fixture
else:
    raise FileNotFoundError(...)
```

và ở cuối data loading nên assert số video/windows kỳ vọng.

Đây cũng đặc biệt quan trọng vì repository GitHub không chứa toàn bộ landmark data; các script thường dựa vào `data/landmarks`/Hugging Face.

---

# 5. Ablation runner cần được audit/re-run sạch

`scripts/run_augmentation_experiments.py` có `find_existing_checkpoint()`.

Với seed khác 42, nó thử những candidate dạng:

```text
seed123/best_Transformer_mix.pt
seed123/best_Transformer_<augmentation>_mix.pt
```

Một checkpoint generic `best_Transformer_mix.pt`, nếu tồn tại, có thể được nhận làm “existing checkpoint” cho **nhiều augmentation task khác nhau**, trừ khi task-specific checkpoint đã tồn tại trước.

Trong report LOO hiện có một dấu hiệu cần kiểm tra: một số seed cho kết quả giống hệt tới hai chữ số, ví dụ `-Mirror = 64.71%` ở cả ba seed và một số configuration khác có hai seed trùng hoàn toàn.

Điều đó **không chứng minh benchmark sai**, vì checkpoint provenance JSON/raw logs không được commit trong `outputs/`. Nhưng kết hợp với reuse logic, nó đủ để yêu cầu audit.

Cách an toàn là mọi run phải có key duy nhất kiểu:

```text
{model}__{feature}__{augment}__seed{seed}__{git_sha}.pt
```

và ablation final nên `force_retrain` toàn bộ 3×variants từ scratch, không reuse legacy checkpoints.

---

# 6. Mirror của `angle2_3d` có khả năng sai dấu

`compute_pair_angles_3d()` định nghĩa elevation có dấu:

\[
\theta_{ij}
=
\operatorname{atan2}
\left(
\Delta y,
\sqrt{\Delta x^2+\Delta z^2}
\right).
\]

Trong `LandmarkAugmenter.mirror()` cho 78-d angles, code hiện chỉ lookup permutation của symmetric pair.

Nhưng permutation được xây dựng bằng cách sort joint indices của cặp mirrored. Nếu symmetry mapping làm thứ tự cặp bị đảo, displacement vector chuyển từ \(p_j-p_i\) thành \(p_i-p_j\). Khi đó \(\Delta y\) đổi dấu và elevation cũng phải đổi dấu.

Tests hiện chỉ kiểm tra shape/bounds:

```text
mirror → vẫn 78 dims
values nằm trong range
```

chứ chưa kiểm tra invariant quan trọng:

```text
extract_angle(mirror(raw_skeleton))
==
mirror_angle(extract_angle(raw_skeleton))
```

Đây là test nên thêm ngay.

---

# 7. Code và manuscript hiện chưa mô tả cùng một thuật toán

Có ít nhất ba lệch pha.

**Coordinate origin.** `src/data/features.py` hiện subtract **hip midpoint** cho relative features. Nhưng feature section trong `preprint/main.tex` lại có phương trình lấy **nose** làm origin. Ngay trong manuscript cũng có đoạn khác nói mid-hip.

**Normalization.** Paper nói “torso-scale normalization ... on each individual window or frame” và tránh dataset-wide parameters. Code không có torso-scale normalization tương ứng; nó dùng **global mean/std từ train set** để z-score train/val/test. Dùng train statistics là hoàn toàn hợp lệ về leakage, nhưng phải mô tả đúng.

**78 angles.** Abstract/C2 có câu “pairwise elevation and azimuth angles”. Code thực tế chỉ tạo một elevation \(\theta\) cho mỗi trong 78 cặp. Manuscript sau đó cũng chỉ concatenate 78 elevation values. Nếu dùng cả elevation và azimuth thì chỉ riêng angular block phải là 156 dims, không phải 78.

Cách đúng là gọi representation:

> 39-d hip-centered relative 3D coordinates + 78-d pairwise elevation angles.

---

# 8. Statistical analysis cần chuyển sang video-cluster-aware

Test set có 2.743 windows nhưng chỉ khoảng 233 valid videos trong phần window/video evaluation. Nhiều windows từ cùng source video không phải independent samples theo nghĩa thống kê.

Hiện `bootstrap_window()` resample từng window độc lập. Điều này có thể đánh giá thấp uncertainty do bỏ qua within-video correlation.

Tương tự, window-level McNemar dùng tất cả windows như độc lập, nên các giá trị như `p < 10^-12` có thể trở nên quá mạnh.

Tốt hơn là **cluster bootstrap theo video**: resample video ID rồi lấy toàn bộ windows thuộc video đó. Hoặc ưu tiên video-level statistical inference là primary.

Ở video level, script chạy Wilcoxon trên vector correctness `{0,1}`. Với paired binary outcome, McNemar exact/sign-based test phù hợp tự nhiên hơn Wilcoxon và paired t-test.

README đề cập Bonferroni-adjusted threshold, nhưng `compute_statistical_tests.py` mình đọc không thấy code thực thi Bonferroni correction. Nếu adjustment chỉ áp dụng thủ công trong paper thì cần ghi rõ; tốt hơn nên đưa correction vào script để audit được.

Một điểm khác: ±SD của 3 seeds chỉ phản ánh **optimization randomness trên fixed split**, không phải uncertainty của population/generalization. Hai khái niệm này nên tách khỏi video-level bootstrap CI.

---

# 9. “External benchmark” hiện chưa thật sự external

`scripts/evaluate_external_benchmark.py` không load SU-EMD/Deyzel external dataset.

Nó lấy **chính test set SkelGym**, lọc bốn class giao nhau:

```text
squat
deadlift
barbell biceps curl
lateral raise
```

sau đó tính open-set/closed-set.

Vì vậy nên gọi là:

> “4-class S&C overlap analysis inspired by Deyzel et al.”

chứ chưa nên gọi “External Benchmark Evaluation”.

Phần “one-shot transfer” cũng lấy support/query từ chính các video internal thuộc những class mà model đã được train để nhận diện. Đây gần với **one-shot prototype/retrieval evaluation on seen classes** hơn là external one-shot transfer learning.

Muốn claim external generalization mạnh, cần thực sự inference/fine-tune trên một dataset bên ngoài chưa tham gia training.

---

# 10. Latency: classifier nhanh, nhưng claim end-to-end đang quá mạnh

`benchmark_hardware_latency.py` benchmark model bằng dummy tensor, warm-up + timed forward pass. Cách này tốt để đo **classifier inference latency**.

Nhưng có ba điểm:

Script gọi kết quả THOP là `MFLOPs` trong khi THOP trả về MACs; cần định nghĩa quy ước rõ.

Latency ensemble được tính bằng cộng mean/median/p95 của từng model. Cộng mean cho serial estimate có thể dùng được; nhưng **sum(median) không phải median của tổng**, và **sum(p95) không phải p95 end-to-end**. Nên benchmark actual ensemble call.

Quan trọng nhất, MediaPipe pose extraction được hard-code khoảng 8–15 ms/frame chứ không được benchmark trong script. Một decision cũng cần 32 frames, tương đương khoảng **1.07 giây observation window ở 30 FPS**. Vì vậy:

```text
0.42–4.33 ms = classifier compute after features are ready
```

không phải:

```text
video → skeleton → 32-frame evidence → prediction = 4.33 ms
```

Paper nên tách “observation horizon”, “pose latency/frame” và “post-window classifier latency”.

---

# 11. Reproducibility/repo engineering

README hiện hướng dẫn:

```bash
pip install -r requirements.txt
```

nhưng trên branch `main` mình kiểm tra không thấy `requirements.txt`.

Cũng không có `pyproject.toml`, `setup.py`, `environment.yml`; không thấy `LICENSE` dù README/paper nói MIT; không thấy `.github/workflows`; `submission_checklist.md` được README nhắc tới nhưng không tồn tại.

Điều này có nghĩa fresh clone hiện chưa đủ để tái lập môi trường.

`configs/default.yaml` cũng khá stale: default feature `12rel_4`, augmentation choices cũ và một số model setting không còn đồng bộ hoàn toàn với benchmark final.

`outputs/EXPERIMENT_RESULTS.md` chứa link kiểu:

```text
file:///Volumes/WorkSpace/Project/...
```

nên không hoạt động với người khác.

Repo cũng đang gom source code, legacy notebooks, generated figures, PDFs, manuscript, outputs và orchestration vào cùng một chỗ. Với research repository thì chấp nhận được, nhưng trước publication nên tách rõ source-of-truth và generated artifacts.

---

# 12. Những phần được làm tốt

Mặc dù có các vấn đề trên, nền tảng dự án khá mạnh.

Dataset split đã được làm ở source-video level và metadata hiện không có filepath overlap giữa train/val/test. Feature/model architecture được tách module rõ. Validation và test loaders không có train augmentation. Weighted ensemble tối ưu weight trên validation thay vì test — đây là lựa chọn đúng. Repo có unit/integration tests cho shapes, graph/sequence forward passes, zero-frame handling, FocalLoss, video aggregation và augmentation basics. Multi-seed final runner cũng tách checkpoint theo seed tốt hơn những runner cũ.

Việc chuyển narrative SkelGym-Aug từ “novel augmentation operator” thành **task-oriented configuration of established transforms** là chỉnh sửa khoa học hợp lý. Final protocol bỏ TimeWarp sau ablation cũng có rationale hợp lý cho exercise cadence.

Model results cũng cho một pattern có ý nghĩa: video aggregation cải thiện đáng kể so với window prediction và heterogeneous ensemble vượt constituent models trong các report hiện tại. Chỉ cần benchmark sạch sau khi sửa các issue critical để các kết luận này đứng vững hơn.

# 13. Thứ tự sửa mình khuyến nghị

1. **Sửa data pipeline trước:** physical augmentation phải xảy ra trước standardization; cấm random fallback ngoài smoke test; thêm geometry-consistency tests cho mirror/yaw/scale.
2. **Sửa experiment provenance:** checkpoint name duy nhất theo model/feature/augmentation/seed/git SHA; re-run toàn bộ final SkelGym-Aug + LOO với `force_retrain`; lưu config/hash/log cho từng run.
3. **Sửa statistical evaluation:** cluster bootstrap theo video, video-level paired test, multiple-comparison correction trong code.
4. **Đồng bộ paper với implementation:** hip midpoint, chỉ elevation 78-d, train-stat z-score; hạ claim “zero leakage” thành “zero source-video overlap”.
5. **Đổi tên external analysis** hoặc thực sự chạy dataset external; sửa latency thành classifier-only/post-window latency.
6. **Đóng gói reproducibility:** `pyproject.toml`/locked requirements, `LICENSE`, CI test workflow, environment versions, checkpoint metadata + normalization stats.
7. **Dọn repository:** tách legacy/generated artifacts, sửa broken local links và xác định một `RESULTS_FINAL.md` duy nhất làm source of truth.

### Kết luận

Về mặt research engineering, **SkelGym có kiến trúc và experimental scope tốt hơn đáng kể so với một project sinh viên thông thường**: có data isolation, nhiều paradigm, multi-seed, ablation, ensemble và manuscript tương đối đầy đủ.

Nhưng ở trạng thái hiện tại, mình xem ba vấn đề **augmentation-after-normalization, silent fake-data fallback, và checkpoint reuse trong ablation runner** là các blocker cần giải quyết trước khi coi benchmark/paper là frozen. Sau đó phải re-run ít nhất các experiment liên quan SkelGym-Aug; không nên chỉ sửa wording vì hai lỗi đầu thay đổi trực tiếp computation.

Nếu mục tiêu tiếp theo là chuẩn bị repository/paper để submit, mình có thể **tiếp tục audit ở mức line-by-line và tạo trực tiếp một branch GitHub sửa các lỗi Critical/High kèm tests**, rồi chuẩn bị PR để bạn review.