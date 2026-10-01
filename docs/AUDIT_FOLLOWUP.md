Mình đã audit repo được trỏ trong file bạn gửi :chatgpt-content-reference{index="0"}, tập trung vào cấu trúc toàn repo, các entrypoint/runtime path chính trong `src/`, toàn bộ nhóm model/data/training/ensemble, các runner trong `scripts/`, test/CI, config, outputs và source Paper ESWA/LNCS.

## Kết luận

**Chưa nên gọi dự án là “clone về là Ready to Run toàn bộ”.**

Đánh giá chính xác hơn là:

| Phần | Trạng thái | Nhận xét |
|---|---|---|
| Python package / import / dependencies | 🟢 Ready | Cài được trên clean GitHub Actions |
| Unit + integration-level tests | 🟢 Ready | **46/46 tests pass** trên Python 3.10 và 3.11 |
| Core models + features + augmentation | 🟢 Ready | Các forward path và feature dimensions đã được test |
| Internal training pipeline | 🟢 Ready | Multi-seed runner và canonical hyperparameter registry khá tốt |
| Internal evaluation / ensemble | 🟢 Ready có điều kiện | Cần landmarks + checkpoints ngoài GitHub |
| Reproducibility từ **fresh GitHub clone** | 🟠 Chưa hoàn toàn | Data/checkpoints/reference artifacts không nằm trong GitHub và bootstrap chưa tự động |
| Full paper reproduction | 🟠 Chưa hoàn toàn | Code chạy được, nhưng artifacts và tài liệu còn lệch nhau |
| MM-Fit external validation | 🔴 Chưa hoàn tất | Data/landmarks/metrics thực tế còn pending |
| Paper consistency / submission freeze | 🟠 Chưa freeze | Có một số claim và số liệu stale giữa checklist ↔ paper ↔ README |
| Full E2E từ clean machine | 🔴 Chưa thể xác nhận Ready | CI hiện chỉ test code, chưa chạy research pipeline thật |

**Tóm lại: Core engineering pipeline đã Ready; project release/reproduction E2E vẫn chưa Ready.**

---

## 1. Source code hiện tại tốt hơn khá nhiều so với các notebook ban đầu

Repo hiện có hai lớp.

`Source Code/` chứa các notebook lịch sử như `LSTM_Training.ipynb`, `ST-GCN.ipynb`, `mediapipe.ipynb`, `Split_data.ipynb`, `Test_model.ipynb`...

Nhưng implementation chính hiện đã được refactor thành library:

```text
src/
├── cli.py
├── constants.py
├── data/
│   ├── dataset.py
│   ├── features.py
│   ├── augmentations.py
│   ├── extractor.py
│   └── report.py
├── models/
│   ├── lstm.py
│   ├── transformer.py
│   ├── stgcn.py
│   ├── aagcn.py
│   └── ensemble.py
├── training/
│   ├── trainer.py
│   └── metrics.py
└── utils/
```

Đây là cấu trúc hợp lý cho một research repo muốn reproduce được. Notebook không còn là dependency bắt buộc của pipeline chính.

---

## 2. Kiến trúc model và Paper nhìn chung khớp nhau

Paper hiện mô tả SkelGym-Full là:

**1 Transformer Mix + 4 AAGCN streams → late fusion → SLSQP validation calibration.**

Code cũng đúng cấu trúc đó:

```text
Transformer
    └─ mix = 117-d

AAGCN
    ├─ bone_3d
    ├─ rel_3d
    ├─ joint_motion_3d
    └─ bone_motion_3d

             ↓
      SLSQP soft voting
             ↓
       SkelGym-Full
```

`src/constants.py` cũng đóng vai trò source-of-truth tương đối tốt cho protocol:

- sequence length = `32`
- train stride = `16`
- validation/test stride = `32`
- Transformer Mix: lr `1e-4`, batch `16`
- AAGCN: lr `1e-3`, batch `32`
- 100 epochs
- patience 10
- early stopping theo `val_macro_f1`

Điểm này rất quan trọng: runner mới không còn phụ thuộc quá nhiều vào các hyperparameter hard-code rải rác như trước.

---

## 3. Feature 117-d hiện khớp Paper

Implementation hiện dùng:

\[
39\text{-d Relative 3D} + 78\text{-d Pairwise Angles}=117\text{-d}
\]

Đây cũng là representation được mô tả trong Paper.

Phần mirror augmentation đã được làm khá cẩn thận. Với 117-d feature, code không chỉ flip dấu X; nó:

1. swap left/right joints;
2. negate trục X;
3. **recompute 78 pairwise angles từ coordinates mới**.

Đây là cách đúng hơn nhiều so với việc reorder một vector angle một cách cơ học.

Test suite cũng có test cho consistency này.

---

## 4. SkelGym-Aug hiện khớp với phiên bản Paper mới

`skel_gym_aug()` hiện mặc định gồm:

```text
Mirror
+ 3D Yaw
+ Scale
+ Jitter
```

và **TimeWarp bị disable**.

Điều này khớp với narrative hiện tại của Paper: time warping được loại sau ablation vì có thể phá cadence/tempo của exercise.

Phần này mình đánh giá là **Ready**.

---

## 5. Data leakage protection trong code được thiết kế khá đúng

`get_dataloaders()` xử lý:

```text
metadata split
    ↓
train / val / test datasets
    ↓
compute μ, σ ONLY from TRAIN
    ↓
apply same μ, σ to VAL and TEST
```

Ngoài ra normalization có thể được freeze thành artifact theo seed.

Đây là một điểm mạnh của implementation hiện tại.

Tuy nhiên, cần phân biệt:

> Code **tuân theo `split` đã có trong metadata**.

Nó không tự chứng minh rằng metadata ban đầu không có cùng subject/session nằm ở hai split. Vì vậy Paper có thể claim **“no source-video overlap”** nếu metadata đã kiểm tra điều đó, nhưng câu kiểu:

> “completely eliminating recording-session memorization”

mạnh hơn những gì source code tự chứng minh được.

---

## 6. Training pipeline hiện đã ở trạng thái tốt

`Trainer` đã có:

- early stopping;
- best + last checkpoint;
- AMP;
- seed handling;
- checkpoint provenance;
- SHA-256;
- normalization provenance;
- optional Hugging Face upload.

Multi-seed runner cũng có:

```python
seeds = [42, 123, 3407]
```

và canonical experiment registry được sử dụng để chọn hyperparameters theo backbone.

Đặc biệt, audit cũ từng phát hiện cross-backbone ablation dùng sai LR/batch size. Phiên bản HEAD đã sửa bằng `CANONICAL_EXPERIMENT_REGISTRY`.

Vì vậy với **data đã có local**, internal training path hiện nhìn khá ổn.

---

# 7. CI thực tế đang xanh

Mình kiểm tra đúng commit HEAD hiện tại:

```text
6760317f918321096ffebdc48ad7437270cedafb
```

commit ngày **01/10/2026**.

GitHub Actions run `36847782394` đã:

```text
Python 3.10   ✅
Python 3.11   ✅
```

Trên Python 3.11:

```text
collected 46 items

46 passed
1 warning
```

Warning duy nhất là SciPy precision-loss trong một statistical unit test với dữ liệu gần như giống nhau; nó không phải runtime failure.

Các tests hiện cover được khá nhiều:

- feature extraction;
- 117-d geometry;
- augmentations;
- model forward;
- parameter budget;
- ensembles;
- sliding windows;
- normalization;
- missing landmarks;
- provenance;
- checkpoint compatibility;
- bootstrap/statistics;
- reproducibility packaging.

Đây là evidence mạnh rằng **core code thực sự chạy**, không chỉ là README nói chạy.

---

# 8. Nhưng CI xanh ≠ E2E research pipeline chạy được

Đây là khác biệt quan trọng nhất.

CI chỉ chạy:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

python -m pytest tests/ -v
```

Nó **không chạy**:

```bash
python scripts/evaluate_local_ensemble.py
python scripts/run_multi_seed_experiments.py
python scripts/evaluate_external.py ...
```

trên dataset thật + checkpoint thật.

Do đó 46/46 pass chứng minh:

> code/API/model logic hoạt động.

Nó chưa chứng minh:

> fresh clone → download artifacts → reproduce paper numbers.

---

# 9. Blocker lớn nhất: fresh GitHub clone không có data/checkpoints

`.gitignore` cố ý loại:

```text
/data/
checkpoints/
mm-fit/
*.pt
*.pth
*.npy
```

Đó là hợp lý vì artifacts lớn nằm trên Hugging Face.

Nhưng vấn đề là main reproduction script mặc định:

```python
--metadata data/Final_dataset_metadata.csv
--landmark_dir data/landmarks
--checkpoint_dir checkpoints
```

Trong GitHub lại chỉ có:

```text
Final_dataset_metadata.csv
```

ở **root**, không phải:

```text
data/Final_dataset_metadata.csv
```

và không có `data/landmarks/`.

`evaluate_local_ensemble.py` cũng **không tự động bootstrap data/checkpoints** trước khi evaluation.

Trong code đã có utility rất tốt:

```python
pull_landmarks_from_hf()
ensure_checkpoint_available()
```

nhưng các main evaluation runners chưa sử dụng chúng như một automatic preflight.

Đây là lý do chính mình không gọi repo hiện tại là:

> “git clone && python evaluate...”

ready.

---

# 10. README Quickstart đang có một lỗi UX/reproducibility khá lớn

README nằm trong **GitHub repo**, nhưng Quickstart hiện hướng dẫn:

```bash
git clone https://huggingface.co/Cuong2004/gym-exercise-classification
cd gym-exercise-classification
pip install -r requirements.txt
```

Tức là user đang đọc source GitHub nhưng được bảo clone **Hugging Face model repo**.

Sau đó README lại yêu cầu:

```bash
python scripts/evaluate_local_ensemble.py
```

Điều này chỉ an toàn nếu HF repository luôn mirror toàn bộ source tree và đúng commit với GitHub.

Với một publication repo, nên có một đường duy nhất:

```text
GitHub = code
Hugging Face = data + weights
```

rồi bootstrap HF artifacts từ GitHub checkout.

README thực ra nói mô hình tổ chức này ở phần khác, nhưng Quickstart chưa thực thi đúng triết lý đó.

---

# 11. Dependency management chạy được nhưng chưa “frozen reproducibility”

Hiện có ba hệ:

```text
requirements.txt
requirements-lock.txt
environment.yml
```

và còn có `uv.lock`.

`requirements.txt` sử dụng lower bounds như:

```text
torch>=2.0
numpy>=1.23
mediapipe>=0.10
...
```

Điều thú vị là CI mới nhất đã kéo các version rất mới và vẫn pass. Đây là điểm tốt cho compatibility.

Nhưng `requirements-lock.txt` lại ghi:

```text
verified environment (Python 3.14 / macOS)
```

trong khi:

```text
environment.yml: Python >=3.10,<3.12
CI: Python 3.10, 3.11
```

Ngoài ra lock file này không phải dependency closure hoàn chỉnh.

Với Paper reproduction, mình sẽ chọn **một canonical environment**, tốt nhất Python 3.11 và một frozen `uv.lock`/lockfile duy nhất.

---

# 12. MM-Fit hiện là blocker rõ ràng nhất cho “Full E2E”

Repo đã có framework khá đầy đủ cho:

```text
MM-Fit
 ├─ native pose protocol
 ├─ MediaPipe protocol
 ├─ open-set
 ├─ closed-set
 └─ one-shot
```

Nhưng artifacts hiện tại nói rất rõ trạng thái thực tế.

`outputs/external/mmfit/data_audit.csv` cho các workout như `w00`, `w05`, `w12`, `w13`, `w20` hiện có:

```text
rgb_video_exists = False
mediapipe_csv_exists = False
frame_alignment_status = CHECK_BOUNDS
```

Trong `segment_metadata.csv`:

```text
rgb_path = N/A
landmark_path = N/A
qc_status = PENDING_LANDMARKS
```

Và `dataset_audit.csv` gần như trống.

Quan trọng hơn, chưa thấy các expected final artifacts như:

```text
metrics_open.csv
metrics_closed.csv
fewshot_summary.csv
confusion_open.png
confusion_closed.png
bootstrap_ci.json
run_manifest.json
```

Tức là:

**MM-Fit adapter đã xây xong khá nhiều, nhưng external experiment chưa được chạy đến đích.**

Điều này cũng khớp với `docs/AUDIT_FOLLOWUP.md`, nơi internal pipeline được đánh dấu READY nhưng MM-Fit Protocol A/B vẫn là **PENDING**.

---

# 13. Paper và current final results khá đồng bộ ở phần chính

Điểm tốt là Paper ESWA hiện dùng:

\[
69.74\%\pm1.04\%
\]

window accuracy và

\[
79.11\%\pm0.25\%
\]

video consensus accuracy.

`outputs/RESULTS_FINAL.md` cũng dùng chính bộ số này.

Macro-F1:

```text
Window: 0.6882 ± 0.0068
Video:  0.7834 ± 0.0082
```

Vì vậy mình xem:

> `paper/paper_eswa.tex` + `outputs/RESULTS_FINAL.md`

là hai nguồn canonical hiện tại.

Paper cũng khớp code ở:

```text
117-d representation
32-frame window
Transformer + four AAGCN streams
SLSQP validation late fusion
SkelGym-Aug without TimeWarp
```

---

# 14. Nhưng `submission_checklist.md` đã stale

Đây là một inconsistency cần sửa trước release.

Checklist hiện nói multi-seed:

```text
Window Accuracy = 70.03% ± 0.70%
Video Accuracy  = 77.68% ± 0.86%
```

Trong khi Paper + `RESULTS_FINAL.md` hiện là:

```text
Window Accuracy = 69.74% ± 1.04%
Video Accuracy  = 79.11% ± 0.25%
```

Trong cùng `RESULTS_FINAL.md` còn có:

```text
70.11% window
77.68% video
```

nhưng đó được ghi rõ là **baseline point run**, không phải 3-seed aggregate.

Có vẻ checklist đang giữ số liệu từ một revision cũ.

Vì thế claim trong checklist rằng:

> “100% Submission-Ready”

hiện không còn chính xác với HEAD.

---

# 15. Còn vài wording issue trong Paper

Mình thấy ít nhất các vấn đề này cần cleanup trước freeze cuối:

- README vẫn có `privacy-preserving deployment`; Paper cũng còn ít nhất một đoạn dùng `privacy-preserving`, dù checklist nói đã đổi toàn bộ sang `privacy-aware`.
- Paper còn cụm **“state-of-the-art mean video consensus accuracy”**, và `RESULTS_FINAL.md` còn “Primary Benchmark SOTA”, trái với checklist nói self-claimed SOTA đã được loại sạch.
- Abstract nói **“sub-millisecond edge classification”**, nhưng full ensemble là khoảng `4.33 ms CPU` và `8.77 ms MPS`; chỉ một số single models / CUDA path dưới 1 ms. Câu này nên narrow lại.
- One-shot comparison với Deyzel dùng protocol tương tự nhưng dataset khác; “matching and outperforming literature baselines” có thể bị reviewer xem là comparison hơi mạnh.
- Window-level McNemar vẫn có intra-video dependence; Paper đã nhận thức được cluster dependence ở bootstrap, nên tốt nhất nhấn video-level inference/statistics là evidence chính.

Đây không phải lỗi làm code crash, nhưng là **publication blockers** nếu mục tiêu là submission-quality freeze.

---

# 16. Một điểm nữa: Paper PDF có sẵn nhưng mình chưa coi đó là fresh-build proof

Trong repo có:

```text
paper_eswa.tex
paper_eswa.pdf
paper_llncs.tex
paper_llncs.pdf
cover_letter.tex
cover_letter.pdf
```

Nghĩa là source đã từng compile thành công.

Nhưng CI hiện tại **không compile LaTeX**. Vì thế chưa có guarantee rằng:

> clean checkout HEAD → pdflatex/bibtex → PDF byte mới

vẫn thành công.

Nên thêm một paper-build CI job nếu Paper là artifact chính thức.

---

## Những việc mình xem là bắt buộc trước khi gắn nhãn `READY_TO_RUN`

1. **Tạo một bootstrap command duy nhất** cho fresh clone: tải landmarks, metadata, checkpoints và reference artifacts từ HF; sau đó verify SHA/version.
2. **Sửa README Quickstart** thành clone GitHub trước, rồi pull artifacts từ Hugging Face; không clone HF thay cho source repo.
3. **Thêm E2E smoke CI**: dùng một fixture nhỏ + một checkpoint thật và thực sự gọi data loader → model → checkpoint → inference → aggregation, thay vì chỉ unit tests.
4. **Hoàn tất MM-Fit** hoặc tách hẳn nó khỏi scope của release hiện tại. Hiện state `PENDING_LANDMARKS/CHECK_BOUNDS` không phù hợp với “full pipeline ready”.
5. **Freeze Paper/repository consistency**: lấy `paper_eswa.tex + RESULTS_FINAL.md` làm source-of-truth, sửa checklist cũ, `privacy-preserving`, self-claimed SOTA và “sub-millisecond” wording; sau đó thêm CI compile Paper.

---

## Verdict cuối

Nếu câu hỏi là:

> **“Source code chính có đủ chất lượng để bắt đầu chạy training trên server không?”**

**Có.** Mình xem internal pipeline là **Ready to rerun**. CI HEAD xanh trên Python 3.10/3.11, 46 tests pass, protocol registry, normalization, augmentation, models và multi-seed runner đã khá chín.

Nếu câu hỏi là:

> **“Một người khác có thể clone GitHub hôm nay và reproduce toàn bộ Paper chỉ bằng vài command README không?”**

**Chưa.**

Nếu câu hỏi là:

> **“Project đã hoàn thành full E2E bao gồm external MM-Fit và đủ sạch để freeze/publication release chưa?”**

**Chưa.** MM-Fit còn pending và paper/reproducibility docs còn một số inconsistency.

Điểm quan trọng nhất hiện tại không phải viết thêm model. **Việc nên làm tiếp theo là biến repo từ “research code chạy được” thành “one-command reproducible release”.**