Commit mới nhất là `481143b6a4c95c388d01f934b5a513fd6a529c9d` — **“Update roadmap and paper revisions”**.

Kết luận của mình: **chưa nên bắt đầu final author review ngay**. Paper đã ở khoảng **97%**, scientific results có thể coi là frozen, nhưng commit mới vẫn chưa sửa hết chính checklist mà roadmap tuyên bố đã “100% resolved”.

Điểm đáng chú ý nhất là **audit script pass không đồng nghĩa manuscript không còn stale prose**. Audit hiện chủ yếu kiểm tra việc các giá trị canonical mới *có xuất hiện* và các bảng khớp artifacts; nó không phát hiện việc một giá trị cũ vẫn cùng tồn tại ở đoạn khác. Vì vậy paper có thể “PASS” nhưng vẫn còn contradiction.

### Những gì commit mới đã sửa đúng

- SkelGym-Lite Methods: `210.57 → 221.72 MFLOPs` ✅
- Explicit discussion `Stacking / SLSQP` trong fusion section đã được bỏ, chuyển thành generic parameterized fusion ✅
- Core tables/results vẫn khớp SOT ✅
- WORLD Mix v2 / Mirror+Yaw / Uniform Soft / Accuracy-Weighted Lite vẫn nhất quán ✅
- Hardware table chính vẫn dùng canonical:
  - Transformer `18.87 MFLOPs`
  - Lite `221.72 MFLOPs`
  - Full `830.29 MFLOPs`
  - Full CPU `3.96 ms`
  - CUDA `4.65 ms`
  - MPS `5.86 ms` ✅
- External Deyzel benchmark có artifact riêng ✅
- Title, figures, 24-d feature definition, augmentation story đều đã ở trạng thái tốt ✅

Nhưng mình vẫn thấy **5 vấn đề cần sửa trước final review**:

| Mức | Vấn đề còn tồn tại |
|---|---|
| 🔴 | Implementation vẫn ghi **`seed = 42`** dù paper report 3 seeds |
| 🔴 | Một đoạn latency vẫn dùng bộ số cực cũ `0.42–4.33 ms CPU / 0.08–0.54 ms CUDA` |
| 🟠 | Parameter counts trong prose vẫn là BiLSTM `367K`, LSTM `378K`, ST-GCN `365K` |
| 🟠 | Statistics vẫn gọi posterior-confidence tests là **“confidence calibration improvements”** |
| 🟠 | Video-level split vẫn có wording mạnh quá về **recording session** |

### 1. Seed contradiction vẫn còn nguyên

`paper.tex` hiện vẫn viết:

> “trained using fixed random seed initialization (\(\text{seed}=42\))”

Trong cùng paper lại nói toàn bộ benchmark được báo cáo trên:

> `42, 123, 3407`

Đây là contradiction rõ.

Nên sửa thành ý:

> Each experiment was independently repeated with seeds 42, 123, and 3407; within each run, the assigned seed was fixed across data loading, initialization, and stochastic operations.

Sau đó mới giải thích seed 42 là canonical run khi cần per-class/confusion matrix.

### 2. Latency cũ vẫn tồn tại trong phần Implementation

Đoạn `Custom Lightweight Architecture Design...` vẫn ghi:

> `0.42--4.33 ms` CPU  
> `0.08--0.54 ms` CUDA

Trong khi hardware artifact và Table Hardware hiện đúng là:

> CPU `0.49–3.96 ms`  
> CUDA `0.59–4.65 ms`  
> MPS `1.36–5.86 ms`

Đây là lỗi số liệu thực sự, không phải style.

Điều đáng nói là audit script không bắt được vì nó chỉ kiểm tra **new canonical range có tồn tại đâu đó trong manuscript**, không kiểm tra **old range không còn tồn tại**.

### 3. Parameter counts prose vẫn stale

Implementation nói:

> Transformer 301K  
> AAGCN 378K  
> BiLSTM **367K**  
> LSTM **378K**  
> ST-GCN **365K**

nhưng các table hiện báo:

- LSTM: **362K**
- BiLSTM Mix v2: **360K**
- ST-GCN: **350K**
- Transformer: 301K
- AAGCN: 378K

Vì vậy reviewer đọc prose rồi nhìn Table sẽ thấy inconsistency.

Mình thậm chí khuyên tránh hard-code toàn bộ số ở prose này. Có thể viết:

> approximately 301K–378K parameters per backbone

rồi để exact counts trong table. Ít cơ hội drift hơn.

### 4. Statistical semantics vẫn chưa sửa

Main statistics paragraph vẫn viết:

> “confirming substantial confidence calibration improvements.”

Trong khi các test này đánh giá **paired ground-truth posterior confidence**, không phải calibration theo định nghĩa ECE/Brier/reliability diagram.

Hơn nữa:

- ST-GCN → Four-Stream: Wilcoxon `p = 0.3491`
- Bone → Full: paired-t `p = 0.2098`

Nên câu chuẩn là:

> Video-level tests evaluate paired differences in ground-truth-class posterior confidence; significance varies across model comparisons.

Caption:

> “pairwise model superiority”

cũng nên đổi thành:

> **pairwise performance differences**

McNemar window results thì vẫn mạnh và hợp lệ: tất cả 5 comparisons significant sau correction.

### 5. Split wording vẫn hơi vượt evidence

Paper vẫn viết:

> “no video file or recording session spans across partitions”

và:

> all clips from a given recording session reside in one split.

Bạn chắc chắn được **source-video identity**, nhưng web data không có reliable session/subject identifiers để chứng minh rằng hai source files khác nhau không cùng một recording session.

Nên giới hạn thành:

> no source video, derived segment, or temporal window crosses partitions.

Đây là claim rất mạnh rồi và hoàn toàn defend được.

---

Ngoài ra còn vài polish rất nhỏ:

- `bilateral arm-splay angle` → **arm-splay angle** vì chỉ có một angle.
- `leading recognition accuracy` → **best-performing configuration in our benchmark**.
- `MediaPipe World coordinates ... true physical scale` → nên giữ wording **estimated metric scale**, như Methods đã dùng.

### Vì sao roadmap nói 100% nhưng mình vẫn tìm thấy lỗi?

Roadmap trong commit này ghi:

> 1,338 checks, 0 discrepancies.

Mình không nghĩ con số đó giả; vấn đề là **coverage của audit**.

Ví dụ audit prose hiện kiểm tra:
- có `76.19 ± 0.16` không;
- có `85.27 ± 0.41` không;
- có canonical latency range mới không;
- có `1.81M`, `679K`, `301K` không.

Nhưng nó không kiểm tra:
- còn `seed=42` stale không;
- còn old latency elsewhere không;
- còn old LSTM/BiLSTM/ST-GCN params không;
- statistical wording có đúng semantics không;
- session-level claim có vượt evidence không.

Do đó “audit passed” và “paper fully semantically clean” là hai tiêu chuẩn khác nhau.

## Verdict

**Experiments:** frozen ✅  
**Canonical numerical tables:** frozen ✅  
**Scientific narrative:** frozen ✅  
**Figures:** essentially frozen ✅  
**Paper prose consistency:** còn một cleanup cuối ⚠️  
**Ready for your final author review:** **chưa — nhưng chỉ còn một commit nhỏ.**

Mình sẽ không chạy thêm experiment hay thay đổi cấu trúc paper nữa. Chỉ sửa khoảng **5–8 dòng/đoạn** nói trên và sync cả:

`paper/paper.tex` → `paper_eswa.tex` → `preprint/main.tex`.

Sau đó mình sẽ nói **YES: sẵn sàng để bạn review cuối**, và từ thời điểm đó chỉ nên sửa readability/grammar/formatting, không động vào scientific design nữa.