# Báo cáo tách thay đổi giữa thầy và bạn — SkelGym Paper

Mình đã đối chiếu trực tiếp **ba phiên bản của bản thảo ESWA**:

| Ký hiệu | Phiên bản | Vai trò |
|---|---|---|
| **A — BASE** | Commit [`6637305c7e`](https://github.com/nxc1802/Gym_Classification/commit/6637305c7e05dde8b38550cd949a1a41eff905d6) | Phiên bản nền có khả năng cao nhất |
| **B — THẦY** | File `Văn bản đã dán (1).txt` | Bản LaTeX đã được thầy chỉnh sửa |
| **C — BẠN** | Commit [`55c8b9885b`](https://github.com/nxc1802/Gym_Classification/commit/55c8b9885bfe795b31e4becfe22487e9271f7c02) | Bản nghiên cứu mới nhất trên GitHub tại thời điểm kiểm tra |

Mình xác nhận **B là bản ESWA**, nên toàn bộ so sánh sử dụng `paper/paper_eswa.tex`, không sử dụng bản LNCS làm nền.

## 1. Tổng quan hai nhánh thay đổi

| Chỉ tiêu | Thầy: A → B | Bạn: A → C |
|---|---:|---:|
| Tổng số dòng phiên bản đích | 1.480 | 1.342 |
| Dòng giữ nguyên theo thứ tự so với A | 1.254 | 1.020 |
| Nhóm thay đổi (diff hunks) | **45** | **158** |
| Dòng được thêm theo diff | 226 | 322 |
| Dòng bị xóa/thay thế theo diff | 130 | 364 |
| Phạm vi chủ yếu | Introduction, Related Work, Dataset | Methods, Experiments, Results, Discussion |
| Bản chất | **Biên tập học thuật** | **Nâng cấp nội dung nghiên cứu** |

Các con số trên là kết quả so sánh dòng LaTeX; việc tách một đoạn văn thành nhiều dòng cũng được tính là thay đổi, không đồng nghĩa từng dòng đều chứa một ý mới.

**Phát hiện quan trọng:** Trong 45 nhóm thay đổi của thầy, có **19 nhóm tác động tới các dòng nền mà bạn cũng đã thay đổi**. Đây là những vùng xung đột về nội dung tiềm năng, không nhất thiết là 19 Git merge conflicts.

Hai nhánh có mức độ bổ sung cho nhau khá cao. Tuy nhiên, chúng không thể merge bằng cách ưu tiên nguyên xi một bên, vì Methods và Results hiện tại đã khác đáng kể so với thời điểm thầy bắt đầu sửa.

---

## 2. Nhánh của thầy: A → B

### 2.1. Phân bố chỉnh sửa

| Khu vực | Nhóm thay đổi | Dòng thêm | Dòng xóa/thay |
|---|---:|---:|---:|
| Frontmatter, Abstract | 2 | 3 | 1 |
| **Introduction** | **11** | **47** | **97** |
| **Related Work** | **8** | **37** | **9** |
| **Dataset & Protocol** | **15** | **37** | **15** |
| Methodology | 8 | 102 | 7 |
| Appendix | 1 | 0 | 1 |
| **Experimental Results** | **0** | **0** | **0** |
| Discussion và Conclusion | **0** | **0** | **0** |

Phần lớn của 102 dòng thêm trong Methodology là **một hình TikZ và phần mô tả đi kèm**, không phải một phương pháp mới.

Điều này xác nhận rằng thầy đang tập trung vào phần trình bày nghiên cứu ở đầu bài, trong khi chưa can thiệp đáng kể vào các bảng kết quả.

### 2.2. Introduction — chỉnh sửa quan trọng nhất

**Vị trí bản thầy: dòng 69–127.**

Thầy đã thay đổi cách mở đầu theo hướng:

**Bản nền:** Giới thiệu vấn đề, sau đó mô tả bốn đóng góp bằng các đoạn `C1–C4` khá dài.

**Bản thầy:** Tổ chức lập luận thành chuỗi:

1. Nhu cầu tập luyện và virtual personal trainer.
2. Hạn chế của RGB-based recognition.
3. Lợi thế của skeleton-based recognition.
4. Ba thách thức nghiên cứu.
5. Danh sách đóng góp của SkelGym.

Thầy cũng chủ động chuyển câu văn dài thành nhiều câu ngắn, làm rõ quan hệ giữa vấn đề và giải pháp.

Chẳng hạn phần đóng góp được thay bằng:

> “The contributions of this work are as follows:”

Sau đó trình bày dưới dạng `itemize`. Văn bản đã dán (1)

**Đây là chỉnh sửa nên bảo tồn.**

Tuy nhiên, các đóng góp mà thầy đang viết vẫn là:

| Nội dung thầy viết | Tình trạng |
|---|---|
| 117-d kinematic descriptor | Không còn đúng |
| Bốn phép augmentation | Không còn là cấu hình cuối |
| SLSQP soft voting | Không còn sử dụng |
| 69.74% / 79.11% | Kết quả benchmark cũ |
| CPU 4.33 ms | Benchmark cũ |

**Hướng kết hợp:** Giữ cấu trúc lập luận và cách tổ chức của thầy, nhưng thay nội dung C1–C4 bằng phương pháp và kết quả hiện tại.

Riêng đoạn thầy ghi chú *“không hiểu lắm”* về bốn phép augmentation, cần xử lý bằng cách giải thích lại contribution mới, không chỉ sửa ngữ pháp. Văn bản đã dán (1)

### 2.3. Related Work — phần đáng giữ gần như nguyên vẹn về cấu trúc

**Vị trí bản thầy: dòng 129–174.**

Thầy thực hiện ba thay đổi đáng kể.

**Một là**, rút gọn lịch sử skeleton-based recognition, tập trung vào các mốc từ RNN/LSTM đến ST-GCN, adaptive GCN và skeleton Transformer.

**Hai là**, diễn đạt phần skeletal augmentation theo ba nhóm:

- Spatial-geometric transformations.
- Temporal transformations.
- Noise injection.

**Ba là**, rút gọn phần fitness recognition bằng cách giới thiệu trực tiếp các công trình tiêu biểu thay vì mô tả quá dài.

Thầy cũng đã comment phần `Ensemble Learning for Skeleton-Based Recognition` trong bản cũ.

Mình đánh giá:

| Thay đổi | Nên xử lý |
|---|---|
| Rút gọn tổng quan RNN → ST-GCN → AAGCN | Giữ |
| Viết lại phần Transformer | Giữ |
| Phân loại augmentation thành ba nhóm | Giữ |
| Rút gọn fitness-related literature | Giữ |
| Bỏ hoàn toàn Ensemble Learning subsection | Cân nhắc |
| Citation placeholders `\cite{}` | Phải bổ sung |

Đặc biệt, **không nên bỏ hết Related Work về fusion**, bởi cross-paradigm fusion vẫn là một đóng góp trung tâm của SkelGym. Có thể giữ ý của thầy về sự súc tích và chỉ viết một đoạn ngắn về late fusion.

Bản thầy còn hai citation placeholders chưa hoàn thiện ở phần augmentation. Văn bản đã dán (1)

### 2.4. Dataset — thầy chỉnh cả văn phong lẫn phạm vi trình bày

**Vị trí bản thầy: dòng 176–318.**

Đây là khu vực cần merge cẩn thận thứ hai, sau Introduction.

| Phần | Thầy thay đổi | Đề xuất |
|---|---|---|
| Dataset overview | Viết lại danh sách 22 bài tập và nguồn dữ liệu | Giữ ý, sửa tiếng Anh |
| Ethical compliance | Rút gọn phần đạo đức và sự đa dạng người tham gia | Giữ sau khi xác minh |
| Source statistics | Giữ số liệu, chỉnh định dạng đoạn | Giữ |
| Inclusion criteria | Chia thành các điều kiện (i)–(iv) | Giữ cách trình bày |
| Exclusion criteria | Chuyển phần lớn nội dung thành comment | Cân nhắc khôi phục |
| Action trimming | Comment subsection và mô tả trimming | **Không nên mất thông tin này** |
| Video-level splitting | Comment phần giải thích chi tiết | **Cần giữ protocol trong paper mới** |
| Image copyright | Đánh dấu để kiểm tra | Bắt buộc xử lý |

Có hai điểm cần phân biệt.

**Điểm tích cực:** Thầy nhấn mạnh rằng đối tượng khoa học của nghiên cứu là *skeletal landmarks extracted from videos*, không phải trực tiếp video RGB. Đây là hướng làm rõ rất tốt.

**Điểm cần điều chỉnh:** Đoạn Ethical Compliance mới vẫn nói rằng video-level partitioning loại bỏ hoàn toàn *subject leakage* và demographic bias. Đây là claim không được chứng minh chỉ bằng video-level splitting, nên không thể đưa nguyên văn vào paper cuối. Văn bản đã dán (1)

### 2.5. Figures — thầy có yêu cầu quan trọng về bản quyền

Thầy đánh dấu hai vị trí:

**Figure Dataset:** Cần kiểm tra các hình minh họa đến từ dataset nào, đặc biệt nếu có hình từ các video công khai.

**Figure MediaPipe:** Cân nhắc chỉ hiển thị skeleton landmarks hoặc ảnh do nhóm tự ghi hình, nhằm hạn chế rủi ro bản quyền. Văn bản đã dán (1)

Đây không phải chỉnh sửa mang tính hình thức.

Mình xếp chúng vào nhóm **cần xác minh trước submission**, và nên giữ nguyên dưới dạng review notes cho đến khi đã quyết định nguồn ảnh.

### 2.6. Methodology — có chỉnh sửa, nhưng không thể copy trực tiếp

Thầy đã:

- Đổi tên `Proposed Methodology` thành `Methodology`.
- Thêm một hình mô tả kiến trúc và đoạn giải thích.
- Chỉnh một số tiêu đề trong phần lập luận toán học, như *Tracking Noise Mitigation*, *Gradient Singularity Elimination* và *Dimensional Compactness*.

Tuy nhiên, các lập luận đó dựa trên **117-d representation gồm 78 pairwise angles**.

Trong bản GitHub mới, Mix v2 chỉ có **24 góc được chọn theo giải phẫu**.

Vì vậy:

**Có thể giữ cách trình bày của thầy, nhưng không thể giữ nguyên các chứng minh, giả định và kết luận định lượng cũ.**

Một vấn đề cấu trúc rõ ràng: bản thầy có hai `\section{Methodology}` tại khoảng dòng 176 và 320. Cần hợp nhất thành một section duy nhất.

Hình mà thầy bổ sung cũng vẫn hiển thị 117-d Mix, Transformer 4L/8H và SLSQP. Do đó **không nên đưa hình này trực tiếp sang paper hiện tại**. Văn bản đã dán (1)

---

## 3. Nhánh của bạn: A → C

Ở nhánh GitHub, thay đổi phân bố ngược lại.

| Khu vực | Nhóm thay đổi |
|---|---:|
| Frontmatter | 2 |
| Introduction | 12 |
| Related Work | 3 |
| Dataset | 3 |
| **Methodology** | **28** |
| **Experimental Results** | **84** |
| Error Analysis | 10 |
| Discussion | 13 |
| References | 3 |
| **Tổng cộng** | **158** |

Điều này giải thích vì sao file của thầy vẫn rất giống bản gốc dù GitHub đã thay đổi lớn.

### 3.1. Representation: từ 117-d sang 63-d

Đây là một trong những thay đổi khoa học lớn nhất.

| Thành phần | BASE / bản thầy | GitHub hiện tại |
|---|---|---|
| Spatial features | Relative 3D | MediaPipe World 3D |
| Điểm tham chiếu | Nose-centered | World landmarks có gốc gần pelvis |
| Spatial dimension | 39 | 39 |
| Angular features | 78 pairwise angles | 24 kinematic angles |
| Total dimensions | **117** | **63** |

Không thể kết hợp hai công thức bằng cách giữ cả hai làm phương pháp đề xuất.

**Nguồn chuẩn phải là GitHub hiện tại.**

Những chỉnh sửa của thầy trong phần lý thuyết có thể được sử dụng để cải thiện cách giải thích 63-d Mix v2, nhưng không nên chuyển các công thức 78 pairwise angles thành chứng minh cho 24 kinematic angles.

### 3.2. Augmentation: từ bốn phép sang Mirror + Yaw

| Nội dung | BASE / bản thầy | GitHub hiện tại |
|---|---|---|
| Candidate pool | Mirror, Yaw, Scale, Jitter, TimeWarp | Giữ candidate pool để ablation |
| Final augmentation | Mirror + Yaw + Scale + Jitter | **Mirror + Yaw** |
| Lý do lựa chọn | Chủ yếu LOO và kiểm tra Test | Validation-window F1/Accuracy và isolated ablations |
| Diễn giải | Các geometric operators nhìn chung có lợi | Tương tác giữa các toán tử không cộng tính |

Đây là chỗ nên kết hợp:

**Giữ văn phong giải thích của thầy về biomechanical constraints, nhưng giữ hoàn toàn bằng chứng lựa chọn cấu hình mới từ GitHub.**

### 3.3. Transformer architecture

| Thành phần | Bản thầy | GitHub hiện tại |
|---|---:|---:|
| Input | 117-d | 63-d |
| Layers | 4 | 3 |
| Heads | 8 | 4 |
| Parameters | Khoảng 399–400K | **301K** |

Các số cũ còn xuất hiện ở hình, bảng hyperparameters và Methods của bản thầy.

Chúng phải được đồng bộ sang kiến trúc mới.

### 3.4. Ensemble fusion — thay đổi căn bản

| Thành phần | Bản thầy | GitHub hiện tại |
|---|---|---|
| SkelGym-Lite | SLSQP | **Accuracy-Weighted Soft** |
| SkelGym-Full | SLSQP | **Uniform Soft** |
| Fusion calibration | Validation NLL optimization | Validation-based protocol selection |
| Video fusion | SLSQP video weights riêng | Temporal consensus sau fusion |
| Learned meta-classifier | Có liên quan trong thảo luận cũ | Không thuộc final protocol |

Đây là xung đột **không thể merge tự động**.

Bản thầy có một subsection riêng:

> `Validation-Calibrated Ensemble Weighting via Constrained Optimization`

Subsection này phải được thay bằng **Cross-Paradigm Ensemble Fusion** của bản GitHub mới.

Các lý giải về ưu thế calibration của SLSQP cũng không nên chuyển sang phần Discussion của Uniform Soft.

### 3.5. Experimental Results — nhánh bạn thay gần như toàn bộ

Đây là khác biệt lớn nhất giữa hai bên.

| Metric | Bản thầy | GitHub hiện tại |
|---|---:|---:|
| SkelGym-Full Window Acc | 69.74 ± 1.04% | **76.19 ± 0.16%** |
| Window Macro-F1 | 0.6882 ± 0.0068 | **0.7537 ± 0.0014** |
| Video Acc | 79.11 ± 0.25% | **85.27 ± 0.41%** |
| Video Macro-F1 | 0.7834 ± 0.0082 | **0.8386 ± 0.0075** |
| Parameters | 1.91M | **1.81M** |
| FLOPs | 418.21 MFLOPs | **830.29 MFLOPs** |
| Full CPU latency | 4.33 ms | **3.96 ms** |

Ngoài kết quả chính, bạn còn cập nhật:

- Feature ablations.
- Augmentation LOO và isolated ablations.
- Graph-stream benchmarks.
- Fusion protocol selection.
- Statistical hypothesis tests.
- Bootstrap confidence intervals.
- Per-class results và confusion analysis.
- Hardware benchmarks và external comparisons.

**Toàn bộ các bảng này phải lấy từ GitHub mới**, không sử dụng bản thầy để khôi phục những giá trị cũ.

### 3.6. Statistical analysis

Đây cũng là nội dung bạn đã sửa đáng kể.

Bản thầy sử dụng kết quả statistical tests thuộc hệ thống SLSQP cũ. GitHub hiện tại sử dụng các kết quả đã tính lại cho những cấu hình cuối.

Trong đó:

- Window-level McNemar đã được cập nhật.
- Video-level posterior-confidence tests đã được diễn giải chính xác hơn.
- Không còn tuyên bố tất cả các video-level comparisons đều có ý nghĩa thống kê.
- Không còn đồng nhất posterior-confidence differences với calibration improvements.

Các chỉnh sửa này cần được giữ nguyên.

### 3.7. Discussion và Error Analysis

Thầy hầu như không sửa phần này so với commit nền.

Trong khi đó, bạn đã cập nhật các phân tích theo mô hình mới, đặc biệt là:

- Per-class errors trên final Uniform Soft.
- Canonical seed 42.
- 200/233 video được phân loại đúng.
- Sự khác nhau giữa single-window và video consensus.
- Hạn chế của monocular pose estimation.
- Diễn giải thống kê và hiệu quả tính toán.

Mình đề xuất giữ GitHub làm nền cho toàn bộ Discussion, rồi mới áp dụng phong cách biên tập mà thầy sử dụng ở Introduction và Related Work.

---

## 4. Ma trận xung đột giữa hai bên

Đây là phần trực tiếp phục vụ công việc combine.

| Khu vực | Thầy đã thay đổi | Bạn đã thay đổi | Mức xung đột | Phương án |
|---|---|---|---|---|
| Title | Không đáng kể | Đổi sang Geometry-Preserving Augmentation | Thấp | Giữ GitHub |
| Abstract | Chủ yếu ghi chú link | Toàn bộ methodology và results | Cao | Giữ GitHub, xử lý ghi chú |
| Introduction | **Viết lại cấu trúc** | Sửa contributions/narrative mới | **Cao** | **Merge thủ công** |
| Contributions | Chuyển sang bullet list | Cập nhật C1–C4 | **Cao** | Dùng bố cục thầy + nội dung GitHub |
| Skeletal Related Work | Rút gọn mạnh | Ít thay đổi | Thấp | Ưu tiên thầy |
| Transformer Related Work | Viết gọn | Ít thay đổi | Thấp | Ưu tiên thầy |
| Augmentation Related Work | Viết gọn, cần citations | Sửa diễn giải khoa học | Trung bình | Kết hợp |
| Ensemble Related Work | Comment subsection | Giữ fusion làm contribution | Trung bình | Viết một đoạn ngắn |
| Dataset overview | Viết lại | Ít thay đổi | Thấp | Ưu tiên thầy sau kiểm chứng |
| Ethics | Viết lại | Sửa leakage claims | Cao | Giữ claim GitHub, cải thiện văn phong |
| Inclusion criteria | Đổi thành dạng điều kiện | Hầu như không đổi | Thấp | Ưu tiên thầy |
| Dataset figures | Ghi chú bản quyền | Chưa giải quyết bằng thay đổi nội dung | Cần kiểm tra | Xử lý thủ công |
| Overall pipeline figure | Thêm hình cũ | Có hình pipeline mới | Cao | Giữ hình GitHub |
| Feature formulation | Chỉnh cách trình bày 117-d | Nâng cấp thành 63-d | **Rất cao** | Giữ GitHub |
| Augmentation Methods | Ít can thiệp trực tiếp | Chọn Mirror + Yaw | Cao về nội dung | Giữ GitHub |
| Transformer Methods | Ít can thiệp | Đổi kiến trúc | Cao về nội dung | Giữ GitHub |
| Fusion Methods | Không sửa | Loại SLSQP | **Rất cao** | Giữ GitHub |
| Experimental tables | Không sửa | Thay đổi lớn | Không có xung đột biên tập trực tiếp | Giữ GitHub |
| Statistical analysis | Không sửa | Tính lại và sửa diễn giải | Không có xung đột biên tập trực tiếp | Giữ GitHub |
| Discussion | Không sửa | Cập nhật theo kết quả mới | Thấp | Giữ GitHub |
| Conclusion | Không sửa | Viết lại theo final system | Thấp | Giữ GitHub |

### Phần có giá trị kết hợp cao nhất

Mình ưu tiên **Introduction → Related Work → Dataset**.

Ba phần này có nhiều đóng góp biên tập từ thầy và ít phụ thuộc trực tiếp vào những thay đổi kiến trúc, nên có thể cải thiện paper cuối mà không phải can thiệp vào thí nghiệm.

---

## 5. Tách riêng các ghi chú mà thầy yêu cầu xử lý

Mình phát hiện **8 vị trí chứa `\rv{...}`** trong file thầy.

| # | Vị trí | Nội dung yêu cầu | Mức ưu tiên |
|---|---|---|---|
| 1 | Abstract, dòng 60 | Cân nhắc đổi hoặc bổ sung liên kết GitHub | Trung bình |
| 2 | Contributions, dòng 110 | Làm rõ câu diễn giải augmentation khó hiểu | Cao |
| 3 | Related Work, dòng 152 | Bổ sung citations cho augmentation | Cao |
| 4 | Dataset figure, dòng 181 | Kiểm tra nguồn hình và quyền sử dụng | **Cao** |
| 5 | Dataset overview, dòng 189 | Làm rõ dữ liệu thực tế là landmarks từ videos | Trung bình |
| 6 | Data availability, dòng 235 | Xem lại mô tả bản quyền và artifact release | **Cao** |
| 7 | MediaPipe figure, dòng 262 | Cân nhắc chỉ dùng landmarks hoặc ảnh nhóm tự ghi | **Cao** |
| 8 | Pipeline description, dòng 406 | Bổ sung đoạn mô tả luồng xử lý hệ thống | Trung bình |

Mình khuyên **không xóa các ghi chú này cho đến khi từng mục có quyết định rõ ràng**.

Đặc biệt, không nên chỉ thay link Hugging Face thành GitHub: repository chứa code và nơi phát hành model/dataset có thể có vai trò khác nhau. Cần xác nhận link nào chứa artifact nào rồi mới mô tả.

---

## 6. Những điểm cần thống nhất với thầy trước khi merge

Có bốn quyết định đáng trao đổi trực tiếp:

**A. Giữ Introduction của thầy ở mức nào?**

Mình khuyên giữ bố cục, cách chia đoạn và phong cách dẫn dắt. Nhưng phải viết lại Contributions theo phiên bản nghiên cứu mới.

**B. Có tiếp tục rút gọn Related Work không?**

Mình đồng ý với hướng của thầy. Tuy nhiên, vẫn cần một đoạn về fusion và các phương pháp liên quan để giải thích research gap.

**C. Cách xử lý các hình có nguồn từ internet?**

Đây là vấn đề bản quyền độc lập với Git merge. Nên chọn các hình có quyền sử dụng rõ ràng hoặc hình tác giả tự tạo.

**D. Có giữ toàn bộ Mathematical Derivation cũ không?**

Mình không khuyên. Một số lập luận của bản thầy liên quan trực tiếp đến 78 pairwise angles, trong khi representation cuối đã thay đổi.

Nên đưa vào paper mới những lập luận có thể chứng minh cho 63-d Mix v2; không giữ những chứng minh chỉ còn đúng với phiên bản 117-d.

---

## 7. Kế hoạch combine cụ thể

Mình đề xuất cách ghép như sau:

**Giai đoạn 1 — Academic editing merge**

Lấy GitHub hiện tại làm nền, lần lượt đưa vào:

- Cách triển khai lập luận của Introduction.
- Cấu trúc bullet-list Contributions.
- Phiên bản Related Work rút gọn.
- Cách mô tả Dataset và tiêu chí inclusion.
- Những câu giải thích rõ ràng hơn của thầy.

**Giai đoạn 2 — Scientific consistency**

Đối chiếu toàn bộ phần vừa ghép với:

- 63-d Mix v2.
- Mirror + Yaw.
- Transformer 3L/4H.
- Accuracy-Weighted Lite và Uniform Soft Full.
- Canonical Tables và statistical artifacts.

**Giai đoạn 3 — Advisor review resolution**

Giải quyết từng ghi chú `\rv{...}` và lưu quyết định cho từng mục.

**Giai đoạn 4 — Final verification**

Kiểm tra lại section hierarchy, figures, references, LaTeX compilation và tính nhất quán của ba bản manuscript.

### Kết luận

**Hai bên đã làm hai loại công việc khác nhau và nhìn chung không triệt tiêu nhau.**

Thầy đã cải thiện đáng kể **cách kể câu chuyện nghiên cứu**, nhất là Introduction, Related Work và Dataset. Bạn đã thay đổi đáng kể **nền tảng khoa học và bằng chứng thực nghiệm**.

Điều quan trọng là không đánh mất phần biên tập của thầy chỉ vì GitHub đã cập nhật mô hình, và cũng không đưa các kết quả 117-d/SLSQP cũ trở lại manuscript cuối.

**Phương án mình chọn: GitHub mới nhất làm scientific base, bản thầy làm academic-editing source, merge thủ công theo section; giữ riêng tám ghi chú của thầy cho tới khi được xử lý.**