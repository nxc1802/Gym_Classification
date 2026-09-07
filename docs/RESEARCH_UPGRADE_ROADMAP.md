# Lộ Trình Nghiên Cứu & Nâng Cấp Hệ Thống Phân Loại Động Tác Gym (Research Upgrade Roadmap)

> **Mục tiêu**: Định hình phương hướng mở rộng và tối ưu hóa chuyên sâu cho hệ thống Gym Exercise Classification trong các giai đoạn nghiên cứu tiếp theo sau khi đã chuẩn hóa toàn diện Pipeline tiền xử lý (Spatial Body Normalization, Feature Scaling độc lập, Temporal Interpolation, Khử triệt để Data Leakage).

---

## 1. Tinh Giản Không Gian Thử Nghiệm (Model Pruning & Resource Efficiency)

### 1.1. Loại Bỏ LSTM và BiLSTM Khỏi Giai Đoạn Thử Nghiệm Tiếp Theo
* **Hạn chế cố hữu của Recurrent Networks (RNN/LSTM/BiLSTM)**:
  * Cơ chế lan truyền tuần tự (sequential recurrence) khiến LSTM khó nắm bắt các mối liên hệ không gian phức tạp giữa 33 khớp xương (chỉ làm phẳng vector $D=99$ hoặc $D=132$ rồi đưa vào unroll theo thời gian).
  * Đã đạt trần hiệu năng trong các thử nghiệm cơ sở (Baseline plateau) và có xu hướng bão hòa gradient ở các chuỗi dài hoặc gặp khó khăn trong việc tách biệt các bài tập có tư thế chuẩn bị tương đồng (ví dụ: Deadlift vs Squat).
* **Chiến lược tối ưu tài nguyên**:
  * Tinh giản không gian tìm kiếm siêu tham số bằng cách dừng mở rộng trên LSTM / BiLSTM.
  * Tập trung 100% GPU / compute budget vào hai họ kiến trúc tiên tiến nhất hiện nay cho Skeleton-based Action Recognition: **Spatial-Temporal Graph Convolutional Networks (ST-GCN)** và **Skeletal Transformer**.

---

## 2. Nâng Cấp Toàn Diện Kiến Trúc Skeletal Transformer

### 2.1. Mã Hóa Vị Trí Thời Gian Hiện Đại (Positional Encodings)
* **Vấn đề**: Sinusoidal Positional Encoding cố định (Vaswani et al.) không biểu diễn tốt cấu trúc chu kỳ động lực học (repetition cycles) của bài tập thể hình.
* **Giải pháp**:
  * **Rotary Position Embedding (RoPE)**: Tích hợp phép quay góc phức trực tiếp vào tích vô hướng giữa Query và Key ($q^T k$), duy trì tính tương đối về khoảng cách thời gian một cách tự nhiên.
  * **Learnable Positional Embedding**: Học ma trận $E_{pos} \in \mathbb{R}^{T \times D}$ theo phân phối bài tập thực tế để bắt trọn từng phase động tác (Eccentric / Isometric / Concentric).

### 2.2. Hiện Đại Hóa Khối Biến Đổi (Feed-Forward Network & Activation)
* Thay thế kích hoạt $\text{ReLU}$ truyền thống bằng $\text{GeLU}$ (Gaussian Error Linear Unit) hoặc $\text{SwiGLU}$, tạo ra độ dốc mượt mà (smooth gradient landscape), giảm hiện tượng "chết nơ-ron" (dead neurons) khi huấn luyện trên dữ liệu chuẩn hóa $[0, 1]$.
* Áp dụng **Pre-LayerNorm (Pre-LN)** thay vì Post-LN để bảo đảm độ ổn định gradient trong các lớp sâu, hạn chế triệt để hiện tượng nổ/biến mất đạo hàm.

### 2.3. Lịch Trình Tốc Độ Học Cải Tiến (Learning Rate Scheduling)
* **Cosine Annealing Learning Rate với Linear Warmup**:
  * $5 - 10$ epochs đầu tiên: Tăng $lr$ tuyến tính từ $10^{-6}$ lên $lr_{base} = 10^{-4}$ (hoặc $3 \times 10^{-4}$) giúp ổn định các trọng số attention ban đầu.
  * Các epochs tiếp theo: Hạ dần $lr$ theo đường cong Cosine về $lr_{min} = 10^{-6}$.

---

## 3. Nâng Cấp Chuyên Sâu Kiến Trúc ST-GCN

### 3.1. Adaptive Graph Convolutional Networks (AAGCN)
* **Hạn chế của ST-GCN tĩnh**: Ma trận kề $A_k$ chỉ dựa trên liên kết giải phẫu tự nhiên (ví dụ: cổ tay nối cùi chỏ). Tuy nhiên, trong động tác gym, sự phối hợp giữa hai cổ tay hoặc giữa tay và vai đối xứng là cốt lõi để nhận diện động tác đúng/sai.
* **Cơ chế đồ thị thích ứng 3 thành phần**:
  $$A = A_{topo} + B + C(X)$$
  * $A_{topo}$: Cấu trúc xương tự nhiên chuẩn MediaPipe 33 joints.
  * $B \in \mathbb{R}^{N \times N}$: Ma trận tham số học được toàn cục (Global learnable matrix), khởi tạo bằng 0 và tự học các liên kết chức năng gián tiếp.
  * $C(X) \in \mathbb{R}^{N \times N}$: Ma trận kề phụ thuộc mẫu động (Sample-dependent attention), tính toán qua tích vô hướng đặc trưng giữa các khớp trong từng frame.

### 3.2. Two-Stream Framework (Joint Stream + Bone Stream)
* **Luồng Khớp (Joint Stream)**: Đầu vào là tọa độ vị trí không gian của các khớp $X_{joint} = [x_j, y_j, z_j]$.
* **Luồng Xương (Bone Stream)**: Đầu vào là vector đoạn xương nối giữa hai khớp liền kề:
  $$e_{u, v} = X_{u} - X_{v}$$
  * Luồng xương bảo toàn tuyệt đối góc định hướng, độ duỗi và tốc độ biến thiên góc của các đoạn chi mà không phụ thuộc vào vị trí tuyệt đối của cơ thể trong không gian.
* **Hợp nhất (Late Fusion)**: Cộng xác suất hoặc ghép đặc trưng trước khi qua lớp phân loại cuối cùng.

---

## 4. Chiến Lược Regularization & Kiểm Soát Overfitting

| Kỹ Thuật | Tham Số Đề Xuất | Mục Tiêu & Cơ Chế Hoạt Động |
| :--- | :--- | :--- |
| **Label Smoothing** | $\epsilon = 0.1$ | Ngăn chặn mô hình quá tự tin vào nhãn 1-hot, làm mượt phân phối xác suất và tăng độ khái quát hóa |
| **Weight Decay** | $10^{-3} \rightarrow 10^{-4}$ | Phạt L2 chuẩn hóa trọng số, ức chế các trọng số quá lớn gây overfit |
| **Spatial Dropout** | $p = 0.3 - 0.4$ | Tắt ngẫu nhiên một số kênh đặc trưng khớp trong quá trình feed-forward |
| **Augmentation Ensemble** | `combined` | Kết hợp Bone Length Scaling ($\pm 10\%$), Temporal Frame Dropping, Jitter, Rotation |

---

## 5. Thực Nghiệm Phân Tích Siêu Tham Số Độ Dài Chuỗi (Sequence Length Ablation)

* **Thiết kế thực nghiệm**:
  * So sánh $sl=20$ (stride 10) vs $sl=32$ (stride 16).
* **Mục tiêu nghiên cứu**:
  * Đánh giá sự đánh đổi giữa độ phân giải thời gian và năng lực bao quát toàn bộ một chu kỳ rep (repetition cycle) trung bình ($1.5 - 2.5$ giây ở $30$ FPS).
  * Làm sáng tỏ giả thuyết: ST-GCN có hoạt động ổn định và chính xác hơn trên các cửa sổ ngắn $sl=20$ do giảm thiểu sự suy hao thông tin liên kết đồ thị qua quá nhiều bước thời gian hay không.

---

## 6. Chiến Lược Ensemble Đột Phá: Weighted Soft Voting

* **Vấn đề của Stacking Classifier**: Meta-model Logistic Regression rất dễ bị overfit trên tập Validation nhỏ hoặc bị thiên lệch nếu một vài mô hình cơ sở có độ phân tán xác suất khác nhau.
* **Giải pháp Weighted Soft Voting**:
  $$\hat{y} = \arg\max_c \sum_{m=1}^M w_m P_m(y=c | X)$$
  * Với ràng buộc $\sum_{m=1}^M w_m = 1, w_m \ge 0$.
  * Tối ưu hóa vector trọng số $w$ trực tiếp trên tập Validation (sử dụng thuật toán tối ưu ràng buộc như SLSQP hoặc Nelder-Mead trên hàm mục tiêu Log-Loss / Negative Accuracy).
  * Đảm bảo tính suy diễn nhanh, không cần huấn luyện thêm meta-classifier và bảo toàn phân phối xác suất gốc.
