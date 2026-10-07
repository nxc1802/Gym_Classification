Được. Từ đây mình sẽ coi **experiment đã freeze** và chuyển trọng tâm sang paper, không tiếp tục “đẻ” thêm thí nghiệm trừ khi phát hiện một lỗi khoa học thực sự.

Mình đã rà manuscript hiện tại và có một kết luận rõ: **`paper/paper.tex` nên được xem là scientific master**, còn `paper_eswa.tex` / `paper_llncs.tex` nên sync sau. `GYM_Publication/manuscript.tex` quá cũ để làm nền chính.

Hiện `paper/paper.tex` chưa chỉ cần “polish”; nó cần một lượt **scientific synchronization** lớn. Ngay Abstract và Introduction vẫn còn:

- Mix v2 = scale-normalized relative 3D → phải đổi thành **39-d metric WORLD 3D + 24-d kinematic angles**.
- SkelGym-Aug = 4 operators → phải đổi thành **Mirror + Yaw**.
- SLSQP fusion → phải đổi thành **Uniform Soft cho SkelGym-Full**, Accuracy-Weighted cho Lite.
- Transformer `3L, 8H, ~399K` → hiện canonical là **3 layers, 4 heads, 301K**.
- kết quả `69.74% / 79.11%` → final Full hiện là **76.19 ± 0.16% window**, **85.27 ± 0.41% video**, F1 **0.7537 / 0.8386**.
- latency `0.42–4.33 ms CPU` và FLOPs cũ cũng cần đồng bộ SOT mới.
- Conclusion vẫn nói Mirror + Yaw + Scale + Jitter.
- Related Work / Ensemble section vẫn xây narrative quanh SLSQP.
- Fig. 1 vẫn có node “SLSQP Calibration” và các số cũ.

Quan trọng hơn, paper hiện có một số section phân tích sâu như Deyzel subset, biomechanical confusion/error analysis với các con số từ hệ thống cũ. Với các phần này mình sẽ áp nguyên tắc rất nghiêm:

> **Nếu số liệu không còn trace được tới canonical artifact hiện tại, hoặc không được regenerate theo final system, thì không giữ claim định lượng đó trong main paper.**

Không nên để một paper có Table 7 dùng final Uniform Soft nhưng Discussion lại phân tích một ensemble SLSQP cũ.

Mình đề xuất sửa theo thứ tự này:

1. **Abstract + Contributions + Fig. 1**  
   Chốt scientific story mới của paper.
2. **Methodology**  
   WORLD Mix v2 → Mirror+Yaw → sequence/graph streams → fusion protocol → video consensus.
3. **Experimental protocol**  
   Làm rõ selection chỉ dựa trên Validation Window Macro-F1/Accuracy; video metrics là downstream/final evaluation.
4. **Results tables 1–10**  
   Đồng bộ trực tiếp từ `RESULTS_FINAL.md`/canonical SOT.
5. **Results narrative + statistical claims**  
   Không dùng test để giải thích lựa chọn method.
6. **Discussion / limitations / biomechanical analysis**  
   Giữ những gì được evidence support, bỏ mechanistic overclaim.
7. **Conclusion + Abstract lần cuối**.
8. Cuối cùng mới làm **language compression, redundancy, venue formatting**.

Scientific story mới của paper theo mình nên rất đơn giản:

```text
Metric WORLD skeleton
        ↓
Biomechanical Mix v2
WORLD XYZ (39) + kinematic angles (24)
        ↓
Mirror + Yaw SkelGym-Aug
(selected only by validation-window evidence)
        ↓
Transformer sequence stream
+
4 complementary AAGCN streams
        ↓
simple late probability fusion
        ↓
SkelGym-Full = Uniform Soft
        ↓
video consensus
```

Điểm hay là paper giờ **sạch hơn bản cũ rất nhiều**: không còn SLSQP, không còn Stacking, không còn phải biện minh cho Scale/Jitter/TimeWarp trong proposed augmentation. Contribution chính trở nên dễ hiểu hơn.

Mình sẽ coi các số sau là “paper anchors” từ giờ:

- **SkelGym-Aug:** Mirror + Yaw
- Val Win: **81.20 ± 0.72%, F1 0.8119 ± 0.0072**
- Transformer + Aug Test: **73.14 ± 0.90%, F1 0.7253**
- Four-stream AAGCN Test: **74.50 ± 0.65%, F1 0.7341**
- SkelGym-Lite: **73.74 ± 0.63% / 82.12 ± 0.40% video**
- **SkelGym-Full Uniform Soft:** **76.19 ± 0.16% Win Acc, 0.7537 ± 0.0014 Win F1, 85.27 ± 0.41% Video Acc, 0.8386 ± 0.0075 Video F1**
- Final model size: **1.81M**
- canonical seeds: **42, 123, 3407**

Bước tiếp theo hợp lý nhất là mình bắt đầu từ **Abstract + 4 Contributions + mô tả Fig. 1**, vì ba phần đó sẽ khóa narrative cho toàn bộ phần còn lại. Sau khi chúng đúng, Methodology và Results sẽ dễ sửa hơn rất nhiều.