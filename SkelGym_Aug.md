### Plan cập nhật cho SkelGym-Aug

1. **Chỉnh wording**

   * Bỏ mô tả/giải thích quá chi tiết từng augmentation.
   * Chỉ mô tả Mirror, Yaw, Scale, TimeWarp, Jitter là **established/basic skeleton augmentation techniques**.
   * Không claim novelty cho từng operator.
   * Viết SkelGym-Aug như **một task-oriented augmentation configuration**.

2. **Chỉnh Contribution**

   * Không xem “novel augmentation methods” là contribution.
   * Chuyển contribution sang **empirical/task-oriented augmentation strategy + systematic ablation/validation**.
   * Giữ claim ở mức vừa phải, tránh overclaim methodological novelty.

3. **Experiments — 3 seeds**

   * **No Aug vs SkelGym-Aug** trên **toàn bộ các model/backbone chính**.
   * Report **Mean ± Std**.
   * **Ablation trên 1 representative model**:

     * Full SkelGym-Aug
     * −Mirror
     * −Yaw
     * −Scale
     * −TimeWarp
     * −Jitter
   * Không cần Individual Augmentation.
   * Không cần class × augmentation analysis.

4. **Discussion**

   * SkelGym-Aug không đóng góp augmentation operator mới.
   * Giá trị nằm ở **việc kết hợp các augmentation đã biết thành một configuration phù hợp với Gym skeleton recognition**.
   * Phân tích Full vs No Aug để chứng minh overall effectiveness.
   * Phân tích Leave-One-Out để xác định component nào cần thiết cho final configuration.
   * Không diễn giải quá mức thành causal/novel contribution của từng augmentation.

### Pipeline cuối

$$
\boxed{
\text{Wording/Contribution}
\rightarrow
\text{3-seed No Aug vs SkelGym-Aug}
\rightarrow
\text{3-seed Leave-One-Out}
\rightarrow
\text{Discussion}
}
$$

**Không thêm augmentation mới, không thêm Generic Aug, không làm class-wise augmentation analysis.**

Đây là scope tôi sẽ giữ cho paper hiện tại.
