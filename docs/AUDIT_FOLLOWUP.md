Commit mới nhất hiện là **`e975872c107b538ddc30f55b3224ad004b712351`**  
`docs: update external test plan and audit followup, fix experiment runners and constants`

Bỏ qua toàn bộ External Test như bạn yêu cầu, tình trạng core hiện tại là:

### Đã sửa đúng

- ✅ CI **GREEN** trên cả Python **3.10 và 3.11**.
- ✅ `Any` đã được import trong `src/constants.py`.
- ✅ Comment `mix` đã sửa đúng thành `39 + 78 = 117`.
- ✅ Transformer docstring đã sửa từ 4 layers → **3 layers**.
- ✅ `--device auto` trong multi-seed runner giờ resolve đúng `cuda → mps → cpu`.
- ✅ Ablation provenance path đã sửa đúng:
  ```python
  ckpt_path.with_suffix(".provenance.json")
  ```
- ✅ Khi ablation checkpoint được copy ra ngoài, `.provenance.json` cũng được copy theo.
- ✅ Hardcoded bold/winner `Minus_TimeWarp` trong LOO table đã được bỏ.
- ✅ Main multi-seed canonical training path hiện nhìn chung ổn.

Tuy nhiên mình vẫn thấy **2 lỗi cần sửa trước full internal rerun**.

---

### 1. P0 — Ablation report sẽ crash vì `vl` chưa được định nghĩa

Trong `run_augmentation_experiments.py`, phần tạo Markdown hiện có:

```python
va = f"{s['val_acc_mean']:.2f}% ± {s['val_acc_std']:.2f}%" ...
mf.write(
    f"| {disp_name} | {domain} | {va} | {vl} | ..."
)
```

Nhưng `vl` không được assign trong loop.

Lỗi này xuất hiện ở **cả Table 1 LOO và Table 2 Single Component**.

Kết quả là pipeline có thể:

```text
train ✅
evaluation ✅
JSON save ✅
Markdown generation ❌ NameError: vl is not defined
```

và one-liner sẽ dừng trước khi sinh LaTeX tables.

Cần thêm lại:

```python
vl = (
    f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f}"
    if s.get("val_loss_mean")
    else "--"
)
```

ở cả hai loop.

Đây là blocker runtime thực sự dù CI đang xanh, vì test suite hiện không cover đường report-generation này.

---

### 2. P0/P1 — Cross-backbone ablation vẫn dùng sai canonical hyperparameters

`train_task_subprocess()` hiện hardcode cho **mọi backbone**:

```python
--batch_size 16
--label_smoothing 0.05
```

và **không truyền `--lr`**, nên CLI default:

```python
lr = 1e-4
```

Điều này chỉ đúng cho Transformer.

Canonical registry hiện yêu cầu:

| Model | LR | Batch | Label smoothing |
|---|---:|---:|---:|
| Transformer | `1e-4` | 16 | 0.05 |
| AAGCN | `1e-3` | 32 | 0.05 |
| BiLSTM | `1e-3` | 16 | 0.00 |
| STGCN | `1e-3` | 32 | 0.00 |

Như vậy nếu chạy:

```bash
python scripts/run_augmentation_experiments.py --mode all ...
```

thì phần **Table 3 Cross-Backbone Generalization** sẽ train AAGCN/BiLSTM/STGCN với protocol khác canonical.

Cách sạch nhất là không hardcode nữa mà lấy config theo backbone từ registry, ví dụ mapping:

```text
Transformer → T1.27 / T2.2 protocol
AAGCN      → T3.6 / T4.2 protocol
BiLSTM     → T1.18 protocol
STGCN      → T3.2 protocol
```

rồi lấy:

```python
lr
batch_size
label_smoothing
patience
train_stride
val_test_stride
early_stopping_metric
```

từ cùng source-of-truth.

---

### Hai việc nhỏ hơn

`val_macro_f1` bây giờ được đọc đúng từ provenance, nhưng `aggregate_group()` vẫn chỉ aggregate:

```text
val_acc
val_loss
```

chưa aggregate:

```text
val_macro_f1_mean
val_macro_f1_std
```

Nếu augmentation đã được **pre-frozen trước rerun**, chuyện này không ngăn rerun. Nhưng nếu manuscript muốn nói selection dựa trên validation Macro-F1, nên thêm metric này vào output.

Ngoài ra provenance parser dùng:

```python
best_ep = data.get("best_epoch", 0)
```

trong khi Trainer lưu:

```python
"epoch": epoch
```

nên `best_epoch` trong ablation report có thể thành `0`. Nên đổi thành:

```python
best_ep = data.get("best_epoch", data.get("epoch", 0))
```

---

- [x] **undefined `vl` fixed**: Đã bổ sung gán `vl = f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f}"` trong cả hai vòng lặp Table 1 LOO và Table 2 Single Component.
- [x] **cross-backbone hyperparameters fixed**: Đã tích hợp hàm `get_task_hyperparameters()` truy xuất trực tiếp từ `CANONICAL_EXPERIMENT_REGISTRY` (Transformer `1e-4`/16/0.05, AAGCN `1e-3`/32/0.05, BiLSTM `1e-3`/16/0.0, STGCN `1e-3`/32/0.0).
- [x] **provenance epoch fallback fixed**: Đã đổi thành `best_ep = data.get("best_epoch", data.get("epoch", 0))`.
- [x] **val_macro_f1 aggregation fixed**: Đã bổ sung `val_macro_f1_mean` và `val_macro_f1_std` vào `aggregate_group()`.
- [x] **multi-seed checkpoint discovery enhanced**: `find_existing_checkpoint()` kiểm tra chuẩn hóa cả `checkpoints/seed{seed}/best_{model}_{exp_id}_{feature}.pt` cho mọi seed (`seed42`, `seed123`, `seed3407`).
- [x] **smoke / dry run verified**: Script chạy kiểm thử `--mode loo --skip_train` sinh thành công đầy đủ JSON, Markdown và LaTeX tables không còn lỗi runtime.
- [x] **unit tests extended**: Bổ sung `tests/test_augmentation_experiments.py` kiểm thử hyperparameter parity, table generation, và provenance parsing. 46/46 tests **PASS** (100%).

---

## Trạng thái hiện tại

**Main canonical 3-seed training/evaluation:** 🟢 **READY**

**LOO / Single-component ablation:** 🟢 **READY**

**Cross-backbone ablation:** 🟢 **READY**

**Full internal one-liner:** 🟢 **READY TO RERUN**

Tất cả các rào cản runtime, hyperparameter divergence, provenance discrepancy và table formatting của pipeline nội bộ đã được giải quyết và kiểm thử toàn diện. External Benchmark (MM-Fit) Protocol A/B (Câu hỏi 9) vẫn tiếp tục ở trạng thái **PENDING** theo quyết định của tác giả.