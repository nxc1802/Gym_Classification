# Pre-Publication Quality & Rigor Checklist

## 🔴 MUST FIX (Completed: 7/7)

- [x] **1. Audit toàn bộ numerical consistency**
  - **Verified Counts**:
    - **1,024** total video recordings (896 internet-harvested + 128 author-recorded).
    - **1,108** segmented repetitions / action clips.
    - **Split counts**: 580 train / 208 val / 236 test videos (100% video-independent partition, 0% video/segment overlap).
    - **Window counts**: 13,136 train ($S=16$, 50% overlap) / 2,075 val ($S=32$) / 2,743 test windows ($S=32$).
    - **233** valid test videos evaluated after 32-frame boundary thresholding (3 ultra-short videos $<32$ frames excluded from test consensus).
  - **Cross-file Consistency**: All occurrences across `paper/paper.tex`, `Final_dataset_metadata.csv`, `outputs/EXPERIMENT_RESULTS.md`, and tables are strictly identical.

- [x] **2. Bỏ “state-of-the-art” nếu chưa benchmark đủ SOTA**
  - **Action**: Removed all generic "state-of-the-art" / SOTA claims throughout the paper (Abstract, Section 1, Section 5, Section 6, Section 7).
  - **Replacement**: Replaced with precise academic language: *"strongest performance among evaluated methods in our benchmark"* or *"consistently outperforms evaluated single-stream and dual-stream backbones"*.
  - **Audit**: Grep verification confirms 0 instances of ungrounded "state-of-the-art" claims in `paper/paper.tex`.

- [x] **3. Bỏ “optimal 117-d”**
  - **Action**: Replaced all normative claims of "optimal" feature representation.
  - **Replacement**: Replaced with *"proposed 117-dimensional Biomechanical Compound Representation"* or *"empirically selected 117-d representation"*.
  - **Remaining "optimal"**: Strictly confined to mathematical optimization contexts (e.g., *"optimal SLSQP voting weights $w^*$"*) and clinical posture descriptions (*"sub-optimal muscular recruitment"* / *"sub-optimal knee travel"*).

- [x] **4. Sửa “strict privacy guarantees”**
  - **Action**: Corrected inaccurate claim that MediaPipe guarantees absolute privacy (since pose estimation still ingests raw RGB camera frames).
  - **Replacement**: Formulated as *"privacy-aware workflow that reduces raw RGB retention and prevents continuous cloud video streaming"*, explaining that edge-native execution processes frames ephemerally in volatile RAM and immediately discards RGB pixels, transmitting only non-identifying coordinate trajectories $(x, y, z)$.

- [x] **5. Sửa latency wording (Classifier Latency ≠ End-to-End Latency)**
  - **Action**: Explicitly separated classifier model inference latency from monocular pose extraction latency across Abstract, Section 1, Section 6.4, Section 6.5, and Conclusion.
  - **Metrics**:
    - **Classifier inference latency**: $1.01\text{ ms}$ per 32-frame window for SkelGym-Full ($0.34\text{ ms}$ for SkelGym-Lite) on Apple MPS, $0.54\text{ ms}$ on CUDA.
    - **Pose extraction latency**: $\approx 8–15\text{ ms}$ per frame with MediaPipe Pose on mobile chipsets.
    - **End-to-end pipeline latency**: $\approx 9–16\text{ ms}$, operating well within the $33.3\text{ ms}$ real-time budget ($30\text{ FPS}$).
  - **Bugfix**: Corrected latency typo in Section 6.4 for SkelGym-Lite from $1.01\text{ ms}$ to $0.34\text{ ms/window}$ on MPS ($3.20\text{ ms}$ on CPU).

- [x] **6. Clarify parameter budget**
  - **Action**: Made parameter constraints explicit across Abstract, Introduction, Section 5.1, and Table captions.
  - **Footprint**: Backbones strictly constrained to $365\text{K}–399\text{K}$ parameters ($\approx 350\text{K} \pm 15\%$):
    - Transformer Mix: $399\text{K}$
    - AAGCN: $378\text{K}$ per stream
    - BiLSTM: $367\text{K}$
    - LSTM: $378\text{K}$
    - ST-GCN: $365\text{K}$
    - SkelGym-Lite Ensemble: $777\text{K}$
    - SkelGym-Full Ensemble: $1.91\text{M}$

- [x] **7. Kiểm tra toàn bộ claims về external benchmark**
  - **Action**: Explicitly qualified all external benchmark statements to reflect the actual evaluated scope.
  - **Scope**: Evaluated on **54 held-out test videos** ($N=529$ temporal windows) encompassing the **4 exact overlapping Strength & Conditioning (S&C) exercises** (*squat, deadlift, barbell biceps curl, lateral raise*) from Deyzel et al. (SU-EMD).
  - **Dual Evaluation Protocol**: Reported both closed-set (re-normalized over the 4 S&C classes, matching Deyzel) and open-set (full 22-class probability distribution) results.

---

## 🟠 SHOULD FIX (Completed: 6/6)

- [x] **8. Bổ sung limitation về MediaPipe failure**
  - **Action**: Added a dedicated paragraph in Section 7 (*Limitations of Monocular Pose Estimation and Failure Modes*):
    - Barbell plate occlusions (e.g., standard $450\text{ mm}$ Olympic bumper plates occluding wrists and torso during deadlifts/rows/bench).
    - Spotter occlusions behind benches.
    - Camera framing limits / out-of-frame limb clipping (feet below frame in RDLs, hands above frame in pull-ups).
    - Monocular depth ambiguity and optical axis $z$-coordinate jitter relative to mid-hip anchor.
    - Proposed future solutions: learned kinematic Kalman smoothing and multi-view consensus.

- [x] **9. Bổ sung limitation về class imbalance**
  - **Action**: Added a dedicated paragraph in Section 7 (*Dataset Class Imbalance and Long-Tail Kinetics*):
    - Documented real-world fitness imbalance: 84 videos for barbell biceps curl / bench press vs. 9 videos for plank (0.9% of dataset).
    - Outlined mitigation strategies employed (Macro F1 validation monitoring, SLSQP loss calibration, label smoothing $\alpha=0.05$).
    - Noted wider confidence intervals on tail classes and highlighted standardized multi-angle video collection as a priority for future benchmark versions.

- [x] **10. Làm rõ 1-shot protocol**
  - **Action**: Fully articulated the 1-shot transfer learning protocol in Section 6.5.1:
    - 100 stochastic trials on the 54 held-out S&C test videos.
    - $K=1$ random exemplar video sampled per class as support (4 support videos total across the 4 classes).
    - Remaining 50 test videos served as query samples.
    - Nearest-neighbor classification using cosine similarity over temporal mean-aggregated feature representations.
    - Results: Transformer Mix ($97.32\% \pm 1.96\%$, 95% CI: $[94.00\%, 98.00\%]$), AAGCN Bone ($92.02\% \pm 7.24\%$), SkelGym-Full ($95.34\% \pm 3.89\%$).

- [x] **11. Kiểm tra toàn bộ references**
  - **Audit**: Python automated citation validator confirms:
    - **43 bibitems defined**, **43 bibitems cited**.
    - **0 missing bibitems** (no undefined `\cite{...}`).
    - **0 unused bibitems** (added `\cite{wolpert1992stacking}` to Section 4.4 for stacked generalization / SLSQP ensemble fusion).

- [x] **12. Kiểm tra numbering Table/Figure/Appendix**
  - **Audit**: Python automated label/ref validator confirms:
    - **13 Tables** defined (`tab:table1` to `tab:table13`), all 13 referenced dynamically in text with `Table~\ref{...}`.
    - **5 Figures** defined (`fig:dataset_samples`, `fig:triplet_vs_pairwise`, `fig:pose_extraction`, `fig:pipeline`, `fig:confmatr`), all dynamically linked with `Fig.~\ref{...}`.
    - 0 broken table references, 0 broken figure references.

- [x] **13. Giảm các câu mang tính marketing → Academic tone**
  - **Edits**:
    - `"unlocks exceptional synergy"` $\to$ `"provides strong complementary synergy"`.
    - `"dramatically reduces validation loss"` $\to$ `"substantially reduces validation loss"`.
    - `"exceptionally effective probability calibrator"` $\to$ `"effective regularizer and probability calibrator"`.
    - `"superior test generalization"` $\to$ `"improved test generalization"`.
    - `"achieves an exceptional 79.83% video accuracy"` $\to$ `"achieves a notable 79.83% video accuracy"`.
    - `"offers an outstanding operational compromise"` $\to$ `"offers a practical operational compromise"`.
    - `"as an exceptionally promising frontier for future research"` $\to$ `"as a promising direction for future research"`.
    - Replaced Unicode en-dashes (`–`) with standard LaTeX dashes (`--`) to resolve font rendering warnings.

---

## 🟢 Compilation & Packaging Status
- **LaTeX Engine**: Tectonic (`/opt/homebrew/bin/tectonic`) compiles `paper/paper.tex` cleanly with exit code 0.
- **Output PDF**: `paper/paper.pdf` (35 pages, 1.68 MB).
- **Overleaf Package**: `paper/paper_overleaf.zip` and `./paper_overleaf.zip` synchronized with all style files, images, and updated `paper.tex`.