# FINAL RETRAINING, HF RECOVERY & PAPER SUBMISSION PLAN

**Project:** Gym_Classification  
**Scope:** Checkpoint recovery → missing-model retraining → artifact archival → Hugging Face sync → `RESULTS_FINAL.md` update → minimal-diff LaTeX update → full submission audit  
**Status:** **PLAN ONLY — NO TRAINING / NO PAPER EDITING STARTED**

## 0. Core Principles

1. Audit first, retrain only what is truly missing.
2. Never substitute a legacy or “similar” checkpoint for the exact paper configuration.
3. All stochastic experiments remain multi-seed: `42`, `123`, `3407`.
4. Every newly trained model must produce complete local artifacts before any paper update.
5. Every uploaded Hugging Face artifact must be verified after upload.
6. `RESULTS_FINAL.md` is updated before LaTeX.
7. LaTeX edits must follow strict minimal-diff strategy.
8. If retraining materially changes results, rankings, statistical conclusions, or discussion claims, stop and report before rewriting prose.
9. Never store Marimo/HF tokens inside source files, Markdown, JSON, logs, checkpoints, or git history.

---

# Phase 0 — Freeze Current Baseline

Before changing anything:

- Record current Git commit SHA.
- Record current Hugging Face repo revision.
- Record dataset metadata hash.
- Record environment / requirements.
- Record current canonical result artifacts.
- Determine the actual source-of-truth LaTeX file.

Current repository candidates include:

- `paper/paper_eswa.tex`
- `paper/paper.tex`
- `preprint/main.tex`
- Overleaf variants

There is currently no guaranteed `final.tex` source-of-truth by filename alone.

### Deliverable

Create a baseline manifest containing:

```text
git_commit
hf_revision
dataset_metadata_sha256
environment
canonical_result_files
latex_source_of_truth
seeds = [42, 123, 3407]
```

**Gate:** no training and no LaTeX edits before this baseline is frozen.

---

# Phase 1 — Paper-to-Checkpoint Inventory

Scan:

- source-of-truth manuscript
- all tables
- supplementary / appendix
- `outputs/RESULTS_FINAL.md`
- canonical JSON artifacts
- experiment configs
- scripts
- checkpoint manifests

Build a complete model/configuration inventory.

For every trainable paper entry record:

| Field | Required |
|---|---|
| Paper table / section | Yes |
| Experiment ID | Yes |
| Architecture | Yes |
| Feature | Yes |
| Feature dimension | Yes |
| Augmentation | Yes |
| Seed | Yes |
| Expected checkpoint name | Yes |
| Expected HF path | Yes |
| Local artifact path | Yes |
| Existing metrics source | Yes |

Separate trainable models from derived ensembles/fusion. Derived models are not retrained as standalone networks; they are recomputed from constituent checkpoint predictions.

---

# Phase 2 — Hugging Face Checkpoint Audit

For every expected trainable model × seed:

1. Search current HF HEAD.
2. Search historical HF revisions if missing.
3. Search GitHub history / manifests.
4. Search archived checkpoint folders.
5. Search prediction `.npz`.
6. Search result JSON.
7. Search provenance JSON.

Classification:

```text
FOUND_CURRENT_HF
FOUND_HISTORICAL_HF
FOUND_LOCAL_ONLY
MISSING
PROVENANCE_MISMATCH
LEGACY_NOT_USABLE
```

A checkpoint is valid only if all of these match:

```text
architecture
feature
feature_dimension
augmentation
seed
model_variant
paper experiment/config
```

Do not treat legacy `mix` 117-d as equivalent to `mix_v2` 63-d.

Create:

```text
artifacts/audit/checkpoint_inventory.md
artifacts/audit/checkpoint_inventory.json
```

with model/config/seed/HF/artifact status.

---

# Phase 3 — Missing-Model Retraining

Only configurations confirmed `MISSING` or unrecoverably corrupted are retrained.

## 3.1 Marimo Server Execution

Training runs on the Marimo server using the Marimo token.

Use the token only via environment/secret manager, e.g.:

```bash
export MARIMO_TOKEN="..."
export HF_TOKEN="..."
```

Never hardcode or log either token.

## 3.2 Parallel Training Strategy

Preferred strategy: **3 seeds of the same configuration in parallel**:

```text
Config A
 ├── seed42
 ├── seed123
 └── seed3407
```

This is preferred because one configuration is completed and aggregated in a single batch.

Alternative strategy: **3 independent models at once** when resource utilization is better:

```text
Model A / seed X
Model B / seed X
Model C / seed X
```

Maximum planned concurrency is 3 jobs unless GPU/RAM checks require fewer.

Before each batch:

1. inspect free GPU memory;
2. inspect RAM;
3. inspect CPU load;
4. estimate model footprint;
5. choose safe concurrency.

---

# Phase 4 — Monitoring & Automatic Job Chaining

Use **event-driven job completion with a 30-minute heartbeat fallback**.

### Primary mechanism

Each job writes a machine-readable status:

```text
RUNNING
SUCCESS
FAILED
```

When all jobs in the current batch finish:

1. validate checkpoint/artifacts;
2. compute metrics;
3. archive/upload;
4. launch the next pending batch immediately.

### 30-minute heartbeat

Every 30 minutes verify:

- process alive/dead;
- latest epoch;
- checkpoint timestamp;
- GPU utilization;
- log growth;
- disk usage;
- stalled/failed processes.

If one seed fails, rerun only that seed; do not retrain successful seeds.

---

# Phase 5 — Mandatory Artifacts Per Model/Seed

Every newly trained run must save:

```text
best_checkpoint.pt
last_checkpoint.pt            # if supported
provenance.json
metrics.json
training_history.json
predictions.npz
per_class_metrics.json
per_class_metrics.csv
confusion_matrix.npy/png
training.log
sha256.txt
```

## Required metrics

### Validation — Window
- val_loss
- val_win_accuracy
- val_win_precision_macro
- val_win_recall_macro
- val_win_f1_macro

### Validation — Video
- val_vid_accuracy
- val_vid_precision_macro
- val_vid_recall_macro
- val_vid_f1_macro

### Test — Window
- test_loss if supported
- test_win_accuracy
- test_win_precision_macro
- test_win_recall_macro
- test_win_f1_macro

### Test — Video
- test_vid_accuracy
- test_vid_precision_macro
- test_vid_recall_macro
- test_vid_f1_macro

Also save:

```text
support
number_of_windows
number_of_videos
best_epoch
train_time
seed
checkpoint_path
checkpoint_sha256
git_commit
```

`predictions.npz` should contain at least:

```text
val_probs
val_targets
val_video_ids
test_probs
test_targets
test_video_ids
```

Per-class metrics must contain Win/Vid Precision, Recall, F1 and Support for both Val and Test.

---

# Phase 6 — Multi-Seed Aggregation

After seeds `42`, `123`, `3407` complete:

- compute Mean ± SD from raw per-seed values;
- compute Val Win / Val Vid / Test Win / Test Vid;
- compute per-class Mean ± SD;
- never aggregate from already rounded Markdown values.

Create canonical multiseed JSON/CSV artifacts.

---

# Phase 7 — Recompute Derived/Ensemble Results

If any constituent backbone is retrained, recompute all dependent:

- Two-stream fusion
- Four-stream fusion
- SkelGym-Lite
- SkelGym-Full
- voting results
- statistical comparisons
- bootstrap confidence intervals

Do not retain old ensemble/statistical metrics when underlying predictions changed.

---

# Phase 8 — Local Archive + HF Upload

Store canonical artifacts locally with a consistent model registry containing:

```text
model
config
seed
checkpoint
sha256
metrics
prediction_artifact
git_commit
training_date
HF_path
HF_revision
```

Upload to HF:

- checkpoint
- provenance
- metrics
- predictions
- per-class results
- config

After upload:

1. list files;
2. download back;
3. verify SHA256;
4. verify JSON/NPZ loading;
5. test checkpoint loading.

Only mark `HF_VERIFIED` when local and downloaded hashes match.

---

# Phase 9 — Canonical Results Freeze

After all missing models are restored and dependent results recomputed, freeze one canonical source containing:

```text
per-seed metrics
multi-seed aggregate
Val/Test
Win/Vid
per-class
checkpoint provenance
HF location
```

No paper editing before this freeze.

---

# Phase 10 — Update `outputs/RESULTS_FINAL.md`

Update `RESULTS_FINAL.md` from canonical artifacts, preferably generated/validated programmatically.

Audit:

- all table values;
- mean ± SD;
- Win/Vid;
- Val/Test;
- model names;
- feature dimensions;
- parameter counts;
- winners/rankings;
- claims;
- statistical results.

---

# Phase 11 — Minimal-Diff LaTeX Update

## Main rule

Use the smallest possible edit.

Do not:

- reformat the whole file;
- run a global formatter;
- alter unrelated whitespace;
- rename labels unnecessarily;
- reorder sections;
- rewrite unaffected paragraphs;
- modify citation keys without need;
- touch unrelated figures.

If only a number changed, change only that number.  
If only a row changed, change only that row.  
If a caption needs one correction, change only that phrase.

## Material-change gate

Before rewriting discussion/conclusion, compare old vs new results.

A change is material if any of these occurs:

1. main model ranking changes;
2. validation/test winner changes;
3. headline metric changes substantially;
4. statistical significance flips;
5. CI changes interpretation;
6. augmentation conclusion changes;
7. an existing discussion claim is no longer supported;
8. abstract/conclusion claim becomes false;
9. external benchmark interpretation changes.

Operational warning threshold:

```text
absolute headline metric change >= 1.0 percentage point
```

This is only a review trigger.

## Required report before rewrite

If material change occurs, create:

```text
artifacts/audit/latex_material_change_report.md
```

For each affected passage report:

```text
File:
Section:
Original claim:
Old value:
New value:
Why original claim is no longer valid:
Minimum required rewrite:
Downstream sections affected:
```

No silent rewrite.

---

# Phase 12 — Paper-Wide Audit

Perform a full audit of:

## Dataset
- class count
- video count
- segment count
- train/val/test split
- test windows/videos
- class support
- feature dimensions
- sequence length
- stride
- normalization
- zero-frame handling

## Numerical consistency
- percentages
- F1 / precision / recall
- N
- parameters
- FLOPs
- p-values
- confidence intervals
- seed counts

## Citations / references
- citation key exists
- bibliography entry exists
- author/title/year/venue correct
- DOI/URL correct when relevant
- cited source supports the sentence
- no broken references
- no accidental duplicate entries

## Methodological consistency
Check consistent use of:

```text
mix_v2
63-d
World 3D
Mirror + Yaw
SkelGym-Aug
Dual-Branch Transformer
42/123/3407
sequence length
stride
video consensus
validation-driven selection
```

## Statistics
If predictions changed, rerun affected:

```text
McNemar
Wilcoxon
paired t-test
Holm-Bonferroni
FDR
bootstrap CI
```

## LaTeX technical audit
Compile and check:

```text
undefined references
undefined citations
duplicate labels
table overflow
figure paths
missing assets
column mismatch
broken equations
bibliography
submission-relevant warnings
```

---

# Phase 13 — Final Submission Report

Create:

```text
artifacts/audit/final_submission_audit.md
```

with:

```text
Checkpoint completeness
HF reproducibility
Metric consistency
Dataset consistency
Statistical validity
Citation/reference audit
LaTeX compile audit
Material-change report
Remaining known limitations
Submission readiness
```

Final status:

```text
READY_TO_SUBMIT
READY_WITH_MINOR_WARNINGS
NOT_READY
```

---

# Execution Order

```text
1. Freeze baseline
2. Determine source-of-truth LaTeX
3. Build paper model inventory
4. Audit HF current + historical revisions
5. Produce missing-checkpoint list
6. Approval gate
7. Launch missing-model retraining on Marimo
8. Run 3 seeds/config in parallel where possible
9. Event-driven completion + 30-min heartbeat
10. Generate detailed Val/Test × Win/Vid artifacts
11. Generate per-class + predictions
12. Aggregate 3 seeds
13. Recompute dependent ensembles/statistics
14. Archive locally
15. Upload to HF
16. Download-back + SHA verification
17. Freeze canonical results
18. Update RESULTS_FINAL.md
19. Compare old/new manuscript numbers
20. Minimal-diff LaTeX update
21. Material-change report before substantial rewrite
22. Dataset/citation/reference/statistical audit
23. Compile
24. Final submission audit
```

---

# Approval Gate Before Training

Before any retraining, report:

```text
TOTAL TRAINABLE CONFIGURATIONS REFERENCED:
...

TOTAL EXPECTED CHECKPOINTS:
...

FOUND CURRENT HF:
...

FOUND HISTORICAL HF:
...

LOCAL ONLY:
...

MISSING:
...

PROVENANCE MISMATCH:
...

RETRAIN REQUIRED:
1. ...
2. ...
3. ...

PLANNED PARALLEL BATCHES:
Batch 1: ...
Batch 2: ...

MARIMO EXECUTION:
Configured / Not configured

PAPER SOURCE OF TRUTH:
...

NO TRAINING STARTED YET
NO LATEX EDITS STARTED YET
```

Training starts only after this inventory is reviewed.

---

# Non-Negotiable Rules

- No fabricated metrics.
- No legacy checkpoint substitution.
- No single-seed result labelled multi-seed.
- No paper update before canonical results freeze.
- No stale ensemble/statistical result after constituent predictions change.
- No token stored in repo.
- No HF upload considered complete until download-back verification passes.
- No broad LaTeX rewrite when a local replacement is sufficient.
- No silent discussion rewrite after material numerical change.
- Every final paper number must be traceable to a reproducible artifact.
