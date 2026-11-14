# FULL RETRAIN, HF REBUILD & FINAL PAPER SUBMISSION PLAN

**Project:** Gym_Classification  
**Final paper source of truth:** `/final.tex` at repository root  
**Seeds:** `42, 123, 3407`  
**Status:** PLAN ONLY — NO HF CLEANUP / NO TRAINING / NO FINAL.TEX EDITS STARTED

---

# 0. Decisions

1. The only Final Paper to edit is `/final.tex` in the repository root.
2. Remove the old checkpoint matching/verification workflow. Historical HF match rate is too low to justify reuse.
3. Full retrain from scratch all trainable model/configurations required by `final.tex` and the final benchmark.
4. Before retraining, clear the HF model repository `Cuong2004/gym-exercise-classification`.
5. Never touch the protected dataset repository `Cuong2004/gym-exercise-landmarks`.
6. Every retrained model must be archived with checkpoint + metrics + predictions + per-class + provenance so normal paper work never needs training or inference again.
7. Update `outputs/RESULTS_FINAL.md` before touching `final.tex`.
8. `final.tex` must be edited with strict minimal diff.
9. If new results materially change a claim/discussion/conclusion, report the impact before rewriting prose.
10. Never store Marimo/HF tokens in source code, Markdown, logs, JSON, checkpoints, or git history.

---

# Phase 0 — Pull Code & Freeze Baseline

Before any destructive action or training:

```bash
git checkout main
git pull origin main
```

Record current Git SHA, dataset metadata hash, environment, current `RESULTS_FINAL.md`, current root `final.tex`, experiment configs/scripts, and current HF model repo revision for archival reference only.

The manuscript source of truth is fixed as:

```text
/final.tex
```

Do not edit other paper/preprint/Overleaf `.tex` variants unless explicitly requested later.

---

# Phase 1 — Build Full Retraining Inventory

Read root `final.tex`, `outputs/RESULTS_FINAL.md`, canonical result JSON, configs, and training scripts.

Build the complete list of ALL trainable model/configuration combinations required by the final paper, not only previously missing checkpoints.

For each entry record: paper table/section, experiment ID, architecture, feature, dimension, augmentation, model variant, seeds, training script, CLI/config, output path, and dependent derived experiments.

Separate:

- Trainable backbones/configs: LSTM, BiLSTM, Transformer, ST-GCN, AAGCN, feature screening, augmentation ablations, graph streams, etc.
- Derived results: voting/fusion, SkelGym-Lite/Full, statistical tests, bootstrap, per-class summaries.

Derived results are recomputed from new predictions; they are not trained as independent neural networks.

Deliverables:

```text
artifacts/audit/full_retrain_inventory.md
artifacts/audit/full_retrain_inventory.json
```

This inventory is the approval gate before HF cleanup.

---

# Phase 2 — Clear Hugging Face Model Repo

Clear only:

```text
Cuong2004/gym-exercise-classification
```

Never delete or modify:

```text
Cuong2004/gym-exercise-landmarks
```

Goal: rebuild HF from a clean state so every model/checkpoint/artifact belongs to this new canonical full-retrain generation.

After cleanup, do not reuse old checkpoints.

---

# Phase 3 — Marimo Full Retraining

Training runs on the Marimo server using secrets via environment/secret manager only:

```bash
export MARIMO_TOKEN=...
export HF_TOKEN=...
```

Training policy:

- Full retrain from scratch.
- Do not resume old weights.
- Use seeds 42, 123, 3407.
- Keep train/val/test split fixed.
- Preserve canonical normalization, seq_len, strides, feature definitions, augmentation and model variants.

---

# Phase 4 — Parallel Training

Preferred: run 3 seeds of the same config simultaneously:

```text
Config A: seed42 + seed123 + seed3407
```

Alternative: run 3 different models simultaneously when this gives better server utilization.

Default max concurrency: 3 jobs, reduced only if GPU VRAM/RAM/CPU/disk limits require it.

Before each batch inspect GPU memory, RAM, CPU load, model footprint, and disk space.

---

# Phase 5 — Monitoring & Job Chaining

Use event-driven completion plus a 30-minute health-check fallback.

Each job reports `RUNNING`, `SUCCESS`, or `FAILED`.

When a batch finishes:

1. confirm expected outputs exist;
2. calculate/save full metrics and predictions;
3. archive locally;
4. upload the completed permanent package to HF;
5. immediately start the next pending batch.

Every 30 minutes check process state, epoch/log progress, checkpoint timestamp, GPU/RAM/disk usage, and stalled jobs.

If one seed fails, rerun only that seed.

---

# Phase 6 — Permanent Artifact Package

Primary objective: after this campaign, canonical models should never need to be trained again, and normal paper/statistical analysis should not need neural-network inference again.

Every model × seed must save:

```text
best_checkpoint.pt
last_checkpoint.pt (if supported)
config.json/yaml
provenance.json
metrics.json
training_history.json
predictions.npz
per_class_metrics.json
per_class_metrics.csv
confusion_matrix.npy
confusion_matrix.png
training.log
environment_manifest.json/txt
dataset_manifest.json
sha256.txt
```

Required provenance: architecture, feature, feature dimension, augmentation, model variant, seed, best epoch, train time, git commit, dataset hash, split identity, normalization, seq_len, strides, hyperparameters, checkpoint filename.

Required overall metrics for BOTH Val and Test, at BOTH Window and Video level:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Loss where supported
- number of windows/videos
- support

`predictions.npz` must contain at least:

```text
val_probs
val_targets
val_preds
val_video_ids
test_probs
test_targets
test_preds
test_video_ids
```

Also store video-aggregated probabilities/targets/predictions whenever possible.

Per-class artifacts must store Precision, Recall, F1, Support for Val/Test × Win/Vid for every class.

---

# Phase 7 — Multi-Seed Aggregation

After seeds 42/123/3407 complete, calculate Mean ± SD from raw per-seed values for Val Win, Val Vid, Test Win, Test Vid, and per-class metrics.

Never aggregate from already rounded Markdown/LaTeX values.

Create multiseed summary JSON/CSV artifacts.

---

# Phase 8 — Recompute All Derived Results

Because this is a full retrain, recompute every derived result referenced by `final.tex`, including applicable fusion/voting, SkelGym-Lite/Full, per-class tables, confusion matrices, McNemar, Wilcoxon, paired t-test, Holm-Bonferroni/FDR, bootstrap CI, and external evaluation dependent on retrained checkpoints.

Do not keep stale derived metrics from the old training generation.

---

# Phase 9 — Local + HF Permanent Archive

Maintain a deterministic local structure plus:

```text
artifacts/results/model_registry.json
```

The registry must map model/config/seed to checkpoint, metrics, predictions, per-class artifact, git commit, dataset hash, training date, HF path, and HF revision.

HF must contain ALL canonical information, not just `.pt` files:

- best/last checkpoints;
- configs;
- provenance;
- training histories;
- full metrics;
- prediction NPZ;
- per-class JSON/CSV;
- confusion matrices;
- dataset/environment manifests;
- multiseed summaries;
- model registry;
- canonical result artifacts;
- required script/config snapshots.

Upload each completed batch to HF as training progresses.

The intended end state is that future work can reconstruct tables, per-class metrics, statistics and ensembles directly from saved artifacts without retraining or inference.

---

# Phase 10 — Canonical Results Freeze

After all training and derived analyses finish, freeze a new canonical generation containing per-seed metrics, multiseed aggregates, Val/Test, Win/Vid, per-class, predictions, ensembles, statistics, provenance and HF locations.

This generation supersedes old benchmark numbers.

No `final.tex` edits before this freeze.

---

# Phase 11 — Update `outputs/RESULTS_FINAL.md`

Regenerate/update `RESULTS_FINAL.md` from the new canonical artifacts, preferably programmatically.

Update affected tables, Mean ± SD, Val/Test, Win/Vid, per-class results, winners/rankings, statistics, model descriptions and narrative claims.

---

# Phase 12 — Minimal-Diff Update of Root `final.tex`

The ONLY Final Paper target is `/final.tex`.

Strict minimal-diff rules:

- no global formatting;
- no unrelated whitespace cleanup;
- no section reordering;
- no unnecessary label/citation changes;
- no rewriting unaffected paragraphs;
- no touching unrelated figures.

If only a number changes, replace only that number. If only one table row changes, replace only that row. If a paragraph remains scientifically correct, leave it untouched.

Before rewriting Abstract/Discussion/Conclusion, flag material changes such as changed winner/ranking, major headline metric shift, changed significance/CI interpretation, changed augmentation conclusion, or an unsupported old claim.

Operational review trigger:

```text
absolute headline metric change >= 1.0 percentage point
```

If substantive rewrite is required, first create:

```text
artifacts/audit/latex_material_change_report.md
```

Report: section, original claim, old result, new result, why rewrite is required, minimum proposed rewrite, and downstream affected sections.

No silent broad rewrite.

---

# Phase 13 — Full Final Paper Audit

Audit root `final.tex` for:

- dataset counts/splits/windows/videos/support;
- feature dimensions, seq_len, stride, normalization;
- every Accuracy/Precision/Recall/F1/N/params/FLOPs/p-value/CI/seed count;
- citation keys, bibliography metadata, DOI/URL where applicable, citation support;
- terminology/method consistency;
- statistical consistency with the new prediction generation;
- undefined references/citations, duplicate labels, table overflow, missing figures/assets, equation/bibliography errors.

Every numeric claim must map to the new canonical generation.

Compile root `final.tex` as the final technical check.

---

# Phase 14 — Final Submission Audit

Create:

```text
artifacts/audit/final_submission_audit.md
```

Report full-retrain completion, HF archive completeness, metric completeness, dataset consistency, statistical validity, citation/reference audit, minimal-diff LaTeX changes, material-change report, compile status, remaining limitations and submission readiness.

Final status: `READY_TO_SUBMIT`, `READY_WITH_MINOR_WARNINGS`, or `NOT_READY`.

---

# Execution Order

```text
1. git pull latest main
2. Freeze baseline
3. Read root final.tex and build full retrain inventory
4. Approval gate
5. Clear Cuong2004/gym-exercise-classification
6. Keep Cuong2004/gym-exercise-landmarks untouched
7. Full retrain on Marimo
8. Prefer 3 seeds/config in parallel
9. Event-driven chaining + 30-minute heartbeat
10. Save complete checkpoint + metrics + predictions + per-class artifacts
11. Upload completed batches to HF
12. Aggregate 3 seeds
13. Recompute all derived ensembles/statistics
14. Complete local model registry and permanent HF archive
15. Freeze canonical results
16. Update outputs/RESULTS_FINAL.md
17. Compare old/new values used by final.tex
18. Apply minimal-diff edits to root final.tex
19. Report before substantive discussion rewrite
20. Audit dataset/numbers/citations/references/method/statistics
21. Compile root final.tex
22. Produce final submission audit
```

---

# Approval Gate Before HF Cleanup / Training

Before clearing HF or training, report:

```text
FINAL PAPER SOURCE: final.tex (repository root)
TOTAL TRAINABLE CONFIGURATIONS: ...
TOTAL PLANNED TRAINING RUNS: ...
MULTISEED POLICY: 42 / 123 / 3407
PLANNED PARALLEL BATCHES: ...
DERIVED RESULTS TO RECOMPUTE: ...
HF MODEL REPO TO CLEAR: Cuong2004/gym-exercise-classification
PROTECTED HF DATASET REPO: Cuong2004/gym-exercise-landmarks
MARIMO EXECUTION: Configured / Not configured
PERMANENT ARTIFACT PACKAGE: checkpoint + config + provenance + full metrics + predictions + per-class + confusion matrices + histories + manifests
NO HF CLEANUP STARTED YET
NO TRAINING STARTED YET
NO FINAL.TEX EDITS STARTED YET
```

---

# Non-Negotiable Rules

- Full retrain means from scratch; do not resume old model weights.
- Do not spend the campaign trying to match legacy checkpoints/results.
- Never touch the protected HF dataset repository.
- No fabricated metrics.
- No single-seed result labelled multiseed.
- No stale ensemble/statistical results after retraining.
- No token stored in the repository.
- HF must retain checkpoints AND downstream predictions/artifacts.
- Store enough prediction/result data so normal paper analysis never requires inference again.
- No `final.tex` update before canonical results freeze.
- No broad LaTeX rewrite when a local replacement is sufficient.
- No silent substantive discussion rewrite after material numerical changes.
- Every number in the final paper must be traceable to the new full-retrain artifact generation.