#!/usr/bin/env python3
"""
Comprehensive Augmentation Ablation Benchmark Runner.

Executes and aggregates two independent, rigorous ablation studies on the representative
sequence model (Transformer mix 117-d) evaluated across multiple random seeds (default: 42, 123, 3407):

1. Table 1: Leave-One-Out (LOO) Ablation Study (Component Subtraction)
   - Evaluates operator necessity by removing one operator at a time from Candidate Full (5-op).
   - Variants:
     * Candidate Full (All 5 Operators: Mirror + Yaw + Scale + TimeWarp + Jitter) [Reference]
     * w/o Sagittal Reflection (-Mirror) [Geometric]
     * w/o Gravitational Yaw (-Yaw) [Geometric]
     * w/o Proportional Scaling (-Scale) [Geometric]
     * w/o Temporal TimeWarp (-TimeWarp / SkelGym-Aug 4-op) [Temporal]
     * w/o Sensor Jitter (-Jitter) [Noise-based]
     * Clean Baseline (No Augmentation) [Control / None]

2. Table 2: Single Component (Individual) Augmentation Study (Component Addition)
   - Evaluates independent efficacy by adding exactly one operator in isolation to the Clean Baseline.
   - Variants:
     * Clean Baseline (No Augmentation) [Control / Reference]
     * + Sagittal Reflection only (single_mirror) [Geometric (Bilateral)]
     * + Gravitational Yaw only (single_yaw) [Geometric (3D Viewpoint)]
     * + Proportional Scaling only (single_scale) [Geometric (Body Scale)]
     * + Temporal TimeWarp only (single_timewarp) [Temporal (Cadence)]
     * + Sensor Jitter only (single_jitter) [Noise-based (Sensor Noise)]
     * SkelGym-Aug (4-op: Mirror+Yaw+Scale+Jitter) [Proposed Task-Oriented]
     * Candidate Full (5-op: +TimeWarp) [Full Candidate]

3. Table 3: Cross-Backbone Generalization (Optional)
   - Compares Clean Baseline vs. SkelGym-Aug across Transformer, AAGCN, BiLSTM, and ST-GCN.

Outputs:
  - outputs/augmentation_ablation_results.json (Full structured metrics)
  - outputs/augmentation_ablation_results.md (Complete Markdown report containing both tables)
  - outputs/table_leave_one_out.tex (Standalone LaTeX table for LOO)
  - outputs/table_single_component.tex (Standalone LaTeX table for Single Component)
"""

import os
import sys
import re
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import NUM_CLASSES, CANONICAL_EXPERIMENT_REGISTRY
from src.data.dataset import get_dataloaders
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.reproducibility import load_checkpoint_weights

SEEDS = [42, 123, 3407]

# ------------------------------------------------------------------------------
# Variant Definitions
# ------------------------------------------------------------------------------
LOO_VARIANTS = [
    ("Candidate_Full_5op", "skel_gym_aug_legacy_5op", "Candidate Full (All 5 Operators)", "None (Reference)"),
    ("Minus_Mirror", "skel_gym_aug_no_mirror", "w/o Sagittal Reflection (-Mirror)", "Geometric"),
    ("Minus_Yaw", "skel_gym_aug_no_yaw", "w/o Gravitational Yaw (-Yaw)", "Geometric"),
    ("Minus_Scale", "skel_gym_aug_no_scale", "w/o Proportional Scaling (-Scale)", "Geometric"),
    ("Minus_TimeWarp", "skel_gym_aug", "w/o Temporal TimeWarp (-TimeWarp / SkelGym-Aug)", "Temporal"),
    ("Minus_Jitter", "skel_gym_aug_no_jitter", "w/o Sensor Jitter (-Jitter)", "Noise-based"),
    ("Clean_Baseline_NoAug", "none", "Clean Baseline (No Augmentation)", "All Operators"),
]

SINGLE_COMPONENT_VARIANTS = [
    ("Clean_Baseline_NoAug", "none", "Clean Baseline (No Augmentation)", "None (Control)"),
    ("Single_Mirror", "single_mirror", "+ Sagittal Reflection only (Mirror)", "Geometric (Bilateral)"),
    ("Single_Yaw", "single_yaw", "+ Gravitational Yaw only (Yaw)", "Geometric (3D Viewpoint)"),
    ("Single_Scale", "single_scale", "+ Proportional Scaling only (Scale)", "Geometric (Body Scale)"),
    ("Single_TimeWarp", "single_timewarp", "+ Temporal TimeWarp only (TimeWarp)", "Temporal (Cadence)"),
    ("Single_Jitter", "single_jitter", "+ Sensor Jitter only (Jitter)", "Noise-based (Sensor Noise)"),
    ("SkelGym_Aug_4op", "skel_gym_aug", "SkelGym-Aug (4-op: Mirror+Yaw+Scale+Jitter)", "Spatial + Sensor (Proposed)"),
    ("Candidate_Full_5op", "skel_gym_aug_legacy_5op", "Candidate Full (5-op: +TimeWarp)", "Spatial + Sensor + Temporal"),
]

BACKBONES = [
    ("Transformer", "mix"),
    ("AAGCN", "bone_3d"),
    ("BiLSTM", "mix"),
    ("STGCN", "rel_3d"),
]

# ------------------------------------------------------------------------------
# Checkpoint & Log Discovery
# ------------------------------------------------------------------------------
def find_existing_checkpoint(task: Dict[str, Any], checkpoint_base: Path, force_retrain: bool = False) -> Optional[Path]:
    """
    Finds existing checkpoint corresponding to task.
    Prevents erroneous cross-variant fallbacks.
    """
    if force_retrain:
        return None

    task_id = task["id"]
    model = task["model"]
    feature = task["feature"]
    aug = task["augment"]
    seed = task["seed"]

    candidates = [
        # Explicit task ID in ablation_aug
        checkpoint_base / "ablation_aug" / f"best_{task_id}.pt",
        checkpoint_base / "ablation_aug" / task_id / f"best_{task_id}.pt",
        checkpoint_base / "ablation_aug" / f"best_{model}_{feature}_aug_{aug}_seed{seed}.pt",
        checkpoint_base / "ablation_aug" / f"best_{model}_{feature}_aug_{aug}.pt",
        checkpoint_base / f"best_{task_id}.pt",
        checkpoint_base / f"seed{seed}" / f"best_{task_id}.pt",
        checkpoint_base / f"seed{seed}" / f"best_{model}_{feature}_aug_{aug}.pt",
    ]

    # Canonical experiment mappings across seeds
    seed_dir = checkpoint_base / f"seed{seed}"
    if model == "Transformer" and feature == "mix" and aug == "none":
        candidates.append(seed_dir / "best_Transformer_T1.27_mix.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_Transformer_T1.27_mix.pt")
    elif model == "Transformer" and feature == "mix" and aug == "skel_gym_aug":
        candidates.append(seed_dir / "best_Transformer_T2.2_mix.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_Transformer_T2.2_mix.pt")
    elif model == "AAGCN" and feature == "bone_3d" and aug == "none":
        candidates.append(seed_dir / "best_AAGCN_T3.6_bone_3d.pt")
        candidates.append(seed_dir / "best_AAGCN_T4.1_bone_3d.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_AAGCN_T4.1_bone_3d.pt")
            candidates.append(checkpoint_base / "best_AAGCN_T3.6_bone_3d.pt")
    elif model == "AAGCN" and feature == "bone_3d" and aug == "skel_gym_aug":
        candidates.append(seed_dir / "best_AAGCN_T4.2_bone_3d.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_AAGCN_T4.2_bone_3d.pt")
    elif model == "BiLSTM" and feature == "mix" and aug == "none":
        candidates.append(seed_dir / "best_BiLSTM_T1.18_mix.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_BiLSTM_T1.18_mix.pt")
    elif model == "STGCN" and feature == "rel_3d" and aug == "none":
        candidates.append(seed_dir / "best_STGCN_T3.2_rel_3d.pt")
        if seed == 42:
            candidates.append(checkpoint_base / "best_STGCN_T3.2_rel_3d.pt")

    for cand in candidates:
        if cand.exists() and cand.stat().st_size > 1000:
            return cand

    return None

def parse_validation_metrics_from_log(log_path: Path) -> Dict[str, Any]:
    """Parses training log to extract validation loss, validation accuracy, and best epoch."""
    if not log_path.exists():
        return {}

    best_val_acc = 0.0
    best_val_loss = 999.0
    best_train_loss = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    min_val_loss = 999.0
    min_loss_epoch = 0

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Epoch (\d+)/\d+ .* - loss: ([\d\.]+) - acc: ([\d\.]+) - val_loss: ([\d\.]+) - val_acc: ([\d\.]+)", line)
            if m:
                ep = int(m.group(1))
                t_loss = float(m.group(2))
                t_acc = float(m.group(3))
                v_loss = float(m.group(4))
                v_acc = float(m.group(5))
                if v_loss < min_val_loss:
                    min_val_loss = v_loss
                    min_loss_epoch = ep
            if "--> Best checkpoint saved" in line:
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_train_loss = t_loss
                best_train_acc = t_acc
                best_epoch = ep

    return {
        "val_acc": round(best_val_acc * 100.0, 2),
        "val_loss": round(best_val_loss, 4),
        "min_val_loss": round(min_val_loss, 4),
        "train_loss": round(best_train_loss, 4),
        "best_epoch": best_epoch,
        "loss_gap": round(best_val_loss - best_train_loss, 4)
    }

def get_validation_metrics(ckpt_path: Optional[Path], log_path: Path) -> Dict[str, Any]:
    """Extracts validation metrics directly from checkpoint provenance sidecar if available, falling back to log parsing."""
    if ckpt_path and ckpt_path.exists():
        prov_candidates = [
            ckpt_path.with_suffix(".provenance.json"),
            ckpt_path.parent / f"{ckpt_path.stem}.provenance.json",
            ckpt_path.with_suffix(ckpt_path.suffix + ".provenance.json"),
            ckpt_path.parent / f"{ckpt_path.name}.provenance.json"
        ]
        for prov_p in prov_candidates:
            if prov_p.exists():
                try:
                    with open(prov_p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    val_acc = data.get("val_acc")
                    val_loss = data.get("val_loss")
                    val_f1 = data.get("val_macro_f1")
                    train_loss = data.get("train_loss")
                    best_ep = data.get("best_epoch", data.get("epoch", 0))
                    if val_acc is not None:
                        acc_pct = float(val_acc) * 100.0 if float(val_acc) <= 1.0 else float(val_acc)
                        v_loss = float(val_loss) if val_loss is not None else 0.0
                        t_loss = float(train_loss) if train_loss is not None else 0.0
                        return {
                            "val_acc": round(acc_pct, 2),
                            "val_loss": round(v_loss, 4),
                            "val_macro_f1": round(float(val_f1), 4) if val_f1 is not None else 0.0,
                            "train_loss": round(t_loss, 4),
                            "best_epoch": int(best_ep),
                            "loss_gap": round(v_loss - t_loss, 4)
                        }
                except Exception:
                    pass
    return parse_validation_metrics_from_log(log_path)

# ------------------------------------------------------------------------------
# Task Execution & Evaluation
# ------------------------------------------------------------------------------
def get_task_hyperparameters(model: str, feature: str, augment: str = "none") -> Dict[str, Any]:
    """
    Resolves hyperparameters from CANONICAL_EXPERIMENT_REGISTRY,
    ensuring each backbone (Transformer, AAGCN, BiLSTM, STGCN) adheres strictly
    to its canonical training protocol.
    """
    # 1. Exact match (model, feature, augment)
    for exp in CANONICAL_EXPERIMENT_REGISTRY.values():
        if exp.get("model") == model and exp.get("feature") == feature and exp.get("augment") == augment:
            return dict(exp)

    # 2. Match (model, feature)
    for exp in CANONICAL_EXPERIMENT_REGISTRY.values():
        if exp.get("model") == model and exp.get("feature") == feature:
            return dict(exp)

    # 3. Canonical defaults by architecture
    is_graph = model in ("AAGCN", "STGCN")
    is_smooth = model in ("Transformer", "AAGCN")
    return {
        "lr": 1e-3 if model in ("AAGCN", "STGCN", "BiLSTM", "LSTM") else 1e-4,
        "batch_size": 32 if is_graph else 16,
        "label_smoothing": 0.05 if is_smooth else 0.0,
        "patience": 10,
        "epochs": 100,
        "early_stopping_metric": "val_macro_f1",
        "train_stride": 16,
        "val_test_stride": 32,
    }

def train_task_subprocess(
    task: Dict[str, Any],
    ckpt_path: Path,
    device: str = "cuda",
    push_to_hf: bool = False,
    hf_repo: str = "Cuong2004/gym-exercise-classification",
    hf_token: Optional[str] = None
) -> bool:
    """Executes training via isolated subprocess."""
    task_ckpt_dir = ckpt_path.parent / task["id"]
    task_ckpt_dir.mkdir(parents=True, exist_ok=True)

    hp = get_task_hyperparameters(task["model"], task["feature"], task["augment"])

    cmd = [
        sys.executable, "-u", "-m", "src.cli", "train",
        "--model", task["model"],
        "--feature", task["feature"],
        "--augment", task["augment"],
        "--seed", str(task["seed"]),
        "--checkpoint_name", task["id"],
        "--epochs", str(hp.get("epochs", 100)),
        "--lr", str(hp.get("lr", 1e-4)),
        "--patience", str(hp.get("patience", 10)),
        "--batch_size", str(hp.get("batch_size", 16)),
        "--label_smoothing", str(hp.get("label_smoothing", 0.05)),
        "--early_stopping_metric", str(hp.get("early_stopping_metric", "val_macro_f1")),
        "--train_stride", str(hp.get("train_stride", 16)),
        "--val_test_stride", str(hp.get("val_test_stride", 32)),
        "--no_test_eval",
        "--device", device,
        "--checkpoint_dir", str(task_ckpt_dir),
        "--output_dir", f"outputs/ablation_runs/{task['id']}",
        "--metadata", task.get("metadata", "Final_dataset_metadata.csv"),
        "--landmark_dir", task.get("landmark_dir", "data/landmarks")
    ]
    if push_to_hf:
        cmd.append("--push_to_hf")
        cmd.extend(["--hf_repo", hf_repo])
        if hf_token:
            cmd.extend(["--hf_token", hf_token])

    log_dir = Path("outputs/ablation_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{task['id']}.log"

    print(f"[START] Training {task['id']} on device {device} -> {ckpt_path.name}", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_file, "w", encoding="utf-8") as lf:
        res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, env=env)

    elapsed = time.time() - t0
    if res.returncode == 0:
        print(f"[DONE] {task['id']} finished in {elapsed:.1f}s", flush=True)
        best_produced = task_ckpt_dir / f"best_{task['id']}.pt"
        if best_produced.exists():
            shutil.copy2(best_produced, ckpt_path)
            best_prov = task_ckpt_dir / f"best_{task['id']}.provenance.json"
            if best_prov.exists():
                shutil.copy2(best_prov, ckpt_path.with_suffix(".provenance.json"))
        return True
    else:
        print(f"[ERROR] {task['id']} failed with code {res.returncode}. See {log_file}", flush=True)
        return False

def evaluate_checkpoint(
    ckpt_path: Path,
    model_type: str,
    feature_method: str,
    device: torch.device,
    metadata_path: str = "Final_dataset_metadata.csv",
    landmark_dir: str = "data/landmarks",
    seed: Optional[int] = None
) -> Dict[str, float]:
    """Evaluates checkpoint on the held-out test partition."""
    state_dict, _ = load_checkpoint_weights(ckpt_path, device="cpu")

    model = build_model(model_type, feature_method, num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method=feature_method,
        batch_size=32,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        landmark_dir=landmark_dir,
        num_workers=0,
        in_memory=True,
        seed=seed,
        strict_norm=True
    )

    test_video_ids = test_loader.dataset.video_ids
    trainer = Trainer(model=model, device=device)
    y_test_true, _, y_test_probs = trainer.predict(test_loader)

    win_preds = np.argmax(y_test_probs, axis=1)
    win_metrics = compute_metrics(y_test_true, win_preds)
    _, _, _, vid_metrics = aggregate_video_level_predictions(y_test_probs, y_test_true, test_video_ids)

    return {
        "win_acc": float(win_metrics["accuracy"] * 100.0),
        "win_f1": float(win_metrics["macro_f1"]),
        "vid_acc": float(vid_metrics["accuracy"] * 100.0),
        "vid_f1": float(vid_metrics["macro_f1"])
    }

# ------------------------------------------------------------------------------
# Task List Builder
# ------------------------------------------------------------------------------
def build_task_list(
    metadata_path: str,
    landmark_dir: str,
    mode: str = "all",
    seeds: List[int] = SEEDS
) -> List[Dict[str, Any]]:
    """Builds task list for Leave-One-Out, Single Component, and Backbone Generalization."""
    tasks = []

    # 1. Leave-One-Out Tasks
    if mode in ("all", "loo"):
        for var_name, aug_method, disp_name, domain in LOO_VARIANTS:
            for seed in seeds:
                tasks.append({
                    "group": "Leave_One_Out",
                    "variant": var_name,
                    "display_name": disp_name,
                    "domain": domain,
                    "id": f"LOO_Trans_{var_name}_seed{seed}",
                    "model": "Transformer",
                    "feature": "mix",
                    "augment": aug_method,
                    "seed": seed,
                    "metadata": metadata_path,
                    "landmark_dir": landmark_dir
                })

    # 2. Single Component Tasks
    if mode in ("all", "single"):
        for var_name, aug_method, disp_name, domain in SINGLE_COMPONENT_VARIANTS:
            for seed in seeds:
                tasks.append({
                    "group": "Single_Component",
                    "variant": var_name,
                    "display_name": disp_name,
                    "domain": domain,
                    "id": f"Single_Trans_{var_name}_seed{seed}",
                    "model": "Transformer",
                    "feature": "mix",
                    "augment": aug_method,
                    "seed": seed,
                    "metadata": metadata_path,
                    "landmark_dir": landmark_dir
                })

    # 3. Cross-Backbone Generalization Tasks
    if mode in ("all", "backbones"):
        for model, feat in BACKBONES:
            for aug in ["none", "skel_gym_aug"]:
                aug_tag = "Aug" if aug == "skel_gym_aug" else "NoAug"
                for seed in seeds:
                    tasks.append({
                        "group": "Backbone_Ablation",
                        "variant": f"{model}_{feat}_{aug_tag}",
                        "display_name": f"{model} ({feat}) [{'SkelGym-Aug' if aug == 'skel_gym_aug' else 'No Aug'}]",
                        "domain": "Backbone Generalization",
                        "id": f"Backbone_{model}_{feat}_{aug_tag}_seed{seed}",
                        "model": model,
                        "feature": feat,
                        "augment": aug,
                        "seed": seed,
                        "metadata": metadata_path,
                        "landmark_dir": landmark_dir
                    })

    return tasks

# ------------------------------------------------------------------------------
# LaTeX Table Generators
# ------------------------------------------------------------------------------
def generate_loo_latex_table(summary_loo: Dict[str, Any]) -> str:
    """Generates standalone LaTeX code for Table 1: Leave-One-Out Ablation."""
    ref_acc = summary_loo.get("Candidate_Full_5op", {}).get("win_acc_mean", 66.41)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Systematic Leave-One-Out (LOO) ablation on candidate augmentation operators across validation and held-out test partitions evaluated across three independent random seeds ($\text{seed} \in \{42, 123, 3407\}$) on the representative sequence model (\texttt{Transformer mix}). All metrics reported as $\text{Mean} \pm \text{Std}$.}",
        r"\label{tab:table2_loo}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"\textbf{Augmentation Configuration} & \textbf{Excluded Domain} & \textbf{Val Window Acc (\%)} & \textbf{Val Loss} & \textbf{Test Window Acc (\%)} & \textbf{Test Macro F1} & \textbf{$\Delta$ vs. Full (Test)} \\",
        r"\midrule"
    ]

    for var_name, _, disp_name, domain in LOO_VARIANTS:
        if var_name not in summary_loo:
            continue
        s = summary_loo[var_name]
        va = f"{s['val_acc_mean']:.2f}\\% $\\pm$ {s['val_acc_std']:.2f}\\%" if s.get("val_acc_mean") else "--"
        vl = f"{s['val_loss_mean']:.4f} $\\pm$ {s['val_loss_std']:.4f}" if s.get("val_loss_mean") else "--"
        ta = f"{s['win_acc_mean']:.2f}\\% $\\pm$ {s['win_acc_std']:.2f}\\%"
        tf = f"{s['win_f1_mean']:.4f} $\\pm$ {s['win_f1_std']:.4f}"
        diff = s['win_acc_mean'] - ref_acc
        diff_str = f"+{diff:.2f}\\%" if diff > 0 else (f"{diff:.2f}\\%" if diff < 0 else "0.00\\% (Ref)")

        lines.append(f"{disp_name} & {domain} & {va} & {vl} & {ta} & {tf} & {diff_str} \\\\")
        if var_name == "Candidate_Full_5op":
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    return "\n".join(lines)

def generate_single_component_latex_table(summary_single: Dict[str, Any]) -> str:
    """Generates standalone LaTeX code for Table 2: Single Component Ablation."""
    base_acc = summary_single.get("Clean_Baseline_NoAug", {}).get("win_acc_mean", 63.35)
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Systematic Single-Component (Individual) Augmentation ablation evaluating each candidate operator in complete isolation against the Clean Baseline on \texttt{Transformer mix} across three seeds ($\text{seed} \in \{42, 123, 3407\}$). All metrics reported as $\text{Mean} \pm \text{Std}$.}",
        r"\label{tab:table2_single}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"\textbf{Augmentation Configuration} & \textbf{Applied Domain} & \textbf{Val Window Acc (\%)} & \textbf{Val Loss} & \textbf{Test Window Acc (\%)} & \textbf{Test Macro F1} & \textbf{$\Delta$ vs. Baseline (Test)} \\",
        r"\midrule"
    ]

    for var_name, _, disp_name, domain in SINGLE_COMPONENT_VARIANTS:
        if var_name not in summary_single:
            continue
        s = summary_single[var_name]
        va = f"{s['val_acc_mean']:.2f}\\% $\\pm$ {s['val_acc_std']:.2f}\\%" if s.get("val_acc_mean") else "--"
        vl = f"{s['val_loss_mean']:.4f} $\\pm$ {s['val_loss_std']:.4f}" if s.get("val_loss_mean") else "--"
        ta = f"{s['win_acc_mean']:.2f}\\% $\\pm$ {s['win_acc_std']:.2f}\\%"
        tf = f"{s['win_f1_mean']:.4f} $\\pm$ {s['win_f1_std']:.4f}"
        diff = s['win_acc_mean'] - base_acc
        diff_str = f"+{diff:.2f}\\%" if diff > 0 else (f"{diff:.2f}\\%" if diff < 0 else "0.00\\% (Ref)")

        if var_name == "SkelGym_Aug_4op":
            lines.append(r"\midrule")
            lines.append(f"{disp_name} & {domain} & {va} & {vl} & {ta} & {tf} & {diff_str} \\\\")
        elif var_name == "Clean_Baseline_NoAug":
            lines.append(f"{disp_name} & {domain} & {va} & {vl} & {ta} & {tf} & {diff_str} \\\\")
            lines.append(r"\midrule")
        else:
            lines.append(f"{disp_name} & {domain} & {va} & {vl} & {ta} & {tf} & {diff_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}"
    ])
    return "\n".join(lines)

# ------------------------------------------------------------------------------
# Main Orchestrator
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Systematic Augmentation Ablation Runner")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "loo", "single", "backbones"], help="Ablation study mode")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for parallel execution")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cuda/mps/cpu/auto)")
    parser.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Directory containing landmark CSVs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Base directory for model checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for reports and results")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining even if checkpoint exists")
    parser.add_argument("--skip_train", action="store_true", help="Skip training phase and only evaluate existing checkpoints")
    parser.add_argument("--push_to_hf", action="store_true", default=False, help="Upload checkpoints to Hugging Face Hub")
    parser.add_argument("--hf_repo", type=str, default="Cuong2004/gym-exercise-classification", help="Hugging Face Model repository ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face authentication token")
    args = parser.parse_args()

    # Device resolution
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device_str = args.device

    ckpt_base = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_task_list(
        metadata_path=args.metadata,
        landmark_dir=args.landmark_dir,
        mode=args.mode,
        seeds=args.seeds
    )
    print(f"\n========================================================")
    print(f"  SkelGym Augmentation Ablation Suite (Mode: {args.mode.upper()})")
    print(f"  Total Tasks: {len(tasks)} | Seeds: {args.seeds} | Device: {device_str} | Push to HF: {args.push_to_hf}")
    print(f"========================================================\n")

    # 1. Resolve Checkpoints & Missing Tasks
    tasks_to_train = []
    task_ckpts = {}

    for t in tasks:
        ckpt = find_existing_checkpoint(t, ckpt_base, force_retrain=args.force_retrain)
        if ckpt and ckpt.exists():
            task_ckpts[t["id"]] = ckpt
        else:
            target_ckpt = ckpt_base / "ablation_aug" / f"best_{t['id']}.pt"
            target_ckpt.parent.mkdir(parents=True, exist_ok=True)
            task_ckpts[t["id"]] = target_ckpt
            tasks_to_train.append((t, target_ckpt))

    print(f"Checkpoint scan: {len(task_ckpts) - len(tasks_to_train)} found, {len(tasks_to_train)} need training.")

    # 2. Training Phase
    if tasks_to_train and not args.skip_train:
        print(f"\n>>> Launching {len(tasks_to_train)} training tasks with {args.workers} workers <<<\n")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(train_task_subprocess, t, ckpt, device_str, args.push_to_hf, args.hf_repo, args.hf_token): t["id"]
                for t, ckpt in tasks_to_train
            }
            for fut in as_completed(futures):
                t_id = futures[fut]
                try:
                    success = fut.result()
                    if not success:
                        print(f"[WARN] Task {t_id} failed.")
                except Exception as e:
                    print(f"[FAIL] Task {t_id} exception: {e}")

    # 3. Evaluation & Metric Aggregation Phase
    eval_device = torch.device(device_str)
    all_results = {}

    print(f"\n>>> Evaluating checkpoints and parsing validation metrics <<<\n")
    for idx_eval, t in enumerate(tasks, 1):
        t_id = t["id"]
        ckpt = task_ckpts.get(t_id)

        # Parse validation metrics from checkpoint sidecar provenance or log
        log_file = Path("outputs/ablation_logs") / f"{t_id}.log"
        val_metrics = get_validation_metrics(ckpt, log_file)

        # Evaluate on test set
        if ckpt and ckpt.exists():
            print(f"[{idx_eval}/{len(tasks)}] Evaluating {t_id}...")
            test_metrics = evaluate_checkpoint(
                ckpt_path=ckpt,
                model_type=t["model"],
                feature_method=t["feature"],
                device=eval_device,
                metadata_path=args.metadata,
                landmark_dir=args.landmark_dir,
                seed=t.get("seed")
            )
        else:
            print(f"[{idx_eval}/{len(tasks)}] Checkpoint not found for {t_id}, using log metrics if available.")
            test_metrics = {}

        t_record = dict(t)
        t_record["checkpoint"] = str(ckpt) if ckpt else None
        t_record["metrics"] = {**val_metrics, **test_metrics}
        all_results[t_id] = t_record

    # 4. Compute Statistical Summaries (Mean ± Std)
    def aggregate_group(group_name: str, variants_list: List[Tuple]) -> Dict[str, Any]:
        summary = {}
        for var_tuple in variants_list:
            v_key = var_tuple[0]
            v_tasks = [r for r in all_results.values() if r.get("group") == group_name and r.get("variant") == v_key]
            if not v_tasks:
                continue

            def get_metric_list(metric_key: str):
                vals = [r["metrics"][metric_key] for r in v_tasks if metric_key in r["metrics"] and r["metrics"][metric_key] is not None]
                return vals

            w_accs = get_metric_list("win_acc")
            w_f1s = get_metric_list("win_f1")
            v_accs = get_metric_list("vid_acc")
            v_f1s = get_metric_list("vid_f1")
            val_accs = get_metric_list("val_acc")
            val_losses = get_metric_list("val_loss")
            val_f1s = get_metric_list("val_macro_f1")

            summary[v_key] = {
                "count": len(v_tasks),
                "display_name": var_tuple[2],
                "domain": var_tuple[3],
                "win_acc_mean": float(np.mean(w_accs)) if w_accs else 0.0,
                "win_acc_std": float(np.std(w_accs)) if w_accs else 0.0,
                "win_f1_mean": float(np.mean(w_f1s)) if w_f1s else 0.0,
                "win_f1_std": float(np.std(w_f1s)) if w_f1s else 0.0,
                "vid_acc_mean": float(np.mean(v_accs)) if v_accs else 0.0,
                "vid_acc_std": float(np.std(v_accs)) if v_accs else 0.0,
                "vid_f1_mean": float(np.mean(v_f1s)) if v_f1s else 0.0,
                "vid_f1_std": float(np.std(v_f1s)) if v_f1s else 0.0,
                "val_acc_mean": float(np.mean(val_accs)) if val_accs else 0.0,
                "val_acc_std": float(np.std(val_accs)) if val_accs else 0.0,
                "val_loss_mean": float(np.mean(val_losses)) if val_losses else 0.0,
                "val_loss_std": float(np.std(val_losses)) if val_losses else 0.0,
                "val_macro_f1_mean": float(np.mean(val_f1s)) if val_f1s else 0.0,
                "val_macro_f1_std": float(np.std(val_f1s)) if val_f1s else 0.0,
            }
        return summary

    summary_loo = aggregate_group("Leave_One_Out", LOO_VARIANTS)
    summary_single = aggregate_group("Single_Component", SINGLE_COMPONENT_VARIANTS)

    # Backbones summary
    bb_variants = set(r["variant"] for r in all_results.values() if r.get("group") == "Backbone_Ablation")
    summary_backbones = {}
    for v in sorted(bb_variants):
        v_tasks = [r for r in all_results.values() if r.get("group") == "Backbone_Ablation" and r.get("variant") == v]
        if v_tasks:
            w_accs = [r["metrics"]["win_acc"] for r in v_tasks if "win_acc" in r["metrics"]]
            w_f1s = [r["metrics"]["win_f1"] for r in v_tasks if "win_f1" in r["metrics"]]
            v_accs = [r["metrics"]["vid_acc"] for r in v_tasks if "vid_acc" in r["metrics"]]
            v_f1s = [r["metrics"]["vid_f1"] for r in v_tasks if "vid_f1" in r["metrics"]]
            summary_backbones[v] = {
                "count": len(v_tasks),
                "win_acc_mean": float(np.mean(w_accs)) if w_accs else 0.0,
                "win_acc_std": float(np.std(w_accs)) if w_accs else 0.0,
                "win_f1_mean": float(np.mean(w_f1s)) if w_f1s else 0.0,
                "win_f1_std": float(np.std(w_f1s)) if w_f1s else 0.0,
                "vid_acc_mean": float(np.mean(v_accs)) if v_accs else 0.0,
                "vid_acc_std": float(np.std(v_accs)) if v_accs else 0.0,
                "vid_f1_mean": float(np.mean(v_f1s)) if v_f1s else 0.0,
                "vid_f1_std": float(np.std(v_f1s)) if v_f1s else 0.0,
            }

    # 5. Save JSON Database
    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": args.seeds,
        "summary_leave_one_out": summary_loo,
        "summary_single_component": summary_single,
        "summary_backbones": summary_backbones,
        "raw_tasks": all_results
    }

    out_json = out_dir / "augmentation_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[SAVED] Structured JSON database written to {out_json}")

    # 6. Generate Markdown Report
    out_md = out_dir / "augmentation_ablation_results.md"
    with open(out_md, "w", encoding="utf-8") as mf:
        mf.write("# Systematic SkelGym Augmentation Ablation Benchmark\n\n")
        mf.write(f"Evaluated across {len(args.seeds)} independent random seeds: `{args.seeds}`.\n\n")

        # Table 1: Leave-One-Out
        mf.write("## Table 1: Systematic Leave-One-Out (LOO) Ablation Study\n\n")
        mf.write("*Evaluates operator necessity by subtracting each operator from Candidate Full (5-op).*\n\n")
        mf.write("| Augmentation Configuration | Excluded Domain | Val Window Acc (%) | Val Loss | Test Window Acc (%) | Test Macro F1 | Delta vs. Full (Test) |\n")
        mf.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        ref_loo = summary_loo.get("Candidate_Full_5op", {}).get("win_acc_mean", 66.41)
        for var_name, _, disp_name, domain in LOO_VARIANTS:
            if var_name in summary_loo:
                s = summary_loo[var_name]
                diff = s["win_acc_mean"] - ref_loo
                diff_str = f"+{diff:.2f}%" if diff > 0 else (f"{diff:.2f}%" if diff < 0 else "Ref (0.00%)")
                va = f"{s['val_acc_mean']:.2f}% ± {s['val_acc_std']:.2f}%" if s.get("val_acc_mean") else "--"
                vl = f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f}" if s.get("val_loss_mean") else "--"
                mf.write(f"| {disp_name} | {domain} | {va} | {vl} | {s['win_acc_mean']:.2f}% ± {s['win_acc_std']:.2f}% | {s['win_f1_mean']:.4f} ± {s['win_f1_std']:.4f} | {diff_str} |\n")

        # Table 2: Single Component
        mf.write("\n## Table 2: Systematic Single-Component (Individual) Augmentation Study\n\n")
        mf.write("*Evaluates independent operator efficacy by adding each operator in isolation to Clean Baseline.*\n\n")
        mf.write("| Augmentation Configuration | Applied Domain | Val Window Acc (%) | Val Loss | Test Window Acc (%) | Test Macro F1 | Delta vs. Baseline (Test) |\n")
        mf.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        base_single = summary_single.get("Clean_Baseline_NoAug", {}).get("win_acc_mean", 63.35)
        for var_name, _, disp_name, domain in SINGLE_COMPONENT_VARIANTS:
            if var_name in summary_single:
                s = summary_single[var_name]
                diff = s["win_acc_mean"] - base_single
                diff_str = f"+{diff:.2f}%" if diff > 0 else (f"{diff:.2f}%" if diff < 0 else "Ref (0.00%)")
                va = f"{s['val_acc_mean']:.2f}% ± {s['val_acc_std']:.2f}%" if s.get("val_acc_mean") else "--"
                vl = f"{s['val_loss_mean']:.4f} ± {s['val_loss_std']:.4f}" if s.get("val_loss_mean") else "--"
                mf.write(f"| {disp_name} | {domain} | {va} | {vl} | {s['win_acc_mean']:.2f}% ± {s['win_acc_std']:.2f}% | {s['win_f1_mean']:.4f} ± {s['win_f1_std']:.4f} | {diff_str} |\n")

        # Table 3: Cross-Backbones
        if summary_backbones:
            mf.write("\n## Table 3: Cross-Backbone Generalization (No-Aug vs. SkelGym-Aug)\n\n")
            mf.write("| Architecture / Stream | Setting | Test Window Acc (%) | Test Window Macro-F1 | Test Video Acc (%) | Test Video Macro-F1 |\n")
            mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for v in sorted(summary_backbones):
                s = summary_backbones[v]
                mf.write(f"| **{v}** | 3-seed Mean | {s['win_acc_mean']:.2f}% ± {s['win_acc_std']:.2f}% | {s['win_f1_mean']:.4f} ± {s['win_f1_std']:.4f} | {s['vid_acc_mean']:.2f}% ± {s['vid_acc_std']:.2f}% | {s['vid_f1_mean']:.4f} ± {s['vid_f1_std']:.4f} |\n")

    print(f"[SAVED] Comprehensive Markdown report written to {out_md}")

    # 7. Generate Standalone LaTeX Tables
    tex_loo = out_dir / "table_leave_one_out.tex"
    with open(tex_loo, "w", encoding="utf-8") as f:
        f.write(generate_loo_latex_table(summary_loo))
    print(f"[SAVED] Standalone LOO LaTeX table written to {tex_loo}")

    tex_single = out_dir / "table_single_component.tex"
    with open(tex_single, "w", encoding="utf-8") as f:
        f.write(generate_single_component_latex_table(summary_single))
    print(f"[SAVED] Standalone Single Component LaTeX table written to {tex_single}")

if __name__ == "__main__":
    main()
