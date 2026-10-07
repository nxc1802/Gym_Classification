#!/usr/bin/env python3
"""
Phase 8-10: Statistical Testing, Cluster-Aware Bootstrap, Per-Class Breakdown, and Canonical SOT Generation.
Generates:
  - Table 7: Unified Window vs Video Multi-Seed Benchmark + Audited Parameter Counts
  - Table 8: Statistical Hypothesis Tests (McNemar with Edwards correction, Wilcoxon, Paired t-test, FDR correction)
  - Table 9: Video-Cluster-Aware Bootstrap (95% CI across video clusters, B=1000)
  - Table 10: Per-Class Granular Breakdown (Precision, Recall, F1 for 7 action classes)
  - Canonical Single Source of Truth: artifacts/results/canonical_results_v2.json
  - Updated Reports: outputs/RESULTS_FINAL.md & outputs/MASTER_BENCHMARK_MATRIX.md
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX
from src.utils.statistics import (
    mcnemar_test,
    cluster_bootstrap_window,
    paired_video_confidence_test,
    adjust_p_values,
    format_p_value,
    get_significance_stars,
)

def send_marimo_toast(msg: str, *args, **kwargs):
    try:
        import marimo as mo
        kind = kwargs.get("kind", "info")
        mo.status.toast(msg, kind=kind)
    except Exception:
        pass

# Trainable parameters audited per architecture
AUDITED_PARAMS = {
    "LSTM": "362K",
    "BiLSTM": "360K",
    "Transformer": "301K",
    "ST-GCN": "350K",
    "AAGCN": "378K",
    "SkelGym-Lite": "679K",
    "Four-Stream AAGCN": "1.51M",
    "SkelGym-Full": "1.81M"
}

def generate_table7(t2: Dict, t3: Dict, t4: Dict, t6: Dict) -> Dict[str, Any]:
    """
    Builds Table 7: Unified Master Benchmark across all canonical architectures.
    """
    rows = []

    # 1. Baselines from Table 2 (Clean Sequences)
    if "LSTM_world_3d" in t2:
        r = t2["LSTM_world_3d"]
        rows.append({
            "model": "LSTM (World 3D Clean)",
            "params": AUDITED_PARAMS["LSTM"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "LSTM_mix_v2" in t2:
        r = t2["LSTM_mix_v2"]
        rows.append({
            "model": "LSTM (Biomechanical Mix v2)",
            "params": AUDITED_PARAMS["LSTM"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "BiLSTM_mix_v2" in t2:
        r = t2["BiLSTM_mix_v2"]
        rows.append({
            "model": "BiLSTM (Biomechanical Mix v2)",
            "params": AUDITED_PARAMS["BiLSTM"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })

    # 2. Transformer from Table 4 (Clean & Proposed)
    if "clean" in t4:
        r = t4["clean"]
        rows.append({
            "model": "Transformer (Biomechanical Mix v2 Clean)",
            "params": AUDITED_PARAMS["Transformer"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "pair_mirror_yaw" in t5:
        r = t5["pair_mirror_yaw"]
        rows.append({
            "model": "Transformer (Biomechanical Mix v2 + SkelGym-Aug)",
            "params": AUDITED_PARAMS["Transformer"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    elif "candidate_minus_time" in t4:
        r = t4["candidate_minus_time"]
        rows.append({
            "model": "Transformer (Biomechanical Mix v2 + SkelGym-Aug)",
            "params": AUDITED_PARAMS["Transformer"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })

    # 3. Graph Streams from Table 3
    if "T3.1" in t3:
        r = t3["T3.1"]
        rows.append({
            "model": "ST-GCN (Raw 3D Clean)",
            "params": AUDITED_PARAMS["ST-GCN"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "T3.4" in t3:
        r = t3["T3.4"]
        rows.append({
            "model": "AAGCN (Bone 3D + SkelGym-Aug)",
            "params": AUDITED_PARAMS["AAGCN"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "T3.9" in t3:
        r = t3["T3.9"]
        rows.append({
            "model": "Four-Stream AAGCN (Uniform Soft Voting)",
            "params": AUDITED_PARAMS["Four-Stream AAGCN"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })

    # 4. Ensembles from Table 6 (Validation Winners)
    winners = t6.get("winners", {})
    w_lite = winners.get("SkelGym-Lite", "uniform_soft")
    w_full = winners.get("SkelGym-Full", "stacking")

    if "SkelGym-Lite" in t6 and w_lite in t6["SkelGym-Lite"]:
        r = t6["SkelGym-Lite"][w_lite]
        desc = r.get("description", f"Lite ({w_lite})")
        rows.append({
            "model": f"SkelGym-Lite (2 Streams, {desc})",
            "params": AUDITED_PARAMS["SkelGym-Lite"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })
    if "SkelGym-Full" in t6 and w_full in t6["SkelGym-Full"]:
        r = t6["SkelGym-Full"][w_full]
        desc = r.get("description", f"Full ({w_full})")
        rows.append({
            "model": f"SkelGym-Full (5 Streams, {desc})",
            "params": AUDITED_PARAMS["SkelGym-Full"],
            "val_win_acc": r["val_win_acc"],
            "val_vid_acc": r["val_vid_acc"],
            "test_win_acc": r["test_win_acc"],
            "test_win_f1": r["test_win_f1"],
            "test_vid_acc": r["test_vid_acc"],
            "test_vid_f1": r["test_vid_f1"]
        })

    return {"table7_rows": rows}

def generate_markdown_table7(t7_data: Dict[str, Any]) -> str:
    lines = [
        "| Architecture / Model | Params | Val Vid Acc (%) | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for r in t7_data["table7_rows"]:
        lines.append(f"| **{r['model']}** | {r['params']} | {r['val_vid_acc']} | {r['test_win_acc']} | {r['test_win_f1']} | **{r['test_vid_acc']}** | **{r['test_vid_f1']}** |")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Phase 8-10: Evaluation & Reports Engine")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 8-10: Comprehensive Evaluation, Statistical Inference, and SOT Compilation")
    print("=" * 80)

    outputs_dir = ROOT_DIR / "outputs"
    artifacts_dir = ROOT_DIR / "artifacts" / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    t2_file = outputs_dir / "table2_clean_sequences.json"
    t3_file = outputs_dir / "table3_graph_streams_results.json"
    t4_file = outputs_dir / "table4_loo_world_mix_v2.json"
    t5_file = outputs_dir / "table5_single_world_mix_v2.json"
    t6_file = outputs_dir / "table6_cross_paradigm_fusion.json"

    # Verify input availability
    missing = []
    for f, name in [(t2_file, "Table 2"), (t3_file, "Table 3"), (t4_file, "Table 4"), (t5_file, "Table 5"), (t6_file, "Table 6")]:
        if not f.exists():
            missing.append(name)

    if missing:
        print(f"Warning: The following table files are not yet generated: {missing}")
        print("Run prior phases before generating full reports.")
        return

    with open(t2_file) as f: t2 = json.load(f)
    with open(t3_file) as f: t3 = json.load(f)
    with open(t4_file) as f: t4 = json.load(f)
    with open(t5_file) as f: t5 = json.load(f)
    with open(t6_file) as f: t6 = json.load(f)

    # 1. Compile Table 7
    print("\n---> Compiling Table 7 (Unified Master Benchmark Matrix)...")
    t7 = generate_table7(t2, t3, t4, t6)
    with open(outputs_dir / "table7_unified_benchmark.json", "w") as f:
        json.dump(t7, f, indent=2)
    print(generate_markdown_table7(t7))

    # 2. Compile Master SOT JSON
    canonical_sot = {
        "metadata": {
            "title": "SkelGym: Cross-Paradigm Biomechanical Skeleton Action Recognition",
            "feature_version": "mix_v2_world_63d",
            "seeds": [42, 123, 3407],
            "dataset": "Final_dataset_metadata.csv",
            "classes": ACTIONS,
            "audited_params": AUDITED_PARAMS
        },
        "table2_clean_sequences": t2,
        "table3_graph_streams": t3,
        "table4_leave_one_out": t4,
        "table5_single_operator": t5,
        "table6_cross_paradigm_fusion": t6,
        "table7_master_benchmark": t7
    }

    sot_path = artifacts_dir / "canonical_results_v2.json"
    with open(sot_path, "w") as f:
        json.dump(canonical_sot, f, indent=2)
    # 3. Run Table 8, 9, 10 Statistical Tests & Bootstrap
    print("\n---> Running Table 8, 9, 10 Statistical Hypothesis Testing and Bootstrap...")
    cmd_t810 = [sys.executable, "-u", str(ROOT_DIR / "scripts" / "run_tables8_9_10_on_server.py"), "--device", args.device]
    res_t810 = subprocess.run(cmd_t810)
    if res_t810.returncode != 0:
        print(f"Warning: run_tables8_9_10_on_server.py exited with code {res_t810.returncode}")

    send_marimo_toast("Phase 8-10 Complete: Canonical SOT & Reports successfully generated!")

if __name__ == "__main__":
    main()
