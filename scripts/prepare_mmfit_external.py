#!/usr/bin/env python3
"""
Prepares and audits the MM-Fit external benchmark dataset.
Runs Quality-Control gate, verifies joint integrity, validates canonical 13-joint mapping,
and outputs external dataset audit reports.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.external.mmfit import MMFitExternalDataset
from src.external.base import QualityControlGate
from src.constants import ACTIONS

def main():
    parser = argparse.ArgumentParser(description="Prepare and audit MM-Fit dataset.")
    parser.add_argument("--root", type=str, default="mm-fit", help="Path to MM-Fit root directory")
    parser.add_argument("--class-set", type=str, default="core4", choices=["core4", "extended5"])
    parser.add_argument("--pose-source", type=str, default="native", choices=["native", "mediapipe"])
    parser.add_argument("--out-dir", type=str, default="outputs/external/mmfit", help="Output directory")
    args = parser.parse_args()

    out_p = PROJECT_ROOT / args.out_dir
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"[MM-Fit Prepare] Loading dataset from: {args.root}")
    print(f"  Class set: {args.class_set} | Pose source: {args.pose_source}")

    qc = QualityControlGate(
        max_nan_ratio=0.20,
        max_zero_ratio=0.20,
        min_frames=16,
        min_pose_success=0.80
    )

    # 1. Unseen Subjects (Primary Test Set)
    print("\n--- 1. Auditing Unseen-Subject Test Set (w00, w05, w12, w13, w20) ---")
    ds_unseen = MMFitExternalDataset(
        root_dir=args.root,
        split_group="unseen_test",
        class_set=args.class_set,
        pose_source=args.pose_source,
        qc_gate=qc,
        apply_geometric_norm=(args.pose_source == "native")
    )
    audit_unseen_file = out_p / "audit_unseen_subjects.csv"
    df_audit_unseen = ds_unseen.export_audit_report(audit_unseen_file)
    print(f"  Accepted records: {len(ds_unseen.records)} / {len(df_audit_unseen)}")
    print(f"  Audit report written to: {audit_unseen_file}")

    # 2. Seen Subjects (Secondary Test Set)
    print("\n--- 2. Auditing Seen-Subject Test Set (w09, w10, w11) ---")
    ds_seen = MMFitExternalDataset(
        root_dir=args.root,
        split_group="seen_test",
        class_set=args.class_set,
        pose_source=args.pose_source,
        qc_gate=qc,
        apply_geometric_norm=(args.pose_source == "native")
    )
    audit_seen_file = out_p / "audit_seen_subjects.csv"
    df_audit_seen = ds_seen.export_audit_report(audit_seen_file)
    print(f"  Accepted records: {len(ds_seen.records)} / {len(df_audit_seen)}")
    print(f"  Audit report written to: {audit_seen_file}")

    # Print class distribution breakdown
    print("\n--- Unseen Test Set Class Distribution ---")
    unseen_counts = {}
    for r in ds_unseen.records:
        unseen_counts[r.action_canonical] = unseen_counts.get(r.action_canonical, 0) + 1
    for act, cnt in sorted(unseen_counts.items()):
        print(f"  {act:<25}: {cnt} action segments")

    print("\nMM-Fit dataset preparation and quality verification complete!")

if __name__ == "__main__":
    main()
