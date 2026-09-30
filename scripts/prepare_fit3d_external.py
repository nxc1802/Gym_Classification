#!/usr/bin/env python3
"""
Prepares and audits the Fit3D external benchmark dataset.
Enforces 50 Hz -> 30 Hz temporal resampling, joint mapping, and repetition parsing.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.external.fit3d import Fit3DExternalDataset
from src.external.base import QualityControlGate

def main():
    parser = argparse.ArgumentParser(description="Prepare and audit Fit3D dataset.")
    parser.add_argument("--root", type=str, default="data_external/fit3d", help="Path to Fit3D root directory")
    parser.add_argument("--class-set", type=str, default="core6", choices=["core6", "extended7"])
    parser.add_argument("--pose-source", type=str, default="native", choices=["native", "mediapipe"])
    parser.add_argument("--out-dir", type=str, default="outputs/external/fit3d", help="Output directory")
    args = parser.parse_args()

    out_p = PROJECT_ROOT / args.out_dir
    out_p.mkdir(parents=True, exist_ok=True)

    print(f"[Fit3D Prepare] Loading dataset from: {args.root}")
    print(f"  Class set: {args.class_set} | Pose source: {args.pose_source}")

    qc = QualityControlGate(
        max_nan_ratio=0.20,
        max_zero_ratio=0.20,
        min_frames=16,
        min_pose_success=0.80
    )

    fit3d_dir = Path(args.root)
    if not fit3d_dir.exists():
        fit3d_dir.mkdir(parents=True, exist_ok=True)
        readme_path = fit3d_dir / "README.md"
        readme_content = """# Fit3D Dataset Acquisition Protocol

Fit3D requires user registration and license agreement under non-commercial research terms.
Official portal: https://fit3d.epfl.ch

### Acquisition Instructions:
1. Register and download the Fit3D dataset (Train + Test sets).
2. Extract subjects (s02, s03, s04, s05, s07, s08, s09, s10, s11) into this directory:
   data_external/fit3d/
   ├── s02/
   │   ├── squat/
   │   │   ├── joints3d_25.npy
   │   │   └── rep_annotations.json
   ...
3. Run:
   python scripts/prepare_fit3d_external.py --root data_external/fit3d
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)
        print(f"Fit3D directory initialized with instructions at: {readme_path}")
        return

    ds = Fit3DExternalDataset(
        root_dir=args.root,
        class_set=args.class_set,
        pose_source=args.pose_source,
        qc_gate=qc,
        apply_geometric_norm=True
    )

    audit_file = out_p / "audit_fit3d.csv"
    df_audit = ds.export_audit_report(audit_file)
    print(f"  Accepted records: {len(ds.records)} / {len(df_audit)}")
    print(f"  Audit report written to: {audit_file}")

if __name__ == "__main__":
    main()
