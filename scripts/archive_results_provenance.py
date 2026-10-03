#!/usr/bin/env python3
"""
Archive Machine-Readable Results & Provenance Manifest.
Saves all experiment JSON artifacts into artifacts/results/ (tracked by git)
with full cryptographic provenance: commit SHA, dataset SHA256, normalization SHAs,
checkpoint SHAs, and random seeds.
"""

import os
import sys
import json
import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hexadecimal digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_info() -> Dict[str, str]:
    """Retrieves current git commit, branch, and status."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        commit = "unknown"
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    except Exception:
        branch = "unknown"
    return {"commit": commit, "branch": branch}

def archive_provenance(output_dir: str = "artifacts/results") -> Path:
    target_dir = PROJECT_ROOT / output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    result_files = [
        "table1_feature_screening.json",
        "graph_streams_results.json",
        "augmentation_ablation_results.json",
        "multi_seed_evaluation_results.json",
        "consensus_gains.json",
        "statistical_tests_report.json",
        "bootstrap_confidence_intervals.json",
        "per_class_results.json",
        "hardware_latency.json",
        "external_benchmark_results.json",
        "canonical_eval_predictions.npz",
        "bootstrap_confidence_intervals.md",
        "statistical_tests_report.md"
    ]

    copied_artifacts = {}
    for fname in result_files:
        src = PROJECT_ROOT / "outputs" / fname
        if src.exists():
            dst = target_dir / fname
            shutil.copy2(src, dst)
            copied_artifacts[fname] = {
                "size_bytes": dst.stat().st_size,
                "sha256": compute_sha256(dst)
            }
            print(f"✅ Archived: {fname} -> {dst.relative_to(PROJECT_ROOT)} ({dst.stat().st_size:,} bytes)")
        else:
            print(f"⚠️ Source missing: {src.relative_to(PROJECT_ROOT)}")

    # Checkpoint hashes
    checkpoint_hashes = {}
    chk_dir = PROJECT_ROOT / "checkpoints"
    if chk_dir.exists():
        for pt_file in sorted(chk_dir.glob("**/*.pt")):
            if not pt_file.name.startswith("._") and "smoke_test" not in pt_file.parts:
                rel = str(pt_file.relative_to(chk_dir))
                checkpoint_hashes[rel] = {
                    "size_bytes": pt_file.stat().st_size,
                    "sha256": compute_sha256(pt_file)
                }

    # Normalization hashes (all seeds)
    norm_hashes = {}
    ref_dir = PROJECT_ROOT / "artifacts" / "reference"
    if ref_dir.exists():
        for npz in sorted(ref_dir.glob("**/*.npz")):
            rel = str(npz.relative_to(ref_dir))
            norm_hashes[rel] = {
                "size_bytes": npz.stat().st_size,
                "sha256": compute_sha256(npz)
            }

    # Dataset metadata hash
    meta_path = PROJECT_ROOT / "data" / "Final_dataset_metadata.csv"
    meta_hash = compute_sha256(meta_path) if meta_path.exists() else None

    git_current = get_git_info()
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_current,
        "provenance_commits": {
            "training_commit": "cb288b6",
            "evaluation_commit": git_current.get("commit"),
            "report_generation_commit": git_current.get("commit")
        },
        "dataset_metadata_sha256": meta_hash,
        "seeds": [42, 123, 3407],
        "normalization_artifacts": norm_hashes,
        "archived_result_artifacts": copied_artifacts,
        "checkpoint_provenance": checkpoint_hashes
    }

    manifest_path = target_dir / "provenance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n✅ Provenance manifest written to: {manifest_path.relative_to(PROJECT_ROOT)}")
    return target_dir

if __name__ == "__main__":
    archive_provenance()
