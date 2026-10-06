#!/usr/bin/env python3
"""
Sync newly trained checkpoints and report artifacts to Hugging Face Hub.
Repo: Cuong2004/gym-exercise-classification
"""

import os
import sys
import glob
from pathlib import Path
from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parent.parent

HF_REPO = "Cuong2004/gym-exercise-classification"
HF_TOKEN = os.environ.get("HF_TOKEN")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sync checkpoints and reports to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=HF_TOKEN, help="Hugging Face User Access Token")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[Error] Hugging Face token must be provided via --token or HF_TOKEN env var.")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Connecting to Hugging Face Hub repo: {HF_REPO}...")

    # 1. Sync Report and Artifact Files in a Single Batch Commit
    from huggingface_hub import CommitOperationAdd
    operations = []
    reports = [
        "outputs/RESULTS_FINAL.md",
        "outputs/MASTER_BENCHMARK_MATRIX.md",
        "outputs/augmentation_ablation_results.json",
        "outputs/table2_clean_sequences.json",
        "outputs/table3_graph_streams_results.json",
        "outputs/table4_loo_world_mix_v2.json",
        "outputs/table5_single_world_mix_v2.json",
        "outputs/table6_cross_paradigm_fusion.json",
        "outputs/table7_unified_benchmark.json",
        "outputs/statistical_tests_report.json",
        "outputs/bootstrap_confidence_intervals.json",
        "outputs/per_class_results.json",
        "outputs/world_3d_benchmark_results.json",
        "artifacts/results/canonical_results_v2.json",
        "configs/augmentation/skelgym_aug_v2.yaml",
    ]
    for r in reports:
        p = ROOT_DIR / r
        if p.exists():
            print(f"Queueing {r} ({p.stat().st_size} bytes)...")
            operations.append(CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)))

    if operations:
        print(f"Committing {len(operations)} reports in a single atomic commit...")
        api.create_commit(
            repo_id=HF_REPO,
            repo_type="model",
            operations=operations,
            commit_message="Sync benchmark reports and JSON results"
        )

    # 2. Sync Checkpoint Directories via upload_folder (1 commit per directory)
    ckpt_dirs = [
        ROOT_DIR / "checkpoints" / "table2",
        ROOT_DIR / "checkpoints" / "ablation_v2",
        ROOT_DIR / "checkpoints" / "graph_streams",
    ]
    for cdir in ckpt_dirs:
        if cdir.exists():
            rel_dir = str(cdir.relative_to(ROOT_DIR))
            print(f"Uploading folder: {rel_dir} via upload_folder (single commit)...")
            api.upload_folder(
                folder_path=str(cdir),
                path_in_repo=rel_dir,
                repo_id=HF_REPO,
                repo_type="model",
                commit_message=f"Sync {rel_dir} checkpoints"
            )

    print("\n[SUCCESS] All reports and checkpoints successfully uploaded to Hugging Face Hub!")

if __name__ == "__main__":
    main()
