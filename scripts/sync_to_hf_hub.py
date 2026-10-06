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

    # 1. Sync Report and Artifact Files
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
        "artifacts/results/canonical_results_v2.json",
        "configs/augmentation/skelgym_aug_v2.yaml",
    ]
    for r in reports:
        p = ROOT_DIR / r
        if p.exists():
            print(f"Uploading {r} ({p.stat().st_size} bytes)...")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=r,
                repo_id=HF_REPO,
                repo_type="model"
            )

    # 2. Sync Checkpoint Directories
    ckpt_dirs = [
        ROOT_DIR / "checkpoints" / "table2",
        ROOT_DIR / "checkpoints" / "ablation_v2",
        ROOT_DIR / "checkpoints" / "graph_streams",
    ]
    for cdir in ckpt_dirs:
        if cdir.exists():
            for fp in sorted(cdir.rglob("*")):
                if fp.is_file() and fp.suffix in [".pt", ".json"]:
                    rel_p = fp.relative_to(ROOT_DIR)
                    print(f"Uploading checkpoint: {rel_p} ({fp.stat().st_size/1e6:.2f} MB)...")
                    api.upload_file(
                        path_or_fileobj=str(fp),
                        path_in_repo=str(rel_p),
                        repo_id=HF_REPO,
                        repo_type="model"
                    )

    print("\n[SUCCESS] All reports and checkpoints successfully uploaded to Hugging Face Hub!")

if __name__ == "__main__":
    main()
