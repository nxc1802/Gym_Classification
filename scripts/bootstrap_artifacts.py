#!/usr/bin/env python3
"""
SkelGym Bootstrap Artifacts & Fresh-Clone Verification Tool.
Enables 1-command environment bootstrap on a fresh git checkout:
  1. Ensures data/Final_dataset_metadata.csv is in place (from root or HF Hub)
  2. Ensures data/landmarks/ has the complete MediaPipe dataset (from HF Hub or local)
  3. Verifies or downloads pretrained model checkpoints (from HF Model Hub or archive)
  4. Validates split counts, video counts, and landmark directory integrity
"""

import os
import sys
import shutil
import tarfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.hf_hub import (
    DEFAULT_MODEL_REPO,
    DEFAULT_DATASET_REPO,
    pull_landmarks_from_hf,
    ensure_checkpoint_available,
    get_hf_token
)

CORE_CHECKPOINTS = [
    "checkpoints/best_Transformer_T2.2_mix.pt",
    "checkpoints/best_AAGCN_T4.2_bone_3d.pt",
    "checkpoints/best_AAGCN_T4.3_rel_3d.pt",
    "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt",
    "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt"
]

def check_metadata(auto_fix: bool = True) -> bool:
    """Verifies that data/Final_dataset_metadata.csv exists and is populated."""
    target = PROJECT_ROOT / "data" / "Final_dataset_metadata.csv"
    root_source = PROJECT_ROOT / "Final_dataset_metadata.csv"

    if target.exists() and target.stat().st_size > 1000:
        print(f"✅ Metadata present: {target.relative_to(PROJECT_ROOT)} ({target.stat().st_size:,} bytes)")
        return True

    if not auto_fix:
        print(f"❌ Missing metadata: {target.relative_to(PROJECT_ROOT)}")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    if root_source.exists() and root_source.stat().st_size > 1000:
        print(f"ℹ️ Copying root metadata to {target.relative_to(PROJECT_ROOT)}...")
        shutil.copy2(root_source, target)
        print(f"✅ Metadata synced: {target.relative_to(PROJECT_ROOT)}")
        return True

    print(f"ℹ️ Downloading metadata from Hugging Face dataset repo ({DEFAULT_DATASET_REPO})...")
    try:
        from huggingface_hub import hf_hub_download
        tok = get_hf_token()
        local_meta = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename="Final_dataset_metadata.csv",
            token=tok
        )
        shutil.copy2(local_meta, target)
        print(f"✅ Metadata downloaded successfully: {target.relative_to(PROJECT_ROOT)}")
        return True
    except Exception as e:
        print(f"❌ Could not download metadata: {e}")
        return False

def check_landmarks(auto_fix: bool = True) -> bool:
    """Verifies that data/landmarks has train, val, and test splits with CSVs."""
    landmarks_dir = PROJECT_ROOT / "data" / "landmarks"
    splits = ["train", "val", "test"]
    
    if landmarks_dir.exists():
        csv_count = len(list(landmarks_dir.glob("*/*/*.csv"))) + len(list(landmarks_dir.glob("*/*.csv")))
        has_all_splits = all((landmarks_dir / s).exists() for s in splits)
        if has_all_splits and csv_count > 100:
            print(f"✅ Landmarks present: {csv_count} CSV files found across splits in data/landmarks/")
            return True

    if not auto_fix:
        print(f"❌ Incomplete landmarks directory in data/landmarks/")
        return False

    print(f"ℹ️ Pulling full landmarks dataset from Hugging Face ({DEFAULT_DATASET_REPO})...")
    try:
        pull_landmarks_from_hf(dest_dir=str(landmarks_dir), repo_id=DEFAULT_DATASET_REPO, pull_metadata=True)
        csv_count = len(list(landmarks_dir.glob("*/*/*.csv"))) + len(list(landmarks_dir.glob("*/*.csv")))
        print(f"✅ Landmarks extracted successfully: {csv_count} CSV files in data/landmarks/")
        return True
    except Exception as e:
        print(f"❌ Failed to download landmarks from Hugging Face: {e}")
        return False

def check_checkpoints(auto_fix: bool = False, from_archive: bool = False) -> Tuple[int, int]:
    """Checks presence of core ensemble checkpoints."""
    ckpts_dir = PROJECT_ROOT / "checkpoints"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    archive_path = PROJECT_ROOT / "archive_pre_clean_results.tar.gz"

    if from_archive and archive_path.exists():
        print(f"ℹ️ Restoring checkpoints from local archive: {archive_path.name}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("checkpoints/") and m.name.endswith(".pt")]
            tar.extractall(path=PROJECT_ROOT, members=members)
        print(f"✅ Extracted {len(members)} checkpoints from archive.")

    present_count = 0
    missing_ckpts = []

    for rel_p in CORE_CHECKPOINTS:
        p = PROJECT_ROOT / rel_p
        if p.exists() and p.stat().st_size > 1000:
            present_count += 1
        else:
            missing_ckpts.append(rel_p)

    if present_count == len(CORE_CHECKPOINTS):
        print(f"✅ All {len(CORE_CHECKPOINTS)} core ensemble checkpoints present in checkpoints/")
        return present_count, len(CORE_CHECKPOINTS)

    print(f"⚠️ {present_count}/{len(CORE_CHECKPOINTS)} core checkpoints found locally.")
    if auto_fix:
        for rel_p in missing_ckpts:
            p = PROJECT_ROOT / rel_p
            print(f"ℹ️ Attempting download of {p.name} from Hugging Face...")
            ensure_checkpoint_available(str(p))
            if p.exists():
                present_count += 1

    return present_count, len(CORE_CHECKPOINTS)

def verify_dataset_integrity() -> bool:
    """Validates video numbers and split definitions in metadata."""
    meta_path = PROJECT_ROOT / "data" / "Final_dataset_metadata.csv"
    if not meta_path.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(meta_path)
        total_rows = len(df)
        splits = df["split"].value_counts().to_dict() if "split" in df.columns else {}
        print(f"📊 Dataset Metadata Verification:")
        print(f"   - Total indexed rows: {total_rows}")
        for s, c in splits.items():
            print(f"   - Split '{s}': {c} entries")
        return total_rows > 0
    except Exception as e:
        print(f"❌ Error verifying dataset metadata: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Bootstrap SkelGym artifacts for fresh clone reproducibility.")
    parser.add_argument("--check-only", action="store_true", help="Only verify existing artifacts without downloading")
    parser.add_argument("--download-checkpoints", action="store_true", help="Download missing checkpoints from Hugging Face")
    parser.add_argument("--from-archive", action="store_true", help="Restore checkpoints from archive_pre_clean_results.tar.gz")
    args = parser.parse_args()

    print("=" * 70)
    print("SKELGYM ARTIFACT BOOTSTRAP & REPRODUCIBILITY VERIFICATION")
    print(f"Workspace: {PROJECT_ROOT}")
    print("=" * 70)

    auto_fix = not args.check_only

    # 1. Metadata
    meta_ok = check_metadata(auto_fix=auto_fix)

    # 2. Landmarks
    landmarks_ok = check_landmarks(auto_fix=auto_fix)

    # 3. Checkpoints
    ckpts_present, ckpts_total = check_checkpoints(
        auto_fix=args.download_checkpoints,
        from_archive=args.from_archive
    )

    # 4. Dataset Integrity
    data_valid = verify_dataset_integrity()

    print("\n" + "=" * 70)
    print("BOOTSTRAP SUMMARY REPORT")
    print(f"  - Metadata:    {'✅ READY' if meta_ok else '❌ MISSING'}")
    print(f"  - Landmarks:   {'✅ READY' if landmarks_ok else '❌ MISSING'}")
    print(f"  - Checkpoints: {ckpts_present}/{ckpts_total} present")
    print(f"  - Integrity:   {'✅ PASSED' if data_valid else '❌ FAILED'}")
    print("=" * 70)

    all_ready = meta_ok and landmarks_ok and data_valid
    if all_ready:
        print("🎉 Ready for data processing and training!")
        if ckpts_present < ckpts_total:
            print("💡 Note: To evaluate pre-trained checkpoints without re-training, run:")
            print("   python scripts/bootstrap_artifacts.py --download-checkpoints")
            print("   or if local archive exists: python scripts/bootstrap_artifacts.py --from-archive")
        return 0
    else:
        print("⚠️ Some artifacts could not be verified. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
