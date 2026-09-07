"""
Script to create a clean, 100% disjoint, stratified dataset split from unique landmark files.
Eliminates all cross-split leakage and ensures balanced representation across 22 classes.
"""

import hashlib
import os
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def compute_file_hash(filepath: Path) -> str:
    """Computes MD5 hash of file content."""
    return hashlib.md5(filepath.read_bytes()).hexdigest()

def reorganize_splits_stratified(landmark_dir: str = "data/landmarks", backup: bool = True):
    """
    1. Collects all 803 unique CSV landmark files across existing train/val/test folders.
    2. Groups them by exercise class.
    3. Performs stratified random splitting (70% train, 15% val, 15% test).
    4. Writes files into clean, completely disjoint train/, val/, test/ directories.
    """
    random.seed(SEED)
    base_dir = Path(landmark_dir)

    print("Collecting all unique landmark files across existing splits...")
    unique_files_by_hash: Dict[str, Path] = {}
    class_to_files: Dict[str, List[Path]] = defaultdict(list)

    for split in ["train", "val", "test"]:
        p = base_dir / split
        if not p.exists():
            continue
        for f in p.glob("**/*.csv"):
            h = compute_file_hash(f)
            if h not in unique_files_by_hash:
                unique_files_by_hash[h] = f
                act_name = f.parent.name
                class_to_files[act_name].append(f)

    total_unique = len(unique_files_by_hash)
    print(f"Discovered {total_unique} unique landmark files across {len(class_to_files)} classes.")

    # Create temporary staging directory
    staging_dir = base_dir.parent / "landmarks_sanitized_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    split_counts = {"train": 0, "val": 0, "test": 0}

    for act_name, files in sorted(class_to_files.items()):
        # Sort files deterministically before shuffling with seed
        files = sorted(files, key=lambda x: x.name)
        random.shuffle(files)
        n = len(files)

        if n == 1:
            train_files = files
            val_files = []
            test_files = []
        elif n == 2:
            train_files = files[:1]
            val_files = files[1:2]
            test_files = []
        elif n <= 4:
            train_files = files[:-2]
            val_files = files[-2:-1]
            test_files = files[-1:]
        else:
            n_val = max(1, int(round(n * VAL_RATIO)))
            n_test = max(1, int(round(n * TEST_RATIO)))
            n_train = n - n_val - n_test

            train_files = files[:n_train]
            val_files = files[n_train:n_train + n_val]
            test_files = files[n_train + n_val:]

        for split_name, s_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dest_dir = staging_dir / split_name / act_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in s_files:
                dest_file = dest_dir / f.name
                # Avoid collision if two files had same name
                if dest_file.exists():
                    dest_file = dest_dir / f"{f.stem}_{compute_file_hash(f)[:6]}.csv"
                shutil.copy2(f, dest_file)
                split_counts[split_name] += 1

    print("\nStratified partition completed:")
    print(f"  - Train: {split_counts['train']} files ({split_counts['train']/total_unique*100:.1f}%)")
    print(f"  - Val:   {split_counts['val']} files ({split_counts['val']/total_unique*100:.1f}%)")
    print(f"  - Test:  {split_counts['test']} files ({split_counts['test']/total_unique*100:.1f}%)")

    # Replace old data/landmarks with sanitized staging
    if backup:
        backup_dir = base_dir.parent / "landmarks_backup_unclean"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        print(f"\nBacking up unclean data/landmarks to {backup_dir} ...")
        shutil.copytree(base_dir, backup_dir)

    print("Overwriting data/landmarks with clean disjoint splits...")
    shutil.rmtree(base_dir)
    shutil.copytree(staging_dir, base_dir)
    shutil.rmtree(staging_dir)
    print("Sanitization and reorganization completed successfully!")

def verify_splits(landmark_dir: str = "data/landmarks") -> bool:
    """Verifies that splits are 100% disjoint."""
    base_dir = Path(landmark_dir)
    train_h = {compute_file_hash(f) for f in (base_dir / "train").glob("**/*.csv")}
    val_h = {compute_file_hash(f) for f in (base_dir / "val").glob("**/*.csv")}
    test_h = {compute_file_hash(f) for f in (base_dir / "test").glob("**/*.csv")}

    tv_overlap = len(train_h & val_h)
    tt_overlap = len(train_h & test_h)
    vt_overlap = len(val_h & test_h)

    print(f"\nSplit Verification:")
    print(f"  - Train unique: {len(train_h)}")
    print(f"  - Val unique:   {len(val_h)}")
    print(f"  - Test unique:  {len(test_h)}")
    print(f"  - Train vs Val overlap:  {tv_overlap}")
    print(f"  - Train vs Test overlap: {tt_overlap}")
    print(f"  - Val vs Test overlap:   {vt_overlap}")

    is_clean = (tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0)
    print("  -> Status:", "CLEAN (100% Disjoint)" if is_clean else "LEAKAGE DETECTED")
    return is_clean

if __name__ == "__main__":
    reorganize_splits_stratified()
    verify_splits()
