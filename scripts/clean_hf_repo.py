#!/usr/bin/env python3
"""
Hugging Face Repository Cleaner for SkelGym.
Safely purges old checkpoints, outputs, and plots from the model repository:
    `Cuong2004/gym-exercise-classification`
While strictly PROTECTING the dataset repository:
    `Cuong2004/gym-exercise-landmarks` (MUST NEVER BE DELETED OR TOUCHED).

Usage:
    python scripts/clean_hf_repo.py --token <HF_TOKEN> [--dry_run] [--force]
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from huggingface_hub import HfApi, get_token
except ImportError:
    print("Error: huggingface_hub is not installed. Run `pip install huggingface_hub`.")
    sys.exit(1)

PROTECTED_DATASET_REPO = "Cuong2004/gym-exercise-landmarks"
TARGET_MODEL_REPO = "Cuong2004/gym-exercise-classification"

def main():
    parser = argparse.ArgumentParser(description="Clean old artifacts from Hugging Face model repository")
    parser.add_argument("--repo_id", type=str, default=TARGET_MODEL_REPO, help="Target HF repo to clean")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token with write permission")
    parser.add_argument("--dry_run", action="store_true", help="List files to delete without actually deleting")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("\n[ERROR] No Hugging Face token found!")
        print("Please provide your token via:")
        print("  1. Argument: python scripts/clean_hf_repo.py --token <YOUR_HF_TOKEN>")
        print("  2. Environment: export HF_TOKEN=<YOUR_HF_TOKEN>")
        print("  3. CLI login: huggingface-cli login\n")
        sys.exit(1)

    # SAFETY CHECK: Never allow cleaning the protected dataset repo
    if args.repo_id == PROTECTED_DATASET_REPO or "landmarks" in args.repo_id.lower():
        print(f"\n[FATAL ERROR] {args.repo_id} is a PROTECTED DATASET REPOSITORY!")
        print("Refusing to delete files from the landmark dataset repository.")
        sys.exit(1)

    api = HfApi(token=token)

    try:
        user_info = api.whoami()
        print(f"Authenticated as user: {user_info.get('name', 'Unknown')}")
    except Exception as e:
        print(f"[ERROR] Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)

    print(f"\nScanning files in target model repository: {args.repo_id}...")
    try:
        files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    except Exception as e:
        print(f"[ERROR] Could not list files for {args.repo_id}: {e}")
        sys.exit(1)

    # Exclude git system files like .gitattributes
    files_to_delete = [f for f in files if f not in [".gitattributes", ".gitignore"]]
    print(f"Found {len(files_to_delete)} files to clean out of {len(files)} total files.")

    if not files_to_delete:
        print("Repository is already clean. Nothing to delete.")
        return

    print("\nSummary of files to delete by directory:")
    dirs = {}
    for f in files_to_delete:
        top = f.split("/")[0] if "/" in f else "root"
        dirs[top] = dirs.get(top, 0) + 1
    for k, v in sorted(dirs.items()):
        print(f"  - {k}/: {v} files")

    if args.dry_run:
        print("\n[DRY RUN] No files were deleted.")
        return

    if not args.force:
        confirm = input(f"\nAre you sure you want to permanently delete {len(files_to_delete)} files from {args.repo_id}? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Operation cancelled by user.")
            return

    print(f"\nDeleting {len(files_to_delete)} files from {args.repo_id}...")
    from huggingface_hub import CommitOperationDelete

    operations = [CommitOperationDelete(path_in_repo=f) for f in files_to_delete]
    
    # Process in chunks of 100 commits to avoid API payload limits
    chunk_size = 100
    for i in range(0, len(operations), chunk_size):
        chunk = operations[i:i+chunk_size]
        print(f"  Committing batch {i+1} to {min(i+chunk_size, len(operations))}...")
        api.create_commit(
            repo_id=args.repo_id,
            repo_type="model",
            operations=chunk,
            commit_message=f"Clean old artifacts batch ({i+1}-{min(i+chunk_size, len(operations))}) before full retraining"
        )

    print(f"\n[SUCCESS] Successfully purged all old artifacts from {args.repo_id}!")
    print(f"Dataset repository {PROTECTED_DATASET_REPO} remains untouched and safe.\n")

if __name__ == "__main__":
    main()
