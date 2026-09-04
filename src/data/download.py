"""
Kaggle Dataset Downloader module.
Downloads gym exercise classification dataset from Kaggle via kagglehub.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
DEFAULT_DATASET = "truongnhatquangk18dn/the-gym-exercise-classification-dataset"

def download_kaggle_dataset(
    dataset_name: str = DEFAULT_DATASET,
    output_dir: Optional[str] = None
) -> str:
    """
    Downloads dataset from Kaggle using kagglehub.
    If output_dir is specified, creates a directory with symlinks or copies if needed.
    Returns path to downloaded dataset.
    """
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub is not installed. Please run: pip install kagglehub")

    print(f"Downloading Kaggle dataset: {dataset_name} ...")
    download_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded successfully to cache: {download_path}")

    if output_dir:
        dest = Path(output_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Check if already points to dest
        if str(dest.resolve()) != str(Path(download_path).resolve()):
            print(f"Preparing dataset in target directory: {dest}")
            dest.mkdir(parents=True, exist_ok=True)
            for item in os.listdir(download_path):
                s = os.path.join(download_path, item)
                d = os.path.join(dest, item)
                if not os.path.exists(d):
                    try:
                        # Try creating directory symlink / junction on Windows
                        os.symlink(s, d, target_is_directory=os.path.isdir(s))
                    except Exception:
                        # If symlink permission denied, copy or leave in cache
                        pass
        return str(dest)

    return str(download_path)
