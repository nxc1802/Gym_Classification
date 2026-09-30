import os
import sys
import json
import random
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Sets deterministic seed across Python, NumPy, and PyTorch for full reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def get_git_commit_hash(short: bool = True) -> str:
    """
    Returns current git commit hash, or 'unknown' if not in a git repository.
    """
    try:
        cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
        result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"

def is_git_repo_dirty() -> bool:
    """
    Returns True if git working tree has uncommitted modifications, False otherwise.
    """
    try:
        cmd = ["git", "status", "--porcelain"]
        result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except Exception:
        return False

def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """
    Computes SHA-256 checksum of a file in 64KB blocks.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for sha256 computation: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def save_provenance_metadata(metadata: Dict[str, Any], json_path: Union[str, Path]) -> Path:
    """
    Saves a JSON provenance metadata sidecar file alongside checkpoints.
    """
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    temp_path.replace(path)
    return path

def load_checkpoint_weights(
    ckpt_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Loads PyTorch checkpoint safely, extracting model_state_dict and provenance metadata.
    Supports both legacy raw state_dict checkpoints and new wrapped provenance dictionaries.
    
    Returns:
        (state_dict, provenance_metadata)
    """
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    loaded = torch.load(path, map_location=device, weights_only=False)

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
        provenance = loaded.get("provenance", loaded.get("metadata", {}))
        return state_dict, provenance
    elif isinstance(loaded, dict):
        # Legacy raw state_dict
        return loaded, {}
    else:
        raise ValueError(f"Unrecognized checkpoint object format in {path}: {type(loaded)}")

