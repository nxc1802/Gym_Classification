"""
Hugging Face Hub Integration Module.
Handles automated uploading and downloading of:
  - Processed MediaPipe Landmarks dataset (ZIP / CSVs)
  - Model checkpoints (both 'best' and 'last' versions)
  - Confusion matrices and comprehensive Markdown / LaTeX reports
With built-in retry logic and connection management to prevent bandwidth bottlenecks.
"""

import os
import time
import zipfile
from pathlib import Path
from typing import Optional, List
from huggingface_hub import HfApi, hf_hub_download, create_repo

DEFAULT_MODEL_REPO = "Cuong2004/gym-exercise-classification"
DEFAULT_DATASET_REPO = "Cuong2004/gym-exercise-landmarks"

def get_hf_token(token: Optional[str] = None) -> str:
    """
    Returns provided token or environment variable HF_TOKEN.
    """
    if token and token.strip():
        return token.strip()
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        # Fallback to local config / cache if logged in
        tok = None
    return tok

def ensure_hf_repo(
    repo_id: str,
    repo_type: str = "model",
    token: Optional[str] = None,
    private: bool = False
) -> bool:
    """
    Ensures that the repository exists on Hugging Face Hub; creates it if not.
    """
    tok = get_hf_token(token)
    api = HfApi(token=tok)
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        return True
    except Exception:
        try:
            print(f"[HF Hub] Creating new {repo_type} repository: {repo_id} (private={private}) ...")
            create_repo(repo_id=repo_id, repo_type=repo_type, token=tok, private=private, exist_ok=True)
            print(f"[HF Hub] Repository {repo_id} created successfully.")
            return True
        except Exception as e:
            print(f"[HF Hub Warning] Could not verify/create repo {repo_id}: {e}")
            return False

def upload_file_to_hf(
    local_path: str,
    path_in_repo: str,
    repo_id: str = DEFAULT_MODEL_REPO,
    repo_type: str = "model",
    token: Optional[str] = None,
    max_retries: int = 3,
    commit_message: Optional[str] = None
) -> Optional[str]:
    """
    Uploads a single file to Hugging Face Hub with exponential backoff retry.
    """
    local_p = Path(local_path)
    if not local_p.exists():
        print(f"[HF Hub Error] Local file not found: {local_p}")
        return None

    tok = get_hf_token(token)
    api = HfApi(token=tok)
    ensure_hf_repo(repo_id=repo_id, repo_type=repo_type, token=tok)

    if commit_message is None:
        commit_message = f"Upload {path_in_repo}"

    for attempt in range(1, max_retries + 1):
        try:
            url = api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message
            )
            print(f"[HF Hub] Successfully uploaded {local_p.name} -> {repo_id}/{path_in_repo}")
            return url
        except Exception as e:
            print(f"[HF Hub] Upload attempt {attempt}/{max_retries} failed for {local_p.name}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                print(f"[HF Hub Error] Failed to upload {local_p.name} after {max_retries} attempts.")
                return None

def upload_checkpoints_to_hf(
    best_ckpt_path: str,
    last_ckpt_path: Optional[str] = None,
    model_name: str = "model",
    repo_id: str = DEFAULT_MODEL_REPO,
    token: Optional[str] = None
) -> dict:
    """
    Uploads both 'best' and 'last' checkpoints for a model to Hugging Face Hub.
    """
    urls = {}
    if best_ckpt_path and Path(best_ckpt_path).exists():
        best_name = Path(best_ckpt_path).name
        url_best = upload_file_to_hf(
            local_path=best_ckpt_path,
            path_in_repo=f"checkpoints/{best_name}",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Add best checkpoint for {model_name}"
        )
        urls["best"] = url_best

    if last_ckpt_path and Path(last_ckpt_path).exists():
        last_name = Path(last_ckpt_path).name
        url_last = upload_file_to_hf(
            local_path=last_ckpt_path,
            path_in_repo=f"checkpoints/{last_name}",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message=f"Add last checkpoint for {model_name}"
        )
        urls["last"] = url_last

    return urls

def push_landmarks_to_hf(
    zip_path: str,
    repo_id: str = DEFAULT_DATASET_REPO,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Uploads processed MediaPipe landmarks ZIP package to Hugging Face dataset repository.
    """
    print(f"[HF Hub] Pushing landmarks archive {zip_path} to dataset repo {repo_id} ...")
    ensure_hf_repo(repo_id=repo_id, repo_type="dataset", token=token)
    return upload_file_to_hf(
        local_path=zip_path,
        path_in_repo=Path(zip_path).name,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Upload processed MediaPipe landmarks: {Path(zip_path).name}"
    )

def pull_landmarks_from_hf(
    dest_dir: str = "data/landmarks",
    repo_id: str = DEFAULT_DATASET_REPO,
    filename: str = "landmarks_dataset.zip",
    token: Optional[str] = None
) -> str:
    """
    Downloads MediaPipe landmarks ZIP package from Hugging Face and extracts to dest_dir.
    """
    dest_p = Path(dest_dir)
    dest_p.mkdir(parents=True, exist_ok=True)
    tok = get_hf_token(token)

    print(f"[HF Hub] Downloading landmarks package '{filename}' from {repo_id} ...")
    local_zip = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        token=tok
    )
    print(f"[HF Hub] Downloaded to: {local_zip}. Extracting into {dest_p} ...")
    with zipfile.ZipFile(local_zip, "r") as zf:
        zf.extractall(dest_p)
    print(f"[HF Hub] Extraction complete! Landmarks ready at {dest_p}")
    return str(dest_p)
