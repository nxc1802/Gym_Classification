#!/usr/bin/env python3
"""
Full End-to-End Pipeline for MediaPipe World Landmarks (pose_world_landmarks):
1. Extract 33 pose_world_landmarks (metric 3D coordinates, mid-hip origin) from dataset videos.
2. Package and upload world_landmarks_dataset.zip to Hugging Face Hub (Cuong2004/gym-exercise-landmarks).
3. Compute training-set normalization statistics (train_mean, train_std) for world_3d (39-d, 13 joints).
4. Train Dual-Branch Transformer across 3 random seeds (42, 123, 3407) on world_3d.
5. Evaluate window-level and video-level consensus metrics on Validation and Held-out Test sets.
6. Compare directly against root-relative (rel_3d, rel_3d_norm) benchmarks.
"""

import os
import sys
import time
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX, RAW_POINTS_13, RAW_POINTS_33
from src.data.extractor import extract_landmarks_from_video
from src.data.dataset import get_dataloaders, save_normalization_artifact, _compute_train_stats
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.hf_hub import upload_file_to_hf, DEFAULT_DATASET_REPO

KAGGLE_DATASET_ID = "nguyenxuancuongk18dn/gym-exercise-classification-dataset"
SEEDS = [42, 123, 3407]

def _extract_worker(task: Tuple[str, str]) -> Tuple[bool, str, str]:
    vid_path, out_csv = task
    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        # Skip if already exists and non-empty (>100 bytes)
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 100:
            return True, Path(vid_path).name, "cached"
        extract_landmarks_from_video(
            video_path=vid_path,
            output_csv_path=out_csv,
            model_complexity=2,
            world_landmarks=True
        )
        return True, Path(vid_path).name, "extracted"
    except Exception as e:
        return False, Path(vid_path).name, str(e)

def extract_all_world_landmarks(
    raw_video_dir: str,
    output_dir: str,
    workers: int = 16
) -> None:
    raw_path = Path(raw_video_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: Extracting MediaPipe pose_world_landmarks (Heavy Complexity=2)")
    print(f"Source video directory: {raw_path}")
    print(f"Target landmark directory: {out_path}")
    print(f"Workers: {workers}")
    print("=" * 80)

    video_exts = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    all_videos = [f for f in raw_path.rglob("*") if f.is_file() and f.suffix.lower() in video_exts]
    print(f"Discovered {len(all_videos)} video files in source directory.")

    tasks = []
    skipped = 0
    for vid in all_videos:
        rel = vid.relative_to(raw_path)
        parts = rel.parts
        split, act, filename = None, None, None
        for idx_p, p in enumerate(parts):
            if p.lower() in {"train", "val", "test"} and len(parts) > idx_p + 1:
                split = p.lower()
                act = parts[idx_p + 1]
                filename = parts[-1]
                break
        if not split or not act or not filename:
            continue
        csv_name = f"{Path(filename).stem}.csv"
        target_csv = out_path / split / act / csv_name
        if target_csv.exists() and target_csv.stat().st_size > 100:
            skipped += 1
        else:
            tasks.append((str(vid), str(target_csv)))

    print(f"Tasks to process: {len(tasks)} (Already cached: {skipped})")

    t0 = time.time()
    success, failed = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_worker, t): t for t in tasks}
        total = len(tasks)
        for i, fut in enumerate(as_completed(futures), 1):
            ok, name, status = fut.result()
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  [FAIL] {name}: {status}")
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"  Progress: [{i}/{total}] | Success: {success} | Failed: {failed} | Elapsed: {elapsed:.1f}s", flush=True)

    print(f"\n[DONE] Extraction completed in {time.time() - t0:.1f}s. Total successful: {success}, Failed: {failed}")
    for sp in ["train", "val", "test"]:
        p = out_path / sp
        cnt = len(list(p.glob("**/*.csv"))) if p.exists() else 0
        print(f"  - {sp:6s}: {cnt:4d} CSV files")

def package_and_upload_to_hf(
    world_landmark_dir: str,
    hf_token: str
) -> None:
    print("\n" + "=" * 80)
    print("STEP 2: Packaging and Uploading world_landmarks_dataset.zip to Hugging Face Hub")
    print(f"Target Repo: {DEFAULT_DATASET_REPO}")
    print("=" * 80)

    zip_base = str(ROOT_DIR / "world_landmarks_dataset")
    zip_path = f"{zip_base}.zip"
    if not os.path.exists(zip_path):
        print(f"Compressing {world_landmark_dir} -> {zip_path} ...")
        shutil.make_archive(zip_base, "zip", world_landmark_dir)
        print(f"Archive created: {zip_path} ({os.path.getsize(zip_path)/(1024**2):.1f} MB)")
    else:
        print(f"Using existing archive: {zip_path} ({os.path.getsize(zip_path)/(1024**2):.1f} MB)")

    print(f"Uploading to Hugging Face Hub ({DEFAULT_DATASET_REPO}) ...")
    upload_file_to_hf(
        local_path=zip_path,
        path_in_repo="world_landmarks_dataset.zip",
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        commit_message="Add MediaPipe Pose World Landmarks (pose_world_landmarks 3D metric coordinates)"
    )
    print("[SUCCESS] world_landmarks_dataset.zip successfully uploaded alongside old landmarks!")

def train_and_eval_world_seed(
    seed: int,
    metadata_path: str,
    landmark_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n" + "-" * 60)
    print(f"--- Training WORLD Transformer (world_3d, Seed {seed}) ---")
    print(f"-" * 60)

    # 1. Dataloaders with normalization statistics computation strictly on train set
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method="world_3d",
        batch_size=16,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method="none",
        landmark_dir=landmark_dir,
        seed=seed,
        strict_norm=False,
        save_norm_artifact=True
    )

    train_mean = train_loader.dataset.norm_mean
    train_std = train_loader.dataset.norm_std
    mean_val = float(train_mean.mean().item()) if train_mean is not None else 0.0
    std_val = float(train_std.mean().item()) if train_std is not None else 1.0
    print(f"Normalization Stats computed on Train Set: Mean={mean_val:.4f}, Std={std_val:.4f}")

    # 2. Build model: Transformer (~301K params budget)
    model = build_model(
        model_type="Transformer",
        feature_method="world_3d",
        num_classes=len(ACTIONS)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: Dual-Branch Transformer | Trainable Parameters: {total_params:,}")

    # 3. Trainer setup
    ckpt_dir = ROOT_DIR / "checkpoints" / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"Transformer_world_3d_seed{seed}"
    best_ckpt_path = ckpt_dir / f"best_{model_name}.pt"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir=str(ckpt_dir),
        model_name=model_name,
        label_smoothing=0.05,
        early_stopping_metric="val_macro_f1",
        use_amp=torch.cuda.is_available(),
        feature_method="world_3d",
        seed=seed
    )

    # 4. Train
    t_start = time.time()
    history = trainer.fit(train_loader, val_loader, epochs=100)
    train_time = time.time() - t_start

    # 5. Evaluate on Validation and Held-out Test
    # Validation
    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    # Test
    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

    # Provenance metrics
    prov_file = best_ckpt_path.with_suffix(".provenance.json")
    val_loss = None
    best_epoch = None
    if prov_file.exists():
        try:
            with open(prov_file) as pf:
                pdata = json.load(pf)
                val_loss = pdata.get("val_loss")
                best_epoch = pdata.get("epoch")
        except Exception:
            pass

    res = {
        "seed": seed,
        "train_time_s": round(train_time, 1),
        "best_epoch": best_epoch or 0,
        "val_loss": round(float(val_loss), 4) if val_loss is not None else 0.0,
        "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
        "val_win_f1": round(float(val_m["macro_f1"]), 4),
        "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
        "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
        "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
        "test_win_f1": round(float(test_m["macro_f1"]), 4),
        "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
        "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
        "train_norm_mean": mean_val,
        "train_norm_std": std_val
    }
    print(f"Results Seed {seed}: ValWin: {res['val_win_acc']}%, ValVid: {res['val_vid_acc']}% | TestWin: {res['test_win_acc']}%, TestVid: {res['test_vid_acc']}% (F1: {res['test_vid_f1']})")
    return res

def main():
    parser = argparse.ArgumentParser(description="End-to-End MediaPipe World Landmarks Pipeline")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw video directory")
    parser.add_argument("--skip_extract", action="store_true", help="Skip landmark extraction step")
    parser.add_argument("--skip_upload", action="store_true", help="Skip Hugging Face Hub upload")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes for landmark extraction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""), help="Hugging Face API token")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Locate Raw Videos
    if not args.skip_extract:
        if args.raw_dir and os.path.exists(args.raw_dir):
            raw_video_dir = args.raw_dir
            print(f"[Dataset] Using provided raw directory: {raw_video_dir}")
        else:
            import kagglehub
            print("[Kaggle] Locating / downloading dataset...")
            raw_video_dir = kagglehub.dataset_download(KAGGLE_DATASET_ID)
            print(f"[Kaggle] Dataset located at: {raw_video_dir}")
        world_landmark_dir = str(ROOT_DIR / "data" / "world_landmarks")
        extract_all_world_landmarks(raw_video_dir, world_landmark_dir, workers=args.workers)
    else:
        world_landmark_dir = str(ROOT_DIR / "data" / "world_landmarks")

    # 2. Upload to HF Hub
    if not args.skip_upload:
        package_and_upload_to_hf(world_landmark_dir, hf_token=args.hf_token)

    # 3. Train & Evaluate across 3 seeds
    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    print("\n" + "=" * 80)
    print("STEP 3: Training & Evaluating WORLD Transformer across 3 Seeds (42, 123, 3407)")
    print("=" * 80)

    seed_results = []
    for s in SEEDS:
        res = train_and_eval_world_seed(
            seed=s,
            metadata_path=str(meta_cand),
            landmark_dir=world_landmark_dir,
            device=device
        )
        seed_results.append(res)

    # 4. Compute Multi-Seed Statistics
    def stats(key):
        vals = [r[key] for r in seed_results]
        return float(np.mean(vals)), float(np.std(vals))

    val_loss_m, val_loss_s = stats("val_loss")
    val_wacc_m, val_wacc_s = stats("val_win_acc")
    val_wf1_m, val_wf1_s = stats("val_win_f1")
    val_vacc_m, val_vacc_s = stats("val_vid_acc")
    val_vf1_m, val_vf1_s = stats("val_vid_f1")

    test_wacc_m, test_wacc_s = stats("test_win_acc")
    test_wf1_m, test_wf1_s = stats("test_win_f1")
    test_vacc_m, test_vacc_s = stats("test_vid_acc")
    test_vf1_m, test_vf1_s = stats("test_vid_f1")

    summary = {
        "feature": "world_3d",
        "dim": 39,
        "model": "Transformer (Dual-Branch)",
        "params": "301K",
        "seeds": seed_results,
        "val_loss": f"{val_loss_m:.4f} ± {val_loss_s:.4f}",
        "val_win_acc": f"{val_wacc_m:.2f}% ± {val_wacc_s:.2f}%",
        "val_win_f1": f"{val_wf1_m:.4f} ± {val_wf1_s:.4f}",
        "val_vid_acc": f"{val_vacc_m:.2f}% ± {val_vacc_s:.2f}%",
        "val_vid_f1": f"{val_vf1_m:.4f} ± {val_vf1_s:.4f}",
        "test_win_acc": f"{test_wacc_m:.2f}% ± {test_wacc_s:.2f}%",
        "test_win_f1": f"{test_wf1_m:.4f} ± {test_wf1_s:.4f}",
        "test_vid_acc": f"{test_vacc_m:.2f}% ± {test_vacc_s:.2f}%",
        "test_vid_f1": f"{test_vf1_m:.4f} ± {test_vf1_s:.4f}",
    }

    out_json = ROOT_DIR / "outputs" / "world_3d_benchmark_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Print Comparison Table
    print("\n" + "=" * 90)
    print("## DIRECT BENCHMARK COMPARISON: WORLD 3D (Metric) vs. RELATIVE 3D (Table 2b)")
    print("=" * 90)
    print("| Representation Paradigm | Dim | Val Loss | Val Win Acc (%) | Val Vid Acc (%) | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **Raw 3D Coordinates** | 39 | 1.1361 | 77.04% ± 0.38% | 78.90% ± 1.39% | 63.36% ± 1.48% | 0.6228 | 72.39% ± 0.88% | 0.7045 |")
    print(f"| **Root-Relative 3D (`rel_3d`)** | 39 | 1.1740 | 77.88% ± 0.89% | — | 63.87% ± 0.73% | 0.6179 | — | — |")
    print(f"| **Scale-Norm Rel 3D (`rel_3d_norm`)** | 39 | 1.1391 | 77.98% ± 0.86% | 81.32% ± 0.46% | 63.46% ± 0.40% | 0.6221 | 73.53% ± 1.07% | 0.7067 |")
    print(f"| **World 3D (pose_world_landmarks)** | 39 | {summary['val_loss']} | {summary['val_win_acc']} | {summary['val_vid_acc']} | {summary['test_win_acc']} | {test_wf1_m:.4f} | **{summary['test_vid_acc']}** | **{test_vf1_m:.4f}** |")
    print("=" * 90)

if __name__ == "__main__":
    main()
