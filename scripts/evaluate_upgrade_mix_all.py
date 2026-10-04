#!/usr/bin/env python3
"""
Comprehensive Evaluation & Sync Script for Upgraded Biomechanical Mix (SkelGym-300K).
Evaluates all 9 checkpoints across 3 features (raw_3d, rel_3d_norm, mix_v2) and 3 seeds (42, 123, 3407)
on both VALIDATION and TEST splits, computing:
  - Window-Level Accuracy (%) & Macro F1
  - Video-Level Accuracy (%) & Macro F1
  - Train Loss, Train Acc (%), Val Loss
And uploads all model weights and provenance files to Hugging Face Hub.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import NUM_CLASSES
from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions
from src.utils.hf_hub import upload_file_to_hf, get_hf_token

HF_REPO = "Cuong2004/gym-exercise-classification"
FEATURES = ["raw_3d", "rel_3d_norm", "mix_v2"]
SEEDS = [42, 123, 3407]

def evaluate_single_checkpoint(
    ckpt_path: Path,
    feature: str,
    seed: int,
    device: str = "cuda"
) -> Dict[str, Any]:
    print(f"\n=======================================================", flush=True)
    print(f"Evaluating {feature} (Seed {seed}) from {ckpt_path.name}", flush=True)
    print(f"=======================================================", flush=True)
    
    variant = "dual_branch" if feature == "mix_v2" else "standard"
    
    # 1. Build Model
    model = build_model(
        model_type="Transformer",
        feature_method=feature,
        num_classes=NUM_CLASSES,
        hidden_dim=112,
        num_layers=3,
        nhead=4,
        transformer_variant=variant
    )
    
    weights = torch.load(ckpt_path, map_location=device)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    
    # 2. Get DataLoaders with exact seed normalization
    _, val_loader, test_loader = get_dataloaders(
        metadata_path="Final_dataset_metadata.csv",
        feature_method=feature,
        batch_size=128,
        seq_len=32,
        landmark_dir="data/landmarks",
        num_workers=0,
        in_memory=True,
        seed=seed
    )
    
    trainer = Trainer(model=model, device=device)
    
    # 3. Evaluate on Validation Split
    y_true_val, y_pred_val, y_prob_val = trainer.predict(val_loader)
    val_win_metrics = compute_metrics(y_true_val, y_pred_val)
    _, _, _, val_vid_metrics = aggregate_video_level_predictions(
        y_prob_val, y_true_val, val_loader.dataset.video_ids
    )
    
    # 4. Evaluate on Test Split
    y_true_test, y_pred_test, y_prob_test = trainer.predict(test_loader)
    test_win_metrics = compute_metrics(y_true_test, y_pred_test)
    _, _, _, test_vid_metrics = aggregate_video_level_predictions(
        y_prob_test, y_true_test, test_loader.dataset.video_ids
    )
    
    # 5. Read Provenance File
    prov_path = ckpt_path.with_name(f"{ckpt_path.stem}.provenance.json")
    prov_data = {}
    if prov_path.exists():
        try:
            prov_data = json.loads(prov_path.read_text())
        except Exception as e:
            print(f"Warning reading provenance: {e}", flush=True)
            
    train_loss = prov_data.get("train_loss", None)
    train_acc = prov_data.get("train_acc", None)
    if train_acc is not None and train_acc <= 1.0:
        train_acc *= 100.0
    val_loss = prov_data.get("val_loss", None)
    best_epoch = prov_data.get("epoch", None)
    
    res = {
        "feature": feature,
        "seed": seed,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_win_acc": val_win_metrics["accuracy"] * 100.0,
        "val_win_f1": val_win_metrics["macro_f1"],
        "val_vid_acc": val_vid_metrics["accuracy"] * 100.0,
        "val_vid_f1": val_vid_metrics["macro_f1"],
        "test_win_acc": test_win_metrics["accuracy"] * 100.0,
        "test_win_f1": test_win_metrics["macro_f1"],
        "test_vid_acc": test_vid_metrics["accuracy"] * 100.0,
        "test_vid_f1": test_vid_metrics["macro_f1"]
    }
    
    print(f"  [VAL]  Win Acc: {res['val_win_acc']:.2f}% | Win F1: {res['val_win_f1']:.4f} | Vid Acc: {res['val_vid_acc']:.2f}% | Vid F1: {res['val_vid_f1']:.4f}", flush=True)
    print(f"  [TEST] Win Acc: {res['test_win_acc']:.2f}% | Win F1: {res['test_win_f1']:.4f} | Vid Acc: {res['test_vid_acc']:.2f}% | Vid F1: {res['test_vid_f1']:.4f}", flush=True)
    return res

def upload_models_to_hub(all_results: List[Dict[str, Any]]):
    tok = get_hf_token()
    if not tok:
        print("[HF Hub] No token found; skipping upload.", flush=True)
        return
        
    print(f"\n[HF Hub] Uploading all checkpoints and metadata to {HF_REPO}...", flush=True)
    for r in all_results:
        ckpt_p = Path(r["checkpoint"])
        feat = r["feature"]
        seed = r["seed"]
        remote_prefix = f"checkpoints/upgrade_mix/{feat}/seed{seed}"
        
        # Upload .pt
        if ckpt_p.exists():
            upload_file_to_hf(
                local_path=str(ckpt_p),
                path_in_repo=f"{remote_prefix}/{ckpt_p.name}",
                repo_id=HF_REPO,
                commit_message=f"Upload {feat} seed{seed} ({r['val_win_acc']:.2f}% Val Acc)"
            )
            
        # Upload provenance
        prov_p = ckpt_p.with_name(f"{ckpt_p.stem}.provenance.json")
        if prov_p.exists():
            upload_file_to_hf(
                local_path=str(prov_p),
                path_in_repo=f"{remote_prefix}/{prov_p.name}",
                repo_id=HF_REPO,
                commit_message=f"Upload provenance for {feat} seed{seed}"
            )

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = PROJECT_ROOT / "checkpoints" / "upgrade_mix"
    
    all_results = []
    summary_by_feature = {}
    
    for feat in FEATURES:
        feat_results = []
        for seed in SEEDS:
            ckpt_path = base_dir / feat / f"seed{seed}" / f"best_Transformer_{feat}.pt"
            if not ckpt_path.exists():
                print(f"Error: Missing checkpoint at {ckpt_path}", flush=True)
                continue
            r = evaluate_single_checkpoint(ckpt_path, feat, seed, device=device)
            feat_results.append(r)
            all_results.append(r)
            
        # Calculate statistics
        if feat_results:
            def calc_stat(key):
                vals = [x[key] for x in feat_results if x.get(key) is not None]
                return float(np.mean(vals)), float(np.std(vals))
                
            summary_by_feature[feat] = {
                "runs": feat_results,
                "val_win_acc_mean": calc_stat("val_win_acc")[0],
                "val_win_acc_sd": calc_stat("val_win_acc")[1],
                "val_win_f1_mean": calc_stat("val_win_f1")[0],
                "val_win_f1_sd": calc_stat("val_win_f1")[1],
                "val_vid_acc_mean": calc_stat("val_vid_acc")[0],
                "val_vid_acc_sd": calc_stat("val_vid_acc")[1],
                "val_vid_f1_mean": calc_stat("val_vid_f1")[0],
                "val_vid_f1_sd": calc_stat("val_vid_f1")[1],
                "test_win_acc_mean": calc_stat("test_win_acc")[0],
                "test_win_acc_sd": calc_stat("test_win_acc")[1],
                "test_win_f1_mean": calc_stat("test_win_f1")[0],
                "test_win_f1_sd": calc_stat("test_win_f1")[1],
                "test_vid_acc_mean": calc_stat("test_vid_acc")[0],
                "test_vid_acc_sd": calc_stat("test_vid_acc")[1],
                "test_vid_f1_mean": calc_stat("test_vid_f1")[0],
                "test_vid_f1_sd": calc_stat("test_vid_f1")[1],
                "train_loss_mean": calc_stat("train_loss")[0],
                "val_loss_mean": calc_stat("val_loss")[0]
            }
            
    # Save JSON summary
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "upgrade_mix_evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(summary_by_feature, f, indent=2)
    print(f"\n[Summary] Evaluation saved to {json_path}", flush=True)
    
    # Upload to HF
    upload_models_to_hub(all_results)
    
    # Upload the summary json itself
    if get_hf_token():
        upload_file_to_hf(
            local_path=str(json_path),
            path_in_repo="reports/upgrade_mix_evaluation_report.json",
            repo_id=HF_REPO,
            commit_message="Upload upgrade mix full evaluation report"
        )
        
    print("\n" + "="*80, flush=True)
    print("FINAL CONSOLIDATED BENCHMARK RESULTS (SkelGym-300K Transformer)", flush=True)
    print("="*80, flush=True)
    for feat, stats in summary_by_feature.items():
        print(f"\nFeature Paradigm: {feat.upper()}")
        print(f"  Validation Window Acc : {stats['val_win_acc_mean']:.2f}% ± {stats['val_win_acc_sd']:.2f}% | Macro F1: {stats['val_win_f1_mean']:.4f} ± {stats['val_win_f1_sd']:.4f}")
        print(f"  Validation Video Acc  : {stats['val_vid_acc_mean']:.2f}% ± {stats['val_vid_acc_sd']:.2f}% | Macro F1: {stats['val_vid_f1_mean']:.4f} ± {stats['val_vid_f1_sd']:.4f}")
        print(f"  Test Window Acc       : {stats['test_win_acc_mean']:.2f}% ± {stats['test_win_acc_sd']:.2f}% | Macro F1: {stats['test_win_f1_mean']:.4f} ± {stats['test_win_f1_sd']:.4f}")
        print(f"  Test Video Acc        : {stats['test_vid_acc_mean']:.2f}% ± {stats['test_vid_acc_sd']:.2f}% | Macro F1: {stats['test_vid_f1_mean']:.4f} ± {stats['test_vid_f1_sd']:.4f}")
        print(f"  Loss (Train / Val)    : {stats['train_loss_mean']:.4f} / {stats['val_loss_mean']:.4f}")

if __name__ == "__main__":
    main()
