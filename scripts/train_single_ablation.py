#!/usr/bin/env python3
"""
Single-Run Worker for Table 4/5 Ablation on WORLD mix_v2.
Used by parallel orchestrators to train or evaluate a single (cfg_id, seed) run in an isolated process.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.dataset import get_dataloaders
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

def main():
    parser = argparse.ArgumentParser(description="Single-Run Ablation Worker")
    parser.add_argument("--cfg_id", type=str, required=True)
    parser.add_argument("--aug", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ablation_v2")
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--force-retrain", action="store_true", default=False)
    args = parser.parse_args()

    resume = args.resume and not args.force_retrain
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    seed_ckpt_dir = checkpoint_dir / f"seed{args.seed}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = Path(args.out_json) if args.out_json else seed_ckpt_dir / f"result_{args.cfg_id}_seed{args.seed}.json"

    # Fast skip if out_json already exists and is non-empty
    if resume and out_json_path.exists():
        try:
            with open(out_json_path) as f:
                cached = json.load(f)
            if cached.get("val_vid_acc") is not None and cached.get("test_vid_acc") is not None:
                print(f"[{args.cfg_id} | Seed {args.seed}] CACHED: Val Vid Acc {cached['val_vid_acc']}%, Test Vid Acc {cached['test_vid_acc']}%")
                return
        except Exception:
            pass

    meta_cand = Path(args.metadata_path) if args.metadata_path else (ROOT_DIR / "data" / "Final_dataset_metadata.csv")
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"\n---> [Worker Seed {args.seed}] Training {args.cfg_id} (aug: {args.aug}) on {device}...")
    t_start = time.time()

    # 1. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=str(meta_cand),
        feature_method="mix_v2",
        batch_size=16,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=args.aug if args.aug != "none" else None,
        zero_frame_handling="interpolate",
        landmark_dir=None,
        in_memory=True,
        seed=args.seed,
        save_norm_artifact=True,
        strict_norm=False
    )

    # 2. Model: Dual-Branch Transformer (301K params)
    model = build_model(
        model_type="Transformer",
        feature_method="mix_v2",
        hidden_dim=112,
        num_layers=3,
        nhead=4,
        dropout=0.2,
        transformer_variant="dual_branch"
    )

    model_name = f"Transformer_mix_v2_{args.cfg_id}_seed{args.seed}"
    best_ckpt_path = seed_ckpt_dir / f"best_{model_name}.pt"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-4,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir=str(seed_ckpt_dir),
        model_name=model_name,
        label_smoothing=0.05,
        early_stopping_metric="val_macro_f1",
        use_amp=torch.cuda.is_available(),
        feature_method="mix_v2",
        augment_method=args.aug,
        seed=args.seed
    )

    # 3. Fit or Load
    if resume and best_ckpt_path.exists():
        print(f"  --> [Worker Seed {args.seed}] Loading existing checkpoint {best_ckpt_path.name}")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        train_time = 0.0
    else:
        history = trainer.fit(train_loader, val_loader, epochs=100)
        train_time = time.time() - t_start

    # 4. Predict Train (unshuffled for Stacking meta-classifier)
    from torch.utils.data import DataLoader
    eval_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=0
    )
    y_train, _, train_probs = trainer.predict(eval_train_loader)

    # 5. Evaluate Validation
    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    # 6. Evaluate Test
    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

    # Save complete prediction tensors to compressed npz
    probs_npz = seed_ckpt_dir / f"probs_{args.cfg_id}_seed{args.seed}.npz"
    np.savez_compressed(
        probs_npz,
        train_probs=train_probs,
        train_targets=y_train,
        val_probs=val_probs,
        val_targets=y_val,
        test_probs=test_probs,
        test_targets=y_test,
        val_video_ids=np.array(val_loader.dataset.video_ids),
        test_video_ids=np.array(test_loader.dataset.video_ids)
    )

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
        "cfg_id": args.cfg_id,
        "aug_method": args.aug,
        "seed": args.seed,
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
        "checkpoint": str(best_ckpt_path)
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"[{args.cfg_id} | Seed {args.seed}] FINISHED in {train_time:.1f}s | Val Vid: {res['val_vid_acc']}%, Test Vid: {res['test_vid_acc']}% (F1: {res['test_vid_f1']})")

if __name__ == "__main__":
    main()
