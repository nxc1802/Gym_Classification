#!/usr/bin/env python3
"""
Phase 0: Freeze SkelGym Reference Artifacts.
Computes and permanently freezes:
1. Canonical class list (22 classes)
2. Normalization mean & std for all feature representations (mix, rel_3d, bone_3d, joint_motion_3d, bone_motion_3d) from train split only
3. Ensemble weights (SkelGym-Lite and SkelGym-Full) fitted strictly on SkelGym validation split
4. Checkpoint manifest with SHA-256 signatures and protocol parameters
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ACTIONS, NUM_CLASSES
from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.models.ensemble import WeightedSoftVotingEnsemble

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_checkpoint_model(model_type: str, feat_type: str, ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    m = build_model(model_type, feat_type, num_classes=NUM_CLASSES)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    return m

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Phase 0] Freezing Reference Artifacts on device: {device}")

    out_dir = PROJECT_ROOT / "artifacts" / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = str(PROJECT_ROOT / "data" / "Final_dataset_metadata.csv")
    landmark_dir = str(PROJECT_ROOT / "data" / "landmarks")

    # 1. Canonical class names
    class_names_path = out_dir / "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(ACTIONS, f, indent=2)
    print(f"[1/4] Saved canonical class names to {class_names_path}")

    # 2. Extract and freeze train normalization statistics
    feature_streams = {
        "mix": "mix_v2",
        "rel_3d": "rel_3d",
        "bone_3d": "bone_3d",
        "joint_motion_3d": "joint_motion_3d",
        "bone_motion_3d": "bone_motion_3d"
    }

    seeds = [42, 123, 3407]
    for seed in seeds:
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        norm_stats = {}
        dataloaders = {}

        print(f"\n[2/4] [Seed {seed}] Extracting train normalization statistics (stride 16) & validation dataloaders...")
        for feat_key, feat_method in feature_streams.items():
            train_l, val_l, test_l = get_dataloaders(
                metadata_path=metadata_path,
                feature_method=feat_method,
                batch_size=32,
                seq_len=32,
                stride=16,
                val_test_stride=32,
                landmark_dir=landmark_dir,
                num_workers=0,
                in_memory=True,
                seed=seed
            )
            dataloaders[feat_key] = (train_l, val_l, test_l)

            # Retrieve train stats
            train_mean = train_l.dataset.train_mean
            train_std = train_l.dataset.train_std
            if isinstance(train_mean, torch.Tensor):
                train_mean = train_mean.cpu().numpy()
            if isinstance(train_std, torch.Tensor):
                train_std = train_std.cpu().numpy()

            stat_path = seed_dir / f"normalization_{feat_key}.npz"
            np.savez(stat_path, mean=train_mean, std=train_std)
            if seed == 42:
                # Also save to base reference dir for default fallback
                np.savez(out_dir / f"normalization_{feat_key}.npz", mean=train_mean, std=train_std)

            norm_stats[feat_key] = {"mean_shape": list(train_mean.shape), "std_shape": list(train_std.shape)}

        # 3. Model predictions on validation set to fit & freeze SLSQP ensemble weights
        print(f"\n[3/4] [Seed {seed}] Checking checkpoints to calibrate ensemble weights...")
        ckpt_base = PROJECT_ROOT / "checkpoints"
        if seed != 42:
            ckpt_base = ckpt_base / f"seed{seed}"

        trans_mix_cand = ckpt_base / "best_Transformer_T2.2_mix_v2.pt"
        if not trans_mix_cand.exists():
            trans_mix_cand = ckpt_base / "best_Transformer_T2.2_mix.pt"

        ensemble_models = {
            "Transformer (Mix)": ("Transformer", "mix_v2", str(trans_mix_cand)),
            "AAGCN (Bone 3D)": ("AAGCN", "bone_3d", str(ckpt_base / "best_AAGCN_T4.2_bone_3d.pt")),
            "AAGCN (Rel 3D)": ("AAGCN", "rel_3d", str(ckpt_base / "best_AAGCN_T4.3_rel_3d.pt")),
            "AAGCN (Joint Mot)": ("AAGCN", "joint_motion_3d", str(ckpt_base / "best_AAGCN_T4.4_joint_motion_3d.pt")),
            "AAGCN (Bone Mot)": ("AAGCN", "bone_motion_3d", str(ckpt_base / "best_AAGCN_T4.5_bone_motion_3d.pt")),
        }
        optional_checkpoints = {
            "ST-GCN (Rel 3D)": ("STGCN", "rel_3d", str(ckpt_base / "best_STGCN_T3.2_rel_3d.pt")),
        }
        checkpoints = dict(ensemble_models)
        for k, v in optional_checkpoints.items():
            if os.path.exists(v[2]):
                checkpoints[k] = v

        all_exist = all(os.path.exists(cp) for _, _, cp in ensemble_models.values())
        if not all_exist:
            print(f"  Checkpoints for seed {seed} not yet fully trained. Saving normalization stats only.")
            continue

        val_probs = {}
        val_l_mix = dataloaders["mix"][1]
        y_val_true = np.array(val_l_mix.dataset.labels)
        val_video_ids = list(val_l_mix.dataset.video_ids)

        ckpt_shas = {}
        for name, (m_type, f_type, ckpt_p) in checkpoints.items():
            ckpt_shas[name] = {
                "path": ckpt_p,
                "sha256": get_file_sha256(ckpt_p) if os.path.exists(ckpt_p) else None
            }
            model = load_checkpoint_model(m_type, f_type, ckpt_p, device)
            _, val_loader, _ = dataloaders[f_type]
            trainer = Trainer(model=model, device=device)
            _, _, probs = trainer.predict(val_loader)
            val_probs[name] = probs

        # Fit SkelGym-Lite (Transformer + AAGCN Bone 3D)
        lite_ens = WeightedSoftVotingEnsemble()
        lite_ens.fit_window([val_probs["Transformer (Mix)"], val_probs["AAGCN (Bone 3D)"]], y_val_true)
        lite_ens.fit_video([val_probs["Transformer (Mix)"], val_probs["AAGCN (Bone 3D)"]], y_val_true, val_video_ids)

        # Fit SkelGym-Full (5 streams)
        full_ens = WeightedSoftVotingEnsemble()
        five_val = [
            val_probs["Transformer (Mix)"],
            val_probs["AAGCN (Bone 3D)"],
            val_probs["AAGCN (Rel 3D)"],
            val_probs["AAGCN (Joint Mot)"],
            val_probs["AAGCN (Bone Mot)"]
        ]
        full_ens.fit_window(five_val, y_val_true)
        full_ens.fit_video(five_val, y_val_true, val_video_ids)

        ensemble_weights = {
            "calibration_source": "skelgym_validation_only",
            "seed": seed,
            "lite": {
                "models": ["Transformer (Mix)", "AAGCN (Bone 3D)"],
                "weights_window": [float(w) for w in lite_ens.weights_window],
                "weights_video": [float(w) for w in lite_ens.weights_video]
            },
            "full": {
                "models": [
                    "Transformer (Mix)",
                    "AAGCN (Bone 3D)",
                    "AAGCN (Rel 3D)",
                    "AAGCN (Joint Mot)",
                    "AAGCN (Bone Mot)"
                ],
                "weights_window": [float(w) for w in full_ens.weights_window],
                "weights_video": [float(w) for w in full_ens.weights_video]
            }
        }

        ensemble_weights_path = seed_dir / "ensemble_weights.json"
        with open(ensemble_weights_path, "w") as f:
            json.dump(ensemble_weights, f, indent=2)
        if seed == 42:
            with open(out_dir / "ensemble_weights.json", "w") as f:
                json.dump(ensemble_weights, f, indent=2)
        print(f"  Saved ensemble weights for seed {seed}")

        # 4. Checkpoint manifest
        manifest = {
            "version": "1.0",
            "dataset_name": "SkelGym",
            "num_classes": NUM_CLASSES,
            "normalization_source": "skelgym_train_only",
            "ensemble_calibration_source": "skelgym_validation_only",
            "seq_len": 32,
            "stride": 16,
            "seed": seed,
            "checkpoints": ckpt_shas,
            "normalization_stats": norm_stats
        }

        manifest_path = seed_dir / "checkpoint_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if seed == 42:
            with open(out_dir / "checkpoint_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
        print(f"  Saved checkpoint manifest for seed {seed}")

    print("\nPhase 0 complete! All reference artifacts permanently frozen per seed.")

if __name__ == "__main__":
    main()
