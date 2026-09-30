#!/usr/bin/env python3
"""
One-Shot Cross-Dataset Transfer Simulation Script.
Evaluates 1-shot metric transfer across 100 trials using frozen penultimate embeddings.
Strictly isolates support and query workouts/subjects to prevent information leakage.
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ACTIONS, NUM_CLASSES, ACTION_TO_IDX
from src.cli import build_model
from src.external.mmfit import MMFitExternalDataset
from src.external.fit3d import Fit3DExternalDataset
from src.external.class_mapping import get_target_skelgym_indices, get_target_skelgym_classes
from src.external.fewshot import extract_penultimate_embeddings, simulate_one_shot_transfer

def load_checkpoint(model_type: str, feat_type: str, ckpt_path: str, device: torch.device):
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    m = build_model(model_type, feat_type, num_classes=NUM_CLASSES)
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()
    return m

def main():
    parser = argparse.ArgumentParser(description="External Benchmark 1-Shot Transfer Simulation")
    parser.add_argument("--config", type=str, default="configs/external/mmfit.yaml", help="Path to config yaml")
    parser.add_argument("--trials", type=int, default=100, help="Number of random 1-shot trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=None, help="Output destination directory")
    args = parser.parse_args()

    cfg_p = PROJECT_ROOT / args.config
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("dataset", "mmfit")
    pose_source = cfg.get("pose_protocol", {}).get("source", "native")
    class_set = cfg.get("class_set", "core4")
    split_group = cfg.get("workout_split", {}).get("split_group", "unseen_test")
    out_dir = Path(args.out_dir or cfg.get("paths", {}).get("output_dir", f"outputs/external/{dataset_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = PROJECT_ROOT / cfg.get("paths", {}).get("reference_dir", "artifacts/reference")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 80)
    print(f"ONE-SHOT CROSS-DATASET METRIC TRANSFER (100 TRIALS, K=1)")
    print(f"Dataset: {dataset_name.upper()} | Pose: {pose_source.upper()} | Class Set: {class_set}")
    print("=" * 80)

    # 1. Load Dataset
    if dataset_name == "mmfit":
        ext_ds = MMFitExternalDataset(
            root_dir=cfg.get("paths", {}).get("data_dir", "mm-fit"),
            split_group=split_group,
            class_set=class_set,
            pose_source=pose_source,
            apply_geometric_norm=True
        )
    elif dataset_name == "fit3d":
        ext_ds = Fit3DExternalDataset(
            root_dir=cfg.get("paths", {}).get("data_dir", "data_external/fit3d"),
            class_set=class_set,
            pose_source=pose_source,
            apply_geometric_norm=True
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    target_class_indices = get_target_skelgym_indices(dataset_name, class_set)

    # 2. Load Normalization
    stat_mix = np.load(ref_dir / "normalization_mix.npz")
    stat_bone = np.load(ref_dir / "normalization_bone_3d.npz")

    # Extract windows
    mix_data = ext_ds.extract_windows("mix", seq_len=32, stride=32)
    bone_data = ext_ds.extract_windows("bone_3d", seq_len=32, stride=32)

    feat_mix = (mix_data["features"] - stat_mix["mean"][None, :, :]) / (stat_mix["std"][None, :, :] + 1e-7)
    feat_bone = (bone_data["features"] - stat_bone["mean"][None, :, :]) / (stat_bone["std"][None, :, :] + 1e-7)

    record_ids = mix_data["record_ids"]
    labels = mix_data["labels"]
    subject_ids = mix_data["subject_ids"]

    # 3. Load Models
    m_trans = load_checkpoint("Transformer", "mix", str(PROJECT_ROOT / "checkpoints/best_Transformer_T2.2_mix.pt"), device)
    m_bone = load_checkpoint("AAGCN", "bone_3d", str(PROJECT_ROOT / "checkpoints/best_AAGCN_T4.2_bone_3d.pt"), device)

    # 4. Extract Penultimate Embeddings
    print("\nExtracting penultimate representation vectors...")
    embs_trans = extract_penultimate_embeddings(m_trans, feat_mix, "Transformer", device)
    embs_bone = extract_penultimate_embeddings(m_bone, feat_bone, "AAGCN", device)

    # Pool embeddings per record
    unique_recs = sorted(list(set(record_ids)))
    rec_to_class = {}
    rec_to_subject = {}
    rec_embs_trans = {}
    rec_embs_bone = {}
    rec_embs_full = {}

    for r in unique_recs:
        mask = [idx for idx, rec_id in enumerate(record_ids) if rec_id == r]
        rec_to_class[r] = labels[mask[0]]
        rec_to_subject[r] = subject_ids[mask[0]]

        mean_trans = np.mean(embs_trans[mask], axis=0)
        norm_t = mean_trans / (np.linalg.norm(mean_trans) + 1e-12)
        rec_embs_trans[r] = norm_t

        mean_bone = np.mean(embs_bone[mask], axis=0)
        norm_b = mean_bone / (np.linalg.norm(mean_bone) + 1e-12)
        rec_embs_bone[r] = norm_b

        # Concatenate normalized embeddings for SkelGym-Full
        c_emb = np.concatenate([norm_t, norm_b])
        rec_embs_full[r] = c_emb / (np.linalg.norm(c_emb) + 1e-12)

    eval_emb_models = {
        "Transformer (Mix)": rec_embs_trans,
        "AAGCN (Bone 3D)": rec_embs_bone,
        "SkelGym-Full": rec_embs_full
    }

    print("\n" + "=" * 80)
    print("TABLE Z: ONE-SHOT CROSS-DATASET TRANSFER SIMULATION")
    print("=" * 80)
    print(f"{'Model':<25} | {'Mean ± SD':<18} | {'95% Percentile CI':<20}")
    print("-" * 80)

    summary_rows = []
    all_trials_data = {}

    for m_name, emb_dict in eval_emb_models.items():
        sim_res = simulate_one_shot_transfer(
            embeddings_by_record=emb_dict,
            record_to_class=rec_to_class,
            record_to_subject=rec_to_subject,
            target_class_indices=target_class_indices,
            n_trials=args.trials,
            seed=args.seed
        )

        mean_acc = sim_res["mean"]
        std_acc = sim_res["std"]
        ci_low, ci_high = sim_res["ci_95"]

        print(f"{m_name:<25} | {mean_acc:>6.2f}% ± {std_acc:<6.2f}% | [{ci_low:>5.2f}%, {ci_high:>5.2f}%]")
        summary_rows.append({
            "model": m_name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high
        })
        all_trials_data[m_name] = sim_res["trials_df"]["accuracy"].tolist()

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "fewshot_summary.csv", index=False)
    pd.DataFrame(all_trials_data).to_csv(out_dir / "fewshot_100_trials.csv", index=False)

    print(f"\nSaved 1-shot transfer trial outputs to: {out_dir / 'fewshot_100_trials.csv'}")

if __name__ == "__main__":
    main()
