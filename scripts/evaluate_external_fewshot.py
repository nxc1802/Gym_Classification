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
from typing import Tuple, Dict, Any, List, Optional
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

def find_checkpoint_path(ckpt_rel: str, seed: int) -> Path:
    p = Path(ckpt_rel)
    stem = p.stem
    suffix = p.suffix
    if seed == 42:
        candidates = [
            PROJECT_ROOT / ckpt_rel,
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}{suffix}",
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}_seed{seed}{suffix}"
        ]
    else:
        candidates = [
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}{suffix}",
            PROJECT_ROOT / "checkpoints" / f"seed{seed}" / f"{stem}_seed{seed}{suffix}"
        ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Strict checkpoint lookup failed for seed {seed}: {ckpt_rel}. "
        f"Checked candidates: {[str(c) for c in candidates]}. No silent fallback allowed."
    )

def load_norm_stats_for_seed(ref_dir: Path, feat: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if seed == 42:
        candidates = [
            ref_dir / f"seed{seed}" / f"normalization_{feat}.npz",
            ref_dir / f"normalization_{feat}.npz"
        ]
    else:
        candidates = [
            ref_dir / f"seed{seed}" / f"normalization_{feat}.npz"
        ]
    for c in candidates:
        if c.exists():
            data = np.load(c)
            return (data["mean"], data["std"])
    raise FileNotFoundError(
        f"Strict normalization artifact lookup failed for feature '{feat}' and seed {seed}. "
        f"Expected: {candidates[0]}."
    )

def load_ensemble_weights_for_seed(ref_dir: Path, seed: int) -> Dict[str, Any]:
    if seed == 42:
        candidates = [
            ref_dir / f"seed{seed}" / "ensemble_weights.json",
            ref_dir / "ensemble_weights.json"
        ]
    else:
        candidates = [
            ref_dir / f"seed{seed}" / "ensemble_weights.json"
        ]
    for c in candidates:
        if c.exists():
            with open(c, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Strict ensemble weights lookup failed for seed {seed}. "
        f"Expected: {candidates[0]}. Run freeze_reference_artifacts.py for seed {seed} first!"
    )

def main():
    parser = argparse.ArgumentParser(description="External Benchmark 1-Shot Transfer Simulation")
    parser.add_argument("--config", type=str, default="configs/external/mmfit.yaml", help="Path to config yaml")
    parser.add_argument("--trials", type=int, default=100, help="Number of random 1-shot trials")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds to evaluate across (e.g. --seeds 42 123 3407)")
    parser.add_argument("--pose-source", type=str, default=None, choices=["native", "mediapipe"], help="Pose protocol source")
    parser.add_argument("--split-group", type=str, default=None, help="Workout split group (e.g. unseen_test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (fallback if --seeds not provided)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output destination directory")
    args = parser.parse_args()

    cfg_p = PROJECT_ROOT / args.config
    with open(cfg_p, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("dataset", "mmfit")
    pose_source = args.pose_source or cfg.get("pose_protocol", {}).get("source", "native")
    class_set = cfg.get("class_set", "core4")
    split_group = args.split_group or cfg.get("workout_split", {}).get("split_group", "unseen_test")
    out_dir = Path(args.out_dir or cfg.get("paths", {}).get("output_dir", f"outputs/external/{dataset_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = PROJECT_ROOT / cfg.get("paths", {}).get("reference_dir", "artifacts/reference")
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    seeds = args.seeds if args.seeds else [args.seed]

    print("=" * 80)
    print(f"ONE-SHOT CROSS-DATASET METRIC TRANSFER ({args.trials} TRIALS, K=1)")
    print(f"Dataset: {dataset_name.upper()} | Pose: {pose_source.upper()} | Class Set: {class_set}")
    print(f"Seeds: {seeds}")
    print("=" * 80)

    # 1. Load Dataset
    if dataset_name == "mmfit":
        ext_ds = MMFitExternalDataset(
            root_dir=cfg.get("paths", {}).get("data_dir", "mm-fit"),
            split_group=split_group,
            class_set=class_set,
            pose_source=pose_source,
            apply_geometric_norm=cfg.get("normalization", {}).get("geometric", False)
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

    # 2. Extract Raw Windows for all 5 Feature Streams Once
    stream_names = ["mix", "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
    raw_window_data = {feat: ext_ds.extract_windows(feat, seq_len=32, stride=32) for feat in stream_names}

    base_stream = raw_window_data["mix"]
    record_ids = base_stream["record_ids"]
    labels = base_stream["labels"]
    subject_ids = base_stream["subject_ids"]

    model_names = ["Transformer (Mix)", "AAGCN (Bone 3D)", "SkelGym-Lite", "SkelGym-Full"]
    all_trials_by_model = {m: [] for m in model_names}
    seed_summaries = []

    five_models_spec = [
        ("Transformer (Mix)", "Transformer", "mix", "checkpoints/best_Transformer_T2.2_mix.pt"),
        ("AAGCN (Bone 3D)", "AAGCN", "bone_3d", "checkpoints/best_AAGCN_T4.2_bone_3d.pt"),
        ("AAGCN (Rel 3D)", "AAGCN", "rel_3d", "checkpoints/best_AAGCN_T4.3_rel_3d.pt"),
        ("AAGCN (Joint Mot)", "AAGCN", "joint_motion_3d", "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt"),
        ("AAGCN (Bone Mot)", "AAGCN", "bone_motion_3d", "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt"),
    ]

    # 3. Multi-Seed Simulation Loop
    for s_idx, seed in enumerate(seeds, 1):
        print(f"\n[{s_idx}/{len(seeds)}] Evaluating 1-Shot Transfer on Seed {seed}...")
        
        # Load norm stats and extract penultimate embeddings for all 5 streams
        embs_by_model = {}
        for (m_name, m_type, f_type, ckpt_p) in five_models_spec:
            mean, std = load_norm_stats_for_seed(ref_dir, f_type, seed)
            raw_feat = raw_window_data[f_type]["features"]
            feat_norm = (raw_feat - mean[None, :, :]) / (std[None, :, :] + 1e-7)

            ckpt_path = find_checkpoint_path(ckpt_p, seed)
            model = load_checkpoint(m_type, f_type, str(ckpt_path), device)
            embs = extract_penultimate_embeddings(model, feat_norm, m_type, device)
            embs_by_model[m_name] = embs

        ens_cfg = load_ensemble_weights_for_seed(ref_dir, seed)
        w_lite = ens_cfg["lite"]["weights_window"]
        w_full = ens_cfg["full"]["weights_window"]

        # Pool embeddings per record
        unique_recs = sorted(list(set(record_ids)))
        rec_to_class = {}
        rec_to_subject = {}
        rec_embs_trans = {}
        rec_embs_bone = {}
        rec_embs_lite = {}
        rec_embs_full = {}

        for r in unique_recs:
            mask = [idx for idx, rec_id in enumerate(record_ids) if rec_id == r]
            rec_to_class[r] = labels[mask[0]]
            rec_to_subject[r] = subject_ids[mask[0]]

            # Normalized mean pooled embeddings for each of the 5 streams
            pooled_norms = []
            for (m_name, _, _, _) in five_models_spec:
                m_emb = np.mean(embs_by_model[m_name][mask], axis=0)
                norm_m = m_emb / (np.linalg.norm(m_emb) + 1e-12)
                pooled_norms.append(norm_m)

            rec_embs_trans[r] = pooled_norms[0]
            rec_embs_bone[r] = pooled_norms[1]

            # Weighted SkelGym-Lite embedding (Transformer + Bone 3D with validation SLSQP weights)
            lite_parts = [
                np.sqrt(max(0.0, float(w_lite[0]))) * pooled_norms[0],
                np.sqrt(max(0.0, float(w_lite[1]))) * pooled_norms[1]
            ]
            lite_concat = np.concatenate(lite_parts)
            rec_embs_lite[r] = lite_concat / (np.linalg.norm(lite_concat) + 1e-12)

            # Weighted SkelGym-Full embedding (All 5 streams with validation SLSQP weights)
            full_parts = [
                np.sqrt(max(0.0, float(w_full[i]))) * pooled_norms[i]
                for i in range(5)
            ]
            full_concat = np.concatenate(full_parts)
            rec_embs_full[r] = full_concat / (np.linalg.norm(full_concat) + 1e-12)

        eval_emb_models = {
            "Transformer (Mix)": rec_embs_trans,
            "AAGCN (Bone 3D)": rec_embs_bone,
            "SkelGym-Lite": rec_embs_lite,
            "SkelGym-Full": rec_embs_full
        }

        for m_name, emb_dict in eval_emb_models.items():
            sim_res = simulate_one_shot_transfer(
                embeddings_by_record=emb_dict,
                record_to_class=rec_to_class,
                record_to_subject=rec_to_subject,
                target_class_indices=target_class_indices,
                n_trials=args.trials,
                seed=seed
            )
            trial_accs = sim_res["trials_df"]["accuracy"].tolist()
            all_trials_by_model[m_name].extend(trial_accs)
            seed_summaries.append({
                "seed": seed,
                "model": m_name,
                "mean_acc": sim_res["mean"],
                "std_acc": sim_res["std"],
                "ci_95_low": sim_res["ci_95"][0],
                "ci_95_high": sim_res["ci_95"][1]
            })

    # 4. Global Aggregation across Trials & Seeds
    print("\n" + "=" * 80)
    print("TABLE Z: ONE-SHOT CROSS-DATASET TRANSFER SIMULATION")
    print(f"Evaluated across {len(seeds)} seed(s) x {args.trials} trials")
    print("=" * 80)
    print(f"{'Model':<25} | {'Mean ± SD':<18} | {'95% Percentile CI':<20}")
    print("-" * 80)

    summary_rows = []
    for m_name in model_names:
        accs = np.array(all_trials_by_model[m_name])
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        ci_low = float(np.percentile(accs, 2.5))
        ci_high = float(np.percentile(accs, 97.5))

        print(f"{m_name:<25} | {mean_acc:>6.2f}% ± {std_acc:<6.2f}% | [{ci_low:>5.2f}%, {ci_high:>5.2f}%]")
        summary_rows.append({
            "model": m_name,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "num_trials": len(accs)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "fewshot_summary.csv", index=False)
    pd.DataFrame(all_trials_by_model).to_csv(out_dir / "fewshot_trials.csv", index=False)
    if len(seeds) > 1:
        pd.DataFrame(seed_summaries).to_csv(out_dir / "fewshot_seed_runs.csv", index=False)

    print(f"\nSaved 1-shot transfer trial outputs to: {out_dir / 'fewshot_trials.csv'}")

if __name__ == "__main__":
    main()
