#!/usr/bin/env python3
"""
Orchestrator and Evaluator for Downstream Experiments with the New SkelGym-Aug
(4-Operator Protocol: Bilateral Mirroring + 3D Yaw Rotation + Scaling + Jitter; No TimeWarp).

Trains all 5 affected backbones across 3 seeds (42, 123, 3407) in parallel:
1. Transformer (mix 117-d) [T2.2]
2. AAGCN (bone_3d)         [T4.2]
3. AAGCN (rel_3d)          [T4.3]
4. AAGCN (joint_motion_3d) [T4.4]
5. AAGCN (bone_motion_3d)  [T4.5]

Then systematically evaluates across all 3 seeds:
- Standalone backbones (Window & Video Acc, Macro F1)
- Two-Stream AAGCN (Joint + Bone)
- Four-Stream AAGCN (Joint + Bone + J-Motion + B-Motion)
- SkelGym-Lite (Transformer + Bone)
- SkelGym-Full (Transformer + 4-Stream AAGCN)
Using validation-calibrated SLSQP soft voting.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model, NUM_CLASSES
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, plot_confusion_matrix
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

MODELS_CONFIG = [
    {"name": "Transformer_mix", "model": "Transformer", "feature": "mix", "exp_id": "T2.2"},
    {"name": "AAGCN_bone_3d", "model": "AAGCN", "feature": "bone_3d", "exp_id": "T4.2"},
    {"name": "AAGCN_rel_3d", "model": "AAGCN", "feature": "rel_3d", "exp_id": "T4.3"},
    {"name": "AAGCN_joint_motion_3d", "model": "AAGCN", "feature": "joint_motion_3d", "exp_id": "T4.4"},
    {"name": "AAGCN_bone_motion_3d", "model": "AAGCN", "feature": "bone_motion_3d", "exp_id": "T4.5"},
]

def get_checkpoint_path(seed: int, model_cfg: Dict[str, str], checkpoint_base: Path) -> Path:
    if seed == 42:
        return checkpoint_base / f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"
    else:
        return checkpoint_base / f"seed{seed}" / f"best_{model_cfg['model']}_{model_cfg['exp_id']}_{model_cfg['feature']}.pt"

def build_training_tasks(seeds: List[int], checkpoint_base: Path) -> List[Dict[str, Any]]:
    tasks = []
    for seed in seeds:
        for m in MODELS_CONFIG:
            ckpt_path = get_checkpoint_path(seed, m, checkpoint_base)
            task_id = f"{m['name']}_seed{seed}"
            tasks.append({
                "id": task_id,
                "seed": seed,
                "model_cfg": m,
                "ckpt_path": ckpt_path,
                "status": "pending"
            })
    return tasks

def run_parallel_training(tasks: List[Dict[str, Any]], max_concurrent: int, device: str, progress_file: Path, log_dir: Path, force_retrain: bool = False):
    log_dir.mkdir(parents=True, exist_ok=True)
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    running_procs: Dict[str, Dict[str, Any]] = {}
    completed = 0
    total = len(tasks)

    print(f"\n==================================================================")
    print(f"STARTING PARALLEL TRAINING: {total} tasks, max {max_concurrent} concurrent")
    print(f"Device: {device} | Checkpoint Protocol: New SkelGym-Aug (No TimeWarp)")
    print(f"==================================================================")

    def save_progress():
        prog = {
            "total": total,
            "completed": completed,
            "running": len(running_procs),
            "pending": sum(1 for t in tasks if t["status"] == "pending"),
            "tasks": {t["id"]: {"status": t["status"], "ckpt": str(t["ckpt_path"])} for t in tasks}
        }
        with open(progress_file, "w") as f:
            json.dump(prog, f, indent=2)

    save_progress()

    task_idx = 0
    while completed < total:
        # Launch tasks up to max_concurrent
        while len(running_procs) < max_concurrent and task_idx < total:
            t = tasks[task_idx]
            task_idx += 1
            t_id = t["id"]
            ckpt_path = t["ckpt_path"]
            m = t["model_cfg"]
            seed = t["seed"]

            # If checkpoint exists and valid size (>100KB), mark done unless force_retrain
            if not force_retrain and ckpt_path.exists() and ckpt_path.stat().st_size > 100000:
                print(f"[REUSE] {t_id} checkpoint already exists: {ckpt_path.name}")
                t["status"] = "completed"
                completed += 1
                save_progress()
                continue

            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{t_id}.log"
            lf = open(log_file, "w", encoding="utf-8")

            cmd = [
                sys.executable, "-m", "src.cli", "train",
                "--model", m["model"],
                "--feature", m["feature"],
                "--augment", "skel_gym_aug",
                "--exp_id", m["exp_id"],
                "--seed", str(seed),
                "--epochs", "100",
                "--patience", "10",
                "--batch_size", "16",
                "--checkpoint_dir", str(ckpt_path.parent),
                "--output_dir", "outputs/downstream_new_aug",
                "--metadata", "data/Final_dataset_metadata.csv",
                "--landmark_dir", "data/landmarks",
                "--device", device,
                "--use_amp",
                "--in_memory"
            ]

            print(f"[LAUNCH] [{len(running_procs) + 1}/{max_concurrent}] {t_id} (Seed {seed}) -> {ckpt_path.name}")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, cwd=str(PROJECT_ROOT))
            running_procs[t_id] = {
                "proc": p,
                "logfile": lf,
                "task": t,
                "start_time": time.time()
            }
            t["status"] = "running"
            save_progress()

        # Poll running tasks
        time.sleep(2)
        finished_ids = []
        for t_id, info in running_procs.items():
            ret = info["proc"].poll()
            if ret is not None:
                info["logfile"].close()
                elapsed = time.time() - info["start_time"]
                t = info["task"]
                if ret == 0:
                    print(f"✅ [FINISHED] {t_id} in {elapsed:.1f}s (Exit 0)")
                    t["status"] = "completed"
                else:
                    print(f"❌ [FAILED] {t_id} with exit code {ret} after {elapsed:.1f}s. Check {log_dir / (t_id + '.log')}")
                    t["status"] = f"failed (code {ret})"
                completed += 1
                finished_ids.append(t_id)
                save_progress()

        for f_id in finished_ids:
            del running_procs[f_id]

    print(f"\n==================================================================")
    print(f"TRAINING COMPLETED: {completed}/{total} tasks processed.")
    print(f"==================================================================")

def evaluate_all(seeds: List[int], device: torch.device, checkpoint_base: Path, metadata_path: str, landmark_dir: str) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"STARTING COMPREHENSIVE EVALUATION ACROSS SEEDS: {seeds}")
    print(f"==================================================================")

    all_seed_eval = {}

    for seed in seeds:
        print(f"\n--- Evaluating Seed {seed} ---")
        val_probs = {}
        test_probs = {}
        y_val_true = None
        y_test_true = None
        val_video_ids = None
        test_video_ids = None

        # Predict with all 5 models
        for m in MODELS_CONFIG:
            feat = m["feature"]
            p = get_checkpoint_path(seed, m, checkpoint_base)
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {p}")

            state_dict = torch.load(p, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            model = build_model(m["model"], feat, num_classes=NUM_CLASSES)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            _, val_loader, test_loader = get_dataloaders(
                metadata_path=metadata_path,
                feature_method=feat,
                batch_size=32,
                seq_len=32,
                stride=32,
                val_test_stride=32,
                landmark_dir=landmark_dir,
                num_workers=0,
                in_memory=True
            )

            if val_video_ids is None:
                val_video_ids = val_loader.dataset.video_ids
            if test_video_ids is None:
                test_video_ids = test_loader.dataset.video_ids

            trainer = Trainer(model=model, device=device)
            y_vt, _, y_vp = trainer.predict(val_loader)
            y_tt, _, y_tp = trainer.predict(test_loader)

            if y_val_true is None:
                y_val_true = y_vt
            if y_test_true is None:
                y_test_true = y_tt

            val_probs[feat] = y_vp
            test_probs[feat] = y_tp

        seed_res = {}

        # 1. Standalone Single Models
        model_names_map = {
            "mix": "Transformer Mix (Aug)",
            "bone_3d": "AAGCN Bone (Aug)",
            "rel_3d": "AAGCN Joint (Aug)",
            "joint_motion_3d": "AAGCN Joint-Motion (Aug)",
            "bone_motion_3d": "AAGCN Bone-Motion (Aug)"
        }
        for feat, name in model_names_map.items():
            prob = test_probs[feat]
            preds = np.argmax(prob, axis=1)
            win_m = compute_metrics(y_test_true, preds)
            _, _, _, vid_m = aggregate_video_level_predictions(prob, y_test_true, test_video_ids)
            seed_res[name] = {
                "win_acc": float(win_m["accuracy"] * 100.0),
                "win_f1": float(win_m["macro_f1"]),
                "vid_acc": float(vid_m["accuracy"] * 100.0),
                "vid_f1": float(vid_m["macro_f1"])
            }

        # 2. Two-Stream AAGCN (Joint + Bone)
        two_feats = ["rel_3d", "bone_3d"]
        ens_two = WeightedSoftVotingEnsemble()
        ens_two.fit_window([val_probs[f] for f in two_feats], y_val_true)
        two_win_preds = ens_two.predict_window([test_probs[f] for f in two_feats])
        two_win_m = compute_metrics(y_test_true, two_win_preds)
        ens_two.fit_video([val_probs[f] for f in two_feats], y_val_true, val_video_ids)
        _, _, _, two_vid_m = ens_two.predict_video([test_probs[f] for f in two_feats], y_test_true, test_video_ids)
        seed_res["Two-Stream AAGCN (Aug)"] = {
            "win_acc": float(two_win_m["accuracy"] * 100.0),
            "win_f1": float(two_win_m["macro_f1"]),
            "vid_acc": float(two_vid_m["accuracy"] * 100.0),
            "vid_f1": float(two_vid_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in ens_two.weights_window],
            "weights_video": [round(float(w), 4) for w in ens_two.weights_video]
        }

        # 3. Four-Stream AAGCN
        four_feats = ["bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
        ens_four = WeightedSoftVotingEnsemble()
        ens_four.fit_window([val_probs[f] for f in four_feats], y_val_true)
        four_win_preds = ens_four.predict_window([test_probs[f] for f in four_feats])
        four_win_m = compute_metrics(y_test_true, four_win_preds)
        ens_four.fit_video([val_probs[f] for f in four_feats], y_val_true, val_video_ids)
        _, _, _, four_vid_m = ens_four.predict_video([test_probs[f] for f in four_feats], y_test_true, test_video_ids)
        seed_res["Four-Stream AAGCN (Aug)"] = {
            "win_acc": float(four_win_m["accuracy"] * 100.0),
            "win_f1": float(four_win_m["macro_f1"]),
            "vid_acc": float(four_vid_m["accuracy"] * 100.0),
            "vid_f1": float(four_vid_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in ens_four.weights_window],
            "weights_video": [round(float(w), 4) for w in ens_four.weights_video]
        }

        # 4. SkelGym-Lite (Transformer + Bone)
        lite_feats = ["mix", "bone_3d"]
        ens_lite = WeightedSoftVotingEnsemble()
        ens_lite.fit_window([val_probs[f] for f in lite_feats], y_val_true)
        lite_win_preds = ens_lite.predict_window([test_probs[f] for f in lite_feats])
        lite_win_m = compute_metrics(y_test_true, lite_win_preds)
        ens_lite.fit_video([val_probs[f] for f in lite_feats], y_val_true, val_video_ids)
        _, _, _, lite_vid_m = ens_lite.predict_video([test_probs[f] for f in lite_feats], y_test_true, test_video_ids)
        seed_res["SkelGym-Lite"] = {
            "win_acc": float(lite_win_m["accuracy"] * 100.0),
            "win_f1": float(lite_win_m["macro_f1"]),
            "vid_acc": float(lite_vid_m["accuracy"] * 100.0),
            "vid_f1": float(lite_vid_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in ens_lite.weights_window],
            "weights_video": [round(float(w), 4) for w in ens_lite.weights_video]
        }

        # 5. SkelGym-Full (Transformer + 4 AAGCN streams)
        full_feats = ["mix", "bone_3d", "rel_3d", "joint_motion_3d", "bone_motion_3d"]
        ens_full = WeightedSoftVotingEnsemble()
        ens_full.fit_window([val_probs[f] for f in full_feats], y_val_true)
        full_win_preds = ens_full.predict_window([test_probs[f] for f in full_feats])
        full_win_m = compute_metrics(y_test_true, full_win_preds)
        ens_full.fit_video([val_probs[f] for f in full_feats], y_val_true, val_video_ids)
        _, _, _, full_vid_m = ens_full.predict_video([test_probs[f] for f in full_feats], y_test_true, test_video_ids)
        seed_res["SkelGym-Full"] = {
            "win_acc": float(full_win_m["accuracy"] * 100.0),
            "win_f1": float(full_win_m["macro_f1"]),
            "vid_acc": float(full_vid_m["accuracy"] * 100.0),
            "vid_f1": float(full_vid_m["macro_f1"]),
            "weights_window": [round(float(w), 4) for w in ens_full.weights_window],
            "weights_video": [round(float(w), 4) for w in ens_full.weights_video]
        }

        # If seed == 42, save final confusion matrix and test predictions
        if seed == 42:
            os.makedirs("outputs/ensemble", exist_ok=True)
            plot_confusion_matrix(y_test_true, full_win_preds, "outputs/ensemble/cm_ensemble_T5.1_weighted_soft.png", normalize=True, title="SkelGym-Full (Cross-Paradigm Ensemble)")
            np.savez("outputs/ensemble/test_predictions.npz", y_true=y_test_true, y_pred=full_win_preds)

        for cfg_name, r in seed_res.items():
            print(f"  {cfg_name:<26} -> Win Acc: {r['win_acc']:>5.2f}% | Win F1: {r['win_f1']:>6.4f} | Vid Acc: {r['vid_acc']:>5.2f}% | Vid F1: {r['vid_f1']:>6.4f}")

        all_seed_eval[str(seed)] = seed_res

    # Aggregated Summary (Mean ± SD)
    all_models = list(all_seed_eval[str(seeds[0])].keys())
    summary = {}
    for m_name in all_models:
        win_accs = [all_seed_eval[str(s)][m_name]["win_acc"] for s in seeds]
        win_f1s = [all_seed_eval[str(s)][m_name]["win_f1"] for s in seeds]
        vid_accs = [all_seed_eval[str(s)][m_name]["vid_acc"] for s in seeds]
        vid_f1s = [all_seed_eval[str(s)][m_name]["vid_f1"] for s in seeds]

        summary[m_name] = {
            "win_acc_mean": float(np.mean(win_accs)),
            "win_acc_sd": float(np.std(win_accs, ddof=1)) if len(seeds) > 1 else 0.0,
            "win_f1_mean": float(np.mean(win_f1s)),
            "win_f1_sd": float(np.std(win_f1s, ddof=1)) if len(seeds) > 1 else 0.0,
            "vid_acc_mean": float(np.mean(vid_accs)),
            "vid_acc_sd": float(np.std(vid_accs, ddof=1)) if len(seeds) > 1 else 0.0,
            "vid_f1_mean": float(np.mean(vid_f1s)),
            "vid_f1_sd": float(np.std(vid_f1s, ddof=1)) if len(seeds) > 1 else 0.0,
        }

    return {"summary": summary, "per_seed": all_seed_eval}

def main():
    parser = argparse.ArgumentParser(description="Downstream Experiments with New SkelGym-Aug")
    parser.add_argument("--max_concurrent", type=int, default=6, help="Maximum concurrent training processes")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda/mps/cpu)")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining even if checkpoints exist")
    parser.add_argument("--metadata", type=str, default="data/Final_dataset_metadata.csv")
    parser.add_argument("--landmark_dir", type=str, default="data/landmarks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_file", type=str, default="outputs/new_aug_downstream_results.json")
    args = parser.parse_args()

    checkpoint_base = Path(args.checkpoint_dir)
    checkpoint_base.mkdir(parents=True, exist_ok=True)
    progress_file = Path("outputs/downstream_progress.json")
    log_dir = Path("outputs/downstream_logs")

    tasks = build_training_tasks(SEEDS, checkpoint_base)

    if not args.skip_train:
        run_parallel_training(tasks, args.max_concurrent, args.device, progress_file, log_dir, force_retrain=args.force_retrain)

    eval_device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else ("mps" if torch.backends.mps.is_available() else "cpu"))
    results = evaluate_all(SEEDS, eval_device, checkpoint_base, args.metadata, args.landmark_dir)

    # Print Summary Table
    print("\n" + "=" * 105)
    print("NEW SKELGYM-AUG DOWNSTREAM BENCHMARK (MEAN ± SD ACROSS SEEDS 42, 123, 3407)")
    print("=" * 105)
    print(f"{'Model Configuration':<28} | {'Window Accuracy':<17} | {'Window Macro-F1':<17} | {'Video Accuracy':<17} | {'Video Macro-F1':<17}")
    print("-" * 105)
    for name, st in results["summary"].items():
        print(f"{name:<28} | {st['win_acc_mean']:>5.2f}% ± {st['win_acc_sd']:>4.2f}%   | {st['win_f1_mean']:>6.4f} ± {st['win_f1_sd']:>6.4f}  | {st['vid_acc_mean']:>5.2f}% ± {st['vid_acc_sd']:>4.2f}%   | {st['vid_f1_mean']:>6.4f} ± {st['vid_f1_sd']:>6.4f}")
    print("=" * 105)

    # Save to JSON
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_path}")

    # Generate Markdown Report
    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# SkelGym-Aug New Formulation Downstream Benchmark Results\n\n")
        f.write("Evaluation across 3 independent random seeds (`42`, `123`, `3407`) using the new 4-operator SkelGym-Aug protocol (Bilateral Mirroring + 3D Yaw Rotation + Scaling + Jitter; No TimeWarp).\n\n")
        f.write("## Main Summary Table (Mean ± SD)\n\n")
        f.write("| Model Configuration | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for name, st in results["summary"].items():
            f.write(f"| **{name}** | {st['win_acc_mean']:.2f}% ± {st['win_acc_sd']:.2f}% | {st['win_f1_mean']:.4f} ± {st['win_f1_sd']:.4f} | {st['vid_acc_mean']:.2f}% ± {st['vid_acc_sd']:.2f}% | {st['vid_f1_mean']:.4f} ± {st['vid_f1_sd']:.4f} |\n")
        
        f.write("\n## Per-Seed Breakdown\n\n")
        for name in results["summary"].keys():
            f.write(f"### {name}\n\n")
            f.write("| Seed | Window Accuracy | Window Macro-F1 | Video Accuracy | Video Macro-F1 |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for s in SEEDS:
                r = results["per_seed"][str(s)][name]
                f.write(f"| {s} | {r['win_acc']:.2f}% | {r['win_f1']:.4f} | {r['vid_acc']:.2f}% | {r['vid_f1']:.4f} |\n")
            f.write("\n")
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()
