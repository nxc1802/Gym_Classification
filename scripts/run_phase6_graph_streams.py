#!/usr/bin/env python3
"""
Phase 6: Table 3 Graph Kinematic Streams Benchmark (AAGCN & ST-GCN).
Evaluates individual graph streams across 3 random seeds (42, 123, 3407):
  T3.1: STGCN Raw 3D (Clean)
  T3.2: STGCN World 3D (Clean)
  T3.3: AAGCN Bone 3D (Clean)
  T3.4: AAGCN Bone 3D + Proposed SkelGym-Aug
  T3.5: AAGCN World Joint + Proposed SkelGym-Aug
  T3.6: AAGCN World Joint Motion (Delta X) + Proposed SkelGym-Aug
  T3.7: AAGCN Bone Motion (Delta B) + Proposed SkelGym-Aug
And multi-stream fusion (Uniform Soft Voting):
  T3.8: Two-Stream AAGCN (T3.5 + T3.4)
  T3.9: Four-Stream AAGCN (T3.5 + T3.4 + T3.6 + T3.7)

Outputs: outputs/table3_graph_streams_results.json
"""

import os
import sys
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.constants import ACTIONS, ACTION_TO_IDX
from src.data.dataset import get_dataloaders
from src.cli import build_model
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics
from src.models.ensemble import aggregate_video_level_predictions

SEEDS = [42, 123, 3407]

def send_marimo_toast(msg: str):
    try:
        import marimo as mo
        mo.status.toast(msg)
    except Exception:
        pass

def train_graph_stream_run(
    exp_id: str,
    model_type: str,
    feature_method: str,
    aug_method: str,
    seed: int,
    metadata_path: str,
    device: torch.device,
    checkpoint_dir: Path,
    resume: bool = True
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n---> Training {exp_id} ({model_type} on {feature_method}, aug: {aug_method}) [Seed {seed}]...")
    t_start = time.time()

    # Graph streams use batch_size 32, lr 1e-3, label_smoothing 0.05
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=metadata_path,
        feature_method=feature_method,
        batch_size=32,
        seq_len=32,
        stride=16,
        val_test_stride=32,
        augment_method=aug_method if aug_method != "none" else None,
        zero_frame_handling="interpolate",
        landmark_dir=None,
        in_memory=True,
        seed=seed,
        save_norm_artifact=True,
        strict_norm=False
    )

    model = build_model(model_type=model_type, feature_method=feature_method)

    seed_ckpt_dir = checkpoint_dir / f"seed{seed}"
    seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"{model_type}_{exp_id}_{feature_method}_seed{seed}"
    best_ckpt_path = seed_ckpt_dir / f"best_{model_name}.pt"

    trainer = Trainer(
        model=model,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir=str(seed_ckpt_dir),
        model_name=model_name,
        label_smoothing=0.05,
        early_stopping_metric="val_macro_f1",
        use_amp=torch.cuda.is_available(),
        feature_method=feature_method,
        augment_method=aug_method,
        seed=seed
    )

    if resume and best_ckpt_path.exists():
        print(f"  --> [Resuming] Found existing checkpoint {best_ckpt_path.name}, skipping training.")
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        train_time = 0.0
    else:
        history = trainer.fit(train_loader, val_loader, epochs=100)
        train_time = time.time() - t_start

    # Evaluate on Validation
    y_val, _, val_probs = trainer.predict(val_loader)
    val_preds = np.argmax(val_probs, axis=1)
    val_m = compute_metrics(y_val, val_preds)
    _, _, _, val_vid_m = aggregate_video_level_predictions(val_probs, y_val, val_loader.dataset.video_ids)

    # Evaluate on Test
    y_test, _, test_probs = trainer.predict(test_loader)
    test_preds = np.argmax(test_probs, axis=1)
    test_m = compute_metrics(y_test, test_preds)
    _, _, _, test_vid_m = aggregate_video_level_predictions(test_probs, y_test, test_loader.dataset.video_ids)

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
        "exp_id": exp_id,
        "model": model_type,
        "feature": feature_method,
        "aug": aug_method,
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
        "val_probs": val_probs.tolist(),
        "test_probs": test_probs.tolist(),
        "val_targets": y_val.tolist(),
        "test_targets": y_test.tolist(),
        "val_video_ids": list(val_loader.dataset.video_ids),
        "test_video_ids": list(test_loader.dataset.video_ids),
        "checkpoint": str(best_ckpt_path)
    }

    print(f"[{exp_id} | Seed {seed}] Val Vid Acc: {res['val_vid_acc']}%, Val Vid F1: {res['val_vid_f1']} | Test Vid Acc: {res['test_vid_acc']}%, Test Vid F1: {res['test_vid_f1']}")
    return res

def evaluate_multi_stream_fusion(
    stream_results: List[Dict[str, Any]],
    stream_keys: List[str],
    fusion_name: str
) -> Dict[str, Any]:
    """
    Evaluates multi-stream fusion (Uniform Soft Voting: w_i = 1/K) across seeds.
    """
    seed_fusions = []
    for s in SEEDS:
        matching_runs = [stream_results[k][s] for k in stream_keys]
        val_probs_list = [np.array(r["val_probs"]) for r in matching_runs]
        test_probs_list = [np.array(r["test_probs"]) for r in matching_runs]
        val_targets = np.array(matching_runs[0]["val_targets"])
        test_targets = np.array(matching_runs[0]["test_targets"])
        val_vids = matching_runs[0]["val_video_ids"]
        test_vids = matching_runs[0]["test_video_ids"]

        # Uniform average soft voting
        val_fused_probs = np.mean(val_probs_list, axis=0)
        test_fused_probs = np.mean(test_probs_list, axis=0)

        val_preds = np.argmax(val_fused_probs, axis=1)
        test_preds = np.argmax(test_fused_probs, axis=1)

        val_m = compute_metrics(val_targets, val_preds)
        test_m = compute_metrics(test_targets, test_preds)

        _, _, _, val_vid_m = aggregate_video_level_predictions(val_fused_probs, val_targets, val_vids)
        _, _, _, test_vid_m = aggregate_video_level_predictions(test_fused_probs, test_targets, test_vids)

        seed_fusions.append({
            "seed": s,
            "val_win_acc": round(float(val_m["accuracy"] * 100.0), 2),
            "val_win_f1": round(float(val_m["macro_f1"]), 4),
            "val_vid_acc": round(float(val_vid_m["accuracy"] * 100.0), 2),
            "val_vid_f1": round(float(val_vid_m["macro_f1"]), 4),
            "test_win_acc": round(float(test_m["accuracy"] * 100.0), 2),
            "test_win_f1": round(float(test_m["macro_f1"]), 4),
            "test_vid_acc": round(float(test_vid_m["accuracy"] * 100.0), 2),
            "test_vid_f1": round(float(test_vid_m["macro_f1"]), 4),
            "val_probs": val_fused_probs.tolist(),
            "test_probs": test_fused_probs.tolist()
        })

    val_w = [r["val_win_acc"] for r in seed_fusions]
    val_wf1 = [r["val_win_f1"] for r in seed_fusions]
    val_v = [r["val_vid_acc"] for r in seed_fusions]
    val_vf1 = [r["val_vid_f1"] for r in seed_fusions]
    test_w = [r["test_win_acc"] for r in seed_fusions]
    test_wf1 = [r["test_win_f1"] for r in seed_fusions]
    test_v = [r["test_vid_acc"] for r in seed_fusions]
    test_vf1 = [r["test_vid_f1"] for r in seed_fusions]

    return {
        "fusion_name": fusion_name,
        "streams": stream_keys,
        "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
        "val_win_f1": f"{np.mean(val_wf1):.4f} ± {np.std(val_wf1):.4f}",
        "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
        "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
        "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
        "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
        "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
        "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
        "runs": seed_fusions
    }

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Table 3 Graph Kinematic Streams")
    parser.add_argument("--aug_method", type=str, default="skel_gym_aug", help="Proposed SkelGym-Aug method name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing checkpoints if available (default: True)")
    parser.add_argument("--force-retrain", action="store_true", default=False, help="Force retrain even if checkpoints exist")
    args = parser.parse_args()

    resume = args.resume and not args.force_retrain

    device = torch.device(args.device)
    print("=" * 80)
    print(f"PHASE 6: Running Table 3 Graph Kinematic Streams ({device}, resume={resume})")
    print(f"Proposed Augmentation: {args.aug_method}")
    print("=" * 80)

    meta_cand = ROOT_DIR / "data" / "Final_dataset_metadata.csv"
    if not meta_cand.exists():
        meta_cand = ROOT_DIR / "Final_dataset_metadata.csv"

    checkpoint_dir = ROOT_DIR / "checkpoints" / "graph_streams"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stream_specs = [
        {"id": "T3.1", "model": "STGCN", "feat": "raw_3d", "aug": "none", "desc": "Raw 3D Joint (ST-GCN Clean)"},
        {"id": "T3.2", "model": "STGCN", "feat": "world_3d", "aug": "none", "desc": "World 3D Joint (ST-GCN Clean)"},
        {"id": "T3.3", "model": "AAGCN", "feat": "bone_3d", "aug": "none", "desc": "Bone 3D Stream (AAGCN Clean)"},
        {"id": "T3.4", "model": "AAGCN", "feat": "bone_3d", "aug": args.aug_method, "desc": "Bone 3D Stream (AAGCN + Aug)"},
        {"id": "T3.5", "model": "AAGCN", "feat": "world_3d", "aug": args.aug_method, "desc": "World Joint Stream (AAGCN + Aug)"},
        {"id": "T3.6", "model": "AAGCN", "feat": "world_joint_motion_3d", "aug": args.aug_method, "desc": "World Joint Motion (Delta X)"},
        {"id": "T3.7", "model": "AAGCN", "feat": "bone_motion_3d", "aug": args.aug_method, "desc": "Bone Motion (Delta B)"},
    ]

    stream_results_by_id = {}
    stream_runs_by_id_and_seed = {}

    for spec in stream_specs:
        eid = spec["id"]
        runs = []
        stream_runs_by_id_and_seed[eid] = {}
        print(f"\n================================================================================")
        print(f"Executing Stream: {eid} ({spec['desc']}) across 3 seeds concurrently...")
        print(f"================================================================================")
        send_marimo_toast(f"⚡ Phase 6 Parallel: Training {eid} ({spec['desc']}) across 3 seeds!")

        worker_script = ROOT_DIR / "scripts" / "train_single_graph_stream.py"
        procs = []
        logs = {}

        for s in SEEDS:
            (checkpoint_dir / f"seed{s}").mkdir(parents=True, exist_ok=True)
            out_json = checkpoint_dir / f"seed{s}" / f"result_{eid}_seed{s}.json"
            log_file = checkpoint_dir / f"seed{s}" / f"train_{eid}_seed{s}.log"
            logs[s] = (log_file, out_json)

            cmd = [
                sys.executable, "-u", str(worker_script),
                "--exp_id", eid,
                "--model_type", spec["model"],
                "--feature_method", spec["feat"],
                "--aug_method", spec["aug"],
                "--seed", str(s),
                "--device", args.device,
                "--checkpoint_dir", str(checkpoint_dir),
                "--metadata_path", str(meta_cand),
                "--out_json", str(out_json)
            ]
            if resume:
                cmd.append("--resume")
            else:
                cmd.append("--force-retrain")

            lf = open(log_file, "w")
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True)
            procs.append((s, p, lf, out_json))

        # Wait for all 3 seeds to finish
        t_start_stream = time.time()
        while True:
            all_done = all(p.poll() is not None for _, p, _, _ in procs)
            if all_done:
                break
            time.sleep(4)
            elapsed = time.time() - t_start_stream
            status_str = " | ".join([f"Seed {s}: {'DONE' if p.poll() is not None else 'RUNNING'}" for s, p, _, _ in procs])
            print(f"  [{elapsed:.0f}s elapsed] {eid} -> {status_str}", end="\r", flush=True)

        print(f"\nAll 3 seeds for {eid} completed in {time.time() - t_start_stream:.1f}s.")

        for s, p, lf, out_json in procs:
            lf.close()
            if p.returncode != 0:
                print(f"Warning: Stream {eid} seed {s} exited with returncode {p.returncode}! Log: {logs[s][0]}")
            if out_json.exists():
                with open(out_json) as f:
                    res = json.load(f)
                probs_npz = checkpoint_dir / f"seed{s}" / f"probs_{eid}_seed{s}.npz"
                if probs_npz.exists():
                    try:
                        pdata = np.load(probs_npz, allow_pickle=True)
                        res["train_probs"] = pdata["train_probs"].tolist()
                        res["train_targets"] = pdata["train_targets"].tolist()
                    except Exception:
                        pass
                runs.append(res)
                stream_runs_by_id_and_seed[eid][s] = res
                print(f"  -> Seed {s}: Val Vid {res['val_vid_acc']}%, Test Vid {res['test_vid_acc']}% (F1: {res['test_vid_f1']})")
            else:
                raise RuntimeError(f"Output JSON missing for stream {eid} seed {s}: {out_json}")

        val_w = [r["val_win_acc"] for r in runs]
        val_wf1 = [r["val_win_f1"] for r in runs]
        val_v = [r["val_vid_acc"] for r in runs]
        val_vf1 = [r["val_vid_f1"] for r in runs]
        test_w = [r["test_win_acc"] for r in runs]
        test_wf1 = [r["test_win_f1"] for r in runs]
        test_v = [r["test_vid_acc"] for r in runs]
        test_vf1 = [r["test_vid_f1"] for r in runs]

        # Strip large prob arrays for compact summary
        clean_runs = [{k: v for k, v in r.items() if not k.endswith("_probs") and not k.endswith("_targets") and not k.endswith("_ids")} for r in runs]

        stream_results_by_id[eid] = {
            "exp_id": eid,
            "description": spec["desc"],
            "model": spec["model"],
            "feature": spec["feat"],
            "aug": spec["aug"],
            "val_win_acc": f"{np.mean(val_w):.2f}% ± {np.std(val_w):.2f}%",
            "val_win_f1": f"{np.mean(val_wf1):.4f} ± {np.std(val_wf1):.4f}",
            "val_vid_acc": f"{np.mean(val_v):.2f}% ± {np.std(val_v):.2f}%",
            "val_vid_f1": f"{np.mean(val_vf1):.4f} ± {np.std(val_vf1):.4f}",
            "test_win_acc": f"{np.mean(test_w):.2f}% ± {np.std(test_w):.2f}%",
            "test_win_f1": f"{np.mean(test_wf1):.4f} ± {np.std(test_wf1):.4f}",
            "test_vid_acc": f"{np.mean(test_v):.2f}% ± {np.std(test_v):.2f}%",
            "test_vid_f1": f"{np.mean(test_vf1):.4f} ± {np.std(test_vf1):.4f}",
            "runs": clean_runs
        }

    # Evaluate Multi-Stream Ensembles
    # T3.8: Two-Stream (T3.5 World Joint + T3.4 Bone)
    two_stream = evaluate_multi_stream_fusion(stream_runs_by_id_and_seed, ["T3.5", "T3.4"], "Two-Stream AAGCN (World Joint + Bone)")
    two_stream_clean_runs = [{k: v for k, v in r.items() if not k.endswith("_probs")} for r in two_stream["runs"]]
    two_stream["runs"] = two_stream_clean_runs
    stream_results_by_id["T3.8"] = two_stream

    # T3.9: Four-Stream (T3.5 + T3.4 + T3.6 + T3.7)
    four_stream = evaluate_multi_stream_fusion(stream_runs_by_id_and_seed, ["T3.5", "T3.4", "T3.6", "T3.7"], "Four-Stream AAGCN (Full)")
    four_stream_clean_runs = [{k: v for k, v in r.items() if not k.endswith("_probs")} for r in four_stream["runs"]]
    four_stream["runs"] = four_stream_clean_runs
    stream_results_by_id["T3.9"] = four_stream

    print("\n" + "=" * 90)
    print("## TABLE 3: GRAPH KINEMATIC STREAMS BENCHMARK RESULTS")
    print("=" * 90)
    print("| Exp ID | Stream / Configuration | Val Vid Acc (%) | Val Vid F1 | Test Win Acc (%) | Test Win F1 | Test Vid Acc (%) | Test Vid F1 |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for eid, rep in stream_results_by_id.items():
        desc = rep.get("description", rep.get("fusion_name", eid))
        print(f"| **{eid}** | {desc} | {rep['val_vid_acc']} | {rep['val_vid_f1']} | {rep['test_win_acc']} | {rep['test_win_f1']} | **{rep['test_vid_acc']}** | **{rep['test_vid_f1']}** |")
    print("=" * 90)

    out_file = ROOT_DIR / "outputs" / "table3_graph_streams_results.json"
    with open(out_file, "w") as f:
        json.dump(stream_results_by_id, f, indent=2)

    pred_cache_file = ROOT_DIR / "outputs" / "table3_stream_predictions.pt"
    torch.save(stream_runs_by_id_and_seed, pred_cache_file)
    print(f"Saved stream predictions cache to {pred_cache_file}")

    send_marimo_toast("Phase 6 Complete: Table 3 Graph Streams benchmark finished successfully!")
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    main()
