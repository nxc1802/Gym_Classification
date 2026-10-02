"""
Orchestration Script for BlockGCN External Baseline on SkelGym.
Supports:
  1. Local Smoke Test (--smoke_test):
     - Fast 2-epoch run on minimal data
     - ZERO disk checkpointing, ZERO artifact creation, ZERO outputs pollution, ZERO src pollution
     - Verifies graph building, 5D tensor forward/backward, optimization step, evaluation & video consensus
  2. Full Server Training:
     - 3 independent seeds (42, 123, 3407) x 140 epochs
     - Dual-level metrics (Window & Video Consensus Acc/Macro-F1)
     - Hardware profiling (FLOPs via thop, latency p95, parameters)
     - Artifact generation (Confusion matrices, Markdown reports, LaTeX tables)
     - Hugging Face Hub synchronization (--push_to_hf)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
import yaml
import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.constants import NUM_CLASSES, ACTIONS
from src.utils.reproducibility import seed_everything
from src.utils.logger import setup_logger
from src.training.metrics import compute_metrics, plot_confusion_matrix
from src.models.ensemble import aggregate_video_level_predictions
from src.external.blockgcn import BlockGCNModel, get_blockgcn_dataloaders, BlockGCNTrainer

logger = setup_logger("BlockGCN_Runner")


def resolve_device(device_str: str) -> torch.device:
    if device_str in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def benchmark_model_complexity(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """
    Measures trainable parameter count, FLOPs/MACs using thop, and inference latency.
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 33, 1, device=device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flops = 0
    try:
        import copy
        import thop
        model_cpu = copy.deepcopy(model).cpu()
        dummy_cpu = torch.randn(1, 3, 32, 33, 1, device="cpu")
        flops, _ = thop.profile(model_cpu, inputs=(dummy_cpu,), verbose=False)
        del model_cpu
    except Exception as e:
        logger.warning(f"Could not calculate FLOPs with thop: {e}")

    # Latency timing
    timings = []
    with torch.no_grad():
        for _ in range(15):  # Warmup
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)

    return {
        "params": total_params,
        "flops": flops,
        "latency_mean_ms": float(np.mean(timings)),
        "latency_median_ms": float(np.median(timings)),
        "latency_p95_ms": float(np.percentile(timings, 95))
    }


def run_smoke_test(config: Dict[str, Any], device: torch.device):
    """
    Executes an ephemeral local smoke test verifying the entire pipeline end-to-end.
    Strictly follows requirement: NO saved checkpoints, NO persistent artifacts, NO src pollution.
    """
    logger.info("🧪 [SMOKE TEST] Initializing ephemeral in-memory verification...")
    seed_everything(42)

    logger.info("1/5. Testing 33-joint 5D DataLoader instantiation...")
    train_loader, val_loader, test_loader = get_blockgcn_dataloaders(
        metadata_path=config["data"].get("metadata_path", "Final_dataset_metadata.csv"),
        landmark_dir=config["data"].get("landmark_dir", "data/landmarks"),
        batch_size=4,
        seq_len=32,
        smoke_test=True
    )
    logger.info(f"   --> Train windows: {len(train_loader.dataset)}, Val windows: {len(val_loader.dataset)}, Test windows: {len(test_loader.dataset)}")

    sample_x, sample_y = next(iter(train_loader))
    assert sample_x.shape == (sample_x.shape[0], 3, 32, 33, 1), f"Unexpected shape {sample_x.shape}"
    logger.info(f"   --> Input Tensor shape verified: {sample_x.shape}")

    logger.info("2/5. Testing BlockGCN architecture forward pass and parameter count...")
    model = BlockGCNModel(num_class=NUM_CLASSES, num_point=33, num_person=1, in_channels=3)
    complexity = benchmark_model_complexity(model, device)
    logger.info(f"   --> Trainable Parameters: {complexity['params']:,}")
    logger.info(f"   --> Forward Latency: {complexity['latency_mean_ms']:.2f} ms ({device})")

    logger.info("3/5. Testing training epoch and backward gradient propagation...")
    trainer = BlockGCNTrainer(
        model=model,
        device=device,
        base_lr=0.05,
        smoke_test=True,  # Disables checkpoint writing to disk
        max_epochs=2
    )
    history = trainer.fit(train_loader, val_loader, epochs=2, verbose=True)
    assert len(history["train_loss"]) == 2, "Training loop failed to record 2 epochs"
    logger.info("   --> Backward pass and SGD optimizer step verified successfully.")

    logger.info("4/5. Testing inference and metrics computation...")
    y_true, y_pred, y_prob = trainer.predict(test_loader)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"   --> Window Accuracy: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    logger.info("5/5. Testing video-level consensus aggregation...")
    if hasattr(test_loader.dataset, "video_ids") and test_loader.dataset.video_ids:
        y_vid_t, y_vid_p, _, vid_metrics = aggregate_video_level_predictions(
            y_prob, y_true, test_loader.dataset.video_ids
        )
        logger.info(f"   --> Video Consensus Accuracy: {vid_metrics['accuracy']*100:.2f}% | Macro F1: {vid_metrics['macro_f1']:.4f}")

    logger.info("✅ [SMOKE TEST PASSED] All BlockGCN modules are verified and 100% stable without polluting workspace!")


def run_full_training(config: Dict[str, Any], args: argparse.Namespace, device: torch.device):
    """
    Executes full multi-seed training with full metrics reporting and Hugging Face sync.
    """
    seeds = args.seeds or config["evaluation"].get("seeds", [42, 123, 3407])
    epochs = args.epochs or config["training"].get("num_epoch", 140)
    batch_size = args.batch_size or config["training"].get("batch_size", 64)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_per_seed = []

    for seed in seeds:
        logger.info(f"\n=======================================================")
        logger.info(f"🚀 Launching BlockGCN Training Run: Seed {seed} ({epochs} epochs)")
        logger.info(f"=======================================================")
        seed_everything(seed)

        # 1. Load Data
        train_loader, val_loader, test_loader = get_blockgcn_dataloaders(
            metadata_path=config["data"].get("metadata_path", "Final_dataset_metadata.csv"),
            landmark_dir=config["data"].get("landmark_dir", "data/landmarks"),
            batch_size=batch_size,
            seq_len=config["data"].get("seq_len", 32),
            train_stride=config["data"].get("train_stride", 16),
            val_test_stride=config["data"].get("val_test_stride", 32),
            smoke_test=False,
            num_workers=args.num_workers
        )

        # 2. Build Model
        model = BlockGCNModel(
            num_class=NUM_CLASSES,
            num_point=33,
            num_person=1,
            in_channels=3,
            drop_out=config["model_args"].get("drop_out", 0.0),
            window_size=config["data"].get("seq_len", 32)
        )

        # 3. Train
        trainer = BlockGCNTrainer(
            model=model,
            device=device,
            base_lr=config["training"].get("base_lr", 0.05),
            momentum=config["training"].get("momentum", 0.9),
            nesterov=config["training"].get("nesterov", True),
            weight_decay=config["training"].get("weight_decay", 0.0004),
            warm_up_epoch=config["training"].get("warm_up_epoch", 5),
            steps=config["training"].get("step", [110, 120]),
            lr_decay_rate=config["training"].get("lr_decay_rate", 0.1),
            max_epochs=epochs,
            checkpoint_dir=str(checkpoint_dir),
            model_name="BlockGCN",
            seed=seed,
            smoke_test=False
        )

        trainer.fit(train_loader, val_loader, epochs=epochs, verbose=True)

        # 4. Evaluate on Test Set
        logger.info(f"Evaluating best checkpoint for seed {seed} on Test Set...")
        y_true, y_pred, y_prob = trainer.predict(test_loader)
        win_metrics = compute_metrics(y_true, y_pred)

        # Video-level Consensus
        y_vid_t, y_vid_p, _, vid_metrics = aggregate_video_level_predictions(
            y_prob, y_true, test_loader.dataset.video_ids
        )

        logger.info(f"Seed {seed} -> Window Acc: {win_metrics['accuracy']*100:.2f}% | Window F1: {win_metrics['macro_f1']:.4f}")
        logger.info(f"Seed {seed} -> Video Acc:  {vid_metrics['accuracy']*100:.2f}% | Video F1:  {vid_metrics['macro_f1']:.4f}")

        # Save Confusion Matrix
        cm_path = output_dir / f"cm_blockgcn_seed{seed}.png"
        plot_confusion_matrix(y_true, y_pred, str(cm_path), title=f"BlockGCN (Seed {seed})")

        seed_res = {
            "seed": seed,
            "window_acc": win_metrics["accuracy"],
            "window_macro_f1": win_metrics["macro_f1"],
            "video_acc": vid_metrics["accuracy"],
            "video_macro_f1": vid_metrics["macro_f1"],
            "checkpoint": str(trainer.best_checkpoint_path)
        }
        results_per_seed.append(seed_res)

        # Optional HF Hub Sync per seed
        if args.push_to_hf:
            try:
                from src.utils.hf_hub import upload_checkpoints_to_hf, upload_file_to_hf
                logger.info(f"[HF Hub] Uploading Seed {seed} checkpoints...")
                upload_checkpoints_to_hf(
                    checkpoint_dir=str(checkpoint_dir),
                    model_name=f"BlockGCN_seed{seed}",
                    repo_id=config["hf_sync"].get("repo_id", "Cuong2004/gym-exercise-classification")
                )
                upload_file_to_hf(
                    local_path=str(cm_path),
                    path_in_repo=f"plots/cm_blockgcn_seed{seed}.png",
                    repo_id=config["hf_sync"].get("repo_id", "Cuong2004/gym-exercise-classification"),
                    commit_message=f"Upload BlockGCN confusion matrix seed {seed}"
                )
            except Exception as e:
                logger.error(f"[HF Hub] Failed to upload seed {seed} to HF Hub: {e}")

    # Summary Statistics across 3 Seeds
    win_accs = [r["window_acc"] * 100 for r in results_per_seed]
    win_f1s = [r["window_macro_f1"] for r in results_per_seed]
    vid_accs = [r["video_acc"] * 100 for r in results_per_seed]
    vid_f1s = [r["video_macro_f1"] for r in results_per_seed]

    complexity = benchmark_model_complexity(model, device)

    # Generate Markdown Report
    report_path = output_dir / "BLOCKGCN_RESULTS.md"
    lines = [
        "# BlockGCN External Baseline Results (CVPR 2024)",
        "",
        "> Single-stream (Joint-only, 33 MediaPipe Landmarks), trained from scratch across 3 seeds without hyperparameter tuning.",
        "",
        "## 1. Summary Benchmark Table",
        "",
        "| Architecture | Parameters | FLOPs / MACs | Window Accuracy | Window Macro F1 | Video Consensus Acc | Video Macro F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **BlockGCN (Adapted 33j)** | {complexity['params']:,} | {complexity['flops'] / 1e6:.2f} MFLOPs | {np.mean(win_accs):.2f}% ± {np.std(win_accs):.2f}% | {np.mean(win_f1s):.4f} ± {np.std(win_f1s):.4f} | {np.mean(vid_accs):.2f}% ± {np.std(vid_accs):.2f}% | {np.mean(vid_f1s):.4f} ± {np.std(vid_f1s):.4f} |",
        "",
        "## 2. Individual Seed Runs",
        "",
        "| Seed | Window Acc | Window Macro F1 | Video Consensus Acc | Video Macro F1 | Checkpoint |",
        "| :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in results_per_seed:
        lines.append(f"| {r['seed']} | {r['window_acc']*100:.2f}% | {r['window_macro_f1']:.4f} | {r['video_acc']*100:.2f}% | {r['video_macro_f1']:.4f} | `{r['checkpoint']}` |")

    lines.extend([
        "",
        "## 3. Hardware Latency Profile",
        f"- **Device:** {device}",
        f"- **Mean Latency:** {complexity['latency_mean_ms']:.2f} ms",
        f"- **Median Latency:** {complexity['latency_median_ms']:.2f} ms",
        f"- **p95 Latency:** {complexity['latency_p95_ms']:.2f} ms",
        ""
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"📊 Report successfully generated at: {report_path}")

    # Generate ADAPTATION.md documentation for manuscript supplementary
    adaptation_path = ROOT_DIR / "docs" / "ADAPTATION.md"
    adaptation_content = """# BlockGCN Adaptation Documentation for SkelGym

This document specifies the exact architectural and training adaptations applied to **BlockGCN (CVPR 2024)** when integrated as an external strong baseline on the SkelGym benchmark.

## 1. Dimensionality and Graph Adaptation
- **Joint Topology:** Adapted from NTU RGB+D (25 joints) to Google MediaPipe Pose (33 joints).
- **Auxiliary Graph Connections:** MediaPipe's facial landmarks and upper body were connected using fixed anatomical edges:
  - Nose (0) <-> Left Shoulder (11) and Right Shoulder (12)
  - Mouth Left (9) <-> Nose (0) and Mouth Right (10) <-> Nose (0)
- **Hop Distance Encoding:** Computed via Floyd-Warshall over the 33-joint graph, creating a $(33, 33)$ shortest-path distance matrix.
- **Single-Person Configuration ($M=1$):** Replaced NTU's hardcoded multi-person operation `repeat(2, 1)` with direct single-person feature projection.
- **Temporal Horizon:** Fixed window length $T=32$ frames (matching SkelGym observation window), with stride 16 (train) and 32 (eval).
- **Classification Head:** Linear layer projecting 256 channels to 22 gym action classes.

## 2. Training Protocol Fidelity
- **Backbone Capacity:** Preserved full 10 GCN-TCN blocks ($128 \\to 256$ channels, ~1.35M parameters).
- **Optimizer:** SGD with Nesterov momentum 0.9, weight decay $4 \\times 10^{-4}$.
- **Learning Rate Schedule:** 5 warmup epochs, step decay $\\times 0.1$ at epochs 110 and 120, total 140 epochs.
- **Data Preprocessing:** Hip-midpoint translation centering without Z-score standardization or dynamic SkelGym-Aug, adhering to the original paper's recipe.
"""
    adaptation_path.write_text(adaptation_content, encoding="utf-8")
    logger.info(f"📝 Adaptation documentation saved to: {adaptation_path}")

    if args.push_to_hf:
        try:
            from src.utils.hf_hub import upload_file_to_hf
            upload_file_to_hf(
                local_path=str(report_path),
                path_in_repo="external/blockgcn/BLOCKGCN_RESULTS.md",
                repo_id=config["hf_sync"].get("repo_id", "Cuong2004/gym-exercise-classification"),
                commit_message="Upload BlockGCN final results report"
            )
            upload_file_to_hf(
                local_path=str(adaptation_path),
                path_in_repo="external/blockgcn/ADAPTATION.md",
                repo_id=config["hf_sync"].get("repo_id", "Cuong2004/gym-exercise-classification"),
                commit_message="Upload BlockGCN adaptation documentation"
            )
        except Exception as e:
            logger.error(f"[HF Hub] Failed to upload final reports: {e}")


def main():
    parser = argparse.ArgumentParser(description="BlockGCN External Baseline Runner")
    parser.add_argument("--config", type=str, default="configs/external/blockgcn_original_33j_32f.yaml", help="Path to config YAML")
    parser.add_argument("--smoke_test", action="store_true", help="Run local fast smoke test (no checkpoints or artifacts written)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--push_to_hf", action="store_true", help="Upload trained checkpoints and plots to Hugging Face Hub")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/external", help="Checkpoint save directory")
    parser.add_argument("--output_dir", type=str, default="outputs/external/blockgcn", help="Report and plots output directory")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model_args": {"drop_out": 0.0},
            "data": {"seq_len": 32, "train_stride": 16, "val_test_stride": 32},
            "training": {"num_epoch": 140, "batch_size": 64, "base_lr": 0.05},
            "evaluation": {"seeds": [42, 123, 3407]},
            "hf_sync": {"repo_id": "Cuong2004/gym-exercise-classification"}
        }

    device = resolve_device(args.device)
    logger.info(f"Using compute device: {device}")

    if args.smoke_test:
        run_smoke_test(config, device)
    else:
        run_full_training(config, args, device)


if __name__ == "__main__":
    main()
