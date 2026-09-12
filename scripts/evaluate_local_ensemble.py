import os
import sys
import time
import logging
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_model
from src.data.dataset import get_dataloaders
from src.training.trainer import Trainer
from src.training.metrics import compute_metrics, plot_confusion_matrix
from src.models.ensemble import WeightedSoftVotingEnsemble, aggregate_video_level_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LocalEval")

def run_local_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using local inference device: {device}")

    metadata_path = "data/Final_dataset_metadata.csv"
    landmark_dir = "data/landmarks"

    if not os.path.exists(metadata_path) or not os.path.exists(landmark_dir):
        logger.error(f"Required data paths not found: metadata={metadata_path}, landmarks={landmark_dir}")
        return

    # Checkpoint configuration for Grand 5-Stream SOTA Ensemble
    component_models = [
        {"name": "Transformer Mix (Aug)", "ckpt": "checkpoints/best_Transformer_T2.2_mix.pt", "model": "Transformer", "feat": "mix"},
        {"name": "AAGCN Joint (Aug)", "ckpt": "checkpoints/best_AAGCN_T4.3_rel_3d.pt", "model": "AAGCN", "feat": "rel_3d"},
        {"name": "AAGCN Bone (Aug)", "ckpt": "checkpoints/best_AAGCN_T4.2_bone_3d.pt", "model": "AAGCN", "feat": "bone_3d"},
        {"name": "AAGCN J-Motion (Aug)", "ckpt": "checkpoints/best_AAGCN_T4.4_joint_motion_3d.pt", "model": "AAGCN", "feat": "joint_motion_3d"},
        {"name": "AAGCN B-Motion (Aug)", "ckpt": "checkpoints/best_AAGCN_T4.5_bone_motion_3d.pt", "model": "AAGCN", "feat": "bone_motion_3d"}
    ]

    val_probs_all = []
    test_probs_all = []
    y_val_true = None
    y_test_true = None
    val_video_ids = None
    test_video_ids = None

    results_table = []

    logger.info("=" * 70)
    logger.info("PHASE 1: LOCAL EVALUATION OF INDIVIDUAL COMPONENT MODELS")
    logger.info("=" * 70)

    for cfg in component_models:
        p = Path(cfg["ckpt"])
        if not p.exists():
            logger.error(f"Checkpoint not found: {p}. Please ensure it is downloaded from HF.")
            return

        logger.info(f"\n--- Evaluating {cfg['name']} ({cfg['model']} - {cfg['feat']}) ---")
        state_dict = torch.load(p, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        m = build_model(cfg["model"], cfg["feat"], num_classes=22)
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()

        # Dataloaders
        train_l, val_l, test_l = get_dataloaders(
            metadata_path=metadata_path,
            feature_method=cfg["feat"],
            batch_size=32,
            seq_len=32,
            stride=32,
            val_test_stride=32,
            landmark_dir=landmark_dir,
            num_workers=0,
            in_memory=True
        )

        if val_video_ids is None and hasattr(val_l.dataset, "video_ids"):
            val_video_ids = val_l.dataset.video_ids
        if test_video_ids is None and hasattr(test_l.dataset, "video_ids"):
            test_video_ids = test_l.dataset.video_ids

        trainer = Trainer(model=m, device=device)

        # 1. Validation Set Inference
        t0 = time.time()
        y_vt, y_vp, y_vprob = trainer.predict(val_l)
        val_time = time.time() - t0
        if y_val_true is None:
            y_val_true = y_vt
        val_probs_all.append(y_vprob)

        val_win_acc = np.mean(y_vt == y_vp) * 100
        _, _, _, val_vid_metrics = aggregate_video_level_predictions(y_vprob, y_vt, val_video_ids)
        val_vid_acc = val_vid_metrics["accuracy"] * 100

        # 2. Test Set Inference
        t0 = time.time()
        y_tt, y_tp, y_tprob = trainer.predict(test_l)
        test_time = time.time() - t0
        if y_test_true is None:
            y_test_true = y_tt
        test_probs_all.append(y_tprob)

        test_win_acc = np.mean(y_tt == y_tp) * 100
        _, _, _, test_vid_metrics = aggregate_video_level_predictions(y_tprob, y_tt, test_video_ids)
        test_vid_acc = test_vid_metrics["accuracy"] * 100

        # Latency (ms / window)
        num_windows = len(y_tt)
        latency_ms = (test_time / num_windows) * 1000
        fps = num_windows / test_time

        logger.info(f"[{cfg['name']}]")
        logger.info(f"  Validation -> Window Acc: {val_win_acc:.2f}% | Video Acc: {val_vid_acc:.2f}%")
        logger.info(f"  Test Set   -> Window Acc: {test_win_acc:.2f}% | Video Acc: {test_vid_acc:.2f}%")
        logger.info(f"  Inference  -> {latency_ms:.2f} ms/window ({fps:.1f} windows/sec on {device})")

        results_table.append({
            "name": cfg["name"],
            "val_win_acc": val_win_acc,
            "val_vid_acc": val_vid_acc,
            "test_win_acc": test_win_acc,
            "test_vid_acc": test_vid_acc,
            "latency_ms": latency_ms
        })

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: ENSEMBLE SELECTION & WEIGHT LEARNING (STRICTLY ON VALIDATION)")
    logger.info("=" * 70)

    ens = WeightedSoftVotingEnsemble()

    # Step A: Solve Window Weights on Val (minimizing Val Window NLL)
    logger.info("Solving Window SLSQP weights on Validation Set...")
    ens.fit_window(val_probs_all, y_val_true)
    w_win = [round(float(w), 4) for w in ens.weights_window]
    logger.info(f"Learned Window Weights (w_win): {w_win}")

    # Validate Window Ensemble on Val
    ens_val_win_preds = ens.predict_window(val_probs_all)
    ens_val_win_acc = np.mean(y_val_true == ens_val_win_preds) * 100
    logger.info(f"🔥 Validation Window Accuracy of Grand Ensemble: {ens_val_win_acc:.2f}%")

    # Step B: Solve Video Weights on Val (minimizing Val Video NLL)
    logger.info("Solving Video SLSQP weights on Validation Set...")
    ens.fit_video(val_probs_all, y_val_true, val_video_ids)
    w_vid = [round(float(w), 4) for w in ens.weights_video]
    logger.info(f"Learned Video Weights (w_vid): {w_vid}")

    # Validate Video Ensemble on Val
    _, _, _, ens_val_vid_metrics = ens.predict_video(val_probs_all, y_val_true, val_video_ids)
    ens_val_vid_acc = ens_val_vid_metrics["accuracy"] * 100
    logger.info(f"🔥 Validation Video Accuracy of Grand Ensemble: {ens_val_vid_acc:.2f}%")

    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: UNBIASED HELD-OUT TEST EVALUATION (FINAL BENCHMARK)")
    logger.info("=" * 70)

    # Step C: Evaluate Window Ensemble on Test
    ens_test_win_preds = ens.predict_window(test_probs_all)
    test_win_metrics = compute_metrics(y_test_true, ens_test_win_preds)
    ens_test_win_acc = test_win_metrics["accuracy"] * 100
    ens_test_win_f1 = test_win_metrics["macro_f1"]

    # Step D: Evaluate Video Ensemble on Test
    y_test_vid_t, y_test_vid_p, _, ens_test_vid_metrics = ens.predict_video(test_probs_all, y_test_true, test_video_ids)
    ens_test_vid_acc = ens_test_vid_metrics["accuracy"] * 100
    ens_test_vid_f1 = ens_test_vid_metrics["macro_f1"]

    logger.info(f"🌟 HELD-OUT TEST Window Accuracy: {ens_test_win_acc:.2f}% | Macro F1: {ens_test_win_f1:.4f}")
    logger.info(f"🌟 HELD-OUT TEST Video Accuracy : {ens_test_vid_acc:.2f}% | Macro F1: {ens_test_vid_f1:.4f}")

    # Save final ensemble confusion matrix
    cm_path = "outputs/ensemble/cm_ensemble_T5.1_weighted_soft.png"
    os.makedirs("outputs/ensemble", exist_ok=True)
    plot_confusion_matrix(y_test_true, ens_test_win_preds, cm_path, title="Grand 5-Stream SOTA Ensemble (Weighted Soft)")
    logger.info(f"Verified & Saved final ensemble confusion matrix: {cm_path}")

    # Summary comparison table
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY COMPARISON TABLE: VALIDATION vs TEST")
    logger.info("=" * 70)
    print(f"{'Model / Architecture':<26} | {'Val Win Acc':<11} | {'Val Vid Acc':<11} | {'Test Win Acc':<12} | {'Test Vid Acc':<12} | {'Lat. (ms)':<9}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['name']:<26} | {r['val_win_acc']:>10.2f}% | {r['val_vid_acc']:>10.2f}% | {r['test_win_acc']:>11.2f}% | {r['test_vid_acc']:>11.2f}% | {r['latency_ms']:>8.2f}ms")
    print("-" * 95)
    print(f"{'Grand 5-Stream Ensemble':<26} | {ens_val_win_acc:>10.2f}% | {ens_val_vid_acc:>10.2f}% | {ens_test_win_acc:>11.2f}% | {ens_test_vid_acc:>11.2f}% | {'~' + str(round(sum(r['latency_ms'] for r in results_table), 2)) + 'ms':>9}")
    print("=" * 95)

if __name__ == '__main__':
    run_local_evaluation()
