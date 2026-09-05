"""
Comprehensive CLI for Gym Exercise Classification.
Commands:
  - download-dataset: download raw dataset from Kaggle via kagglehub.
  - data-report: generate statistics verifying Paper Table 1, and package MediaPipe landmarks.
  - push-landmarks-hf: upload processed landmarks ZIP to Hugging Face dataset repo.
  - pull-landmarks-hf: download processed landmarks ZIP from Hugging Face dataset repo.
  - preprocess: extract landmarks, build feature representations, verify dataset.
  - train: high-throughput training with AMP (bfloat16), RAM caching, HF sync, dual checkpointing.
  - evaluate: evaluate a trained checkpoint on test/val set and plot confusion matrix.
  - ensemble: combine multiple trained models with hard voting, soft voting, or stacking.
  - reproduce: automated reproduction runner matching specific tables in the publication.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from src.constants import (
    ACTIONS,
    NUM_CLASSES,
    FEATURE_DIMS,
    DEFAULT_SEQ_LEN,
    DEFAULT_TRAIN_STRIDE,
    DEFAULT_VAL_TEST_STRIDE
)
from src.utils.reproducibility import seed_everything
from src.utils.logger import setup_logger
from src.utils.config import load_config
from src.utils.hf_hub import (
    upload_file_to_hf,
    upload_checkpoints_to_hf,
    push_landmarks_to_hf,
    pull_landmarks_from_hf,
    DEFAULT_MODEL_REPO,
    DEFAULT_DATASET_REPO
)
from src.data.extractor import extract_landmarks_from_video, batch_extract_landmarks
from src.data.dataset import get_dataloaders, build_dataset_from_csvs
from src.data.download import download_kaggle_dataset
from src.data.report import generate_dataset_report, run_mediapipe_extraction_pipeline
from src.models import (
    LSTMModel,
    BiLSTMModel,
    BranchConcatModel,
    TransformerModel,
    BranchConcatTransformer,
    STGCNModel,
    HardVotingEnsemble,
    SoftVotingEnsemble,
    StackingEnsemble
)
from src.training import Trainer, compute_metrics, plot_confusion_matrix, export_latex_table7

logger = setup_logger("GymCLI")

def resolve_device(device_str: str) -> torch.device:
    """
    Intelligently resolves device string ('cuda', 'mps', 'cpu', 'auto') to active torch.device.
    """
    if device_str in ("cuda", "auto") and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            logger.info("CUDA not available. Detected Apple Silicon MPS! Accelerating on MPS device.")
            return torch.device("mps")
        else:
            return torch.device("cpu")
    if device_str == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device("cuda" if device_str in ("cuda", "auto") and torch.cuda.is_available() else device_str)

def build_model(
    model_type: str,
    feature_method: str,
    num_classes: int = NUM_CLASSES,
    hidden_dim: Optional[int] = None,
    num_layers: Optional[int] = None,
    nhead: int = 8,
    dropout: Optional[float] = None
) -> nn.Module:
    """
    Model Factory instantiating the requested architecture with compatible input dimensions.
    Defaults are calibrated to a fair ~350K parameter budget.
    """
    feat_dim = FEATURE_DIMS.get(feature_method, 36)

    if model_type == "LSTM":
        if feature_method == "branch_concat":
            return BranchConcatModel(dim1=49, dim2=286, num_classes=num_classes, hidden_dim=hidden_dim or 64, dropout=dropout or 0.3)
        h_dim = hidden_dim if hidden_dim is not None else 160
        n_layers = num_layers if num_layers is not None else 2
        drop = dropout if dropout is not None else 0.3
        return LSTMModel(feat_dim=feat_dim, num_classes=num_classes, hidden_dim=h_dim, num_layers=n_layers, dropout=drop)

    elif model_type == "BiLSTM":
        if feature_method == "branch_concat":
            return BranchConcatModel(dim1=49, dim2=286, num_classes=num_classes, hidden_dim=hidden_dim or 64, dropout=dropout or 0.3)
        h_dim = hidden_dim if hidden_dim is not None else 96
        n_layers = num_layers if num_layers is not None else 2
        drop = dropout if dropout is not None else 0.3
        return BiLSTMModel(feat_dim=feat_dim, num_classes=num_classes, hidden_dim=h_dim, num_layers=n_layers, dropout=drop)

    elif model_type == "Transformer":
        if feature_method == "branch_concat":
            return BranchConcatTransformer(dim1=49, dim2=286, num_classes=num_classes, d_model=hidden_dim or 128, nhead=nhead, num_layers=num_layers or 3, dropout=dropout or 0.2)
        h_dim = hidden_dim if hidden_dim is not None else 128
        n_layers = num_layers if num_layers is not None else 3
        drop = dropout if dropout is not None else 0.2
        return TransformerModel(feat_dim=feat_dim, num_classes=num_classes, d_model=h_dim, nhead=nhead, num_layers=n_layers, dim_feedforward=192, dropout=drop)

    elif model_type == "STGCN":
        drop = dropout if dropout is not None else 0.3
        return STGCNModel(feat_dim=feat_dim, num_classes=num_classes, dropout=drop)

    elif model_type == "BranchConcat":
        return BranchConcatModel(dim1=49, dim2=286, num_classes=num_classes, hidden_dim=hidden_dim or 64, dropout=dropout or 0.3)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def update_experiment_markdown(
    report_file: str,
    exp_id: str,
    record: Dict[str, Any]
) -> bool:
    """
    Updates the specific row in EXPERIMENT_RESULTS.md matching **{exp_id}**.
    """
    rep_path = Path(report_file)
    if not rep_path.exists():
        return False

    content = rep_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    updated = False
    new_lines = []

    for line in lines:
        if f"**{exp_id}**" in line and line.strip().startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            # Table 1: Exp ID | Model | Feature | Dim | Train Loss | Val Loss | Val Acc | Test Acc | Macro F1 | Checkpoint | Status
            if exp_id.startswith("T1."):
                if len(parts) >= 12:
                    parts[5] = f"{record.get('train_loss', 0.0):.4f}"
                    parts[6] = f"{record.get('val_loss', 0.0):.4f}"
                    parts[7] = f"{record.get('val_acc', 0.0) * 100:.2f}%"
                    parts[8] = f"{record.get('accuracy', 0.0) * 100:.2f}%"
                    parts[9] = f"{record.get('macro_f1', 0.0):.4f}"
                    parts[10] = f"`checkpoints/{record.get('checkpoint', '')}`"
                    parts[11] = "Done"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                    updated = True
            # Table 2: Exp ID | Augmentation | Config | Train Loss | Val Loss | Val Acc | Test Acc | Macro F1 | Checkpoint | Status
            elif exp_id.startswith("T2."):
                if len(parts) >= 11:
                    parts[4] = f"{record.get('train_loss', 0.0):.4f}"
                    parts[5] = f"{record.get('val_loss', 0.0):.4f}"
                    parts[6] = f"{record.get('val_acc', 0.0) * 100:.2f}%"
                    parts[7] = f"{record.get('accuracy', 0.0) * 100:.2f}%"
                    parts[8] = f"{record.get('macro_f1', 0.0):.4f}"
                    parts[9] = f"`checkpoints/{record.get('checkpoint', '')}`"
                    parts[10] = "Done"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                    updated = True
            # Table 4: Exp ID | Model | Graph Stream | Tensor Shape | Train Loss | Val Loss | Val Acc | Test Acc | Macro F1 | Checkpoint | Status
            elif exp_id.startswith("T4."):
                if len(parts) >= 12:
                    parts[5] = f"{record.get('train_loss', 0.0):.4f}"
                    parts[6] = f"{record.get('val_loss', 0.0):.4f}"
                    parts[7] = f"{record.get('val_acc', 0.0) * 100:.2f}%"
                    parts[8] = f"{record.get('accuracy', 0.0) * 100:.2f}%"
                    parts[9] = f"{record.get('macro_f1', 0.0):.4f}"
                    parts[10] = f"`checkpoints/{record.get('checkpoint', '')}`"
                    parts[11] = "Done"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                    updated = True
            # Table 5: Exp ID | Ensemble Strategy | Component Models | Test Acc | Macro F1 | Weighted F1 | Checkpoint / Artifact | Status
            elif exp_id.startswith("T5."):
                if len(parts) >= 9:
                    parts[4] = f"{record.get('accuracy', 0.0) * 100:.2f}%"
                    parts[5] = f"{record.get('macro_f1', 0.0):.4f}"
                    parts[6] = f"{record.get('weighted_f1', record.get('macro_f1', 0.0)):.4f}"
                    if "checkpoint" in record:
                        parts[7] = f"`{record['checkpoint']}`"
                    parts[8] = "Done"
                    line = "| " + " | ".join(parts[1:-1]) + " |"
                    updated = True
        new_lines.append(line)

    if updated:
        rep_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info(f"Successfully auto-updated Table row **{exp_id}** in {rep_path}")
    return updated

def append_experiment_result(
    report_file: str,
    record: Dict[str, Any],
    push_to_hf: bool = False,
    hf_repo: str = DEFAULT_MODEL_REPO,
    hf_token: Optional[str] = None
) -> None:
    """
    Appends experiment results to a single consolidated Markdown file.
    Optionally pushes the updated report to Hugging Face Hub.
    """
    rep_path = Path(report_file)
    rep_path.parent.mkdir(parents=True, exist_ok=True)

    header_needed = not rep_path.exists()
    with open(rep_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Master Experiment Results: Deep Learning for Gym Exercise Classification\n\n")
            f.write("This file aggregates all experimental evaluation results, tracking accuracy, Macro F1, and checkpoints.\n\n")
            f.write("| Timestamp | Model | Feature | Augment | Epochs | Batch | Train Loss | Val Loss | Test Acc (%) | Macro F1 | Best Checkpoint | HF Synced |\n")
            f.write("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |\n")

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"| {ts} | {record['model']} | {record['feature']} | {record['augment']} | "
            f"{record['epochs']} | {record['batch_size']} | {record.get('train_loss', 0.0):.4f} | "
            f"{record.get('val_loss', 0.0):.4f} | {record['accuracy']*100:.2f}% | "
            f"{record['macro_f1']:.4f} | `{record['checkpoint']}` | {'Yes' if push_to_hf else 'No'} |\n"
        )

    logger.info(f"Consolidated results updated in: {rep_path}")

    if push_to_hf:
        upload_file_to_hf(
            local_path=str(rep_path),
            path_in_repo="reports/EXPERIMENT_RESULTS.md",
            repo_id=hf_repo,
            token=hf_token,
            commit_message="Update master experiment results report"
        )

def cmd_download(args):
    """
    Downloads Kaggle gym exercise classification dataset.
    """
    logger.info(f"Downloading Kaggle dataset '{args.dataset}' ...")
    path = download_kaggle_dataset(args.dataset, args.output_dir)
    logger.info(f"Dataset ready at: {path}")

def cmd_data_report(args):
    """
    Generates dataset statistics report (verifying Paper Table 1) and optionally
    runs MediaPipe extraction pipeline to create lightweight server-ready CSVs.
    """
    logger.info("Generating dataset description report (verifying Paper Table 1)...")
    report_info = generate_dataset_report(args.metadata, args.output_dir)
    logger.info(f"Report generated: {report_info['md_path']}")
    logger.info(f"LaTeX Table 1 code saved: {report_info['tex_path']}")

    if args.extract_mediapipe:
        logger.info(f"Running MediaPipe pose extraction pipeline (smoke_test={args.smoke_test})...")
        pkg_path = run_mediapipe_extraction_pipeline(
            raw_dataset_dir=args.raw_dir,
            output_landmark_dir=args.landmark_dir,
            metadata_path=args.metadata,
            smoke_test=args.smoke_test,
            smoke_class=args.smoke_class,
            zip_output=True,
            num_workers=getattr(args, "num_workers", 8)
        )
        logger.info(f"Landmark packaging complete: {pkg_path}")
        if getattr(args, "push_to_hf", False) and pkg_path:
            logger.info(f"Uploading packaged landmarks {pkg_path} to HF dataset repo...")
            push_landmarks_to_hf(
                zip_path=pkg_path,
                repo_id=getattr(args, "hf_dataset_repo", DEFAULT_DATASET_REPO),
                token=getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
            )

def cmd_push_landmarks_hf(args):
    """
    Uploads processed landmarks ZIP to Hugging Face dataset repository.
    """
    token = args.token or os.environ.get("HF_TOKEN")
    logger.info(f"Pushing landmarks {args.zip_path} to HF dataset repo {args.repo_id} ...")
    url = push_landmarks_to_hf(args.zip_path, repo_id=args.repo_id, token=token)
    if url:
        logger.info(f"Landmarks pushed successfully: {url}")
    else:
        logger.error("Failed to push landmarks to HF.")

def cmd_pull_landmarks_hf(args):
    """
    Downloads and extracts processed landmarks ZIP from Hugging Face dataset repository.
    """
    token = args.token or os.environ.get("HF_TOKEN")
    logger.info(f"Pulling landmarks from HF dataset repo {args.repo_id} to {args.dest_dir} ...")
    path = pull_landmarks_from_hf(
        dest_dir=args.dest_dir,
        repo_id=args.repo_id,
        filename=args.filename,
        token=token
    )
    logger.info(f"Landmarks ready at: {path}")

def cmd_preprocess(args):
    """
    Handles preprocessing commands: extract-landmarks, verify-data, or report.
    """
    if args.subcommand == "extract-landmarks":
        logger.info(f"Running MediaPipe pose extraction (smoke_test={getattr(args, 'smoke_test', False)})...")
        pkg_path = run_mediapipe_extraction_pipeline(
            raw_dataset_dir=args.video_dir,
            output_landmark_dir=args.output_dir,
            metadata_path=getattr(args, "metadata", "Final_dataset_metadata.csv"),
            smoke_test=getattr(args, "smoke_test", False),
            smoke_class=getattr(args, "smoke_class", "barbell biceps curl"),
            zip_output=True
        )
        logger.info(f"Extraction & Packaging complete! Saved to {pkg_path}")

    elif args.subcommand == "verify-data":
        meta_p = Path(args.metadata)
        if not meta_p.exists():
            logger.error(f"Metadata file not found: {meta_p}")
            return
        df = pd.read_csv(meta_p)
        logger.info(f"Metadata Loaded: {len(df)} entries.")
        logger.info(f"Class distribution across splits:\n{df.groupby(['class', 'split']).size().unstack(fill_value=0)}")

        if args.landmark_dir:
            ld = Path(args.landmark_dir)
            if ld.exists():
                csvs = list(ld.glob("**/*.csv"))
                logger.info(f"Total landmark CSV files in {ld}: {len(csvs)}")
            else:
                logger.warning(f"Landmark dir {args.landmark_dir} does not exist.")

    elif args.subcommand == "report":
        cmd_data_report(args)

def cmd_train(args):
    """
    Trains a model according to CLI specifications.
    Optimized for high-throughput 96GB VRAM server execution with AMP and in-memory caching.
    """
    seed_everything(args.seed)

    is_smoke = getattr(args, "smoke_test", False)
    smoke_class = getattr(args, "smoke_class", "barbell biceps curl")
    if is_smoke:
        logger.info(f"[SMOKE TEST MODE ACTIVATED] Minimal dataset for '{smoke_class}', epochs=2, batch_size=4!")
        args.epochs = min(args.epochs, 2)
        args.batch_size = min(args.batch_size, 4)
        args.patience = 2

    device_str = args.device
    if device_str in ("cuda", "auto") and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            logger.info("CUDA not available. Detected Apple Silicon MPS! Accelerating on MPS device.")
            device_str = "mps"
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
    device = torch.device(device_str)
    if device_str == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    elif device_str == "mps":
        gpu_name = "Apple Silicon GPU (Metal Performance Shaders)"
    else:
        gpu_name = "CPU"
    logger.info(f"Using device: {device} ({gpu_name})")

    # Dataloaders
    in_mem = getattr(args, "in_memory", True)
    zero_frame_handling = getattr(args, "zero_frame", "zero")
    logger.info(f"Loading data: Feature={args.feature}, Augment={args.augment}, ZeroFrame={zero_frame_handling}, BatchSize={args.batch_size}, InMemory={in_mem}")
    train_loader, val_loader, test_loader = get_dataloaders(
        metadata_path=args.metadata,
        feature_method=args.feature,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.train_stride,
        val_test_stride=getattr(args, "val_test_stride", 32),
        augment_method=args.augment,
        zero_frame_handling=zero_frame_handling,
        landmark_dir=args.landmark_dir,
        num_workers=args.num_workers,
        smoke_test=is_smoke,
        smoke_class=smoke_class,
        in_memory=in_mem
    )
    logger.info(f"Dataset windows -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = build_model(
        model_type=args.model,
        feature_method=args.feature,
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Initialized {args.model} model. Trainable parameters: {num_params:,}")

    model_name = f"{args.model}_{args.feature}_aug_{args.augment}"
    if getattr(args, "exp_id", None):
        model_name = f"{args.model}_{args.exp_id}_{args.feature}"

    use_amp = getattr(args, "use_amp", True)
    amp_dtype = getattr(args, "amp_dtype", "bfloat16")
    push_to_hf = getattr(args, "push_to_hf", False)
    hf_repo = getattr(args, "hf_repo", DEFAULT_MODEL_REPO)
    hf_token = getattr(args, "hf_token", None)
    use_cw = getattr(args, "use_class_weights", False)
    label_smoothing = getattr(args, "label_smoothing", 0.0)

    trainer = Trainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        model_name=model_name,
        use_class_weights=use_cw,
        label_smoothing=label_smoothing,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        push_to_hf=push_to_hf,
        hf_repo=hf_repo,
        hf_token=hf_token
    )

    logger.info(f"Starting training for {args.epochs} epochs (EarlyStopping patience={args.patience}, AMP={use_amp} [{amp_dtype}], LabelSmoothing={label_smoothing})...")
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs, verbose=True)

    # Evaluate on test set
    logger.info("Evaluating best checkpoint on Test set...")
    y_true, y_pred, y_prob = trainer.predict(test_loader)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    # Plot confusion matrix
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_path = out_dir / f"cm_{model_name}.png"
    plot_confusion_matrix(y_true, y_pred, str(cm_path), title=f"Confusion Matrix: {model_name}")
    logger.info(f"Saved confusion matrix plot: {cm_path}")

    # Update consolidated Master Experiment Report
    report_file = getattr(args, "report_file", "outputs/EXPERIMENT_RESULTS.md")
    val_loss = history["val_loss"][-1] if history.get("val_loss") else 0.0
    val_acc = history["val_acc"][-1] if history.get("val_acc") else 0.0
    train_loss = history["train_loss"][-1] if history.get("train_loss") else 0.0

    record = {
        "model": args.model,
        "feature": args.feature,
        "augment": args.augment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "checkpoint": f"best_{model_name}.pt"
    }

    exp_id = getattr(args, "exp_id", None)
    if exp_id:
        update_experiment_markdown(report_file, exp_id, record)
    else:
        append_experiment_result(
            report_file=report_file,
            record=record,
            push_to_hf=push_to_hf,
            hf_repo=hf_repo,
            hf_token=hf_token
        )

    if push_to_hf:
        upload_file_to_hf(
            local_path=str(cm_path),
            path_in_repo=f"plots/cm_{model_name}.png",
            repo_id=hf_repo,
            token=hf_token,
            commit_message=f"Add confusion matrix for {model_name}"
        )

    return metrics

def cmd_evaluate(args):
    """
    Evaluates an existing checkpoint on test or validation split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = build_model(
        model_type=args.model,
        feature_method=args.feature,
        num_classes=NUM_CLASSES,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    _, val_loader, test_loader = get_dataloaders(
        metadata_path=args.metadata,
        feature_method=args.feature,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        landmark_dir=args.landmark_dir,
        num_workers=0
    )
    loader = test_loader if args.split == "test" else val_loader

    trainer = Trainer(model=model, device=device)
    y_true, y_pred, y_prob = trainer.predict(loader)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Evaluation on {args.split.upper()} - Acc: {metrics['accuracy']*100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    if args.save_cm:
        plot_confusion_matrix(y_true, y_pred, args.save_cm, title=f"Confusion Matrix: {args.model}")
        logger.info(f"Confusion matrix saved to {args.save_cm}")

    if args.save_table7:
        export_latex_table7(metrics["report_dict"], args.save_table7)
        logger.info(f"Table 7 exported to {args.save_table7}")

    return metrics

def cmd_ensemble(args):
    """
    Ensemble multiple trained models using hard voting, soft voting, or stacking.
    Dynamically loads appropriate dataloader for each model architecture and feature type.
    """
    device = resolve_device(args.device)
    logger.info(f"Running Ensemble method: {args.method.upper()} on {len(args.checkpoints)} checkpoints using device: {device}.")

    model_entries = []
    models_only = []
    for ckpt in args.checkpoints:
        p = Path(ckpt)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        state_dict = torch.load(p, map_location="cpu")

        # Identify model architecture
        if "STGCN" in p.name or any("gcn_conv" in k for k in state_dict.keys()):
            m_type = "STGCN"
        elif "BiLSTM" in p.name:
            m_type = "BiLSTM"
        elif "LSTM" in p.name:
            m_type = "LSTM"
        else:
            m_type = "Transformer"

        # Identify feature method from filename
        f_type = None
        for feat in ["mix", "angle_3d", "angle_2d", "raw_3d", "raw_2d", "rel_3d", "rel_2d", "branch_concat", "direct_concat", "full_rel_4", "full_4", "12rel_4", "13_4"]:
            if f"_{feat}." in p.name or f"_{feat}_" in p.name or p.name.endswith(f"_{feat}") or feat in p.name:
                f_type = feat
                break

        # Fallback: deduce from tensor dimensions in state_dict
        if f_type is None:
            if "input_proj.weight" in state_dict:
                in_dim = state_dict["input_proj.weight"].shape[1]
                for k, v in FEATURE_DIMS.items():
                    if v == in_dim:
                        f_type = k
                        break
            elif "lstm.weight_ih_l0" in state_dict:
                in_dim = state_dict["lstm.weight_ih_l0"].shape[1]
                for k, v in FEATURE_DIMS.items():
                    if v == in_dim:
                        f_type = k
                        break
            elif "data_bn.weight" in state_dict:
                c_dim = state_dict["data_bn.weight"].shape[0]
                f_type = "raw_3d" if c_dim == 3 else "raw_2d"

        if f_type is None:
            f_type = "mix" if m_type == "Transformer" else "raw_3d"

        logger.info(f"Loaded {p.name} -> Model: {m_type}, Feature: {f_type}")
        m = build_model(m_type, f_type, num_classes=NUM_CLASSES)
        m.load_state_dict(state_dict)
        m.to(device)
        m.eval()
        model_entries.append((m, m_type, f_type, p.name))
        models_only.append(m)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_preds_list = []
    all_probs_list = []
    val_probs_list = []
    y_val_final = None
    y_test_final = None

    for m, m_type, f_type, ckpt_name in model_entries:
        logger.info(f"Generating predictions for {m_type} ({f_type}) from {ckpt_name} ...")
        _, val_loader, test_loader = get_dataloaders(
            metadata_path=args.metadata,
            feature_method=f_type,
            batch_size=getattr(args, "batch_size", 64),
            landmark_dir=args.landmark_dir,
            num_workers=0,
            in_memory=True
        )
        tr = Trainer(model=m, device=device)
        if args.method == "stacking":
            y_vt, _, y_vpr = tr.predict(val_loader)
            val_probs_list.append(y_vpr)
            if y_val_final is None:
                y_val_final = y_vt

        y_t, y_p, y_pr = tr.predict(test_loader)
        all_preds_list.append(y_p)
        all_probs_list.append(y_pr)
        if y_test_final is None:
            y_test_final = y_t

    if args.method == "hard":
        ens = HardVotingEnsemble()
        final_preds = ens.predict(all_preds_list)
    elif args.method == "soft":
        ens = SoftVotingEnsemble()
        final_preds = ens.predict(all_probs_list)
    elif args.method == "stacking":
        ens = StackingEnsemble()
        ens.fit(val_probs_list, y_val_final)
        final_preds = ens.predict(all_probs_list)

    metrics = compute_metrics(y_test_final, final_preds)
    logger.info(f"Ensemble ({args.method.upper()}) Test Accuracy: {metrics['accuracy'] * 100:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    cm_path = out_dir / f"cm_ensemble_{args.method}.png"
    plot_confusion_matrix(y_test_final, final_preds, str(cm_path), title=f"Ensemble ({args.method.upper()})")
    logger.info(f"Saved ensemble confusion matrix: {cm_path}")

    report_file = getattr(args, "report_file", "outputs/EXPERIMENT_RESULTS.md")
    exp_id = getattr(args, "exp_id", None)
    if report_file and exp_id:
        update_experiment_markdown(
            report_file,
            exp_id,
            {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics.get("weighted_f1", metrics["macro_f1"]),
                "checkpoint": f"outputs/ensemble/cm_ensemble_{args.method}.png"
            }
        )

    # Auto-update Table 6 if method is stacking or requested
    if args.method == "stacking" and report_file and Path(report_file).exists():
        try:
            from sklearn.metrics import classification_report as sk_clf_report
            rep_dict = sk_clf_report(y_test_final, final_preds, target_names=ACTIONS, output_dict=True, zero_division=0)
            rep_text = Path(report_file).read_text(encoding="utf-8")
            new_lines = []
            for line in rep_text.splitlines():
                matched_act = None
                for act in ACTIONS:
                    if line.strip().startswith(f"| {act} "):
                        matched_act = act
                        break
                if matched_act and matched_act in rep_dict:
                    row_data = rep_dict[matched_act]
                    p = row_data["precision"]
                    r = row_data["recall"]
                    f1 = row_data["f1-score"]
                    supp = int(row_data["support"])
                    line = f"| {matched_act} | {p:.4f} | {r:.4f} | {f1:.4f} | {supp} |"
                elif line.strip().startswith("| **Accuracy**"):
                    acc = rep_dict.get("accuracy", metrics["accuracy"])
                    tot_supp = len(y_test_final)
                    line = f"| **Accuracy** | | | **{acc*100:.2f}%** | **{tot_supp}** |"
                elif line.strip().startswith("| **Macro avg**"):
                    m_data = rep_dict.get("macro avg", {})
                    line = f"| **Macro avg** | **{m_data.get('precision', 0):.4f}** | **{m_data.get('recall', 0):.4f}** | **{m_data.get('f1-score', 0):.4f}** | **{int(m_data.get('support', 0))}** |"
                elif line.strip().startswith("| **Weighted avg**"):
                    w_data = rep_dict.get("weighted avg", {})
                    line = f"| **Weighted avg** | **{w_data.get('precision', 0):.4f}** | **{w_data.get('recall', 0):.4f}** | **{w_data.get('f1-score', 0):.4f}** | **{int(w_data.get('support', 0))}** |"
                new_lines.append(line)
            Path(report_file).write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            logger.info(f"Updated Table 6 Classification Report in {report_file}")
        except Exception as e:
            logger.warning(f"Could not auto-update Table 6: {e}")

    if getattr(args, "push_to_hf", False):
        token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
        repo = getattr(args, "hf_repo", DEFAULT_MODEL_REPO)
        upload_file_to_hf(str(cm_path), repo_id=repo, path_in_repo=f"plots/{cm_path.name}", token=token)
        if os.path.exists(report_file):
            upload_file_to_hf(report_file, repo_id=repo, path_in_repo="reports/EXPERIMENT_RESULTS.md", token=token)

    return metrics

def cmd_reproduce(args):
    """
    Automated reproduction runner to generate results for specific paper tables.
    """
    logger.info(f"Starting reproduction runner for: {args.table.upper()}")
    target_table = args.table.lower()

    if target_table in ["table2", "all"]:
        logger.info("=== REPRODUCING TABLE 2: Temporal Models on Landmark Features ===")
        for m in ["Transformer", "BiLSTM", "LSTM"]:
            for f in ["12rel_4", "full_rel_4", "13_4", "full_4", "angle3"]:
                logger.info(f"Running Table 2 experiment: Model={m}, Feature={f}")
                args.model = m
                args.feature = f
                args.augment = "none"
                args.epochs = 1 if args.dry_run else 100
                cmd_train(args)

    if target_table in ["table3", "all"]:
        logger.info("=== REPRODUCING TABLE 3: Transformer with Augmentations ===")
        for aug in ["jitter", "rotate", "joint_dropout", "time_warp"]:
            logger.info(f"Running Table 3 experiment: Augmentation={aug}")
            args.model = "Transformer"
            args.feature = "12rel_4"
            args.augment = aug
            args.epochs = 1 if args.dry_run else 100
            cmd_train(args)

    if target_table in ["table4", "all"]:
        logger.info("=== REPRODUCING TABLE 4: Concatenated Features (Branch vs Direct) ===")
        for m in ["Transformer", "BiLSTM", "LSTM"]:
            for concat_mode in ["branch_concat", "direct_concat"]:
                logger.info(f"Running Table 4 experiment: Model={m}, Mode={concat_mode}")
                args.model = m
                args.feature = concat_mode
                args.augment = "none"
                args.epochs = 1 if args.dry_run else 100
                cmd_train(args)

    if target_table in ["table5", "all"]:
        logger.info("=== REPRODUCING TABLE 5: ST-GCN on Raw vs Relative ===")
        for f in ["full_4", "full_rel_4"]:
            logger.info(f"Running Table 5 experiment: ST-GCN, Feature={f}")
            args.model = "STGCN"
            args.feature = f
            args.augment = "none"
            args.epochs = 1 if args.dry_run else 100
            cmd_train(args)

    logger.info("Reproduction sequence finished.")

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gym Exercise Classification CLI: High-Throughput Training, HF Sync, Preprocessing & Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # Preprocess
    p_pre = subparsers.add_parser("preprocess", help="Data preprocessing and landmark extraction")
    p_pre_subs = p_pre.add_subparsers(dest="subcommand", required=True)
    
    p_extract = p_pre_subs.add_parser("extract-landmarks", help="Extract MediaPipe landmarks from videos")
    p_extract.add_argument("--video_dir", type=str, default="data/raw", help="Directory containing raw .mp4 videos")
    p_extract.add_argument("--output_dir", type=str, default="data/landmarks", help="Output directory for landmark CSVs")
    p_extract.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    p_extract.add_argument("--smoke_test", action="store_true", help="Extract minimal samples for 1 class for rapid testing")
    p_extract.add_argument("--smoke_class", type=str, default="barbell biceps curl", help="Class to extract during smoke test")
    p_extract.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")

    p_verify = p_pre_subs.add_parser("verify-data", help="Verify dataset metadata and file integrity")
    p_verify.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Path to metadata CSV")
    p_verify.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Landmarks directory")

    p_rep_sub = p_pre_subs.add_parser("report", help="Generate dataset description and verify Paper Table 1")
    p_rep_sub.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    p_rep_sub.add_argument("--output_dir", type=str, default="outputs/dataset_report", help="Directory for summary reports")
    p_rep_sub.add_argument("--extract_mediapipe", action="store_true", help="Extract MediaPipe pose landmarks and package for server")
    p_rep_sub.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw video files")
    p_rep_sub.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Output directory for landmark CSVs")
    p_rep_sub.add_argument("--smoke_test", action="store_true", help="Smoke test extraction: minimal samples for 1 class")
    p_rep_sub.add_argument("--smoke_class", type=str, default="barbell biceps curl", help="Target class for smoke test")

    # Download Dataset
    p_down = subparsers.add_parser("download-dataset", help="Download raw video dataset from Kaggle")
    p_down.add_argument("--dataset", type=str, default="truongnhatquangk18dn/the-gym-exercise-classification-dataset", help="Kaggle dataset slug")
    p_down.add_argument("--output_dir", type=str, default="data/raw", help="Target output directory")

    # Data Report
    p_rep_data = subparsers.add_parser("data-report", help="Generate dataset description, verify Paper Table 1, and extract/package landmarks")
    p_rep_data.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    p_rep_data.add_argument("--output_dir", type=str, default="outputs/dataset_report", help="Directory for summary reports")
    p_rep_data.add_argument("--extract_mediapipe", action="store_true", help="Extract MediaPipe pose landmarks and package for server")
    p_rep_data.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw video files")
    p_rep_data.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Output directory for landmark CSVs")
    p_rep_data.add_argument("--smoke_test", action="store_true", help="Smoke test extraction: minimal samples for 1 class")
    p_rep_data.add_argument("--smoke_class", type=str, default="barbell biceps curl", help="Target class for smoke test")
    p_rep_data.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker processes for MediaPipe extraction")
    p_rep_data.add_argument("--push_to_hf", action="store_true", default=False, help="Upload extracted landmarks ZIP to Hugging Face Hub")
    p_rep_data.add_argument("--hf_dataset_repo", type=str, default=DEFAULT_DATASET_REPO, help="Hugging Face dataset repo ID")
    p_rep_data.add_argument("--hf_token", type=str, default=None, help="Hugging Face authentication token")

    # Hugging Face Hub Landmarks Push / Pull
    p_push_hf = subparsers.add_parser("push-landmarks-hf", help="Upload landmarks archive to Hugging Face dataset repository")
    p_push_hf.add_argument("--zip_path", type=str, default="outputs/landmarks_smoketest.zip", help="Path to landmark ZIP archive")
    p_push_hf.add_argument("--repo_id", type=str, default=DEFAULT_DATASET_REPO, help="Hugging Face dataset repository ID")
    p_push_hf.add_argument("--token", type=str, default=None, help="Hugging Face auth token")

    p_pull_hf = subparsers.add_parser("pull-landmarks-hf", help="Download landmarks archive from Hugging Face dataset repository")
    p_pull_hf.add_argument("--dest_dir", type=str, default="data/landmarks", help="Directory to unpack landmarks into")
    p_pull_hf.add_argument("--repo_id", type=str, default=DEFAULT_DATASET_REPO, help="Hugging Face dataset repository ID")
    p_pull_hf.add_argument("--filename", type=str, default="landmarks_smoketest.zip", help="Landmarks ZIP filename in repository")
    p_pull_hf.add_argument("--token", type=str, default=None, help="Hugging Face auth token")

    # Train
    p_train = subparsers.add_parser("train", help="Train a deep learning model")
    p_train.add_argument("--model", type=str, default="Transformer", choices=["LSTM", "BiLSTM", "Transformer", "STGCN", "BranchConcat"], help="Model architecture")
    p_train.add_argument(
        "--feature", type=str, default="mix",
        choices=["raw_2d", "raw_3d", "rel_2d", "rel_3d", "angle_2d", "angle_3d", "mix", "full_4", "full_rel_4", "13_4", "12rel_4", "angle3", "angle2", "direct_concat", "branch_concat"],
        help="Feature representation method"
    )
    p_train.add_argument("--augment", type=str, default="none", choices=["none", "jitter", "rotate", "joint_dropout", "time_warp", "combined"], help="Augmentation method")
    p_train.add_argument("--zero_frame", type=str, default="zero", choices=["zero", "ffill", "linear"], help="Missing/zero-frame handling strategy")
    p_train.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing regularization parameter (e.g. 0.1)")
    p_train.add_argument("--exp_id", type=str, default=None, help="Experiment ID to automatically update outputs/EXPERIMENT_RESULTS.md (e.g. T1.1, T2.1, T2b.1, A3, PROPOSED)")
    p_train.add_argument("--smoke_test", action="store_true", help="Smoke test mode: minimal 2 epochs and minimal dataset for quick debugging")
    p_train.add_argument("--smoke_class", type=str, default="barbell biceps curl", help="Target class for smoke test debug")
    p_train.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    p_train.add_argument("--batch_size", type=int, default=16, help="Batch size (e.g. 16 for clean gradient dynamics)")
    p_train.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p_train.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    p_train.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p_train.add_argument("--seq_len", type=int, default=32, help="Sequence length in frames")
    p_train.add_argument("--train_stride", type=int, default=16, help="Stride for training sliding window")
    p_train.add_argument("--val_test_stride", type=int, default=32, help="Stride for validation and testing sliding window")
    p_train.add_argument("--use_class_weights", action="store_true", default=False, help="Use class weighting in CrossEntropyLoss")
    p_train.add_argument("--device", type=str, default="auto", choices=["cuda", "cpu", "mps", "auto"], help="Computation device")
    p_train.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv", help="Metadata CSV path")
    p_train.add_argument("--landmark_dir", type=str, default="data/landmarks", help="Directory of landmark CSVs")
    p_train.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p_train.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model weights")
    p_train.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension (None uses calibrated ~350K budget)")
    p_train.add_argument("--num_layers", type=int, default=None, help="Number of model layers (None uses calibrated ~350K budget)")
    p_train.add_argument("--nhead", type=int, default=8, help="Number of attention heads (for Transformer)")
    p_train.add_argument("--dropout", type=float, default=None, help="Dropout probability (None uses calibrated ~350K budget)")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")
    p_train.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")

    # High-Performance Acceleration & HF Options
    p_train.add_argument("--use_amp", action="store_true", default=True, help="Enable Automatic Mixed Precision (AMP)")
    p_train.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable AMP (run FP32)")
    p_train.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Precision format for AMP")
    p_train.add_argument("--in_memory", action="store_true", default=True, help="Cache all dataset tensors in RAM")
    p_train.add_argument("--push_to_hf", action="store_true", default=False, help="Upload checkpoints and results to Hugging Face Hub")
    p_train.add_argument("--hf_repo", type=str, default=DEFAULT_MODEL_REPO, help="Hugging Face Model repository ID")
    p_train.add_argument("--hf_token", type=str, default=None, help="Hugging Face authentication token")
    p_train.add_argument("--report_file", type=str, default="outputs/EXPERIMENT_RESULTS.md", help="Single consolidated results file")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    p_eval.add_argument("--model", type=str, default="Transformer", choices=["LSTM", "BiLSTM", "Transformer", "STGCN", "BranchConcat"])
    p_eval.add_argument(
        "--feature", type=str, default="mix",
        choices=["raw_2d", "raw_3d", "rel_2d", "rel_3d", "angle_2d", "angle_3d", "mix", "full_4", "full_rel_4", "13_4", "12rel_4", "angle3", "angle2", "direct_concat", "branch_concat"]
    )
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_eval.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv")
    p_eval.add_argument("--landmark_dir", type=str, default="data/landmarks")
    p_eval.add_argument("--batch_size", type=int, default=128)
    p_eval.add_argument("--seq_len", type=int, default=32)
    p_eval.add_argument("--device", type=str, default="cuda")
    p_eval.add_argument("--save_cm", type=str, default=None, help="File path to save confusion matrix image")
    p_eval.add_argument("--save_table7", type=str, default=None, help="File path to save Table 7 LaTeX code")
    p_eval.add_argument("--hidden_dim", type=int, default=128)
    p_eval.add_argument("--num_layers", type=int, default=4)
    p_eval.add_argument("--nhead", type=int, default=8)

    # Ensemble
    p_ens = subparsers.add_parser("ensemble", help="Ensemble multiple models")
    p_ens.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint .pt file paths")
    p_ens.add_argument("--method", type=str, default="stacking", choices=["hard", "soft", "stacking"], help="Ensemble method")
    p_ens.add_argument("--exp_id", type=str, default=None, help="Experiment ID to automatically update report (e.g. T5.1, T5.2, T5.3)")
    p_ens.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv")
    p_ens.add_argument("--landmark_dir", type=str, default="data/landmarks")
    p_ens.add_argument("--output_dir", type=str, default="outputs/ensemble")
    p_ens.add_argument("--batch_size", type=int, default=128)
    p_ens.add_argument("--device", type=str, default="cuda")
    p_ens.add_argument("--push_to_hf", action="store_true", default=False, help="Push ensemble plots and report to Hugging Face")
    p_ens.add_argument("--hf_repo", type=str, default=DEFAULT_MODEL_REPO, help="Hugging Face model repository ID")
    p_ens.add_argument("--hf_token", type=str, default=None, help="Hugging Face authentication token")
    p_ens.add_argument("--report_file", type=str, default="outputs/EXPERIMENT_RESULTS.md", help="Single consolidated results file")

    # Reproduce
    p_rep = subparsers.add_parser("reproduce", help="Automated reproduction for paper tables")
    p_rep.add_argument("--table", type=str, default="all", choices=["table2", "table3", "table4", "table5", "table6", "all"])
    p_rep.add_argument("--metadata", type=str, default="Final_dataset_metadata.csv")
    p_rep.add_argument("--landmark_dir", type=str, default="data/landmarks")
    p_rep.add_argument("--output_dir", type=str, default="outputs/reproduce")
    p_rep.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p_rep.add_argument("--dry_run", action="store_true", help="Quick run with 1 epoch for pipeline verification")
    p_rep.add_argument("--device", type=str, default="cuda")
    p_rep.add_argument("--batch_size", type=int, default=128)
    p_rep.add_argument("--lr", type=float, default=1e-4)
    p_rep.add_argument("--weight_decay", type=float, default=1e-4)
    p_rep.add_argument("--patience", type=int, default=20)
    p_rep.add_argument("--seq_len", type=int, default=32)
    p_rep.add_argument("--train_stride", type=int, default=16)
    p_rep.add_argument("--hidden_dim", type=int, default=128)
    p_rep.add_argument("--num_layers", type=int, default=4)
    p_rep.add_argument("--nhead", type=int, default=8)
    p_rep.add_argument("--dropout", type=float, default=0.1)
    p_rep.add_argument("--seed", type=int, default=42)
    p_rep.add_argument("--num_workers", type=int, default=0)
    p_rep.add_argument("--use_amp", action="store_true", default=True)
    p_rep.add_argument("--amp_dtype", type=str, default="bfloat16")
    p_rep.add_argument("--in_memory", action="store_true", default=True)
    p_rep.add_argument("--push_to_hf", action="store_true", default=False)
    p_rep.add_argument("--hf_repo", type=str, default=DEFAULT_MODEL_REPO)
    p_rep.add_argument("--hf_token", type=str, default=None)
    p_rep.add_argument("--report_file", type=str, default="outputs/EXPERIMENT_RESULTS.md")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "download-dataset":
        cmd_download(args)
    elif args.command == "data-report":
        cmd_data_report(args)
    elif args.command == "push-landmarks-hf":
        cmd_push_landmarks_hf(args)
    elif args.command == "pull-landmarks-hf":
        cmd_pull_landmarks_hf(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "ensemble":
        cmd_ensemble(args)
    elif args.command == "reproduce":
        cmd_reproduce(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
