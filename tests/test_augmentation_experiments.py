import pytest
import json
import tempfile
from pathlib import Path

from scripts.run_augmentation_experiments import (
    get_task_hyperparameters,
    generate_loo_latex_table,
    generate_single_component_latex_table,
    get_validation_metrics,
    build_task_list
)
from src.constants import CANONICAL_EXPERIMENT_REGISTRY

def test_task_hyperparameters_cross_backbone_parity():
    """Verify that get_task_hyperparameters matches canonical registry values for all backbones."""
    # 1. Transformer
    trans_hp = get_task_hyperparameters("Transformer", "mix", "skel_gym_aug")
    assert trans_hp["lr"] == 1e-4
    assert trans_hp["batch_size"] == 16
    assert trans_hp["label_smoothing"] == 0.05
    assert trans_hp["early_stopping_metric"] == "val_macro_f1"

    # 2. AAGCN
    aagcn_hp = get_task_hyperparameters("AAGCN", "bone_3d", "skel_gym_aug")
    assert aagcn_hp["lr"] == 1e-3
    assert aagcn_hp["batch_size"] == 32
    assert aagcn_hp["label_smoothing"] == 0.05

    # 3. BiLSTM
    bilstm_hp = get_task_hyperparameters("BiLSTM", "mix", "none")
    assert bilstm_hp["lr"] == 1e-3
    assert bilstm_hp["batch_size"] == 16
    assert bilstm_hp["label_smoothing"] == 0.0

    # 4. STGCN
    stgcn_hp = get_task_hyperparameters("STGCN", "rel_3d", "none")
    assert stgcn_hp["lr"] == 1e-3
    assert stgcn_hp["batch_size"] == 32
    assert stgcn_hp["label_smoothing"] == 0.0

def test_latex_table_generators_smoke():
    """Verify that table generators produce valid LaTeX tables without errors."""
    mock_summary_loo = {
        "Candidate_Full_5op": {
            "win_acc_mean": 66.41, "win_acc_std": 0.45,
            "win_f1_mean": 0.6550, "win_f1_std": 0.0050,
            "val_acc_mean": 68.20, "val_acc_std": 0.30,
            "val_loss_mean": 0.9500, "val_loss_std": 0.0200
        },
        "Minus_TimeWarp": {
            "win_acc_mean": 67.82, "win_acc_std": 0.38,
            "win_f1_mean": 0.6690, "win_f1_std": 0.0040,
            "val_acc_mean": 69.10, "val_acc_std": 0.25,
            "val_loss_mean": 0.9200, "val_loss_std": 0.0150
        }
    }
    loo_tex = generate_loo_latex_table(mock_summary_loo)
    assert r"\begin{table*}" in loo_tex
    assert "Candidate Full (All 5 Operators)" in loo_tex
    assert "+1.41\\%" in loo_tex
    assert "(Optimal)" not in loo_tex

    mock_summary_single = {
        "Clean_Baseline_NoAug": {
            "win_acc_mean": 63.35, "win_acc_std": 0.50,
            "win_f1_mean": 0.6200, "win_f1_std": 0.0060,
            "val_acc_mean": 65.00, "val_acc_std": 0.40,
            "val_loss_mean": 1.0500, "val_loss_std": 0.0300
        },
        "Single_Mirror": {
            "win_acc_mean": 64.50, "win_acc_std": 0.40,
            "win_f1_mean": 0.6350, "win_f1_std": 0.0050,
            "val_acc_mean": 66.10, "val_acc_std": 0.35,
            "val_loss_mean": 1.0100, "val_loss_std": 0.0250
        }
    }
    single_tex = generate_single_component_latex_table(mock_summary_single)
    assert r"\begin{table*}" in single_tex
    assert "Clean Baseline (No Augmentation)" in single_tex
    assert "+1.15\\%" in single_tex

def test_provenance_epoch_parsing():
    """Verify that get_validation_metrics extracts epoch when best_epoch or epoch is present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        ckpt_file = tmp_dir / "best_model.pt"
        ckpt_file.write_bytes(b"dummy")

        prov_file = tmp_dir / "best_model.provenance.json"
        prov_file.write_text(json.dumps({
            "val_acc": 0.75,
            "val_loss": 0.85,
            "val_macro_f1": 0.74,
            "train_loss": 0.45,
            "epoch": 42
        }))

        metrics = get_validation_metrics(ckpt_file, tmp_dir / "dummy.log")
        assert metrics["val_acc"] == 75.0
        assert metrics["val_loss"] == 0.85
        assert metrics["val_macro_f1"] == 0.74
        assert metrics["best_epoch"] == 42
