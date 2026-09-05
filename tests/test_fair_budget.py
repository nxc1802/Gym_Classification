"""
Unit and Integration Tests for Controlled Parameter Budget (~350K) and Advanced Features.
Validates:
  1. Parameter counts for LSTM, BiLSTM, Transformer, STGCN all within ~350K +- 15%
  2. New feature extraction (raw_2d, raw_3d, rel_2d, rel_3d, angle_2d, angle_3d, mix)
  3. Zero-frame handling (zero, ffill, linear interpolation)
  4. Label smoothing loss computation
  5. Auto-updating EXPERIMENT_RESULTS.md via update_experiment_markdown
"""

import unittest
import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.constants import NUM_CLASSES, RAW_POINTS_13, RAW_POINTS_33, FEATURE_DIMS
from src.data.features import (
    extract_raw_features,
    extract_relative_features,
    compute_triplet_angles_2d,
    compute_triplet_angles_3d,
    extract_mix_features,
    extract_features_by_method
)
from src.data.dataset import handle_zero_frames
from src.models import (
    LSTMModel,
    BiLSTMModel,
    TransformerModel,
    STGCNModel
)
from src.cli import build_model, update_experiment_markdown

class TestFairBudgetAndFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_frames = 60
        cols = ["Frame"]
        for pt in RAW_POINTS_33:
            cols.extend([f"{pt}_x", f"{pt}_y", f"{pt}_z", f"{pt}_visibility"])
        data = np.random.uniform(0.1, 0.9, size=(cls.n_frames, len(cols)))
        data[:, 0] = np.arange(cls.n_frames)
        cls.df = pd.DataFrame(data, columns=cols)

    def test_01_feature_shapes(self):
        # raw_2d: 13 * 2 = 26
        raw_2d = extract_features_by_method(self.df, "raw_2d")
        self.assertEqual(raw_2d.shape, (self.n_frames, 26))

        # raw_3d: 13 * 3 = 39
        raw_3d = extract_features_by_method(self.df, "raw_3d")
        self.assertEqual(raw_3d.shape, (self.n_frames, 39))

        # rel_2d: 12 * 2 = 24
        rel_2d = extract_features_by_method(self.df, "rel_2d")
        self.assertEqual(rel_2d.shape, (self.n_frames, 24))

        # rel_3d: 12 * 3 = 36
        rel_3d = extract_features_by_method(self.df, "rel_3d")
        self.assertEqual(rel_3d.shape, (self.n_frames, 36))

        # angle_2d: 286
        ang_2d = extract_features_by_method(self.df, "angle_2d")
        self.assertEqual(ang_2d.shape, (self.n_frames, 286))

        # angle_3d: 286
        ang_3d = extract_features_by_method(self.df, "angle_3d")
        self.assertEqual(ang_3d.shape, (self.n_frames, 286))

        # mix: 36 + 286 = 322
        mix = extract_features_by_method(self.df, "mix")
        self.assertEqual(mix.shape, (self.n_frames, 322))
        self.assertFalse(np.isnan(mix).any())

    def test_02_zero_frame_handling(self):
        df_corrupt = self.df.copy()
        # Corrupt frames 10..15 by setting coordinates to 0.0
        coord_cols = [c for c in df_corrupt.columns if any(c.endswith(f"_{d}") for d in ["x", "y", "z"])]
        df_corrupt.loc[10:15, coord_cols] = 0.0

        # Linear interpolation
        df_linear = handle_zero_frames(df_corrupt, method="linear")
        # Ensure interpolated frames are not zero
        self.assertTrue((df_linear.loc[10:15, coord_cols] != 0.0).any().any())
        self.assertFalse(df_linear[coord_cols].isna().any().any())

        # Forward fill
        df_ffill = handle_zero_frames(df_corrupt, method="ffill")
        self.assertTrue((df_ffill.loc[10:15, coord_cols] != 0.0).any().any())

    def test_03_model_parameter_budgets(self):
        # Target: ~350K parameters +- 15% (297K to 402K)
        min_p = 290_000
        max_p = 405_000

        # LSTM on rel_3d (dim=36)
        lstm = build_model("LSTM", "rel_3d")
        p_lstm = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
        print(f"LSTM params: {p_lstm:,}")
        self.assertTrue(min_p <= p_lstm <= max_p, f"LSTM params {p_lstm} outside range [{min_p}, {max_p}]")

        # BiLSTM on rel_3d (dim=36)
        bilstm = build_model("BiLSTM", "rel_3d")
        p_bilstm = sum(p.numel() for p in bilstm.parameters() if p.requires_grad)
        print(f"BiLSTM params: {p_bilstm:,}")
        self.assertTrue(min_p <= p_bilstm <= max_p, f"BiLSTM params {p_bilstm} outside range [{min_p}, {max_p}]")

        # Transformer on rel_3d (dim=36)
        trans = build_model("Transformer", "rel_3d")
        p_trans = sum(p.numel() for p in trans.parameters() if p.requires_grad)
        print(f"Transformer params: {p_trans:,}")
        self.assertTrue(min_p <= p_trans <= max_p, f"Transformer params {p_trans} outside range [{min_p}, {max_p}]")

        # STGCN on rel_3d (dim=36)
        stgcn = build_model("STGCN", "rel_3d")
        p_stgcn = sum(p.numel() for p in stgcn.parameters() if p.requires_grad)
        print(f"STGCN params: {p_stgcn:,}")
        self.assertTrue(min_p <= p_stgcn <= max_p, f"STGCN params {p_stgcn} outside range [{min_p}, {max_p}]")

    def test_04_forward_pass_all_models(self):
        x = torch.randn(2, 32, 36)  # (B=2, T=32, D=36)
        
        lstm = build_model("LSTM", "rel_3d")
        out_lstm = lstm(x)
        self.assertEqual(out_lstm.shape, (2, NUM_CLASSES))

        bilstm = build_model("BiLSTM", "rel_3d")
        out_bilstm = bilstm(x)
        self.assertEqual(out_bilstm.shape, (2, NUM_CLASSES))

        trans = build_model("Transformer", "rel_3d")
        out_trans = trans(x)
        self.assertEqual(out_trans.shape, (2, NUM_CLASSES))

        stgcn = build_model("STGCN", "rel_3d")
        out_stgcn = stgcn(x)
        self.assertEqual(out_stgcn.shape, (2, NUM_CLASSES))

    def test_05_label_smoothing_loss(self):
        criterion_standard = nn.CrossEntropyLoss(label_smoothing=0.0)
        criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

        logits = torch.randn(4, NUM_CLASSES)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        loss_std = criterion_standard(logits, targets)
        loss_sm = criterion_smooth(logits, targets)

        self.assertTrue(torch.is_tensor(loss_std))
        self.assertTrue(torch.is_tensor(loss_sm))
        self.assertFalse(torch.isnan(loss_sm))

    def test_06_markdown_table_update(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tf:
            tf.write("""# Test Benchmark

| Exp ID | Model Architecture | Capacity / Config | # Params | Train Loss | Val Loss | Val Acc (%) | Test Acc (%) | Macro F1 | Checkpoint Path | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **T1.1** | **LSTM** | 2 layers, `hidden_dim = 160`, dropout 0.3 | $\approx 360\text{K}$ | - | - | - | - | - | `checkpoints/best_LSTM_table1.pt` | Pending |
""")
            tf_path = tf.name

        record = {
            "train_loss": 0.8234,
            "val_loss": 1.2345,
            "val_acc": 0.5678,
            "accuracy": 0.5512,
            "macro_f1": 0.5432,
            "checkpoint": "best_LSTM_table1.pt"
        }

        updated = update_experiment_markdown(tf_path, "T1.1", record)
        self.assertTrue(updated)

        content = Path(tf_path).read_text()
        self.assertIn("0.8234", content)
        self.assertIn("55.12%", content)
        self.assertIn("Done", content)
        Path(tf_path).unlink()

if __name__ == "__main__":
    unittest.main()
