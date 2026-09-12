"""
Automated unit and integration test suite for Gym Exercise Classification Pipeline.
Validates:
  1. Feature engineering (Raw, Relative, Triplet Angle, Pair Angle, Concat)
  2. Landmark sequence augmentations (Jitter, Rotate, Dropout, Time Warping)
  3. Windowing & Dataset batching
  4. Model architectures forward pass (LSTM, BiLSTM, Transformer, STGCN, BranchConcat)
  5. Ensembling (Hard, Soft, Stacking)
  6. Evaluation metrics & LaTeX export
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch

from src.constants import ACTIONS, NUM_CLASSES, RAW_POINTS_13, RAW_POINTS_33
from src.data.features import (
    extract_raw_features,
    extract_relative_features,
    compute_triplet_angles,
    compute_pair_angles,
    extract_features_by_method
)
from src.data.augmentations import LandmarkAugmenter
from src.data.dataset import sliding_windows, GymDataset
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
from src.training.metrics import compute_metrics, export_latex_table7

class TestGymPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate a synthetic DataFrame mimicking 75 frames of MediaPipe 33 landmarks
        cls.n_frames = 75
        cols = ["Frame"]
        for pt in RAW_POINTS_33:
            cols.extend([f"{pt}_x", f"{pt}_y", f"{pt}_z", f"{pt}_visibility"])
        
        data = np.random.uniform(0.1, 0.9, size=(cls.n_frames, len(cols)))
        data[:, 0] = np.arange(cls.n_frames)
        cls.df = pd.DataFrame(data, columns=cols)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_01_feature_engineering(self):
        # Test 12rel_4 (hip-midpoint-relative, all 13 joints: 13*4 + 1 avg_hip_vis = 53)
        rel_12 = extract_relative_features(self.df, RAW_POINTS_13, ["x", "y", "z", "visibility"], include_origin_vis=True)
        self.assertEqual(rel_12.shape, (self.n_frames, 53))

        # Test full_rel_4 (hip-midpoint-relative, all 33 joints: 33*4 + 1 = 133)
        rel_32 = extract_relative_features(self.df, RAW_POINTS_33, ["x", "y", "z", "visibility"], include_origin_vis=True)
        self.assertEqual(rel_32.shape, (self.n_frames, 133))

        # Test angles
        ang3 = compute_triplet_angles(self.df, RAW_POINTS_13)
        self.assertEqual(ang3.shape, (self.n_frames, 286))
        self.assertTrue(np.all(ang3 >= 0.0) and np.all(ang3 <= np.pi + 1e-5))

        ang2 = compute_pair_angles(self.df, RAW_POINTS_13)
        self.assertEqual(ang2.shape, (self.n_frames, 78))

        # Test direct concat (53 + 286 = 339)
        direct = extract_features_by_method(self.df, "direct_concat")
        self.assertEqual(direct.shape, (self.n_frames, 339))

        # Test branch concat
        b1, b2 = extract_features_by_method(self.df, "branch_concat")
        self.assertEqual(b1.shape, (self.n_frames, 53))
        self.assertEqual(b2.shape, (self.n_frames, 286))

    def test_02_augmentations(self):
        aug = LandmarkAugmenter()
        x = torch.randn(32, 49)

        # Jitter
        x_jit = aug.apply(x, "jitter")
        self.assertEqual(x_jit.shape, x.shape)
        self.assertFalse(torch.equal(x, x_jit))

        # Rotate
        x_rot = aug.apply(x, "rotate")
        self.assertEqual(x_rot.shape, x.shape)

        # Dropout
        x_drop = aug.apply(x, "joint_dropout")
        self.assertEqual(x_drop.shape, x.shape)

        # Time warp
        x_warp = aug.apply(x, "time_warp")
        self.assertEqual(x_warp.shape, x.shape)

        # Mirror
        x_mir = aug.apply(x, "mirror")
        self.assertEqual(x_mir.shape, x.shape)

        # Speed perturb
        x_sp = aug.apply(x, "speed_perturb")
        self.assertEqual(x_sp.shape, x.shape)

        # Generate augmented variants (should return 3 variants)
        variants = aug.generate_augmented_variants(x, "combined")
        self.assertEqual(len(variants), 3)
        for v in variants:
            self.assertEqual(v.shape, x.shape)

    def test_03_sliding_windows_and_padding(self):
        # Normal sequence 75 frames, seq_len 32, stride 16
        arr = np.random.randn(75, 53)
        wins = sliding_windows(arr, seq_len=32, stride=16)
        # Windows: [0:32], [16:48], [32:64], [48:75 -> 27 frames >= 16 -> interpolated]
        self.assertGreaterEqual(len(wins), 3)
        for w in wins:
            self.assertEqual(w.shape, (32, 53))

        # Short sequence 20 frames >= 50% of 32 -> should be interpolated to 32
        arr_short = np.random.randn(20, 53)
        wins_short = sliding_windows(arr_short, seq_len=32, stride=16)
        self.assertEqual(len(wins_short), 1)
        self.assertEqual(wins_short[0].shape, (32, 53))

        # Very short sequence 10 frames < 50% of 32 -> should be discarded
        arr_tiny = np.random.randn(10, 53)
        wins_tiny = sliding_windows(arr_tiny, seq_len=32, stride=16)
        self.assertEqual(len(wins_tiny), 0)

    def test_04_model_forward_passes(self):
        B, T = 4, 32

        # 1. LSTM
        m_lstm = LSTMModel(feat_dim=53, num_classes=NUM_CLASSES).to(self.device)
        out_lstm = m_lstm(torch.randn(B, T, 53).to(self.device))
        self.assertEqual(out_lstm.shape, (B, NUM_CLASSES))

        # 2. BiLSTM
        m_bilstm = BiLSTMModel(feat_dim=53, num_classes=NUM_CLASSES).to(self.device)
        out_bilstm = m_bilstm(torch.randn(B, T, 53).to(self.device))
        self.assertEqual(out_bilstm.shape, (B, NUM_CLASSES))

        # 3. Transformer
        m_tf = TransformerModel(feat_dim=53, num_classes=NUM_CLASSES).to(self.device)
        out_tf = m_tf(torch.randn(B, T, 53).to(self.device))
        self.assertEqual(out_tf.shape, (B, NUM_CLASSES))

        # 4. BranchConcat (LSTM & Transformer)
        m_bc_lstm = BranchConcatModel(dim1=53, dim2=286, num_classes=NUM_CLASSES).to(self.device)
        x_tuple = (torch.randn(B, T, 53).to(self.device), torch.randn(B, T, 286).to(self.device))
        out_bc_lstm = m_bc_lstm(x_tuple)
        self.assertEqual(out_bc_lstm.shape, (B, NUM_CLASSES))

        m_bc_tf = BranchConcatTransformer(dim1=53, dim2=286, num_classes=NUM_CLASSES).to(self.device)
        out_bc_tf = m_bc_tf(x_tuple)
        self.assertEqual(out_bc_tf.shape, (B, NUM_CLASSES))

        # 5. STGCN
        m_stgcn_33 = STGCNModel(feat_dim=132, num_classes=NUM_CLASSES, num_joints=33).to(self.device)
        out_stgcn = m_stgcn_33(torch.randn(B, T, 132).to(self.device))
        self.assertEqual(out_stgcn.shape, (B, NUM_CLASSES))

    def test_05_ensembles(self):
        N = 50
        y_true = np.random.randint(0, NUM_CLASSES, size=N)

        p1 = np.random.uniform(size=(N, NUM_CLASSES))
        p1 /= p1.sum(axis=1, keepdims=True)
        p2 = np.random.uniform(size=(N, NUM_CLASSES))
        p2 /= p2.sum(axis=1, keepdims=True)
        p3 = np.random.uniform(size=(N, NUM_CLASSES))
        p3 /= p3.sum(axis=1, keepdims=True)

        # Hard voting
        hard_ens = HardVotingEnsemble()
        preds_hard = hard_ens.predict([np.argmax(p1, 1), np.argmax(p2, 1), np.argmax(p3, 1)])
        self.assertEqual(len(preds_hard), N)

        # Soft voting
        soft_ens = SoftVotingEnsemble()
        preds_soft = soft_ens.predict([p1, p2, p3])
        self.assertEqual(len(preds_soft), N)

        # Stacking
        stack_ens = StackingEnsemble()
        stack_ens.fit([p1, p2, p3], y_true)
        preds_stack = stack_ens.predict([p1, p2, p3])
        self.assertEqual(len(preds_stack), N)

    def test_06_metrics_and_latex_export(self):
        N = 100
        y_true = np.random.randint(0, NUM_CLASSES, size=N)
        y_pred = y_true.copy()
        y_pred[:10] = (y_pred[:10] + 1) % NUM_CLASSES

        metrics = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.90, places=2)
        self.assertIn("report_dict", metrics)

        # LaTeX Table 7 export
        out_tex = "outputs/test_table7.tex"
        tex_code = export_latex_table7(metrics["report_dict"], out_tex)
        self.assertIn("Classification report for the stacking ensemble", tex_code)
        self.assertIn("barbell biceps curl", tex_code)

    def test_07_data_report_and_smoke_dataset(self):
        from src.data.report import generate_dataset_report
        from src.data.dataset import get_dataloaders

        rep = generate_dataset_report(metadata_path="Final_dataset_metadata.csv", output_report_dir="outputs/test_report")
        self.assertIn(rep["frame_stats"]["total_videos"], (1024, 1026))
        self.assertTrue(Path(rep["md_path"]).exists())
        self.assertTrue(Path(rep["tex_path"]).exists())

        # 2. Test smoke_test dataloaders
        tr_l, va_l, te_l = get_dataloaders(
            metadata_path="Final_dataset_metadata.csv",
            feature_method="12rel_4",
            batch_size=4,
            smoke_test=True,
            smoke_class="barbell biceps curl"
        )
        # Smoke test should load only a few windows (minimal dataset)
        self.assertLess(len(tr_l.dataset), 50)
        self.assertGreater(len(tr_l.dataset), 0)
        self.assertGreater(len(va_l.dataset), 0)
        self.assertGreater(len(te_l.dataset), 0)

    def test_08_dynamic_features_and_dual_target_ensemble(self):
        from src.constants import get_feature_dimension
        from src.models import WeightedSoftVotingEnsemble

        # Test dynamic feature dimensions
        self.assertEqual(get_feature_dimension("rel_3d+angle2_3d"), 117)
        self.assertEqual(get_feature_dimension("mix:rel_3d,angle2_3d"), 117)
        self.assertEqual(get_feature_dimension("rel_3d"), 39)
        self.assertEqual(get_feature_dimension("angle2_3d"), 78)

        # Test feature extraction with compound string
        feat_compound = extract_features_by_method(self.df, "rel_3d+angle2_3d")
        self.assertEqual(feat_compound.shape, (self.n_frames, 117))

        # Test Dual-Target Weighted Soft Voting
        N_val = 60
        N_test = 40
        M = 2  # 2 models

        val_labels = np.random.randint(0, NUM_CLASSES, size=N_val)
        val_p1 = np.eye(NUM_CLASSES)[val_labels] * 0.8 + 0.2 / NUM_CLASSES
        val_p2 = np.random.uniform(size=(N_val, NUM_CLASSES))
        val_p2 /= val_p2.sum(axis=1, keepdims=True)

        val_video_ids = [f"vid_{i // 3}" for i in range(N_val)]
        test_video_ids = [f"vid_test_{i // 2}" for i in range(N_test)]

        test_p1 = np.random.uniform(size=(N_test, NUM_CLASSES))
        test_p1 /= test_p1.sum(axis=1, keepdims=True)
        test_p2 = np.random.uniform(size=(N_test, NUM_CLASSES))
        test_p2 /= test_p2.sum(axis=1, keepdims=True)

        ens = WeightedSoftVotingEnsemble()
        
        # Window-level optimization
        ens.fit_window([val_p1, val_p2], val_labels)
        self.assertEqual(len(ens.weights_window), M)
        self.assertAlmostEqual(sum(ens.weights_window), 1.0, places=4)
        # Model 1 is much better, so weight of model 1 should be significantly higher
        self.assertGreater(ens.weights_window[0], ens.weights_window[1])

        win_preds = ens.predict_window([test_p1, test_p2])
        self.assertEqual(len(win_preds), N_test)

        # Video-level optimization
        test_labels = np.random.randint(0, NUM_CLASSES, size=N_test)
        ens.fit_video([val_p1, val_p2], val_labels, val_video_ids)
        self.assertEqual(len(ens.weights_video), M)
        self.assertAlmostEqual(sum(ens.weights_video), 1.0, places=4)

        y_vid_true, y_vid_pred, final_vid_probs, metrics = ens.predict_video([test_p1, test_p2], test_labels, test_video_ids)
        self.assertEqual(len(y_vid_true), N_test // 2)
        self.assertEqual(len(y_vid_pred), N_test // 2)
        self.assertIn("accuracy", metrics)

if __name__ == "__main__":
    unittest.main()

