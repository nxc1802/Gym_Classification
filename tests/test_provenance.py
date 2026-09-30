"""
Unit and integration tests for Experiment Provenance and Checkpoint Integrity (Roadmap Part 2).
Validates:
  1. Git commit hash extraction and repo cleanliness reporting.
  2. SHA-256 computation of checkpoint files.
  3. Provenance metadata generation and .provenance.json sidecar persistence.
  4. Backward-compatible loading of both legacy flat state dicts and rich provenance checkpoints.
  5. Strict checkpoint lookup in ablation runners (preventing cross-variant contamination).
  6. Force-retrain semantics across runners.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer
from src.utils.reproducibility import (
    get_git_commit_hash,
    is_git_repo_dirty,
    compute_file_sha256,
    save_provenance_metadata,
    load_checkpoint_weights,
)
from scripts.run_augmentation_experiments import find_existing_checkpoint

class DummyModel(nn.Module):
    def __init__(self, in_features=10, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.fc(x)

class TestProvenanceAndCheckpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ckpt_dir = self.temp_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_git_utilities(self):
        """Validates that git commit hash is non-empty and is_git_repo_dirty returns a bool."""
        git_sha = get_git_commit_hash(short=True)
        self.assertIsInstance(git_sha, str)
        self.assertTrue(len(git_sha) > 0)
        self.assertNotEqual(git_sha, "")

        is_dirty = is_git_repo_dirty()
        self.assertIsInstance(is_dirty, bool)

    def test_02_sha256_computation(self):
        """Validates SHA-256 computation against a known byte sequence."""
        test_file = self.temp_dir / "sample.bin"
        test_bytes = b"SkelGym Biomechanical Consistency and Provenance Verification"
        test_file.write_bytes(test_bytes)

        import hashlib
        expected_sha = hashlib.sha256(test_bytes).hexdigest()
        computed_sha = compute_file_sha256(test_file)
        self.assertEqual(computed_sha, expected_sha)

    def test_03_trainer_provenance_sidecar(self):
        """Validates that Trainer saves both .pt with metadata and .provenance.json with SHA-256."""
        device = torch.device("cpu")
        model = DummyModel(in_features=10, num_classes=3)
        
        # Synthetic tiny dataset
        x_train = torch.randn(8, 32, 10)
        y_train = torch.randint(0, 3, (8,))
        x_val = torch.randn(4, 32, 10)
        y_val = torch.randint(0, 3, (4,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=4)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=4)

        norm_stats = {
            "mean_shape": [117],
            "is_normalized": True,
            "mean_norm": 0.42,
            "std_norm": 1.15
        }

        trainer = Trainer(
            model=model,
            device=device,
            lr=1e-3,
            checkpoint_dir=str(self.ckpt_dir),
            model_name="DummyTestModel",
            use_amp=False,
            feature_method="mix",
            augment_method="skel_gym_aug",
            seed=123,
            normalization_stats=norm_stats,
            provenance_metadata={"notes": "unit_test_run"}
        )

        history = trainer.fit(train_loader, val_loader, epochs=2, verbose=False)
        self.assertIn("train_loss", history)

        best_ckpt = self.ckpt_dir / "best_DummyTestModel.pt"
        sidecar_json = self.ckpt_dir / "best_DummyTestModel.provenance.json"

        self.assertTrue(best_ckpt.exists(), "best_DummyTestModel.pt was not saved!")
        self.assertTrue(sidecar_json.exists(), "best_DummyTestModel.provenance.json sidecar was not saved!")

        # Verify sidecar JSON contents
        with open(sidecar_json, "r", encoding="utf-8") as f:
            prov = json.load(f)

        self.assertEqual(prov["model_name"], "DummyTestModel")
        self.assertEqual(prov["feature_method"], "mix")
        self.assertEqual(prov["augment_method"], "skel_gym_aug")
        self.assertEqual(prov["seed"], 123)
        self.assertIn("git_sha", prov)
        self.assertIn("checkpoint_sha256", prov)
        self.assertEqual(prov["checkpoint_file"], "best_DummyTestModel.pt")
        self.assertEqual(prov["normalization_stats"], norm_stats)
        self.assertEqual(prov["notes"], "unit_test_run")

        # Verify checkpoint SHA-256 matches sidecar
        actual_sha = compute_file_sha256(best_ckpt)
        self.assertEqual(prov["checkpoint_sha256"], actual_sha)

    def test_04_backward_compatible_checkpoint_loading(self):
        """Validates that load_checkpoint_weights loads both new rich dicts and legacy flat state dicts."""
        model = DummyModel(in_features=10, num_classes=3)
        
        # 1. Legacy flat state dict
        legacy_path = self.ckpt_dir / "legacy_model.pt"
        torch.save(model.state_dict(), legacy_path)

        loaded_weights, prov = load_checkpoint_weights(legacy_path, device="cpu")
        self.assertIn("fc.weight", loaded_weights)
        self.assertEqual(prov, {})

        # 2. Wrapped rich payload
        rich_path = self.ckpt_dir / "rich_model.pt"
        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": 10,
            "provenance": {"seed": 3407, "model_name": "TestRich"}
        }
        torch.save(payload, rich_path)

        loaded_rich_weights, rich_prov = load_checkpoint_weights(rich_path, device="cpu")
        self.assertIn("fc.weight", loaded_rich_weights)
        self.assertEqual(rich_prov.get("seed"), 3407)
        self.assertEqual(rich_prov.get("model_name"), "TestRich")

    def test_05_strict_checkpoint_matching_prevents_cross_contamination(self):
        """
        Validates that an ablation task (e.g. -Mirror or -Yaw) will NEVER match
        a generic baseline checkpoint like seed123/best_Transformer_mix.pt.
        """
        # Create a generic baseline checkpoint in seed123
        seed_dir = self.ckpt_dir / "seed123"
        seed_dir.mkdir(parents=True, exist_ok=True)
        generic_ckpt = seed_dir / "best_Transformer_mix.pt"
        generic_ckpt.write_bytes(b"0" * 2000)

        # LOO task: -Mirror on seed 123
        task_no_mirror = {
            "id": "LOO_Trans_NoMirror_seed123",
            "model": "Transformer",
            "feature": "mix",
            "augment": "skel_gym_aug_no_mirror",
            "seed": 123
        }

        # Must NOT match the generic checkpoint!
        matched = find_existing_checkpoint(task_no_mirror, self.ckpt_dir, force_retrain=False)
        self.assertIsNone(matched, "CRITICAL: LOO_Trans_NoMirror_seed123 erroneously matched generic best_Transformer_mix.pt!")

        # Create exact task checkpoint
        exact_ckpt = self.ckpt_dir / "ablation_aug" / "best_LOO_Trans_NoMirror_seed123.pt"
        exact_ckpt.parent.mkdir(parents=True, exist_ok=True)
        exact_ckpt.write_bytes(b"1" * 2000)

        matched_exact = find_existing_checkpoint(task_no_mirror, self.ckpt_dir, force_retrain=False)
        self.assertEqual(matched_exact, exact_ckpt, "Failed to match exact task checkpoint!")

        # With force_retrain=True, must return None even if exact exists
        matched_forced = find_existing_checkpoint(task_no_mirror, self.ckpt_dir, force_retrain=True)
        self.assertIsNone(matched_forced, "force_retrain=True must return None to trigger clean retraining!")

if __name__ == "__main__":
    unittest.main()
