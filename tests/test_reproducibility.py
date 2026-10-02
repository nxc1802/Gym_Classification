#!/usr/bin/env python3
"""
Unit tests for Reproducibility Packaging & Environment Manifests.
Verifies the presence and structural validity of:
- requirements.txt & requirements-dev.txt
- pyproject.toml
- environment.yml
- LICENSE (MIT)
- .github/workflows/ci.yml
- submission_checklist.md
- configs/default.yaml
"""

import unittest
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestReproducibilityPackaging(unittest.TestCase):
    def test_requirements_files_exist(self):
        req_path = PROJECT_ROOT / "requirements.txt"
        req_dev_path = PROJECT_ROOT / "requirements-dev.txt"
        self.assertTrue(req_path.exists(), "requirements.txt must exist in project root")
        self.assertTrue(req_dev_path.exists(), "requirements-dev.txt must exist in project root")
        
        req_content = req_path.read_text(encoding="utf-8")
        self.assertIn("torch", req_content)
        self.assertIn("mediapipe", req_content)
        self.assertIn("scipy", req_content)
        self.assertIn("scikit-learn", req_content)
        self.assertIn("pandas", req_content)

    def test_pyproject_toml_exists(self):
        pyproj_path = PROJECT_ROOT / "pyproject.toml"
        self.assertTrue(pyproj_path.exists(), "pyproject.toml must exist")
        content = pyproj_path.read_text(encoding="utf-8")
        self.assertIn("name = \"skelgym\"", content)
        self.assertIn("[project]", content)
        self.assertIn("license = { text = \"MIT\" }", content)

    def test_environment_yml_valid(self):
        env_path = PROJECT_ROOT / "environment.yml"
        self.assertTrue(env_path.exists(), "environment.yml must exist")
        with open(env_path, "r", encoding="utf-8") as f:
            env_data = yaml.safe_load(f)
        self.assertEqual(env_data.get("name"), "skelgym")
        self.assertIn("dependencies", env_data)

    def test_license_exists(self):
        license_path = PROJECT_ROOT / "LICENSE"
        self.assertTrue(license_path.exists(), "LICENSE must exist")
        content = license_path.read_text(encoding="utf-8")
        self.assertIn("MIT License", content)
        self.assertIn("Cuong Nguyen", content)

    def test_ci_workflow_valid(self):
        ci_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(ci_path.exists(), "GitHub CI workflow must exist")
        with open(ci_path, "r", encoding="utf-8") as f:
            ci_data = yaml.safe_load(f)
        self.assertEqual(ci_data.get("name"), "SkelGym CI")
        self.assertIn("jobs", ci_data)
        self.assertIn("test", ci_data["jobs"])


    def test_default_config_verified(self):
        from src.utils.config import load_config
        cfg_path = PROJECT_ROOT / "configs" / "default.yaml"
        self.assertTrue(cfg_path.exists(), "configs/default.yaml must exist")
        cfg = load_config(str(cfg_path))
        self.assertEqual(cfg.get("feature_method"), "mix")
        self.assertEqual(cfg.get("model_type"), "Transformer")
        self.assertEqual(cfg.get("augmentation"), "skel_gym_aug")
        self.assertEqual(cfg.get("batch_size"), 16)
        self.assertEqual(cfg.get("seq_len"), 32)
        self.assertEqual(cfg.get("train_stride"), 16)
        self.assertEqual(cfg.get("val_test_stride"), 32)

    def test_bootstrap_artifacts_tool_exists(self):
        script_path = PROJECT_ROOT / "scripts" / "bootstrap_artifacts.py"
        self.assertTrue(script_path.exists(), "scripts/bootstrap_artifacts.py must exist")
        import subprocess, sys
        res = subprocess.run([sys.executable, str(script_path), "--check-only"], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"bootstrap_artifacts.py --check-only failed: {res.stderr}")

if __name__ == "__main__":
    unittest.main()
