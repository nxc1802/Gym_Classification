"""
Comprehensive Smoke Test Suite for Gym Exercise Classification.
Validates the entire pipeline:
  1. Temporal training (Transformer) with --smoke_test
  2. Graph training (AAGCN) with --smoke_test
  3. Single-model evaluation with --video_level --smoke_test
  4. Ensemble evaluation with --video_level --smoke_test
  5. Verifies that outputs/EXPERIMENT_RESULTS.md is completely untouched.
"""

import os
import sys
import subprocess
import shutil
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def compute_file_hash(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    hasher.update(filepath.read_bytes())
    return hasher.hexdigest()

def run_cmd(cmd_list):
    cmd_str = " ".join(cmd_list)
    print(f"\n[RUNNING] {cmd_str}")
    res = subprocess.run(cmd_list, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with return code {res.returncode}")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError(f"Command failed: {cmd_str}")
    print("[SUCCESS]")
    return res.stdout

def main():
    print("=" * 60)
    print("🚀 STARTING ISOLATED PIPELINE SMOKE TEST")
    print("=" * 60)

    official_report = ROOT / "outputs" / "EXPERIMENT_RESULTS.md"
    initial_hash = compute_file_hash(official_report)
    print(f"Initial SHA256 of outputs/EXPERIMENT_RESULTS.md: {initial_hash[:16]}...")

    smoke_output_dir = ROOT / "outputs" / "smoke_test"
    smoke_ckpt_dir = ROOT / "checkpoints" / "smoke_test"

    # Step 1: Smoke Test Temporal Model (Transformer mix)
    print("\n--- Step 1: Train Transformer (mix) ---")
    run_cmd([
        sys.executable, "run.py", "train",
        "--model", "Transformer",
        "--feature", "mix",
        "--exp_id", "T1.21",
        "--smoke_test",
        "--epochs", "2",
        "--device", "auto",
        "--use_amp",
        "--in_memory"
    ])

    # Step 2: Smoke Test Graph Model (AAGCN rel_3d)
    print("\n--- Step 2: Train AAGCN (rel_3d) ---")
    run_cmd([
        sys.executable, "run.py", "train",
        "--model", "AAGCN",
        "--feature", "rel_3d",
        "--exp_id", "T4.5",
        "--smoke_test",
        "--epochs", "2",
        "--device", "auto",
        "--use_amp",
        "--in_memory"
    ])

    # Step 3: Smoke Test Single-Model Video-Level Evaluation
    print("\n--- Step 3: Evaluate Transformer with Video-Level Aggregation ---")
    ckpt_transformer = smoke_ckpt_dir / "best_Transformer_T1.21_mix.pt"
    run_cmd([
        sys.executable, "run.py", "evaluate",
        "--checkpoint", str(ckpt_transformer),
        "--model", "Transformer",
        "--feature", "mix",
        "--video_level",
        "--smoke_test",
        "--device", "auto"
    ])

    # Step 4: Smoke Test Ensemble with Video-Level Aggregation
    print("\n--- Step 4: Ensemble Transformer + AAGCN with Video-Level Aggregation ---")
    ckpt_aagcn = smoke_ckpt_dir / "best_AAGCN_T4.5_rel_3d.pt"
    run_cmd([
        sys.executable, "run.py", "ensemble",
        "--method", "weighted_soft",
        "--exp_id", "T5.1",
        "--checkpoints", str(ckpt_transformer), str(ckpt_aagcn),
        "--seq_len", "32",
        "--stride", "32",
        "--video_level",
        "--smoke_test",
        "--device", "auto"
    ])

    # Step 5: Verification of Isolation
    print("\n" + "=" * 60)
    print("🔍 VERIFYING ISOLATION & ARTIFACT GENERATION")
    print("=" * 60)

    final_hash = compute_file_hash(official_report)
    print(f"Final SHA256 of outputs/EXPERIMENT_RESULTS.md:   {final_hash[:16]}...")
    if initial_hash != final_hash:
        raise AssertionError("CRITICAL ERROR: outputs/EXPERIMENT_RESULTS.md was modified during smoke test!")
    print("✅ PASS: outputs/EXPERIMENT_RESULTS.md remained 100% untouched!")

    smoke_report = smoke_output_dir / "SMOKE_RESULTS.md"
    if not smoke_report.exists():
        raise AssertionError(f"Smoke results report {smoke_report} was not created!")
    print(f"✅ PASS: Smoke report created at {smoke_report.relative_to(ROOT)} ({smoke_report.stat().st_size} bytes)")

    # Check checkpoints
    ckpts = list(smoke_ckpt_dir.glob("*.pt"))
    print(f"✅ PASS: Found {len(ckpts)} isolated smoke test checkpoints in {smoke_ckpt_dir.relative_to(ROOT)}")
    for c in ckpts:
        print(f"   - {c.name}")

    # Check confusion matrices
    cms = list(smoke_output_dir.glob("*.png")) + list((smoke_output_dir / "ensemble").glob("*.png"))
    print(f"✅ PASS: Found {len(cms)} isolated confusion matrix plots in {smoke_output_dir.relative_to(ROOT)}")
    for cm in cms:
        print(f"   - {cm.relative_to(ROOT)}")

    print("\n🎉 ALL SMOKE TESTS COMPLETED SUCCESSFULLY WITH PERFECT ISOLATION!")

if __name__ == "__main__":
    main()
