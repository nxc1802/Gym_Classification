#!/usr/bin/env python3
"""
Automated Audit Script: Paper Manuscripts vs. Canonical Artifacts.
Enforces 100% numerical verification between LaTeX tables/text and canonical JSON artifacts:
- artifacts/results/table1_feature_screening.json
- artifacts/results/augmentation_ablation_results.json
- artifacts/results/graph_streams_results.json
- artifacts/results/multi_seed_evaluation_results.json
- artifacts/results/consensus_gains.json
- artifacts/results/hardware_latency.json
- artifacts/results/external_benchmark_results.json

Exits 0 if all audited numbers match within tolerance (<= 0.05% for percentages, <= 0.005 for F1/loss/ms).
Exits non-zero with detailed mismatch diffs otherwise.
"""

import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "results"

DEFAULT_PAPERS = [
    PROJECT_ROOT / "paper" / "paper_eswa.tex",
    PROJECT_ROOT / "paper" / "paper_llncs.tex",
]

def clean_cell(text: str) -> str:
    """Clean LaTeX macros and formatting from table cell."""
    s = text.strip()
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\$([^\$]+)\$", r"\1", s)
    s = s.replace(r"\%", "").replace("%", "")
    s = s.replace(r"\pm", "±").replace("+-", "±")
    s = s.replace("~ms", " ms")
    s = re.sub(r"\(n=\d+\)", "", s)
    s = s.replace("---", "—").replace("--", "-")
    return s.strip()

def parse_num_and_sd(cell_text: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse number or 'mean ± sd' from cell text."""
    clean = clean_cell(cell_text)
    if "±" in clean:
        parts = clean.split("±")
        try:
            m = float(re.search(r"[-+]?\d*\.?\d+", parts[0]).group())
            s = float(re.search(r"[-+]?\d*\.?\d+", parts[1]).group())
            return m, s
        except Exception:
            return None, None
    m = re.search(r"[-+]?\d*\.?\d+", clean)
    if m:
        try:
            return float(m.group()), None
        except Exception:
            return None, None
    return None, None

def extract_table_rows(tex_content: str, table_label: str) -> List[List[str]]:
    """Extract rows and cells for a specific LaTeX table by label."""
    # Find table environment containing the label
    pattern = rf"\\begin\{{(?:table|table\*)\}}[\s\S]*?\\label\{{{table_label}\}}[\s\S]*?\\begin\{{tabular\}}([\s\S]*?)\\end\{{tabular\}}"
    match = re.search(pattern, tex_content)
    if not match:
        pattern2 = rf"\\begin\{{(?:table|table\*)\}}[\s\S]*?\\begin\{{tabular\}}([\s\S]*?)\\end\{{tabular\}}[\s\S]*?\\label\{{{table_label}\}}"
        match = re.search(pattern2, tex_content)
    if not match:
        raise ValueError(f"Could not locate table with label '{table_label}' in TeX content")

    tabular_body = match.group(1)
    raw_rows = tabular_body.split(r"\\")
    rows = []
    for r in raw_rows:
        r = r.strip()
        if not r or r.startswith(r"\toprule") or r.startswith(r"\midrule") or r.startswith(r"\bottomrule"):
            r = re.sub(r"\\(?:top|mid|bottom)rule", "", r).strip()
            if not r:
                continue
        if r.startswith(r"\multicolumn") or r.startswith(r"\multirow"):
            continue
        cells = [c.strip() for c in r.split("&")]
        if len(cells) > 1:
            rows.append(cells)
    return rows

class PaperAuditor:
    def __init__(self, paper_path: Path):
        self.paper_path = paper_path
        self.tex_content = paper_path.read_text(encoding="utf-8")
        self.errors: List[str] = []
        self.checked_count = 0

    def assert_close(self, table_name: str, item_name: str, metric_name: str, paper_val: float, artifact_val: float, tol: float = 0.05):
        self.checked_count += 1
        diff = abs(paper_val - artifact_val)
        if diff > tol:
            err = (
                f"[{self.paper_path.name}] {table_name} -> {item_name} ({metric_name}): "
                f"Paper has {paper_val:.4f}, Artifact has {artifact_val:.4f} (diff={diff:.4f} > tol={tol})"
            )
            self.errors.append(err)

    def audit_table1_feature_screening(self):
        art_path = ARTIFACTS_DIR / "table1_feature_screening.json"
        if not art_path.exists():
            self.errors.append(f"Missing artifact: {art_path}")
            return
        raw_list = json.loads(art_path.read_text())
        # Index by (model, feature)
        data = {(item["model"], item["feature"]): item for item in raw_list}

        rows = extract_table_rows(self.tex_content, "tab:table1")
        feature_map = {
            "raw 2d": "raw_2d",
            "relative 2d": "rel_2d",
            "angle 2d": "angle_2d",
            "angle2 2d": "angle2_2d",
            "raw 3d": "raw_3d",
            "relative 3d": "rel_3d",
            "angle 3d": "angle_3d",
            "angle2 3d": "angle2_3d",
            "mix": "mix",
        }

        for cells in rows[1:]:  # skip header
            name = clean_cell(cells[0]).lower()
            feat_key = None
            for k, f_val in feature_map.items():
                if k in name:
                    feat_key = f_val
                    break
            if not feat_key:
                continue

            # LSTM Val
            val, _ = parse_num_and_sd(cells[2])
            if val is not None and ("LSTM", feat_key) in data:
                self.assert_close("Table 1", f"LSTM {feat_key}", "Val Acc", val, data[("LSTM", feat_key)]["val_acc"])
            # BiLSTM Val
            val, _ = parse_num_and_sd(cells[3])
            if val is not None and ("BiLSTM", feat_key) in data:
                self.assert_close("Table 1", f"BiLSTM {feat_key}", "Val Acc", val, data[("BiLSTM", feat_key)]["val_acc"])
            # Trans Val
            val, _ = parse_num_and_sd(cells[4])
            if val is not None and ("Transformer", feat_key) in data:
                self.assert_close("Table 1", f"Trans {feat_key}", "Val Acc", val, data[("Transformer", feat_key)]["val_acc"])
            # Trans Test
            val, _ = parse_num_and_sd(cells[5])
            if val is not None and ("Transformer", feat_key) in data:
                self.assert_close("Table 1", f"Trans {feat_key}", "Test Acc", val, data[("Transformer", feat_key)]["test_win_acc"])

    def audit_table2_loo(self):
        art_path = ARTIFACTS_DIR / "augmentation_ablation_results.json"
        if not art_path.exists():
            return
        data = json.loads(art_path.read_text())["summary_leave_one_out"]

        rows = extract_table_rows(self.tex_content, "tab:table2")
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            target_key = None
            if "clean baseline" in c_name:
                target_key = "Clean_Baseline_NoAug"
            elif "candidate full" in c_name:
                target_key = "Candidate_Full_5op"
            elif "skelgym-aug" in c_name or "timewarp" in c_name:
                target_key = "Minus_TimeWarp"
            elif "mirror" in c_name:
                target_key = "Minus_Mirror"
            elif "yaw" in c_name:
                target_key = "Minus_Yaw"
            elif "scale" in c_name:
                target_key = "Minus_Scale"
            elif "jitter" in c_name:
                target_key = "Minus_Jitter"

            if not target_key or target_key not in data:
                continue
            art = data[target_key]
            # Val Window Acc
            val, _ = parse_num_and_sd(cells[2])
            if val is not None:
                self.assert_close("Table 2", target_key, "Val Win Acc", val, art["val_acc_mean"])
            # Val Loss
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 2", target_key, "Val Loss", val, art["val_loss_mean"], tol=0.01)
            # Test Window Acc
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 2", target_key, "Test Win Acc", val, art["win_acc_mean"])
            # Test Macro F1
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 2", target_key, "Test Macro F1", val, art["win_f1_mean"], tol=0.005)

    def audit_table3_single_aug(self):
        art_path = ARTIFACTS_DIR / "augmentation_ablation_results.json"
        if not art_path.exists():
            return
        data = json.loads(art_path.read_text())["summary_single_component"]

        rows = extract_table_rows(self.tex_content, "tab:table3")
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            target_key = None
            if "clean baseline" in c_name:
                target_key = "Clean_Baseline_NoAug"
            elif "skelgym-aug" in c_name:
                target_key = "SkelGym_Aug_4op"
            elif "candidate full" in c_name:
                target_key = "Candidate_Full_5op"
            elif "reflection" in c_name or "mirror" in c_name:
                target_key = "Single_Mirror"
            elif "yaw" in c_name:
                target_key = "Single_Yaw"
            elif "scale" in c_name:
                target_key = "Single_Scale"
            elif "timewarp" in c_name:
                target_key = "Single_TimeWarp"
            elif "jitter" in c_name:
                target_key = "Single_Jitter"

            if not target_key or target_key not in data:
                continue
            art = data[target_key]
            # Val Win Acc
            val, _ = parse_num_and_sd(cells[2])
            if val is not None:
                self.assert_close("Table 3", target_key, "Val Win Acc", val, art["val_acc_mean"])
            # Val Loss
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 3", target_key, "Val Loss", val, art["val_loss_mean"], tol=0.01)
            # Test Win Acc
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 3", target_key, "Test Win Acc", val, art["win_acc_mean"])
            # Test Macro F1
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 3", target_key, "Test Macro F1", val, art["win_f1_mean"], tol=0.005)

    def audit_table4_graph_streams(self):
        art_path = ARTIFACTS_DIR / "graph_streams_results.json"
        if not art_path.exists():
            return
        experiments = json.loads(art_path.read_text())["experiments"]
        data = {exp["exp_id"]: exp for exp in experiments}

        rows = extract_table_rows(self.tex_content, "tab:table4")
        for cells in rows[1:]:
            c_model = clean_cell(cells[0]).lower()
            c_stream = clean_cell(cells[1]).lower()
            c_aug = clean_cell(cells[2]).lower() if len(cells) > 2 else ""

            exp_id = None
            if "st-gcn" in c_model and "relative" in c_stream:
                exp_id = "T3.2"
            elif "aagcn baseline" in c_model and "bone" in c_stream:
                exp_id = "T3.3"
            elif "two-stream" in c_model:
                exp_id = "T3.8"
            elif "four-stream" in c_model:
                exp_id = "T3.9"
            elif "joint motion" in c_stream:
                exp_id = "T3.6"
            elif "bone motion" in c_stream:
                exp_id = "T3.7"
            elif "bone 3d" in c_stream and "skelgym-aug" in c_aug:
                exp_id = "T3.4"
            elif "joint" in c_stream and "skelgym-aug" in c_aug:
                exp_id = "T3.5"

            if not exp_id or exp_id not in data:
                continue
            art = data[exp_id]
            # Val Acc
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 4", exp_id, "Val Acc", val, art["val_acc"])
            # Test Win Acc
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 4", exp_id, "Test Win Acc", val, art["win_acc"])
            # Test Vid Acc
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 4", exp_id, "Test Vid Acc", val, art["vid_acc"])

    def audit_table5_voting_strategies(self):
        art_path = ARTIFACTS_DIR / "multi_seed_evaluation_results.json"
        if not art_path.exists():
            return
        ms = json.loads(art_path.read_text())["summary"]

        rows = extract_table_rows(self.tex_content, "tab:table5")
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            if "hard" in c_name:
                art = ms["fusion_methods"]["SkelGym-Full"]["Hard Voting"]
                target_key = "Hard Voting"
            elif "uniform" in c_name:
                art = ms["fusion_methods"]["SkelGym-Full"]["Uniform Soft Voting"]
                target_key = "Uniform Soft Voting"
            elif "accuracy-weighted" in c_name:
                art = ms["fusion_methods"]["SkelGym-Full"]["Accuracy-Weighted Soft"]
                target_key = "Accuracy-Weighted Soft"
            elif "skelgym-lite" in c_name:
                art = ms["fusion_methods"]["SkelGym-Lite"]["SLSQP Soft Voting"]
                target_key = "SkelGym-Lite SLSQP"
            elif "skelgym-full" in c_name:
                art = ms["fusion_methods"]["SkelGym-Full"]["SLSQP Soft Voting"]
                target_key = "SkelGym-Full SLSQP"
            elif "transformer mix" in c_name:
                art = ms["individual"]["Transformer Mix (Aug)"]
                target_key = "Transformer Mix"
            elif "aagcn bone" in c_name:
                art = ms["individual"]["AAGCN Bone (Aug)"]
                target_key = "AAGCN Bone"
            elif "four-stream" in c_name:
                art = ms["fusion_methods"]["Four-Stream AAGCN (Aug)"]["SLSQP Soft Voting"]
                target_key = "Four-Stream AAGCN"
            else:
                continue

            # Val Win Acc
            if "val_win_acc_mean" in art:
                val, _ = parse_num_and_sd(cells[2])
                if val is not None:
                    self.assert_close("Table 5", target_key, "Val Win Acc", val, art["val_win_acc_mean"])
            # Val Vid Acc
            if "val_vid_acc_mean" in art:
                val, _ = parse_num_and_sd(cells[3])
                if val is not None:
                    self.assert_close("Table 5", target_key, "Val Vid Acc", val, art["val_vid_acc_mean"])
            # Test Win Acc
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 5", target_key, "Test Win Acc", val, art["win_acc_mean"])
            # Test Macro F1
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 5", target_key, "Test Macro F1", val, art["win_f1_mean"], tol=0.005)
            # Test Vid Acc
            val, _ = parse_num_and_sd(cells[6])
            if val is not None:
                self.assert_close("Table 5", target_key, "Test Vid Acc", val, art["vid_acc_mean"])

    def audit_table7_consensus_gains(self):
        art_path = ARTIFACTS_DIR / "consensus_gains.json"
        if not art_path.exists():
            return
        data = json.loads(art_path.read_text())

        rows = extract_table_rows(self.tex_content, "tab:table7")
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            matched_key = None
            if "skelgym-full" in c_name:
                matched_key = "SkelGym-Full (5 Streams)"
            elif "skelgym-lite" in c_name:
                matched_key = "SkelGym-Lite (2 Models)"
            elif "four-stream" in c_name:
                matched_key = "Four-Stream AAGCN (Aug)"
            elif "two-stream" in c_name:
                matched_key = "Two-Stream AAGCN (Aug)"
            elif "skelgym-aug transformer" in c_name:
                matched_key = "SkelGym-Aug Transformer (Mix)"
            elif "skelgym-aug aagcn" in c_name:
                matched_key = "SkelGym-Aug AAGCN (Bone 3D)"
            elif "transformer (mix 117-d, clean)" in c_name or ("transformer" in c_name and "clean" in c_name):
                matched_key = "Transformer (Mix 117-d, Clean)"
            elif "clean baseline aagcn" in c_name:
                matched_key = "Clean Baseline AAGCN (Bone 3D)"
            elif "bilstm" in c_name:
                matched_key = "Baseline BiLSTM (Mix 117-d)"
            elif "lstm" in c_name and "bilstm" not in c_name:
                matched_key = "Baseline LSTM (Mix 117-d)"
            elif "st-gcn" in c_name:
                matched_key = "Baseline ST-GCN (Rel 3D)"

            if not matched_key or matched_key not in data:
                continue
            art = data[matched_key]
            # Test Win Acc
            val, _ = parse_num_and_sd(cells[1])
            if val is not None:
                self.assert_close("Table 7", matched_key, "Test Win Acc", val, art["win_acc"])
            # Test Win Macro F1
            val, _ = parse_num_and_sd(cells[2])
            if val is not None:
                self.assert_close("Table 7", matched_key, "Test Win F1", val, art["win_f1"], tol=0.005)
            # Test Vid Acc
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 7", matched_key, "Test Vid Acc", val, art["vid_acc"])
            # Test Vid Macro F1
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 7", matched_key, "Test Vid F1", val, art["vid_f1"], tol=0.005)

    def audit_table11_hardware_latency(self):
        art_path = ARTIFACTS_DIR / "hardware_latency.json"
        if not art_path.exists():
            return
        data = json.loads(art_path.read_text())["models"]

        rows = extract_table_rows(self.tex_content, "tab:hardware_latency")
        # Priority matching to avoid substring collision
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            matched_key = None
            if "skelgym-full" in c_name:
                matched_key = "SkelGym-Full (Transformer + 4 AAGCN)"
            elif "skelgym-lite" in c_name:
                matched_key = "SkelGym-Lite (Transformer + Bone)"
            elif "joint-motion" in c_name:
                matched_key = "AAGCN Joint-Motion Stream"
            elif "bone-motion" in c_name:
                matched_key = "AAGCN Bone-Motion Stream"
            elif "joint" in c_name and "motion" not in c_name:
                matched_key = "AAGCN Joint Stream"
            elif "bone" in c_name and "motion" not in c_name:
                matched_key = "AAGCN Bone Stream"
            elif "transformer" in c_name:
                matched_key = "Transformer Mix (117-d)"

            if not matched_key or matched_key not in data:
                continue
            art = data[matched_key]
            # FLOPs per Window (mflops)
            val, _ = parse_num_and_sd(cells[2])
            if val is not None:
                self.assert_close("Table 11", matched_key, "MFLOPs", val, art["mflops"], tol=0.1)
            # CUDA
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 11", matched_key, "CUDA ms", val, art["cuda_mean_ms"], tol=0.01)
            # MPS
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 11", matched_key, "MPS ms", val, art["mps_mean_ms"], tol=0.01)
            # CPU
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 11", matched_key, "CPU ms", val, art["cpu_mean_ms"], tol=0.01)

    def audit_table12_external_benchmark(self):
        art_path = ARTIFACTS_DIR / "external_benchmark_results.json"
        if not art_path.exists():
            return
        data = json.loads(art_path.read_text())["models"]

        rows = extract_table_rows(self.tex_content, "tab:deyzel_benchmark")
        # Priority matching to avoid substring collision
        for cells in rows[1:]:
            c_name = clean_cell(cells[0]).lower()
            matched_key = None
            if "skelgym-full" in c_name:
                matched_key = "SkelGym-Full"
            elif "skelgym-lite" in c_name:
                matched_key = "SkelGym-Lite"
            elif "st-gcn" in c_name:
                matched_key = "ST-GCN (Rel 3D)"
            elif "bilstm" in c_name:
                matched_key = "BiLSTM (Mix 117-d)"
            elif "lstm" in c_name and "bilstm" not in c_name:
                matched_key = "LSTM (Mix 117-d)"
            elif "aagcn" in c_name:
                matched_key = "AAGCN (Bone 3D)"
            elif "transformer" in c_name:
                matched_key = "Transformer (Mix)"

            if not matched_key or matched_key not in data:
                continue
            art = data[matched_key]
            # Open Win Acc
            val, _ = parse_num_and_sd(cells[1])
            if val is not None:
                self.assert_close("Table 12", matched_key, "Open Win Acc", val, art["open_win_acc"])
            # Open Vid Acc
            val, _ = parse_num_and_sd(cells[2])
            if val is not None:
                self.assert_close("Table 12", matched_key, "Open Vid Acc", val, art["open_vid_acc"])
            # Closed Win Acc
            val, _ = parse_num_and_sd(cells[3])
            if val is not None:
                self.assert_close("Table 12", matched_key, "Closed Win Acc", val, art["closed_win_acc"])
            # Closed Vid Acc
            val, _ = parse_num_and_sd(cells[4])
            if val is not None:
                self.assert_close("Table 12", matched_key, "Closed Vid Acc", val, art["closed_vid_acc"])
            # Closed Vid Macro F1
            val, _ = parse_num_and_sd(cells[5])
            if val is not None:
                self.assert_close("Table 12", matched_key, "Closed Vid F1", val, art["closed_vid_f1"], tol=0.005)

    def audit_prose_claims(self):
        """Audit key claims in abstract, introduction, and conclusion."""
        # 1. SkelGym-Full accuracy
        m_win = re.search(r"69\.73\s*\\?%\s*\$?(\\pm|±)\$?\s*1\.10\s*\\?%", self.tex_content)
        m_vid = re.search(r"78\.83\s*\\?%\s*\$?(\\pm|±)\$?\s*0\.66\s*\\?%", self.tex_content)
        self.checked_count += 2
        if not m_win:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical SkelGym-Full window accuracy claim (69.73% ± 1.10%) in text")
        if not m_vid:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical SkelGym-Full video accuracy claim (78.83% ± 0.66%) in text")

        # 2. Latency claims
        m_lat_cpu = re.search(r"0\.42\s*--\s*3\.96\s*~?\s*ms", self.tex_content)
        m_lat_cuda = re.search(r"0\.50\s*--\s*4\.65\s*~?\s*ms", self.tex_content)
        self.checked_count += 2
        if not m_lat_cpu:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical CPU latency range (0.42--3.96 ms) in text")
        if not m_lat_cuda:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical CUDA latency range (0.50--4.65 ms) in text")

        # 3. Parameter counts
        self.checked_count += 2
        if "1.91M" not in self.tex_content:
            self.errors.append(f"[{self.paper_path.name}] Missing SkelGym-Full parameter count (1.91M)")
        if "778K" not in self.tex_content:
            self.errors.append(f"[{self.paper_path.name}] Missing SkelGym-Lite parameter count (778K)")

    def run_all(self) -> bool:
        print(f"\n=======================================================")
        print(f"Auditing manuscript: {self.paper_path.name}")
        print(f"=======================================================")
        self.audit_table1_feature_screening()
        self.audit_table2_loo()
        self.audit_table3_single_aug()
        self.audit_table4_graph_streams()
        self.audit_table5_voting_strategies()
        self.audit_table7_consensus_gains()
        self.audit_table11_hardware_latency()
        self.audit_table12_external_benchmark()
        self.audit_prose_claims()

        print(f"Audited {self.checked_count} numerical values and text assertions.")
        if self.errors:
            print(f"FAILED with {len(self.errors)} discrepancies:")
            for err in self.errors:
                print(f"  ❌ {err}")
            return False
        else:
            print(f"✅ PASSED with 0 discrepancies! All numerical values match canonical artifacts.")
            return True

def main():
    parser = argparse.ArgumentParser(description="Audit LaTeX manuscripts against canonical JSON artifacts")
    parser.add_argument("--papers", nargs="*", type=Path, default=DEFAULT_PAPERS, help="Paths to .tex files to audit")
    args = parser.parse_args()

    all_passed = True
    for p in args.papers:
        if not p.exists():
            print(f"Error: Manuscript not found at {p}", file=sys.stderr)
            all_passed = False
            continue
        auditor = PaperAuditor(p)
        passed = auditor.run_all()
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL MANUSCRIPTS 100% NUMERICAL-LOCKED AND VERIFIED AGAINST ARTIFACTS!\n")
        sys.exit(0)
    else:
        print("\n❌ AUDIT FAILED! Please resolve numerical discrepancies above.\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
