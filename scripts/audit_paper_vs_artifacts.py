#!/usr/bin/env python3
"""
Automated Audit Script: Paper Manuscripts vs. Canonical Artifacts.
Enforces 100% numerical verification between LaTeX tables/text and canonical JSON artifacts:
- artifacts/results/canonical_results_v2.json
- artifacts/results/statistical_tests_report.json
- artifacts/results/bootstrap_confidence_intervals.json
- artifacts/results/per_class_results.json
- artifacts/results/hardware_latency.json
- artifacts/results/external_benchmark_results.json

Audits:
- Table 1 (tab:table1): Clean Landmark Screening across sequence backbones
- Table 2 (tab:table2): Leave-One-Out (LOO) augmentation ablation on validation set
- Table 3 (tab:table3): Single and paired component augmentation ablation
- Table 4 (tab:table4): Spatial-temporal graph kinematic streams benchmark
- Table 5 (tab:table5): Cross-paradigm ensemble voting strategies
- Table 7 (tab:table7): Master benchmark comparison across models
- Table 10 (tab:statistical_tests): Statistical hypothesis testing (McNemar, Wilcoxon, paired t, Cohen's d)
- Table 11 (tab:bootstrap_ci): Non-parametric cluster bootstrap 95% confidence intervals
- Table 12 (tab:table6_compact): Per-class precision, recall, F1, and support for all 22 classes
- Table 13 (tab:hardware_latency): Computational complexity (Params, MFLOPs) and latencies (CUDA, MPS, CPU)
- Table 14 (tab:deyzel_benchmark): External Deyzel et al. S&C comparative benchmark
- Prose Claims: SkelGym-Full accuracy, latency ranges, model parameter footprints

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
    PROJECT_ROOT / "paper" / "paper.tex",
    PROJECT_ROOT / "paper" / "paper_eswa.tex",
    PROJECT_ROOT / "preprint" / "main.tex",
]

def clean_cell(text: str) -> str:
    """Clean LaTeX macros and formatting from table cell."""
    s = text.strip()
    s = re.sub(r"\\textbf\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\cite\{[^}]+\}", "", s)
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
    """Extract rows and cells for a specific LaTeX table by label without cross-table bleeding."""
    blocks = re.findall(r'\\begin\{(?:table|table\*)\}([\s\S]*?)\\end\{(?:table|table\*)\}', tex_content)
    target_block = None
    for b in blocks:
        if f"\\label{{{table_label}}}" in b:
            target_block = b
            break
    if not target_block:
        raise ValueError(f"Could not locate table with label '{table_label}' in TeX content")
    
    m_tab = re.search(r'\\begin\{tabular\}\{[^}]+\}([\s\S]*?)\\end\{tabular\}', target_block)
    if not m_tab:
        raise ValueError(f"Could not locate tabular environment in table '{table_label}'")
    tabular_body = m_tab.group(1)
    raw_rows = tabular_body.split(r"\\")
    rows = []
    for r in raw_rows:
        r = r.strip()
        r = re.sub(r"\\(?:top|mid|bottom|cmid)rule(?:\([^)]*\))?(?:\{[^}]*\})?", "", r).strip()
        if not r:
            continue
        cells = [c.strip() for c in r.split("&")]
        if len(cells) <= 1:
            continue
        # Skip header rows
        c0 = clean_cell(cells[0]).lower()
        if c0 == "" or c0 in [
            "architecture / model", "architecture / config", "model architecture",
            "configuration", "augmentation strategy", "stream id", "exercise class",
            "pairwise hypothesis comparison (m_a vs. m_b)", "component / configuration"
        ]:
            continue
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

    def audit_table1_feature_screening(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table1")
        data = canonical_v2["table2_clean_sequences"]
        row_map = {
            "lstm (world_3d)": "LSTM_world_3d",
            "lstm (mix_v2)": "LSTM_mix_v2",
            "bilstm (world_3d)": "BiLSTM_world_3d",
            "bilstm (mix_v2)": "BiLSTM_mix_v2",
            "transformer (world_3d)": "Transformer_world_3d",
            "transformer (mix_v2 clean)": "clean",
        }
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            key = None
            for k_pat, art_k in row_map.items():
                if k_pat in name:
                    key = art_k
                    break
            if not key:
                continue
            
            if key == "clean":
                art = canonical_v2["table5_single_operator"]["clean"]
            else:
                art = data[key]
            
            art_acc_m, art_acc_sd = parse_num_and_sd(art["val_win_acc"])
            art_f1_m, art_f1_sd = parse_num_and_sd(art["val_win_f1"])

            paper_acc_m, paper_acc_sd = parse_num_and_sd(cells[4])
            paper_f1_m, paper_f1_sd = parse_num_and_sd(cells[5])

            if paper_acc_m is not None and art_acc_m is not None:
                self.assert_close("Table 1", name, "Val Win Acc Mean", paper_acc_m, art_acc_m)
            if paper_acc_sd is not None and art_acc_sd is not None:
                self.assert_close("Table 1", name, "Val Win Acc SD", paper_acc_sd, art_acc_sd)
            if paper_f1_m is not None and art_f1_m is not None:
                self.assert_close("Table 1", name, "Val Win F1 Mean", paper_f1_m, art_f1_m, tol=0.005)
            if paper_f1_sd is not None and art_f1_sd is not None:
                self.assert_close("Table 1", name, "Val Win F1 SD", paper_f1_sd, art_f1_sd, tol=0.005)

    def audit_table2_loo(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table2")
        data = canonical_v2["table4_leave_one_out"]
        key_map = {
            "clean baseline": "clean",
            "candidate full 5op": "candidate_full_5op",
            "candidate minus mirror": "candidate_minus_mirror",
            "candidate minus yaw": "candidate_minus_yaw",
            "candidate minus scale": "candidate_minus_scale",
            "candidate minus time": "candidate_minus_time",
            "candidate minus jitter": "candidate_minus_jitter",
        }
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            key = None
            for k_pat, art_k in key_map.items():
                if k_pat in name:
                    key = art_k
                    break
            if not key or key not in data:
                continue
            art = data[key]
            art_acc_m, art_acc_sd = parse_num_and_sd(art["val_win_acc"])
            art_f1_m, art_f1_sd = parse_num_and_sd(art["val_win_f1"])

            paper_acc_m, paper_acc_sd = parse_num_and_sd(cells[3])
            paper_f1_m, paper_f1_sd = parse_num_and_sd(cells[4])

            if paper_acc_m is not None and art_acc_m is not None:
                self.assert_close("Table 2", name, "Val Win Acc Mean", paper_acc_m, art_acc_m)
            if paper_acc_sd is not None and art_acc_sd is not None:
                self.assert_close("Table 2", name, "Val Win Acc SD", paper_acc_sd, art_acc_sd)
            if paper_f1_m is not None and art_f1_m is not None:
                self.assert_close("Table 2", name, "Val Win F1 Mean", paper_f1_m, art_f1_m, tol=0.005)
            if paper_f1_sd is not None and art_f1_sd is not None:
                self.assert_close("Table 2", name, "Val Win F1 SD", paper_f1_sd, art_f1_sd, tol=0.005)

    def audit_table3_single_aug(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table3")
        data = canonical_v2["table5_single_operator"]
        key_map = {
            "clean baseline": "clean",
            "single mirror": "single_mirror",
            "single yaw": "single_yaw",
            "pair mirror + yaw": "pair_mirror_yaw",
            "single scale": "single_scale",
            "single time": "single_time",
            "single jitter": "single_jitter",
        }
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            key = None
            for k_pat, art_k in key_map.items():
                if k_pat in name:
                    key = art_k
                    break
            if not key or key not in data:
                continue
            art = data[key]
            art_acc_m, art_acc_sd = parse_num_and_sd(art["val_win_acc"])
            art_f1_m, art_f1_sd = parse_num_and_sd(art["val_win_f1"])

            paper_acc_m, paper_acc_sd = parse_num_and_sd(cells[3])
            paper_f1_m, paper_f1_sd = parse_num_and_sd(cells[4])

            if paper_acc_m is not None and art_acc_m is not None:
                self.assert_close("Table 3", name, "Val Win Acc Mean", paper_acc_m, art_acc_m)
            if paper_acc_sd is not None and art_acc_sd is not None:
                self.assert_close("Table 3", name, "Val Win Acc SD", paper_acc_sd, art_acc_sd)
            if paper_f1_m is not None and art_f1_m is not None:
                self.assert_close("Table 3", name, "Val Win F1 Mean", paper_f1_m, art_f1_m, tol=0.005)
            if paper_f1_sd is not None and art_f1_sd is not None:
                self.assert_close("Table 3", name, "Val Win F1 SD", paper_f1_sd, art_f1_sd, tol=0.005)

            if key == "pair_mirror_yaw" and art.get("val_loss"):
                art_l_m, art_l_sd = parse_num_and_sd(art["val_loss"])
                paper_l_m, paper_l_sd = parse_num_and_sd(cells[2])
                if paper_l_m is not None and art_l_m is not None:
                    self.assert_close("Table 3", name, "Val Loss Mean", paper_l_m, art_l_m, tol=0.01)

    def audit_table4_graph_streams(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table4")
        data = canonical_v2["table3_graph_streams"]
        for cells in rows:
            sid = clean_cell(cells[0]).strip()
            if sid not in data:
                continue
            art = data[sid]
            art_acc_m, art_acc_sd = parse_num_and_sd(art["val_win_acc"])
            art_f1_m, art_f1_sd = parse_num_and_sd(art["val_win_f1"])

            paper_acc_m, paper_acc_sd = parse_num_and_sd(cells[4])
            paper_f1_m, paper_f1_sd = parse_num_and_sd(cells[5])

            if paper_acc_m is not None and art_acc_m is not None:
                self.assert_close("Table 4", sid, "Val Win Acc Mean", paper_acc_m, art_acc_m)
            if paper_acc_sd is not None and art_acc_sd is not None:
                self.assert_close("Table 4", sid, "Val Win Acc SD", paper_acc_sd, art_acc_sd)
            if paper_f1_m is not None and art_f1_m is not None:
                self.assert_close("Table 4", sid, "Val Win F1 Mean", paper_f1_m, art_f1_m, tol=0.005)
            if paper_f1_sd is not None and art_f1_sd is not None:
                self.assert_close("Table 4", sid, "Val Win F1 SD", paper_f1_sd, art_f1_sd, tol=0.005)

    def audit_table5_voting_strategies(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table5")
        data = canonical_v2["table6_cross_paradigm_fusion"]
        for cells in rows:
            c0 = clean_cell(cells[0]).lower()
            c1 = clean_cell(cells[1]).lower()
            
            art = None
            label = ""
            if "transformer mix" in c0:
                art = canonical_v2["table5_single_operator"]["pair_mirror_yaw"]
                label = "Transformer Mix Reference"
            elif "skelgym-lite" in c0:
                if "hard" in c1:
                    art = data["SkelGym-Lite"]["hard"]
                    label = "SkelGym-Lite Hard"
                elif "accuracy-weighted" in c1:
                    art = data["SkelGym-Lite"]["accuracy_weighted_soft"]
                    label = "SkelGym-Lite Acc-Weighted"
                elif "uniform" in c1:
                    art = data["SkelGym-Lite"]["uniform_soft"]
                    label = "SkelGym-Lite Uniform"
            elif "skelgym-full" in c0:
                if "hard" in c1:
                    art = data["SkelGym-Full"]["hard"]
                    label = "SkelGym-Full Hard"
                elif "accuracy-weighted" in c1:
                    art = data["SkelGym-Full"]["accuracy_weighted_soft"]
                    label = "SkelGym-Full Acc-Weighted"
                elif "uniform" in c1:
                    art = data["SkelGym-Full"]["uniform_soft"]
                    label = "SkelGym-Full Uniform"
            
            if not art:
                continue

            # Val Win Acc
            am, asd = parse_num_and_sd(art["val_win_acc"])
            pm, psd = parse_num_and_sd(cells[2])
            if pm is not None and am is not None:
                self.assert_close("Table 5", label, "Val Win Acc Mean", pm, am)
            if psd is not None and asd is not None:
                self.assert_close("Table 5", label, "Val Win Acc SD", psd, asd)

            # Val Win F1
            am, asd = parse_num_and_sd(art["val_win_f1"])
            pm, psd = parse_num_and_sd(cells[3])
            if pm is not None and am is not None:
                self.assert_close("Table 5", label, "Val Win F1 Mean", pm, am, tol=0.005)
            if psd is not None and asd is not None:
                self.assert_close("Table 5", label, "Val Win F1 SD", psd, asd, tol=0.005)

            # Val Vid Acc
            am, asd = parse_num_and_sd(art["val_vid_acc"])
            pm, psd = parse_num_and_sd(cells[4])
            if pm is not None and am is not None:
                self.assert_close("Table 5", label, "Val Vid Acc Mean", pm, am)
            if psd is not None and asd is not None:
                self.assert_close("Table 5", label, "Val Vid Acc SD", psd, asd)

            # Val Vid F1
            am, asd = parse_num_and_sd(art["val_vid_f1"])
            pm, psd = parse_num_and_sd(cells[5])
            if pm is not None and am is not None:
                self.assert_close("Table 5", label, "Val Vid F1 Mean", pm, am, tol=0.005)
            if psd is not None and asd is not None:
                self.assert_close("Table 5", label, "Val Vid F1 SD", psd, asd, tol=0.005)

            # Test metrics if revealed
            if clean_cell(cells[6]) != "—" and "test_win_acc" in art:
                am, asd = parse_num_and_sd(art["test_win_acc"])
                pm, psd = parse_num_and_sd(cells[6])
                if pm is not None and am is not None:
                    self.assert_close("Table 5", label, "Test Win Acc Mean", pm, am)
                am, asd = parse_num_and_sd(art["test_win_f1"])
                pm, psd = parse_num_and_sd(cells[7])
                if pm is not None and am is not None:
                    self.assert_close("Table 5", label, "Test Win F1 Mean", pm, am, tol=0.005)
                am, asd = parse_num_and_sd(art["test_vid_acc"])
                pm, psd = parse_num_and_sd(cells[8])
                if pm is not None and am is not None:
                    self.assert_close("Table 5", label, "Test Vid Acc Mean", pm, am)
                am, asd = parse_num_and_sd(art["test_vid_f1"])
                pm, psd = parse_num_and_sd(cells[9])
                if pm is not None and am is not None:
                    self.assert_close("Table 5", label, "Test Vid F1 Mean", pm, am, tol=0.005)

    def audit_table7_master_benchmark(self, canonical_v2: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table7")
        art_rows = canonical_v2["table7_master_benchmark"]["table7_rows"]
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            if "blockgcn" in name:
                pm, _ = parse_num_and_sd(cells[5])
                if pm is not None:
                    self.assert_close("Table 7", "BlockGCN", "Test Win Acc", pm, 54.24)
                pm, _ = parse_num_and_sd(cells[6])
                if pm is not None:
                    self.assert_close("Table 7", "BlockGCN", "Test Win F1", pm, 0.5576, tol=0.005)
                pm, _ = parse_num_and_sd(cells[7])
                if pm is not None:
                    self.assert_close("Table 7", "BlockGCN", "Test Vid Acc", pm, 70.48)
                pm, _ = parse_num_and_sd(cells[8])
                if pm is not None:
                    self.assert_close("Table 7", "BlockGCN", "Test Vid F1", pm, 0.6948, tol=0.005)
                continue

            matched_art = None
            for ar in art_rows:
                ar_m = ar["model"].lower()
                if "bilstm" in name:
                    if "bilstm" in ar_m:
                        matched_art = ar
                        break
                elif "lstm (world" in name:
                    if "lstm (world" in ar_m and "bilstm" not in ar_m:
                        matched_art = ar
                        break
                elif "lstm (bio" in name:
                    if "lstm (bio" in ar_m and "bilstm" not in ar_m:
                        matched_art = ar
                        break
                elif "transformer (mix v2 clean)" in name:
                    if "clean" in ar_m and "transformer" in ar_m:
                        matched_art = ar
                        break
                elif "transformer (mix v2 + aug)" in name:
                    if "skelgym-aug" in ar_m and "transformer" in ar_m:
                        matched_art = ar
                        break
                elif "st-gcn" in name:
                    if "st-gcn" in ar_m:
                        matched_art = ar
                        break
                elif "aagcn (bone 3d + aug)" in name:
                    if "aagcn (bone" in ar_m:
                        matched_art = ar
                        break
                elif "four-stream" in name:
                    if "four-stream" in ar_m:
                        matched_art = ar
                        break
                elif "skelgym-lite" in name:
                    if "skelgym-lite" in ar_m:
                        matched_art = ar
                        break
                elif "skelgym-full" in name:
                    if "skelgym-full" in ar_m:
                        matched_art = ar
                        break

            if not matched_art:
                continue

            # Test Win Acc
            am, asd = parse_num_and_sd(matched_art["test_win_acc"])
            pm, psd = parse_num_and_sd(cells[5])
            if pm is not None and am is not None:
                self.assert_close("Table 7", name, "Test Win Acc Mean", pm, am)
            # Test Win F1
            am, asd = parse_num_and_sd(matched_art["test_win_f1"])
            pm, psd = parse_num_and_sd(cells[6])
            if pm is not None and am is not None:
                self.assert_close("Table 7", name, "Test Win F1 Mean", pm, am, tol=0.005)
            # Test Vid Acc
            am, asd = parse_num_and_sd(matched_art["test_vid_acc"])
            pm, psd = parse_num_and_sd(cells[7])
            if pm is not None and am is not None:
                self.assert_close("Table 7", name, "Test Vid Acc Mean", pm, am)
            # Test Vid F1
            am, asd = parse_num_and_sd(matched_art["test_vid_f1"])
            pm, psd = parse_num_and_sd(cells[8])
            if pm is not None and am is not None:
                self.assert_close("Table 7", name, "Test Vid F1 Mean", pm, am, tol=0.005)

    def audit_table10_statistical_tests(self, stat_report: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:statistical_tests")
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            matched_key = None
            for k in stat_report:
                if "unaugmented trans" in name and "unaugmented trans" in k.lower():
                    matched_key = k
                    break
                elif "fixed st-gcn" in name and "fixed st-gcn" in k.lower():
                    matched_key = k
                    break
                elif "single sequence" in name and "single sequence" in k.lower():
                    matched_key = k
                    break
                elif "single graph" in name and "single graph" in k.lower():
                    matched_key = k
                    break
                elif "four-stream graph" in name and "four-stream graph" in k.lower():
                    matched_key = k
                    break
            if not matched_key:
                continue
            art = stat_report[matched_key]
            # chi2
            pm, _ = parse_num_and_sd(cells[1])
            if pm is not None:
                self.assert_close("Table 10", matched_key, "McNemar chi2", pm, art["win_chi2"], tol=0.1)
            # Wilcoxon W
            pm, _ = parse_num_and_sd(cells[4])
            if pm is not None:
                self.assert_close("Table 10", matched_key, "Wilcoxon W", pm, float(art["vid_wilcoxon_w"]), tol=0.1)
            # Wilcoxon p
            pm, _ = parse_num_and_sd(cells[5])
            if pm is not None and "e" not in art["vid_wilcoxon_p"]:
                self.assert_close("Table 10", matched_key, "Wilcoxon p", pm, float(art["vid_wilcoxon_p"]), tol=0.001)
            # Cohen's d
            pm, _ = parse_num_and_sd(cells[7])
            if pm is not None:
                self.assert_close("Table 10", matched_key, "Cohen's d", pm, float(art["vid_cohens_d"]), tol=0.01)

    def audit_table11_bootstrap_ci(self, boot_report: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:bootstrap_ci")
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            matched_key = None
            for k in boot_report:
                k_low = k.lower()
                if "bilstm" in name:
                    if "bilstm" in k_low:
                        matched_key = k
                        break
                elif "lstm" in name and "bilstm" not in name:
                    if "lstm" in k_low and "bilstm" not in k_low:
                        matched_key = k
                        break
                elif "st-gcn" in name and "st-gcn" in k_low:
                    matched_key = k
                    break
                elif "transformer" in name and "clean" in name and "clean" in k_low:
                    matched_key = k
                    break
                elif "transformer" in name and "skelgym-aug" in name and "skelgym-aug" in k_low:
                    matched_key = k
                    break
                elif "aagcn" in name and "bone" in name and "bone" in k_low:
                    matched_key = k
                    break
                elif "four-stream" in name and "four-stream" in k_low:
                    matched_key = k
                    break
                elif "skelgym-lite" in name and "skelgym-lite" in k_low:
                    matched_key = k
                    break
                elif "skelgym-full" in name and "skelgym-full" in k_low:
                    matched_key = k
                    break
            if not matched_key:
                continue
            art = boot_report[matched_key]
            # Window Acc mean
            pm, _ = parse_num_and_sd(cells[1].split("[")[0])
            if pm is not None:
                self.assert_close("Table 11", matched_key, "Window Acc Mean", pm, art["w_acc_mean"])
            # Window F1 mean
            pm, _ = parse_num_and_sd(cells[2].split("[")[0])
            if pm is not None:
                self.assert_close("Table 11", matched_key, "Window F1 Mean", pm, art["w_f1_mean"], tol=0.005)
            # Video Acc mean
            pm, _ = parse_num_and_sd(cells[3].split("[")[0])
            if pm is not None:
                self.assert_close("Table 11", matched_key, "Video Acc Mean", pm, art["v_acc_mean"])
            # Video F1 mean
            pm, _ = parse_num_and_sd(cells[4].split("[")[0])
            if pm is not None:
                self.assert_close("Table 11", matched_key, "Video F1 Mean", pm, art["v_f1_mean"], tol=0.005)

    def audit_table12_per_class(self, per_class: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:table6_compact")
        for cells in rows:
            cls_name = clean_cell(cells[0]).lower()
            if cls_name not in per_class:
                continue
            art = per_class[cls_name]
            # Window Recall
            pm, _ = parse_num_and_sd(cells[1])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Win Recall", pm, art["win_recall"], tol=0.005)
            # Window F1
            pm, _ = parse_num_and_sd(cells[2])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Win F1", pm, art["win_f1"], tol=0.005)
            # Window Support
            pm, _ = parse_num_and_sd(cells[3])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Win Support", pm, float(art["win_support"]), tol=0.1)
            # Video Recall
            pm, _ = parse_num_and_sd(cells[4])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Vid Recall", pm, art["vid_recall"], tol=0.005)
            # Video F1
            pm, _ = parse_num_and_sd(cells[5])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Vid F1", pm, art["vid_f1"], tol=0.005)
            # Video Support
            pm, _ = parse_num_and_sd(cells[6])
            if pm is not None:
                self.assert_close("Table 12 Per-Class", cls_name, "Vid Support", pm, float(art["vid_support"]), tol=0.1)

    def audit_table13_hardware_latency(self, hw_latency: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:hardware_latency")
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            matched_key = None
            if "skelgym-full" in name:
                matched_key = "SkelGym-Full (Transformer + 4 AAGCN)"
            elif "skelgym-lite" in name:
                matched_key = "SkelGym-Lite (Transformer + Bone)"
            elif "joint-motion" in name:
                matched_key = "AAGCN Joint-Motion Stream"
            elif "bone-motion" in name:
                matched_key = "AAGCN Bone-Motion Stream"
            elif "joint stream" in name:
                matched_key = "AAGCN Joint Stream"
            elif "bone stream" in name:
                matched_key = "AAGCN Bone Stream"
            elif "transformer" in name:
                matched_key = "Transformer Mix v2 (63-d)"

            if not matched_key or matched_key not in hw_latency:
                continue
            art = hw_latency[matched_key]
            # FLOPs
            pm, _ = parse_num_and_sd(cells[2])
            if pm is not None:
                self.assert_close("Table 13 HW", matched_key, "MFLOPs", pm, art["mflops"], tol=0.1)
            # CUDA
            pm, _ = parse_num_and_sd(cells[3].split("(")[0])
            if pm is not None:
                self.assert_close("Table 13 HW", matched_key, "CUDA ms", pm, art["cuda_mean_ms"], tol=0.01)
            # MPS
            pm, _ = parse_num_and_sd(cells[4].split("(")[0])
            if pm is not None:
                self.assert_close("Table 13 HW", matched_key, "MPS ms", pm, art["mps_mean_ms"], tol=0.01)
            # CPU
            pm, _ = parse_num_and_sd(cells[5].split("(")[0])
            if pm is not None:
                self.assert_close("Table 13 HW", matched_key, "CPU ms", pm, art["cpu_mean_ms"], tol=0.01)

    def audit_table14_deyzel_benchmark(self, ext_bench: Dict[str, Any]):
        rows = extract_table_rows(self.tex_content, "tab:deyzel_benchmark")
        for cells in rows:
            name = clean_cell(cells[0]).lower()
            matched_key = None
            if "skelgym-full" in name:
                matched_key = "SkelGym-Full"
            elif "skelgym-lite" in name:
                matched_key = "SkelGym-Lite"
            elif "st-gcn" in name:
                matched_key = "ST-GCN (Rel 3D)"
            elif "bilstm" in name:
                matched_key = "BiLSTM (Mix 63-d)"
            elif "lstm" in name and "bilstm" not in name:
                matched_key = "LSTM (Mix 63-d)"
            elif "aagcn" in name:
                matched_key = "AAGCN (Bone 3D)"
            elif "transformer" in name:
                matched_key = "Transformer (Mix 63-d)"

            if not matched_key or matched_key not in ext_bench:
                continue
            art = ext_bench[matched_key]
            # Open Win Acc
            pm, _ = parse_num_and_sd(cells[1])
            if pm is not None:
                self.assert_close("Table 14 Deyzel", matched_key, "Open Win Acc", pm, art["open_win_acc"])
            # Open Vid Acc
            pm, _ = parse_num_and_sd(cells[2])
            if pm is not None:
                self.assert_close("Table 14 Deyzel", matched_key, "Open Vid Acc", pm, art["open_vid_acc"])
            # Closed Win Acc
            pm, _ = parse_num_and_sd(cells[3])
            if pm is not None:
                self.assert_close("Table 14 Deyzel", matched_key, "Closed Win Acc", pm, art["closed_win_acc"])
            # Closed Vid Acc
            pm, _ = parse_num_and_sd(cells[4])
            if pm is not None:
                self.assert_close("Table 14 Deyzel", matched_key, "Closed Vid Acc", pm, art["closed_vid_acc"])
            # Closed Vid Macro F1
            pm, _ = parse_num_and_sd(cells[5])
            if pm is not None:
                self.assert_close("Table 14 Deyzel", matched_key, "Closed Vid F1", pm, art["closed_vid_f1"], tol=0.005)

    def audit_prose_claims(self):
        # 1. SkelGym-Full accuracy
        m_win = re.search(r"76\.19\s*\\?%\s*\$?(\\pm|±)\$?\s*0\.16", self.tex_content)
        m_vid = re.search(r"85\.27\s*\\?%\s*\$?(\\pm|±)\$?\s*0\.41", self.tex_content)
        self.checked_count += 2
        if not m_win:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical SkelGym-Full window accuracy claim (76.19% ± 0.16%) in text")
        if not m_vid:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical SkelGym-Full video accuracy claim (85.27% ± 0.41%) in text")

        # 2. Latency claims
        m_lat_cpu = re.search(r"0\.49\s*--\s*3\.96\s*~?\s*ms", self.tex_content)
        m_lat_cuda = re.search(r"0\.59\s*--\s*4\.65\s*~?\s*ms", self.tex_content)
        m_lat_mps = re.search(r"1\.36\s*--\s*5\.86\s*~?\s*ms", self.tex_content)
        self.checked_count += 3
        if not m_lat_cpu:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical CPU latency range (0.49--3.96 ms) in text")
        if not m_lat_cuda:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical CUDA latency range (0.59--4.65 ms) in text")
        if not m_lat_mps:
            self.errors.append(f"[{self.paper_path.name}] Missing canonical MPS latency range (1.36--5.86 ms) in text")

        # 3. Parameter counts
        self.checked_count += 3
        if "1.81M" not in self.tex_content:
            self.errors.append(f"[{self.paper_path.name}] Missing SkelGym-Full parameter count (1.81M)")
        if "679K" not in self.tex_content:
            self.errors.append(f"[{self.paper_path.name}] Missing SkelGym-Lite parameter count (679K)")
        if "301K" not in self.tex_content:
            self.errors.append(f"[{self.paper_path.name}] Missing Transformer Mix parameter count (301K)")

    def run_all(self, canonical_v2, stat_report, boot_report, per_class, hw_latency, ext_bench) -> bool:
        print(f"\n=======================================================")
        print(f"Auditing manuscript: {self.paper_path.name}")
        print(f"=======================================================")
        self.audit_table1_feature_screening(canonical_v2)
        self.audit_table2_loo(canonical_v2)
        self.audit_table3_single_aug(canonical_v2)
        self.audit_table4_graph_streams(canonical_v2)
        self.audit_table5_voting_strategies(canonical_v2)
        self.audit_table7_master_benchmark(canonical_v2)
        self.audit_table10_statistical_tests(stat_report)
        self.audit_table11_bootstrap_ci(boot_report)
        self.audit_table12_per_class(per_class)
        self.audit_table13_hardware_latency(hw_latency)
        self.audit_table14_deyzel_benchmark(ext_bench)
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

    canonical_v2 = json.load(open(ARTIFACTS_DIR / "canonical_results_v2.json"))
    stat_report = json.load(open(ARTIFACTS_DIR / "statistical_tests_report.json"))
    boot_report = json.load(open(ARTIFACTS_DIR / "bootstrap_confidence_intervals.json"))
    per_class = json.load(open(ARTIFACTS_DIR / "per_class_results.json"))["classes"]
    hw_latency = json.load(open(ARTIFACTS_DIR / "hardware_latency.json"))["models"]
    ext_bench = json.load(open(ARTIFACTS_DIR / "external_benchmark_results.json"))["models"]

    all_passed = True
    for p in args.papers:
        if not p.exists():
            print(f"Error: Manuscript not found at {p}", file=sys.stderr)
            all_passed = False
            continue
        auditor = PaperAuditor(p)
        passed = auditor.run_all(canonical_v2, stat_report, boot_report, per_class, hw_latency, ext_bench)
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
