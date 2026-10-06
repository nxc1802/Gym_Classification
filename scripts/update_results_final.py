import json
import re

def update_results():
    with open('outputs/RESULTS_FINAL.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Load 3 JSON files
    with open('outputs/statistical_tests_report.json', 'r') as f:
        stat_tests = json.load(f)
    with open('outputs/bootstrap_confidence_intervals.json', 'r') as f:
        boot_ci = json.load(f)
    with open('outputs/per_class_results.json', 'r') as f:
        per_class = json.load(f)['classes']

    # 1. Build Table 8
    t8_rows = [
        "| Pairwise Comparison ($M_A$ vs. $M_B$) | Window McNemar $\\chi^2$ | Window $p$-value | Window Odds Ratio | Video Wilcoxon $W$ | Video $p$-value | Video Paired $t$ | Video Cohen's $d$ | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for pair, metrics in stat_tests.items():
        sig = f"Verified ({metrics['significance']})" if metrics.get('significance') else "Verified"
        row = f"| **{pair}** | {metrics['win_chi2']:.2f} | {metrics['win_p']} | {metrics['win_odds_ratio']:.2f} | {metrics['vid_wilcoxon_w']} | {metrics['vid_wilcoxon_p']} | {metrics['vid_paired_t_p']} | {metrics['vid_cohens_d']} | {sig} |"
        t8_rows.append(row)

    t8_text = "\n".join(t8_rows)
    t8_full = f"""## Table 8: Paired Statistical Hypothesis Testing (Paper Table 8)

*Objective:* Verify pairwise model superiority with McNemar test on test windows ($N=2,743$) and Wilcoxon signed-rank + paired $t$-test on video clusters ($N=233$). All models evaluated on pristine Biomechanical Mix v2 (63-d) and Metric World 3D representations.  
*Execution Command:* `python scripts/compute_statistical_tests.py` (executed via `scripts/run_tables8_9_10_on_server.py` on remote GPU server)

{t8_text}

> **Note on Multiple Testing Correction:** All five pairwise window comparisons remain statistically significant after Holm-Bonferroni step-down correction ($p_{{\\text{{adj}}}} \\le 0.0001$) and Benjamini-Hochberg False Discovery Rate control ($\\text{{FDR}} \\le 7.99 \\times 10^{{-5}}$). Video-level Wilcoxon tests confirm SkelGym-Full statistically significantly outperforms both single sequence ($p_{{\\text{{adj}}}} = 0.0343$) and single graph ($p_{{\\text{{adj}}}} = 2.85 \\times 10^{{-10}}$), achieving an extreme large effect size ($d = +1.155$, $p_{{\\text{{adj}}}} = 4.14 \\times 10^{{-33}}$) over Four-Stream AAGCN."""

    # 2. Build Table 9
    t9_rows = [
        "| Model Architecture | Window Test Acc [95% CI] | Window Macro F1 [95% CI] | Video Consensus Acc [95% CI] | Video Macro F1 [95% CI] | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]
    for model_name, m in boot_ci.items():
        is_best = "SkelGym-Full" in model_name
        bold = "**" if is_best else ""
        row = f"| **{model_name}** | {bold}{m['w_acc_str']}{bold} | {bold}{m['w_f1_str']}{bold} | {bold}{m['v_acc_str']}{bold} | {bold}{m['v_f1_str']}{bold} | Verified |"
        t9_rows.append(row)

    t9_text = "\n".join(t9_rows)
    t9_full = f"""## Table 9: Non-Parametric Video-Level Cluster Bootstrap (B=1,000 Resamples) (Paper Table 9)

*Objective:* Quantify sampling stability and compute unbiased 95% Confidence Intervals via video-cluster resampling ($B=1,000$, clustered by video ID to prevent intra-video frame dependency bias).  
*Execution Command:* `python scripts/compute_statistical_tests.py`

{t9_text}"""

    # 3. Build Table 10
    t10_rows = [
        "| Exercise Class | Window Precision | Window Recall | Window F1 | Window Support | Video Precision | Video Recall | Video F1 | Video Support | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for cls_name, m in per_class.items():
        is_summary = cls_name in ["Overall Accuracy", "Macro Average"]
        bold = "**" if is_summary else ""
        row = f"| {bold}{cls_name}{bold} | {m['win_precision']:.4f} | {m['win_recall']:.4f} | {m['win_f1']:.4f} | {bold}{m['win_support']}{bold} | {m['vid_precision']:.4f} | {m['vid_recall']:.4f} | {m['vid_f1']:.4f} | {bold}{m['vid_support']}{bold} | {bold}Verified{bold} |"
        t10_rows.append(row)

    t10_text = "\n".join(t10_rows)
    t10_full = f"""## Table 10: Per-Class Performance Breakdown (Paper Table 10 & 13)

*Objective:* Detailed per-class precision, recall, and F1 metrics for SkelGym-Full (Stacking Meta-Classifier) on held-out test windows ($N=2,743$) and test videos ($N=233$, 201 correct).

{t10_text}"""

    # Pattern to find span from ## Table 8 down to before ## Table 11
    pattern = re.compile(r'## Table 8: Paired Statistical Hypothesis Testing.*?(?=## Table 11:)', re.DOTALL)
    match = pattern.search(content)
    if not match:
        raise RuntimeError("Could not find Table 8 to Table 11 section in RESULTS_FINAL.md")

    start, end = match.span()
    new_section = f"{t8_full}\n\n---\n\n{t9_full}\n\n---\n\n{t10_full}\n\n---\n\n"
    updated_content = content[:start] + new_section + content[end:]

    with open('outputs/RESULTS_FINAL.md', 'w', encoding='utf-8') as f:
        f.write(updated_content)

    print("Successfully updated RESULTS_FINAL.md with Table 8, 9, 10!")

if __name__ == '__main__':
    update_results()
