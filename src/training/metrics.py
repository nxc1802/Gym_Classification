"""
Evaluation metrics, classification report, confusion matrix plotting, and LaTeX export.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.constants import ACTIONS, NUM_CLASSES

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes overall accuracy, macro/weighted precision, recall, and F1.
    """
    if target_names is None:
        target_names = ACTIONS

    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = list(range(len(target_names)))
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f1_weight),
        "report_dict": report_dict
    }

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    target_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> None:
    """
    Plots and saves publication-quality confusion matrix in both high-res PNG and vector PDF.
    """
    if target_names is None:
        target_names = ACTIONS
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    # Format annotations: display values >= 0.01 in bold, suppress 0.00 clutter for visual clarity
    annot_data = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if normalize:
                annot_data[i, j] = f"{val:.2f}" if val >= 0.005 else ""
            else:
                annot_data[i, j] = f"{int(val)}" if val > 0 else ""

    fig, ax = plt.subplots(figsize=(15, 13))
    sns.set_theme(style="white", font_scale=0.95)
    
    sns.heatmap(
        cm,
        annot=annot_data,
        fmt="",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
        cbar_kws={
            "label": "Normalized Recall / Accuracy" if normalize else "Sample Count",
            "shrink": 0.82,
            "pad": 0.02
        },
        linewidths=0.6,
        linecolor="#dbeafe",
        annot_kws={"size": 9.5, "weight": "bold", "va": "center", "ha": "center"},
        ax=ax
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted Exercise Class", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_ylabel("True Exercise Class", fontsize=13, fontweight="bold", labelpad=12)
    ax.set_xticklabels(target_names, rotation=45, ha="right", fontsize=11, fontweight="semibold")
    ax.set_yticklabels(target_names, rotation=0, fontsize=11, fontweight="semibold")

    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Save high-res PNG (400 DPI, tight bounding box)
    plt.savefig(str(out_p), dpi=400, bbox_inches="tight")
    
    # Save crisp Vector PDF for publication
    pdf_p = out_p.with_suffix(".pdf")
    plt.savefig(str(pdf_p), format="pdf", bbox_inches="tight")
    
    plt.close()

def export_latex_table7(report_dict: Dict[str, Any], output_tex_path: str) -> str:
    """
    Exports classification metrics in exact Springer LNCS Table 7 format.
    """
    lines = [
        "\\begin{table}[!h]",
        "\\centering",
        "\\caption{Classification report for the stacking ensemble on the test set}",
        "\\label{tab:stacking-report}",
        "\\begin{tabular}{l c c c r}",
        "\\hline",
        "\\textbf{Exercise Class}   & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-score} & \\textbf{Support} \\\\",
        "\\hline"
    ]

    for act in ACTIONS:
        if act in report_dict:
            stats = report_dict[act]
            p = f"{stats['precision']:.4f}"
            r = f"{stats['recall']:.4f}"
            f1 = f"{stats['f1-score']:.4f}"
            sup = int(stats["support"])
            lines.append(f"{act:<25} & {p:<18} & {r:<15} & {f1:<17} & {sup:<16} \\\\")

    lines.append("\\hline")
    acc = report_dict.get("accuracy", 0.0)
    total_sup = int(report_dict.get("macro avg", {}).get("support", 0))
    lines.append(f"\\multicolumn{{1}}{{l}}{{\\textbf{{Accuracy}}}}      &                   &           &          {acc:.4f}         & {total_sup:<16} \\\\")

    macro = report_dict.get("macro avg", {})
    lines.append(f"\\multicolumn{{1}}{{l}}{{\\textbf{{Macro avg}}}}     & {macro.get('precision', 0):.4f}            & {macro.get('recall', 0):.4f}          & {macro.get('f1-score', 0):.4f}            & {total_sup:<16} \\\\")

    weighted = report_dict.get("weighted avg", {})
    lines.append(f"\\multicolumn{{1}}{{l}}{{\\textbf{{Weighted avg}}}}  & {weighted.get('precision', 0):.4f}            & {weighted.get('recall', 0):.4f}          & {weighted.get('f1-score', 0):.4f}            & {total_sup:<16} \\\\")

    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])

    tex_str = "\n".join(lines)
    out_p = Path(output_tex_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(tex_str)

    return tex_str
