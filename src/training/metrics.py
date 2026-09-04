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
    Plots and saves confusion matrix figure.
    """
    if target_names is None:
        target_names = ACTIONS
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    plt.figure(figsize=(14, 12))
    sns.set_theme(font_scale=0.9)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), dpi=300)
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
