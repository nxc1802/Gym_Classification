"""
Metrics, Statistical Bootstrapping, and Domain-Gap Diagnostics for External Benchmarks.
Implements:
1. Open-Set Evaluation (22-class argmax over external samples)
2. Closed-Set Evaluation (Core-K hypothesis subspace conditioning)
3. Hierarchical Aggregation: Window -> Segment/Repetition -> Recording/Workout
4. Subject/Recording-level Non-Parametric Bootstrap 95% Confidence Intervals (B=1000)
5. Paired Model Comparison Bootstrap Differences (Delta Accuracy, Delta F1)
6. Domain-Gap Diagnostics (Class Centroid Cosine Distance between SkelGym and External Benchmark)
7. Error Analysis Manifest (misclassified records with margin and second-guess prediction)
"""

from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.constants import ACTIONS, NUM_CLASSES, ACTION_TO_IDX, IDX_TO_ACTION

def evaluate_window_level(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    target_class_indices: List[int],
    mode: str = "open_set"
) -> Dict[str, Any]:
    """
    Evaluates predictions at window level.
    mode:
      - 'open_set': Argmax over all 22 SkelGym classes.
      - 'closed_set': Probability re-normalized across target_class_indices only.
    """
    N = len(y_true)
    if N == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "preds": np.array([])}

    if mode == "open_set":
        preds = np.argmax(y_probs, axis=1)
        acc = accuracy_score(y_true, preds) * 100.0
        # Compute macro F1 strictly on the evaluated target classes
        f1 = f1_score(y_true, preds, labels=target_class_indices, average="macro", zero_division=0)
        rec = recall_score(y_true, preds, labels=target_class_indices, average=None, zero_division=0)
    elif mode == "closed_set":
        # Restrict columns and re-normalize
        sub_probs = y_probs[:, target_class_indices]
        sub_probs_norm = sub_probs / (np.sum(sub_probs, axis=1, keepdims=True) + 1e-12)
        closed_preds_local = np.argmax(sub_probs_norm, axis=1)
        preds = np.array([target_class_indices[p] for p in closed_preds_local])

        acc = accuracy_score(y_true, preds) * 100.0
        f1 = f1_score(y_true, preds, labels=target_class_indices, average="macro", zero_division=0)
        rec = recall_score(y_true, preds, labels=target_class_indices, average=None, zero_division=0)
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    per_class_rec = {ACTIONS[idx]: float(rec[i] * 100.0) for i, idx in enumerate(target_class_indices)}

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "per_class_recall": per_class_rec,
        "preds": preds,
        "probs": y_probs
    }

def aggregate_hierarchical_predictions(
    y_probs: np.ndarray,
    y_true: np.ndarray,
    group_ids: List[str],
    target_class_indices: List[int],
    mode: str = "open_set"
) -> Dict[str, Any]:
    """
    Aggregates window-level probabilities to group level (e.g. segment or workout/recording).
    P_group = (1/K) * sum_{k=1}^K P_window_k
    """
    unique_groups = []
    group_to_indices = {}
    for idx, g in enumerate(group_ids):
        if g not in group_to_indices:
            unique_groups.append(g)
            group_to_indices[g] = []
        group_to_indices[g].append(idx)

    group_probs = []
    group_trues = []

    for g in unique_groups:
        indices = group_to_indices[g]
        avg_prob = np.mean(y_probs[indices], axis=0)
        group_probs.append(avg_prob)
        group_trues.append(y_true[indices[0]])

    group_probs_arr = np.array(group_probs, dtype=np.float32)
    group_trues_arr = np.array(group_trues, dtype=np.int64)

    eval_res = evaluate_window_level(
        y_probs=group_probs_arr,
        y_true=group_trues_arr,
        target_class_indices=target_class_indices,
        mode=mode
    )

    eval_res["group_ids"] = unique_groups
    eval_res["group_trues"] = group_trues_arr
    eval_res["group_probs"] = group_probs_arr
    return eval_res

def compute_recording_level_bootstrap_ci(
    y_probs_by_group: np.ndarray,
    y_trues_by_group: np.ndarray,
    target_class_indices: List[int],
    mode: str = "open_set",
    cluster_ids: Optional[List[str]] = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Non-parametric bootstrap confidence interval at recording/group level.
    If cluster_ids (e.g. workout IDs ['w00', 'w05', ...]) are provided:
      Performs cluster bootstrap resampling: resamples entire workout clusters with replacement.
    Otherwise:
      Resamples individual recordings with replacement (B=1000 iterations).
    """
    rng = np.random.default_rng(seed)
    N = len(y_trues_by_group)
    if N < 2:
        return {"acc_mean": 0.0, "acc_ci": [0.0, 0.0], "f1_mean": 0.0, "f1_ci": [0.0, 0.0]}

    boot_accs = []
    boot_f1s = []

    alpha_low = ((1.0 - confidence_level) / 2.0) * 100.0
    alpha_high = (1.0 - (1.0 - confidence_level) / 2.0) * 100.0

    if cluster_ids is not None:
        cluster_arr = np.array(cluster_ids)
        unique_clusters = np.unique(cluster_arr)
        K = len(unique_clusters)
        cluster_to_indices = {c: np.where(cluster_arr == c)[0] for c in unique_clusters}
    else:
        unique_clusters = None

    for _ in range(n_bootstraps):
        if unique_clusters is not None:
            sampled_clusters = rng.choice(unique_clusters, size=K, replace=True)
            sampled_indices = []
            for sc in sampled_clusters:
                sampled_indices.extend(cluster_to_indices[sc])
            boot_idx = np.array(sampled_indices, dtype=np.int64)
        else:
            boot_idx = rng.choice(N, size=N, replace=True)

        sub_probs = y_probs_by_group[boot_idx]
        sub_trues = y_trues_by_group[boot_idx]

        res = evaluate_window_level(sub_probs, sub_trues, target_class_indices, mode=mode)
        boot_accs.append(res["accuracy"])
        boot_f1s.append(res["macro_f1"])

    return {
        "acc_mean": float(np.mean(boot_accs)),
        "acc_std": float(np.std(boot_accs)),
        "acc_ci": [float(np.percentile(boot_accs, alpha_low)), float(np.percentile(boot_accs, alpha_high))],
        "f1_mean": float(np.mean(boot_f1s)),
        "f1_std": float(np.std(boot_f1s)),
        "f1_ci": [float(np.percentile(boot_f1s, alpha_low)), float(np.percentile(boot_f1s, alpha_high))]
    }

def compute_paired_bootstrap_difference(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    y_trues: np.ndarray,
    target_class_indices: List[int],
    mode: str = "open_set",
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired bootstrap difference Delta = Model_A - Model_B.
    """
    rng = np.random.default_rng(seed)
    N = len(y_trues)
    diff_accs = []
    diff_f1s = []

    for _ in range(n_bootstraps):
        idx = rng.choice(N, size=N, replace=True)
        res_a = evaluate_window_level(probs_a[idx], y_trues[idx], target_class_indices, mode=mode)
        res_b = evaluate_window_level(probs_b[idx], y_trues[idx], target_class_indices, mode=mode)
        diff_accs.append(res_a["accuracy"] - res_b["accuracy"])
        diff_f1s.append(res_a["macro_f1"] - res_b["macro_f1"])

    return {
        "delta_acc_mean": float(np.mean(diff_accs)),
        "delta_acc_ci": [float(np.percentile(diff_accs, 2.5)), float(np.percentile(diff_accs, 97.5))],
        "delta_f1_mean": float(np.mean(diff_f1s)),
        "delta_f1_ci": [float(np.percentile(diff_f1s, 2.5)), float(np.percentile(diff_f1s, 97.5))]
    }

def compute_domain_gap_diagnostics(
    skelgym_embs: np.ndarray,
    skelgym_labels: np.ndarray,
    external_embs: np.ndarray,
    external_labels: np.ndarray,
    target_class_indices: List[int]
) -> Dict[str, Any]:
    """
    Calculates class centroid cosine similarities between SkelGym test set and External benchmark.
    Quantifies semantic domain divergence for each shared exercise.
    """
    results = {}
    similarities = []

    for c_idx in target_class_indices:
        c_name = ACTIONS[c_idx]
        mask_skel = (skelgym_labels == c_idx)
        mask_ext = (external_labels == c_idx)

        if not np.any(mask_skel) or not np.any(mask_ext):
            continue

        cent_skel = np.mean(skelgym_embs[mask_skel], axis=0)
        cent_ext = np.mean(external_embs[mask_ext], axis=0)

        norm_s = np.linalg.norm(cent_skel)
        norm_e = np.linalg.norm(cent_ext)
        cos_sim = float(np.dot(cent_skel, cent_ext) / ((norm_s * norm_e) + 1e-12))

        results[c_name] = {
            "cosine_similarity": cos_sim,
            "cosine_distance": float(1.0 - cos_sim),
            "n_skelgym": int(np.sum(mask_skel)),
            "n_external": int(np.sum(mask_ext))
        }
        similarities.append(cos_sim)

    results["mean_cross_domain_similarity"] = float(np.mean(similarities)) if similarities else 0.0
    return results

def plot_external_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_class_indices: List[int],
    dest_path: Union[str, Path],
    title: str = "Confusion Matrix"
) -> None:
    """
    Plots and exports high-resolution confusion matrix for publication.
    """
    dest_p = Path(dest_path)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    labels = target_class_indices
    names = [ACTIONS[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-12) * 100.0

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={'label': 'Recall (%)'}
    )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("Ground Truth Class", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(dest_p, dpi=300)
    plt.close()

def generate_error_analysis_table(
    y_probs: np.ndarray,
    y_trues: np.ndarray,
    record_ids: List[str],
    subject_ids: List[str],
    dataset_name: str,
    pose_source: str,
    dest_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Constructs detailed misclassified_records.csv for audit and inspection.
    """
    dest_p = Path(dest_path)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    preds = np.argmax(y_probs, axis=1)

    for i in range(len(y_trues)):
        t_cls = y_trues[i]
        p_cls = preds[i]
        if t_cls != p_cls:
            prob_vec = y_probs[i]
            sorted_idx = np.argsort(prob_vec)[::-1]
            top1_cls = sorted_idx[0]
            top2_cls = sorted_idx[1]
            conf = float(prob_vec[top1_cls])
            conf2 = float(prob_vec[top2_cls])
            margin = conf - conf2

            rows.append({
                "dataset": dataset_name,
                "subject": subject_ids[i],
                "record_id": record_ids[i],
                "true_class": ACTIONS[t_cls],
                "pred_class": ACTIONS[p_cls],
                "confidence": conf,
                "second_class": ACTIONS[top2_cls],
                "second_confidence": conf2,
                "margin": margin,
                "pose_source": pose_source
            })

    df_err = pd.DataFrame(rows)
    df_err.to_csv(dest_p, index=False)
    return df_err
