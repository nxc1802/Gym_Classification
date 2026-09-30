"""
Statistical Inference and Hypothesis Testing for SkelGym Biomechanical Classification.
Supports:
  - Video-Cluster-Aware Bootstrap Resampling (accounts for intra-video correlation)
  - McNemar's Test (Exact Binomial and Asymptotic Chi-Squared with Edwards Continuity Correction)
  - Video-Level Paired Tests (Wilcoxon Signed-Rank, Paired t-Test, Effect Sizes)
  - Multiple Testing Corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR)
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    exact: bool = True
) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired binary classification outcomes.
    Discordant pairs:
      b: Model 1 correct, Model 2 incorrect
      c: Model 1 incorrect, Model 2 correct

    Returns:
      b, c, chi2, p_value, exact_p_value, odds_ratio, n_discordant
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    c01 = int(np.sum((y_pred1 != y_true) & (y_pred2 == y_true)))  # Model 1 wrong, Model 2 right
    c10 = int(np.sum((y_pred1 == y_true) & (y_pred2 != y_true)))  # Model 1 right, Model 2 wrong
    b, c = c10, c01
    n_discordant = b + c

    # Asymptotic Chi-squared statistic with Edwards continuity correction
    if n_discordant > 0:
        chi2 = float((abs(b - c) - 1.0)**2 / n_discordant)
        asymp_p = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        asymp_p = 1.0

    # Exact binomial test under H0: p = 0.5
    if n_discordant > 0:
        exact_res = stats.binomtest(k=min(b, c), n=n_discordant, p=0.5, alternative="two-sided")
        exact_p = float(exact_res.pvalue)
    else:
        exact_p = 1.0

    odds_ratio = float(c / b) if b > 0 else (float("inf") if c > 0 else 1.0)
    p_val = exact_p if exact else asymp_p

    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "chi2": chi2,
        "p_value": p_val,
        "asymp_p_value": asymp_p,
        "exact_p_value": exact_p,
        "odds_ratio": odds_ratio,
        "acc1": float(np.mean(y_pred1 == y_true) * 100.0),
        "acc2": float(np.mean(y_pred2 == y_true) * 100.0),
        "delta_acc": float((np.mean(y_pred2 == y_true) - np.mean(y_pred1 == y_true)) * 100.0)
    }

def cluster_bootstrap_window(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    video_ids: List[str],
    B: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs Cluster Bootstrap Resampling clustered by source video ID.
    Windows from the same video are sampled together as a cluster, preserving
    intra-video temporal and biomechanical correlation.

    Returns:
      acc_mean, acc_ci, acc_std, f1_mean, f1_ci, f1_std
    """
    np.random.seed(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    video_ids = np.asarray(video_ids)

    unique_vids = np.unique(video_ids)
    num_clusters = len(unique_vids)

    # Pre-index windows per unique video for O(1) cluster extraction
    vid_to_indices = {vid: np.where(video_ids == vid)[0] for vid in unique_vids}

    accs = []
    f1s = []

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(B):
        sampled_vids = np.random.choice(unique_vids, size=num_clusters, replace=True)
        # Gather all window indices for sampled videos
        sampled_indices = np.concatenate([vid_to_indices[v] for v in sampled_vids])

        yt = y_true[sampled_indices]
        yp = y_pred[sampled_indices]

        acc = float(np.mean(yt == yp) * 100.0)
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

        accs.append(acc)
        f1s.append(f1)

    accs = np.array(accs)
    f1s = np.array(f1s)

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_ci": (float(np.percentile(accs, lower_pct)), float(np.percentile(accs, upper_pct))),
        "acc_std": float(np.std(accs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_ci": (float(np.percentile(f1s, lower_pct)), float(np.percentile(f1s, upper_pct))),
        "f1_std": float(np.std(f1s, ddof=1)),
        "num_clusters": num_clusters,
        "total_windows": len(y_true),
        "B": B
    }

def bootstrap_video(
    y_true_vid: np.ndarray,
    y_pred_vid: np.ndarray,
    B: int = 1000,
    seed: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Non-parametric bootstrap resampling at video level (where each video is an independent unit).
    """
    np.random.seed(seed)
    y_true_vid = np.asarray(y_true_vid)
    y_pred_vid = np.asarray(y_pred_vid)
    N = len(y_true_vid)

    accs = []
    f1s = []

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for _ in range(B):
        idx = np.random.choice(N, size=N, replace=True)
        yt = y_true_vid[idx]
        yp = y_pred_vid[idx]

        accs.append(float(np.mean(yt == yp) * 100.0))
        f1s.append(float(f1_score(yt, yp, average="macro", zero_division=0)))

    accs = np.array(accs)
    f1s = np.array(f1s)

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_ci": (float(np.percentile(accs, lower_pct)), float(np.percentile(accs, upper_pct))),
        "acc_std": float(np.std(accs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_ci": (float(np.percentile(f1s, lower_pct)), float(np.percentile(f1s, upper_pct))),
        "f1_std": float(np.std(f1s, ddof=1)),
        "N": N,
        "B": B
    }

def paired_video_confidence_test(
    y_true_vid: np.ndarray,
    prob1: np.ndarray,
    prob2: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates continuous posterior probability assigned to ground truth class:
      p_true_1 vs p_true_2
    Performs Wilcoxon signed-rank test and paired Student's t-test on probability margins.
    """
    y_true = np.asarray(y_true_vid)
    N = len(y_true)

    # Extract probability assigned to the ground-truth class
    p_true1 = prob1[np.arange(N), y_true]
    p_true2 = prob2[np.arange(N), y_true]

    diff = p_true2 - p_true1
    n_nonzero = int(np.sum(diff != 0))

    if n_nonzero > 0 and N > 1:
        try:
            w_res = stats.wilcoxon(p_true2, p_true1, zero_method="pratt")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0
    else:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0

    if N > 1:
        t_res = stats.ttest_rel(p_true2, p_true1)
        ttest_stat = float(t_res.statistic) if not np.isnan(t_res.statistic) else 0.0
        ttest_p = float(t_res.pvalue) if not np.isnan(t_res.pvalue) else 1.0
        s_d = float(np.std(diff, ddof=1))
        cohens_d = float(np.mean(diff) / s_d) if s_d > 0 else 0.0
    else:
        ttest_stat = 0.0
        ttest_p = 1.0
        cohens_d = 0.0

    return {
        "mean_prob1": float(np.mean(p_true1)),
        "mean_prob2": float(np.mean(p_true2)),
        "mean_delta_prob": float(np.mean(diff)),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "ttest_stat": ttest_stat,
        "ttest_p": ttest_p,
        "cohens_d": cohens_d,
        "n_nonzero": n_nonzero,
        "N": N
    }

def adjust_p_values(p_values: List[float], method: str = "holm") -> List[float]:
    """
    Applies multiple comparison adjustment to a collection of p-values.
    Methods:
      - 'bonferroni': Single-step Bonferroni FWER control (p * m)
      - 'holm': Step-down Holm-Bonferroni FWER control (strictly more powerful than Bonferroni)
      - 'fdr_bh': Benjamini-Hochberg False Discovery Rate control
    """
    m = len(p_values)
    if m <= 1:
        return [float(p) for p in p_values]

    p_arr = np.asarray(p_values, dtype=float)
    method = method.lower()

    if method == "bonferroni":
        adjusted = np.clip(p_arr * m, 0.0, 1.0)
        return [float(p) for p in adjusted]

    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adj_sorted = np.empty(m, dtype=float)

    if method == "holm":
        # Step-down: (m - i) * p_(i) with monotonicity enforcement
        running_max = 0.0
        for i in range(m):
            mult = m - i
            val = min(1.0, sorted_p[i] * mult)
            running_max = max(running_max, val)
            adj_sorted[i] = running_max
    elif method in ("fdr_bh", "bh", "fdr"):
        # Step-up: (m / (i + 1)) * p_(i) with reverse monotonicity enforcement
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            mult = float(m) / float(i + 1)
            val = min(1.0, sorted_p[i] * mult)
            running_min = min(running_min, val)
            adj_sorted[i] = running_min
    else:
        raise ValueError(f"Unknown adjustment method: {method}. Choose from 'bonferroni', 'holm', 'fdr_bh'.")

    # Invert sort order back to original order
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    return [float(p) for p in adjusted]

def format_p_value(p: float) -> str:
    """Formats p-values into clean publication string with scientific notation for small values."""
    if p < 0.0001:
        return f"{p:.2e}"
    else:
        return f"{p:.4f}"

def get_significance_stars(p: float) -> str:
    """Returns standard APA/academic significance markers: *** p < 0.001, ** p < 0.01, * p < 0.05, n.s."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."
