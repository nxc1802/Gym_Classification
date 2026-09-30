"""
Unit tests for Statistical Inference & Hypothesis Testing (Roadmap Part 3).
Validates:
  1. McNemar's test (Exact binomial & asymptotic chi-squared with continuity correction).
  2. Cluster bootstrap resampling (clustered by source video ID).
  3. Video-level bootstrap resampling.
  4. Video-level continuous probability paired tests (Wilcoxon and paired t-test).
  5. Multiple comparison corrections (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR).
"""

import unittest
import numpy as np

from src.utils.statistics import (
    mcnemar_test,
    cluster_bootstrap_window,
    bootstrap_video,
    paired_video_confidence_test,
    adjust_p_values,
    format_p_value,
    get_significance_stars,
)

class TestStatisticalInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_01_mcnemar_identity_and_symmetry(self):
        """McNemar test between identical predictions yields p=1.0 and chi2=0."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred1 = y_true.copy()
        y_pred2 = y_true.copy()

        res = mcnemar_test(y_true, y_pred1, y_pred2)
        self.assertEqual(res["b"], 0)
        self.assertEqual(res["c"], 0)
        self.assertEqual(res["n_discordant"], 0)
        self.assertEqual(res["chi2"], 0.0)
        self.assertEqual(res["p_value"], 1.0)
        self.assertEqual(res["odds_ratio"], 1.0)

    def test_02_mcnemar_discordant_pairs(self):
        """Validates McNemar test with known discordant counts."""
        # 10 samples:
        # y_true = all 1s
        # pred1: 8 correct, 2 wrong
        # pred2: 4 correct, 6 wrong
        y_true = np.ones(10, dtype=int)
        y_pred1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
        y_pred2 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        res = mcnemar_test(y_true, y_pred1, y_pred2)
        # b = pred1 right, pred2 wrong (indices 4, 5, 6, 7 -> 4)
        # c = pred1 wrong, pred2 right (none -> 0)
        self.assertEqual(res["b"], 4)
        self.assertEqual(res["c"], 0)
        self.assertEqual(res["n_discordant"], 4)
        # With b=4, c=0, continuity corrected chi2 = (|4-0|-1)^2 / 4 = 9/4 = 2.25
        self.assertAlmostEqual(res["chi2"], 2.25, places=4)
        # Exact two-sided binomial p-value for k=0, n=4, p=0.5 is 2 * (0.5)^4 = 2 * 0.0625 = 0.125
        self.assertAlmostEqual(res["exact_p_value"], 0.125, places=4)

    def test_03_cluster_bootstrap_window(self):
        """Validates that cluster bootstrap runs and yields valid confidence interval bounds."""
        # 20 windows across 4 videos (5 windows per video)
        video_ids = [f"vid_{i // 5}" for i in range(20)]
        y_true = np.array([i % 3 for i in range(20)])
        y_pred = y_true.copy()
        # introduce 2 errors in vid_0
        y_pred[0] = (y_pred[0] + 1) % 3
        y_pred[1] = (y_pred[1] + 1) % 3

        boot = cluster_bootstrap_window(y_true, y_pred, video_ids, B=100, seed=42)

        self.assertEqual(boot["num_clusters"], 4)
        self.assertEqual(boot["total_windows"], 20)
        self.assertTrue(boot["acc_ci"][0] <= boot["acc_mean"] <= boot["acc_ci"][1])
        self.assertTrue(boot["f1_ci"][0] <= boot["f1_mean"] <= boot["f1_ci"][1])
        self.assertTrue(0.0 <= boot["acc_ci"][0] <= 100.0)

    def test_04_bootstrap_video(self):
        """Validates video-level bootstrap returns valid confidence interval bounds."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = np.array([0, 1, 2, 0, 1, 2, 0, 0, 2, 0])  # 9/10 correct = 90%

        boot = bootstrap_video(y_true, y_pred, B=100, seed=42)

        self.assertEqual(boot["N"], 10)
        self.assertTrue(boot["acc_ci"][0] <= boot["acc_mean"] <= boot["acc_ci"][1])
        self.assertTrue(boot["f1_ci"][0] <= boot["f1_mean"] <= boot["f1_ci"][1])

    def test_05_paired_video_confidence_test(self):
        """Validates paired Wilcoxon and t-test on continuous prediction probabilities."""
        N = 15
        y_true = np.random.randint(0, 3, size=N)
        # Model 1 assigns modest probability to true class (~0.4)
        prob1 = np.ones((N, 3)) * 0.3
        for i in range(N):
            prob1[i, y_true[i]] = 0.4
        prob1 /= prob1.sum(axis=1, keepdims=True)

        # Model 2 assigns much higher probability to true class (~0.8)
        prob2 = np.ones((N, 3)) * 0.1
        for i in range(N):
            prob2[i, y_true[i]] = 0.8
        prob2 /= prob2.sum(axis=1, keepdims=True)

        res = paired_video_confidence_test(y_true, prob1, prob2)
        self.assertTrue(res["mean_delta_prob"] > 0)
        self.assertTrue(res["wilcoxon_p"] < 0.01)
        self.assertTrue(res["ttest_p"] < 0.01)
        self.assertTrue(res["cohens_d"] > 1.0)

    def test_06_multiple_comparison_corrections(self):
        """Validates Bonferroni, Holm-Bonferroni, and Benjamini-Hochberg FDR adjustments."""
        raw_p = [0.001, 0.01, 0.03, 0.05, 0.20]
        m = len(raw_p)

        # 1. Bonferroni: p_adj = min(1.0, p * m)
        bonf_adj = adjust_p_values(raw_p, method="bonferroni")
        self.assertAlmostEqual(bonf_adj[0], 0.001 * 5, places=5)
        self.assertAlmostEqual(bonf_adj[1], 0.01 * 5, places=5)
        self.assertAlmostEqual(bonf_adj[2], 0.03 * 5, places=5)
        self.assertAlmostEqual(bonf_adj[3], 0.05 * 5, places=5)
        self.assertAlmostEqual(bonf_adj[4], 1.0, places=5)

        # 2. Holm-Bonferroni (strictly monotone and <= Bonferroni)
        holm_adj = adjust_p_values(raw_p, method="holm")
        for i in range(m):
            self.assertTrue(holm_adj[i] <= bonf_adj[i] + 1e-9)
        # Check step-down values:
        # rank 1: p=0.001 * 5 = 0.005
        # rank 2: p=0.01 * 4 = 0.04
        # rank 3: p=0.03 * 3 = 0.09
        # rank 4: p=0.05 * 2 = 0.10
        # rank 5: p=0.20 * 1 = 0.20
        self.assertAlmostEqual(holm_adj[0], 0.005, places=5)
        self.assertAlmostEqual(holm_adj[1], 0.04, places=5)
        self.assertAlmostEqual(holm_adj[2], 0.09, places=5)
        self.assertAlmostEqual(holm_adj[3], 0.10, places=5)
        self.assertAlmostEqual(holm_adj[4], 0.20, places=5)

        # 3. Benjamini-Hochberg FDR (strictly <= Holm)
        fdr_adj = adjust_p_values(raw_p, method="fdr_bh")
        for i in range(m):
            self.assertTrue(fdr_adj[i] <= holm_adj[i] + 1e-9)

    def test_07_formatting_utilities(self):
        """Validates publication string formatters."""
        self.assertEqual(format_p_value(1.23e-8), "1.23e-08")
        self.assertEqual(format_p_value(0.02456), "0.0246")
        self.assertEqual(get_significance_stars(0.0001), "***")
        self.assertEqual(get_significance_stars(0.005), "**")
        self.assertEqual(get_significance_stars(0.03), "*")
        self.assertEqual(get_significance_stars(0.15), "n.s.")

if __name__ == "__main__":
    unittest.main()
