"""
Ensemble methods for gym exercise classification.
Includes Hard Voting, Soft Voting, and Stacking Meta-Classifier.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class HardVotingEnsemble:
    """
    Hard voting ensemble: picks the class with majority vote across models.
    """
    def predict(self, model_predictions: List[np.ndarray]) -> np.ndarray:
        # model_predictions: list of M arrays of shape (N,)
        stacked = np.stack(model_predictions, axis=0)  # (M, N)
        final_preds = []
        for i in range(stacked.shape[1]):
            col = stacked[:, i]
            vals, counts = np.unique(col, return_counts=True)
            final_preds.append(vals[np.argmax(counts)])
        return np.array(final_preds, dtype=np.int64)

class SoftVotingEnsemble:
    """
    Soft voting ensemble: averages the predicted probability distributions across models.
    Optionally weights models according to their validation performance.
    """
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights

    def predict_proba(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        # model_probabilities: list of M arrays of shape (N, num_classes)
        if self.weights is not None:
            w = np.array(self.weights) / np.sum(self.weights)
            weighted_probs = sum(p * weight for p, weight in zip(model_probabilities, w))
            return weighted_probs
        else:
            return np.mean(model_probabilities, axis=0)

    def predict(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        avg_probs = self.predict_proba(model_probabilities)
        return np.argmax(avg_probs, axis=1)

class StackingEnsemble:
    """
    Stacking ensemble: trains a meta-classifier on base models' probability vectors.
    """
    def __init__(self, c_param: float = 1.0, max_iter: int = 1000):
        self.meta_classifier = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            solver="lbfgs"
        )
        self.is_fitted = False

    def fit(self, val_probabilities: List[np.ndarray], val_labels: np.ndarray) -> "StackingEnsemble":
        """
        Trains the meta-learner on validation probabilities.
        Concatenates probabilities along feature dimension: (N, M * num_classes)
        """
        X_meta = np.concatenate(val_probabilities, axis=1)
        self.meta_classifier.fit(X_meta, val_labels)
        self.is_fitted = True
        return self

    def predict(self, test_probabilities: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predict!")
        X_meta = np.concatenate(test_probabilities, axis=1)
        return self.meta_classifier.predict(X_meta)

    def predict_proba(self, test_probabilities: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predict_proba!")
        X_meta = np.concatenate(test_probabilities, axis=1)
        return self.meta_classifier.predict_proba(X_meta)

class WeightedSoftVotingEnsemble:
    """
    Weighted Soft Voting Ensemble with continuous weights optimized on Validation set via SLSQP.
    Directly implements Section 6 of RESEARCH_UPGRADE_ROADMAP.md.
    Objective: minimize negative log-likelihood on validation probabilities.
    """
    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights
        self.is_fitted = (weights is not None)

    def fit(self, val_probabilities: List[np.ndarray], val_labels: np.ndarray) -> "WeightedSoftVotingEnsemble":
        """
        Optimizes weights w in [0, 1] with sum(w) = 1 to minimize cross-entropy / negative log-likelihood on validation set.
        """
        from scipy.optimize import minimize
        M = len(val_probabilities)
        init_weights = np.ones(M, dtype=np.float64) / M
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        eps = 1e-12

        def nll_objective(w):
            w = np.array(w, dtype=np.float64)
            w_sum = np.sum(w)
            if w_sum <= 0:
                return 1e6
            w_norm = w / w_sum
            blended = sum(p * weight for p, weight in zip(val_probabilities, w_norm))
            prob_true = blended[np.arange(len(val_labels)), val_labels]
            return -np.mean(np.log(np.clip(prob_true, eps, 1.0)))

        res = minimize(nll_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = np.array(res.x, dtype=np.float64)
        self.weights = list(opt_w / np.sum(opt_w))
        self.is_fitted = True
        return self

    def predict_proba(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted or self.weights is None:
            return np.mean(model_probabilities, axis=0)
        w = np.array(self.weights, dtype=np.float64)
        w_norm = w / np.sum(w)
        return sum(p * weight for p, weight in zip(model_probabilities, w_norm))

    def predict(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        avg_probs = self.predict_proba(model_probabilities)
        return np.argmax(avg_probs, axis=1)

