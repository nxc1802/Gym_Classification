"""
Ensemble methods for gym exercise classification.
Includes Hard Voting, Soft Voting, and Stacking Meta-Classifier.
"""

from typing import List, Dict, Any, Optional, Tuple
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
        if len(np.unique(val_labels)) < 2:
            # Fallback for smoke test / single class debugging: mark fitted without training LogisticRegression
            self.is_fitted = True
            return self

        X_meta = np.concatenate(val_probabilities, axis=1)
        self.meta_classifier.fit(X_meta, val_labels)
        self.is_fitted = True
        return self

    def predict(self, test_probabilities: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predict!")
        if not hasattr(self.meta_classifier, "classes_"):
            return np.argmax(np.mean(test_probabilities, axis=0), axis=1)
        X_meta = np.concatenate(test_probabilities, axis=1)
        return self.meta_classifier.predict(X_meta)

    def predict_proba(self, test_probabilities: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble must be fitted before predict_proba!")
        if not hasattr(self.meta_classifier, "classes_"):
            return np.mean(test_probabilities, axis=0)
        X_meta = np.concatenate(test_probabilities, axis=1)
        return self.meta_classifier.predict_proba(X_meta)

class WeightedSoftVotingEnsemble:
    """
    Weighted Soft Voting Ensemble with Dual-Target Optimization (Window-Level and Video-Level).
    Uses SLSQP constrained optimization to minimize negative log-likelihood (NLL).
    """
    def __init__(
        self,
        weights_window: Optional[List[float]] = None,
        weights_video: Optional[List[float]] = None
    ):
        self.weights_window = weights_window
        self.weights_video = weights_video
        self.weights = weights_window  # Default backward compatibility
        self.is_fitted = (weights_window is not None)

    def fit_window(self, val_probabilities: List[np.ndarray], val_labels: np.ndarray) -> "WeightedSoftVotingEnsemble":
        """
        Phase 1: Optimizes weights w_win in [0, 1] with sum(w) = 1 to minimize NLL on validation windows.
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
        self.weights_window = list(opt_w / np.sum(opt_w))
        self.weights = self.weights_window
        self.is_fitted = True
        return self

    def fit_video(
        self,
        val_probabilities: List[np.ndarray],
        val_labels: np.ndarray,
        val_video_ids: List[str]
    ) -> "WeightedSoftVotingEnsemble":
        """
        Phase 2: Optimizes weights w_vid in [0, 1] with sum(w) = 1 to minimize NLL on validation video consensus.
        First aggregates each model's validation windows to video level, then optimizes w_vid.
        """
        from scipy.optimize import minimize
        M = len(val_probabilities)
        unique_vids = []
        vid_to_indices = {}
        for idx, v in enumerate(val_video_ids):
            if v not in vid_to_indices:
                unique_vids.append(v)
                vid_to_indices[v] = []
            vid_to_indices[v].append(idx)

        # Aggregate each model's probabilities to video level
        # Shape per model: (N_unique_videos, num_classes)
        val_video_probs_per_model = []
        video_true_labels = []
        for v in unique_vids:
            indices = vid_to_indices[v]
            video_true_labels.append(val_labels[indices[0]])
        video_true_labels = np.array(video_true_labels, dtype=np.int64)

        for m_idx in range(M):
            m_probs = val_probabilities[m_idx]
            v_probs_m = [np.mean(m_probs[vid_to_indices[v]], axis=0) for v in unique_vids]
            val_video_probs_per_model.append(np.array(v_probs_m, dtype=np.float64))

        init_weights = np.ones(M, dtype=np.float64) / M
        bounds = [(0.0, 1.0) for _ in range(M)]
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        eps = 1e-12

        def nll_video_objective(w):
            w = np.array(w, dtype=np.float64)
            w_sum = np.sum(w)
            if w_sum <= 0:
                return 1e6
            w_norm = w / w_sum
            blended = sum(p * weight for p, weight in zip(val_video_probs_per_model, w_norm))
            prob_true = blended[np.arange(len(video_true_labels)), video_true_labels]
            return -np.mean(np.log(np.clip(prob_true, eps, 1.0)))

        res = minimize(nll_video_objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = np.array(res.x, dtype=np.float64)
        self.weights_video = list(opt_w / np.sum(opt_w))
        return self

    def fit(self, val_probabilities: List[np.ndarray], val_labels: np.ndarray) -> "WeightedSoftVotingEnsemble":
        return self.fit_window(val_probabilities, val_labels)

    def predict_proba_window(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        w = self.weights_window if self.weights_window is not None else self.weights
        if w is None:
            return np.mean(model_probabilities, axis=0)
        w_arr = np.array(w, dtype=np.float64)
        w_norm = w_arr / np.sum(w_arr)
        return sum(p * weight for p, weight in zip(model_probabilities, w_norm))

    def predict_window(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        avg_probs = self.predict_proba_window(model_probabilities)
        return np.argmax(avg_probs, axis=1)

    def predict_proba(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        return self.predict_proba_window(model_probabilities)

    def predict(self, model_probabilities: List[np.ndarray]) -> np.ndarray:
        return self.predict_window(model_probabilities)

    def predict_video(
        self,
        model_probabilities: List[np.ndarray],
        y_trues: np.ndarray,
        video_ids: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Evaluates at Video level using weights_video.
        First aggregates each model's test windows to video level, then blends with weights_video.
        """
        unique_vids = []
        vid_to_indices = {}
        for idx, v in enumerate(video_ids):
            if v not in vid_to_indices:
                unique_vids.append(v)
                vid_to_indices[v] = []
            vid_to_indices[v].append(idx)

        M = len(model_probabilities)
        w = self.weights_video if self.weights_video is not None else self.weights_window
        if w is None:
            w = [1.0 / M] * M
        w_norm = np.array(w, dtype=np.float64) / np.sum(w)

        # Video-level probabilities per model
        video_probs_per_model = []
        y_video_true = []
        for v in unique_vids:
            indices = vid_to_indices[v]
            y_video_true.append(int(y_trues[indices[0]]))

        for m_idx in range(M):
            m_probs = model_probabilities[m_idx]
            v_probs_m = [np.mean(m_probs[vid_to_indices[v]], axis=0) for v in unique_vids]
            video_probs_per_model.append(np.array(v_probs_m, dtype=np.float64))

        final_video_probs = sum(p * weight for p, weight in zip(video_probs_per_model, w_norm))
        y_video_pred = np.argmax(final_video_probs, axis=1)
        y_video_true = np.array(y_video_true, dtype=np.int64)

        from src.training.metrics import compute_metrics
        metrics = compute_metrics(y_video_true, y_video_pred)
        return y_video_true, y_video_pred, final_video_probs.astype(np.float32), metrics

def aggregate_video_level_predictions(
    y_probs: np.ndarray,
    y_trues: np.ndarray,
    video_ids: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Aggregates window-level prediction probabilities into video-level decisions.
    Uses Soft Probability Average Pooling: P_video = (1/K) * sum_{k=1}^K P_window_k
    Returns:
      - y_video_true: true class per video (N_videos,)
      - y_video_pred: predicted class per video (N_videos,)
      - y_video_prob: predicted probability distribution per video (N_videos, num_classes)
      - metrics: video-level accuracy, macro F1, weighted F1
    """
    from src.training.metrics import compute_metrics
    unique_vids = []
    vid_to_indices = {}
    for idx, v in enumerate(video_ids):
        if v not in vid_to_indices:
            unique_vids.append(v)
            vid_to_indices[v] = []
        vid_to_indices[v].append(idx)

    y_video_true = []
    y_video_pred = []
    y_video_prob = []

    for v in unique_vids:
        indices = vid_to_indices[v]
        v_probs = np.mean(y_probs[indices], axis=0)
        v_pred = int(np.argmax(v_probs))
        v_true = int(y_trues[indices[0]])

        y_video_true.append(v_true)
        y_video_pred.append(v_pred)
        y_video_prob.append(v_probs)

    y_video_true = np.array(y_video_true, dtype=np.int64)
    y_video_pred = np.array(y_video_pred, dtype=np.int64)
    y_video_prob = np.array(y_video_prob, dtype=np.float32)

    metrics = compute_metrics(y_video_true, y_video_pred)
    return y_video_true, y_video_pred, y_video_prob, metrics


