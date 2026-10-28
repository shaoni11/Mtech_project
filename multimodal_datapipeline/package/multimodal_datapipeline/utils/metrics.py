"""NumPy-only evaluation metrics shared by the baseline workflows."""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    loss = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(np.mean(loss))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks where tied values share their average rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def roc_auc_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(probs, dtype=np.float64))
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    return float((pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Step-wise area under the precision-recall curve (same definition as sklearn)."""
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-np.asarray(probs, dtype=np.float64), kind="mergesort")
    sorted_probs = np.asarray(probs)[order]
    sorted_true = np.asarray(y_true)[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    # Evaluate only at the last position of each tied-score block.
    threshold_ends = np.r_[np.where(np.diff(sorted_probs))[0], len(sorted_probs) - 1]
    tp = tp[threshold_ends]
    fp = fp[threshold_ends]
    precision = tp / (tp + fp)
    recall = tp / positives
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def binary_classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(np.int32)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = (probs >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
