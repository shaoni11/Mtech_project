"""Small metric helpers without a scikit-learn dependency."""

from __future__ import annotations

import math


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def binary_confusion(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_pred = [int(score >= threshold) for score in y_score]
    tp = sum(true == 1 and pred == 1 for true, pred in zip(y_true, y_pred))
    tn = sum(true == 0 and pred == 0 for true, pred in zip(y_true, y_pred))
    fp = sum(true == 0 and pred == 1 for true, pred in zip(y_true, y_pred))
    fn = sum(true == 1 and pred == 0 for true, pred in zip(y_true, y_pred))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    """Compute ROC-AUC using average ranks for ties."""
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        positives_in_tie = sum(label for _, label in pairs[i:j])
        rank_sum_pos += positives_in_tie * avg_rank
        rank += j - i
        i = j

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pr_auc_score(y_true: list[int], y_score: list[float]) -> float:
    """Compute average precision, a standard PR-AUC summary for imbalanced tasks."""
    n_pos = sum(y_true)
    if n_pos == 0:
        return float("nan")

    ordered = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    precision_sum = 0.0
    for _, label in ordered:
        if label == 1:
            tp += 1
            precision_sum += tp / max(1, tp + fp)
        else:
            fp += 1
    return precision_sum / n_pos


def binary_classification_metrics(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    metrics = binary_confusion(y_true, y_score)
    metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    metrics["pr_auc"] = pr_auc_score(y_true, y_score)
    metrics["n_examples"] = float(len(y_true))
    metrics["n_active"] = float(sum(y_true))
    metrics["n_inactive"] = float(len(y_true) - sum(y_true))
    return metrics


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    if not y_true:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    ss_res = sum(error * error for error in errors)
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "r2": r2}

