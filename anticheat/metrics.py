"""Detection metrics, in numpy so the pipeline needs nothing beyond torch and pandas.

An anti-cheat system is an intrusion detector, not a classifier competition, so the
headline numbers here are false-positive rate at a useful recall and precision at a
realistic prevalence rather than accuracy on a balanced test set.
"""

from __future__ import annotations

import numpy as np


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so that tied scores cannot inflate the AUC."""

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1

    return ranks


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    r = _ranks(np.asarray(scores, dtype=np.float64))
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, threshold) with one point per distinct score."""

    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    order = np.argsort(-scores, kind="mergesort")
    y, scores = y[order], scores[order]

    # Collapse runs of equal scores; splitting them would invent operating points
    # no threshold can actually reach.
    distinct = np.flatnonzero(np.diff(scores))
    cut = np.r_[distinct, len(y) - 1]

    tp = np.cumsum(y)[cut]
    fp = np.cumsum(1.0 - y)[cut]

    n_pos, n_neg = tp[-1], fp[-1]
    tpr = tp / n_pos if n_pos else np.zeros_like(tp)
    fpr = fp / n_neg if n_neg else np.zeros_like(fp)

    return np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, scores[cut]]


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under precision-recall, the honest summary when positives are rare."""

    y = np.asarray(y, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = y[order]

    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos

    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fpr_at_recall(y: np.ndarray, scores: np.ndarray, target: float) -> dict:
    """Cheapest false-positive rate that still catches `target` of the cheating."""

    fpr, tpr, thr = roc_curve(y, scores)
    ok = np.flatnonzero(tpr >= target)
    if len(ok) == 0:
        return {"fpr": float("nan"), "tpr": float("nan"), "threshold": float("nan")}

    i = ok[int(np.argmin(fpr[ok]))]
    return {"fpr": float(fpr[i]), "tpr": float(tpr[i]), "threshold": float(thr[i])}


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, target: float) -> dict:
    """How much cheating is caught if the alarm budget is fixed."""

    fpr, tpr, thr = roc_curve(y, scores)
    ok = np.flatnonzero(fpr <= target)
    if len(ok) == 0:
        return {"fpr": float("nan"), "tpr": float("nan"), "threshold": float("nan")}

    i = ok[int(np.argmax(tpr[ok]))]
    return {"fpr": float(fpr[i]), "tpr": float(tpr[i]), "threshold": float(thr[i])}


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Base-rate correction.

    A test set balanced at 20% cheaters flatters precision. This re-states it for a
    population where cheaters are as rare as they actually are.
    """

    hits = prevalence * tpr
    alarms = hits + (1.0 - prevalence) * fpr
    return float(hits / alarms) if alarms > 0 else float("nan")


def confusion(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=np.float64)
    pred = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.float64)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": (tp + tn) / len(y) if len(y) else float("nan"),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": 1.0 - specificity if specificity == specificity else float("nan"),
        "f1": f1,
    }
