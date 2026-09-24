"""Evaluation metrics: multi-class, multi-label, calibration and OOD detection."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins (Guo et al., 2017)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(mask.sum()),
                    "confidence": float(conf[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return out


def multiclass_report(y_true: np.ndarray, probs: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro/weighted P/R/F1, per-class metrics, confusion matrix, calibration."""
    y_pred = probs.argmax(axis=1)
    idx = list(range(len(labels)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=idx, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=idx, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=idx, average="weighted", zero_division=0)
    onehot = np.eye(len(labels))[y_true]
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "log_loss": float(log_loss(y_true, np.clip(probs, 1e-12, 1.0), labels=idx)),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "ece": expected_calibration_error(probs, y_true),
        "per_class": {
            labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in idx
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
        "labels": labels,
    }


def multilabel_report(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, labels: list[str]) -> dict:
    """Micro/macro F1, Hamming loss, subset accuracy, P@1 and R@3 for multi-label output."""
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0]
    p_at_1 = float(np.mean(y_true[np.arange(len(y_true)), top1]))
    top3 = order[:, :3]
    hits3 = np.array([y_true[i, top3[i]].sum() for i in range(len(y_true))])
    r_at_3 = float(np.mean(hits3 / np.maximum(y_true.sum(axis=1), 1)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    return {
        "n": int(len(y_true)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "subset_accuracy": float(np.mean((y_true == y_pred).all(axis=1))),
        "precision_at_1": p_at_1,
        "recall_at_3": r_at_3,
        "per_label": {
            labels[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i in range(len(labels))
        },
    }


def ood_report(in_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Scores are *in-distribution* scores (higher = more in-distribution)."""
    y = np.concatenate([np.ones(len(in_scores)), np.zeros(len(ood_scores))])
    s = np.concatenate([in_scores, ood_scores])
    thr95 = np.quantile(in_scores, 0.05)  # keeps 95% of in-distribution
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr_ood": float(average_precision_score(1 - y, -s)),
        "fpr_at_95_tpr": float(np.mean(ood_scores >= thr95)),
        "n_in": int(len(in_scores)),
        "n_ood": int(len(ood_scores)),
    }


def binary_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y_true, p))
