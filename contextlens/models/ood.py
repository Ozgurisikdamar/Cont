"""Out-of-taxonomy detectors (docs/HARDENING.md item 3, decisions.md D-34).

Every detector returns an **in-domain score**: higher = more like the training
topics. A text is flagged when its score is below the detector's threshold.

  centroid     max cosine similarity to the general-topic centroids (v1.0 gate)
  msp          maximum calibrated softmax probability (Hendrycks & Gimpel, 2017)
  energy       T * logsumexp(logits / T) (Liu et al., 2020)
  mahalanobis  minus the smallest Mahalanobis distance to a class mean, shared
               shrunk covariance (Lee et al., 2018)
  knn          mean cosine similarity to the k nearest training embeddings
               (Sun et al., 2022)
  binary       logistic regression "in-domain vs off-topic" on embeddings
  other_class  a 9-way softmax with an extra "other" class; score = 1 - P(other)

``binary`` and ``other_class`` need off-topic training texts (CLINC150 train,
scripts/build_ood_conversational.py); the others use in-domain data only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

from contextlens.models.heads import fit_softmax

DETECTORS = ("centroid", "msp", "energy", "mahalanobis", "knn", "binary", "other_class")


@dataclass
class OODDetector:
    """Fitted detector. ``state`` holds the arrays / models the method needs."""

    method: str
    state: dict[str, Any] = field(default_factory=dict)

    def score(
        self, X: np.ndarray, general_probs: np.ndarray, logits: np.ndarray | None, temperature: float
    ) -> np.ndarray:
        m, s = self.method, self.state
        if m == "centroid":
            return (X @ s["centroids"].T).max(axis=1)
        if m == "msp":
            return general_probs.max(axis=1)
        if m == "energy":
            if logits is None:
                raise ValueError("energy needs logits")
            finite = np.where(np.isfinite(logits), logits, -np.inf)
            return temperature * logsumexp(finite / temperature, axis=1)
        if m == "mahalanobis":
            dist = [np.einsum("nd,nd->n", (X - mu) @ s["precision"], X - mu) for mu in s["means"]]
            return -np.min(np.stack(dist, axis=1), axis=1)
        if m == "knn":
            sims = X @ s["bank"].T.astype(np.float32)
            k = int(s["k"])
            top = np.partition(sims, -k, axis=1)[:, -k:]
            return top.mean(axis=1)
        if m == "binary":
            return s["clf"].predict_proba(X)[:, 1]
        if m == "other_class":
            p = s["clf"].predict_proba(X)
            other = list(s["clf"].classes_).index(s["other_label"])
            return 1.0 - p[:, other]
        raise ValueError(f"unknown OOD method {m!r}")


def fit_detector(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    X_offtopic: np.ndarray | None = None,
    k: int = 10,
    bank_size: int = 8000,
    seed: int = 42,
) -> OODDetector:
    """Fit ``method`` on training embeddings (L2-normalised) and general labels."""
    if method not in DETECTORS:
        raise ValueError(f"unknown OOD method {method!r}")
    if method in ("msp", "energy"):
        return OODDetector(method)
    if method == "centroid":
        cents = np.stack([X_train[y_train == g].mean(axis=0) for g in range(n_classes)])
        return OODDetector(method, {"centroids": cents / np.linalg.norm(cents, axis=1, keepdims=True)})
    if method == "mahalanobis":
        means = np.stack([X_train[y_train == g].mean(axis=0) for g in range(n_classes)])
        centred = X_train - means[y_train]
        cov = LedoitWolf().fit(centred)
        return OODDetector(method, {"means": means, "precision": cov.precision_})
    if method == "knn":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=min(bank_size, len(X_train)), replace=False)
        return OODDetector(method, {"bank": X_train[np.sort(idx)].astype(np.float16), "k": k})
    if X_offtopic is None:
        raise ValueError(f"{method} needs off-topic training texts")
    if method == "binary":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=min(len(X_train), 3 * len(X_offtopic)), replace=False)
        X = np.vstack([X_train[idx], X_offtopic])
        y = np.concatenate([np.ones(len(idx), dtype=int), np.zeros(len(X_offtopic), dtype=int)])
        return OODDetector(method, {"clf": fit_softmax(X, y, 4.0)})
    if method == "other_class":
        other = n_classes
        X = np.vstack([X_train, X_offtopic])
        y = np.concatenate([y_train, np.full(len(X_offtopic), other)])
        return OODDetector(method, {"clf": fit_softmax(X, y, 4.0), "other_label": other})
    raise ValueError(f"unknown OOD method {method!r}")
