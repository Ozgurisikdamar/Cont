"""Classifier heads shared by experiments and the production model.

* General topic: multinomial logistic regression (softmax) + temperature scaling.
* Subtopics: multi-label one-vs-rest logistic regression (sigmoid per label),
  either *hierarchical* (one head per general topic, trained only on that
  topic's rows, giving P(subtopic | general)) or *flat* (one head over all
  subtopics).
* Alternative to both: :class:`FlatSubtopicSoftmax`, one softmax over the 28
  subtopics (trained on each passage's primary subtopic); P(general) is the sum
  of its children's probabilities and P(subtopic | general) their share.
The choice is made experimentally (docs/EXPERIMENTS.md, E-7).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, log_softmax, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from contextlens.config import RANDOM_SEED

MAX_ITER = 2000


def fit_softmax(X: np.ndarray, y: np.ndarray, C: float, class_weight: str | None = "balanced") -> LogisticRegression:
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, class_weight=class_weight, random_state=RANDOM_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)
    return clf


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T minimising validation NLL of softmax(logits / T) (Guo et al., 2017)."""

    def nll(t: float) -> float:
        return float(-log_softmax(logits / t, axis=1)[np.arange(len(y)), y].mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def calibrated_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(logits / temperature, axis=1)


def grouped_probs(
    logits: np.ndarray, temperature: float, parent_col: np.ndarray, n_general: int
) -> tuple[np.ndarray, np.ndarray]:
    """From subtopic logits: (P(general), P(subtopic | general)).

    P(general) sums the softmax probabilities of the general topic's subtopics;
    the conditional is each subtopic's share of its parent's mass.
    """
    sub = softmax(logits / temperature, axis=1)
    general = np.stack([sub[:, parent_col == g].sum(axis=1) for g in range(n_general)], axis=1)
    conditional = sub / np.maximum(general[:, parent_col], 1e-12)
    return general, conditional


def fit_grouped_temperature(logits: np.ndarray, y_general: np.ndarray, parent_col: np.ndarray) -> float:
    """Temperature minimising the validation NLL of the *general* topic under grouped_probs."""
    n_general = int(parent_col.max()) + 1

    def nll(t: float) -> float:
        general, _ = grouped_probs(logits, t, parent_col, n_general)
        return float(-np.log(np.maximum(general[np.arange(len(y_general)), y_general], 1e-12)).mean())

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


@dataclass
class FlatSubtopicSoftmax:
    """Multinomial logistic regression over all subtopics (the primary label of each passage)."""

    n_subtopics: int
    C: float = 1.0
    clf: LogisticRegression | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> FlatSubtopicSoftmax:
        # Primary subtopic = first positive column; every positive column of a
        # passage belongs to its general topic (checked in tests/test_leakage.py).
        self.clf = fit_softmax(X, Y.argmax(axis=1), self.C)
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Logits for every subtopic column; subtopics never seen in training get -inf."""
        if self.clf is None:
            raise RuntimeError("FlatSubtopicSoftmax is not fitted")
        out = np.full((X.shape[0], self.n_subtopics), -np.inf)
        out[:, self.clf.classes_] = self.clf.decision_function(X)
        return out


@dataclass
class MultiLabelHead:
    """One sigmoid logistic regression per label (labels with no positives are skipped)."""

    labels: list[str]
    C: float = 1.0
    models: dict[str, LogisticRegression] = field(default_factory=dict)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> MultiLabelHead:
        for j, label in enumerate(self.labels):
            y = Y[:, j]
            if 0 < y.sum() < len(y):
                self.models[label] = fit_softmax(X, y, self.C, class_weight="balanced")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.labels)), dtype=np.float64)
        for j, label in enumerate(self.labels):
            m = self.models.get(label)
            if m is not None:
                out[:, j] = expit(m.decision_function(X))
        return out


@dataclass
class HierarchicalSubtopics:
    """P(subtopic | general) from one multi-label head per general topic."""

    children: dict[str, list[str]]  # general id -> subtopic ids
    C: float = 1.0
    heads: dict[str, MultiLabelHead] = field(default_factory=dict)

    def fit(
        self, X: np.ndarray, general: np.ndarray, Y: np.ndarray, sub_index: dict[str, int], general_ids: list[str]
    ) -> HierarchicalSubtopics:
        for gi, gid in enumerate(general_ids):
            rows = general == gi
            cols = [sub_index[s] for s in self.children[gid]]
            self.heads[gid] = MultiLabelHead(self.children[gid], self.C).fit(X[rows], Y[rows][:, cols])
        return self

    def conditional(self, X: np.ndarray, sub_index: dict[str, int], n_subs: int) -> np.ndarray:
        """Matrix of P(s | parent(s)) for every subtopic column."""
        out = np.zeros((X.shape[0], n_subs))
        for gid, head in self.heads.items():
            cols = [sub_index[s] for s in self.children[gid]]
            out[:, cols] = head.predict_proba(X)
        return out


def joint_subtopic_probs(general_probs: np.ndarray, conditional: np.ndarray, parent_col: np.ndarray) -> np.ndarray:
    """P(s) = P(parent(s)) * P(s | parent(s)); ``parent_col[j]`` = general index of subtopic j."""
    return general_probs[:, parent_col] * conditional


def decide_subtopics(
    probs: np.ndarray, parent_col: np.ndarray, general_pred: np.ndarray, threshold: float
) -> np.ndarray:
    """Multi-label decision with hierarchy validation.

    Only children of the predicted general topic are eligible; the best child is
    always returned (every in-taxonomy text has at least one subtopic) and
    further children are added when their score reaches ``threshold``.
    """
    out = np.zeros_like(probs, dtype=int)
    for i in range(probs.shape[0]):
        cols = np.where(parent_col == general_pred[i])[0]
        scores = probs[i, cols]
        out[i, cols[int(np.argmax(scores))]] = 1
        out[i, cols[scores >= threshold]] = 1
    return out
