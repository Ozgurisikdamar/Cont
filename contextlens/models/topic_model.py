"""Production topic model: featurizer + calibrated general head + subtopic heads + OOD gate.

Prediction flow for one text::

    normalise -> features -> P(general) (temperature-scaled softmax)
              -> P(subtopic | general) (hierarchical heads) -> subtopics of the
                 predicted general whose probability passes the tuned threshold
              -> OOD score (cosine similarity to class centroids) -> uncertain flag

Artifacts are written by ``train.py`` and loaded with integrity checks (see
:mod:`contextlens.models.artifact`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contextlens.models.heads import HierarchicalSubtopics, calibrated_softmax
from contextlens.preprocessing.text import is_informative, normalize

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubtopicScore:
    id: str
    probability: float  # P(subtopic | predicted general)


@dataclass(frozen=True)
class Prediction:
    text: str
    status: str  # "ok" | "uncertain" | "uninformative"
    general: str | None
    confidence: float  # calibrated P(general)
    general_probs: dict[str, float]
    subtopics: tuple[SubtopicScore, ...]
    subtopic_probs: dict[str, float]  # joint P(subtopic) = P(general) * P(sub | general)
    ood_score: float
    reasons: tuple[str, ...] = ()

    @property
    def uncertain(self) -> bool:
        return self.status != "ok"


@dataclass
class TopicModel:
    general_ids: list[str]
    subtopic_ids: list[str]
    parent_col: np.ndarray
    general_head: Any  # sklearn LogisticRegression
    temperature: float
    subtopic_heads: HierarchicalSubtopics
    subtopic_threshold: float
    centroids: np.ndarray  # L2-normalised class centroids in embedding space
    ood_threshold: float
    min_confidence: float
    encoder: Any  # object with .encode(list[str]) -> np.ndarray
    metadata: dict = field(default_factory=dict)
    max_subtopics: int = 3  # at most this many subtopics are reported per text

    @property
    def sub_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.subtopic_ids)}

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch probabilities: (P(general), P(sub | general), ood_scores). Texts must be normalised."""
        X = self.encoder.encode(texts)
        general = calibrated_softmax(self.general_head.decision_function(X), self.temperature)
        conditional = self.subtopic_heads.conditional(X, self.sub_index, len(self.subtopic_ids))
        ood = (X @ self.centroids.T).max(axis=1)
        return general, conditional, ood

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        cleaned = [normalize(t) for t in texts]
        informative = [i for i, t in enumerate(cleaned) if is_informative(t)]
        results: list[Prediction | None] = [None] * len(texts)
        for i, t in enumerate(cleaned):
            if i not in informative:
                results[i] = Prediction(t, "uninformative", None, 0.0, {}, (), {}, 0.0, ("no content words",))
        if informative:
            t0 = time.perf_counter()
            general, conditional, ood = self.predict_proba([cleaned[i] for i in informative])
            log.debug("inference for %d texts took %.1f ms", len(informative), (time.perf_counter() - t0) * 1000)
            for row, i in enumerate(informative):
                results[i] = self._decide(cleaned[i], general[row], conditional[row], float(ood[row]))
        return [r for r in results if r is not None]

    def _decide(self, text: str, gp: np.ndarray, cond: np.ndarray, ood: float) -> Prediction:
        g = int(np.argmax(gp))
        cols = np.where(self.parent_col == g)[0]
        order = cols[np.argsort(-cond[cols])]
        chosen = [order[0], *[c for c in order[1:] if cond[c] >= self.subtopic_threshold]][: max(1, self.max_subtopics)]
        reasons = []
        if ood < self.ood_threshold:
            reasons.append("far from all training topics (possible out-of-taxonomy input)")
        if gp[g] < self.min_confidence:
            reasons.append(f"low confidence ({gp[g]:.0%})")
        joint = gp[self.parent_col] * cond
        return Prediction(
            text=text,
            status="uncertain" if reasons else "ok",
            general=self.general_ids[g],
            confidence=float(gp[g]),
            general_probs={gid: float(p) for gid, p in zip(self.general_ids, gp, strict=True)},
            subtopics=tuple(SubtopicScore(self.subtopic_ids[c], float(cond[c])) for c in chosen),
            subtopic_probs={sid: float(p) for sid, p in zip(self.subtopic_ids, joint, strict=True)},
            ood_score=ood,
            reasons=tuple(reasons),
        )
