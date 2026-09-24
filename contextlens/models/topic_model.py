"""Production topic model: featurizer + calibrated general head + subtopic heads + OOD gate.

Prediction flow for one text::

    normalise -> features -> P(general), P(subtopic | general)
                 (head_type "hierarchical": temperature-scaled general softmax +
                 one multi-label head per general topic; "flat_softmax": one
                 temperature-scaled softmax over the 28 subtopics, P(general) =
                 sum of its subtopics) -> subtopics of the predicted general
                 whose probability passes the tuned threshold
              -> OOD score (cosine similarity to class centroids) -> uncertain flag

Artifacts are written by ``train.py`` and loaded with integrity checks (see
:mod:`contextlens.models.artifact`).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contextlens.models.heads import FlatSubtopicSoftmax, HierarchicalSubtopics, calibrated_softmax, grouped_probs
from contextlens.preprocessing.text import is_informative, normalize

log = logging.getLogger(__name__)

WORD = re.compile(r"[^\W\d_]+")


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
    general_head: Any  # LogisticRegression (hierarchical) or FlatSubtopicSoftmax (flat_softmax)
    temperature: float
    subtopic_heads: HierarchicalSubtopics | None  # None for head_type "flat_softmax"
    subtopic_threshold: float
    centroids: np.ndarray  # L2-normalised class centroids in embedding space
    ood_threshold: float
    min_confidence: float
    encoder: Any  # object with .encode(list[str]) -> np.ndarray
    metadata: dict = field(default_factory=dict)
    max_subtopics: int = 3  # at most this many subtopics are reported per text
    head_type: str = "hierarchical"  # "hierarchical" | "flat_softmax" (docs/EXPERIMENTS.md, E-7)
    # Language gate: English words known to the model (training vocabulary + stop
    # words). A text of >= 3 words of which fewer than this share are known is
    # answered "uncertain" (decisions.md D-29). Empty set = gate off.
    known_words: frozenset[str] = frozenset()
    min_known_word_share: float = 0.4

    @property
    def sub_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.subtopic_ids)}

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))

    def predict_proba(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch probabilities: (P(general), P(sub | general), ood_scores). Texts must be normalised."""
        if not texts:
            n_gen, n_sub = len(self.general_ids), len(self.subtopic_ids)
            return np.zeros((0, n_gen)), np.zeros((0, n_sub)), np.zeros(0)
        X = self.encoder.encode(texts)
        if self.head_type == "flat_softmax":
            assert isinstance(self.general_head, FlatSubtopicSoftmax)
            general, conditional = grouped_probs(
                self.general_head.logits(X), self.temperature, self.parent_col, len(self.general_ids)
            )
        else:
            assert self.subtopic_heads is not None
            general = calibrated_softmax(self.general_head.decision_function(X), self.temperature)
            conditional = self.subtopic_heads.conditional(X, self.sub_index, len(self.subtopic_ids))
        ood = (X @ self.centroids.T).max(axis=1)
        return general, conditional, ood

    def looks_english(self, text: str) -> bool:
        """False when most words of a (normalised) text are unknown English words."""
        if not self.known_words:
            return True
        words = [w for w in WORD.findall(text.lower()) if len(w) >= 2]
        if len(words) < 3:
            return True
        return sum(w in self.known_words for w in words) / len(words) >= self.min_known_word_share

    def uncertain_mask(self, texts: list[str], general: np.ndarray, ood: np.ndarray) -> np.ndarray:
        """The deployed "uncertain" rule for a batch (used by evaluation and decay tuning)."""
        english = np.array([self.looks_english(t) for t in texts], dtype=bool)
        return (ood < self.ood_threshold) | (general.max(axis=1) < self.min_confidence) | ~english

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
        if not self.looks_english(text):
            reasons.append("does not look like English (the model covers English only)")
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
