"""Fit the production :class:`TopicModel` (used by ``train.py`` and the tests).

Every data-dependent choice uses held-out *development* data only: temperature
(NLL on validation), subtopic threshold (macro-F1 sweep on validation) and the
OOD threshold (quantile of in-distribution scores on the calibration texts -
validation passages, or real user questions from Stack Exchange ext_dev when
``ood_calibration_texts`` is given). Test data is never seen here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from contextlens.data.dataset import LabelSpace
from contextlens.evaluation.metrics import multiclass_report, multilabel_report
from contextlens.models.heads import (
    FlatSubtopicSoftmax,
    HierarchicalSubtopics,
    calibrated_softmax,
    decide_subtopics,
    fit_grouped_temperature,
    fit_softmax,
    fit_temperature,
    grouped_probs,
)
from contextlens.models.topic_model import TopicModel

log = logging.getLogger(__name__)


HEAD_TYPES = ("hierarchical", "flat_softmax")


@dataclass(frozen=True)
class TrainConfig:
    encoder: str
    general_C: float
    subtopic_C: float
    thresholds: tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.20, 0.71, 0.05))
    ood_keep_quantile: float = 0.05  # 5% of in-distribution calibration texts fall below the gate
    min_confidence: float = 0.40
    vocabulary_size: int = 40000
    head_type: str = "hierarchical"  # or "flat_softmax" (general_C is unused then)

    def __post_init__(self) -> None:
        if self.head_type not in HEAD_TYPES:
            raise ValueError(f"head_type must be one of {HEAD_TYPES}, got {self.head_type!r}")


def centroids_of(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    cents = np.stack([X[y == g].mean(axis=0) for g in range(n_classes)])
    return cents / np.linalg.norm(cents, axis=1, keepdims=True)


def build_vocabulary(texts: list[str], labels: np.ndarray, size: int) -> dict[str, float]:
    """Keyword weight of content words in the training corpus (used for search-query keywords).

    weight = IDF x topic concentration, where the concentration is the largest
    class-size-normalised share of the word's documents in one general topic
    (1/n_classes for a word spread evenly, 1.0 for a word of a single topic).
    IDF alone favours words that are rare in an encyclopedia but generic in
    conversation ("things", "carry"); the concentration keeps the topical ones.
    """
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=size, token_pattern=r"(?u)\b[a-z][a-z\-]+\b")  # noqa: S106
    X = vec.fit_transform(texts)
    present = (X > 0).astype(np.float64)
    n_classes = int(labels.max()) + 1
    # document frequency per class, normalised by class size -> P(class | word) up to a constant
    per_class = np.vstack(
        [
            np.asarray(present[labels == c].sum(axis=0)).ravel() / max(int((labels == c).sum()), 1)
            for c in range(n_classes)
        ]
    )
    concentration = per_class.max(axis=0) / np.maximum(per_class.sum(axis=0), 1e-12)
    weights = vec.idf_ * concentration
    return {w: round(float(x), 4) for w, x in zip(vec.get_feature_names_out(), weights, strict=True)}


def fit_topic_model(
    encoder: Any,
    space: LabelSpace,
    children: dict[str, list[str]],
    train: tuple[list[str], np.ndarray, np.ndarray],
    val: tuple[list[str], np.ndarray, np.ndarray],
    cfg: TrainConfig,
    ood_calibration_texts: list[str] | None = None,
) -> tuple[TopicModel, dict]:
    """Train on ``train`` = (texts, general labels, subtopic matrix); tune on ``val``.

    ``ood_calibration_texts`` - in-distribution texts whose score quantile sets the
    OOD threshold (default: the validation passages).
    """
    Xtr, Xva = encoder.encode(train[0]), encoder.encode(val[0])
    ytr, Ytr = train[1], train[2]
    yva, Yva = val[1], val[2]
    n_general = len(space.general_ids)

    general_head: Any
    subs: HierarchicalSubtopics | None
    if cfg.head_type == "flat_softmax":
        general_head = FlatSubtopicSoftmax(len(space.subtopic_ids), cfg.subtopic_C).fit(Xtr, Ytr)
        logits_val = general_head.logits(Xva)
        temperature = fit_grouped_temperature(logits_val, yva, space.parent_col)
        gp_val, cond_val = grouped_probs(logits_val, temperature, space.parent_col, n_general)
        subs = None
    else:
        general_head = fit_softmax(Xtr, ytr, cfg.general_C)
        temperature = fit_temperature(general_head.decision_function(Xva), yva)
        gp_val = calibrated_softmax(general_head.decision_function(Xva), temperature)
        subs = HierarchicalSubtopics(children, cfg.subtopic_C).fit(Xtr, ytr, Ytr, space.sub_index, space.general_ids)
        cond_val = subs.conditional(Xva, space.sub_index, len(space.subtopic_ids))
    sweep = []
    for thr in cfg.thresholds:
        dec = decide_subtopics(cond_val, space.parent_col, gp_val.argmax(1), thr)
        sweep.append((multilabel_report(Yva, dec, cond_val, space.subtopic_ids)["macro_f1"], -thr, thr))
    best_macro, _, threshold = max(sweep)

    cents = centroids_of(Xtr, ytr, n_general)
    Xcal = encoder.encode(ood_calibration_texts) if ood_calibration_texts else Xva
    ood_threshold = float(np.quantile((Xcal @ cents.T).max(axis=1), cfg.ood_keep_quantile))

    model = TopicModel(
        general_ids=space.general_ids,
        subtopic_ids=space.subtopic_ids,
        parent_col=space.parent_col,
        general_head=general_head,
        temperature=temperature,
        subtopic_heads=subs,
        subtopic_threshold=float(threshold),
        centroids=cents,
        ood_threshold=ood_threshold,
        min_confidence=cfg.min_confidence,
        encoder=encoder,
        head_type=cfg.head_type,
    )
    val_report = {
        "general": multiclass_report(yva, gp_val, space.general_ids),
        "subtopic_macro_f1": best_macro,
        "subtopic_threshold_sweep": [{"threshold": t, "macro_f1": round(m, 4)} for m, _, t in sweep],
        "temperature": temperature,
        "ood_threshold": ood_threshold,
        "head_type": cfg.head_type,
    }
    log.info(
        "val acc=%.4f macroF1=%.4f T=%.3f sub_thr=%.2f sub_macroF1=%.4f ood_thr=%.4f",
        val_report["general"]["accuracy"],
        val_report["general"]["macro_f1"],
        temperature,
        threshold,
        best_macro,
        ood_threshold,
    )
    return model, val_report
