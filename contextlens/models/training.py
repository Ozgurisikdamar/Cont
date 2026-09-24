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
    HierarchicalSubtopics,
    calibrated_softmax,
    decide_subtopics,
    fit_softmax,
    fit_temperature,
)
from contextlens.models.topic_model import TopicModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    encoder: str
    general_C: float
    subtopic_C: float
    thresholds: tuple[float, ...] = tuple(round(x, 2) for x in np.arange(0.20, 0.71, 0.05))
    ood_keep_quantile: float = 0.05  # 5% of in-distribution calibration texts fall below the gate
    min_confidence: float = 0.40
    vocabulary_size: int = 40000


def centroids_of(X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    cents = np.stack([X[y == g].mean(axis=0) for g in range(n_classes)])
    return cents / np.linalg.norm(cents, axis=1, keepdims=True)


def build_vocabulary(texts: list[str], size: int) -> dict[str, float]:
    """IDF of content words in the training corpus (used for query keywords)."""
    vec = TfidfVectorizer(stop_words="english", min_df=3, max_features=size, token_pattern=r"(?u)\b[a-z][a-z\-]+\b")  # noqa: S106
    vec.fit(texts)
    return {w: round(float(idf), 4) for w, idf in zip(vec.get_feature_names_out(), vec.idf_, strict=True)}


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

    general = fit_softmax(Xtr, ytr, cfg.general_C)
    temperature = fit_temperature(general.decision_function(Xva), yva)
    gp_val = calibrated_softmax(general.decision_function(Xva), temperature)

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
        general_head=general,
        temperature=temperature,
        subtopic_heads=subs,
        subtopic_threshold=float(threshold),
        centroids=cents,
        ood_threshold=ood_threshold,
        min_confidence=cfg.min_confidence,
        encoder=encoder,
    )
    val_report = {
        "general": multiclass_report(yva, gp_val, space.general_ids),
        "subtopic_macro_f1": best_macro,
        "subtopic_threshold_sweep": [{"threshold": t, "macro_f1": round(m, 4)} for m, _, t in sweep],
        "temperature": temperature,
        "ood_threshold": ood_threshold,
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
