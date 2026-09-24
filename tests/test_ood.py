"""Out-of-taxonomy detectors (contextlens.models.ood, decisions.md D-34)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from contextlens.data.dataset import LabelSpace
from contextlens.models.artifact import load_artifact, save_artifact
from contextlens.models.ood import DETECTORS, OODDetector, fit_detector

from .conftest import FakeEncoder, tiny_corpus

CHAT = [
    "set an alarm for seven tomorrow morning",
    "what is my bank account balance",
    "book a table for two tonight",
    "turn up the volume please",
    "remind me to call mom",
    "how is the traffic on my commute",
]
IN = ["gene dna chromosome allele heredity", "football goal striker league club", "poem poetry verse rhyme stanza"]


@pytest.fixture(scope="module")
def train_data(taxonomy):
    space = LabelSpace.from_taxonomy(taxonomy)
    texts, generals, _ = tiny_corpus(0, 12)
    enc = FakeEncoder()
    return enc, enc.encode(texts), space.encode_general(pd.Series(generals)), len(space.general_ids)


@pytest.mark.parametrize("method", DETECTORS)
def test_detector_ranks_topical_text_above_chat(method, train_data, fake_model):
    enc, X, y, n = train_data
    off = enc.encode([f"{c} {w}" for c in CHAT for w in ("please", "now", "today")])
    det = fit_detector(method, X, y, n, off, k=3)
    Xi, Xo = enc.encode(IN), enc.encode(CHAT)
    gi, _, li = fake_model.head_outputs(Xi)
    go, _, lo = fake_model.head_outputs(Xo)
    si = det.score(Xi, gi, li, fake_model.temperature)
    so = det.score(Xo, go, lo, fake_model.temperature)
    assert si.shape == (3,) and so.shape == (6,)
    assert np.all(np.isfinite(si)) and np.all(np.isfinite(so))
    if method not in ("msp", "energy"):  # probability-based scores are not reliable on a 4k-dim toy encoder
        assert si.mean() > so.mean()


def test_detectors_that_learn_from_offtopic_text_need_it(train_data):
    _, X, y, n = train_data
    for method in ("binary", "other_class"):
        with pytest.raises(ValueError, match="off-topic"):
            fit_detector(method, X, y, n, None)
    with pytest.raises(ValueError, match="unknown"):
        fit_detector("nope", X, y, n)
    with pytest.raises(ValueError, match="logits"):
        OODDetector("energy").score(X[:1], np.ones((1, n)) / n, None, 1.0)


@pytest.mark.parametrize("method", ["mahalanobis", "knn", "binary", "other_class"])
def test_detector_round_trips_through_the_artifact(method, train_data, fake_model, tmp_path):
    enc, X, y, n = train_data
    off = enc.encode(CHAT)
    model = dataclasses.replace(fake_model, ood_detector=fit_detector(method, X, y, n, off, k=3))
    save_artifact(model, tmp_path / "m", {}, save_encoder=False)
    loaded = load_artifact(tmp_path / "m", encoder=FakeEncoder())
    assert loaded.ood_detector is not None and loaded.ood_detector.method == method
    assert loaded.metadata["ood_method"] == method
    texts = IN + CHAT
    np.testing.assert_allclose(model.predict_proba(texts)[2], loaded.predict_proba(texts)[2], rtol=1e-5)
