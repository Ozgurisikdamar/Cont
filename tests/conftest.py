"""Shared fixtures.

``FakeEncoder`` is a deterministic bag-of-words hashing encoder, so the whole
pipeline (training, artifact round-trip, conversation, database) can be tested
offline in a few seconds without downloading a sentence-transformer.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from contextlens.config import load_settings
from contextlens.data.dataset import LabelSpace
from contextlens.models.training import TrainConfig, fit_topic_model
from contextlens.taxonomy import load_taxonomy

DIM = 4096
TOKEN = re.compile(r"[a-z]+")


class FakeEncoder:
    key = "fake"
    dim = DIM

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in TOKEN.findall(t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % DIM] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def save(self, path: Path) -> None:  # the fake encoder has no weights
        path.mkdir(parents=True, exist_ok=True)


# Small hand-written training set: 3 sentences per subtopic (vocabulary chosen
# so that each subtopic is separable by the bag-of-words fake encoder).
VOCAB = {
    "quantum_mechanics": "particle wave function entanglement superposition quantum state measurement",
    "relativity": "spacetime relativity einstein gravity time dilation lorentz",
    "astrophysics": "star galaxy supernova black hole nebula stellar cosmic",
    "classical_mechanics": "newton force momentum pendulum friction velocity acceleration",
    "genetics": "gene dna chromosome allele heredity mutation genome",
    "evolution": "evolution natural selection species darwin fossil ancestor adaptation",
    "cell_biology": "cell membrane mitochondria organelle cytoplasm nucleus protein",
    "ecology": "ecosystem habitat food web predator population biodiversity niche",
    "organic_chemistry": "carbon organic compound alkane benzene functional group ester",
    "chemical_reactions": "reaction catalyst oxidation reduction reagent yield equilibrium",
    "periodic_table": "element periodic table atomic number noble gas halogen metal",
    "quantum_computing": "qubit quantum processor gate algorithm quantum computer circuit",
    "artificial_intelligence": "neural network machine learning model training artificial intelligence",
    "software": "software code program developer compiler library bug",
    "hardware": "cpu motherboard memory chip transistor hardware disk",
    "scientific_method": "hypothesis experiment observation test falsifiable method evidence",
    "history_of_science": "history science galileo copernicus revolution natural philosophy era",
    "scientific_research": "research paper peer review journal funding laboratory study",
    "novels": "novel plot character narrator chapter fiction story",
    "science_books": "popular science book explains readers universe bestseller",
    "poetry": "poem poetry verse rhyme stanza poet sonnet",
    "authors": "author writer wrote biography literary career published",
    "football": "football goal striker league club match penalty",
    "basketball": "basketball dunk hoop nba court rebound point guard",
    "olympics": "olympic games medal athletes olympics gold torch",
    "ottoman_history": "ottoman sultan empire istanbul pasha janissary",
    "world_wars": "war soldiers battle allied axis trench army",
    "ancient_history": "ancient rome greece pharaoh empire antiquity temple",
}


def tiny_corpus(seed: int, n: int) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    tax = load_taxonomy()
    texts, generals, subs = [], [], []
    for sid, words in VOCAB.items():
        w = words.split()
        for _ in range(n):
            texts.append(" ".join(rng.choice(w, size=6)))
            generals.append(tax.parent_of(sid))
            subs.append(sid)
    return texts, generals, subs


@pytest.fixture(scope="session")
def taxonomy():
    return load_taxonomy()


@pytest.fixture(scope="session", params=["hierarchical", "flat_softmax"])
def fake_model(taxonomy, request):
    """A tiny trained TopicModel; every test using it runs once per head type."""
    import pandas as pd

    space = LabelSpace.from_taxonomy(taxonomy)
    children = {g: taxonomy.children_of(g) for g in taxonomy.general_ids}
    tr = tiny_corpus(0, 12)
    va = tiny_corpus(1, 4)

    def pack(c):
        return (c[0], space.encode_general(pd.Series(c[1])), space.encode_subtopics(pd.Series(c[2])))

    cfg = TrainConfig(
        encoder="fake",
        general_C=8.0,
        subtopic_C=8.0,
        min_confidence=0.2,
        ood_keep_quantile=0.02,
        head_type=request.param,
    )
    model, _ = fit_topic_model(FakeEncoder(), space, children, pack(tr), pack(va), cfg)
    model.metadata = {
        "model_name": "contextlens-topic",
        "model_version": "test",
        "trained_at": "2026-01-01",
        "dataset_version": "tiny",
        "encoder": "fake",
    }
    return model


@pytest.fixture()
def settings(tmp_path):
    return load_settings(db_path=tmp_path / "test.db", web_enabled=False)
