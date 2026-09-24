"""Data-integrity checks on the committed corpus (skipped when the data is not built)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from contextlens.config import PATHS

PASSAGES = PATHS.data_processed / "passages.parquet"
CASES = json.loads((Path(__file__).parent / "acceptance_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    if not PASSAGES.exists():
        pytest.skip("corpus not built (python scripts/download_data.py --corpus)")
    return pd.read_parquet(PASSAGES)


def test_acceptance_sentences_are_not_training_data(corpus):
    texts = corpus.text.str.lower()
    probes = [c["text"] for c in CASES["single"] + CASES["ood"]] + CASES["conversation"]["messages"]
    for probe in probes:
        assert not texts.str.contains(probe.lower(), regex=False).any(), probe


def test_splits_are_disjoint_by_article(corpus):
    by_split = {s: set(corpus[corpus.split == s].wiki_id) for s in ("train", "val", "test")}
    assert not by_split["train"] & by_split["val"]
    assert not by_split["train"] & by_split["test"]
    assert not by_split["val"] & by_split["test"]


def test_every_class_is_in_every_split(corpus):
    for split in ("train", "val", "test"):
        part = corpus[corpus.split == split]
        assert part.general.nunique() == 8
        assert part.subtopics.str.split("|").explode().nunique() == 28


def test_no_exact_duplicate_passages(corpus):
    assert not corpus.text.str.lower().duplicated().any()


def test_labels_respect_the_hierarchy(corpus, taxonomy):
    for general, subs in corpus[["general", "subtopics"]].drop_duplicates().itertuples(index=False):
        assert taxonomy.is_consistent(general, subs.split("|"))
