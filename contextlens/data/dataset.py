"""Load the processed corpus and the external evaluation sets as arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import logging

from contextlens.config import PATHS
from contextlens.preprocessing.text import normalize
from contextlens.taxonomy import Taxonomy, load_taxonomy

log = logging.getLogger(__name__)


class DataMissingError(FileNotFoundError):
    """Raised when processed data has not been built yet."""


@dataclass
class LabelSpace:
    general_ids: list[str]
    subtopic_ids: list[str]
    parent_col: np.ndarray  # general index of each subtopic column

    @classmethod
    def from_taxonomy(cls, tax: Taxonomy) -> LabelSpace:
        gids = tax.general_ids
        sids = tax.subtopic_ids
        return cls(gids, sids, np.array([gids.index(tax.parent_of(s)) for s in sids]))

    @property
    def sub_index(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.subtopic_ids)}

    def encode_general(self, values: pd.Series) -> np.ndarray:
        index = {g: i for i, g in enumerate(self.general_ids)}
        return values.map(index).to_numpy()

    def encode_subtopics(self, values: pd.Series) -> np.ndarray:
        index = self.sub_index
        Y = np.zeros((len(values), len(self.subtopic_ids)), dtype=int)
        for i, subs in enumerate(values):
            for s in (subs.split("|") if isinstance(subs, str) else subs):
                if s:
                    Y[i, index[s]] = 1
        return Y


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataMissingError(f"{path} not found - run: python scripts/download_data.py --all")
    return path


def load_passages() -> pd.DataFrame:
    return pd.read_parquet(_require(PATHS.data_processed / "passages.parquet"))


def load_ood_passages() -> pd.DataFrame:
    return pd.read_parquet(_require(PATHS.data_processed / "ood_passages.parquet"))


def load_jsonl(path: Path) -> pd.DataFrame:
    with open(_require(path), encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh])


def load_se_general() -> pd.DataFrame:
    return load_jsonl(PATHS.data_external / "se_general_eval.jsonl")


def load_se_subtopic() -> pd.DataFrame:
    return load_jsonl(PATHS.data_external / "se_subtopic_eval.jsonl")


class BenchmarkData:
    """All splits as normalised texts + encoded labels (used by the benchmark scripts)."""

    def __init__(self) -> None:
        tax = load_taxonomy()
        self.tax = tax
        self.space = LabelSpace.from_taxonomy(tax)
        df = load_passages()
        self.splits = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
        self.text = {s: [normalize(t) for t in d.text] for s, d in self.splits.items()}
        self.yg = {s: self.space.encode_general(d.general) for s, d in self.splits.items()}
        self.ys = {s: self.space.encode_subtopics(d.subtopics) for s, d in self.splits.items()}
        ood = load_ood_passages()
        for s in ("val", "test"):
            self.text[f"ood_{s}"] = [normalize(t) for t in ood[ood.split == s].text]
        seg = load_se_general()
        for s in ("ext_dev", "ext_test"):
            part = seg[seg.split == s]
            ind = part[~part.is_ood]
            self.text[f"se_{s}"] = [normalize(t) for t in ind.text]
            self.yg[f"se_{s}"] = self.space.encode_general(ind.general)
            self.text[f"se_ood_{s}"] = [normalize(t) for t in part[part.is_ood].text]
        ses = load_se_subtopic()
        for s in ("ext_dev", "ext_test"):
            part = ses[ses.split == s].reset_index(drop=True)
            self.text[f"sesub_{s}"] = [normalize(t) for t in part.text]
            self.yg[f"sesub_{s}"] = self.space.encode_general(part.general)
            self.ys[f"sesub_{s}"] = self.space.encode_subtopics(part.subtopics)
        log.info("sizes: %s", {k: len(v) for k, v in self.text.items()})
