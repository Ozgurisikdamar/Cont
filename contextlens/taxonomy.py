"""Taxonomy loading and hierarchy helpers.

The taxonomy (``configs/taxonomy.json``) is the single source of truth for
label ids, display names, the parent/child relation used for hierarchy
validation, and the composition metadata (role, broader topic, query phrase)
used by the conversation layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from contextlens.config import PATHS


@dataclass(frozen=True)
class Subtopic:
    id: str
    name: str
    phrase: str
    parent: str
    seeds: tuple[tuple[str, int], ...]  # (Wikipedia category, max crawl depth)
    exclude: tuple[str, ...]  # named pattern groups from the taxonomy file
    exclude_patterns: tuple[str, ...]  # subtopic-specific category regexes
    stackexchange: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class GeneralTopic:
    id: str
    name: str
    role: str  # "domain" or "format"
    phrase: str
    broader: str | None
    perspective_template: str | None
    subtopics: tuple[Subtopic, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: str
    generals: tuple[GeneralTopic, ...]
    raw: dict

    @property
    def general_ids(self) -> list[str]:
        return [g.id for g in self.generals]

    @property
    def subtopic_ids(self) -> list[str]:
        return [s.id for g in self.generals for s in g.subtopics]

    def general(self, gid: str) -> GeneralTopic:
        for g in self.generals:
            if g.id == gid:
                return g
        raise KeyError(f"unknown general topic: {gid}")

    def subtopic(self, sid: str) -> Subtopic:
        for g in self.generals:
            for s in g.subtopics:
                if s.id == sid:
                    return s
        raise KeyError(f"unknown subtopic: {sid}")

    def parent_of(self, sid: str) -> str:
        return self.subtopic(sid).parent

    def children_of(self, gid: str) -> list[str]:
        return [s.id for s in self.general(gid).subtopics]

    def is_consistent(self, general_id: str, subtopic_ids: list[str]) -> bool:
        """True if every subtopic belongs to ``general_id`` (hierarchy validation)."""
        children = set(self.children_of(general_id))
        return all(s in children for s in subtopic_ids)

    def display(self, label_id: str) -> str:
        try:
            return self.general(label_id).name
        except KeyError:
            return self.subtopic(label_id).name


def _parse_seed(seed: str) -> tuple[str, int]:
    name, _, depth = seed.partition("@")
    return name, int(depth) if depth else 0


def _parse(raw: dict) -> Taxonomy:
    generals = []
    seen: set[str] = set()
    for g in raw["general_topics"]:
        subs = []
        for s in g["subtopics"]:
            if s["id"] in seen:
                raise ValueError(f"duplicate label id {s['id']}")
            seen.add(s["id"])
            subs.append(
                Subtopic(
                    id=s["id"],
                    name=s["name"],
                    phrase=s["phrase"],
                    parent=g["id"],
                    seeds=tuple(_parse_seed(x) for x in s.get("seeds", [])),
                    exclude=tuple(s.get("exclude", [])),
                    exclude_patterns=tuple(s.get("exclude_patterns", [])),
                    stackexchange=tuple((site, tag) for site, tag in s.get("stackexchange", [])),
                )
            )
        if g["role"] not in {"domain", "format"}:
            raise ValueError(f"invalid role for {g['id']}: {g['role']}")
        generals.append(
            GeneralTopic(
                id=g["id"],
                name=g["name"],
                role=g["role"],
                phrase=g["phrase"],
                broader=g.get("broader"),
                perspective_template=g.get("perspective_template"),
                subtopics=tuple(subs),
            )
        )
    tax = Taxonomy(version=raw["version"], generals=tuple(generals), raw=raw)
    for g in tax.generals:
        if g.broader is not None and g.broader not in tax.general_ids:
            raise ValueError(f"{g.id}.broader references unknown topic {g.broader}")
    return tax


@lru_cache(maxsize=4)
def load_taxonomy(path: Path = PATHS.taxonomy) -> Taxonomy:
    with open(path, encoding="utf-8") as fh:
        return _parse(json.load(fh))
