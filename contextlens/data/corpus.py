"""Corpus construction: labels (category graph) + text (pinned dump) -> passages.

Outputs (see docs/DATASET_CARD.md):

* ``data/manifest/articles.csv``     one row per labelled article (committed)
* ``data/manifest/ood_articles.csv`` out-of-taxonomy articles (committed)
* ``data/manifest/crawl_stats.json`` per-subtopic crawl statistics (committed)
* ``data/processed/passages.parquet``      in-taxonomy passages with split
* ``data/processed/ood_passages.parquet``  OOD passages with split (val/test)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from contextlens.config import DataConfig, Paths
from contextlens.data.dbpedia import SparqlClient, articles_in, crawl_categories, redirect_aliases
from contextlens.data.labels import (
    LabelledArticle,
    assign_labels,
    crawl_membership,
    select_balanced,
    stable_hash,
)
from contextlens.data.passages import make_passages
from contextlens.data.wikipedia_dump import extract_articles
from contextlens.taxonomy import Taxonomy

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


@dataclass
class OODArticle:
    title: str
    category: str
    split: str


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(articles: list[LabelledArticle], val_frac: float, test_frac: float) -> dict[str, str]:
    """Deterministic grouped split: the unit is the article, stratified by
    (general topic, first subtopic) so every class appears in every split."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for art in articles:
        strata[(art.general, art.subtopics[0])].append(art.title)
    split_of: dict[str, str] = {}
    for key in sorted(strata):
        titles = sorted(strata[key], key=stable_hash)
        n = len(titles)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        for i, title in enumerate(titles):
            if i < n_test:
                split_of[title] = "test"
            elif i < n_test + n_val:
                split_of[title] = "val"
            else:
                split_of[title] = "train"
    return split_of


def crawl_ood(
    taxonomy: Taxonomy, client: SparqlClient, in_taxonomy_titles: set[str], per_category: int
) -> list[OODArticle]:
    """Articles from out-of-taxonomy categories (depth <= 1), never in-taxonomy."""
    raw = taxonomy.raw["ood_evaluation_seeds"]
    patterns = [re.compile(p) for p in taxonomy.raw["global_exclude_category_patterns"]]
    out: list[OODArticle] = []
    taken: set[str] = set()
    for split in ("val", "test"):
        for category in raw[split]:
            tree = crawl_categories(client, category, 1, patterns)
            titles = [
                t
                for t, _ in sorted(articles_in(client, tree).items(), key=lambda kv: (kv[1], stable_hash(kv[0])))
                if t not in in_taxonomy_titles and t not in taken
            ]
            for title in titles[:per_category]:
                taken.add(title)
                out.append(OODArticle(title, category, split))
    return out


NEAR_DUPLICATE_COSINE = 0.9


def near_duplicate_ids(passages: list[dict], threshold: float = NEAR_DUPLICATE_COSINE) -> set[str]:
    """Validation/test passages whose TF-IDF cosine to some training passage is >= ``threshold``.

    Sibling articles share templated sentences ("A period 2 element is one of the
    chemical elements in the second row ..."); across splits such passages would
    let the model be tested on text it has effectively seen.
    """
    train = [p["text"] for p in passages if p["split"] == "train"]
    others = [p for p in passages if p["split"] != "train"]
    if not train or not others:
        return set()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(train))
    dist, _ = nn.kneighbors(vec.transform([p["text"] for p in others]))
    return {others[i]["passage_id"] for i in np.where(1.0 - dist[:, 0] >= threshold)[0]}


def recover_missing(
    client: SparqlClient, wanted: set[str], texts: dict[str, dict], passes: list[tuple[str, dict[str, Any], bool]]
) -> dict[str, int]:
    """Find text for titles the first exact-title pass missed; mutates ``texts``.

    DBpedia (labels) reflects a newer Wikipedia than the dumps (text), and a
    dump can lack articles its parser dropped. ``passes`` is a list of
    ``(name, extract_articles kwargs, exact_title_done)``; for each dump, in
    order, the still-missing titles are looked up

    1. by exact title (``<name>``), unless that was already done, then
    2. by redirect alias (``<name>_redirect``): an article renamed after the
       snapshot sits in the dump under a title that now redirects to it. A
       merge can match several aliases: the longest pre-merge article wins.

    An alias that is itself a wanted title is never borrowed, and no page id is
    assigned to two articles, so no text is duplicated across labels.
    """
    aliases = redirect_aliases(client, sorted(wanted - set(texts)))
    used_ids = {doc["wiki_id"] for doc in texts.values()}
    counts: Counter[str] = Counter()
    for name, dump, exact_done in passes:
        missing = sorted(wanted - set(texts))
        alias_titles = {a for m in missing for a in aliases.get(m, []) if a not in wanted}
        lookup = alias_titles | (set() if exact_done else set(missing))
        if not lookup:
            shutil.rmtree(dump["download_dir"], ignore_errors=True)
            continue
        found = extract_articles(lookup, **dump, keep_shards=False)  # also deletes the shards
        for title in missing:
            if not exact_done and title in found and found[title]["wiki_id"] not in used_ids:
                source, dump_title = name, title
            else:
                options = [a for a in aliases.get(title, []) if a in found and found[a]["wiki_id"] not in used_ids]
                if not options:
                    continue
                source = f"{name}_redirect"
                dump_title = max(sorted(options), key=lambda a: len(found[a]["text"]))  # first longest
            doc = found[dump_title]
            used_ids.add(doc["wiki_id"])
            texts[title] = {**doc, "dump_title": dump_title, "text_source": source}
            counts[source] += 1
        log.info("%s pass: recovered %s; %d titles still missing", name, dict(counts), len(wanted - set(texts)))
    return dict(sorted(counts.items()))


def build_corpus(taxonomy: Taxonomy, paths: Paths, cfg: DataConfig) -> dict:
    client = SparqlClient(cfg.sparql_endpoint, paths.data_raw / "sparql_cache")
    membership, stats = crawl_membership(taxonomy, client)
    labelled, ambiguous = assign_labels(taxonomy, membership)
    selected = select_balanced(taxonomy, labelled, membership, cfg.max_articles_per_subtopic)
    log.info("labelled=%d ambiguous=%d selected=%d", len(labelled), len(ambiguous), len(selected))

    ood = crawl_ood(taxonomy, client, set(membership), cfg.ood_articles_per_category)
    log.info("ood articles=%d", len(ood))

    wanted = {a.title for a in selected} | {o.title for o in ood}
    dump: dict[str, Any] = {
        "repo": cfg.wiki_repo,
        "config": cfg.wiki_config,
        "revision": cfg.wiki_revision,
        "num_shards": cfg.wiki_num_shards,
        "download_dir": paths.data_raw / "wikipedia_shards",
    }
    legacy: dict[str, Any] = {
        "repo": cfg.legacy_wiki_repo,
        "config": cfg.legacy_wiki_config,
        "revision": cfg.legacy_wiki_revision,
        "num_shards": cfg.legacy_wiki_num_shards,
        "download_dir": paths.data_raw / "legacy_wikipedia_shards",
    }
    texts = extract_articles(wanted, **dump, keep_shards=True)
    log.info("found text for %d/%d titles (exact title)", len(texts), len(wanted))
    recovered = recover_missing(client, wanted, texts, [("dump", dump, True), ("legacy_dump", legacy, False)])
    log.info("found text for %d/%d titles; recovered: %s", len(texts), len(wanted), recovered)

    kept = [a for a in selected if a.title in texts]
    split_of = assign_splits(kept, cfg.val_fraction, cfg.test_fraction)

    passages, articles_rows = [], []
    seen_text: set[str] = set()
    duplicate_passages = 0
    for art in kept:
        doc = texts[art.title]
        chunks = make_passages(
            doc["text"],
            min_words=cfg.min_passage_words,
            max_words=cfg.max_passage_words,
            max_passages=cfg.max_passages_per_article,
        )
        n_kept = 0
        for idx, chunk in enumerate(chunks):
            key = chunk.lower()
            if key in seen_text:  # exact duplicate across articles
                duplicate_passages += 1
                continue
            seen_text.add(key)
            n_kept += 1
            passages.append(
                {
                    "passage_id": f"{doc['wiki_id']}-{idx}",
                    "wiki_id": int(doc["wiki_id"]),
                    "title": art.title,
                    "general": art.general,
                    "subtopics": "|".join(art.subtopics),
                    "depth": art.depth,
                    "split": split_of[art.title],
                    "text_source": doc.get("text_source", "dump"),
                    "text": chunk,
                }
            )
        articles_rows.append(
            {
                "title": art.title,
                "wiki_id": doc["wiki_id"],
                "url": doc["url"],
                "general": art.general,
                "subtopics": "|".join(art.subtopics),
                "depth": art.depth,
                "split": split_of[art.title],
                "text_source": doc.get("text_source", "dump"),
                "dump_title": doc.get("dump_title", art.title),
                "n_passages": n_kept,
                "text_sha256": sha256(doc["text"]),
            }
        )

    near_dups = near_duplicate_ids(passages)
    passages = [p for p in passages if p["passage_id"] not in near_dups]
    per_title = Counter(p["title"] for p in passages)
    for row in articles_rows:
        row["n_passages"] = per_title.get(row["title"], 0)

    ood_passages, ood_rows = [], []
    for o in ood:
        if o.title not in texts:
            continue
        doc = texts[o.title]
        chunks = make_passages(
            doc["text"], min_words=cfg.min_passage_words, max_words=cfg.max_passage_words, max_passages=2
        )
        for idx, chunk in enumerate(chunks):
            if chunk.lower() in seen_text:
                continue
            ood_passages.append(
                {
                    "passage_id": f"{doc['wiki_id']}-{idx}",
                    "wiki_id": int(doc["wiki_id"]),
                    "title": o.title,
                    "category": o.category,
                    "split": o.split,
                    "text": chunk,
                }
            )
        ood_rows.append(
            {
                "title": o.title,
                "wiki_id": doc["wiki_id"],
                "url": doc["url"],
                "category": o.category,
                "split": o.split,
                "text_sha256": sha256(doc["text"]),
            }
        )

    paths.data_manifest.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    _write_csv(paths.data_manifest / "articles.csv", articles_rows)
    _write_csv(paths.data_manifest / "ood_articles.csv", ood_rows)
    (paths.data_manifest / "ambiguous_titles.txt").write_text("\n".join(sorted(ambiguous)) + "\n", encoding="utf-8")
    summary = {
        "wiki_snapshot": {"repo": cfg.wiki_repo, "config": cfg.wiki_config, "revision": cfg.wiki_revision},
        "wiki_snapshot_fallback": {
            "repo": cfg.legacy_wiki_repo,
            "config": cfg.legacy_wiki_config,
            "revision": cfg.legacy_wiki_revision,
        },
        "label_source": cfg.sparql_endpoint,
        "crawl_stats": stats,
        "labelled_articles": len(labelled),
        "ambiguous_articles": len(ambiguous),
        "selected_articles": len(selected),
        "articles_with_text": len(kept),
        "recovered_by_source": recovered,
        "missing_text": sorted({a.title for a in selected} - set(texts)),
        "passages": len(passages),
        "duplicate_passages_removed": duplicate_passages,
        "near_duplicate_passages_removed": len(near_dups),
        "ood_articles": len(ood_rows),
        "ood_passages": len(ood_passages),
    }
    (paths.data_manifest / "crawl_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(passages).to_parquet(paths.data_processed / "passages.parquet", index=False)
    pd.DataFrame(ood_passages).to_parquet(paths.data_processed / "ood_passages.parquet", index=False)
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
