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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from contextlens.config import DataConfig, Paths
from contextlens.data.dbpedia import SparqlClient, articles_in, crawl_categories
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
                t for t, _ in sorted(articles_in(client, tree).items(), key=lambda kv: (kv[1], stable_hash(kv[0])))
                if t not in in_taxonomy_titles and t not in taken
            ]
            for title in titles[:per_category]:
                taken.add(title)
                out.append(OODArticle(title, category, split))
    return out


def build_corpus(taxonomy: Taxonomy, paths: Paths, cfg: DataConfig) -> dict:
    client = SparqlClient(cfg.sparql_endpoint, paths.data_raw / "sparql_cache")
    membership, stats = crawl_membership(taxonomy, client)
    labelled, ambiguous = assign_labels(taxonomy, membership)
    selected = select_balanced(taxonomy, labelled, membership, cfg.max_articles_per_subtopic)
    log.info("labelled=%d ambiguous=%d selected=%d", len(labelled), len(ambiguous), len(selected))

    ood = crawl_ood(taxonomy, client, set(membership), cfg.ood_articles_per_category)
    log.info("ood articles=%d", len(ood))

    wanted = {a.title for a in selected} | {o.title for o in ood}
    texts = extract_articles(
        wanted,
        repo=cfg.wiki_repo,
        config=cfg.wiki_config,
        revision=cfg.wiki_revision,
        num_shards=cfg.wiki_num_shards,
        download_dir=paths.data_raw / "wikipedia_shards",
    )
    log.info("found text for %d/%d titles", len(texts), len(wanted))

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
            passages.append({
                "passage_id": f"{doc['wiki_id']}-{idx}",
                "wiki_id": int(doc["wiki_id"]),
                "title": art.title,
                "general": art.general,
                "subtopics": "|".join(art.subtopics),
                "depth": art.depth,
                "split": split_of[art.title],
                "text": chunk,
            })
        articles_rows.append({
            "title": art.title,
            "wiki_id": doc["wiki_id"],
            "url": doc["url"],
            "general": art.general,
            "subtopics": "|".join(art.subtopics),
            "depth": art.depth,
            "split": split_of[art.title],
            "n_passages": n_kept,
            "text_sha256": sha256(doc["text"]),
        })

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
            ood_passages.append({
                "passage_id": f"{doc['wiki_id']}-{idx}", "wiki_id": int(doc["wiki_id"]), "title": o.title,
                "category": o.category, "split": o.split, "text": chunk,
            })
        ood_rows.append({"title": o.title, "wiki_id": doc["wiki_id"], "url": doc["url"],
                         "category": o.category, "split": o.split, "text_sha256": sha256(doc["text"])})

    paths.data_manifest.mkdir(parents=True, exist_ok=True)
    paths.data_processed.mkdir(parents=True, exist_ok=True)
    _write_csv(paths.data_manifest / "articles.csv", articles_rows)
    _write_csv(paths.data_manifest / "ood_articles.csv", ood_rows)
    (paths.data_manifest / "ambiguous_titles.txt").write_text("\n".join(sorted(ambiguous)) + "\n", encoding="utf-8")
    summary = {
        "wiki_snapshot": {"repo": cfg.wiki_repo, "config": cfg.wiki_config, "revision": cfg.wiki_revision},
        "label_source": cfg.sparql_endpoint,
        "crawl_stats": stats,
        "labelled_articles": len(labelled),
        "ambiguous_articles": len(ambiguous),
        "selected_articles": len(selected),
        "articles_with_text": len(kept),
        "missing_text": sorted({a.title for a in selected} - set(texts)),
        "passages": len(passages),
        "duplicate_passages_removed": duplicate_passages,
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
