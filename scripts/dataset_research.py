"""Measure candidate public datasets against the ContextLens taxonomy.

Downloads (small) evaluation splits of candidate datasets from the Hugging Face
Hub at pinned revisions and reports rows, label inventory, exact-duplicate
rate, empty texts, text length and **taxonomy coverage** (how many of our 8
general topics / 28 subtopics each dataset can supply labelled examples for).
Coverage mappings are hand-written below and justified in
docs/DATASET_RESEARCH.md. Output: reports/dataset_research.json

    python scripts/dataset_research.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from huggingface_hub import hf_hub_download

from contextlens.config import PATHS

CACHE = PATHS.data_raw / "dataset_research"

# (repo, revision, file, loader, text columns, label column)
CANDIDATES = {
    "AG News": (
        "fancyzhx/ag_news",
        "eb185aade064a813bc0b7f42de02595523103ca4",
        "data/test-00000-of-00001.parquet",
        "parquet",
        ["text"],
        "label",
    ),
    "20 Newsgroups (UCI #113, SetFit mirror)": (
        "SetFit/20_newsgroups",
        "f1b91292074e7cfb69be58b642d583ec262f30ed",
        "test.jsonl",
        "jsonl",
        ["text"],
        "label_text",
    ),
    "DBpedia-14": (
        "fancyzhx/dbpedia_14",
        "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
        "dbpedia_14/test-00000-of-00001.parquet",
        "parquet",
        ["title", "content"],
        "label",
    ),
    "DBPedia Classes (Kaggle danofer mirror)": (
        "DeveloperOats/DBPedia_Classes",
        "4d0aa96069f24063697e4df63b95be78d3f7fb7d",
        "DBPEDIA_val.csv",
        "csv",
        ["text"],
        "l3",
    ),
    "News Category / HuffPost (Kaggle rmisra mirror)": (
        "heegyu/news-category-dataset",
        "304a05a55bc6abc0446d8fae0d0771716b6a271a",
        "data.json",
        "jsonl",
        ["headline", "short_description"],
        "category",
    ),
}

# Which of OUR labels each dataset's labels can supply (hand mapping; see docs).
COVERAGE = {
    "AG News": {"general": ["sports", "technology"], "subtopics": []},
    "20 Newsgroups (UCI #113, SetFit mirror)": {
        "general": ["technology", "physics", "sports"],
        "subtopics": ["hardware", "software", "astrophysics"],
    },
    "DBpedia-14": {"general": ["sports", "books"], "subtopics": []},
    "DBPedia Classes (Kaggle danofer mirror)": {
        "general": ["sports", "books", "history"],
        "subtopics": ["football", "basketball", "olympics", "poetry", "authors", "world_wars"],
    },
    "News Category / HuffPost (Kaggle rmisra mirror)": {
        "general": ["science", "technology", "sports"],
        "subtopics": [],
    },
}

# Reported (not downloaded) candidates, numbers from the official sources.
REPORTED = [
    {
        "dataset": "Yahoo! Answers Topics",
        "source": "HF community-datasets/yahoo_answers_topics",
        "rows": 1460000,
        "classes": 10,
        "license": "unknown",
        "general_covered": 3,
        "subtopics_covered": 0,
    },
    {
        "dataset": "Web of Science WOS-46985",
        "source": "Mendeley Data 9rw3vkcfy4 (Kowsari et al. 2017)",
        "rows": 46985,
        "classes": "7 / 134",
        "license": "CC BY 4.0",
        "general_covered": 3,
        "subtopics_covered": 5,
    },
    {
        "dataset": "arXiv metadata",
        "source": "Kaggle Cornell-University/arxiv",
        "rows": ">2,000,000",
        "classes": "~150 (multi-label)",
        "license": "CC0 (metadata)",
        "general_covered": 5,
        "subtopics_covered": 14,
    },
    {
        "dataset": "LSHTC (Wikipedia)",
        "source": "Kaggle competition lshtc",
        "rows": ">2,000,000",
        "classes": 325056,
        "license": "competition terms",
        "general_covered": "n/a (no raw text: pre-tokenised feature ids)",
        "subtopics_covered": "n/a",
    },
    {
        "dataset": "Stack Exchange (MTEB clustering + API tags)",
        "source": "HF mteb/stackexchange-clustering; api.stackexchange.com",
        "rows": "~75,000 titles (121 sites)",
        "classes": "site + tags",
        "license": "CC BY-SA 4.0 (content)",
        "general_covered": 8,
        "subtopics_covered": 25,
    },
    {
        "dataset": "Wikipedia 20231101.en + category graph (built here)",
        "source": "HF wikimedia/wikipedia + DBpedia SPARQL",
        "rows": "6.4M articles",
        "classes": "any (category graph)",
        "license": "CC BY-SA 3.0 / GFDL",
        "general_covered": 8,
        "subtopics_covered": 28,
    },
]


def load(repo: str, rev: str, fname: str, kind: str) -> pd.DataFrame:
    path = hf_hub_download(repo, fname, repo_type="dataset", revision=rev, local_dir=CACHE / repo.replace("/", "__"))
    if kind == "parquet":
        return pd.read_parquet(path)
    if kind == "csv":
        return pd.read_csv(path)
    return pd.read_json(path, lines=True)


def main() -> None:
    results = []
    for name, (repo, rev, fname, kind, text_cols, label_col) in CANDIDATES.items():
        df = load(repo, rev, fname, kind)
        text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        words = text.str.split().str.len()
        cov = COVERAGE[name]
        results.append(
            {
                "dataset": name,
                "hub_repo": repo,
                "revision": rev,
                "file_measured": fname,
                "rows_measured": int(len(df)),
                "classes": int(df[label_col].nunique()),
                "largest_class_share": round(float(df[label_col].value_counts(normalize=True).iloc[0]), 4),
                "smallest_class_share": round(float(df[label_col].value_counts(normalize=True).iloc[-1]), 4),
                "exact_duplicate_rate": round(float(text.duplicated().mean()), 4),
                "empty_text_rate": round(float((text == "").mean()), 4),
                "median_words": float(words.median()),
                "p95_words": float(words.quantile(0.95)),
                "general_covered": len(cov["general"]),
                "subtopics_covered": len(cov["subtopics"]),
                "coverage": cov,
            }
        )
        print(
            f"{name}: rows={len(df)} classes={results[-1]['classes']} dup={results[-1]['exact_duplicate_rate']}"
            f" median_words={results[-1]['median_words']} coverage={len(cov['general'])}/8,"
            f" {len(cov['subtopics'])}/28"
        )
    out = PATHS.reports / "dataset_research.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measured": results, "reported": REPORTED}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
