"""Exploratory data analysis and leakage checks (docs/DATASET_CARD.md, docs/EXPERIMENTS.md).

    python scripts/eda.py

Writes reports/eda.json and figures in reports/figures/eda_*.png.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from contextlens.config import PATHS
from contextlens.data.dataset import (
    acceptance_texts,
    load_ood_passages,
    load_passages,
    load_se_general,
    load_se_subtopic,
)
from contextlens.evaluation.plots import plot_bars
from contextlens.taxonomy import load_taxonomy

FIG = PATHS.figures
NEAR_DUP_COSINE = 0.9
PATTERNS = {
    "html_tag": re.compile(r"<[a-zA-Z/][^>]{0,80}>"),
    "url": re.compile(r"https?://|www\."),
    "mention": re.compile(r"(?<!\w)@\w+"),
    "hashtag": re.compile(r"(?<!\w)#\w+"),
    "emoji": re.compile("[\U0001f300-\U0001faff☀-➿]"),
    "mojibake": re.compile("Ã.|â€|Â "),
    "replacement_char": re.compile("�"),
    "non_ascii": re.compile(r"[^\x00-\x7f]"),
    "latex": re.compile(r"\$[^$]+\$|\\\w+\{"),
}


def text_stats(texts: pd.Series) -> dict:
    words = texts.str.split().str.len()
    chars = texts.str.len()
    return {
        "n": int(len(texts)),
        "words": {
            "min": int(words.min()),
            "p5": float(words.quantile(0.05)),
            "median": float(words.median()),
            "mean": round(float(words.mean()), 2),
            "p95": float(words.quantile(0.95)),
            "max": int(words.max()),
        },
        "chars": {"median": float(chars.median()), "mean": round(float(chars.mean()), 1), "max": int(chars.max())},
        "anomalies": {k: int(texts.str.contains(p).sum()) for k, p in PATTERNS.items()},
        "empty": int((texts.str.strip() == "").sum()),
        "exact_duplicates": int(texts.str.lower().duplicated().sum()),
    }


def top_tokens(texts: pd.Series, k: int = 15) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(w for w in re.findall(r"[a-z][a-z\-]+", t.lower()) if w not in ENGLISH_STOP_WORDS)
    return counter.most_common(k)


def near_duplicates(train: list[str], other: list[str]) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(train)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(train))
    dist, idx = nn.kneighbors(vec.transform(other))
    sims = 1 - dist[:, 0]
    worst = np.argsort(-sims)[:5]
    return {
        "n": len(other),
        "max_cosine": round(float(sims.max()), 4),
        "p99_cosine": round(float(np.quantile(sims, 0.99)), 4),
        f"count_cosine_ge_{NEAR_DUP_COSINE}": int((sims >= NEAR_DUP_COSINE).sum()),
        "closest_pairs": [
            {"text": other[i], "nearest_train": train[idx[i, 0]], "cosine": round(float(sims[i]), 4)} for i in worst
        ],
    }


def main() -> None:
    tax = load_taxonomy()
    df = load_passages()
    ood = load_ood_passages()
    seg, ses = load_se_general(), load_se_subtopic()
    report: dict = {
        "shape": list(df.shape),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "nulls": {c: int(v) for c, v in df.isna().sum().items()},
        "articles": int(df.wiki_id.nunique()),
        "passages_per_article": df.groupby("wiki_id").size().describe().round(3).to_dict(),
        "split_passages": df.split.value_counts().to_dict(),
        "split_articles": df.groupby("split").wiki_id.nunique().to_dict(),
    }
    gen = df.groupby(["general", "split"]).size().unstack(fill_value=0)
    report["general_distribution"] = gen.to_dict(orient="index")
    counts = df.general.value_counts()
    report["general_imbalance_ratio"] = round(float(counts.max() / counts.min()), 3)
    subs = df.subtopics.str.split("|").explode()
    sub_counts = subs.value_counts()
    report["subtopic_distribution"] = sub_counts.to_dict()
    report["subtopic_imbalance_ratio"] = round(float(sub_counts.max() / sub_counts.min()), 3)
    report["labels_per_passage"] = df.subtopics.str.split("|").str.len().value_counts().sort_index().to_dict()
    report["depth_distribution"] = df.depth.value_counts().sort_index().to_dict()
    report["text"] = text_stats(df.text)
    report["text_by_general"] = {g: text_stats(df[df.general == g].text)["words"] for g in tax.general_ids}
    vec = TfidfVectorizer(min_df=1).fit(df[df.split == "train"].text)
    report["vocabulary_size_train"] = len(vec.vocabulary_)
    report["top_tokens_by_general"] = {g: top_tokens(df[df.general == g].text) for g in tax.general_ids}
    report["ood"] = {
        "passages": int(len(ood)),
        "by_split": ood.split.value_counts().to_dict(),
        "by_category": ood.category.value_counts().to_dict(),
        "text": text_stats(ood.text),
    }
    report["stackexchange_general"] = {
        "rows": int(len(seg)),
        "ood_rows": int(seg.is_ood.sum()),
        "by_general": seg[~seg.is_ood].general.value_counts().to_dict(),
        "by_split": seg.split.value_counts().to_dict(),
        "text": text_stats(seg.text),
    }
    report["stackexchange_subtopic"] = {
        "rows": int(len(ses)),
        "by_general": ses.general.value_counts().to_dict(),
        "by_subtopic": ses.subtopics.explode().value_counts().to_dict(),
        "text": text_stats(ses.text),
    }

    # ---- leakage checks
    by_split = {s: set(df[df.split == s].wiki_id) for s in ("train", "val", "test")}
    train_texts = df[df.split == "train"].text.str.lower()
    leakage = {
        "article_overlap_train_val": len(by_split["train"] & by_split["val"]),
        "article_overlap_train_test": len(by_split["train"] & by_split["test"]),
        "article_overlap_val_test": len(by_split["val"] & by_split["test"]),
        "exact_text_overlap_train_test": int(df[df.split == "test"].text.str.lower().isin(set(train_texts)).sum()),
        "exact_text_overlap_train_val": int(df[df.split == "val"].text.str.lower().isin(set(train_texts)).sum()),
        "ood_titles_in_corpus": int(ood.title.isin(set(df.title)).sum()),
        "se_titles_in_corpus": int(seg.text.str.lower().isin(set(df.text.str.lower())).sum()),
    }
    acc = json.loads((PATHS.root / "tests" / "acceptance_cases.json").read_text(encoding="utf-8"))
    probe_texts = [t.lower() for t in acceptance_texts(acc)]
    corpus_lower = df.text.str.lower()
    leakage["acceptance_sentences_in_corpus"] = int(
        sum(corpus_lower.str.contains(re.escape(p)).any() for p in probe_texts)
    )
    tr = df[df.split == "train"].text.tolist()
    leakage["near_duplicates_test_vs_train"] = near_duplicates(tr, df[df.split == "test"].text.tolist())
    leakage["near_duplicates_val_vs_train"] = near_duplicates(tr, df[df.split == "val"].text.tolist())
    report["leakage"] = leakage

    FIG.mkdir(parents=True, exist_ok=True)
    plot_bars(counts.to_dict(), FIG / "eda_general_distribution.png", "Passages per general topic", "passages")
    plot_bars(
        sub_counts.to_dict(), FIG / "eda_subtopic_distribution.png", "Passages per subtopic (multi-label)", "passages"
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.text.str.split().str.len(), bins=40, color="#2f6fdb", alpha=0.8, label="Wikipedia passages")
    ax.hist(seg.text.str.split().str.len(), bins=40, color="#e07b39", alpha=0.6, label="Stack Exchange titles")
    ax.set_xlabel("words per text")
    ax.set_ylabel("count")
    ax.set_title("Text length: training passages vs. real user questions", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_length_hist.png", dpi=130)
    plt.close(fig)
    out = PATHS.reports / "eda.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "shape",
                    "articles",
                    "split_passages",
                    "general_imbalance_ratio",
                    "subtopic_imbalance_ratio",
                    "labels_per_passage",
                )
            },
            default=str,
        )
    )
    print(json.dumps({k: v for k, v in leakage.items() if not isinstance(v, dict)}))
    print("near dup test:", {k: v for k, v in leakage["near_duplicates_test_vs_train"].items() if k != "closest_pairs"})


if __name__ == "__main__":
    main()
