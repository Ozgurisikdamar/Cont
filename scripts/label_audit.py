"""Manual label audit: print a reproducible sample, summarise recorded verdicts.

    python scripts/label_audit.py --print      # show the sample to judge
    python scripts/label_audit.py               # summarise reports/label_audit.json

The sample is 6 random TRAINING passages per general topic (random_state=42).
Verdicts were written by reading each passage and its article title:
  correct       - the article belongs to the labelled topic and the passage shows it
  weak_passage  - the article label is right but this passage alone carries no topical signal
  wrong_label   - the article itself belongs to another general topic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from contextlens.config import PATHS

PER_TOPIC = 6
OUT = PATHS.reports / "label_audit.json"


def sample() -> pd.DataFrame:
    df = pd.read_parquet(PATHS.data_processed / "passages.parquet")
    train = df[df.split == "train"]
    parts = [train[train.general == g].sample(PER_TOPIC, random_state=42) for g in sorted(train.general.unique())]
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    if args.print:
        for i, r in enumerate(sample().itertuples()):
            print(f"[{i}] {r.passage_id} | {r.general} | {r.subtopics} | {r.title}\n    {r.text}")
        return
    audit = json.loads(OUT.read_text(encoding="utf-8"))
    counts = Counter(item["verdict"] for item in audit["items"])
    n = len(audit["items"])
    print({k: f"{v}/{n} ({v / n:.1%})" for k, v in sorted(counts.items())})


if __name__ == "__main__":
    main()
