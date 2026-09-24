"""Build the language-identification evaluation set (docs/HARDENING.md, item 6).

Source: Tatoeba sentence exports (CC BY 2.0 FR, https://tatoeba.org) - short,
conversational sentences in many languages, which is exactly the input the
language gate has to judge. English sentences are the positives; 17 other
languages (13 in Latin script, where confusion with English is possible) are
the negatives.

Every sentence is assigned to ``dev`` or ``locked`` by a hash of its Tatoeba id,
so the split is stable and independent of sampling. ``dev`` is used to choose the
language-ID method and its threshold; ``locked`` is only evaluated once, after
all decisions are frozen (scripts/locked_eval.py).

Besides the full sentence, the first 1, 2 and 3 words of each sentence are added
as separate rows (``length`` = 1/2/3/full), because short inputs are where the
old gate failed.

    python scripts/build_language_eval.py      # -> data/external/language_eval.jsonl
"""

from __future__ import annotations

import bz2
import hashlib
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextlens.config import RANDOM_SEED  # noqa: E402

RAW = ROOT / "data" / "raw" / "tatoeba"
OUT = ROOT / "data" / "external" / "language_eval.jsonl"
URL = "https://downloads.tatoeba.org/exports/per_language/{0}/{0}_sentences.tsv.bz2"
NON_ENGLISH = [
    "tur", "fra", "spa", "deu", "ita", "por", "nld", "pol", "swe", "ron",
    "ind", "fin", "hun", "ces", "dan", "rus", "ell",
]  # fmt: skip
PER_LANGUAGE = 400  # sentences per split and non-English language
ENGLISH = 3000  # English sentences per split
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def split_of(sentence_id: str) -> str:
    return "dev" if int(hashlib.sha256(sentence_id.encode()).hexdigest(), 16) % 2 == 0 else "locked"


def load(lang: str) -> list[tuple[str, str]]:
    path = RAW / f"{lang}.tsv.bz2"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(lang), path)  # noqa: S310 - constant https URL
    rows = []
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3 and 2 <= len(parts[2].split()) <= 25:
                rows.append((parts[0], parts[2]))
    return rows


def main() -> int:
    rng = random.Random(RANDOM_SEED)
    out = []
    for lang in ["eng", *NON_ENGLISH]:
        rows = load(lang)
        rng.shuffle(rows)
        quota = ENGLISH if lang == "eng" else PER_LANGUAGE
        taken = {"dev": 0, "locked": 0}
        for sid, text in rows:
            split = split_of(sid)
            if taken[split] >= quota:
                if all(v >= quota for v in taken.values()):
                    break
                continue
            taken[split] += 1
            words = WORD.findall(text)
            base = {"source": "tatoeba", "id": sid, "lang": lang, "is_english": lang == "eng", "split": split}
            out.append({**base, "length": "full", "text": text})
            for k in (1, 2, 3):
                if len(words) > k:
                    out.append({**base, "length": str(k), "text": " ".join(words[:k])})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
