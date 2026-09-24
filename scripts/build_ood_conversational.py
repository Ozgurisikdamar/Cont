"""Conversational out-of-taxonomy evaluation data (docs/HARDENING.md, item 3).

Source: CLINC150 (Larson et al., 2019; CC BY 3.0), Hugging Face dataset
``clinc/clinc_oos`` config ``plus`` at a pinned revision. Its 150 in-scope
intents are short assistant requests and chit-chat ("set a timer for ten
minutes", "what's my credit score", "tell me a joke") - exactly the off-topic
chat the v1.0 gate let through.

What is used, and why:
* the in-scope intents are the OOD negatives, **except** intents whose
  utterances can be about our topics (``EXCLUDED_INTENTS``: fun facts,
  definitions, the meaning of life, vaccines, unit conversion);
* CLINC's own ``oos`` class is **not** used: it is "out of the assistant's
  scope", not out of ours - a sample of its validation utterances includes
  "where do black holes come from", "is autism a genetic disease" and "who has
  the best record in the nfl", which are Physics, Biology and Sports questions.

Splits: CLINC ``validation`` -> ``dev`` (detector choice and thresholds),
CLINC ``test`` -> ``locked`` (evaluated once after the freeze,
evaluate.py --stage locked). CLINC ``train`` is kept for training a dedicated OOD
classifier (``train``); it never reaches an evaluation.

    python scripts/build_ood_conversational.py   # -> data/external/ood_conversational.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from huggingface_hub import hf_hub_download

from contextlens.config import PATHS

REPO = "clinc/clinc_oos"
REVISION = "155b9c710419136e17307b80d0a13e68cd46b4ec"
SPLITS = {"train": "train", "validation": "dev", "test": "locked"}
EXCLUDED_INTENTS = {"oos", "fun_fact", "definition", "meaning_of_life", "vaccines", "measurement_conversion"}
OUT = PATHS.root / "data" / "external" / "ood_conversational.jsonl"


def intent_names() -> list[str]:
    readme = Path(hf_hub_download(REPO, "README.md", repo_type="dataset", revision=REVISION)).read_text()
    section = readme[readme.find("config_name: plus") :]
    section = section[: section.find("config_name: small")]
    names = [n.strip("'") for _, n in re.findall(r"'(\d+)': (\S+)", section)]
    if len(names) != 151:
        raise SystemExit(f"expected 151 intent names in the dataset card, found {len(names)}")
    return names


def main() -> int:
    names = intent_names()
    rows = []
    for source, split in SPLITS.items():
        path = hf_hub_download(REPO, f"plus/{source}-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        df = pd.read_parquet(path)
        for text, intent in zip(df.text, df.intent, strict=True):
            name = names[int(intent)]
            if name not in EXCLUDED_INTENTS:
                rows.append({"source": "clinc150", "intent": name, "split": split, "text": text})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = pd.DataFrame(rows).groupby("split").size().to_dict()
    print(f"wrote {len(rows)} utterances to {OUT}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
