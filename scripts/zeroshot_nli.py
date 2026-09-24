"""Experiment: zero-shot NLI classification (facebook/bart-large-mnli), no training data.

Runs on stratified samples (NLI costs one forward pass per label per text, so
the full splits would take hours on CPU). Output: reports/experiments/zeroshot_nli.json

    python scripts/zeroshot_nli.py --per-class 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from transformers import pipeline

from contextlens.config import PATHS, RANDOM_SEED
from contextlens.data.dataset import BenchmarkData
from contextlens.evaluation.metrics import multiclass_report
from contextlens.models.encoders import best_device

REPO, REVISION = "facebook/bart-large-mnli", "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"


def stratified(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    idx = [
        rng.choice(np.where(y == c)[0], size=min(per_class, int((y == c).sum())), replace=False) for c in np.unique(y)
    ]
    return np.sort(np.concatenate(idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()
    data = BenchmarkData()
    labels = [data.tax.general(g).name for g in data.space.general_ids]
    clf = pipeline(
        "zero-shot-classification", model=REPO, revision=REVISION, device=0 if best_device() == "cuda" else -1
    )
    rng = np.random.default_rng(RANDOM_SEED)
    out: dict = {
        "model": REPO,
        "revision": REVISION,
        "hypothesis_template": "This text is about {}.",
        "per_class": args.per_class,
    }
    for split in ("val", "se_ext_dev"):
        idx = stratified(data.yg[split], args.per_class, rng)
        texts = [data.text[split][i] for i in idx]
        t0 = time.perf_counter()
        res = clf(texts, candidate_labels=labels, hypothesis_template="This text is about {}.", multi_label=False)
        secs = time.perf_counter() - t0
        probs = np.zeros((len(texts), len(labels)))
        for i, r in enumerate(res):
            for lab, score in zip(r["labels"], r["scores"], strict=True):
                probs[i, labels.index(lab)] = score
        rep = multiclass_report(data.yg[split][idx], probs, data.space.general_ids)
        out[split] = {k: round(v, 4) for k, v in rep.items() if isinstance(v, float)} | {
            "n": len(texts),
            "ms_per_text": round(secs * 1000 / len(texts), 1),
        }
        print(split, out[split])
    target = PATHS.reports / "experiments" / "zeroshot_nli.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
