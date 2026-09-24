"""Compare language-identification methods for the English gate (docs/HARDENING.md, item 6).

Data (dev only - the ``locked`` half is reserved for scripts/locked_eval.py):

* data/external/language_eval.jsonl, split ``dev``: Tatoeba sentences in English
  (positives) and 17 other languages (negatives), full and cut to 1/2/3 words;
* English in-domain text the model really receives: 1,000 Stack Exchange ext_dev
  question titles and 1,000 Wikipedia validation passages, also cut to 1/2/3 words.

Only texts that reach the gate are scored: at least one word that is not an
English stop word (``needs_language_check``).

Methods
  known_words      the v1.0 gate (share of known words, texts of >= 3 words only)
  py3langid        P(en) >= t (97 languages, 4.6 MB)
  lingua           confidence(en) >= t (75 languages, 96 MB)
  fasttext         P(en) >= t (lid.176.ftz, 176 languages, 0.9 MB + fastText)
  fasttext_reject  per word-count bucket: *not English* only when fastText's top
                   language is not English with probability >= c AND not all
                   words are in the English lexicon (every word type of the
                   Wikipedia training split). One or two words are often
                   genuinely ambiguous, so a rejection needs positive evidence
                   of another language.

Selection rule (per bucket for fasttext_reject): highest balanced accuracy
(mean of English acceptance and non-English rejection) subject to English
acceptance >= 0.99 on BOTH the in-domain texts and the Tatoeba English
sentences - the gate must not cost the model its own users.

The comparison libraries are experiment-only:
    pip install --no-deps py3langid==0.4.0 lingua-language-detector==2.1.1
    python scripts/language_gate_experiment.py   # -> reports/experiments/language_gate.json
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextlens.config import RANDOM_SEED, load_settings  # noqa: E402
from contextlens.models.artifact import known_words  # noqa: E402
from contextlens.models.language import (  # noqa: E402
    BUCKETS,
    FASTTEXT_MODEL,
    build_lexicon,
    ensure_fasttext_model,
    load_fasttext,
    word_bucket,
)
from contextlens.preprocessing.text import WORD_RE, needs_language_check, normalize  # noqa: E402

OUT = ROOT / "reports" / "experiments" / "language_gate.json"
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REJECT_CONF = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
MIN_ENGLISH_ACCEPT = 0.99
FOOTPRINT_MB = {"known_words": 0.3, "py3langid": 4.6, "fasttext": 0.9 + 4.4, "lingua": 96.2}


def prefixes(texts: list[str], source: str) -> list[dict]:
    rows = []
    for t in texts:
        words = WORD_RE.findall(t)
        rows.append({"source": source, "lang": "eng", "is_english": True, "length": "full", "text": t})
        for k in (1, 2, 3):
            if len(words) > k:
                rows.append(
                    {"source": source, "lang": "eng", "is_english": True, "length": str(k), "text": " ".join(words[:k])}
                )
    return rows


def dev_frame() -> pd.DataFrame:
    tat = pd.read_json(ROOT / "data/external/language_eval.jsonl", lines=True)
    tat = tat[tat.split == "dev"]
    se = pd.read_json(ROOT / "data/external/se_general_eval.jsonl", lines=True)
    se = se[(se.split == "ext_dev") & (~se.is_ood)].sample(1000, random_state=RANDOM_SEED)
    wiki = pd.read_parquet(ROOT / "data/processed/passages.parquet", columns=["text", "split"])
    wiki = wiki[wiki.split == "val"].sample(1000, random_state=RANDOM_SEED)
    rows = tat.to_dict("records") + prefixes(se.text.tolist(), "se_ext_dev") + prefixes(wiki.text.tolist(), "wiki_val")
    df = pd.DataFrame(rows)
    df["norm"] = [normalize(t) for t in df.text]
    df = df[[needs_language_check(t) for t in df.norm]].reset_index(drop=True)
    df["bucket"] = [word_bucket(len(WORD_RE.findall(t))) for t in df.norm]
    return df


def summarise(df: pd.DataFrame, accept: np.ndarray) -> dict:
    out: dict = {}
    eng = df.is_english.to_numpy()
    tat = (df.source == "tatoeba").to_numpy()
    for b in [*BUCKETS, "all"]:
        m = np.ones(len(df), bool) if b == "all" else (df.bucket == b).to_numpy()
        tat_en, ind_en, non = m & eng & tat, m & eng & ~tat, m & ~eng
        r_n = float((~accept[non]).mean())
        out[b] = {
            "english_accept_tatoeba": round(float(accept[tat_en].mean()), 4),
            "english_accept_indomain": round(float(accept[ind_en].mean()), 4),
            "non_english_reject": round(r_n, 4),
            "balanced_accuracy": round((float(accept[m & eng].mean()) + r_n) / 2, 4),
            "n_english": int((m & eng).sum()),
            "n_non_english": int(non.sum()),
        }
    return out


def english_ok(s: dict) -> bool:
    return s["english_accept_tatoeba"] >= MIN_ENGLISH_ACCEPT and s["english_accept_indomain"] >= MIN_ENGLISH_ACCEPT


def timed(fn: Callable[[str], object], texts: list[str]) -> tuple[list, float]:
    t0 = time.perf_counter()
    out = [fn(t) for t in texts]
    return out, (time.perf_counter() - t0) / len(texts) * 1000


def main() -> int:
    df = dev_frame()
    texts = df.norm.tolist()
    print(df.groupby(["source", "bucket"]).size().to_string())
    results = []

    def add(method: str, params: dict, accept: np.ndarray, ms: float, footprint: float) -> dict:
        s = summarise(df, accept)
        row = {
            "method": method,
            **params,
            "ms_per_text": round(ms, 3),
            "footprint_mb": footprint,
            "english_ok_every_bucket": all(english_ok(s[b]) for b in BUCKETS),
            "by_bucket": s,
        }
        results.append(row)
        print(method, params, s["all"])
        return row

    words_known = known_words(load_settings().model_dir)

    def old_gate(t: str) -> bool:
        words = [w for w in WORD_RE.findall(t.lower()) if len(w) >= 2]
        return len(words) < 3 or sum(w in words_known for w in words) / len(words) >= 0.4

    acc, ms = timed(old_gate, texts)
    add("known_words", {"threshold": 0.4}, np.array(acc), ms, FOOTPRINT_MB["known_words"])

    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    sc, ms = timed(lambda t: dict(ident.rank(t)).get("en", 0.0), texts)
    for t in THRESHOLDS:
        add("py3langid", {"threshold": t}, np.array(sc) >= t, ms, FOOTPRINT_MB["py3langid"])

    from lingua import Language, LanguageDetectorBuilder

    det = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    sc, ms = timed(lambda t: float(det.compute_language_confidence(t, Language.ENGLISH)), texts)
    for t in THRESHOLDS:
        add("lingua", {"threshold": t}, np.array(sc) >= t, ms, FOOTPRINT_MB["lingua"])

    model = load_fasttext(ensure_fasttext_model(FASTTEXT_MODEL))

    def ft(t: str) -> tuple[float, str, float]:
        labels, probs = model.predict(t, k=3)
        d = {lab.replace("__label__", ""): float(p) for lab, p in zip(labels, probs, strict=True)}
        return d.get("en", 0.0), labels[0].replace("__label__", ""), float(probs[0])

    sc3, ms = timed(ft, texts)
    p_en = np.array([x[0] for x in sc3])
    other = np.array([x[1] != "en" for x in sc3])
    conf = np.array([x[2] for x in sc3])
    for t in THRESHOLDS:
        add("fasttext", {"threshold": t}, p_en >= t, ms, FOOTPRINT_MB["fasttext"])

    passages = pd.read_parquet(ROOT / "data/processed/passages.parquet", columns=["text", "split"])
    lexicon = build_lexicon(passages[passages.split == "train"].text)
    all_known = np.array([all(w in lexicon for w in WORD_RE.findall(t.lower())) for t in texts])
    buckets = df.bucket.to_numpy()
    chosen: dict[str, float] = {}
    sweep: dict[str, list] = {}
    for b in BUCKETS:
        sweep[b] = []
        best: tuple[float, float] | None = None
        for c in REJECT_CONF:
            s = summarise(df, ~(other & (conf >= c) & ~all_known))[b]
            ok = english_ok(s)
            sweep[b].append({"reject_confidence": c, "eligible": ok, **s})
            if ok and (best is None or s["balanced_accuracy"] > best[1]):
                best = (c, s["balanced_accuracy"])
        chosen[b] = best[0] if best is not None else max(REJECT_CONF)
        print("bucket", b, "chosen reject confidence", chosen[b], "eligible", best is not None)
    c_vec = np.array([chosen[b] for b in buckets])
    final = add(
        "fasttext_reject+lexicon",
        {"reject_confidence": chosen, "lexicon_words": len(lexicon)},
        ~(other & (conf >= c_vec) & ~all_known),
        ms,
        FOOTPRINT_MB["fasttext"],
    )
    OUT.write_text(
        json.dumps(
            {
                "protocol": __doc__,
                "n_rows": len(df),
                "rows_by_bucket": {k: int(v) for k, v in df.groupby("bucket").size().items()},
                "selected": {"method": final["method"], "reject_confidence": chosen},
                "bucket_sweep": sweep,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("selected", chosen, json.dumps(final["by_bucket"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
