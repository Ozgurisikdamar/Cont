"""Evaluate the trained artifact and write the report data.

    python evaluate.py                     # development stage (default)
    python evaluate.py --stage locked      # final holdout - once, after scripts/freeze.py

Stages (docs/HARDENING.md item 2, decisions.md D-36):

* ``dev``    - data used to take decisions: Wikipedia ``val``, Stack Exchange
  ``ext_dev`` (general + subtopic), Wikipedia OOD ``val``, CLINC150 ``dev``
  chat, Tatoeba ``dev`` -> reports/evaluation_dev.json + figures ``*_dev.png``.
* ``locked`` - the locked final holdout, evaluated once with a frozen
  configuration: new Wikipedia articles and 2026 Stack Exchange questions
  (scripts/build_locked_sets.py), CLINC150 ``test``, Tatoeba ``locked``; plus
  the v1.0 test splits reported as **legacy** (they were seen during v1.0
  development). Refuses to run unless reports/locked/FREEZE.json matches the
  current configuration, artifact and data, and refuses a second run unless
  ``--rerun-reason`` is given (the reason is recorded). Writes
  reports/locked/results.json and the raw predictions to reports/locked/raw/.

Both stages report: general topic (multi-class) and subtopics (multi-label),
the deployed uncertain gate on in-domain and off-topic data, off-topic
detection (AUROC, AUPRC, FPR@95TPR, recall), language gate, calibration,
error breakdowns, latency and the acceptance probes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from contextlens.config import PATHS, load_settings
from contextlens.data.dataset import (
    DataMissingError,
    LabelSpace,
    load_jsonl,
    load_ood_passages,
    load_passages,
    load_se_general,
    load_se_subtopic,
)
from contextlens.evaluation.metrics import multiclass_report, multilabel_report, ood_report, reliability_curve
from contextlens.evaluation.plots import plot_confusion, plot_per_class_f1, plot_reliability
from contextlens.logging_setup import setup_logging
from contextlens.models.artifact import ArtifactError, load_artifact
from contextlens.models.heads import decide_subtopics, joint_subtopic_probs
from contextlens.preprocessing.text import normalize
from contextlens.services.composer import compose
from contextlens.services.tracker import ConversationTracker
from contextlens.taxonomy import load_taxonomy

log = logging.getLogger("evaluate")
ACCEPTANCE = PATHS.root / "tests" / "acceptance_cases.json"
N_ERROR_EXAMPLES = 40


def evaluate_split(model, space: LabelSpace, texts: list[str], yg: np.ndarray, Ys: np.ndarray | None) -> dict:
    gp, cond, ood = model.predict_proba(texts)
    out: dict = {"general": multiclass_report(yg, gp, space.general_ids), "ood_scores": ood}
    if Ys is not None:
        dec = decide_subtopics(cond, space.parent_col, gp.argmax(1), model.subtopic_threshold)
        out["subtopics"] = multilabel_report(
            Ys, dec, joint_subtopic_probs(gp, cond, space.parent_col), space.subtopic_ids
        )
        # hierarchy-conditional view: subtopic metrics only where the general topic was right
        right = gp.argmax(1) == yg
        joint = joint_subtopic_probs(gp, cond, space.parent_col)
        out["subtopics_given_correct_general"] = multilabel_report(
            Ys[right], dec[right], joint[right], space.subtopic_ids
        )
    out["deployed_gate"] = gate_summary(model, texts, gp, ood, yg)
    out["_gp"] = gp
    return out


def gate_summary(model, texts: list[str], gp: np.ndarray, ood: np.ndarray, yg: np.ndarray | None = None) -> dict:
    """What the user sees: share of texts answered 'uncertain' by the deployed rule
    (OOD score below threshold OR calibrated confidence below min_confidence OR
    not English)."""
    conf = gp.max(axis=1)
    by_ood, by_conf = ood < model.ood_threshold, conf < model.min_confidence
    uncertain = model.uncertain_mask(texts, gp, ood)
    out: dict = {
        "n": int(len(conf)),
        "uncertain_rate": round(float(uncertain.mean()), 4),
        "uncertain_by_ood_score": round(float(by_ood.mean()), 4),
        "uncertain_by_low_confidence": round(float(by_conf.mean()), 4),
        "uncertain_by_language": round(float(np.mean([not model.looks_english(t) for t in texts])), 4),
    }
    if yg is not None:
        correct = gp.argmax(axis=1) == yg
        out["accuracy_all"] = round(float(correct.mean()), 4)
        out["accuracy_answered"] = round(float(correct[~uncertain].mean()), 4) if (~uncertain).any() else None
        out["accuracy_uncertain"] = round(float(correct[uncertain].mean()), 4) if uncertain.any() else None
    return out


def by_group(correct: np.ndarray, groups: list[str]) -> dict:
    """Accuracy (or flag rate) and count per group, sorted by rate ascending."""
    out = {}
    for g in sorted(set(groups)):
        mask = np.array([x == g for x in groups])
        out[g] = {"n": int(mask.sum()), "rate": round(float(correct[mask].mean()), 4)}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["rate"]))


def length_bucket(text: str, edges: tuple[int, ...]) -> str:
    n = len(text.split())
    for lo, hi in zip((0, *edges), (*edges, 10_000), strict=True):
        if lo < n <= hi:
            return f"{lo + 1}-{hi}" if hi < 10_000 else f">{lo}"
    return f">{edges[-1]}"


def confusion_pairs(yg: np.ndarray, gp: np.ndarray, labels: list[str], k: int = 10) -> list[dict]:
    pred = gp.argmax(axis=1)
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(yg, pred, strict=True):
        if t != p:
            pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:k]
    return [
        {"true": labels[t], "pred": labels[p], "count": c, "share_of_true_class": round(c / int((yg == t).sum()), 4)}
        for (t, p), c in top
    ]


def errors(texts: list[str], yg: np.ndarray, gp: np.ndarray, labels: list[str], k: int) -> list[dict]:
    wrong = np.where(gp.argmax(1) != yg)[0]
    wrong = wrong[np.argsort(-gp[wrong].max(axis=1))]  # most confident mistakes first
    return [
        {
            "text": texts[i],
            "true": labels[yg[i]],
            "pred": labels[int(gp[i].argmax())],
            "confidence": round(float(gp[i].max()), 3),
        }
        for i in wrong[:k]
    ]


def acceptance(model, tax) -> dict:
    cases = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
    single = []
    for c in cases["single"]:
        p = model.predict(c["text"])
        top_sub = p.subtopics[0].id if p.subtopics else None
        ok = p.general == c["general"] and ("subtopic" not in c or top_sub == c["subtopic"])
        single.append(
            {
                **c,
                "pred_general": p.general,
                "confidence": round(p.confidence, 3),
                "pred_subtopics": [(s.id, round(s.probability, 3)) for s in p.subtopics],
                "status": p.status,
                "pass": bool(ok),
            }
        )
    ood = []
    for c in cases["ood"]:
        p = model.predict(c["text"])
        ood.append(
            {
                **c,
                "pred_general": p.general,
                "confidence": round(p.confidence, 3),
                "status": p.status,
                "reasons": list(p.reasons),
                "ood_score": round(p.ood_score, 4),
                "pass": p.status == "uncertain",
            }
        )
    settings = load_settings()
    tracker = ConversationTracker(
        decay=settings.decay,
        min_share=settings.theme_min_share,
        max_topics=settings.max_theme_topics,
        expire_after=settings.theme_expire_after,
        confirm_turns=settings.theme_confirm_turns,
    )
    children = {g: tax.children_of(g) for g in tax.general_ids}
    turns = []
    for i, msg in enumerate(cases["conversation"]["messages"], start=1):
        p = model.predict(msg)
        tracker.update(p.general_probs, p.subtopic_probs, 0.0 if p.uncertain else 1.0)
        theme = compose(tracker.theme(children), tax)
        expected = next((e for e in cases["conversation"]["expected_after"] if e["after_turn"] == i), None)
        turn = {
            "turn": i,
            "text": msg,
            "pred": p.general,
            "theme_label": theme.label,
            "phrase": theme.phrase,
            "generals": list(theme.generals),
        }
        if expected:
            turn["expected_phrase"] = expected["phrase"]
            turn["pass"] = set(expected["generals"]) == set(theme.generals) and theme.phrase == expected["phrase"]
        turns.append(turn)
    return {"single": single, "ood": ood, "conversation": turns}


def measure_latency(model, texts: list[str]) -> dict:
    model.predict(texts[0])
    times = []
    for t in texts[:200]:
        t0 = time.perf_counter()
        model.predict(t)
        times.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    t0 = time.perf_counter()
    model.predict_many(texts[:500])
    batch_ms = (time.perf_counter() - t0) * 1000 / min(500, len(texts))
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return {
        "single_ms_median": round(float(np.median(times)), 2),
        "single_ms_p95": round(float(np.percentile(times, 95)), 2),
        "batch_ms_per_text": round(batch_ms, 3),
        "python_heap_peak_mb_during_batch": round(peak, 1),
    }


def strip_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_") and k != "ood_scores"}


LOCKED_DIR = PATHS.reports / "locked"


def _read_jsonl(path: Path) -> pd.DataFrame:
    return load_jsonl(path)


def _labels(df: pd.DataFrame) -> pd.Series:
    """Subtopics as the "a|b" strings LabelSpace expects (lists in the locked files)."""
    return df.subtopics.map(lambda v: "|".join(v) if isinstance(v, list) else v)


def load_stage(stage: str) -> dict[str, dict]:
    """Named evaluation sections -> their data frames (normalised later)."""
    df, ood, seg, ses = load_passages(), load_ood_passages(), load_se_general(), load_se_subtopic()
    clinc = _read_jsonl(PATHS.data_external / "ood_conversational.jsonl")
    lang = _read_jsonl(PATHS.data_external / "language_eval.jsonl")
    legacy = {
        "wiki": df[df.split == "test"],
        "se_general": seg[seg.split == "ext_test"],
        "se_subtopic": ses[ses.split == "ext_test"],
        "ood_wiki": ood[ood.split == "test"],
    }
    if stage == "dev":
        return {
            "dev": {
                "wiki": df[df.split == "val"],
                "se_general": seg[seg.split == "ext_dev"],
                "se_subtopic": ses[ses.split == "ext_dev"],
                "ood_wiki": ood[ood.split == "val"],
                "chat": clinc[clinc.split == "dev"],
                "language": lang[lang.split == "dev"],
            }
        }
    se_locked = _read_jsonl(PATHS.root / "data" / "locked" / "se_locked.jsonl")
    se_locked = se_locked.assign(subtopics=_labels(se_locked))
    wiki_locked = _read_jsonl(PATHS.root / "data" / "locked" / "wiki_locked.jsonl")
    return {
        "locked": {
            "wiki": wiki_locked.assign(subtopics=_labels(wiki_locked)),
            "se_general": se_locked[se_locked.set == "general"],
            "se_subtopic": se_locked[(se_locked.subtopics != "") & ~se_locked.is_ood],
            "chat": clinc[clinc.split == "locked"],
            "language": lang[lang.split == "locked"],
        },
        "legacy_seen_in_v1": legacy,
    }


def language_summary(model, lang: pd.DataFrame) -> dict:
    ok = np.array([model.looks_english(normalize(t)) for t in lang.text])
    out = {}
    for length in ("1", "2", "3", "full"):
        m = (lang.length == length).to_numpy()
        eng, other = m & lang.is_english.to_numpy(), m & ~lang.is_english.to_numpy()
        out[length] = {
            "english_accepted": round(float(ok[eng].mean()), 4),
            "non_english_rejected": round(float((~ok[other]).mean()), 4),
            "n_english": int(eng.sum()),
            "n_non_english": int(other.sum()),
        }
    return out


def evaluate_section(model, space: LabelSpace, data: dict, raw_dir: Path | None) -> dict:
    out: dict = {}
    raw: dict[str, list[dict]] = {}

    def keep_raw(name: str, texts: list[str], gp, ood, yg=None) -> None:
        if raw_dir is None:
            return
        unc = model.uncertain_mask(texts, gp, ood)
        raw[name] = [
            {
                "text": t,
                "gold": None if yg is None else space.general_ids[int(yg[i])],
                "pred": space.general_ids[int(gp[i].argmax())],
                "confidence": round(float(gp[i].max()), 4),
                "ood_score": round(float(ood[i]), 5),
                "uncertain": bool(unc[i]),
            }
            for i, t in enumerate(texts)
        ]

    wiki = data["wiki"]
    w_texts = [normalize(t) for t in wiki.text]
    yg_w = space.encode_general(wiki.general)
    res_w = evaluate_split(model, space, w_texts, yg_w, space.encode_subtopics(wiki.subtopics))
    out["wiki"] = strip_private(res_w)
    out["wiki_errors"] = errors(w_texts, yg_w, res_w["_gp"], space.general_ids, N_ERROR_EXAMPLES)
    keep_raw("wiki", w_texts, res_w["_gp"], res_w["ood_scores"], yg_w)

    seg = data["se_general"]
    se_in = seg[~seg.is_ood.astype(bool)]
    s_texts = [normalize(t) for t in se_in.text]
    yg_s = space.encode_general(se_in.general)
    res_s = evaluate_split(model, space, s_texts, yg_s, None)
    out["se_general"] = strip_private(res_s)
    # decisions.md D-32: hsm is "history of science AND mathematics" and the only
    # Science site of the general set - reported with and without it. Without hsm
    # there is no Science question left, so macro-F1 is over the present classes.
    no_hsm = np.array([str(site).split(".")[0] != "hsm" for site in se_in.site])
    present = sorted(set(yg_s[no_hsm].tolist()))
    pred = res_s["_gp"].argmax(axis=1)
    out["se_general_without_hsm"] = {
        "n": int(no_hsm.sum()),
        "accuracy": float(np.mean(pred[no_hsm] == yg_s[no_hsm])),
        "macro_f1_present_classes": float(
            f1_score(yg_s[no_hsm], pred[no_hsm], labels=present, average="macro", zero_division=0)
        ),
        "classes": [space.general_ids[i] for i in present],
    }
    out["se_general_errors"] = errors(s_texts, yg_s, res_s["_gp"], space.general_ids, N_ERROR_EXAMPLES)
    keep_raw("se_general", s_texts, res_s["_gp"], res_s["ood_scores"], yg_s)

    sub = data["se_subtopic"]
    sub_texts = [normalize(t) for t in sub.text]
    res_sub = evaluate_split(
        model, space, sub_texts, space.encode_general(sub.general), space.encode_subtopics(sub.subtopics)
    )
    out["se_subtopic"] = strip_private(res_sub)

    offtopic: dict = {}
    sources = {"se_sites": [normalize(t) for t in seg[seg.is_ood.astype(bool)].text]}
    if "ood_wiki" in data:
        sources["wiki_categories"] = [normalize(t) for t in data["ood_wiki"].text]
    if "chat" in data:
        sources["chat"] = [normalize(t) for t in data["chat"].text]
    for name, texts in sources.items():
        gp_o, _, sc_o = model.predict_proba(texts)
        in_scores = res_w["ood_scores"] if name == "wiki_categories" else res_s["ood_scores"]
        offtopic[name] = ood_report(in_scores, sc_o) | {
            "in_domain_reference": "wiki" if name == "wiki_categories" else "se_general",
            "deployed_gate": gate_summary(model, texts, gp_o, sc_o),
        }
        keep_raw(f"offtopic_{name}", texts, gp_o, sc_o)
    out["offtopic"] = offtopic
    if "language" in data:
        out["language"] = language_summary(model, data["language"])

    out["calibration"] = {
        "wiki": reliability_curve(res_w["_gp"], yg_w),
        "se_general": reliability_curve(res_s["_gp"], yg_s),
    }
    unc_s = model.uncertain_mask(s_texts, res_s["_gp"], res_s["ood_scores"])
    site_col = list(se_in.site)
    out["analysis"] = {
        "wiki_accuracy_by_words": by_group(
            res_w["_gp"].argmax(1) == yg_w, [length_bucket(t, (15, 25, 40)) for t in w_texts]
        ),
        "wiki_accuracy_by_label_count": by_group(
            res_w["_gp"].argmax(1) == yg_w, [f"{len(str(s).split('|'))} subtopic(s)" for s in wiki.subtopics]
        ),
        "wiki_confusion_pairs": confusion_pairs(yg_w, res_w["_gp"], space.general_ids),
        "se_accuracy_by_words": by_group(res_s["_gp"].argmax(1) == yg_s, [length_bucket(t, (7, 12)) for t in s_texts]),
        "se_accuracy_by_site": by_group(res_s["_gp"].argmax(1) == yg_s, site_col),
        "se_uncertain_rate_by_site": by_group(unc_s, site_col),
        "se_confusion_pairs": confusion_pairs(yg_s, res_s["_gp"], space.general_ids),
    }
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in raw.items():
            with (raw_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def freeze_fingerprint(model_dir: Path) -> dict[str, str]:
    """SHA-256 of everything that defines the frozen system (compared with FREEZE.json)."""
    files = [
        PATHS.root / "configs" / "model.json",
        PATHS.taxonomy,
        PATHS.root / "contextlens" / "config.py",
        PATHS.data_processed / "passages.parquet",
        PATHS.root / "data" / "locked" / "MANIFEST.json",
        model_dir / "metadata.json",
    ]
    return {str(p.relative_to(PATHS.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_freeze(model_dir: Path, rerun_reason: str | None) -> dict:
    freeze_path = LOCKED_DIR / "FREEZE.json"
    if not freeze_path.exists():
        raise SystemExit("no reports/locked/FREEZE.json - run scripts/freeze.py before the locked evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    now = freeze_fingerprint(model_dir)
    changed = [k for k, v in freeze["files"].items() if now.get(k) != v]
    if changed:
        raise SystemExit(f"configuration changed since the freeze: {changed}")
    results = LOCKED_DIR / "results.json"
    if results.exists() and not rerun_reason:
        raise SystemExit("the locked holdout was already evaluated; pass --rerun-reason to record a second run")
    return freeze


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "locked"), default="dev")
    parser.add_argument("--rerun-reason", default=None)
    args = parser.parse_args(argv)
    setup_logging("INFO", PATHS.root / "logs" / "evaluate.log")
    settings = load_settings()
    freeze = check_freeze(settings.model_dir, args.rerun_reason) if args.stage == "locked" else None
    t0 = time.perf_counter()
    try:
        model = load_artifact(settings.model_dir)
    except ArtifactError as exc:
        log.error("%s", exc)
        return 2
    load_seconds = time.perf_counter() - t0
    tax = load_taxonomy()
    space = LabelSpace.from_taxonomy(tax)
    try:
        sections = load_stage(args.stage)
    except DataMissingError as exc:
        log.error("%s", exc)
        return 2
    report: dict = {
        "stage": args.stage,
        "model": {
            k: model.metadata.get(k)
            for k in ("model_name", "model_version", "trained_at", "dataset_version", "encoder", "encoder_revision")
        }
        | {"head_type": model.head_type, "ood_method": model.metadata.get("ood_method", "centroid")},
        "load_seconds": round(load_seconds, 2),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if freeze is not None:
        report["freeze"] = {"frozen_at": freeze.get("frozen_at"), "commit": freeze.get("commit")}
        if args.rerun_reason:
            report["rerun_reason"] = args.rerun_reason
    for name, data in sections.items():
        raw_dir = LOCKED_DIR / "raw" / name if args.stage == "locked" else None
        report[name] = evaluate_section(model, space, data, raw_dir)
    first = next(iter(sections))
    report["latency"] = measure_latency(model, [normalize(t) for t in sections[first]["wiki"].text])
    report["acceptance"] = acceptance(model, tax)

    tag = args.stage
    figs = PATHS.figures
    main_sec = report[first]
    plot_confusion(main_sec["wiki"]["general"], figs / f"confusion_wiki_{tag}.png", f"General topic - Wikipedia {tag}")
    plot_confusion(
        main_sec["se_general"]["general"],
        figs / f"confusion_se_{tag}.png",
        f"General topic - Stack Exchange {tag}",
    )
    plot_per_class_f1(
        main_sec["wiki"]["subtopics"]["per_label"],
        figs / f"subtopic_f1_wiki_{tag}.png",
        f"Subtopic F1 - Wikipedia {tag}",
    )
    plot_reliability(main_sec["calibration"], figs / f"reliability_{tag}.png")
    if args.stage == "locked":
        LOCKED_DIR.mkdir(parents=True, exist_ok=True)
        out = LOCKED_DIR / "results.json"
    else:
        out = PATHS.reports / "evaluation_dev.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    g, s = main_sec["wiki"]["general"], main_sec["se_general"]["general"]
    log.info(
        "%s: wiki acc=%.4f macroF1=%.4f | SE acc=%.4f macroF1=%.4f | wrote %s",
        tag,
        g["accuracy"],
        g["macro_f1"],
        s["accuracy"],
        s["macro_f1"],
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
