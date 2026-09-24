"""Evaluate the trained artifact on every held-out set and write the final report data.

    python evaluate.py

Outputs reports/evaluation.json and figures in reports/figures/:
  * Wikipedia test split: general topic (multi-class) + subtopics (multi-label)
  * Stack Exchange ext_test: real user questions (general + subtopics)
  * OOD: Wikipedia out-of-taxonomy categories + Stack Exchange off-topic sites
  * calibration (ECE, reliability), latency, acceptance probes, error samples
"""

from __future__ import annotations

import json
import logging
import sys
import time
import tracemalloc

import numpy as np

from contextlens.config import PATHS, load_settings
from contextlens.data.dataset import (
    DataMissingError,
    LabelSpace,
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


def main() -> int:
    setup_logging("INFO", PATHS.root / "logs" / "evaluate.log")
    settings = load_settings()
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
        df, ood_df, seg, ses = load_passages(), load_ood_passages(), load_se_general(), load_se_subtopic()
    except DataMissingError as exc:
        log.error("%s", exc)
        return 2
    report: dict = {
        "model": {
            k: model.metadata.get(k)
            for k in ("model_name", "model_version", "trained_at", "dataset_version", "encoder", "encoder_revision")
        },
        "load_seconds": round(load_seconds, 2),
    }

    test = df[df.split == "test"]
    t_texts = [normalize(t) for t in test.text]
    res_test = evaluate_split(
        model, space, t_texts, space.encode_general(test.general), space.encode_subtopics(test.subtopics)
    )
    report["wiki_test"] = strip_private(res_test)
    report["wiki_test_errors"] = errors(
        t_texts, space.encode_general(test.general), res_test["_gp"], space.general_ids, N_ERROR_EXAMPLES
    )

    se_test = seg[(seg.split == "ext_test") & (~seg.is_ood)]
    s_texts = [normalize(t) for t in se_test.text]
    res_se = evaluate_split(model, space, s_texts, space.encode_general(se_test.general), None)
    report["se_general_ext_test"] = strip_private(res_se)
    report["se_general_errors"] = errors(
        s_texts, space.encode_general(se_test.general), res_se["_gp"], space.general_ids, N_ERROR_EXAMPLES
    )
    sub_test = ses[ses.split == "ext_test"]
    res_sesub = evaluate_split(
        model,
        space,
        [normalize(t) for t in sub_test.text],
        space.encode_general(sub_test.general),
        space.encode_subtopics(sub_test.subtopics),
    )
    report["se_subtopic_ext_test"] = strip_private(res_sesub)

    ood_test = [normalize(t) for t in ood_df[ood_df.split == "test"].text]
    se_ood = [normalize(t) for t in seg[(seg.split == "ext_test") & seg.is_ood].text]
    gp_ood_wiki, _, ood_scores_wiki = model.predict_proba(ood_test)
    gp_ood_se, _, ood_scores_se = model.predict_proba(se_ood)
    report["ood"] = {
        "threshold": model.ood_threshold,
        "wiki": ood_report(res_test["ood_scores"], ood_scores_wiki)
        | {
            "id_kept": float(np.mean(res_test["ood_scores"] >= model.ood_threshold)),
            "ood_flagged": float(np.mean(ood_scores_wiki < model.ood_threshold)),
            "deployed_gate": gate_summary(model, ood_test, gp_ood_wiki, ood_scores_wiki),
        },
        "stackexchange": ood_report(res_se["ood_scores"], ood_scores_se)
        | {
            "id_kept": float(np.mean(res_se["ood_scores"] >= model.ood_threshold)),
            "ood_flagged": float(np.mean(ood_scores_se < model.ood_threshold)),
            "deployed_gate": gate_summary(model, se_ood, gp_ood_se, ood_scores_se),
        },
    }
    yg_test = space.encode_general(test.general)
    report["calibration"] = {
        "wiki_test": reliability_curve(res_test["_gp"], yg_test),
        "se_ext_test": reliability_curve(res_se["_gp"], space.encode_general(se_test.general)),
    }
    # --- error-analysis breakdowns (docs/ERROR_ANALYSIS.md) ---------------------------
    gp_se = res_se["_gp"]
    yg_se = space.encode_general(se_test.general)
    unc_se = model.uncertain_mask(s_texts, gp_se, res_se["ood_scores"])
    ood_wiki_flag = model.uncertain_mask(ood_test, gp_ood_wiki, ood_scores_wiki)
    ood_se_flag = model.uncertain_mask(se_ood, gp_ood_se, ood_scores_se)
    report["analysis"] = {
        "wiki_test_accuracy_by_words": by_group(
            res_test["_gp"].argmax(1) == yg_test, [length_bucket(t, (15, 25, 40)) for t in t_texts]
        ),
        "wiki_test_accuracy_by_label_count": by_group(
            res_test["_gp"].argmax(1) == yg_test, [f"{len(s.split('|'))} subtopic(s)" for s in test.subtopics]
        ),
        "wiki_test_confusion_pairs": confusion_pairs(yg_test, res_test["_gp"], space.general_ids),
        "se_ext_test_accuracy_by_words": by_group(
            gp_se.argmax(1) == yg_se, [length_bucket(t, (7, 12)) for t in s_texts]
        ),
        "se_ext_test_accuracy_by_site": by_group(gp_se.argmax(1) == yg_se, list(se_test.site)),
        "se_ext_test_uncertain_rate_by_site": by_group(unc_se, list(se_test.site)),
        "se_ext_test_confusion_pairs": confusion_pairs(yg_se, gp_se, space.general_ids),
        "ood_wiki_uncertain_rate_by_category": by_group(ood_wiki_flag, list(ood_df[ood_df.split == "test"].category)),
        "ood_se_uncertain_rate_by_site": by_group(ood_se_flag, list(seg[(seg.split == "ext_test") & seg.is_ood].site)),
    }
    report["latency"] = measure_latency(model, t_texts)
    report["acceptance"] = acceptance(model, tax)

    figs = PATHS.figures
    plot_confusion(report["wiki_test"]["general"], figs / "confusion_wiki_test.png", "General topic - Wikipedia test")
    plot_confusion(
        report["se_general_ext_test"]["general"],
        figs / "confusion_se_ext_test.png",
        "General topic - Stack Exchange ext_test",
    )
    plot_per_class_f1(
        report["wiki_test"]["subtopics"]["per_label"],
        figs / "subtopic_f1_wiki_test.png",
        "Subtopic F1 - Wikipedia test",
    )
    plot_reliability(report["calibration"], figs / "reliability_final.png")
    out = PATHS.reports / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    g, s = report["wiki_test"]["general"], report["se_general_ext_test"]["general"]
    log.info(
        "wiki test acc=%.4f macroF1=%.4f | SE ext_test acc=%.4f macroF1=%.4f | wrote %s",
        g["accuracy"],
        g["macro_f1"],
        s["accuracy"],
        s["macro_f1"],
        out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
