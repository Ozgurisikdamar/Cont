"""Model benchmark for ContextLens (docs/EXPERIMENTS.md, docs/MODEL_REPORT.md).

Selection protocol (no test-set peeking):
  * hyper-parameters are chosen on the Wikipedia *validation* split;
  * model families are compared on validation AND on ``ext_dev`` (real Stack
    Exchange questions - the style users actually type);
  * ``test`` / ``ext_test`` are NOT read (hardening item 2, decisions.md D-36):
    they were seen during v1.0 development; the final held-out numbers come
    from the locked holdout (``evaluate.py --stage locked``) after a freeze.

Sections (run all by default):
  general      featurizer x head grid for the 8 general topics
  zeroshot     label-description similarity with sentence encoders (no training)
  ensemble     concatenated embeddings of two encoders + logistic regression
  hierarchy    flat vs hierarchical subtopic modelling, threshold sweep
  calibration  ECE / reliability before and after temperature scaling
  selective    accuracy vs. coverage for confidence thresholds (choice of min_confidence)
  ood          out-of-distribution scores (MSP, energy, centroid, kNN)

    python scripts/run_experiments.py [--sections general hierarchy ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from contextlens.config import PATHS, RANDOM_SEED
from contextlens.data.dataset import BenchmarkData as Data
from contextlens.evaluation.metrics import (
    multiclass_report,
    multilabel_report,
    ood_report,
    reliability_curve,
)
from contextlens.logging_setup import setup_logging
from contextlens.models.encoders import SentenceEncoder, available_encoders, cached_encode
from contextlens.models.featurizers import TFIDF_FEATURIZERS
from contextlens.models.heads import (
    FlatSubtopicSoftmax,
    HierarchicalSubtopics,
    MultiLabelHead,
    calibrated_softmax,
    decide_subtopics,
    fit_grouped_temperature,
    fit_softmax,
    fit_temperature,
    grouped_probs,
    joint_subtopic_probs,
)
from contextlens.reproducibility import seed_everything

log = logging.getLogger("experiments")
OUT = PATHS.reports / "experiments"
EMB_CACHE = PATHS.data_processed / "embeddings"
C_GRID_TFIDF = [1.0, 4.0, 16.0]
C_GRID_EMB = [0.5, 2.0, 8.0, 32.0]
NB_ALPHAS = [0.01, 0.1, 0.5]
SVM_C = [0.1, 0.5, 2.0]
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.71, 0.05)]
LATENCY_SAMPLES = 200


OUT_SUFFIX = ""


def dump(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}{OUT_SUFFIX}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compact(report: dict) -> dict:
    keys = ("n", "accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "ece", "log_loss")
    return {k: round(report[k], 4) if isinstance(report[k], float) else report[k] for k in keys if k in report}


# ------------------------------------------------------------------ featurizers
def embed(data: Data, key: str, split: str) -> np.ndarray:
    enc = _encoder(key)
    return cached_encode(enc, data.text[split], EMB_CACHE, split)


_ENCODER_CACHE: dict[str, SentenceEncoder] = {}


def _encoder(key: str) -> SentenceEncoder:
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = SentenceEncoder(key)
    return _ENCODER_CACHE[key]


def encoder_size_mb(key: str) -> float:
    model = _encoder(key).model
    return round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1)


def features(data: Data, feat: str, splits: list[str]) -> tuple[dict[str, object], object | None, float]:
    """Return per-split feature matrices, the fitted vectorizer (TF-IDF only), fit seconds."""
    t0 = time.perf_counter()
    if feat in TFIDF_FEATURIZERS:
        vec = TFIDF_FEATURIZERS[feat]()
        X = {"train": vec.fit_transform(data.text["train"])}  # fit on TRAIN only (no leakage)
        for s in splits:
            if s != "train":
                X[s] = vec.transform(data.text[s])
        return X, vec, time.perf_counter() - t0
    X = {s: embed(data, feat, s) for s in ["train", *splits] if s}
    return X, None, time.perf_counter() - t0


# ---------------------------------------------------------------- section: general
def latency(predict_one, texts: list[str]) -> dict:
    sample = texts[:LATENCY_SAMPLES]
    predict_one(sample[:5])  # warm-up
    times = []
    for t in sample:
        t0 = time.perf_counter()
        predict_one([t])
        times.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    predict_one(sample)
    batch = (time.perf_counter() - t0) * 1000 / len(sample)
    return {
        "single_ms_median": round(float(np.median(times)), 2),
        "single_ms_p95": round(float(np.percentile(times, 95)), 2),
        "batch_ms_per_text": round(batch, 3),
    }


def general_candidates(feat: str) -> list[tuple[str, dict, object]]:
    if feat in TFIDF_FEATURIZERS:
        out: list[tuple[str, dict, object]] = []
        out += [("logreg", {"C": c}, lambda X, y, c=c: fit_softmax(X, y, c)) for c in C_GRID_TFIDF]
        if feat == "tfidf-word":
            out += [
                ("multinomial_nb", {"alpha": a}, lambda X, y, a=a: MultinomialNB(alpha=a).fit(X, y)) for a in NB_ALPHAS
            ]
            out += [
                ("complement_nb", {"alpha": a}, lambda X, y, a=a: ComplementNB(alpha=a).fit(X, y)) for a in NB_ALPHAS
            ]
            out += [
                (
                    "linear_svm_platt",
                    {"C": c},
                    lambda X, y, c=c: CalibratedClassifierCV(
                        LinearSVC(C=c, class_weight="balanced", random_state=RANDOM_SEED), method="sigmoid", cv=3
                    ).fit(X, y),
                )
                for c in SVM_C
            ]
        return out
    return [("logreg", {"C": c}, lambda X, y, c=c: fit_softmax(X, y, c)) for c in C_GRID_EMB]


def section_general(data: Data, feats: list[str] | None = None) -> dict:
    labels = data.space.general_ids
    eval_splits = ["val", "se_ext_dev"]
    results: dict[str, dict] = {}
    majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(data.yg["train"]), 1)), data.yg["train"])
    results["majority"] = {
        "feature": "-",
        "head": "majority",
        "params": {},
        **{
            s: compact(multiclass_report(data.yg[s], majority.predict_proba(np.zeros((len(data.yg[s]), 1))), labels))
            for s in eval_splits
        },
    }
    for feat in feats or [*TFIDF_FEATURIZERS, *available_encoders()]:
        X, vec, fit_feat_s = features(data, feat, eval_splits)
        best_by_head: dict[str, dict] = {}
        for head, params, fit in general_candidates(feat):
            t0 = time.perf_counter()
            clf = fit(X["train"], data.yg["train"])
            train_s = time.perf_counter() - t0
            val = multiclass_report(data.yg["val"], clf.predict_proba(X["val"]), labels)
            cur = best_by_head.get(head)
            log.info("%s %s %s val acc=%.4f mF1=%.4f", feat, head, params, val["accuracy"], val["macro_f1"])
            if cur is None or val["macro_f1"] > cur["val_macro_f1"]:
                best_by_head[head] = {"val_macro_f1": val["macro_f1"], "clf": clf, "params": params, "train_s": train_s}
        for head, best in best_by_head.items():
            clf = best["clf"]
            name = f"{feat}|{head}"
            row = {
                "feature": feat,
                "head": head,
                "params": best["params"],
                "train_seconds": round(best["train_s"] + (fit_feat_s if vec is not None else 0), 1),
            }
            for s in eval_splits:
                row[s] = compact(multiclass_report(data.yg[s], clf.predict_proba(X[s]), labels))
            train_rep = multiclass_report(data.yg["train"], clf.predict_proba(X["train"]), labels)
            row["train"] = compact(train_rep)
            if vec is not None:
                row["size_mb"] = round(len(pickle.dumps((vec, clf))) / 1e6, 2)
                row["latency"] = latency(lambda t, v=vec, c=clf: c.predict_proba(v.transform(t)), data.text["val"])
            else:
                row["size_mb"] = round(encoder_size_mb(feat) + len(pickle.dumps(clf)) / 1e6, 2)
                row["latency"] = latency(
                    lambda t, f=feat, c=clf: c.predict_proba(_encoder(f).encode(t)), data.text["val"]
                )
            results[name] = row
            log.info("%s -> val %s | ext_dev %s", name, row["val"], row["se_ext_dev"])
    dump("general", results)
    return results


# --------------------------------------------------------------- section: ensemble
ENSEMBLE_PAIRS = [("bge-small", "e5-small"), ("bge-small", "mpnet-base"), ("e5-small", "mpnet-base")]


def section_ensemble(data: Data) -> dict:
    """Concatenated embeddings of two encoders + one logistic-regression head."""
    labels = data.space.general_ids
    eval_splits = ["val", "se_ext_dev"]
    results: dict[str, dict] = {}
    pairs = list(ENSEMBLE_PAIRS)
    if "minilm-l6-ft" in available_encoders():
        pairs.append(("minilm-l6-ft", "bge-small"))
    for a, b in pairs:
        X = {s: np.hstack([embed(data, a, s), embed(data, b, s)]) for s in ["train", *eval_splits]}
        best = None
        for c in C_GRID_EMB:
            t0 = time.perf_counter()
            clf = fit_softmax(X["train"], data.yg["train"], c)
            train_s = time.perf_counter() - t0
            val = multiclass_report(data.yg["val"], clf.predict_proba(X["val"]), labels)
            log.info("%s+%s C=%s val mF1=%.4f", a, b, c, val["macro_f1"])
            if best is None or val["macro_f1"] > best[0]:
                best = (val["macro_f1"], c, clf, train_s)
        assert best is not None
        _, c, clf, train_s = best
        row: dict = {"feature": f"{a}+{b}", "head": "logreg", "params": {"C": c}, "train_seconds": round(train_s, 1)}
        for s in eval_splits:
            row[s] = compact(multiclass_report(data.yg[s], clf.predict_proba(X[s]), labels))
        row["train"] = compact(multiclass_report(data.yg["train"], clf.predict_proba(X["train"]), labels))
        row["size_mb"] = round(encoder_size_mb(a) + encoder_size_mb(b) + len(pickle.dumps(clf)) / 1e6, 2)
        row["latency"] = latency(
            lambda t, c=clf, a=a, b=b: c.predict_proba(np.hstack([_encoder(a).encode(t), _encoder(b).encode(t)])),
            data.text["val"],
        )
        results[f"{a}+{b}|logreg"] = row
        log.info("%s+%s|logreg -> val %s | ext_dev %s", a, b, row["val"], row["se_ext_dev"])
    dump("ensemble", results)
    return results


# --------------------------------------------------------------- section: zeroshot
def label_descriptions(data: Data) -> list[str]:
    out = []
    for g in data.tax.generals:
        subs = ", ".join(s.phrase for s in g.subtopics)
        out.append(f"{g.name}: a text about {g.phrase}, including {subs}.")
    return out


def section_zeroshot(data: Data) -> dict:
    labels = data.space.general_ids
    results = {}
    for key in ("minilm-l6", "bge-small", "e5-small", "mpnet-base"):
        enc = _encoder(key)
        L = enc.encode(label_descriptions(data))
        row = {}
        for s in ("val", "se_ext_dev"):
            sims = embed(data, key, s) @ L.T
            row[s] = compact(multiclass_report(data.yg[s], softmax(sims / 0.05, axis=1), labels))
        results[f"{key}|label-similarity"] = row
        log.info("zero-shot %s: %s", key, row)
    dump("zeroshot", results)
    return results


# -------------------------------------------------------------- section: hierarchy
def section_hierarchy(data: Data, feats: list[str]) -> dict:
    space = data.space
    gids, sids, parent = space.general_ids, space.subtopic_ids, space.parent_col
    sub_idx = space.sub_index
    children = {g: data.tax.children_of(g) for g in gids}
    eval_splits = ["val", "sesub_ext_dev"]
    results: dict[str, dict] = {}
    for feat in feats:
        X, _, _ = features(data, feat, eval_splits)
        C = 16.0 if feat in TFIDF_FEATURIZERS else 8.0
        gen = fit_softmax(X["train"], data.yg["train"], C)
        # --- H1: hierarchical (general head + one multi-label head per general)
        hier = HierarchicalSubtopics(children, C).fit(X["train"], data.yg["train"], data.ys["train"], sub_idx, gids)
        # --- H2: flat multi-label over all subtopics
        flat = MultiLabelHead(sids, C).fit(X["train"], data.ys["train"])
        # --- H3: flat softmax over primary subtopic, hierarchy derived
        primary = data.ys["train"].argmax(axis=1)
        flat_sm = fit_softmax(X["train"], primary, C)

        def run(variant: str, split: str, X=X, gen=gen, hier=hier, flat=flat, flat_sm=flat_sm) -> tuple:
            if variant == "hierarchical":
                gp = gen.predict_proba(X[split])
                cond = hier.conditional(X[split], sub_idx, len(sids))
                return gp, cond, joint_subtopic_probs(gp, cond, parent)
            if variant == "flat_multilabel":
                sp = flat.predict_proba(X[split])
                gp = np.stack([sp[:, parent == g].max(axis=1) for g in range(len(gids))], axis=1)
                gp = gp / gp.sum(axis=1, keepdims=True)
                return gp, sp, sp
            sm = np.zeros((X[split].shape[0], len(sids)))
            sm[:, flat_sm.classes_] = flat_sm.predict_proba(X[split])
            gp = np.stack([sm[:, parent == g].sum(axis=1) for g in range(len(gids))], axis=1)
            cond = sm / np.maximum(gp[:, parent], 1e-12)
            return gp, cond, sm

        for variant in ("hierarchical", "flat_multilabel", "flat_softmax"):
            gp_val, cond_val, _ = run(variant, "val")
            sweep = []
            for thr in THRESHOLDS:
                dec = decide_subtopics(cond_val, parent, gp_val.argmax(1), thr)
                rep = multilabel_report(data.ys["val"], dec, cond_val, sids)
                sweep.append(
                    {
                        "threshold": thr,
                        "macro_f1": round(rep["macro_f1"], 4),
                        "micro_f1": round(rep["micro_f1"], 4),
                        "samples_f1": round(rep["samples_f1"], 4),
                    }
                )
            best_thr = max(sweep, key=lambda r: (r["macro_f1"], -r["threshold"]))["threshold"]
            row: dict = {"threshold": best_thr, "sweep_val": sweep}
            for s in eval_splits:
                gp, cond, joint = run(variant, s)
                dec = decide_subtopics(cond, parent, gp.argmax(1), best_thr)
                ml = multilabel_report(data.ys[s], dec, joint, sids)
                row[s] = {
                    "general": compact(multiclass_report(data.yg[s], gp, gids)),
                    "subtopics": {k: round(v, 4) for k, v in ml.items() if isinstance(v, float)},
                }
                if s == "val":
                    row[s]["subtopics_per_label"] = ml["per_label"]
            results[f"{feat}|{variant}"] = row
            log.info("%s %s thr=%.2f val=%s", feat, variant, best_thr, row["val"])
    dump("hierarchy", results)
    return results


# ------------------------------------------------------------ section: calibration
def general_scorers(data: Data, feat: str, X: dict) -> dict:
    """Calibrated P(general) for both production head types (fitted on train, T on val).

    ``logreg``: 8-way softmax head + temperature. ``flat_softmax``: 28-way subtopic
    softmax + temperature fitted on the general NLL, P(general) = sum of children.
    Returns name -> (temperature, raw-probs fn, calibrated-probs fn).
    """
    C = 16.0 if feat in TFIDF_FEATURIZERS else 8.0
    parent, n_gen = data.space.parent_col, len(data.space.general_ids)
    clf = fit_softmax(X["train"], data.yg["train"], C)
    T = fit_temperature(clf.decision_function(X["val"]), data.yg["val"])
    flat = FlatSubtopicSoftmax(len(data.space.subtopic_ids), C).fit(X["train"], data.ys["train"])
    Tf = fit_grouped_temperature(flat.logits(X["val"]), data.yg["val"], parent)
    return {
        "logreg": (
            T,
            lambda Z, clf=clf: softmax(clf.decision_function(Z), axis=1),
            lambda Z, clf=clf, T=T: calibrated_softmax(clf.decision_function(Z), T),
        ),
        "flat_softmax": (
            Tf,
            lambda Z, flat=flat: grouped_probs(flat.logits(Z), 1.0, parent, n_gen)[0],
            lambda Z, flat=flat, Tf=Tf: grouped_probs(flat.logits(Z), Tf, parent, n_gen)[0],
        ),
    }


def section_calibration(data: Data, feats: list[str]) -> dict:
    gids = data.space.general_ids
    splits = ("val", "se_ext_dev")
    results = {}
    for feat in feats:
        X, _, _ = features(data, feat, list(splits))
        for head, (T, raw_fn, cal_fn) in general_scorers(data, feat, X).items():
            row: dict = {"temperature": round(T, 4)}
            for s in splits:
                raw_p, cal_p = raw_fn(X[s]), cal_fn(X[s])
                raw = multiclass_report(data.yg[s], raw_p, gids)
                cal = multiclass_report(data.yg[s], cal_p, gids)
                row[s] = {
                    "ece_raw": round(raw["ece"], 4),
                    "ece_temp": round(cal["ece"], 4),
                    "nll_raw": round(raw["log_loss"], 4),
                    "nll_temp": round(cal["log_loss"], 4),
                    "reliability_raw": reliability_curve(raw_p, data.yg[s]),
                    "reliability_temp": reliability_curve(cal_p, data.yg[s]),
                }
            name = feat if head == "logreg" else f"{feat}|flat_softmax"
            results[name] = row
            log.info(
                "calibration %s: T=%.3f %s",
                name,
                T,
                {s: (row[s]["ece_raw"], row[s]["ece_temp"]) for s in splits},
            )
    dump("calibration", results)
    return results


# ------------------------------------------------------------- section: selective
CONFIDENCE_GRID = [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]


def section_selective(data: Data, feats: list[str]) -> dict:
    """Accuracy of the kept predictions vs. the share kept, per confidence threshold."""
    results: dict = {}
    for feat in feats:
        X, _, _ = features(data, feat, ["val", "se_ext_dev"])
        for head, (T, _raw, cal_fn) in general_scorers(data, feat, X).items():
            row: dict = {"temperature": round(T, 4)}
            for split in ("val", "se_ext_dev"):
                probs = cal_fn(X[split])
                conf, correct = probs.max(axis=1), probs.argmax(axis=1) == data.yg[split]
                row[split] = [
                    {
                        "min_confidence": t,
                        "coverage": round(float(np.mean(conf >= t)), 4),
                        "accuracy_kept": round(float(np.mean(correct[conf >= t])), 4) if (conf >= t).any() else None,
                        "accuracy_rejected": round(float(np.mean(correct[conf < t])), 4) if (conf < t).any() else None,
                    }
                    for t in CONFIDENCE_GRID
                ]
            name = feat if head == "logreg" else f"{feat}|flat_softmax"
            results[name] = row
            log.info("selective %s: %s", name, row["se_ext_dev"])
    dump("selective", results)
    return results


# --------------------------------------------------------------------- section: ood
def section_ood(data: Data, feats: list[str]) -> dict:
    results = {}
    pairs = {
        "wiki_val": ("val", "ood_val"),
        "se_ext_dev": ("se_ext_dev", "se_ood_ext_dev"),
    }
    for feat in feats:
        splits = sorted({s for p in pairs.values() for s in p})
        X, _, _ = features(data, feat, splits)
        C = 16.0 if feat in TFIDF_FEATURIZERS else 8.0
        clf = fit_softmax(X["train"], data.yg["train"], C)
        T = fit_temperature(clf.decision_function(X["val"]), data.yg["val"])
        scorers = {
            "msp": lambda Z, clf=clf, T=T: calibrated_softmax(clf.decision_function(Z), T).max(axis=1),
            "energy": lambda Z, clf=clf, T=T: logsumexp(clf.decision_function(Z) / T, axis=1),
        }
        if feat not in TFIDF_FEATURIZERS:
            Xtr = X["train"]
            cents = np.stack([Xtr[data.yg["train"] == g].mean(axis=0) for g in range(len(data.space.general_ids))])
            cents /= np.linalg.norm(cents, axis=1, keepdims=True)
            scorers["centroid_cos"] = lambda Z, cents=cents: (Z @ cents.T).max(axis=1)
            scorers["knn10_cos"] = lambda Z, Xtr=Xtr: np.sort(Z @ Xtr.T, axis=1)[:, -10:].mean(axis=1)
        row: dict = {}
        for sname, fn in scorers.items():
            row[sname] = {}
            # Two calibration choices for the threshold (keep 95% of in-distribution texts):
            # Wikipedia validation passages vs. real user questions (Stack Exchange ext_dev).
            thresholds = {
                "wiki_val": float(np.quantile(fn(X["val"]), 0.05)),
                "se_ext_dev": float(np.quantile(fn(X["se_ext_dev"]), 0.05)),
            }
            row[sname]["thresholds_at_95"] = {k: round(v, 4) for k, v in thresholds.items()}
            for pname, (ind, ood) in pairs.items():
                a, b = fn(X[ind]), fn(X[ood])
                rep = ood_report(a, b)
                for tname, thr in thresholds.items():
                    rep[f"id_kept@{tname}"] = round(float(np.mean(a >= thr)), 4)
                    rep[f"ood_flagged@{tname}"] = round(float(np.mean(b < thr)), 4)
                row[sname][pname] = {k: round(v, 4) if isinstance(v, float) else v for k, v in rep.items()}
            log.info("ood %s %s: %s", feat, sname, {p: row[sname][p]["auroc"] for p in pairs})
        results[feat] = row
    dump("ood", results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sections", nargs="+", default=["general", "zeroshot", "hierarchy", "calibration", "ood"])
    parser.add_argument(
        "--feats",
        nargs="+",
        default=["tfidf-word+char", "bge-small", "mpnet-base"],
        help="featurizers for the hierarchy/calibration/ood sections",
    )
    parser.add_argument(
        "--general-feats", nargs="+", default=None, help="featurizers for the general section (default: all)"
    )
    parser.add_argument("--out-suffix", default="", help="suffix for the output JSON names (partial re-runs)")
    args = parser.parse_args()
    global OUT_SUFFIX
    OUT_SUFFIX = args.out_suffix
    setup_logging("INFO", PATHS.root / "logs" / "experiments.log")
    seed_everything(RANDOM_SEED)
    data = Data()
    for section in args.sections:
        t0 = time.perf_counter()
        if section == "general":
            section_general(data, args.general_feats)
        elif section == "zeroshot":
            section_zeroshot(data)
        elif section == "ensemble":
            section_ensemble(data)
        elif section == "hierarchy":
            section_hierarchy(data, args.feats)
        elif section == "calibration":
            section_calibration(data, args.feats)
        elif section == "selective":
            section_selective(data, args.feats)
        elif section == "ood":
            section_ood(data, args.feats)
        else:
            parser.error(f"unknown section {section}")
        log.info("section %s done in %.0fs", section, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
