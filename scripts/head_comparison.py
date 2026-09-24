"""Subtopic head comparison: softmax vs genuine multi-label heads (docs/HARDENING.md, item 1).

About 6% of the passages carry two or three subtopics (always siblings under
one general topic). This script compares, on the same fine-tuned encoder
embeddings (configs/model.json ``encoder``):

  flat_softmax      one 28-way softmax over the primary subtopic; P(general) is
                    the sum of its children; siblings are added when their share
                    of the parent mass passes a threshold (the v1.0 head)
  flat_sigmoid      general softmax head + 28 independent one-vs-rest sigmoid
                    outputs (binary cross-entropy per label) over all subtopics
  hier_sigmoid      general softmax head + one set of sigmoid outputs per general
                    topic trained only on that topic's rows: P(s) = P(g) P(s | g)

Every candidate uses the same decision (children of the predicted general
topic; the best child always, further children at >= threshold), so the
comparison isolates the learned objective. For each candidate the
regularisation C and the threshold are chosen on the validation split.

Development data only: Wikipedia ``val`` (and its multi-label subset) and Stack
Exchange ``se_ext_dev`` / ``sesub_ext_dev``. ``test`` / ``ext_test`` are not
read (they are evaluated once after the freeze, evaluate.py --stage locked).

Decision rule (fixed before running):
  1. score = mean(val subtopic macro-F1, sesub_ext_dev macro-F1 over supported labels)
  2. eligible = general macro-F1 on val and on se_ext_dev each within 0.01 of the best candidate
  3. pick the highest score; within 0.005 prefer the higher share of multi-label
     validation passages where at least two gold subtopics are predicted (the
     property the audit asked for), then lower general ECE
     on se_ext_dev, then lower latency.

    python scripts/head_comparison.py        # -> reports/experiments/head_comparison.json
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from contextlens.config import PATHS
from contextlens.data.dataset import BenchmarkData
from contextlens.evaluation.metrics import multiclass_report, multilabel_report
from contextlens.logging_setup import setup_logging
from contextlens.models.encoders import SentenceEncoder, cached_encode
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
)

log = logging.getLogger("head_comparison")

C_GRID = [1.0, 4.0, 8.0, 16.0]
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.96, 0.05)]
DEV = ("val", "se_ext_dev", "sesub_ext_dev")
TOL_GENERAL, TOL_SCORE = 0.01, 0.005


class Candidate:
    """Fitted head: ``scores(X)`` -> (P(general), P(sub | general), joint P(sub))."""

    def __init__(self, name: str, C: float, data: dict, space, children) -> None:
        self.name, self.C = name, C
        X, yg, Ys, Xva, ygva = (
            data["X"]["train"],
            data["yg"]["train"],
            data["Ys"]["train"],
            data["X"]["val"],
            data["yg"]["val"],
        )
        pc, n_g, n_s = space.parent_col, len(space.general_ids), len(space.subtopic_ids)
        self.space = space
        t0 = time.perf_counter()
        if name == "flat_softmax":
            self.head = FlatSubtopicSoftmax(n_s, C).fit(X, Ys)
            self.T = fit_grouped_temperature(self.head.logits(Xva), ygva, pc)
        else:
            self.general = fit_softmax(X, yg, data["general_C"])
            self.T = fit_temperature(self.general.decision_function(Xva), ygva)
            if name == "flat_sigmoid":
                self.head = MultiLabelHead(space.subtopic_ids, C).fit(X, Ys)
            else:
                self.head = HierarchicalSubtopics(children, C).fit(X, yg, Ys, space.sub_index, space.general_ids)
        self.fit_seconds = round(time.perf_counter() - t0, 1)
        self.n_g, self.n_s = n_g, n_s

    def scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pc = self.space.parent_col
        if self.name == "flat_softmax":
            gp, cond = grouped_probs(self.head.logits(X), self.T, pc, self.n_g)
            return gp, cond, gp[:, pc] * cond
        gp = calibrated_softmax(self.general.decision_function(X), self.T)
        if self.name == "flat_sigmoid":
            sig = self.head.predict_proba(X)
            return gp, sig, gp[:, pc] * sig
        cond = self.head.conditional(X, self.space.sub_index, self.n_s)
        return gp, cond, gp[:, pc] * cond

    def size_bytes(self) -> int:
        parts = [self.head] + ([self.general] if self.name != "flat_softmax" else [])
        return len(pickle.dumps(parts))


def multi_label_rows(Y: np.ndarray, pred: np.ndarray) -> dict:
    """Behaviour on passages with >= 2 gold subtopics, and extra labels on single-label ones."""
    rows = Y.sum(axis=1) >= 2
    single = Y.sum(axis=1) == 1
    if not rows.any():
        return {"n": 0}
    return {
        "n": int(rows.sum()),
        # share of gold labels predicted, over the multi-label rows (micro)
        "label_recall": round(float((pred[rows] * Y[rows]).sum() / Y[rows].sum()), 4),
        # share of multi-label rows where more than one gold label is predicted
        "rows_with_2plus_gold_found": round(float(np.mean((pred[rows] * Y[rows]).sum(axis=1) >= 2)), 4),
        "subset_accuracy": round(float(np.mean((Y[rows] == pred[rows]).all(axis=1))), 4),
        "single_label_rows_with_extra_labels": round(float(np.mean(pred[single].sum(axis=1) > 1)), 4),
    }


def round_floats(d: dict) -> dict:
    return {k: round(v, 4) for k, v in d.items() if isinstance(v, float)}


def evaluate(cand: Candidate, data: dict, thr: float) -> dict:
    out: dict = {}
    for split in DEV:
        gp, cond, joint = cand.scores(data["X"][split])
        row = {"general": round_floats(multiclass_report(data["yg"][split], gp, cand.space.general_ids))}
        if split in data["Ys"]:
            Y = data["Ys"][split]
            dec = decide_subtopics(cond, cand.space.parent_col, gp.argmax(1), thr)
            ml = multilabel_report(Y, dec, joint, cand.space.subtopic_ids)
            row["subtopics"] = round_floats(ml)
            row["subtopics_per_label"] = {
                k: {m: round(x, 4) if isinstance(x, float) else x for m, x in v.items()}
                for k, v in ml["per_label"].items()
            }
            row["multi_label_subset"] = multi_label_rows(Y, dec)
            row["subtopic_brier"] = round(float(np.mean((joint - Y) ** 2)), 5)
        out[split] = row
    X1 = data["X"]["val"][:1]
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        cand.scores(X1)
        times.append((time.perf_counter() - t0) * 1000)
    out["head_latency_ms_median"] = round(float(np.median(times)), 3)
    out["head_size_kb"] = round(cand.size_bytes() / 1024, 1)
    out["fit_seconds"] = cand.fit_seconds
    return out


def main() -> int:
    setup_logging("INFO")
    conf = json.loads((PATHS.root / "configs" / "model.json").read_text(encoding="utf-8"))
    bench = BenchmarkData()
    space = bench.space
    children = {g.id: [s.id for s in g.subtopics] for g in bench.tax.generals}
    enc = SentenceEncoder(conf["encoder"])
    cache = PATHS.data_processed / "embeddings"  # same cache as scripts/run_experiments.py
    splits = ("train", *DEV)
    data = {
        "X": {s: cached_encode(enc, bench.text[s], cache, s) for s in splits},
        "yg": {s: bench.yg[s] for s in splits},
        "Ys": {s: bench.ys[s] for s in ("train", "val", "sesub_ext_dev")},
    }
    # the general head of the sigmoid candidates: C chosen on val general macro-F1
    gscores = {}
    for C in C_GRID:
        clf = fit_softmax(data["X"]["train"], data["yg"]["train"], C)
        gp = calibrated_softmax(clf.decision_function(data["X"]["val"]), 1.0)
        gscores[C] = multiclass_report(data["yg"]["val"], gp, space.general_ids)["macro_f1"]
    data["general_C"] = max(C_GRID, key=lambda c: (gscores[c], -c))
    log.info("general C sweep %s -> %s", gscores, data["general_C"])

    results = {}
    for name in ("flat_softmax", "flat_sigmoid", "hier_sigmoid"):
        best = None
        sweep = []
        for C in C_GRID:
            cand = Candidate(name, C, data, space, children)
            gp, cond, _ = cand.scores(data["X"]["val"])
            for thr in THRESHOLDS:
                dec = decide_subtopics(cond, space.parent_col, gp.argmax(1), thr)
                f = multilabel_report(data["Ys"]["val"], dec, cond, space.subtopic_ids)["macro_f1"]
                sweep.append({"C": C, "threshold": thr, "val_macro_f1": round(f, 4)})
                if best is None or f > best[0]:
                    best = (f, C, thr, cand)
            log.info("%s C=%s done", name, C)
        _, C, thr, cand = best
        res = evaluate(cand, data, thr)
        results[name] = {"C": C, "threshold": thr, "temperature": round(cand.T, 4), **res, "sweep": sweep}
        log.info(
            "%s C=%s thr=%s val sub macroF1=%.4f sesub=%.4f",
            name, C, thr, res["val"]["subtopics"]["macro_f1"], res["sesub_ext_dev"]["subtopics"]["macro_f1_supported"],
        )  # fmt: skip

    def score(r: dict) -> float:
        return (r["val"]["subtopics"]["macro_f1"] + r["sesub_ext_dev"]["subtopics"]["macro_f1_supported"]) / 2

    best_g = {s: max(r[s]["general"]["macro_f1"] for r in results.values()) for s in ("val", "se_ext_dev")}
    eligible = {
        n: r for n, r in results.items() if all(r[s]["general"]["macro_f1"] >= best_g[s] - TOL_GENERAL for s in best_g)
    }
    top = max(score(r) for r in eligible.values())
    close = [n for n, r in eligible.items() if score(r) >= top - TOL_SCORE]
    chosen = max(
        close,
        key=lambda n: (
            results[n]["val"]["multi_label_subset"]["rows_with_2plus_gold_found"],
            -results[n]["se_ext_dev"]["general"]["ece"],
            -results[n]["head_latency_ms_median"],
        ),
    )
    summary = {
        n: {
            "score": round(score(r), 4),
            "eligible": n in eligible,
            "val_general_macro_f1": r["val"]["general"]["macro_f1"],
            "se_ext_dev_general_macro_f1": r["se_ext_dev"]["general"]["macro_f1"],
            "se_ext_dev_general_ece": r["se_ext_dev"]["general"]["ece"],
            "val_sub_macro_f1": r["val"]["subtopics"]["macro_f1"],
            "val_sub_micro_f1": r["val"]["subtopics"]["micro_f1"],
            "sesub_ext_dev_macro_f1_supported": r["sesub_ext_dev"]["subtopics"]["macro_f1_supported"],
            "val_multilabel_rows_2plus_found": r["val"]["multi_label_subset"]["rows_with_2plus_gold_found"],
            "val_extra_labels_on_single": r["val"]["multi_label_subset"]["single_label_rows_with_extra_labels"],
            "head_latency_ms": r["head_latency_ms_median"],
            "head_size_kb": r["head_size_kb"],
        }
        for n, r in results.items()
    }
    out = {
        "protocol": __doc__,
        "encoder": conf["encoder"],
        "general_C_sigmoid_candidates": data["general_C"],
        "summary": summary,
        "chosen": chosen,
        "results": results,
    }
    target = PATHS.reports / "experiments" / "head_comparison.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
