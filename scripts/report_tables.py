"""Render benchmark JSON (reports/experiments/*.json, reports/evaluation.json) as Markdown tables.

    python scripts/report_tables.py        # writes reports/tables.md

The model report and final report copy their tables from this output, so no
number in the documentation is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlens.config import PATHS

EXP = PATHS.reports / "experiments"


# Partial re-runs write <section><suffix>.json (run_experiments.py --out-suffix);
# "_ft" holds the runs with the fine-tuned encoder. They are merged into one table.
SUFFIXES = ("", "_ft")


def load(name: str) -> dict | None:
    merged: dict = {}
    for suffix in SUFFIXES:
        path = EXP / f"{name}{suffix}.json"
        if path.exists():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    return merged or None


def f(x: float | None, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def general_table(general: dict) -> str:
    rows = []
    ordered = sorted(general.items(), key=lambda kv: -kv[1]["se_ext_dev"]["macro_f1"])
    for name, r in ordered:
        lat = r.get("latency", {})
        rows.append(
            [
                f"`{name}`",
                ", ".join(f"{k}={v}" for k, v in r.get("params", {}).items()) or "–",
                f(r["val"]["accuracy"]),
                f(r["val"]["macro_f1"]),
                f(r["val"].get("ece")),
                f(r["se_ext_dev"]["accuracy"]),
                f(r["se_ext_dev"]["macro_f1"]),
                f(r["test"]["macro_f1"]),
                f(r["se_ext_test"]["macro_f1"]),
                f(r.get("train", {}).get("macro_f1")),
                f(lat.get("single_ms_median"), 1),
                f(r.get("size_mb"), 1),
                f(r.get("train_seconds"), 0),
            ]
        )
    return table(
        [
            "model",
            "params",
            "val acc",
            "val macro-F1",
            "val ECE",
            "ext_dev acc",
            "ext_dev macro-F1",
            "test macro-F1 †",
            "ext_test macro-F1 †",
            "train macro-F1",
            "ms / text",
            "MB",
            "train s",
        ],
        rows,
    )


def hierarchy_table(hier: dict) -> str:
    rows = []
    for name, r in hier.items():
        v, d = r["val"]["subtopics"], r["sesub_ext_dev"]["subtopics"]
        rows.append(
            [
                f"`{name}`",
                f(r["threshold"], 2),
                f(v["macro_f1"]),
                f(v["micro_f1"]),
                f(v["samples_f1"]),
                f(v["hamming_loss"], 4),
                f(v["subset_accuracy"]),
                f(v["precision_at_1"]),
                f(v["recall_at_3"]),
                f(d.get("macro_f1_supported", d["macro_f1"])),
                f(d["micro_f1"]),
                f(r["val"]["general"]["macro_f1"]),
                f(r["test"]["subtopics"]["macro_f1"]),
            ]
        )
    return table(
        [
            "model | variant",
            "τ",
            "val macro-F1",
            "val micro-F1",
            "val samples-F1",
            "val Hamming",
            "val subset acc",
            "val P@1",
            "val R@3",
            "ext_dev macro-F1 (25 labels)",
            "ext_dev micro-F1",
            "val general macro-F1",
            "test macro-F1 †",
        ],
        rows,
    )


def calibration_table(cal: dict) -> str:
    rows = []
    for feat, r in cal.items():
        for split in ("val", "test", "se_ext_dev", "se_ext_test"):
            c = r[split]
            rows.append(
                [
                    f"`{feat}`",
                    split,
                    f(r["temperature"], 3),
                    f(c["ece_raw"]),
                    f(c["ece_temp"]),
                    f(c["nll_raw"]),
                    f(c["nll_temp"]),
                ]
            )
    return table(["model", "split", "T", "ECE raw", "ECE after T", "NLL raw", "NLL after T"], rows)


def ood_table(ood: dict) -> str:
    rows = []
    for feat, scorers in ood.items():
        for sname, r in scorers.items():
            thr = r["thresholds_at_95"]
            for pair in ("wiki_val", "wiki_test", "se_ext_dev", "se_ext_test"):
                p = r[pair]
                rows.append(
                    [
                        f"`{feat}`",
                        sname,
                        pair,
                        f(p["auroc"]),
                        f(p["fpr_at_95_tpr"]),
                        f(p.get("id_kept@se_ext_dev")),
                        f(p.get("ood_flagged@se_ext_dev")),
                        f(p.get("id_kept@wiki_val")),
                        f(p.get("ood_flagged@wiki_val")),
                    ]
                )
            rows.append(
                [
                    f"`{feat}`",
                    sname,
                    "thresholds",
                    f"wiki_val={f(thr['wiki_val'], 4)}",
                    f"se_ext_dev={f(thr['se_ext_dev'], 4)}",
                    "",
                    "",
                    "",
                    "",
                ]
            )
    return table(
        [
            "model",
            "score",
            "pair",
            "AUROC",
            "FPR@95TPR",
            "ID kept @SE-thr",
            "OOD flagged @SE-thr",
            "ID kept @wiki-thr",
            "OOD flagged @wiki-thr",
        ],
        rows,
    )


def simple_rows(name: str, res: dict, splits: tuple[str, ...]) -> list[list[str]]:
    return [[f"`{name}`", s, f(res[s]["accuracy"]), f(res[s]["macro_f1"])] for s in splits if s in res]


# ------------------------------------------------------------------ experiment log
FEATURE_TEXT = {
    "-": "none (predicts the most frequent training class)",
    "tfidf-word": "TF-IDF word 1-2-grams (lower-cased, sublinear tf, min_df 2, max_df 0.9, <=200k features)",
    "tfidf-word-nostop": "as tfidf-word, English stop words removed",
    "tfidf-word+char": "tfidf-word + TF-IDF char_wb 3-5-grams (<=300k features)",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2 embeddings (frozen, 384-d, L2-normalised)",
    "bge-small": "BAAI/bge-small-en-v1.5 embeddings (frozen, 384-d, L2-normalised)",
    "e5-small": "intfloat/e5-small-v2 embeddings ('query: ' prefix, frozen, 384-d)",
    "mpnet-base": "sentence-transformers/all-mpnet-base-v2 embeddings (frozen, 768-d)",
    "minilm-l6-ft": "all-MiniLM-L6-v2 fine-tuned on the training split (E-5), exported, 384-d",
}
HEAD_TEXT = {
    "majority": "majority-class baseline",
    "logreg": "multinomial logistic regression, class_weight=balanced",
    "multinomial_nb": "multinomial naive Bayes",
    "complement_nb": "complement naive Bayes",
    "linear_svm_platt": "linear SVM (class_weight=balanced) + Platt scaling (3-fold)",
}
GRID_TEXT = {
    "logreg": "C over {1, 4, 16} (TF-IDF) or {0.5, 2, 8, 32} (embeddings)",
    "multinomial_nb": "alpha over {0.01, 0.1, 0.5}",
    "complement_nb": "alpha over {0.01, 0.1, 0.5}",
    "linear_svm_platt": "C over {0.1, 0.5, 2}",
    "majority": "-",
}
# Written after reading the results; no numbers here (they come from the JSON).
DECISION = {
    "majority": "Floor reference only.",
    "tfidf-word|": "Not selected: strong on Wikipedia, weak transfer to real questions (lexical overlap with encyclopedic text).",
    "tfidf-word-nostop|": "Not selected: removing stop words does not close the transfer gap.",
    "tfidf-word+char|": "Not selected: character n-grams add cost without improving transfer; kept as the lexical reference in later sections.",
    "minilm-l6|": "Not selected as a frozen encoder: weakest transfer of the four; used as the base for fine-tuning (E-5).",
    "bge-small|": "Not selected: best frozen encoder on questions, but below the fine-tuned MiniLM on both selection sets and slower.",
    "e5-small|": "Not selected: close to bge-small; below the fine-tuned MiniLM on both selection sets.",
    "mpnet-base|": "Not selected: best frozen encoder on Wikipedia val, not on questions; 4-5x slower and 3x larger than the small encoders.",
    "minilm-l6-ft|": "SELECTED (docs/MODEL_REPORT.md): best on Wikipedia val and on Stack Exchange ext_dev, smallest and fastest encoder.",
    "+": "Not selected: concatenating two encoders doubles latency and size for no gain over the fine-tuned single encoder.",
}


def _metrics(r: dict | None, keys: tuple[str, ...] = ("accuracy", "macro_f1", "weighted_f1", "ece")) -> str:
    if not r:
        return "–"
    names = {"accuracy": "acc", "macro_f1": "macro-F1", "weighted_f1": "weighted-F1", "ece": "ECE"}
    return ", ".join(f"{names[k]} {f(r.get(k))}" for k in keys if k in r)


def _decision(name: str) -> str:
    for key, text in DECISION.items():
        if key == name or (key.endswith("|") and name.startswith(key)) or (key == "+" and "+" in name.split("|")[0]):
            return text
    return "See docs/MODEL_REPORT.md."


def card(eid: str, name: str, r: dict) -> str:
    feat, head = r.get("feature", "-"), r.get("head", "-")
    lat = r.get("latency", {})
    gap = None
    if r.get("train") and r.get("val"):
        gap = r["train"]["macro_f1"] - r["val"]["macro_f1"]
    rows = [
        ["Model", HEAD_TEXT.get(head, head)],
        [
            "Features",
            " + ".join(FEATURE_TEXT.get(x, x) for x in feat.split("+"))
            if feat != "tfidf-word+char"
            else FEATURE_TEXT[feat],
        ],
        [
            "Hyper-parameters",
            f"{', '.join(f'{k}={v}' for k, v in r.get('params', {}).items()) or '-'} (chosen on val macro-F1; grid: {GRID_TEXT.get(head, '-')})",
        ],
        ["Train metrics", _metrics(r.get("train"))],
        [
            "Validation metrics",
            f"Wikipedia val: {_metrics(r.get('val'))} · Stack Exchange ext_dev: {_metrics(r.get('se_ext_dev'))}",
        ],
        [
            "Test metrics †",
            f"Wikipedia test: {_metrics(r.get('test'))} · Stack Exchange ext_test: {_metrics(r.get('se_ext_test'))}",
        ],
        [
            "Training time",
            "–"
            if r.get("train_seconds") is None
            else f"{f(r.get('train_seconds'), 1)} s "
            + (
                "(vectorizer + head fit)"
                if feat.startswith("tfidf")
                else "head fit (embedding extraction is cached and not included)"
            ),
        ],
        [
            "Inference time",
            f"median {f(lat.get('single_ms_median'), 1)} ms / p95 {f(lat.get('single_ms_p95'), 1)} ms per text, batch {f(lat.get('batch_ms_per_text'), 2)} ms/text; size {f(r.get('size_mb'), 1)} MB"
            if lat
            else "–",
        ],
        [
            "Notes",
            "train-val macro-F1 gap "
            + f(gap)
            + (" (memorises the training set)" if gap is not None and gap > 0.1 else "")
            if gap is not None
            else "–",
        ],
        ["Decision", _decision(name)],
    ]
    return f"### {eid} · `{name}`\n\n" + table(["field", "value"], rows)


def experiment_log() -> str:
    parts = [
        "# Experiment log (generated by scripts/report_tables.py)",
        "One card per trained general-topic model, in the format of the project brief. "
        "Protocol, the other experiments (subtopics, calibration, OOD, decay) and the discussion: "
        "docs/EXPERIMENTS.md and docs/MODEL_REPORT.md. † report only - not used for any decision.",
    ]
    counters: dict[str, int] = {}

    def next_id(group: str) -> str:
        counters[group] = counters.get(group, 0) + 1
        return f"{group}.{counters[group]}"

    for name, r in (load("general") or {}).items():
        feat = r.get("feature", "-")
        group = "E-0" if name == "majority" else ("E-1" if feat.startswith("tfidf") else "E-2")
        parts.append(card(next_id(group), name, r))
    for name, r in (load("ensemble") or {}).items():
        parts.append(card(next_id("E-6"), name, r))
    for ft in sorted(EXP.glob("finetune_*.json")):
        r = json.loads(ft.read_text(encoding="utf-8"))
        hp = r["hyperparameters"]
        rows = [
            [
                "Model",
                "shared transformer encoder (mean pooling) + general head (class-weighted cross-entropy) + subtopic head (BCE)",
            ],
            ["Features", FEATURE_TEXT.get(r["encoder"], r["encoder"]) + ", fine-tuned end to end"],
            [
                "Hyper-parameters",
                f"lr {hp['lr']}, epochs {hp['epochs']} (best epoch by val macro-F1), batch {hp['batch']}, max {hp['max_len']} tokens, AdamW, linear warm-up; subtopic threshold {r.get('subtopic_threshold')} (val)",
            ],
            ["Train metrics", "per-epoch training loss: " + ", ".join(f"{h['train_loss']}" for h in r["history"])],
            [
                "Validation metrics",
                f"Wikipedia val: {_metrics(r['val']['general'])}, subtopic macro-F1 {f(r['val']['subtopics'].get('macro_f1'))} · Stack Exchange ext_dev: {_metrics(r['se_ext_dev']['general'])}",
            ],
            [
                "Test metrics †",
                f"Wikipedia test: {_metrics(r['test']['general'])} · Stack Exchange ext_test: {_metrics(r['se_ext_test']['general'])}",
            ],
            [
                "Training time",
                f"{r['train_seconds']} s on {r['device']} ("
                + ", ".join(f"epoch {h['epoch']}: {h['epoch_seconds']} s" for h in r["history"])
                + ")",
            ],
            ["Inference time", f"median {r['latency_single_ms_median']} ms per text; {r['size_mb']} MB"],
            ["Notes", "val macro-F1 by epoch: " + ", ".join(str(h["val_macro_f1"]) for h in r["history"])],
            [
                "Decision",
                "The fine-tuned encoder is exported and used with logistic-regression heads (row minilm-l6-ft above), which keeps the artifact format, temperature scaling and OOD gate of the frozen encoders.",
            ],
        ]
        parts.append(f"### E-5 · `{ft.stem}`\n\n" + table(["field", "value"], rows))
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parts = [
        "# Benchmark tables (generated by scripts/report_tables.py)",
        "† test / ext_test columns are for reporting only; no decision used them.",
    ]
    if (g := load("general")) is not None:
        parts += ["## General topic — featurizer × head (best hyper-parameters on validation)", general_table(g)]
    if (e := load("ensemble")) is not None:
        parts += ["## General topic — concatenated encoder embeddings", general_table(e)]
    if (z := load("zeroshot")) is not None:
        rows = [
            r for name, res in z.items() for r in simple_rows(name, res, ("val", "se_ext_dev", "test", "se_ext_test"))
        ]
        parts += ["## Zero-shot label similarity (no training)", table(["model", "split", "acc", "macro-F1"], rows)]
    if (n := load("zeroshot_nli")) is not None:
        rows = [
            [f"`{n['model']}`", s, f(n[s]["accuracy"]), f(n[s]["macro_f1"]), str(n[s]["n"]), f(n[s]["ms_per_text"], 1)]
            for s in ("val", "se_ext_dev")
            if s in n
        ]
        parts += [
            "## Zero-shot NLI (stratified sample)",
            table(["model", "split", "acc", "macro-F1", "n", "ms / text"], rows),
        ]
    for ft in sorted(EXP.glob("finetune_*.json")):
        r = json.loads(ft.read_text(encoding="utf-8"))
        rows = [
            [
                f"`{ft.stem}`",
                s,
                f(r[s]["general"].get("accuracy")),
                f(r[s]["general"].get("macro_f1")),
                f(r[s].get("subtopics", {}).get("macro_f1_supported", r[s].get("subtopics", {}).get("macro_f1"))),
            ]
            for s in ("val", "se_ext_dev", "sesub_ext_dev", "test", "se_ext_test", "sesub_ext_test")
            if s in r
        ]
        parts += [
            f"## Fine-tuned two-head transformer ({r['encoder']}, {r['train_seconds']} s, "
            f"{r['latency_single_ms_median']} ms/text, {r['size_mb']} MB)",
            table(["model", "split", "general acc", "general macro-F1", "subtopic macro-F1"], rows),
            "history: " + json.dumps(r["history"]),
        ]
    if (h := load("hierarchy")) is not None:
        parts += ["## Subtopics — hierarchical vs flat", hierarchy_table(h)]
    if (c := load("calibration")) is not None:
        parts += ["## Calibration — temperature scaling", calibration_table(c)]
    if (sel := load("selective")) is not None:
        rows = [
            [
                f"`{feat}`",
                split,
                f(r["min_confidence"], 2),
                f(r["coverage"]),
                f(r["accuracy_kept"]),
                f(r["accuracy_rejected"]),
            ]
            for feat, res in sel.items()
            for split in ("val", "se_ext_dev")
            for r in res[split]
        ]
        parts += [
            "## Selective prediction — confidence threshold",
            table(["model", "split", "min confidence", "coverage", "accuracy of kept", "accuracy of rejected"], rows),
        ]
    if (o := load("ood")) is not None:
        parts += ["## Out-of-taxonomy detection", ood_table(o)]
    if (d := load("decay")) is not None:
        cols = ["decay", "min_share", "switch_rule", "confirm_turns", "expire_after"]
        mets = ["theme_accuracy", "switch_lag", "tangent_robust", "false_switch_rate", "accumulation_3",
                "spurious_topics", "stale_theme_rate", "premature_expiry"]  # fmt: skip

        def row(label: str, r: dict) -> list[str]:
            return [label, *[str(r.get(c, "")) for c in cols], *[f(r.get(m)) for m in mets]]

        rows = [row("v1.0 baseline", d["baseline_v1_0"]), row("**selected**", d["selected_metrics"])]
        rows += [row("", r) for r in sorted(d["results"], key=lambda r: -r["theme_accuracy"])[:15]]
        parts += [
            f"## Conversation tracker (validation conversations; rule: {d['selection_rule']}; "
            f"constraints met: {d.get('constraints_met')}) - top 15 of {len(d['results'])} by theme accuracy",
            table(["", *cols, *mets], rows),
        ]
    out = PATHS.reports / "tables.md"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    log_path = PATHS.reports / "experiment_log.md"
    log_path.write_text(experiment_log(), encoding="utf-8")
    print(f"wrote {out} and {log_path}")


if __name__ == "__main__":
    main()
