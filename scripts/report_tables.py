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


def load(name: str) -> dict | None:
    path = EXP / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


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
        rows = [
            [
                split,
                f(r["decay"], 1),
                f(r["theme_accuracy"]),
                f(r["switch_lag"], 2),
                f(r["tangent_robust"]),
                f(r["accumulation_3"]),
            ]
            for split, rs in d["results"].items()
            for r in rs
        ]
        parts += [
            f"## Conversation decay (selected: {d['selected_decay']}; rule: {d['selection_rule']})",
            table(["split", "decay", "theme acc", "switch lag", "tangent robust", "3-topic accumulation"], rows),
        ]
    out = PATHS.reports / "tables.md"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
