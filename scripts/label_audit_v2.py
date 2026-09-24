"""Stratified manual label audit (docs/HARDENING.md, item 8).

    python scripts/label_audit_v2.py --print    # print the sample to judge
    python scripts/label_audit_v2.py            # summarise reports/label_audit_v2.json

Sample (TRAINING split of the current corpus, random_state=42, no duplicates):
10 passages per subtopic (a passage counts for every subtopic it carries), plus
5 more for each subtopic the audit brief marks as problematic: the three Science
subtopics, quantum mechanics, quantum computing and authors -> 280 + 30 = 310
passages, over-sampling the weak classes.

Verdicts (written by reading the passage and its article title):
  correct       the general topic and at least one subtopic are right, and the
                passage itself shows it
  weak          the label is defensible but the passage alone carries little or
                no topical signal, or the article is borderline between topics
  incorrect     the general topic is wrong for this article

Each verdict is stored with the labels the passage carried when it was judged
(taxonomy 1.2.0 before the fixes of decisions.md D-32), so the summary does not
depend on the current corpus. For every ``incorrect`` item the summary also
records what the current corpus does with it (removed / relabelled / unchanged).

The auditor is the agent that built the corpus, not an independent human
annotator; the estimate is therefore a lower bound on disagreement and is
reported with Wilson 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from contextlens.config import PATHS, RANDOM_SEED

PER_SUBTOPIC = 10
OVERSAMPLE = {
    "scientific_method": 5,
    "scientific_research": 5,
    "history_of_science": 5,
    "quantum_mechanics": 5,
    "quantum_computing": 5,
    "authors": 5,
}
OUT = PATHS.reports / "label_audit_v2.json"
VERDICTS = ("correct", "weak", "incorrect")


def sample() -> pd.DataFrame:
    df = pd.read_parquet(PATHS.data_processed / "passages.parquet")
    train = df[df.split == "train"].copy()
    train["subs"] = train.subtopics.str.split("|")
    chosen: list[str] = []
    by_sub: dict[str, list[str]] = defaultdict(list)
    subtopics = sorted({s for subs in train.subs for s in subs})
    for sid in subtopics:
        need = PER_SUBTOPIC + OVERSAMPLE.get(sid, 0)
        pool = train[train.subs.apply(lambda s, sid=sid: sid in s) & ~train.passage_id.isin(chosen)]
        pick = pool.sample(min(need, len(pool)), random_state=RANDOM_SEED)
        chosen += pick.passage_id.tolist()
        by_sub[sid] += pick.passage_id.tolist()
    out = train.set_index("passage_id").loc[chosen].reset_index()
    out["stratum"] = [next(s for s, ids in by_sub.items() if pid in ids) for pid in out.passage_id]
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def summarise(items: list[dict], current: pd.DataFrame) -> dict:
    n = len(items)
    counts = Counter(i["verdict"] for i in items)
    overall = {v: {"n": counts[v], "rate": round(counts[v] / n, 4), "ci95": wilson(counts[v], n)} for v in VERDICTS}
    groups: dict[str, dict[str, Counter]] = {"general": defaultdict(Counter), "stratum": defaultdict(Counter)}
    for it in items:
        groups["general"][it["general"]][it["verdict"]] += 1
        groups["stratum"][it["stratum"]][it["verdict"]] += 1
    per = {
        name: {
            key: {
                "n": sum(c.values()),
                **{v: c[v] for v in VERDICTS},
                "incorrect_ci95": wilson(c["incorrect"], sum(c.values())),
            }
            for key, c in sorted(g.items())
        }
        for name, g in groups.items()
    }
    now = current.set_index("passage_id")
    fate = []
    for it in items:
        if it["verdict"] != "incorrect":
            continue
        if it["passage_id"] not in now.index:
            state = "removed"
        else:
            row = now.loc[it["passage_id"]]
            state = (
                "unchanged"
                if (row.general, row.subtopics) == (it["general"], it["subtopics"])
                else (f"relabelled {row.general}/{row.subtopics}")
            )
        fate.append({"title": it["title"], "was": f"{it['general']}/{it['subtopics']}", "now": state})
    # fixed = the article was removed or moved to another general topic
    fixed = sum(
        f["now"] == "removed"
        or (f["now"].startswith("relabelled") and f["now"].split()[1].split("/")[0] != f["was"].split("/")[0])
        for f in fate
    )
    return {
        "n": n,
        "overall_unweighted": overall,
        **per,
        "incorrect_after_fixes": {"fixed": fixed, "remaining": len(fate) - fixed, "items": fate},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()
    if args.print:
        frame = sample()
        for i, r in enumerate(frame.itertuples()):
            print(f"[{i}] {r.passage_id} | {r.general} | {r.subtopics} | {r.title}\n    {r.text}")
        return 0
    audit = json.loads(OUT.read_text(encoding="utf-8"))
    items = audit["items"]
    bad = [i["passage_id"] for i in items if i["verdict"] not in VERDICTS]
    if bad:
        raise SystemExit(f"invalid verdicts: {bad[:5]}")
    audit["summary"] = summarise(items, pd.read_parquet(PATHS.data_processed / "passages.parquet"))
    OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit["summary"]["overall_unweighted"], indent=2))
    print(json.dumps({k: v for k, v in audit["summary"]["incorrect_after_fixes"].items() if k != "items"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
