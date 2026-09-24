"""Re-apply the labelling rules to the committed corpus after a taxonomy fix.

    python scripts/relabel_corpus.py            # writes data/, reports/relabel.json

Taxonomy 1.1.0 removed the seed ``Quantum_information_science@0`` from the
subtopic *quantum_computing*: that category holds foundational physics
(quantum entanglement, Bell's theorem, Bell states) at depth 0, so the
minimum-depth rule labelled those articles Technology instead of Physics
(found by error analysis of the acceptance test; decisions.md D-27).

The label rules (contextlens.data.labels) are re-run on the cached category
crawl (data/raw/sparql_cache, no network). For every article already in the
corpus the new label replaces the old one; the article keeps its split, so the
grouped split stays leak-free. Articles that no longer receive a label (or
become ambiguous) are removed. Articles that a from-scratch build would *add*
are not added (that needs the full dump download); docs/DATASET_CARD.md says so.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from contextlens.config import DATA, PATHS
from contextlens.data.dbpedia import SparqlClient
from contextlens.data.labels import assign_labels, crawl_membership
from contextlens.logging_setup import setup_logging
from contextlens.taxonomy import load_taxonomy


def main() -> int:
    setup_logging("INFO")
    tax = load_taxonomy()
    client = SparqlClient(DATA.sparql_endpoint, PATHS.data_raw / "sparql_cache")
    membership, _ = crawl_membership(tax, client)
    labelled, _ambiguous = assign_labels(tax, membership)
    new = {a.title: a for a in labelled}

    man_path = PATHS.data_manifest / "articles.csv"
    pas_path = PATHS.data_processed / "passages.parquet"
    man, pas = pd.read_csv(man_path), pd.read_parquet(pas_path)
    changes, keep = [], []
    for row in man.itertuples(index=False):
        art = new.get(row.title)
        old = (row.general, row.subtopics)
        now = (art.general, "|".join(art.subtopics)) if art else None
        if now != old:
            changes.append(
                {"title": row.title, "split": row.split, "old": list(old), "new": list(now) if now else None}
            )
        keep.append(art is not None)
    man = man[keep].copy()
    pas = pas[pas.title.isin(set(man.title))].copy()
    for frame in (man, pas):
        frame["general"] = [new[t].general for t in frame.title]
        frame["subtopics"] = ["|".join(new[t].subtopics) for t in frame.title]
        frame["depth"] = [new[t].depth for t in frame.title]
    man.to_csv(man_path, index=False, lineterminator="\r\n")  # same format as the corpus builder (csv module)
    pas.to_parquet(pas_path, index=False)

    summary = {
        "taxonomy_version": tax.version,
        "articles_changed": len(changes),
        "articles_removed": sum(1 for c in changes if c["new"] is None),
        "transitions": {
            f"{k[0]} -> {k[1]}": v
            for k, v in Counter((c["old"][0], c["new"][0] if c["new"] else "removed") for c in changes).items()
        },
        "by_split": dict(Counter(c["split"] for c in changes)),
        "articles_after": len(man),
        "passages_after": len(pas),
        "changes": changes,
    }
    out = PATHS.reports / "relabel.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "changes"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
