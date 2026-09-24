"""Build the locked final holdout (docs/HARDENING.md item 2, decisions.md D-36).

The v1.0 ``test`` / ``ext_test`` splits were computed during development
(they appeared in benchmark tables), so they are no longer blind. This script
builds new data that no development script reads; ``scripts/locked_eval.py``
evaluates it exactly once, after the configuration is frozen.

1. ``wiki_locked``   Wikipedia articles that the taxonomy 1.2.0 category crawl
                     labels but that are in neither the current nor the v1.x
                     corpus (the corpus capped each subtopic); up to
                     ``PER_SUBTOPIC`` per subtopic; text from the same pinned
                     Wikipedia dump as the corpus, first passage cut exactly like
                     the corpus.
2. ``se_locked``     Stack Exchange question titles created 2026-01-01 ..
                     2026-09-20, top-voted per site (in-taxonomy sites of the
                     general set, off-topic sites) and per (site, tag) pair of
                     the subtopic set; question ids already in any evaluation
                     set are dropped. The v1.0 external sets were top-voted of
                     all time, so these are new questions.
3. Already built with a locked half: Tatoeba (``language_eval.jsonl``, split
   ``locked``) and CLINC150 test (``ood_conversational.jsonl``, split ``locked``).

Output: data/locked/wiki_locked.jsonl, data/locked/se_locked.jsonl and
data/locked/MANIFEST.json (sizes + SHA-256 of every locked file).

    python scripts/build_locked_sets.py
    python scripts/build_locked_sets.py --repair   # re-derive view flags only (no network)
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from contextlens.config import DATA, PATHS, RANDOM_SEED
from contextlens.data.dbpedia import SparqlClient
from contextlens.data.labels import assign_labels, crawl_membership
from contextlens.data.passages import make_passages
from contextlens.data.wikipedia_dump import extract_articles
from contextlens.logging_setup import setup_logging
from contextlens.net import get_json, make_session
from contextlens.taxonomy import load_taxonomy

log = logging.getLogger("build_locked_sets")
OUT = PATHS.root / "data" / "locked"
PER_SUBTOPIC = 15
FROM_DATE, TO_DATE = "2026-01-01", "2026-09-20"
SE_PAGE = 100


def epoch(day: str) -> int:
    return int(pd.Timestamp(day, tz="UTC").timestamp())


def corpus_titles() -> set[str]:
    """Titles in the current manifest and in every committed earlier version of it."""
    titles = set(pd.read_csv(PATHS.data_manifest / "articles.csv").title)
    revs = subprocess.run(
        ["git", "log", "--format=%H", "--", "data/manifest/articles.csv"],  # noqa: S607 - git from PATH
        cwd=PATHS.root, capture_output=True, text=True, check=True,
    ).stdout.split()  # fmt: skip
    for rev in revs:
        blob = subprocess.run(
            ["git", "show", f"{rev}:data/manifest/articles.csv"],  # noqa: S607 - git from PATH
            cwd=PATHS.root,
            capture_output=True,
            text=True,
        )
        if blob.returncode == 0:
            titles |= set(pd.read_csv(StringIO(blob.stdout)).title)
    return titles


def fetch_texts(titles: list[str]) -> dict[str, str]:
    """Article text from the same pinned Wikipedia dump as the corpus (the MediaWiki
    API rate-limits shared cloud IPs - HTTP 429 on the first request here)."""
    found = extract_articles(
        titles,
        repo=DATA.wiki_repo,
        config=DATA.wiki_config,
        revision=DATA.wiki_revision,
        num_shards=DATA.wiki_num_shards,
        download_dir=PATHS.data_raw / "wikipedia_shards",
    )
    return {t: doc["text"] for t, doc in found.items()}


def build_wiki(tax) -> list[dict]:
    client = SparqlClient(DATA.sparql_endpoint, PATHS.data_raw / "sparql_cache")
    membership, _ = crawl_membership(tax, client)
    labelled, _ = assign_labels(tax, membership)
    seen = corpus_titles()
    by_sub: dict[str, list] = defaultdict(list)
    for art in labelled:
        if art.title not in seen:
            by_sub[art.subtopics[0]].append(art)
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = []
    for sid in sorted(by_sub):
        pool = sorted(by_sub[sid], key=lambda a: a.title)
        idx = rng.choice(len(pool), size=min(PER_SUBTOPIC * 2, len(pool)), replace=False)
        chosen += [pool[i] for i in sorted(idx)]  # oversample: some pages have no usable lead
    texts = fetch_texts([a.title for a in chosen])
    cfg = DATA
    rows, per_sub = [], defaultdict(int)
    for art in chosen:
        if per_sub[art.subtopics[0]] >= PER_SUBTOPIC or art.title not in texts:
            continue
        chunks = make_passages(
            texts[art.title], min_words=cfg.min_passage_words, max_words=cfg.max_passage_words, max_passages=1
        )
        if not chunks:
            continue
        per_sub[art.subtopics[0]] += 1
        rows.append(
            {"source": "wikipedia_api", "title": art.title, "general": art.general,
             "subtopics": art.subtopics, "text": chunks[0]}
        )  # fmt: skip
    log.info("wiki_locked: %d passages, per subtopic %s", len(rows), dict(per_sub))
    return rows


def se_titles(site: str, tag: str | None, session) -> list[dict]:
    params = {
        "site": site, "pagesize": SE_PAGE, "sort": "votes", "order": "desc",
        "fromdate": epoch(FROM_DATE), "todate": epoch(TO_DATE),
    }  # fmt: skip
    if tag:
        params["tagged"] = tag
    payload = get_json(f"{DATA.se_api}/questions", params, session=session, timeout=30, retries=2)
    if payload.get("backoff"):
        time.sleep(float(payload["backoff"]))
    time.sleep(0.3)
    log.info("SE %s/%s: %d items, quota %s", site, tag, len(payload.get("items", [])), payload.get("quota_remaining"))
    return [
        {"site": site, "question_id": it["question_id"], "text": html.unescape(it["title"]).strip(),
         "tags": it.get("tags", []), "created": it.get("creation_date")}
        for it in payload.get("items", [])
    ]  # fmt: skip


def assign_views(rows: list[dict], site_cfg: dict) -> list[dict]:
    """Mark which evaluation view each question belongs to (independent flags).

    * general view (``in_general``): every question from a site of the general
      set, labelled by the site (``site_general``; None = off-topic site) - the
      same rule as the v1 general set;
    * subtopic view: every in-taxonomy question with at least one subtopic,
      labelled by its (site, tag) pairs (``general`` + ``subtopics``).

    Run 1 of the locked evaluation used an exclusive ``set`` field instead: a
    general-set question also returned by a subtopic query was moved to the
    subtopic view, which emptied the general view of every Technology and Science
    site (decisions.md D-36). ``--repair`` re-derives the flags from the saved
    file without any network request.
    """
    site_general = {f.split(".")[0]: g for f, g in site_cfg["in_taxonomy"].items()}
    ood_sites = {f.split(".")[0] for f in site_cfg["ood"]}
    for r in rows:
        r.pop("set", None)
        r["in_general"] = r["site"] in site_general or r["site"] in ood_sites
        r["site_general"] = site_general.get(r["site"])
    return rows


def build_se(tax) -> list[dict]:
    known = set()
    for name in ("se_subtopic_eval.jsonl",):
        for line in (PATHS.data_external / name).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            known.add((r["site"], r["question_id"]))
    session = make_session()
    site_cfg = tax.raw["stackexchange_general_sites"]
    rows: dict[tuple[str, int], dict] = {}
    for fname, general in site_cfg["in_taxonomy"].items():
        site = fname.split(".")[0]
        for q in se_titles(site, None, session):
            rows[(site, q["question_id"])] = {
                **q,
                "general": general,
                "subtopics": [],
                "is_ood": False,
            }
    for fname in site_cfg["ood"]:
        site = fname.split(".")[0]
        for q in se_titles(site, None, session):
            rows[(site, q["question_id"])] = {**q, "general": None, "subtopics": [], "is_ood": True}
    for g in tax.generals:
        for sub in g.subtopics:
            for site, tag in sub.stackexchange:
                for q in se_titles(site, tag, session):
                    key = (site, q["question_id"])
                    row = rows.setdefault(key, {**q, "general": g.id, "subtopics": [], "is_ood": False})
                    if row["general"] == g.id and sub.id not in row["subtopics"]:
                        row["subtopics"].append(sub.id)
    # every other matching (site, tag) pair of the same general topic
    for row in rows.values():
        if row["general"] is None or not row["subtopics"]:
            continue
        for sub in tax.general(row["general"]).subtopics:
            for site, tag in sub.stackexchange:
                if site == row["site"] and (tag is None or tag in row["tags"]) and sub.id not in row["subtopics"]:
                    row["subtopics"].append(sub.id)
    out = assign_views([r for k, r in rows.items() if k not in known], site_cfg)
    log.info("se_locked: %d questions (%d dropped as already known)", len(out), len(rows) - len(out))
    return sorted(out, key=lambda r: (r["site"], r["question_id"]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    setup_logging("INFO")
    tax = load_taxonomy()
    OUT.mkdir(parents=True, exist_ok=True)
    if "--repair" in sys.argv:  # re-derive the view flags of the saved files, no network
        se = assign_views(read_jsonl(OUT / "se_locked.jsonl"), tax.raw["stackexchange_general_sites"])
        files = {"wiki_locked.jsonl": read_jsonl(OUT / "wiki_locked.jsonl"), "se_locked.jsonl": se}
    else:
        files = {"wiki_locked.jsonl": build_wiki(tax), "se_locked.jsonl": build_se(tax)}
    for name, rows in files.items():
        with (OUT / name).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    external = {
        "language_eval.jsonl (split=locked)": PATHS.data_external / "language_eval.jsonl",
        "ood_conversational.jsonl (split=locked)": PATHS.data_external / "ood_conversational.jsonl",
    }
    manifest = {
        "built_with": "scripts/build_locked_sets.py",
        "taxonomy_version": tax.version,
        "se_date_range": [FROM_DATE, TO_DATE],
        "files": {name: {"rows": len(rows), "sha256": sha256(OUT / name)} for name, rows in files.items()},
        "external_locked_halves": {k: sha256(p) for k, p in external.items()},
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["files"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
