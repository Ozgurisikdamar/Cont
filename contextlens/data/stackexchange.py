"""External evaluation data from Stack Exchange (real user-written questions).

Two sources, both used **only for evaluation** (never for training):

* ``fetch_tagged_questions`` — Stack Exchange API, one request per
  (site, tag) pair from the taxonomy. Provides general topic *and* subtopic
  labels (from tags). Raw API responses are cached on disk so rebuilding does
  not consume the (small, per-IP) anonymous API quota again.
* ``load_cluster_titles`` — the MTEB StackExchangeClustering files (question
  titles labelled with their site), pinned to a Hub revision. Provides general
  topic labels for in-taxonomy sites and genuine out-of-taxonomy (OOD) titles.

Content is CC BY-SA (Stack Exchange); links are kept for attribution.
"""

from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path

from contextlens.net import NetworkError, get_json, make_session
from contextlens.taxonomy import Taxonomy

log = logging.getLogger(__name__)

API_PAGE_SIZE = 100


def _cache_file(cache_dir: Path, site: str, tag: str | None) -> Path:
    return cache_dir / f"{site}__{tag or 'ALL'}.json"


def fetch_tagged_questions(
    api_base: str, site: str, tag: str | None, cache_dir: Path
) -> list[dict]:
    """Return the top-voted questions for ``site``/``tag`` (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_file(cache_dir, site, tag)
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        params: dict[str, object] = {
            "site": site,
            "pagesize": API_PAGE_SIZE,
            "sort": "votes",
            "order": "desc",
        }
        if tag:
            params["tagged"] = tag
        payload = get_json(f"{api_base}/questions", params, session=make_session(), timeout=30, retries=2)
        cache.write_text(json.dumps(payload), encoding="utf-8")
        log.info("SE %s/%s quota_remaining=%s", site, tag, payload.get("quota_remaining"))
        if payload.get("backoff"):
            time.sleep(float(payload["backoff"]))
    questions = []
    for item in payload.get("items", []):
        questions.append(
            {
                "source": "stackexchange_api",
                "site": site,
                "question_id": item["question_id"],
                "text": html.unescape(item["title"]).strip(),
                "tags": item.get("tags", []),
                "link": item.get("link", ""),
                "score": item.get("score", 0),
            }
        )
    return questions


def build_subtopic_eval_set(
    taxonomy: Taxonomy, api_base: str, cache_dir: Path, missing: list[str] | None = None
) -> list[dict]:
    """Questions labelled with a general topic and one or more subtopics.

    Pairs that cannot be fetched (e.g. API throttling) are skipped and reported
    in ``missing``; rerunning later resumes from the on-disk cache.
    """
    by_key: dict[tuple[str, int], dict] = {}
    for general in taxonomy.generals:
        for sub in general.subtopics:
            for site, tag in sub.stackexchange:
                try:
                    questions = fetch_tagged_questions(api_base, site, tag, cache_dir)
                except NetworkError as exc:
                    log.warning("skipping SE %s/%s: %s", site, tag, exc)
                    if missing is not None:
                        missing.append(f"{site}/{tag}")
                    continue
                for q in questions:
                    key = (q["site"], q["question_id"])
                    row = by_key.setdefault(key, {**q, "general": general.id, "subtopics": []})
                    if row["general"] != general.id:
                        # Same question reachable from two general topics: ambiguous.
                        row["general"] = None
                    if sub.id not in row["subtopics"]:
                        row["subtopics"].append(sub.id)
    # A question also carries every other subtopic whose (site, tag) matches its tags.
    for row in by_key.values():
        if row["general"] is None:
            continue
        for sub in taxonomy.general(row["general"]).subtopics:
            for site, tag in sub.stackexchange:
                if site == row["site"] and (tag is None or tag in row["tags"]) and sub.id not in row["subtopics"]:
                    row["subtopics"].append(sub.id)
    rows = [r for r in by_key.values() if r["general"] is not None]
    dropped = len(by_key) - len(rows)
    if dropped:
        log.warning("dropped %d cross-topic ambiguous SE questions", dropped)
    return sorted(rows, key=lambda r: (r["site"], r["question_id"]))


def load_cluster_titles(jsonl_paths: list[Path], site_map: dict[str, str], ood_sites: list[str]) -> list[dict]:
    """Titles from MTEB StackExchangeClustering mapped to general topics / OOD."""
    rows: dict[tuple[str, str], dict] = {}
    for path in jsonl_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                block = json.loads(line)
                for text, site in zip(block["sentences"], block["labels"], strict=True):
                    if site in site_map:
                        general: str | None = site_map[site]
                    elif site in ood_sites:
                        general = None
                    else:
                        continue
                    text = html.unescape(text).strip()
                    rows.setdefault((site, text), {
                        "source": "mteb_stackexchange_clustering",
                        "site": site.replace(".txt", ""),
                        "text": text,
                        "general": general,
                        "is_ood": general is None,
                    })
    return sorted(rows.values(), key=lambda r: (r["site"], r["text"]))
