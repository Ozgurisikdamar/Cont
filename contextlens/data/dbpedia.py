"""Wikipedia category-graph crawl via the DBpedia SPARQL endpoint.

DBpedia mirrors Wikipedia's category graph (``dct:subject`` for article→category
and ``skos:broader`` for category→parent). We use it only for *labels*: which
articles belong to which taxonomy subtopic. Article text comes from the pinned
Wikipedia dump (see :mod:`contextlens.data.wikipedia_dump`).

Responses are cached on disk (keyed by a hash of the query) so the crawl can be
re-run without hitting the endpoint again. The final label manifest is
committed to the repository, which pins the crawl result even though the live
endpoint changes over time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import requests

from contextlens.net import NetworkError, make_session

log = logging.getLogger(__name__)

CATEGORY_PREFIX = "http://dbpedia.org/resource/Category:"
RESOURCE_PREFIX = "http://dbpedia.org/resource/"
SPARQL_PAGE = 10000
VALUES_BATCH = 40
CONCURRENCY = 4  # parallel requests to the public endpoint (fair-use friendly)
MAX_ATTEMPTS = 8
SPLIT_AFTER_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 15.0
PARTIAL_RESULT_STATE = "S1TAT"
EXCLUDED_ARTICLE_PATTERNS = (
    re.compile(r"^(List|Lists|Index|Outline|Timeline|Glossary)_of_"),
    re.compile(r"\(disambiguation\)"),
)


class SparqlClient:
    def __init__(self, endpoint: str, cache_dir: Path, timeout: float = 180.0) -> None:
        self.endpoint = endpoint
        self.cache_dir = cache_dir
        self.timeout = timeout
        self._local = threading.local()
        cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> requests.Session:
        """One HTTP session per thread (batches are fetched concurrently)."""
        if not hasattr(self._local, "session"):
            session = make_session()
            # Virtuoso answers 406 to a plain ``application/json`` Accept header.
            session.headers["Accept"] = "application/sparql-results+json"
            self._local.session = session
        return self._local.session

    def select_many(self, template: str, batches: list[list[str]], order_by: str) -> list[list[dict[str, str]]]:
        """Run several batched SELECTs with bounded concurrency; results keep input order."""
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(lambda b: self.select_values(template, b, order_by), batches))

    def select(self, query: str, attempts: int = MAX_ATTEMPTS) -> list[dict[str, str]]:
        cache = self._cache_path(query)
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        params = {"query": query, "format": "application/sparql-results+json"}
        last = ""
        for attempt in range(attempts):
            try:
                resp = self.session.get(self.endpoint, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                # Virtuoso "anytime" queries return HTTP 200 with *partial*
                # results when they hit the server-side time limit.
                if resp.status_code == 200 and resp.headers.get("X-SQL-State") != PARTIAL_RESULT_STATE:
                    try:
                        payload = resp.json()
                    except ValueError as exc:
                        raise NetworkError("malformed SPARQL JSON") from exc
                    rows = [{k: v["value"] for k, v in b.items()} for b in payload["results"]["bindings"]]
                    cache.write_text(json.dumps(rows), encoding="utf-8")
                    return rows
                last = f"HTTP {resp.status_code} {resp.headers.get('X-SQL-State', '')} {resp.text[:120]!r}"
            log.warning("SPARQL attempt %d failed: %s", attempt + 1, last)
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise NetworkError(f"SPARQL query failed after {attempts} attempts: {last}")

    def select_all(self, query_body: str, order_by: str, attempts: int = MAX_ATTEMPTS) -> list[dict[str, str]]:
        """Complete result of a SELECT.

        Virtuoso caps a response at 10k rows. Sorting large results is slow on
        the public endpoint, so an unsorted first page is requested and the
        sorted, paginated form is used only when that page is full.
        """
        sorted_first = f"{query_body} ORDER BY {order_by} LIMIT {SPARQL_PAGE} OFFSET 0"
        if not self._cache_path(sorted_first).exists():
            rows = self.select(f"{query_body} LIMIT {SPARQL_PAGE}", attempts)
            if len(rows) < SPARQL_PAGE:
                return rows
        out: list[dict[str, str]] = []
        offset = 0
        while True:
            rows = self.select(f"{query_body} ORDER BY {order_by} LIMIT {SPARQL_PAGE} OFFSET {offset}", attempts)
            out.extend(rows)
            if len(rows) < SPARQL_PAGE:
                return out
            offset += SPARQL_PAGE

    def select_values(self, template: str, names: list[str], order_by: str) -> list[dict[str, str]]:
        """Run ``template`` (containing ``{values}``) for a batch of categories.

        When the endpoint keeps failing on a batch (timeouts / HTTP 500 on
        heavy queries) the batch is split in half and each half retried, down
        to single categories, which then get the full retry budget.
        """
        body = template.format(values=_values(_cat_uri(n) for n in names))
        if len(names) == 1:
            return self.select_all(body, order_by)
        try:
            return self.select_all(body, order_by, attempts=SPLIT_AFTER_ATTEMPTS)
        except NetworkError:
            mid = len(names) // 2
            log.warning("splitting a %d-category SPARQL batch after repeated failures", len(names))
            return self.select_values(template, names[:mid], order_by) + self.select_values(
                template, names[mid:], order_by
            )

    def _cache_path(self, query: str) -> Path:
        return self.cache_dir / f"{hashlib.sha256(query.encode('utf-8')).hexdigest()[:24]}.json"


def category_name(uri: str) -> str:
    return unquote(uri[len(CATEGORY_PREFIX) :]) if uri.startswith(CATEGORY_PREFIX) else unquote(uri)


def article_title(uri: str) -> str:
    return unquote(uri[len(RESOURCE_PREFIX) :]).replace("_", " ")


def _cat_uri(name: str) -> str:
    return f"<{CATEGORY_PREFIX}{name}>"


def _values(uris: Iterable[str]) -> str:
    return " ".join(u if u.startswith("<") else f"<{u}>" for u in uris)


def _batched(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass
class CategoryTree:
    seed: str
    depth_of: dict[str, int] = field(default_factory=dict)  # category name -> depth
    excluded: dict[str, str] = field(default_factory=dict)  # category name -> reason


def crawl_categories(client: SparqlClient, seed: str, max_depth: int, exclude: list[re.Pattern[str]]) -> CategoryTree:
    """Breadth-first walk of sub-categories up to ``max_depth`` (seed = depth 0)."""
    tree = CategoryTree(seed=seed)
    tree.depth_of[seed] = 0
    frontier = [seed]
    for depth in range(1, max_depth + 1):
        next_frontier: list[str] = []
        template = "SELECT DISTINCT ?c WHERE {{ VALUES ?p {{ {values} }} ?c skos:broader ?p . }}"
        batches = list(_batched(sorted(frontier), VALUES_BATCH))
        for row in (r for rows in client.select_many(template, batches, "?c") for r in rows):
            name = category_name(row["c"])
            if name in tree.depth_of or name in tree.excluded:
                continue
            reason = next((p.pattern for p in exclude if p.search(name)), None)
            if reason:
                tree.excluded[name] = reason
                continue
            tree.depth_of[name] = depth
            next_frontier.append(name)
        frontier = next_frontier
    return tree


def articles_in(client: SparqlClient, tree: CategoryTree) -> dict[str, int]:
    """Map article title -> minimum category depth at which it was reached."""
    result: dict[str, int] = {}
    cats = sorted(tree.depth_of)
    template = "SELECT DISTINCT ?a ?c WHERE {{ VALUES ?c {{ {values} }} ?a dct:subject ?c . }}"
    for rows in client.select_many(template, list(_batched(cats, VALUES_BATCH)), "?a ?c"):
        for row in rows:
            uri = row["a"]
            if not uri.startswith(RESOURCE_PREFIX):
                continue
            local = unquote(uri[len(RESOURCE_PREFIX) :])
            if local.startswith("Category:") or any(p.search(local) for p in EXCLUDED_ARTICLE_PATTERNS):
                continue
            depth = tree.depth_of[category_name(row["c"])]
            title = article_title(uri)
            if depth < result.get(title, 10**6):
                result[title] = depth
    return result
