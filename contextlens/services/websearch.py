"""Key-less web search: English Wikipedia first, DuckDuckGo Instant Answer as fallback.

Robustness: every provider call goes through :func:`contextlens.net.get_json`
(timeouts, bounded retries, ``Retry-After``); responses are parsed
defensively (missing keys, wrong types, HTML in snippets); URLs are validated
(https + expected host); results are cached in SQLite. Any failure yields an
empty result with a status instead of an exception, so classification,
conversation tracking and logging keep working offline.

Privacy: only the generated query string is sent (never the raw message).
"""

from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from contextlens.net import NetworkError, get_json

log = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
DDG_API = "https://api.duckduckgo.com/"
MAX_TITLE_CHARS = 200
MAX_SUMMARY_CHARS = 500
_TAG = re.compile(r"<[^>]{0,500}>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class SearchResult:
    rank: int
    title: str
    summary: str
    url: str
    source: str


@dataclass(frozen=True)
class SearchOutcome:
    query: str
    status: str  # "ok" | "cached" | "no_results" | "offline" | "disabled"
    provider: str
    results: tuple[SearchResult, ...] = ()
    error: str = ""


class SearchCache(Protocol):
    def get_cached(self, query: str, provider: str, ttl_hours: int) -> str | None: ...

    def put_cached(self, query: str, provider: str, payload: str) -> None: ...


def clean_snippet(text: Any, limit: int) -> str:
    """Strip HTML/control characters and bound the length of untrusted text."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(_TAG.sub("", text))
    text = _CTRL.sub("", " ".join(text.split()))
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def is_safe_url(url: Any, allowed_suffixes: tuple[str, ...]) -> bool:
    if not isinstance(url, str) or len(url) > 2000:
        return False
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(host == s or host.endswith("." + s) for s in allowed_suffixes)


def parse_wikipedia(payload: Any, limit: int) -> list[SearchResult]:
    pages = payload.get("query", {}).get("pages", []) if isinstance(payload, dict) else []
    if isinstance(pages, dict):  # formatversion=1 shape
        pages = list(pages.values())
    if not isinstance(pages, list):
        return []
    ranked = sorted((p for p in pages if isinstance(p, dict)), key=lambda p: p.get("index", 10**6))
    out: list[SearchResult] = []
    for page in ranked:
        title = clean_snippet(page.get("title"), MAX_TITLE_CHARS)
        url = page.get("fullurl") or (
            f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}" if title else ""
        )
        if not title or not is_safe_url(url, ("wikipedia.org",)):
            continue
        out.append(
            SearchResult(len(out) + 1, title, clean_snippet(page.get("extract"), MAX_SUMMARY_CHARS), url, "Wikipedia")
        )
        if len(out) >= limit:
            break
    return out


def parse_duckduckgo(payload: Any, limit: int) -> list[SearchResult]:
    if not isinstance(payload, dict):
        return []
    out: list[SearchResult] = []
    abstract = clean_snippet(payload.get("AbstractText"), MAX_SUMMARY_CHARS)
    url = payload.get("AbstractURL")
    if abstract and is_safe_url(url, ("wikipedia.org", "duckduckgo.com")):
        out.append(
            SearchResult(
                1,
                clean_snippet(payload.get("Heading"), MAX_TITLE_CHARS) or "Result",
                abstract,
                url,
                clean_snippet(payload.get("AbstractSource"), 60) or "DuckDuckGo",
            )
        )
    topics = payload.get("RelatedTopics", [])
    flat: list[dict] = []
    for t in topics if isinstance(topics, list) else []:
        if isinstance(t, dict) and "Topics" in t and isinstance(t["Topics"], list):
            flat.extend(x for x in t["Topics"] if isinstance(x, dict))
        elif isinstance(t, dict):
            flat.append(t)
    for t in flat:
        if len(out) >= limit:
            break
        text = clean_snippet(t.get("Text"), MAX_SUMMARY_CHARS)
        link = t.get("FirstURL")
        if text and is_safe_url(link, ("duckduckgo.com", "wikipedia.org")):
            title = text.split(" - ")[0][:MAX_TITLE_CHARS]
            out.append(SearchResult(len(out) + 1, title, text, link, "DuckDuckGo"))
    return out


@dataclass
class WebSearcher:
    cache: SearchCache | None = None
    enabled: bool = True
    timeout: float = 6.0
    retries: int = 2
    backoff: float = 1.0
    max_results: int = 3
    cache_ttl_hours: int = 72
    fetch: Callable[..., Any] = get_json  # injectable for tests

    def _providers(self) -> list[tuple[str, Callable[[str], Any], Callable[[Any, int], list[SearchResult]]]]:
        return [
            ("wikipedia", self._wikipedia_request, parse_wikipedia),
            ("duckduckgo", self._ddg_request, parse_duckduckgo),
        ]

    def _wikipedia_request(self, query: str) -> Any:
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": str(self.max_results),
            "prop": "extracts|info",
            "exintro": "1",
            "explaintext": "1",
            "exsentences": "2",
            "inprop": "url",
        }
        return self.fetch(WIKIPEDIA_API, params, timeout=self.timeout, retries=self.retries, backoff=self.backoff)

    def _ddg_request(self, query: str) -> Any:
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        return self.fetch(DDG_API, params, timeout=self.timeout, retries=self.retries, backoff=self.backoff)

    def search(self, queries: list[str]) -> SearchOutcome:
        """Try each query with each provider until something returns results."""
        if not self.enabled:
            return SearchOutcome(queries[0] if queries else "", "disabled", "")
        errors: list[str] = []
        last_query = ""
        for query in queries:
            last_query = query
            for name, request, parse in self._providers():
                cached = self.cache.get_cached(query, name, self.cache_ttl_hours) if self.cache else None
                if cached is not None:
                    results = [SearchResult(**r) for r in json.loads(cached)]
                    if results:
                        return SearchOutcome(query, "cached", name, tuple(results))
                    continue
                try:
                    results = parse(request(query), self.max_results)
                except NetworkError as exc:
                    log.warning("%s search failed for %r: %s", name, query, exc)
                    errors.append(f"{name}: {exc}")
                    continue
                if self.cache:
                    self.cache.put_cached(query, name, json.dumps([asdict(r) for r in results]))
                if results:
                    return SearchOutcome(query, "ok", name, tuple(results))
        status = "offline" if errors else "no_results"
        return SearchOutcome(last_query, status, "", (), "; ".join(errors)[:500])
