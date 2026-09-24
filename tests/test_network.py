"""Tests against the real internet. Both kinds are deselected by default.

* ``pytest -m network`` - **resilience**: whatever the providers do (answer,
  rate-limit, fail), the searcher returns a clean status and only safe results.
  It passes when both providers are down, so it does not prove that search works.
* ``pytest -m network_live`` - **live smoke**: a deterministic query must get at
  least one valid result from at least one provider (HTTPS, allow-listed host,
  non-empty title). It fails when search does not work; run it before a release.
"""

from urllib.parse import urlparse

import pytest

from contextlens.services.websearch import WebSearcher, parse_duckduckgo, parse_wikipedia

ALLOWED_HOSTS = ("wikipedia.org", "duckduckgo.com")
LIVE_QUERY = "Albert Einstein"  # stable article on both providers


def _valid(result) -> bool:
    host = urlparse(result.url).hostname or ""
    return (
        result.url.startswith("https://")
        and any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)
        and bool(result.title.strip())
    )


@pytest.mark.network
def test_live_search_returns_safe_results_or_a_clean_status():
    out = WebSearcher(timeout=10.0, retries=1).search(["quantum computing"])
    assert out.status in {"ok", "no_results", "offline"}
    if out.status == "ok":
        assert out.provider in {"wikipedia", "duckduckgo"}
        assert all(_valid(r) for r in out.results)


@pytest.mark.network_live
def test_at_least_one_provider_returns_a_valid_result():
    searcher = WebSearcher(timeout=10.0, retries=2)
    per_provider: dict[str, str] = {}
    valid_found = False
    for name, request, parse in (
        ("wikipedia", searcher._wikipedia_request, parse_wikipedia),
        ("duckduckgo", searcher._ddg_request, parse_duckduckgo),
    ):
        try:
            results = parse(request(LIVE_QUERY), 3)
        except Exception as exc:  # recorded in the failure message
            per_provider[name] = f"error: {exc}"
            continue
        good = [r for r in results if _valid(r)]
        per_provider[name] = f"{len(good)}/{len(results)} valid"
        valid_found = valid_found or bool(good)
    assert valid_found, f"no provider returned a valid result for {LIVE_QUERY!r}: {per_provider}"


@pytest.mark.network_live
def test_searcher_end_to_end_returns_ok():
    out = WebSearcher(timeout=10.0, retries=2).search([LIVE_QUERY])
    assert out.status == "ok", (out.status, out.error)
    assert out.results and all(_valid(r) for r in out.results)
