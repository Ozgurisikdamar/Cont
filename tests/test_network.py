"""Live web-search check (``pytest -m network``). Needs internet access."""

import pytest

from contextlens.services.websearch import WebSearcher

pytestmark = pytest.mark.network


def test_live_search_returns_safe_results_or_a_clean_status():
    out = WebSearcher(timeout=10.0, retries=1).search(["quantum computing"])
    assert out.status in {"ok", "no_results", "offline"}
    if out.status == "ok":
        assert out.provider in {"wikipedia", "duckduckgo"}
        assert all(r.url.startswith("https://") and r.title for r in out.results)
