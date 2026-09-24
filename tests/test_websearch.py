import json

import pytest
import requests

from contextlens.net import NetworkError, get_json
from contextlens.services.websearch import (
    WebSearcher,
    clean_snippet,
    is_safe_url,
    parse_duckduckgo,
    parse_wikipedia,
)

WIKI_OK = {
    "query": {
        "pages": [
            {
                "index": 2,
                "title": "Qubit",
                "extract": "A qubit is a <b>unit</b>.",
                "fullurl": "https://en.wikipedia.org/wiki/Qubit",
            },
            {
                "index": 1,
                "title": "Quantum computing",
                "extract": "Quantum computing uses qubits.",
                "fullurl": "https://en.wikipedia.org/wiki/Quantum_computing",
            },
            {"index": 3, "title": "Evil", "extract": "x", "fullurl": "javascript:alert(1)"},
        ]
    }
}
DDG_OK = {
    "Heading": "Qubit",
    "AbstractText": "In quantum computing, a qubit is...",
    "AbstractURL": "https://en.wikipedia.org/wiki/Qubit",
    "AbstractSource": "Wikipedia",
    "RelatedTopics": [
        {"Text": "Quantum gate - a basic circuit", "FirstURL": "https://duckduckgo.com/Quantum_gate"},
        {"Name": "group", "Topics": [{"Text": "Bloch sphere - geometry", "FirstURL": "https://duckduckgo.com/Bloch"}]},
        {"Text": "bad", "FirstURL": "http://insecure.example.com"},
    ],
}


def test_parse_wikipedia_orders_by_search_rank_and_drops_unsafe_urls():
    results = parse_wikipedia(WIKI_OK, limit=5)
    assert [r.title for r in results] == ["Quantum computing", "Qubit"]
    assert results[1].summary == "A qubit is a unit."  # HTML stripped
    assert [r.rank for r in results] == [1, 2]


def test_parse_wikipedia_handles_malformed_payloads():
    for payload in (None, [], "text", {"query": {"pages": "nope"}}, {"query": {"pages": [1, None]}}, {}):
        assert parse_wikipedia(payload, 3) == []
    dict_shape = {"query": {"pages": {"1": {"title": "Qubit", "extract": "x"}}}}
    assert parse_wikipedia(dict_shape, 3)[0].url == "https://en.wikipedia.org/wiki/Qubit"


def test_parse_duckduckgo_abstract_and_nested_topics():
    results = parse_duckduckgo(DDG_OK, limit=5)
    assert [r.title for r in results] == ["Qubit", "Quantum gate", "Bloch sphere"]
    assert all(r.url.startswith("https://") for r in results)
    assert parse_duckduckgo({"RelatedTopics": "x"}, 3) == []
    assert parse_duckduckgo(None, 3) == []


def test_clean_snippet_and_url_validation():
    assert clean_snippet("<script>x</script>Hi &amp; bye\x07", 50) == "xHi & bye"
    assert clean_snippet("a" * 20, 10).endswith("…") and len(clean_snippet("a" * 20, 10)) == 10
    assert clean_snippet(123, 10) == ""
    assert is_safe_url("https://en.wikipedia.org/wiki/X", ("wikipedia.org",))
    assert not is_safe_url("http://en.wikipedia.org/wiki/X", ("wikipedia.org",))
    assert not is_safe_url("https://wikipedia.org.evil.com/x", ("wikipedia.org",))
    assert not is_safe_url("javascript:alert(1)", ("wikipedia.org",))


class MemoryCache:
    def __init__(self):
        self.store = {}

    def get_cached(self, query, provider, ttl_hours):
        return self.store.get((query.lower(), provider))

    def put_cached(self, query, provider, payload):
        self.store[(query.lower(), provider)] = payload


def fetch_factory(wiki=WIKI_OK, ddg=DDG_OK, fail=()):
    calls = []

    def fetch(url, params, **kwargs):
        provider = "wikipedia" if "wikipedia" in url else "duckduckgo"
        calls.append(provider)
        if provider in fail:
            raise NetworkError(f"{provider} down")
        return wiki if provider == "wikipedia" else ddg

    return fetch, calls


def test_search_uses_wikipedia_first_and_caches():
    fetch, calls = fetch_factory()
    cache = MemoryCache()
    ws = WebSearcher(cache=cache, fetch=fetch, max_results=2)
    out = ws.search(["quantum computing qubits", "quantum computing"])
    assert out.status == "ok" and out.provider == "wikipedia" and len(out.results) == 2
    again = ws.search(["quantum computing qubits"])
    assert again.status == "cached" and calls == ["wikipedia"]


def test_search_falls_back_to_duckduckgo():
    fetch, calls = fetch_factory(fail=("wikipedia",))
    out = WebSearcher(fetch=fetch).search(["qubit"])
    assert out.provider == "duckduckgo" and out.status == "ok"
    assert calls == ["wikipedia", "duckduckgo"]


def test_search_tries_fallback_query_when_first_has_no_results():
    empty = {"query": {"pages": []}}
    fetch, _ = fetch_factory(wiki=empty, ddg={})
    out = WebSearcher(fetch=fetch).search(["very specific query", "broad"])
    assert out.status == "no_results" and out.results == ()


def test_offline_never_raises():
    fetch, _ = fetch_factory(fail=("wikipedia", "duckduckgo"))
    out = WebSearcher(fetch=fetch).search(["qubit"])
    assert out.status == "offline" and "down" in out.error


def test_disabled():
    assert WebSearcher(enabled=False).search(["x"]).status == "disabled"


# ---- contextlens.net.get_json ---------------------------------------------------
class FakeResponse:
    def __init__(self, status, body=b"{}", headers=None):
        self.status_code = status
        self.content = body
        self.headers = headers or {}

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_get_json_retries_on_429_with_retry_after():
    sleeps = []
    s = FakeSession([FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, b'{"ok": 1}')])
    assert get_json("https://x", session=s, sleep=sleeps.append) == {"ok": 1}
    assert sleeps == [3.0]


def test_get_json_retries_timeouts_then_gives_up():
    s = FakeSession([requests.Timeout(), requests.ConnectionError(), requests.Timeout()])
    with pytest.raises(NetworkError, match="giving up"):
        get_json("https://x", session=s, retries=2, sleep=lambda _s: None)
    assert s.calls == 3


def test_get_json_does_not_retry_404_and_rejects_bad_json():
    with pytest.raises(NetworkError, match="404"):
        get_json("https://x", session=FakeSession([FakeResponse(404)]), sleep=lambda _s: None)
    with pytest.raises(NetworkError, match="malformed"):
        get_json("https://x", session=FakeSession([FakeResponse(200, b"<html>")]), sleep=lambda _s: None)


def test_failed_provider_is_skipped_until_its_cooldown_ends():
    fetch, calls = fetch_factory(wiki={"query": {"pages": []}}, ddg={}, fail=("wikipedia",))
    now = [0.0]
    ws = WebSearcher(fetch=fetch, provider_cooldown_s=300, clock=lambda: now[0])
    out = ws.search(["a", "b", "c"])
    assert calls == ["wikipedia", "duckduckgo", "duckduckgo", "duckduckgo"]  # wikipedia tried once
    assert out.status == "no_results"  # duckduckgo answered (empty): not "offline"
    now[0] = 301.0
    ws.search(["d"])
    assert calls[-2:] == ["wikipedia", "duckduckgo"]  # circuit closed again after the cooldown
