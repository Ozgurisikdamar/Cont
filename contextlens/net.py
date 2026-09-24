"""Small, defensive HTTP helper shared by data acquisition and web search.

Handles timeouts, bounded retries with exponential backoff, ``Retry-After`` on
HTTP 429, non-JSON bodies and oversized responses. Every failure surfaces as
:class:`NetworkError` so callers can degrade gracefully instead of crashing.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from contextlens.config import USER_AGENT

log = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 5_000_000
MAX_RETRY_AFTER_SECONDS = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class NetworkError(RuntimeError):
    """Raised when a request cannot produce a usable JSON response."""


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    retries: int = 2,
    backoff: float = 1.0,
    sleep: Any = time.sleep,
) -> Any:
    """GET ``url`` and decode JSON, retrying transient failures.

    ``sleep`` is injectable so tests do not wait in real time.
    """
    sess = session or make_session()
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = sess.get(url, params=params, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            log.warning("request to %s failed (%s), attempt %d", url, type(exc).__name__, attempt + 1)
        else:
            if resp.status_code == 200:
                if len(resp.content) > MAX_RESPONSE_BYTES:
                    raise NetworkError(f"response from {url} exceeds {MAX_RESPONSE_BYTES} bytes")
                try:
                    return resp.json()
                except ValueError as exc:
                    raise NetworkError(f"malformed JSON from {url}") from exc
            last_error = NetworkError(f"HTTP {resp.status_code} from {url}")
            if resp.status_code not in RETRYABLE_STATUS:
                raise last_error
            log.warning("HTTP %s from %s, attempt %d", resp.status_code, url, attempt + 1)
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit() and attempt < retries:
                sleep(min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
                continue
        if attempt < retries:
            sleep(backoff * (2**attempt))
    raise NetworkError(f"giving up on {url}: {last_error}")
