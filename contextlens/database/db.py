"""SQLite persistence: sessions, analysed texts, conversation topics, search results.

Design (docs/architecture.md#database):

* Foreign keys enforced (``PRAGMA foreign_keys = ON``), indexes on every
  foreign key, WAL journal and a busy timeout so a second process does not get
  an immediate "database is locked".
* All statements are parameterised; every write runs in a transaction
  (``with self.conn:``).
* Schema versioning with ``PRAGMA user_version``: ``MIGRATIONS[i]`` upgrades
  version ``i`` to ``i + 1``. No migration framework needed at this scale.
* Every prediction row references the ``model_versions`` row that produced it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000

MIGRATIONS: list[str] = [
    # v0 -> v1: initial schema
    """
    CREATE TABLE model_versions (
        id INTEGER PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        trained_at TEXT NOT NULL,
        dataset_version TEXT NOT NULL,
        metrics_json TEXT NOT NULL DEFAULT '{}',
        UNIQUE (model_name, model_version)
    );
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        model_version_id INTEGER REFERENCES model_versions(id)
    );
    CREATE TABLE texts (
        id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        turn INTEGER NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        general_topic TEXT,
        general_confidence REAL,
        subtopics_json TEXT NOT NULL DEFAULT '[]',
        general_probs_json TEXT NOT NULL DEFAULT '{}',
        uncertain INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        model_version_id INTEGER REFERENCES model_versions(id)
    );
    CREATE INDEX idx_texts_session ON texts(session_id, turn);
    CREATE TABLE conversation_topics (
        id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        text_id INTEGER NOT NULL REFERENCES texts(id) ON DELETE CASCADE,
        theme_label TEXT NOT NULL,
        theme_phrase TEXT NOT NULL,
        general_scores_json TEXT NOT NULL,
        subtopic_scores_json TEXT NOT NULL,
        query TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_topics_session ON conversation_topics(session_id);
    CREATE INDEX idx_topics_text ON conversation_topics(text_id);
    CREATE TABLE search_results (
        id INTEGER PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        conversation_topic_id INTEGER NOT NULL REFERENCES conversation_topics(id) ON DELETE CASCADE,
        query TEXT NOT NULL,
        provider TEXT NOT NULL,
        rank INTEGER NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        url TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_results_session ON search_results(session_id);
    CREATE INDEX idx_results_topic ON search_results(conversation_topic_id);
    CREATE TABLE search_cache (
        query TEXT NOT NULL,
        provider TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (query, provider)
    );
    CREATE INDEX idx_cache_fetched ON search_cache(fetched_at);
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DatabaseError(RuntimeError):
    """Raised when the database cannot be opened or written."""


@dataclass(frozen=True)
class TextRecord:
    turn: int
    content: str
    general_topic: str | None
    general_confidence: float | None
    subtopics: list[dict]
    uncertain: bool
    status: str
    created_at: str


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        try:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT_MS / 1000)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            if self.path != ":memory:":
                self.conn.execute("PRAGMA journal_mode = WAL")
            self._migrate()
        except (sqlite3.Error, OSError) as exc:
            raise DatabaseError(f"cannot open database {self.path}: {exc}") from exc

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        try:
            with self.conn:
                yield self.conn
        except sqlite3.Error as exc:
            raise DatabaseError(f"database write failed: {exc}") from exc

    def _read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        try:
            return self.conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DatabaseError(f"database read failed: {exc}") from exc

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise DatabaseError(f"database schema v{version} is newer than this code (v{SCHEMA_VERSION})")
        for target in range(version, SCHEMA_VERSION):
            log.info("migrating database schema v%d -> v%d", target, target + 1)
            # executescript commits implicitly; the user_version bump is part of the script.
            self.conn.executescript(f"BEGIN; {MIGRATIONS[target]} PRAGMA user_version = {target + 1}; COMMIT;")

    @property
    def schema_version(self) -> int:
        return int(self.conn.execute("PRAGMA user_version").fetchone()[0])

    # -- model versions & sessions ---------------------------------------
    def register_model(self, name: str, version: str, trained_at: str, dataset_version: str, metrics: dict) -> int:
        with self._tx() as c:
            c.execute(
                "INSERT OR IGNORE INTO model_versions (model_name, model_version, trained_at, dataset_version,"
                " metrics_json) VALUES (?, ?, ?, ?, ?)",
                (name, version, trained_at, dataset_version, json.dumps(metrics)),
            )
            row = c.execute(
                "SELECT id FROM model_versions WHERE model_name = ? AND model_version = ?", (name, version)
            ).fetchone()
        return int(row["id"])

    def start_session(self, model_version_id: int | None) -> str:
        session_id = str(uuid.uuid4())
        with self._tx() as c:
            c.execute(
                "INSERT INTO sessions (id, started_at, model_version_id) VALUES (?, ?, ?)",
                (session_id, utcnow(), model_version_id),
            )
        return session_id

    def end_session(self, session_id: str) -> None:
        with self._tx() as c:
            c.execute("UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL", (utcnow(), session_id))

    # -- writes ------------------------------------------------------------
    def add_text(
        self,
        session_id: str,
        turn: int,
        content: str,
        *,
        general_topic: str | None,
        general_confidence: float | None,
        subtopics: list[dict],
        general_probs: dict[str, float],
        uncertain: bool,
        status: str,
        model_version_id: int | None,
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO texts (session_id, turn, content, created_at, general_topic, general_confidence,"
                " subtopics_json, general_probs_json, uncertain, status, model_version_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    turn,
                    content,
                    utcnow(),
                    general_topic,
                    general_confidence,
                    json.dumps(subtopics),
                    json.dumps(general_probs),
                    int(uncertain),
                    status,
                    model_version_id,
                ),
            )
        return int(cur.lastrowid or 0)

    def add_conversation_topic(
        self,
        session_id: str,
        text_id: int,
        theme_label: str,
        theme_phrase: str,
        general_scores: dict[str, float],
        subtopic_scores: dict[str, float],
        query: str,
    ) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO conversation_topics (session_id, text_id, theme_label, theme_phrase,"
                " general_scores_json, subtopic_scores_json, query, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    text_id,
                    theme_label,
                    theme_phrase,
                    json.dumps(general_scores),
                    json.dumps(subtopic_scores),
                    query,
                    utcnow(),
                ),
            )
        return int(cur.lastrowid or 0)

    def add_search_results(self, session_id: str, topic_id: int, query: str, provider: str, results: list[dict]) -> int:
        rows = [
            (
                session_id,
                topic_id,
                query,
                provider,
                r["rank"],
                r["title"],
                r["summary"],
                r["url"],
                r["source"],
                utcnow(),
            )
            for r in results
        ]
        with self._tx() as c:
            c.executemany(
                "INSERT INTO search_results (session_id, conversation_topic_id, query, provider, rank, title,"
                " summary, url, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    # -- search cache (implements websearch.SearchCache) -------------------
    # The cache is best-effort: a database problem must never block a search.
    def get_cached(self, query: str, provider: str, ttl_hours: int) -> str | None:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
        try:
            rows = self._read(
                "SELECT payload_json FROM search_cache WHERE query = ? AND provider = ? AND fetched_at >= ?",
                (query.lower(), provider, cutoff),
            )
        except DatabaseError as exc:
            log.warning("search cache read skipped: %s", exc)
            return None
        return str(rows[0]["payload_json"]) if rows else None

    def put_cached(self, query: str, provider: str, payload: str) -> None:
        try:
            with self._tx() as c:
                c.execute(
                    "INSERT INTO search_cache (query, provider, payload_json, fetched_at) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(query, provider) DO UPDATE SET payload_json = excluded.payload_json,"
                    " fetched_at = excluded.fetched_at",
                    (query.lower(), provider, payload, utcnow()),
                )
        except DatabaseError as exc:
            log.warning("search cache write skipped: %s", exc)

    # -- reads -------------------------------------------------------------
    def session_texts(self, session_id: str) -> list[TextRecord]:
        rows = self._read(
            "SELECT turn, content, general_topic, general_confidence, subtopics_json, uncertain, status, created_at"
            " FROM texts WHERE session_id = ? ORDER BY turn",
            (session_id,),
        )
        return [
            TextRecord(
                r["turn"],
                r["content"],
                r["general_topic"],
                r["general_confidence"],
                json.loads(r["subtopics_json"]),
                bool(r["uncertain"]),
                r["status"],
                r["created_at"],
            )
            for r in rows
        ]

    def count(self, table: str) -> int:
        if table not in {
            "sessions",
            "texts",
            "conversation_topics",
            "search_results",
            "search_cache",
            "model_versions",
        }:
            raise ValueError(f"unknown table {table!r}")
        return int(self._read(f"SELECT COUNT(*) FROM {table}")[0][0])  # noqa: S608 - allow-listed table name
