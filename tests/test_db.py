import json
import sqlite3

import pytest

from contextlens.database.db import SCHEMA_VERSION, Database, DatabaseError


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "t.db")
    yield d
    d.close()


def add_text(db, sid, turn=1, content="hello"):
    return db.add_text(
        sid,
        turn,
        content,
        general_topic="physics",
        general_confidence=0.9,
        subtopics=[{"id": "relativity", "probability": 0.8}],
        general_probs={"physics": 0.9},
        uncertain=False,
        status="ok",
        model_version_id=None,
    )


def test_schema_and_pragmas(db):
    assert db.schema_version == SCHEMA_VERSION
    assert db.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"sessions", "texts", "conversation_topics", "search_results", "search_cache", "model_versions"} <= tables
    indexes = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"idx_texts_session", "idx_topics_session", "idx_results_topic"} <= indexes


def test_round_trip(db):
    mv = db.register_model("m", "1.0", "2026-01-01", "abc", {"acc": 0.9})
    assert db.register_model("m", "1.0", "2026-01-01", "abc", {}) == mv  # idempotent
    sid = db.start_session(mv)
    tid = add_text(db, sid)
    topic = db.add_conversation_topic(sid, tid, "Physics", "physics", {"physics": 1.0}, {}, "physics")
    n = db.add_search_results(
        sid,
        topic,
        "physics",
        "wikipedia",
        [
            {
                "rank": 1,
                "title": "Physics",
                "summary": "s",
                "url": "https://en.wikipedia.org/wiki/Physics",
                "source": "Wikipedia",
            }
        ],
    )
    assert n == 1
    rows = db.session_texts(sid)
    assert rows[0].general_topic == "physics" and rows[0].subtopics[0]["id"] == "relativity"
    assert db.count("search_results") == 1
    db.end_session(sid)
    assert db.conn.execute("SELECT ended_at FROM sessions WHERE id=?", (sid,)).fetchone()[0] is not None


def test_foreign_keys_enforced(db):
    with pytest.raises(DatabaseError):
        add_text(db, "no-such-session")


def test_sql_injection_is_stored_literally(db):
    sid = db.start_session(None)
    evil = "x'); DROP TABLE texts; --"
    add_text(db, sid, content=evil)
    assert db.session_texts(sid)[0].content == evil
    assert db.count("texts") == 1


def test_count_rejects_unknown_table(db):
    with pytest.raises(ValueError):
        db.count("texts; DROP TABLE texts")


def test_search_cache_ttl(db):
    db.put_cached("Qubit", "wikipedia", json.dumps([{"rank": 1}]))
    assert db.get_cached("qubit", "wikipedia", ttl_hours=1) is not None  # case-insensitive key
    db.conn.execute("UPDATE search_cache SET fetched_at = '2000-01-01T00:00:00+00:00'")
    assert db.get_cached("qubit", "wikipedia", ttl_hours=1) is None


def test_reopen_does_not_remigrate(tmp_path):
    path = tmp_path / "t.db"
    Database(path).close()
    d = Database(path)
    assert d.schema_version == SCHEMA_VERSION
    d.close()


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    conn.close()
    with pytest.raises(DatabaseError, match="newer"):
        Database(path)


def test_unwritable_location_raises_database_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(DatabaseError):
        Database(blocker / "sub" / "t.db")


def test_locked_database_surfaces_as_database_error(tmp_path, monkeypatch):
    import contextlens.database.db as dbmod

    path = tmp_path / "t.db"
    d = Database(path)
    sid = d.start_session(None)
    other = sqlite3.connect(path, timeout=0.1)
    other.execute("BEGIN EXCLUSIVE")
    d.conn.execute("PRAGMA busy_timeout = 100")
    with pytest.raises(dbmod.DatabaseError):
        add_text(d, sid)
    other.rollback()
    other.close()
    d.close()


def test_stored_message_length_is_bounded(tmp_path):
    from contextlens.database.db import MAX_CONTENT_CHARS, Database

    db = Database(tmp_path / "c.db")
    sid = db.start_session(None)
    db.add_text(
        sid,
        1,
        "x" * (MAX_CONTENT_CHARS * 3),
        general_topic=None,
        general_confidence=None,
        subtopics=[],
        general_probs={},
        uncertain=True,
        status="uncertain",
        model_version_id=None,
    )
    assert len(db.session_texts(sid)[0].content) == MAX_CONTENT_CHARS
    db.close()
