"""Settings: defaults, environment overrides and type coercion."""

from pathlib import Path

from contextlens.config import load_settings


def test_min_confidence_defaults_to_the_artifact_value():
    assert load_settings().min_confidence is None


def test_environment_overrides_are_coerced(monkeypatch):
    monkeypatch.setenv("CONTEXTLENS_MIN_CONFIDENCE", "0.55")
    monkeypatch.setenv("CONTEXTLENS_WEB_ENABLED", "no")
    monkeypatch.setenv("CONTEXTLENS_MAX_SUBTOPICS", "2")
    monkeypatch.setenv("CONTEXTLENS_DB_PATH", "~/topics.db")
    s = load_settings()
    assert s.min_confidence == 0.55 and s.web_enabled is False and s.max_subtopics == 2
    assert s.db_path == Path("~/topics.db").expanduser()


def test_min_confidence_can_be_reset_to_the_artifact_value(monkeypatch):
    monkeypatch.setenv("CONTEXTLENS_MIN_CONFIDENCE", "model")
    assert load_settings().min_confidence is None


def test_explicit_overrides_win_over_environment(monkeypatch):
    monkeypatch.setenv("CONTEXTLENS_DECAY", "0.3")
    assert load_settings(decay=0.9).decay == 0.9
