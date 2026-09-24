"""End-to-end chain with a tiny model trained on the fly (fake encoder, no network)."""

import dataclasses
import json

import numpy as np
import pytest

from contextlens.cli import app as cli
from contextlens.database.db import Database
from contextlens.models.artifact import ArtifactError, load_artifact, save_artifact
from contextlens.net import NetworkError
from contextlens.services.pipeline import ConversationSession
from contextlens.services.websearch import WebSearcher

from .conftest import FakeEncoder

WIKI = {
    "query": {
        "pages": [
            {
                "index": 1,
                "title": "Qubit",
                "extract": "A qubit is a unit.",
                "fullurl": "https://en.wikipedia.org/wiki/Qubit",
            }
        ]
    }
}


def fake_fetch(fail=False):
    def fetch(url, params, **kw):
        if fail:
            raise NetworkError("offline")
        return WIKI if "wikipedia" in url else {}

    return fetch


def session(model, taxonomy, settings, db=None, fail_web=False):
    searcher = WebSearcher(cache=db, fetch=fake_fetch(fail_web), max_results=3)
    return ConversationSession(model, taxonomy, settings, db, searcher, {"qubit": 5.0, "processor": 4.0})


def test_prediction_and_hierarchy(fake_model, taxonomy):
    p = fake_model.predict("qubit quantum processor gate algorithm circuit")
    assert p.general == "technology" and p.subtopics[0].id == "quantum_computing"
    assert taxonomy.is_consistent(p.general, [s.id for s in p.subtopics])
    assert abs(sum(p.general_probs.values()) - 1) < 1e-6
    p2 = fake_model.predict("particle wave function entanglement superposition")
    assert p2.general == "physics" and p2.subtopics[0].id == "quantum_mechanics"


def test_uninformative_and_out_of_vocabulary(fake_model):
    for text in ["", "   ", "!!!", "12345", "and the of"]:
        assert fake_model.predict(text).status == "uninformative"
    far = fake_model.predict("zzz yyy xxx www vvv")  # no overlap with any training vocabulary
    assert far.status == "uncertain"


def test_full_chain_persists_everything(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    s = session(fake_model, taxonomy, settings, db)
    r = s.process("qubit quantum processor gate algorithm circuit")
    assert r.theme.phrase == "quantum computing"
    assert r.query is not None and r.query.primary.startswith("quantum computing")
    assert r.search is not None and r.search.status == "ok" and r.search.results[0].title == "Qubit"
    assert db.count("texts") == 1 and db.count("conversation_topics") == 1 and db.count("search_results") == 1
    stored = db.conn.execute("SELECT general_topic, status, model_version_id FROM texts").fetchone()
    assert stored["general_topic"] == "technology" and stored["model_version_id"] is not None
    s.close()
    db.close()


def test_offline_web_does_not_break_classification_or_logging(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    s = session(fake_model, taxonomy, settings, db, fail_web=True)
    r = s.process("star galaxy supernova black hole nebula")
    assert r.prediction.general == "physics"
    assert r.search.status == "offline" and r.search.results == ()
    assert db.count("texts") == 1 and db.count("conversation_topics") == 1 and db.count("search_results") == 0
    db.close()


def test_database_failure_is_a_warning_not_a_crash(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    s = session(fake_model, taxonomy, settings, db)
    db.conn.close()  # simulate the database becoming unusable mid-session
    r = s.process("football goal striker league club match")
    assert r.prediction.general == "sports"
    assert any("database" in w for w in r.warnings)
    assert s.history()[0].general_topic == "sports"  # local history still works


def test_conversation_theme_and_reset(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    s = session(fake_model, taxonomy, settings, db)
    s.process("novel plot character narrator chapter fiction")
    s.process("hypothesis experiment observation test method evidence")
    assert set(s.current_theme().generals) == {"books", "science"}
    assert s.current_theme().phrase == "science books"
    old = s.session_id
    s.reset()
    assert s.session_id != old and s.current_theme().generals == ()
    assert len(s.history()) == 0
    assert db.count("texts") == 2  # reset keeps stored records
    assert db.conn.execute("SELECT ended_at FROM sessions WHERE id=?", (old,)).fetchone()[0] is not None
    db.close()


def test_uncertain_turn_does_not_change_theme(fake_model, taxonomy, settings):
    s = session(fake_model, taxonomy, settings)
    s.process("football goal striker league club match")
    before = s.current_theme().generals
    r = s.process("zzz yyy xxx www vvv qubit")
    assert r.prediction.uncertain
    assert s.current_theme().generals == before
    # the theme is still searched, but the off-topic message adds no keywords ("qubit" is in the vocabulary)
    assert r.query is not None and r.query.primary == r.query.fallback == s.current_theme().phrase


def test_database_that_cannot_start_a_session_falls_back_to_local_history(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    db.conn.close()
    s = session(fake_model, taxonomy, settings, db)
    assert s.session_id == "offline-session"
    r = s.process("football goal striker league club match")
    assert r.prediction.general == "sports" and r.warnings
    assert [h.general_topic for h in s.history()] == ["sports"]


def test_artifact_round_trip_and_integrity(fake_model, tmp_path):
    out = tmp_path / "model"
    save_artifact(fake_model, out, {"qubit": 5.0}, save_encoder=False)
    loaded = load_artifact(out, encoder=FakeEncoder())
    text = "gene dna chromosome allele heredity"
    a, b = fake_model.predict(text), loaded.predict(text)
    assert a.general == b.general and np.isclose(a.confidence, b.confidence)
    # tampering is detected
    (out / "vocabulary.json").write_text(json.dumps({"x": 1.0}))
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(out, encoder=FakeEncoder())


class _OtherEncoder(FakeEncoder):
    """Same dimension, different embedding space (a stand-in for the wrong encoder)."""

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return np.roll(super().encode(texts), 7, axis=1)


def test_artifact_refuses_an_encoder_it_was_not_trained_with(fake_model, tmp_path):
    out = tmp_path / "model"
    save_artifact(fake_model, out, {}, save_encoder=False)
    with pytest.raises(ArtifactError, match="does not reproduce"):
        load_artifact(out, encoder=_OtherEncoder())


def test_artifact_without_encoder_fingerprint_is_refused(fake_model, tmp_path):
    out = tmp_path / "model"
    save_artifact(fake_model, out, {}, save_encoder=False)
    meta = json.loads((out / "metadata.json").read_text())
    del meta["encoder_probe"]
    (out / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="fingerprint"):
        load_artifact(out, encoder=FakeEncoder())


def test_unloadable_encoder_becomes_an_artifact_error(fake_model, tmp_path, monkeypatch):
    from contextlens.models import artifact

    out = tmp_path / "model"
    save_artifact(fake_model, out, {}, save_encoder=False)
    meta = json.loads((out / "metadata.json").read_text())
    meta["encoder"] = "minilm-l6-ft"
    (out / "metadata.json").write_text(json.dumps(meta))

    def missing(*args, **kwargs):
        raise FileNotFoundError("fine-tuned encoder not found")

    monkeypatch.setattr(artifact, "SentenceEncoder", missing)
    with pytest.raises(ArtifactError, match="cannot load the sentence encoder"):
        load_artifact(out)


def test_missing_artifact_explains_how_to_build(tmp_path):
    with pytest.raises(ArtifactError, match="python train.py"):
        load_artifact(tmp_path / "nothing")


def test_cli_script_mode(fake_model, tmp_path, monkeypatch):
    out = tmp_path / "model"
    save_artifact(fake_model, out, {"qubit": 5.0}, save_encoder=False)
    monkeypatch.setattr(cli, "load_artifact", lambda d, **kw: load_artifact(d, encoder=FakeEncoder(), **kw))
    script = tmp_path / "msgs.txt"
    script.write_text("help\nqubit quantum processor gate algorithm circuit\nhistory\nreset\n!!!\nq\n")
    lines: list[str] = []
    code = cli.main(
        ["--model-dir", str(out), "--db", str(tmp_path / "c.db"), "--no-web", "--script", str(script)],
        print_=lines.append,
    )
    text = "\n".join(lines)
    assert code == 0
    assert "General topic : Technology" in text
    assert "Quantum Computing" in text
    assert "Conversation reset" in text
    assert "Nothing topical" in text
    assert "Goodbye" in text
    db = Database(tmp_path / "c.db")
    assert db.count("texts") == 1
    db.close()


def test_cli_missing_model_exit_code(tmp_path):
    lines: list[str] = []
    assert cli.main(["--model-dir", str(tmp_path / "none"), "--once", "hi"], print_=lines.append) == 2
    assert any("train.py" in ln for ln in lines)


def test_cli_eof_and_interrupt(fake_model, taxonomy, settings, monkeypatch):
    s = session(fake_model, taxonomy, settings)
    for exc in (EOFError, KeyboardInterrupt):

        def raiser(_prompt, exc=exc):
            raise exc

        monkeypatch.setattr("builtins.input", raiser)
        lines: list[str] = []
        assert cli.run_loop(s, None, lines.append) == 0
        assert "oodbye" in lines[-1]


def test_cli_unreadable_script_is_an_error_not_a_traceback(fake_model, tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    save_artifact(fake_model, model_dir, {}, save_encoder=False)
    monkeypatch.setattr(cli, "load_artifact", lambda d, **kw: load_artifact(d, encoder=FakeEncoder(), **kw))
    out: list[str] = []
    code = cli.main(
        [
            "--model-dir",
            str(model_dir),
            "--db",
            str(tmp_path / "x.db"),
            "--no-web",
            "--script",
            str(tmp_path / "nope.txt"),
        ],
        print_=out.append,
    )
    assert code == 2 and any("cannot read script file" in line for line in out)


def test_reported_subtopics_are_capped(fake_model):
    loose = dataclasses.replace(fake_model, subtopic_threshold=0.0, max_subtopics=2)  # every sibling passes
    assert len(loose.predict("particle wave function entanglement spacetime newton star").subtopics) == 2
    capped = dataclasses.replace(fake_model, subtopic_threshold=0.0, max_subtopics=0)
    assert len(capped.predict("particle wave function entanglement").subtopics) == 1  # best one is always kept


def test_artifact_with_unknown_head_type_is_refused(fake_model, tmp_path):
    import skops.io as sio

    from contextlens.models import artifact

    out = tmp_path / "model"
    save_artifact(fake_model, out, {}, save_encoder=False)
    heads = sio.load(out / "heads.skops", trusted=sorted(sio.get_untrusted_types(file=out / "heads.skops")))
    heads["head_type"] = "mystery"
    sio.dump(heads, out / "heads.skops")
    meta = json.loads((out / "metadata.json").read_text())
    meta["checksums"]["heads.skops"] = artifact.sha256_file(out / "heads.skops")
    (out / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="unknown head_type"):
        load_artifact(out, encoder=FakeEncoder())


def test_empty_batch_is_handled(fake_model):
    gp, cond, ood = fake_model.predict_proba([])
    assert gp.shape == (0, len(fake_model.general_ids)) and cond.shape[0] == 0 and ood.shape == (0,)
    assert fake_model.predict_many([]) == []
