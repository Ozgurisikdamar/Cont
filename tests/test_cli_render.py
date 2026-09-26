"""Console rendering: an uncertain message must read as uncertain, never as a confident topic."""

import dataclasses

from contextlens.cli import app as cli
from contextlens.database.db import Database
from contextlens.models.language import LanguageGate
from contextlens.services.pipeline import ConversationSession

ON_TOPIC = "qubit quantum processor gate algorithm circuit"
NOT_ENGLISH = "bu aksam sinemaya gidecegim"


class _TurkishWhenIyor:
    """fastText stand-in: Turkish when a word ends in -iyor/-ecegim, else English."""

    def predict(self, text, k=1):
        tr = any(w.endswith(("iyor", "ecegim")) for w in text.lower().split())
        return (["__label__tr"], [0.95]) if tr else (["__label__en"], [0.9])


def _text(result, taxonomy) -> str:
    lines: list[str] = []
    cli.render_turn(result, taxonomy.display, lines.append)
    return "\n".join(lines)


def test_uncertain_turn_leads_with_the_verdict_not_the_rejected_topic(fake_model, taxonomy, settings):
    strict = dataclasses.replace(fake_model, ood_threshold=2.0)  # every message is now "uncertain"
    result = ConversationSession(strict, taxonomy, settings, db=None, searcher=None).process(ON_TOPIC)
    pred = result.prediction
    assert pred.status == "uncertain" and pred.general is not None
    text = _text(result, taxonomy)
    topic = taxonomy.display(pred.general)
    assert "General topic : uncertain - no confident topic." in text
    assert f"General topic : {topic}" not in text  # the rejected guess is not the headline
    assert "Confidence    :" not in text and "Subtopics     :" not in text
    assert "Reason        : far from all training topics" in text
    guess = next(line for line in text.splitlines() if line.startswith("Best guess    : "))
    assert guess.startswith(f"Best guess    : {topic} (") and guess.endswith(" - not used")
    assert taxonomy.display(pred.subtopics[0].id) in guess  # information is kept, only demoted
    assert text.index("uncertain") < text.index("Best guess")
    assert "This message was not added to the conversation theme." in text


def test_confident_turn_keeps_the_topic_headline(fake_model, taxonomy, settings):
    result = ConversationSession(fake_model, taxonomy, settings, db=None, searcher=None).process(ON_TOPIC)
    text = _text(result, taxonomy)
    assert result.prediction.status == "ok"
    assert "General topic : Technology" in text and "Confidence    : " in text
    assert "  - Quantum Computing (" in text
    assert "uncertain" not in text and "Best guess" not in text


def test_best_guess_without_subtopics():
    assert cli.best_guess("books", 0.53, [], str.title) == "Books (53.0%)"
    assert cli.best_guess("books", 0.53, ["Novels (94.3%)"], str.title) == "Books (53.0%) > Novels (94.3%)"


def test_history_marks_uncertain_and_non_english_messages(fake_model, taxonomy, settings):
    db = Database(settings.db_path)
    session = ConversationSession(fake_model, taxonomy, settings, db, searcher=None)
    session.process(ON_TOPIC)
    session.model = dataclasses.replace(fake_model, ood_threshold=2.0)
    session.process(ON_TOPIC)
    gate = LanguageGate(_TurkishWhenIyor(), frozenset(), {"1": 0.5, "2": 0.5, "3": 0.3, "4+": 0.3})
    session.model = dataclasses.replace(fake_model, language_gate=gate)
    session.process(NOT_ENGLISH)
    lines: list[str] = []
    cli.render_history(session, taxonomy.display, lines.append)
    confident, uncertain, foreign = (line.split("-> ", 1)[1] for line in lines if "-> " in line)
    assert confident.startswith("Technology (") and "Quantum Computing" in confident
    assert uncertain.startswith("uncertain (best guess: Technology (") and "Quantum Computing (" in uncertain
    assert foreign == "not English (not analysed)"
    session.close()
    db.close()
