"""Acceptance and edge-case tests against the REAL trained artifact.

Run with ``pytest -m model``; skipped when models/contextlens-topic is absent.
The cases come from the project brief (tests/acceptance_cases.json) and are
never part of the training data (tests/test_leakage.py).
"""

import json
import statistics
import time
from pathlib import Path

import pytest

from contextlens.config import load_settings
from contextlens.models.artifact import ArtifactError, load_artifact
from contextlens.services.pipeline import ConversationSession

pytestmark = pytest.mark.model
CASES = json.loads((Path(__file__).parent / "acceptance_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def model():
    try:
        return load_artifact(load_settings().model_dir)
    except ArtifactError:
        pytest.skip("trained artifact not available (python train.py)")


@pytest.mark.parametrize("case", CASES["single"], ids=lambda c: c["id"])
def test_single_messages(model, case):
    p = model.predict(case["text"])
    assert p.status == "ok", p.reasons
    assert p.general == case["general"]
    if "subtopic" in case:
        assert p.subtopics[0].id == case["subtopic"]


def test_quantum_is_disambiguated_by_context(model):
    physics = model.predict(CASES["single"][0]["text"])
    tech = model.predict(CASES["single"][1]["text"])
    assert (physics.general, physics.subtopics[0].id) == ("physics", "quantum_mechanics")
    assert (tech.general, tech.subtopics[0].id) == ("technology", "quantum_computing")


@pytest.mark.parametrize("case", CASES["ood"], ids=lambda c: c["id"])
def test_out_of_taxonomy_is_uncertain(model, case):
    assert model.predict(case["text"]).status == "uncertain"


def test_conversation_theme_accumulates(model, taxonomy):
    session = ConversationSession(model, taxonomy, load_settings(), db=None, searcher=None)
    expected = {e["after_turn"]: e for e in CASES["conversation"]["expected_after"]}
    for turn, message in enumerate(CASES["conversation"]["messages"], start=1):
        session.process(message)
        if turn in expected:
            theme = session.current_theme()
            assert set(theme.generals) == set(expected[turn]["generals"])
            assert theme.phrase == expected[turn]["phrase"]


def test_reset_clears_the_theme(model, taxonomy):
    session = ConversationSession(model, taxonomy, load_settings(), db=None, searcher=None)
    session.process(CASES["single"][0]["text"])
    session.reset()
    assert session.current_theme().generals == ()


@pytest.mark.parametrize("text", ["", "   ", "!!!", "12345", "🙂🙂🙂", "the and of", "https://example.com"])
def test_uninformative_input(model, text):
    assert model.predict(text).status == "uninformative"


def test_markup_and_case_do_not_change_the_answer(model):
    clean = model.predict("Quantum processors can speed up certain algorithms by using qubits.")
    noisy = model.predict("<b>QUANTUM PROCESSORS</b> can speed up certain algorithms by using qubits!!! @user #tech")
    assert noisy.general == clean.general == "technology"


def test_very_long_input_is_handled(model):
    p = model.predict("Cells carry DNA and living things diversify through evolution. " * 2000)
    assert p.general == "biology" and p.status == "ok"


def test_non_english_input_is_not_confidently_classified(model):
    assert model.predict("Bu akşam arkadaşlarımla sinemaya gideceğim.").status == "uncertain"


def test_probabilities_are_well_formed(model):
    p = model.predict(CASES["single"][2]["text"])
    assert abs(sum(p.general_probs.values()) - 1.0) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in p.subtopic_probs.values())
    assert all(s.id in {x for x in p.subtopic_probs} for s in p.subtopics)


def test_single_message_latency_on_cpu(model):
    texts = [c["text"] for c in CASES["single"]] * 4
    model.predict(texts[0])  # warm-up
    times = []
    for t in texts:
        t0 = time.perf_counter()
        model.predict(t)
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.5  # seconds; measured values are in docs/TEST_REPORT.md
