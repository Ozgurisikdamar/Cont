from contextlens.services.query import build_query, salient_keywords

VOCAB = {"qubits": 6.1, "processors": 4.2, "algorithms": 3.9, "quantum": 3.0, "computing": 2.5}


def test_query_combines_theme_and_rare_keywords():
    q = build_query("quantum computing", "Quantum processors use qubits to speed up algorithms.", VOCAB)
    assert q.primary == "quantum computing qubits processors"
    assert q.fallback == "quantum computing"
    assert q.candidates() == ["quantum computing qubits processors", "quantum computing"]


def test_words_outside_vocabulary_are_never_sent():
    # Names and rare personal tokens are not in the public training vocabulary.
    q = build_query("novels", "My friend Zeynep Yilmaz lent me her novel", {"novel": 3.0})
    assert "zeynep" not in q.primary.lower() and "yilmaz" not in q.primary.lower()


def test_theme_words_are_not_repeated():
    assert salient_keywords("quantum computing with qubits", VOCAB, "quantum computing", 2) == ["qubits"]


def test_empty_theme_gives_empty_query():
    q = build_query("", "anything", VOCAB)
    assert q.candidates() == []


def test_keyword_refinement_can_be_disabled():
    q = build_query("quantum computing", "qubits processors", VOCAB, max_keywords=0)
    assert q.primary == q.fallback == "quantum computing"
    assert q.candidates() == ["quantum computing"]
