from contextlens.data.passages import article_paragraphs, clean_text, is_heading, make_passages, split_sentences

ARTICLE = """Qubit

A qubit ( ; ) is the basic unit of quantum information in quantum computing. It is a two-state quantum-mechanical system, e.g. the spin of an electron.
Unlike a bit, a qubit can be in a superposition of both states at the same time, which U.S. and European labs exploit.

History
The term was coined by Benjamin Schumacher in 1995 while he worked on quantum coding problems at Kenyon College.

References
Schumacher, B. (1995). Quantum coding. Physical Review A.
"""


def test_clean_text_removes_empty_parentheses():
    assert clean_text("A qubit ( ; ) is a unit ,here") == "A qubit is a unit,here"


def test_heading_detection():
    assert is_heading("History")
    assert is_heading("Early life and career")
    assert not is_heading("A qubit is the basic unit of quantum information.")


def test_paragraphs_skip_headings_and_stop_at_back_matter():
    paras = article_paragraphs(ARTICLE)
    assert len(paras) == 3
    assert all("Physical Review" not in p for p in paras)


def test_sentence_split_respects_abbreviations():
    sents = split_sentences("It is a system, e.g. the spin of an electron. Labs in the U.S. use it. Dr. Smith agrees.")
    assert sents == ["It is a system, e.g. the spin of an electron.", "Labs in the U.S. use it.", "Dr. Smith agrees."]


def test_initials_are_not_sentence_boundaries():
    assert len(split_sentences("J. R. R. Tolkien wrote novels. He was a philologist.")) == 2


def test_make_passages_bounds():
    passages = make_passages(ARTICLE, min_words=12, max_words=60, max_passages=4)
    assert 1 <= len(passages) <= 4
    for p in passages:
        assert 12 <= len(p.split()) <= 60
    assert passages[0].startswith("A qubit is the basic unit")


def test_max_passages_respected():
    assert len(make_passages(ARTICLE, min_words=5, max_words=60, max_passages=1)) == 1
