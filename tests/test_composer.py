from contextlens.services.composer import EMPTY_THEME, compose
from contextlens.services.tracker import Theme, ThemeTopic


def theme(*generals, subs=None):
    n = len(generals)
    return Theme([ThemeTopic(g, 1 / n) for g in generals], subs or {})


def test_empty(taxonomy):
    assert compose(Theme([], {}), taxonomy) == EMPTY_THEME


def test_single_topic_uses_dominant_subtopic(taxonomy):
    c = compose(theme("technology", subs={"technology": [ThemeTopic("quantum_computing", 0.8)]}), taxonomy)
    assert c.label == "Technology > Quantum Computing"
    assert c.phrase == "quantum computing"


def test_single_topic_without_dominant_subtopic(taxonomy):
    c = compose(theme("sports", subs={"sports": [ThemeTopic("football", 0.4)]}), taxonomy)
    assert c.phrase == "sports"


def test_books_plus_science_is_science_books(taxonomy):
    assert compose(theme("books", "science"), taxonomy).phrase == "science books"


def test_books_plus_history_is_history_books(taxonomy):
    assert compose(theme("history", "books"), taxonomy).phrase == "history books"


def test_books_science_biology_narrows(taxonomy):
    c = compose(theme("books", "science", "biology"), taxonomy)
    assert c.phrase == "science books about biology"
    assert c.label == "Books + Science + Biology"


def test_books_plus_biology_without_science(taxonomy):
    assert compose(theme("biology", "books"), taxonomy).phrase == "biology books"


def test_history_is_a_perspective(taxonomy):
    assert compose(theme("history", "physics"), taxonomy).phrase == "history of physics"
    assert compose(theme("science", "history"), taxonomy).phrase == "history of science"


def test_narrowing_without_format(taxonomy):
    assert compose(theme("science", "biology"), taxonomy).phrase == "biology (science)"


def test_unrelated_pair(taxonomy):
    assert compose(theme("sports", "technology"), taxonomy).phrase == "sports and technology"


def test_concepts_are_the_single_ideas_behind_the_phrase(taxonomy):
    c = compose(theme("books", "science", "biology"), taxonomy)
    assert c.concepts == ("biology", "science")  # narrower domain first, the format is not a concept
    assert compose(theme("sports"), taxonomy).concepts == ()  # nothing simpler than the phrase itself
    assert compose(theme("history", "physics"), taxonomy).concepts == ("history", "physics")
