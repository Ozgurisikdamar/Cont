import pytest

from contextlens.services.tracker import ConversationTracker

CHILDREN = {"books": ["novels", "science_books"], "science": ["scientific_method"], "biology": ["genetics"]}


def test_decay_update_formula():
    t = ConversationTracker(decay=0.7)
    t.update({"books": 0.8, "science": 0.2}, {})
    t.update({"books": 0.1, "science": 0.9}, {})
    assert t.general_scores["books"] == pytest.approx(0.8 * 0.7 + 0.1)
    assert t.general_scores["science"] == pytest.approx(0.2 * 0.7 + 0.9)
    assert t.turns == 2


def test_zero_weight_turn_only_ages_context():
    t = ConversationTracker(decay=0.5)
    t.update({"books": 1.0}, {})
    t.update({"sports": 1.0}, {}, weight=0.0)
    assert t.general_scores["books"] == pytest.approx(0.5)
    assert t.general_scores["sports"] == 0.0


def test_theme_keeps_topics_above_share():
    t = ConversationTracker(decay=0.7, min_share=0.2)
    t.update({"books": 1.0}, {"novels": 1.0})
    t.update({"science": 1.0}, {"scientific_method": 1.0})
    theme = t.theme(CHILDREN)
    assert [g.id for g in theme.generals] == ["science", "books"]
    assert theme.subtopics["books"][0].id == "novels"


def test_old_topics_fade_out():
    t = ConversationTracker(decay=0.3, min_share=0.2)
    t.update({"books": 1.0}, {})
    for _ in range(3):
        t.update({"biology": 1.0}, {})
    assert [g.id for g in t.theme(CHILDREN).generals] == ["biology"]


def test_reset_clears_state():
    t = ConversationTracker()
    t.update({"books": 1.0}, {"novels": 1.0})
    t.reset()
    assert t.theme(CHILDREN).is_empty
    assert t.turns == 0


def test_invalid_decay():
    with pytest.raises(ValueError):
        ConversationTracker(decay=1.5)
