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
    assert {g.id for g in theme.generals} == {"science", "books"}
    assert theme.generals[0].id == "books"  # one Science message does not take over the conversation
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


# --- expiry and tangent handling (decisions.md D-31) -----------------------

PHYS, BOOKS = {"physics": 0.9, "books": 0.1}, {"books": 0.9, "physics": 0.1}
OFF = {"sports": 0.5, "books": 0.5}  # an uncertain, off-topic message


def _run(tracker, seq):
    for probs, weight in seq:
        tracker.update(probs, {}, weight)
    return tracker


@pytest.mark.parametrize("n_uncertain, empty", [(1, False), (2, False), (5, True), (20, True)])
def test_theme_expires_after_an_uncertain_streak(n_uncertain, empty):
    t = _run(ConversationTracker(expire_after=3), [(PHYS, 1.0)] + [(OFF, 0.0)] * n_uncertain)
    assert t.theme(CHILDREN).is_empty is empty


def test_decay_alone_never_expires_the_theme():
    # the v1.0 behaviour this rule fixes: shares survive any number of zero-weight turns
    t = _run(ConversationTracker(expire_after=0), [(PHYS, 1.0)] + [(OFF, 0.0)] * 20)
    assert [g.id for g in t.theme(CHILDREN).generals] == ["physics"]


def test_a_confident_message_resets_the_uncertain_streak():
    t = _run(ConversationTracker(expire_after=3), [(PHYS, 1.0), (OFF, 0.0), (OFF, 0.0), (PHYS, 1.0), (OFF, 0.0)])
    assert not t.theme(CHILDREN).is_empty and t.idle_turns == 1


def test_one_message_tangent_does_not_switch_the_dominant_topic():
    t = _run(ConversationTracker(), [(PHYS, 1.0)] * 5 + [(BOOKS, 1.0)] + [(PHYS, 1.0)])
    assert t.dominant == "physics"
    t = _run(ConversationTracker(), [(PHYS, 1.0)] * 3 + [(BOOKS, 1.0)])
    assert t.dominant == "physics" and t.theme(CHILDREN).generals[0].id == "physics"


def test_a_real_switch_is_followed():
    t = _run(ConversationTracker(), [(PHYS, 1.0)] * 5 + [(BOOKS, 1.0)] * 5)
    assert t.dominant == "books" and t.theme(CHILDREN).generals[0].id == "books"
    t = _run(ConversationTracker(), [(PHYS, 1.0)] * 2 + [(BOOKS, 1.0)] * 3)
    assert t.dominant == "books"


def test_switch_needs_consecutive_agreement():
    t = _run(ConversationTracker(), [(PHYS, 1.0)] * 3 + [(BOOKS, 1.0), (PHYS, 1.0), (BOOKS, 1.0)])
    assert t.dominant == "physics"


def test_invalid_tracker_settings():
    with pytest.raises(ValueError):
        ConversationTracker(confirm_turns=0)
    with pytest.raises(ValueError):
        ConversationTracker(switch_rule="other")
