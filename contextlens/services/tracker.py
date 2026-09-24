"""Conversation-level topic tracking with exponential decay.

Each turn the accumulated score of every label is updated as

    score_t = score_{t-1} * decay + weight_t * p_t(label)

where ``p_t`` is the calibrated probability from the classifier. ``weight_t``
is 1.0 for a confident prediction and 0.0 for an uncertain / out-of-taxonomy /
non-English turn (the conversation still ages, but an off-topic remark does not
inject a spurious topic). Uninformative input (empty, stop words only) does not
count as a turn at all.

Two rules on top of the decay (decisions.md D-31):

* **Expiry.** Decay alone never empties the theme: every score is multiplied by
  the same factor, so the *shares* - and with them the theme - survive any
  number of off-topic turns. After ``expire_after`` consecutive turns without a
  confident topical message the context is cleared.
* **Hysteresis.** The *dominant* topic (the one the theme is about) changes only
  after ``confirm_turns`` consecutive confident messages agree on the same new
  topic (``switch_rule="votes"``; with ``"leader"`` the accumulated-score leader
  must differ from it that many turns in a row). A one-message tangent adds a
  secondary topic but does not take over the conversation; a real switch is
  followed after the second message.

All parameters were selected on simulated conversations built from the
validation split (scripts/tune_decay.py, docs/EXPERIMENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThemeTopic:
    id: str
    share: float  # fraction of the accumulated mass


@dataclass
class Theme:
    generals: list[ThemeTopic]  # the dominant topic first, then by share
    subtopics: dict[str, list[ThemeTopic]]  # general id -> active subtopics

    @property
    def is_empty(self) -> bool:
        return not self.generals


@dataclass
class ConversationTracker:
    decay: float = 0.7
    min_share: float = 0.2
    max_topics: int = 3
    # Within an active general topic, a subtopic is part of the theme when it
    # holds at least this share of that topic's subtopic mass.
    sub_min_share: float = 0.35
    max_subtopics: int = 2
    # Clear the context after this many consecutive turns without a confident
    # topical message (0 = never).
    expire_after: int = 5
    # Consecutive confident turns a new score leader needs to become dominant.
    confirm_turns: int = 2
    switch_rule: str = "votes"  # "votes" | "leader"
    general_scores: dict[str, float] = field(default_factory=dict)
    subtopic_scores: dict[str, float] = field(default_factory=dict)
    turns: int = 0
    idle_turns: int = 0  # consecutive turns without a confident topical message
    dominant: str | None = None
    expired: bool = False  # True right after a turn that cleared the context
    _challenger: str | None = None
    _streak: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.expire_after < 0 or self.confirm_turns < 1:
            raise ValueError("expire_after must be >= 0 and confirm_turns >= 1")
        if self.switch_rule not in {"votes", "leader"}:
            raise ValueError("switch_rule must be 'votes' or 'leader'")

    def update(self, general_probs: dict[str, float], subtopic_probs: dict[str, float], weight: float = 1.0) -> None:
        for scores, probs in ((self.general_scores, general_probs), (self.subtopic_scores, subtopic_probs)):
            for key in set(scores) | set(probs):
                scores[key] = scores.get(key, 0.0) * self.decay + weight * float(probs.get(key, 0.0))
        self.turns += 1
        self.expired = False
        if weight > 0:
            self.idle_turns = 0
            vote = max(general_probs, key=lambda k: (general_probs[k], k)) if general_probs else None
            self._follow(vote)
        else:
            self.idle_turns += 1
            if self.expire_after and self.idle_turns >= self.expire_after and self.general_scores:
                self._clear()
                self.expired = True

    def _follow(self, vote: str | None) -> None:
        if not self.general_scores:
            return
        leader = max(self.general_scores, key=lambda k: (self.general_scores[k], k))
        total = sum(self.general_scores.values())
        dom_share = self.general_scores.get(self.dominant, 0.0) / total if self.dominant and total > 0 else 0.0
        if self.dominant is None or dom_share < self.min_share:
            # no dominant yet, or it has faded out of the theme: take the leader
            self.dominant, self._challenger, self._streak = leader, None, 0
            return
        candidate = vote if self.switch_rule == "votes" else leader
        if candidate is None or candidate == self.dominant:
            self._challenger, self._streak = None, 0
            return
        self._streak = self._streak + 1 if candidate == self._challenger else 1
        self._challenger = candidate
        if self._streak >= self.confirm_turns:
            self.dominant, self._challenger, self._streak = candidate, None, 0

    def _clear(self) -> None:
        self.general_scores.clear()
        self.subtopic_scores.clear()
        self.dominant, self._challenger, self._streak = None, None, 0

    def reset(self) -> None:
        self._clear()
        self.turns = 0
        self.idle_turns = 0
        self.expired = False

    @staticmethod
    def active_topics(scores: dict[str, float], min_share: float, limit: int) -> list[ThemeTopic]:
        total = sum(scores.values())
        if total <= 0:
            return []
        ranked = sorted(((k, v / total) for k, v in scores.items()), key=lambda kv: (-kv[1], kv[0]))
        return [ThemeTopic(k, round(s, 4)) for k, s in ranked if s >= min_share][:limit]

    def theme(self, children: dict[str, list[str]]) -> Theme:
        """Active general topics and, for each, its active subtopics.

        ``children`` maps a general topic id to its subtopic ids (taxonomy).
        """
        generals = self.active_topics(self.general_scores, self.min_share, self.max_topics)
        generals.sort(key=lambda t: t.id != self.dominant)  # stable: dominant first, rest by share
        subs: dict[str, list[ThemeTopic]] = {}
        for g in generals:
            within = {s: self.subtopic_scores.get(s, 0.0) for s in children.get(g.id, [])}
            subs[g.id] = self.active_topics(within, self.sub_min_share, self.max_subtopics)
        return Theme(generals, subs)
