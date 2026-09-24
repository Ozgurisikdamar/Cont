"""Conversation-level topic tracking with exponential decay.

Each turn the accumulated score of every label is updated as

    score_t = score_{t-1} * decay + weight_t * p_t(label)

where ``p_t`` is the calibrated probability from the classifier. ``weight_t``
is 1.0 for a confident prediction and 0.0 for an uncertain / out-of-taxonomy
turn (the conversation still ages, but an off-topic remark does not inject a
spurious topic). Uninformative input (empty, stop words only) does not count
as a turn at all. The decay value was selected on simulated conversations built
from the validation split (scripts/tune_decay.py, docs/EXPERIMENTS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThemeTopic:
    id: str
    share: float  # fraction of the accumulated mass


@dataclass
class Theme:
    generals: list[ThemeTopic]
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
    general_scores: dict[str, float] = field(default_factory=dict)
    subtopic_scores: dict[str, float] = field(default_factory=dict)
    turns: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")

    def update(self, general_probs: dict[str, float], subtopic_probs: dict[str, float], weight: float = 1.0) -> None:
        for scores, probs in ((self.general_scores, general_probs), (self.subtopic_scores, subtopic_probs)):
            for key in set(scores) | set(probs):
                scores[key] = scores.get(key, 0.0) * self.decay + weight * float(probs.get(key, 0.0))
        self.turns += 1

    def reset(self) -> None:
        self.general_scores.clear()
        self.subtopic_scores.clear()
        self.turns = 0

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
        subs: dict[str, list[ThemeTopic]] = {}
        for g in generals:
            within = {s: self.subtopic_scores.get(s, 0.0) for s in children.get(g.id, [])}
            subs[g.id] = self.active_topics(within, self.sub_min_share, self.max_subtopics)
        return Theme(generals, subs)
