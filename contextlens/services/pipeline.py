"""One conversation turn, end to end.

    text -> TopicModel -> DB(texts) -> ConversationTracker -> composer -> query
         -> WebSearcher -> DB(conversation_topics, search_results)

Failure isolation (requirement: the ML layer must not depend on the web or
database layers): a web failure yields an empty result with a status; a
database failure is logged and reported as a warning while classification,
tracking and search continue. Only the model is required.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from contextlens.config import Settings
from contextlens.database.db import Database, DatabaseError, TextRecord
from contextlens.models.topic_model import Prediction, TopicModel
from contextlens.services.composer import EMPTY_THEME, ComposedTheme, compose
from contextlens.services.query import SearchQuery, build_query
from contextlens.services.tracker import ConversationTracker
from contextlens.services.websearch import SearchOutcome, WebSearcher
from contextlens.taxonomy import Taxonomy

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    prediction: Prediction
    theme: ComposedTheme
    query: SearchQuery | None
    search: SearchOutcome | None
    warnings: list[str] = field(default_factory=list)


class ConversationSession:
    """Stateful conversation: owns the tracker and the current session id."""

    def __init__(
        self,
        model: TopicModel,
        taxonomy: Taxonomy,
        settings: Settings,
        db: Database | None,
        searcher: WebSearcher | None,
        vocabulary: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.tax = taxonomy
        self.settings = settings
        self.db = db
        self.searcher = searcher
        self.vocabulary = vocabulary or {}
        self.children = {g: taxonomy.children_of(g) for g in taxonomy.general_ids}
        self.tracker = self._new_tracker()
        self.turn = 0
        self.model_version_id: int | None = None
        self.local_history: list[TextRecord] = []  # used when the database is unavailable
        self.session_id = self._start_session()

    # -- lifecycle -------------------------------------------------------
    def _new_tracker(self) -> ConversationTracker:
        s = self.settings
        return ConversationTracker(decay=s.decay, min_share=s.theme_min_share, max_topics=s.max_theme_topics)

    def _db_call(self, what: str, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call ``Database.<method>``; a database failure becomes a warning, never a crash."""
        if self.db is None:
            return None
        try:
            return getattr(self.db, method)(*args, **kwargs)
        except DatabaseError as exc:
            log.error("database %s failed: %s", what, exc)
            self._warnings.append(f"could not save {what} to the database")
            return None

    def _start_session(self) -> str:
        self._warnings: list[str] = []
        meta = self.model.metadata
        if self.db is not None and self.model_version_id is None:
            self.model_version_id = self._db_call(
                "model version",
                "register_model",
                meta.get("model_name", "contextlens-topic"),
                self.model.version,
                meta.get("trained_at", ""),
                meta.get("dataset_version", ""),
                meta.get("test_metrics", {}),
            )
        session_id = self._db_call("session", "start_session", self.model_version_id)
        return session_id or "offline-session"

    def reset(self) -> None:
        """Start a new conversation context; stored records are kept."""
        self._db_call("session end", "end_session", self.session_id)
        self.tracker = self._new_tracker()
        self.turn = 0
        self.local_history = []
        self.session_id = self._start_session()

    def close(self) -> None:
        self._db_call("session end", "end_session", self.session_id)

    # -- one turn ----------------------------------------------------------
    def process(self, text: str) -> TurnResult:
        self._warnings = []
        pred = self.model.predict(text)
        if pred.status == "uninformative":
            return TurnResult(pred, self.current_theme(), None, None, ["nothing topical to analyse"])
        self.turn += 1
        subs = [{"id": s.id, "probability": round(s.probability, 4)} for s in pred.subtopics]
        text_id = self._db_call(
            "text",
            "add_text",
            self.session_id,
            self.turn,
            text,
            general_topic=pred.general,
            general_confidence=pred.confidence,
            subtopics=subs,
            general_probs={k: round(v, 4) for k, v in pred.general_probs.items()},
            uncertain=pred.uncertain,
            status=pred.status,
            model_version_id=self.model_version_id,
        )
        self.local_history.append(
            TextRecord(self.turn, text, pred.general, pred.confidence, subs, pred.uncertain, pred.status, "")
        )
        # Uncertain / off-topic turns age the context but add no topic mass.
        self.tracker.update(pred.general_probs, pred.subtopic_probs, weight=0.0 if pred.uncertain else 1.0)
        theme = self.current_theme()
        query = build_query(theme.phrase, text, self.vocabulary) if theme.phrase else None
        outcome = None
        if query is not None and self.searcher is not None:
            outcome = self.searcher.search(query.candidates())
        if text_id is not None:
            topic_id = self._db_call(
                "conversation topic",
                "add_conversation_topic",
                self.session_id,
                text_id,
                theme.label,
                theme.phrase,
                {k: round(v, 4) for k, v in self.tracker.general_scores.items()},
                {k: round(v, 4) for k, v in self.tracker.subtopic_scores.items() if v >= 0.01},
                outcome.query if outcome else (query.primary if query else ""),
            )
            if topic_id is not None and outcome is not None and outcome.results:
                self._db_call(
                    "search results",
                    "add_search_results",
                    self.session_id,
                    topic_id,
                    outcome.query,
                    outcome.provider,
                    [asdict(r) for r in outcome.results],
                )
        return TurnResult(pred, theme, query, outcome, list(self._warnings))

    def current_theme(self) -> ComposedTheme:
        theme = self.tracker.theme(self.children)
        return compose(theme, self.tax) if not theme.is_empty else EMPTY_THEME

    def history(self) -> list[TextRecord]:
        if self.db is not None:
            try:
                return self.db.session_texts(self.session_id)
            except DatabaseError as exc:
                log.error("history read failed: %s", exc)
        return list(self.local_history)
