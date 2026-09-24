"""Build a web-search query from the conversation theme.

The query is the composed theme phrase, optionally refined with up to
``max_keywords`` salient words from the latest message. Only words present in
the public training vocabulary (``vocabulary``: word -> keyword weight = IDF x
topic concentration, see contextlens.models.training.build_vocabulary) are
eligible, so names, typos and other personal tokens a user types are never
sent to the search provider. Words are ranked by weight.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextlens.preprocessing.text import content_words

DEFAULT_MAX_KEYWORDS = 2
MIN_KEYWORD_LENGTH = 4
MAX_QUERIES = 4  # bounded number of provider round-trips per turn


@dataclass(frozen=True)
class SearchQuery:
    primary: str  # theme phrase + keywords (tried first)
    fallback: str  # theme phrase only
    concepts: tuple[str, ...] = ()  # single concepts behind the theme (last resort)

    def candidates(self) -> list[str]:
        ordered = [self.primary, self.fallback, *self.concepts]
        return [q for q in dict.fromkeys(ordered) if q][:MAX_QUERIES]


def _stem(word: str) -> str:
    """Crude plural folding so "novel" does not repeat "novels" in a query."""
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def salient_keywords(message: str, vocabulary: dict[str, float], exclude: str, limit: int) -> list[str]:
    excluded = {_stem(w) for w in content_words(exclude)}
    seen: set[str] = set()
    candidates = []
    for word in content_words(message):
        stem = _stem(word)
        if stem in excluded or stem in seen or len(word) < MIN_KEYWORD_LENGTH or word not in vocabulary:
            continue
        seen.add(stem)
        candidates.append(word)
    candidates.sort(key=lambda w: (-vocabulary[w], w))
    return candidates[:limit]


def build_query(
    theme_phrase: str,
    latest_message: str,
    vocabulary: dict[str, float] | None,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    concepts: tuple[str, ...] = (),
) -> SearchQuery:
    phrase = " ".join(theme_phrase.split())
    if not phrase:
        return SearchQuery("", "")
    keywords = salient_keywords(latest_message, vocabulary or {}, phrase, max_keywords) if max_keywords > 0 else []
    primary = f"{phrase} {' '.join(keywords)}".strip()
    return SearchQuery(primary=primary, fallback=phrase, concepts=tuple(" ".join(c.split()) for c in concepts))
