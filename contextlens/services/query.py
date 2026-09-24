"""Build a web-search query from the conversation theme.

The query is the composed theme phrase, optionally refined with up to
``max_keywords`` salient words from the latest message. Only words present in
the public training vocabulary (``vocabulary``: word -> IDF) are eligible, so
names, typos and other personal tokens a user types are never sent to the
search provider. Words are ranked by IDF (rarer = more specific).
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


def salient_keywords(message: str, vocabulary: dict[str, float], exclude: str, limit: int) -> list[str]:
    excluded = set(content_words(exclude))
    seen: set[str] = set()
    candidates = []
    for word in content_words(message):
        if word in excluded or word in seen or len(word) < MIN_KEYWORD_LENGTH or word not in vocabulary:
            continue
        seen.add(word)
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
