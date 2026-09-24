"""Text normalisation for model input.

English-specific notes (see decisions.md, ADR-006):

* Lower-casing uses ``str.casefold`` (locale-independent). The Turkish
  dotted/dotless-I problem from the original brief does not arise in English,
  but NFKC normalisation still folds compatibility characters (ligatures,
  full-width forms) so "ﬁnance" and "finance" share a token.
* URLs, @mentions and HTML are removed; hashtags keep their word
  (``#quantum`` -> ``quantum``) because the word is topical.
* Emoji and other symbols are dropped: they carry sentiment, not topic.
* No stemming/lemmatisation: the sentence encoder uses WordPiece sub-words and
  the TF-IDF baseline gets character n-grams, both of which already share
  morphology (measured in docs/EXPERIMENTS.md).
"""

from __future__ import annotations

import html
import re
import unicodedata

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_MENTION = re.compile(r"(?<!\w)@\w+")
_HASHTAG = re.compile(r"(?<!\w)#(\w+)")
_HTML_TAG = re.compile(r"<[^>]{1,200}>")
_WORD = re.compile(r"[a-z][a-z'\-]*")
_SPACES = re.compile(r"\s+")

MAX_INPUT_CHARS = 5000


def _strip_symbols(text: str) -> str:
    # Keep letters, marks, numbers, punctuation and separators; drop symbols
    # (emoji, pictographs, dingbats) and control characters.
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in {"S", "C"} or ch in "\n\t")


def normalize(text: str, *, lowercase: bool = False) -> str:
    """Clean user or corpus text. Safe on any string (never raises)."""
    if not text:
        return ""
    text = text[:MAX_INPUT_CHARS]
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(r"\1", text)
    text = _strip_symbols(text)
    text = _SPACES.sub(" ", text).strip()
    return text.casefold() if lowercase else text


def content_words(text: str) -> list[str]:
    """Lower-cased alphabetic tokens that are not stop words."""
    return [w for w in _WORD.findall(normalize(text, lowercase=True)) if w not in ENGLISH_STOP_WORDS and len(w) > 1]


def is_informative(text: str) -> bool:
    """False for empty, punctuation-only, numeric-only or stop-word-only input."""
    return bool(content_words(text))
