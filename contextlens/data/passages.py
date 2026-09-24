"""Split Wikipedia article text into short, chat-sized passages.

Users type one or two sentences, so the classifier is trained on passages of
roughly 12-60 words rather than whole articles. Passages come from the lead
section first (the most on-topic part of an article) and then from the
following sections, stopping at back-matter sections.
"""

from __future__ import annotations

import re

STOP_SECTIONS = {
    "see also",
    "references",
    "external links",
    "further reading",
    "notes",
    "bibliography",
    "sources",
    "citations",
    "footnotes",
    "works cited",
    "selected works",
    "works",
    "publications",
    "discography",
    "filmography",
    "gallery",
}
_TERMINAL = tuple('.!?:;"”)]')
_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "cf.",
    "ca.",
    "c.",
    "approx.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "St.",
    "Jr.",
    "Sr.",
    "No.",
    "Nos.",
    "Fig.",
    "Vol.",
    "U.S.",
    "U.K.",
    "U.N.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Gen.",
    "Col.",
    "Lt.",
    "Capt.",
    "Sgt.",
    "Gov.",
    "Rev.",
    "Mt.",
    "al.",
    "op.",
    "ed.",
    "eds.",
)
_ABBR_TOKEN = "⁣"  # noqa: S105 - not a secret: invisible separator protecting abbreviation dots
_EMPTY_PARENS = re.compile(r"\(\s*[;,:\s]*\s*\)")
_LEADING_PUNCT_IN_PARENS = re.compile(r"\(\s*[;,]\s*")
_SPACES = re.compile(r"[ \t ]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[]?[A-Z0-9])")
_INITIALS = re.compile(r"\b([A-Z])\.(?=\s?[A-Z]\.?)")


def clean_text(text: str) -> str:
    """Remove dump artefacts (empty pronunciation parentheses, odd spacing)."""
    text = _EMPTY_PARENS.sub("", text)
    text = _LEADING_PUNCT_IN_PARENS.sub("(", text)
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text.strip()


def is_heading(line: str) -> bool:
    words = line.split()
    return 0 < len(words) <= 8 and not line.rstrip().endswith(_TERMINAL)


def article_paragraphs(text: str) -> list[str]:
    """Content paragraphs in reading order, stopping at back-matter sections."""
    paragraphs: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if is_heading(line):
            if line.lower() in STOP_SECTIONS:
                break
            continue
        if len(line.split()) < 6:  # captions, list fragments
            continue
        paragraphs.append(clean_text(line))
    return paragraphs


def split_sentences(paragraph: str) -> list[str]:
    protected = paragraph
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _ABBR_TOKEN))
    protected = _INITIALS.sub(lambda m: m.group(1) + _ABBR_TOKEN, protected)
    parts = _SENT_SPLIT.split(protected)
    return [p.replace(_ABBR_TOKEN, ".").strip() for p in parts if p.strip()]


def make_passages(text: str, *, min_words: int, max_words: int, max_passages: int) -> list[str]:
    """Group consecutive sentences into passages of ``min_words``..``max_words``."""
    passages: list[str] = []
    for paragraph in article_paragraphs(text):
        current: list[str] = []
        count = 0
        for sentence in split_sentences(paragraph):
            words = sentence.split()
            if len(words) > max_words:
                words = words[:max_words]
                sentence = " ".join(words)
            if current and count + len(words) > max_words:
                if count >= min_words:
                    passages.append(" ".join(current))
                current, count = [], 0
            current.append(sentence)
            count += len(words)
            if count >= min_words:
                passages.append(" ".join(current))
                current, count = [], 0
            if len(passages) >= max_passages:
                return passages
        # A short paragraph tail is dropped rather than glued onto the next
        # paragraph (different paragraphs often change subject).
    return passages[:max_passages]
