"""Feature extractors compared in the benchmark (TF-IDF variants and sentence embeddings)."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

TFIDF_MAX_FEATURES = 200_000
CHAR_MAX_FEATURES = 300_000


def tfidf_word(stop_words: str | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, strip_accents="unicode",
        lowercase=True, stop_words=stop_words, max_features=TFIDF_MAX_FEATURES, dtype=float,
    )


def tfidf_word_char() -> FeatureUnion:
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, strip_accents="unicode",
        lowercase=True, max_features=CHAR_MAX_FEATURES, dtype=float,
    )
    return FeatureUnion([("word", tfidf_word()), ("char", char)])


TFIDF_FEATURIZERS = {
    "tfidf-word": lambda: tfidf_word(),
    "tfidf-word-nostop": lambda: tfidf_word("english"),
    "tfidf-word+char": tfidf_word_char,
}
