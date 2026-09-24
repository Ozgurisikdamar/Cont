"""English-language gate (decisions.md D-30).

The model only knows English. A text is answered *uncertain* when it is
confidently another language. The decision combines two local, deterministic
signals:

* **fastText lid.176** (Joulin et al., 176 languages, 0.9 MB, CC BY-SA 3.0):
  its top-1 language and probability;
* an **English lexicon**: every word type of the Wikipedia training split.

A text is *not English* only when fastText's top language is not English with
probability >= ``reject_confidence[bucket]`` **and** at least one of its words is
outside the lexicon. The confidence depends on the number of words (1, 2, 3,
4+): one or two words are often genuinely ambiguous ("Tom", "La", "Die"), and
rare English technical terms ("titration") are exactly what a character-n-gram
model gets wrong and a lexicon gets right. Thresholds are chosen on the dev
split (scripts/language_gate_experiment.py) and stored in the artifact.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from contextlens.config import PATHS
from contextlens.preprocessing.text import WORD_RE

log = logging.getLogger(__name__)

FASTTEXT_MODEL = PATHS.data_raw / "langid" / "lid.176.ftz"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
BUCKETS = ("1", "2", "3", "4+")


def load_fasttext(path: Path) -> Any:
    """Load a fastText model without its stderr deprecation notice."""
    import fasttext

    fasttext.FastText.eprint = lambda *args, **kwargs: None
    return fasttext.load_model(str(path))


def build_lexicon(train_texts: Iterable[str]) -> frozenset[str]:
    """Every lower-cased word type of the training texts."""
    return frozenset(w for t in train_texts for w in WORD_RE.findall(str(t).lower()))


def ensure_fasttext_model(path: Path = FASTTEXT_MODEL) -> Path:
    """Download lid.176.ftz once (0.9 MB) if it is not cached."""
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", FASTTEXT_URL)
        urllib.request.urlretrieve(FASTTEXT_URL, path)  # noqa: S310 - constant https URL
    return path


def word_bucket(n_words: int) -> str:
    return str(n_words) if n_words <= 3 else "4+"


@dataclass
class LanguageGate:
    model: Any  # fastText model (predict(text, k) -> (labels, probs))
    lexicon: frozenset[str]
    reject_confidence: dict[str, float] = field(default_factory=lambda: dict.fromkeys(BUCKETS, 0.7))
    source: Path = FASTTEXT_MODEL  # the .ftz file, copied into the artifact

    def detect(self, text: str) -> tuple[str, float]:
        labels, probs = self.model.predict(text.replace("\n", " "), k=1)
        return labels[0].replace("__label__", ""), float(probs[0])

    def is_english(self, text: str) -> bool:
        words = WORD_RE.findall(text.lower())
        if not words or all(w in self.lexicon for w in words):
            return True
        lang, prob = self.detect(text)
        return not (lang != "en" and prob >= self.reject_confidence.get(word_bucket(len(words)), 1.1))


def write_lexicon(words: Iterable[str], path: Path) -> None:
    path.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")


def read_lexicon(path: Path) -> frozenset[str]:
    return frozenset(w for w in path.read_text(encoding="utf-8").split("\n") if w)
