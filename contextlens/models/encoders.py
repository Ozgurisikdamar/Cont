"""Sentence-embedding encoders (local inference, CPU/GPU agnostic)."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Candidate encoders evaluated in docs/EXPERIMENTS.md. Revisions are pinned so
# results are reproducible; prefixes follow each model card.
ENCODERS: dict[str, dict[str, str]] = {
    "minilm-l6": {"repo": "sentence-transformers/all-MiniLM-L6-v2",
                  "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", "prefix": ""},
    "bge-small": {"repo": "BAAI/bge-small-en-v1.5",
                  "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "prefix": ""},
    "e5-small": {"repo": "intfloat/e5-small-v2",
                 "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52", "prefix": "query: "},
    "mpnet-base": {"repo": "sentence-transformers/all-mpnet-base-v2",
                   "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130", "prefix": ""},
}


def best_device() -> str:
    """'cuda' when a GPU is available, otherwise 'cpu' (the system never requires a GPU)."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"


@dataclass
class SentenceEncoder:
    """Thin wrapper around sentence-transformers with an optional on-disk cache."""

    key: str
    local_path: Path | None = None
    batch_size: int = 64
    max_seq_length: int = 128

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        spec = ENCODERS[self.key]
        self.prefix = spec["prefix"]
        source = str(self.local_path) if self.local_path and self.local_path.exists() else spec["repo"]
        kwargs = {} if source != spec["repo"] else {"revision": spec["revision"]}
        self.model = SentenceTransformer(source, device=best_device(), **kwargs)
        self.model.max_seq_length = self.max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self.model.encode(
            [self.prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

    def save(self, path: Path) -> None:
        self.model.save(str(path))


def cached_encode(encoder: SentenceEncoder, texts: list[str], cache_dir: Path, name: str) -> np.ndarray:
    """Encode ``texts`` once and reuse the result (keyed by encoder + content hash)."""
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / encoder.key / f"{name}-{digest}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    emb = encoder.encode(texts, show_progress=False)
    np.save(path, emb)
    return emb
