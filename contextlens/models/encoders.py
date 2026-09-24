"""Sentence-embedding encoders (local inference, CPU/GPU agnostic)."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contextlens.config import REPO_ROOT

log = logging.getLogger(__name__)

# Candidate encoders evaluated in docs/EXPERIMENTS.md. Revisions are pinned so
# results are reproducible; prefixes follow each model card.
ENCODERS: dict[str, dict[str, str]] = {
    "minilm-l6": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "prefix": "",
    },
    "bge-small": {
        "repo": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "prefix": "",
    },
    "e5-small": {
        "repo": "intfloat/e5-small-v2",
        "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
        "prefix": "query: ",
    },
    "mpnet-base": {
        "repo": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "prefix": "",
    },
    # all-MiniLM-L6-v2 fine-tuned on the training split with a general + subtopic
    # head (scripts/finetune_transformer.py --export). Weights are local, not on the Hub.
    "minilm-l6-ft": {
        "repo": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "prefix": "",
        "local": "models/finetuned/minilm-l6",
    },
}


def available_encoders() -> list[str]:
    """Registry keys that can be loaded now (a local fine-tuned encoder must have been exported)."""
    return [k for k, spec in ENCODERS.items() if "local" not in spec or (REPO_ROOT / spec["local"]).exists()]


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
        found = self._resolve_local(spec)
        if found is not None:
            self.model = SentenceTransformer(str(found), device=best_device())
            self.source = str(found)
            self.cache_tag = f"{self.key}@{_weights_digest(found)}"
        elif "local" in spec:
            # A fine-tuned encoder has no Hub fallback: silently loading the base
            # model would give embeddings the heads were never trained on.
            raise FileNotFoundError(
                f"fine-tuned encoder {self.key!r} not found (looked in: "
                f"{', '.join(str(p) for p in self._candidates(spec))}); create it with "
                f"python scripts/finetune_transformer.py --encoder minilm-l6 --export {spec['local']}"
            )
        else:  # pinned revision from the Hugging Face Hub (cached after the first download)
            self.model = SentenceTransformer(spec["repo"], device=best_device(), revision=spec["revision"])
            self.source = f"{spec['repo']}@{spec['revision']}"
            self.cache_tag = f"{self.key}@{spec['revision'][:8]}"
        self.model.max_seq_length = self.max_seq_length
        self.dim = int(self.model.get_sentence_embedding_dimension() or 0)

    def _candidates(self, spec: dict[str, str]) -> list[Path]:
        paths = [self.local_path] if self.local_path is not None else []
        if "local" in spec:
            paths.append(REPO_ROOT / spec["local"])
        return paths

    def _resolve_local(self, spec: dict[str, str]) -> Path | None:
        """First candidate directory that holds a saved sentence-transformers model."""
        for path in self._candidates(spec):
            if (path / "modules.json").is_file():
                return path
        return None

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
        """Save the encoder; weights that are exactly representable in float16 are stored as float16.

        The fine-tuned encoder is exported in float16 (halving the file), so its
        float32 copy in memory round-trips losslessly; a Hub encoder in float32 is
        saved unchanged.
        """
        import torch

        params = list(self.model.parameters())
        half_exact = all(torch.equal(p, p.detach().half().float()) for p in params if p.is_floating_point())
        if half_exact:
            self.model.half()
        try:
            self.model.save(str(path))
        finally:
            if half_exact:
                self.model.float()


def _weights_digest(directory: Path) -> str:
    """Short content hash of a saved model's weight files (identifies a local export)."""
    h = hashlib.sha256()
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix in {".safetensors", ".bin"}:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:8]


def cached_encode(encoder: SentenceEncoder, texts: list[str], cache_dir: Path, name: str) -> np.ndarray:
    """Encode ``texts`` once and reuse the result.

    The key is the encoder's identity (Hub revision, or a hash of the local
    weights - so a re-exported fine-tuned encoder never reuses stale vectors)
    plus a hash of the texts.
    """
    digest = hashlib.sha1("\n".join(texts).encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    path = cache_dir / encoder.cache_tag / f"{name}-{digest}.npy"
    if path.exists():
        return np.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    emb = encoder.encode(texts, show_progress=False)
    log.info("encoded %d texts (%s, %s) in %.0fs", len(texts), encoder.key, name, time.perf_counter() - t0)
    np.save(path, emb)
    return emb


@dataclass
class CachingEncoder:
    """Development-script wrapper: ``encode`` goes through :func:`cached_encode`.

    Lets experiment scripts reuse the production training code
    (``fit_topic_model``) without re-encoding the corpus on every run. Never
    saved into an artifact.
    """

    inner: Any  # a SentenceEncoder (anything with encode(); cached only when it has a cache_tag)
    cache_dir: Path

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts or getattr(self.inner, "cache_tag", None) is None:  # e.g. the test suite's fake encoder
            return self.inner.encode(texts)
        return cached_encode(self.inner, texts, self.cache_dir, "texts")
