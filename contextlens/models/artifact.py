"""Save and load the model artifact safely.

Layout of ``models/contextlens-topic/``::

    metadata.json      model name/version, training date, dataset fingerprint,
                       labels, hyper-parameters, thresholds, metrics, SHA-256
                       of every other file
    heads.skops        general head + subtopic heads (skops, not pickle)
    centroids.npy      class centroids for the OOD gate (allow_pickle=False)
    vocabulary.json    word -> IDF, used to pick query keywords
    encoder/           local copy of the sentence encoder (safetensors); every
                       file is listed in metadata.json -> encoder_manifest
                       (relative path, size, SHA-256)

Loading refuses to continue when a file is missing, a checksum does not match,
an encoder file is missing, changed or unexpected,
the skops file contains types outside an explicit allow-list, or the encoder
does not reproduce the fingerprint recorded at training time (the embedding of
a fixed probe sentence) - a corrupt, tampered or mismatched artifact never runs
silently.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import skops.io as sio

from contextlens.models.encoders import ENCODERS, SentenceEncoder
from contextlens.models.heads import FlatSubtopicSoftmax, HierarchicalSubtopics, MultiLabelHead
from contextlens.models.topic_model import TopicModel

log = logging.getLogger(__name__)

TRUSTED_TYPES = {
    "contextlens.models.heads.FlatSubtopicSoftmax",
    "contextlens.models.heads.HierarchicalSubtopics",
    "contextlens.models.heads.MultiLabelHead",
    "sklearn.linear_model._logistic.LogisticRegression",
    "numpy.dtype",
    "builtins.dict",
    "builtins.list",
    "builtins.str",
}
CHECKSUMMED_FILES = ("heads.skops", "centroids.npy", "vocabulary.json")

# Encoder fingerprint: the heads only make sense on the embeddings they were
# trained on. Loading a different encoder (e.g. the Hub base model instead of the
# fine-tuned one) raises no error by itself, so the artifact stores the embedding
# of this sentence and load_artifact checks that the loaded encoder reproduces it.
PROBE_TEXT = "Entangled qubits, the Ottoman Empire, a novel about football and the periodic table."
PROBE_MIN_COSINE = 0.999


class ArtifactError(RuntimeError):
    """Raised when the model artifact is missing, corrupt or untrusted."""


HOW_TO_BUILD = (
    "Build it with:\n"
    "    python scripts/download_data.py --all   # data (once)\n"
    "    python train.py                          # trains and writes models/contextlens-topic/"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoder_manifest(directory: Path) -> list[dict[str, Any]]:
    """Relative path, size and SHA-256 of every file below ``directory``, sorted by path."""
    return [
        {"path": p.relative_to(directory).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    ]


def verify_encoder_files(directory: Path, manifest: Any) -> None:
    """Raise ArtifactError unless the encoder directory matches ``manifest`` exactly.

    The embedding fingerprint (verify_encoder) proves the encoder is *compatible*;
    this check proves the files are the ones written at training time - including
    files the fingerprint cannot see (e.g. tokenizer settings used only for rare
    inputs, or an unexpected extra module)."""
    if not isinstance(manifest, list) or not manifest:
        raise ArtifactError(f"the artifact has no encoder manifest; retrain it.\n{HOW_TO_BUILD}")
    expected = {str(e["path"]): e for e in manifest}
    present = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*") if p.is_file()}
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    if missing:
        raise ArtifactError(f"encoder files missing: {missing}\n{HOW_TO_BUILD}")
    if extra:
        raise ArtifactError(f"unexpected files in {directory}: {extra}; the encoder was modified")
    for rel, entry in expected.items():
        path = present[rel]
        if path.stat().st_size != int(entry["size"]) or sha256_file(path) != entry["sha256"]:
            raise ArtifactError(f"checksum mismatch for {path}; the encoder is corrupt or was modified")


def encoder_probe(encoder: Any) -> list[float]:
    """Embedding of PROBE_TEXT, rounded for storage in metadata.json."""
    return [round(float(x), 6) for x in np.asarray(encoder.encode([PROBE_TEXT]))[0]]


def verify_encoder(encoder: Any, meta: dict[str, Any], centroids: np.ndarray) -> None:
    """Raise ArtifactError unless ``encoder`` is the one the heads were trained with."""
    expected = meta.get("encoder_probe")
    if not expected:
        raise ArtifactError(f"the artifact has no encoder fingerprint; retrain it.\n{HOW_TO_BUILD}")
    ref = np.asarray(expected, dtype=np.float64)
    got = np.asarray(encoder.encode([PROBE_TEXT]), dtype=np.float64)[0]
    if got.shape != ref.shape or ref.shape[0] != centroids.shape[1]:
        raise ArtifactError(
            f"encoder dimension {got.shape[0]} does not match the artifact ({ref.shape[0]}); "
            "the encoder is not the one the model was trained with"
        )
    cosine = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-12))
    if cosine < PROBE_MIN_COSINE:
        raise ArtifactError(
            f"the loaded encoder ({getattr(encoder, 'source', meta.get('encoder'))}) does not reproduce the "
            f"embedding recorded at training time (cosine {cosine:.4f} < {PROBE_MIN_COSINE}); "
            f"the classifier heads would receive embeddings they were never trained on.\n{HOW_TO_BUILD}"
        )


def save_artifact(model: TopicModel, directory: Path, vocabulary: dict[str, float], save_encoder: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    heads: dict[str, Any] = {"head_type": model.head_type, "general_head": model.general_head}
    if model.subtopic_heads is not None:
        heads["subtopic_heads"] = model.subtopic_heads
    sio.dump(heads, directory / "heads.skops")
    np.save(directory / "centroids.npy", model.centroids.astype(np.float32), allow_pickle=False)
    (directory / "vocabulary.json").write_text(json.dumps(vocabulary, sort_keys=True), encoding="utf-8")
    if save_encoder:
        model.encoder.save(directory / "encoder")
    meta = dict(model.metadata)
    if (directory / "encoder").is_dir():
        meta["encoder_manifest"] = encoder_manifest(directory / "encoder")
    meta.update(
        {
            "general_ids": model.general_ids,
            "subtopic_ids": model.subtopic_ids,
            "parent_col": model.parent_col.tolist(),
            "head_type": model.head_type,
            "temperature": model.temperature,
            "subtopic_threshold": model.subtopic_threshold,
            "ood_threshold": model.ood_threshold,
            "min_confidence": model.min_confidence,
            "min_known_word_share": model.min_known_word_share,
            "encoder_probe": encoder_probe(model.encoder),
            "checksums": {name: sha256_file(directory / name) for name in CHECKSUMMED_FILES},
        }
    )
    (directory / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _load_heads(path: Path) -> dict[str, Any]:
    untrusted = set(sio.get_untrusted_types(file=path))
    unexpected = untrusted - TRUSTED_TYPES
    if unexpected:
        raise ArtifactError(f"{path} contains untrusted types: {sorted(unexpected)}")
    heads = sio.load(path, trusted=sorted(untrusted))
    if not isinstance(heads, dict) or "general_head" not in heads:
        raise ArtifactError(f"{path} does not contain the expected heads")
    head_type = heads.setdefault("head_type", "hierarchical")
    if head_type == "flat_softmax":
        if not isinstance(heads["general_head"], FlatSubtopicSoftmax):
            raise ArtifactError(f"{path} has an unexpected head type for flat_softmax")
        heads["subtopic_heads"] = None
    elif head_type == "hierarchical":
        subs = heads.get("subtopic_heads")
        if not isinstance(subs, HierarchicalSubtopics) or not all(
            isinstance(h, MultiLabelHead) for h in subs.heads.values()
        ):
            raise ArtifactError(f"{path} has unexpected subtopic head types")
    else:
        raise ArtifactError(f"{path} has an unknown head_type {head_type!r}")
    return heads


def load_artifact(
    directory: Path, min_confidence: float | None = None, encoder: Any | None = None, max_subtopics: int = 3
) -> TopicModel:
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        raise ArtifactError(f"model artifact not found at {directory}.\n{HOW_TO_BUILD}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read {meta_path}: {exc}") from exc
    for name in CHECKSUMMED_FILES:
        path = directory / name
        if not path.exists():
            raise ArtifactError(f"artifact file missing: {path}\n{HOW_TO_BUILD}")
        if sha256_file(path) != meta.get("checksums", {}).get(name):
            raise ArtifactError(f"checksum mismatch for {path}; the artifact is corrupt or was modified")
    heads = _load_heads(directory / "heads.skops")
    centroids = np.load(directory / "centroids.npy", allow_pickle=False)
    if encoder is None:
        key = meta.get("encoder")
        if key not in ENCODERS:
            raise ArtifactError(f"unknown encoder {key!r} in metadata")
        encoder_dir = directory / "encoder"
        if encoder_dir.is_dir():
            verify_encoder_files(encoder_dir, meta.get("encoder_manifest"))
        try:
            encoder = SentenceEncoder(key, local_path=directory / "encoder")
        except (OSError, ValueError) as exc:  # FileNotFoundError is an OSError
            raise ArtifactError(f"cannot load the sentence encoder {key!r}: {exc}\n{HOW_TO_BUILD}") from exc
    verify_encoder(encoder, meta, centroids)
    return TopicModel(
        general_ids=list(meta["general_ids"]),
        subtopic_ids=list(meta["subtopic_ids"]),
        parent_col=np.asarray(meta["parent_col"], dtype=int),
        general_head=heads["general_head"],
        temperature=float(meta["temperature"]),
        subtopic_heads=heads["subtopic_heads"],
        subtopic_threshold=float(meta["subtopic_threshold"]),
        centroids=centroids,
        ood_threshold=float(meta["ood_threshold"]),
        min_confidence=float(min_confidence if min_confidence is not None else meta["min_confidence"]),
        encoder=encoder,
        metadata=meta,
        max_subtopics=max_subtopics,
        head_type=heads["head_type"],
        known_words=known_words(directory),
        min_known_word_share=float(meta.get("min_known_word_share", 0.4)),
    )


def known_words(directory: Path) -> frozenset[str]:
    """Training vocabulary + English stop words (the language gate's word list)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(load_vocabulary(directory)) | frozenset(ENGLISH_STOP_WORDS)


def load_vocabulary(directory: Path) -> dict[str, float]:
    path = directory / "vocabulary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("vocabulary unavailable (%s); queries will not be refined with keywords", exc)
        return {}
    return {str(k): float(v) for k, v in data.items()}
