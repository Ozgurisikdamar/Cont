"""Save and load the model artifact safely.

Layout of ``models/contextlens-topic/``::

    metadata.json      model name/version, training date, dataset fingerprint,
                       labels, hyper-parameters, thresholds, metrics, SHA-256
                       of every other file
    heads.skops        general head + subtopic heads (skops, not pickle)
    centroids.npy      class centroids for the OOD gate (allow_pickle=False)
    vocabulary.json    word -> IDF, used to pick query keywords
    encoder/           optional local copy of the sentence encoder

Loading refuses to continue when a file is missing, a checksum does not match
or the skops file contains types outside an explicit allow-list - a corrupt or
tampered artifact never runs silently.
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
from contextlens.models.heads import HierarchicalSubtopics, MultiLabelHead
from contextlens.models.topic_model import TopicModel

log = logging.getLogger(__name__)

TRUSTED_TYPES = {
    "contextlens.models.heads.HierarchicalSubtopics",
    "contextlens.models.heads.MultiLabelHead",
    "sklearn.linear_model._logistic.LogisticRegression",
    "numpy.dtype",
    "builtins.dict",
    "builtins.list",
    "builtins.str",
}
CHECKSUMMED_FILES = ("heads.skops", "centroids.npy", "vocabulary.json")


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


def save_artifact(model: TopicModel, directory: Path, vocabulary: dict[str, float], save_encoder: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sio.dump({"general_head": model.general_head, "subtopic_heads": model.subtopic_heads}, directory / "heads.skops")
    np.save(directory / "centroids.npy", model.centroids.astype(np.float32), allow_pickle=False)
    (directory / "vocabulary.json").write_text(json.dumps(vocabulary, sort_keys=True), encoding="utf-8")
    if save_encoder:
        model.encoder.save(directory / "encoder")
    meta = dict(model.metadata)
    meta.update(
        {
            "general_ids": model.general_ids,
            "subtopic_ids": model.subtopic_ids,
            "parent_col": model.parent_col.tolist(),
            "temperature": model.temperature,
            "subtopic_threshold": model.subtopic_threshold,
            "ood_threshold": model.ood_threshold,
            "min_confidence": model.min_confidence,
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
    if not isinstance(heads, dict) or not {"general_head", "subtopic_heads"} <= set(heads):
        raise ArtifactError(f"{path} does not contain the expected heads")
    if not isinstance(heads["subtopic_heads"], HierarchicalSubtopics) or not all(
        isinstance(h, MultiLabelHead) for h in heads["subtopic_heads"].heads.values()
    ):
        raise ArtifactError(f"{path} has unexpected subtopic head types")
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
        key = meta["encoder"]
        if key not in ENCODERS:
            raise ArtifactError(f"unknown encoder {key!r} in metadata")
        encoder = SentenceEncoder(key, local_path=directory / "encoder")
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
    )


def load_vocabulary(directory: Path) -> dict[str, float]:
    path = directory / "vocabulary.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("vocabulary unavailable (%s); queries will not be refined with keywords", exc)
        return {}
    return {str(k): float(v) for k, v in data.items()}
