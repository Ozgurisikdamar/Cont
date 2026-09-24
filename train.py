"""Train the production ContextLens model and write models/contextlens-topic/.

    python train.py                  # uses configs/model.json
    python train.py --no-save-encoder

Requires the processed corpus (python scripts/download_data.py --all).
Hyper-parameters come from configs/model.json, which records the choice made in
the benchmark (docs/MODEL_REPORT.md). Training and tuning use development data
only; the test splits are evaluated once after the freeze (evaluate.py --stage locked).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contextlens.config import PATHS, RANDOM_SEED, load_settings
from contextlens.data.dataset import DataMissingError, LabelSpace, load_jsonl, load_passages, load_se_general
from contextlens.logging_setup import setup_logging
from contextlens.models.artifact import save_artifact
from contextlens.models.encoders import ENCODERS, CachingEncoder, SentenceEncoder
from contextlens.models.language import (
    FASTTEXT_MODEL,
    LanguageGate,
    build_lexicon,
    ensure_fasttext_model,
    load_fasttext,
)
from contextlens.models.ood import fit_detector
from contextlens.models.topic_model import TopicModel
from contextlens.models.training import TrainConfig, build_vocabulary, fit_topic_model
from contextlens.preprocessing.text import normalize
from contextlens.reproducibility import seed_everything
from contextlens.taxonomy import load_taxonomy

log = logging.getLogger("train")
MODEL_CONFIG = PATHS.root / "configs" / "model.json"


def dataset_fingerprint() -> str:
    h = hashlib.sha256()
    for path in (PATHS.data_processed / "passages.parquet", PATHS.data_manifest / "articles.csv", PATHS.taxonomy):
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def train_config(conf: dict) -> TrainConfig:
    return TrainConfig(
        encoder=conf["encoder"],
        general_C=float(conf["general_C"]),
        subtopic_C=float(conf["subtopic_C"]),
        ood_keep_quantile=float(conf["ood_keep_quantile"]),
        min_confidence=0.0 if conf["min_confidence"] == "auto" else float(conf["min_confidence"]),
        vocabulary_size=int(conf["vocabulary_size"]),
        head_type=conf.get("head_type", "hierarchical"),
    )


def offtopic_training_texts() -> list[str]:
    """CLINC150 train utterances (scripts/build_ood_conversational.py); used only by
    detectors that learn from off-topic examples."""
    df = load_jsonl(PATHS.root / "data" / "external" / "ood_conversational.jsonl")
    return [normalize(t) for t in df[df.split == "train"].text]


def fit_from_config(conf: dict, encoder: Any) -> tuple[TopicModel, dict]:
    """Fit the full production model (heads, OOD gate, confidence, language gate) from ``conf``.

    Development data only: Wikipedia ``train`` / ``val`` and Stack Exchange
    ``ext_dev``. The test splits are evaluated once after the freeze
    (evaluate.py --stage locked). ``encoder`` may be a :class:`CachingEncoder` in
    development scripts; the caller puts the real encoder on the model.
    """
    cfg = train_config(conf)
    df = load_passages()
    tax = load_taxonomy()
    space = LabelSpace.from_taxonomy(tax)
    children = {g: tax.children_of(g) for g in tax.general_ids}

    def split(name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        part = df[df.split == name]
        texts = [normalize(t) for t in part.text]
        return texts, space.encode_general(part.general), space.encode_subtopics(part.subtopics)

    train, val = split("train"), split("val")
    log.info("train=%d val=%d passages", len(train[0]), len(val[0]))

    se = load_se_general()
    se_dev = se[(se.split == "ext_dev") & ~se.is_ood]
    ood_conf = conf.get("ood", {"method": "centroid", "calibration": conf.get("ood_calibration", "wiki_val")})
    if ood_conf["calibration"] == "se_ext_dev":  # real user questions, in-distribution only (never ext_test)
        calibration_texts = [normalize(t) for t in se_dev.text]
    elif ood_conf["calibration"] == "wiki_val":
        calibration_texts = val[0]
    else:
        raise ValueError(f"unknown OOD calibration {ood_conf['calibration']!r} (expected wiki_val or se_ext_dev)")
    t0 = time.perf_counter()
    model, val_report = fit_topic_model(
        encoder, space, children, train, val, cfg, ood_calibration_texts=calibration_texts
    )
    method = ood_conf["method"]
    if method != "centroid":
        # decisions.md D-34: the detector chosen by scripts/ood_experiment.py on dev data
        Xtr = encoder.encode(train[0])
        offtopic = encoder.encode(offtopic_training_texts()) if method in ("binary", "other_class") else None
        model.ood_detector = fit_detector(method, Xtr, train[1], len(space.general_ids), offtopic)
        Xcal = encoder.encode(calibration_texts)
        gp_cal, _, logits_cal = model.head_outputs(Xcal)
        model.ood_threshold = float(np.quantile(model.ood_scores(Xcal, gp_cal, logits_cal), cfg.ood_keep_quantile))
    log.info("OOD detector %s, threshold %.4f", method, model.ood_threshold)
    train_seconds = time.perf_counter() - t0

    min_conf_sweep = None
    if conf["min_confidence"] == "auto":
        # Largest threshold that still answers >= min_coverage of real in-domain
        # questions (Stack Exchange ext_dev; never ext_test).
        gp_dev, _, _ = model.predict_proba([normalize(t) for t in se_dev.text])
        conf_dev, correct = gp_dev.max(axis=1), gp_dev.argmax(axis=1) == space.encode_general(se_dev.general)
        min_conf_sweep = [
            {
                "min_confidence": t,
                "coverage": round(float(np.mean(conf_dev >= t)), 4),
                "accuracy_answered": round(float(np.mean(correct[conf_dev >= t])), 4),
            }
            for t in conf["min_confidence_grid"]
        ]
        ok = [r["min_confidence"] for r in min_conf_sweep if r["coverage"] >= conf["min_confidence_min_coverage"]]
        model.min_confidence = float(max(ok)) if ok else float(min(conf["min_confidence_grid"]))
        log.info("min_confidence=%.2f (sweep on ext_dev: %s)", model.min_confidence, min_conf_sweep)

    # Language gate (decisions.md D-30): thresholds chosen on dev data by
    # scripts/language_gate_experiment.py and recorded in configs/model.json.
    ft_path = ensure_fasttext_model(FASTTEXT_MODEL)
    model.language_gate = LanguageGate(
        load_fasttext(ft_path),
        build_lexicon(train[0]),
        {k: float(v) for k, v in conf["language_gate"]["reject_confidence"].items()},
        source=ft_path,
    )
    info = {
        "cfg": cfg,
        "val_report": val_report,
        "min_conf_sweep": min_conf_sweep,
        "ood": ood_conf,
        "train_seconds": train_seconds,
        "train": train,
    }
    return model, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG)
    parser.add_argument("--out", type=Path, default=None, help="artifact directory (default: settings.model_dir)")
    parser.add_argument("--no-save-encoder", action="store_true", help="do not copy encoder weights into the artifact")
    args = parser.parse_args(argv)
    setup_logging("INFO", PATHS.root / "logs" / "train.log")
    seed_everything(RANDOM_SEED)
    out_dir = args.out or load_settings().model_dir
    conf = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        load_passages()
    except DataMissingError as exc:
        log.error("%s", exc)
        return 2
    # embeddings are cached on disk, keyed by the encoder weights and the texts
    encoder = CachingEncoder(SentenceEncoder(conf["encoder"]), PATHS.data_processed / "embeddings")
    try:
        model, info = fit_from_config(conf, encoder)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    model.encoder = encoder.inner
    cfg, val_report, train = info["cfg"], info["val_report"], info["train"]
    model.metadata = {
        "model_name": conf["model_name"],
        "model_version": conf["model_version"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset_fingerprint(),
        "encoder": cfg.encoder,
        "encoder_repo": ENCODERS[cfg.encoder]["repo"],
        "encoder_revision": ENCODERS[cfg.encoder]["revision"],
        "hyperparameters": {
            "head_type": cfg.head_type,
            "general_C": cfg.general_C,
            "subtopic_C": cfg.subtopic_C,
            "ood_keep_quantile": cfg.ood_keep_quantile,
            "ood": info["ood"],
            "language_gate": conf["language_gate"],
        },
        "preprocessing": "contextlens.preprocessing.text.normalize (NFKC, HTML/URL/mention removal, no lower-casing)",
        "train_seconds": round(info["train_seconds"], 1),
        "val_metrics": {k: val_report["general"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")}
        | {"subtopic_macro_f1": val_report["subtopic_macro_f1"]},
        "subtopic_threshold_sweep": val_report["subtopic_threshold_sweep"],
        "min_confidence_sweep_ext_dev": info["min_conf_sweep"],
    }
    vocabulary = build_vocabulary([t.lower() for t in train[0]], train[1], cfg.vocabulary_size)
    save_artifact(model, out_dir, vocabulary, save_encoder=not args.no_save_encoder)
    log.info("artifact written to %s (%.0fs)", out_dir, info["train_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
