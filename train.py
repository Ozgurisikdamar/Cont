"""Train the production ContextLens model and write models/contextlens-topic/.

    python train.py                  # uses configs/model.json
    python train.py --no-save-encoder

Requires the processed corpus (python scripts/download_data.py --all).
Hyper-parameters come from configs/model.json, which records the choice made in
the benchmark (docs/MODEL_REPORT.md). Training never looks at the test split;
test metrics are computed afterwards for the metadata only.
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

from contextlens.config import PATHS, RANDOM_SEED, load_settings
from contextlens.data.dataset import DataMissingError, LabelSpace, load_passages
from contextlens.evaluation.metrics import multiclass_report
from contextlens.logging_setup import setup_logging
from contextlens.models.artifact import save_artifact
from contextlens.models.encoders import ENCODERS, SentenceEncoder
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
    cfg = TrainConfig(
        encoder=conf["encoder"],
        general_C=float(conf["general_C"]),
        subtopic_C=float(conf["subtopic_C"]),
        ood_keep_quantile=float(conf["ood_keep_quantile"]),
        min_confidence=float(conf["min_confidence"]),
        vocabulary_size=int(conf["vocabulary_size"]),
    )
    try:
        df = load_passages()
    except DataMissingError as exc:
        log.error("%s", exc)
        return 2
    tax = load_taxonomy()
    space = LabelSpace.from_taxonomy(tax)
    children = {g: tax.children_of(g) for g in tax.general_ids}

    def split(name: str) -> tuple[list[str], object, object]:
        part = df[df.split == name]
        texts = [normalize(t) for t in part.text]
        return texts, space.encode_general(part.general), space.encode_subtopics(part.subtopics)

    train, val, test = split("train"), split("val"), split("test")
    log.info("train=%d val=%d test=%d passages", len(train[0]), len(val[0]), len(test[0]))

    t0 = time.perf_counter()
    encoder = SentenceEncoder(cfg.encoder)
    model, val_report = fit_topic_model(encoder, space, children, train, val, cfg)  # type: ignore[arg-type]
    train_seconds = time.perf_counter() - t0

    gp_test, _, _ = model.predict_proba(test[0])
    test_report = multiclass_report(test[1], gp_test, space.general_ids)  # type: ignore[arg-type]
    log.info("test (report only): acc=%.4f macroF1=%.4f", test_report["accuracy"], test_report["macro_f1"])

    model.metadata = {
        "model_name": conf["model_name"],
        "model_version": conf["model_version"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_version": dataset_fingerprint(),
        "encoder": cfg.encoder,
        "encoder_repo": ENCODERS[cfg.encoder]["repo"],
        "encoder_revision": ENCODERS[cfg.encoder]["revision"],
        "hyperparameters": {"general_C": cfg.general_C, "subtopic_C": cfg.subtopic_C,
                            "ood_keep_quantile": cfg.ood_keep_quantile},
        "preprocessing": "contextlens.preprocessing.text.normalize (NFKC, HTML/URL/mention removal, no lower-casing)",
        "train_seconds": round(train_seconds, 1),
        "val_metrics": {k: val_report["general"][k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")}
        | {"subtopic_macro_f1": val_report["subtopic_macro_f1"]},
        "test_metrics": {k: test_report[k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")},
        "subtopic_threshold_sweep": val_report["subtopic_threshold_sweep"],
    }
    vocabulary = build_vocabulary([t.lower() for t in train[0]], cfg.vocabulary_size)
    save_artifact(model, out_dir, vocabulary, save_encoder=not args.no_save_encoder)
    log.info("artifact written to %s (%.0fs)", out_dir, train_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
