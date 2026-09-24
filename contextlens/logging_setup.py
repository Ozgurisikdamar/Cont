"""Logging configuration shared by all entry points."""

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger once; noisy third-party loggers are capped at WARNING."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT, handlers=handlers, force=True
    )
    for noisy in ("urllib3", "filelock", "huggingface_hub", "sentence_transformers", "transformers", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
