"""Central configuration.

Every tunable value lives here (no magic numbers scattered across modules).
Values can be overridden with ``CONTEXTLENS_<NAME>`` environment variables,
e.g. ``CONTEXTLENS_WEB_TIMEOUT=3``. Paths are resolved relative to the
repository root so the scripts work from any working directory.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42

# User agent sent to every third-party HTTP service (Wikimedia policy requires a
# descriptive UA with contact information).
USER_AGENT = "ContextLensNLP/1.0 (https://github.com/Ozgurisikdamar/Cont)"


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    taxonomy: Path = REPO_ROOT / "configs" / "taxonomy.json"
    data_raw: Path = REPO_ROOT / "data" / "raw"
    data_processed: Path = REPO_ROOT / "data" / "processed"
    data_external: Path = REPO_ROOT / "data" / "external"
    data_manifest: Path = REPO_ROOT / "data" / "manifest"
    models: Path = REPO_ROOT / "models"
    reports: Path = REPO_ROOT / "reports"
    figures: Path = REPO_ROOT / "reports" / "figures"


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Instantiate via :func:`load_settings`."""

    db_path: Path = REPO_ROOT / "contextlens.db"
    model_dir: Path = REPO_ROOT / "models" / "contextlens-topic"
    log_level: str = "WARNING"
    # --- classification -------------------------------------------------
    # Below this calibrated max-probability the input is flagged "uncertain".
    # None = use the value chosen in the benchmark and stored in the model
    # artifact (the behaviour that evaluate.py measures); set a number to override.
    min_confidence: float | None = None
    # Subtopics are shown when their (per-label calibrated) probability is at or
    # above the validation-tuned threshold stored in the model artifact; this is
    # an upper bound on how many are displayed.
    max_subtopics: int = 3
    # --- conversation tracking ------------------------------------------
    decay: float = 0.7
    # A topic takes part in the conversation theme when its share of the
    # accumulated mass is at least this value.
    theme_min_share: float = 0.2
    max_theme_topics: int = 3
    # --- web search ------------------------------------------------------
    web_enabled: bool = True
    web_timeout: float = 6.0
    web_retries: int = 2
    web_backoff: float = 1.0
    max_results: int = 3
    cache_ttl_hours: int = 72


def _coerce(value: str, target_type: type) -> object:
    if target_type is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if issubclass(target_type, Path):
        return Path(value).expanduser()
    convert: Callable[[str], object] = target_type
    return convert(value)


def load_settings(**overrides: object) -> Settings:
    """Build settings from defaults, environment variables and explicit overrides."""
    base = Settings()
    values: dict[str, object] = {}
    for f in fields(Settings):
        default = getattr(base, f.name)
        raw = os.environ.get(f"CONTEXTLENS_{f.name.upper()}")
        if raw is None:
            values[f.name] = default
        elif default is None:  # optional float setting; "", "none" or "model" keep the default
            values[f.name] = None if raw.strip().lower() in {"", "none", "model"} else float(raw)
        else:
            values[f.name] = _coerce(raw, type(default))
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


PATHS = Paths()


@dataclass(frozen=True)
class DataConfig:
    """Corpus construction parameters (see docs/DATASET_CARD.md)."""

    # Pinned Wikipedia snapshot on the Hugging Face Hub.
    wiki_repo: str = "wikimedia/wikipedia"
    wiki_config: str = "20231101.en"
    wiki_revision: str = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    wiki_num_shards: int = 41
    # Second pinned snapshot, different parser: some articles are missing from
    # 20231101.en (e.g. "Gold", "Spacetime") but present here. Used only for
    # titles the first snapshot does not have (docs/DATASET_CARD.md).
    legacy_wiki_repo: str = "legacy-datasets/wikipedia"
    legacy_wiki_config: str = "data/20220301.en"
    legacy_wiki_revision: str = "97a0b052c326b45fb68593a14972d9eed884cd17"
    legacy_wiki_num_shards: int = 41
    # External Stack Exchange titles (MTEB StackExchangeClustering), pinned.
    se_cluster_repo: str = "mteb/stackexchange-clustering"
    se_cluster_revision: str = "9006e0189d5dfd6b10255843bbdac6bd1676a3bf"
    sparql_endpoint: str = "https://dbpedia.org/sparql"
    se_api: str = "https://api.stackexchange.com/2.3"
    max_articles_per_subtopic: int = 450
    max_passages_per_article: int = 4
    min_passage_words: int = 12
    max_passage_words: int = 60
    ood_articles_per_category: int = 60
    # Split ratios (grouped by article, stratified by general topic).
    val_fraction: float = 0.15
    test_fraction: float = 0.15


DATA = DataConfig()
