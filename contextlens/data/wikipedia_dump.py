"""Extract article text from the pinned Wikipedia dump on the Hugging Face Hub.

``wikimedia/wikipedia`` config ``20231101.en`` is a fixed snapshot (41 parquet
shards, ~11.6 GB). Pinning the dataset *revision* makes the text reproducible,
unlike the live MediaWiki API (which is also heavily rate-limited from shared
IPs). Shards are streamed one at a time and deleted after filtering so the
peak disk usage stays around one shard.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

# Only the beginning of an article is used (lead + first sections), so keep a
# bounded prefix to limit memory and disk usage.
MAX_STORED_CHARS = 12000


def shard_filename(config: str, index: int, total: int) -> str:
    return f"{config}/train-{index:05d}-of-{total:05d}.parquet"


def extract_articles(
    titles: Iterable[str],
    *,
    repo: str,
    config: str,
    revision: str,
    num_shards: int,
    download_dir: Path,
    keep_shards: bool = False,
) -> dict[str, dict]:
    """Return ``{title: {"wiki_id", "url", "text"}}`` for every title found."""
    wanted = sorted(set(titles))
    wanted_arr = pc.cast(wanted, "string") if wanted else None
    found: dict[str, dict] = {}
    for i in range(num_shards):
        fname = shard_filename(config, i, num_shards)
        path = Path(
            hf_hub_download(repo, fname, repo_type="dataset", revision=revision, local_dir=download_dir)
        )
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["id", "url", "title", "text"], batch_size=20000):
            mask = pc.is_in(batch.column("title"), value_set=wanted_arr)
            hits = batch.filter(mask)
            for row in hits.to_pylist():
                found[row["title"]] = {
                    "wiki_id": row["id"],
                    "url": row["url"],
                    "text": row["text"][:MAX_STORED_CHARS],
                }
        log.info("shard %d/%d: %d/%d titles found so far", i + 1, num_shards, len(found), len(wanted))
        if not keep_shards:
            path.unlink(missing_ok=True)
    return found
