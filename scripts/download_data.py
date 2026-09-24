"""Acquire all data used by ContextLens.

    python scripts/download_data.py --corpus         # Wikipedia training corpus (~12 GB streamed, ~15 min)
    python scripts/download_data.py --stackexchange  # external evaluation sets
    python scripts/download_data.py --all

The committed manifests under data/manifest/ pin the result; see
docs/DATASET_CARD.md for provenance and licences.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from contextlens.config import DATA, PATHS
from contextlens.data.corpus import build_corpus
from contextlens.data.labels import stable_hash
from contextlens.data.stackexchange import build_subtopic_eval_set, load_cluster_titles
from contextlens.logging_setup import setup_logging
from contextlens.taxonomy import load_taxonomy

log = logging.getLogger("download_data")


def _dev_or_test(key: str) -> str:
    """Deterministic 50/50 split of external data into ext_dev / ext_test."""
    return "ext_dev" if int(stable_hash(key), 16) % 2 == 0 else "ext_test"


# Deterministic caps keep the external set balanced across topics and cheap to
# embed with several encoders (the MTEB files hold ~84k titles).
EXT_CAP_PER_TOPIC = 1500
EXT_CAP_OOD = 6000


def cap_external(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["split"], r["general"] or "__ood__"), []).append(r)
    kept: list[dict] = []
    for (_split, topic), members in sorted(groups.items()):
        cap = EXT_CAP_OOD if topic == "__ood__" else EXT_CAP_PER_TOPIC
        kept += sorted(members, key=lambda r: stable_hash(r["site"] + r["text"]))[:cap]
    return sorted(kept, key=lambda r: (r["site"], r["text"]))


def build_stackexchange() -> dict:
    tax = load_taxonomy()
    out_dir = PATHS.data_external
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    sub_rows = build_subtopic_eval_set(tax, DATA.se_api, out_dir / "stackexchange" / "api_cache", missing)
    for r in sub_rows:
        r["split"] = _dev_or_test(f"{r['site']}:{r['question_id']}")
    with open(out_dir / "se_subtopic_eval.jsonl", "w", encoding="utf-8") as fh:
        for r in sub_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    site_cfg = tax.raw["stackexchange_general_sites"]
    files = [
        Path(
            hf_hub_download(
                DATA.se_cluster_repo,
                name,
                repo_type="dataset",
                revision=DATA.se_cluster_revision,
                local_dir=PATHS.data_raw / "mteb_stackexchange",
            )
        )
        for name in ("valid.jsonl", "test.jsonl")
    ]
    gen_rows = load_cluster_titles(files, site_cfg["in_taxonomy"], site_cfg["ood"])
    for r in gen_rows:
        r["split"] = _dev_or_test(f"{r['site']}:{r['text']}")
    gen_rows = cap_external(gen_rows)
    with open(out_dir / "se_general_eval.jsonl", "w", encoding="utf-8") as fh:
        for r in gen_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "se_subtopic_questions": len(sub_rows),
        "se_subtopic_missing_pairs": missing,
        "se_general_titles": len(gen_rows),
        "se_general_caps": {"per_topic_per_split": EXT_CAP_PER_TOPIC, "ood_per_split": EXT_CAP_OOD},
        "se_general_ood_titles": sum(r["is_ood"] for r in gen_rows),
    }
    (PATHS.data_manifest / "external_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", action="store_true", help="build the Wikipedia training corpus")
    parser.add_argument("--stackexchange", action="store_true", help="build the external evaluation sets")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    if not (args.corpus or args.stackexchange or args.all):
        parser.error("choose --corpus, --stackexchange or --all")
    PATHS.data_manifest.mkdir(parents=True, exist_ok=True)
    if args.corpus or args.all:
        summary = build_corpus(load_taxonomy(), PATHS, DATA)
        log.info("corpus: %s", {k: v for k, v in summary.items() if k != "crawl_stats"})
    if args.stackexchange or args.all:
        log.info("stackexchange: %s", build_stackexchange())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
