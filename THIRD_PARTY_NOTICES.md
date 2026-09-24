# Third-party notices

ContextLens code is released under the Apache License 2.0 (`LICENSE`). It uses,
and its repository contains data derived from, the third-party material below.

## Data

| material | used for | licence / terms | attribution |
|---|---|---|---|
| Wikipedia article text — `wikimedia/wikipedia` 20231101.en (rev `b04c8d1`) and `legacy-datasets/wikipedia` 20220301.en (rev `97a0b05`) | training/evaluation passages in `data/processed/` | CC BY-SA 3.0 and GFDL | © Wikipedia contributors. Each article's URL is listed in `data/manifest/articles.csv`. Derived passages are distributed under CC BY-SA 3.0. |
| DBpedia (public SPARQL endpoint) — Wikipedia category graph | labels (`data/manifest/articles.csv`) | CC BY-SA 3.0 and GFDL | © DBpedia Association and Wikipedia contributors |
| Stack Exchange questions — Stack Exchange API | `data/external/se_subtopic_eval.jsonl` (evaluation only) | CC BY-SA (2.5 / 3.0 / 4.0 depending on the post date) | Each record carries the question link; content by the respective Stack Exchange users |
| MTEB `StackExchangeClustering` titles (rev `9006e01`) | general-topic and OOD evaluation (not committed; regenerated) | derived from the Stack Exchange data dump, CC BY-SA | Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, 2022; Stack Exchange users |
| AG News, 20 Newsgroups, DBpedia-14, DBPedia Classes, News Category (HuffPost) | measured once for the dataset comparison only; nothing redistributed | see `docs/DATASET_RESEARCH.md` | respective authors |

## Models (downloaded at pinned revisions, not redistributed in git)

| model | licence |
|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| `BAAI/bge-small-en-v1.5` | MIT |
| `intfloat/e5-small-v2` | MIT |
| `sentence-transformers/all-mpnet-base-v2` | Apache-2.0 |
| `facebook/bart-large-mnli` (zero-shot baseline only) | MIT |

The trained artifact stores a local copy of the selected encoder in
`models/contextlens-topic/encoder/` (git-ignored); its licence is the one above.

## Online services (runtime, optional)

| service | terms |
|---|---|
| Wikipedia / MediaWiki Action API | Wikimedia Terms of Use and API etiquette (descriptive User-Agent, modest request rate) |
| DuckDuckGo Instant Answer API | DuckDuckGo terms; results are shown with their source |

The application sends only the composed theme phrase and at most two
vocabulary words to these services (see `docs/MODEL_CARD.md`, privacy).

## Python libraries (versions from `requirements.txt`)

| package | licence |
|---|---|
| numpy, scipy, pandas, scikit-learn | BSD-3-Clause |
| pyarrow, requests, transformers, sentence-transformers, huggingface_hub | Apache-2.0 |
| torch | BSD-3-Clause |
| skops | MIT |
| matplotlib | Matplotlib licence (PSF-based) |
| pytest, ruff, mypy (development only) | MIT |
