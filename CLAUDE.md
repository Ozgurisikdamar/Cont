# ContextLens — CLAUDE.md

Quick reference for anyone (human or agent) working in this repository.
Everything in the project is **English**: data, input, output, code, docs.

## What this is

A local, conversation-aware topic analyser. For every message it predicts a
**general topic** (8) and **subtopics** (28) with calibrated confidence, says
*uncertain* for off-topic text, keeps a **decayed conversation context**,
composes the active topics into one theme ("science books about biology"),
builds a **web-search query** (Wikipedia, DuckDuckGo fallback, no API keys) and
stores everything in **SQLite**.

## Read first

| question | file |
|---|---|
| how to install / run / train / test | `README.md` |
| why this data (candidate comparison) | `docs/DATASET_RESEARCH.md` |
| what the data is, how labels were made | `docs/DATASET_CARD.md` |
| the 8 × 28 taxonomy, composition rules | `docs/TAXONOMY.md` (+ `configs/taxonomy.json`) |
| components, one-turn sequence, schema | `docs/architecture.md` |
| CLI, settings, Python API, scripts, file formats | `docs/api.md` |
| benchmark: every model tried, and the winner | `docs/MODEL_REPORT.md`, `docs/EXPERIMENTS.md` |
| where the model fails | `docs/ERROR_ANALYSIS.md` |
| intended use, limits | `docs/MODEL_CARD.md`, `KNOWN_ISSUES.md` |
| why a decision was taken | `decisions.md` (ADR-lite, D-N) |
| what was done, what is open | `sprints.md`, `handover.md`, `PROJECT_STATE.md` |
| test evidence | `docs/TEST_REPORT.md` |
| the final write-up | `docs/FINAL_REPORT.md` |

## Commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python project.py                       # interactive app (needs the trained artifact)
python project.py --no-web --once "text"

python scripts/download_data.py --all   # corpus + Stack Exchange sets (network, ~20 min)
python scripts/eda.py
python scripts/run_experiments.py       # benchmark (CPU: about an hour)
python train.py && python scripts/tune_decay.py && python evaluate.py

ruff check . && ruff format --check . && mypy contextlens project.py train.py evaluate.py
pytest -q                               # offline suite (fake encoder, no downloads)
pytest -q -m model                      # needs the trained artifact
pytest -q -m network                    # needs internet
```

## Layout

```
contextlens/      package: config, taxonomy, preprocessing, data/ (offline acquisition),
                  models/ (encoders, heads, TopicModel, artifact, training),
                  services/ (tracker, composer, query, websearch, pipeline),
                  database/, evaluation/, cli/
configs/          taxonomy.json (source of truth), model.json (chosen by the benchmark)
scripts/          data acquisition, EDA, experiments, decay tuning
data/manifest/    committed label manifest (articles.csv, crawl stats)
data/processed/   committed passages (parquet)
data/external/    Stack Exchange evaluation sets
reports/          experiment JSON, evaluation JSON, figures
models/           trained artifact (encoder weights git-ignored)
tests/            unit, integration, acceptance, edge cases
```

## Rules

* **No fabricated numbers.** Every metric in a document comes from a file in
  `reports/`. If something was not run, say so.
* **Test data is report-only.** Hyper-parameters and model choice use the
  Wikipedia *validation* split and Stack Exchange *ext_dev*; `test` and
  `ext_test` are never used for a decision.
* **Acceptance sentences are never training data** (`tests/acceptance_cases.json`;
  `scripts/eda.py` checks the corpus for them).
* The taxonomy lives in `configs/taxonomy.json` only; composition and labels
  are driven by its metadata — do not hard-code topic pairs in Python.
* The model is never retrained at application start-up.
* SQL is always parameterised; the database layer must never crash a turn.
* Keep `random_state` / `RANDOM_SEED = 42` everywhere randomness appears.

## Pitfalls (learned the hard way)

* **Two Wikipedia snapshots are needed.** `wikimedia/wikipedia` 20231101.en
  silently lacks some articles (e.g. "Gold", "Spacetime"); the corpus builder
  falls back to `legacy-datasets/wikipedia` 20220301.en, then to redirect
  aliases for renamed articles. Case-insensitive title matching is **unsafe**
  ("GOLD", "TIN" are different pages).
* **The public DBpedia endpoint** has no abstracts and no `dbo:wikiPageID`,
  answers partial results with HTTP 200 + `X-SQL-State: S1TAT`, and goes into
  maintenance (HTTP 502). The client retries, splits batches and caches every
  response in `data/raw/sparql_cache`.
* **Wikipedia's API rate-limits shared IPs** (HTTP 429 even for the first
  request from some cloud machines). Runtime search falls back to DuckDuckGo
  and a circuit breaker skips a failing provider for 5 minutes.
* **ruff excludes must be anchored** (`./data`, not `data`): an unanchored
  pattern also excluded `contextlens/data/` and `contextlens/models/`.
* `tests/conftest.py` fixtures `taxonomy` and `fake_model` are session-scoped:
  never mutate them in a test — use `dataclasses.replace`.
* `pkill -f <pattern>` can match the calling shell's own command line; kill by
  PID instead.

## Git

* Branch: `claude/ecstatic-lovelace-arcc4x`; do not open pull requests unless asked.
* Commit identity (repository-local config): `Ozgurisikdamar
  <74007174+Ozgurisikdamar@users.noreply.github.com>`, `commit.gpgsign false`.
  Verify author **and** committer with
  `git log --format='%h A:%an <%ae> C:%cn <%ce>' -5` before every push.
* Commit messages, PR texts and comments carry **no** co-author lines, session
  links or "generated with" signatures.
