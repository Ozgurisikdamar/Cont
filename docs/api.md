# API reference

ContextLens has no network server. Its interfaces are a **command-line app**,
a small **Python API**, the **scripts** that build data and models, the
**SQLite schema**, and the **file formats** it reads and writes.

## 1. Command line — `python project.py`

```
python project.py [--model-dir DIR] [--db PATH] [--no-web] [--debug]
                  [--script FILE | --once TEXT]
```

| option | default | effect |
|---|---|---|
| `--model-dir DIR` | `models/contextlens-topic` | trained artifact to load (never retrained at start-up) |
| `--db PATH` | `contextlens.db` | SQLite database file (created and migrated if missing) |
| `--no-web` | web on | fully local: no network requests at all |
| `--debug` | off | verbose logs to stderr and `logs/contextlens.log` |
| `--script FILE` | — | read messages (one per line, blank lines skipped) instead of the keyboard |
| `--once TEXT` | — | analyse one message and exit |

**Interactive commands** (case-insensitive):

| command | effect |
|---|---|
| `help` | list commands |
| `history` | this conversation's messages with their classification |
| `reset` | start a new conversation context; **stored records are kept** |
| `exit`, `quit`, `q` | leave (Ctrl+C and Ctrl+D / EOF also leave cleanly) |

**Exit codes:** `0` normal end · `2` artifact missing/corrupt or script file
unreadable (the message says how to fix it).

**Output of one turn** (real output is in the README):

```
[Text analysis]
General topic : <general topic>
Confidence    : <calibrated probability>
Subtopics     :
  - <General > Subtopic> (<P(subtopic | general)>)

[Conversation]
Theme         : <active topics, e.g. Books + Science + Biology>
In words      : <theme phrase, e.g. science books about biology>
Search query  : "<query that produced the results>"

[Web results]
1. <title> (<source>)
   <summary>
   <url>

Saved to the database.
```

An `uncertain` message shows `Note: uncertain - <reason>` and is not added to
the conversation theme; neither is a `non_english` message (the language gate is
confident the text is not English). An uninformative message (empty, symbols,
only stop words) is not a turn: nothing is stored.

## 2. Runtime settings (environment)

Every field of `contextlens.config.Settings` can be overridden with an
environment variable `CONTEXTLENS_<FIELD>` (upper case); values are coerced to
the field's type. Command-line options win over the environment.

| setting | default | meaning |
|---|---|---|
| `db_path` | `contextlens.db` | database file |
| `model_dir` | `models/contextlens-topic` | artifact directory |
| `log_level` | `WARNING` | logging level when `--debug` is not given |
| `min_confidence` | `None` | below this calibrated probability a prediction is `uncertain`; `None` uses the value tuned in the benchmark and stored in the artifact (`CONTEXTLENS_MIN_CONFIDENCE=0.5` overrides) |
| `max_subtopics` | `3` | at most this many subtopics are reported per message (the primary subtopic + sibling suggestions) |
| `decay` | `0.7` | conversation decay (`score = score·decay + weight·p`); chosen by `scripts/tune_decay.py` |
| `theme_min_share` | `0.2` | a topic is part of the theme when it holds ≥ this share of the decayed mass |
| `max_theme_topics` | `3` | at most this many topics in a theme |
| `theme_expire_after` | `4` | clear the context after this many consecutive messages without a confident topic (`0` = never); chosen on validation conversations (D-31) |
| `theme_confirm_turns` | `2` | the dominant topic changes only after this many consecutive confident messages agree on a new one (hysteresis against one-message tangents, D-31) |
| `web_enabled` | `true` | web search on/off (`--no-web`) |
| `web_timeout` | `6.0` | seconds per HTTP request |
| `web_retries` | `2` | retries for timeouts, 429 and 5xx (with back-off, `Retry-After` honoured up to 30 s) |
| `web_backoff` | `1.0` | back-off base in seconds (`backoff · 2^attempt`) |
| `max_results` | `3` | results kept per search |
| `cache_ttl_hours` | `72` | lifetime of cached search responses |

Example: `CONTEXTLENS_DECAY=0.8 CONTEXTLENS_WEB_ENABLED=false python project.py`.

## 3. Python API

### Load and predict

```python
from contextlens.config import load_settings
from contextlens.models.artifact import load_artifact

model = load_artifact(load_settings().model_dir)          # raises ArtifactError if missing/corrupt
p = model.predict("Quantum processors can speed up certain algorithms by using qubits.")
p.status          # "ok" | "uncertain" | "non_english" | "uninformative"
p.general         # "technology"
p.confidence      # calibrated P(general)
p.subtopics       # primary subtopic first, then sibling suggestions:
                  # (SubtopicScore(id="quantum_computing", probability=P(sub | general)), ...)
p.general_probs   # {"physics": ..., ...} — sums to 1
p.subtopic_probs  # joint P(sub) = P(general) · P(sub | general), all 28 subtopics
p.ood_score       # in-domain score of the off-topic detector (v1.1: negative Mahalanobis
                  # distance to the nearest general-topic mean; higher = more in-domain)
p.reasons         # why it is uncertain, e.g. ("low confidence (35%)",) or the language gate
model.looks_english(text)                  # language gate: fastText lid.176 + training lexicon,
                                           # per-length reject confidence (configs/model.json)
model.predict_many(texts)                  # batched
model.predict_proba(normalised_texts)      # (P(general), P(sub | general), ood score) as arrays
model.head_outputs(embeddings)             # (P(general), P(sub | general), logits)
model.ood_scores(embeddings, general, logits)
```

`load_artifact(directory, min_confidence=None, encoder=None, max_subtopics=3)` —
`min_confidence` overrides the value stored in the artifact; `encoder` injects
an alternative encoder (used by tests).

### A conversation

```python
from contextlens.database.db import Database
from contextlens.models.artifact import load_vocabulary
from contextlens.services.pipeline import ConversationSession
from contextlens.services.websearch import WebSearcher
from contextlens.taxonomy import load_taxonomy

settings = load_settings()
db = Database(settings.db_path)
session = ConversationSession(model, load_taxonomy(), settings, db,
                              WebSearcher(cache=db), load_vocabulary(settings.model_dir))
r = session.process("I read a novel about the history of the Ottoman Empire.")
r.prediction   # Prediction (above)
r.theme        # ComposedTheme(label, phrase, generals, subtopics, concepts)
r.query        # SearchQuery(primary, fallback, concepts) or None
r.search       # SearchOutcome(query, status, provider, results, error) or None
r.warnings     # e.g. ["could not save text to the database"]
session.history(); session.reset(); session.close(); db.close()
```

`SearchOutcome.status`: `ok` · `cached` · `no_results` · `offline` · `disabled`.

### Building blocks

| call | purpose |
|---|---|
| `preprocessing.text.normalize(text, lowercase=False)` | the exact normalisation used in training and at runtime |
| `services.tracker.ConversationTracker(decay, min_share, max_topics).update(general_probs, subtopic_probs, weight)` | decayed topic scores; `.theme(children)` → active topics |
| `services.composer.compose(theme, taxonomy)` | active topics → `ComposedTheme` |
| `services.query.build_query(phrase, message, vocabulary, max_keywords=2, concepts=())` | theme → `SearchQuery` (only vocabulary words are sent) |
| `services.websearch.WebSearcher(...).search(queries)` | Wikipedia → DuckDuckGo, cache, circuit breaker |
| `taxonomy.load_taxonomy()` | validated taxonomy (`general_ids`, `subtopic_ids`, `children_of`, `parent_of`, `display`) |

## 4. Scripts

| command | reads | writes |
|---|---|---|
| `python scripts/dataset_research.py` | candidate datasets (pinned) | `reports/dataset_research.json` |
| `python scripts/download_data.py --corpus` | DBpedia, Wikipedia dumps (pinned) | `data/manifest/*`, `data/processed/*.parquet` |
| `python scripts/download_data.py --stackexchange` | Stack Exchange API, MTEB titles (pinned) | `data/external/se_*_eval.jsonl` |
| `python scripts/eda.py` | processed data | `reports/eda.json`, `reports/figures/eda_*.png` |
| `python scripts/run_experiments.py [--sections ...] [--feats ...]` | processed + external data (development splits only) | `reports/experiments/{general,zeroshot,hierarchy,calibration,ood}.json` |
| `python scripts/finetune_transformer.py --encoder minilm-l6` | processed + external data | `reports/experiments/finetune_<encoder>.json` |
| `python scripts/zeroshot_nli.py --per-class 40` | processed + external data | `reports/experiments/zeroshot_nli.json` |
| `python train.py` | `configs/model.json`, processed data, `data/external/ood_conversational.jsonl` (train half) | `models/contextlens-topic/` (no test-set evaluation) |
| `python scripts/tune_decay.py` | artifact, processed data | `reports/experiments/decay.json` |
| `python evaluate.py --stage dev` | artifact, development data | `reports/evaluation_dev.json`, `reports/figures/*_dev.png` |
| `python evaluate.py --stage locked` | artifact, `data/locked/*`, locked halves; refuses without `reports/locked/FREEZE.json` or with a fingerprint mismatch | `reports/locked/results.json`, `reports/locked/raw/`, `reports/figures/*_locked.png` |
| `python scripts/freeze.py` | clean working tree | `reports/locked/FREEZE.json` (SHA-256 of config, taxonomy, settings, passages, locked manifest, artifact metadata); refuses once `results.json` exists |
| `python scripts/build_locked_sets.py [--repair]` | Wikipedia dump (pinned), Stack Exchange API (questions created in 2026) | `data/locked/{wiki_locked,se_locked}.jsonl`, `data/locked/MANIFEST.json` |
| `python scripts/build_ood_conversational.py` | CLINC150 (pinned) | `data/external/ood_conversational.jsonl` (train / dev / locked) |
| `python scripts/build_language_eval.py` | Tatoeba (pinned) | `data/external/language_eval.jsonl` (dev / locked) |
| `python scripts/language_gate_experiment.py` | language eval dev half | `reports/experiments/language_gate.json` |
| `python scripts/head_comparison.py` | fine-tuned encoder, dev data | `reports/experiments/head_comparison.json` |
| `python scripts/ood_experiment.py` | fine-tuned encoder, dev data | `reports/experiments/ood_detectors.json` |
| `python scripts/label_audit_v2.py` | processed data | `reports/label_audit_v2.json` (310 items, Wilson intervals) |
| `bash scripts/ci.sh` | – | ruff, format check, mypy, offline pytest, model tests (the steps of `.github/workflows/ci.yml`) |
| `python scripts/finetune_transformer.py --encoder minilm-l6 --export DIR` | as above | + fine-tuned encoder (float16) in `DIR` |
| `python scripts/relabel_corpus.py` | cached crawl, `configs/taxonomy.json` | relabelled manifest/passages (splits kept), `reports/relabel_taxonomy_<version>.json` |
| `python scripts/report_tables.py` | `reports/experiments/*.json` | `reports/tables.md`, `reports/experiment_log.md` |
| `python scripts/hardware_info.py` | – | `reports/hardware.json` |
| `python scripts/fp16_storage_check.py` | fine-tuned encoder | `reports/fp16_storage.json` |
| `python scripts/test_table.py` | `reports/tests/*.xml` | `reports/tests/results.md` |

## 5. Database schema (`contextlens.db`, `PRAGMA user_version = 1`)

| table | key columns | notes |
|---|---|---|
| `model_versions` | `id`, `model_name`, `model_version` (unique pair), `trained_at`, `dataset_version`, `metrics_json` | one row per artifact version used |
| `sessions` | `id` (uuid4), `started_at`, `ended_at`, `model_version_id` → `model_versions` | `reset` ends one session and starts another |
| `texts` | `id`, `session_id` → `sessions`, `turn`, `content`, `general_topic`, `general_confidence`, `subtopics_json`, `general_probs_json`, `uncertain`, `status`, `model_version_id` | one row per analysed message |
| `conversation_topics` | `id`, `session_id`, `text_id` → `texts`, `theme_label`, `theme_phrase`, `general_scores_json`, `subtopic_scores_json`, `query` | theme state after each message |
| `search_results` | `id`, `session_id`, `conversation_topic_id` → `conversation_topics`, `query`, `provider`, `rank`, `title`, `summary`, `url`, `source` | results shown to the user |
| `search_cache` | (`query`, `provider`) primary key, `payload_json`, `fetched_at` | response cache, TTL checked on read |

Indexes: `texts(session_id, turn)`, `conversation_topics(session_id)`,
`conversation_topics(text_id)`, `search_results(session_id)`,
`search_results(conversation_topic_id)`, `search_cache(fetched_at)`.
Foreign keys are enforced with `ON DELETE CASCADE` from sessions.

```sql
-- the last conversation, turn by turn
SELECT t.turn, t.content, t.general_topic, round(t.general_confidence, 3), c.theme_phrase, c.query
FROM texts t JOIN conversation_topics c ON c.text_id = t.id
WHERE t.session_id = (SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1)
ORDER BY t.turn;
```

## 6. File formats

* `configs/taxonomy.json` — see [TAXONOMY.md](TAXONOMY.md).
* `configs/model.json` — training configuration chosen by the benchmark
  (encoder key, `head_type` — `flat_softmax` or `hierarchical` —, regularisation `C`,
  OOD calibration set and quantile, `ood.method` of the off-topic detector,
  minimum confidence `"auto"` + grid and coverage, `language_gate.reject_confidence`
  per input length, vocabulary size).
* `data/processed/passages.parquet` — `passage_id, wiki_id, title, general,
  subtopics ("a|b"), depth, split, text_source, text`.
* `data/processed/ood_passages.parquet` — `passage_id, wiki_id, title, category, split, text`.
* `data/manifest/articles.csv` — one row per article: title, page id, URL, labels,
  split, text source, dump title, number of passages, SHA-256 of the text used.
* `data/external/se_subtopic_eval.jsonl` — `{"source", "site", "question_id", "text", "tags",
  "link", "score", "general", "subtopics": [...], "split"}` (split: `ext_dev` / `ext_test`).
* `data/external/se_general_eval.jsonl` — `{"source", "site", "text", "general" | null,
  "is_ood", "split"}`.
* `data/locked/wiki_locked.jsonl` — `{"source", "title", "general", "subtopics", "text"}`;
  `data/locked/se_locked.jsonl` — `{"site", "question_id", "text", "tags", "created",
  "general", "subtopics", "is_ood", "in_general", "site_general"}` (`in_general` and the
  subtopic view are independent flags, decisions.md D-36). Checksums in `MANIFEST.json`.
* `data/external/ood_conversational.jsonl` — `{"source", "intent", "split", "text"}`.
* Artifact files — see [architecture.md §3](architecture.md#artifact-modelscontextlens-topic).
