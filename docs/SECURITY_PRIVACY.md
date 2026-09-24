# Security and privacy review

Scope: the application (`project.py` and `contextlens/`), the training/data
scripts, and the committed data. Each item states the risk, what the code does,
and the evidence (test or file).

## 1. Threat model

ContextLens is a single-user, local console application. Inputs are the user's
messages (untrusted text), responses from two public web APIs (untrusted JSON),
the model artifact on disk (trusted only after verification) and the local SQLite
file. There is no server, no authentication and no secret: the web APIs used are
key-less.

## 2. Findings and controls

| # | area | risk | control | evidence |
|---|---|---|---|---|
| 1 | SQL | injection through message text or table names | every statement is parameterised; the only dynamic identifier (`count(table)`) is checked against an allow-list | `tests/test_db.py::test_sql_injection_is_stored_literally`, `::test_count_rejects_unknown_table` |
| 2 | database integrity | orphaned rows, partial writes, concurrent access | `PRAGMA foreign_keys=ON`, one transaction per write, WAL, `busy_timeout=5000`; a locked or unwritable database becomes a `DatabaseError` → a warning, never a crash | `::test_foreign_keys_enforced`, `::test_locked_database_surfaces_as_database_error`, `tests/test_integration.py::test_database_failure_is_a_warning_not_a_crash` |
| 3 | schema | an older program opening a newer database | `PRAGMA user_version`; newer schemas are refused | `::test_newer_schema_is_refused` |
| 4 | model artifact | code execution through pickle; tampered or truncated files | heads stored with **skops** and loaded with an explicit allow-list of types; `.npy` loaded with `allow_pickle=False`; SHA-256 of every file checked against `metadata.json`; every encoder file (weights, tokenizer, configs) checked against a path/size/SHA-256 manifest — missing, changed or extra files are refused — and the encoder verified by a probe-embedding fingerprint (a swapped or missing encoder is refused, no silent Hub fallback) | `contextlens/models/artifact.py`, `tests/test_integration.py::test_artifact_round_trip_and_integrity`, `::test_artifact_refuses_an_encoder_it_was_not_trained_with`, `::test_encoder_files_are_verified`, `tests/test_encoders.py` |
| 5 | input size | memory/time exhaustion from a huge paste | the model reads at most 5,000 characters; the database stores at most 10,000 | `contextlens/preprocessing/text.py`, `tests/test_db.py::test_stored_message_length_is_bounded`, `tests/test_acceptance.py::test_very_long_input_is_handled` |
| 6 | input content | HTML/script, control characters, odd Unicode | NFKC normalisation, HTML unescape + tag removal, control characters removed; nothing user-supplied is ever rendered as HTML (terminal output only) | `tests/test_text.py` |
| 7 | web responses | malicious links, oversized or malformed payloads | HTTPS-only URLs on allow-listed domains (`wikipedia.org`, `duckduckgo.com`); snippets stripped of markup and truncated (200/500 chars); responses > 5 MB rejected; malformed JSON → provider treated as failed | `tests/test_websearch.py::test_parse_wikipedia_orders_by_search_rank_and_drops_unsafe_urls`, `::test_clean_snippet_and_url_validation`, `::test_parse_wikipedia_handles_malformed_payloads` |
| 8 | web availability | hangs, retry storms, rate limits | per-request timeout (6 s), bounded retries with exponential back-off, `Retry-After` capped at 30 s, 5-minute circuit breaker per provider, response cache (72 h) | `tests/test_websearch.py::test_get_json_*`, `::test_failed_provider_is_skipped_until_its_cooldown_ends` |
| 9 | privacy — outbound data | leaking what the user typed | the query is the theme phrase plus at most two words that appear in the public training vocabulary; names, numbers, typos, e-mail addresses and other out-of-vocabulary tokens are never sent; off-topic messages add no words; `--no-web` makes the app fully offline | `contextlens/services/query.py`, `tests/test_query.py`, `tests/test_integration.py::test_uncertain_turn_does_not_change_theme` |
| 10 | privacy — local data | the database contains everything the user typed, in plain text | documented; the file is local (`contextlens.db`, git-ignored); `reset` does not delete (by design of the brief) — delete the file to erase history | README §Privacy, `.gitignore` |
| 11 | logs | user text in log files | the application logs never include message text; search failures log the query (theme phrase + vocabulary words only); `--debug` writes to `logs/` (git-ignored) | `grep -rn "log\." contextlens/services contextlens/cli contextlens/database` |
| 12 | secrets | credentials in the repository | none are needed; `.env` is git-ignored; no tokens appear in the code or data | repository scan |
| 13 | supply chain | unpinned dependencies or models | Python packages pinned in `requirements.txt`; datasets and models pinned by revision hash | `requirements.txt`, `contextlens/models/encoders.py`, `contextlens/config.py` |
| 14 | committed data | personal data in the corpus | Wikipedia passages are public encyclopedic text; biographies were largely excluded by category rules; Stack Exchange records contain question titles and links only (no user names) | `docs/DATASET_CARD.md` |

## 3. Residual risks

* The database is not encrypted; anyone with access to the file can read the
  conversation history.
* Search results are third-party content shown verbatim (after cleaning); their
  correctness is not verified.
* The classifier's output can be wrong with high confidence on text unlike its
  training data; the OOD gate reduces but does not remove this
  ([MODEL_CARD.md](MODEL_CARD.md)).
* Checksums detect corruption and accidental edits, not a deliberate attacker:
  someone who can write `metadata.json` can also rewrite the hashes. There is no
  signature; the safety of loading still rests on skops' type allow-list and on
  safetensors (neither executes code).
