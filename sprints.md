# Sprints

Status of the work, sprint by sprint. The detailed "why" of each choice is in
`decisions.md`; the session log is `handover.md`.

## S1 — Scaffold and taxonomy ✅
- Repository layout, pinned `requirements.txt` (CPU PyTorch), ruff / mypy / pytest configuration.
- Taxonomy (8 general topics, 28 subtopics) in `configs/taxonomy.json` with role /
  broader / perspective metadata and curated category seeds.
- Settings with `CONTEXTLENS_*` environment overrides; logging; reproducibility helpers.

## S2 — Dataset research ✅
- Five candidate datasets measured at pinned revisions, four assessed from official
  documentation (`scripts/dataset_research.py`, `reports/dataset_research.json`).
- Finding: best existing coverage 3/8 general, 6/28 subtopics (measured) — decision to
  build a Wikipedia corpus; Stack Exchange kept for evaluation only.
  → `docs/DATASET_RESEARCH.md`, D-2, D-3.

## S3 — Corpus ✅
- DBpedia category crawl with retries, batch splitting, partial-result detection and
  an on-disk cache; label rules (minimum depth, ties excluded, depth ≤ 1 secondaries).
- Text from the pinned 2023 dump; discovered that it lacks some articles entirely →
  redirect aliases and the pinned 2022 dump as fallbacks (+963 articles).
  A REST-API backfill was tried and dropped (429 with 30 s Retry-After).
- Passages, grouped stratified split, near-duplicate filter; Stack Exchange general,
  subtopic and OOD sets. → `docs/DATASET_CARD.md`, D-4 … D-8.

## S4 — EDA and data quality ✅
- `scripts/eda.py`: distributions, length, anomalies, leakage (article, exact text,
  TF-IDF near-duplicates, acceptance sentences). All leakage checks at 0.
- Manual label audit of 48 training passages: 89.6% correct, 6.2% weak passage,
  4.2% wrong label (`reports/label_audit.json`).
- Fixed on the way: OOD test seed `Automobiles` is empty in DBpedia → `Cars`.

## S5 — Model benchmark ✅
- `scripts/run_experiments.py`: majority baseline; TF-IDF (3 featurizers) × LR / NB /
  Complement NB / linear SVM + Platt; four sentence encoders × LR; zero-shot label
  similarity; hierarchical vs flat subtopic heads with threshold sweep; temperature
  scaling; OOD scores (MSP, energy, centroid cosine, kNN) with thresholds calibrated
  on Wikipedia and on user-style questions.
- `scripts/finetune_transformer.py` (shared encoder, two heads), `scripts/zeroshot_nli.py`.
- Selection on validation + ext_dev only. → `docs/MODEL_REPORT.md`, `docs/EXPERIMENTS.md`.
- Result: fine-tuned MiniLM + flat 28-way softmax (D-22, D-23); centroid gate on
  questions (D-24); confidence floor by coverage (D-25). E-6 (ensemble) not run.
- Quantum acceptance failure traced to a label seed → taxonomy 1.1, relabel,
  re-fine-tune (D-27); language gate (D-29); decay tuned jointly with theme share (D-28).

## S6 — Application ✅
- `TopicModel` (calibrated flat subtopic softmax or hierarchical heads, OOD, confidence and language gates),
  safe artifact (skops allow-list, checksums).
- Conversation tracker (decay), metadata-driven theme composer, privacy-preserving
  query builder with concept fallbacks, web search (Wikipedia → DuckDuckGo, cache,
  circuit breaker), SQLite with migrations, CLI with `history` / `reset` / `exit`.

## S7 — Tests ✅
- Offline suite (fake encoder, no downloads): unit + integration + CLI.
- Data integrity (`tests/test_leakage.py`), real-artifact acceptance and edge cases
  (`tests/test_acceptance.py`, marker `model`), live search (`tests/test_network.py`,
  marker `network`). Last run: 162 passed, 0 failed. → `docs/TEST_REPORT.md`.

## S8 — Documentation and delivery ✅
- Written: DATASET_RESEARCH, DATASET_CARD, TAXONOMY, architecture, api, decisions,
  CLAUDE.md, THIRD_PARTY_NOTICES.
- With results: README, MODEL_REPORT, MODEL_CARD, EXPERIMENTS, ERROR_ANALYSIS,
  TEST_REPORT, FINAL_REPORT, KNOWN_ISSUES, PROJECT_STATE, handover.

## Backlog (not planned for v1)
- Owner decision: drop the superseded 45 MB model from git history (force push).
- E-6: fine-tune e5-small-v2 and compare on ext_dev.
- Rework the Science class; explicit off-topic class trained on real data.
- Re-embed with a newer Wikipedia snapshot when one is published on the Hub.
- Human-annotated conversational test set (the Stack Exchange titles are questions,
  not chat turns).
- More providers for key-less search; per-user search opt-out setting.
