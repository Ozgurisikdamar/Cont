# Decisions (ADR-lite)

Each entry: **Decision · Why · Consequence**. Numbers quoted here come from
files in `reports/` or `data/manifest/`; nothing is estimated.
Decisions that depend on measurements reference the report that justifies them.

---

## D-1 · The whole system is English
**Decision.** The project brief was written for Turkish; this implementation is
English end to end: data, taxonomy names, input, output, commands
(`exit/q`, `history`, `reset`), documentation.
**Why.** Owner's requirement ("tek fark İngilizce").
**Consequence.** Preprocessing is English-specific (stop words, abbreviations in
the sentence splitter). Non-English input is expected to be *uncertain*
(tested in `tests/test_acceptance.py`).

## D-2 · Build the training corpus from Wikipedia instead of reusing a labelled dataset
**Decision.** Train on Wikipedia passages labelled through Wikipedia's own
category graph.
**Why.** Measured in `reports/dataset_research.json`: the best existing dataset
covers 3/8 general topics and 6/28 subtopics (DBPedia Classes); the best reported
one 5/8 and 14/28 (arXiv, research abstracts only, no sports/books/history).
Wikipedia covers 8/8 and 28/28, is English, CC BY-SA and reproducible from a
pinned dump. Full comparison: `docs/DATASET_RESEARCH.md`.
**Consequence.** Labels are *distant supervision* (noisy) and the text style is
encyclopedic; both are measured (D-3, `docs/ERROR_ANALYSIS.md`).

## D-3 · Real user questions (Stack Exchange) for evaluation only
**Decision.** Stack Exchange question titles are never trained on. They form
`ext_dev` (model selection) and `ext_test` (report only), split 50/50 by a
SHA-1 hash of each item.
**Why.** Runtime input is short, conversational text; a model chosen only on
Wikipedia passages would be chosen for the wrong distribution. Keeping the set
out of training makes it an honest estimate of transfer.
**Consequence.** Two headline numbers are reported: in-domain (Wikipedia test)
and style-shifted (Stack Exchange ext_test).

## D-4 · Labels from the category graph via DBpedia, with curated seeds
**Decision.** Seeds are `Category@depth` per subtopic; the tree is crawled with
exclusion patterns; general topic = minimum crawl depth, ties are *excluded*
(831 ambiguous articles); secondary subtopics only at depth ≤ 1; at most 450
articles per subtopic, shallow first.
**Why.** Deep category crawls drift (a periodic-table crawl reached songs, a
classical-mechanics crawl reached acoustics and instruments). Excluding ties
instead of guessing keeps the label noise down where the graph itself is
ambiguous.
**Consequence.** 132,746 labelled articles → 11,937 selected. Seeds and
exclusions live in `configs/taxonomy.json` and are documented in `docs/TAXONOMY.md`.

## D-5 · Article text from two pinned dumps, not from the live API
**Decision.** Text comes from `wikimedia/wikipedia` 20231101.en (pinned
revision); titles it lacks are looked up by redirect alias, then in
`legacy-datasets/wikipedia` 20220301.en (pinned), exact title then redirect alias.
**Why.** (1) The live MediaWiki API returns HTTP 429 from shared cloud IPs and
is not reproducible. (2) The 2023 dump silently lacks some articles — measured
by scanning all 41 shards: "Gold", "Neon", "Silicon", "Spacetime",
"Quantum chromodynamics" are absent under any capitalisation; case-variant hits
("GOLD", "TIN", "XENON") are *different pages*, so case-insensitive matching was
rejected. (3) DBpedia reflects a newer Wikipedia, so renamed articles exist in the
dumps under an old title that now redirects. A REST-API backfill was tried and
dropped: uncached pages returned 429 with a 30 s `Retry-After`.
**Consequence.** 11,169 of the 11,937 selected in-taxonomy articles have text:
10,270 exact, 43 by redirect (2023), 846 exact + 10 by redirect (2022). The
remaining 768 are mostly articles created after March 2022; they are listed in
`data/manifest/crawl_stats.json → missing_text`. Every article's source is in
`articles.csv → text_source`.

## D-6 · Passages, not whole articles
**Decision.** Lead-first passages of 12–60 words, at most 4 per article, stop
sections (References, See also, …) skipped, exact duplicates removed.
**Why.** User messages are one or two sentences; the lead of an article is its
most topical part; capping passages per article stops long articles from
dominating a class.
**Consequence.** Median passage 23 words (Stack Exchange titles: 10).

## D-7 · Grouped, stratified split plus a near-duplicate filter
**Decision.** Split by *article* (70/15/15), stratified by (general topic, first
subtopic), deterministic (titles ordered by SHA-1 within each stratum). Validation/test passages with a
TF-IDF cosine ≥ 0.9 to any training passage are dropped.
**Why.** Passages of one article must not be on both sides. Sibling articles
share templated sentences ("A period 2 element is one of the chemical elements in
the second row …"): the EDA found 5 such test and 7 such validation passages
before the filter.
**Consequence.** Zero article overlap and zero exact-text overlap across splits
(`reports/eda.json → leakage`).

## D-8 · Out-of-taxonomy evaluation data
**Decision.** Wikipedia passages from 12 unrelated categories (6 validation, 6
disjoint test) and titles from 15 unrelated Stack Exchange sites.
**Why.** "Uncertain" must be measured, not assumed; disjoint categories stop the
OOD threshold from being tuned to the test domain.
**Consequence.** OOD detection is reported with AUROC/FPR@95 and with the
actual flag rate at the deployed threshold.

## D-9 · Selection protocol
**Decision.** Hyper-parameters on Wikipedia validation; model families on
validation **and** Stack Exchange ext_dev; `test` / `ext_test` only for the
final report.
**Why.** Prevents optimistic numbers; ext_dev keeps the choice honest for
conversational input.

## D-10 · Theme composition from taxonomy metadata
**Decision.** Topics carry `role` (domain / format), `broader` and an optional
`perspective_template`; four templates in `taxonomy.json` compose any
combination ("science books", "science books about biology", "history of
physics", "biology (science)", "sports and technology").
**Why.** A table of topic pairs grows quadratically and silently misses
combinations; metadata generalises (every rule is a test in
`tests/test_composer.py`).

## D-11 · Uncertain and uninformative messages
**Decision.** *Uninformative* (empty, symbols, only stop words, only a URL) is
not a turn: nothing is stored. *Uncertain* (low confidence or far from every
training topic) is stored and ages the context (decay applies) but adds no topic
mass (weight 0) and no search keywords.
**Why.** "I'm going to order pizza tonight" must not steer a conversation about
biology, but the user did say it.

## D-12 · Search query privacy
**Decision.** The query is the theme phrase plus at most two words from the
latest message, and only words that occur in the public training vocabulary
(IDF table in the artifact). Names, typos and other personal tokens are never
sent. Component concepts of the theme are last-resort queries (at most 4 queries
per turn).
**Why.** Privacy by construction; entity-only providers (DuckDuckGo Instant
Answers) cannot answer composed phrases such as "science books about biology".

## D-13 · Search providers, caching, circuit breaker
**Decision.** Wikipedia search API first, DuckDuckGo Instant Answer API second,
no API keys; responses cached in SQLite for 72 h; a provider that fails is
skipped for 5 minutes; timeouts, 429 (`Retry-After`, capped at 30 s) and 5xx are
retried with exponential back-off; URLs must be HTTPS on an allow-listed domain.
**Why.** Robustness without keys; a rate-limited provider must not add seconds
to every turn.

## D-14 · SQLite design
**Decision.** Tables `model_versions`, `sessions`, `texts`,
`conversation_topics`, `search_results`, `search_cache`; foreign keys on, WAL,
5 s busy timeout, `PRAGMA user_version` migrations, parameterised SQL, one
transaction per write. `reset` ends a session and starts a new one; nothing is
deleted.
**Why.** The brief asks for a history that survives resets; sessions make a
reset an explicit, queryable event.

## D-15 · Artifact format
**Decision.** Heads in `skops` loaded with an explicit trusted-type allowlist;
centroids as `.npy` without pickle; SHA-256 of every file in `metadata.json`,
verified at load; encoder weights copied next to the heads (git-ignored).
**Why.** Pickle executes code on load; checksums catch partial copies and edits.

## D-16 · Reproducibility
**Decision.** `RANDOM_SEED = 42` everywhere; every external artifact pinned by
revision (datasets, encoders, NLI model); every HTTP response of the data build
cached on disk; the label manifest and passages committed.
**Why.** The public endpoints change; the committed manifest is the ground truth
for this version.

## D-17 · No synthetic data
**Decision.** No generated, paraphrased or hand-written training examples.
**Why.** The brief's zero-fabrication rule, and hand-written examples would
leak the evaluator's intuition into the model.
