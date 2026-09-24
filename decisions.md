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
verified at load; encoder weights copied next to the heads (committed since D-26) and checked by fingerprint (D-18).
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

## D-18 · The encoder is verified, not trusted
**Decision.** `metadata.json` stores the embedding of a fixed probe sentence
produced at training time; `load_artifact` re-encodes the probe and refuses to
run if the cosine to the stored vector is below 0.999. A fine-tuned encoder
(`"local"` in the encoder registry) never falls back to the Hugging Face Hub.
**Why.** A missing or different encoder raises no error by itself: the heads
would silently receive embeddings they were never trained on. The first version
of the loader did exactly that — when the artifact's encoder copy was absent it
loaded the *base* model from the Hub.
**Alternatives.** Hash the weight files (breaks when the same weights are saved
in another format or dtype); trust the metadata `encoder` key (does not detect
the fallback).
**Consequence.** One extra forward pass at start-up. Tests:
`tests/test_integration.py::test_artifact_refuses_an_encoder_it_was_not_trained_with`,
`tests/test_encoders.py`.

## D-19 · Fine-tuned encoder weights are stored in float16
**Decision.** `scripts/finetune_transformer.py --export` writes float16 weights;
they load back as float32. `SentenceEncoder.save` keeps float16 only when the
weights are exactly representable (so a Hub encoder is never rounded).
**Why.** Halves the file (about 45 MB instead of 91 MB) so the trained model can
live in the repository. Measured on the pinned base MiniLM
(`reports/fp16_storage.json`, 300 validation passages): embeddings after the
float16 round trip have cosine ≥ 0.9999993 to the float32 ones. Every later step
(benchmark rows `minilm-l6-ft`, `train.py`, `evaluate.py`) uses exactly the
rounded weights, so reported numbers describe the shipped model.
**Alternatives.** float32 (twice the size, no measurable benefit); int8
quantisation (needs a different runtime, accuracy not measured).

## D-20 · Embedding cache keyed by encoder identity
**Decision.** `cached_encode` stores vectors under `<encoder>@<revision>` for Hub
encoders and `<encoder>@<hash of the weight files>` for local ones.
**Why.** The first key was the encoder name only; re-exporting the fine-tuned
encoder would have reused stale vectors without any error.

## D-21 · The artifact's `min_confidence` is the default at run time
**Decision.** `Settings.min_confidence` defaults to `None` = use the value tuned
in the benchmark and stored in the artifact; `CONTEXTLENS_MIN_CONFIDENCE`
overrides it.
**Why.** The console used to override the artifact with a hard-coded 0.40, so the
application could behave differently from what `evaluate.py` measured.

## D-22 · Encoder: all-MiniLM-L6-v2 fine-tuned on the training split
**Decision.** The production encoder is MiniLM-L6 fine-tuned for 3 epochs with a
general + subtopic head (`scripts/finetune_transformer.py`), exported without
its heads and used like a frozen encoder under the production heads.
**Why.** Best general-topic macro-F1 on both selection sets (Wikipedia val
0.843, Stack Exchange ext_dev 0.752 vs 0.836 / 0.743 for the best frozen
encoders), and the smallest and fastest candidate (13 ms per text, 91 MB).
Owner's instruction at the end of the benchmark: go with MiniLM.
**Alternatives.** Frozen e5-small / bge-small (−1 to −2 points, 2× latency);
mpnet-base (best frozen on Wikipedia, not on questions, 5× latency); TF-IDF
(does not transfer to questions); zero-shot NLI (0.51 on questions, ~2 s per
text). Table: docs/MODEL_REPORT.md §1.
**Consequence.** The encoder is not on the Hub; its weights ship with the
artifact (D-26). The fine-tuned model's own subtopic head was weak (flat BCE),
hence the exported encoder + production heads.

## D-23 · Head: one softmax over the 28 subtopics
**Decision.** `head_type = flat_softmax`: P(general) = sum of the
subtopics' probabilities; P(subtopic | general) = share within the parent.
**Why.** Won E-7 for every embedding encoder on val **and** on the Stack
Exchange subtopic set, for subtopics and for the general topic
(docs/MODEL_REPORT.md §3).
**Alternatives.** H1 hierarchical (separate general head + one multi-label head
per general topic, the original design, kept as `head_type = hierarchical`);
H2 flat multi-label (worst; over-confident sigmoids).

## D-24 · Out-of-taxonomy gate: centroid cosine, calibrated on real questions
**Decision.** A message is *uncertain* when its maximum cosine to the 8 class
centroids is below the value that keeps 95% of Stack Exchange ext_dev
in-domain questions.
**Why.** Best AUROC on questions among the scores that need no training data
at run time; a Wikipedia-calibrated threshold answered only 66–92% of genuine
questions (E-9).

## D-25 · Minimum confidence chosen by a coverage rule
**Decision.** `min_confidence` = the largest value in {0.30 … 0.60} that still
answers ≥ 90% of ext_dev in-domain questions (computed in `train.py`, stored in
the artifact with the whole sweep).
**Why.** The confidence floor trades wrong answers for "uncertain" answers; a
fixed number has no justification, a coverage target does and is tuned on
development data only.

## D-26 · The trained model is committed
**Decision.** `models/contextlens-topic/` (heads, centroids, vocabulary,
float16 encoder, metadata with checksums) is in the repository.
**Why.** The encoder is fine-tuned (not downloadable), and rebuilding it costs
~50 minutes of CPU; with the artifact committed a clone runs immediately and the
shipped model is exactly the evaluated one.
**Alternatives.** Git LFS or a release asset (extra tooling for users);
download from the Hub (only possible for frozen encoders).

## D-27 · Fix the label rule behind the failed quantum test, not the test
**Decision.** Taxonomy 1.1.0 drops the seed `Quantum_information_science@0`
from *quantum_computing*; `scripts/relabel_corpus.py` re-labels the corpus
(136 of 11,154 articles: 49 Technology → Physics, 6 → Hardware, 81 removed),
keeping every article's split; the encoder is fine-tuned again and the model
retrained on v1.1.
**Why.** The first production model answered Technology > Quantum Computing for
the entanglement sentence of the brief. Error analysis showed the training data
said so: "Quantum entanglement", "Bell's theorem" and ~50 similar articles were
labelled Technology because that category holds them at depth 0.
**Alternatives.** Adding the sentence or paraphrases to training (forbidden by
the brief, and it would hide the real cause); a keyword rule for "entanglement"
(hard-coding); tie-breaking by hand for single articles (not reproducible).
**Consequence.** Quantum computing is now the smallest subtopic (539
passages); the benchmark tables E-0 … E-9 were measured on v1.0 labels
(DATASET_CARD §9).

## D-28 · Conversation decay 0.7, theme share 0.2
**Decision.** `decay = 0.7`, `theme_min_share = 0.2` (defaults in `Settings`).
**Why.** `scripts/tune_decay.py` on simulated conversations from the
validation split (400 conversations per setting, grid decay × share): best
theme accuracy (0.757) among the settings that keep a Books → Science →
Biology style 3-topic theme ≥ 80% of the time (0.825), with 0.54 spurious
theme topics per turn and a 1.3-turn lag after a topic switch
(`reports/experiments/decay.json`; test split: 0.760 / 0.80).
**Note.** The first probe drew any passage of the topic, so its ceiling was
the classifier's accuracy³ (~0.67) whatever the decay; it now draws correctly
and confidently classified passages — it measures the decay, not the model.

## D-29 · Language gate (superseded by D-30)
**Decision.** A text of ≥ 3 words of which < 40% are known English words
(training vocabulary + stop words) is answered *uncertain*.
**Why.** The centroid gate did not catch non-English text (a Turkish sentence
was classified with confidence). Threshold from development data: it flags
0.05% of Wikipedia validation passages and 0.10% of English Stack Exchange
ext_dev questions; Turkish, German, Spanish and French example sentences score
0.00–0.33.
**Alternatives.** A language-identification model (another dependency for a
one-language system); character n-gram heuristics (less transparent).

## D-30 · Language gate: fastText lid.176 + training lexicon, per-length thresholds (replaces D-29)
**Decision.** A text is *non-English* (new status `non_english`, reported as
"not English", never classified) when fastText lid.176's top language is not
English with probability ≥ c(n) **and** at least one of its words is outside
the lexicon of the Wikipedia training split (46,488 word types). c = 0.5 for
1–2 words, 0.3 for 3 and more. The check runs before the informativeness
check, so a text in another script is reported as non-English rather than
"nothing to analyse"; a text of English stop words only stays uninformative.
**Why.** D-29's gate skipped texts of fewer than 3 words and rejected only
39.6% of non-English dev texts. On the dev half of a Tatoeba set (18
languages, full sentences and 1/2/3-word cuts) plus 2,000 in-domain English
texts, `reports/experiments/language_gate.json`:

| words | English accepted (Tatoeba / in-domain) | non-English rejected, D-29 → D-30 |
|---|---|---|
| 1 | 0.998 / 0.993 | 0.000 → 0.505 |
| 2 | 0.997 / 0.991 | 0.000 → 0.755 |
| 3 | 0.996 / 0.991 | 0.790 → 0.932 |
| ≥ 4 | 0.999 / 1.000 | 0.866 → 0.986 |
| all | 0.997 / 0.994 | 0.396 → 0.788 |

Selection rule: per length bucket, highest balanced accuracy subject to ≥ 0.99
English acceptance on both English sets. Plain classifiers could not meet the
English constraint at every length (best balanced accuracy: fastText 0.938,
py3langid 0.867, lingua 0.850 — but in-domain one-word English acceptance
0.63 or lower): one or two words are often ambiguous ("Tom", "La", "Die"),
and rare English terms ("titration": French 0.998) are what a character
n-gram model gets wrong and a lexicon gets right. 0.015 ms per text, 0.9 MB.
**Alternatives.** lingua (96 MB, weaker on short text), py3langid (weak on
short text; its package pins numpy ≥ 2, which conflicts with the pinned 1.26),
a per-word dictionary vote (no source of foreign vocabulary without new data).
**Limits.** Single words are only half caught; "guten tag" (German 0.49) passes
the gate. The locked Tatoeba half is evaluated once after the model freeze.

## D-31 · Conversation context expires; the dominant topic switches only on agreement (amends D-28)
**Decision.** On top of the decay (0.7, share 0.2, unchanged): (1) the
context is cleared after **3** consecutive turns without a confident topical
message (uncertain, off-topic or non-English); (2) the **dominant** theme topic
changes only when **2** consecutive confident messages agree on the same new
topic (`switch_rule = "votes"`). A single tangent still enters the theme as a
secondary topic (a Books → Science → Biology run still composes "science books
about biology"), but it no longer takes over the conversation.
**Why.** The audit was right on both counts. Decay multiplies every score by the
same factor, so the *shares* — and the theme — survived any number of
uncertain turns (stale-theme rate after 10 uncertain turns: **1.00**). And 50%
of one-message tangents made the tangent topic the top of the theme. Grid over
decay × share × confirm × expiry × switch rule (360 settings) on validation
conversations (`reports/experiments/decay.json`, `scripts/tune_decay.py`):

| | v1.0 | selected |
|---|---:|---:|
| theme accuracy (dominant = segment topic) | 0.756 | 0.716 |
| switch lag (turns) | 1.34 | 1.96 |
| tangent robustness | 0.456 | 0.863 |
| false switch rate after a tangent | 0.498 | 0.118 |
| stale theme after 10 uncertain turns | 1.000 | 0.000 |
| premature expiry after 2 uncertain turns | 0.000 | 0.000 |
| 3-topic accumulation | 0.820 | 0.820 |

**Trade-off (not hidden).** Theme accuracy drops by 0.04 and a real switch is
followed about 0.6 turns later: one message cannot tell a tangent from a
switch, so the rule waits for the second. Theme accuracy alone always prefers
switching on every message, which is why the rule now also requires a false
switch rate ≤ 0.15. The "votes" rule beat waiting for the accumulated-score
leader (0.716 vs 0.655 theme accuracy at the same robustness).
**Alternatives.** An absolute mass floor (equivalent to the streak rule here,
since only uncertain turns lower the mass); time-based decay (the console has
no meaningful inter-message time); a KL/embedding change-point detector (needs
labelled conversations to tune).
**Test split.** No longer used by the tuning script; conversation metrics on
held-out data are reported once by the locked evaluation.

## D-32 · Science means the scientific enterprise itself; audit-driven label fixes (taxonomy 1.2.0)
**Decision.** The general topic *Science* covers science as an activity —
method, philosophy of science, research practice and publishing, misconduct,
the history of science *as such* — and no longer anything a science-history
category happens to contain. Taxonomy 1.2.0:
* `history_of_science`: seed `History_of_science@1` (was @2) with exclusions for
  per-discipline histories, natural history, naturalists, hoaxes, museums,
  libraries, expeditions, instruments, timelines, measurement, organisations and
  books;
* `scientific_research`: drops `Research_and_development@0` and
  `Academic_publishing@0` (business R&D and publishing-industry articles);
* audit fixes (categories located with the cached crawl, one per systematic
  error): global `recipients_of` (politicians reached through the Olympic Order),
  `relativity` excludes `…critics` and the category `Dimension` (algebraic
  geometry), `scientific_method` excludes `Research_methods` and
  `Scientific_observation` sub-categories (instruments, cell-biology
  techniques, Euler's laws), `world_wars` excludes
  `Science_and_technology_during_…` (radar articles), `ancient_history`
  excludes `Historians_of_…` (modern scholars).

**Why.** Root cause of the weak Science class (dev data only: val recall 0.688;
ext_dev hsm recall 0.324, its false negatives going to Physics): the depth-2
crawl under `History_of_science` pulled in the history of physics, biology,
chemistry, astronomy and mathematics, so "Science" was trained as a mixture of
every other class. The 310-passage stratified audit
(`reports/label_audit_v2.json`, `scripts/label_audit_v2.py`) measured the rest:
correct 74.5% [69.4, 79.1], weak 20.3% [16.2, 25.2], incorrect **5.2% [3.2,
8.2]** (Wilson 95%); Science was the worst general topic (8/45 incorrect,
[9.3%, 31.3%]). Each incorrect item was traced to the category that brought it
in; the fixes above remove 7 of the 16. The 9 left are single articles filed
directly in a seed category (Supersymmetry, "Ice age" and an ornithological
collection sit in `Category:History_of_science` itself; a judicial institute in
`Category:Research`) and cannot be removed by a category rule without dropping
the seed. `Research_methods@0` was kept for *scientific_research*: removing it
also removed literature reviews, cross-sectional studies, grounded theory and
research synthesis (68 articles, most of them in scope), which is worse than the
two instruments it contained.

**Effect on the corpus** (`reports/relabel_taxonomy_1.2.0.json`, relabelled
from the committed 1.1 corpus, splits kept): 883 articles changed, 730 removed
(Science 313, Sports 207, History 125, Physics 85), 10,358 articles / 40,112
passages left; Science keeps 2,388 / 463 / 465 train / val / test passages.

**Evaluation rule for Stack Exchange.** The general external set labels every
`hsm` question *Science*, but hsm is "history of science **and mathematics**"
and most of its questions ask about a specific field. It is therefore reported
twice from now on: the legacy number (all sites) and the number without hsm;
Science on questions is measured on the subtopic set (hsm tagged
scientific-method / philosophy-of-science, philosophy scientific-method,
academia research-process). Test sets are not re-cut by this rule — the same
rule applies to dev and test.
**Not done.** The auditor is the agent that built the corpus, not an
independent annotator; the audit is a lower bound on disagreement.

## D-35 · CI: one workflow, manual trigger on this account; the same steps locally
**Decision.** `.github/workflows/ci.yml` has two jobs — *fast* (ruff check,
ruff format --check, mypy, the offline pytest suite with `-m "not model"`) and
*model* (the tests that need the committed artifact) — and is triggered by
`workflow_dispatch` only. `scripts/ci.sh` runs exactly the same steps locally
(`bash scripts/ci.sh fast` for the first job). Network tests stay out of CI
(`network` / `network_live`, run before releases).
**Why not push / pull_request.** GitHub Actions on this account ends every run
in `startup_failure` before a runner starts (billing; measured on the
account's other repositories, 55 of 55 runs). An automatic trigger would mark
every commit red without running a single step, which is worse than no badge:
it trains readers to ignore red. The trigger lines are in the workflow as a
comment; enabling them is a two-line change once Actions minutes exist.
**Consequence (not hidden).** The audit asked for CI on every push; on this
account that is not achievable today. What is guaranteed is that the CI
definition exists, is correct, and passes locally (`reports/tests/`).
