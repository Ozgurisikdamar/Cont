# ContextLens — final report (v1.1.0)

Every number in this report is copied from a file under `reports/`
(`locked/results.json`, `evaluation_dev.json`, `experiments/*.json`,
`eda.json`, `label_audit_v2.json`, `hardware.json`, `tests/`). Detailed
documents are linked per section. v1.1.0 is the result of a hardening pass over
v1.0.0; what changed and why is in [HARDENING.md](HARDENING.md) and
decisions D-30 … D-36.

## 1. Executive summary

ContextLens is a local console application that, for every English message,
predicts one of **8 general topics** and a **primary subtopic** out of 28 (with
sibling suggestions) with calibrated confidence, answers *uncertain* when a
message is off-topic and *non_english* when it is not English, keeps a decayed
**conversation theme** that ignores one-message tangents and expires after
off-topic runs ("science books about biology"), turns the theme into a
privacy-preserving **web-search query** (Wikipedia, DuckDuckGo fallback, no
keys) and stores everything in **SQLite**.

No existing labelled dataset covered the taxonomy, so a corpus of **40,112
Wikipedia passages** was built with labels from Wikipedia's category graph
(audited on 310 passages: 5.2% wrong). Real **Stack Exchange questions**,
**CLINC150** assistant chat and **Tatoeba** sentences measure how the model
handles user-style, off-topic and non-English text. The model is a
**fine-tuned all-MiniLM-L6-v2** with one softmax over the 28 subtopics. On a
**locked holdout** built before any v1.1 decision and evaluated once after a
fingerprinted freeze, it reaches macro-F1 **0.838** on unseen Wikipedia
articles and **0.730** on 2026 questions (**0.818** without the
history-of-science site), median **19.9 ms** per message on a 4-vCPU CPU,
47 MB on disk. All acceptance cases pass; 230 tests pass.

## 2. Problem definition

Input: a stream of free-text English messages from one user. Output per
message: general topic + confidence, primary subtopic (+ sibling suggestions)
with probabilities, or *uncertain* / *non_english* / *uninformative*; per
conversation: the current theme, a natural-language phrase for it, a search
query and up to three web results. Topics can combine (Books + Science →
"science books"), so the theme is not a single-message decision.

## 3. Requirements

From the brief: English only; 8 general topics and 28 subtopics; subtopics
under their general topic; calibrated confidence and an explicit uncertain
answer; context across messages with decay and a `reset` that keeps stored
data; composed themes; web search without API keys; SQLite persistence; no
training at start-up; benchmarked model choice with justified data; zero
fabricated numbers; full test and documentation set. The hardening brief
added: an honest multi-label decision, an untouched final holdout, stronger
off-topic and language gates, context expiry, CI, a larger label audit,
encoder checksums. Derived data requirements: docs/DATASET_RESEARCH.md §1.

## 4. Dataset research

Nine candidates were assessed, five of them downloaded and measured at pinned
revisions (`reports/dataset_research.json`). The best measured one (DBPedia
Classes) covers 3 of 8 general topics and 6 of 28 subtopics; the best reported
one (arXiv) 5/8 and 14/28, without sports, books or history. Merging datasets
would mix text styles class by class, letting a model learn the source instead
of the topic. Stack Exchange covers 8/8 and 25/28 but holds question titles
only → used for evaluation. Details: docs/DATASET_RESEARCH.md.

## 5. Selected data

| part | size | source / use |
|---|---|---|
| training corpus (taxonomy 1.2.0) | 40,112 passages / 10,344 articles: train 28,075 · val 5,984 · legacy test 6,053 | English Wikipedia, two pinned dumps; labels from the DBpedia category graph |
| development | Stack Exchange ext_dev 10,124 in-taxonomy + 6,000 off-topic, 1,630 subtopic questions; CLINC150 dev 2,900; Tatoeba dev half; Wikipedia OOD val 652 | every threshold and model choice |
| locked holdout | 374 Wikipedia passages (unseen articles), 2,713 Stack Exchange questions created in 2026, CLINC150 test 4,350, Tatoeba locked half | read once by `evaluate.py --stage locked` |
| off-topic training | CLINC150 train half | only for the two detectors that need off-topic examples (not chosen) |

Licences: CC BY-SA (Wikipedia, Stack Exchange), CC BY 3.0 (CLINC150), CC BY
2.0 FR (Tatoeba). Card: docs/DATASET_CARD.md.

## 6. Exploratory data analysis

`scripts/eda.py` → `reports/eda.json`, figures `reports/figures/eda_*.png`.

* Passage length: min 12 · median 23 · mean 25.5 · p95 47 · max 60 words;
  Stack Exchange titles median 10 words — the style gap.
* Class balance: general imbalance ratio 1.97, subtopic 3.67.
* Labels per passage: 1 → 37,958, 2 → 2,099, 3 → 55 (5.4% multi-label).
* Label depth: 0 → 16,939, 1 → 21,101, 2 → 2,072. Training vocabulary 44,777.
* Label audit (310 passages, stratified): 74.5% correct, 20.3% weak, **5.2%
  wrong [3.2, 8.2]**; Science worst (8 of 45 wrong) — traced to its seeds
  (`reports/label_audit_v2.json`).

## 7. Data cleaning

* Back-matter sections, headings and lines under 6 words dropped at extraction;
  exact duplicates across articles removed; validation/test passages with
  TF-IDF cosine ≥ 0.9 to a training passage removed.
* Split grouped by article and stratified by label; the relabels of taxonomy
  1.1 and 1.2 keep every article in its split.
* Leakage checks all 0: article overlap, exact text overlap, near-duplicates
  (max cosine test→train 0.870), OOD / Stack Exchange titles in the corpus,
  acceptance sentences in the corpus.
* Label fixes: taxonomy 1.1 (quantum seed, 136 articles, D-27); taxonomy 1.2
  (Science = scientific enterprise, audit exclusions; 883 passages relabelled,
  730 removed, D-32).

## 8. NLP preprocessing

`contextlens.preprocessing.text.normalize`: Unicode NFKC, HTML unescape and tag
removal, URLs / e-mail / @mentions removed, control characters removed,
whitespace collapsed, at most 5,000 characters. Gate order: no content words →
*uninformative*; language gate confident it is not English → *non_english*;
English stop words only → *uninformative*; otherwise classified. Tokenisation
belongs to the encoder (uncased WordPiece; 64 tokens during fine-tuning, 128 at
inference). A training vocabulary weighted by IDF × topic concentration drives
the query builder; the training lexicon backs up the language identifier.

## 9. Taxonomy

8 general topics (Physics, Biology, Chemistry, Technology, Science, Books,
Sports, History) × 28 subtopics, defined only in `configs/taxonomy.json` with
composition metadata (role: *format* for Books, *perspective* for History,
*field* for the sciences). 1.1.0 removed the `Quantum_information_science`
seed; 1.2.0 redefined Science as the scientific enterprise (method, research
practice, history of science as a discipline) and added audit exclusions.
docs/TAXONOMY.md.

## 10. Models considered

Majority baseline; TF-IDF (word, word without stop words, word + char) × LR /
multinomial NB / complement NB / linear SVM + Platt; zero-shot label similarity;
zero-shot NLI (bart-large-mnli); frozen sentence encoders (all-MiniLM-L6-v2,
bge-small-en-v1.5, e5-small-v2, all-mpnet-base-v2) × LR; fine-tuned MiniLM with
two heads; subtopic heads (hierarchical, flat multi-label, flat softmax — and,
on the production encoder, softmax vs one-vs-rest sigmoid vs per-parent
sigmoid); seven off-topic detectors; three language-gate variants. A
two-encoder ensemble (E-6) was not run.

## 11. Experiments

Registry: docs/EXPERIMENTS.md; one card per model: `reports/experiment_log.md`;
tables: `reports/tables.md`. Selection on development data only (Wikipedia
val, Stack Exchange ext_dev, CLINC150 / Tatoeba dev halves).

| model (taxonomy 1.2.0) | Wikipedia val | questions (ext_dev) | ms / text | MB |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.805 | 0.593 | 1.5 | 10 |
| TF-IDF + complement naive Bayes | 0.813 | 0.636 | 1.5 | 17 |
| zero-shot NLI (bart-large-mnli, v1.0 sample) | 0.623 | 0.511 | 2,556 | – |
| MiniLM (frozen) + LR | 0.830 | 0.712 | 14.1 | 91 |
| mpnet-base (frozen) + LR | 0.849 | 0.713 | 66.3 | 438 |
| bge-small (frozen) + LR | 0.838 | 0.736 | 27.8 | 133 |
| e5-small (frozen) + LR | 0.842 | 0.747 | 25.3 | 133 |
| **MiniLM fine-tuned** | **0.848** | **0.749** | **13.5** | **91** |

Findings: lexical models memorise and lose ~20 points on questions; the
biggest frozen encoder wins on Wikipedia but not on questions; fine-tuning the
smallest encoder is best on questions and ties the best on Wikipedia at the
lowest cost. The benchmark was rerun on 1.2.0 after the freeze (development
splits only); it confirms the frozen choice with narrower margins than on v1.0
(MODEL_REPORT §1). Flat softmax over 28 subtopics was best for every encoder
(E-7); temperature scaling lowered ECE on questions for every model (E-8).

Hardening experiments on the production encoder:

| experiment | outcome | decision |
|---|---|---|
| subtopic heads (E-15) | softmax val sub macro-F1 0.660 vs one-vs-rest 0.649, per-parent 0.657; every sigmoid optimum predicts one label | softmax, objective "primary + sibling suggestions" (D-33) |
| off-topic detectors (E-16) | Mahalanobis: AUROC 0.904 / 0.904 / 0.958 (Wikipedia / SE / chat), recall at 95% question retention 0.24 / 0.56 / 0.79; centroid (v1.0) 0.27 / 0.45 / 0.58 | Mahalanobis (D-34) |
| language gate (E-13) | fastText + lexicon, reject confidence 0.5 (1–2 words) / 0.3 (3+) | deployed (D-30) |
| tracker (E-14) | 360 settings; expiry 4 + 2 agreeing turns: false switches 0.488 → 0.104, stale theme 1.0 → 0.0 | deployed (D-31) |

## 12. Hyperparameter search

| parameter | grid | chosen on | value |
|---|---|---|---|
| LR C (embeddings, benchmark) | 0.5, 2, 8, 32 | val | per encoder |
| production head C | 1, 4, 8, 16 | val | 16 |
| fine-tuning | lr 5e-5, 3 epochs, batch 32, 64 tokens, warm-up 6% | val per epoch | 0.843 → 0.845 → 0.848 |
| sibling threshold τ | 0.20 … 0.95 | val subtopic macro-F1 | 0.40 |
| temperature | NLL on val (general topic) | val | 1.428 |
| off-topic threshold | keep 95% of ext_dev questions | ext_dev | 5th percentile of the Mahalanobis score |
| min_confidence | 0.30 … 0.60, largest with coverage ≥ 90% | ext_dev | 0.50 (coverage 0.926) |
| language reject confidence | per length bucket | Tatoeba dev + in-domain English | 0.5 / 0.5 / 0.3 / 0.3 |
| tracker | decay × share × confirm × expiry × switch rule | simulated val conversations | 0.7 × 0.20 × 2 × 4 × votes |

## 13. Final model

Fine-tuned all-MiniLM-L6-v2 (3 epochs, ~40 min on 4 vCPU, float16 export) →
L2-normalised 384-d embedding → multinomial LR over 28 subtopics (C = 16,
class-weighted, trained on the primary subtopic), T = 1.428, P(general) = sum
of its subtopics. Output: the primary subtopic of the predicted general topic
+ siblings with P(sub | general) ≥ 0.40, at most 3. Uncertain if the
Mahalanobis score is below the ext_dev 5th percentile or confidence < 0.50;
*non_english* from the language gate. Artifact: 47 MB, skops heads and
detector with a type allow-list, SHA-256 checksums of every artifact and
encoder file, encoder fingerprint. docs/MODEL_CARD.md.

## 14. Evaluation — locked holdout

`scripts/freeze.py` recorded the SHA-256 of the configuration, taxonomy,
settings, passages, locked manifest and artifact metadata
(`reports/locked/FREEZE.json`); `evaluate.py --stage locked` checked the
fingerprint and ran once. It ran twice in total: run 1 exposed a
data-partition bug in the locked Stack Exchange view (questions belonging to
both views were dropped from one); the data was repaired offline, the state
re-frozen and run 2 recorded with its reason. Run 1 is kept
(`results_run1_partition_bug.json`); only the Stack Exchange general view
differs (D-36).

| set | n | accuracy | macro-F1 | weighted-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Wikipedia, unseen articles | 374 | 0.848 | 0.838 | 0.847 | 0.058 |
| Stack Exchange questions (2026) | 1,623 | 0.804 | 0.730 | 0.796 | 0.056 |
| … without hsm (7 classes) | 1,524 | 0.843 | 0.818 | – | – |
| Stack Exchange subtopic questions (general view) | 1,212 | 0.823 | 0.772 | 0.810 | 0.050 |

| subtopics | micro-F1 | macro-F1 (supported) | samples-F1 | Hamming | subset acc | P@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wikipedia | 0.659 | 0.664 (25) | 0.667 | 0.025 | 0.626 | 0.668 | 0.853 |
| … general topic right | 0.782 | 0.784 | 0.787 | 0.016 | 0.738 | 0.789 | 0.966 |
| questions | 0.603 | 0.507 (24) | 0.606 | 0.030 | 0.557 | 0.624 | 0.830 |

| gate (locked) | result |
|---|---|
| in-domain flagged uncertain | 4.0% (Wikipedia), 9.3% (questions); accuracy of the answered 0.861 / 0.844, of the flagged 0.53 / 0.40 |
| off-topic flagged | 78.1% of CLINC150 chat (AUROC 0.961, FPR@95TPR 0.20), 48.5% of off-topic questions (AUROC 0.887) |
| language (Tatoeba) | non-English rejected 48% / 76% / 93% / 97% (1 / 2 / 3 words / sentence); English accepted ≥ 99.6% |

Legacy (v1.0 test splits, seen during v1.0 development): Wikipedia 0.847 /
0.839, Stack Exchange 0.773 / 0.746 (accuracy / macro-F1). Development
numbers: `reports/evaluation_dev.json`. Reliability diagrams and confusion
matrices: `reports/figures/*_locked.png`.

## 15. Confusion matrix

Largest off-diagonal cells (share of the true class, locked):

| Wikipedia | | questions | |
|---|---:|---|---:|
| Science → Books | 13.8% | Science → Physics | 42.4% |
| Biology → History | 8.3% | Science → Books | 15.2% |
| Biology → Science | 6.7% | Science → Technology | 14.1% |
| Biology → Chemistry | 6.7% | History → Books | 10.1% |
| Physics → Technology | 6.7% | History → Science | 10.1% |

## 16. Error analysis

Per-class F1 (Wikipedia / questions): Sports 0.933 / 0.765, Books 0.884 /
0.760, Technology 0.876 / 0.847, History 0.870 / 0.749, Physics 0.867 / 0.866,
Chemistry 0.831 / 0.831, Biology 0.789 / 0.785, **Science 0.656 / 0.241**. All
Science questions come from the history-of-science site, which asks about the
history of one field (accuracy 0.19). Confident errors are arguable labels,
physics in everyday settings (billiards → Sports) and site-defined labels.
Weak Wikipedia subtopics: classical mechanics 0.33, evolution 0.42, cell
biology 0.43. docs/ERROR_ANALYSIS.md.

## 17. Conversation tracking

Each certain message adds its calibrated general-topic probabilities to a
score vector that decays by 0.7 per message; uncertain, non-English and
uninformative messages add nothing. Topics holding ≥ 20% of the total form the
theme. The dominant topic changes only when 2 consecutive confident messages
agree on a new one, and the context is cleared after 4 consecutive messages
without a confident topic (the CLI says so). Simulated validation
conversations: theme accuracy 0.729 (v1.0 tracker 0.767), switch lag 1.92
(1.34), tangent robustness 0.874 (0.460), false switch rate 0.104 (0.488),
stale-theme rate 0.00 (1.00), three-topic accumulation 0.80 (0.80). The
composer turns the theme into a phrase from taxonomy metadata; `reset` clears
the tracker and keeps all stored rows.

## 18. Web search

Query = theme phrase + at most two salient words that exist in the public
training vocabulary (never names, numbers or e-mail addresses). Providers:
Wikipedia search API, then DuckDuckGo Instant Answer; HTTPS-only URLs on
allow-listed domains; snippets cleaned and truncated; 6 s timeout, bounded
retries, `Retry-After` capped at 30 s, 5-minute circuit breaker per provider,
72 h cache in SQLite. `--no-web` disables all network access. Two test kinds:
resilience (a clean status whatever the providers do) and live smoke (fails
unless a provider returns a valid result).

## 19. Database design

SQLite, `PRAGMA user_version = 1`, foreign keys on, WAL, one transaction per
write, parameterised SQL. Tables: `model_versions`, `sessions`, `texts`,
`conversation_topics`, `search_results`, `search_cache`. A locked or
unwritable database is a warning, never a crashed turn. docs/api.md §5.

## 20. Software architecture

`project.py` (console) → `ConversationSession` (pipeline) → `TopicModel`
(language gate, encoder, subtopic softmax, off-topic detector, decision) →
`ConversationTracker` → `compose()` → `build_query()` → `WebSearcher` →
`Database`. Offline data pipeline in `contextlens/data` and `scripts/`;
training in `train.py`; evaluation in `evaluate.py` (dev / locked); freeze in
`scripts/freeze.py`. Each component fails in isolation. docs/architecture.md,
docs/api.md.

## 21. Testing

230 passed, 0 failed (227 offline + trained model, 1 network resilience, 2
live smoke); ruff and mypy clean. Acceptance: the brief's sentences, the
quantum pair, pizza → uncertain, the three-message conversation, tangent /
switch / expiry with the real model, reset, uninformative inputs, short
non-English inputs, markup / case, long input, latency. `scripts/ci.sh` run
while the benchmark used all cores failed the latency test once (median 0.51 s
vs 0.5 s); rerun on the idle machine, all checks passed (194 offline + 36 model tests; `reports/tests/ci_sh_idle.log`). docs/TEST_REPORT.md.

## 22. Performance

Reference machine (`reports/hardware.json`): Linux, Intel Xeon 2.8 GHz, 4 vCPU,
15.7 GB RAM, no GPU, Python 3.11.15, torch 2.5.1+cpu.

| measure | value |
|---|---:|
| single message, median / p95 | 19.9 ms / 30.3 ms |
| batch, per text | 5.7 ms |
| model load | 2.6 s |
| artifact size | 47 MB |
| fine-tuning (3 epochs) | 39.8 min |
| `train.py` (heads + detector, cached embeddings) | 41 s |

float16 storage changes embeddings by at most 7e-7 in cosine
(`reports/fp16_storage.json`, measured on the base encoder).

## 23. Security

Parameterised SQL with an allow-listed identifier; skops heads and detector
with a type allow-list and `allow_pickle=False`; SHA-256 checksums of every
artifact and encoder file (missing, changed or extra files are refused) and an
encoder fingerprint (no silent Hub fallback); bounded input; sanitised,
allow-listed web results; queries built only from public vocabulary; no
secrets; pinned dependencies. Residual risk: the local database is plain text.
docs/SECURITY_PRIVACY.md.

## 24. Limitations

Science on real questions (0.241 F1); partial off-topic detection, especially
for encyclopedia-style off-topic text; one-word non-English inputs; subtopics
are a primary label + suggestions, not multi-label predictions; label noise
(5.2% wrong); switches followed one turn later; free web endpoints can
rate-limit; CI has no automatic trigger; locked sets are small (14–60 per
class on Wikipedia); E-6 not run. KNOWN_ISSUES.md.

## 25. Future improvements

A human-annotated set of real chat messages (in- and off-topic) for training
the off-topic detector and testing; a second general topic for "history of a
field" questions; fine-tuning e5-small (E-6); more seeds for weak subtopics and
other histories; a change-point detector for faster theme switches; ONNX
export; database encryption.

## 26. Conclusion

The system meets the brief and the hardening brief with measured, disclosed
limits: it classifies messages into the 8 × 28 taxonomy with calibrated
confidence, recognises most assistant chat and short non-English text,
follows a conversation without being derailed by tangents, searches the web
without keys and stores everything locally. Three lessons: encyclopedia-only
validation would have chosen the wrong model (real questions changed the
ranking); a weak class was a label-definition problem, fixed in the labels
rather than in the metric; and a holdout is only as good as the process around
it — the one rerun the process needed is recorded with its reason and the
first run is kept next to it.
