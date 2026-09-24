# ContextLens — final report

Every number in this report is copied from a file under `reports/`
(`evaluation.json`, `experiments/*.json`, `eda.json`, `label_audit.json`,
`hardware.json`, `tests/`). Detailed documents are linked per section.

## 1. Executive summary

ContextLens is a local console application that, for every English message,
predicts one of **8 general topics** and **1–3 of 28 subtopics** with calibrated
confidence, answers *uncertain* when a message is off-topic or not English,
keeps a decayed **conversation theme** ("science books about biology"), turns
it into a privacy-preserving **web-search query** (Wikipedia, DuckDuckGo
fallback, no keys) and stores everything in **SQLite**.

No existing labelled dataset covered the taxonomy, so a corpus of **42,942
Wikipedia passages** was built with labels from Wikipedia's category graph;
real **Stack Exchange questions** were used as an external test of how the
model handles user-style text. Eleven model families were compared; the winner
is a **fine-tuned all-MiniLM-L6-v2** with a single softmax over the 28
subtopics: test macro-F1 **0.834** on Wikipedia and **0.756** on real questions,
median **19.9 ms** per message on a 4-vCPU CPU, 45 MB on disk. All acceptance
cases of the brief pass; 162 tests pass, 0 fail.

## 2. Problem definition

Input: a stream of free-text English messages from one user. Output per
message: general topic + confidence, subtopics + probabilities, or
*uncertain* / *uninformative*; per conversation: the current theme, a
natural-language phrase for it, a search query and up to three web results.
Topics can combine (Books + Science → "science books"), so the theme is not a
single-message decision.

## 3. Requirements

From the brief: English only; 8 general topics and 28 subtopics; multi-label
subtopics; calibrated confidence and an explicit uncertain answer; context
across messages with decay and a `reset` that keeps stored data; composed
themes; web search without API keys; SQLite persistence; no training at
start-up; benchmarked model choice with justified data; zero fabricated
numbers; full test and documentation set. Derived data requirements:
docs/DATASET_RESEARCH.md §1.

## 4. Dataset research

Nine candidates were assessed, five of them downloaded and measured at pinned
revisions (`reports/dataset_research.json`). The best measured one (DBPedia
Classes) covers 3 of 8 general topics and 6 of 28 subtopics; the best reported
one (arXiv) 5/8 and 14/28, without sports, books or history. Merging datasets
would mix text styles class by class, letting a model learn the source instead
of the topic. Stack Exchange covers 8/8 and 25/28 but holds question titles
only → used for evaluation. Details: docs/DATASET_RESEARCH.md.

## 5. Selected dataset

| part | size | source |
|---|---|---|
| training corpus (labels v1.1) | 42,942 passages / 11,073 articles: train 30,051 · val 6,464 · test 6,427 | English Wikipedia, two pinned dumps (20231101.en, fallback 20220301.en) |
| labels | general by minimum category depth from curated seeds; secondary subtopics at depth ≤ 1 | DBpedia category graph |
| external | 10,124 / 10,087 in-taxonomy questions (ext_dev / ext_test), 1,630 / 1,642 subtopic questions, 6,000 / 6,000 off-topic | Stack Exchange (11 sites + off-topic sites) |
| out-of-taxonomy | 1,327 passages from 12 categories (val 652, test 675) | Wikipedia |

Licences: CC BY-SA. Card: docs/DATASET_CARD.md.

## 6. Exploratory data analysis

`scripts/eda.py` → `reports/eda.json`, figures in `reports/figures/eda_*.png`.

* Passage length: min 12 · median 23 · mean 25.5 · p95 47 · max 60 words;
  Stack Exchange titles median 10 words — the style gap.
* Class balance: general imbalance ratio 1.589 (largest/smallest), subtopic
  3.805 (smallest: quantum computing, 539 passages).
* Labels per passage: 1 → 40,295, 2 → 2,596, 3 → 51.
* Label depth: 0 → 17,360, 1 → 23,642, 2 → 1,940.
* Training vocabulary: 46,819 tokens.
* Manual audit of 48 training passages: 89.6% correct, 6.2% weak passage,
  4.2% wrong label (`reports/label_audit.json`).

## 7. Data cleaning

* Back-matter sections, headings and lines under 6 words dropped at extraction.
* 38 exact duplicates across articles removed; 12 validation/test passages with
  TF-IDF cosine ≥ 0.9 to a training passage removed.
* Split grouped by article and stratified by label.
* Leakage checks all 0: article overlap between splits, exact text overlap,
  test→train near-duplicates ≥ 0.9 (max cosine 0.869), out-of-taxonomy and
  Stack Exchange titles in the corpus, acceptance sentences in the corpus.
* Residual markup: 9 HTML tags, 7 URLs, 15 LaTeX fragments in 42,942 passages —
  removed or harmless at load time.
* Label fix (taxonomy 1.1, D-27): 136 of 11,154 articles relabelled or removed.

## 8. NLP preprocessing

`contextlens.preprocessing.text.normalize`: Unicode NFKC, HTML unescape and tag
removal, URLs / e-mail / @mentions removed, control characters removed,
whitespace collapsed, at most 5,000 characters. Messages without content words
(empty, symbols, numbers, stop words only, a bare URL) are *uninformative* and
not classified. Tokenisation belongs to the encoder (uncased WordPiece; 64 tokens during
fine-tuning, 128 at inference). A training vocabulary weighted by IDF × topic
concentration drives the query builder and the language gate.

## 9. Taxonomy

8 general topics (Physics, Biology, Chemistry, Technology, Science, Books,
Sports, History) × 28 subtopics, defined only in `configs/taxonomy.json`
together with composition metadata (role: *format* for Books, *perspective* for
History, *field* for the sciences; broader/narrower links). Version 1.1.0
removed the `Quantum_information_science` seed. docs/TAXONOMY.md.

## 10. Models considered

Majority baseline; TF-IDF (word, word without stop words, word + char) × LR /
multinomial NB / complement NB / linear SVM + Platt; zero-shot label similarity
with four encoders; zero-shot NLI (bart-large-mnli); frozen sentence encoders
(all-MiniLM-L6-v2, bge-small-en-v1.5, e5-small-v2, all-mpnet-base-v2) × LR;
fine-tuned MiniLM with two heads; three subtopic designs (hierarchical, flat
multi-label, flat softmax). A two-encoder ensemble (E-6) was planned and not
run (the owner chose to proceed with MiniLM).

## 11. Experiments

Registry: docs/EXPERIMENTS.md; one card per model: `reports/experiment_log.md`.
Selection on Wikipedia val **and** Stack Exchange ext_dev; tests report only.

| model (labels v1.0) | val macro-F1 | ext_dev macro-F1 | ms / text | MB |
|---|---:|---:|---:|---:|
| majority | 0.033 | 0.021 | – | – |
| zero-shot NLI (sampled) | 0.623 | 0.511 | 2,556 | – |
| label similarity (mpnet) | 0.708 | 0.640 | – | – |
| TF-IDF + LR | 0.793 | 0.595 | 1.6 | 10.6 |
| TF-IDF + complement NB | 0.803 | 0.643 | 1.8 | 17.8 |
| TF-IDF word+char + LR | 0.800 | 0.618 | 4.7 | 21.1 |
| MiniLM frozen + LR | 0.813 | 0.679 | 14.9 | 90.9 |
| bge-small frozen + LR | 0.823 | 0.735 | 27.9 | 133.4 |
| e5-small frozen + LR | 0.823 | 0.743 | 27.1 | 133.4 |
| mpnet-base frozen + LR | 0.836 | 0.737 | 66.7 | 437.9 |
| **MiniLM fine-tuned** | **0.843** | **0.752** | **13.0** | **90.9** |

Findings: lexical models memorise (train macro-F1 0.95–1.00) and lose 15–20
points on questions; the biggest frozen encoder wins on Wikipedia but not on
questions; fine-tuning the smallest encoder wins on both. Flat softmax over
28 subtopics was best for every embedding encoder (E-7). Temperature scaling
never raised ECE (E-8). Centroid cosine is the best off-topic score on
questions, and its threshold must be calibrated on questions (E-9).

## 12. Hyperparameter search

| parameter | grid | chosen on | value |
|---|---|---|---|
| LR C (TF-IDF) | 1, 4, 16 | val | per featurizer (`general.json`) |
| LR C (embeddings) | 0.5, 2, 8, 32 | val | 8 |
| NB α | 0.01, 0.1, 0.5 | val | per model |
| SVM C | 0.1, 0.5, 2 | val | per featurizer |
| fine-tuning | lr 5e-5, 3 epochs, batch 32, 64 tokens, warm-up 6% | val per epoch | 0.8288 → 0.8349 → 0.8368 (v1.1) |
| sibling threshold τ | 0.20 … 0.70 | val subtopic macro-F1 | 0.40 |
| temperature | NLL on val (general topic) | val | 1.431 |
| OOD threshold | keep 95% of ext_dev questions | ext_dev | 0.705 |
| min_confidence | 0.30 … 0.60, largest with coverage ≥ 90% | ext_dev | 0.50 (coverage 0.932) |
| decay × theme share | decay grid × 0.10 … 0.25 | simulated val conversations | 0.7 × 0.20 |

## 13. Final model

Fine-tuned all-MiniLM-L6-v2 (3 epochs, 30.5 min on 4 vCPU, float16 export) →
L2-normalised 384-d embedding → multinomial LR over 28 subtopics (C = 8,
class-weighted), T = 1.431, P(general) = sum of its subtopics. Subtopics: best
child of the predicted topic + siblings with P(sub | general) ≥ 0.40, at most 3.
Uncertain if centroid cosine < 0.705, confidence < 0.50 or < 40% known English
words. Artifact: 45 MB, skops heads with a type allow-list, SHA-256 checksums,
encoder fingerprint. `train.py`: 217.8 s. docs/MODEL_CARD.md.

## 14. Evaluation

| set | n | accuracy | macro-F1 | weighted-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Wikipedia test | 6,427 | 0.835 | 0.834 | 0.835 | 0.031 |
| Stack Exchange ext_test | 10,087 | 0.775 | 0.756 | 0.773 | 0.067 |
| Stack Exchange subtopic ext_test (general) | 1,642 | 0.878 | 0.862 | 0.879 | 0.028 |

| subtopics | macro-F1 | micro-F1 | samples-F1 | Hamming | subset acc | P@1 | R@3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wikipedia test | 0.647 | 0.647 | 0.652 | 0.027 | 0.598 | 0.666 | 0.862 |
| … general topic right | 0.775 | 0.775 | 0.781 | 0.017 | 0.716 | 0.791 | 0.975 |
| Stack Exchange (25 labels) | 0.701 | 0.694 | 0.701 | 0.023 | 0.640 | 0.717 | 0.902 |

Uncertain gate: 6.8% of Wikipedia test and 9.2% of question answers become
uncertain; accuracy of the rest 0.868 / 0.812, of the flagged ones 0.378 /
0.402. Off-topic flagged: 33.3% (Wikipedia categories), 43.6% (questions);
AUROC 0.850 / 0.845. Validation (for reference): accuracy 0.842, macro-F1
0.841. Reliability diagram: `reports/figures/reliability_final.png`.

## 15. Confusion matrix

Figures: `reports/figures/confusion_wiki_test.png`,
`reports/figures/confusion_se_ext_test.png`. Largest off-diagonal cells (share
of the true class):

| Wikipedia test | | Stack Exchange ext_test | |
|---|---:|---|---:|
| Science → Technology | 8.2% | Science → Physics | 43.0% |
| Science → Books | 7.6% | Chemistry → Physics | 12.0% |
| History → Books | 6.4% | History → Science | 11.5% |
| History → Science | 6.4% | History → Books | 11.1% |
| Biology → Science | 5.4% | Science → Books | 10.5% |

## 16. Error analysis

Per-class F1 (Wikipedia / questions): Sports 0.912 / 0.927, Chemistry 0.881 /
0.813, Biology 0.859 / 0.827, Technology 0.856 / 0.854, History 0.846 / 0.776,
Physics 0.842 / 0.716, Books 0.814 / 0.766, **Science 0.661 / 0.369**. Science
is weak by definition (its texts are about other fields). Short questions
(≤ 7 words) reach 0.737 accuracy. Confident errors are label noise, passages
without topical content, and questions whose topic is not in the words. One
error was traced to a label seed and fixed (quantum entanglement labelled
Technology, D-27). docs/ERROR_ANALYSIS.md.

## 17. Conversation tracking

Each informative, certain message adds its calibrated general-topic
probabilities to a score vector that decays by 0.7 per message; uncertain and
uninformative messages add nothing. Topics holding at least 20% of the total
form the theme; subtopic scores are tracked the same way. The composer turns
the theme into a phrase from taxonomy metadata (format + field → "science
books", + narrower field → "science books about biology", perspective →
"history of …"). `reset` clears the tracker, starts a new database session and
keeps all stored rows. Simulated conversations (validation): theme accuracy
0.757, switch lag 1.34 turns, three-topic accumulation 0.825, spurious topics
0.537; test 0.760 / 1.32 / 0.80 / 0.528.

## 18. Web search

Query = theme phrase + at most two salient words that exist in the public
training vocabulary (plural-folded, never names, numbers or e-mail addresses).
Providers: Wikipedia search API, then the DuckDuckGo Instant Answer API; HTTPS-only
URLs on allow-listed domains; snippets cleaned and truncated; 6 s timeout,
bounded retries, `Retry-After` capped at 30 s, 5-minute circuit breaker per
provider, 72 h cache in SQLite. `--no-web` disables all network access.

## 19. Database design

SQLite, `PRAGMA user_version = 1`, foreign keys on, WAL, one transaction per
write, parameterised SQL. Tables: `model_versions`, `sessions`, `texts`,
`conversation_topics`, `search_results`, `search_cache`. A locked or
unwritable database is a warning, never a crashed turn. docs/api.md §5.

## 20. Software architecture

`project.py` (console) → `ConversationSession` (pipeline) → `TopicModel`
(encoder + heads + gates) → `ConversationTracker` → `compose()` (theme
phrase) → `build_query()` → `WebSearcher` → `Database`. Offline data pipeline in
`contextlens/data` and `scripts/`; training in `train.py`; evaluation in
`evaluate.py`. Each component fails in isolation (no web, no database, no
results — the turn still answers). docs/architecture.md, docs/api.md.

## 21. Testing

162 passed, 0 failed (161 offline + trained model, 1 network); ruff and mypy
clean. Acceptance: all brief sentences, the quantum pair, pizza → uncertain,
the three-message conversation ("science books" → "science books about
biology"), reset, uninformative inputs, markup/case invariance, long input,
non-English input, latency. Failures met during development and their fixes
are listed in docs/TEST_REPORT.md §3.

## 22. Performance

Reference machine (`reports/hardware.json`): Linux, Intel Xeon 2.8 GHz, 4 vCPU,
15.7 GB RAM, no GPU, Python 3.11.15, torch 2.5.1+cpu, transformers 4.46.3,
sentence-transformers 3.2.1, scikit-learn 1.5.2.

| measure | value |
|---|---:|
| single message, median / p95 | 19.9 ms / 28.3 ms |
| batch, per text | 5.5 ms |
| model load | 2.8 s |
| artifact size | 45 MB |
| fine-tuning (3 epochs) | 30.5 min |
| `train.py` | 217.8 s |

float16 storage changes embeddings by at most 7e-7 in cosine (min cosine
0.9999993, `reports/fp16_storage.json`).

## 23. Security

Parameterised SQL with an allow-listed identifier; skops heads with a type
allow-list and `allow_pickle=False`; SHA-256 checksums and an encoder
fingerprint (no silent Hub fallback); bounded input; sanitised, allow-listed
web results; queries built only from public vocabulary; no secrets; pinned
dependencies. Residual risk: the local database is plain text.
docs/SECURITY_PRIVACY.md.

## 24. Limitations

Off-topic detection is partial; Science is weak; short messages are harder;
label noise about 4%; small Quantum Computing class; English only; free web
endpoints can rate-limit; the external test consists of question titles, not
chat messages; E-6 not run. KNOWN_ISSUES.md.

## 25. Future improvements

An annotated set of real chat messages (in- and off-topic) for fine-tuning and
testing; an explicit "other" class; reworking Science; fine-tuning e5-small
(E-6); more seeds for weak subtopics and other histories; a new-topic detector
for faster theme switches; ONNX export; database encryption.

## 26. Conclusion

The system meets the brief: it classifies messages into the 8 × 28 taxonomy
with calibrated confidence, recognises part of the off-topic and non-English
input, follows a conversation and composes themes, searches the web without
keys and stores everything locally. The main lesson of the benchmark is that
encyclopedia-only validation would have chosen the wrong model — real
questions changed the ranking — and the main lesson of the error analysis is
that a failing test was a data problem, fixed in the labels rather than hidden.
