# Model card — contextlens-topic 1.0.0

| | |
|---|---|
| **name / version** | `contextlens-topic` 1.0.0, trained 2026-09-24 (`models/contextlens-topic/metadata.json`) |
| **problem** | English topic classification: 1 of 8 general topics + 1–3 of 28 subtopics (hierarchical, multi-label), with calibrated confidence and an *uncertain* answer |
| **architecture** | all-MiniLM-L6-v2 fine-tuned on the training split (float16 weights) → mean-pooled L2-normalised 384-d embedding → multinomial logistic regression over the 28 subtopics with temperature scaling; P(general) = sum over its subtopics; OOD gate = cosine to 8 class centroids; language gate |
| **artifact** | 45 MB; heads in skops (type allow-list), SHA-256 checksums, encoder fingerprint (docs/architecture.md §3) |
| **licence** | code: Apache-2.0 (LICENSE); encoder derived from all-MiniLM-L6-v2 (Apache-2.0); training text CC BY-SA ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) |

## Intended use

Tagging messages in a local, single-user console so that a conversation theme
and a web-search query can be derived. Topics are broad (physics, biology,
chemistry, technology, science, books, sports, history).

**Not intended for:** content moderation, decisions about people, languages
other than English, domains outside the taxonomy (the model will map them to
the nearest topic unless the gate catches them).

## Data

Training: 30,051 passages from English Wikipedia (labels v1.1), labelled
through Wikipedia's category graph from curated seeds
([DATASET_CARD.md](DATASET_CARD.md)). Selection data: Wikipedia validation
(6,464) and Stack Exchange ext_dev questions (10,124). Test: Wikipedia test
(6,427), Stack Exchange ext_test (10,087 general, 1,642 subtopic), 675
out-of-taxonomy passages and 6,000 off-topic questions.

## Preprocessing

`contextlens.preprocessing.text.normalize`: Unicode NFKC, HTML unescape and tag
removal, URLs / e-mail / @mentions removed, control characters removed,
whitespace collapsed, at most 5,000 characters; no explicit lower-casing (the
MiniLM tokenizer is uncased). Texts with no content words (empty,
symbols, numbers, stop words only) are *uninformative* and not classified.

## Training

1. Fine-tune MiniLM with a general head (class-weighted cross-entropy) and a
   subtopic head (BCE): AdamW, lr 5e-5, 3 epochs, batch 32, max 64 tokens,
   warm-up 6%, seed 42 — 30.5 min on 4 CPU cores.
2. Export the encoder; fit the production head on its embeddings (C = 8,
   class_weight balanced); temperature on validation NLL (T = 1.431).
3. Thresholds on development data only: τ_sub = 0.40 (val macro-F1 sweep
   0.20–0.70), OOD 0.705 (keeps 95% of ext_dev questions), min_confidence 0.50
   (largest value answering ≥ 90% of ext_dev questions: 93.2%), language gate
   0.40 known-word share.

## Metrics (test sets, never used for choices)

| set | accuracy | macro-F1 | weighted-F1 | ECE |
|---|---:|---:|---:|---:|
| Wikipedia test | 0.835 | 0.834 | 0.835 | 0.031 |
| Stack Exchange questions | 0.775 | 0.756 | 0.773 | 0.067 |

Subtopics: macro-F1 0.647 (Wikipedia), 0.701 (questions, 25 labels);
P@1 0.666 / R@3 0.862 on Wikipedia. Full tables:
[MODEL_REPORT.md §6](MODEL_REPORT.md), per class:
[ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

## Thresholds and what they do

| rule | effect on test data |
|---|---|
| uncertain (any of: centroid cosine < 0.705, confidence < 0.50, < 40% known English words) | 6.8% of Wikipedia / 9.2% of question answers; accuracy of the rest 0.868 / 0.812 |
| same rule on off-topic text | 33.3% (Wikipedia categories) / 43.6% (Stack Exchange sites) flagged |
| subtopics | best child of the predicted general topic always; siblings ≥ 0.40 |

## Limitations and known failure cases

* **Science** (the study of science itself) is weak: F1 0.661 on Wikipedia,
  0.369 on questions; history-of-science questions go to Physics.
* **Short questions** (≤ 7 words): accuracy 0.737.
* **Off-topic detection is partial**: half to two thirds of off-topic texts get
  a topic with apparent confidence (e.g. pizza → Books 0.77 is caught only by
  the centroid gate).
* Labels are distant supervision; a manual audit found 4.2% wrong and 6.2%
  weak labels.
* Quantum computing has little training data after the v1.1 fix (539 passages).

## Hardware and latency

Trained and measured on 4 vCPU (Intel Xeon 2.8 GHz), 15.7 GB RAM, no GPU,
Linux, Python 3.11.15, torch 2.5.1+cpu (`reports/hardware.json`). Inference:
median 19.9 ms, p95 28.3 ms per message; 5.5 ms per text in batches; load 2.8 s.
A GPU is used automatically when available.

## Ethical and bias considerations

* Wikipedia's coverage and category structure reflect its editors (English,
  Western-centric); e.g. the History topic is seeded with Ottoman history,
  world wars and ancient history only, so other histories are covered less.
* Biographies are largely excluded by category rules, which reduces but does
  not remove personal data in the corpus.
* Stack Exchange evaluation reflects the users of 11 English sites.
* The model can be confidently wrong; it should inform, not decide.
