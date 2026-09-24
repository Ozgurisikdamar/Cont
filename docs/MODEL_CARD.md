# Model card — contextlens-topic 1.1.0

| | |
|---|---|
| **name / version** | `contextlens-topic` 1.1.0 (`models/contextlens-topic/metadata.json`), frozen before the locked evaluation (`reports/locked/FREEZE.json`) |
| **problem** | English topic analysis: 1 of 8 general topics, a **primary subtopic** of 28 plus **secondary sibling suggestions**, calibrated confidence, and an *uncertain* / *non_english* / *uninformative* answer when the model should not commit |
| **architecture** | all-MiniLM-L6-v2 fine-tuned on the training split (float16) → mean-pooled L2-normalised 384-d embedding → one multinomial logistic regression over the 28 subtopics (trained on each passage's primary subtopic) with temperature scaling; P(general) = sum of its subtopics; off-topic gate = Mahalanobis distance to the 8 general-topic means; language gate = fastText lid.176 + training lexicon |
| **artifact** | 47 MB; heads + off-topic detector in skops (type allow-list), SHA-256 checksums of every artifact file and every encoder file (manifest), encoder fingerprint (docs/architecture.md §3) |
| **licence** | code: Apache-2.0 (LICENSE); encoder derived from all-MiniLM-L6-v2 (Apache-2.0); training text CC BY-SA; lid.176 CC BY-SA 3.0 ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)) |

## What the subtopic output means

The data is mostly single-label: 6% of passages carry two or three subtopics.
Three heads were compared on development data (decisions.md D-33): the 28-way
softmax, 28 independent sigmoids (one-vs-rest) and one set of sigmoids per
general topic. The softmax scored best (val subtopic macro-F1 0.660 vs 0.649 /
0.657), and every sigmoid head's best threshold was the highest on the grid,
i.e. its best decision was to predict one label. The model is therefore
described as **primary subtopic classification with secondary sibling
suggestions**: siblings whose share of the parent's probability mass is at
least 0.40 are shown as suggestions. It is not a true multi-label classifier
(a second gold subtopic is found for 5.7% of multi-label validation passages).

## Intended use

Tagging messages in a local, single-user console so that a conversation theme
and a web-search query can be derived. Topics are broad (physics, biology,
chemistry, technology, science, books, sports, history).

**Not intended for:** content moderation, decisions about people, languages
other than English, domains outside the taxonomy (the off-topic gate catches
78% of assistant chat and about half of off-topic questions — the rest is
mapped to the nearest topic).

## Data

Training: 28,075 passages from English Wikipedia (taxonomy 1.2.0), labelled
through Wikipedia's category graph from curated seeds, audited on 310
passages ([DATASET_CARD.md](DATASET_CARD.md)). Development (all choices):
Wikipedia validation (5,984), Stack Exchange ext_dev (10,124 general, 1,630
subtopic questions), CLINC150 dev chat (2,900), Tatoeba dev. **Locked
holdout** (evaluated once after the freeze, decisions.md D-36): 374 unseen
Wikipedia passages, 2,713 Stack Exchange questions created in 2026, 4,350
CLINC150 test utterances, the Tatoeba locked half. The v1.0 test splits are
reported as *legacy (seen during v1.0 development)*.

## Preprocessing

`contextlens.preprocessing.text.normalize`: Unicode NFKC, HTML unescape and tag
removal, URLs / e-mail / @mentions removed, control characters removed,
whitespace collapsed, at most 5,000 characters; no explicit lower-casing (the
MiniLM tokenizer is uncased). Order of the gates: no content words →
*uninformative*; language gate confident it is not English → *non_english*;
English stop words only → *uninformative*; otherwise classified.

## Training

1. Fine-tune MiniLM with a general head (class-weighted cross-entropy) and a
   subtopic head (BCE): AdamW, lr 5e-5, 3 epochs, batch 32, max 64 tokens,
   warm-up 6%, seed 42; best epoch by validation macro-F1
   (`reports/experiments/finetune_minilm-l6.json`).
2. Export the encoder; fit the production head on its embeddings (C = 16,
   class_weight balanced, chosen on val); temperature on validation NLL
   (T = 1.428).
3. Thresholds on development data only: τ_sub = 0.40 (val macro-F1 sweep
   0.20–0.95); off-topic gate = 5th percentile of the Mahalanobis score of the
   ext_dev questions (keeps 95%); min_confidence 0.50 (largest value answering
   ≥ 90% of ext_dev questions: 92.6%); language gate reject confidence
   0.5 / 0.5 / 0.3 / 0.3 for 1 / 2 / 3 / 4+ words.

## Metrics — locked holdout (`reports/locked/results.json`)

| set | n | accuracy | macro-F1 | weighted-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Wikipedia, unseen articles | 374 | 0.848 | 0.838 | 0.847 | 0.058 |
| Stack Exchange questions (2026) | 1,623 | 0.804 | 0.730 | 0.796 | 0.056 |
| same, without the hsm site (7 topics) | 1,524 | 0.843 | 0.818 | – | – |

Subtopics (locked): Wikipedia micro-F1 0.659, macro-F1 0.664 over supported
labels, P@1 0.668, R@3 0.853; questions micro-F1 0.604, macro-F1 0.507 (24
supported labels, several with fewer than 10 questions).
Legacy test splits (seen in v1.0 development): Wikipedia 0.847 / 0.839,
Stack Exchange 0.773 / 0.746 (accuracy / macro-F1).

## Gates and what they do (locked holdout)

| rule | effect |
|---|---|
| uncertain (Mahalanobis score below threshold, confidence < 0.50, or not English) on in-domain text | 4.0% of Wikipedia passages, 9.3% of questions; accuracy of the answered ones 0.861 / 0.844 |
| same rule on off-topic text | 78.1% of CLINC150 chat, 48.5% of off-topic Stack Exchange questions flagged; AUROC 0.961 / 0.887 |
| language gate (Tatoeba locked) | English accepted ≥ 99.6% at every length; non-English rejected 48% (1 word), 76% (2), 93% (3), 97% (full sentence) |
| subtopics | best child of the predicted general topic always; siblings ≥ 0.40 as suggestions |

## Limitations and known failure cases

* **Science on questions:** F1 0.656 on Wikipedia but 0.241 on questions — the
  hsm site asks mostly about the history of one field (accuracy 0.19; 42% go
  to Physics). Science was redefined as the scientific enterprise itself
  (decisions.md D-32); this is the visible cost.
* **Off-topic detection is partial:** half of off-topic questions and a fifth
  of assistant chat still get a topic; out-of-taxonomy *encyclopedia
  paragraphs* are caught only ~24% of the time (development data).
* **Short non-English inputs:** half of one-word non-English inputs pass the
  language gate (the topic gates catch some, e.g. "guten tag").
* **Labels:** distant supervision; 310-passage audit: 5.2% wrong [3.2, 8.2],
  20.3% weak.
* **Weak subtopics:** classical mechanics, evolution, cell biology, history of
  science and relativity score F1 < 0.50 on the locked Wikipedia passages
  (14–17 passages each — small samples).

## Hardware and latency

Trained and measured on 4 vCPU (Intel Xeon 2.8 GHz), 15.7 GB RAM, no GPU,
Linux, Python 3.11, torch 2.5.1+cpu (`reports/hardware.json`). Inference
(locked run): median 19.9 ms, p95 30.3 ms per message; 5.7 ms per text in
batches. A GPU is used automatically when available.

## Ethical and bias considerations

* Wikipedia's coverage and category structure reflect its editors (English,
  Western-centric); e.g. the History topic is seeded with Ottoman history,
  world wars and ancient history only.
* Biographies are largely excluded by category rules; the audit removed
  politicians reached through sports and physics categories (D-32).
* Stack Exchange and CLINC150 reflect the users of a few English sites and
  of one assistant-intent dataset.
* The model can be confidently wrong; it should inform, not decide.
