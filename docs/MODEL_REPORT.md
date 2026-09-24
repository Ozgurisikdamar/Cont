# Model report — v1.1.0

Which model ContextLens uses and why. Every number is copied from
`reports/experiments/*.json` (rendered in `reports/tables.md` and, one card per
model, `reports/experiment_log.md` by `scripts/report_tables.py`) or from the
locked evaluation (`reports/locked/results.json`). Protocol and experiment
registry: [EXPERIMENTS.md](EXPERIMENTS.md).

**Selection rule.** Hyper-parameters on Wikipedia *val*; model families on *val*
**and** Stack Exchange *ext_dev* (real questions — the input the application
actually receives). Tie-breakers: latency, size, calibration. No test data is
read by any development script; the final model is measured once on the
locked holdout (§6).

**Timeline.** The encoder was chosen in v1.0 (D-22) and fine-tuned again on
taxonomy 1.2.0 before the freeze. The benchmark below was **rerun on the 1.2.0
corpus after the freeze**, development splits only, so that the tables describe
the data the model was trained on; §1 checks that the rerun does not overturn
the frozen choice.

## 1. Result in one table (general topic, macro-F1, taxonomy 1.2.0)

| model | Wikipedia val | Stack Exchange ext_dev | ms / text | MB |
|---|---:|---:|---:|---:|
| majority class | 0.035 | 0.021 | – | – |
| zero-shot NLI (bart-large-mnli, sample of 320; labels v1.0, not rerun) | 0.623 | 0.511 | 2,556 | – |
| zero-shot label similarity (best: mpnet-base) | 0.728 | 0.640 | – | – |
| TF-IDF word + logistic regression | 0.805 | 0.593 | 1.5 | 9.9 |
| TF-IDF word + complement NB (best lexical on questions) | 0.813 | 0.636 | 1.5 | 16.7 |
| TF-IDF word+char + logistic regression | 0.808 | 0.614 | 4.5 | 20.0 |
| all-MiniLM-L6-v2 (frozen) + LR | 0.830 | 0.712 | 14.1 | 90.9 |
| all-mpnet-base-v2 (frozen) + LR | **0.849** | 0.713 | 66.3 | 437.9 |
| bge-small-en-v1.5 (frozen) + LR | 0.838 | 0.736 | 27.8 | 133.4 |
| e5-small-v2 (frozen) + LR | 0.842 | 0.747 | 25.3 ¹ | 133.4 |
| **all-MiniLM-L6-v2 fine-tuned + LR** | 0.848 | **0.749** | **13.5** | **90.9** |

¹ re-measured on an idle machine (`general_latency_idle.json`); the benchmark
run measured 695 ms because the test suite was running at the same time.

Latency: median single-text prediction on the reference CPU; MB: encoder + head
in memory. Full tables (accuracy, weighted-F1, ECE, training time):
`reports/tables.md`.

**Does the rerun change the choice?** No. The fine-tuned MiniLM is the best
model on real questions (0.749; e5-small 0.747 is within noise) and ties
mpnet-base on Wikipedia val (0.848 vs 0.849), while being 5× faster and
5× smaller than mpnet-base and 2× faster than e5-small. On the v1.0 labels it
led on both sets (0.843 / 0.752); on 1.2.0 the margins are narrower and the
tie-breakers decide. The frozen selection stands; a fine-tuned e5-small (E-6)
remains the obvious untested alternative.

## 2. What the benchmark showed

1. **Lexical models do not transfer.** TF-IDF + LR / NB / SVM reach 0.80–0.81
   macro-F1 on Wikipedia but 0.59–0.64 on real questions; their training
   macro-F1 is 0.94–1.00, i.e. they memorise encyclopedic vocabulary.
2. **Sentence embeddings transfer much better**, and the ranking changes with
   the input style: mpnet-base is the best frozen encoder on Wikipedia but
   among the worst on questions (0.713), where the small bge / e5 encoders
   lead. Choosing on Wikipedia alone would pick the wrong, slowest model.
3. **Zero-shot is not good enough**: label similarity ≤ 0.73 / 0.64; NLI
   0.62 / 0.51 and ~200× slower.
4. **Fine-tuning the smallest encoder** (3 epochs, ~40 min on CPU) lifts
   MiniLM from 0.830 / 0.712 frozen to 0.848 / 0.749 — the largest gain of any
   single change, at no inference cost. Validation macro-F1 per epoch
   0.843 → 0.845 → 0.848.

## 3. Subtopics: one primary subtopic + sibling suggestions

**Benchmark heads (E-7, rerun on 1.2.0).** Subtopic macro-F1, val / SE
subtopic questions (τ tuned on val, grid 0.20–0.70):

| encoder | H1 hierarchical | H2 flat multi-label (sigmoids) | H3 flat softmax |
|---|---|---|---|
| TF-IDF word+char | 0.656 / 0.624 | 0.665 / 0.641 | 0.665 / 0.639 |
| bge-small | 0.669 / 0.692 | 0.612 / 0.658 | **0.676 / 0.711** |
| mpnet-base | 0.698 / 0.688 | 0.661 / 0.671 | **0.703 / 0.700** |

H3 also gives the best general-topic macro-F1 for every encoder. The sigmoid
heads' thresholds sat at the top of the grid (0.70).

**Production encoder (E-15, D-33).** Because 6% of passages carry two
subtopics, the softmax was re-examined against genuine multi-label heads on
the fine-tuned encoder, with a wider threshold grid (0.20–0.95):

| head | val sub macro-F1 | val sub micro-F1 | ext_dev sub macro-F1 | 2nd gold label found | extra label on single-label | ext_dev general macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| **flat softmax** | **0.660** | **0.669** | **0.691** | 5.7% | 5.3% | **0.747** |
| one-vs-rest sigmoid | 0.649 | 0.657 | 0.678 | 11.8% | 9.4% | 0.741 |
| per-parent sigmoid | 0.657 | 0.666 | 0.685 | 0.0% | 0.0% | 0.741 |

The softmax wins on every quality metric. The one-vs-rest head finds more
second labels but adds wrong ones almost as often; the per-parent head's best
threshold predicts exactly one label. **Decision:** keep the softmax and
describe the output honestly as **primary subtopic classification with
secondary sibling suggestions** (siblings with P(sub | general) ≥ 0.40).

## 4. Calibration (E-8)

Temperature scaling (T fitted on val by NLL) lowers ECE on questions for every
model, e.g. bge-small 0.072 → 0.051, mpnet-base 0.047 → 0.026, TF-IDF 0.053 →
0.027 (`calibration.json`). The production head applies temperature scaling
(T = 1.428, fitted on the general-topic NLL); locked ECE 0.058 (Wikipedia) /
0.056 (questions).

## 5. Saying "uncertain" and "non_english"

**Benchmark scores (E-9, rerun).** On questions vs off-topic questions,
embedding-geometry scores beat probability scores: bge-small centroid AUROC
0.894, mpnet-base kNN 0.934 vs MSP 0.73–0.76. A threshold calibrated on
Wikipedia passages keeps only 65–91% of genuine questions, so thresholds are
calibrated on questions (D-24).

**Detector comparison on the production model (E-16, D-34).** Seven detectors
on development data, threshold keeping 95% of ext_dev questions:

| detector | AUROC Wikipedia / SE / chat | recall Wikipedia OOD / SE off-topic / chat |
|---|---|---|
| centroid cosine (v1.0) | 0.846 / 0.851 / 0.904 | 0.27 / 0.45 / 0.58 |
| **Mahalanobis** | 0.904 / 0.904 / 0.958 | 0.24 / 0.56 / 0.79 |

(the other five — MSP, energy, kNN, binary classifier, explicit "other" class
— are in `reports/tables.md`; the two trained on CLINC chat catch chat well
but do not generalise to off-topic questions.) Mahalanobis is deployed.

**Language gate (E-13, D-30).** fastText lid.176 with the training lexicon as a
tie-breaker, reject confidence per input length chosen on the Tatoeba dev
half with ≥ 99% English kept.

## 6. Final model and locked result

| | |
|---|---|
| encoder | all-MiniLM-L6-v2 fine-tuned 3 epochs on the 1.2.0 training split (39.8 min on 4 vCPU), exported without its heads, float16 on disk |
| head | `flat_softmax`: multinomial LR (C = 16, class-weighted) over the 28 subtopics; T = 1.428 |
| subtopic rule | primary subtopic of the predicted general topic + siblings with P(sub \| general) ≥ 0.40, at most 3 |
| uncertain | Mahalanobis score below the ext_dev 5th percentile **or** confidence < 0.50 (coverage rule, D-25) |
| non_english | language gate (D-30) |
| artifact | 47 MB, `train.py` 41 s with cached embeddings, `configs/model.json` |

**Locked holdout** (`reports/locked/results.json`, evaluated once after
`scripts/freeze.py`; run 2 after a data-partition fix, D-36):

| set | n | accuracy | macro-F1 | weighted-F1 | ECE |
|---|---:|---:|---:|---:|---:|
| Wikipedia, unseen articles | 374 | 0.848 | 0.838 | 0.847 | 0.058 |
| Stack Exchange questions (2026) | 1,623 | 0.804 | 0.730 | 0.796 | 0.056 |
| … without history of science | 1,524 | 0.843 | 0.818 | – | – |

| subtopics | micro-F1 | macro-F1 (supported) | P@1 | R@3 |
|---|---:|---:|---:|---:|
| Wikipedia | 0.659 | 0.664 | 0.668 | 0.853 |
| questions | 0.603 | 0.507 | 0.624 | 0.830 |

| gate | result |
|---|---|
| in-domain flagged | 4.0% (Wikipedia) / 9.3% (questions); accuracy of the answered 0.861 / 0.844, of the flagged 0.53 / 0.40 |
| off-topic flagged | CLINC150 chat 78.1% (AUROC 0.961), off-topic questions 48.5% (AUROC 0.887) |
| non-English rejected | 48% / 76% / 93% / 97% for 1 / 2 / 3 words / sentences; English kept ≥ 99.6% |

The locked Wikipedia numbers match development (val macro-F1 0.850) closely;
questions are lower than ext_dev only through the history-of-science site
(accuracy 0.19), whose questions are about the history of one field — the
visible cost of D-32 (docs/ERROR_ANALYSIS.md).

**Cost.** Median 19.9 ms per message (p95 30.3 ms), 5.7 ms per text in
batches, model load 2.6 s — on a 4-vCPU CPU, no GPU.

**Acceptance.** All five brief sentences, the quantum pair, the pizza
sentence, the three-message conversation and the tangent / switch / expiry
cases pass (docs/TEST_REPORT.md).
