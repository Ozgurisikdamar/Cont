# Model report

Which model ContextLens uses and why. Every number is copied from
`reports/experiments/*.json` (rendered in `reports/tables.md` and, one card per
model, `reports/experiment_log.md` by `scripts/report_tables.py`) or from
`reports/evaluation.json` (the final model, `evaluate.py`). Protocol and
experiment registry: [EXPERIMENTS.md](EXPERIMENTS.md).

**Selection rule.** Hyper-parameters on Wikipedia *val*; model families on *val*
**and** Stack Exchange *ext_dev* (real questions — the input the application
actually receives). *test* and *ext_test* are reported, never used to choose.
Tie-breakers: latency, size, calibration.

## 1. Result in one table (general topic, macro-F1)

| model | Wikipedia val | Stack Exchange ext_dev | ms / text | MB |
|---|---:|---:|---:|---:|
| majority class | 0.033 | 0.021 | – | – |
| zero-shot NLI (bart-large-mnli, sample of 320) | 0.623 | 0.511 | 2,556 | – |
| zero-shot label similarity (best: mpnet-base) | 0.708 | 0.640 | – | – |
| TF-IDF word + logistic regression | 0.793 | 0.595 | 1.6 | 10.6 |
| TF-IDF word + complement NB (best lexical on questions) | 0.803 | 0.643 | 1.8 | 17.8 |
| TF-IDF word+char + logistic regression | 0.800 | 0.618 | 4.7 | 21.1 |
| all-MiniLM-L6-v2 (frozen) + LR | 0.813 | 0.679 | 14.9 | 90.9 |
| bge-small-en-v1.5 (frozen) + LR | 0.823 | 0.735 | 27.9 | 133.4 |
| e5-small-v2 (frozen) + LR | 0.823 | 0.743 | 27.1 | 133.4 |
| all-mpnet-base-v2 (frozen) + LR | 0.836 | 0.737 | 66.7 | 437.9 |
| **all-MiniLM-L6-v2 fine-tuned (two heads)** | **0.843** | **0.752** | **13.0** | **90.9** |

(latency: median single-text prediction on the reference CPU; MB: encoder +
head in memory. Full tables with accuracy, weighted-F1, ECE, test columns and
training time: `reports/tables.md`.)

## 2. What the benchmark showed

1. **Lexical models do not transfer.** TF-IDF + LR/NB/SVM reach 0.79–0.80
   macro-F1 on Wikipedia but only 0.59–0.64 on real questions; their training
   macro-F1 is 0.95–1.00, i.e. they memorise encyclopedic vocabulary. The same ranking shows on *test* / *ext_test*.
2. **Sentence embeddings transfer much better**: frozen encoders + a linear
   head lose 8–13 points between Wikipedia and questions instead of 16–20.
   Among them the largest (mpnet-base) is best on Wikipedia but **not** on
   questions — choosing on Wikipedia alone would have picked the wrong model
   and paid 2.5× the latency of e5-small.
3. **Zero-shot is not good enough.** Label-description similarity needs no
   training but stays ≤ 0.71 / 0.64; zero-shot NLI is worse (0.62 / 0.51) and
   150–200× slower than the fine-tuned encoder. (NLI was measured on a stratified
   sample of 40 texts per class per split — wider uncertainty.)
4. **Fine-tuning pays.** Fine-tuning the smallest encoder (MiniLM, 3 epochs on
   CPU, 51 min) beats every frozen encoder on both selection sets while staying
   the fastest and smallest. Validation macro-F1 per epoch 0.833 → 0.838 → 0.843
   (best epoch = last; the gain per epoch is shrinking).

## 3. Subtopics: hierarchical vs flat (E-7)

Three ways to predict the 28 subtopics were compared for every encoder; τ is
the sibling threshold tuned on val (0.20–0.70). Subtopic macro-F1; "SE" is the
Stack Exchange subtopic set (25 labels with questions).

| encoder | H1 hierarchical (val / SE) | H2 flat multi-label (val / SE) | H3 flat softmax (val / SE) |
|---|---|---|---|
| TF-IDF word+char | 0.651 / 0.628 | **0.665 / 0.650** | 0.660 / 0.641 |
| bge-small | 0.662 / 0.688 | 0.607 / 0.660 | **0.676 / 0.709** |
| e5-small | 0.674 / 0.695 | 0.615 / 0.659 | **0.688 / 0.716** |
| mpnet-base | 0.692 / 0.690 | 0.656 / 0.687 | **0.699 / 0.712** |

H3 (one softmax over the 28 subtopics; the general topic is the sum of its
subtopics' probabilities) also gives the **best general-topic** macro-F1 for
every encoder (e.g. e5-small 0.840 vs 0.821 on val, 0.871 vs 0.847 on the SE
subtopic questions). Reading: training on 28 finer classes gives the general
decision more structure, and a single softmax keeps the subtopics of one
general topic in competition, which a set of independent sigmoids does not.
H2's thresholds sat at the top of the allowed range (0.70), a sign that its
independent sigmoids are over-confident. **Decision: H3 for the production
model** (decisions.md D-23). Multi-label output is kept: the best subtopic of
the predicted general topic is always returned, siblings are added when their
conditional probability reaches τ.

## 4. Calibration (E-8)

Temperature scaling (T fitted on val by NLL) lowers ECE where the model is
miscalibrated and never raises it: bge-small on questions 0.067 → 0.046,
mpnet-base 0.041 → 0.017, TF-IDF 0.055 → 0.024; e5-small is already calibrated
(T = 1.03). The production model applies temperature scaling to its head
(for H3 the temperature is fitted on the NLL of the *general* topic).

## 5. Saying "uncertain" (E-9)

In-distribution scores compared: max softmax probability, energy, maximum
cosine to the 8 class centroids, mean cosine to the 10 nearest training
passages. Two threshold calibrations were compared (keep 95% of in-domain
texts): on Wikipedia val passages, or on Stack Exchange ext_dev questions.

* On **questions vs off-topic questions** the embedding-geometry scores are
  best: bge-small centroid cosine AUROC 0.894, kNN 0.885, vs MSP 0.768.
* A threshold calibrated on **Wikipedia** passages is wrong for questions:
  with bge-small centroid cosine it answers only 92.0% of genuine questions
  (e5-small 65.9%, mpnet-base 66.2%) instead of the intended 95%, because a
  10-word question lies farther from every centroid than a 25-word passage.
* **Decision:** centroid cosine, threshold calibrated on ext_dev questions
  (D-24). kNN scores are similar but need the training embeddings at run time.

## 6. Final model

⟪FINAL⟫
