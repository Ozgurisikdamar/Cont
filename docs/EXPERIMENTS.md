# Experiments

The registry of every experiment run for ContextLens: what was tested, why, how,
and where the raw output is. Result tables are generated from the JSON files by
`scripts/report_tables.py` (→ `reports/tables.md`) and discussed in
[MODEL_REPORT.md](MODEL_REPORT.md).

## Protocol

| rule | detail |
|---|---|
| data | Wikipedia passages (train 30,266 · val 6,515 · test 6,479), Stack Exchange questions (general: ext_dev 10,124 · ext_test 10,087; subtopic: ext_dev 1,630 · ext_test 1,642), OOD (Wikipedia categories, 15 Stack Exchange sites) — [DATASET_CARD.md](DATASET_CARD.md) |
| preprocessing | `contextlens.preprocessing.text.normalize` for every split and at runtime (no lower-casing for encoders; TF-IDF lower-cases itself) |
| fitting | featurizers and heads are fitted on **train only** |
| selection | hyper-parameters: Wikipedia **val**; model families: **val + Stack Exchange ext_dev** |
| reporting | **test** and **ext_test** are computed for the report only |
| seed | `RANDOM_SEED = 42` (Python, NumPy, PyTorch, scikit-learn `random_state`) |
| hardware | 4 vCPU, 15 GB RAM, no GPU (see FINAL_REPORT §hardware) |
| class imbalance | `class_weight="balanced"` for every logistic regression / SVM; class-weighted cross-entropy for fine-tuning |
| metrics | accuracy, macro/weighted precision/recall/F1, per-class report, confusion matrix, ECE (15 bins), log loss; multi-label: micro/macro/weighted/samples F1, Hamming loss, subset accuracy, P@1, R@3; OOD: AUROC, AUPR(OOD), FPR@95%TPR and the flag rate at the deployed threshold; latency (median/p95 single text, batch), model size, training time |

## Registry

| id | question | setup | command | output |
|---|---|---|---|---|
| E-0 | how far does a trivial model get? | majority class | `run_experiments.py --sections general` | `general.json → majority` |
| E-1 | lexical baselines | TF-IDF word (1–2-grams), word without stop words, word + char (3–5 char_wb) × multinomial LR (C ∈ {1, 4, 16}); on word features also MultinomialNB / ComplementNB (α ∈ {0.01, 0.1, 0.5}) and linear SVM + Platt (C ∈ {0.1, 0.5, 2}) | same | `general.json` |
| E-2 | do sentence embeddings transfer better to real questions? | frozen encoders all-MiniLM-L6-v2, bge-small-en-v1.5, e5-small-v2 (`query:` prefix), all-mpnet-base-v2 (pinned revisions, L2-normalised, max 128 tokens) × multinomial LR (C ∈ {0.5, 2, 8, 32}) | same | `general.json` |
| E-3 | can labels alone classify (no training)? | cosine similarity of the text to a label description ("Physics: a text about physics, including quantum mechanics, …"), softmax with τ = 0.05 | `--sections zeroshot` | `zeroshot.json` |
| E-4 | zero-shot NLI | facebook/bart-large-mnli, hypothesis "This text is about {label}.", 40 texts per class from val and ext_dev (NLI costs 8 forward passes per text) | `zeroshot_nli.py --per-class 40` | `zeroshot_nli.json` |
| E-5 | does fine-tuning beat frozen embeddings? | shared all-MiniLM-L6-v2 encoder, mean pooling, general head (class-weighted CE) + subtopic head (BCE), AdamW lr 5e-5, 3 epochs, batch 32, max 64 tokens, linear warm-up 6%; best epoch by val macro-F1 | `finetune_transformer.py --encoder minilm-l6` | `finetune_minilm-l6.json` |
| E-6 | do two encoders help? | concatenated embeddings of two encoders × LR (C grid of E-2) — **planned, not run** | `--sections ensemble` | – |
| E-7 | how to predict subtopics? | H1 hierarchical: P(general) × P(subtopic \| general) with one one-vs-rest head per general topic; H2 flat multi-label over 28 subtopics; H3 flat softmax over the primary subtopic with the general topic derived. Decision rule: best child of the predicted general topic always, siblings added above τ; τ swept 0.20–0.70 on val | `--sections hierarchy` | `hierarchy.json` |
| E-8 | are the probabilities trustworthy? | ECE / NLL before and after temperature scaling (T fitted on val by NLL), for the 8-way softmax head and for the H3 subtopic softmax (T fitted on the general-topic NLL) | `--sections calibration` | `calibration.json` |
| E-8b | which confidence floor? | coverage and accuracy of kept / rejected predictions for min_confidence ∈ {0 … 0.7}; the production value is chosen in `train.py` by the rule of D-25 | `train.py` (the `--sections selective` variant exists but was not run for the report) | artifact `metadata.json → min_confidence_sweep_ext_dev` |
| E-9 | how to say "uncertain"? | in-distribution scores: max softmax probability (MSP), energy (logsumexp of logits / T), max cosine to class centroids, mean cosine to the 10 nearest training passages; threshold at 95% in-distribution kept, calibrated on Wikipedia val **or** on Stack Exchange ext_dev | `--sections ood` | `ood.json` |
| E-10 | which conversation decay? | simulated conversations from held-out passages with the trained model's predictions: 3 segments × 3–6 turns, 15% tangents, 10% OOD; decay ∈ {0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9}; select max val theme accuracy subject to 3-topic accumulation ≥ 0.80 | `tune_decay.py` | `decay.json` |
| E-11 | final model on everything held out | trained artifact on test, ext_test (general + subtopic), OOD test, calibration, latency, acceptance probes | `evaluate.py` | `reports/evaluation.json` |

## Results and decisions

The per-model cards required by the brief (ID, model, features,
hyper-parameters, train / validation / test metrics, training and inference
time, notes, decision) are generated from the JSON files:
**[`reports/experiment_log.md`](../reports/experiment_log.md)**. The discussion
and the final choice are in [MODEL_REPORT.md](MODEL_REPORT.md).

| id | outcome | decision |
|---|---|---|
| E-0 | macro-F1 0.03 | floor |
| E-1 | 0.79–0.80 on Wikipedia, 0.59–0.64 on questions; train macro-F1 0.95–1.00 | lexical models rejected (no transfer) |
| E-2 | 0.81–0.84 / 0.68–0.74; mpnet best on Wikipedia, e5/bge best on questions | embeddings, selection must look at questions |
| E-3 | ≤ 0.71 / 0.64 | rejected |
| E-4 | 0.62 / 0.51, ~2 s per text | rejected |
| E-5 | 0.843 / 0.752, 13 ms per text | **selected encoder** (D-22) |
| E-6 | **not run** — dropped when the owner decided to proceed with MiniLM after E-5 | – |
| E-7 | H3 flat softmax best for every encoder | **selected head** (D-23) |
| E-8 | temperature scaling never raises ECE | applied |
| E-9 | centroid cosine best on questions; Wikipedia-calibrated threshold answers only 66–92% of genuine questions | **gate + calibration on questions** (D-24) |
| E-10 | see `decay.json` | decay (below) |
| E-11 | final numbers | `reports/evaluation.json`, FINAL_REPORT |

## Notes on the protocol

* **Why ext_dev takes part in selection.** The deployed model reads short
  questions, not encyclopedia passages; E-2 showed that the ranking of models
  changes between the two (mpnet-base is best on Wikipedia val, not on
  questions). Choosing on Wikipedia alone would optimise for the wrong input.
* **Why the OOD threshold is calibrated on questions.** A cosine-to-centroid
  score is systematically lower for a 10-word question than for a 25-word passage;
  a threshold set on passages flags too many genuine questions (E-9).
* **Sampled experiments.** Only E-4 is sampled (320 texts per split); its numbers
  have wider uncertainty and are marked as such.
