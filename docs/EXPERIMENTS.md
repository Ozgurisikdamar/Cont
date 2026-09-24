# Experiments

The registry of every experiment run for ContextLens: what was tested, why, how,
and where the raw output is. Result tables are generated from the JSON files by
`scripts/report_tables.py` (→ `reports/tables.md`) and discussed in
[MODEL_REPORT.md](MODEL_REPORT.md).

## Protocol

| rule | detail |
|---|---|
| data | Wikipedia passages, taxonomy **1.2.0** (train 28,075 · val 5,984; the old `test` split, 6,053, is legacy), Stack Exchange questions (general ext_dev 10,124; subtopic ext_dev 1,630), off-topic: Wikipedia categories (val 652), 15 Stack Exchange sites (ext_dev 6,000), CLINC150 assistant chat (dev 2,900, train half for detectors only), Tatoeba (dev half) — [DATASET_CARD.md](DATASET_CARD.md). v1.0 ran E-0 … E-11 on labels 1.0 / 1.1; after the hardening pass the benchmark was **rerun on 1.2.0 with development splits only** (after the freeze; it does not change the frozen choice). |
| preprocessing | `contextlens.preprocessing.text.normalize` for every split and at runtime (no lower-casing for encoders; TF-IDF lower-cases itself) |
| fitting | featurizers and heads are fitted on **train only** |
| selection | hyper-parameters: Wikipedia **val**; model families: **val + Stack Exchange ext_dev** |
| reporting | development scripts compute **no** test numbers any more. The final model is measured once on the **locked holdout** (built before any v1.1 decision, `data/locked/`) after `scripts/freeze.py` (E-17). `test` / `ext_test` were reported during v1.0 development and are kept only as a *legacy* section of the locked report |
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
| E-11 | v1.0 final model on the (then) held-out sets | superseded by E-17 | `evaluate.py` (v1.0) | `reports/evaluation_v1.0.json` |
| E-12 | how wrong are the labels, and why is Science weak? | 310-passage stratified audit (10 per subtopic + 5 for six weak subtopics), verdict per item, Wilson 95% CIs, root-cause trace of every wrong Science label | `label_audit_v2.py` | `reports/label_audit_v2.json`, `relabel_taxonomy_1.2.0.json` |
| E-13 | can short non-English text be rejected? | fastText lid.176 vs lexicon vs both; reject confidence per input length (1 / 2 / 3 / 4+ words) on the Tatoeba dev half + in-domain English; rule: max rejection subject to ≥ 99% English accepted | `language_gate_experiment.py` | `language_gate.json` |
| E-14 | expiry and tangent robustness | E-10 grid extended: decay × share × confirm turns × expiry × switch rule (360 settings); new metrics: false switch rate, stale theme, premature expiry; rule in `decay.json` | `tune_decay.py` | `decay.json` |
| E-15 | is the data multi-label enough for a multi-label head? | same fine-tuned encoder: flat softmax vs one-vs-rest sigmoid vs per-parent sigmoid; C and thresholds (0.20–0.95) on val; fixed selection score | `head_comparison.py` | `head_comparison.json` (first run with a 0.70 grid cap: `head_comparison_grid070.json`) |
| E-16 | a better off-topic detector | centroid, MSP, energy, Mahalanobis, kNN, binary in-vs-out classifier, explicit "other" class; in-domain Wikipedia val + ext_dev vs Wikipedia OOD val, off-topic SE ext_dev, CLINC150 dev; threshold keeps 95% of ext_dev questions | `ood_experiment.py` | `ood_detectors.json` |
| E-17 | the final model, once, on data never seen | locked holdout (374 Wikipedia passages, 2,713 SE questions from 2026, CLINC150 test, Tatoeba locked half) + legacy test splits; raw predictions stored | `freeze.py`, `evaluate.py --stage locked` | `reports/locked/` |

## Results and decisions

The per-model cards required by the brief (ID, model, features,
hyper-parameters, train / validation / test metrics, training and inference
time, notes, decision) are generated from the JSON files:
**[`reports/experiment_log.md`](../reports/experiment_log.md)**. The discussion
and the final choice are in [MODEL_REPORT.md](MODEL_REPORT.md).

| id | outcome | decision |
|---|---|---|
| E-0 | macro-F1 0.035 / 0.021 | floor |
| E-1 | 0.80–0.81 on Wikipedia, 0.59–0.64 on questions; train macro-F1 0.94–1.00 | lexical models rejected (no transfer) |
| E-2 | 0.83–0.85 / 0.71–0.75; mpnet best on Wikipedia, e5/bge best on questions | embeddings, selection must look at questions |
| E-3 | ≤ 0.73 / 0.64 | rejected |
| E-4 | 0.62 / 0.51, ~2 s per text (v1.0 labels, not rerun) | rejected |
| E-5 | v1.0 labels 0.843 / 0.752; taxonomy 1.2.0 0.848 / 0.749 with the production LR head, 13.5 ms per text | **selected encoder** (D-22); confirmed by the 1.2.0 rerun on tie-breakers |
| E-6 | **not run** — dropped when the owner decided to proceed with MiniLM after E-5 | – |
| E-7 | H3 flat softmax best for every encoder | **selected head** (D-23) |
| E-8 | temperature scaling never raises ECE | applied |
| E-9 | centroid cosine best on questions; Wikipedia-calibrated threshold answers only 66–92% of genuine questions | **gate + calibration on questions** (D-24) |
| E-10 | see `decay.json` | decay (below) |
| E-11 | superseded | `reports/evaluation_v1.0.json` |
| E-12 | 5.2% wrong [3.2, 8.2], 20.3% weak; Science worst (8 of 45 wrong) — its crawl pulled in every field's history and R&D pages | taxonomy 1.2.0, Science = scientific enterprise (D-32) |
| E-13 | fastText + lexicon; on the locked half non-English is rejected 48–97% (1 word → sentence), English accepted ≥ 99.6% | deployed (D-30) |
| E-14 | expiry 4, confirm 2: theme accuracy 0.767 → 0.729, false switches 0.488 → 0.104 | deployed (D-31) |
| E-15 | softmax 0.660 val sub macro-F1 vs 0.649 / 0.657; sigmoid thresholds at the top of the grid | softmax kept, objective renamed (D-33) |
| E-16 | Mahalanobis: chat recall 0.79, SE off-topic 0.56, Wikipedia OOD 0.24 at 95% question retention | deployed (D-34) |
| E-17 | locked: Wikipedia 0.848 / 0.838, questions 0.804 / 0.730 (0.843 / 0.818 without hsm) | reported (D-36) |

## Notes on the protocol

* **Why ext_dev takes part in selection.** The deployed model reads short
  questions, not encyclopedia passages; E-2 showed that the ranking of models
  changes between the two (mpnet-base is best on Wikipedia val, not on
  questions). Choosing on Wikipedia alone would optimise for the wrong input.
* **Why the OOD threshold is calibrated on questions.** A cosine-to-centroid
  score is systematically lower for a 10-word question than for a 25-word passage;
  a threshold set on passages flags too many genuine questions (E-9).
* **Why the benchmark was rerun after the freeze.** Its tables are the
  documentation's evidence; after the 1.2.0 relabel they had to describe the
  data the model was trained on. The rerun uses development splits only and
  its outcome was checked against the frozen choice (MODEL_REPORT §1).
* **Sampled experiments.** Only E-4 is sampled (320 texts per split); its numbers
  have wider uncertainty and are marked as such.
