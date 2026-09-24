# Project state — 2026-09-24

**Status: complete (v1.0.0).** All brief requirements are implemented,
trained, evaluated, tested and documented. Open items are limitations, not
unfinished work (KNOWN_ISSUES.md).

## Completed

- Dataset research: 9 candidates, 5 measured (docs/DATASET_RESEARCH.md).
- Corpus v1.1: 42,942 Wikipedia passages / 11,073 articles, 8 general topics,
  28 subtopics, grouped split, leakage checks 0; Stack Exchange external sets;
  out-of-taxonomy sets; manual label audit (docs/DATASET_CARD.md).
- Benchmark E-0 … E-10 except E-6: lexical, zero-shot, frozen encoders,
  fine-tuned MiniLM, subtopic head designs, calibration, OOD gates, decay
  (docs/MODEL_REPORT.md, reports/experiment_log.md).
- Final model: fine-tuned all-MiniLM-L6-v2 + flat 28-way softmax, temperature,
  centroid gate, confidence floor, language gate — `models/contextlens-topic/`.
- Application: conversation tracker, theme composer, query builder, web search
  with fallback/cache/circuit breaker, SQLite, console (`project.py`).
- Tests: 162 passed, 0 failed; ruff and mypy clean (docs/TEST_REPORT.md).
- Documentation: README, CLAUDE.md, decisions (D-1 … D-29), architecture, api,
  model/dataset cards, error analysis, test report, final report, security.

## In progress

Nothing.

## Decisions (headline; all in decisions.md)

| id | decision |
|---|---|
| D-2 | build a Wikipedia corpus (no existing dataset covers the taxonomy) |
| D-3 | Stack Exchange questions for evaluation only |
| D-9 | choose on Wikipedia val **and** real questions; test sets report only |
| D-22 | encoder: MiniLM fine-tuned (best on both selection sets, fastest) |
| D-23 | head: one softmax over 28 subtopics |
| D-24 | uncertain gate: centroid cosine calibrated on questions |
| D-27 | fix the label seed behind the failed quantum test, not the test |
| D-28 | decay 0.7, theme share 0.2 |
| D-29 | language gate |

## Metrics (reports/evaluation.json)

| | Wikipedia test | Stack Exchange ext_test |
|---|---:|---:|
| general accuracy | 0.835 | 0.775 |
| general macro-F1 / weighted-F1 | 0.834 / 0.835 | 0.756 / 0.773 |
| ECE | 0.031 | 0.067 |
| subtopic macro-F1 | 0.647 | 0.701 (25 labels) |
| uncertain share / accuracy of answered | 6.8% / 0.868 | 9.2% / 0.812 |
| off-topic flagged | 33.3% | 43.6% |

Latency: median 19.9 ms per message on 4 vCPU; artifact 45 MB.

## Known problems

Off-topic detection partial; Science class weak (F1 0.369 on questions);
short messages harder; label noise ~4%; small Quantum Computing class. Details: KNOWN_ISSUES.md.

## Next actions

1. Collect a small annotated set of real chat messages (in- and off-topic).
2. Run E-6 (fine-tune e5-small-v2) and compare on ext_dev.
3. Rework the Science class (issue 2).
