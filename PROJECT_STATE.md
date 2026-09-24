# Project state — 2026-09-24

**Status: v1.1.0 complete — hardening pass finished, with known limitations.**
The audit of v1.0.0 listed ten gaps; each was worked on in its own commit and
is recorded in [docs/HARDENING.md](docs/HARDENING.md). "Done" below means the
work was carried out and measured — not that the weakness disappeared; the
measured residual is given next to each item.

| # | gap (v1.0) | state | measured result (v1.1.0) |
|---|---|---|---|
| 1 | softmax objective on multi-label data | **done** — three heads compared on dev, softmax kept, objective renamed *primary subtopic + secondary sibling suggestions* everywhere (D-33) | val subtopic macro-F1 0.660 (softmax) vs 0.649 / 0.657; a second gold subtopic is suggested for only 5.7% of multi-label passages |
| 2 | no untouched final holdout | **done** — locked holdout built before any v1.1 decision, freeze fingerprint, evaluated once after the freeze (D-36) | Wikipedia 0.848 acc / 0.838 macro-F1; questions 0.804 / 0.730. Run twice: run 1 had a data-partition bug in the Stack Exchange view; both runs kept |
| 3 | off-topic detection 33–44% | **improved, still partial** — CLINC150 chat added, seven detectors compared, Mahalanobis gate (D-34) | locked: 78.1% of chat, 48.5% of off-topic questions flagged (AUROC 0.961 / 0.887); dev Wikipedia out-of-taxonomy passages ~24% |
| 4 | Science weak | **root cause fixed in the labels, weakness on questions remains** — Science redefined, taxonomy 1.2.0 (D-32) | locked Science F1 0.656 (Wikipedia) / 0.241 (questions); hsm accuracy 0.19, reported separately |
| 5 | theme never expires; tangents move it | **done** — expiry after 4 unconfident turns, switch after 2 agreeing turns (D-31) | false switches 0.488 → 0.104; theme accuracy 0.767 → 0.729 and switch lag 1.34 → 1.92 turns are the price |
| 6 | language gate skips < 3 words | **done** — fastText lid.176 + lexicon, per-length thresholds (D-30) | locked Tatoeba: non-English rejected 48% (1 word) … 97% (sentence); English accepted ≥ 99.6% |
| 7 | no CI; network test cannot fail | **done with a limitation** — resilience vs live smoke tests; CI workflow + `scripts/ci.sh` (D-35) | Actions does not start on this account (billing): manual trigger only, the same steps run locally |
| 8 | label audit of 48 passages | **done** — 310 passages, Wilson CIs | 5.2% wrong [3.2, 8.2], 20.3% weak; 9 wrong labels cannot be removed by a category rule |
| 9 | encoder files without checksums | **done** — SHA-256 manifest, loader refuses changes | tested (3 cases) |
| 10 | repository name `Cont` | **done** — `Ozgurisikdamar/ContexLens` | the frozen User-Agent string still names the intermediate name (redirects) |

## Completed

- Corpus (taxonomy 1.2.0): 40,112 passages / 10,344 articles, train 28,075 ·
  val 5,984; development sets (Stack Exchange ext_dev, CLINC150 dev, Tatoeba
  dev, Wikipedia OOD val); locked holdout (374 + 2,713 + CLINC150 test +
  Tatoeba locked half).
- Benchmark E-0 … E-17 except E-6 (docs/EXPERIMENTS.md); rerun on 1.2.0 with
  development splits only.
- Final model v1.1.0: fine-tuned all-MiniLM-L6-v2, 28-way subtopic softmax
  (C = 16, T = 1.428), Mahalanobis off-topic gate, confidence floor 0.50,
  fastText language gate — `models/contextlens-topic/` (47 MB).
- Application: tracker with decay, hysteresis and expiry; theme composer; query
  builder; web search with fallback / cache / circuit breaker; SQLite; console.
- Tests: 230 passed, 0 failed; ruff and mypy clean (docs/TEST_REPORT.md).
- Documentation regenerated from `reports/` (README, cards, reports, API,
  architecture, decisions D-1 … D-36, hardening log).

## Metrics (locked holdout, `reports/locked/results.json`)

| | Wikipedia (374) | questions (1,623) |
|---|---:|---:|
| general accuracy / macro-F1 | 0.848 / 0.838 | 0.804 / 0.730 |
| without hsm (1,524, 7 classes) | – | 0.843 / 0.818 |
| weighted-F1 / ECE | 0.847 / 0.058 | 0.796 / 0.056 |
| subtopic micro-F1 / P@1 / R@3 | 0.659 / 0.668 / 0.853 | 0.603 / 0.624 / 0.830 (1,212 subtopic questions) |
| uncertain share / accuracy of answered | 4.0% / 0.861 | 9.3% / 0.844 |
| off-topic flagged | – | 48.5% (questions) · 78.1% (chat) |

Latency: median 19.9 ms, p95 30.3 ms per message on 4 vCPU; artifact 47 MB.

## Known problems

Science on real questions; partial off-topic detection (encyclopedia-style
off-topic text especially); one-word non-English inputs; subtopics are a
primary label + suggestions; label noise (5.2% wrong); switch lag; no automatic
CI. Details and measurements: [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Next actions

1. Collect a small human-labelled set of real chat messages (in- and
   off-topic) and train the off-topic detector on it.
2. Allow a second general topic for "history of a field" questions (hsm).
3. Run E-6 (fine-tune e5-small-v2) and compare on ext_dev.
4. Enable automatic CI triggers when Actions minutes are available.
