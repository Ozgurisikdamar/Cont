# Test report — v1.1.0

Evidence: JUnit XML written by pytest — `reports/tests/pytest_offline_and_model.xml`,
`reports/tests/pytest_network.xml`, `reports/tests/pytest_network_live.xml` —
and the table rendered from them, `reports/tests/results.md`
(`scripts/test_table.py`). Static checks: `reports/tests/ruff.txt`,
`reports/tests/mypy.txt`. Only tests that were actually run and passed are
marked PASS. All runs: 2026-09-24, on the reference machine (4 vCPU Intel Xeon
2.8 GHz, 15.7 GB RAM, no GPU, Python 3.11.15 — `reports/hardware.json`), with
the frozen v1.1.0 artifact.

## 1. Summary

| run | command | result |
|---|---|---|
| offline + trained model | `pytest -q --junitxml=…` (default markers: everything except `network*`) | **227 passed**, 0 failed, 0 skipped, 0 xfailed |
| network resilience | `pytest -q -m network --junitxml=…` | **1 passed** |
| live search smoke | `pytest -q -m network_live --junitxml=…` | **2 passed** (a real provider returned a valid HTTPS result) |
| lint / format | `ruff check .` · `ruff format --check .` | all checks passed · 82 files formatted |
| types | `mypy contextlens project.py train.py evaluate.py` | no issues in 41 source files |
| CI steps | `bash scripts/ci.sh` | run 1 (while the benchmark rerun used all 4 cores): ruff, format, mypy and 194 offline tests passed; model job 35 passed, **1 failed** — `test_single_message_latency_on_cpu`, median 0.51 s against the 0.5 s bound. Rerun on an idle machine: see below |

Total: **230 passed, 0 failed.** The GitHub Actions workflow runs the steps of
`scripts/ci.sh`; it cannot start on this account (billing, D-35), so the local
run is the evidence.

| module | tests | what it covers |
|---|---:|---|
| test_integration | 58 | artifact save/load, file checksums, **encoder manifest** (missing / changed / extra files, no manifest), encoder fingerprint, unknown head type, pipeline turn, database writes, empty batch, language gate, both head types (fake encoder) |
| test_acceptance | 36 | **real trained model**: the brief's sentences, the quantum pair, pizza → uncertain, conversation accumulation, reset, uninformative inputs, markup / case, long input, non-English (sentences and 1–2-word inputs), short English accepted, "guten tag", tangent / sustained switch / expiry with the real model, probabilities, latency |
| test_ood | 23 | all seven off-topic detectors: topical text ranked above chat, detectors that need off-topic training text refuse to fit without it, round trip through the artifact |
| test_tracker | 16 | decay, theme share, hysteresis (confirm turns), expiry, uncertain turns carry no weight |
| test_websearch | 13 | providers, fallback, circuit breaker, result sanitising (mocked HTTP) |
| test_composer · test_db | 11 · 11 | theme composition from taxonomy metadata · schema, pragmas, parameterised SQL, failures never crash a turn |
| test_heads_metrics | 10 | heads, grouped temperature, metrics |
| test_text · test_passages · test_query · test_taxonomy | 8 · 7 · 7 · 7 | normalisation, corpus construction, query building, taxonomy validation |
| test_labels · test_leakage | 5 · 5 | label rules; no split / acceptance leakage |
| test_encoders · test_settings · test_train_script | 4 · 4 · 2 | local encoder loading and cache identity; settings; `train.py`, `evaluate.py --stage dev`, `tune_decay.py` smoke and **`--stage locked` refuses without a freeze file** |
| test_network | 1 + 2 | resilience (clean status whatever the providers do) · live smoke (fails unless a provider returns a valid result) |

## 2. Acceptance cases (brief and hardening)

Values from the frozen artifact (`reports/locked/results.json → acceptance`
and the test run).

| test | expected | actual | result |
|---|---|---|---|
| "In quantum entanglement, the wave functions of particles can change together." | Physics > Quantum Mechanics | Physics 0.942 · Quantum Mechanics 0.847 | PASS |
| "Quantum processors can speed up certain algorithms by using qubits." | Technology > Quantum Computing | Technology 0.966 · Quantum Computing 0.698 | PASS |
| Test A: "I read the novel I borrowed from the library; …" | Books | Books 0.935 · Novels 0.657 | PASS |
| Test B: "Scientists test their hypotheses using experiments and observation." | Science | Science 0.919 · Scientific Method 0.576 | PASS |
| Test C: "Cells carry DNA and living things diversify through evolution." | Biology | Biology 0.931 · Evolution 0.470 | PASS |
| quantum pair | different general topics | Physics vs Technology | PASS |
| "I'm going to order pizza tonight." | uncertain | uncertain — Mahalanobis gate ("far from all training topics"); raw guess Books 0.53 | PASS |
| conversation A → B → C | "science books" → "science books about biology" | "Books + Science" / "science books" → "Books + Biology + Science" / "science books about biology" | PASS |
| reset | theme empty, records kept | empty theme | PASS |
| one Books message inside five Physics messages | Physics stays dominant (tangent) | Physics dominant | PASS |
| two Physics then three Books messages | theme switches to Books | Books dominant | PASS |
| ten off-topic messages after a Physics theme | old theme expires | theme empty | PASS |
| empty, spaces, "!!!", "12345", emoji, "the and of", a URL | uninformative | uninformative | PASS (7) |
| HTML markup / upper case | same answer as plain text | same general topic | PASS |
| very long input (~128,000 characters) | answered | Biology, status ok | PASS |
| Turkish and German sentences | `non_english` | `non_english` | PASS |
| "merhaba", "merhaba dünya", "bonjour", "bonjour monde", "hola amigo" | rejected by the language gate | `non_english` | PASS (5) |
| "guten tag" | no confident topic | passes the language gate, flagged uncertain by the topic gates | PASS |
| "hello", "hello world", "quantum", "quantum physics", "titration", "photosynthesis" | accepted as English | accepted | PASS (6) |
| probabilities | general sums to 1, subtopics in [0, 1] | yes | PASS |
| single-message latency on CPU | median < 0.5 s | median 19.9 ms, p95 30.3 ms (locked evaluation run) | PASS |

## 3. Failures seen during development (fixed, not hidden)

| test | first result | cause | fix |
|---|---|---|---|
| quantum entanglement → Physics | FAIL: Technology > Quantum Computing 0.61 | seed `Quantum_information_science` labelled entanglement articles as Technology | taxonomy 1.1, relabel, retrain (D-27) |
| non-English input | FAIL: a Turkish sentence answered with a topic | no language check | language gate (D-29), fastText per length (D-30) |
| "guten tag" | xfail through H6: Sports 0.58, caught by no gate | two-word German greeting; v1.0 centroid gate | caught by the topic gates since the Mahalanobis gate (H7); now a regular test |
| latency < 0.5 s | FAIL once: 0.52 s | measured while a fine-tuning job used all 4 cores | re-run on an idle machine; the threshold was not changed |
| train/evaluate smoke | FAIL: empty batch crashed `predict_proba` | zero-row arrays | empty batch returns empty arrays |
| `fit_detector` unknown method | FAIL: error raised only after fitting | method validated late | validate first |
| live network test | could not fail when both providers failed | one test for two purposes | split into `network` (resilience) and `network_live` (smoke) |

## 4. How to reproduce

```bash
pytest -q --junitxml=reports/tests/pytest_offline_and_model.xml
pytest -q -m network --junitxml=reports/tests/pytest_network.xml
pytest -q -m network_live --junitxml=reports/tests/pytest_network_live.xml
python scripts/test_table.py        # → reports/tests/results.md
bash scripts/ci.sh                  # lint + types + offline + model tests
```

Without the trained artifact the `model`-marked tests are skipped; without
internet the `network` test passes on a clean `offline` status and the
`network_live` tests fail — that is their purpose.
