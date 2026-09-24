# Test report

Evidence: `reports/tests/pytest_offline_and_model.xml`,
`reports/tests/pytest_network.xml` (JUnit XML written by pytest) and the table
rendered from them, `reports/tests/results.md` (`scripts/test_table.py`).
Static checks: `reports/tests/ruff.txt`, `reports/tests/mypy.txt`. Only tests
that were actually run and passed are marked PASS.

## 1. Summary

| run | command | result |
|---|---|---|
| offline + trained model | `pytest -q -m "not network" --junitxml=…` | **161 passed**, 0 failed |
| network | `pytest -q -m network --junitxml=…` | **1 passed**, 0 failed |
| lint / format | `ruff check .` · `ruff format --check .` | all checks passed · 70 files formatted |
| types | `mypy contextlens project.py train.py evaluate.py` | no issues in 39 source files |

Total: **162 passed, 0 failed.** Environment: 4 vCPU Intel Xeon 2.8 GHz,
15.7 GB RAM, no GPU, Python 3.11.15 (`reports/hardware.json`).

| module | tests | what it covers |
|---|---:|---|
| test_integration | 40 | artifact save/load, checksums, encoder fingerprint and refusal, unknown head type, pipeline turn, database writes, empty batch, language gate, both head types (fake encoder) |
| test_acceptance | 21 | the brief's sentences, conversation, edge cases, latency — **real trained model** |
| test_websearch | 13 | providers, fallback, circuit breaker, result sanitising (mocked HTTP) |
| test_composer | 11 | theme composition rules from taxonomy metadata |
| test_db | 11 | schema, pragmas, parameterised SQL, failure never crashes a turn |
| test_heads_metrics | 10 | heads, grouped temperature, metrics |
| test_text | 8 | normalisation, uninformative detection |
| test_passages, test_labels, test_leakage | 7 · 5 · 5 | corpus construction, label rules, no split/acceptance leakage |
| test_query, test_taxonomy, test_tracker | 7 · 7 · 6 | query building, taxonomy validation, decayed context |
| test_encoders, test_settings, test_train_script | 4 · 4 · 2 | local encoder loading and cache identity, settings, train/evaluate/tune_decay smoke |
| test_network | 1 | live Wikipedia/DuckDuckGo search returns safe results or a clean status |

## 2. Acceptance cases (brief)

Actual values are from the trained artifact (`models/contextlens-topic`,
`reports/evaluation.json → acceptance`).

| test | expected | actual | result |
|---|---|---|---|
| "In quantum entanglement, the wave functions of particles can change together." | Physics > Quantum Mechanics | Physics 0.950 · Quantum Mechanics 0.744 | PASS |
| "Quantum processors can speed up certain algorithms by using qubits." | Technology > Quantum Computing | Technology 0.970 · Quantum Computing 0.632 | PASS |
| Test A: "I read the novel I borrowed from the library; …" | Books | Books 0.924 · Novels 0.560 | PASS |
| Test B: "Scientists test their hypotheses using experiments and observation." | Science | Science 0.920 · Scientific Method 0.663 | PASS |
| Test C: "Cells carry DNA and living things diversify through evolution." | Biology | Biology 0.938 · Evolution 0.455 | PASS |
| quantum pair disambiguated | different general topics for the two quantum sentences | Physics vs Technology | PASS |
| "I'm going to order pizza tonight." | uncertain | uncertain (centroid cosine 0.547 < 0.705; raw guess Books 0.772) | PASS |
| conversation after A, B | theme contains Books + Science, phrase "science books" | "Science + Books", "science books" | PASS |
| conversation after A, B, C | Books + Science + Biology, phrase "science books about biology" | "Biology + Science + Books", "science books about biology" | PASS |
| reset | theme empty | empty theme (database history kept — shown in the demo run, not in this test) | PASS |
| empty, spaces, "!!!", "12345", emoji, "the and of", a URL | uninformative, no classification | uninformative | PASS (7 cases) |
| HTML markup / upper case | same answer as plain text | same general topic | PASS |
| very long input (~128,000 characters, truncated to 5,000) | answered, Biology | Biology, status ok | PASS |
| non-English input | not confidently classified | Turkish and German sentences → uncertain ("does not look like English") | PASS |
| probabilities | general sums to 1, subtopics within [0, 1] | yes | PASS |
| single-message latency on CPU | median < 0.5 s | median 19.9 ms, p95 28.3 ms (`evaluate.py`, idle machine) | PASS |

## 3. Failures seen during development (fixed, not hidden)

| test | first result | cause | fix |
|---|---|---|---|
| quantum entanglement → Physics | FAIL: Technology > Quantum Computing 0.61 | seed `Quantum_information_science` labelled entanglement articles as Technology | taxonomy 1.1, relabel, retrain (D-27) |
| non-English input | FAIL: a Turkish sentence answered with a topic | no language check | language gate, 40% known words (D-29) |
| latency < 0.5 s | FAIL once: 0.52 s | measured while a fine-tuning job used all 4 cores | re-run on an idle machine; the threshold was not changed |
| train/evaluate smoke | FAIL: empty batch crashed `predict_proba` | zero-row arrays | empty batch returns empty arrays |

## 4. How to reproduce

```bash
pytest -q -m "not network" --junitxml=reports/tests/pytest_offline_and_model.xml
pytest -q -m network --junitxml=reports/tests/pytest_network.xml
python scripts/test_table.py        # → reports/tests/results.md
```

Without the trained artifact the `model`-marked tests are skipped; without
internet the `network` test is skipped.
