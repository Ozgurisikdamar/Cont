# Hardening pass (after v1.0.0)

An audit of v1.0.0 listed methodological and implementation gaps. This file
records, per item, what was found, what was done and where the evidence is.
It is filled in as the work proceeds; every item has its own commit.

| # | audit item | action | evidence | state |
|---|---|---|---|---|
| 1 | multi-label data, softmax objective | – | – | open |
| 2 | no untouched final holdout | – | – | open |
| 3 | off-topic detection 33–44% | – | – | open |
| 4 | Science class weak | – | – | open |
| 5 | theme never expires; tangents | expiry after 3 turns without a confident topic; dominant topic switches only when 2 consecutive confident messages agree; tuned on validation conversations with new metrics (false switch rate, stale-theme rate, premature expiry); CLI says when the theme expired | D-31, `reports/experiments/decay.json`, `contextlens/services/tracker.py`, `tests/test_tracker.py`, `tests/test_acceptance.py` (tangent, switch, expiry with the real model) | done; theme accuracy 0.756 → 0.716 is the price (reported) |
| 6 | language gate skips short texts | fastText lid.176 + training lexicon, per-length reject thresholds chosen on the Tatoeba dev half + in-domain English; new `non_english` status; runs before the informativeness check (other scripts are no longer "uninformative") | D-30, `reports/experiments/language_gate.json`, `contextlens/models/language.py`, `tests/test_acceptance.py` (short-text cases) | done; "guten tag" not caught by this gate (xfail until H7) |
| 7 | no CI; network test cannot fail | network tests split: `network` (resilience, passes on a clean offline state) and `network_live` (a deterministic query must return a valid HTTPS result on an allow-listed host from at least one provider); both deselected by default. CI: see below | `tests/test_network.py`, `reports/tests/pytest_network_live.xml` (2 passed) | network done; CI open |
| 8 | label audit of 48 passages | – | – | open |
| 9 | encoder files without checksums | `encoder_manifest` in metadata.json (path, size, SHA-256 of all 11 encoder files); `load_artifact` refuses missing, changed or extra files; probe fingerprint kept as the semantic check | `contextlens/models/artifact.py`, `tests/test_integration.py::test_encoder_files_are_verified` (3 cases), `::test_encoder_directory_without_manifest_is_refused` | done |
| 10 | repository name | – | – | open |
