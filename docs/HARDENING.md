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
| 5 | theme never expires; tangents | – | – | open |
| 6 | language gate skips short texts | – | – | open |
| 7 | no CI; network test cannot fail | – | – | open |
| 8 | label audit of 48 passages | – | – | open |
| 9 | encoder files without checksums | `encoder_manifest` in metadata.json (path, size, SHA-256 of all 11 encoder files); `load_artifact` refuses missing, changed or extra files; probe fingerprint kept as the semantic check | `contextlens/models/artifact.py`, `tests/test_integration.py::test_encoder_files_are_verified` (3 cases), `::test_encoder_directory_without_manifest_is_refused` | done |
| 10 | repository name | – | – | open |
