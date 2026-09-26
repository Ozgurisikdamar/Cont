# Handover

Read §1–§3 before starting a session; rewrite §1–§2 and add a §5 entry at the end
of a session.

## 1. Current state

See [PROJECT_STATE.md](PROJECT_STATE.md) for the one-page status and
[sprints.md](sprints.md) for the sprint board. Headline numbers live in
[docs/FINAL_REPORT.md](docs/FINAL_REPORT.md); every number is generated from
`reports/` (`python scripts/report_tables.py`).

## 2. Open items

See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) (limitations with measurements) and the
backlog at the end of `sprints.md`.

## 3. Things that cost time (read before touching data or benchmarks)

1. **The 2023 Wikipedia dump is incomplete.** Absent titles are not a matching
   bug: "Gold", "Spacetime", "Quantum chromodynamics" are simply not in
   `wikimedia/wikipedia` 20231101.en. The 2022 legacy dump has them. Never match
   titles case-insensitively ("GOLD" is a different page).
2. **Live endpoints misbehave from cloud IPs.** Wikipedia's action and REST APIs
   return 429 intermittently (the REST API with a 30 s Retry-After on cache
   misses); DBpedia goes into maintenance (502) and returns partial results with
   HTTP 200. Everything the data build fetches is cached under `data/raw/`.
3. **DBpedia's public endpoint has no abstracts and no `dbo:wikiPageID`.**
4. **DBpedia category names drift.** `Category:Automobiles` is empty; the
   current name is `Cars`. An empty seed fails silently — check
   `reports/eda.json → ood.by_category` after a rebuild.
5. **`pkill -f` / `pgrep -f` match their own shell.** A wait loop written as
   `until ! pgrep -f "download_data"` never ends (it matches itself), and a
   `pkill -f` with the same pattern kills the calling shell (exit 144). Use PIDs
   or check a log line instead.
6. **`tail -F log | grep | cut` monitors stay silent**: `cut` buffers. Use
   `grep --line-buffered` as the last stage.
7. **Benchmark runtime on 4 vCPU:** encoding the corpus + external sets takes
   ~7 min (MiniLM), ~13 min (bge/e5-small), ~30 min (mpnet-base); fine-tuning
   MiniLM for 3 epochs ~35–50 min. Embeddings are cached in
   `data/processed/embeddings/` (keyed by content hash), so re-running a section
   is cheap.

8. **Locked data is read once.** `evaluate.py --stage locked` needs
   `reports/locked/FREEZE.json`; files in the fingerprint (`configs/*.json`,
   `contextlens/config.py`, passages, artifact metadata, locked manifest) must
   not change afterwards. Stack Exchange views are independent flags — the one
   partition bug of this project was an exclusive field (D-36).
9. **Actions never start on this account** (billing): the workflow is manual
   only; run `bash scripts/ci.sh` locally.
10. **Monitor-type watchers expire after a few minutes;** for runs of an hour
   use a background shell that waits and writes a log line.
11. **Embedding caches are keyed by the whole split's text hash:** a relabel
   that drops passages re-encodes the whole split for every encoder (~1.5 h for
   the full benchmark on 4 vCPU).

## 4. How to continue

```bash
. .venv/bin/activate
pytest -q                          # offline suite
pytest -q -m model                 # needs models/contextlens-topic
bash scripts/ci.sh                 # lint + types + offline + model tests
python scripts/report_tables.py    # regenerate reports/tables.md
python evaluate.py --stage dev     # development evaluation (never --stage locked again without a D-entry)
```

## 5. Session log

### 2026-09-24 — build, benchmark, application, documentation
- Built the corpus (two-dump text, redirect recovery, near-duplicate filter),
  the Stack Exchange evaluation sets and EDA; manual label audit (48 passages).
- Ran the benchmark (E-0 … E-9), trained the final artifact, tuned the decay,
  evaluated on all held-out sets, wrote the documentation set.
- Fixed on the way: ruff excluded `contextlens/data` and `contextlens/models`;
  uncertain turns leaked words into search queries; history ignored local
  records when the DB session could not start; `max_subtopics` was unused; the
  database stored unbounded message text; an unreadable `--script` raised a
  traceback; OOD test seed `Automobiles` was empty.

### 2026-09-24 (later) — final model, evaluation, delivery
- Selected fine-tuned MiniLM (owner: "pick MiniLM and move on"); flat 28-way
  softmax head won E-7; OOD threshold and confidence floor chosen on ext_dev.
- The quantum-entanglement acceptance sentence failed (Technology 0.61): the
  seed `Quantum_information_science` labelled entanglement articles as
  Technology. Taxonomy 1.1 + `scripts/relabel_corpus.py` (splits kept), MiniLM
  fine-tuned again (30.5 min). Now Physics 0.95.
- Added the language gate (a Turkish sentence was answered with a topic).
- Decay tuning probe measured classifier errors (ceiling = accuracy³); fixed to
  use confidently correct messages; grid now decay × theme share.
- Encoder safety: probe-embedding fingerprint, no Hub fallback for local
  encoders, embedding cache keyed by encoder identity.
- Lessons: write run metadata only after the expensive export (a
  `PosixPath` in `json.dump` cost a 35-minute run); measure latency on an idle
  machine; a CLI default must not silently override an artifact value.
- The superseded v1.0 artifact (45 MB) was removed from git history on the
  owner's request (filter-branch on the two commits before the final model,
  force push); the working tree did not change.

### 2026-09-24 (hardening) — v1.1.0
- Applied the final-hardening brief item by item (docs/HARDENING.md, D-30 …
  D-36): encoder manifest, network test split, fastText language gate, tracker
  expiry + hysteresis, 310-passage audit, taxonomy 1.2.0 (Science root cause),
  head comparison, seven off-topic detectors, CI, locked holdout + freeze.
- Locked evaluation ran twice: run 1 dropped Stack Exchange questions that
  belonged to both views (exclusive field). Repaired offline, re-frozen, run 2
  recorded with its reason; run 1 kept next to it.
- hsm accuracy 0.19 on the locked set: reported, not tuned.
- Repository renamed twice (ContexLens-NLP, then ContexLens); GitHub redirects
  the old names to the same refs. The HTTP User-Agent still names
  ContexLens-NLP because `contextlens/config.py` is in the frozen fingerprint.
- Benchmark rerun on the 1.2.0 corpus after the freeze, development splits
  only; it does not change the frozen selection.


### 2026-09-26 — console output for uncertain messages
- An `uncertain` turn printed `General topic : Books` and `Confidence : 53.0%`
  before its "uncertain" note, so off-topic input read like a confident wrong
  answer. The block now leads with `General topic : uncertain - no confident
  topic.`, then the reason and the rejected prediction as `Best guess : … - not
  used`; `history` marks such rows `uncertain (best guess: …)` and non-English
  rows `not English (not analysed)` (`tests/test_cli_render.py`). Console
  output only: model, thresholds and the frozen fingerprint are unchanged.
- `test_evaluate_and_decay_scripts_run` now skips when the git-ignored Stack
  Exchange set (`data/external/se_general_eval.jsonl`) is missing, like its
  sibling test; it failed on a fresh clone. Offline suite 197 passed, 4
  skipped; model suite 36 passed.
