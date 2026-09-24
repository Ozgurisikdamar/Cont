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

## 4. How to continue

```bash
. .venv/bin/activate
pytest -q                          # offline suite
pytest -q -m model                 # needs models/contextlens-topic
python scripts/report_tables.py    # regenerate reports/tables.md
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
