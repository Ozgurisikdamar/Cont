# Known issues

Every measurement below comes from `reports/evaluation.json`,
`reports/experiments/*.json` or `reports/label_audit.json`. Severity:
**high** = visible to most users, **medium** = visible in some situations,
**low** = cosmetic or operational.

| # | issue | severity | cause | impact (measured) | attempted | recommended |
|---|---|---|---|---|---|---|
| 1 | **Off-topic text often gets a topic** | high | the model has only seen the 8 topics; the gate is a distance rule with no off-topic training data | only 33.3% of out-of-taxonomy Wikipedia passages and 43.6% of off-topic questions are answered "uncertain" (AUROC 0.850 / 0.845); e.g. pizza → Books 0.77, caught by the centroid gate only | MSP, energy, centroid cosine, kNN (E-9); threshold calibrated on questions; confidence floor 0.50 | collect real off-topic messages and train a ninth "other" class or a dedicated detector; show the confidence prominently |
| 2 | **"Science" is weak** | high | the class means *science about science*; its texts talk about physics, biology … | F1 0.661 (Wikipedia), 0.369 (questions); 43% of history-of-science questions → Physics | none beyond the benchmark | merge into a "Science (general)" perspective like History, or allow two general topics per message |
| 3 | **Short conversational messages** | medium | training text is encyclopedic | accuracy 0.737 for questions ≤ 7 words vs 0.835 on Wikipedia passages | fine-tuning, selection on real questions (ext_dev) | human-labelled chat messages for fine-tuning and testing |
| 4 | **Label noise** | medium | distant supervision through Wikipedia's category graph | manual audit of 48 passages: 4.2% wrong, 6.2% weak | minimum depth, ties excluded, depth ≤ 1 secondaries; seed fix D-27 | larger audit; relabel the confident-error list in `evaluation.json` |
| 5 | **Quantum Computing is small** | medium | removing the `Quantum_information_science` seed (D-27) moved 49 articles to Physics and dropped 81 | 539 passages; subtopic F1 0.58; 0.650 accuracy on quantum-computing questions | relabelled, fine-tuned again | add curated seeds (e.g. quantum algorithms, qubit hardware) |
| 6 | **Weak subtopics** | medium | overlapping definitions | history of science 0.41, scientific method 0.50, scientific research 0.56, quantum mechanics 0.57, authors 0.57 (Wikipedia F1) | H1/H2/H3 heads, threshold sweep (E-7) | more distinctive seeds; allow "general only" answers when no subtopic is confident |
| 7 | **Theme may carry a stale topic** | medium | exponential decay keeps a topic for a few turns | simulated conversations: theme accuracy 0.757 (val), switch lag 1.34 turns, 0.537 extra topics on average; a one-message tangent moves the top topic 67% of the time (tangent robustness 0.334) | grid over decay × minimum share (D-28) | a "new topic" detector that resets decay on a sharp switch |
| 8 | **English only** | medium | English training data and encoder | non-English text is answered "uncertain" by the language gate; a sentence with ≥ 40% English-looking words can still be classified | language gate (D-29) | a language identifier; multilingual encoder if needed |
| 9 | **Web search depends on free endpoints** | medium | Wikipedia and DuckDuckGo have no key-less SLA | HTTP 429 from Wikipedia seen from cloud IPs; the app then uses DuckDuckGo or shows no results | fallback, 72 h cache, 5-minute circuit breaker | a keyed provider for production use |
| 10 | **History topic coverage is narrow** | low | seeds: Ottoman history, world wars, ancient history | not measured; other histories can only be mapped to the nearest seeded subtopic | – | add seeds per region / period |
| 11 | **Stack Exchange sets are question titles, not chats** | low | no public chat corpus with these labels; three subtopics have no site | external numbers approximate real use; subtopic set has 25 of 28 labels | – | build a small annotated chat set |
| 12 | **Repository history holds an extra 45 MB** | low | an intermediate model (labels v1.0) was committed before the label fix | clone size ~45 MB larger than needed | – | rewrite history (needs a force push; owner's decision) |
| 13 | **Latency depends on machine load** | low | CPU inference | median 19.9 ms idle; one test run measured 0.52 s while fine-tuning used all cores | – | none for a single-user app |
| 14 | **E-6 (larger fine-tuned encoder) not run** | low | owner decided to proceed with MiniLM | unknown whether a fine-tuned e5/bge would be better on questions | – | fine-tune e5-small-v2 with the same script and compare on ext_dev |
