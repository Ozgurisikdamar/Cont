# Dataset card — ContextLens topic corpus (taxonomy 1.2.0)

Why this data was chosen over existing datasets: [DATASET_RESEARCH.md](DATASET_RESEARCH.md).
Numbers below come from `data/manifest/crawl_stats.json`, `reports/eda.json` and
`reports/label_audit_v2.json` (the 48-passage v1.0 audit is kept in `reports/label_audit.json`).

## 1. Summary

| | |
|---|---|
| language | English |
| task | hierarchical topic classification: 1 of 8 general topics + 1–3 of 28 subtopics (always children of the general topic; 94.6% of passages carry exactly one subtopic) |
| unit | passage (12–60 words) from the lead and first sections of a Wikipedia article |
| size | **40,112 passages from 10,344 articles** — train 28,075 · validation 5,984 · test 6,053 (taxonomy 1.2.0; v1.1 had 42,942 / 11,073, v1.0 43,260 / 11,154, see §9–§10) |
| labels | distant supervision from Wikipedia's category graph (via DBpedia) |
| text | Wikipedia dumps pinned on the Hugging Face Hub (see §3) |
| external sets | development: Stack Exchange ext_dev, CLINC150 dev chat, Tatoeba dev, out-of-taxonomy Wikipedia val; **locked holdout** (§11): unseen Wikipedia passages, 2026 Stack Exchange questions, CLINC150 test, Tatoeba locked half; legacy (seen in v1.0 development): test, ext_test |
| licence | CC BY-SA (Wikipedia text: CC BY-SA 3.0 / GFDL; Stack Exchange content: CC BY-SA) — see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
| build | `python scripts/download_data.py --all` (deterministic given the cached/pinned sources) |

## 2. Label construction

1. For every subtopic, seed categories with a crawl depth (`configs/taxonomy.json`)
   are expanded through `skos:broader`; branches matching exclusion patterns
   (stubs, lists, people, fiction, media works, awards, …) are pruned.
2. Articles are collected with `dct:subject`; list/index/outline/timeline/
   glossary pages and disambiguations are dropped.
3. **General topic** = the one whose crawl reaches the article at the smallest
   depth. **Ties are excluded** (831 ambiguous articles, listed in
   `data/manifest/ambiguous_titles.txt`).
4. **Subtopics** = subtopics of that general topic at the minimum depth, plus
   others of the same general topic reached at depth ≤ 1.
5. **Selection** — at most 450 articles per (primary) subtopic, shallowest
   first, ties broken by SHA-1 of the title: 132,746 labelled → 11,937 selected.

Per-subtopic crawl statistics (categories, excluded categories, articles per
depth) are in `data/manifest/crawl_stats.json → crawl_stats`.

## 3. Text acquisition

| step | source | articles |
|---|---|---:|
| exact title | `wikimedia/wikipedia` `20231101.en` @ `b04c8d1` | 10,270 in-taxonomy |
| redirect alias (renamed after the snapshot) | same dump | 43 |
| exact title | `legacy-datasets/wikipedia` `20220301.en` @ `97a0b05` | 846 |
| redirect alias | same dump | 10 |
| **total with text** | | **11,169 of 11,937 (93.6%)** |

The 2023 dump lacks some long-standing articles entirely (e.g. "Gold",
"Spacetime", "Quantum chromodynamics"; verified by scanning all 41 shards), so
the 2022 snapshot is used as a fallback. The 768 selected articles without text
are mostly articles created after March 2022; they are listed in
`crawl_stats.json → missing_text`. 15 articles yielded no usable passage.
The source of every article is recorded in `articles.csv → text_source`
(passages: 39,750 from the 2023 dump, 162 via 2023 redirects, 3,309 from the
2022 dump, 39 via 2022 redirects).

## 4. Passages

* Lead-first; reading stops at the first back-matter section (*See also*,
  *References*, *External links*, *Further reading*, *Notes*, *Bibliography*, …);
  headings and lines under 6 words (captions, list fragments) are dropped.
* Sentence splitting protects abbreviations and initials; sentences are packed
  into passages of **12–60 words**, at most **4 per article** (mean 3.88).
* Exact duplicates across articles removed (38); validation/test passages with a
  TF-IDF cosine ≥ 0.9 to any training passage removed (12) — templated sibling
  articles such as "A period *n* element is …".
* Length (words): min 12 · median 23 · mean 25.5 · p95 47 · max 60.
  Stack Exchange titles: median 10 words — the style gap the external test measures.

## 5. Splits

Grouped by article (no article has passages in two splits), stratified by
(general topic, first subtopic), 70 / 15 / 15, deterministic.

| general topic | train | val | test |
|---|---:|---:|---:|
| Physics | 4,420 | 955 | 945 |
| Biology | 4,515 | 961 | 971 |
| Chemistry | 2,912 | 643 | 630 |
| Technology | 3,630 | 774 | 744 |
| Science | 2,388 | 463 | 465 |
| Books | 4,569 | 989 | 988 |
| Sports | 2,675 | 589 | 652 |
| History | 2,966 | 610 | 658 |

* General-topic imbalance (largest / smallest, all splits): **1.97** (Science
  is smallest after the 1.2.0 redefinition); subtopic imbalance: **3.67**
  (smallest: quantum computing 539, history of science 595; largest 1,978).
  Handled with `class_weight="balanced"`.
* Labels per passage: 1 subtopic 37,958 · 2 subtopics 2,099 · 3 subtopics 55.
* Crawl depth of the label: depth 0 16,939 · depth 1 21,101 · depth 2 2,072 passages.
* Every general topic and every subtopic occurs in every split
  (`tests/test_leakage.py`).

## 6. Quality and leakage checks (`reports/eda.json`)

| check | result |
|---|---|
| article overlap train/val/test | 0 / 0 / 0 |
| exact text overlap train–val, train–test | 0, 0 |
| near-duplicates (cosine ≥ 0.9) test→train after filtering | 0 (max cosine 0.870) |
| acceptance sentences of the brief in the corpus | 0 |
| OOD titles in the corpus · Stack Exchange titles in the corpus | 0 · 0 |
| empty texts · exact duplicates | 0 · 0 |
| residual markup (HTML tags · URLs · LaTeX fragments) | 8 · 7 · 14 passages — removed at load time by `normalize` (tags, URLs) or harmless |

**v1.0 manual label audit** (superseded by the 310-passage audit in §10; `reports/label_audit.json`, `scripts/label_audit.py`):
48 training passages (6 per general topic, `random_state=42`), each read with
its article title:

| verdict | count |
|---|---:|
| correct | 43 (89.6%) |
| weak passage — article label right, passage carries no topical signal | 3 (6.2%) |
| wrong label — article belongs to another general topic | 2 (4.2%) |

The two wrong labels are boundary cases of the category graph
(*Amyloid (mycology)* filed under chemical reactions; *Spin model* under quantum
computing). With 48 items the 95% Wilson interval of the wrong-label rate is wide (1.2%–14.0%);
the audit shows the order of magnitude, not a precise rate.

## 7. External evaluation sets (never trained on)

| set | source | size | use |
|---|---|---|---|
| Stack Exchange general | MTEB `StackExchangeClustering` titles @ `9006e01`, site → general topic (11 sites) | ext_dev 10,124 · ext_test 10,087 (≤ 1,500 per topic per split) | general topic on real questions |
| Stack Exchange subtopic | Stack Exchange API, questions tagged per subtopic (`site[tag]` in TAXONOMY.md) | ext_dev 1,630 · ext_test 1,642, 25 subtopics | subtopics on real questions |
| Stack Exchange OOD | 15 unrelated sites (cooking, travel, pets, …) | 6,000 per split | out-of-taxonomy detection |
| Wikipedia OOD | 12 unrelated categories, 6 for validation, 6 disjoint for test | see §8 | out-of-taxonomy detection |

Splits of the external sets are a deterministic 50/50 hash. `se_general_eval.jsonl`
is not committed (regenerated from the pinned MTEB revision); the subtopic set
is committed because the live API changes.

## 8. Out-of-taxonomy Wikipedia passages

Categories — validation: Cooking techniques, Music genres, Fashion, Tourism,
Dog breeds, Gardening; test: Cuisine, Painting, Dance, Cars, Personal finance,
Beer styles. At most 60 articles per category, 2 passages per article, articles
reachable from any taxonomy seed excluded.

## 9. Known limitations

* **Encyclopedic register.** Passages are descriptive third-person prose;
  users write questions and opinions. Measured by the Stack Exchange sets.
* **Distant labels.** 5.2% wrong [3.2, 8.2] and 20.3% weak labels in the
  310-passage audit (§10); some passages carry no topical signal on their own.
* **Snapshot age.** Text is from Nov 2023 (fallback Mar 2022); newer topics
  (e.g. 2024 AI products) are under-represented or missing.
* **Western/English-Wikipedia coverage bias** of the category graph (e.g.
  sports coverage is dominated by association football, basketball and the
  Olympics as defined by the taxonomy).
* **Science is a hard class by design**: it covers the scientific enterprise
  itself (method, philosophy, research practice, the history of science as
  such), which overlaps lexically with every natural science (D-32).

## 9b. Version 1.1 — label fix found by error analysis

The first trained model classified the brief's acceptance sentence *"In quantum
entanglement, the wave functions of particles can change together."* as
Technology > Quantum Computing. The cause was a label rule, not the model: the
seed `Quantum_information_science@0` of *quantum_computing* contains
foundational physics (Quantum entanglement, Bell's theorem, Bell states,
cat states…) at depth 0, where the Quantum-mechanics tree reaches them only at
depth ≥ 1, so the minimum-depth rule labelled them Technology.

Taxonomy 1.1.0 removes that seed. `scripts/relabel_corpus.py` re-ran the label
rules on the cached crawl and applied them to the existing articles, keeping
each article's split (`reports/relabel_taxonomy_1.1.0.json`; the 1.2.0 relabel is `reports/relabel_taxonomy_1.2.0.json`, decisions.md D-32):

| change | articles |
|---|---:|
| Technology → Physics (e.g. Quantum entanglement, Bell's theorem, Bell state) | 49 |
| Technology → Technology, other subtopic (hardware) | 6 |
| removed (only reached through the dropped seed; e.g. Density matrix, LOCC) | 81 |
| total (train 91 · val 19 · test 26) | 136 of 11,154 |

The acceptance sentence itself was **not** added to the data. Articles that a
from-scratch build with taxonomy 1.1 would additionally select (previously
ambiguous physics/technology ties) were not added — that needs the full dump
download; a rebuild with `download_data.py --all` may therefore differ from the
committed corpus by those articles. (v1.0 frozen-encoder benchmark on v1.0 labels; model v1.0 on v1.1 labels.)

## 10. Taxonomy 1.2.0 — Science redefined, 310-passage label audit (D-32)

**Audit** (`scripts/label_audit_v2.py`, `reports/label_audit_v2.json`): 10
training passages per subtopic plus 5 more for six weak subtopics (scientific
method, scientific research, history of science, quantum mechanics, quantum
computing, authors) = 310 passages, judged against the passage text and the
article title. Every verdict is stored with the labels it judged.

| verdict | n | rate | 95% Wilson CI |
|---|---:|---:|---:|
| correct | 231 | 74.5% | 69.4–79.1% |
| weak (label defensible, passage carries little signal or the article is borderline) | 63 | 20.3% | 16.2–25.2% |
| incorrect (wrong general topic) | 16 | 5.2% | 3.2–8.2% |

Science was the worst general topic (8 of 45 incorrect, CI 9.3–31.3%). The
auditor is the agent that built the corpus, not an independent annotator, so
this is a lower bound on disagreement.

**Root cause of the weak Science class:** `History_of_science@2` reached the
history of physics, biology, chemistry, astronomy and mathematics, natural
history, museums and instruments; `Research_and_development@0` and
`Academic_publishing@0` brought business and publishing articles. Science is
now the scientific enterprise itself; the seeds were narrowed and every
systematic error of the audit was traced to its category and excluded
([TAXONOMY.md](TAXONOMY.md)). 7 of the 16 incorrect items were removed or moved
to the right topic; the other 9 sit directly in a seed category (e.g.
*Supersymmetry* and *Ice age* are filed in `Category:History_of_science`) and
are documented, not special-cased.

**Relabel** (`scripts/relabel_corpus.py`, `reports/relabel_taxonomy_1.2.0.json`,
splits kept): 883 articles changed, 730 removed (Science 313, Sports 207,
History 125, Physics 85); 10,358 articles / 40,112 passages remain (10,344
articles after the passage filters).

## 11. Evaluation data: development vs. locked holdout (D-36)

| set | size | role |
|---|---|---|
| Wikipedia `val` | 5,984 | development (every choice) |
| Stack Exchange `ext_dev` | 10,124 general · 6,000 off-topic · 1,630 subtopic | development |
| CLINC150 `dev` (`data/external/ood_conversational.jsonl`) | 2,900 assistant-chat utterances; `train` half (14,500) only for detectors that learn from off-topic text | development |
| Tatoeba `dev` (`language_eval.jsonl`) | 18 languages, 1/2/3-word prefixes + full sentences | development (language gate) |
| **Wikipedia locked** (`data/locked/wiki_locked.jsonl`) | 374 passages, first passage of articles in no corpus version; 27 subtopics × 14–15 (periodic table: none usable) | locked |
| **Stack Exchange locked** (`data/locked/se_locked.jsonl`) | 2,713 questions created 2026-01-01..2026-09-20: 1,623 in-domain + 1,029 off-topic in the general view (labelled by site), 1,212 in the subtopic view | locked |
| **CLINC150 `locked`** | 4,350 utterances (CLINC test) | locked |
| **Tatoeba `locked`** | the other hash half | locked |
| Wikipedia `test`, Stack Exchange `ext_test` | 6,053 · 10,087 | legacy — computed during v1.0 development, reported as *seen* |

CLINC150: the in-scope intents are used except six that can be topical (fun
facts, definitions, the meaning of life, vaccines, unit conversion); CLINC's own
`oos` class is not used because it contains physics, biology and sports
questions (`scripts/build_ood_conversational.py`). The locked Wikipedia text
comes from the same pinned dump as the corpus (the MediaWiki API answered HTTP
429 on the first request). `data/locked/MANIFEST.json` holds the SHA-256 of
every locked file; `reports/locked/FREEZE.json` pins the configuration that
was evaluated.
