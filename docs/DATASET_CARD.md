# Dataset card — ContextLens topic corpus v1

Why this data was chosen over existing datasets: [DATASET_RESEARCH.md](DATASET_RESEARCH.md).
Numbers below come from `data/manifest/crawl_stats.json`, `reports/eda.json` and
`reports/label_audit.json`.

## 1. Summary

| | |
|---|---|
| language | English |
| task | hierarchical topic classification: 1 of 8 general topics + 1–3 of 28 subtopics (multi-label, always children of the general topic) |
| unit | passage (12–60 words) from the lead and first sections of a Wikipedia article |
| size | **43,260 passages from 11,154 articles** — train 30,266 · validation 6,515 · test 6,479 |
| labels | distant supervision from Wikipedia's category graph (via DBpedia) |
| text | Wikipedia dumps pinned on the Hugging Face Hub (see §3) |
| external test sets | Stack Exchange questions (general + subtopic), out-of-taxonomy Wikipedia passages and Stack Exchange questions |
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
| Physics | 4,518 | 975 | 961 |
| Biology | 4,507 | 961 | 967 |
| Chemistry | 2,912 | 643 | 630 |
| Technology | 3,953 | 841 | 836 |
| Science | 3,229 | 688 | 681 |
| Books | 4,561 | 989 | 984 |
| Sports | 3,296 | 716 | 705 |
| History | 3,290 | 702 | 715 |

* General-topic imbalance (largest / smallest, all splits): **1.56**; subtopic
  imbalance: **1.90** (smallest: quantum computing and periodic table, whose
  category trees are small). Handled with `class_weight="balanced"`.
* Labels per passage: 1 subtopic 40,613 · 2 subtopics 2,596 · 3 subtopics 51.
* Crawl depth of the label: depth 0 17,942 · depth 1 23,406 · depth 2 1,912 passages.
* Every general topic and every subtopic occurs in every split
  (`tests/test_leakage.py`).

## 6. Quality and leakage checks (`reports/eda.json`)

| check | result |
|---|---|
| article overlap train/val/test | 0 / 0 / 0 |
| exact text overlap train–val, train–test | 0, 0 |
| near-duplicates (cosine ≥ 0.9) test→train after filtering | 0 (max cosine 0.869) |
| acceptance sentences of the brief in the corpus | 0 |
| OOD titles in the corpus · Stack Exchange titles in the corpus | 0 · 0 |
| empty texts · exact duplicates | 0 · 0 |
| residual markup (HTML tags · URLs · LaTeX fragments) | 9 · 7 · 16 passages — removed at load time by `normalize` (tags, URLs) or harmless |

**Manual label audit** (`reports/label_audit.json`, `scripts/label_audit.py`):
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
* **Distant labels.** About 4% wrong article labels in the audit sample, and
  some passages carry no topical signal on their own.
* **Snapshot age.** Text is from Nov 2023 (fallback Mar 2022); newer topics
  (e.g. 2024 AI products) are under-represented or missing.
* **Western/English-Wikipedia coverage bias** of the category graph (e.g.
  sports coverage is dominated by association football, basketball and the
  Olympics as defined by the taxonomy).
* **Science is a hard class by design**: it covers *science about science*
  (method, history, research practice), which overlaps lexically with every
  natural science.
