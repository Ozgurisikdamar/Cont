# Dataset research — why this data

This document answers one question: **which labelled English data can train and
honestly evaluate a hierarchical topic classifier for 8 general topics and 28
subtopics?** Every number below is either *measured* by
`scripts/dataset_research.py` (output: `reports/dataset_research.json`) or
*reported* from the official source page — the two are kept apart.

## 1. Requirements (derived from the brief)

| # | requirement | why it matters |
|---|---|---|
| R1 | English text | the whole system is English (input, output, data) |
| R2 | labels for **all 8 general topics** (physics, biology, chemistry, technology, science, books, sports, history) | a class without training examples cannot be predicted |
| R3 | labels for the **28 subtopics**, ideally more than one per text | the brief asks for hierarchical + multi-label output |
| R4 | enough examples per class (hundreds, not tens) | stable per-class metrics and thresholds |
| R5 | a licence that allows redistribution of derived data | the label manifest is committed to a public repository |
| R6 | reproducible acquisition (pinned revision, deterministic processing) | results must be re-creatable |
| R7 | a second, **independent** source of real user text | training and testing on the same source overstates quality |
| R8 | out-of-taxonomy text | the system must say "uncertain" for pizza, not guess "sports" |
| R9 | no fabricated or model-generated examples | zero-fabrication policy of the brief |

## 2. Search process

Sources searched: Hugging Face Hub, Kaggle, the UCI Machine Learning Repository,
Mendeley Data, MTEB benchmark tasks and the Wikipedia/DBpedia knowledge graph.
Five candidates with a downloadable evaluation split were **downloaded at a pinned
revision and measured**; four more were assessed from their official
documentation only (too large to download for a coverage check, or no raw text).

For every candidate the class inventory was mapped by hand onto our taxonomy
(`COVERAGE` in `scripts/dataset_research.py`). A class counts only when it is a
clean match: AG News' "Sci/Tech" is *not* counted as physics, biology or
chemistry because it cannot tell them apart.

## 3. Measured candidates

| dataset | source · revision | licence | rows (split measured) | classes | median words | dup. rate | general covered | subtopics covered |
|---|---|---|---:|---:|---:|---:|---:|---:|
| AG News | HF `fancyzhx/ag_news` @`eb185aa` | unknown (academic use) | 7,600 (test) | 4 | 37 | 0.0% | 2/8 (sports, technology) | 0/28 |
| 20 Newsgroups (UCI #113) | HF `SetFit/20_newsgroups` @`f1b9129` | not stated on mirror | 7,532 (test) | 20 | 82 | 3.1% (+2.9% empty) | 3/8 | 3/28 (hardware, software, astrophysics) |
| DBpedia-14 | HF `fancyzhx/dbpedia_14` @`9abd46c` | CC BY-SA 3.0 | 70,000 (test) | 14 | 49 | 0.0% | 2/8 (sports, books) | 0/28 |
| DBPedia Classes (Kaggle mirror) | HF `DeveloperOats/DBPedia_Classes` @`4d0aa96` | CC0 (mirror) | 36,003 (val) | 219 (L3) | 73 | 0.0% | 3/8 (sports, books, history) | 6/28 |
| News Category / HuffPost | HF `heegyu/news-category-dataset` @`304a05a` | CC BY 4.0 | 209,527 | 42 | 28 | 0.2% | 3/8 (science, technology, sports) | 0/28 |

## 4. Reported candidates (not downloaded)

| dataset | source | licence | size | classes | general covered | subtopics covered | note |
|---|---|---|---|---|---|---|---|
| Yahoo! Answers Topics | HF `community-datasets/yahoo_answers_topics` | unknown | 1.46M | 10 | 3/8 | 0/28 | "Science & Mathematics" mixes all sciences |
| Web of Science WOS-46985 | Mendeley Data 9rw3vkcfy4 | CC BY 4.0 | 46,985 | 7 / 134 | 3/8 | 5/28 | abstracts; no sports, books, history |
| arXiv metadata | Kaggle `Cornell-University/arxiv` | CC0 (metadata) | >2M | ~150 multi-label | 5/8 | 14/28 | research abstracts only; no sports, books, history |
| LSHTC (Wikipedia) | Kaggle competition | competition terms | >2M | 325,056 | n/a | n/a | only pre-tokenised feature ids — no raw text |
| Stack Exchange | HF `mteb/stackexchange-clustering` + api.stackexchange.com | CC BY-SA (user content) | ~75k titles, 121 sites | site + tags | 8/8 | 25/28 | real user questions — used for **evaluation** |
| Wikipedia + category graph (built here) | HF `wikimedia/wikipedia` 20231101.en + DBpedia SPARQL | CC BY-SA 3.0 / GFDL | 6.4M articles | any | 8/8 | 28/28 | used for **training** |

## 5. Finding

**No existing labelled dataset covers the taxonomy.** The best measured candidate
(DBPedia Classes) supplies 3 of 8 general topics and 6 of 28 subtopics; the best
reported one (arXiv) reaches 5/8 and 14/28 but has no sports, books or history and
contains only research abstracts. Combining datasets would not help either: their
label definitions do not align (e.g. AG News "Sci/Tech", Yahoo "Science &
Mathematics") and a merged set would mix text styles class by class, which lets a
model learn *the source* instead of *the topic*.

## 6. Decision

| role | data | why |
|---|---|---|
| **training / validation / test** | Wikipedia passages (pinned dump `20231101.en`, revision `b04c8d1`) labelled through Wikipedia's own category graph (via DBpedia) | the only source meeting R1–R6 for all 36 labels; editor-maintained categories are a real, independent labelling signal; CC BY-SA allows committing the label manifest |
| **external test (style shift)** | Stack Exchange question titles: MTEB clustering titles (site → general topic) and API questions (tag → subtopic) | real user-written questions (R7) from a source the model never sees in training |
| **out-of-taxonomy (OOD)** | Wikipedia articles from 12 unrelated categories (cooking, dance, finance, …) and 15 Stack Exchange sites (cooking, travel, pets, …) | measures whether "uncertain" is said when it should be (R8) |

Nothing was generated, paraphrased or hand-written for training (R9). The five
acceptance sentences of the brief are **never** in the training data; the EDA
script checks this (`leakage.acceptance_sentences_in_corpus` in `reports/eda.json`).

### How the labels are built (distant supervision)

Each subtopic has curated seed categories with a crawl depth
(`configs/taxonomy.json`, e.g. `Quantum_mechanics@2`). The category tree is
crawled through `skos:broader`, noisy branches are pruned by exclusion patterns
(fiction, people, awards, stubs, lists, …), and each article is labelled with the
general topic whose crawl reaches it at the **smallest depth**. Articles reached
at the same minimum depth by two general topics are **ambiguous and excluded**
(not guessed). Secondary subtopics are kept only when reached at depth ≤ 1, which
limits the drift that deep category crawls are known for. Full rules:
[DATASET_CARD.md](DATASET_CARD.md).

## 7. Known risks and how they are handled

| risk | effect | mitigation |
|---|---|---|
| category-graph label noise | some articles carry a weak or wrong label | depth limits, exclusion patterns, ambiguity exclusion, depth ≤ 1 for secondary labels; noise is measured in [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md) |
| style mismatch (encyclopedic vs. conversational) | Wikipedia accuracy overstates chat accuracy | Stack Exchange external test is reported separately and used for model selection (ext_dev half) |
| label snapshot newer than text snapshot | DBpedia reflects a newer Wikipedia than the Nov-2023 dump; renamed or new articles have no text | renamed articles are recovered through `dbo:wikiPageRedirects` aliases; articles created after the snapshot are dropped and counted in `data/manifest/crawl_stats.json` |
| live endpoints change | the crawl cannot be re-run bit-for-bit later | the label manifest (`data/manifest/articles.csv`, with text SHA-256) is committed; text comes from a pinned dump revision |
| licence | CC BY-SA share-alike | attribution in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md); derived data is released under the same terms |
