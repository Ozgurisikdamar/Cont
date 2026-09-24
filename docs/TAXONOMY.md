# Taxonomy — 8 general topics, 28 subtopics

Source of truth: [`configs/taxonomy.json`](../configs/taxonomy.json) (version 1.1.0 — 1.0.0 also seeded *quantum_computing* with Quantum information science@0; removed because it labelled entanglement physics as Technology, [DATASET_CARD.md §9](DATASET_CARD.md)).
Everything below is read from that file at runtime; nothing about the taxonomy is
hard-coded in Python. `contextlens.taxonomy.load_taxonomy()` validates it
(unique ids, every `broader` refers to an existing topic).

## 1. General topics

| general topic | role | broader | perspective | meaning of the role |
|---|---|---|---|---|
| **Physics** (`physics`) | domain | science | — | |
| **Biology** (`biology`) | domain | science | — | |
| **Chemistry** (`chemistry`) | domain | science | — | |
| **Technology** (`technology`) | domain | — | — | |
| **Science** (`science`) | domain | — | — | about science itself: method, history, research practice |
| **Books** (`books`) | format | — | — | a *medium* that wraps any domain |
| **Sports** (`sports`) | domain | — | — | |
| **History** (`history`) | domain | — | `history of {topic}` | can frame another domain |

`broader: science` does **not** make physics a child class of science in the
classifier — all eight are sibling classes. It is composition metadata: when
Science and Physics are both active in a conversation, Physics is the focus.

## 2. Subtopics and how their labels are obtained

Each subtopic lists Wikipedia seed categories as `Category@depth` — the category
tree is crawled down to that many levels (`skos:broader` in DBpedia). "Excl." is
the number of per-subtopic exclusion rules (named categories + regex patterns)
on top of the global exclusions. "Stack Exchange" is where the external test
questions come from: `site[tag]`, or the whole site when no tag is given.

| general | id | name | query phrase | seed categories | excl. | Stack Exchange |
|---|---|---|---|---|---:|---|
| physics | `quantum_mechanics` | Quantum Mechanics | quantum mechanics | Quantum mechanics@2 | 2 | physics[quantum-mechanics] |
| physics | `relativity` | Relativity | theory of relativity | Theory of relativity@2 | 1 | physics[general-relativity], physics[special-relativity] |
| physics | `astrophysics` | Astrophysics | astrophysics | Astrophysics@2 | 1 | physics[astrophysics] |
| physics | `classical_mechanics` | Classical Mechanics | classical mechanics | Classical mechanics@2 | 2 | physics[classical-mechanics], physics[newtonian-mechanics] |
| biology | `genetics` | Genetics | genetics | Genetics@2 | 1 | biology[genetics] |
| biology | `evolution` | Evolution | evolution | Evolutionary biology@2, Evolution@2 | 1 | biology[evolution] |
| biology | `cell_biology` | Cell Biology | cell biology | Cell biology@2 | 1 | biology[cell-biology] |
| biology | `ecology` | Ecology | ecology | Ecology@2 | 1 | biology[ecology] |
| chemistry | `organic_chemistry` | Organic Chemistry | organic chemistry | Organic chemistry@2 | 1 | chemistry[organic-chemistry] |
| chemistry | `chemical_reactions` | Chemical Reactions | chemical reactions | Chemical reactions@2 | 1 | chemistry[reaction-mechanism], chemistry[reaction-control] |
| chemistry | `periodic_table` | Periodic Table | periodic table of elements | Periodic table@2, Chemical elements@0 | 2 | chemistry[periodic-trends], chemistry[periodic-table], chemistry[elements] |
| technology | `quantum_computing` | Quantum Computing | quantum computing | Quantum computing@2, Quantum gates@1 | 2 | quantumcomputing |
| technology | `artificial_intelligence` | Artificial Intelligence | artificial intelligence | Artificial intelligence@2, Machine learning@1 | 2 | ai |
| technology | `software` | Software | software | Software engineering@2, Software@1 | 1 | softwareengineering |
| technology | `hardware` | Hardware | computer hardware | Computer hardware@2 | 1 | superuser[cpu], superuser[motherboard] |
| science | `scientific_method` | Scientific Method | scientific method | Scientific method@1, Philosophy of science@0 | 2 | philosophy[scientific-method], hsm[scientific-method] |
| science | `history_of_science` | History of Science | history of science | History of science@2 | 0 | hsm |
| science | `scientific_research` | Scientific Research | scientific research | Research@0, Metascience@1, Research methods@0, Design of experiments@0, Scientific misconduct@1, Peer review@0, Open science@0, Research ethics@0, Research and development@0, Academic publishing@0 | 1 | academia[research-process] |
| books | `novels` | Novels | novels | Novels@1 | 0 | — |
| books | `science_books` | Science Books | popular science books | Science books@1, Popular science books@1 | 1 | — |
| books | `poetry` | Poetry | poetry | Poetry@1, Genres of poetry@1, Poems@2, Poetics@1 | 1 | literature[poetry] |
| books | `authors` | Authors | authors and writers | Novelists@2, Writers@2 | 0 | — |
| sports | `football` | Football (Soccer) | association football | Association football@2 | 0 | sports[football] |
| sports | `basketball` | Basketball | basketball | Basketball@2 | 0 | sports[basketball] |
| sports | `olympics` | Olympics | Olympic Games | Olympic Games@2 | 0 | sports[olympics] |
| history | `ottoman_history` | Ottoman History | Ottoman Empire history | History of the Ottoman Empire@2, Ottoman Empire@2 | 0 | history[ottoman-empire] |
| history | `world_wars` | World Wars | World War I and World War II | World War I@2, World War II@2 | 1 | history[world-war-two], history[world-war-one] |
| history | `ancient_history` | Ancient History | ancient history | Ancient history@2 | 0 | history[ancient-history], history[ancient-rome], history[ancient-greece] |

Three subtopics (novels, science books, authors) have no Stack Exchange tag with
a clean mapping, so the external **subtopic** benchmark covers 25/28 subtopics.
The external **general-topic** benchmark covers all eight (Literature SE → books).

### Why these seeds

Seeds were chosen by inspecting the crawl, not by name alone. Examples of what the
inspection found and fixed (details in [decisions.md](../decisions.md)):

* A deep crawl of `Research` produced many off-topic articles; it was replaced by
  ten precise depth-0/1 seeds about research practice (methods, peer review,
  metascience, research ethics, …).
* `Periodic_table@2` alone reached songs and paintings named after elements
  (e.g. a "herring earring" article); `Chemical_elements@0` plus exclusions fixed it.
* `Classical_mechanics@2` reached acoustics and musical instruments; excluded.
* `Artificial_intelligence@2` reached AI in fiction and films; global
  exclusions (`fiction`, `films`, `television`, …) remove those branches.

Global exclusions (`global_exclude_category_patterns`) remove maintenance and
non-topical categories everywhere: stubs, lists, templates, portals, images,
disambiguation, people, awards, popular culture, fiction and media works.
Scientists are removed from science subtopics (`science_person_exclude_patterns`)
because a biography is about a person, not about the field.

## 3. Label rules (hierarchical + multi-label)

1. **General topic** = the general topic whose crawl reaches the article at the
   **smallest depth**. A tie between two general topics → the article is
   *ambiguous* and excluded (831 articles; listed in
   `data/manifest/ambiguous_titles.txt`).
2. **Subtopics** = every subtopic of that general topic reaching the article at
   the minimum depth, plus others of the same general topic reached at depth ≤ 1.
   A passage therefore has 1–4 subtopics, always children of its general topic
   (the hierarchy is consistent by construction; `Taxonomy.is_consistent`).
3. Every passage inherits the labels of its article.

## 4. Theme composition (conversation level)

The conversation tracker keeps a decayed score per topic; the active topics are
turned into one phrase by `contextlens.services.composer`, driven only by the
metadata above plus four templates in `taxonomy.json → composition`
(every row is a test in `tests/test_composer.py`):

| active topics | rule | phrase |
|---|---|---|
| Physics (+ Quantum Mechanics dominant) | single topic → focus subtopic | quantum mechanics |
| Books + Science | format wraps domain: `{domain} {format}` | science books |
| Books + Science + Biology | biology is narrower than science: `{broader} {format} about {narrow}` | science books about biology |
| Books + History | format wraps domain | history books |
| Science + Biology | narrowing without a format: `{narrow} ({broader})` | biology (science) |
| History + Physics | History is a perspective: `history of {topic}` | history of physics |
| Sports + Technology | none of the above: `{a} and {b}` | sports and technology |

## 5. Out of taxonomy

Text that belongs to none of the eight topics must be reported as *uncertain*.
Evaluation sets (never used for training):

* Wikipedia categories — validation: Cooking techniques, Music genres, Fashion,
  Tourism, Dog breeds, Gardening; test (disjoint): Cuisine, Painting, Dance,
  Cars, Personal finance, Beer styles (`Automobiles` was replaced by
  `Cars`: the former category is empty in DBpedia, the renamed one has 686 articles).
* Stack Exchange sites — cooking, travel, pets, gardening, bicycles, coffee,
  beer, parenting, money, woodworking, music, photo, fitness, diy, movies.

Borderline by design: `fitness` and `bicycles` are close to sports; they measure
how the OOD gate behaves near the boundary, not far from it.
