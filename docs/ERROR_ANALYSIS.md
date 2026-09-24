# Error analysis

Where the final model (v1.1.0) fails, why, and what was done about it. Numbers
come from the **locked holdout** (`reports/locked/results.json`, run 2,
evaluated once after the freeze — decisions.md D-36) unless a line says
*dev* (`reports/evaluation_dev.json`) or names an experiment file. The locked
sets are small (374 Wikipedia passages, 14–60 per class; 1,623 Stack Exchange
questions), so single-class numbers carry wide uncertainty.

## 1. Errors found during development and fixed at the source

| found by | error | fix | evidence |
|---|---|---|---|
| brief sentence | *"In quantum entanglement, …"* → Technology > Quantum Computing (0.61) | seed `Quantum_information_science` removed, corpus relabelled (136 articles) | D-27 |
| acceptance tests | non-English text answered with confidence | language gate (lexicon), then fastText lid.176 per length | D-29, D-30 |
| audit | Science pulled in every field's history and R&D / publishing pages | Science redefined as the scientific enterprise; taxonomy 1.2.0; 883 passages relabelled, 730 removed | D-32, `reports/relabel_taxonomy_1.2.0.json` |
| 310-passage audit | 16 wrong labels (5.2%) | 7 removed by category rules; 9 filed directly in a seed category remain | `reports/label_audit_v2.json` |
| locked run 1 | the locked Stack Exchange view dropped questions that belong to both views | independent flags, offline repair, run 2 with a written reason | D-36 |

The acceptance sentences were never added to the data; every fix changed a
label rule or a gate, and the model was retrained from the rules.

## 2. General topic: which classes fail

| class | Wikipedia F1 (P / R) | questions F1 (P / R) | n (wiki / questions) |
|---|---:|---:|---:|
| Sports | 0.933 (0.933 / 0.933) | 0.765 (0.660 / 0.912) | 45 / 34 |
| Books | 0.884 (0.826 / 0.950) | 0.760 (0.702 / 0.828) | 60 / 128 |
| Technology | 0.876 (0.886 / 0.867) | 0.847 (0.837 / 0.857) | 45 / 300 |
| History | 0.870 (0.851 / 0.889) | 0.749 (0.849 / 0.670) | 45 / 109 |
| Physics | 0.867 (0.925 / 0.817) | 0.866 (0.860 / 0.873) | 60 / 653 |
| Chemistry | 0.831 (0.771 / 0.900) | 0.831 (0.803 / 0.861) | 30 / 180 |
| Biology | 0.789 (0.878 / 0.717) | 0.785 (0.795 / 0.775) | 60 / 120 |
| **Science** | **0.656** (0.625 / 0.690) | **0.241** (0.322 / 0.192) | 29 / 99 |

**Science on questions is still the weak point, and the reason is the data,
not the model.** All 99 locked Science questions come from the history of
science site (hsm); **42.4% are predicted Physics**, 15.2% Books, 14.1%
Technology. They ask about the history *of one field* (*"Why Euler didn't
include viscosity in equations of fluid dynamics?"* → Physics, 0.96). After
D-32 Science means the scientific enterprise (method, research practice,
history of science as a discipline), so a question about Euler's fluid
dynamics is arguably Physics. hsm is therefore reported separately: without
it the questions reach accuracy 0.843 / macro-F1 0.818 (7 classes).

**Biology has low recall on Wikipedia (0.717):** 8.3% go to History (forests,
regions — *"Hambach Forest is an ancient forest …"*), 6.7% to Science and 6.7%
to Chemistry (proteins: *"Angiogenin (ANG) … is a small 123 amino acid
protein"* → Biology is the prediction, Chemistry the label).

## 3. Confusion pairs

| Wikipedia (true → predicted) | share of true class |
|---|---:|
| Science → Books | 13.8% |
| Biology → History | 8.3% |
| Biology → Science / Chemistry | 6.7% each |
| Physics → Technology | 6.7% |
| Technology → Science | 6.7% |

| questions | share of true class |
|---|---:|
| Science → Physics | 42.4% |
| Science → Books | 15.2% |
| Science → Technology | 14.1% |
| History → Books / History → Science | 10.1% each |
| Technology → Physics | 9.0% |
| Chemistry → Physics | 7.8% |

The pairs are semantically adjacent, not random: a book about history, the
history of physics, AI applied to physics. A single general label is sometimes
a forced choice (KNOWN_ISSUES.md #2).

## 4. Style gap: encyclopedia → questions

| input | accuracy |
|---|---:|
| Wikipedia, 1–15 / 16–25 / 26–40 / > 40 words | 0.915 / 0.818 / 0.853 / 0.879 |
| questions, 1–7 / 8–12 / > 12 words | 0.780 / 0.809 / 0.812 |

General macro-F1 drops from 0.838 (Wikipedia) to 0.730 (questions; 0.818
without hsm). Short questions remain the hardest bucket. By site: software
engineering 0.92, AI 0.91, sports 0.91, physics 0.88, chemistry 0.86,
literature 0.83, astronomy 0.81, biology 0.78, quantum computing 0.74,
history 0.67, hsm 0.19.

## 5. Confident mistakes

The most confident errors (`wiki_errors`, `se_general_errors`) fall into three
groups:

1. **Arguable or noisy labels** — *"Numerical Recipes is the generic title of
   a series of books on algorithms …"* labelled Books, predicted Technology
   (0.92); *"Interdigitation is the interlinking of biological components …"*
   labelled Science, predicted Biology (0.96). The audit estimates 5.2% wrong
   [3.2, 8.2] and 20.3% weak labels.
2. **Physics in everyday settings** — *"Understanding the behaviour of a cue
   ball in billiards after it strikes another ball"* and *"Maximum horizontal
   distance obtained by throwing ball from a height"* (physics site) →
   Sports (0.97). The words are about sport; the question is mechanics.
3. **Site ≠ topic** — *"Options to improve current visual filter"* (physics
   site) → Technology (0.98); *"Was there a MIX (assembly language) user
   group"* (hsm) → Technology (0.98). The label comes from the site, and the
   text genuinely reads as another topic.

## 6. Subtopics

Primary subtopic + sibling suggestions (not a multi-label classifier, D-33).

| set | micro-F1 | macro-F1 (supported) | P@1 | R@3 |
|---|---:|---:|---:|---:|
| Wikipedia | 0.659 | 0.664 (25 labels) | 0.668 | 0.853 |
| … when the general topic is right | 0.782 | 0.784 | 0.789 | 0.966 |
| questions | 0.603 | 0.507 (24 labels) | 0.624 | 0.830 |

Hardest on Wikipedia (14–17 passages each): classical mechanics 0.33,
evolution 0.42, cell biology 0.43, history of science 0.44, relativity 0.46,
quantum mechanics 0.50. Best: novels 0.93, hardware 0.90, ottoman history /
football / basketball 0.83–0.84. On questions history of science reaches 0.17
(99 questions — the hsm effect again), while the large physics subtopics are
reasonable (relativity 0.71, classical mechanics 0.74, 172–199 questions).
Several question subtopics have fewer than 10 examples (olympics 1, periodic
table 2, basketball 2) and their F1 carries no information.
Passages with two gold subtopics are not harder at the general level (11 of 11
correct, a tiny sample). Only 5.7% of the multi-label validation passages get
their second gold subtopic as a suggestion (*dev*,
`reports/experiments/head_comparison.json`).

## 7. Off-topic and non-English text

| input (locked) | flagged | by distance | by low confidence | by language |
|---|---:|---:|---:|---:|
| CLINC150 assistant chat (4,350) | 78.1% | 75.5% | 33.8% | 0.2% |
| off-topic Stack Exchange questions (1,029) | 48.5% | 40.4% | 25.6% | 0.6% |
| in-domain questions (1,623) | 9.3% | 4.0% | 6.5% | 0.5% |
| in-domain Wikipedia passages (374) | 4.0% | 0.0% | 3.7% | 0.3% |

(The rules overlap; a text can be flagged by several.) The Mahalanobis gate
works on short text — chat and questions — and flags almost no encyclopedia
passage, in-domain or not: on the *dev* out-of-taxonomy Wikipedia passages it
flags 0.9% and the confidence floor catches most of the 23.9% that are
flagged. So *"I'm going to order pizza tonight."* is now caught by distance
(v1.0 caught it only with the centroid gate), but a paragraph about cooking
techniques is usually mapped to the nearest topic. The flagged in-domain
answers are mostly wrong ones (accuracy 0.53 on Wikipedia, 0.40 on questions,
vs 0.86 / 0.84 for the answered ones), so the gate still does useful work on
in-domain text.

Language gate (Tatoeba locked half): non-English rejected 48% at one word,
76% at two, 93% at three, 97% for full sentences; English accepted ≥ 99.6% at
every length. *"guten tag"* passes the language gate and is flagged by the
topic gates.

## 8. What was not done

* No data augmentation (paraphrasing, back translation): it would not address
  the dominant errors (site-defined labels, overlapping classes, style gap),
  and D-17 rules out synthetic training data.
* No human-labelled chat data: the external sets are question titles and
  assistant commands (CLINC150).
* No tuning on the locked results: every weakness above is reported as
  measured (the hsm number was not "fixed" after it was seen).
