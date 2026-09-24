# Error analysis

Where the final model fails, why, and what was done about it. Numbers come
from `reports/evaluation.json` (sections `analysis`, `wiki_test_errors`,
`se_general_errors`, per-class reports) unless stated otherwise.

## 1. One error found and fixed: entanglement was "Technology"

The first trained model answered **Technology > Quantum Computing (0.61)** for
the brief's sentence *"In quantum entanglement, the wave functions of particles
can change together."* The model was right about its data: the corpus labelled
"Quantum entanglement", "Bell's theorem", "Bell state", "Cat state" and ~45
similar articles as Technology, because the seed category
`Quantum_information_science` of *quantum_computing* contains them at depth 0.

Fix (D-27): the seed was removed and the corpus relabelled from the cached
crawl (136 of 11,154 articles). After fine-tuning and training again the same
sentence is **Physics > Quantum Mechanics (0.95 / 0.74)** while *"Quantum
processors can speed up certain algorithms by using qubits."* stays
**Technology > Quantum Computing (0.97 / 0.63)**. The sentence was never added
to the data. This is the train → evaluate → analyse → fix → retrain loop the
brief asks for; its price is a smaller *quantum_computing* class (539 passages).

Two further problems were found the same way: non-English text was answered
with confidence (fixed by the language gate, D-29) and the decay-tuning probe
measured classifier errors instead of the decay (D-28).

## 2. General topic: which classes fail

| class | Wikipedia test F1 | Stack Exchange ext_test F1 (P / R) |
|---|---:|---:|
| Sports | 0.912 | 0.927 (0.929 / 0.925) |
| Chemistry | 0.881 | 0.813 (0.834 / 0.792) |
| Biology | 0.859 | 0.827 (0.870 / 0.789) |
| Technology | 0.856 | 0.854 (0.851 / 0.856) |
| History | 0.846 | 0.776 (0.879 / 0.694) |
| Physics | 0.842 | 0.716 (0.629 / 0.833) |
| Books | 0.814 | 0.766 (0.719 / 0.820) |
| **Science** | **0.661** | **0.369** (0.414 / 0.333) |

**Science is the weak class, and it is weak by definition.** It means *science
about science* (method, history of science, research practice), so its texts
talk about a field: *"What did Einstein learn in his university E&M courses?"*
(true: science, predicted physics, 0.98). On Stack Exchange 43.0% of Science
questions (history-of-science site) are predicted Physics. Its subtopics are
the weakest ones too: history of science F1 0.41 (Wikipedia) / 0.35 (Stack
Exchange), scientific method 0.50, scientific research 0.56.

**Physics has low precision on questions (0.63):** it absorbs astronomy-style
chemistry and science questions (*"What does the surface of Mercury look
like?"*, chemistry → physics 12% of chemistry questions).

## 3. Confusion pairs

| Wikipedia test (true → predicted) | share of true class |
|---|---:|
| Science → Technology | 8.2% |
| Science → Books | 7.6% |
| History → Books / History → Science | 6.4% each |
| Biology → Science | 5.4% |

| Stack Exchange ext_test | share of true class |
|---|---:|
| Science → Physics | 43.0% |
| Chemistry → Physics | 12.0% |
| History → Science | 11.5% |
| History → Books | 11.1% |
| Science → Books | 10.5% |

The pairs are *semantically adjacent*, not random: the taxonomy has real
overlaps (a book about history, the history of physics), and a single
general label is sometimes a forced choice.

## 4. Style gap: encyclopedia → questions

| input length | accuracy |
|---|---:|
| Wikipedia test, 1–15 / 16–25 / 26–40 / > 40 words | 0.832 / 0.826 / 0.843 / 0.855 |
| Stack Exchange ext_test, 1–7 / 8–12 / > 12 words | 0.737 / 0.784 / 0.796 |

General macro-F1 drops from 0.834 (Wikipedia) to 0.756 (questions). Short
questions are the hardest (0.737 for ≤ 7 words). By site, accuracy ranges from
0.926 (sports) and 0.895 (AI) to 0.694 (history) and 0.333 (history of
science); quantum computing questions reach 0.650 (200 questions — the class
lost training data in D-27).

## 5. Confident mistakes

The most confident errors (`wiki_test_errors`, `se_general_errors`, sorted by
confidence) fall into three groups:

1. **Label noise / arguable labels** — *"A trapped-ion quantum computer is one
   proposed approach to a large-scale quantum computer."* labelled Physics,
   predicted Technology (0.97). The manual audit estimated 4.2% wrong and 6.2%
   weak labels (`reports/label_audit.json`).
2. **Passages without topical content** — *"The Hatta number (Ha) was developed
   by Shirôji Hatta, who taught at Tohoku University."* (chemistry → history).
3. **Questions whose topic is not in the words** — *"How would the Strikers
   re-emerge if there were so few competent people left in the world?"* (a
   novel on the literature site → sports, 0.99); *"Why do musketeers shoot in
   volley?"* (history → sports).

## 6. Subtopics

Macro-F1 0.647 on Wikipedia test, 0.775 when the general topic is right, 0.701
on the Stack Exchange subtopic questions. Hardest: history of science (0.41),
scientific method (0.50), scientific research (0.56), quantum mechanics (0.57),
authors (0.57), quantum computing (0.58, 59 test passages). Best: hardware,
basketball, poetry, football (0.76–0.77). Passages with 2 labels are not
harder than single-label ones (general accuracy 0.866 vs 0.833).

## 7. Out-of-taxonomy text

At the operating point that keeps 95% of genuine questions, the deployed rule
answers "uncertain" for 33.3% of out-of-taxonomy Wikipedia passages and 43.6%
of off-topic Stack Exchange questions. Per source it ranges from Dance 17% /
movies 18% (vocabulary overlaps Books and Sports) to personal finance 46% /
woodworking 53%. The texts it lets through are still mapped to the nearest
topic (pizza → Books 0.77, flagged only by the centroid gate). On in-domain text
the flagged answers are mostly wrong ones (accuracy 0.38–0.40 vs 0.81–0.87 for
the answered ones), so the gate does useful work, but it is not a reliable
off-topic detector. Improving it needs real off-topic training data (KNOWN_ISSUES.md).

## 8. What was not done

* No data augmentation (paraphrasing, back translation): it would not address
  the dominant errors (overlapping classes, style gap) and D-17 rules out
  synthetic data.
* No human-labelled chat data: the external test sets are question titles.
