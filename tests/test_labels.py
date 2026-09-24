from contextlens.data.corpus import assign_splits
from contextlens.data.labels import LabelledArticle, assign_labels, select_balanced


def test_general_topic_is_the_shallowest_crawl(taxonomy):
    membership = {
        "Qubit": {"quantum_computing": 0, "quantum_mechanics": 2},
        "Tie": {"quantum_computing": 1, "quantum_mechanics": 1},
        "Gene": {"genetics": 0, "evolution": 1, "cell_biology": 2},
    }
    labelled, ambiguous = assign_labels(taxonomy, membership)
    by_title = {a.title: a for a in labelled}
    assert by_title["Qubit"].general == "technology"
    assert by_title["Qubit"].subtopics == ["quantum_computing"]
    assert ambiguous == ["Tie"]
    # secondary labels only up to depth 1 (cell_biology at depth 2 is ignored)
    assert by_title["Gene"].subtopics == ["evolution", "genetics"]


def test_balanced_selection_caps_and_prefers_shallow(taxonomy):
    membership = {f"A{i}": {"genetics": i % 3} for i in range(30)}
    labelled, _ = assign_labels(taxonomy, membership)
    chosen = select_balanced(taxonomy, labelled, membership, cap=10)
    assert len(chosen) == 10
    assert all(membership[a.title]["genetics"] == 0 for a in chosen)
    assert chosen == select_balanced(taxonomy, labelled, membership, cap=10)  # deterministic


def test_grouped_split_is_disjoint_and_stratified():
    arts = [LabelledArticle(f"T{i}", "physics", ["relativity"], 0) for i in range(100)]
    arts += [LabelledArticle(f"B{i}", "books", ["poetry"], 0) for i in range(40)]
    split = assign_splits(arts, 0.15, 0.15)
    assert set(split) == {a.title for a in arts}  # each article in exactly one split
    for prefix in ("T", "B"):
        values = [v for k, v in split.items() if k.startswith(prefix)]
        assert {"train", "val", "test"} == set(values)
    assert assign_splits(arts, 0.15, 0.15) == split
