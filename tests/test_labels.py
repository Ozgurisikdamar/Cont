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


def test_missing_articles_are_recovered_from_redirects_and_the_second_dump(monkeypatch, tmp_path):
    from contextlens.data import corpus

    aliases = {"Merged": ["Short old", "Long old", "Wanted too"], "Gold": [], "Brand new": []}
    dumps = {
        "d23": {  # first snapshot: renamed/merged articles under their old titles, "Gold" absent
            "Short old": {"wiki_id": "1", "url": "u", "text": "x"},
            "Long old": {"wiki_id": "2", "url": "u", "text": "much longer main article"},
        },
        "d22": {  # second snapshot: has "Gold"; "Long old" again (same page id) must not be reused
            "Gold": {"wiki_id": "12240", "url": "u", "text": "gold text"},
        },
    }
    monkeypatch.setattr(corpus, "redirect_aliases", lambda client, titles: aliases)

    def fake_extract(titles, repo, **kw):
        assert "Wanted too" not in titles  # a wanted title is never borrowed as an alias
        return {t: d for t, d in dumps[repo].items() if t in titles}

    monkeypatch.setattr(corpus, "extract_articles", fake_extract)
    wanted = {"Merged", "Gold", "Wanted too", "Brand new"}
    texts = {"Wanted too": {"wiki_id": "3", "url": "u", "text": "w"}}
    passes = [
        ("dump", {"repo": "d23", "download_dir": tmp_path}, True),
        ("legacy", {"repo": "d22", "download_dir": tmp_path}, False),
    ]
    counts = corpus.recover_missing(None, wanted, texts, passes)
    assert counts == {"dump_redirect": 1, "legacy": 1}
    assert texts["Merged"]["dump_title"] == "Long old"  # merge: the longest pre-merge article wins
    assert texts["Gold"]["text_source"] == "legacy" and texts["Gold"]["text"] == "gold text"
    assert "Brand new" not in texts  # created after both snapshots: stays missing


def test_near_duplicates_across_splits_are_found():
    from contextlens.data.corpus import near_duplicate_ids

    tpl = "A period {} element is one of the chemical elements in the {} row of the periodic table."
    passages = [
        {"passage_id": "a", "split": "train", "text": tpl.format(1, "first")},
        {"passage_id": "b", "split": "train", "text": "Football is a team sport played with a ball."},
        {"passage_id": "c", "split": "test", "text": tpl.format(2, "first")},  # templated sibling
        {"passage_id": "d", "split": "val", "text": "Mitochondria produce energy inside the cell."},
    ]
    assert near_duplicate_ids(passages, threshold=0.8) == {"c"}
