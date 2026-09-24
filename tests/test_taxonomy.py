import copy

import pytest

from contextlens.taxonomy import _parse, _parse_seed


def test_shape(taxonomy):
    assert len(taxonomy.general_ids) == 8
    assert len(taxonomy.subtopic_ids) == 28
    assert set(taxonomy.children_of("physics")) == {
        "quantum_mechanics", "relativity", "astrophysics", "classical_mechanics"}
    assert taxonomy.parent_of("quantum_computing") == "technology"


def test_hierarchy_validation(taxonomy):
    assert taxonomy.is_consistent("physics", ["quantum_mechanics", "relativity"])
    assert not taxonomy.is_consistent("physics", ["quantum_computing"])


def test_composition_metadata(taxonomy):
    assert taxonomy.general("books").role == "format"
    assert taxonomy.general("biology").broader == "science"
    assert taxonomy.general("history").perspective_template == "history of {topic}"


def test_display_names(taxonomy):
    assert taxonomy.display("technology") == "Technology"
    assert taxonomy.display("football") == "Football (Soccer)"
    with pytest.raises(KeyError):
        taxonomy.display("cooking")


def test_seed_parsing():
    assert _parse_seed("Quantum_mechanics@2") == ("Quantum_mechanics", 2)
    assert _parse_seed("Chemical_elements") == ("Chemical_elements", 0)


def test_rejects_duplicate_ids(taxonomy):
    raw = copy.deepcopy(taxonomy.raw)
    raw["general_topics"][1]["subtopics"].append(copy.deepcopy(raw["general_topics"][0]["subtopics"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        _parse(raw)


def test_rejects_unknown_broader(taxonomy):
    raw = copy.deepcopy(taxonomy.raw)
    raw["general_topics"][0]["broader"] = "nonexistent"
    with pytest.raises(ValueError, match="broader"):
        _parse(raw)
