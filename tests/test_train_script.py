"""train.py end to end on a small sample with the fake encoder (no download, seconds)."""

import json

import train
from contextlens.data import dataset
from contextlens.models.artifact import load_artifact

from .conftest import FakeEncoder


def test_train_script_writes_a_loadable_artifact(tmp_path, monkeypatch):
    try:
        passages, se = dataset.load_passages(), dataset.load_se_general()
    except dataset.DataMissingError:
        import pytest

        pytest.skip("processed corpus not available")
    small = passages.groupby(["split", "general"], group_keys=False).head(40)
    se_small = se.groupby(["split", "is_ood", "general"], group_keys=False, dropna=False).head(30)
    monkeypatch.setattr(train, "load_passages", lambda: small)
    monkeypatch.setattr(train, "load_se_general", lambda: se_small)
    monkeypatch.setattr(train, "SentenceEncoder", lambda key: FakeEncoder())
    monkeypatch.setitem(train.ENCODERS, "fake", {"repo": "fake", "revision": "0", "prefix": ""})
    conf = json.loads(train.MODEL_CONFIG.read_text())
    conf["encoder"] = "fake"
    cfg_path = tmp_path / "model.json"
    cfg_path.write_text(json.dumps(conf))
    out = tmp_path / "artifact"
    assert train.main(["--config", str(cfg_path), "--out", str(out)]) == 0
    model = load_artifact(out, encoder=FakeEncoder())
    meta = model.metadata
    assert meta["hyperparameters"]["head_type"] == conf["head_type"]
    if conf["min_confidence"] == "auto":
        sweep = meta["min_confidence_sweep_ext_dev"]
        assert [r["min_confidence"] for r in sweep] == conf["min_confidence_grid"]
        assert model.min_confidence in conf["min_confidence_grid"]
    assert model.predict("Quantum processors can speed up algorithms by using qubits.").general is not None


def _small_artifact(tmp_path, monkeypatch):
    """Train a small fake-encoder artifact with train.py (see the test above)."""
    passages, se = dataset.load_passages(), dataset.load_se_general()
    small = passages.groupby(["split", "general"], group_keys=False).head(40)
    se_small = se.groupby(["split", "is_ood", "general"], group_keys=False, dropna=False).head(30)
    monkeypatch.setattr(train, "load_passages", lambda: small)
    monkeypatch.setattr(train, "load_se_general", lambda: se_small)
    monkeypatch.setattr(train, "SentenceEncoder", lambda key: FakeEncoder())
    monkeypatch.setitem(train.ENCODERS, "fake", {"repo": "fake", "revision": "0", "prefix": ""})
    conf = json.loads(train.MODEL_CONFIG.read_text())
    conf["encoder"] = "fake"
    (tmp_path / "model.json").write_text(json.dumps(conf))
    out = tmp_path / "artifact"
    assert train.main(["--config", str(tmp_path / "model.json"), "--out", str(out)]) == 0
    return out, small, se_small


def test_evaluate_and_decay_scripts_run(tmp_path, monkeypatch):
    import dataclasses
    import importlib
    import sys
    from pathlib import Path

    import pytest

    try:
        dataset.load_passages()
    except dataset.DataMissingError:
        pytest.skip("processed corpus not available")
    import evaluate

    out, small, se_small = _small_artifact(tmp_path, monkeypatch)
    ood = dataset.load_ood_passages().groupby("split", group_keys=False).head(30)
    sub = dataset.load_se_subtopic().groupby("split", group_keys=False).head(40)
    fake_paths = dataclasses.replace(evaluate.PATHS, reports=tmp_path / "reports", figures=tmp_path / "figs")
    (tmp_path / "reports" / "experiments").mkdir(parents=True)

    def load(d, **kw):
        return load_artifact(out, encoder=FakeEncoder(), **kw)

    monkeypatch.setattr(evaluate, "PATHS", fake_paths)
    monkeypatch.setattr(evaluate, "load_artifact", load)
    for name, value in [
        ("load_passages", small),
        ("load_ood_passages", ood),
        ("load_se_general", se_small),
        ("load_se_subtopic", sub),
    ]:
        monkeypatch.setattr(evaluate, name, lambda v=value: v)
    real_jsonl = evaluate.load_jsonl
    monkeypatch.setattr(evaluate, "load_jsonl", lambda p: real_jsonl(p).groupby("split", group_keys=False).head(40))
    assert evaluate.main(["--stage", "dev"]) == 0
    report = json.loads((tmp_path / "reports" / "evaluation_dev.json").read_text())
    assert {"stage", "dev", "acceptance", "latency"} <= set(report)
    assert {"wiki", "se_general", "se_general_without_hsm", "se_subtopic", "offtopic", "language"} <= set(report["dev"])
    assert {"se_sites", "wiki_categories", "chat"} <= set(report["dev"]["offtopic"])
    # the locked stage refuses to run without a freeze
    monkeypatch.setattr(evaluate, "LOCKED_DIR", tmp_path / "reports" / "locked")
    with pytest.raises(SystemExit, match="FREEZE"):
        evaluate.main(["--stage", "locked"])

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    tune_decay = importlib.import_module("tune_decay")
    monkeypatch.setattr(tune_decay, "PATHS", fake_paths)
    monkeypatch.setattr(tune_decay, "load_artifact", load)
    monkeypatch.setattr(tune_decay, "load_passages", lambda: small)
    monkeypatch.setattr(tune_decay, "load_ood_passages", lambda: ood)
    monkeypatch.setattr(tune_decay, "N_CONVERSATIONS", 20)
    tune_decay.main()
    decay = json.loads((tmp_path / "reports" / "experiments" / "decay.json").read_text())
    assert {"decay", "confirm_turns", "expire_after"} <= set(decay["selected"])
