"""SentenceEncoder loading rules and float16 saving (no model download)."""

import pytest
import torch

from contextlens.models import encoders
from contextlens.models.encoders import ENCODERS, SentenceEncoder


def test_fine_tuned_encoder_never_falls_back_to_the_hub(tmp_path, monkeypatch):
    spec = {"repo": "unused/base-model", "revision": "0" * 40, "prefix": "", "local": "no/such/dir"}
    monkeypatch.setitem(ENCODERS, "tiny-ft", spec)
    monkeypatch.setattr(encoders, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="--export no/such/dir"):
        SentenceEncoder("tiny-ft", local_path=tmp_path / "artifact" / "encoder")


def test_empty_directory_is_not_mistaken_for_a_model(tmp_path, monkeypatch):
    spec = {"repo": "unused/base-model", "revision": "0" * 40, "prefix": "", "local": "ft"}
    monkeypatch.setitem(ENCODERS, "tiny-ft", spec)
    monkeypatch.setattr(encoders, "REPO_ROOT", tmp_path)
    (tmp_path / "ft").mkdir()  # exists but holds no modules.json
    with pytest.raises(FileNotFoundError):
        SentenceEncoder("tiny-ft")


class _TinyModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(2, 2, bias=False)
        self.lin.weight.data = weight.clone()
        self.saved_dtype: torch.dtype | None = None

    def save(self, path: str) -> None:
        self.saved_dtype = self.lin.weight.dtype


def _encoder_with(model: torch.nn.Module) -> SentenceEncoder:
    enc = SentenceEncoder.__new__(SentenceEncoder)  # skip __post_init__ (no download)
    enc.model = model
    return enc


def test_half_exact_weights_are_saved_as_float16_and_restored(tmp_path):
    weight = torch.tensor([[0.5, 0.25], [1.0, -2.0]])
    model = _TinyModel(weight)
    _encoder_with(model).save(tmp_path)
    assert model.saved_dtype == torch.float16
    assert model.lin.weight.dtype == torch.float32 and torch.equal(model.lin.weight, weight)


def test_float32_weights_are_saved_unchanged(tmp_path):
    weight = torch.tensor([[0.1, 0.2], [0.3, 1e-8]])
    model = _TinyModel(weight)
    _encoder_with(model).save(tmp_path)
    assert model.saved_dtype == torch.float32 and torch.equal(model.lin.weight, weight)
