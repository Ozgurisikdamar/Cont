"""Measure what storing encoder weights in float16 does to the embeddings.

    python scripts/fp16_storage_check.py     # writes reports/fp16_storage.json

The fine-tuned encoder is exported with float16 weights (half the file size) and
loaded back as float32. This compares embeddings of 300 validation passages from
the pinned all-MiniLM-L6-v2 in float32 against the same model after a float16
save/load round trip. (The shipped artifact is additionally protected by the
encoder fingerprint check, contextlens/models/artifact.py.)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from contextlens.config import PATHS
from contextlens.data.dataset import load_passages
from contextlens.models.encoders import ENCODERS
from contextlens.preprocessing.text import normalize

N = 300


def main() -> int:
    from sentence_transformers import SentenceTransformer

    spec = ENCODERS["minilm-l6"]
    df = load_passages()
    texts = [normalize(t) for t in df[df.split == "val"].text[:N]]
    model = SentenceTransformer(spec["repo"], revision=spec["revision"], device="cpu")
    ref = model.encode(texts, normalize_embeddings=True)
    with tempfile.TemporaryDirectory() as tmp:
        model.half()
        model.save(tmp)
        reloaded = SentenceTransformer(tmp, device="cpu")
        dtype = str(next(reloaded.parameters()).dtype)
        got = reloaded.encode(texts, normalize_embeddings=True)
    cos = (ref * got).sum(axis=1)
    result = {
        "encoder": f"{spec['repo']}@{spec['revision']}",
        "n_texts": len(texts),
        "reloaded_dtype": dtype,
        "cosine_min": round(float(cos.min()), 7),
        "cosine_mean": round(float(cos.mean()), 7),
        "max_abs_diff": round(float(np.abs(ref - got).max()), 7),
    }
    out = PATHS.reports / "fp16_storage.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
