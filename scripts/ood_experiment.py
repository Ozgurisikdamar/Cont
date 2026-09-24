"""Out-of-taxonomy detector comparison (docs/HARDENING.md item 3, decisions.md D-34).

The production heads are fitted exactly as ``train.py`` does (``fit_from_config``
with configs/model.json), then seven detectors (contextlens.models.ood) are
compared on development data only:

  in-domain     Wikipedia ``val`` passages; Stack Exchange ``ext_dev`` in-taxonomy
                questions (the input style users actually type)
  off-topic     Wikipedia OOD ``val`` (out-of-taxonomy categories); Stack Exchange
                ``ext_dev`` off-topic sites; CLINC150 ``dev`` assistant chat
                (scripts/build_ood_conversational.py)

Pairs: wiki val vs wiki OOD, SE in vs SE off-topic, SE in vs CLINC chat.
Per pair: AUROC, AUPRC (off-topic = positive), FPR@95TPR (share of off-topic
accepted when 95% of in-domain is kept). At the deployed operating point
(threshold = 5th percentile of the SE ext_dev in-domain scores, combined with
the ``min_confidence`` gate): off-topic recall (= TNR of the gate), in-domain
retention, coverage and accuracy on the answered in-domain questions.

Selection rule (fixed before running): the highest mean AUROC over the three
pairs; within 0.005, the higher mean off-topic recall at the operating point,
then the simpler detector (order in DETECTORS).

    python scripts/ood_experiment.py      # -> reports/experiments/ood_detectors.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from contextlens.config import PATHS, RANDOM_SEED
from contextlens.data.dataset import LabelSpace, load_jsonl, load_ood_passages, load_passages, load_se_general
from contextlens.logging_setup import setup_logging
from contextlens.models.encoders import CachingEncoder, SentenceEncoder
from contextlens.models.ood import DETECTORS, fit_detector
from contextlens.preprocessing.text import normalize
from contextlens.reproducibility import seed_everything
from contextlens.taxonomy import load_taxonomy
from train import MODEL_CONFIG, fit_from_config, offtopic_training_texts

log = logging.getLogger("ood_experiment")
KEEP = 0.95
TOL = 0.005


def pair_metrics(s_in: np.ndarray, s_out: np.ndarray) -> dict:
    y = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])  # off-topic = positive
    s = -np.concatenate([s_in, s_out])
    thr = np.quantile(s_in, 1 - KEEP)
    return {
        "auroc": round(float(roc_auc_score(y, s)), 4),
        "auprc": round(float(average_precision_score(y, s)), 4),
        "fpr_at_95tpr": round(float(np.mean(s_out >= thr)), 4),
        "n_in": int(len(s_in)),
        "n_out": int(len(s_out)),
    }


def main() -> int:
    setup_logging("INFO")
    seed_everything(RANDOM_SEED)
    conf = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    conf = {**conf, "ood": {"method": "centroid", "calibration": "se_ext_dev"}}
    enc = CachingEncoder(SentenceEncoder(conf["encoder"]), PATHS.data_processed / "embeddings")
    model, info = fit_from_config(conf, enc)
    space = LabelSpace.from_taxonomy(load_taxonomy())
    train_texts, ytr = info["train"][0], info["train"][1]

    se, ood = load_se_general(), load_ood_passages()
    clinc = load_jsonl(PATHS.root / "data" / "external" / "ood_conversational.jsonl")
    df = load_passages()
    val = df[df.split == "val"]
    sets = {
        "wiki_in": [normalize(t) for t in val.text],
        "wiki_ood": [normalize(t) for t in ood[ood.split == "val"].text],
        "se_in": [normalize(t) for t in se[(se.split == "ext_dev") & ~se.is_ood].text],
        "se_ood": [normalize(t) for t in se[(se.split == "ext_dev") & se.is_ood].text],
        "chat_ood": [normalize(t) for t in clinc[clinc.split == "dev"].text],
    }
    y_se = space.encode_general(se[(se.split == "ext_dev") & ~se.is_ood].general)
    emb = {k: enc.encode(v) for k, v in sets.items()}
    heads = {k: model.head_outputs(X) for k, X in emb.items()}
    Xtr = enc.encode(train_texts)
    Xoff = enc.encode(offtopic_training_texts())
    english = {k: np.array([model.looks_english(t) for t in v]) for k, v in sets.items()}
    pairs = (("wiki_in", "wiki_ood"), ("se_in", "se_ood"), ("se_in", "chat_ood"))

    results = {}
    for method in DETECTORS:
        det = fit_detector(
            method, Xtr, ytr, len(space.general_ids), Xoff if method in ("binary", "other_class") else None
        )
        scores = {k: det.score(emb[k], heads[k][0], heads[k][2], model.temperature) for k in sets}
        thr = float(np.quantile(scores["se_in"], 1 - KEEP))
        row: dict = {"pairs": {f"{a}_vs_{b}": pair_metrics(scores[a], scores[b]) for a, b in pairs}}
        # operating point: detector threshold + the deployed confidence and language gates
        flagged = {k: (scores[k] < thr) | (heads[k][0].max(axis=1) < model.min_confidence) | ~english[k] for k in sets}
        answered_se = ~flagged["se_in"]
        row["operating_point"] = {
            "threshold": round(thr, 5),
            "offtopic_recall": {k: round(float(flagged[k].mean()), 4) for k in ("wiki_ood", "se_ood", "chat_ood")},
            "in_domain_retention_detector_only": {
                k: round(float(np.mean(scores[k] >= thr)), 4) for k in ("wiki_in", "se_in")
            },
            "coverage": {k: round(float(1 - flagged[k].mean()), 4) for k in ("wiki_in", "se_in")},
            "se_in_accuracy_answered": round(
                float(np.mean(heads["se_in"][0].argmax(axis=1)[answered_se] == y_se[answered_se])), 4
            ),
        }
        row["mean_auroc"] = round(float(np.mean([p["auroc"] for p in row["pairs"].values()])), 4)
        row["mean_offtopic_recall"] = round(float(np.mean(list(row["operating_point"]["offtopic_recall"].values()))), 4)
        results[method] = row
        log.info("%s mean AUROC %.4f recall %s", method, row["mean_auroc"], row["operating_point"]["offtopic_recall"])

    top = max(r["mean_auroc"] for r in results.values())
    close = [m for m in DETECTORS if results[m]["mean_auroc"] >= top - TOL]
    chosen = max(close, key=lambda m: (results[m]["mean_offtopic_recall"], -DETECTORS.index(m)))
    out = {
        "protocol": __doc__,
        "encoder": conf["encoder"],
        "head_type": conf["head_type"],
        "min_confidence": model.min_confidence,
        "sizes": {k: len(v) for k, v in sets.items()},
        "results": results,
        "chosen": chosen,
    }
    target = PATHS.reports / "experiments" / "ood_detectors.json"
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for m, r in results.items():
        op = r["operating_point"]
        print(f"{m:12s} AUROC {r['mean_auroc']:.4f}  recall {op['offtopic_recall']}  coverage {op['coverage']}")
    print("chosen:", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
