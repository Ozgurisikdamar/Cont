"""Choose the conversation decay on simulated conversations (docs/EXPERIMENTS.md, E-DECAY).

Conversations are simulated from held-out passages with the trained model's
real predictions:

* a conversation has 3 segments; each segment has a true topic and lasts 3-6
  turns; 75% of turns are on-topic, 15% are tangents (another topic) and 10%
  are out-of-taxonomy passages;
* the tracker sees exactly what the app would see (calibrated general
  probabilities, weight 0 for uncertain turns).

Metrics per decay value:
  theme_accuracy   top theme topic == segment topic (turns after the first of a segment)
  switch_lag       turns needed after a topic switch until the theme follows
  tangent_robust   share of tangent turns where the theme stays on the segment topic
  accumulation_3   P(all three topics of a Books -> Science -> Biology style
                   3-message sequence of distinct topics are in the theme)

Selection uses conversations built from the validation split; the test split
only reports. Output: reports/experiments/decay.json

    python scripts/tune_decay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from contextlens.config import PATHS, RANDOM_SEED, load_settings
from contextlens.data.dataset import LabelSpace, load_ood_passages, load_passages
from contextlens.models.artifact import load_artifact
from contextlens.preprocessing.text import normalize
from contextlens.services.tracker import ConversationTracker
from contextlens.taxonomy import load_taxonomy

DECAYS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
N_CONVERSATIONS = 400
P_TANGENT, P_OOD = 0.15, 0.10


def predictions(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gp, _, ood = model.predict_proba(texts)
    uncertain = (ood < model.ood_threshold) | (gp.max(axis=1) < model.min_confidence)
    return gp, uncertain


def simulate(decay: float, pools: dict[int, np.ndarray], gp: np.ndarray, unc: np.ndarray, ood_gp: np.ndarray,
             ood_unc: np.ndarray, ids: list[str], min_share: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_classes = len(pools)
    hits = total = tangent_hits = tangents = 0
    lags: list[int] = []
    for _ in range(N_CONVERSATIONS):
        tracker = ConversationTracker(decay=decay, min_share=min_share)
        prev = -1
        for _segment in range(3):
            topic = int(rng.choice([c for c in range(n_classes) if c != prev]))
            length = int(rng.integers(3, 7))
            lag = None
            for turn in range(length):
                r = rng.random()
                is_tangent = False
                if r < P_OOD:
                    j = int(rng.integers(len(ood_gp)))
                    probs, uncertain = ood_gp[j], ood_unc[j]
                elif r < P_OOD + P_TANGENT:
                    other = int(rng.choice([c for c in range(n_classes) if c != topic]))
                    j = int(rng.choice(pools[other]))
                    probs, uncertain, is_tangent = gp[j], unc[j], True
                else:
                    j = int(rng.choice(pools[topic]))
                    probs, uncertain = gp[j], unc[j]
                tracker.update(dict(zip(ids, probs, strict=True)), {}, 0.0 if uncertain else 1.0)
                top = max(tracker.general_scores, key=tracker.general_scores.get) if tracker.general_scores else None
                on_topic = top == ids[topic]
                if lag is None and on_topic:
                    lag = turn
                if turn > 0:
                    total += 1
                    hits += on_topic
                if is_tangent and turn > 0:
                    tangents += 1
                    tangent_hits += on_topic
            if prev != -1:
                lags.append(lag if lag is not None else length)
            prev = topic
    # accumulation: three distinct-topic confident messages in a row
    acc_hits = 0
    for _ in range(N_CONVERSATIONS):
        tracker = ConversationTracker(decay=decay, min_share=min_share)
        topics = rng.choice(n_classes, size=3, replace=False)
        for t in topics:
            j = int(rng.choice(pools[int(t)]))
            tracker.update(dict(zip(ids, gp[j], strict=True)), {}, 1.0)
        active = {x.id for x in tracker.active_topics(tracker.general_scores, min_share, 3)}
        acc_hits += all(ids[int(t)] in active for t in topics)
    return {"decay": decay, "theme_accuracy": round(hits / total, 4), "switch_lag": round(float(np.mean(lags)), 3),
            "tangent_robust": round(tangent_hits / max(tangents, 1), 4),
            "accumulation_3": round(acc_hits / N_CONVERSATIONS, 4)}


def main() -> None:
    settings = load_settings()
    model = load_artifact(settings.model_dir)
    space = LabelSpace.from_taxonomy(load_taxonomy())
    df, ood = load_passages(), load_ood_passages()
    out: dict = {"protocol": __doc__, "results": {}}
    for split in ("val", "test"):
        part = df[df.split == split]
        gp, unc = predictions(model, [normalize(t) for t in part.text])
        y = space.encode_general(part.general)
        pools = {c: np.where((y == c) & ~unc)[0] for c in range(len(space.general_ids))}
        ood_gp, ood_unc = predictions(model, [normalize(t) for t in ood[ood.split == split].text])
        rows = [simulate(d, pools, gp, unc, ood_gp, ood_unc, space.general_ids, settings.theme_min_share,
                         RANDOM_SEED) for d in DECAYS]
        out["results"][split] = rows
        for r in rows:
            print(split, r)
    val = out["results"]["val"]
    # Primary: theme accuracy; must keep a 3-topic theme (the Books/Science/Biology use case) >= 80% of the time.
    eligible = [r for r in val if r["accumulation_3"] >= 0.8] or val
    best = max(eligible, key=lambda r: (r["theme_accuracy"], -abs(r["decay"] - 0.7)))
    out["selected_decay"] = best["decay"]
    out["selection_rule"] = "max validation theme_accuracy subject to accumulation_3 >= 0.80"
    print("selected decay:", best["decay"])
    target = PATHS.reports / "experiments" / "decay.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
