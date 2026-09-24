"""Tune the conversation tracker on simulated conversations (docs/EXPERIMENTS.md, E-10).

Conversations are simulated from **validation** passages with the trained
model's real predictions (the test split is not used here):

* a conversation has 3 segments; each segment has a true topic and lasts 3-6
  turns; 75% of turns are on-topic, 15% are tangents (another topic) and 10%
  are out-of-taxonomy texts;
* the tracker sees exactly what the app would see (calibrated general
  probabilities, weight 0 for uncertain turns); on-topic turns are drawn from
  all passages of the topic, including ones the model is unsure about.

Metrics
  theme_accuracy     dominant theme topic == segment topic (turns after the first of a segment)
  switch_lag         turns after a topic switch until the dominant topic follows
  tangent_robust     share of tangent turns after which the dominant topic is unchanged
  false_switch_rate  share of tangent turns after which the tangent became dominant
  accumulation_3     P(all three topics of a Books -> Science -> Biology style
                     3-message sequence are in the theme); confident, correct messages
  spurious_topics    mean number of theme topics that are neither the current nor
                     the previous segment topic
  stale_theme_rate   P(theme still non-empty after 10 consecutive uncertain turns)
  premature_expiry   P(theme already empty after 2 consecutive uncertain turns)
  expiry_turns       mean number of uncertain turns until the theme is empty

Grid: decay x theme_min_share x confirm_turns x expire_after x switch_rule.
Rule: maximise theme_accuracy among settings with accumulation_3 >= 0.80,
stale_theme_rate <= 0.05, premature_expiry <= 0.05 and false_switch_rate <=
0.15; ties -> higher tangent_robust, fewer spurious topics. Theme accuracy alone
always prefers switching on every message (it scores the turns of a real
switch and ignores what a tangent does to the conversation), which is exactly
the v1.0 weakness, hence the false-switch constraint. The v1.0 tracker
(confirm 1, no expiry) is reported as the baseline. Output: reports/experiments/decay.json

    python scripts/tune_decay.py
"""

from __future__ import annotations

import itertools
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

DECAYS = [0.5, 0.6, 0.7, 0.8]
MIN_SHARES = [0.15, 0.2, 0.25]
CONFIRM = [1, 2, 3]
SWITCH_RULES = ["votes", "leader"]
MAX_FALSE_SWITCH = 0.15
EXPIRE = [0, 3, 4, 5, 6, 8]
N_CONVERSATIONS = 400
P_TANGENT, P_OOD = 0.15, 0.10
STALE_AFTER, PREMATURE_AFTER = 10, 2


def predictions(model, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    gp, _, ood = model.predict_proba(texts)
    uncertain = model.uncertain_mask(texts, gp, ood)
    return gp, uncertain


def simulate(params: dict, pools, confident, gp, unc, ood_gp, ood_unc, ids, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_classes = len(pools)

    def new() -> ConversationTracker:
        return ConversationTracker(
            decay=params["decay"],
            min_share=params["min_share"],
            confirm_turns=params["confirm_turns"],
            expire_after=params["expire_after"],
            switch_rule=params["switch_rule"],
        )

    def feed(tracker: ConversationTracker, probs: np.ndarray, uncertain: bool) -> None:
        tracker.update(dict(zip(ids, probs, strict=True)), {}, 0.0 if uncertain else 1.0)

    hits = total = tangents = tangent_kept = false_switch = spurious = 0
    lags: list[int] = []
    for _ in range(N_CONVERSATIONS):
        tracker, prev = new(), -1
        for _segment in range(3):
            topic = int(rng.choice([c for c in range(n_classes) if c != prev]))
            length = int(rng.integers(3, 7))
            lag = None
            for turn in range(length):
                r = rng.random()
                before = tracker.dominant
                tangent_to = None
                if r < P_OOD:
                    j = int(rng.integers(len(ood_gp)))
                    feed(tracker, ood_gp[j], bool(ood_unc[j]))
                elif r < P_OOD + P_TANGENT:
                    tangent_to = int(rng.choice([c for c in range(n_classes) if c != topic]))
                    j = int(rng.choice(pools[tangent_to]))
                    feed(tracker, gp[j], bool(unc[j]))
                else:
                    j = int(rng.choice(pools[topic]))
                    feed(tracker, gp[j], bool(unc[j]))
                on_topic = tracker.dominant == ids[topic]
                if lag is None and on_topic:
                    lag = turn
                if turn > 0:
                    total += 1
                    hits += on_topic
                    active = {x.id for x in tracker.active_topics(tracker.general_scores, params["min_share"], 3)}
                    spurious += len(active - {ids[topic], ids[prev] if prev >= 0 else ""})
                    if tangent_to is not None and before is not None and before != ids[tangent_to]:
                        tangents += 1
                        tangent_kept += tracker.dominant == before
                        false_switch += tracker.dominant == ids[tangent_to]
            if prev != -1:
                lags.append(lag if lag is not None else length)
            prev = topic

    acc_hits = 0
    for _ in range(N_CONVERSATIONS):
        tracker = new()
        topics = rng.choice(n_classes, size=3, replace=False)
        for t in topics:
            feed(tracker, gp[int(rng.choice(confident[int(t)]))], False)
        active = {x.id for x in tracker.active_topics(tracker.general_scores, params["min_share"], 3)}
        acc_hits += all(ids[int(t)] in active for t in topics)

    # staleness probe: a confident segment, then a streak of uncertain turns
    uncertain_pool = np.where(ood_unc)[0]
    stale = premature = 0
    expiry: list[int] = []
    for _ in range(N_CONVERSATIONS):
        tracker = new()
        topic = int(rng.integers(n_classes))
        for _ in range(int(rng.integers(3, 7))):
            feed(tracker, gp[int(rng.choice(confident[topic]))], False)
        emptied_at = None
        for k in range(1, STALE_AFTER + 1):
            feed(tracker, ood_gp[int(rng.choice(uncertain_pool))], True)
            if emptied_at is None and not tracker.general_scores:
                emptied_at = k
            if k == PREMATURE_AFTER:
                premature += not tracker.general_scores
        stale += emptied_at is None
        expiry.append(emptied_at if emptied_at is not None else STALE_AFTER + 1)

    return {
        **params,
        "theme_accuracy": round(hits / total, 4),
        "switch_lag": round(float(np.mean(lags)), 3),
        "tangent_robust": round(tangent_kept / max(tangents, 1), 4),
        "false_switch_rate": round(false_switch / max(tangents, 1), 4),
        "accumulation_3": round(acc_hits / N_CONVERSATIONS, 4),
        "spurious_topics": round(spurious / total, 4),
        "stale_theme_rate": round(stale / N_CONVERSATIONS, 4),
        "premature_expiry": round(premature / N_CONVERSATIONS, 4),
        "expiry_turns": round(float(np.mean(expiry)), 2),
    }


def eligible(r: dict) -> bool:
    return (
        r["accumulation_3"] >= 0.8
        and r["stale_theme_rate"] <= 0.05
        and r["premature_expiry"] <= 0.05
        and r["false_switch_rate"] <= MAX_FALSE_SWITCH
    )


def main() -> None:
    settings = load_settings()
    model = load_artifact(settings.model_dir)
    space = LabelSpace.from_taxonomy(load_taxonomy())
    df, ood = load_passages(), load_ood_passages()
    part = df[df.split == "val"]
    gp, unc = predictions(model, [normalize(t) for t in part.text])
    y = space.encode_general(part.general)
    pools = {c: np.where(y == c)[0] for c in range(len(space.general_ids))}
    confident = {c: np.where((y == c) & ~unc & (gp.argmax(axis=1) == c))[0] for c in range(len(space.general_ids))}
    ood_gp, ood_unc = predictions(model, [normalize(t) for t in ood[ood.split == "val"].text])
    args = (pools, confident, gp, unc, ood_gp, ood_unc, space.general_ids, RANDOM_SEED)
    baseline = simulate(
        {"decay": 0.7, "min_share": 0.2, "confirm_turns": 1, "expire_after": 0, "switch_rule": "leader"}, *args
    )
    print("baseline v1.0", baseline)
    rows = []
    for d, m, c, e, rule in itertools.product(DECAYS, MIN_SHARES, CONFIRM, EXPIRE, SWITCH_RULES):
        if c == 1 and rule == "votes":
            continue  # identical to switching on every message; covered by "leader" with confirm 1
        params = {"decay": d, "min_share": m, "confirm_turns": c, "expire_after": e, "switch_rule": rule}
        rows.append(simulate(params, *args))
    ok = [r for r in rows if eligible(r)]
    pool = ok or rows
    best = max(pool, key=lambda r: (r["theme_accuracy"], r["tangent_robust"], -r["spurious_topics"]))
    print("selected:", best, "constraints met:", bool(ok))
    out = {
        "protocol": __doc__,
        "split": "val",
        "baseline_v1_0": baseline,
        "results": rows,
        "selected": {k: best[k] for k in ("decay", "min_share", "confirm_turns", "expire_after", "switch_rule")},
        "selected_metrics": best,
        "constraints_met": bool(ok),
        "selection_rule": "max val theme_accuracy s.t. accumulation_3 >= 0.80, stale_theme_rate <= 0.05, "
        "premature_expiry <= 0.05, false_switch_rate <= 0.15 (ties: tangent_robust, fewer spurious topics)",
    }
    target = PATHS.reports / "experiments" / "decay.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
