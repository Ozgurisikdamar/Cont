"""Experiment: fine-tuned transformer with a shared encoder and two heads.

    encoder (mean pooling) --+-- general head   (softmax, cross-entropy)
                             +-- subtopic head  (sigmoid, binary cross-entropy)

This is the "shared encoder + two heads" alternative from the brief, compared
against frozen-embedding + logistic-regression heads in docs/EXPERIMENTS.md.
Early stopping / best-epoch selection uses validation general macro-F1 only.
CPU training is supported (device auto-selected); results go to
reports/experiments/finetune_<encoder>.json. Weights are not kept unless
--save is given (the benchmark decides whether they are worth shipping).

    python scripts/finetune_transformer.py --encoder minilm-l6 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from contextlens.config import PATHS, RANDOM_SEED
from contextlens.data.dataset import BenchmarkData
from contextlens.evaluation.metrics import multiclass_report, multilabel_report
from contextlens.logging_setup import setup_logging
from contextlens.models.encoders import ENCODERS, best_device
from contextlens.models.heads import decide_subtopics
from contextlens.reproducibility import seed_everything

log = logging.getLogger("finetune")


class TwoHead(nn.Module):
    def __init__(self, repo: str, revision: str, n_general: int, n_sub: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(repo, revision=revision)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.general = nn.Linear(hidden, n_general)
        self.sub = nn.Linear(hidden, n_sub)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = self.dropout((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
        return self.general(pooled), self.sub(pooled)


@torch.no_grad()
def predict(model: TwoHead, tok, texts: list[str], prefix: str, device: str, max_len: int,
            batch: int = 128) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    gps, sps = [], []
    for i in range(0, len(texts), batch):
        enc = tok([prefix + t for t in texts[i:i + batch]], truncation=True, max_length=max_len, padding=True,
                  return_tensors="pt").to(device)
        g, s = model(enc["input_ids"], enc["attention_mask"])
        gps.append(torch.softmax(g, -1).cpu().numpy())
        sps.append(torch.sigmoid(s).cpu().numpy())
    return np.concatenate(gps), np.concatenate(sps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="minilm-l6", choices=sorted(ENCODERS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    setup_logging("INFO", PATHS.root / "logs" / f"finetune_{args.encoder}.log")
    seed_everything(RANDOM_SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = best_device()
    data = BenchmarkData()
    space = data.space
    spec = ENCODERS[args.encoder]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    model = TwoHead(spec["repo"], spec["revision"], len(space.general_ids), len(space.subtopic_ids)).to(device)
    texts, yg, ys = data.text["train"], data.yg["train"], data.ys["train"]
    counts = np.bincount(yg, minlength=len(space.general_ids))
    ce = nn.CrossEntropyLoss(weight=torch.tensor(len(yg) / (len(counts) * counts), dtype=torch.float32).to(device))
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(texts) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    rng = np.random.default_rng(RANDOM_SEED)
    history, best, best_state = [], -1.0, None
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(texts))
        t0, running = time.perf_counter(), 0.0
        for bi, i in enumerate(range(0, len(order), args.batch)):
            idx = order[i:i + args.batch]
            enc = tok([spec["prefix"] + texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            g, s = model(enc["input_ids"], enc["attention_mask"])
            loss = ce(g, torch.tensor(yg[idx]).to(device)) + bce(s, torch.tensor(ys[idx], dtype=torch.float32).to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss)
            if bi % 200 == 0:
                log.info("epoch %d batch %d loss %.4f (%.0fs)", epoch, bi, running / (bi + 1), time.perf_counter() - t0)
        gp, _ = predict(model, tok, data.text["val"], spec["prefix"], device, args.max_len)
        rep = multiclass_report(data.yg["val"], gp, space.general_ids)
        history.append({"epoch": epoch, "train_loss": round(running / (bi + 1), 4), "val_accuracy": round(rep["accuracy"], 4),
                        "val_macro_f1": round(rep["macro_f1"], 4), "epoch_seconds": round(time.perf_counter() - t0, 1)})
        log.info("epoch %d: %s", epoch, history[-1])
        if rep["macro_f1"] > best:
            best = rep["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - t_start
    results: dict = {"encoder": args.encoder, "hyperparameters": vars(args), "history": history,
                     "train_seconds": round(train_seconds, 1), "device": device}
    thr_grid = [round(x, 2) for x in np.arange(0.2, 0.71, 0.05)]
    gp_val, sp_val = predict(model, tok, data.text["val"], spec["prefix"], device, args.max_len)
    sweep = [(multilabel_report(data.ys["val"], decide_subtopics(sp_val, space.parent_col, gp_val.argmax(1), t), sp_val,
                                space.subtopic_ids)["macro_f1"], t) for t in thr_grid]
    thr = max(sweep)[1]
    results["subtopic_threshold"] = thr
    for split in ("val", "test", "se_ext_dev", "se_ext_test", "sesub_ext_dev", "sesub_ext_test"):
        gp, sp = predict(model, tok, data.text[split], spec["prefix"], device, args.max_len)
        row = {"general": {k: round(v, 4) for k, v in multiclass_report(data.yg[split], gp, space.general_ids).items()
                           if isinstance(v, float)}}
        if split in data.ys:
            ml = multilabel_report(data.ys[split], decide_subtopics(sp, space.parent_col, gp.argmax(1), thr), sp,
                                   space.subtopic_ids)
            row["subtopics"] = {k: round(v, 4) for k, v in ml.items() if isinstance(v, float)}
        results[split] = row
        log.info("%s: %s", split, row)
    sample = data.text["val"][:200]
    times = []
    for t in sample:
        t0 = time.perf_counter()
        predict(model, tok, [t], spec["prefix"], device, args.max_len)
        times.append((time.perf_counter() - t0) * 1000)
    results["latency_single_ms_median"] = round(float(np.median(times)), 2)
    results["size_mb"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1)
    out = PATHS.reports / "experiments" / f"finetune_{args.encoder}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.save:
        torch.save(model.state_dict(), PATHS.models / f"finetuned_{args.encoder}.pt")
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
