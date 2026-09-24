"""Report figures (matplotlib, non-interactive backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6fdb"
ACCENT_2 = "#e07b39"


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(report: dict, path: Path, title: str) -> None:
    cm = np.asarray(report["confusion_matrix"], dtype=float)
    labels = report["labels"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else INK,
            )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{title} (row-normalised colour, counts shown)", fontsize=10)
    _save(fig, path)


def plot_per_class_f1(per_label: dict, path: Path, title: str) -> None:
    items = sorted(per_label.items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(items) + 1))
    ax.barh([k for k, _ in items], [v["f1"] for _, v in items], color=ACCENT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#e4e7eb")
    _save(fig, path)


def plot_reliability(curves: dict[str, list[dict]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, label="perfect calibration")
    for (name, curve), color in zip(curves.items(), (ACCENT, ACCENT_2, INK), strict=False):
        ax.plot([b["confidence"] for b in curve], [b["accuracy"] for b in curve], "o-", color=color, label=name)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability diagram (general topic)", fontsize=10)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_bars(values: dict[str, float], path: Path, title: str, xlabel: str) -> None:
    items = sorted(values.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(items) + 1))
    ax.barh([k for k, _ in items], [v for _, v in items], color=ACCENT)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", color="#e4e7eb")
    _save(fig, path)
