#!/usr/bin/env python3
"""Regenerate plots from results/data/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data" / "metrics.json"
OUT = ROOT / "results" / "plots"


def series(data, run_key, field):
    xs, ys = [], []
    for row in data[run_key]["history"]:
        x, y = row.get("step"), row.get(field)
        if x is None or y is None:
            continue
        try:
            x, y = float(x), float(y)
        except Exception:
            continue
        if y != y:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def main():
    data = json.loads(DATA.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 11})

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in [
        ("vanilla", "Vanilla (full L0 attn + residual)", "#2ca02c"),
        ("l0_abl", "L0-abl (no L0 attn, no L0 residual)", "#9467bd"),
    ]:
        xs, ys = series(data, key, "train_loss")
        ax.plot(xs, ys, label=label, color=color, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("Pretraining train loss — d8 on 2×T4 (22k steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_22k.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in [
        ("vanilla", "Vanilla", "#2ca02c"),
        ("l0_abl", "L0-abl", "#9467bd"),
    ]:
        xs, ys = series(data, key, "val_bpb")
        ax.plot(xs, ys, label=label, color=color, linewidth=2, marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation bpb (bits per byte)")
    ax.set_title("Pretraining validation bpb — d8 on 2×T4 (22k steps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "val_bpb_22k.png", dpi=160)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label, color in [
        ("vanilla_pilot", "Vanilla (1.5k pilot)", "#2ca02c"),
        ("l0_abl_pilot", "L0-abl (1.5k pilot)", "#9467bd"),
    ]:
        xs, ys = series(data, key, "train_loss")
        ax.plot(xs, ys, label=label, color=color, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("Pilot (~25 min) — train loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_pilot.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    pairs = [
        (axes[0], "Final train loss ↓", [data["vanilla"]["summary"]["train_loss"], data["l0_abl"]["summary"]["train_loss"]]),
        (axes[1], "Final val bpb ↓", [data["vanilla"]["summary"]["val_bpb"], data["l0_abl"]["summary"]["val_bpb"]]),
    ]
    for ax, title, vals in pairs:
        bars = ax.bar(["Vanilla", "L0-abl"], vals, color=["#2ca02c", "#9467bd"], width=0.55)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom")
        ax.set_ylim(min(vals) * 0.98, max(vals) * 1.01)
    fig.suptitle("Final metrics after 22k pretrain steps (lower is better)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "final_metrics_bars.png", dpi=160, bbox_inches="tight")
    plt.close()
    print("Wrote plots to", OUT)


if __name__ == "__main__":
    main()
