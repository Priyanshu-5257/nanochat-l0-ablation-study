#!/usr/bin/env python3
"""Regenerate all comparison plots from results/data/metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data" / "metrics.json"
OUT = ROOT / "results" / "plots"

ORDER = ["vanilla", "l0_r4", "l0_r6", "l0_mid_router"]
COLORS = {
    "vanilla": "#2ca02c",
    "l0_r4": "#9467bd",
    "l0_r6": "#ff7f0e",
    "l0_mid_router": "#1f77b4",
}


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

    # train loss
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in ORDER:
        xs, ys = series(data, key, "train_loss")
        ax.plot(
            xs,
            ys,
            label=data[key]["label"],
            color=COLORS[key],
            linewidth=2,
            linestyle="--" if key == "l0_r6" else "-",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Train loss")
    ax.set_title("Pretraining train loss — all variants (d8, 2×T4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "train_loss_all.png", dpi=160)
    plt.close()

    # val bpb
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for key in ORDER:
        xs, ys = series(data, key, "val_bpb")
        ax.plot(
            xs,
            ys,
            label=data[key]["label"],
            color=COLORS[key],
            linewidth=2,
            marker="o",
            markersize=2.5,
            linestyle="--" if key == "l0_r6" else "-",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation bpb ↓")
    ax.set_title("Pretraining validation bpb — all variants (d8, 2×T4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "val_bpb_all.png", dpi=160)
    plt.close()

    # final bars
    keys_bar = ORDER
    labels = [data[k]["label"] + ("*" if not data[k].get("complete", True) else "") for k in keys_bar]
    loss_v = [data[k]["summary"]["train_loss"] for k in keys_bar]
    bpb_v = [data[k]["summary"]["val_bpb"] for k in keys_bar]
    cols = [COLORS[k] for k in keys_bar]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, title, vals in [
        (axes[0], "Final train loss ↓", loss_v),
        (axes[1], "Final / last val bpb ↓", bpb_v),
    ]:
        bars = ax.bar(range(len(labels)), vals, color=cols, width=0.65)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(min(vals) * 0.985, max(vals) * 1.008)
    fig.suptitle("* incomplete run (crashed before 22k)", y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "final_metrics_all.png", dpi=160, bbox_inches="tight")
    plt.close()

    # gap vs vanilla
    fig, ax = plt.subplots(figsize=(10, 5))
    van_x, van_y = series(data, "vanilla", "val_bpb")
    van = dict(zip(van_x, van_y))
    for key in ["l0_r4", "l0_r6", "l0_mid_router"]:
        xs, ys = series(data, key, "val_bpb")
        gap_x, gap_y = [], []
        for x, y in zip(xs, ys):
            if not van_x:
                continue
            nearest = min(van_x, key=lambda t: abs(t - x))
            if abs(nearest - x) > 300:
                continue
            gap_x.append(x)
            gap_y.append(y - van[nearest])
        ax.plot(gap_x, gap_y, label=f"{data[key]['label']} − vanilla", color=COLORS[key], linewidth=2)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Δ val bpb (variant − vanilla)")
    ax.set_title("Validation gap vs vanilla (positive = worse)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "val_bpb_gap_vs_vanilla.png", dpi=160)
    plt.close()
    print("Wrote plots to", OUT)


if __name__ == "__main__":
    main()
