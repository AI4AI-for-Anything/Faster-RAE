#!/usr/bin/env python3
"""Render the README's H200 comparison from the recorded benchmark.

Run from any directory with Python and matplotlib; no GPU or network is needed.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parent
RECORD = ROOT / "h200-benchmark.json"


def main():
    data = json.loads(RECORD.read_text())
    baseline = data["baseline_weighted_ms"]
    optimized = data["runs"]["faster"]["weighted_ms_per_step"]
    speedup = baseline / optimized
    assert data["complete"]
    assert all(math.isfinite(value) and value > 0 for value in (baseline, optimized))
    assert math.isclose(speedup, data["speedup"])
    assert data["global_batch_size"] == (
        data["world_size"] * data["micro_batch_size"] * data["gradient_accumulation_steps"]
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "svg.fonttype": "none",
        "svg.hashsalt": "faster-rae-h200-benchmark",
    })
    fg, muted, accent = "#292122", "#79696B", "#8C0000"
    fig = plt.figure(figsize=(9, 4.5), facecolor="white")
    fig.text(.06, .88, "FASTER-RAE, fully optimized by AI agents", fontsize=19, weight="bold", color=accent)
    batch = (
        f"RAE Stage-1  ·  {data['world_size']} H200 GPUs  ·  "
        f"{data['image_size']} × {data['image_size']} images  ·  "
        f"global batch {data['global_batch_size']}"
    )
    fig.text(.06, .78, batch, fontsize=10.5, color=muted)
    fig.text(.85, .54, f"{speedup:.2f}×", fontsize=35, weight="bold", color=accent, ha="center")
    fig.text(.85, .43, "training speedup", fontsize=12, color=muted, ha="center")

    ax = fig.add_axes([.20, .24, .43, .45])
    values = [baseline, optimized]
    ax.barh([1, 0], values, height=.24, color=["#D7BDBF", accent], zorder=3)
    for y, value in zip([1, 0], values):
        ax.text(value + baseline * .025, y, f"{value:,.2f} ms", va="center", color=fg, fontsize=11.5, weight="bold")
    ax.set_yticks([1, 0], ["Baseline\n(preset off)", "Faster-RAE\n(preset on)"])
    ax.set_xlim(0, baseline * 1.30)
    ax.set_ylim(-.6, 1.6)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Weighted time per effective iteration (ms) ↓", labelpad=12, color=muted, fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#EADFE0")
    ax.tick_params(axis="both", length=0, pad=9, colors=muted, labelsize=10)
    ax.grid(axis="x", color="#F0E7E8", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)

    for ext in ("svg", "png"):
        path = ROOT / f"04-h200-benchmark.{ext}"
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    print(f"Rendered H200 benchmark: {baseline:,.2f} → {optimized:,.2f} ms ({speedup:.2f}×).")


if __name__ == "__main__":
    main()
