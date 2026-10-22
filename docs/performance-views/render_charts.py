#!/usr/bin/env python3
"""Render compact crimson optimization charts from recorded measurements.

Requires matplotlib >= 3.8. Remeasurements remain in the source data but are
omitted from the displayed sequence. Parent gains retain their recorded values.
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
RED = "#8C0000"
LIGHT = dict(bg="#FFFFFF", fg="#292122", muted="#79696B", accent=RED,
             grid="#EADFE0", card="#FAF5F5")
DARK = dict(bg="#290D13", fg="#FFF5F2", muted="#D1AEB4", accent="#FFA4A4",
            grid="#59313A", card="#3A1720")
WARM = dict(bg="#FBF7F3", fg="#302326", muted="#806B6F", accent=RED,
            grid="#E8DADD", card="#F3E9E6")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                     "svg.fonttype": "none", "pdf.fonttype": 42,
                     "svg.hashsalt": "faster-rae-crimson"})


def load():
    data = json.loads((ROOT / "measurement-history.json").read_text())
    stages = [dict(s) for s in data["stages"] if s["kind"] != "remeasurement"]
    assert stages[0]["kind"] == "reference"
    for i, stage in enumerate(stages):
        stage["display_id"] = "B" if i == 0 else str(i)
        if stage["kind"] == "reference":
            stage["label"] = "Baseline"
    assert len({s["id"] for s in stages}) == len(stages)
    assert all(math.isfinite(s["iter_ms"]) and s["iter_ms"] > 0 for s in stages)
    assert all(math.isfinite(s["speedup"]) and s["speedup"] > 0 for s in stages)
    assert all(s["retained"] for s in stages)
    assert all(s["parent_speedup"] > 0 for s in stages[1:])
    return data, stages


def save(fig, name):
    for ext in ("svg", "png", "pdf"):
        metadata = {"Date": None} if ext == "svg" else (
            {"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        path = ROOT / f"{name}.{ext}"
        fig.savefig(path, dpi=200, facecolor=fig.get_facecolor(), metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def frame(ax, theme):
    ax.set_facecolor(theme["bg"])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0, pad=8, labelsize=12)
    ax.grid(axis="y", color=theme["grid"], lw=.7, zorder=0)
    ax.set_axisbelow(True)


def heading(fig, theme, title, final):
    fig.text(.09, .96, "FASTER–RAE", fontsize=11, weight="bold", color=theme["accent"])
    fig.text(.09, .913, title, fontsize=21.5, weight="bold", color=theme["fg"])
    fig.text(.94, .928, f"{final:.2f}×", fontsize=29, ha="right", weight="bold", color=theme["accent"])
    fig.text(.94, .902, "total speedup", fontsize=10.5, ha="right", color=theme["muted"])


def speed_axis(ax, stages, theme):
    values = [s["speedup"] for s in stages]
    frame(ax, theme)
    ax.plot(range(len(values)), values, color=theme["accent"], lw=2.3,
            marker="o", markersize=6.5, zorder=3)
    for i, value in enumerate(values):
        offset = -23 if i in (2, 4, 6) else 11
        ax.annotate(f"{value:.2f}×", (i, value), xytext=(0, offset),
                    textcoords="offset points", ha="center", fontsize=12,
                    color=theme["accent"] if i == len(values)-1 else theme["fg"],
                    weight="bold")
    ax.set_ylim(.7, 6)
    ax.set_xlim(-.3, len(values)-.65)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xticks(range(len(values)), [s["display_id"] for s in stages])


def gains_axis(ax, stages, theme):
    frame(ax, theme)
    gains = [100*(s["parent_speedup"]-1) for s in stages[1:]]
    ax.bar(range(1, len(stages)), gains, color=theme["accent"], width=.20, zorder=3)
    for i, value in enumerate(gains, 1):
        ax.text(i, value+6, f"+{value:.1f}%", ha="center", va="bottom",
                fontsize=11.5, weight="bold", color=theme["fg"])
    ax.set_xlim(-.3, len(stages)-.65)
    ax.set_ylim(0, 185)
    ax.set_yticks([0, 75, 150])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xticks(range(1, len(stages)), [s["display_id"] for s in stages[1:]])


def key(fig, stages, theme, top=.263, step=.061):
    for i, stage in enumerate(stages):
        x, y = .09 + (i % 2)*.44, top - (i // 2)*step
        fig.add_artist(FancyBboxPatch((x, y-.034), .41, .05,
                       boxstyle="round,pad=0.003,rounding_size=0.006",
                       transform=fig.transFigure, facecolor=theme["card"], edgecolor="none"))
        fig.text(x+.015, y-.009, stage["display_id"], fontsize=13, color=theme["accent"], weight="bold")
        fig.text(x+.054, y-.009, stage["label"], fontsize=12.5, color=theme["fg"], weight="bold")


def paper(data, stages, theme=LIGHT, name="01-paper"):
    fig = plt.figure(figsize=(9, 8.8), facecolor=theme["bg"])
    heading(fig, theme, "Agent Optimization History", stages[-1]["speedup"])
    fig.text(.09, .855, "Cumulative speedup", fontsize=12, color=theme["muted"])
    speed_axis(fig.add_axes([.09, .598, .85, .235]), stages, theme)
    fig.text(.09, .523, "Throughput gain vs. measured parent", fontsize=12, color=theme["muted"])
    gains_axis(fig.add_axes([.09, .352, .85, .15]), stages, theme)
    key(fig, stages, theme)
    save(fig, name)


def editorial(data, stages):
    theme = WARM
    fig = plt.figure(figsize=(9, 8.8), facecolor=theme["bg"])
    heading(fig, theme, "Agent Optimization History", stages[-1]["speedup"])
    for x, label, stage in [(.09, "BASELINE", stages[0]), (.53, "FINAL", stages[-1])]:
        fig.text(x, .835, label, fontsize=11, color=theme["muted"], weight="bold")
        fig.text(x, .792, f"{stage['iter_ms']:,.2f} ms", fontsize=24, color=theme["accent"], weight="bold")
    speed_axis(fig.add_axes([.09, .487, .85, .25]), stages, theme)
    # Two columns leave enough room for readable labels and timing values.
    for i, stage in enumerate(stages):
        x, y = .09+(i % 2)*.44, .395-(i // 2)*.095
        fig.add_artist(FancyBboxPatch((x, y-.047), .41, .075,
                       boxstyle="round,pad=0.004,rounding_size=0.007",
                       transform=fig.transFigure, facecolor=theme["card"], edgecolor="none"))
        fig.text(x+.015, y, stage["display_id"], fontsize=13, color=theme["accent"], weight="bold")
        fig.text(x+.054, y, stage["label"], fontsize=12.5, color=theme["fg"], weight="bold")
        fig.text(x+.054, y-.027, f"{stage['iter_ms']:,.2f} ms", fontsize=12, color=theme["muted"])
    save(fig, "03-editorial")


if __name__ == "__main__":
    data, stages = load()
    paper(data, stages)
    paper(data, stages, DARK, "02-kda-dark")
    editorial(data, stages)
    print("Rendered three compact crimson styles as SVG, PNG and PDF.")
