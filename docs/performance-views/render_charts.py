#!/usr/bin/env python3
"""Render three views of the same measured optimization history (CPU only).

python3 docs/performance-views/render_charts.py
Requires matplotlib >= 3.8. All values come from measurement-history.json; no fitted data.
"""
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "svg.fonttype": "none", "pdf.fonttype": 42})


def load():
    data = json.loads((ROOT / "measurement-history.json").read_text())
    stages = data["stages"]
    assert stages[0]["kind"] == "reference"
    for stage in stages:
        stage["latency_ms"] = stage["iter_ms"]
        stage["short_label"] = stage["label"]
        stage["display_id"] = "B" if stage["order"] == 0 else str(stage["order"])
    data["hardware_note"] = "Historical reported full checks · 2 Hopper GPUs · local batch 64 · global batch 128 · phase weighting 6:2:8."
    data["interpretation"] = "Sequential configurations, not isolated causal ablations. Diamond = remeasurement. Archived aggregates; not a new rerun."
    assert len({s["id"] for s in stages}) == len(stages)
    assert all(math.isfinite(s["latency_ms"]) and s["latency_ms"] > 0 for s in stages)
    assert all(math.isfinite(s["speedup"]) and s["speedup"] > 0 for s in stages)
    assert all(s["retained"] for s in stages)
    assert all(s["parent_speedup"] > 0 for s in stages[1:])
    return data, stages


def save(fig, name):
    for ext in ("svg", "png", "pdf"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=180, facecolor=fig.get_facecolor())
        if ext == "svg":
            output = ROOT / f"{name}.{ext}"
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


def frame(ax, fg, grid):
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=fg, length=0, pad=8)
    ax.yaxis.label.set_color(fg)
    ax.xaxis.label.set_color(fg)
    ax.grid(axis="y", color=grid, lw=.7, zorder=0)
    ax.set_axisbelow(True)


def speed_axis(ax, stages, color, fg, grid):
    y = [s["speedup"] for s in stages]
    ax.plot(range(len(y)), y, color=color, lw=2.6, zorder=3)
    for i, (s, val) in enumerate(zip(stages, y)):
        repeat = s["kind"] == "remeasurement"
        ax.scatter(i, val, s=65, marker="D" if repeat else "o",
                   facecolor=ax.get_facecolor() if repeat else color,
                   edgecolor=fg if repeat else color, linewidth=1.5, zorder=4)
        ax.annotate(f"{val:.2f}×", (i, val), xytext=(0, 12 if i % 2 == 0 else -23),
                    textcoords="offset points", ha="center", fontsize=10,
                    color=fg, weight="bold")
    frame(ax, fg, grid)
    ax.axhline(1, color=grid, ls="--", lw=1)
    ax.set_ylim(.65, max(y) + .75)
    ax.set_xlim(-.3, len(y)-.7)
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylabel("Cumulative speedup")
    ax.set_xticks(range(len(y)), [s["display_id"] for s in stages])
    return y


def footnote(fig, data, fg, y=.022):
    fig.text(.07, y, data["hardware_note"] + "\n" + data["interpretation"],
             color=fg, fontsize=8.2, linespacing=1.7, va="bottom")


def paper(data, stages):
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(14, 9.6),
                                  gridspec_kw={"height_ratios": [2, 1.5]})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.075, right=.97, top=.83, bottom=.22, hspace=.55)
    fig.text(.075, .945, "Faster-RAE  /  optimization history", fontsize=22, weight="bold", color="#12263a")
    fig.text(.075, .899, "A measured sequence of retained configurations", fontsize=12, color="#536574")
    speed_axis(ax, stages, "#087c88", "#203543", "#dce3e8")
    ax.set_xlabel("Configuration in recorded order")
    # Reported same-session parent ratios, not ratios of adjacent archived rows.
    ratios = [s["parent_speedup"] for s in stages[1:]]
    cols = ["#b4bcc5" if s["kind"] == "remeasurement" else "#087c88" for s in stages[1:]]
    bars.bar(range(1,len(stages)), [100*(r-1) for r in ratios], color=cols, width=.5)
    for i,r in enumerate(ratios,1):
        label = f"+{100*(r-1):.1f}%"
        if stages[i]["kind"] == "remeasurement":
            label += "\n(remeasurement)"
        bars.text(i,100*(r-1)+4, label, ha="center", va="bottom", fontsize=9, color="#203543")
    bars.set_ylim(0,180)
    bars.set_ylabel("Throughput change (%)")
    bars.set_xticks(range(1,len(stages)), [s["display_id"] for s in stages[1:]])
    bars.set_title("Reported same-session parent comparisons · includes remeasurement", loc="left", fontsize=11)
    frame(bars,"#203543","#dce3e8")
    labels = [f"{s['display_id']}  {s['short_label']}" for s in stages]
    for i,label in enumerate(labels):
        fig.text(.075+(i%3)*.31,.158-(i//3)*.029,label,fontsize=8.5,color="#304655")
    footnote(fig,data,"#536574")
    save(fig,"01-paper")


def dark(data, stages):
    bg, fg, muted, green = "#111914", "#eef4ec", "#9baaa0", "#c7ff3d"
    fig, ax = plt.subplots(figsize=(14,8.6), facecolor=bg)
    ax.set_facecolor(bg)
    fig.subplots_adjust(left=.075,right=.97,bottom=.35,top=.7)
    fig.text(.075,.94,"HOW THE RESULT EVOLVED",color=green,fontsize=10,weight="bold")
    fig.text(.075,.878,f"From {stages[0]["speedup"]:.2f}× to {stages[-1]["speedup"]:.2f}×",color=fg,fontsize=30,weight="bold")
    fig.text(.075,.826,"Historical H200-class benchmark · retained configurations · archived report",color=muted,fontsize=12)
    y = speed_axis(ax,stages,green,fg,"#334539")
    ax.fill_between(range(len(y)),1,y,color=green,alpha=.045)
    for idx, label, tx in [(1,"TF32 / cuBLAS",.12),(5,"Compilation",.48),(7,"Local BF16",.79)]:
        ax.annotate(label,xy=(idx,y[idx]),xycoords="data",xytext=(tx,1.15),textcoords="axes fraction",
                    color=green,weight="bold",ha="center",fontsize=10,
                    arrowprops={"arrowstyle":"->","color":green,"linestyle":"--","lw":1,
                                "connectionstyle":"arc3,rad=.15"})
    fig.text(.075,.265,"CONFIGURATION KEY",color=muted,fontsize=9,weight="bold")
    for i,s in enumerate(stages):
        fig.text(.075+(i%3)*.31,.227-(i//3)*.038,f"{s['display_id']}  {s['short_label']}",
                 color=muted if s["kind"]=="remeasurement" else fg,fontsize=9)
    footnote(fig,data,muted,.03)
    save(fig,"02-kda-dark")


def editorial(data, stages):
    bg, fg, accent, grid = "#f5f2e9", "#25362e", "#b14d32", "#d8ddd1"
    fig = plt.figure(figsize=(14,9.8),facecolor=bg)
    fig.text(.07,.94,"RAE, made faster.",fontsize=31,color=fg,weight="bold")
    fig.text(.07,.89,"The measured path, with every retained configuration visible.",fontsize=12,color="#657065")
    base, final=stages[0]["latency_ms"],stages[-1]["latency_ms"]
    for xpos,title,value in [(.07,"REFERENCE",f"{base:,.2f} ms"),(.38,"RETAINED FINAL",f"{final:,.2f} ms"),(.71,"CUMULATIVE",f"{stages[-1]["speedup"]:.2f}×")]:
        fig.text(xpos,.825,title,fontsize=9,color="#657065",weight="bold")
        fig.text(xpos,.779,value,fontsize=25,color=accent if xpos==.71 else fg,weight="bold")
    ax=fig.add_axes([.075,.42,.89,.29],facecolor=bg)
    speed_axis(ax,stages,accent,fg,grid)
    ax.set_xlabel("Configuration in recorded order")
    for i,s in enumerate(stages[1:]):
        col,row=i%4,i//4
        x,y=.075+col*.226,.294-row*.09
        fig.text(x,y,f"{s['display_id']}   {s['latency_ms']} ms",fontsize=11,weight="bold",color=fg)
        fig.text(x,y-.025,s["short_label"],fontsize=8.5,color="#657065")
        ratio=s["parent_speedup"]
        note="Remeasurement only" if s["kind"]=="remeasurement" else f"{ratio:.3f}× vs same-session parent"
        fig.text(x,y-.046,note,fontsize=7.8,color=accent)
    footnote(fig,data,"#657065",.035)
    save(fig,"03-editorial")


if __name__ == "__main__":
    data, stages = load()
    paper(data, stages)
    dark(data, stages)
    editorial(data, stages)
    print("Rendered three styles as SVG, PNG and PDF.")
