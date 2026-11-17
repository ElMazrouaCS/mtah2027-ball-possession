"""Static figures of the README.

    python tools/make_figures.py      # writes docs/assets/pipeline.png and docs/assets/bench.png

bench.png plots the leave-one-game-out results recorded from
    python -m extval.bench                 (baseline per game)
    python -m extval.tune --holdout {1,2,3} (untuned and tuned scores on the held-out game)
"""

import os

import matplotlib

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6b66", "#d9d8d2", "#fcfcfb"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b8b7b0"
plt.rcParams.update({"font.family": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})

# spellF1 on the held-out game, open-data bench (mean of 4 degradation scenarios)
BENCH = {
    "Game 1": {"baseline": 0.291, "untuned": 0.573, "tuned": 0.659},
    "Game 2": {"baseline": 0.274, "untuned": 0.571, "tuned": 0.614},
    "Game 3": {"baseline": 0.247, "untuned": 0.581, "tuned": 0.626},
}


def bench_figure():
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = (("baseline", "Organisers' baseline (smooth50)", GREY),
              ("untuned", "Our pipeline, a-priori parameters", "#86b6ef"),
              ("tuned", "Our pipeline, tuned on the two other games", BLUE))
    w = 0.26
    for i, (key, label, color) in enumerate(series):
        xs = [g + (i - 1) * w for g in range(len(BENCH))]
        ys = [BENCH[name][key] for name in BENCH]
        ax.bar(xs, ys, width=w - 0.03, color=color, label=label, zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, "{:.2f}".format(y), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(BENCH))); ax.set_xticklabels(["Held-out " + g.lower() for g in BENCH])
    ax.set_ylim(0, 0.75); ax.set_ylabel("spellF1")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    ax.set_title("Every gain is measured on a game the tuning never saw", loc="left",
                 fontsize=12.5, fontweight="semibold", pad=14)
    fig.savefig(os.path.join(ASSETS, "bench.png"), bbox_inches="tight", facecolor=SURFACE)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(10, 3.4), dpi=150, facecolor=SURFACE)
    ax.set_xlim(-1, 101); ax.set_ylim(0, 37); ax.axis("off")
    # x, y, width, height, title, lines, highlighted
    boxes = [
        (0, 9.5, 19, 14, "Inputs", ["automatic tactical tracking", "22 players + ball, 25 fps", "Smart Tagging"], False),
        (25, 18.5, 24, 13, "Dead-ball evidence", ["long ball gaps, slow players,", "ball past a line, set-piece", "window, fast ball (against)"], False),
        (25, 1, 24, 13, "Home-vs-away evidence", ["player within 1.5 m of the ball,", "direction of play,", "Smart Tagging phase team"], False),
        (55, 9.5, 23, 14, "3-state HMM", ["H, A, D decoded by exact", "Viterbi: total evidence minus", "a cost per state change"], True),
        (84, 9.5, 16, 14, "N7_TACTIC.csv", ["199,319 frames", "H, A, D when graded,", "X elsewhere"], False),
    ]
    for x, y, w, h, title, lines, main in boxes:
        if main:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                        fc=BLUE, alpha=0.12, ec="none"))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc="none", ec=BLUE if main else MUTED, lw=1.6 if main else 1))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", fontweight="semibold", fontsize=9.5)
        for k, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 6.8 - 2.6 * k, ln, ha="center", fontsize=7.6, color=MUTED)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1, mutation_scale=10)
    ax.annotate("", xy=(24.4, 25), xytext=(19.6, 19), arrowprops=arrow)
    ax.annotate("", xy=(24.4, 7.5), xytext=(19.6, 14), arrowprops=arrow)
    ax.annotate("", xy=(54.4, 19), xytext=(49.6, 25), arrowprops=arrow)
    ax.annotate("", xy=(54.4, 14), xytext=(49.6, 7.5), arrowprops=arrow)
    ax.annotate("", xy=(83.4, 16.5), xytext=(78.6, 16.5), arrowprops=arrow)
    ax.text(0, 35, "A possession changes only when sustained evidence pays the switching cost",
            fontsize=12.5, fontweight="semibold")
    fig.savefig(os.path.join(ASSETS, "pipeline.png"), bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    pipeline_figure()
    bench_figure()
    print("written", ASSETS)
