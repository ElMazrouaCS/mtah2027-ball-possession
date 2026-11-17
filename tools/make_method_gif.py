"""Animated illustration of the method on an OPEN match (Metrica sample game 3).

    python tools/make_method_gif.py          # writes docs/assets/method.gif

Uses only open data (github.com/metrica-sports/sample-data, cloned in opendata/), degraded exactly
as in the validation bench. Nothing from the competition data is drawn or published.
The 60-second window is picked by a fixed rule (first window after minute 10 of the first half
with at least one stoppage and three possessions), not by hand.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from extval import bench  # noqa: E402
from extval import opendata as od  # noqa: E402
from solution import config, evidence, pipeline  # noqa: E402
from solution.evidence import centred_nanmean  # noqa: E402

OUT = os.path.join(ROOT, "docs", "assets", "method.gif")
WINDOW_S = 60
STEP = 10                      # draw every 10th frame (0.4 s of match per GIF frame)
FPS_GIF = 10

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6b66", "#d9d8d2", "#fcfcfb"
HOME, AWAY, DEAD = "#2a78d6", "#eb6834", "#b8b7b0"
COLORS = {"H": HOME, "A": AWAY, "D": DEAD}
plt.rcParams.update({"font.family": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def spells(lab):
    st = np.concatenate(([0], np.flatnonzero(lab[1:] != lab[:-1]) + 1, [len(lab)]))
    return [(st[i], st[i + 1], lab[st[i]]) for i in range(len(st) - 1)]


def pick_window(gt, blocks):
    first, last = blocks[0]
    L = WINDOW_S * od.FPS
    for a in range(first + 10 * 60 * od.FPS, last - L, od.FPS):
        sp = spells(gt[a:a + L])
        if sum(k == "D" for _, _, k in sp) >= 1 and sum(k in "HA" for _, _, k in sp) >= 3:
            return a, a + L
    raise RuntimeError("no window matches the rule")


def main():
    P = config.params()
    case = bench.build_cases(games=(3,), scenarios={"central": bench.SCENARIOS["central"]})[0]
    trk, gt, blocks = case["trk"], case["lab"], case["blocks"]
    f = pipeline.features(trk, case["tags"], blocks, od.LENGTH, od.WIDTH, P)
    ours = pipeline.decode_from_features(f, blocks, P)
    base = bench.baseline_labels(trk)
    dead_e = evidence.dead_evidence(f["ok"], f["speed"], f["oob"], f["vcoll"], f["gaplen"],
                                    f["sp"][0], P)
    own_e = evidence.owner_evidence(f["ok"], f["speed"], f["dist"][0], f["dist"][1],
                                    f["tagprior"], P, *f["terr"])
    dead_s, own_s = centred_nanmean(dead_e, od.FPS), centred_nanmean(own_e, od.FPS)

    a, b = pick_window(gt, blocks)
    t = (np.arange(a, b) - a) / od.FPS

    fig = plt.figure(figsize=(9, 7.6), dpi=100, facecolor=SURFACE)
    gs = fig.add_gridspec(5, 1, height_ratios=[5.2, 1.0, 1.0, 0.15, 1.05], hspace=0.42,
                          left=0.15, right=0.89, top=0.9, bottom=0.05)
    fig.text(0.15, 0.955, "Evidence per frame, one decision per possession",
             fontsize=14, fontweight="semibold")
    fig.text(0.15, 0.925, "Open Metrica sample game 3, tracking degraded like the provided ATD, "
             "60 s window at 4x speed.", color=MUTED)

    # pitch
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
    ax.add_patch(Rectangle((0, 0), od.LENGTH, od.WIDTH, fill=False, ec=GRID, lw=1.2))
    ax.plot([od.LENGTH / 2] * 2, [0, od.WIDTH], color=GRID, lw=1)
    ax.add_patch(plt.Circle((od.LENGTH / 2, od.WIDTH / 2), 9.15, fill=False, ec=GRID, lw=1))
    for x0 in (0, od.LENGTH - 16.5):
        ax.add_patch(Rectangle((x0, od.WIDTH / 2 - 20.16), 16.5, 40.32, fill=False, ec=GRID, lw=1))
    ax.set_xlim(-4, od.LENGTH + 4); ax.set_ylim(-4, od.WIDTH + 4); ax.set_aspect("equal")
    ax.axis("off")
    home = ax.scatter([], [], s=46, c=HOME, ec=SURFACE, lw=1.2, zorder=3, label="home")
    away = ax.scatter([], [], s=46, c=AWAY, ec=SURFACE, lw=1.2, zorder=3, label="away")
    ball = ax.scatter([], [], s=26, c=INK, zorder=4, label="ball")
    status = ax.text(od.LENGTH / 2, od.WIDTH + 2.2, "", ha="center", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.05), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.2)
    team = trk["teams"]

    # evidence curves
    def curve(axc, y, label, pos, neg, color_pos, color_neg):
        axc.set_facecolor(SURFACE)
        axc.axhline(0, color=GRID, lw=1)
        yy = y[a:b]
        axc.fill_between(t, 0, yy, where=yy > 0, color=color_pos, alpha=0.35, lw=0)
        axc.fill_between(t, 0, yy, where=yy < 0, color=color_neg, alpha=0.35, lw=0)
        axc.plot(t, yy, color=INK, lw=1)
        axc.set_xlim(0, WINDOW_S); lim = np.nanmax(np.abs(yy)) * 1.1 or 1
        axc.set_ylim(-lim, lim); axc.set_yticks([]); axc.set_xticks([])
        axc.text(0, 1.12, label, transform=axc.transAxes, fontweight="semibold")
        axc.text(1, 1.12, pos + "  /  " + neg, transform=axc.transAxes, ha="right", color=MUTED)
        for s in ("top", "right"):
            axc.spines[s].set_visible(False)
        return axc.axvline(0, color=INK, lw=1.2)

    cur1 = curve(fig.add_subplot(gs[1]), dead_s, "Dead-ball evidence", "above: stoppage",
                 "below: live play", DEAD, "#e7e6e0")
    cur2 = curve(fig.add_subplot(gs[2]), own_s, "Home-vs-away evidence", "above: home",
                 "below: away", HOME, AWAY)

    # label strips
    axl = fig.add_subplot(gs[4]); axl.set_facecolor(SURFACE)
    rows = (("Ground truth", gt), ("Our decoder", ours), ("Baseline", base))
    for i, (name, lab) in enumerate(rows):
        y = 2 - i
        for s, e, k in spells(lab[a:b]):
            if k in COLORS:
                axl.add_patch(Rectangle((s / od.FPS, y - 0.36), (e - s) / od.FPS, 0.72,
                                        color=COLORS[k], lw=0))
        n = sum(k in "HA" for _, _, k in spells(lab[a:b]))
        axl.text(-0.8, y, name, ha="right", va="center")
        axl.text(WINDOW_S + 0.8, y, "{} poss.".format(n), va="center", color=MUTED)
    axl.set_xlim(0, WINDOW_S); axl.set_ylim(-0.6, 2.6); axl.set_yticks([])
    axl.set_xticks(range(0, WINDOW_S + 1, 10))
    axl.set_xticklabels(["{} s".format(x) for x in range(0, WINDOW_S + 1, 10)])
    for s in ("top", "right", "left"):
        axl.spines[s].set_visible(False)
    cur3 = axl.axvline(0, color=INK, lw=1.2)
    for k, nm, x in (("H", "home", 0), ("A", "away", 0.12), ("D", "dead ball", 0.24)):
        axl.add_patch(Rectangle((x, 1.2), 0.025, 0.12, transform=axl.transAxes, color=COLORS[k],
                                clip_on=False))
        axl.text(x + 0.035, 1.26, nm, transform=axl.transAxes, va="center", color=MUTED)

    frames = list(range(a, b, STEP))

    def draw(fr):
        px, py = trk["px"][fr], trk["py"][fr]
        vis = np.isfinite(px)
        home.set_offsets(np.c_[px[vis & (team == 0)], py[vis & (team == 0)]])
        away.set_offsets(np.c_[px[vis & (team == 1)], py[vis & (team == 1)]])
        if np.isfinite(trk["bx"][fr]):
            ball.set_offsets([[trk["bx"][fr], trk["by"][fr]]]); status.set_text("")
        else:
            ball.set_offsets(np.empty((0, 2))); status.set_text("ball not tracked")
        x = (fr - a) / od.FPS
        for c in (cur1, cur2, cur3):
            c.set_xdata([x, x])
        return home, away, ball, status, cur1, cur2, cur3

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    FuncAnimation(fig, draw, frames=frames, blit=False).save(OUT, writer=PillowWriter(fps=FPS_GIF))
    print("written", OUT, "window frames", a, b)


if __name__ == "__main__":
    main()
