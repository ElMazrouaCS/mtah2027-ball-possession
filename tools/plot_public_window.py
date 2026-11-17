"""Timeline of the public 5-minute window: public ground truth vs our submission vs baseline.

    python tools/plot_public_window.py       # writes docs/assets/public_window.png

Reads only files of the public starter repository (evaluation/gt/*_public.csv,
evaluation/results/baseline_smooth50.csv) and our own N7_TACTIC.csv. For display only:
nothing here feeds back into the method.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6b66", "#d9d8d2", "#fcfcfb"
COLORS = {"H": "#2a78d6", "A": "#eb6834", "D": "#b8b7b0"}
FPS = 25
plt.rcParams.update({"font.family": ["Inter", "Segoe UI", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10, "text.color": INK,
                     "axes.edgecolor": GRID, "xtick.color": MUTED})


def letters(path):
    """frame-indexed array of H/A/D/X from a frame,label CSV."""
    df = pm.read_label_csv(path)
    out = np.full(int(df["frame"].max()) + 1, "X", dtype="<U1")
    out[df["frame"].to_numpy()] = df["label"].astype(str).str.strip().str.upper().to_numpy()
    return out


def spells(lab):
    st = np.concatenate(([0], np.flatnonzero(lab[1:] != lab[:-1]) + 1, [len(lab)]))
    return [(st[i], st[i + 1], lab[st[i]]) for i in range(len(st) - 1)]


def main():
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    rows = [("Public ground truth", gt_path),
            ("N7_TACTIC", os.path.join(ROOT, "N7_TACTIC.csv")),
            ("Baseline smooth50", os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))]
    labs = [letters(p) for _, p in rows]
    gt = labs[0]
    graded = np.flatnonzero(np.isin(gt, ["H", "A", "D"]))
    a, b = graded[0], graded[-1] + 1

    fig, ax = plt.subplots(figsize=(9, 2.3), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, ((name, _), lab) in enumerate(zip(rows, labs)):
        lab = lab[a:b]
        y = len(rows) - 1 - i
        n = 0
        for s, e, k in spells(lab):
            if k in COLORS:
                ax.add_patch(Rectangle((s / FPS / 60, y - 0.34), (e - s) / FPS / 60, 0.68,
                                       color=COLORS[k], lw=0))
                n += k in "HA"
        ax.text(-0.08, y, name, ha="right", va="center", fontweight="semibold" if i == 1 else None)
        ax.text((b - a) / FPS / 60 + 0.08, y, "{} poss.".format(n), va="center", color=MUTED)
    ax.set_xlim(0, (b - a) / FPS / 60); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([]); ax.set_xticks(range(0, 6)); ax.set_xticklabels(["{}:00".format(m) for m in range(6)])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    for k, nm, x in (("H", "Barcelona", 0.0), ("A", "Athletic Club", 0.14), ("D", "dead ball", 0.31)):
        ax.add_patch(Rectangle((x, 1.08), 0.02, 0.07, transform=ax.transAxes, color=COLORS[k], clip_on=False))
        ax.text(x + 0.028, 1.115, nm, transform=ax.transAxes, va="center", color=MUTED, fontsize=9)
    ax.set_title("Public five-minute window", loc="left", fontsize=12.5, fontweight="semibold", pad=26)
    out = os.path.join(ROOT, "docs", "assets", "public_window.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print("written", out)


if __name__ == "__main__":
    main()
