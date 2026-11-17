"""IDSSE / Sportec open data (7 Bundesliga matches, DFL) as extra bench games.

Bassek, Rein, Weber, Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195. CC-BY-4.0. Loaded with kloppy (it downloads the files).

Why these matches matter: their TRACAB position data carries, for every frame, the DFL channels
BallStatus (alive/dead) and BallPossession (home/away). Labels are therefore built the same way
as the challenge ground truth ("D when the ball is dead, otherwise the owning team"), from
corrected tracking, with no event-to-frame rules of ours in between.

    pip install "kloppy>=3.19" lxml
    python -m extval.idsse_probe          # check the download and print label statistics
"""

import os

import numpy as np

from . import opendata as od

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "idsse")


def _convert(match_id):
    from kloppy import sportec

    ds = sportec.load_open_tracking_data(match_id=match_id, coordinates="kloppy")
    df = ds.to_df()
    home, away = ds.metadata.teams

    cols_x, teams = [], []
    for t, team in enumerate((home, away)):
        for p in team.players:
            c = "{}_x".format(p.player_id)
            if c in df.columns:
                cols_x.append(c)
                teams.append(t)
    px = df[cols_x].to_numpy(float) * od.LENGTH
    py = df[[c[:-2] + "_y" for c in cols_x]].to_numpy(float) * od.WIDTH
    bx = df["ball_x"].to_numpy(float) * od.LENGTH
    by = df["ball_y"].to_numpy(float) * od.WIDTH

    state = df["ball_state"].astype(str).str.lower().to_numpy()
    owner = df["ball_owning_team_id"].astype(str).to_numpy()
    raw = np.where(state == "dead", "D", np.where(owner == str(home.team_id), "H", "A"))

    period = df["period_id"].to_numpy()
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(idx[0]), int(idx[-1])))
    return dict(px=px.astype(np.float32), py=py.astype(np.float32), bx=bx, by=by,
                teams=np.array(teams, dtype=np.int8), raw=raw.astype("<U1"),
                blocks=np.array(blocks), fps=np.array(ds.metadata.frame_rate))


CONFIRM_S = 3.0   # same CHALLENGE.md control rule as extval.opendata (passes are not available here)


def control_rule(lab, blocks, confirm_s=CONFIRM_S):
    """A turnover counts only if the new team keeps the ball >= confirm_s (CHALLENGE.md).

    The DFL BallPossession channel switches on every touch (25-37% of its spells last under 2 s),
    while the challenge only counts *controlled* possession. A spell that starts with a turnover
    and lasts less than confirm_s is given back to the previous owner; spells that start right
    after a stoppage are kept (clear possessor after a restart). Same rule as the Metrica labels.
    """
    lab = lab.copy()
    for first, last in blocks:
        changed = True
        while changed:
            changed = False
            w = lab[first:last + 1]
            st = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
            for i in range(1, len(st) - 1):
                k, prev = w[st[i]], w[st[i] - 1]
                if k in "HA" and prev in "HA" and (st[i + 1] - st[i]) < confirm_s * od.FPS:
                    w[st[i]:st[i + 1]] = prev
                    changed = True
                    break
    return lab


def load(match_id, rule=True):
    """(trk, labels, blocks) with trk in metres, like extval.opendata.

    rule=True applies the CHALLENGE.md control rule to the DFL channel (default, as for Metrica);
    rule=False returns the channel as delivered.
    """
    path = os.path.join(CACHE, match_id + ".npz")
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, **_convert(match_id))
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    if int(d["fps"]) != od.FPS:
        raise ValueError("expected 25 fps, got {}".format(int(d["fps"])))
    blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
    trk = dict(px=d["px"], py=d["py"], bx=d["bx"], by=d["by"], teams=d["teams"],
               n=len(d["bx"]))
    lab = control_rule(d["raw"], blocks) if rule else d["raw"]
    return trk, lab, blocks


def merge_short(lab, blocks, min_s):
    """Possession spells shorter than min_s between two spells of the same team are merged.

    Not applied by default: the DFL channel is used as delivered. Kept for the probe, to see how
    much flicker the channel itself contains.
    """
    lab = lab.copy()
    for first, last in blocks:
        w = lab[first:last + 1]
        st = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(1, len(st) - 2):
            a, b = st[i], st[i + 1]
            if w[a] in "HA" and w[st[i - 1]] == w[b] and w[b] in "HA" and (b - a) < min_s * od.FPS:
                w[a:b] = w[b]
    return lab


def restarts(trk, lab, blocks):
    """[(frame, kind, team, x_m, y_m)] at every D -> alive change; kind from the ball's position.

    Same vocabulary as the Smart Tagging so the simulated tags can be drawn with realistic recall.
    """
    out = []
    L, W = od.LENGTH, od.WIDTH
    for first, last in blocks:
        w = lab[first:last + 1]
        for i in np.flatnonzero((w[:-1] == "D") & (w[1:] != "D")) + 1:
            f = first + int(i)
            x, y = trk["bx"][f], trk["by"][f]
            team = 0 if lab[f] == "H" else 1
            if not (np.isfinite(x) and np.isfinite(y)):
                kind = "other"
            elif min(x, L - x) < 2.0 and min(y, W - y) < 3.0:
                kind = "Corner Kick"
            elif min(y, W - y) < 1.5:
                kind = "Throw-in"
            elif min(x, L - x) < 6.5 and abs(y - W / 2) < 10.5:
                kind = "Goal Kick"
            elif np.hypot(x - L / 2, y - W / 2) < 2.0:
                kind = "Kick Off"
            else:
                kind = "Free Kick"
            out.append((f, kind, team, x, y))
    return out
