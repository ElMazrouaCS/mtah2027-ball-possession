"""Metrica open sample games (github.com/metrica-sports/sample-data): tracking + H/A/D labels.

These are OTHER games, with human event data, which the rules explicitly allow for training and
validation. Nothing here touches the competition data.

Labels are built from the events, following the spirit of CHALLENGE.md:
  - D from the moment play stops (ball crosses a line, foul whistled, offside, referee stop)
    until the next on-ball event (the restart: set piece, goal kick, ...).
  - otherwise the team of the most recent possession-defining event (PASS, SET PIECE, SHOT,
    RECOVERY, BALL LOST, BALL OUT). CHALLENGE alone never changes possession.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.environ.get("MTAH_OPENDATA_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opendata", "sample-data", "data"))
LENGTH, WIDTH = 105.0, 68.0
FPS = 25

ON_BALL = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}
STOP_LOST = ("OFFSIDE", "FORCED", "END HALF", "HAND BALL")


# --------------------------------------------------------------------------- events
def _events_csv(g):
    e = pd.read_csv(os.path.join(ROOT, "Sample_Game_%d" % g, "Sample_Game_%d_RawEventsData.csv" % g))
    out = pd.DataFrame({
        "team": (e["Team"] == "Away").astype(int),          # 0 = Home, 1 = Away
        "type": e["Type"].str.upper(),
        "subtype": e["Subtype"].fillna("").str.upper(),
        "period": e["Period"].astype(int),
        "start": e["Start Frame"].astype(int) - 1,          # frames counted from 1 in the file
        "end": e["End Frame"].astype(int) - 1,
        "x": e["Start X"].astype(float),
        "y": e["Start Y"].astype(float),
    })
    return out


def _events_json(g):
    d = json.load(open(os.path.join(ROOT, "Sample_Game_%d" % g, "Sample_Game_%d_events.json" % g)))
    rows = []
    for ev in d["data"]:
        sub = ev.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        else:
            sub = ""
        rows.append({
            "team": 0 if ev["team"]["id"].endswith("A") else 1,
            "type": ev["type"]["name"].upper(),
            "subtype": (sub or "").upper(),
            "period": int(ev["period"]),
            "start": int(ev["start"]["frame"]) - 1,
            "end": int(ev["end"]["frame"] or ev["start"]["frame"]) - 1,
            "x": np.nan if ev["start"]["x"] is None else float(ev["start"]["x"]),
            "y": np.nan if ev["start"]["y"] is None else float(ev["start"]["y"]),
        })
    return pd.DataFrame(rows)


def events(g):
    e = _events_csv(g) if g in (1, 2) else _events_json(g)
    e.loc[e["end"] < e["start"], "end"] = e["start"]
    return e.sort_values(["start"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- labels
def stop_frame(row):
    """Frame at which this event stops play, or None."""
    t, s = row["type"], row["subtype"]
    if t == "BALL OUT":
        return row["end"]
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return row["end"]
    if t == "FAULT RECEIVED":
        return row["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return row["start"]
    if t == "BALL LOST" and any(k in s for k in STOP_LOST):
        return row["end"]
    return None


CONFIRM_S = 3.0      # CHALLENGE.md proxies for "controlled": keeps the ball a few seconds ...
CONFIRM_PASSES = 3   # ... or completes more than two consecutive passes


def apply_control_rule(lab, e, blocks, confirm_s=CONFIRM_S, confirm_passes=CONFIRM_PASSES):
    """Merge possession spells that never reach *controlled* possession into the previous owner.

    A spell that starts right after a stoppage is always kept (CHALLENGE.md: after a restart there
    is a clear possessor). A spell that starts with a turnover is kept only if that team keeps the
    ball >= confirm_s or makes >= confirm_passes passes in it; otherwise possession never changed.
    """
    lab = lab.copy()
    passes = e[e["type"] == "PASS"]
    for first, last in blocks:
        changed = True
        while changed:
            changed = False
            w = lab[first:last + 1]
            st = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
            for i in range(1, len(st) - 1):
                k, prev = w[st[i]], w[st[i] - 1]
                if k not in "HA" or prev not in "HA":
                    continue
                a, b = first + st[i], first + st[i + 1]
                team = 0 if k == "H" else 1
                npass = int(((passes["start"] >= a) & (passes["start"] < b) &
                             (passes["team"] == team)).sum())
                if (b - a) / FPS < confirm_s and npass < confirm_passes:
                    lab[a:b] = prev
                    changed = True
                    break
    return lab


def labels(e, n, control_rule=True):
    """(labels array of 'H','A','D','X', [(first, last_inclusive)] graded blocks per period)."""
    lab = np.full(n, "X", dtype="<U1")
    blocks = []
    for p in sorted(e["period"].unique()):
        ep = e[e["period"] == p].reset_index(drop=True)
        onball = ep[ep["type"].isin(ON_BALL)]
        first = int(onball["start"].min())
        last = int(min(max(ep["end"].max(), ep["start"].max()), n - 1))
        blocks.append((first, last))

        owner = np.full(n, -1, dtype=np.int8)
        for _, r in onball.iterrows():
            owner[r["start"]:] = r["team"]           # forward fill, later events overwrite
        dead = np.zeros(n, dtype=bool)
        starts = onball["start"].to_numpy()
        for _, r in ep.iterrows():
            f = stop_frame(r)
            if f is None:
                continue
            nxt = starts[starts > f]
            restart = int(nxt[0]) if len(nxt) else last + 1
            dead[f:restart] = True
        seg = slice(first, last + 1)
        lab[seg] = np.where(dead[seg], "D", np.where(owner[seg] == 0, "H",
                                                     np.where(owner[seg] == 1, "A", "D")))
    if control_rule:
        lab = apply_control_rule(lab, e, blocks)
    return lab, blocks


# --------------------------------------------------------------------------- tracking
def _tracking_csv(g):
    frames = []
    for side in ("Home", "Away"):
        t = pd.read_csv(os.path.join(ROOT, "Sample_Game_%d" % g,
                                     "Sample_Game_%d_RawTrackingData_%s_Team.csv" % (g, side)),
                        skiprows=2)
        frames.append(t)
    home, away = frames
    n = len(home)
    hx = home.iloc[:, 3:-2:2].to_numpy(float); hy = home.iloc[:, 4:-2:2].to_numpy(float)
    ax = away.iloc[:, 3:-2:2].to_numpy(float); ay = away.iloc[:, 4:-2:2].to_numpy(float)
    px = np.hstack([hx, ax]) * LENGTH
    py = np.hstack([hy, ay]) * WIDTH
    teams = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    bx = home.iloc[:, -2].to_numpy(float) * LENGTH
    by = home.iloc[:, -1].to_numpy(float) * WIDTH
    return dict(px=px, py=py, bx=bx, by=by, teams=teams, n=n)


def _tracking_epts(g):
    meta = ET.parse(os.path.join(ROOT, "Sample_Game_%d" % g, "Sample_Game_%d_metadata.xml" % g))
    root = meta.getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    players = [(p.get("id"), p.get("teamId")) for p in root.iter("Player")]
    # player order in the raw line = order of PlayerChannelRef groups in the first DataSpec
    spec = next(root.iter("DataFormatSpecification"))
    order = []
    for ref in spec.iter("PlayerChannelRef"):
        pid = ref.get("playerChannelId").rsplit("_", 1)[0]
        if pid not in order:
            order.append(pid)
    team_of = dict(players)
    chan_player = {}
    for pc in root.iter("PlayerChannel"):
        chan_player[pc.get("id").rsplit("_", 1)[0]] = pc.get("playerId")
    teams = np.array([0 if team_of[chan_player.get(c, c)] == team_ids[0] else 1 for c in order],
                     dtype=np.int8)
    rows_p, rows_b, idx = [], [], []
    with open(os.path.join(ROOT, "Sample_Game_%d" % g, "Sample_Game_%d_tracking.txt" % g)) as fh:
        for line in fh:
            fr, pl, ball = line.rstrip("\n").split(":")
            idx.append(int(fr))
            vals = []
            for grp in pl.split(";"):
                a = grp.split(",")
                vals += [float(a[0]) if a[0] not in ("NaN", "") else np.nan,
                         float(a[1]) if a[1] not in ("NaN", "") else np.nan]
            rows_p.append(vals)
            b = ball.split(",")
            rows_b.append([float(b[0]) if b[0] not in ("NaN", "") else np.nan,
                           float(b[1]) if b[1] not in ("NaN", "") else np.nan])
    idx = np.array(idx)
    n = idx.max() + 1
    P = np.full((n, 2 * len(order)), np.nan); B = np.full((n, 2), np.nan)
    P[idx] = np.array(rows_p); B[idx] = np.array(rows_b)
    return dict(px=P[:, 0::2] * LENGTH, py=P[:, 1::2] * WIDTH, bx=B[:, 0] * LENGTH,
                by=B[:, 1] * WIDTH, teams=teams, n=n)


def tracking(g):
    return _tracking_csv(g) if g in (1, 2) else _tracking_epts(g)


def player_speed(px, py, smooth=12):
    """m/s from positions, centred finite difference over +/- smooth frames."""
    v = np.full(px.shape, np.nan)
    v[smooth:-smooth] = np.hypot(px[2 * smooth:] - px[:-2 * smooth],
                                 py[2 * smooth:] - py[:-2 * smooth]) * FPS / (2 * smooth)
    return v


def restarts(e, lab, blocks):
    """[(frame, kind, team, x_m, y_m)] for every D -> alive change inside graded blocks.

    kind follows the Smart Tagging vocabulary where one exists, else 'other'.
    """
    out = []
    onball = e[e["type"].isin(ON_BALL)]
    for first, last in blocks:
        w = lab[first:last + 1]
        for i in np.flatnonzero((w[:-1] == "D") & (w[1:] != "D")) + 1:
            f = first + int(i)
            at = onball[(onball["start"] >= f - 2) & (onball["start"] <= f + 2)]
            kind, team, x, y = "other", None, np.nan, np.nan
            for _, r in at.iterrows():
                t, s = r["type"], r["subtype"]
                team = int(r["team"]); x, y = r["x"], r["y"]
                if t == "SET PIECE":
                    kind = ("Throw-in" if "THROW" in s else "Corner Kick" if "CORNER" in s else
                            "Kick Off" if "KICK OFF" in s else "Free Kick")
                    break
                if "GOAL KICK" in s:
                    kind = "Goal Kick"
                    break
            out.append((f, kind, team, x * LENGTH if np.isfinite(x) else np.nan,
                        y * WIDTH if np.isfinite(y) else np.nan))
    return out
