"""Turn clean open-data tracking into tracking that looks like the provided automatic ATD.

Every knob is a scenario parameter, because we do NOT know some properties of the automatic
tracking (e.g. how often the ball is lost during live play vs during stoppages). We validate over
a GRID of scenarios and prefer settings that are good across all of them.

What is calibrated, and on what (label-free statistics of the provided tactical ATD, diag_step1/2):
  - ball present ~59% of graded frames, missing almost only in gaps >= 3 s (median ~12 s);
  - short occlusions are filled by the tracker (no 1-3 s gaps) -> we interpolate them linearly;
  - at tagged restarts the ball is missing ~69%, at random instants ~42%;
  - ~20.5 of 22 players visible on average; position jitter of a few cm;
  - spare balls / far-out detections beyond the touchline;
  - Smart Tagging: set pieces opening ~3.5 s before the restart, low recall on free kicks.
"""

from dataclasses import dataclass, field

import numpy as np

from . import opendata as od


@dataclass
class Scenario:
    alive_missing: float = 0.12        # fraction of live-play frames inside a ball gap
    gap_median_s: float = 10.0
    gap_sigma: float = 0.9             # lognormal shape of gap lengths
    dead_visible_prob: float = 0.4    # a stoppage shows the placed ball before the restart
    dead_visible_s: tuple = (2.0, 8.0)
    dead_head_prob: float = 0.5        # the ball stays visible a moment after play stops
    dead_head_s: tuple = (0.0, 4.0)
    short_occl_per_min: float = 3.0    # filled by linear interpolation
    spare_ball_prob: float = 0.1       # per stoppage: a far-out detection appears
    player_visible: float = 0.93
    player_off_mean_s: float = 3.0
    jitter_m: float = 0.05
    ball_offset_m: float = 0.0       # slowly varying ball position error (std per axis)
    ball_offset_tau_s: float = 1.0
    sp_recall: dict = field(default_factory=lambda: {
        "Throw-in": 0.7, "Corner Kick": 1.0, "Goal Kick": 0.75, "Free Kick": 0.2,
        "Kick Off": 1.0, "other": 0.0})
    sp_offset_s: tuple = (2.5, 4.5)
    tag_own_prob: float = 0.7
    tag_other_prob: float = 0.4
    tag_accuracy: float = 0.85
    seed: int = 0


def _runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def degrade(trk, lab, blocks, rst, sc):
    """Returns (trk_degraded, tagging). Deterministic given sc.seed."""
    rng = np.random.default_rng(sc.seed)
    fps = od.FPS
    n = trk["n"]
    bx = trk["bx"].copy(); by = trk["by"].copy()
    graded = np.zeros(n, dtype=bool)
    for a, b in blocks:
        graded[a:b + 1] = True
    alive = graded & np.isin(lab, ["H", "A"])
    dead = graded & (lab == "D")

    # 1. stoppages: open data already has no ball there; optionally show the placed ball
    for a, b in _runs(dead):
        # ball position at the restart = first clean ball position after the stoppage
        after = np.flatnonzero(np.isfinite(trk["bx"][b:b + 2 * fps]))
        if len(after) and rng.random() < sc.dead_visible_prob:
            k = int(rng.uniform(*sc.dead_visible_s) * fps)
            s = max(a, b - k)
            bx[a:s] = np.nan; by[a:s] = np.nan
            bx[s:b] = trk["bx"][b + after[0]] + rng.normal(0, 0.3)
            by[s:b] = trk["by"][b + after[0]] + rng.normal(0, 0.3)
        else:
            bx[a:b] = np.nan; by[a:b] = np.nan
        before = np.flatnonzero(np.isfinite(trk["bx"][max(a - 2 * fps, 0):a]))
        if len(before) and rng.random() < sc.dead_head_prob:
            k = int(rng.uniform(*sc.dead_head_s) * fps)
            lx = trk["bx"][max(a - 2 * fps, 0) + before[-1]]
            ly = trk["by"][max(a - 2 * fps, 0) + before[-1]]
            e2 = min(b, a + k)
            bx[a:e2] = lx + np.linspace(0, rng.normal(0, 2), e2 - a)
            by[a:e2] = ly + np.linspace(0, rng.normal(0, 2), e2 - a)
        if rng.random() < sc.spare_ball_prob and b - a > 2 * fps:
            s = a + int(rng.integers(0, b - a - fps))
            e = min(b, s + int(rng.uniform(1, 5) * fps))
            bx[s:e] = rng.uniform(0, od.LENGTH)
            by[s:e] = rng.choice([-1, 1]) * rng.uniform(8, 20) + (od.WIDTH if rng.random() < .5 else 0)

    # 2. short occlusions during live play, filled by linear interpolation (tracker behaviour)
    n_short = int(alive.sum() / fps / 60 * sc.short_occl_per_min)
    idx_alive = np.flatnonzero(alive)
    for s in rng.choice(idx_alive, size=min(n_short, len(idx_alive)), replace=False):
        e = s + int(rng.uniform(0.2, 2.5) * fps)
        if e + 1 >= n or not (np.isfinite(bx[s - 1]) and np.isfinite(bx[e])):
            continue
        t = np.linspace(0, 1, e - s + 2)[1:-1]
        bx[s:e] = bx[s - 1] + (bx[e] - bx[s - 1]) * t
        by[s:e] = by[s - 1] + (by[e] - by[s - 1]) * t

    # 3. long ball gaps during live play
    target = sc.alive_missing * alive.sum()
    lost = 0
    mu = np.log(sc.gap_median_s * fps)
    while lost < target:
        s = int(rng.choice(idx_alive))
        L = int(np.clip(rng.lognormal(mu, sc.gap_sigma), 3 * fps, 120 * fps))
        e = min(n, s + L)
        newly = np.isfinite(bx[s:e]) & alive[s:e]
        lost += int(newly.sum())
        bx[s:e] = np.nan; by[s:e] = np.nan

    # 4. slowly varying position error + jitter on the ball
    if sc.ball_offset_m > 0:
        w = max(1, int(sc.ball_offset_tau_s * fps))
        for arr in (bx, by):
            z = rng.normal(0, 1, n + w)
            z = (np.cumsum(z)[w:] - np.cumsum(z)[:-w]) / np.sqrt(w)
            arr += sc.ball_offset_m * z
    ok = np.isfinite(bx)
    bx[ok] += rng.normal(0, sc.jitter_m, ok.sum()); by[ok] += rng.normal(0, sc.jitter_m, ok.sum())

    # 5. players: Markov dropouts + jitter, speeds estimated from the noisy positions
    px = trk["px"].copy(); py = trk["py"].copy()
    mean_on = sc.player_off_mean_s * fps * sc.player_visible / (1 - sc.player_visible)
    mean_off = sc.player_off_mean_s * fps
    for j in range(px.shape[1]):
        t, on = 0, True
        while t < n:                              # alternate geometric on/off runs
            L = int(rng.geometric(1.0 / (mean_on if on else mean_off)))
            if not on:
                px[t:t + L, j] = np.nan; py[t:t + L, j] = np.nan
            t += L; on = not on
    vis = np.isfinite(px)
    px[vis] += rng.normal(0, sc.jitter_m, vis.sum()); py[vis] += rng.normal(0, sc.jitter_m, vis.sum())
    ps = od.player_speed(np.where(np.isfinite(trk["px"]), trk["px"], np.nan),
                         np.where(np.isfinite(trk["py"]), trk["py"], np.nan))
    ps[~vis] = np.nan

    out = dict(px=px.astype(np.float32), py=py.astype(np.float32), ps=ps.astype(np.float32),
               bx=bx, by=by, teams=trk["teams"])
    return out, simulate_tagging(lab, blocks, rst, sc, rng)


def simulate_tagging(lab, blocks, rst, sc, rng):
    fps = od.FPS
    tags = []
    for f, kind, team, _x, _y in rst:
        if rng.random() < sc.sp_recall.get(kind, 0.0):
            s = f - int(rng.uniform(*sc.sp_offset_s) * fps)
            tags.append((s, s + 35 * fps, "SET PIECES",
                         None if kind == "Throw-in" else team, kind))
    for a, b in blocks:
        w = lab[a:b + 1]
        starts = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(len(starts) - 1):
            s, e, k = a + starts[i], a + starts[i + 1], w[starts[i]]
            if k not in ("H", "A") or e - s < 2 * fps:
                continue
            owner = 0 if k == "H" else 1
            L = e - s
            for code, prob, team_true in (("PROGRESSION", sc.tag_own_prob, owner),
                                          ("DEFENDING IN MIDDLE THIRD", sc.tag_other_prob,
                                           1 - owner)):
                if rng.random() < prob:
                    ts = s + int(rng.uniform(0, 0.3) * L + rng.normal(0, fps))
                    te = e - int(rng.uniform(0, 0.3) * L + rng.normal(0, fps))
                    team = team_true if rng.random() < sc.tag_accuracy else 1 - team_true
                    if te > ts:
                        tags.append((ts, te, code, team, None))
    return tags
