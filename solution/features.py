"""Per-frame feature matrices for the two learned models.

    X_dead, X_own, names_dead, names_own = build(trk, tagging, blocks, length, width, P)

X_dead : features that say "stoppage or live play"       -> model P(D | x)
X_own  : features that say "home or away in control"    -> model P(H | x, alive)
         Built ANTISYMMETRIC: swapping the teams flips the sign of every column, so the model has
         no notion of "home" and cannot learn a home bias from the training games.

Everything is computed from the tracking and the Smart Tagging only. Non-linear effects are
captured with one-hot bins, so a logistic model learns a piecewise-constant curve per feature.
"""

import warnings

import numpy as np

from . import evidence
from .evidence import centred_nanmean, runs_of


def _bins(x, edges, name, valid=None):
    """One-hot encode x into len(edges)+1 bins (NaN or ~valid -> all zeros)."""
    x = np.asarray(x, dtype=float)
    k = np.digitize(np.nan_to_num(x, nan=-1e9), edges)
    cols, names = [], []
    for i in range(len(edges) + 1):
        c = (k == i) & np.isfinite(x)
        if valid is not None:
            c &= valid
        cols.append(c.astype(np.float32))
        lo = "-inf" if i == 0 else edges[i - 1]
        hi = "inf" if i == len(edges) else edges[i]
        names.append("{}[{},{})".format(name, lo, hi))
    return cols, names


def gap_position(ok, blocks, fps):
    """For frames without ball: seconds since the ball was lost and until it re-appears."""
    n = len(ok)
    since = np.full(n, np.nan)
    until = np.full(n, np.nan)
    for first, last in blocks:
        for a, b in runs_of(~ok[first:last + 1]):
            idx = np.arange(b - a)
            since[first + a:first + b] = idx / fps
            until[first + a:first + b] = (b - a - idx) / fps
    return since, until


def attack_direction(px, teams, blocks, n):
    """+1 where home attacks towards x = length (i.e. defends x = 0), -1 otherwise, per half.

    The defending team sits closer to its own goal, so over a whole half the team with the lower
    median centroid defends the x = 0 goal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ch = np.nanmean(np.where(teams[None, :] == 0, px, np.nan), axis=1)
        ca = np.nanmean(np.where(teams[None, :] == 1, px, np.nan), axis=1)
    d = np.zeros(n)
    for first, last in blocks:
        diff = np.nanmedian(ca[first:last + 1] - ch[first:last + 1])
        d[first:last + 1] = 1.0 if diff > 0 else -1.0
    return d, ch, ca


def build(trk, tagging, blocks, length, width, P):
    fps = P.FPS
    n = len(trk["bx"])
    px, py, teams = trk["px"], trk["py"], trk["teams"]

    ok, sx, sy, speed, oob = evidence.ball_features(trk["bx"], trk["by"], length, width, P)
    vcoll = evidence.collective_speed(trk["ps"], P)
    vcoll3 = centred_nanmean(vcoll, 3 * fps)
    dv = np.full(n, np.nan)
    s2 = 2 * fps
    dv[s2:-s2] = vcoll[2 * s2:] - vcoll[:-2 * s2]
    since, until = gap_position(ok, blocks, fps)
    gaplen = evidence.gap_length(ok, blocks) / fps
    sp_mask, _ = evidence.set_piece_windows(tagging, ok, speed, n, P)
    tagp = evidence.tag_prior(tagging, n, P)
    dh, da = evidence.team_distances(px, py, teams, sx, sy)

    # ball geometry
    dist_touch = np.fmin(np.abs(sy), np.abs(width - sy))
    dist_goal = np.fmin(np.abs(sx), np.abs(length - sx))
    in_box6 = ok & (dist_goal < 5.5) & (np.abs(sy - width / 2) < 9.16)
    near_center = ok & (np.hypot(sx - length / 2, sy - width / 2) < 3.0)
    slow_ball = centred_nanmean(np.where(ok, np.nan_to_num(speed), np.nan), fps)

    # territory: where is the play, seen from home's attacking direction
    direction, ch, ca = attack_direction(px, teams, blocks, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        call = np.nanmean(px, axis=1)
    terr = direction * (centred_nanmean(call, fps) - length / 2) / (length / 2)       # -1 .. 1
    terr_v = np.full(n, np.nan)
    terr_v[s2:-s2] = (terr[2 * s2:] - terr[:-2 * s2]) / 4.0 * (length / 2)            # m/s
    gapdiff = direction * centred_nanmean(ca - ch, fps)          # >0: home deeper than away
    # kick-off formation: every tracked player in his own half
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        own_half_home = np.nanmean(np.where(teams[None, :] == 0,
                                            (direction[:, None] * (px - length / 2)) < 0, np.nan), axis=1)
        own_half_away = np.nanmean(np.where(teams[None, :] == 1,
                                            (direction[:, None] * (px - length / 2)) > 0, np.nan), axis=1)
    kickoff = np.nan_to_num(np.fmin(own_half_home, own_half_away))

    # ---------------- dead-ball design matrix
    cols, names = [], []

    def add(c, nm):
        cols.extend(c if isinstance(c, list) else [c]); names.extend(nm if isinstance(nm, list) else [nm])

    add(np.ones(n, np.float32), "bias")
    add((~ok).astype(np.float32), "missing")
    add(*_bins(gaplen, [1, 3, 6, 10, 20, 40], "gaplen", ~ok))
    add(*_bins(since, [1, 3, 6, 10, 20], "since_lost", ~ok))
    add(*_bins(until, [1, 3, 6, 10, 20], "until_back", ~ok))
    add(*_bins(vcoll, [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8], "vcoll"))
    add(*_bins(vcoll3, [0.7, 1.0, 1.4, 1.9, 2.5], "vcoll3"))
    add(*_bins(dv, [-1.0, -0.4, -0.1, 0.1, 0.4, 1.0], "dvcoll"))
    add(*_bins(slow_ball, [0.5, 1.5, 3, 6, 12], "ballspeed1s", ok))
    add(*_bins(dist_touch, [0.5, 1.5, 3, 8], "dist_touch", ok))
    add(*_bins(dist_goal, [0.5, 1.5, 3, 8], "dist_goal", ok))
    add(oob.astype(np.float32), "oob")
    add((in_box6 & (np.nan_to_num(slow_ball) < 1.0)).astype(np.float32), "static_in_box6")
    add((near_center & (np.nan_to_num(slow_ball) < 1.0)).astype(np.float32), "static_center")
    add(*_bins(kickoff, [0.8, 0.95], "kickoff_formation"))
    add(sp_mask.astype(np.float32), "set_piece_window")
    add(*_bins(np.fmin(dh, da), [1, 2, 4, 8], "dmin", ok))
    X_dead = np.stack(cols, axis=1).astype(np.float32)
    names_dead = names

    # ---------------- ownership design matrix (antisymmetric: home <-> away flips every column)
    cols, names = [], []
    diff = np.clip(np.nan_to_num(da - dh), -10, 10)                # >0: home closer
    dmin = np.fmin(dh, da)
    valid = ok & np.isfinite(dmin)
    fast = np.nan_to_num(speed) > 8
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 99)):
        m = valid & (dmin >= lo) & (dmin < hi)
        add((np.where(m & ~fast, np.clip(diff, -3, 3), 0)).astype(np.float32),
            "diff_clip3*[dmin {}-{}]".format(lo, hi))
        add((np.where(m & ~fast, np.sign(diff), 0)).astype(np.float32),
            "sign_diff*[dmin {}-{}]".format(lo, hi))
    add((np.where(valid & fast, np.sign(diff), 0)).astype(np.float32), "sign_diff*[fast ball]")
    add(np.nan_to_num(terr).astype(np.float32), "territory")
    add(np.nan_to_num(np.clip(terr_v, -5, 5)).astype(np.float32), "territory_velocity")
    add(np.nan_to_num(np.clip(gapdiff, -30, 30) / 10).astype(np.float32), "block_depth_diff")
    add(np.where(~ok, np.nan_to_num(terr), 0).astype(np.float32), "territory*[missing]")
    add(np.where(~ok, np.nan_to_num(np.clip(terr_v, -5, 5)), 0).astype(np.float32),
        "territory_velocity*[missing]")
    add(tagp.astype(np.float32), "tag_prior")
    X_own = np.stack(cols, axis=1).astype(np.float32)
    return X_dead, X_own, names_dead, names
