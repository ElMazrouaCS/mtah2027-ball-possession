"""Per-frame evidence: dead-ball log-odds and home-vs-away log-odds.

Interfaces (the contract between modules):
    dead_evidence(...)  -> float array (frames,)  > 0 means "looks dead",  < 0 "looks alive"
    owner_evidence(...) -> float array (frames,)  > 0 means "home has it", < 0 "away has it"
Both are 0 where there is no information. The decoder adds them up over time.
Every function takes P, the parameter set (see config.py).
"""

import warnings

import numpy as np


# --------------------------------------------------------------------------- helpers
def runs_of(mask):
    """[(start, stop_exclusive)] of True runs."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def centred_nanmean(x, w):
    """Centred moving average of width w that ignores NaN (NaN where the window has no value)."""
    v = np.where(np.isfinite(x), x, 0.0)
    c = np.isfinite(x).astype(float)
    h = int(w) // 2
    cv = np.concatenate(([0.0], np.cumsum(v)))
    cc = np.concatenate(([0.0], np.cumsum(c)))
    idx = np.arange(len(x))
    lo = np.clip(idx - h, 0, len(x))
    hi = np.clip(idx + h + 1, 0, len(x))
    s, n = cv[hi] - cv[lo], cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


# --------------------------------------------------------------------------- features
def ball_features(bx, by, length, width, P):
    """Clean ball track: ok mask, smoothed position, speed (m/s), out-of-bounds-near mask."""
    present = np.isfinite(bx) & np.isfinite(by)
    far = present & ((bx < -P.BALL_FAR_OUT_M) | (bx > length + P.BALL_FAR_OUT_M) |
                     (by < -P.BALL_FAR_OUT_M) | (by > width + P.BALL_FAR_OUT_M))
    ok = present & ~far

    step = np.full(len(bx), np.nan)
    step[1:] = np.hypot(np.diff(bx), np.diff(by)) * P.FPS
    ok &= ~(np.nan_to_num(step, nan=0.0) > P.BALL_MAX_SPEED)

    sx = np.full(len(bx), np.nan)
    sy = np.full(len(bx), np.nan)
    for a, b in runs_of(ok):
        sx[a:b] = centred_nanmean(bx[a:b], P.BALL_SMOOTH_FRAMES)
        sy[a:b] = centred_nanmean(by[a:b], P.BALL_SMOOTH_FRAMES)
    speed = np.full(len(bx), np.nan)
    both = ok[1:] & ok[:-1]
    speed[1:][both] = np.hypot(np.diff(sx), np.diff(sy))[both] * P.FPS
    speed = centred_nanmean(speed, P.BALL_SMOOTH_FRAMES)
    speed[~ok] = np.nan

    m = P.OOB_MARGIN_M
    oob_raw = ok & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    oob = np.zeros(len(bx), dtype=bool)
    for a, b in runs_of(oob_raw):
        if b - a >= P.OOB_MIN_FRAMES:
            oob[a:b] = True
    return ok, sx, sy, speed, oob


def collective_speed(ps, P):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = np.nanmean(ps, axis=1)
    return centred_nanmean(v, P.COLLECTIVE_SMOOTH_FRAMES)


def gap_length(ok, blocks):
    """For each frame without ball: length (frames) of the gap containing it, inside its half."""
    out = np.zeros(len(ok), dtype=np.int32)
    for first, last in blocks:
        for a, b in runs_of(~ok[first:last + 1]):
            out[first + a:first + b] = b - a
    return out


def team_distances(px, py, teams, sx, sy):
    """Distance (m) from the ball to the nearest home and nearest away tracked player."""
    d = np.hypot(px - sx[:, None], py - sy[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dh = np.nanmin(np.where(teams[None, :] == 0, d, np.nan), axis=1)
        da = np.nanmin(np.where(teams[None, :] == 1, d, np.nan), axis=1)
    return dh, da


# --------------------------------------------------------------------------- Smart Tagging
def set_piece_windows(tagging, ok, speed, n, P):
    """Boolean mask: from SP_BEFORE_S before each tagged set piece to its detected restart kick."""
    mask = np.zeros(n, dtype=bool)
    n_detected = 0
    for start, _end, code, _team, _typ in tagging:
        if code != "SET PIECES":
            continue
        a = max(start - int(P.SP_BEFORE_S * P.FPS), 0)
        lo = max(start, 0)
        hi = min(start + int(P.SP_SEARCH_S * P.FPS), n)
        moving = np.flatnonzero(ok[lo:hi] & (np.nan_to_num(speed[lo:hi]) > P.SP_KICK_SPEED))
        if len(moving):
            restart = lo + int(moving[0])
            n_detected += 1
        else:
            restart = min(start + int(P.SP_FALLBACK_S * P.FPS), n)
        mask[a:restart] = True
    return mask, n_detected


def tag_prior(tagging, n, P):
    """+1 where a phase says home has the ball, -1 for away (clipped)."""
    p = np.zeros(n)
    for start, end, code, team, _typ in tagging:
        if team is None:
            continue
        if code in P.TAG_OWN:
            owner = team
        elif code in P.TAG_OTHER:
            owner = 1 - team
        else:
            continue
        p[max(start, 0):min(end, n)] += 1.0 if owner == 0 else -1.0
    return np.clip(p, -1, 1)


# --------------------------------------------------------------------------- evidence
def dead_evidence(ok, speed, oob, vcoll, gaplen, sp_mask, P):
    e = np.full(len(ok), float(P.DEAD_BIAS))
    missing = ~ok
    e += np.where(missing & (gaplen >= P.GAP_MIN_S * P.FPS), P.GAP_W, 0.0)
    e += np.where(missing & (gaplen >= P.GAP_LONG_S * P.FPS), P.GAP_LONG_W, 0.0)
    slow = np.clip((P.SPEED_REF - np.nan_to_num(vcoll, nan=P.SPEED_REF)) / P.SPEED_SCALE, -2, 2)
    e += P.SPEED_W * slow
    e += np.where(oob, P.OOB_W, 0.0)
    e += np.where(sp_mask, P.SP_W, 0.0)
    e -= np.where(ok & (np.nan_to_num(speed) > P.FAST_BALL_SPEED) & ~oob, P.FAST_BALL_W, 0.0)
    return e


def territory(px, teams, blocks, length, P):
    """(terr, terr_v): play location in [-1, 1] and its speed (m/s), seen from home's attack."""
    from .features import attack_direction
    n = px.shape[0]
    direction, _, _ = attack_direction(px, teams, blocks, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmean(px, axis=1)
    terr = direction * (centred_nanmean(c, P.FPS) - length / 2) / (length / 2)
    s2 = 2 * P.FPS
    terr_v = np.zeros(n)
    terr_v[s2:-s2] = (terr[2 * s2:] - terr[:-2 * s2]) / 4.0 * (length / 2)
    return np.nan_to_num(terr), np.nan_to_num(np.clip(terr_v, -5, 5))


def owner_evidence(ok, speed, dh, da, tagprior, P, terr=None, terr_v=None):
    valid = ok & np.isfinite(dh) & np.isfinite(da)
    dmin = np.fmin(dh, da)
    strength = np.where(dmin < P.CONTROL_DIST_M, P.STRENGTH_CONTROL,
                        np.where(dmin < P.NEAR_DIST_M, P.STRENGTH_NEAR, P.STRENGTH_FAR))
    strength = np.where(np.nan_to_num(speed) > P.FLIGHT_SPEED, strength * P.FLIGHT_FACTOR,
                        strength)
    diff = np.clip((np.nan_to_num(da) - np.nan_to_num(dh)) / P.DIFF_SCALE_M, -2, 2)
    e = np.where(valid, strength * diff, 0.0)
    e = e + P.TAG_W * tagprior
    if terr is not None:
        e = e + P.TERR_W * terr + P.TERRV_W * terr_v
    return e
