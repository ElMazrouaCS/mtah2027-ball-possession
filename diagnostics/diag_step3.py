"""Step 3 -- statistics used to calibrate the open-data simulator (extval/degrade.py).

    python diagnostics/diag_step3.py

Ball-to-nearest-player distances, agreement of Smart Tagging team tags with the team in clear
control, collective speed with/without ball. Aggregates only, no ground truth.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np

from solution import config, data, evidence


def pct(a, ps):
    a = a[np.isfinite(a)]
    return "  ".join("p{}={:.2f}".format(p, np.percentile(a, p)) for p in ps) + "  (n={})".format(len(a))


def main():
    P = config.params()
    trk = data.load_tracking()
    blocks = data.graded_blocks()
    n = len(trk["bx"])
    g = np.zeros(n, dtype=bool)
    for a, b in blocks:
        g[a:b + 1] = True

    ok, sx, sy, speed, oob = evidence.ball_features(trk["bx"], trk["by"], data.LENGTH, data.WIDTH, P)
    dh, da = evidence.team_distances(trk["px"], trk["py"], trk["teams"], sx, sy)
    dmin = np.fmin(dh, da)
    m = g & ok & np.isfinite(dmin)

    print("1. DISTANCE BALLE -> JOUEUR LE PLUS PROCHE (m), frames notees avec balle")
    print("   tous          : " + pct(dmin[m], (10, 25, 50, 75, 90)))
    slow = m & (np.nan_to_num(speed) < 3)
    print("   balle lente   : " + pct(dmin[slow], (10, 25, 50, 75, 90)))
    for t in (1.0, 1.5, 2.0, 3.0):
        print("   part < {:.1f} m : {:5.1f}%".format(t, 100 * np.mean(dmin[m] < t)))
    print("   ecart 2e equipe - 1re (m) : " + pct(np.abs(dh - da)[m], (10, 25, 50, 75, 90)))

    print("\n2. ACCORD SMART TAGGING <-> EQUIPE QUI CONTROLE")
    control = m & (dmin < 1.5) & (np.abs(dh - da) > 3.0)
    ctrl_team = np.where(dh < da, 0, 1)
    tagging = data.load_tagging(P.FPS)
    for label, codes, flip in (("phases 'en possession'", P.TAG_OWN, False),
                               ("phases 'sans ballon'", P.TAG_OTHER, True)):
        agree = total = 0
        for s, e, code, team, _ in tagging:
            if team is None or code not in codes:
                continue
            owner = (1 - team) if flip else team
            w = control[max(s, 0):min(e, n)]
            ct = ctrl_team[max(s, 0):min(e, n)][w]
            agree += int(np.sum(ct == owner)); total += len(ct)
        print("   {:24s}: accord {:5.1f}%  sur {} frames de controle".format(
            label, 100 * agree / max(total, 1), total))
    print("   (frames de controle hors de toute phase taguee : {:.1f}%)".format(
        100 * np.mean(tag_cover(tagging, n, P)[control] == 0)))

    print("\n3. VITESSE COLLECTIVE (m/s) SELON LA BALLE")
    v = evidence.collective_speed(trk["ps"], P)
    print("   balle visible : " + pct(v[g & ok], (10, 25, 50, 75, 90)))
    print("   balle absente : " + pct(v[g & ~ok], (10, 25, 50, 75, 90)))


def tag_cover(tagging, n, P):
    c = np.zeros(n, dtype=np.int8)
    for s, e, code, team, _ in tagging:
        if team is not None and (code in P.TAG_OWN or code in P.TAG_OTHER):
            c[max(s, 0):min(e, n)] = 1
    return c


if __name__ == "__main__":
    main()
