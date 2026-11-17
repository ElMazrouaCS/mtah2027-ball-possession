"""Step 1 -- statistics of the dead-ball cues. Prints aggregates only, no ground truth.

    python diagnostics/diag_step1.py

Speed units (ball speed recomputed from positions vs the file field), ball-gap lengths,
out-of-bounds ball positions, impossible jumps, collective player speed. Graded frames only.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N = GAMES[GAME]["n_video_frames"]
BLOCKS = [(a, b) for _, a, b in half_video_blocks(GAME)]  # bornes inclusives


def to_float(s):
    return np.nan if s[0] == "N" else float(s)


def load_raw():
    """Tableaux par frame video : balle x,y,s ; vitesse moyenne et nb de joueurs presents."""
    bx = np.full(N, np.nan); by = np.full(N, np.nan); bs = np.full(N, np.nan)
    pmean = np.full(N, np.nan); pcount = np.zeros(N, dtype=np.int16)
    psample = []  # vitesses individuelles, 1 frame sur 5 (pour les percentiles)
    with open(asset_paths()["atd_raw"]) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            vf = int(parts[0]) - 1
            speeds = []
            for g in parts[1].split(";"):
                f = g.split(",")
                if f[0][0] != "N":
                    pcount[vf] += 1
                    if f[2][0] != "N":
                        speeds.append(float(f[2]))
            if speeds:
                pmean[vf] = np.mean(speeds)
                if vf % 5 == 0:
                    psample.extend(speeds)
            b = parts[2].split(",")
            bx[vf], by[vf], bs[vf] = to_float(b[0]), to_float(b[1]), to_float(b[2])
    return bx, by, bs, pmean, pcount, np.array(psample)


def graded_mask():
    m = np.zeros(N, dtype=bool)
    for a, b in BLOCKS:
        m[a:b + 1] = True
    return m


def q(arr, ps=(10, 50, 90, 99)):
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return "aucune valeur"
    return "  ".join("p{}={:.2f}".format(p, np.percentile(arr, p)) for p in ps) + \
        "   (n={})".format(len(arr))


def runs(mask):
    """Longueurs des suites de True dans mask, calculees separement dans chaque mi-temps."""
    out = []
    for a, b in BLOCKS:
        w = mask[a:b + 1].astype(np.int8)
        e = np.diff(np.concatenate(([0], w, [0])))
        out.extend((np.flatnonzero(e == -1) - np.flatnonzero(e == 1)).tolist())
    return np.array(out, dtype=int)


def main():
    bx, by, bs, pmean, pcount, psample = load_raw()
    g = graded_mask()
    ng = int(g.sum())
    present = g & np.isfinite(bx) & np.isfinite(by)
    print("frames notees : {}   balle presente : {:.2f}%".format(ng, 100 * present.sum() / ng))

    # ---- 1. Unite des vitesses
    print("\n1. UNITE DES VITESSES")
    # vitesse recalculee : deux frames consecutives avec balle, dans la meme mi-temps
    both = present[1:] & present[:-1]
    dx = np.diff(bx) * LENGTH; dy = np.diff(by) * WIDTH
    v_calc = np.full(N, np.nan)
    v_calc[1:][both] = np.hypot(dx, dy)[both] * FPS  # m/s
    print("   balle, vitesse recalculee (m/s)   : " + q(v_calc[g]))
    print("   balle, champ vitesse du fichier   : " + q(bs[g]))
    both_known = g & np.isfinite(v_calc) & np.isfinite(bs)
    if both_known.sum() > 100:
        ratio = np.median(bs[both_known]) / max(np.median(v_calc[both_known]), 1e-9)
        print("   ratio median champ / recalcule    : {:.2f}   (~1 => m/s, ~3.6 => km/h)"
              .format(ratio))
    print("   joueurs, vitesse individuelle     : " + q(psample))
    print("   (repere : un joueur qui marche ~1.5 m/s = 5 km/h, sprint ~9 m/s = 32 km/h)")

    # ---- 2. Trous de balle
    print("\n2. TROUS DE BALLE (suites de frames sans balle, par mi-temps)")
    gaps = runs(g & ~present)
    bins = [(0, 0.2), (0.2, 1), (1, 3), (3, 10), (10, 30), (30, 1e9)]
    total_missing = gaps.sum()
    print("   nb de trous : {}   frames manquantes : {}".format(len(gaps), total_missing))
    print("   duree (s)       nb_trous   % des frames manquantes")
    for lo, hi in bins:
        sel = (gaps >= lo * FPS) & (gaps < hi * FPS)
        label = "{}-{}".format(lo, hi) if hi < 1e9 else ">{}".format(lo)
        print("   {:12s}  {:9d}   {:6.2f}%".format(
            label, int(sel.sum()), 100 * gaps[sel].sum() / max(total_missing, 1)))

    # ---- 3. Balle hors terrain
    print("\n3. BALLE HORS DU TERRAIN (parmi les frames ou elle est presente)")
    for margin in (0.0, 0.01, 0.03):
        out = present & ((bx < -margin) | (bx > 1 + margin) | (by < -margin) | (by > 1 + margin))
        print("   au-dela de {:4.2f} (~{:.1f} m) : {:6.2f}%".format(
            margin, margin * LENGTH, 100 * out.sum() / max(present.sum(), 1)))
    out_x = present & ((bx < 0) | (bx > 1)); out_y = present & ((by < 0) | (by > 1))
    print("   dont hors ligne de but (x) : {:.2f}%   hors ligne de touche (y) : {:.2f}%".format(
        100 * out_x.sum() / max(present.sum(), 1), 100 * out_y.sum() / max(present.sum(), 1)))
    print("   bornes observees : x dans [{:.3f}, {:.3f}]  y dans [{:.3f}, {:.3f}]".format(
        np.nanmin(bx[present]), np.nanmax(bx[present]),
        np.nanmin(by[present]), np.nanmax(by[present])))

    # ---- 4. Sauts impossibles (erreurs du tracker)
    print("\n4. SAUTS DE BALLE (vitesse recalculee)")
    vc = v_calc[g & np.isfinite(v_calc)]
    for thr in (35, 50, 100):
        print("   > {:3d} m/s : {:6.2f}% des paires de frames".format(
            thr, 100 * (vc > thr).mean() if len(vc) else 0))

    # ---- 5. Vitesse collective
    print("\n5. VITESSE MOYENNE DES JOUEURS PAR FRAME (unite du fichier)")
    print("   " + q(pmean[g], ps=(5, 10, 25, 50, 75, 90)))
    print("   joueurs presents par frame : " + q(pcount[g].astype(float), ps=(5, 50, 95)))
    print("   frames notees avec < 10 joueurs visibles : {:.2f}%".format(
        100 * (pcount[g] < 10).mean()))


if __name__ == "__main__":
    main()
