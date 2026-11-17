"""Step 2 -- do the dead-ball cues agree with the Smart Tagging restarts? Aggregates only.

    python diagnostics/diag_step2.py

Around each tagged set piece (start and end) vs 500 random instants (control group, fixed seed):
ball missing, end of a ball gap nearby, ball out of bounds before, collective speed profile.
Then the reverse direction: are ball gaps followed by a tagged restart? No ground truth used.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np

from diag_step1 import BLOCKS, FPS, N, graded_mask, load_raw
from games import asset_paths

MIN_GAP_S = 3          # on ne s'interesse qu'aux trous de balle >= 3 s
OOB_MARGIN = 0.01      # balle hors terrain si x ou y sort de [-0.01, 1.01]
OFFSETS_S = (-20, -10, -5, -2, 0, 2, 5)
N_CONTROL = 500
SEED = 0


def set_pieces():
    """[(frame_debut, frame_fin, type)] des instances SET PIECES, en frames video."""
    root = ET.parse(asset_paths()["smart_tagging"]).getroot()
    out = []
    for inst in root.iter("instance"):
        code = (inst.findtext("code") or "").strip().upper()
        if code != "SET PIECES":
            continue
        typ = "?"
        for lab in inst.iter("label"):
            if (lab.findtext("group") or "").strip().lower() == "type":
                typ = (lab.findtext("text") or "?").strip()
        start = int(round(float(inst.findtext("start")) * FPS))
        end = int(round(float(inst.findtext("end")) * FPS))
        out.append((start, end, typ))
    return out


def gaps(missing):
    """[(debut, fin_exclue)] des trous de balle >= MIN_GAP_S, par mi-temps."""
    out = []
    for a, b in BLOCKS:
        w = missing[a:b + 1].astype(np.int8)
        e = np.diff(np.concatenate(([0], w, [0])))
        for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if t - s >= MIN_GAP_S * FPS:
                out.append((a + s, a + t))
    return out


def in_graded(f):
    return any(a <= f <= b for a, b in BLOCKS)


def describe(times, missing, oob, pmean, gap_list, label):
    """Mesures autour d'une liste d'instants t0 (frames)."""
    times = [t for t in times if in_graded(t)]
    n = len(times)
    if not n:
        print("   {} : aucun instant dans les mi-temps notees".format(label))
        return
    gap_ends = np.array([t for _, t in gap_list])
    inside_gap = sum(1 for t in times if any(s <= t < e for s, e in gap_list))
    near_end = []
    for t in times:
        d = (gap_ends - t) / FPS
        d = d[np.abs(d) <= 15]
        if len(d):
            near_end.append(d[np.argmin(np.abs(d))])
    oob_before = sum(1 for t in times if oob[max(t - 15 * FPS, 0):t + 1].any())
    miss_at = sum(1 for t in times if missing[t])

    print("   {} (n={})".format(label, n))
    print("     balle absente a t0                       : {:6.1f}%".format(100 * miss_at / n))
    print("     t0 a l'interieur d'un trou >= {} s        : {:6.1f}%".format(
        MIN_GAP_S, 100 * inside_gap / n))
    print("     une fin de trou a +/-15 s de t0          : {:6.1f}%".format(
        100 * len(near_end) / n))
    if near_end:
        ne = np.array(near_end)
        print("       ecart (fin de trou - t0), en s : p10={:.1f}  p50={:.1f}  p90={:.1f}".format(
            *np.percentile(ne, (10, 50, 90))))
    print("     balle hors terrain dans les 15 s avant   : {:6.1f}%".format(
        100 * oob_before / n))
    prof = []
    for o in OFFSETS_S:
        vals = [pmean[t + o * FPS] for t in times
                if 0 <= t + o * FPS < N and np.isfinite(pmean[t + o * FPS])]
        prof.append("{:+d}s:{:.2f}".format(o, np.median(vals)) if vals else "{:+d}s:-".format(o))
    print("     vitesse collective mediane (m/s)         : " + "  ".join(prof))


def main():
    bx, by, _bs, pmean, _pc, _ps = load_raw()
    g = graded_mask()
    present = np.isfinite(bx) & np.isfinite(by)
    missing = g & ~present
    oob = present & ((bx < -OOB_MARGIN) | (bx > 1 + OOB_MARGIN) |
                     (by < -OOB_MARGIN) | (by > 1 + OOB_MARGIN))
    gap_list = gaps(missing)
    sp = set_pieces()

    print("coups de pied arretes : {}   dans les mi-temps notees : {}".format(
        len(sp), sum(in_graded(s) for s, _, _ in sp)))
    print("types : {}".format(dict(Counter(t for _, _, t in sp))))
    dur = np.array([(e - s) / FPS for s, e, _ in sp])
    print("duree des instances (s) : p10={:.1f}  p50={:.1f}  p90={:.1f}".format(
        *np.percentile(dur, (10, 50, 90))))
    print("trous de balle >= {} s : {}".format(MIN_GAP_S, len(gap_list)))

    print("\nA. AUTOUR DU DEBUT des coups de pied arretes (t0 = start)")
    describe([s for s, _, _ in sp], missing, oob, pmean, gap_list, "tous")
    for typ in sorted({t for _, _, t in sp}):
        describe([s for s, _, t in sp if t == typ], missing, oob, pmean, gap_list, typ)

    print("\nB. AUTOUR DE LA FIN des coups de pied arretes (t0 = end)")
    describe([e for _, e, _ in sp], missing, oob, pmean, gap_list, "tous")

    print("\nC. TEMOIN : instants au hasard dans les mi-temps notees")
    rng = np.random.default_rng(SEED)
    graded_idx = np.flatnonzero(g)
    describe(rng.choice(graded_idx, N_CONTROL, replace=False).tolist(),
             missing, oob, pmean, gap_list, "hasard")

    print("\nD. SENS INVERSE : les trous de balle sont-ils suivis d'une reprise taguee ?")
    starts = np.array([s for s, _, _ in sp])
    for win in (5, 10, 20):
        hit = sum(1 for _, e in gap_list if (np.abs(starts - e) <= win * FPS).any())
        print("   trous dont la fin a un debut de coup de pied arrete a +/-{:2d} s : {:5.1f}%"
              .format(win, 100 * hit / max(len(gap_list), 1)))
    med_all = np.nanmedian(pmean[g])
    low = sum(1 for s, e in gap_list if np.nanmedian(pmean[s:e]) < med_all)
    print("   trous ou la vitesse collective mediane < mediane du match : {:5.1f}%".format(
        100 * low / max(len(gap_list), 1)))
    print("   (mediane du match : {:.2f} m/s)".format(med_all))


if __name__ == "__main__":
    main()
