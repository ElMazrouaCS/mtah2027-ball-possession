"""Regenerate the submission file from the provided data. Deterministic: no randomness anywhere.

    python make_submission.py            # writes N7_TACTIC.csv at the repository root

Writes <team>.csv and prints AGGREGATES only (counts, durations).
"""

import argparse
import os

import numpy as np

from solution import config, data, pipeline

ROOT = os.path.dirname(os.path.abspath(__file__))


def possession_stats(labels, blocks, fps):
    n_spells, alive = 0, 0
    for first, last in blocks:
        w = labels[first:last + 1]
        starts = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1))
        n_spells += int(np.sum(np.isin(w[starts], ["H", "A"])))
        alive += int(np.sum(np.isin(w, ["H", "A"])))
    return n_spells, alive / fps / max(n_spells, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="N7_TACTIC")
    args = ap.parse_args()

    P = config.params()
    trk = data.load_tracking()
    tagging = data.load_tagging(P.FPS)
    blocks = data.graded_blocks()

    f = pipeline.features(trk, tagging, blocks, data.LENGTH, data.WIDTH, P)
    labels = pipeline.decode_from_features(f, blocks, P)

    out = os.path.join(ROOT, "{}.csv".format(args.team))
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))

    # ---- aggregates only
    graded = np.zeros(len(labels), dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    g = labels[graded]
    n_spells, mean_dur = possession_stats(labels, blocks, P.FPS)
    print("written: {}".format(os.path.basename(out)))
    print("graded frames: H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
        *(100 * np.mean(g == x) for x in "HAD")))
    print("possessions: {}   mean duration {:.1f} s".format(n_spells, mean_dur))
    print("set pieces with a detected restart kick: {}".format(f["sp"][1]))
    print("frames in a set-piece dead window: {:.1f}% of graded".format(
        100 * np.mean(f["sp"][0][graded])))
    print("frames with persistent out-of-bounds ball: {:.2f}% of graded".format(
        100 * np.mean(f["oob"][graded])))


if __name__ == "__main__":
    main()
