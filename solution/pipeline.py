"""The whole method as one function, independent of which match it runs on.

    labels = run(trk, tagging, blocks, length, width, P)

trk     : dict with px, py, ps (frames x slots, metres, m/s), bx, by (frames, metres), teams (slots)
tagging : [(start_frame, end_frame, CODE, team 0/1/None, type)]
blocks  : [(first, last_inclusive)] graded frames of each half
P       : parameter set (config.params())
Returns a '<U1' array of H/A/D on graded frames, X elsewhere.
"""

import numpy as np

from . import decode, evidence


def features(trk, tagging, blocks, length, width, P):
    n = len(trk["bx"])
    ok, sx, sy, speed, oob = evidence.ball_features(trk["bx"], trk["by"], length, width, P)
    return dict(
        n=n, ok=ok, speed=speed, oob=oob,
        vcoll=evidence.collective_speed(trk["ps"], P),
        gaplen=evidence.gap_length(ok, blocks),
        dist=evidence.team_distances(trk["px"], trk["py"], trk["teams"], sx, sy),
        sp=evidence.set_piece_windows(tagging, ok, speed, n, P),
        tagprior=evidence.tag_prior(tagging, n, P),
        terr=evidence.territory(trk["px"], trk["teams"], blocks, length, P),
    )


def decode_from_features(f, blocks, P):
    dead_e = evidence.dead_evidence(f["ok"], f["speed"], f["oob"], f["vcoll"], f["gaplen"],
                                    f["sp"][0], P)
    owner_e = evidence.owner_evidence(f["ok"], f["speed"], f["dist"][0], f["dist"][1],
                                      f["tagprior"], P, *f["terr"])
    labels = np.full(f["n"], "X", dtype="<U1")
    trans = decode.transition_matrix(P)
    for first, last in blocks:
        emis = decode.emissions(dead_e[first:last + 1], owner_e[first:last + 1])
        labels[first:last + 1] = decode.LETTERS[decode.viterbi(emis, trans)]
    return labels


def run(trk, tagging, blocks, length, width, P):
    return decode_from_features(features(trk, tagging, blocks, length, width, P), blocks, P)
