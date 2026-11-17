"""Train the two logistic models on degraded open games, then tune the HMM scaling/costs.

    python -m extval.train --holdout 3     # train+tune on games 1,2; report game 3 untouched
    python -m extval.train --write         # train+tune on all 3 games, write solution/weights.json

Labels come from the open games' event data (extval/opendata.py). Nothing touches the
competition data.
"""

import argparse
import json

import numpy as np

from solution import config, features, model
from . import bench
from . import opendata as od

SUB = 5          # use every 5th frame to fit the logistic models (frames are highly correlated)
HMM_GRID = {
    "alpha_dead": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
    "alpha_own": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],
    "dead_offset": [-1.5, -1, -0.5, 0, 0.5, 1, 1.5],
    "switch_cost": [2, 4, 6, 8, 12, 16, 24],
    "to_dead_cost": [2, 4, 8, 12, 16, 24],
    "from_dead_cost": [2, 4, 8, 12, 16, 24],
}


def featurise(cases, P):
    for c in cases:
        c["Xd"], c["Xo"], c["nd"], c["no"] = features.build(c["trk"], c["tags"], c["blocks"],
                                                            od.LENGTH, od.WIDTH, P)


def training_rows(cases):
    Xd, yd, Xo, yo = [], [], [], []
    for c in cases:
        g = np.zeros(len(c["lab"]), bool)
        for a, b in c["blocks"]:
            g[a:b + 1] = True
        idx = np.flatnonzero(g)[::SUB]
        lab = c["lab"][idx]
        Xd.append(c["Xd"][idx]); yd.append(lab == "D")
        alive = np.isin(lab, ["H", "A"])
        # antisymmetric augmentation: every alive frame also appears with teams swapped
        Xo.append(c["Xo"][idx][alive]); yo.append(lab[alive] == "H")
        Xo.append(-c["Xo"][idx][alive]); yo.append(lab[alive] == "A")
    return np.vstack(Xd), np.concatenate(yd), np.vstack(Xo), np.concatenate(yo)


def score_all(cases, W):
    rows = []
    for c in cases:
        lab = model.decode_labels(c["Xd"], c["Xo"], c["blocks"], W)
        rows.append((c["name"], c["game"]) + bench.score(c["gt"], lab))
    return rows


def tune_hmm(cases, W, sweeps=3, log=print):
    best = dict(W)
    rows = score_all(cases, best)
    best_s = (np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]))
    log("hmm start spellF1 {:.4f} macroF1 {:.4f}".format(*best_s))
    for sw in range(sweeps):
        improved = False
        for k, vals in HMM_GRID.items():
            for v in vals:
                trial = dict(best, **{k: v})
                rows = score_all(cases, trial)
                s = (np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows]))
                if s > best_s:
                    best, best_s, improved = trial, s, True
                    log("  sweep {} {:14s} -> {:<5} spellF1 {:.4f} macroF1 {:.4f}".format(sw, k, v, *s))
        if not improved:
            break
    return best, best_s


def train(cases, l2_dead=10.0, l2_own=10.0):
    Xd, yd, Xo, yo = training_rows(cases)
    wd = model.fit_logistic(Xd, yd, l2=l2_dead)
    wo = model.fit_logistic(Xo, yo, l2=l2_own)
    return dict(dead=wd.tolist(), own=wo.tolist(), dead_offset=0.0, alpha_dead=0.2,
                alpha_own=0.3, switch_cost=8.0, to_dead_cost=12.0, from_dead_cost=12.0,
                names_dead=cases[0]["nd"], names_own=cases[0]["no"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    P = config.params()
    train_games = tuple(g for g in bench.GAMES if g != args.holdout)
    cases = bench.build_cases(games=train_games)
    featurise(cases, P)
    W = train(cases)
    W, s = tune_hmm(cases, W)
    print("TRAIN games {}: spellF1 {:.4f} macroF1 {:.4f}".format(train_games, *s))
    print("dead weights:", {n: round(w, 2) for n, w in zip(W["names_dead"], W["dead"]) if abs(w) > 0.3})
    print("own weights:", {n: round(w, 3) for n, w in zip(W["names_own"], W["own"])})
    print("hmm:", {k: W[k] for k in HMM_GRID})
    if args.holdout:
        hc = bench.build_cases(games=(args.holdout,))
        featurise(hc, P)
        bench.summarise([(c["name"], c["game"]) + bench.score(c["gt"], bench.baseline_labels(c["trk"]))
                         for c in hc], "HOLDOUT game {} - BASELINE".format(args.holdout))
        bench.features_for(hc, P)
        bench.summarise(bench.evaluate(hc, P), "HOLDOUT game {} - hand-made evidence".format(args.holdout))
        bench.summarise(score_all(hc, W), "HOLDOUT game {} - LEARNED models + HMM".format(args.holdout))
    if args.write:
        with open(model.WEIGHTS, "w") as fh:
            json.dump(W, fh, indent=1)
        print("written", model.WEIGHTS)


if __name__ == "__main__":
    main()
