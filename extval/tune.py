"""Coordinate-descent tuning of the decoder/evidence parameters on the open-data bench.

    python -m extval.tune                        # tune on all games, print the best set
    python -m extval.tune --holdout 3            # tune on games 1,2 then score game 3 untouched

Objective: mean official spellF1 over every (game, scenario) case; macroF1 breaks ties.
Only parameters that do not change the cached features are searched here.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from solution import config
from . import bench

GRID = {
    "SWITCH_COST": [4, 6, 8, 12, 16, 20, 28, 40],
    "TO_DEAD_COST": [2, 4, 8, 12, 16, 24],
    "FROM_DEAD_COST": [2, 4, 8, 12, 16, 24],
    "DEAD_BIAS": [-2.5, -2, -1.5, -1, -0.5, 0],
    "GAP_W": [0, 0.5, 1, 1.5, 2, 3],
    "GAP_LONG_W": [0, 0.25, 0.5, 1, 1.5],
    "GAP_MIN_S": [0.5, 1, 2, 3, 5],
    "GAP_LONG_S": [5, 10, 20],
    "SPEED_W": [0, 0.25, 0.5, 1, 1.5, 2],
    "SPEED_REF": [0.9, 1.1, 1.3, 1.5, 1.8],
    "OOB_W": [0, 1, 2, 3, 5],
    "SP_W": [0, 1, 2, 3, 5],
    "FAST_BALL_W": [0, 0.5, 1, 1.5, 2.5, 4],
    "FAST_BALL_SPEED": [2, 3, 4, 6, 8],
    "STRENGTH_CONTROL": [0.5, 1, 1.5, 2, 3, 4],
    "STRENGTH_NEAR": [0, 0.2, 0.4, 0.8, 1.2],
    "STRENGTH_FAR": [0, 0.05, 0.1, 0.2, 0.4],
    "CONTROL_DIST_M": [1, 1.5, 2, 3],
    "NEAR_DIST_M": [2, 3, 5, 8],
    "DIFF_SCALE_M": [0.5, 1, 1.5, 3, 5],
    "FLIGHT_SPEED": [5, 8, 12, 20],
    "FLIGHT_FACTOR": [0, 0.2, 0.5, 1],
    "TAG_W": [0, 0.05, 0.1, 0.2, 0.4],
    "TERR_W": [0, 0.1, 0.2, 0.4, 0.8, 1.2],
    "TERRV_W": [0, 0.05, 0.1, 0.2, 0.4],
}

_CASES = None


def _init(games, seeds, scenarios):
    global _CASES
    _CASES = bench.build_cases(games=games, seeds=seeds,
                               scenarios={k: bench.SCENARIOS[k] for k in scenarios})
    bench.features_for(_CASES, config.params())


def _eval(overrides):
    rows = bench.evaluate(_CASES, config.params(**overrides))
    return float(np.mean([r[2] for r in rows])), float(np.mean([r[3] for r in rows]))


def descend(pool, start, sweeps=3, log=print):
    best = dict(start)
    best_score = pool.submit(_eval, best).result()
    log("start  spellF1 {:.4f}  macroF1 {:.4f}".format(*best_score))
    for sweep in range(sweeps):
        improved = False
        for name, values in GRID.items():
            trials = [dict(best, **{name: v}) for v in values]
            scores = list(pool.map(_eval, trials))
            k = max(range(len(values)), key=lambda i: scores[i])
            if scores[k] > best_score:
                best, best_score, improved = trials[k], scores[k], True
                log("  sweep {} {:16s} -> {:<6} spellF1 {:.4f} macroF1 {:.4f}".format(
                    sweep, name, values[k], *best_score))
        if not improved:
            break
    return best, best_score


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="1,2,3", help="training games: 1,2,3 | idsse | all | ids")
    ap.add_argument("--holdout", default="", help="games scored untouched, same syntax")
    ap.add_argument("--scenarios", default=",".join(bench.SCENARIOS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    seeds = tuple(range(args.seeds))
    scen = tuple(args.scenarios.split(","))
    hold = bench.parse_games(args.holdout) if args.holdout else ()
    # backwards compatible: "--holdout 3" alone means "tune on the other Metrica games"
    train = tuple(g for g in bench.parse_games(args.games) if g not in hold)
    start = {k: getattr(config, k) for k in GRID}
    print("train", train, "| holdout", hold, "| scenarios", scen, "| workers", args.workers)

    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=(train, seeds, scen)) as pool:
        best, score = descend(pool, start)
    print("BEST on games {}: spellF1 {:.4f} macroF1 {:.4f}".format(train, *score))
    print(json.dumps(best))
    if hold:
        hc = bench.build_cases(games=hold, seeds=seeds,
                               scenarios={k: bench.SCENARIOS[k] for k in scen})
        bench.features_for(hc, config.params())
        base = [(c["name"], c["game"]) + bench.score(c["gt"], bench.baseline_labels(c["trk"]))
                for c in hc]
        cur = bench.evaluate(hc, config.params())
        tuned = bench.evaluate(hc, config.params(**best))
        bench.per_game(base, "HOLDOUT - baseline")
        bench.per_game(cur, "HOLDOUT - current config.py")
        bench.per_game(tuned, "HOLDOUT - tuned")
    else:
        base, cur, tuned = [], [], []
    from . import records
    best_hash, _ = records.params_hash(config.params(**best))
    cur_hash, _ = records.params_hash(config.params())
    records.save("tune", tuple(train) + tuple(hold), scen, config.params(**best),
                 records.rows_from("holdout_baseline_smooth50", base)
                 + records.rows_from("holdout_current_config", cur)
                 + records.rows_from("holdout_tuned", tuned),
                 extra=dict(train_games=[str(g) for g in train], holdout_games=[str(g) for g in hold],
                            train_spell_f1=score[0], train_macro_f1=score[1],
                            best_overrides=best, best_params_hash=best_hash,
                            current_params_hash=cur_hash))


if __name__ == "__main__":
    main()
