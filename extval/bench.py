"""Validation bench: open games x degradation scenarios, scored with the OFFICIAL metric code.

    python -m extval.bench                 # default parameters vs baseline, every scenario

Nothing here reads the competition data.
"""

import os
import sys
import warnings

import numpy as np

from solution import config, evidence, pipeline
from . import degrade as dg
from . import opendata as od

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "evaluation"))
import possession_metrics as pm  # noqa: E402

GAMES = (1, 2, 3)
# Calibrated on label-free statistics of the provided ATD (diag_step1/2/3): ball-to-nearest-player
# distances -> ball_offset_m 0.8; Smart Tagging team agreement with clear control ~70-81%.
_CAL = dict(ball_offset_m=0.8, tag_accuracy=0.75)
SCENARIOS = {
    "central":      dict(_CAL, dead_visible_prob=0.3),
    "ball~dead":    dict(_CAL, alive_missing=0.06, dead_visible_prob=0.15, dead_head_prob=0.3),
    "ball!~dead":   dict(_CAL, alive_missing=0.22, dead_visible_prob=0.6, dead_head_prob=0.7),
    "noisy":        dict(_CAL, dead_visible_prob=0.3, tag_accuracy=0.65, tag_own_prob=0.5,
                         ball_offset_m=1.1,
                         sp_recall={"Throw-in": 0.5, "Corner Kick": 0.8, "Goal Kick": 0.5,
                                    "Free Kick": 0.1, "Kick Off": 1.0, "other": 0.0}),
}

_RAW = {}


def raw_game(g):
    if g not in _RAW:
        e = od.events(g)
        trk = od.tracking(g)
        lab, blocks = od.labels(e, trk["n"])
        _RAW[g] = (trk, lab, blocks, od.restarts(e, lab, blocks))
    return _RAW[g]


def build_cases(scenarios=SCENARIOS, games=GAMES, seeds=(0,)):
    cases = []
    for name, kw in scenarios.items():
        for g in games:
            trk, lab, blocks, rst = raw_game(g)
            for s in seeds:
                dtrk, tags = dg.degrade(trk, lab, blocks, rst, dg.Scenario(seed=100 * g + s, **kw))
                cases.append(dict(name=name, game=g, seed=s, trk=dtrk, tags=tags, lab=lab,
                                  blocks=blocks, gt=pm.label_codes(lab)))
    return cases


def score(gt_codes, labels):
    runs = pm.valid_runs(gt_codes)
    pred = pm.label_codes(labels)
    sp = pm.spell_scores(gt_codes, pm.fill_abstentions(pred, runs), runs)
    cm = pm.scored_confusion(pm.confusion(gt_codes, pred))
    return sp["mean_f1"], pm.macro_f1(cm), sp["n_pred"], sp["n_gt"]


def features_for(cases, P):
    for c in cases:
        c["feat"] = pipeline.features(c["trk"], c["tags"], c["blocks"], od.LENGTH, od.WIDTH, P)


def evaluate(cases, P):
    rows = []
    for c in cases:
        labels = pipeline.decode_from_features(c["feat"], c["blocks"], P)
        rows.append((c["name"], c["game"]) + score(c["gt"], labels))
    return rows


def baseline_labels(trk, window=50):
    """Same algorithm as starter/baseline.py: nearest tracked player, centred majority vote."""
    bx, by = trk["bx"], trk["by"]
    d = np.hypot(trk["px"] - bx[:, None], trk["py"] - by[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        allnan = np.all(~np.isfinite(d), axis=1)
        j = np.nanargmin(np.where(allnan[:, None], 0, d), axis=1)
    codes = np.where(np.isfinite(bx) & ~allnan, trk["teams"][j], -1)
    hc = np.concatenate(([0], np.cumsum(codes == 0))); ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes)); lo = np.maximum(0, i - window); hi = np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    lab = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    lab[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return lab


def summarise(rows, title):
    names = sorted({r[0] for r in rows}, key=list(SCENARIOS).index)
    print(title)
    for nm in names:
        rr = [r for r in rows if r[0] == nm]
        print("   {:12s} spellF1 {:.3f}  macroF1 {:.3f}   poss pred/gt {:.0f}/{:.0f}".format(
            nm, np.mean([r[2] for r in rr]), np.mean([r[3] for r in rr]),
            np.mean([r[4] for r in rr]), np.mean([r[5] for r in rr])))
    print("   {:12s} spellF1 {:.3f}  macroF1 {:.3f}".format(
        "ALL", np.mean([r[2] for r in rows]), np.mean([r[3] for r in rows])))


TUNED_H3 = {"SWITCH_COST": 8, "TO_DEAD_COST": 24, "FROM_DEAD_COST": 24, "DEAD_BIAS": -0.5,
            "GAP_LONG_W": 0.25, "GAP_MIN_S": 1, "OOB_W": 5, "SP_W": 2, "FAST_BALL_W": 1.5,
            "FAST_BALL_SPEED": 2, "STRENGTH_CONTROL": 1, "STRENGTH_NEAR": 0, "STRENGTH_FAR": 0.05,
            "CONTROL_DIST_M": 1, "DIFF_SCALE_M": 0.5, "FLIGHT_SPEED": 12, "FLIGHT_FACTOR": 0,
            "TAG_W": 0.2}

if __name__ == "__main__":
    P = config.params()
    cases = build_cases()
    base = [(c["name"], c["game"]) + score(c["gt"], baseline_labels(c["trk"])) for c in cases]
    summarise(base, "BASELINE smooth50 (for reference: 0.2433 / 0.6724 on the real full match)")
    features_for(cases, P)
    summarise(evaluate(cases, P), "OURS (config.py defaults)")
    summarise(evaluate(cases, config.params(**TUNED_H3)), "OURS (tuned on the OLD simulator)")
