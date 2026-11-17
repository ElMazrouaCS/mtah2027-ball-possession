"""Logistic models (numpy only, deterministic) + their use as HMM emissions.

Weights are trained ONLY on the open Metrica sample games (other matches, allowed by the rules)
by extval/train.py and stored in solution/weights.json. make_submission.py never trains: it reads
the stored weights, so the submission is reproducible byte for byte.
"""

import json
import os

import numpy as np

from . import decode, features

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.json")


def fit_logistic(X, y, l2=1.0, iters=50, w=None):
    """L2-regularised logistic regression by Newton's method (IRLS). y in {0,1}."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    d = X.shape[1]
    w = np.zeros(d) if w is None else w
    reg = l2 * np.eye(d)
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (p - y) + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


def load_weights(path=WEIGHTS):
    with open(path) as fh:
        return json.load(fh)


def emissions(X_dead, X_own, W):
    """Per-frame scores (frames, 3) for H, A, D from the two models' log-odds."""
    ld = X_dead @ np.array(W["dead"]) - W["dead_offset"]       # log-odds of D vs alive
    lo = X_own @ np.array(W["own"])                             # log-odds of H vs A
    e = np.zeros((len(ld), 3))
    e[:, decode.H] = 0.5 * W["alpha_own"] * lo
    e[:, decode.A] = -0.5 * W["alpha_own"] * lo
    e[:, decode.D] = W["alpha_dead"] * ld
    return e


def transition_matrix(W):
    t = np.zeros((3, 3))
    H, A, D = decode.H, decode.A, decode.D
    t[H, A] = t[A, H] = -W["switch_cost"]
    t[H, D] = t[A, D] = -W["to_dead_cost"]
    t[D, H] = t[D, A] = -W["from_dead_cost"]
    return t


def decode_labels(X_dead, X_own, blocks, W):
    n = len(X_dead)
    labels = np.full(n, "X", dtype="<U1")
    trans = transition_matrix(W)
    for first, last in blocks:
        e = emissions(X_dead[first:last + 1], X_own[first:last + 1], W)
        labels[first:last + 1] = decode.LETTERS[decode.viterbi(e, trans)]
    return labels


def run(trk, tagging, blocks, length, width, P, W):
    Xd, Xo, _, _ = features.build(trk, tagging, blocks, length, width, P)
    return decode_labels(Xd, Xo, blocks, W)
