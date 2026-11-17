"""Where do we lose? Characterise the ground-truth possessions we miss (IoU < 0.5)."""
import numpy as np
from solution import config
from . import bench, train, opendata as od
import possession_metrics as pm

def spells(lab, blocks):
    out = []
    for a, b in blocks:
        w = lab[a:b + 1]; st = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(len(st) - 1):
            if w[st[i]] in "HA": out.append((a + st[i], a + st[i + 1], w[st[i]]))
    return out

def best_iou(s, pred):
    best = 0
    for (a, b, k) in pred:
        if k != s[2] or b <= s[0] or a >= s[1]: continue
        inter = min(b, s[1]) - max(a, s[0]); best = max(best, inter / (max(b, s[1]) - min(a, s[0])))
    return best

if __name__ == "__main__":
    import json, sys
    W = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    P = config.params()
    hc = bench.build_cases(games=(3,), scenarios={"central": bench.SCENARIOS["central"]})
    train.featurise(hc, P)
    c = hc[0]
    lab = od.labels(od.events(3), len(c["lab"]))[0]
    from solution import model
    pred = model.decode_labels(c["Xd"], c["Xo"], c["blocks"], W)
    gs, ps = spells(lab, c["blocks"]), spells(pred, c["blocks"])
    ok = np.isfinite(c["trk"]["bx"])
    rows = []
    for s in gs:
        L = (s[1] - s[0]) / 25
        prev_dead = lab[s[0] - 1] == "D"; next_dead = lab[s[1]] == "D" if s[1] < len(lab) else True
        rows.append((L, best_iou(s, ps), 1 - ok[s[0]:s[1]].mean(), prev_dead, next_dead,
                     np.mean(pred[s[0]:s[1]] == s[2]), np.mean(pred[s[0]:s[1]] == "D")))
    r = np.array(rows, dtype=float)
    hit = r[:, 1] >= 0.5
    print("gt spells", len(r), "found(IoU>=.5)", hit.sum(), " pred spells", len(ps))
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 999)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        print("  dur {:>3}-{:<3}s  n={:3d}  found {:5.1f}%  ball-missing {:4.0f}%  frames right-team {:4.0f}%  frames-pred-D {:4.0f}%".format(
            lo, hi, m.sum(), 100 * hit[m].mean() if m.any() else 0, 100 * r[m, 2].mean() if m.any() else 0,
            100 * r[m, 5].mean() if m.any() else 0, 100 * r[m, 6].mean() if m.any() else 0))
    for nm, m in (("starts after D", r[:, 3] == 1), ("starts after turnover", r[:, 3] == 0)):
        print("  {:22s} n={:3d} found {:5.1f}%".format(nm, m.sum(), 100 * hit[m].mean()))
    # dead segmentation
    g = np.zeros(len(lab), bool)
    for a, b in c["blocks"]: g[a:b + 1] = True
    tD, pD = (lab == "D")[g], (pred == "D")[g]
    print("  D frames: precision {:.2f} recall {:.2f}".format((tD & pD).sum() / pD.sum(), (tD & pD).sum() / tD.sum()))
    alive = ~tD & ~pD
    print("  team accuracy on frames both alive: {:.3f}".format(np.mean(lab[g][alive] == pred[g][alive])))
    def nseg(m):
        return int(np.sum(np.diff(np.concatenate(([0], m.astype(int), [0]))) == 1))
    print("  D segments true {} pred {}".format(nseg(tD), nseg(pD)))
