"""Download the 7 IDSSE matches (first run) and print their label statistics.

    python -m extval.idsse_probe

Open data of other matches: everything printed here can be shared freely.
"""

import time
from collections import Counter

import numpy as np

from . import idsse
from . import opendata as od


def spells(lab, blocks):
    out = []
    for a, b in blocks:
        w = lab[a:b + 1]
        st = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        out += [(st[i + 1] - st[i]) / od.FPS for i in range(len(st) - 1) if w[st[i]] in "HA"]
    return np.array(out)


def main():
    import kloppy
    print("kloppy", kloppy.__version__)
    for m in idsse.MATCHES:
        t0 = time.time()
        try:
            trk, lab, blocks = idsse.load(m, rule=False)
        except Exception as exc:                       # keep going, report the failure
            print(m, "FAILED:", type(exc).__name__, str(exc)[:200])
            continue
        g = np.concatenate([np.arange(a, b + 1) for a, b in blocks])
        sp = spells(lab, blocks)
        sp3 = spells(idsse.merge_short(lab, blocks, 3.0), blocks)
        ctrl = idsse.control_rule(lab, blocks)
        spc = spells(ctrl, blocks)
        ok = np.isfinite(trk["bx"][g])
        dead = lab[g] == "D"
        rst = idsse.restarts(trk, ctrl, blocks)
        print("{}  {:.1f} min  slots {}  players visible {:.1f}  ball present {:.1f}%  "
              "(in D {:.1f}%)".format(m, len(g) / od.FPS / 60, np.bincount(trk["teams"]).tolist(),
                                     np.isfinite(trk["px"][g]).sum(1).mean(), 100 * ok.mean(),
                                     100 * ok[dead].mean() if dead.any() else float("nan")))
        print("   D {:.1f}%  H {:.1f}%  A {:.1f}%  possessions {} mean {:.1f}s  <2s {:.0f}%  "
              "| merged <3s: {} mean {:.1f}s".format(
                  100 * dead.mean(), 100 * np.mean(lab[g] == "H"), 100 * np.mean(lab[g] == "A"),
                  len(sp), sp.mean(), 100 * np.mean(sp < 2), len(sp3), sp3.mean()))
        print("   control rule (used by the bench): possessions {} mean {:.1f}s  <2s {:.0f}%".format(
            len(spc), spc.mean(), 100 * np.mean(spc < 2)))
        print("   restarts {}  ({:.0f}s)".format(dict(Counter(k for _, k, _, _, _ in rst)),
                                               time.time() - t0))


if __name__ == "__main__":
    main()
