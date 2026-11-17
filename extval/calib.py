"""Compare label-free statistics of degraded open data with those measured on the provided ATD."""
import numpy as np
from solution import evidence
from . import opendata as od, degrade as dg

TARGET = dict(present=59.3, gap_3_10=14.6, gap_10_30=41.4, gap_30=43.8, sp_missing=69.0,
              rand_missing=42.2, players=20.5)


def stats(trk, tags, blocks):
    n = len(trk["bx"]); g = np.zeros(n, bool)
    for a, b in blocks: g[a:b + 1] = True
    ok = np.isfinite(trk["bx"])
    gl = []
    for a, b in blocks:
        gl += [e - s for s, e in evidence.runs_of(~ok[a:b + 1])]
    gl = np.array(gl); tot = gl.sum()
    sp = [s for s, _, c, _, _ in tags if c == "SET PIECES" and 0 <= s < n and g[s]]
    rng = np.random.default_rng(0); rnd = rng.choice(np.flatnonzero(g), 500, replace=False)
    pl = np.isfinite(trk["px"][g]).sum(1).mean() * 22 / trk["px"].shape[1] if False else \
        np.isfinite(trk["px"][g]).sum(1).mean()
    return dict(present=100 * ok[g].mean(),
                gap_3_10=100 * gl[(gl >= 75) & (gl < 250)].sum() / tot,
                gap_10_30=100 * gl[(gl >= 250) & (gl < 750)].sum() / tot,
                gap_30=100 * gl[gl >= 750].sum() / tot,
                sp_missing=100 * np.mean(~ok[sp]) if sp else np.nan,
                rand_missing=100 * np.mean(~ok[rnd]), players=pl)


if __name__ == "__main__":
    import sys
    sc_kw = eval("dict(%s)" % (sys.argv[1] if len(sys.argv) > 1 else ""))
    print("target      ", {k: round(v, 1) for k, v in TARGET.items()})
    for g in (1, 2, 3):
        e = od.events(g); trk = od.tracking(g); lab, blocks = od.labels(e, trk["n"])
        rst = od.restarts(e, lab, blocks)
        dtrk, tags = dg.degrade(trk, lab, blocks, rst, dg.Scenario(seed=g, **sc_kw))
        print("game", g, "    ", {k: round(v, 1) for k, v in stats(dtrk, tags, blocks).items()})
