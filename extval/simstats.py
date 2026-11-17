"""Same statistics as diag_step3.py, on degraded open data (to compare simulator vs reality)."""
import numpy as np
from solution import config, evidence
from . import bench, opendata as od

def pct(a, ps):
    a = a[np.isfinite(a)]
    return "  ".join("p{}={:.2f}".format(p, np.percentile(a, p)) for p in ps)

if __name__ == "__main__":
    import sys
    P = config.params()
    kw = eval("dict(%s)" % (sys.argv[1] if len(sys.argv) > 1 else ""))
    for c in bench.build_cases(scenarios={"x": dict(bench.SCENARIOS["central"], **kw)}):
        trk, blocks = c["trk"], c["blocks"]; n = len(trk["bx"]); g = np.zeros(n, bool)
        for a, b in blocks: g[a:b + 1] = True
        ok, sx, sy, speed, oob = evidence.ball_features(trk["bx"], trk["by"], od.LENGTH, od.WIDTH, P)
        dh, da = evidence.team_distances(trk["px"], trk["py"], trk["teams"], sx, sy)
        dmin = np.fmin(dh, da); m = g & ok & np.isfinite(dmin)
        v = evidence.collective_speed(trk["ps"], P)
        print("game", c["game"])
        print("  dmin       ", pct(dmin[m], (10, 25, 50, 75, 90)))
        print("  dmin slow  ", pct(dmin[m & (np.nan_to_num(speed) < 3)], (10, 25, 50, 75, 90)))
        print("  share <1/1.5/2/3 m", [round(100 * np.mean(dmin[m] < t), 1) for t in (1, 1.5, 2, 3)])
        print("  |dh-da|    ", pct(np.abs(dh - da)[m], (10, 25, 50, 75, 90)))
        print("  vcoll ball visible", pct(v[g & ok], (10, 50, 90)), "| missing", pct(v[g & ~ok], (10, 50, 90)))
