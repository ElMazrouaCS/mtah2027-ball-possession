"""Aggregate every evaluation record in extval/results/.

    python -m extval.report                 # one table per dataset, every record listed
    python -m extval.report --params <hash> # only the runs of one parameter set

For each record: mean spellF1 / macroF1 per dataset (metrica, idsse) and per method, so results
of different datasets are evaluated independently and only then put side by side.
"""

import argparse
from collections import defaultdict

import numpy as np

from . import records


def summarise(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["method"], r["dataset"])].append(r)
    return {k: (np.mean([r["spell_f1"] for r in v]), np.mean([r["macro_f1"] for r in v]),
                len({r["game"] for r in v}), len({r["scenario"] for r in v}))
            for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    args = ap.parse_args()
    recs = [r for r in records.load_all() if args.params in (None, r["params_hash"])]
    if not recs:
        print("no record in extval/results/ yet: run python -m extval.bench or extval.tune")
        return
    print("{:34s} {:6s} {:13s} {:7s} {:24s} {:>7s} {:>7s} {:>5s} {:>5s}".format(
        "record", "kind", "params", "dataset", "method", "spellF1", "macroF1", "games", "scen."))
    for rec in recs:
        for (method, ds), (sf, mf, ng, ns) in sorted(summarise(rec["rows"]).items(),
                                                     key=lambda kv: (kv[0][1], kv[0][0])):
            print("{:34s} {:6s} {:13s} {:7s} {:24s} {:7.3f} {:7.3f} {:5d} {:5d}".format(
                rec["file"][:34], rec["kind"], rec["params_hash"], ds, method, sf, mf, ng, ns))
        if rec["kind"] == "tune":
            print("{:34s}   trained on {} -> best params hash {}".format(
                "", ",".join(rec.get("train_games", [])), rec.get("best_params_hash", "?")))


if __name__ == "__main__":
    main()
