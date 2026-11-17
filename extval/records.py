"""Evaluation records: every bench or tuning run is written to its own JSON file, never overwritten.

    extval/results/<UTC time>_<kind>_<games>.json

Each record carries what is needed to trust and reproduce it: the git commit, the games and
scenarios, the exact parameter values (and their hash), and one row per (method, game, scenario).
`python -m extval.report` aggregates every record.
"""

import datetime
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "extval", "results")


def dataset_of(game):
    return "metrica" if isinstance(game, int) else "idsse"


def params_hash(P):
    values = {k: v for k, v in sorted(vars(P).items())}
    blob = json.dumps(values, sort_keys=True, default=list).encode()
    return hashlib.sha256(blob).hexdigest()[:12], values


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "solution"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+modified-solution" if dirty else "")
    except Exception:
        return "unknown"


def rows_from(method, rows):
    """bench.evaluate rows (scenario, game, spellF1, macroF1, n_pred, n_gt) -> dicts."""
    return [dict(method=method, scenario=r[0], game=str(r[1]), dataset=dataset_of(r[1]),
                 spell_f1=round(float(r[2]), 6), macro_f1=round(float(r[3]), 6),
                 n_pred=int(r[4]), n_gt=int(r[5])) for r in rows]


def save(kind, games, scenarios, P, rows, extra=None):
    h, values = params_hash(P)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "-".join(sorted({dataset_of(g) for g in games}))
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "{}_{}_{}.json".format(stamp, kind, tag))
    k = 1
    while os.path.exists(path):                      # never overwrite
        path = os.path.join(RESULTS, "{}_{}_{}_{}.json".format(stamp, kind, tag, k)); k += 1
    record = dict(kind=kind, created_utc=stamp, git=git_commit(), games=[str(g) for g in games],
                  scenarios=list(scenarios), params_hash=h, params=values, rows=rows)
    record.update(extra or {})
    with open(path, "w") as fh:
        json.dump(record, fh, indent=1, default=list)
    print("record written:", os.path.relpath(path, ROOT))
    return path


def load_all():
    out = []
    if os.path.isdir(RESULTS):
        for name in sorted(os.listdir(RESULTS)):
            if name.endswith(".json"):
                with open(os.path.join(RESULTS, name)) as fh:
                    rec = json.load(fh)
                rec["file"] = name
                out.append(rec)
    return out
