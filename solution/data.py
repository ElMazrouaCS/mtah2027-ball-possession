"""Read the provided assets into numpy arrays. Standard library + numpy only.

The parsed tracking is cached in cache/atd.npz (gitignored: derived data must never be published).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, asset_paths, half_video_blocks

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "atd.npz")
N_FRAMES = GAMES[GAME]["n_video_frames"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def team_of_slots():
    """0 = home, 1 = away, per track slot (first Team in the metadata is home, as in baseline.py)."""
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_raw():
    teams = team_of_slots()
    n_slots = len(teams)
    px = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    py = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    ps = np.full((N_FRAMES, n_slots), np.nan, dtype=np.float32)
    bx = np.full(N_FRAMES, np.nan)
    by = np.full(N_FRAMES, np.nan)
    with open(asset_paths()["atd_raw"]) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1          # ATD frame_id counts from 1, video frames from 0
            for j, g in enumerate(parts[1].split(";")):
                if g[0] == "N":
                    continue
                f = g.split(",")
                px[vf, j] = float(f[0]) * LENGTH
                py[vf, j] = float(f[1]) * WIDTH
                if f[2][0] != "N":
                    ps[vf, j] = float(f[2])
            b = parts[2].split(",")
            if b[0][0] != "N":
                bx[vf] = float(b[0]) * LENGTH
                by[vf] = float(b[1]) * WIDTH
    return dict(px=px, py=py, ps=ps, bx=bx, by=by, teams=teams)


def load_tracking():
    """dict of arrays in METRES: px, py, ps (frames x slots), bx, by (frames), teams (slots)."""
    if os.path.isfile(CACHE):
        with np.load(CACHE) as z:
            return {k: z[k] for k in z.files}
    data = _parse_raw()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **data)
    return data


def load_tagging(fps):
    """[(start_frame, end_frame, CODE, team or None, type or None)] from the Smart Tagging XML.

    team is 0 (home) / 1 (away) / None (N/A). Home is matched by name, Barcelona per DATA.md.
    """
    out = []
    for inst in ET.parse(asset_paths()["smart_tagging"]).iter("instance"):
        tags = {(lab.findtext("group") or "").strip().lower(): (lab.findtext("text") or "").strip()
                for lab in inst.iter("label")}
        team_txt = tags.get("team", "").lower()
        team = 0 if "barcelona" in team_txt else (1 if "athletic" in team_txt
                                                   or "atletic" in team_txt else None)
        out.append((int(round(float(inst.findtext("start")) * fps)),
                    int(round(float(inst.findtext("end")) * fps)),
                    (inst.findtext("code") or "").strip().upper(), team,
                    tags.get("type")))
    return out


def graded_blocks():
    """[(first, last_inclusive)] video frames of each graded half."""
    return [(a, b) for _, a, b in half_video_blocks(GAME)]
