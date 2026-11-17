"""Step 0 -- STRUCTURE of the provided data. Prints aggregates only.

    python diagnostics/diag_step0.py

Tag paths, channel names (digits masked), field counts, % NaN, number of distinct values.
Never prints a position, a speed, a label or a player name. Main question: does the automatic
tracking carry a ball-alive/dead or an owning-team channel? (Answer: no.)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import re
import xml.etree.ElementTree as ET
from collections import Counter

from games import GAME, asset_paths, half_video_blocks

PATHS = asset_paths()
KEYWORDS = ("state", "status", "alive", "dead", "live", "owner", "owning", "possess", "inplay",
            "in_play")
MAX_DISTINCT = 20  # au-dela on affiche seulement ">20"


def graded_mask_fn():
    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    return lambda vf: any(a <= vf <= b for a, b in blocks)


def anonymise(text):
    """'player12_x' -> 'player#_x' : on garde la forme du nom, pas les numeros."""
    return re.sub(r"\d+", "#", text or "")


def pct(n, d):
    return "{:6.2f}%".format(100.0 * n / d) if d else "   n/a"


# ---------------------------------------------------------------- 1. XML de metadonnees ATD
def diag_atd_xml():
    print("=" * 70)
    print("1. ATD XML (metadonnees)")
    root = ET.parse(PATHS["atd_meta"]).getroot()

    # Arborescence des balises (chemins anonymises, avec comptes)
    paths = Counter()

    def walk(el, prefix):
        tag = el.tag.split("}")[-1]
        path = prefix + "/" + tag
        paths[path] += 1
        for child in el:
            walk(child, path)
    walk(root, "")
    print("\n1a. Arborescence (chemin : nombre d'occurrences)")
    for path, n in sorted(paths.items()):
        print("   {:5d}  {}".format(n, path))

    # Noms d'attributs par balise
    attrs = {}
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        attrs.setdefault(tag, set()).update(el.attrib.keys())
    print("\n1b. Noms d'attributs par balise")
    for tag in sorted(attrs):
        if attrs[tag]:
            print("   {:30s} {}".format(tag, sorted(attrs[tag])))

    # Canaux : identifiants anonymises (structure, pas des donnees)
    channels = Counter()
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if "channel" in tag.lower():
            ident = el.get("id") or el.get("channelId") or el.get("playerChannelId") or ""
            name = el.findtext("Name") or el.get("name") or ""
            channels["{} | id={} | name={}".format(tag, anonymise(ident), anonymise(name))] += 1
    print("\n1c. Canaux (numeros remplaces par #)")
    for key, n in sorted(channels.items()):
        print("   {:5d}  {}".format(n, key))

    # Recherche de mots-cles dans balises / attributs / identifiants
    hits = Counter()
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        for kw in KEYWORDS:
            if kw in tag.lower():
                hits["balise " + tag] += 1
            for k, v in el.attrib.items():
                if kw in k.lower():
                    hits["attribut {}@{}".format(tag, k)] += 1
                if kw in (v or "").lower():
                    hits["valeur d'attribut {}@{} contient '{}'".format(tag, k, kw)] += 1
            if el.text and kw in el.text.lower():
                hits["texte de <{}> contient '{}'".format(tag, kw)] += 1
    print("\n1d. Mots-cles etat/proprietaire trouves dans le XML")
    if not hits:
        print("   aucun")
    for key, n in sorted(hits.items()):
        print("   {:5d}  {}".format(n, key))


# ---------------------------------------------------------------- 2. TXT brut ATD
def diag_atd_raw():
    print("\n" + "=" * 70)
    print("2. ATD TXT brut (stats sur les frames notees uniquement)")
    is_graded = graded_mask_fn()

    n_lines = n_graded = 0
    n_parts = Counter()           # nb de blocs separes par ':'
    n_player_groups = Counter()   # nb de groupes joueurs par ligne
    player_fields = Counter()     # nb de champs par groupe joueur
    ball_fields = Counter()       # nb de champs du groupe balle
    present_players = 0           # somme des joueurs non-NaN (frames notees)
    ball_nan = Counter()          # par position de champ balle
    ball_numeric = Counter()
    ball_distinct = {}            # position -> set (plafonne)
    extra_parts_distinct = {}     # si plus de 3 blocs ':'

    with open(PATHS["atd_raw"]) as fh:
        for line in fh:
            line = line.rstrip("\n")
            parts = line.split(":")
            n_lines += 1
            n_parts[len(parts)] += 1
            try:
                vf = int(parts[0]) - 1
            except ValueError:
                continue
            if not is_graded(vf):
                continue
            n_graded += 1

            groups = parts[1].split(";") if len(parts) > 1 else []
            n_player_groups[len(groups)] += 1
            for g in groups:
                f = g.split(",")
                player_fields[len(f)] += 1
                if not f[0].startswith("N"):
                    present_players += 1

            if len(parts) > 2:
                bf = parts[2].split(",")
                ball_fields[len(bf)] += 1
                for i, v in enumerate(bf):
                    v = v.strip()
                    if v == "" or v.lower() == "nan":
                        ball_nan[i] += 1
                        continue
                    try:
                        float(v)
                        ball_numeric[i] += 1
                    except ValueError:
                        pass
                    s = ball_distinct.setdefault(i, set())
                    if len(s) <= MAX_DISTINCT:
                        s.add(v)

            for j in range(3, len(parts)):
                s = extra_parts_distinct.setdefault(j, set())
                if len(s) <= MAX_DISTINCT:
                    s.add(parts[j])

    print("   lignes totales : {}   frames notees : {}".format(n_lines, n_graded))
    print("   blocs ':' par ligne (nb_blocs: nb_lignes) : {}".format(dict(n_parts)))
    print("   groupes joueurs par ligne : {}".format(dict(n_player_groups)))
    print("   champs par groupe joueur  : {}".format(dict(player_fields)))
    print("   joueurs presents par frame notee (moyenne) : {:.2f}".format(
        present_players / max(n_graded, 1)))
    print("   champs du groupe balle    : {}".format(dict(ball_fields)))
    print("\n   Par champ du groupe balle (frames notees) :")
    print("   champ   %NaN      %numerique  nb_valeurs_distinctes")
    n_fields = max(ball_fields) if ball_fields else 0
    for i in range(n_fields):
        d = len(ball_distinct.get(i, ()))
        print("   {:5d}  {}   {}     {}".format(
            i, pct(ball_nan[i], n_graded), pct(ball_numeric[i], n_graded),
            ">{}".format(MAX_DISTINCT) if d > MAX_DISTINCT else d))
    if extra_parts_distinct:
        print("\n   Blocs ':' supplementaires (au-dela de frame:joueurs:balle) :")
        for j, s in sorted(extra_parts_distinct.items()):
            d = len(s)
            print("   bloc {} : nb_valeurs_distinctes = {}".format(
                j, ">{}".format(MAX_DISTINCT) if d > MAX_DISTINCT else d))
    print("\n   Lecture : un champ avec peu de valeurs distinctes (2 a 5) et non numerique")
    print("   ressemblerait a un etat de balle ou a une equipe. x, y, vitesse en ont des milliers.")


# ---------------------------------------------------------------- 3. Vue kloppy (colonnes)
def diag_kloppy():
    print("\n" + "=" * 70)
    print("3. kloppy : colonnes du DataFrame (echantillon 1 frame / seconde)")
    try:
        from kloppy import metrica
    except ImportError:
        print("   kloppy absent -> section sautee (pip install -r requirements-starter.txt)")
        return
    ds = metrica.load_tracking_epts(meta_data=PATHS["atd_meta"], raw_data=PATHS["atd_raw"],
                                    sample_rate=1 / 25)
    df = ds.to_df()
    is_graded = graded_mask_fn()
    if "frame_id" in df.columns:
        df = df[[is_graded(int(f) - 1) for f in df["frame_id"]]]
    print("   frames echantillonnees notees : {}".format(len(df)))

    player_cols = [c for c in df.columns if re.match(r"^.*\d+_[xyzsd]$", str(c))]
    other_cols = [c for c in df.columns if c not in player_cols]
    print("   colonnes joueurs : {} (forme : {})".format(
        len(player_cols), sorted({anonymise(str(c)) for c in player_cols})))
    if player_cols:
        print("   %NaN moyen colonnes joueurs : {:.2f}%".format(
            100 * df[player_cols].isna().mean().mean()))
    print("\n   autres colonnes : nom | type | %NaN | nb_valeurs_distinctes")
    for c in other_cols:
        s = df[c]
        nun = s.nunique(dropna=True)
        print("   {:22s} {:10s} {:7.2f}%  {}".format(
            str(c), str(s.dtype), 100 * s.isna().mean(),
            ">{}".format(MAX_DISTINCT) if nun > MAX_DISTINCT else nun))
    print("\n   A regarder : ball_state et ball_owning_team. Si 100% NaN (ou 1 seule valeur),")
    print("   l'ATD automatique ne donne PAS ces canaux.")


# ---------------------------------------------------------------- 4. Smart Tagging
def diag_smart_tagging():
    print("\n" + "=" * 70)
    print("4. Smart Tagging : structure")
    root = ET.parse(PATHS["smart_tagging"]).getroot()
    tags = Counter(el.tag for el in root.iter())
    print("   balises : {}".format(dict(tags)))
    groups = Counter()
    for inst in root.iter("instance"):
        for lab in inst.iter("label"):
            groups[lab.findtext("group") or "(sans groupe)"] += 1
    print("   groupes de tags (nb d'occurrences) : {}".format(dict(groups)))


# ---------------------------------------------------------------- 5. Homographie
def diag_homography():
    print("\n" + "=" * 70)
    print("5. Homographie : structure")
    with open(PATHS["homography"]) as fh:
        header = fh.readline().strip().split(",")
        n = valid = 0
        vi = header.index("valid") if "valid" in header else None
        for line in fh:
            n += 1
            if vi is not None and line.rstrip("\n").split(",")[vi].strip() in ("1", "1.0", "True"):
                valid += 1
    print("   colonnes : {}".format(header))
    print("   lignes : {}   valid=1 : {}".format(n, pct(valid, n)))


if __name__ == "__main__":
    for step in (diag_atd_xml, diag_atd_raw, diag_kloppy, diag_smart_tagging, diag_homography):
        try:
            step()
        except Exception as exc:  # message sans valeurs : type + nom de l'etape
            print("\n   !! {} a echoue : {}".format(step.__name__, type(exc).__name__))
