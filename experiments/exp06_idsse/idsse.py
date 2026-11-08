"""Loader for the IDSSE / DFL Bundesliga dataset (Bassek et al. 2025, CC BY 4.0).

    python experiments/exp06_idsse/idsse.py --report J03WMX

Why this dataset
----------------
CHALLENGE.md says the ground truth is derived mechanically from *corrected tracking data*: a
frame is D where the corrected tracking marks the ball dead, otherwise the team the tracking names
as owning it. IDSSE ships exactly those two flags per frame, from DFL's corrected TRACAB tracking
at 25 fps: BallPossession (1 home, 2 away) and BallStatus (0 inactive, 1 active). It is not the
provided match, so tuning on it is what the rules ask for.

Format
------
The positions file is one XML <FrameSet> per person (players, referees, ball), each holding every
frame of one half, in metres from the pitch centre. It is parsed line by line with a regex rather
than a XML parser because each <Frame/> is one line and the files are 350-420 MB.

Data lives in data/external/idsse/, which is git-ignored, and is downloaded from figshare
(DOI 10.6084/m9.figshare.28196177) by download.py.
"""

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)

DATA_DIR = os.path.join(ROOT, "data", "external", "idsse")

FRAMESET = re.compile(r'<FrameSet\s([^>]*)>')
FRAME = re.compile(r'<Frame\s([^>]*)/>')
ATTR = re.compile(r'(\w+)="([^"]*)"')

HOME, AWAY = 0, 1
FPS = 25


def path(match_id, kind, data_dir=None):
    """Path of one of a match's three files: 'info', 'events' or 'positions'."""
    data_dir = data_dir or DATA_DIR
    prefix = {"info": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
              "positions": "DFL_04_03_positions_raw_observed"}[kind]
    for name in sorted(os.listdir(data_dir)):
        if name.startswith(prefix) and match_id in name:
            return os.path.join(data_dir, name)
    raise FileNotFoundError("no {} file for {} in {}".format(kind, match_id, data_dir))


def matches(data_dir=None):
    """Match ids that have all three files downloaded."""
    data_dir = data_dir or DATA_DIR
    ids = sorted({re.search(r'(DFL-MAT-\w+)\.xml', n).group(1)
                  for n in os.listdir(data_dir) if n.startswith("DFL_04_03")})
    out = []
    for match in ids:
        try:
            for kind in ("info", "events", "positions"):
                path(match, kind, data_dir)
            out.append(match)
        except FileNotFoundError:
            pass
    return out


def teams(match_id, data_dir=None):
    """{'home': team_id, 'away': team_id} read from the match information file."""
    text = open(path(match_id, "info", data_dir), encoding="utf-8").read()
    general = dict(ATTR.findall(re.search(r'<General\s([^>]*)/?>', text).group(1)))
    return {"home": general["HomeTeamId"], "away": general["GuestTeamId"],
            "home_name": general["HomeTeamName"], "away_name": general["GuestTeamName"]}


def load(match_id, data_dir=None, keep_attrs=False):
    """Parse a positions file.

    Returns {half_name: {"frames": int array, "tracks": [{"team", "person", "x", "y", ...}],
    "ball": {...}}} where team is HOME / AWAY / 'referee' and x, y are float arrays aligned to
    `frames`. The ball's extra per-frame fields (possession, status, z) are returned as arrays too.
    """
    side = teams(match_id, data_dir)
    team_code = {side["home"]: HOME, side["away"]: AWAY}
    halves = {}
    current = None
    with open(path(match_id, "positions", data_dir), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                attrs = dict(ATTR.findall(FRAMESET.match(line).group(1)))
                current = {"half": attrs["GameSection"], "team": attrs["TeamId"],
                           "person": attrs.get("PersonId"), "rows": []}
                continue
            if line.startswith("</FrameSet"):
                if current is not None:
                    halves.setdefault(current["half"], []).append(current)
                current = None
                continue
            if current is not None and line.startswith("<Frame"):
                current["rows"].append(dict(ATTR.findall(FRAME.match(line).group(1))))

    out = {}
    for half, sets in halves.items():
        numbers = sorted({int(r["N"]) for s in sets for r in s["rows"]})
        index = {n: i for i, n in enumerate(numbers)}
        entry = {"frames": np.array(numbers, dtype=np.int64), "tracks": [], "ball": None}
        for s in sets:
            n = len(numbers)
            x = np.full(n, np.nan)
            y = np.full(n, np.nan)
            extra = {}
            for r in s["rows"]:
                i = index[int(r["N"])]
                x[i], y[i] = float(r["X"]), float(r["Y"])
                if s["team"] == "BALL" or keep_attrs:
                    for key, value in r.items():
                        if key in ("N", "T", "X", "Y"):
                            continue
                        extra.setdefault(key, np.full(n, np.nan))[i] = _num(value)
            track = {"team": team_code.get(s["team"], s["team"]), "person": s["person"],
                     "x": x, "y": y, "extra": extra}
            if s["team"] == "BALL":
                entry["ball"] = track
            else:
                entry["tracks"].append(track)
        out[half] = entry
    return out


def _num(value):
    try:
        return float(value)
    except ValueError:
        return float("nan")


def report(match_id, data_dir=None):
    side = teams(match_id, data_dir)
    print("{}  {} (home) v {} (away)".format(match_id, side["home_name"], side["away_name"]))
    data = load(match_id, data_dir)
    for half, entry in data.items():
        ball = entry["ball"]
        n = len(entry["frames"])
        print("\n{}: {} frames ({:.1f} min), frame numbers {}..{}".format(
            half, n, n / FPS / 60.0, entry["frames"][0], entry["frames"][-1]))
        kinds = {}
        for t in entry["tracks"]:
            kinds[t["team"]] = kinds.get(t["team"], 0) + 1
        print("  tracks:", kinds)
        if ball is None:
            print("  no ball track")
            continue
        print("  ball fields:", sorted(ball["extra"]))
        for key, arr in sorted(ball["extra"].items()):
            vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
            if len(vals) <= 6:
                print("    {:<16} {}".format(key, dict(zip(vals.tolist(),
                                                           (counts / n).round(3).tolist()))))
            else:
                print("    {:<16} {} distinct, range {:.2f}..{:.2f}".format(
                    key, len(vals), np.nanmin(arr), np.nanmax(arr)))
        print("  ball missing x/y: {:.1f}%".format(100.0 * np.isnan(ball["x"]).mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", metavar="MATCH", help="e.g. J03WMX")
    args = parser.parse_args()
    if args.report:
        report(args.report)
    else:
        print("downloaded:", matches())
