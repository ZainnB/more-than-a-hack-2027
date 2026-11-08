"""One IDSSE match as a single clean scene, laid out like the provided match.

    python experiments/exp06_idsse/scene.py J03WMX

The ATD is graded from kickoff to the end of each half, with an ungraded gap at half time. A scene
is built the same way so the same scorer and the same pipeline run on it unchanged:

  - each half is trimmed to its live span: first to last frame with BallStatus = 1. DFL's raw
    frames start well before kickoff and the dead frames there would flatter the dead-ball score.
  - the two halves are concatenated with GAP_FRAMES of nothing between them, labelled X in the
    ground truth, so valid_runs() splits them exactly as it does at the real half time.

Everything is in metres from the pitch centre. Ground truth is the IDSSE flags: D where
BallStatus is 0, otherwise H or A from BallPossession (1 home, 2 away). See day1_check.py for how
that compares with the public ground truth.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
# Frames of nothing between the halves. Longer than any smoothing window used downstream, so no
# filter leaks across, and long enough to read as a real gap.
GAP_FRAMES = 200

HALF_ORDER = ("firstHalf", "secondHalf")


def build(match_id, data_dir=None):
    """The clean scene for one match, or raises if a half has no live frames."""
    data = idsse.load(match_id, data_dir)

    # One column per person across both halves: substitutes only exist in one of them.
    persons = {}
    for half in HALF_ORDER:
        for t in data[half]["tracks"]:
            if t["team"] in (idsse.HOME, idsse.AWAY):
                persons.setdefault(t["person"], t["team"])
    order = sorted(persons)
    column = {p: i for i, p in enumerate(order)}
    team = np.array([persons[p] for p in order], dtype=np.int8)

    parts_ball, parts_z, parts_players, parts_gt, blocks = [], [], [], [], []
    cursor = 0
    for number, half in enumerate(HALF_ORDER, start=1):
        entry = data[half]
        extra = entry["ball"]["extra"]
        status, poss = extra["BallStatus"], extra["BallPossession"]
        live = np.flatnonzero(status == 1)
        a, b = int(live[0]), int(live[-1]) + 1
        n = b - a

        gt = np.full(n, pm.X, dtype=np.int8)
        s, p = status[a:b], poss[a:b]
        gt[(s == 1) & (p == 1)] = pm.H
        gt[(s == 1) & (p == 2)] = pm.A
        gt[s == 0] = pm.D

        players = np.full((n, len(order), 2), np.nan)
        for t in entry["tracks"]:
            if t["team"] in (idsse.HOME, idsse.AWAY):
                players[:, column[t["person"]], 0] = t["x"][a:b]
                players[:, column[t["person"]], 1] = t["y"][a:b]

        parts_ball.append(np.stack([entry["ball"]["x"][a:b], entry["ball"]["y"][a:b]], axis=1))
        parts_z.append(extra["Z"][a:b])
        parts_players.append(players)
        parts_gt.append(gt)
        blocks.append((number, cursor, cursor + n - 1))
        cursor += n

        if number == 1:
            parts_ball.append(np.full((GAP_FRAMES, 2), np.nan))
            parts_z.append(np.full(GAP_FRAMES, np.nan))
            parts_players.append(np.full((GAP_FRAMES, len(order), 2), np.nan))
            parts_gt.append(np.full(GAP_FRAMES, pm.X, dtype=np.int8))
            cursor += GAP_FRAMES

    return {"match": match_id, "n": cursor, "blocks": blocks, "team": team,
            "ball": np.concatenate(parts_ball), "ball_z": np.concatenate(parts_z),
            "players": np.concatenate(parts_players), "gt": np.concatenate(parts_gt)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("match")
    args = parser.parse_args()
    s = build(args.match)
    print("{}: {} frames, blocks {}, {} persons ({} home, {} away)".format(
        s["match"], s["n"], s["blocks"], len(s["team"]), int((s["team"] == 0).sum()),
        int((s["team"] == 1).sum())))
    g = s["gt"]
    for name, code in (("H", pm.H), ("A", pm.A), ("D", pm.D), ("X", pm.X)):
        print("  {} {:>6.1f}%".format(name, 100.0 * (g == code).mean()))
