"""Every ATD track slot's position on every frame, cached as arrays.

    python experiments/exp07_team_colour/players.py          # build the cache, print a summary

exp01's features keep only per-team distances to the ball. The colour step needs to know where
each individual player is, so it can be found in the video, so this parses the raw ATD once into
(frames, slots) arrays of normalised 0-1 coordinates -- the coordinates the provided homography
takes directly. Derived match data: the cache lives in cache/, which is git-ignored.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, require, source_frames  # noqa: E402
from baseline import FIELD_SEP, GROUP_SEP, MISSING, team_of_each_track  # noqa: E402

CACHE = os.path.join(HERE, "cache", "players_{}.npz")


def extract(game=GAME, data_dir=None):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_track(meta_path)
    n = source_frames("tactical", game)
    x = np.full((n, len(team)), np.nan, dtype=np.float32)
    y = np.full((n, len(team)), np.nan, dtype=np.float32)
    bx = np.full(n, np.nan, dtype=np.float32)
    by = np.full(n, np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # ATD frame ids count from 1, submissions from 0
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if not group or group[0] == MISSING or slot >= len(team):
                    continue
                fields = group.split(",")
                x[frame, slot] = float(fields[0])
                y[frame, slot] = float(fields[1])
            ball = parts[2].strip()
            if ball and ball[0] != MISSING:
                fields = ball.split(",")
                bx[frame], by[frame] = float(fields[0]), float(fields[1])
    return {"x": x, "y": y, "team": team, "ball_x": bx, "ball_y": by}


def load(game=GAME, data_dir=None, rebuild=False):
    path = CACHE.format(game)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as h:
            return {k: h[k] for k in h.files}
    out = extract(game, data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


if __name__ == '__main__':
    p = load()
    tracked = ~np.isnan(p["x"])
    print("{} frames x {} slots; home slots {}, away slots {}".format(
        p["x"].shape[0], p["x"].shape[1], int((p["team"] == 0).sum()), int((p["team"] == 1).sum())))
    print("players tracked per frame, median over frames with any: {:.0f}".format(
        np.median(tracked.sum(axis=1)[tracked.any(axis=1)])))
