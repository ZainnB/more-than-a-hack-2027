"""exp01's feature extraction, re-run with the keeper corrections applied.

    python experiments/exp03_keeper_fix/features.py --report

Only the per-team distances change. The ball track, the tracked-player count and the mean player
speed are all team-agnostic and come out identical; what moves is `d_home` and `d_away` on the
frames where a keeper slot was tracking somebody from the other side.

Cached separately from exp01's features, because the two differ and silently sharing a cache
between them would be the kind of bug that takes an afternoon.
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
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import HOME, AWAY, team_of_each_track  # noqa: E402

try:
    from . import keeper
except ImportError:
    import keeper

# exp01's derived views work unchanged on the corrected dict -- they only read the arrays.
from exp01_geometry_hysteresis.features import (  # noqa: E402
    ball_seen, ball_inside, ball_speed_mps, team_margin, NEAR_RADII)

CACHE_NAME = "features_keeper_{}.npz"


def cache_path(game=GAME, folder=None):
    return os.path.join(folder or os.path.join(HERE, "cache"), CACHE_NAME.format(game))


def extract(game=GAME, data_dir=None, apply_correction=True):
    """Same output shape as exp01's extract(), with keeper-corrected team assignment."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    slot_team = team_of_each_track(meta_path)

    frames = keeper.parse(raw_path, slot_team)
    slot_x = np.full((n_frames, len(slot_team)), np.nan, dtype=np.float32)
    for frame, positions, _ball in frames:
        for slot, x, _y, _team in positions:
            if slot < len(slot_team):
                slot_x[frame, slot] = x
    keepers = keeper.keeper_slots(slot_x) if apply_correction else []

    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    d_home = np.full(n_frames, np.nan, dtype=np.float32)
    d_away = np.full(n_frames, np.nan, dtype=np.float32)
    near = {r: (np.zeros(n_frames, np.int8), np.zeros(n_frames, np.int8)) for r in NEAR_RADII}
    n_tracked = np.zeros(n_frames, dtype=np.int16)
    n_corrected = 0

    # Player speeds are not re-derived here; they are team-agnostic, so exp01's are reused below.
    for frame, positions, ball_xy in frames:
        n_tracked[frame] = len(positions)
        teams, changed = keeper.correct_frame(positions, keeper.half_of(frame, game), keepers,
                                              game) if apply_correction else (
            [t for _s, _x, _y, t in positions], [])
        n_corrected += len(changed)

        if ball_xy is None:
            continue
        ball[frame, 0], ball[frame, 1] = ball_xy
        ball_mx, ball_my = ball_xy[0] * length, ball_xy[1] * width

        best = [np.inf, np.inf]
        counts = {r: [0, 0] for r in NEAR_RADII}
        for (_slot, x, y, _old), team in zip(positions, teams):
            dx, dy = x * length - ball_mx, y * width - ball_my
            distance = (dx * dx + dy * dy) ** 0.5
            side = 0 if team == HOME else 1
            if distance < best[side]:
                best[side] = distance
            for radius in NEAR_RADII:
                if distance < radius:
                    counts[radius][side] += 1
        if best[0] < np.inf:
            d_home[frame] = best[0]
        if best[1] < np.inf:
            d_away[frame] = best[1]
        for radius in NEAR_RADII:
            near[radius][0][frame] = counts[radius][0]
            near[radius][1][frame] = counts[radius][1]

    out = {"ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_speed": ball[:, 2],
           "d_home": d_home, "d_away": d_away, "n_tracked": n_tracked,
           "n_corrected": np.array([n_corrected])}
    for radius in NEAR_RADII:
        key = str(int(radius))
        out["n_home_" + key] = near[radius][0]
        out["n_away_" + key] = near[radius][1]

    # mean_speed is independent of team assignment, so take exp01's rather than reparse for it.
    from exp01_geometry_hysteresis import features as exp01_features
    out["mean_speed"] = exp01_features.load(game, data_dir)["mean_speed"]
    return out


def load(game=GAME, data_dir=None, rebuild=False, folder=None, apply_correction=True):
    if not apply_correction:
        return extract(game, data_dir, apply_correction=False)
    path = cache_path(game, folder)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}
    features = extract(game, data_dir)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    np.savez_compressed(path, **features)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    corrected = load(rebuild=args.rebuild)
    print("keeper-corrected features -> {}".format(os.path.relpath(cache_path(), ROOT)))
    print("  {} player-frames reassigned".format(int(corrected["n_corrected"][0])))

    if args.report:
        from exp01_geometry_hysteresis import features as exp01_features
        base = exp01_features.load()
        for key in ("ball_x", "n_tracked", "mean_speed"):
            same = np.array_equal(np.nan_to_num(base[key], nan=-999),
                                  np.nan_to_num(corrected[key], nan=-999))
            print("  {:<12} identical to exp01: {}".format(key, same))
        for key in ("d_home", "d_away"):
            a, b = base[key], corrected[key]
            differ = ~np.isclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999))
            print("  {:<12} differs on {} frames".format(key, int(differ.sum())))
        nearest_a, _m, has_a = exp01_features.team_margin(base)
        nearest_b, _m2, has_b = team_margin(corrected)
        flipped = has_a & has_b & (nearest_a != nearest_b)
        print("\n  the nearest-player TEAM flips on {} frames ({:.2f}% of all frames)".format(
            int(flipped.sum()), 100.0 * flipped.mean()))
        print("  those are the frames where possession could change hands differently.")
