"""exp06's pipeline and tuned values, with near-ball team labels checked against shirt colour.

    python experiments/exp07_team_colour/run.py          # -> evaluation/results/exp07.csv

Only frames where decide.py changed a label have their per-team ball distances recomputed;
every other frame's features are exp03's, byte for byte, so any difference from exp06 is the
colour step's.
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
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import decide  # noqa: E402
import players as players_mod  # noqa: E402
from exp06_idsse import run as run06  # noqa: E402  (applies exp06's tuned values at import)
from exp03_keeper_fix import features as feat3  # noqa: E402
from exp01_geometry_hysteresis import team as exp01_team  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp07.csv")


def patched_features(game=GAME, data_dir=None):
    """exp03's features with team distances recomputed where colour changed a label."""
    features = {k: v.copy() for k, v in feat3.load(game, data_dir).items()}
    p = players_mod.load(game, data_dir)
    fixed, base, stats = decide.decide(game, data_dir)
    length, width = GAMES[game]["pitch"]
    frames = np.unique(np.argwhere(fixed != base)[:, 0])
    for f in frames:
        if np.isnan(p["ball_x"][f]):
            continue
        d = np.hypot((p["x"][f] - p["ball_x"][f]) * length, (p["y"][f] - p["ball_y"][f]) * width)
        for side, key in ((0, "d_home"), (1, "d_away")):
            mine = d[(fixed[f] == side) & ~np.isnan(d)]
            features[key][f] = mine.min() if mine.size else np.nan
            for radius in (3, 5):
                features["n_{}_{}".format("home" if side == 0 else "away", radius)][f] = \
                    int((mine < radius).sum())
    return features, len(frames), stats


def build_labels(game=GAME, data_dir=None):
    features, n_patched, stats = patched_features(game, data_dir)
    votes = exp01_team.tag_team_votes(features, game, data_dir)
    labels, detail = run04.labels_from_features(
        features, half_video_blocks(game), game, GAMES[game]["n_video_frames"],
        votes=votes, mode="none", data_dir=data_dir)
    detail.update(n_patched=n_patched, colour_stats=stats)
    return labels, detail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    labels, detail = build_labels()
    path = write_submission(args.out, labels)
    print("wrote {} rows -> {}   ({} frames re-labelled by colour)".format(
        len(labels), os.path.relpath(path, ROOT), detail["n_patched"]))
