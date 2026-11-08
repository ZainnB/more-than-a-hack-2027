"""exp06's pipeline and tuned values, with stoppages anchored and added by the whistle.

    python experiments/exp05_whistle/run.py          # -> evaluation/results/exp05.csv
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

import stoppage  # noqa: E402
from exp06_idsse import run as run06  # noqa: E402  (applies exp06's tuned values at import)
from exp03_keeper_fix import features as feat3  # noqa: E402
from exp01_geometry_hysteresis import team as exp01_team  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp05.csv")


def build_labels(game=GAME, data_dir=None):
    features = feat3.load(game, data_dir)
    votes = exp01_team.tag_team_votes(features, game, data_dir)
    stats = {}
    labels, detail = run04.labels_from_features(
        features, half_video_blocks(game), game, GAMES[game]["n_video_frames"], votes=votes,
        mode="none", data_dir=data_dir, dead_hook=stoppage.make_hook(game, stats))
    detail["whistle_stats"] = stats
    return labels, detail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    labels, detail = build_labels()
    path = write_submission(args.out, labels)
    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    print("whistles: {}".format(detail["whistle_stats"]))
