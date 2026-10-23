"""exp02's pipeline, run on keeper-corrected tracking. Nothing else changes.

    python experiments/exp03_keeper_fix/run.py
    python experiments/exp03_keeper_fix/run.py --no-correction   # ablation: exp02 exactly

Every layer is exp02's, imported rather than copied -- the dead-ball rules, the touch detector,
the possession state machine and every threshold. The only difference is the feature dict they
are handed, which has the keeper identity switches corrected (see keeper.py). So the difference
in score is attributable to the correction and to nothing else.
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

try:
    from . import features as feat
except ImportError:
    import features as feat

from exp02_touch_gated import config, features as exp02_features, team as team_layer, \
    touch as touch_layer  # noqa: E402
from exp01_geometry_hysteresis import deadball, team as exp01_team  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp03.csv")


def build_labels(game=GAME, data_dir=None, correct=True, use_touch=None):
    features = feat.load(game, data_dir, apply_correction=correct)
    kin = exp02_features.kinematics(features, game)
    dead, _parts, restarts = deadball.dead_mask(features, game, data_dir)

    touches, touch_team = touch_layer.detect(features, kin, extra=restarts)
    contested = touch_layer.contested(features)
    votes = exp01_team.tag_team_votes(features, game, data_dir)

    codes, why = team_layer.possession_state(
        features, kin, dead, touches, touch_team, contested, votes, game=game,
        use_touch=config.USE_TOUCH if use_touch is None else use_touch)
    codes = team_layer.merge_short(
        codes, dead, int(round(config.MIN_POSSESSION_S * GAMES[game]["fps"])))

    labels = np.full(GAMES[game]["n_video_frames"], "X", dtype="<U1")
    for _half, first, last in half_video_blocks(game):
        window = slice(first, last + 1)
        labels[window] = np.where(dead[window], "D",
                                  np.where(codes[window] == team_layer.HOME, "H",
                                           np.where(codes[window] == team_layer.AWAY, "A", "X")))
    return labels, {"why": why, "dead": dead, "features": features}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--no-correction", action="store_true")
    parser.add_argument("--touch", action="store_true")
    args = parser.parse_args()

    labels, detail = build_labels(correct=not args.no_correction,
                                  use_touch=True if args.touch else None)
    path = write_submission(args.out, labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    for name in ("H", "A", "D", "X"):
        count = int(np.sum((labels == name) & graded))
        print("  {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / graded.sum()))
    if not args.no_correction:
        print("keeper corrections applied: {} player-frames".format(
            int(detail["features"]["n_corrected"][0])))
