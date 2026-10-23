"""Build exp02's submission: exp01's dead-ball layer, a touch-gated possession layer.

    python experiments/exp02_touch_gated/run.py                  # the configured exp02
    python experiments/exp02_touch_gated/run.py --touch          # ablation: add the touch path
    python experiments/exp02_touch_gated/run.py --k-fallback 75  # ablation: the planned 3 s
    python experiments/exp02_touch_gated/run.py --no-backdate    # ablation: label changes late

The ablations exist because exp02 changed three things at once relative to exp01 -- touch gating,
a stricter fallback, and backdating -- and the comparison has to say which one earned what. The
answer, measured on 20 annotated minutes, is in README.md: backdating and the stricter fallback
did; touch gating did not.
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
    from . import config, features as feat, team as team_layer, touch as touch_layer
except ImportError:
    import config
    import features as feat
    import team as team_layer
    import touch as touch_layer

from exp01_geometry_hysteresis import deadball, team as exp01_team  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp02.csv")


def build_labels(game=GAME, data_dir=None, use_touch=True, use_fallback=True):
    features = feat.load(game, data_dir)
    kin = feat.kinematics(features, game)
    dead, _parts, restarts = deadball.dead_mask(features, game, data_dir)

    # Restarts are unioned in as touches: the bend detector measurably misses most of them (see
    # touch.py), and possession changes hands at a restart more reliably than anywhere else.
    touches, touch_team = touch_layer.detect(features, kin, extra=restarts)
    contested = touch_layer.contested(features)
    votes = exp01_team.tag_team_votes(features, game, data_dir)

    codes, why = team_layer.possession_state(
        features, kin, dead, touches, touch_team, contested, votes, game=game,
        use_touch=use_touch, use_fallback=use_fallback)
    codes = team_layer.merge_short(
        codes, dead, int(round(config.MIN_POSSESSION_S * GAMES[game]["fps"])))

    labels = np.full(GAMES[game]["n_video_frames"], "X", dtype="<U1")
    for _half, first, last in half_video_blocks(game):
        window = slice(first, last + 1)
        labels[window] = np.where(dead[window], "D",
                                  np.where(codes[window] == team_layer.HOME, "H",
                                           np.where(codes[window] == team_layer.AWAY, "A", "X")))
    return labels, {"why": why, "touches": touches, "dead": dead}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--touch", action="store_true",
                        help="turn the touch path on (off by default, see config.USE_TOUCH)")
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--k-fallback", type=int,
                        help="override config.K_FALLBACK_FRAMES, e.g. 75 for the planned value")
    parser.add_argument("--no-backdate", action="store_true")
    args = parser.parse_args()
    if args.k_fallback:
        config.K_FALLBACK_FRAMES = args.k_fallback
    if args.no_backdate:
        config.BACKDATE_FALLBACK = False

    labels, detail = build_labels(use_touch=(config.USE_TOUCH or args.touch) and not args.no_touch,
                                  use_fallback=not args.no_fallback)
    path = write_submission(args.out, labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    why = detail["why"]
    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    for name in ("H", "A", "D", "X"):
        count = int(np.sum((labels == name) & graded))
        print("  {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / graded.sum()))
    print("possession changes: {} by touch, {} by fallback, {} acquired after a restart".format(
        int(((why == 1) & graded).sum()), int(((why == 2) & graded).sum()),
        int(((why == 3) & graded).sum())))
