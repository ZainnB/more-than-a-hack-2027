"""exp03's pipeline with every possession boundary snapped onto the event that caused it.

    python experiments/exp04_boundary_timing/run.py
    python experiments/exp04_boundary_timing/run.py --mode none      # reproduces exp03 exactly
    python experiments/exp04_boundary_timing/run.py --mode crossing
    python experiments/exp04_boundary_timing/run.py --margin 1.0 --k 25

Everything up to the boundary refinement is exp03's, imported rather than copied, so `--mode
none` is byte-identical to exp03 and the difference is attributable to the refinement alone.
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
    from . import config as exp04_config, refine
except ImportError:
    import config as exp04_config
    import refine

from exp03_keeper_fix import features as feat4  # noqa: E402
from exp02_touch_gated import config, features as feat2, team as team_layer, \
    touch as touch_layer  # noqa: E402
from exp01_geometry_hysteresis import deadball, team as exp01_team  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp04.csv")
DEFAULT_MODE = exp04_config.SNAP_MODE

# exp04 inherits exp02's possession layer and overrides two of its thresholds. Applied at import
# so every entry point -- run.py, refine.py --report, an interactive session -- sees the same
# configuration, rather than only the command line.
config.MARGIN_M = exp04_config.MARGIN_M
config.K_FALLBACK_FRAMES = exp04_config.K_FALLBACK_FRAMES


def build_labels(game=GAME, data_dir=None, mode=DEFAULT_MODE, window=None):
    features = feat4.load(game, data_dir)
    kin = feat2.kinematics(features, game)
    dead, _parts, restarts = deadball.dead_mask(features, game, data_dir)

    touches, touch_team = touch_layer.detect(features, kin, extra=restarts)
    contested = touch_layer.contested(features)
    votes = exp01_team.tag_team_votes(features, game, data_dir)

    codes, _why = team_layer.possession_state(
        features, kin, dead, touches, touch_team, contested, votes, game=game,
        use_touch=config.USE_TOUCH)
    codes = team_layer.merge_short(
        codes, dead, int(round(config.MIN_POSSESSION_S * GAMES[game]["fps"])))

    nearest, _margin, has_answer = feat2.team_margin(features)
    codes, moves = refine.snap(codes, dead, touches, touch_team, nearest, has_answer,
                               mode=mode,
                               window=exp04_config.SNAP_WINDOW if window is None else window)

    labels = np.full(GAMES[game]["n_video_frames"], "X", dtype="<U1")
    for _half, first, last in half_video_blocks(game):
        sl = slice(first, last + 1)
        labels[sl] = np.where(dead[sl], "D",
                              np.where(codes[sl] == team_layer.HOME, "H",
                                       np.where(codes[sl] == team_layer.AWAY, "A", "X")))
    return labels, {"moves": moves, "dead": dead}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--mode", default=DEFAULT_MODE,
                        choices=["none", "crossing", "touch"])
    parser.add_argument("--window", type=int, help="snap search radius in frames")
    parser.add_argument("--margin", type=float, help="override config.MARGIN_M")
    parser.add_argument("--k", type=int, help="override config.K_FALLBACK_FRAMES")
    args = parser.parse_args()

    if args.margin is not None:
        config.MARGIN_M = args.margin
    if args.k is not None:
        config.K_FALLBACK_FRAMES = args.k

    labels, detail = build_labels(mode=args.mode, window=args.window)
    path = write_submission(args.out, labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} rows -> {}  (mode={}, margin={}, k={})".format(
        len(labels), os.path.relpath(path, ROOT), args.mode,
        config.MARGIN_M, config.K_FALLBACK_FRAMES))
    for name in ("H", "A", "D", "X"):
        count = int(np.sum((labels == name) & graded))
        print("  {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / graded.sum()))
    shifts = [abs(n - o) for o, n, _a in detail["moves"]]
    if shifts:
        print("boundaries moved: {}  median shift {} frames".format(
            len(shifts), int(np.median(shifts))))
