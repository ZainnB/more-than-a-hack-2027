"""Build the submission: tracking in, scored frame,label CSV out.

    python experiments/exp01_geometry_hysteresis/run.py
    python evaluation/validate_submission.py evaluation/results/exp01.csv
    python evaluation/run_evaluation.py

Three layers, in order, each measurable on its own:

    features.py   one pass over the raw ATD -> cached per-frame arrays
    deadball.py   which frames are stoppages        (precision 97.5%, recall 74.5%)
    team.py       who has the ball on the rest      (team accuracy 0.8621)

Everything outside the two graded halves is written as X. Those frames are never scored -- the
ground truth is X there too -- so an abstention costs nothing, and writing a real label would be
inventing one.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

try:
    from . import config, deadball, features as feat, team as team_layer
except ImportError:
    import config
    import deadball
    import features as feat
    import team as team_layer
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp01.csv")


def build_labels(game=GAME, data_dir=None, rebuild=False):
    """The full 199319-label array, plus the intermediate masks for reporting."""
    features = feat.load(game, data_dir, rebuild=rebuild)
    dead, parts, restarts = deadball.dead_mask(features, game, data_dir)
    votes = team_layer.tag_team_votes(features, game, data_dir)
    codes = team_layer.possession_state(features, dead, votes, game=game)

    n_frames = GAMES[game]["n_video_frames"]
    labels = np.full(n_frames, "X", dtype="<U1")

    # Only inside the graded halves. half_video_blocks is derived from the tracking data rather
    # than hardcoded, and the frames outside it are halftime and the pre/post-match camera.
    for _half, first, last in half_video_blocks(game):
        window = slice(first, last + 1)
        block = np.where(dead[window], "D",
                         np.where(codes[window] == team_layer.HOME, "H",
                                  np.where(codes[window] == team_layer.AWAY, "A", "X")))
        labels[window] = block

    return labels, {"features": features, "dead": dead, "parts": parts,
                    "restarts": restarts, "codes": codes}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the submission")
    parser.add_argument("--rebuild", action="store_true", help="re-extract the ATD features")
    args = parser.parse_args()

    labels, detail = build_labels(rebuild=args.rebuild)
    path = write_submission(args.out, labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    print("\nlabel counts over the {} graded frames".format(int(graded.sum())))
    for name in ("H", "A", "D", "X"):
        count = int(np.sum((labels == name) & graded))
        print("  {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / graded.sum()))

    print("\ndead-ball rules, frames claimed inside the graded halves")
    for name in deadball.ALL_RULES:
        if name in detail["parts"]:
            print("  {:<10} {:>7}".format(name, int((detail["parts"][name] & graded).sum())))
    print("  {:<10} {:>7}   restarts detected: {}".format(
        "union", int((detail["dead"] & graded).sum()), len(detail["restarts"])))

    print("\nNow check it, then score it:")
    print("  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("  python evaluation/run_evaluation.py")
