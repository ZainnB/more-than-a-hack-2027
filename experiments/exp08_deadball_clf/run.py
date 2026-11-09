"""exp08 on the provided match: exp06's possession layer, stoppages from a dead-ball classifier
trained on IDSSE, and the referee's whistle lowering the classifier's threshold.

    python experiments/exp08_deadball_clf/run.py                   # -> evaluation/results/exp08.csv
    python experiments/exp08_deadball_clf/run.py --no-whistle      # classifier alone
    python experiments/exp08_deadball_clf/run.py --possession retuned --no-whistle   # 1.5% re-tune
    python experiments/exp08_deadball_clf/run.py --dead rules --possession retuned --no-whistle

The default is the configuration adopted in README.md. The other options reproduce its tables.
"""

import argparse
import json
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
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp07_team_colour"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import deadclf  # noqa: E402
import deadfeat  # noqa: E402
import harness  # noqa: E402  (knob table; also applies exp04's a priori overrides)
import players as players_mod  # noqa: E402
from exp03_keeper_fix import features as feat3  # noqa: E402
from exp01_geometry_hysteresis import team as exp01_team  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp08.csv")

# exp06's possession values, kept because IDSSE could not tell them from the 1.5% re-tune
# (0.420 against 0.422 with the classifier, inside seed noise). See README.md.
POSSESSION = {"exp06": {"use_touch": True, "margin_m": 1.0, "k_fallback": 75},
              "retuned": {"use_touch": False, "margin_m": 0.5, "k_fallback": 15}}

# The dead-ball rules' values from retune.py; they only matter with --dead rules.
RULES = {"slow_speed": 1.5, "slow_min_frames": 500, "max_dead": 250, "ball_lost_frames": 125}


def apply(values):
    for name, value in values.items():
        module, attr = harness.KNOBS[name]
        setattr(module, attr, value)


def classifier_dead(game=GAME, data_dir=None, use_whistle=True):
    """Dead mask for the provided match from deadclf's final model (trained on all of IDSSE)."""
    with open(deadclf.MODEL_PATH) as h:
        model = json.load(h)
    p = players_mod.load(game, data_dir)
    length, width = GAMES[game]["pitch"]
    ball = np.stack([(p["ball_x"] - 0.5) * length, (p["ball_y"] - 0.5) * width], axis=1)
    players = np.stack([(p["x"] - 0.5) * length, (p["y"] - 0.5) * width], axis=2)
    X = deadfeat.compute(ball.astype(np.float64), players.astype(np.float64),
                         length / 2.0, width / 2.0)
    prob = deadclf.proba(model, X)
    d = model["decode"]
    if not use_whistle:
        return deadclf.decode(prob, half_video_blocks(game), **d)
    import whistle  # PyAV only if the cache is missing; never in the same process as opencv
    frames = [int(e[0] * 25) for e in whistle.load(game)["events"]]
    return deadclf.decode_with_whistle(prob, half_video_blocks(game), frames, d["smooth"],
                                       d["min_dead"], d["min_live"])


def build_labels(dead_source="classifier", use_whistle=True, game=GAME, data_dir=None):
    features = feat3.load(game, data_dir)
    votes = exp01_team.tag_team_votes(features, game, data_dir)
    blocks = half_video_blocks(game)
    hook = None
    if dead_source == "classifier":
        import tune
        hook = tune.replace_dead(classifier_dead(game, data_dir, use_whistle), blocks)
    return run04.labels_from_features(features, blocks, game, GAMES[game]["n_video_frames"],
                                      votes=votes, mode="none", data_dir=data_dir,
                                      dead_hook=hook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", choices=["classifier", "rules"], default="classifier")
    parser.add_argument("--possession", choices=list(POSSESSION), default="exp06")
    parser.add_argument("--no-whistle", action="store_true",
                        help="classifier threshold 0.5 everywhere, no whistle")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    apply(RULES)
    apply(POSSESSION[args.possession])
    labels, _detail = build_labels(args.dead, not args.no_whistle)
    path = write_submission(args.out, labels)
    graded = np.zeros(len(labels), dtype=bool)
    for _h, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} -> {}  (dead: {}, whistle: {}, possession: {})".format(
        len(labels), os.path.relpath(path, ROOT), args.dead, not args.no_whistle,
        args.possession))
    for name in ("H", "A", "D", "X"):
        print("  {}  {:5.1f}%".format(name, 100.0 * np.mean(labels[graded] == name)))
