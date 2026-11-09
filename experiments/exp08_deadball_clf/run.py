"""exp08 on the provided match: possession values re-tuned at 1.5% flips, and stoppages from the
dead-ball classifier trained on IDSSE instead of the hand-written rules.

    python experiments/exp08_deadball_clf/run.py                   # -> evaluation/results/exp08.csv
    python experiments/exp08_deadball_clf/run.py --dead rules      # re-tuned rules instead
    python experiments/exp08_deadball_clf/run.py --whistle         # anchor stoppages on whistles
    python experiments/exp08_deadball_clf/run.py --possession exp06 --out evaluation/results/exp08b.csv

Every value here was chosen on IDSSE (retune.py, deadclf.py). The whistle anchor is off by
default: IDSSE has no audio, so it cannot be validated on external data.
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
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp05_whistle"))
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

# From retune.py at 1.5% flips. The possession values are the same with the classifier
# (retune.py --classifier) and with the rules; the rule values only matter with --dead rules.
TUNED = {"use_touch": False, "margin_m": 0.5, "k_fallback": 15,
         "slow_speed": 1.5, "slow_min_frames": 500, "max_dead": 250, "ball_lost_frames": 125}
ANCHOR_S = 5.0

# exp06's possession values. With the classifier, IDSSE scores them 0.420 against 0.422 for the
# re-tuned ones (retune.py --classifier), a difference inside seed noise. See README.md for why
# this option exists and why it was added after looking at the provided match.
EXP06_POSSESSION = {"use_touch": True, "margin_m": 1.0, "k_fallback": 75}

for _name, _value in TUNED.items():
    _module, _attr = harness.KNOBS[_name]
    setattr(_module, _attr, _value)


def classifier_dead(game=GAME, data_dir=None, whistle_threshold=False):
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
    if whistle_threshold:
        import whistle
        frames = [int(e[0] * 25) for e in whistle.load(game)["events"]]
        d = model["decode"]
        return deadclf.decode_with_whistle(prob, half_video_blocks(game), frames, d["smooth"],
                                           d["min_dead"], d["min_live"])
    return deadclf.decode(prob, half_video_blocks(game), **model["decode"])


def whistle_anchor(dead, game=GAME):
    """Move each stoppage that starts within ANCHOR_S after a whistle back to the whistle."""
    import whistle
    out = dead.copy()
    for start_s, _e, _hz, _db in whistle.load(game)["events"]:
        w = int(start_s * 25)
        if w >= len(out) or out[w]:
            continue
        ahead = out[w:w + int(ANCHOR_S * 25)]
        if ahead.any():
            out[w:w + int(np.argmax(ahead))] = True
    return out


def build_labels(dead_source="classifier", anchor=False, game=GAME, data_dir=None,
                 whistle_threshold=False):
    features = feat3.load(game, data_dir)
    votes = exp01_team.tag_team_votes(features, game, data_dir)
    blocks = half_video_blocks(game)
    hook = None
    if dead_source == "classifier":
        mask = classifier_dead(game, data_dir, whistle_threshold)
        if anchor:
            mask = whistle_anchor(mask, game)
        import tune
        hook = tune.replace_dead(mask, blocks)
    elif anchor:
        def hook(dead, restarts, _features):
            return whistle_anchor(dead, game), restarts
    return run04.labels_from_features(features, blocks, game, GAMES[game]["n_video_frames"],
                                      votes=votes, mode="none", data_dir=data_dir,
                                      dead_hook=hook)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dead", choices=["classifier", "rules"], default="classifier")
    parser.add_argument("--whistle", action="store_true")
    parser.add_argument("--possession", choices=["retuned", "exp06"], default="retuned")
    parser.add_argument("--whistle-threshold", action="store_true",
                        help="threshold 0.6 normally, 0.3 for 10 s after a whistle")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.possession == "exp06":
        for _name, _value in EXP06_POSSESSION.items():
            _module, _attr = harness.KNOBS[_name]
            setattr(_module, _attr, _value)
    labels, _detail = build_labels(args.dead, args.whistle, whistle_threshold=args.whistle_threshold)
    path = write_submission(args.out, labels)
    graded = np.zeros(len(labels), dtype=bool)
    for _h, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} -> {}  (dead: {}, whistle anchor: {}, possession: {})".format(
        len(labels), os.path.relpath(path, ROOT), args.dead, args.whistle, args.possession))
    for name in ("H", "A", "D", "X"):
        print("  {}  {:5.1f}%".format(name, 100.0 * np.mean(labels[graded] == name)))
