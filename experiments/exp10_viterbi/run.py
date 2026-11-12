"""exp10 on the provided match: a Viterbi decoder over H / A / D -> evaluation/results/exp10_viterbi.csv

    python experiments/exp10_viterbi/run.py                    # everything on
    python experiments/exp10_viterbi/run.py --no-touch         # -> exp10_viterbi_notouch.csv
    python experiments/exp10_viterbi/run.py --no-tagging       # -> exp10_viterbi_notag.csv
    python experiments/exp10_viterbi/run.py --a-priori         # config.A_PRIORI, not IDSSE's choice

The evidence is exp08's and exp09's (see emissions.py); only the decision rule is new. Values come
from config.chosen(): IDSSE's choice if idsse_eval.py has filled config.CHOSEN, else the a priori
fallback. The match is never scored here.

Deterministic: no randomness anywhere on this path, and Viterbi breaks ties by a fixed rule, so
two runs write byte-identical files.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
for path in (ROOT, EXPERIMENTS, os.path.join(ROOT, "starter")):
    if path not in sys.path:
        sys.path.insert(0, path)

from exp10_viterbi import config, decode, emissions  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")
LABEL = np.array(["H", "A", "D", "X"])


def build_labels(params, use_touch=True, use_tagging=True, game=GAME, data_dir=None):
    features, p_dead, touches, smooth = emissions.match_inputs(game, data_dir, with_touch=use_touch)
    E = emissions.build(features, game, p_dead, smooth, params["ha_scale"],
                        params["touch_gain"] if use_touch else 0.0, touches)
    n = GAMES[game]["n_video_frames"]
    relief_h = relief_a = None
    if use_tagging:
        relief_h, relief_a = emissions.tag_relief(game, n, GAMES[game]["fps"], data_dir)
    codes = decode.decode_halves(E, half_video_blocks(game), params["c_switch"],
                                 params["c_dead"], params["c_dead"], relief_h, relief_a)
    return LABEL[codes]


def default_out(use_touch, use_tagging, a_priori):
    name = "exp10_viterbi"
    if not use_touch:
        name += "_notouch"
    if not use_tagging:
        name += "_notag"
    if a_priori:
        name += "_apriori"
    return os.path.join(RESULTS, name + ".csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-touch", action="store_true", help="drop exp09's touch bonus")
    parser.add_argument("--no-tagging", action="store_true",
                        help="drop the Smart Tagging transition relief")
    parser.add_argument("--a-priori", action="store_true", help="use config.A_PRIORI")
    parser.add_argument("--out")
    args = parser.parse_args()

    params = dict(config.A_PRIORI) if args.a_priori else config.chosen()
    source = "a priori" if args.a_priori or not config.CHOSEN else "IDSSE"
    labels = build_labels(params, not args.no_touch, not args.no_tagging)
    out = args.out or default_out(not args.no_touch, not args.no_tagging, args.a_priori)
    path = write_submission(out, labels)

    graded = np.zeros(len(labels), dtype=bool)
    for _h, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} -> {}  ({} values: {}; touch {}, tagging {})".format(
        len(labels), os.path.relpath(path, ROOT), source, params,
        not args.no_touch, not args.no_tagging))
    for name in ("H", "A", "D", "X"):
        print("  {}  {:5.1f}%".format(name, 100.0 * np.mean(labels[graded] == name)))
    assert not (labels[graded] == "X").any(), "X inside a graded half"
    live = np.isin(labels, ["H", "A"]) & graded
    edges = np.flatnonzero(live[1:] & ((labels[1:] != labels[:-1]) | ~live[:-1])) + 1
    print("  possessions over the full match: {}".format(len(edges) + int(live[0])))
