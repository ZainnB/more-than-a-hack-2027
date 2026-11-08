"""The exp04 pipeline with the parameters tuned on IDSSE, run on the provided match.

    python experiments/exp06_idsse/run.py          # -> evaluation/results/exp06.csv

The pipeline is exp04's, unchanged. Only the values in tuned.TUNED differ, and every one of
them came from external data (see tuned.py and tune.py).
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

import harness  # noqa: E402  (holds the knob table)
from tuned import TUNED  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402
from baseline import write_submission  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp06.csv")

# Applied at import, as exp04 does with its own overrides, so every entry point sees the same
# configuration.
for _name, _value in TUNED.items():
    _module, _attr = harness.KNOBS[_name]
    setattr(_module, _attr, _value)


def build_labels():
    return run04.build_labels(mode="none")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    labels, _detail = build_labels()
    path = write_submission(args.out, labels)
    graded = np.zeros(len(labels), dtype=bool)
    for _half, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    print("config: {}".format(TUNED))
    for name in ("H", "A", "D", "X"):
        count = int(np.sum((labels == name) & graded))
        print("  {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / graded.sum()))
