"""Turn hand-stamped possession boundaries into a full-length frame,label ground truth.

    python validation/expand.py
    python validation/expand.py --out validation/gt/val15.csv

Input: one small CSV per window in validation/labels/, each row a boundary -- the frame at which
a label *starts*. Output: a single 199319-row file in exactly the submission format, so it drops
into the existing scorer with no changes at all:

    python evaluation/run_evaluation.py --gt validation/gt/val15.csv

Everything outside the annotated windows is X, which the scorer never grades. That is the same
shape the public ground truth has, and it is why a full-match submission can be scored against
15 minutes of labels without being truncated first.

On trusting this file
---------------------
It is hand-made, so it is wrong in the ways humans are wrong, and the checks below are what stop
the *mechanical* errors from hiding among the judgement calls. A boundary file is rejected if its
frames are not strictly ascending, if any frame falls outside the window it claims, if two
consecutive stamps carry the same label (which means a boundary was stamped where nothing
changed), or if the first stamp is not at the window's first frame -- without that the window
would open on an undefined label.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import windows as win  # noqa: E402
from games import GAME, GAMES  # noqa: E402

# Params
LABELS_DIR = os.path.join(HERE, "labels")
GT_DIR = os.path.join(HERE, "gt")
DEFAULT_OUT = os.path.join(GT_DIR, "val15.csv")
VALID = ("H", "A", "D")  # X is never stamped: it means "not annotated", not "unsure"


def label_path(name):
    return os.path.join(LABELS_DIR, "{}.csv".format(name))


def read_boundaries(name):
    """[(frame, label)] for one window, validated. Raises ValueError with a usable message."""
    path = label_path(name)
    if not os.path.isfile(path):
        raise ValueError("{}: no such file. Annotate it first:\n"
                         "    python validation/annotate.py --window {}".format(path, name))

    start, stop = win.frames(name)
    rows = []
    with open(path) as handle:
        for number, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.lower().startswith("frame"):
                continue  # header or blank
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 2:
                raise ValueError("{}:{}: expected 'frame,label', got {!r}".format(
                    path, number, text))
            try:
                frame = int(parts[0])
            except ValueError:
                raise ValueError("{}:{}: frame {!r} is not an integer".format(
                    path, number, parts[0]))
            label = parts[1].upper()
            if label not in VALID:
                raise ValueError("{}:{}: label {!r} is not one of {}".format(
                    path, number, parts[1], list(VALID)))
            if not (start <= frame < stop):
                raise ValueError("{}:{}: frame {} is outside {} [{}, {})".format(
                    path, number, frame, name, start, stop))
            rows.append((frame, label))

    if not rows:
        raise ValueError("{}: no boundaries".format(path))
    for (earlier, _), (later, _) in zip(rows, rows[1:]):
        if later <= earlier:
            raise ValueError("{}: frames must ascend, but {} follows {}".format(
                path, later, earlier))
    for (_, first), (frame, second) in zip(rows, rows[1:]):
        if first == second:
            raise ValueError("{}: frame {} stamps {} again, but it is already {} -- a boundary "
                             "has to change the label".format(path, frame, second, first))
    if rows[0][0] != start:
        raise ValueError("{}: the first boundary is at {}, but the window opens at {}. Stamp the "
                         "opening label on the window's first frame.".format(
                             path, rows[0][0], start))
    return rows


def expand(names=None, n_frames=None, game=GAME):
    """The full-length label array. X everywhere that was not annotated."""
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    labels = np.full(n_frames, "X", dtype="<U1")

    for name in (names or list(win.WINDOWS)):
        start, stop = win.frames(name)
        boundaries = read_boundaries(name)
        # Each boundary holds until the next one; the last holds to the end of the window.
        edges = [frame for frame, _ in boundaries] + [stop]
        for (frame, label), end in zip(boundaries, edges[1:]):
            labels[frame:min(end, n_frames)] = label
    return labels


def write(path, labels):
    """Identical format to a submission: header, then one frame,label row per video frame."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--window", action="append", dest="names",
                        help="expand only this window; repeatable")
    args = parser.parse_args()

    try:
        labels = expand(args.names)
    except ValueError as error:
        print("FAIL  {}".format(error))
        sys.exit(1)

    path = write(args.out, labels)
    fps = GAMES[GAME]["fps"]

    print("wrote {} rows -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    annotated = int(np.sum(labels != "X"))
    print("\nannotated {} frames ({:.1f} minutes)".format(annotated, annotated / float(fps) / 60))
    for name in (args.names or list(win.WINDOWS)):
        start, stop = win.frames(name)
        block = labels[start:stop]
        counts = {k: int(np.sum(block == k)) for k in VALID}
        alive = counts["H"] + counts["A"]
        changes = int(np.sum(block[1:] != block[:-1]))
        print("  {:<6} H {:>5}  A {:>5}  D {:>5}   home share {:.2f}   {} segments".format(
            name, counts["H"], counts["A"], counts["D"],
            counts["H"] / float(alive) if alive else 0.0, changes + 1))

    counts = {k: int(np.sum(labels == k)) for k in VALID}
    alive = counts["H"] + counts["A"]
    print("\n  overall  H {}  A {}  D {}   home share {:.3f}   dead share {:.3f}".format(
        counts["H"], counts["A"], counts["D"],
        counts["H"] / float(alive) if alive else 0.0,
        counts["D"] / float(annotated) if annotated else 0.0))
    print("\nScore against it with:")
    print("  python evaluation/run_evaluation.py --gt {} --bootstrap 1000".format(
        os.path.relpath(path, ROOT)))
