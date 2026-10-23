"""Compare a blind hand annotation of the public window with the organisers' ground truth.

    python validation/annotate.py --window public        # annotate first -- blind, see below
    python validation/calibrate.py

Why this exists
---------------
The 15 minutes in WINDOWS are labelled by us, and exp02's 2 s threshold was chosen on them. The
score that decides the competition is computed against the organisers' labels, which come from
*corrected tracking*, mechanically: dead where the corrected tracking marks the ball dead, else
whichever team it names as owning the ball. If our convention differs -- dead balls stamped a
second later, possession handed over at the first touch rather than at control -- then exp02
learned our convention, not the scorer's, and every number in compare.py inherits the gap.

Re-annotating the one window that has an official answer measures that gap directly:

  1. agreement        how often the two label the same frame the same way
  2. human ceiling    the hand labels scored as if they were a submission. Roughly the best a
                      system can expect to reach, since it is the official labels' agreement
                      with a careful human watching the video
  3. boundary bias    for each official boundary, the matching hand boundary and its offset in
                      frames, by kind. A consistent sign is a convention difference, and the
                      fix is to shift the validation labels, not the model
  4. model re-score   every submission scored against both labellings of the same 5 minutes.
                      If an experiment's rank changes between them, it is fitted to a convention

Blind means blind
-----------------
Annotate *before* opening the official CSV, the scoreboard detail for this window, or any
experiment's output for it. A calibration made after seeing the answer measures nothing. The
annotation tool shows raw tracking only, for exactly this reason.
"""

import argparse
import collections
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import windows as win  # noqa: E402
import expand as expander  # noqa: E402
import compare  # noqa: E402

# Params
NAME = "public"
SEARCH_FRAMES = 125   # how far either side to look for a matching hand boundary (5 s)
FPS = 25


def boundaries(codes, start, stop):
    """[(frame, from_label, to_label)] where the label changes inside [start, stop)."""
    window = codes[start:stop]
    change = np.flatnonzero(window[1:] != window[:-1]) + 1
    return [(start + int(i), int(window[i - 1]), int(window[i])) for i in change]


def kind(before, after):
    if after == pm.D:
        return "dead starts"
    if before == pm.D:
        return "dead ends"
    return "possession changes"


def match_boundaries(official, hand):
    """Greedy nearest matching of each official boundary to an unused hand boundary of the same
    from/to labels within SEARCH_FRAMES. Returns (matched [(kind, offset)], missed, extra)."""
    used = set()
    matched, missed = [], []
    for frame, a, b in official:
        best = None
        for j, (hf, ha, hb) in enumerate(hand):
            if j in used or (ha, hb) != (a, b) or abs(hf - frame) > SEARCH_FRAMES:
                continue
            if best is None or abs(hf - frame) < abs(hand[best][0] - frame):
                best = j
        if best is None:
            missed.append((frame, a, b))
        else:
            used.add(best)
            matched.append((kind(a, b), hand[best][0] - frame))
    extra = [h for j, h in enumerate(hand) if j not in used]
    return matched, missed, extra


def label(code):
    return pm.LABELS[code]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()

    start, stop = win.frames(NAME)
    try:
        hand_full = pm.label_codes(expander.expand([NAME]))
    except ValueError as error:
        print("FAIL  {}".format(error))
        sys.exit(1)
    official_full = pm.label_codes(pm.read_label_csv(compare.PUBLIC_GT))

    # Both restricted to the window, X elsewhere, so the scorer grades exactly these frames.
    hand = np.full(len(hand_full), pm.X, dtype=np.int8)
    hand[start:stop] = hand_full[start:stop]
    official = np.full(len(official_full), pm.X, dtype=np.int8)
    official[start:stop] = official_full[start:stop]

    o, h = official[start:stop], hand[start:stop]

    # -- 1. frame agreement ---------------------------------------------------------------------
    print("frame agreement, public window [{}, {}), {} frames".format(start, stop, stop - start))
    print("  overall {:.1%}\n".format((o == h).mean()))
    print("  rows = official, cols = yours      H       A       D")
    for k in (pm.H, pm.A, pm.D):
        print("    {}                        {:>7} {:>7} {:>7}".format(
            label(k), *[int(((o == k) & (h == c)).sum()) for c in (pm.H, pm.A, pm.D)]))
    for k in (pm.H, pm.A, pm.D):
        print("  {}  official {:5.1%}   yours {:5.1%}".format(label(k), (o == k).mean(),
                                                            (h == k).mean()))

    # -- 2. human ceiling ---------------------------------------------------------------------
    r = pm.evaluate(official, hand)
    print("\nyour labels scored as a submission against the official ones (the human ceiling)")
    print("  macroF1 {:.4f}   spellF1 {:.4f}   possessions {} vs official {}   transF1@25 {:.4f}"
          .format(r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"],
                  r["transitions"][25]["f1"]))

    # -- 3. boundary bias ---------------------------------------------------------------------
    matched, missed, extra = match_boundaries(boundaries(official, start, stop),
                                              boundaries(hand, start, stop))
    print("\nboundary offsets, yours minus official, in frames (+ = you stamp later)")
    by = collections.defaultdict(list)
    for k, offset in matched:
        by[k].append(offset)
    for k in ("dead starts", "dead ends", "possession changes"):
        values = np.array(by.get(k, []))
        if not values.size:
            print("  {:<20} none matched".format(k))
            continue
        print("  {:<20} n={:<3} mean {:+6.1f}  median {:+6.1f}  ({:+.2f} s)   offsets {}".format(
            k, values.size, values.mean(), np.median(values), np.median(values) / FPS,
            sorted(values.tolist())))
    print("\n  official boundaries you have no match for (within {:.0f} s):".format(
        SEARCH_FRAMES / float(FPS)))
    for frame, a, b in missed:
        print("    {:>6}  {} -> {}".format(frame, label(a), label(b)))
    print("  your boundaries with no official match:")
    for frame, a, b in extra:
        print("    {:>6}  {} -> {}".format(frame, label(a), label(b)))

    # -- 4. does any submission's verdict depend on whose labels? -----------------------------
    print("\nevery submission against both labellings of these 5 minutes")
    print("  {:<20} {:>16} {:>16}".format("submission", "official", "yours"))
    rows = []
    for name, pred in compare.submissions():
        a = pm.evaluate(official, pred)
        b = pm.evaluate(hand, pred)
        rows.append((name, a, b))
    for name, a, b in sorted(rows, key=lambda t: -t[1]["spell_f1_mean"]):
        print("  {:<20} {:>7.3f} {:>3}/{:<3} {:>7.3f} {:>3}/{:<3}".format(
            name, a["spell_f1_mean"], a["n_pred_spells"], a["n_gt_spells"],
            b["spell_f1_mean"], b["n_pred_spells"], b["n_gt_spells"]))
    print("\n  spellF1  predicted/true possessions. If an experiment ranks well against yours and")
    print("  badly against the official labels, it has learned your convention, not theirs.")
