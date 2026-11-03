"""Move each possession boundary onto the event that caused it.

    python experiments/exp04_boundary_timing/refine.py --report

Why timing and not counting
---------------------------
exp03's errors are mostly not invented possessions. Of its spurious spells, 60-80% overlap a
real possession of the same team and simply miss the IoU 0.5 bar -- so one badly-placed boundary
is charged twice, once as a miss and once as a spurious. Suppressing spurious spells by
tightening a threshold would delete the overlapping ones too; moving them lifts both counts at
once.

And the problem is spread, not lag. Matched against the hand labels, exp03's transitions sit at a
median of **-4 frames** -- essentially unbiased -- with a 10th-to-90th percentile range of
**-72 to +33 frames**. There is nothing to correct with a constant offset. Only 32 of 79 true
transitions have a prediction within 4 s at all.

What decides a boundary now
---------------------------
exp03 dates a change of possession to the first frame of the evidence streak that triggered it:
the first frame the opponent was nearer by MARGIN_M. That is a threshold crossing on a noisy
distance signal, which is exactly the kind of quantity that lands anywhere within a couple of
seconds of the real event.

Two better-justified anchors, both already computed elsewhere in the pipeline:

  TOUCH     the nearest detected touch by the incoming team. A possession begins when somebody
            touches the ball, and touch.py finds those from ball acceleration and direction
            change -- 2000 across the match, the right rate for football. The touch path was
            rejected as a *trigger* in exp02 because it also fires on presses and deflections
            that never take possession; as a *timing* anchor it carries none of that risk,
            because the decision to switch has already been made by other means.

  CROSSING  the frame where the incoming team actually became the nearer one, rather than the
            frame where it became nearer by a full MARGIN_M. Always at or before the trigger,
            and a weaker claim than a touch, so it is only used where no touch is found.

Neither invents or removes a possession. Both only move a boundary that exp03 had already
decided on, which is why this can be measured cleanly against it.
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

HOME, AWAY, NOBODY = 0, 1, -1

# Params
# How far either side of a boundary to look for a better anchor. 50 frames is 2 s, which covers
# the bulk of the measured spread (10th-90th percentile -72..+33) without letting a boundary
# wander into the neighbouring possession.
SNAP_WINDOW = 50

# A boundary is never moved closer than this to the one before or after it, so refinement cannot
# collapse a short possession to nothing.
MIN_GAP = 5


def boundaries(codes, dead):
    """[(index, frame, from_team, to_team)] for every live H<->A change, in order.

    Changes into or out of a dead ball are left alone: those are the dead-ball layer's business
    and it already dates them from the ball leaving the pitch or being struck.
    """
    out = []
    previous = NOBODY
    previous_frame = None
    for frame in range(len(codes)):
        if dead[frame]:
            previous = NOBODY
            continue
        current = codes[frame]
        if current == NOBODY:
            continue
        if previous in (HOME, AWAY) and current != previous:
            out.append((len(out), frame, previous, current))
        previous, previous_frame = current, frame
    return out


def snap(codes, dead, touches, touch_team, nearest, has_answer,
         mode="touch", window=None, min_gap=MIN_GAP):
    """Codes with every live possession boundary moved onto its best anchor.

    mode: "touch" prefers a touch and falls back to the crossing, "crossing" uses only the
    crossing, "none" returns the input unchanged. Returns (codes, moves) where moves is a list
    of (old_frame, new_frame, anchor) for reporting.
    """
    if mode == "none":
        return codes.copy(), []
    window = SNAP_WINDOW if window is None else window

    out = codes.copy()
    marks = boundaries(codes, dead)
    moves = []
    previous_new = None

    for position, (_i, frame, _old_team, new_team) in enumerate(marks):
        low = frame - window
        high = frame + window
        # Never cross a neighbouring boundary, and never cross a dead stretch.
        if previous_new is not None:
            low = max(low, previous_new + min_gap)
        if position + 1 < len(marks):
            high = min(high, marks[position + 1][1] - min_gap)
        low = max(low, 0)
        high = min(high, len(out) - 1)
        if high <= low:
            previous_new = frame
            continue
        span = np.arange(low, high + 1)
        span = span[~dead[low:high + 1]]
        if not span.size:
            previous_new = frame
            continue

        target, anchor = None, None

        if mode in ("touch", "both"):
            candidates = span[touches[span] & (touch_team[span] == new_team)]
            if candidates.size:
                target = int(candidates[np.argmin(np.abs(candidates - frame))])
                anchor = "touch"

        if target is None and mode in ("touch", "crossing", "both"):
            # The frame the incoming team actually became nearest, walking back from the trigger.
            usable = has_answer[span] & (nearest[span] == new_team)
            if usable.any():
                # earliest frame of the unbroken run of "new team is nearest" containing frame
                run = span[usable]
                at_or_before = run[run <= frame]
                if at_or_before.size:
                    start = at_or_before[-1]
                    while start - 1 >= low and has_answer[start - 1] \
                            and nearest[start - 1] == new_team and not dead[start - 1]:
                        start -= 1
                    target, anchor = int(start), "crossing"

        if target is not None and target != frame:
            lo, hi = (target, frame) if target < frame else (frame, target)
            out[lo:hi] = np.where(dead[lo:hi], NOBODY,
                                  new_team if target < frame else _old_or(codes, lo))
            moves.append((frame, target, anchor))
            previous_new = target
        else:
            previous_new = frame

    return out, moves


def _old_or(codes, lo):
    """The label in force just before `lo`, used when a boundary moves later."""
    return codes[lo - 1] if lo > 0 else codes[lo]


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    sys.path.insert(0, os.path.join(ROOT, "validation"))
    import possession_metrics as pm  # noqa: E402
    import compare  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    try:
        from . import run as run04
    except ImportError:
        import run as run04

    for mode in ("none", "crossing", "touch"):
        labels, detail = run04.build_labels(mode=mode)
        moves = detail["moves"]
        pred = pm.label_codes(labels)
        r = compare.summarise(compare.ground_truths()["all20"], pred)
        anchors = {}
        for _old, _new, anchor in moves:
            anchors[anchor] = anchors.get(anchor, 0) + 1
        shifts = [abs(n - o) for o, n, _a in moves]
        print("mode={:<9} moved {:>3} boundaries {:<22} median shift {:>4} fr   "
              "spellF1 {:.4f}  macroF1 {:.4f}".format(
                  mode, len(moves), str(anchors), int(np.median(shifts)) if shifts else 0,
                  r["spellF1"], r["macroF1"]))
