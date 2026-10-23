"""Shift the hand labels' dead-ball ends onto the organisers' convention.

    python validation/recalibrate.py --dry-run     # show every edit, change nothing
    python validation/recalibrate.py               # apply, keeping the originals
    python validation/recalibrate.py --revert      # put the originals back

What this corrects, and why
---------------------------
validation/calibrate.py compared a blind hand annotation of the public window with the
organisers' own labels for it. Dead-ball *starts* matched almost exactly -- a median offset of 0
frames -- but dead-ball *ends* were consistently early: offsets of -12, -15, -29 and -29 frames,
median -22, about 0.9 s.

That is a convention difference, not a mistake. A human ends a stoppage when play visibly
resumes: the taker picks the ball up and walks to the spot. The official labels come from
corrected tracking and end it when the ball is actually struck. Both are defensible; only one is
what the competition scores against.

Left uncorrected it bends the model. exp01's dead-ball layer was given a rule that ended
stoppages at the first sign of movement, because that scored better on these labels. Against the
official labels the same rule cost 16 points of dead-ball recall.

So every D-to-live boundary moves later by SHIFT_FRAMES. Starts of stoppages are not touched.

Limits worth knowing
--------------------
The offset is measured from four boundaries in one window. It is a median, applied uniformly, and
real boundaries vary (-12 to -29). It cannot fix the other difference calibrate.py found -- the
hand labels have 35% fewer possessions, because a human does not register a 0.8 s touch as a
possession -- and no mechanical shift can, since those possessions were never marked.

The originals are kept beside the corrected files, so this is reversible and the raw annotation
is never lost.
"""

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import windows as win  # noqa: E402
import expand as expander  # noqa: E402

# Params
# Median of the four dead-ball-end offsets measured by calibrate.py on the public window
# (-29, -29, -15, -12). Positive because the correction moves the boundary later.
SHIFT_FRAMES = 22
SUFFIX = ".asannotated.csv"
TARGETS = ("win1", "win2", "win3")  # not "public": that window has an official answer already


def original_path(name):
    return expander.label_path(name).replace(".csv", SUFFIX)


def shift(name, frames=SHIFT_FRAMES):
    """[(frame, label, moved_from_or_None)] with every D-to-live boundary moved later."""
    rows = expander.read_boundaries(name)
    start, stop = win.frames(name)
    out = []
    for index, (frame, label) in enumerate(rows):
        previous = rows[index - 1][1] if index else None
        if previous == "D" and label != "D":
            # The restart. Move it later, but never past the next boundary or the window's end.
            ceiling = rows[index + 1][0] if index + 1 < len(rows) else stop
            moved = min(frame + frames, ceiling - 1)
            out.append((moved, label, frame))
        else:
            out.append((frame, label, None))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--frames", type=int, default=SHIFT_FRAMES)
    args = parser.parse_args()

    if args.revert:
        for name in TARGETS:
            backup = original_path(name)
            if os.path.isfile(backup):
                shutil.copyfile(backup, expander.label_path(name))
                print("restored {}".format(os.path.relpath(expander.label_path(name), ROOT)))
            else:
                print("no backup for {}; nothing to restore".format(name))
        raise SystemExit(0)

    total = 0
    for name in TARGETS:
        rows = shift(name, args.frames)
        moved = [(f, l, o) for f, l, o in rows if o is not None]
        total += len(moved)
        print("{}: {} dead-ball ends moved +{} frames".format(name, len(moved), args.frames))
        for frame, label, origin in moved:
            print("    {:>7} -> {:>7}  D -> {}{}".format(
                origin, frame, label, "   (clamped)" if frame - origin != args.frames else ""))

        if args.dry_run:
            continue
        backup = original_path(name)
        if not os.path.isfile(backup):
            shutil.copyfile(expander.label_path(name), backup)
        with open(expander.label_path(name), "w") as handle:
            handle.write("frame,label\n")
            handle.writelines("{},{}\n".format(f, l) for f, l, _ in rows)

    if args.dry_run:
        print("\n{} boundaries would move. Nothing was changed.".format(total))
    else:
        print("\n{} boundaries moved. Originals kept as *{}".format(total, SUFFIX))
        print("Rebuild and re-score:")
        print("  python validation/expand.py")
        print("  python validation/compare.py --bootstrap 1000")
