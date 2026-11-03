"""Lists of moments to go and watch, with video and match timestamps.

    python validation/towatch.py                 # both lists
    python validation/towatch.py --missed        # only: official possessions our labels merged
    python validation/towatch.py --nearmiss      # only: the model's near-miss possessions
    python validation/towatch.py --out notes.txt

Two different questions, two different lists.

**--missed** compares our blind annotation of the public window with the organisers' own labels
for it. Every official possession we failed to reproduce is a case where their convention counts
something ours did not, and watching them is the fastest way to learn the difference. This list
can only ever cover the public 5 minutes, because that is the only window with official labels.

**--nearmiss** is for the rest of the match, where there is no ground truth to be missing from.
It lists the places the model *almost* changed possession -- a run of evidence for the other team
that fell short of the configured threshold -- which are the most likely places a short
possession is being dropped. They are candidates, not errors: watching them is how to find out
which.

Timestamps are given twice. **video** is frame / 25, which is what validation/annotate.py shows
and what to seek to. **match** is the burnt-in clock, counting from each kickoff (frame 11185 and
frame 113742), which is what the broadcast shows.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.join(ROOT, "experiments")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import windows as win  # noqa: E402
import expand as expander  # noqa: E402
import compare  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
KICKOFFS = {1: 11185, 2: 113742}  # published in DATA.md and FAQ.md
NEAR_MISS_MIN = 10  # a run shorter than this is not evidence of anything


def video_time(frame, fps=25):
    return "{:02d}:{:05.2f}".format(int(frame / fps) // 60, frame / float(fps) % 60)


def match_time(frame, fps=25):
    """The burnt-in clock: 0:00 at each kickoff, second half counting from 45:00."""
    half = 1 if frame < KICKOFFS[2] else 2
    seconds = (frame - KICKOFFS[half]) / float(fps) + (0 if half == 1 else 45 * 60)
    if seconds < 0:
        return "pre-ko"
    return "{:02d}:{:04.1f}".format(int(seconds) // 60, seconds % 60)


def missed_possessions():
    """[(start, stop, official_label, what_we_said)] official possessions our labels do not have."""
    start, stop = win.frames("public")
    official = pm.label_codes(pm.read_label_csv(compare.PUBLIC_GT))
    hand = pm.label_codes(expander.expand(["public"]))

    def spells(codes):
        window = np.full(len(codes), pm.X, dtype=np.int8)
        window[start:stop] = codes[start:stop]
        return pm.possession_spells(window, [(start, stop)])

    ours = spells(hand)
    out = []
    for a, b, k in spells(official):
        overlap = max([min(b, y) - max(a, x) for x, y, _t in ours
                       if min(b, y) > max(a, x)] or [0])
        said = sorted({pm.LABELS[c] for c in hand[a:b]})
        if overlap / float(b - a) < 0.5 or pm.LABELS[k] not in said:
            out.append((a, b, pm.LABELS[k], "/".join(said)))
    return out, len(spells(official)), len(ours)


def near_misses(game=GAME):
    """[(start, frames, team)] where evidence for the other team fell short of the threshold."""
    from exp03_keeper_fix import features as feat3, run as run3
    from exp02_touch_gated import features as feat2, team as team2, config as config2
    from exp01_geometry_hysteresis import deadball

    features = feat3.load()
    kin = feat2.kinematics(features, game)
    dead, _parts, _restarts = deadball.dead_mask(features, game)
    evidence, _nearest, _has = team2.evidence(features, kin, np.zeros(len(dead), dtype=bool))
    codes = pm.label_codes(run3.build_labels(game)[0])

    out = []
    for _half, first, last in half_video_blocks(game):
        run, start = 0, None
        for frame in range(first, last + 1):
            if dead[frame] or codes[frame] not in (pm.H, pm.A):
                run = 0
                continue
            held = team2.HOME if codes[frame] == pm.H else team2.AWAY
            opponent = team2.AWAY if held == team2.HOME else team2.HOME
            if evidence[frame] == opponent:
                if run == 0:
                    start = frame
                run += 1
            else:
                if NEAR_MISS_MIN <= run < config2.K_FALLBACK_FRAMES:
                    out.append((start, run, "A" if opponent == team2.AWAY else "H"))
                run = 0
    return sorted(out, key=lambda r: -r[1]), config2.K_FALLBACK_FRAMES


def render(show_missed=True, show_nearmiss=True, limit=None):
    lines = []
    if show_missed:
        rows, n_official, n_ours = missed_possessions()
        lines.append("OFFICIAL POSSESSIONS OUR ANNOTATION DOES NOT REPRODUCE")
        lines.append("public window only -- the one place the organisers' labels exist")
        lines.append("their {} possessions against our {}\n".format(n_official, n_ours))
        lines.append("  {:<10} {:<9} {:>6}  {:<9} {}".format(
            "video", "match", "dur", "official", "we said"))
        for a, b, label, said in rows:
            lines.append("  {:<10} {:<9} {:>5.1f}s  {:<9} {}".format(
                video_time(a), match_time(a), (b - a) / 25.0, label, said))
        lines.append("\n  {} to watch. These are the convention gap, frame-accurate.".format(
            len(rows)))

    if show_nearmiss:
        rows, threshold = near_misses()
        if show_missed:
            lines.append("\n" + "-" * 72 + "\n")
        lines.append("NEAR-MISS POSSESSIONS -- candidates, not known errors")
        lines.append("places the model saw {}-{} frames of evidence for the other team and".format(
            NEAR_MISS_MIN, threshold - 1))
        lines.append("needed {} to act. the most likely dropped short possessions.\n".format(
            threshold))
        lines.append("  {:<10} {:<9} {:>6}  {}".format("video", "match", "frames", "would be"))
        for start, run, team in (rows[:limit] if limit else rows):
            lines.append("  {:<10} {:<9} {:>6}  {}".format(
                video_time(start), match_time(start), run, team))
        lines.append("\n  {} in the whole match{}.".format(
            len(rows), ", showing the {} longest".format(limit) if limit else ""))
        lines.append("  Note that is far short of the ~118 possessions needed to reach the")
        lines.append("  official convention's ~360, so the frame threshold is not the main")
        lines.append("  constraint -- the 2 m evidence margin is. See the README.")
    return "\n".join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--missed", action="store_true")
    parser.add_argument("--nearmiss", action="store_true")
    parser.add_argument("--limit", type=int, help="show only the N longest near-misses")
    parser.add_argument("--out", help="write to this file as well as printing")
    args = parser.parse_args()

    both = not (args.missed or args.nearmiss)
    text = render(show_missed=both or args.missed,
                  show_nearmiss=both or args.nearmiss,
                  limit=args.limit)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print("\nwrote {}".format(os.path.relpath(args.out, ROOT)))
