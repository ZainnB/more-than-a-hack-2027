"""Decide which frames are dead balls, from pitch geometry first and Smart Tagging second.

    python experiments/exp01_geometry_hysteresis/deadball.py        # measure it on the public GT

What this replaces
------------------
starter/baseline.py has exactly one dead-ball rule: the tracker lost the ball, therefore D. That
rule is 55.8% precise on the public window, because the tracker loses the ball constantly during
open play -- it detects occlusion, not the referee. It is the baseline's weakest class, at D
recall 0.214 for baseline_smooth50.

The rules here, in the order they are applied and in descending order of how much they can be
trusted:

  1. OUT OF BOUNDS. The ball is tracked outside the unit square. 98.1% of such frames are dead.
     Play is then dead until a restart, which is *detected* -- the ball is back inside the pitch
     and moving -- rather than assumed from a fixed duration.

  2. BALL LOST AND THE PITCH IS WALKING. The ball untracked for 3 seconds *and* the mean player
     speed low. Both conditions matter: the first alone is the baseline's rule, and the second is
     what distinguishes a stoppage from a crowded box. Deliberately the weakest rule, and it
     fires last.

What it does not detect, and why that is a choice
-------------------------------------------------
FOULS. A foul stops play without the ball leaving the pitch, and nothing in the ATD marks the
whistle. Two of the four dead stretches in the public window (5.5 s and 20.4 s) have no
out-of-bounds frame anywhere near them, and both are almost certainly fouls.

The obvious candidate signal was measured and rejected: a duel -- players from both teams within
3 m of the ball -- is dead on 1.6% of frames against a 19.6% base rate. A duel is contested
*live* play, which is the opposite of what was wanted. Dilating it forward 200 frames still gives
17.5%, below base rate.

So fouls are left undetected rather than guessed at. On the ranking metric that is the cheaper
error: spellF1 never sees an abstention, and a wrong D inside a real possession splits it into two
and costs a match outright.
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
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

try:  # as a package, so exp01's own config/features win over any other experiment's
    from . import config, features as feat
except ImportError:  # run directly as a script
    import config
    import features as feat
import load_smart_tagging as tagging  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402


def _runs_of(mask):
    """[(start, stop)] half-open spans where a boolean array is True."""
    flagged = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flagged, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def out_of_bounds(features, margin=None):
    """True where the tracked ball is outside the pitch.

    Only where the ball is actually seen: a NaN comparison is False, but saying so explicitly is
    what stops "we lost the ball" from being read as "the ball left the pitch". They are
    different claims and the first one is far weaker.
    """
    margin = config.OUT_OF_BOUNDS_MARGIN if margin is None else margin
    seen = feat.ball_seen(features)
    x, y = features["ball_x"], features["ball_y"]
    return seen & ((x < -margin) | (x > 1.0 + margin) | (y < -margin) | (y > 1.0 + margin))


def find_restart(features, start, limit, speed=None):
    """(frame, how) for the restart at or after `start`, or (None, "none").

    A restart is the ball back inside the pitch, *settled*, and *then struck*. All three parts are
    needed, and the middle one is the subtle one: after the ball goes out it is retrieved and
    thrown or rolled back into play, which puts a fast-moving ball inside the pitch while play is
    still dead. At the goal kick at frame 18389 that return throw reaches 6.5 m/s, which clears
    the strike threshold on its own and ends the dead stretch 262 frames early. A ball that was
    slow for a second beforehand is on the spot; a ball that was already moving is in transit.

    Two outcomes, distinguished because they deserve different trust:

      "struck"   a settled ball was struck. This is the real restart.
      "reentry"  no such strike inside the bound, so the frame the ball came back onto the pitch
                 is returned instead. A weaker answer, and deliberately the conservative one --
                 it ends the dead stretch early, which costs recall rather than precision.

    Searching for the restart rather than adding a fixed duration is the point of the layer.
    Stoppages in the public window run from 5.5 s to 20.4 s, so any constant is wrong by ten
    seconds somewhere.
    """
    speed = feat.ball_speed_mps(features) if speed is None else speed
    inside = feat.ball_inside(features)
    stop = min(len(inside), limit)

    reentry = None
    for frame in range(start, stop):
        if not inside[frame]:
            continue
        if reentry is None:
            reentry = frame
        if speed[frame] >= config.BALL_RESTART_SPEED_MPS and _was_settled(speed, frame):
            return _snap_to_strike(inside, speed, frame, start), "struck"
    return (reentry, "reentry") if reentry is not None else (None, "none")


def _snap_to_strike(inside, speed, frame, floor):
    """Walk back to the first frame of the strike, when the ball's track begins mid-flight.

    A ball's speed is differenced from the previous frame, so the first frame of a tracked run has
    no speed at all and the strike is detected one frame late. At both kickoffs the ball is
    untracked until it is already travelling, which puts the detected restart 3 frames after the
    real one -- cheap on spellF1, but free to fix and the same correction applies wherever the
    tracker picks the ball up in flight.

    Only frames that are themselves part of the strike are absorbed: still moving at full speed,
    or unmeasurable because the track starts there. A ball that was merely rolling slowly before
    this is the ball being returned to the spot, and walking back through it is the 262-frame
    error that config.BALL_SETTLED_SPEED_MPS exists to prevent.
    """
    while frame - 1 >= floor and inside[frame - 1] and (
            np.isnan(speed[frame - 1]) or speed[frame - 1] >= config.BALL_RESTART_SPEED_MPS):
        frame -= 1
    return frame


def _was_settled(speed, frame):
    """Was the ball stationary in the second before `frame`?

    Untracked counts as settled. A ball the tracker cannot see is not a ball visibly travelling
    across the pitch, and the kickoff of each half is exactly this case -- the ball is NaN until
    the frame it is struck, so demanding a tracked slow approach would reject every kickoff.
    """
    window = speed[max(0, frame - config.BALL_SETTLED_FRAMES):frame]
    if not window.size:
        return True
    known = window[~np.isnan(window)]
    return not known.size or float(np.median(known)) < config.BALL_SETTLED_SPEED_MPS


def find_stoppage_onset(features, restart, limit, speed=None):
    """First dead frame of the stoppage that ends at `restart`, walking backwards.

    The onset is just after the last struck ball: whatever action stopped play -- the shot that
    went in, the pass that went out, the foul -- is the last time the ball moved at speed. Walking
    back to it means a stoppage's length is measured rather than assumed, which matters most where
    the assumption would be worst: a goal celebration and a quick throw-in differ by a minute.

    `limit` is a floor on how far back to look and only bounds the search. For the second-half
    kickoff there is no struck ball to find -- the ball is untracked through halftime -- so the
    walk runs to the bound, which is what correctly covers the 2934 graded dead frames before that
    kickoff that FAQ.md documents.
    """
    speed = feat.ball_speed_mps(features) if speed is None else speed
    inside = feat.ball_inside(features)
    floor = max(0, limit)
    for frame in range(restart - 1, floor - 1, -1):
        if inside[frame] and speed[frame] >= config.BALL_RESTART_SPEED_MPS:
            return frame + 1
    return floor


def from_out_of_bounds(features):
    """Dead stretches that begin with the ball leaving the pitch. The primary rule."""
    oob = out_of_bounds(features)
    speed = feat.ball_speed_mps(features)
    dead = np.zeros(len(oob), dtype=bool)

    restarts = []
    for start, stop in _runs_of(oob):
        # The stretch runs from the ball leaving until play restarts. stop is where the ball came
        # back inside the pitch, which is not the same thing as play restarting -- somebody still
        # has to take the throw -- so the search for the restart starts there.
        limit = start + config.MAX_DEAD_FRAMES
        restart, how = find_restart(features, stop, limit, speed)
        end = restart if restart is not None else limit
        end = max(end, start + config.MIN_DEAD_FRAMES)
        dead[start:min(end, len(dead))] = True
        if restart is not None and how == "struck":
            restarts.append(restart)
    return dead, restarts


def from_slow_pitch(features):
    """Dead stretches where all 22 players are walking at once. The only foul detector here.

    This is the rule that catches a stoppage with the ball still on the pitch, which is what a
    foul is. Both of the public window's undetected dead stretches are visible in it and in
    nothing else: over the 20.4 s stretch at frame 15889 the ball is tracked on 100% of frames and
    never leaves the pitch, so no geometric rule can see it, but the mean player speed there is
    1.13 m/s against a match median of 1.52.

    The pitch speed is smoothed first and then required to stay low for several seconds. Both
    steps matter: without the smoothing a single mistracked player drags the mean down, and
    without the duration the rule fires on every lull in open play.
    """
    mean_speed = features["mean_speed"]
    # Gaps are filled with the match median rather than zero. A NaN run filled with zero would
    # read as a motionless pitch and this rule would call it dead on no evidence at all.
    filled = np.nan_to_num(mean_speed, nan=float(np.nanmedian(mean_speed)))
    window = np.ones(config.SLOW_PITCH_WINDOW) / config.SLOW_PITCH_WINDOW
    smoothed = np.convolve(filled, window, mode="same")

    dead = np.zeros(len(mean_speed), dtype=bool)
    for start, stop in _runs_of(smoothed < config.SLOW_PITCH_SPEED):
        if stop - start >= config.SLOW_PITCH_MIN_FRAMES:
            dead[start:stop] = True
    return dead


def from_ball_lost(features):
    """Dead stretches where the ball is gone and nobody is running. The weakest rule, applied last.

    The baseline's version of this is unconditional and 55.8% precise. The two conditions here are
    what lift it: a sustained loss, so a one-second occlusion in a crowded box does not qualify,
    and a slow pitch, because players walk during a stoppage and at no other time do all 22 slow
    down together.
    """
    seen = feat.ball_seen(features)
    mean_speed = features["mean_speed"]
    dead = np.zeros(len(seen), dtype=bool)

    for start, stop in _runs_of(~seen):
        if stop - start < config.BALL_LOST_DEAD_FRAMES:
            continue
        window = mean_speed[start:stop]
        if not window.size or np.all(np.isnan(window)):
            continue
        # The mean over the stretch, not a per-frame test: one slow frame is noise, and what the
        # rule is actually looking for is a pitch that has stopped running for seconds at a time.
        if np.nanmean(window) < config.SLOW_PITCH_SPEED:
            dead[start:stop] = True
    return dead


ALL_RULES = ("oob", "slow", "lost")


def dead_mask(features, game=GAME, data_dir=None, events=None, rules=ALL_RULES):
    """Union of the enabled dead-ball rules, as a boolean array over video frames.

    `rules` is a tuple so each rule can be switched off and measured on its own -- which is how
    the numbers in this module's docstring were produced, and how the next one should be.
    """
    parts = {}
    restarts = []
    if "oob" in rules:
        parts["oob"], found = from_out_of_bounds(features)
        restarts.extend(found)
    if "slow" in rules:
        parts["slow"] = from_slow_pitch(features)
    if "lost" in rules:
        parts["lost"] = from_ball_lost(features)

    total = np.zeros(len(features["ball_x"]), dtype=bool)
    for mask in parts.values():
        total |= mask

    # A detected restart is positive evidence that play is alive, and it outranks every rule
    # above. It is applied last and as a veto for one specific failure: the players stand still in
    # formation waiting for a kickoff, so from_slow_pitch reads the pitch as walking at the exact
    # moment the half begins, and the first second of each half comes out dead.
    for restart in sorted(set(restarts)):
        total[restart:restart + config.RESTART_ALIVE_FRAMES] = False
    return total, parts, sorted(set(restarts))


if __name__ == '__main__':
    import possession_metrics as pm

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=os.path.join(
        ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    args = parser.parse_args()

    features = feat.load()
    total, parts, restarts = dead_mask(features)

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    runs = pm.valid_runs(gt)
    graded = np.zeros(len(gt), dtype=bool)
    for start, stop in runs:
        graded[start:stop] = True
    truth_dead = graded & (gt == pm.D)

    print("dead-ball rules, measured on {}".format(os.path.basename(args.gt)))
    print("  graded frames {}   of which dead {} ({:.1f}%)\n".format(
        int(graded.sum()), int(truth_dead.sum()),
        100.0 * truth_dead.sum() / max(1, graded.sum())))
    print("  {:<28} {:>7} {:>10} {:>9}".format("rule", "frames", "precision", "recall"))

    for name in ALL_RULES:
        if name not in parts:
            continue
        fired = parts[name] & graded
        hit = fired & truth_dead
        print("  {:<28} {:>7} {:>9.1f}% {:>8.1f}%".format(
            name, int(fired.sum()),
            100.0 * hit.sum() / max(1, fired.sum()),
            100.0 * hit.sum() / max(1, truth_dead.sum())))

    fired = total & graded
    hit = fired & truth_dead
    print("  {:<28} {:>7} {:>9.1f}% {:>8.1f}%".format(
        "ALL (union)", int(fired.sum()),
        100.0 * hit.sum() / max(1, fired.sum()),
        100.0 * hit.sum() / max(1, truth_dead.sum())))
    print("\n  baseline_smooth50 for comparison:  precision 65.7%  recall 21.4%")

    print("\n  dead stretches found in the graded window:")
    for start, stop in _runs_of(total & graded):
        overlap = truth_dead[start:stop].mean()
        print("    {:>6} - {:>6}  {:5.1f} s   {:5.1f}% truly dead".format(
            start, stop, (stop - start) / 25.0, 100.0 * overlap))
    print("\n  true dead stretches, for comparison:")
    for start, stop in _runs_of(truth_dead):
        found = total[start:stop].mean()
        print("    {:>6} - {:>6}  {:5.1f} s   {:5.1f}% detected".format(
            start, stop, (stop - start) / 25.0, 100.0 * found))
