"""Find the frames where somebody touched the ball.

    python experiments/exp02_touch_gated/touch.py --report
    python experiments/exp02_touch_gated/touch.py --sweep

Why this module exists
----------------------
It is the single highest-value idea in the research note. Link & Hoernig (PLOS ONE 2017) measured
naive proximity at precision 0.68; requiring a change of possession to coincide with a *kick*
raised it to 0.86, and adding ball prediction to 0.91. Timeline agreement went 0.82 -> 0.87 ->
0.92.

exp01 has no notion of a touch. It concedes possession when the opposing team has been nearer for
25 consecutive frames, which both lags the real moment by up to a second and fires on balls that
merely rolled past somebody. Its transition score shows it: transF1 0.35 at +/- 1 s, with 13
missed and 13 spurious against 20 real transitions.

A touch here is a frame where the ball's trajectory bends sharply, near a player, while the ball
is actually moving.

The published thresholds survive, but not for the obvious reason
---------------------------------------------------------------
Link & Hoernig's kick threshold is an acceleration local maximum of 4 m/s^2 and Vidal-Codina's
direction-change threshold is 10 degrees, both measured on multi-camera optical tracking. On this
data the *median* ball acceleration is 6.7 m/s^2, so taken alone a 4 m/s^2 threshold is true on
64% of all frames and detects nothing.

That is not noise, and smoothing does not remove it: widening the Savitzky-Golay window from 7
frames to 31 moves the median only from 6.7 to 6.1. The cause is that this ball is tracked in 2D
pitch coordinates, so a ball in the air is projected onto the pitch plane through the homography
and its height appears as horizontal motion. The projected path genuinely curves and accelerates
hard. The 4 m/s^2 figure describes a ball located in space; ours is located on a plane.

What makes the threshold work anyway is that a touch is not "acceleration is high" but "accel-
eration *peaks*" -- a local maximum -- and that one kick may only produce one touch. With the
local-maximum test and a 10-frame refractory window, the published values give **2006 touches over
the graded match, one every 3.2 seconds**, which is squarely inside the 1500-2500 a real match
contains. `--sweep` shows the whole surface; the count is flat across a wide band of thresholds,
which is the reassuring part.

Measured limits of the bend detector
------------------------------------
`--report` runs a check that needs no labels: every restart exp01's dead-ball layer found is a
struck ball, and that layer found them from pitch geometry with no reference to acceleration. Two
independent routes to the same event.

The bend detector only agrees with **18% of them within 3 frames, and 62% within 25**. Measured
causes, within +/- 3 frames of the 55 graded restarts:

    acceleration over threshold     60%
    ball displaced >= 10 cm         55%   <- the binding constraint
    a tracked player within 2 m     75%
    heading change over threshold   13%

The displacement test fails because the ball's track frequently *begins* at the restart, and the
Savitzky-Golay window is undefined for the first three frames of a run, so the ball has no
measurable displacement at the exact moment it is struck. The proximity test fails because the
player taking a throw-in stands off the pitch, where the tracker often does not see them at all.

Neither is worth contorting this detector to fix, because the restarts are already known by a
better route. `detect()` therefore takes an `extra` argument, and the state machine unions in the
dead-ball layer's restarts -- 97.5% precise, independently derived -- so a touch exists at every
restart by construction. This detector's job is the touches *during* open play, where the ball is
tracked continuously and both players are on the pitch, and that is what it is good at.
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

from games import GAME, GAMES, half_video_blocks  # noqa: E402

try:
    from . import config, features as feat
except ImportError:
    import config
    import features as feat

HOME, AWAY = 0, 1


def local_maxima(values, threshold):
    """True where a value is at or above `threshold` and no lower than either neighbour.

    A plateau counts at its first frame, because `>=` on the left and `>` on the right makes the
    test asymmetric on purpose: a flat-topped acceleration peak is one touch, not four.
    """
    out = np.zeros(len(values), dtype=bool)
    middle = values[1:-1]
    out[1:-1] = (middle >= threshold) & (middle >= values[:-2]) & (middle > values[2:])
    return out


def nearest_player(features):
    """(team, distance) of the player closest to the ball on each frame.

    Reuses exp01's team_margin, which already resolves the two per-team distances and the
    one-team-tracked case.
    """
    nearest, _margin, has_answer = feat.team_margin(features)
    home, away = features["d_home"], features["d_away"]
    distance = np.where(nearest == HOME,
                        np.where(np.isnan(home), np.inf, home),
                        np.where(np.isnan(away), np.inf, away))
    distance = np.where(has_answer, distance, np.inf)
    return nearest, distance


def detect(features, kinematics=None, accel_mps2=None, heading_deg=None, extra=()):
    """Boolean mask of touch frames, plus the team credited with each.

    `extra` is a list of frames that are known to be touches by some other route, and in practice
    it is the dead-ball layer's restarts. They bypass every test below, because they were derived
    from pitch geometry at 97.5% precision and this detector measurably cannot see most of them --
    see the module docstring for why. Possession changes hands at restarts more reliably than
    anywhere else in a match, so a touch has to exist there.

    Four conditions, all required:

      1. the trajectory bends -- an acceleration local maximum, or a heading change above
         threshold. Either alone is enough; they catch different things. A pass struck from a
         standing ball is an acceleration spike with little direction change, while a ball
         deflected at speed changes direction with little change of pace.
      2. the ball actually moved over the measurement window. The heading of a stationary ball is
         undefined, and its jitter will produce any direction change you ask for.
      3. somebody is close enough to have done it.
      4. no other touch in the preceding refractory window, so one kick is one touch rather than
         a burst across every frame its acceleration stayed high.
    """
    kin = feat.kinematics(features) if kinematics is None else kinematics
    accel_mps2 = config.TOUCH_ACCEL_MPS2 if accel_mps2 is None else accel_mps2
    heading_deg = config.TOUCH_HEADING_DEG if heading_deg is None else heading_deg

    bends = local_maxima(np.nan_to_num(kin["accel"], nan=0.0), accel_mps2)
    turns = local_maxima(np.nan_to_num(kin["heading_change"], nan=0.0), heading_deg)
    moved = np.nan_to_num(kin["displacement"], nan=0.0) >= config.TOUCH_MIN_DISP_M

    team, distance = nearest_player(features)
    close = distance <= config.TOUCH_MAX_PLAYER_DIST_M

    candidates = (bends | turns) & moved & close

    # Refractory thinning, in one pass so the gap is measured against the touch that was kept and
    # not against the candidate that was dropped.
    touches = np.zeros(len(candidates), dtype=bool)
    last = -config.TOUCH_REFRACTORY_FRAMES - 1
    for frame in np.flatnonzero(candidates):
        if frame - last >= config.TOUCH_REFRACTORY_FRAMES:
            touches[frame] = True
            last = frame

    # Known touches are added after thinning, never dropped by it: a restart is not a candidate
    # competing with its neighbours, it is a fact about the match.
    for frame in extra:
        if 0 <= frame < len(touches):
            touches[frame] = True
    return touches, team


def contested(features):
    """True where both teams have a player inside the duel zone.

    Vidal-Codina's rule is that a duel prevails over possession, and CHALLENGE.md says the same
    thing from the other side: while the ball is contested, possession stays with whoever already
    had it. A touch on one of these frames is not allowed to hand the ball over.
    """
    radius = str(int(config.R_DZ_M)) if config.R_DZ_M == int(config.R_DZ_M) else None
    home_key, away_key = "n_home_" + (radius or "3"), "n_away_" + (radius or "3")
    if home_key not in features:  # radius not among the cached counts; fall back to distances
        return (np.nan_to_num(features["d_home"], nan=np.inf) <= config.R_DZ_M) \
            & (np.nan_to_num(features["d_away"], nan=np.inf) <= config.R_DZ_M)
    return (features[home_key] > 0) & (features[away_key] > 0)


def graded_mask(n_frames, game=GAME):
    mask = np.zeros(n_frames, dtype=bool)
    for _half, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


if __name__ == '__main__':
    from exp01_geometry_hysteresis import deadball

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    features = feat.load()
    kin = feat.kinematics(features)
    graded = graded_mask(len(features["ball_x"]))
    fps = GAMES[GAME]["fps"]

    if args.sweep:
        print("touch count against threshold, over {} graded frames".format(int(graded.sum())))
        print("a match has roughly 1500-2500 touches: one every {:.0f}-{:.0f} frames\n".format(
            graded.sum() / 2500.0, graded.sum() / 1500.0))
        print("  {:>8} {:>8} {:>10} {:>14}".format("accel", "heading", "touches", "1 per N fr"))
        for accel in (4.0, 10.0, 20.0, 30.0, 50.0, 80.0):
            for heading in (10.0, 30.0, 60.0, 90.0):
                touches, _team = detect(features, kin, accel, heading)
                count = int((touches & graded).sum())
                flag = "  <- in range" if 1500 <= count <= 2500 else ""
                print("  {:>8.0f} {:>8.0f} {:>10} {:>14.0f}{}".format(
                    accel, heading, count, graded.sum() / max(count, 1), flag))
        raise SystemExit(0)

    touches, team = detect(features, kin)
    count = int((touches & graded).sum())
    print("touch detection  accel >= {:.0f} m/s^2, heading >= {:.0f} deg".format(
        config.TOUCH_ACCEL_MPS2, config.TOUCH_HEADING_DEG))
    print("  {} touches over {} graded frames -- one every {:.0f} frames ({:.1f} s)".format(
        count, int(graded.sum()), graded.sum() / max(count, 1),
        graded.sum() / max(count, 1) / float(fps)))
    print("  a real match has roughly 1500-2500 touches")

    duels = contested(features)
    print("  {} of them are contested (both teams inside {:.1f} m)".format(
        int((touches & graded & duels).sum()), config.R_DZ_M))

    if args.report:
        # The label-free check: every restart the dead-ball layer found is a struck ball, and it
        # found them from pitch geometry with no reference to acceleration at all.
        _dead, _parts, restarts = deadball.dead_mask(features)
        restarts = [r for r in restarts if graded[r]]
        print("\nagreement with the {} restarts exp01 found independently".format(len(restarts)))
        print("  (a restart is a struck ball, so each one should coincide with a touch)")
        for tolerance in (3, 6, 12, 25):
            hit = sum(1 for r in restarts
                      if touches[max(0, r - tolerance):r + tolerance + 1].any())
            print("    within +/-{:>2} frames: {:>3}/{:<3}  {:.0f}%".format(
                tolerance, hit, len(restarts), 100.0 * hit / max(len(restarts), 1)))
        print("\n  this needs no ground truth: two independent routes to the same event.")
