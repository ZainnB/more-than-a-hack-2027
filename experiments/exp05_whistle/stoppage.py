"""Use the referee's whistle to start stoppages: a dead_hook for exp04's pipeline.

Two things the tracking-only dead-ball layer cannot do, and the whistle can:

  anchor   A stoppage the tracking does find usually starts late, because it is read from the
           players slowing down afterwards. If one starts within ANCHOR_S after a whistle, its
           start moves back to the whistle.
  add      A foul or offside with the ball still on the pitch and play resuming quickly is often
           not found at all (exp06's dead-ball recall on the public window is 0.38). A whistle in
           live play that is followed by play visibly stopping becomes a stoppage, from the
           whistle until the restart.

"Play visibly stops" is the guard against the crowd, which also whistles: within SETTLE_S after
the whistle the ball settles (median speed under the settled speed for a second) or goes untracked,
AND the mean player speed over the following seconds falls by at least SLOWDOWN against the
seconds before. A crowd whistle in the middle of an attack fails both.

The restart is found exactly as exp01 finds one after the ball leaves the pitch: a settled ball,
inside the pitch, struck at the restart speed. Without one inside MAX_STOP_S nothing is added.

Every threshold here is set a priori from what a stoppage looks like; none is tuned on our labels.
"""

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

import whistle  # noqa: E402
from exp01_geometry_hysteresis import config as exp01_config, deadball, \
    features as exp01_features  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402

# Params
FPS = 25
ANCHOR_S = 5.0
SETTLE_S = 4.0
SLOWDOWN = 0.10
BEFORE_S, AFTER_S = 3.0, 3.0
MAX_STOP_S = 30.0
MIN_GAP_S = 1.0     # restart search begins this long after the whistle


def _settles(speed, seen, a, b):
    """Does the ball settle, or go untracked for most of a second, inside [a, b)?"""
    win = FPS
    for start in range(a, max(a + 1, b - win)):
        s = speed[start:start + win]
        known = s[~np.isnan(s)]
        if (~seen[start:start + win]).mean() >= 0.5:
            return True
        if known.size and np.median(known) < exp01_config.BALL_SETTLED_SPEED_MPS:
            return True
    return False


def make_hook(game=GAME, stats=None):
    events = whistle.load(game)["events"]
    blocks = half_video_blocks(game)
    stats = {} if stats is None else stats

    def hook(dead, restarts, features):
        dead = dead.copy()
        restarts = list(restarts)
        speed = exp01_features.ball_speed_mps(features, game)
        seen = exp01_features.ball_seen(features)
        pitch_speed = features["mean_speed"]
        n = len(dead)
        anchored = added = ignored = already = 0
        for start_s, _end_s, _hz, _db in events:
            w = int(start_s * FPS)
            if not any(first <= w <= last for _h, first, last in blocks):
                continue
            if dead[w]:
                already += 1
                continue
            ahead = dead[w:min(n, w + int(ANCHOR_S * FPS))]
            if ahead.any():
                onset = w + int(np.argmax(ahead))
                dead[w:onset] = True
                anchored += 1
                continue
            before = pitch_speed[max(0, w - int(BEFORE_S * FPS)):w]
            after = pitch_speed[w + FPS:w + FPS + int(AFTER_S * FPS)]
            slowed = (np.isfinite(before).any() and np.isfinite(after).any()
                      and np.nanmean(after) <= (1.0 - SLOWDOWN) * np.nanmean(before))
            if not (slowed and _settles(speed, seen, w, w + int(SETTLE_S * FPS))):
                ignored += 1
                continue
            restart, how = deadball.find_restart(features, w + int(MIN_GAP_S * FPS),
                                                 w + int(MAX_STOP_S * FPS), speed)
            if restart is None or how != "struck":
                ignored += 1
                continue
            dead[w:restart] = True
            restarts.append(restart)
            added += 1
        stats.update(anchored=anchored, added=added, ignored=ignored, already_dead=already)
        return dead, sorted(set(restarts))

    return hook
