"""True touches on CLEAN IDSSE tracking: the training target for the touch model.

    python experiments/exp09_touch_model/labels.py --report          # every match
    python experiments/exp09_touch_model/labels.py --report J03WMX

What a touch is here
--------------------
The clean TRACAB ball is 25 Hz, unsmoothed, with height. A free ball only slows down (rolling
friction, air drag) and falls under gravity; it never turns or speeds up on its own. So a touch
is a frame where the ball's velocity changes by more than free flight can explain, measured as the
difference between the mean velocity over the K frames after and the K frames before:

  - the horizontal change always counts;
  - the vertical change counts only when the ball is clear of the ground (z > Z_GROUND_M), after
    removing gravity, so a bounce is not a touch but a header is;

and a player is within reach at that moment: horizontally within REACH_M of the ball, and the
ball below HEAD_M (players are tracked as points on the ground, so reach is a cylinder).
Every team with a player in reach is credited (see detect). `header` = ball above HEADER_Z_M.

Thresholds are physical, not fitted: DV_MPS is several times what drag and friction can remove
in 2K frames (< 1 m/s even for a 30 m/s ball), and well under the gentlest trap (~3 m/s).
They are then *checked* against the event log (every Play / Shot / Tackle / Claim should have a
touch by the same player nearby) and against the BallPossession flips, which is external data
and so legal to look at, but no value here was picked because it scored better.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import events as events_mod  # noqa: E402
import idsse  # noqa: E402
import tune  # noqa: E402
import possession_metrics as pm  # noqa: E402

# Params
K = 3                 # frames each side for the before / after velocity
DV_MPS = 2.0          # velocity change no free ball can make in 2K frames
G = 9.81
Z_GROUND_M = 0.4      # below this a vertical change may be a bounce
REACH_M = 1.5         # horizontal reach of a tracked player point (legs, body, lunge)
HEAD_M = 2.7          # highest a player can meet the ball (jumping header)
HEADER_Z_M = 1.4
REFRACTORY = 5        # one contact is one touch
MATCH_FRAMES = 12     # +-0.5 s when checking against the BallPossession flips
EVENT_FRAMES = 50     # +-2 s against the hand-timed event log

CACHE = os.path.join(HERE, "cache", "touches_{}.npz")


def velocity_change(ball, z):
    """|v_after - v_before| per frame, gravity removed from the vertical part when airborne."""
    n = len(ball)
    p = np.column_stack([ball, z])
    dv = np.full(n, np.nan)
    before = (p[K:-K] - p[:-2 * K]) / K * 25.0          # frame t: p[t] - p[t-K]
    after = (p[2 * K:] - p[K:-K]) / K * 25.0            # frame t: p[t+K] - p[t]
    d = after - before
    d[:, 2] += G * K / 25.0                             # free fall changes vz by -g*K/25
    airborne = z[K:-K] > Z_GROUND_M
    d[~airborne, 2] = 0.0
    dv[K:-K] = np.sqrt((d ** 2).sum(axis=1))
    return dv


def detect(scene):
    """Every true touch in a clean scene: frame, team, credited column, header, contested.

    A velocity change with players of BOTH teams in reach gives one touch per team, each flagged
    `contested`. Crediting only the nearest player covered 84-86% of BallPossession flips with a
    touch by the new team; crediting each team in reach covers 94-96%. The rest are duels where
    the nearest player at the peak is not the one who played it, and the tracking cannot say
    which foot met the ball first. Refractory periods are kept per team for the same reason.
    """
    ball, z, players, team = scene["ball"], scene["ball_z"], scene["players"], scene["team"]
    z = np.nan_to_num(z, nan=0.0)
    dv = np.nan_to_num(velocity_change(ball, z), nan=0.0)
    peak = np.zeros(len(dv), dtype=bool)
    peak[1:-1] = (dv[1:-1] >= DV_MPS) & (dv[1:-1] >= dv[:-2]) & (dv[1:-1] > dv[2:])
    ok = peak & (z <= HEAD_M) & (scene["gt"] != pm.X)

    dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    per_team = {}
    for tm in (0, 1):
        cols = np.flatnonzero(team == tm)
        per_team[tm] = (dist[:, cols].min(axis=1), cols[dist[:, cols].argmin(axis=1)])

    rows = []
    for tm in (0, 1):
        d, col = per_team[tm]
        last = -REFRACTORY - 1
        for f in np.flatnonzero(ok & (d <= REACH_M)):
            if f - last >= REFRACTORY:
                rows.append((f, tm, col[f]))
                last = f
    rows.sort()
    frames = np.array([r[0] for r in rows], dtype=np.int64)
    teams = np.array([r[1] for r in rows], dtype=np.int8)
    contested = (per_team[0][0][frames] <= REACH_M) & (per_team[1][0][frames] <= REACH_M)
    return {"frame": frames, "team": teams, "column": np.array([r[2] for r in rows]),
            "header": z[frames] > HEADER_Z_M, "contested": contested, "dv": dv[frames]}


def load(match, rebuild=False):
    path = CACHE.format(match)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as h:
            return {k: h[k] for k in h.files}
    t = detect(tune.cached_scene(match))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **t)
    return t


def possession_flips(gt):
    """Frames where the true team changes between consecutive live (H/A) frames, ignoring D."""
    live = np.flatnonzero((gt == pm.H) | (gt == pm.A))
    codes = gt[live]
    i = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    contiguous = live[i] - live[i - 1] == 1          # a flip across a stoppage is a restart
    return live[i][contiguous], (codes[i][contiguous] == pm.A).astype(np.int8)


def report(match):
    s = tune.cached_scene(match)
    t = load(match, rebuild=True)
    ev = events_mod.load(match)
    tf, tc, tt = t["frame"], t["column"], t["team"]

    def hit(frame, want, key, span=MATCH_FRAMES):
        lo, hi = np.searchsorted(tf, frame - span), np.searchsorted(tf, frame + span + 1)
        return bool((key[lo:hi] == want).any())

    # The event log is hand-timed: at its stamp the player is a median 3.7 m from the ball and the
    # best constant lag differs by match (-0.6 s, +0.8 s). It checks *who*, in a +-2 s window;
    # the frame-accurate check is the BallPossession flips below.
    by_player = np.mean([hit(e["frame"], e["column"], tc, EVENT_FRAMES) for e in ev])
    by_team = np.mean([hit(e["frame"], e["team"], tt, EVENT_FRAMES) for e in ev])
    heads = [e for e in ev if e["header_hint"]]
    head_rec = np.mean([hit(e["frame"], e["column"], tc, EVENT_FRAMES) for e in heads]) \
        if heads else np.nan
    ff, fteam = possession_flips(s["gt"])
    flip_cov = np.mean([hit(f, w, tt) for f, w in zip(ff, fteam)])
    live_min = (s["gt"] != pm.X).sum() / 25.0 / 60.0
    print("{}  touches {:>5} ({:.0f}/live min, {:.0f}% headers, {:.0f}% contested)  "
          "event recall +-2s: player {:.3f} team {:.3f} aerial {:.3f} (n={})  "
          "flips covered {:.3f} (n={})".format(
              match[-6:], len(tf), len(tf) / live_min, 100 * t["header"].mean(),
              100 * t["contested"].mean(), by_player, by_team, head_rec, len(heads), flip_cov,
              len(ff)))
    return by_player, flip_cov


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("matches", nargs="*")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    wanted = [m for m in idsse.matches()
              if not args.matches or any(m.endswith(w) for w in args.matches)]
    for m in wanted:
        if args.report:
            report(m)
        else:
            print(m, len(load(m, rebuild=True)["frame"]))
