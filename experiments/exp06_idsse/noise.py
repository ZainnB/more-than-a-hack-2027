"""Degrade clean IDSSE tracking so it looks like the provided automatic tracking (the ATD).

    python experiments/exp06_idsse/noise.py        # prints what each switch does to a match

Why it exists
-------------
IDSSE is corrected optical tracking: the ball is always there, always sharp, and has a height.
The provided match is automatic tactical-camera output. A pipeline tuned on the first and run on
the second finds its parameters in the wrong place. The simulator is how that gap is closed, so
how faithful it is decides how far anything tuned here can be trusted. Every figure it uses comes
from the provided *tracking* (input data), none from the ground truth or the video.

What it does, and where each number comes from
----------------------------------------------
  ball smoothing   The ATD ball is far smoother than a real ball: its residual against a
                   7-frame quadratic fit is 1.3 mm at the 99th percentile, against 300 mm for
                   IDSSE's. It is effectively a curve drawn through the kicks. A 51-frame
                   Savitzky-Golay filter is used, chosen because it makes exp02's touch detector
                   fire at the ATD's rate: 28.1 touches per visible-ball minute, against 32.0 and
                   24.0 on the two IDSSE matches measured (mean 28.0). This is the one parameter
                   fitted to a provided-tracking statistic, and it is a rate, not a label.
  ball height      Dropped. Only x and y are ever used, as in the ATD.
  ball past line   IDSSE parks the ball just outside the pitch for a whole stoppage; the ATD sees
                   it leave and loses it after a median of 21 frames (p90 122). Each exit is kept
                   visible for a lognormal draw matching those quantiles, then hidden.
  ball gaps        The ATD ball is missing on 40.7% of graded frames, in long runs: mean 431
                   frames, median 268, none shorter than 75. It is also missing mostly when play
                   is dead. So gaps are a two-state Markov chain: a missing run lasts 431 frames
                   on average, and a visible run is much shorter during a stoppage than during
                   play. The live missing rate (16%) was measured on the ATD with the pipeline's
                   own dead mask, which is partly circular -- it uses the ball-lost rule -- so
                   treat it as an upper bound on how well hidden the ball is in play. The dead
                   rate is then solved to reach 40.7% overall.
  team flips       A player's team label is wrong for a stretch of time on 7% of player-frames
                   (the pipeline's published ~93% team attribution), in segments averaging 4 s.
  player dropout   4.5% of player-frames untracked (the ATD's median is 21 of 22), in segments of
                   2 s.

Not simulated yet, and it matters: goalkeeper identity switches, more than 11 players on a team,
position jitter on players, and ball gaps that cluster in crowded moments rather than only in
stoppages.
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
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from exp02_touch_gated import features as feat2  # noqa: E402

# Params
BALL_SMOOTH_WINDOW = 51
BALL_MISSING_RATE = 0.407          # DATA.md / CHALLENGE.md: 41% of graded frames
BALL_MISSING_LIVE = 0.16           # measured on the ATD, see above
BALL_MEAN_GAP_FRAMES = 431         # 0.407 * 158857 graded frames / 150 gaps; used during stoppages
# How long a missing run lasts once play is live. Not measured: it is the one free choice in the
# gap model, and it exists because a 431-frame run that starts in a stoppage would otherwise carry
# 15 s into the play that follows and make a 16% live missing rate unreachable (a 30% floor).
# 100 frames (4 s) is "the ball is re-acquired shortly after a restart".
BALL_MEAN_GAP_LIVE_FRAMES = 100
# How long the ATD keeps seeing the ball once it has left the pitch: median 21 frames, p90 122,
# over 81 exits. IDSSE instead parks the ball just past the line for the whole stoppage (median
# runs of 350-400 frames), which made the out-of-bounds rule fire on 35% of simulated frames
# against 20% on the real match. Modelled as a lognormal matching those two quantiles.
OUTSIDE_KEEP_MEDIAN = 21
OUTSIDE_KEEP_P90 = 122
PITCH_HALF = (52.5, 34.0)
TEAM_FLIP_RATE = 0.07
TEAM_FLIP_MEAN_FRAMES = 100
PLAYER_DROPOUT_RATE = 0.045
PLAYER_DROPOUT_MEAN_FRAMES = 50

CLEAN = {"smooth": False, "gaps": False, "flips": False, "dropout": False}
ATD_LIKE = {"smooth": True, "gaps": True, "flips": True, "dropout": True}


def _two_state(n, rate, mean_on, rng):
    """Boolean mask, True for a fraction `rate` of frames, in segments averaging `mean_on`."""
    mean_off = mean_on * (1.0 - rate) / rate
    mask = np.zeros(n, dtype=bool)
    i = 0
    state = rng.random() < rate
    while i < n:
        length = max(1, int(rng.exponential(mean_on if state else mean_off)))
        mask[i:i + length] = state
        i += length
        state = not state
    return mask


def smooth_ball(ball, window=BALL_SMOOTH_WINDOW):
    """SG-smoothed ball, with the unsmoothed value kept where the filter leaves a NaN rim."""
    out = np.empty_like(ball)
    for axis in (0, 1):
        s = feat2.sg_smooth(ball[:, axis], window, 2)
        rim = np.isnan(s) & ~np.isnan(ball[:, axis])
        s[rim] = ball[rim, axis]
        out[:, axis] = s
    return out


def _run_chain(dead, rng, visible, mean_gap, start_missing):
    n = len(dead)
    draw = rng.random(n)
    hidden = np.zeros(n, dtype=bool)
    missing = start_missing
    for i in range(n):
        if missing:
            if draw[i] < 1.0 / mean_gap[bool(dead[i])]:
                missing = False
        elif draw[i] < 1.0 / visible[bool(dead[i])]:
            missing = True
        hidden[i] = missing
    return hidden


def outside_hidden(ball, rng, median=OUTSIDE_KEEP_MEDIAN, p90=OUTSIDE_KEEP_P90):
    """Frames where the ball has been outside the pitch longer than the camera keeps it."""
    outside = (np.abs(ball[:, 0]) > PITCH_HALF[0]) | (np.abs(ball[:, 1]) > PITCH_HALF[1])
    mu, sigma = np.log(median), (np.log(p90) - np.log(median)) / 1.2816
    edges = np.diff(np.concatenate(([0], outside.astype(np.int8), [0])))
    hidden = np.zeros(len(ball), dtype=bool)
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        keep = int(rng.lognormal(mu, sigma))
        hidden[a + keep:b] = True
    return hidden


def ball_gaps(dead, rng, rate=BALL_MISSING_RATE, live=BALL_MISSING_LIVE,
              mean_gap=BALL_MEAN_GAP_FRAMES, mean_gap_live=BALL_MEAN_GAP_LIVE_FRAMES, rounds=12,
              extra=None):
    """Boolean mask of frames with the ball hidden: a two-state chain keyed on dead vs live.

    A missing run that starts in a stoppage carries on into the play that follows, so the rate
    realised in play comes out well above what the visible-run lengths alone imply (40% against a
    16% target on the first attempt). The two visible-run means are therefore calibrated by
    simulation, on a fixed generator, until the realised live and dead rates hit their targets.
    `extra` frames are hidden regardless (the ball gone past the line) and count towards them.
    """
    extra = np.zeros(len(dead), dtype=bool) if extra is None else extra
    share_dead = max(dead.mean(), 1e-6)
    target = {False: live,
              True: float(np.clip((rate - live * (1.0 - share_dead)) / share_dead, 0.05, 0.97))}
    gap = {False: mean_gap_live, True: mean_gap}
    visible = {k: gap[k] * (1.0 - t) / t for k, t in target.items()}
    for _ in range(rounds):
        trial = _run_chain(dead, np.random.default_rng(12345), visible, gap, False) | extra
        for k in (False, True):
            mask = dead == k
            realised = max(trial[mask].mean(), 1e-3) if mask.any() else target[k]
            visible[k] = float(np.clip(visible[k] * realised / target[k], 5.0, 1e6))
    return _run_chain(dead, rng, visible, gap, bool(rng.random() < rate)) | extra


def degrade(scene, seed=0, switches=ATD_LIKE, ball_missing=None, flip_rate=None,
            protect_near_m=None):
    """(ball, players, team_matrix) after the chosen degradations. Deterministic in `seed`.

    The three keyword overrides exist for what-if runs (whatif.py), and leave the defaults alone:

      ball_missing    overall fraction of graded frames with the ball hidden. The in-play rate is
                      scaled in proportion, so a lower overall rate also means a better-seen game.
      flip_rate       fraction of player-frames carrying the wrong team.
      protect_near_m  team labels are always right for a player within this distance of the ball
                      (the true ball, not the degraded one). A what-if for a tracker that gets
                      identity right in duels, which is where identity switches are most likely
                      and do the most damage.
    """
    rng = np.random.default_rng(seed)
    ball = scene["ball"].copy()
    true_ball = scene["ball"]
    players = scene["players"].copy()
    n, persons = players.shape[:2]
    team = np.broadcast_to(scene["team"], (n, persons)).copy()
    true_team = team.copy()

    graded = scene["gt"] != pm.X
    if switches["smooth"]:
        ball = smooth_ball(ball)
    if switches["gaps"]:
        dead = scene["gt"] == pm.D
        rate = BALL_MISSING_RATE if ball_missing is None else ball_missing
        live = BALL_MISSING_LIVE * rate / BALL_MISSING_RATE
        hidden = ball_gaps(dead, rng, rate=rate, live=live, extra=outside_hidden(ball, rng))
        ball[hidden & graded] = np.nan
    rate = TEAM_FLIP_RATE if flip_rate is None else flip_rate
    if switches["flips"] and rate > 0:
        for p in range(persons):
            team[_two_state(n, rate, TEAM_FLIP_MEAN_FRAMES, rng), p] ^= 1
        if protect_near_m is not None:
            near = np.hypot(players[..., 0] - true_ball[:, None, 0],
                            players[..., 1] - true_ball[:, None, 1]) <= protect_near_m
            team[near] = true_team[near]
    if switches["dropout"]:
        for p in range(persons):
            players[_two_state(n, PLAYER_DROPOUT_RATE, PLAYER_DROPOUT_MEAN_FRAMES, rng), p] = np.nan
    return ball, players, team


if __name__ == '__main__':
    import scene as sc  # noqa: E402

    s = sc.build("J03WMX")
    graded = s["gt"] != pm.X
    for name, sw in (("clean", CLEAN), ("atd-like", ATD_LIKE)):
        ball, players, team = degrade(s, 0, sw)
        missing = np.isnan(ball[:, 0]) & graded
        flipped = (team != np.broadcast_to(s["team"], team.shape)) & ~np.isnan(players[..., 0])
        print("{:<9} ball missing {:5.1f}%  (live {:4.1f}%, dead {:4.1f}%)   players tracked {:4.1f}   "
              "team-flipped {:4.1f}% of tracked".format(
                  name, 100 * missing.sum() / graded.sum(),
                  100 * (missing & (s["gt"] != pm.D)).sum() / max(1, (graded & (s["gt"] != pm.D)).sum()),
                  100 * (missing & (s["gt"] == pm.D)).sum() / max(1, (s["gt"] == pm.D).sum()),
                  np.mean((~np.isnan(players[..., 0])).sum(axis=1)[graded]),
                  100 * flipped.sum() / (~np.isnan(players[..., 0])).sum()))
