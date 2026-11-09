"""Per-frame features for the dead-ball classifier, computed the same way on IDSSE and on the ATD.

Tracking only: the players' movement and shape, and what the ball is doing. After Lang et al.
2022 (in-play status from player positions, 0.92 frame accuracy on TRACAB) and the stoppage
signatures in Vidal-Codina et al. 2022. The pipeline is offline, so windows look both back and
ahead: a stoppage is a slowdown *followed by* walking, and a restart is walking followed by a
burst, and both directions carry information.

Player speed is differenced from positions on both datasets, never read from the ATD's own speed
field, so the two are defined identically.
"""

import numpy as np

FPS = 25

# Thresholds that turn per-player speeds into shares. Walking, and running hard.
SLOW_MPS = 1.0
FAST_MPS = 4.0
SPEED_HALF_SPAN = 2

NAMES = []


def _speed(xy):
    """(n, P) speed in m/s from (n, P, 2) positions, central difference over 2 * SPEED_HALF_SPAN."""
    n = len(xy)
    h = SPEED_HALF_SPAN
    out = np.full(xy.shape[:2], np.nan)
    d = xy[2 * h:] - xy[:n - 2 * h]
    out[h:n - h] = np.hypot(d[..., 0], d[..., 1]) * FPS / (2 * h)
    return out


def window_mean(x, a, b):
    """Mean of x over frames [t + a, t + b] inclusive, ignoring NaN. NaN where none."""
    n = len(x)
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(n)
    lo = np.clip(t + a, 0, n)
    hi = np.clip(t + b + 1, 0, n)
    total, count = cs[hi] - cs[lo], cn[hi] - cn[lo]
    out = np.full(n, np.nan)
    out[count > 0] = total[count > 0] / count[count > 0]
    return out


def _nanmean_rows(m):
    ok = ~np.isnan(m)
    count = ok.sum(axis=1)
    out = np.full(len(m), np.nan)
    out[count > 0] = np.where(ok, m, 0.0).sum(axis=1)[count > 0] / count[count > 0]
    return out


def compute(ball, players, half_length, half_width):
    """(n, F) features from ball (n, 2) and players (n, P, 2), metres from the pitch centre."""
    s = lambda sec: int(round(sec * FPS))  # noqa: E731
    speed = _speed(players)
    tracked = ~np.isnan(speed)
    pitch = _nanmean_rows(speed)
    n_tr = tracked.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        slow = np.where(n_tr > 0, ((speed < SLOW_MPS) & tracked).sum(axis=1) / n_tr, np.nan)
        fast = np.where(n_tr > 0, ((speed > FAST_MPS) & tracked).sum(axis=1) / n_tr, np.nan)
    centre = np.stack([_nanmean_rows(players[..., 0]), _nanmean_rows(players[..., 1])], axis=1)
    spread = _nanmean_rows(np.hypot(players[..., 0] - centre[:, None, 0],
                                    players[..., 1] - centre[:, None, 1]))

    seen = (~np.isnan(ball[:, 0])).astype(float)
    outside = np.where(seen > 0, ((np.abs(ball[:, 0]) > half_length) |
                                  (np.abs(ball[:, 1]) > half_width)).astype(float), 0.0)
    bspeed = np.full(len(ball), np.nan)
    h = SPEED_HALF_SPAN
    d = ball[2 * h:] - ball[:-2 * h]
    bspeed[h:-h] = np.hypot(d[:, 0], d[:, 1]) * FPS / (2 * h)
    line = np.minimum(half_length - np.abs(ball[:, 0]), half_width - np.abs(ball[:, 1]))
    near = np.nanmin(np.where(np.isnan(players[..., 0]), np.inf,
                              np.hypot(players[..., 0] - ball[:, None, 0],
                                       players[..., 1] - ball[:, None, 1])), axis=1)
    near = np.where(np.isfinite(near) & (seen > 0), near, np.nan)

    cols = {}
    cols["pitch_now"] = window_mean(pitch, -s(0.5), s(0.5))
    cols["pitch_past3"] = window_mean(pitch, -s(3), -1)
    cols["pitch_next3"] = window_mean(pitch, 1, s(3))
    cols["pitch_past8"] = window_mean(pitch, -s(8), -1)
    cols["pitch_next8"] = window_mean(pitch, 1, s(8))
    cols["slowdown"] = cols["pitch_past3"] - cols["pitch_next3"]
    cols["slow_now"] = window_mean(slow, -s(0.5), s(0.5))
    cols["slow_next3"] = window_mean(slow, 1, s(3))
    cols["fast_now"] = window_mean(fast, -s(0.5), s(0.5))
    cols["spread_now"] = window_mean(spread, -s(0.5), s(0.5))
    cols["seen_now"] = window_mean(seen, -s(1), s(1))
    cols["seen_past5"] = window_mean(seen, -s(5), -1)
    cols["seen_next5"] = window_mean(seen, 1, s(5))
    cols["out_now"] = window_mean(outside, -s(1), s(1))
    cols["out_past5"] = window_mean(outside, -s(5), -1)
    cols["out_next5"] = window_mean(outside, 0, s(5))
    cols["ball_speed"] = window_mean(bspeed, -s(0.5), s(0.5))
    cols["ball_line"] = window_mean(line, -s(0.5), s(0.5))
    cols["ball_near"] = window_mean(near, -s(0.5), s(0.5))
    # a few transforms so a linear model can bend
    cols["log_pitch_now"] = np.log(cols["pitch_now"] + 0.1)
    cols["log_pitch_next8"] = np.log(cols["pitch_next8"] + 0.1)
    cols["log_ball_speed"] = np.log(cols["ball_speed"] + 0.1)
    cols["ball_missing"] = np.isnan(cols["ball_speed"]).astype(float)
    NAMES[:] = list(cols)
    return np.stack([cols[k] for k in NAMES], axis=1).astype(np.float32)
