"""Scene -> the feature dict the exp01..exp04 pipeline consumes.

The pipeline was written against the ATD, so this reproduces that dict key for key: normalised
0-1 ball coordinates, the distance from the ball to the nearest player of each team, how many
players of each team are inside 3 m and 5 m, how many players are tracked, and the mean player
speed. See exp01_geometry_hysteresis/features.extract for the original.

Takes the team of every player on every frame as a matrix rather than one label per player, so
the noise simulator can flip a player's team for a stretch of time.
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

# Params
PITCH = (105.0, 68.0)   # IDSSE reports 105 x 68; the ATD's 105.3 is a measured value
FPS = 25
NEAR_RADII = (3.0, 5.0)

# Half-width, in frames, of the central difference used for player speed. 2 gives a 5-frame
# window, enough that one noisy position does not fake a sprint. The ATD supplies a smoothed
# speed per player; this stands in for it.
SPEED_HALF_SPAN = 2


def player_speed(players):
    """(frames, persons) speed in m/s from positions, NaN where either end is untracked."""
    n = len(players)
    out = np.full(players.shape[:2], np.nan)
    h = SPEED_HALF_SPAN
    delta = players[2 * h:] - players[:n - 2 * h]
    out[h:n - h] = np.hypot(delta[..., 0], delta[..., 1]) * FPS / (2 * h)
    return out


def to_features(ball, players, team_matrix, pitch=PITCH):
    """ball (n,2) metres, players (n,P,2) metres, team_matrix (n,P) of 0/1 -> feature dict."""
    n = len(ball)
    length, width = pitch

    distance = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])

    out = {
        "ball_x": (ball[:, 0] / length + 0.5).astype(np.float32),
        "ball_y": (ball[:, 1] / width + 0.5).astype(np.float32),
        "ball_speed": np.full(n, np.nan, dtype=np.float32),
    }
    for side, key in ((0, "d_home"), (1, "d_away")):
        d = np.where(team_matrix == side, distance, np.nan)
        has = ~np.all(np.isnan(d), axis=1)
        best = np.full(n, np.nan)
        best[has] = np.nanmin(d[has], axis=1)
        out[key] = best.astype(np.float32)
    for radius in NEAR_RADII:
        near = distance < radius
        out["n_home_{}".format(int(radius))] = np.sum(near & (team_matrix == 0), axis=1).astype(np.int8)
        out["n_away_{}".format(int(radius))] = np.sum(near & (team_matrix == 1), axis=1).astype(np.int8)

    tracked = ~np.isnan(players[..., 0])
    out["n_tracked"] = tracked.sum(axis=1).astype(np.int16)
    speed = player_speed(players)
    known = ~np.isnan(speed)
    total = np.where(known, speed, 0.0).sum(axis=1)
    count = known.sum(axis=1)
    mean_speed = np.full(n, np.nan)
    mean_speed[count > 0] = total[count > 0] / count[count > 0]
    out["mean_speed"] = mean_speed.astype(np.float32)
    return out
