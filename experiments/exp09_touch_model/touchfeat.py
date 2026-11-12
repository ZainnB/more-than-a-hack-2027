"""Candidate touch moments and their features, from tracking that looks like the ATD.

A candidate is (frame, team): a moment where that team's nearest player makes a closest approach
to the ball (a local minimum of the team's ball distance, within CANDIDATE_M), thinned so one
approach is one candidate. Every touch needs a player at the ball, so the candidates should
contain nearly every true touch; the model's job is to say which approaches were contact.

Works on one common shape for both sources:
    ball (n, 2) metres from the centre, NaN where unseen (smoothed, as the ATD is)
    players (n, P, 2) metres from the centre
    team (n, P) 0 home / 1 away, per frame (degraded IDSSE flips it; the ATD's is per slot)

Features describe what the smoothed track still shows around a contact: the closest approach,
the opposition's distance, what the ball did before and after at three time scales (smoothing
hides the instant but not the before/after), whether the ball arrived toward the player and left
away from them or with them, and what the player's own body did (braking, turning).
"""

import numpy as np

FPS = 25.0
CANDIDATE_M = 4.0           # the 2 s ball smoothing drags a struck ball metres ahead of the kicker
NMS_FRAMES = 10            # one candidate per team per 0.4 s
SCALES = (6, 12, 25)       # before/after windows, frames
NEAR_M = (1.5, 3.0)

NAMES = (
    ["d_team", "d_opp", "d_opp_min12", "d_gap", "n_team_3", "n_opp_3", "n_team_1.5", "n_opp_1.5",
     "ball_speed", "ball_seen_12", "ball_seen_25", "d_team_m12", "d_team_p12", "d_team_m25",
     "d_team_p25", "frames_within_2m"]
    + ["turn_{}".format(k) for k in SCALES]
    + ["speed_ratio_{}".format(k) for k in SCALES]
    + ["dv_{}".format(k) for k in SCALES]
    + ["arrive_{}".format(k) for k in SCALES]
    + ["leave_{}".format(k) for k in SCALES]
    + ["carry_{}".format(k) for k in SCALES]
    + ["p_speed", "p_brake", "p_turn", "opp_speed"]
)


def team_distances(ball, players, team):
    """(n,2) nearest distance per team and (n,2) the player column achieving it."""
    d = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
    d = np.where(np.isnan(d), np.inf, d)
    out_d = np.full((len(ball), 2), np.inf)
    out_j = np.zeros((len(ball), 2), dtype=np.int64)
    for tm in (0, 1):
        dt = np.where(team == tm, d, np.inf)
        out_j[:, tm] = dt.argmin(axis=1)
        out_d[:, tm] = dt[np.arange(len(ball)), out_j[:, tm]]
    return out_d, out_j, d


def candidates(dist, live):
    """(frames, teams) of closest approaches, NMS-thinned per team."""
    frames, teams = [], []
    for tm in (0, 1):
        d = dist[:, tm]
        ok = np.isfinite(d) & (d <= CANDIDATE_M) & live
        dm = np.where(np.isfinite(d), d, np.inf)
        minima = np.zeros(len(d), dtype=bool)
        minima[1:-1] = ok[1:-1] & (dm[1:-1] <= dm[:-2]) & (dm[1:-1] < dm[2:])
        # plateaus and the edge of a ball gap: the first frame of a run within reach counts too
        enter = np.zeros(len(d), dtype=bool)
        enter[1:] = ok[1:] & ~ok[:-1]
        idx = np.flatnonzero(minima | enter)
        # NMS: keep the closest approach in every NMS window
        order = idx[np.argsort(dm[idx], kind="stable")]
        taken = np.zeros(len(d), dtype=bool)
        keep = []
        for f in order:
            lo, hi = max(0, f - NMS_FRAMES), f + NMS_FRAMES + 1
            if not taken[lo:hi].any():
                keep.append(f)
                taken[f] = True
        keep.sort()
        frames += keep
        teams += [tm] * len(keep)
    order = np.lexsort((teams, frames))
    return np.array(frames, dtype=np.int64)[order], np.array(teams, dtype=np.int8)[order]


def _at(arr, idx):
    return arr[np.clip(idx, 0, len(arr) - 1)]


def _unit(v):
    n = np.hypot(v[:, 0], v[:, 1])[:, None]
    return np.where(n > 1e-6, v / np.maximum(n, 1e-6), 0.0)


def features(ball, players, team, frames, teams):
    """(len(frames), len(NAMES)) feature matrix."""
    n = len(ball)
    dist, col, d_all = team_distances(ball, players, team)
    seen = ~np.isnan(ball[:, 0])
    cs = np.concatenate(([0], np.cumsum(seen)))
    F, T = frames, teams.astype(np.int64)
    O = 1 - T
    j = col[F, T]                                   # the candidate player
    rows = np.arange(len(F))
    cols = {}

    cols["d_team"] = dist[F, T]
    cols["d_opp"] = dist[F, O]
    win = np.stack([_at(dist[:, 0], F + k) for k in range(-12, 13)], axis=1), \
        np.stack([_at(dist[:, 1], F + k) for k in range(-12, 13)], axis=1)
    cols["d_opp_min12"] = np.where(O == 0, win[0].min(axis=1), win[1].min(axis=1))
    cols["d_gap"] = cols["d_opp"] - cols["d_team"]
    for r in NEAR_M:
        near = d_all[F] <= r
        tF = team[F]
        cols["n_team_{}".format(r if r != 3.0 else 3)] = (near & (tF == T[:, None])).sum(axis=1)
        cols["n_opp_{}".format(r if r != 3.0 else 3)] = (near & (tF == O[:, None])).sum(axis=1)

    def ball_vel(a, b):
        """Mean ball velocity between frames a and b (arrays), NaN if either end unseen."""
        pa, pb = _at(ball, a), _at(ball, b)
        return (pb - pa) * FPS / np.maximum(b - a, 1)[:, None]

    v_now = ball_vel(F - 2, F + 2)
    cols["ball_speed"] = np.hypot(v_now[:, 0], v_now[:, 1])
    for k in (12, 25):
        lo, hi = np.clip(F - k, 0, n), np.clip(F + k + 1, 0, n)
        cols["ball_seen_{}".format(k)] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)

    def player_ball(offset):
        f = np.clip(F + offset, 0, n - 1)
        return np.hypot(players[f, j, 0] - ball[f, 0], players[f, j, 1] - ball[f, 1])

    for k in (12, 25):
        cols["d_team_m{}".format(k)] = player_ball(-k)
        cols["d_team_p{}".format(k)] = player_ball(k)
    within = np.stack([player_ball(k) for k in range(-25, 26, 2)], axis=1)
    cols["frames_within_2m"] = np.nansum(within <= 2.0, axis=1) * 2.0

    p_now = players[F, j]
    for k in SCALES:
        vb, va = ball_vel(F - k, F), ball_vel(F, F + k)
        sb, sa = np.hypot(vb[:, 0], vb[:, 1]), np.hypot(va[:, 0], va[:, 1])
        cosang = (vb * va).sum(axis=1) / np.maximum(sb * sa, 1e-6)
        cols["turn_{}".format(k)] = np.where((sb > 0.5) & (sa > 0.5), 1.0 - cosang, 0.0)
        cols["speed_ratio_{}".format(k)] = np.log((sa + 0.5) / (sb + 0.5))
        cols["dv_{}".format(k)] = np.hypot(*(va - vb).T)
        to_player = _unit(p_now - ball[F])
        cols["arrive_{}".format(k)] = (vb * to_player).sum(axis=1)       # ball coming toward
        cols["leave_{}".format(k)] = -(va * to_player).sum(axis=1)       # ball going away
        pv = (_at(players[:, :, :], F + k)[rows, j] - p_now) * FPS / k
        cols["carry_{}".format(k)] = np.hypot(*(va - pv).T)              # ball moves with him?

    pb = (p_now - _at(players, F - 6)[rows, j]) * FPS / 6
    pa = (_at(players, F + 6)[rows, j] - p_now) * FPS / 6
    sb, sa = np.hypot(*pb.T), np.hypot(*pa.T)
    cols["p_speed"] = sb
    cols["p_brake"] = sb - sa
    cosang = (pb * pa).sum(axis=1) / np.maximum(sb * sa, 1e-6)
    cols["p_turn"] = np.where((sb > 1) & (sa > 1), 1.0 - cosang, 0.0)
    jo = col[F, O]
    po = (_at(players, F + 6)[rows, jo] - _at(players, F - 6)[rows, jo]) * FPS / 12
    cols["opp_speed"] = np.hypot(*po.T)

    X = np.column_stack([np.asarray(cols[name], dtype=np.float64) for name in NAMES])
    X[~np.isfinite(X)] = np.nan
    return X
