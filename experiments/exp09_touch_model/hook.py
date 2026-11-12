"""The touch model as a touch_hook for exp04's labels_from_features.

The hook replaces exp02's hand-set detector. Every candidate approach the model scores at or above
`tau` becomes a touch by that team, dated to the candidate frame; where both teams fire on the
same frame the likelier one wins. Restarts stay touches, as in exp02. Nothing is marked
contested: the model has already weighed the duel, and "duel prevails" is what kept presser-
versus-carrier calls stuck on the wrong side.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp08_deadball_clf"))

import touchfeat  # noqa: E402
import deadclf  # noqa: E402
from exp02_touch_gated import touch as touch_rule  # noqa: E402


def score_candidates(model, ball, players, team, inside):
    """(frames, teams, p) for every candidate approach inside the halves."""
    dist, _col, _d = touchfeat.team_distances(ball, players, team)
    frames, teams = touchfeat.candidates(dist, inside)
    X = touchfeat.features(ball, players, team, frames, teams)
    return frames, teams, deadclf.proba(model, X)


def second_touch(frames, teams, window):
    """Keep a change-of-team touch only if the other side does not touch next within `window`.

    A touch by B after A's is confirmed when the next touch inside the window is B's again (a
    second touch, or a teammate receiving), or when nobody touches inside it. If A touches next,
    B's touch was a block or a poke that A recovered: it is dropped, and with it the 1-3 s
    flip-and-back. Real switches that A wins straight back are lost too; that is the trade.
    """
    keep = np.ones(len(frames), dtype=bool)
    prev = -1
    for i in range(len(frames)):
        t = teams[i]
        if prev >= 0 and t != prev:
            j = i + 1
            while j < len(frames) and frames[j] == frames[i]:
                j += 1
            if j < len(frames) and frames[j] - frames[i] <= window and teams[j] != t:
                keep[i] = False
                continue
        prev = t
    return keep


def make(frames, teams, p, tau, confirm_window=0):
    """A touch_hook from scored candidates; `confirm_window` > 0 applies the second-touch check."""
    sel = p >= tau
    frames, teams, p = frames[sel], teams[sel], p[sel]
    if confirm_window:
        keep = second_touch(frames, teams, confirm_window)
        frames, teams, p = frames[keep], teams[keep], p[keep]

    def hook(features, _kin, dead, restarts):
        n = len(dead)
        touch_team, _dist = touch_rule.nearest_player(features)
        touch_team = touch_team.copy()
        touches = np.zeros(n, dtype=bool)
        best = np.zeros(n)
        for f, t, q in zip(frames, teams, p):
            if not dead[f] and q > best[f]:
                touches[f], touch_team[f], best[f] = True, t, q
        for r in restarts:
            if 0 <= r < n:
                touches[r] = True
        return touches, touch_team, np.zeros(n, dtype=bool)
    return hook
