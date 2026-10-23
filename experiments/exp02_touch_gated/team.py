"""Touch-gated possession: the ball changes hands at a touch, and the change is dated to it.

**As shipped, the touch path is OFF** (config.USE_TOUCH) and only the fallback path runs, at 2 s,
backdated. That was the outcome of measuring this module on 20 annotated minutes: backdating and
the stricter fallback improved pooled spellF1 from 0.328 (exp01) to 0.529, while the touch path
as planned took it back down to 0.356. README.md has the full ablation. The design below is what
was planned, kept intact so the measurement can be re-run.

What changes from exp01
-----------------------
exp01 concedes possession once the opposition has been nearer by 2 m for 25 consecutive frames,
and labels the change at the *end* of that second. On the 15-minute hand-annotated validation set
that produced two measurable failures:

  - flip-backs. 14 possessions of ~2 s sandwiched between the other team's, almost all wholly
    wrong in the hand labels: a presser gets within 2 m for a second, is credited with the ball,
    and loses it again. Nobody touched it.
  - lateness. Every change is reported a second after the evidence started.

This module gates the change on a touch and backdates it:

  1. TOUCH PATH. At a touch by the team not in possession, look ahead K_CONFIRM frames. If the new
     team is nearer on most of the frames where that can be judged, the change happened *at the
     touch*, and the labels from the touch onwards say so. If not, it was a deflection or a
     tackle that did not come away with the ball, and nothing changes. A *contested* touch --
     both teams inside the duel zone -- never transfers possession: duel prevails (Vidal-Codina),
     and CHALLENGE.md says possession stays with the last team that had it.

  2. FALLBACK PATH. The touch detector finds ~2000 touches and the ball is untracked on 43% of
     frames, so many real changes have no detected touch. Without a fallback possession would
     stick to one team. The fallback is exp01's rule made stricter (3 s, not 1 s, because the
     flip-backs last up to ~5 s) and backdated to the start of the streak.

  3. A ball in flight is never evidence. Above IN_FLIGHT_SPEED_MPS with no touch, whoever stands
     under it did not touch it -- the research note's answer to having no z coordinate.

Everything else is exp01's: the dead-ball layer, state reset at every stoppage, holding the team
through ball-less frames, and Smart Tagging votes only after a long ball-less stretch.
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
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, half_video_blocks  # noqa: E402

try:
    from . import config, features as feat, touch as touch_layer
except ImportError:
    import config
    import features as feat
    import touch as touch_layer

from exp01_geometry_hysteresis import config as exp01_config  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1


def evidence(features, kinematics, touches):
    """Per-frame team whose proximity counts as possession evidence, or NOBODY.

    A frame is evidence for a team when that team is nearest to the ball by at least MARGIN_M and
    the ball is not in flight. "No evidence" is different from "evidence for the other team":
    the first pauses a streak, the second breaks it.
    """
    nearest, margin, has_answer = feat.team_margin(features)
    speed = np.nan_to_num(kinematics["speed"], nan=0.0)
    in_flight = (speed > config.IN_FLIGHT_SPEED_MPS) & ~touches
    usable = has_answer & (margin >= config.MARGIN_M) & ~in_flight
    return np.where(usable, nearest, NOBODY).astype(np.int8), nearest, has_answer


def _confirmed(nearest, has_answer, start, stop, team):
    """Did `team` keep the ball over [start, stop)? Majority of judgeable frames, with a floor."""
    window = slice(start, stop)
    seen = has_answer[window]
    n_seen = int(seen.sum())
    if n_seen < config.CONFIRM_MIN_SEEN:
        return False
    return (nearest[window][seen] == team).mean() >= config.CONFIRM_FRACTION


def possession_state(features, kinematics, dead, touches, touch_team, contested,
                     votes=None, blocks=None, game=GAME, use_touch=True, use_fallback=True):
    """Per-frame HOME / AWAY / NOBODY. `dead` frames stay NOBODY; the caller writes D there.

    Also returns a per-frame record of *why* possession changed -- "touch" or "fallback" -- which
    is what lets the validation compare say which path is earning its keep.
    """
    blocks = half_video_blocks(game) if blocks is None else blocks
    ev, nearest, has_answer = evidence(features, kinematics, touches)
    votes_home, votes_away = (None, None) if votes is None else votes

    out = np.full(len(dead), NOBODY, dtype=np.int8)
    why = np.zeros(len(dead), dtype=np.int8)  # 0 none, 1 touch, 2 fallback, 3 acquire

    for _half, first, last in blocks:
        current = NOBODY
        candidate, streak_start, streak = NOBODY, None, 0
        unseen = 0
        frame = first
        while frame <= last:
            if dead[frame]:
                current, candidate, streak, unseen = NOBODY, NOBODY, 0, 0
                frame += 1
                continue

            # -- acquiring after a restart -------------------------------------------------------
            if current == NOBODY:
                if has_answer[frame]:
                    current = nearest[frame]
                    why[frame] = 3
                elif votes_home is not None and votes_home[frame] != votes_away[frame]:
                    current = HOME if votes_home[frame] > votes_away[frame] else AWAY
                out[frame] = current
                frame += 1
                continue

            # -- touch path: the change is dated to the touch ------------------------------------
            if use_touch and touches[frame] and touch_team[frame] != current \
                    and not contested[frame]:
                stop = min(last + 1, frame + 1 + config.K_CONFIRM)
                # never confirm across a stoppage
                dead_ahead = np.flatnonzero(dead[frame + 1:stop])
                if dead_ahead.size:
                    stop = frame + 1 + int(dead_ahead[0])
                new = touch_team[frame]
                if _confirmed(nearest, has_answer, frame + 1, stop, new):
                    current = new
                    why[frame] = 1
                    candidate, streak = NOBODY, 0

            # -- fallback path: sustained evidence, backdated to where it began ------------------
            team_here = ev[frame]
            if use_fallback and team_here != NOBODY and team_here != current:
                if team_here == candidate:
                    streak += 1
                else:
                    candidate, streak_start, streak = team_here, frame, 1
                if streak >= config.K_FALLBACK_FRAMES:
                    current = candidate
                    if config.BACKDATE_FALLBACK:
                        out[streak_start:frame] = np.where(dead[streak_start:frame],
                                                           NOBODY, current)
                    why[streak_start] = 2
                    candidate, streak = NOBODY, 0
            elif team_here == current:
                candidate, streak = NOBODY, 0
            elif team_here == NOBODY and not config.FALLBACK_PAUSE_ON_UNSEEN:
                candidate, streak = NOBODY, 0

            # -- long ball-less stretch: the held label goes stale -------------------------------
            if has_answer[frame]:
                unseen = 0
            else:
                unseen += 1
                if unseen > exp01_config.HOLD_WITHOUT_BALL_FRAMES and votes_home is not None \
                        and votes_home[frame] != votes_away[frame]:
                    current = HOME if votes_home[frame] > votes_away[frame] else AWAY

            out[frame] = current
            frame += 1

    return out, why


def merge_short(codes, dead, min_frames):
    """Fold possessions shorter than `min_frames` into identical neighbours on both sides."""
    out = codes.copy()
    labels = np.where(dead, 2, out)
    edges = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate(([0], edges, [len(labels)]))
    for i in range(1, len(bounds) - 2):
        a, b = bounds[i], bounds[i + 1]
        if labels[a] in (HOME, AWAY) and b - a < min_frames \
                and labels[a - 1] == labels[b] and labels[b] in (HOME, AWAY):
            out[a:b] = labels[b]
    return out
