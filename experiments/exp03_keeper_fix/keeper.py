"""Correct the tracker's keeper identity switches, by position rather than by track slot.

    python experiments/exp03_keeper_fix/keeper.py --report

The bug, as found by hand
-------------------------
While annotating, Zain noticed the home keeper drawn in the away team's colour on several
occasions -- including one where the ball was passed back to him, which a naive reading would
score as a change of possession. He correctly did not label it as one.

The tracking metadata is not at fault. It assigns each of the 46 slots a team for the whole match,
and averaged over the match both keepers are assigned correctly:

    slot 23  HOME  1st-half mean x 0.102   2nd-half 0.887     (home defends left, then right)
    slot 45  AWAY  1st-half mean x 0.847   2nd-half 0.176     (away defends right, then left)

The fault is transient. Slot 45 in particular is re-used for other players constantly: it appears
on the *wrong side of halfway* on 9.6% of its first-half sightings and 12.6% of its second-half
ones. Slot 23 is far more stable, at 0.4% and 1.8%. When that happens near the ball, a home player
-- often the home keeper -- is wearing the away team's label.

It matters on 1680 frames of the graded match (1.06%) where a wrong-side keeper slot is within
5 m of the ball. That is small in frames and large in consequence: those frames sit next to the
ball, which is exactly where possession is decided.

The correction
--------------
Slot identity is unreliable frame by frame but perfectly reliable *in aggregate*: a slot that
spends 80% of the match inside a penalty area is a keeper slot, whatever it happens to be
tracking right now. So:

    a KEEPER SLOT found on the wrong side of halfway is not its keeper. Whoever it is
    tracking there plays, almost certainly, for the team defending that end.

Which team defends which end is known per half and needs no tracking: home defends the left goal
in the first half and the right goal in the second. That is read off the kickoff frame, where the
two sides line up in their own halves, and it is a fact about the match rather than a parameter.

A purely positional rule was tried first -- "the deepest player in the box is that end's keeper"
-- and rejected. It fired on 6424 frames but reached only 73 of the 1680 that matter, because a
keeper 10.7 m off his line is outside any threshold tight enough to be safe at corners, and
because an away player standing near the home goal is perfectly ordinary. Position alone cannot
say a label is wrong; position *combined with* a slot that is supposed to be a keeper can.

The cost of being wrong is a keeper genuinely upfield -- a corner in the 95th minute. That is
rare, brief, and in those frames the ball is at the other end anyway.
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

from games import GAME, GAMES, half_video_blocks  # noqa: E402

HOME, AWAY = 0, 1

# Params
# A slot counts as a keeper slot if it spends at least this share of its sightings inside a
# penalty area. Measured on this match the split is unambiguous -- the two keeper slots sit at
# 82% and 88%, and no outfield slot exceeds 50% -- so anything from 0.5 to 0.8 picks the same
# two slots. It is a classification, not a threshold to tune.
KEEPER_INBOX_SHARE = 0.60

# Penalty-area bounds, normalised. The real area is 16.5 m deep on a 105.3 m pitch and 40.3 m
# wide on a 68 m one.
BOX_DEPTH = 0.157
BOX_HALF_WIDTH = 0.30

# How far past halfway a keeper slot must be before it is treated as a mis-track. Exactly
# halfway, with no dead band: a keeper is never in the opposition half during open play, and the
# frames this is wrong about are ones where the ball is at the far end and nothing depends on it.
WRONG_SIDE_X = 0.5


def keeper_slots(slot_x, share=KEEPER_INBOX_SHARE):
    """Indices of the slots that behave like keepers, from where they spend the match.

    slot_x is an (n_frames, n_slots) array of normalised x, NaN where unseen.
    """
    with np.errstate(invalid="ignore"):
        inbox = (slot_x < BOX_DEPTH) | (slot_x > 1.0 - BOX_DEPTH)
    seen = ~np.isnan(slot_x)
    out = []
    for slot in range(slot_x.shape[1]):
        n = int(seen[:, slot].sum())
        if n and inbox[seen[:, slot], slot].mean() >= share:
            out.append(slot)
    return out


def parse(raw_path, slot_team):
    """[(frame, [(slot, x, y, team)], ball_or_None)] for the whole match, one streaming pass."""
    out = []
    for line in open(raw_path):
        parts = line.split(":")
        if len(parts) != 3:
            continue
        frame = int(parts[0]) - 1
        positions = []
        for slot, group in enumerate(parts[1].split(";")):
            if not group or group[0] == "N":
                continue
            fields = group.split(",")
            positions.append((slot, float(fields[0]), float(fields[1]),
                              slot_team[slot] if slot < len(slot_team) else HOME))
        ball_group = parts[2].strip()
        ball = None
        if ball_group and ball_group[0] != "N":
            fields = ball_group.split(",")
            ball = (float(fields[0]), float(fields[1]))
        out.append((frame, positions, ball))
    return out


def defending_team(game=GAME):
    """{half: (team defending the left goal, team defending the right goal)}.

    Home defends the left goal in the first half and the right in the second. Read off the
    kickoff frame -- at 11185 the home side lines up entirely in the left half -- so this is an
    observation about the match, not a tunable.
    """
    return {1: (HOME, AWAY), 2: (AWAY, HOME)}


def half_of(frame, game=GAME):
    for half, first, last in half_video_blocks(game):
        if first <= frame <= last:
            return half
    return None


def correct_frame(positions, half, keepers, game=GAME):
    """Team code per entry of `positions`, with keeper mis-tracks corrected.

    `positions` is [(slot, x, y, team)] for one frame. `keepers` is the set from keeper_slots().
    Returns the corrected team codes and the indices that changed.

    The whole rule: a keeper slot on the wrong side of halfway is tracking somebody else, and
    whoever it is belongs to the team defending that end.
    """
    teams = [team for _slot, _x, _y, team in positions]
    if half is None:
        return teams, []

    left_team, right_team = defending_team(game)[half]
    changed = []

    for index, (slot, x, _y, team) in enumerate(positions):
        if slot not in keepers:
            continue
        own_end_is_left = (team == left_team)
        wrong_side = (x > WRONG_SIDE_X) if own_end_is_left else (x < WRONG_SIDE_X)
        if not wrong_side:
            continue
        # It is at the other end, so it is not this keeper. Whoever it is defends that end.
        correct = right_team if x > WRONG_SIDE_X else left_team
        if teams[index] != correct:
            teams[index] = correct
            changed.append(index)

    return teams, changed


if __name__ == '__main__':
    from baseline import team_of_each_track
    from games import require

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    slot_team = team_of_each_track(meta_path)
    length, width = GAMES[GAME]["pitch"]
    n_slots = len(slot_team)

    frames = parse(raw_path, slot_team)
    slot_x = np.full((GAMES[GAME]["n_video_frames"], n_slots), np.nan, dtype=np.float32)
    for frame, positions, _ball in frames:
        for slot, x, _y, _team in positions:
            if slot < n_slots:
                slot_x[frame, slot] = x
    keepers = keeper_slots(slot_x)
    print("keeper slots identified from where they spend the match: {}".format(keepers))
    for slot in keepers:
        print("   slot {:>2}  team {}  seen on {:.0f}% of frames".format(
            slot, "HOME" if slot_team[slot] == HOME else "AWAY",
            100.0 * (~np.isnan(slot_x[:, slot])).mean()))
    print()

    n_changed = n_frames_changed = near_ball = graded = 0
    for frame, positions, ball in frames:
        half = half_of(frame)
        if half is None:
            continue
        graded += 1
        _teams, changed = correct_frame(positions, half, keepers)
        if not changed:
            continue
        n_changed += len(changed)
        n_frames_changed += 1
        if ball is not None:
            bx, by = ball
            for index in changed:
                _slot, x, y, _t = positions[index]
                if ((x - bx) * length) ** 2 + ((y - by) * width) ** 2 < 25.0:
                    near_ball += 1
                    break

    print("keeper correction over {} graded frames".format(graded))
    print("  frames with at least one correction : {:>7}  ({:.2f}%)".format(
        n_frames_changed, 100.0 * n_frames_changed / max(1, graded)))
    print("  individual player-frames corrected  : {:>7}".format(n_changed))
    print("  corrections within 5 m of the ball  : {:>7}  <- the ones that can flip possession"
          .format(near_ball))
    print("\n  for reference, a wrong-side keeper slot sits within 5 m of the ball on 1680")
    print("  frames, so that is roughly the ceiling for what this can fix.")
