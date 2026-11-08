"""Check the team label of every player near the ball against the shirts in the same frame.

    python experiments/exp07_team_colour/decide.py          # label-free summary

Same-frame colour consensus. Nothing is fitted to the video:

  1. Start from the ATD's team labels, with exp03's goalkeeper correction applied.
  2. For each frame, the other players (not goalkeepers, not the near-ball candidates, only crops
     that are mostly body) are split into the two colour groups they form in that frame, and each
     group is named by which team label is more common in it. A plain per-team median was the
     first version, and it broke exactly where it mattered: early in the second half the ATD labels
     15 players home and 6 away, the home median became a mix of both kits, and real Barcelona
     players were then "neither" and dropped. The split uses this frame only and keeps nothing, so
     nothing is fitted to the video.
  3. Each player within CANDIDATE_M of the ball is then checked:
       ghost     the box is almost all grass. Counted, but the label is kept.
       neither   far from both references (the referee in yellow). Counted, label kept.
       flip      at least RATIO times closer to the other team's colour AND within PURITY of
                 the other team's reference, so the crop looks like a typical member of that
                 team rather than a mix of two shirts. Team swapped.
     Ghost and neither originally dropped the player. Looking at a random sample, three of four
     drops were real players: in a close camera zoom, and lying on the ground. Removing a real
     player is worse than keeping a wrong label, so they now only count.
       keep      otherwise, including every ambiguous case. The step can abstain, never invent.
     A candidate whose torso overlaps another player's box in the image is not judged at all: in a
     duel the crop catches the neighbour, which is how the first version flipped players wrongly.
  4. A per-track majority over +/- SMOOTH_FRAMES frames, so one bad crop cannot flicker a label.

The thresholds below are set a priori, conservatively, and are not tuned on our video or labels.
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

import colour  # noqa: E402
import load_homography as homography  # noqa: E402
import players as players_mod  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from exp03_keeper_fix import keeper  # noqa: E402

# Params
CANDIDATE_M = colour.CANDIDATE_M
RATIO = 0.5            # "clearly closer": at least twice as close to the other team's colour
NEITHER = 1.0          # farther from both references than they are from each other
GHOST_FILL = 0.15      # under 15% of the torso band is not grass
MIN_REFERENCE = 3      # players per team needed to form a reference
MIN_SEPARATION = 0.15  # references closer than this: the frame cannot tell the kits apart
SMOOTH_FRAMES = 25     # +/- 1 s
SPLIT_ROUNDS = 10      # 2-means iterations per frame
PURITY = 0.35          # within this fraction of the reference separation of the new team
DROP = -1              # team code for "not a player here"
APPLY_DROPS = False

# Descriptor -> comparable vector. Chroma only (see colour.py); a and b are scaled so a typical
# kit difference is the same order as one in saturation or whiteness.
def vector(desc):
    return np.stack([desc[..., 0], desc[..., 3], desc[..., 1] / 20.0, desc[..., 2] / 20.0],
                    axis=-1)


def references(vecs, labels):
    """Two reference colours from one frame's players, named by the ATD labels' majority.

    2-means seeded from the per-label medians, then each group is called home or away by which
    has the larger share of home labels. Returns None if a group ends up with fewer than
    MIN_REFERENCE players.
    """
    centres = np.array([np.median(vecs[labels == t], axis=0) for t in (0, 1)])
    for _ in range(SPLIT_ROUNDS):
        d = np.linalg.norm(vecs[:, None, :] - centres[None, :, :], axis=2)
        group = d.argmin(axis=1)
        if min((group == 0).sum(), (group == 1).sum()) < MIN_REFERENCE:
            return None
        new = np.array([np.median(vecs[group == g], axis=0) for g in (0, 1)])
        if np.allclose(new, centres):
            break
        centres = new
    home_share = [np.mean(labels[group == g] == 0) for g in (0, 1)]
    if home_share[0] == home_share[1]:
        return None
    home = int(np.argmax(home_share))
    return [centres[home], centres[1 - home]]


def boxes(H, xs, ys, length):
    """(x0, y0, x1, y1, torso_y0, torso_y1) image boxes for every slot on one frame."""
    u, v = homography.project(H, xs, ys)
    u2, _ = homography.project(H, xs + 1.0 / length, ys)
    ppm = np.abs(u2 - u)
    w, h = colour.BOX_WIDTH_M * ppm, colour.BOX_HEIGHT_M * ppm
    top = v - h
    return u - w / 2, top, u + w / 2, v, top + colour.TORSO[0] * h, top + colour.TORSO[1] * h


def overlapped(b, s, others):
    """Does slot s's torso band overlap any other slot's whole box?"""
    x0, y0, x1, y1, t0, t1 = b
    for o in others:
        if o == s or np.isnan(x0[o]):
            continue
        if x0[s] < x1[o] and x0[o] < x1[s] and t0[s] < y1[o] and y0[o] < t1[s]:
            return True
    return False


def base_teams(p, game=GAME):
    """(frames, slots) team codes: the ATD's, with exp03's keeper correction."""
    n, slots = p["x"].shape
    team = np.broadcast_to(p["team"], (n, slots)).copy()
    keepers = keeper.keeper_slots(p["x"])
    sides = keeper.defending_team(game)
    for half, first, last in half_video_blocks(game):
        left, right = sides[half]
        sl = slice(first, last + 1)
        for k in keepers:
            x = p["x"][sl, k]
            own_left = p["team"][k] == left
            with np.errstate(invalid="ignore"):
                wrong = (x > keeper.WRONG_SIDE_X) if own_left else (x < keeper.WRONG_SIDE_X)
                correct = np.where(x > keeper.WRONG_SIDE_X, right, left)
            team[sl, k] = np.where(wrong, correct, team[sl, k])
    return team, keepers


def decide(game=GAME, data_dir=None):
    p = players_mod.load(game, data_dir)
    c = colour.load(game, data_dir)
    matrices, _err, _valid = homography.load(game, data_dir)
    team, keepers = base_teams(p, game)
    length, width = GAMES[game]["pitch"]
    frames, desc = c["frames"], c["desc"]
    vec = vector(desc)
    fill = desc[..., 4]
    is_keeper = np.zeros(p["x"].shape[1], dtype=bool)
    is_keeper[keepers] = True

    raw = {}  # slot -> list of (frame, verdict) ; verdict: 0/1 team, DROP, or None (keep)
    stats = {"frames": len(frames), "no_reference": 0, "too_similar": 0, "candidates": 0,
             "overlap": 0, "ghost": 0, "neither": 0, "flip": 0}
    for i, f in enumerate(frames):
        dx = (p["x"][f] - p["ball_x"][f]) * length
        dy = (p["y"][f] - p["ball_y"][f]) * width
        dist = np.hypot(dx, dy)
        cand = np.flatnonzero(np.nan_to_num(dist, nan=np.inf) <= CANDIDATE_M)
        usable = ~np.isnan(vec[i, :, 0]) & (np.nan_to_num(fill[i]) >= 0.3) & ~is_keeper
        usable[cand] = False
        members = np.flatnonzero(usable & np.isin(team[f], (0, 1)))
        refs = []
        if (team[f, members] == 0).sum() >= MIN_REFERENCE and \
                (team[f, members] == 1).sum() >= MIN_REFERENCE:
            refs = references(vec[i, members], team[f, members]) or []
        if len(refs) < 2:
            stats["no_reference"] += 1
            continue
        separation = float(np.linalg.norm(refs[0] - refs[1]))
        if separation < MIN_SEPARATION:
            stats["too_similar"] += 1
            continue
        b = boxes(matrices[f], p["x"][f].astype(np.float64), p["y"][f].astype(np.float64), length)
        tracked = np.flatnonzero(~np.isnan(p["x"][f]))
        for s in cand:
            if is_keeper[s]:
                continue
            stats["candidates"] += 1
            if overlapped(b, s, tracked):
                stats["overlap"] += 1
                continue
            own = team[f, s]
            if own not in (0, 1):
                continue
            verdict = None
            if np.nan_to_num(fill[i, s], nan=0.0) < GHOST_FILL:
                verdict = DROP if APPLY_DROPS else None
                stats["ghost"] += 1
            elif not np.isnan(vec[i, s, 0]):
                d = [float(np.linalg.norm(vec[i, s] - r)) for r in refs]
                if min(d) > NEITHER * separation:
                    verdict = DROP if APPLY_DROPS else None
                    stats["neither"] += 1
                elif d[1 - own] < RATIO * d[own] and d[1 - own] < PURITY * separation:
                    verdict = 1 - own
                    stats["flip"] += 1
            raw.setdefault(s, []).append((f, verdict))

    # per-track majority over +/- SMOOTH_FRAMES, counting "keep" as a vote for the base label
    out = team.copy()
    changed = 0
    for s, entries in raw.items():
        fr = np.array([e[0] for e in entries])
        code = np.array([-9 if e[1] is None else e[1] for e in entries])
        for j, f in enumerate(fr):
            window = code[np.abs(fr - f) <= SMOOTH_FRAMES]
            values, counts = np.unique(window, return_counts=True)
            winner = values[np.argmax(counts)]
            if winner != -9:
                out[f, s] = winner
                changed += 1
    stats["changed_after_smoothing"] = changed
    return out, team, stats


if __name__ == '__main__':
    fixed, base, stats = decide()
    print("frames examined         {}".format(stats["frames"]))
    print("  no reference (<{} per team)  {}".format(MIN_REFERENCE, stats["no_reference"]))
    print("  kits too similar        {}".format(stats["too_similar"]))
    print("near-ball checks        {}".format(stats["candidates"]))
    for k in ("overlap", "flip", "neither", "ghost"):
        print("  {:<8} {:>7}  {:5.2f}%".format(k, stats[k], 100.0 * stats[k] / max(1, stats["candidates"])))
    print("player-frames changed after smoothing  {}  ({:.2f}% of near-ball checks)".format(
        stats["changed_after_smoothing"], 100.0 * stats["changed_after_smoothing"] / max(1, stats["candidates"])))
