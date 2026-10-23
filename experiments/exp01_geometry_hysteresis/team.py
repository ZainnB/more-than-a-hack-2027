"""Decide which team is in possession, by holding a state rather than classifying each frame.

    python experiments/exp01_geometry_hysteresis/team.py      # measure it on the public GT

What this replaces
------------------
starter/baseline.py answers "who has the ball?" once per frame, independently, as "whoever is
nearest to it", and then smooths the answer with a majority vote. The smoothing is the only thing
standing between that and a flickering mess, and it is a single blunt knob: at +/-0 the baseline
reports 131 possessions in a match that had around 17 in five minutes, and at +/-125 it reports 16
but loses the short ones.

The problem is not the smoothing width. It is that a possession is a *state* and the baseline has
none. So this module holds one, and changes it only on evidence:

  CONCEDE possession when the opposing team has been nearer to the ball, by at least
  config.MARGIN_M metres, for config.K_FRAMES consecutive frames.

Both conditions come out of CHALLENGE.md rather than out of a sweep. The margin is there because
"a simple touch of the ball without control" is explicitly not a change of possession, and a
player half a metre nearer to the ball than an opponent has not controlled it. The duration is
there because CHALLENGE.md's own proxy for control is "a player keeps the ball for a few seconds".

Measured on the public window, against nearest-player's 0.7784 team accuracy and 128 possessions:

    margin >= 1.0 m                 0.8078    86 possessions
    margin >= 2.0 m                 0.8189    66
    margin >= 2.0 m, k = 25         0.8627    16        <- the configured default
    margin >= 1.0 m, k = 50         0.8942     9        (over-merged, only 9 of 17 survive)

The margin is a real confidence signal and not a tuning knob: on frames where nearest-player gets
the team wrong, the median |d_home - d_away| is 1.78 m, and where it gets it right, 4.06 m.

Why the ball going missing is not a dead ball
---------------------------------------------
The ball is untracked on 43% of frames, and the baseline emits D on every one of them. That is the
single largest difference here: this module *holds* the current team instead.

It is nearly free on the ranking metric. spellF1 never sees an abstention -- the scorer forward-
fills it with the previous label before matching possessions -- so going quiet inside a possession
costs nothing, while a wrong D in the middle of one splits it in two and loses it outright.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

try:
    from . import config, features as feat
except ImportError:
    import config
    import features as feat
import load_smart_tagging as tagging  # noqa: E402
from baseline import team_names  # noqa: E402
from games import GAME, half_video_blocks, require  # noqa: E402

# Params
# Codes as they appear in the XML, upper case, matched case-insensitively.
HOME, AWAY, NOBODY = 0, 1, -1


def tag_team_votes(features, game=GAME, data_dir=None, events=None):
    """(votes_home, votes_away) per frame from the Smart Tagging codes worth trusting.

    Only config.TEAM_VOTE_CODES contribute, and read that list's comment before widening it: the
    codes left out are not merely weaker, they are at or below chance. BALL IN THE BOX attributes
    the right team on 44% of frames, which is worse than a coin flip, and PROGRESSION's 72.6% is
    below the nearest-player call this would be overriding.

    Club names are resolved against the ATD metadata rather than hardcoded, because the XML and
    the metadata are two independent spellings of the same two clubs and only one of them decides
    which side is home.
    """
    events = tagging.load(game, data_dir) if events is None else events
    home_name, away_name = team_names(require(["atd_meta"], game, data_dir)[0])
    by_name = {str(home_name).strip().lower(): HOME, str(away_name).strip().lower(): AWAY}

    n_frames = len(features["ball_x"])
    votes = np.zeros((n_frames, 2), dtype=np.int16)
    wanted = {code.strip().upper() for code in config.TEAM_VOTE_CODES}

    for event in events:
        if (event["code"] or "").strip().upper() not in wanted:
            continue
        side = by_name.get(str(event["tags"].get("Team") or "").strip().lower())
        if side is None:
            continue  # "N/A", which is every throw-in -- see DATA.md
        start = max(0, event["start_frame"])
        stop = min(n_frames, event["end_frame"])
        if stop > start:
            votes[start:stop, side] += 1
    return votes[:, HOME], votes[:, AWAY]


def possession_state(features, dead, votes=None, blocks=None, game=GAME,
                     margin_m=None, k_frames=None, reset_on_dead=True):
    """Per-frame team code (HOME / AWAY / NOBODY) from the hysteresis state machine.

    `dead` frames are left as NOBODY -- the caller turns those into D. The state machine runs
    inside each half separately and never carries a possession across the halftime gap.

    reset_on_dead drops the held team at every stoppage, so the team that restarts is decided by
    the geometry at the restart rather than inherited from before the whistle. That is the
    conservative reading of CHALLENGE.md, which says a restart has "a clear possessor" without
    saying it is the same side -- and usually it is not, since the ball normally goes to the team
    that did not put it out.
    """
    margin_m = config.MARGIN_M if margin_m is None else margin_m
    k_frames = config.K_FRAMES if k_frames is None else k_frames
    blocks = half_video_blocks(game) if blocks is None else blocks

    nearest, margin, has_answer = feat.team_margin(features)
    votes_home, votes_away = (None, None) if votes is None else votes

    out = np.full(len(nearest), NOBODY, dtype=np.int8)

    for _half, first, last in blocks:
        current = NOBODY     # the team we believe has the ball
        candidate = NOBODY   # the team currently building a case to take it
        streak = 0           # how long that case has been running
        unseen = 0           # consecutive frames without usable ball evidence

        for frame in range(first, last + 1):
            if dead[frame]:
                # A stoppage. Nothing is in possession, and the state is dropped so that the
                # restart is read from the pitch instead of inherited across the whistle.
                if reset_on_dead:
                    current = candidate = NOBODY
                    streak = unseen = 0
                continue

            if not has_answer[frame]:
                # No ball, or nobody tracked near it. Hold the possession rather than guessing.
                unseen += 1
                candidate, streak = NOBODY, 0
                if (current == NOBODY or unseen > config.HOLD_WITHOUT_BALL_FRAMES) \
                        and votes_home is not None:
                    # The held label has gone stale, or there never was one. This is the only
                    # place Smart Tagging is consulted about the team.
                    if votes_home[frame] > votes_away[frame]:
                        current = HOME
                    elif votes_away[frame] > votes_home[frame]:
                        current = AWAY
                out[frame] = current
                continue

            unseen = 0
            seen_team = nearest[frame]
            if current == NOBODY:
                # Acquiring, not conceding: the first evidence after a restart sets the state
                # outright, because there is no possession yet for the margin to protect.
                current = seen_team
                candidate, streak = NOBODY, 0
            elif seen_team != current and margin[frame] >= margin_m:
                # The opposition is nearer by a margin that means something. Count it.
                if seen_team == candidate:
                    streak += 1
                else:
                    candidate, streak = seen_team, 1
                if streak >= k_frames:
                    current, candidate, streak = candidate, NOBODY, 0
            else:
                # Either our own team is nearest, or the margin is inside the noise. Both reset
                # the case against us: a challenge has to be won continuously to count.
                candidate, streak = NOBODY, 0
            out[frame] = current

    return out


if __name__ == '__main__':
    import possession_metrics as pm
    import deadball

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=os.path.join(
        ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    parser.add_argument("--sweep", action="store_true",
                        help="print a margin/k grid instead of the configured value alone")
    args = parser.parse_args()

    features = feat.load()
    dead, _parts, _restarts = deadball.dead_mask(features)
    votes = tag_team_votes(features)

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    runs = pm.valid_runs(gt)
    graded = np.zeros(len(gt), dtype=bool)
    for start, stop in runs:
        graded[start:stop] = True
    alive = graded & ((gt == pm.H) | (gt == pm.A))

    def score(margin_m, k_frames):
        codes = possession_state(features, dead, votes, margin_m=margin_m, k_frames=k_frames)
        answered = alive & (codes >= 0)
        correct = (codes[answered] == gt[answered]).mean() if answered.sum() else 0.0
        labels = np.where(dead, pm.D, np.where(codes == HOME, pm.H,
                                               np.where(codes == AWAY, pm.A, pm.X)))
        spells = pm.possession_spells(labels.astype(np.int8), runs)
        return correct, int(answered.sum()), len(spells)

    print("team accuracy on GT-alive frames, and possessions produced in the graded window")
    print("  the public window has 17 possessions; nearest-player alone scores 0.7784 with 128\n")

    if args.sweep:
        print("  {:>8} {:>4} {:>10} {:>8} {:>12}".format("margin", "k", "accuracy", "frames",
                                                         "possessions"))
        for margin_m in (0.0, 1.0, 2.0, 3.0):
            for k_frames in (1, 12, 25, 50):
                correct, answered, spells = score(margin_m, k_frames)
                flag = "  <- configured" if (margin_m == config.MARGIN_M
                                             and k_frames == config.K_FRAMES) else ""
                print("  {:>8.1f} {:>4} {:>10.4f} {:>8} {:>12}{}".format(
                    margin_m, k_frames, correct, answered, spells, flag))
        print("\n  Do not read this grid as a leaderboard. 17 possessions cannot rank 16 cells,")
        print("  and EVALUATION.md shows a single hyperparameter picked this way choosing wrong.")
        print("  It is here to check the configured value is not perched on a cliff.")
    else:
        correct, answered, spells = score(config.MARGIN_M, config.K_FRAMES)
        print("  margin {:.1f} m, k = {} frames".format(config.MARGIN_M, config.K_FRAMES))
        print("    team accuracy  {:.4f}  over {} graded alive frames".format(correct, answered))
        print("    possessions    {}".format(spells))
