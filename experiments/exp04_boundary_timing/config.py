"""exp04's overrides: the pipeline with every match-tuned value removed.

Everything not listed here is inherited from exp01's and exp02's configs. Those two files are left
as they were, as the record of what those experiments did; exp04 overrides them at import (see
run.py) so it can run without any value that was chosen by scoring against this match.

The rule, from experiments/PLAN.md section 1: the public GT and our hand-annotated windows may
measure the pipeline, but no value, threshold or on/off switch may be chosen because it scored
better on them. Each value below was chosen that way, and each one goes back to the value it had
*before* it was scored, which was itself set for a reason that never used the labels.

Kept as inherited, because they never came from a score: the published touch and zone values
(Link & Hoernig, Vidal-Codina), the Savitzky-Golay settings, K_CONFIRM, MIN_POSSESSION_S, the
restart speeds (read off the tracking at one goal kick, not scored), MIN_DEAD_FRAMES, the
slow-pitch window and duration, the keeper classification (an automatic check on the provided
tracking, which PLAN.md section 1 allows), and pitch geometry.

One judgement call is flagged rather than reverted: exp01's SLOW_PITCH_SPEED = 1.3 m/s. It was
re-checked against the public dead stretches after a parsing bug, but the value is human walking
pace, which is where it would be set from first principles too. There is no earlier, unscored
value to go back to.
"""

# -- exp02 overrides -------------------------------------------------------------------------------
# Possession evidence margin. 1.5 m (exp02) and 1.25 m (exp04 as first shipped) were both picked by
# sweeping against the official public labels. 2.0 m is the value it had before any sweep, from
# physics alone: a player's reach is about a metre and a tackle happens inside two, so below 2 m
# "who is nearest" is not evidence of who is in control.
MARGIN_M = 2.0

# Frames of sustained evidence before a change. 35 was picked from the same joint sweep. 25 frames
# (1 s) is CHALLENGE.md's own proxy for controlled possession, "a player keeps the ball for a few
# seconds", at its conservative end. It is the value exp01 started from, before anything was
# scored.
K_FALLBACK_FRAMES = 25

# The touch path. exp02 switched it off because it scored worse on the hand-annotated windows,
# which is selecting a component by its score on the match. Restored to the design it was built
# from: Link & Hoernig report that requiring a change of possession to coincide with a detected
# kick raises precision from 0.68 to 0.86, and every threshold in it is published.
USE_TOUCH = True

# -- exp01 overrides -------------------------------------------------------------------------------
# How far past the ball leaving the pitch to search for the restart. 375 was picked from a sweep
# against both labellings ("the last value that costs the official labels nothing"). 1500 frames
# (60 s) is the original value: past every non-goal stoppage.
MAX_DEAD_FRAMES = 1500

# Smart Tagging codes allowed to vote on the team in possession, used only where the ball is not
# tracked. exp01's list was selected by measuring each code's accuracy against the public GT,
# which is feature selection on the GT. Replaced by the a priori list: the codes whose Team tag
# names the team *in possession* by definition -- the attacking phases, and set pieces, whose Team
# is the side taking them. The DEFENDING IN * codes name the other team and are excluded for that
# reason, not because of a score.
TEAM_VOTE_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "LONG BALL", "SET PIECES")

# -- boundary refinement ---------------------------------------------------------------------------
# Snap each possession boundary onto a nearby touch. See refine.py. Off: the refinement has no
# setting that was chosen without the match labels. It was kept only for a +0.013 gain on the
# official window, its +/-8 frame radius was picked by score, and refine.py's own default of 50
# frames was sized to the boundary spread measured against the hand labels. With tuning removed
# it has nothing to stand on, so exp04 runs it as "none".
SNAP_MODE = "none"
SNAP_WINDOW = 8
