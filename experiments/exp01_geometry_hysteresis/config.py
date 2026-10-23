"""Every threshold this experiment uses, in one place, with the reason for its value.

The reasons matter more than the numbers. The public ground truth is 5 minutes and 17
possessions, and EVALUATION.md demonstrates that tuning even a single hyperparameter against it
picks the wrong value: baseline_smooth125 wins the public window and loses the full match. So
nothing here was chosen by grid search over the public window. Each value is either a physical
quantity (a player's reach, a restart's duration) or a rule stated in CHALLENGE.md, and the
public window is used only to check that the value is not sitting on a cliff.

If you change one of these, change it for a reason you can write down, and re-check the
possession count rather than the score -- the count is the signal the 17-possession window can
actually support.
"""

# -- pitch geometry ----------------------------------------------------------------------------

# The ATD gives normalised 0-1 coordinates, so "outside the pitch" is literally outside [0, 1].
# Measured on this match: 98.1% of frames where the tracked ball is outside the unit square are
# dead in the public ground truth, which makes this the highest-precision signal in the data.
#
# No tolerance band is added. A band of 0.01 drops precision from 98.1% to 45.8%, because a ball
# genuinely in play runs along the touchline constantly. The hard boundary is the whole value.
OUT_OF_BOUNDS_MARGIN = 0.0

# Ball speed, in m/s, above which the ball counts as *struck* rather than carried or placed.
#
# Read features.ball_speed_mps() before changing this. The ATD has a third field for the ball and
# it is NaN on all 113488 frames where the ball is seen -- the column exists and is empty -- so
# this speed is differenced from consecutive positions instead. Player speeds are populated; the
# ball's are not.
#
# 6 m/s separates the two things that happen after the ball leaves the pitch. Measured on the goal
# kick at video frame 18389: the ball is retrieved and carried back at 0.6-3.7 m/s for ten
# seconds, then struck at 7.1 m/s. A player walking the ball into position cannot reach 6 m/s;
# a throw-in or goal kick comfortably exceeds it.
BALL_RESTART_SPEED_MPS = 6.0

# A restart is a ball that was *settled and then struck*, not merely a ball moving fast, and this
# is the speed below which it counts as settled. Both halves of that are needed: at the goal kick
# at frame 18389 the ball is thrown back towards the keeper at up to 6.5 m/s, which clears the
# strike threshold on its own and ends the dead stretch 262 frames early. Requiring it to have
# been slow first rejects a ball in transit and accepts a ball on the spot.
#
# Every restart in football is taken from a stationary ball -- throw-in, goal kick, corner, free
# kick, kickoff -- so this costs no generality.
BALL_SETTLED_SPEED_MPS = 2.0

# How long the ball must have been settled before a strike counts as a restart. 25 frames is 1 s.
BALL_SETTLED_FRAMES = 25

# -- dead-ball stretch bounds ------------------------------------------------------------------

# The shortest plausible stoppage. The public ground truth's shortest dead stretch is 138 frames
# (5.5 s) and a quick throw-in is faster than that, so 50 frames (2 s) sits below anything real
# and only suppresses single-frame tracking noise.
MIN_DEAD_FRAMES = 50

# How far past the ball leaving the pitch the restart may be searched for.
#
# Originally 1500 (60 s), chosen to be past every non-goal stoppage. That was too generous: when
# a throw-in is taken quickly, without the ball visibly settling, find_restart scans past the real
# restart and locks onto a later one, and a single missed restart then marks up to a minute of
# live play dead. On the hand-annotated windows that produced stoppages of 24, 34 and 44 s and
# dropped the oob rule's precision to 51.9%.
#
# 375 frames (15 s) bounds the damage while still covering an ordinary stoppage. Measured against
# both labellings of the data, dead-ball precision / recall:
#
#     cap      official public      hand-annotated 15 min
#     1500     97.5% / 74.5%        70.6% / 89.0%
#      500     97.5% / 74.5%        76.8% / 84.1%
#      375     97.5% / 74.5%        83.7% / 84.5%    <- chosen
#      250     95.1% / 59.0%        90.9% / 83.7%
#
# 375 is the last value that costs the official labels nothing at all. Below it, recall falls off
# a cliff there -- real stoppages genuinely run longer than 10 s -- even though the hand labels
# keep improving, because they end dead balls about 0.9 s early (see validation/calibrate.py).
#
# Goal celebrations exceed this and are covered by the slow-pitch rule instead.
MAX_DEAD_FRAMES = 375

# How long the ball must be untracked, together with a slow pitch, before the ball-lost fallback
# calls a dead ball. The baseline's unconditional version -- "no ball, therefore D" -- is only
# 55.8% precise, because the tracker loses the ball constantly during open play. Requiring 3
# seconds of loss AND a slow pitch is what makes it usable, and it is deliberately the weakest of
# the three dead-ball rules.
BALL_LOST_DEAD_FRAMES = 75

# After a restart has been found, play is alive by definition for at least this long, and the
# dead-ball rules are overruled across it. The slow-pitch rule is wrong in exactly one
# predictable place: the players stand still in formation waiting for a kickoff, so the pitch
# reads as walking at the precise moment play begins.
RESTART_ALIVE_FRAMES = 50

# Mean speed of all 22 tracked players, in m/s, below which the pitch counts as "walking".
#
# This is the foul detector, and it is the only rule here that catches a stoppage with the ball
# still on the pitch. Players slow down together during a stoppage and at no other time: 1.3 m/s
# is walking pace, against a median of 1.52 m/s over graded frames.
#
# Unlike most thresholds measured on 4 dead stretches, this one is on a plateau rather than a
# spike -- across smoothing windows of 25/50/75 frames and thresholds of 1.1 to 1.3, precision
# stays in a 78-93% band. That stability is the reason to trust it; the single best cell in that
# sweep is not.
SLOW_PITCH_SPEED = 1.3

# Frames to average the pitch speed over before testing it. 50 frames is 2 s, enough that one
# player's tracking glitch cannot make the squad look stationary.
SLOW_PITCH_WINDOW = 50

# How long the pitch must stay slow to count as a stoppage rather than a lull. 200 frames is 8
# seconds. Play genuinely slows -- a keeper holding the ball, a back line passing sideways -- so
# the duration is what separates a stoppage from a quiet passage, and shortening it is the fastest
# way to start calling live play dead.
SLOW_PITCH_MIN_FRAMES = 200

# -- possession hysteresis ---------------------------------------------------------------------

# How much nearer, in metres, the opposing team must be to the ball before possession is conceded.
# This is the core of the experiment. A player's reach is about a metre and a tackle happens
# inside two, so below this distance "who is nearest" is not evidence about who is in control.
#
# It is also what the data says: on frames where nearest-player gets the team wrong, the median
# |d_home - d_away| is 1.78 m; where it gets it right, 4.06 m. A third of its errors are on
# frames where the margin is under a metre.
MARGIN_M = 2.0

# How many consecutive frames the opposing team must hold that margin before possession changes.
# CHALLENGE.md's own proxy for controlled possession is "a player keeps the ball for a few
# seconds", and this is the conservative end of that: 25 frames is exactly 1 second at 25 fps.
#
# This is the knob that decides the possession count, so it is the one to be most careful with.
# Raising it merges real possessions; lowering it splits them. At 1 s the public window produces
# 16 possessions against a true 17, which is the right order without having been fitted to it.
K_FRAMES = 25

# How long the ball can stay untracked before the held possession label stops being trusted and
# the Smart Tagging vote is consulted instead. Below this, holding is strictly better: an
# abstention inside a real possession is free on spellF1, while a wrong switch splits it in two.
# 100 frames is 4 seconds, by which point the ball could be anywhere.
HOLD_WITHOUT_BALL_FRAMES = 100

# -- Smart Tagging -------------------------------------------------------------------------------

# Codes whose Team tag is reliable enough to vote on the team in possession, measured against the
# public ground truth on GT-alive frames:
#
#     BUILD UP            100.0%   (1 instance in the public window)
#     CREATING CHANCES    100.0%   (1 instance)
#     SET PIECES           96.2%   (3 instances)
#     LONG BALL            90.2%   (1 instance)
#
# Read those instance counts before trusting the percentages -- three of the four rest on a single
# tagged interval. They are used only as a tiebreak on frames where the ball is not tracked at
# all, never to override a tracked ball, which is why a one-instance sample is acceptable here.
TEAM_VOTE_CODES = ("BUILD UP", "CREATING CHANCES", "SET PIECES", "LONG BALL")

# Codes deliberately NOT used for the team call, and why:
#
#     PROGRESSION          72.6%   below the baseline's own 85.8% nearest-player accuracy
#     BALL IN FINAL THIRD  53.7%   coin flip
#     PLAYERS IN THE BOX   45.9%   coin flip
#     BALL IN THE BOX      44.0%   coin flip
#     DEFENDING IN *               the same intervals mirrored -- each one's accuracy sums to
#                                  exactly 100% with its attacking twin, so it carries no
#                                  independent information
#
# Kept as a list rather than a comment so the reasoning survives a refactor.
TEAM_VOTE_REJECTED = ("PROGRESSION", "BALL IN FINAL THIRD", "PLAYERS IN THE BOX",
                      "BALL IN THE BOX", "ATTACKING TRANSITION", "DEFENSIVE TRANSITION",
                      "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
                      "DEFENDING IN DEFENSIVE THIRD", "GOALS")

# Seed the possession after a detected restart from the set piece's Team tag.
#
# OFF by default, on purpose. The agreed rule for this experiment is that Smart Tagging informs
# the team call only where the ball is not tracked, and a restart usually has a tracked ball. The
# lever is here because it is the most promising single extension -- restarts are exactly where
# spellF1's transitions live, and the SET PIECES Team tag is 96.2% accurate -- but turning it on
# is a separate experiment with its own measurement, not a default.
#
# Note before trying it: every Throw-in (26 of the 59 set pieces) is tagged Team = N/A, so this
# can only ever seed corners, free kicks and goal kicks.
USE_TAG_TEAM_AT_RESTART = False

# -- abstention --------------------------------------------------------------------------------

# Write X rather than a guess where the system has no evidence at all.
#
# This is cheap on the ranking metric and worth understanding before changing: spellF1 never sees
# an X, because the scorer forward-fills it with the last label in the half before matching
# possessions. So an X inside a possession costs nothing, while a wrong D there splits one real
# possession into two and costs a match. macroF1 does charge an X as a miss, but macroF1 is the
# tie-break, not the ranking.
ABSTAIN_OUTSIDE_HALVES = True
