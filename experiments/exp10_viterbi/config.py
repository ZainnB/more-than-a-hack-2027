"""exp10's parameters. Every value has a one-line reason; none was chosen by scoring the match.

Two kinds of value live here:

  A PRIORI   set from physics, the laws of the game, or the shape of the metric, before anything
             was scored. These never change.
  IDSSE      chosen by idsse_eval.py on 7 degraded Bundesliga matches x 3 noise seeds, leave-one-
             match-out, against exp08 on the identical inputs. The A priori fallback for each is
             kept beside it, and is what run.py uses if the IDSSE data is not downloaded.

How the costs and the emissions trade off -- the one idea needed to read the numbers below
------------------------------------------------------------------------------------------
Viterbi keeps a short possession of L frames, wedged between two possessions of the other team,
exactly when the evidence it collects over those L frames beats paying two switches:

        sum over the L frames of (E_new - E_old)   >   2 * C_SWITCH

So C_SWITCH is measured in "nats of evidence", and the emissions say how many nats a frame is
worth. A frame where one team is clearly nearer is worth up to 4 * HA_SCALE nats (see
emissions.py); a frame where the ball is unseen is worth 0. With HA_SCALE 0.1 and C_SWITCH 4, a
1 s interception needs about 20 clearly-won frames out of its 25 to survive, and a 1 s flicker
where the challenger is only marginally nearer (a few tenths of a nat a frame) is absorbed.
That is the trade a local "margin + k frames" rule cannot make: here strength and duration
substitute for each other.
"""

# -- emissions: home vs away (A PRIORI) ----------------------------------------------------------

# A difference in nearest-player distance beyond this says nothing more about control: a tackle
# happens inside 2 m (exp04's physical argument for its margin), so 2 m nearer is "clearly nearer"
# and 10 m nearer is no more so. Clipping also stops one frame dominating a whole possession.
Z_MAX_M = 2.0

# Softness of the logistic on the signed margin. Half a metre is about the tracking's position
# noise near the ball, so a margin of 0.5 m is one "unit" of evidence and a near tie is ~neutral.
Z_SOFT_M = 0.5

# If even the nearest player is further than this from the ball, nobody is playing it (a pass or
# clearance is travelling); which team happens to be nearer is positioning, not possession.
# 5 m is two strides plus reach, and matches exp09's candidate radius (4 m) plus tracking noise.
FAR_M = 5.0

# Ball speeds above this mean the ball is in flight, so proximity is not evidence. Imported, not
# re-set: exp02's IN_FLIGHT_SPEED_MPS (8 m/s, a firm pass) -- see emissions.py.

# -- emissions: dead vs live (A PRIORI) ----------------------------------------------------------

# exp08's classifier probability is clipped away from 0 and 1 so one frame cannot contribute
# more than ~7 nats; the classifier is calibrated on IDSSE but not to the third decimal.
P_CLIP = 1e-3

# Weight on log p(dead) vs log(1 - p(dead)). 1.0 = take the classifier's log-odds at face value;
# the D-side costs below set how much of it is needed, so a second scale would be redundant.
W_DEAD = 1.0

# The probability is smoothed over the same +-12 frames exp08's decode uses (chosen on IDSSE by
# deadclf.py), read from model.json at run time rather than copied here.

# -- touch bonus (exp09) -------------------------------------------------------------------------

# Only candidates the model believes (p > 0.5) add evidence, and they add only for their own team:
# a weak candidate is "no touch seen", not "the other team touched it".
TOUCH_MIN_P = 0.5

# A candidate's bonus is spread over +-12 frames (+-0.5 s), because on the smoothed ATD ball the
# contact is only located to within about a second (exp09 README, "Candidates").
TOUCH_SPREAD = 12

# -- Smart Tagging transitions (A PRIORI only: IDSSE has no tagging, so nothing can tune it) ----

# Window around an ATTACKING / DEFENSIVE TRANSITION start in which switching to the team gaining
# the ball is cheaper. 3 s: the tagger marks phases to a few seconds, not frames.
TAG_WINDOW_S = 3.0

# Fraction of the switch cost forgiven inside that window. A half: a tag is evidence, not an
# answer (DATA.md), so it can tip a close call but never create a switch on its own.
TAG_RELIEF = 0.5

# -- transition costs and scales (IDSSE; a priori fallback alongside) ---------------------------

# The grid idsse_eval.py searches. Only ratios matter for H vs A (C_SWITCH / HA_SCALE is "frames
# of clear evidence per switch"), so HA_SCALE is held to two values and C_SWITCH spans a factor
# of 16. C_DEAD spans "one second of confident dead" to "several seconds".
#
# The first grid (c_switch 1-16, ha_scale 0.1/0.25, c_dead 10-60, touch_gain 0-4) put its best
# cell on three edges at once (ha_scale 0.1, c_dead 60, touch_gain 4), so, as exp06 and exp09 did,
# it was extended past every edge before anything was chosen. The first grid's CSV is kept as
# results/idsse_grid_first.csv.
GRID = {"c_switch": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        "ha_scale": [0.05, 0.1, 0.25],
        "c_dead": [10.0, 30.0, 60.0, 120.0],
        "touch_gain": [0.0, 2.0, 4.0, 8.0]}

A_PRIORI = {
    # 1 s of clearly-won ball, with ~40% of it unseen, is ~15 frames x 0.4 nats = 6 nats; a real
    # 1 s interception should just survive, so two switches must cost a little less: 2 x 2.5.
    # Rounded to the grid's 2.0 / 4.0 neighbourhood -> 4 with HA_SCALE 0.1 (0.4 nats/frame).
    "c_switch": 4.0,
    "ha_scale": 0.1,
    # exp08's decode drops stoppages shorter than 100 frames; at a typical smoothed p of 0.8
    # (log-odds 1.4) a 1 s stoppage earns ~35 nats, a 4 s one ~140. 30 per edge (60 in and out)
    # keeps the shortest real whistle stoppages (~2 s) and drops sub-second blips.
    "c_dead": 30.0,
    # A confident touch (p 0.9, log-odds 2.2) times 2 is ~4.4 nats: worth about one switch, so
    # a touch plus a little proximity flips possession, a touch alone on noise does not.
    "touch_gain": 2.0,
}

# Values chosen on IDSSE by idsse_eval.py: the best mean spellF1 over all 7 matches x 3 seeds
# (macroF1 breaks ties), from results/idsse_grid.csv. The extended grid's best is still on two
# edges (ha_scale 0.05, touch_gain 8), but the top cells are near-copies of each other scaled
# together -- (4, 0.05, 60, 4) and (8, 0.1, 60, 8) score within 0.001 -- so the surface is a flat
# ridge, not a slope running off the grid, and extending again would only walk along it.
# Held-out gain over exp08: +0.004 spellF1, 4/7 matches, inside seed noise. See README.md.
CHOSEN = {"c_switch": 8.0, "ha_scale": 0.05, "c_dead": 60.0, "touch_gain": 8.0}


def chosen():
    """The values run.py uses: IDSSE's choice if it exists, else the a priori fallback."""
    return dict(CHOSEN) if CHOSEN else dict(A_PRIORI)
