"""Thresholds for the touch-gated state machine, each with the work it came from.

Unlike exp01, most of these are not guesses calibrated against 4 dead stretches -- they are
published values from two papers that solved this problem on better data, and the research note in
experiments/research/ collects them. That is worth more than any sweep we could run on 17
possessions.

The one adjustment made to the published numbers: both papers worked on multi-camera optical
tracking (TRACAB, Hawk-Eye) with sub-metre ball error. Ours is automatic tactical-camera output,
uncorrected, with the ball missing on 43% of frames. The research note's advice for exactly this
case is to enlarge the metric radii -- "use metric thresholds of 1.5-2 m rather than optical-data
values of 0.5-1 m" -- and that is the only place these differ from source.
"""

# -- possession and duel zones -------------------------------------------------------------------
# Vidal-Codina et al. (MIT/FIFA, arXiv 2202.00804): a possession zone of radius R_pz around each
# player, a duel zone of radius R_dz around the ball, and **duel prevails over possession**.
#
# Their chosen radii were 0.5 m (Track160) and 1.0 m (Tracab/Hawk-Eye), searched over 0.5-1.5 m on
# FIFA experts' advice, justified by the observation that 99% of Tracab passes have a minimum
# passer-ball distance under 1 m but only 80% under 0.5 m. For broadcast-quality input the
# research note recommends 1.5-2.0 m, and our data sits between the two.
R_PZ_M = 1.5

# R_dz >= R_pz by construction. Two opponents inside it at a touch means a duel, and a duel does
# not transfer possession -- which is the same rule CHALLENGE.md states from the other direction:
# "In phases where the ball is contested or loose, possession stays with the last team that had
# it."
R_DZ_M = 2.0

# -- touch detection -------------------------------------------------------------------------------
# A touch is where the ball's trajectory bends. This is the single highest-value idea in the
# research note: Link & Hoernig (PLOS ONE 2017) report naive proximity at precision 0.68, which
# rises to 0.86 once a change of possession must coincide with a kick, and to 0.91 with ball
# prediction added. Timeline agreement goes 0.82 -> 0.87 -> 0.92.
#
# Their kick detection is a local maximum of ball acceleration at or above 4 m/s^2, on 25 Hz data
# smoothed with a Rauch-Tung-Striebel smoother. We smooth with Savitzky-Golay instead (see
# features.sg_smooth) because it needs no model of the dynamics, but the threshold is theirs.
TOUCH_ACCEL_MPS2 = 4.0

# The second, independent touch signature: the ball changes direction. Vidal-Codina use
# epsilon_theta = cos 10 degrees. A ball that bends by more than this without being touched is a
# tracking error, not football.
TOUCH_HEADING_DEG = 10.0

# Minimum ball displacement over the detection window, Vidal-Codina's epsilon_s. Without it a
# stationary ball's tracking jitter produces a heading change of any size you like, because the
# heading of a ball that is not moving is undefined.
TOUCH_MIN_DISP_M = 0.10

# A touch must happen near somebody. Outside this radius a bend in the trajectory is a bounce, a
# deflection off nobody, or noise -- not a player's touch.
TOUCH_MAX_PLAYER_DIST_M = R_DZ_M

# Frames either side used to measure the heading change. 3 frames is 0.12 s, long enough that a
# genuine deflection has turned measurably and short enough not to smear two touches together.
TOUCH_HEADING_SPAN = 3

# Minimum frames between two separate touches. Without it one kick produces a burst of touches
# across the frames where acceleration stays above threshold. 10 frames is 0.4 s, comfortably
# below the fastest real one-two.
TOUCH_REFRACTORY_FRAMES = 10

# -- Savitzky-Golay --------------------------------------------------------------------------------
# 2nd order over 7 frames, the research note's recommendation for 25 Hz, and the reason it matters
# is blunt: "At 25 Hz broadcast, smooth first (Savitzky-Golay or RTS); raw second differences are
# noise-dominated." exp01's ball speed was a raw first difference, so acceleration from it would
# have been unusable.
SG_WINDOW = 7
SG_ORDER = 2

# -- the state machine -----------------------------------------------------------------------------
# How long the new team must keep the ball after a touch before the change of possession is
# accepted. This is CHALLENGE.md's own proxy -- "a player keeps the ball for a few seconds" -- at
# its conservative end, and it is the same 25 frames exp01 used.
#
# The difference from exp01 is what happens once it is confirmed: exp01 changed the label here,
# a full second late. This backdates the change to the touch, so confirmation decides *whether*
# possession changed without also deciding *when*. That is what should move the transition metrics,
# where exp01 scores transF1 0.35 with 13 missed and 13 spurious.
# Whether a touch must be confirmed by K_CONFIRM frames of nearest-team evidence before it hands
# the ball over. True is the design as built. exp09 turns it off when its learned touch model
# supplies the touches: confirming with proximity is what favoured pressers (LEARNINGS lesson 7).
TOUCH_CONFIRM = True

K_CONFIRM = 25

# Possessions shorter than this are merged into their neighbour unless a confirmed touch bounds
# them. Research note, Rung 2. The public ground truth does contain real possessions of 0.8 s, so
# this is set below that and is a noise filter rather than a smoother.
MIN_POSSESSION_S = 0.3

# How much nearer, in metres, a team must be for a frame to count as evidence it has the ball.
#
# Held at exp01's 2.0 through exp01-exp03 so each comparison isolated one change. Swept properly
# once under-segmentation became the main error, and it turned out to be the binding constraint
# all along -- not the frame threshold everything else had been tuned on. Scored against the
# organisers' own labels:
#
#     margin   official spellF1   poss (17)   full-match poss
#     2.0      0.503              14          242
#     1.5      0.542              14          266     <- chosen
#     1.0      0.542              14          287
#     0.5      0.497              16          319
#     0.0      0.469              18          345
#
# 1.5 and 1.0 tie at the top and 1.5 is the more conservative of the two. The reason a looser
# margin helps is now understood rather than merely measured: watching the public window's
# missed possessions showed the official labels count an **interception or a clearance** as a
# possession, with no requirement that the player bring the ball under control. Those are
# exactly the contacts a 2 m margin suppresses.
#
# Note this reading contradicts a plain reading of CHALLENGE.md, which says "a simple touch of
# the ball without control" is not a change of possession. The ground truth is the authority --
# CHALLENGE.md says so itself -- and the ground truth counts them.
MARGIN_M = 1.5

# What "the new team retained the ball" means when confirming a touch: over the K_CONFIRM frames
# after it, at least this fraction of the frames where the team call can be made favour the new
# team, and at least CONFIRM_MIN_SEEN such frames exist. Without the floor, a touch followed by
# the ball vanishing for a second would confirm on one lucky frame.
CONFIRM_FRACTION = 0.6
CONFIRM_MIN_SEEN = 5

# The fallback for a change of possession no detected touch explains. Needed because the touch
# detector finds ~2000 touches and misses many -- the ball is untracked on 43% of frames -- so
# requiring a touch for every change would leave possession stuck with one team.
#
# It is exp01's rule (opposition nearer by MARGIN_M) with two differences, both earned on the
# 15-minute validation set:
#   - 3 s instead of 1 s. exp01's single biggest error class there was 14 short possessions
#     between the other team's, almost all wholly wrong in the hand labels: an opponent pressing
#     within 2 m for a second, credited with the ball, losing it again ~2 s later. The flip-backs
#     last 1.8-4.8 s. A change with no touch at all has to be sustained past that.
#   - backdated to the start of the streak, so the extra certainty costs no latency.
#
# Planned at 75 (3 s) from the flip-back durations, then set to 50 (2 s) because that scored best
# on the hand-annotated windows, then **reverted to 25 (1 s)** once those windows were themselves
# calibrated. The last step is the one worth reading.
#
# validation/calibrate.py re-annotated the public window blind and compared it with the official
# labels. The hand labels contain 11 possessions where the official ones contain 17: a human does
# not register a 0.8 s touch as a possession, while the official labels -- derived mechanically
# from corrected tracking -- do. A 35% under-count, and it systematically rewards merging. That
# is what selected 2 s.
#
# Judged by each labelling, fallback only, backdated:
#
#     K          OFFICIAL public        hand-annotated 15 min
#     25 (1 s)   0.485   16 / 17        0.421   49 / 28
#     50 (2 s)   0.462    9 / 17        0.554   37 / 28
#     75 (3 s)   0.267    7 / 17        0.557   33 / 28
#
# They disagree, and the official convention is the one the competition scores against -- the
# private ground truth is built the same mechanical way as the public file. 1 s also lands the
# possession count far closer on both: 16/17 against 9/17 officially, and 264 full-match
# possessions against the ~360 the official rate implies, where 2 s gives 203.
#
# The hand labels still earn their keep on dead ball and on second-half coverage. They cannot
# adjudicate a possession threshold, and this value is no longer chosen on them.
K_FALLBACK_FRAMES = 25

# Whether the touch path runs at all. OFF by default, because it was measured and it does not
# earn its keep on this data -- see README.md. Pooled over 20 minutes, adding it to the 2 s
# fallback took spellF1 from 0.529 to at best 0.451 (with the strictest confirmation tried) and
# the possession count from 46 to 57+; with a 3 s fallback it at best ties. It helps only on the public window, whose rapid short
# exchanges are exactly where a detected touch is informative, and hurts on all three others.
USE_TOUCH = False

# Label a fallback change from where its streak began rather than where it completed. Togglable
# so the ablation can separate what the stricter threshold earns from what the backdating earns.
BACKDATE_FALLBACK = True

# Unseen frames pause a streak rather than resetting it. With the ball missing on 43% of frames a
# 75-frame streak of *consecutive* sightings would almost never complete.
FALLBACK_PAUSE_ON_UNSEEN = True

# A ball travelling this fast with no touch is in flight, and nobody possesses it. This is the
# answer to having no z coordinate: a lofted pass passes "over" players who never touched it, and
# the research note's rule is "never assign a player when the ball keeps a constant velocity
# through their zone".
IN_FLIGHT_SPEED_MPS = 8.0
