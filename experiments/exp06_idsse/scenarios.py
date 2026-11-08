"""The what-if scenarios: the ATD-like baseline with one thing at a time made better.

Shared by whatif.py (score a fixed pipeline on each) and tune.py --scenario (re-tune the
pipeline for each). Each entry is (name, noise switches, degrade() overrides).

On the ball-missing rows: the "target" is the overall fraction of graded frames with the ball
hidden, but the simulator hides every ball that has been outside the pitch for more than about a
second, as the ATD does, and that alone is roughly 27% of frames. So 20% and 15% targets realise
as 29% and 28%. Only the in-play part of the missing ball can improve.
"""

import noise

NO_SMOOTH = dict(noise.ATD_LIKE, smooth=False)

# name -> (noise switches, degrade() overrides). The first row is the baseline everything is
# compared with.
SCENARIOS = [
    ("baseline (ATD-like)", noise.ATD_LIKE, {}),
    # -- the ball is seen more often --------------------------------------------------------
    ("ball missing: target 30%", noise.ATD_LIKE, {"ball_missing": 0.30}),
    ("ball missing: target 20%", noise.ATD_LIKE, {"ball_missing": 0.20}),
    ("ball missing: target 15%", noise.ATD_LIKE, {"ball_missing": 0.15}),
    # -- team identity gets better ----------------------------------------------------------
    ("team flips 3.5% (half)", noise.ATD_LIKE, {"flip_rate": 0.035}),
    ("team flips 0%", noise.ATD_LIKE, {"flip_rate": 0.0}),
    ("flips fixed within 1.5 m of ball", noise.ATD_LIKE, {"protect_near_m": 1.5}),
    ("flips fixed within 3 m of ball", noise.ATD_LIKE, {"protect_near_m": 3.0}),
    # -- the ball keeps its sharp kicks -----------------------------------------------------
    ("ball not smoothed (sharp kicks)", NO_SMOOTH, {}),
    # -- combinations -----------------------------------------------------------------------
    ("ball 30% + flips fixed within 1.5 m", noise.ATD_LIKE,
     {"ball_missing": 0.30, "protect_near_m": 1.5}),
    ("ball 30% + flips 0%", noise.ATD_LIKE, {"ball_missing": 0.30, "flip_rate": 0.0}),
    ("ball 30% + flips 0% + sharp kicks", NO_SMOOTH, {"ball_missing": 0.30, "flip_rate": 0.0}),
    ("everything clean (ceiling)", noise.CLEAN, {}),
]

