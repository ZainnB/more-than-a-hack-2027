"""exp04's overrides. Everything not listed here is inherited from exp02's config.

Two changes from exp03, both from the sweeps in README.md.
"""

# -- the evidence threshold, and why it moved -----------------------------------------------------
# exp03 ran at margin 1.5 m / 25 frames. Those were chosen separately, at different times, each
# with the other held fixed -- and they interact, which neither sweep could see.
#
# A looser margin admits weaker evidence, so it needs more of that evidence to be trusted. Swept
# jointly against both labellings, k is the parameter that matters and the margin is nearly flat
# across 0.75-1.25 m once k is right:
#
#     k      public spellF1   pooled spellF1   full-match possessions
#     25     0.555            0.347            287
#     30     0.621            0.393            258
#     35     0.629            0.396            247    <- chosen
#     40     0.629            0.389            232
#     50     0.600            0.389            208
#     60     0.462            0.343            199
#
# 30 to 50 is a broad plateau and both label sets agree on it, which is the reason to trust it;
# the single best cell is not. 1.25 m is the top of the margin's flat range, so it is the most
# conservative of the tied settings.
MARGIN_M = 1.25
K_FALLBACK_FRAMES = 35

# -- boundary refinement ---------------------------------------------------------------------------
# Snap each live possession boundary onto the nearest touch by the incoming team. See refine.py.
#
# This barely earns its place: +0.013 on the official window, flat pooled. It is kept because the
# gain is consistent in sign across every k in the plateau and it cannot change the possession
# count by construction, so it carries no risk of the kind the threshold does. It is not what
# made exp04 better -- the sweep above is.
#
# The search radius is the load-bearing part. At +/-8 frames it helps slightly; at +/-50 it
# clearly hurts (pooled 0.353 -> 0.340), because a boundary allowed to travel two seconds finds
# the wrong touch.
SNAP_MODE = "touch"
SNAP_WINDOW = 8
