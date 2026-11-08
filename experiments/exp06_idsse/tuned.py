"""exp06's parameters: exp04's pipeline, tuned on degraded IDSSE. Nothing from the provided match.

Each value was chosen by tune.py on 7 IDSSE matches x 2 noise seeds, scored against IDSSE's own
BallPossession / BallStatus flags, with the tracking degraded by noise.py to look like the ATD.
The CSVs behind every number are in results/.

    stage            grid                                       best mean spellF1   held-out
    A                margin x k x touch (98 configs)            0.339               0.337
    B2 after A       slow speed x slow duration x restart
                     search x ball-lost (90 configs)            0.355               --
    A after B2       margin x k x touch, again                  0.355               0.353

For comparison on the same 14 runs: exp04's a priori values score 0.330, and the values that were
tuned on the public window (margin 1.25, k 35, touch off) score 0.309.

The surface is flat near the top -- the best ten configurations of each stage sit within 0.002 of
each other, inside seed-to-seed noise -- so where the best cell is tied, the value in the middle of
the plateau is taken rather than the cell at its edge.
"""

TUNED = {
    # possession layer
    "use_touch": True,       # every one of stage A's top 10 has it on
    "margin_m": 1.0,         # tied at 0.355 with 0.5; plateau spans 0.5-1.5, 1.0 is its centre
    "k_fallback": 75,        # 50-75 on the plateau once the touch path does the switching
    # dead-ball layer
    "slow_speed": 1.5,       # B2 best; 1.4 within 0.002
    "slow_min_frames": 500,  # past B's grid edge of 300, so B2 extended it; 500 is B2's best
    "max_dead": 250,         # flat from 250 to 500 (0.353-0.355)
    "ball_lost_frames": 125,  # insensitive: 50-125 within 0.001
}
