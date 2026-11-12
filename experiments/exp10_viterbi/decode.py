"""Viterbi decoding over the three states H, A, D.

    python experiments/exp10_viterbi/decode.py        # runs the unit test

Why a global decoder
--------------------
Every earlier experiment decides a switch locally: once the challenger has been nearer by a
margin for k frames, possession changes. That rule has one knob for two different jobs. Set it
strict and real 1-2 s possessions (interceptions, clearances, which the official labels count)
are merged into the incumbent; set it loose and every press becomes a flicker.

Viterbi looks at the whole half at once and picks the single label sequence with the best total
score: per-frame evidence (emissions.py) minus a cost for every change of state. A short
possession is kept when the evidence inside it is strong enough to pay for entering and leaving
it, and dropped when it is not -- so strength and duration trade against each other instead of
both having to clear a fixed bar.

The algorithm
-------------
delta[t, j] is the best score of any path that ends in state j at frame t. It satisfies

    delta[t, j] = E[t, j] + max_i ( delta[t-1, i] + T[t, i, j] )

where T is the log transition score (0 to stay, minus a cost to move). We remember which i won
(the back-pointer) and, at the end, walk the pointers back from the best final state. Everything
is in log space, so products of probabilities become sums and nothing underflows over 80,000
frames. Work is O(frames x 3 x 3).

States use the scorer's own codes (evaluation/possession_metrics.py): H 0, A 1, D 2.
"""

import numpy as np

H, A, D = 0, 1, 2


def viterbi(E, c_switch_to_h, c_switch_to_a, c_dead, c_restart):
    """Best state path for one contiguous stretch (one half).

    E                (n, 3) per-frame log-scores for H, A, D.
    c_switch_to_h    (n,) cost of an A -> H change landing on frame t (per frame, so Smart
                     Tagging can make a switch cheaper near a tagged transition).
    c_switch_to_a    (n,) the same for H -> A.
    c_dead           cost of live -> D (H -> D or A -> D).
    c_restart        cost of D -> live (D -> H or D -> A). The same for both teams: who restarts
                     is left entirely to the evidence that follows.

    Ties are broken towards staying in the same state, then towards the lower state index, so the
    output is deterministic.
    """
    n = len(E)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    back = np.zeros((n, 3), dtype=np.int8)
    # Plain Python floats in the loop: for a 3x3 step, numpy's per-call overhead costs more than
    # the arithmetic, and this keeps a full half to well under a second.
    eh, ea, ed = (E[:, k].tolist() for k in range(3))
    to_h, to_a = c_switch_to_h.tolist(), c_switch_to_a.tolist()
    dh, da, dd = eh[0], ea[0], ed[0]   # uniform start: a half can open live or dead
    bh, ba, bd = back[:, 0], back[:, 1], back[:, 2]
    for t in range(1, n):
        # into H: stay, switch from A, or restart from D
        s, f = dh, H
        v = da - to_h[t]
        if v > s:
            s, f = v, A
        v = dd - c_restart
        if v > s:
            s, f = v, D
        nh, bh[t] = s + eh[t], f
        # into A
        s, f = da, A
        v = dh - to_a[t]
        if v > s:
            s, f = v, H
        v = dd - c_restart
        if v > s:
            s, f = v, D
        na, ba[t] = s + ea[t], f
        # into D: from either live state pays the same cost
        s, f = dd, D
        v = dh - c_dead
        if v > s:
            s, f = v, H
        v = da - c_dead
        if v > s:
            s, f = v, A
        nd, bd[t] = s + ed[t], f
        dh, da, dd = nh, na, nd

    path = np.zeros(n, dtype=np.int8)
    final = (dh, da, dd)
    state = int(np.argmax(final))   # first maximum -> lowest index on a tie
    path[-1] = state
    for t in range(n - 1, 0, -1):
        state = back[t, state]
        path[t - 1] = state
    return path


def decode_halves(E, blocks, c_switch, c_dead, c_restart, relief_h=None, relief_a=None):
    """Full-length code array: Viterbi inside each (half, first, last) block, X (3) outside.

    relief_h / relief_a are optional (n,) fractions of c_switch forgiven on each frame (Smart
    Tagging); None means none.
    """
    n = len(E)
    out = np.full(n, 3, dtype=np.int8)   # X
    to_h = np.full(n, float(c_switch))
    to_a = np.full(n, float(c_switch))
    if relief_h is not None:
        to_h *= 1.0 - relief_h
    if relief_a is not None:
        to_a *= 1.0 - relief_a
    for _half, first, last in blocks:
        sl = slice(first, last + 1)
        out[sl] = viterbi(E[sl], to_h[sl], to_a[sl], c_dead, c_restart)
    return out


# -- unit test -----------------------------------------------------------------------------------

def _test():
    """A strong 1 s interception is kept; a weak 1 s flicker is not.

    Ten seconds of home possession, then one second where away is the evidence's choice, then ten
    more seconds of home. Each frame's evidence is a log-score gap between H and A. With a switch
    cost of 4 nats, entering and leaving the one-second away stretch costs 8 nats, so:

      strong: 25 frames x 0.6 nats = 15 nats  > 8  -> kept, and exactly where it happened
      weak:   25 frames x 0.2 nats =  5 nats  < 8  -> absorbed into the home possession
    """
    fps, c = 25, 4.0

    def run(gap):
        n = 21 * fps
        E = np.zeros((n, 3))
        E[:, D] = -10.0                      # play is clearly live throughout
        E[:, H] = 0.4                        # home nearer, moderately, most of the time
        E[10 * fps:11 * fps, H] = 0.0
        E[10 * fps:11 * fps, A] = gap        # the one-second away stretch
        cost = np.full(n, c)
        return viterbi(E, cost, cost, 30.0, 30.0)

    strong = run(0.6)
    assert (strong[:10 * fps] == H).all() and (strong[11 * fps:] == H).all(), "home misread"
    assert (strong[10 * fps:11 * fps] == A).all(), "strong 1 s interception was not kept"
    weak = run(0.2)
    assert (weak == H).all(), "weak 1 s flicker was not absorbed"

    # Dead ball: a 4 s confident stoppage is found; a 0.4 s blip of the same strength is not.
    n = 20 * fps
    E = np.zeros((n, 3))
    E[:, H] = 0.1
    E[:, D] = -3.0
    E[5 * fps:9 * fps, D] = 1.5               # p(dead) ~ 0.95 for 4 s
    E[14 * fps:14 * fps + 10, D] = 1.5        # the same for 10 frames
    cost = np.full(n, c)
    path = viterbi(E, cost, cost, 30.0, 30.0)
    assert (path[5 * fps:9 * fps] == D).all(), "4 s stoppage missed"
    assert not (path[14 * fps:14 * fps + 10] == D).any(), "0.4 s blip became a stoppage"
    print("decode.py: all tests passed (strong interception kept, weak flicker absorbed, "
          "4 s stoppage found, 0.4 s blip ignored)")


if __name__ == '__main__':
    _test()
