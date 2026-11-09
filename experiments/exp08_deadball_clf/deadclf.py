"""Train a dead-ball classifier on IDSSE's BallStatus, and compare it with the hand-tuned rules.

    python experiments/exp08_deadball_clf/deadclf.py          # train, evaluate, write model.json

The rules exp01-exp06 use for stoppages (ball out, a slow pitch for 20 s, a lost ball on a slow
pitch) are hand-written and carry hand-tuned durations. exp06 tuned them as far as they go and
still misses short whistle stoppages. This replaces them with a model trained on external data:
per-frame features from deadfeat.py, labels from IDSSE's own BallStatus, tracking degraded by
exp06's simulator (at exp08's realistic 1.5% team-flip rate) so it learns from what the ATD looks
like rather than from clean optical tracking.

Model: L2-regularised logistic regression, fitted by Newton's method in numpy, so the submission
needs no new dependency to run it. Probabilities become stoppages by smoothing, a threshold, and
minimum lengths; those decode settings are chosen on IDSSE by leave-one-match-out, never on the
provided match.

Leave-one-match-out throughout: the model and the decode settings scored on a match never saw it.
"""

import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import deadfeat  # noqa: E402
import harness  # noqa: E402
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import tune  # noqa: E402
import possession_metrics as pm  # noqa: E402
from exp01_geometry_hysteresis import deadball  # noqa: E402

MODEL_PATH = os.path.join(HERE, "model.json")
FLIP_RATE = 0.015
SEEDS = (0, 1)
STRIDE = 5          # training subsample: every 5th frame (0.2 s apart, still ~400k rows)
L2 = 1.0
HALF = (adapter.PITCH[0] / 2.0, adapter.PITCH[1] / 2.0)

DECODE_GRID = {"tau": [0.3, 0.4, 0.5, 0.6],
               "smooth": [0, 12, 25, 50],      # frames either side of the moving average
               "min_dead": [25, 50, 100],      # shortest stoppage kept
               "min_live": [12, 25, 50]}       # shortest live stretch kept between stoppages

# Hand-tuned rule values from exp08's re-tune, for the comparison.
RULE_PARAMS = {"slow_speed": 1.5, "slow_min_frames": 500, "max_dead": 250, "ball_lost_frames": 125}


def dataset(matches=None, seeds=SEEDS):
    """{(match, seed): (X, y, graded, blocks, adapter_features)} on degraded IDSSE."""
    out = {}
    for match in matches or idsse.matches():
        s = tune.cached_scene(match)
        for seed in seeds:
            ball, players, team = noise.degrade(s, seed, noise.ATD_LIKE, flip_rate=FLIP_RATE)
            X = deadfeat.compute(ball, players, *HALF)
            graded = s["gt"] != pm.X
            out[(match, seed)] = (X, s["gt"] == pm.D, graded, s["blocks"],
                                  adapter.to_features(ball, players, team))
    return out


def fit(X, y, l2=L2, rounds=30):
    med = np.nanmedian(X, axis=0)
    Xf = np.where(np.isnan(X), med, X)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0) + 1e-6
    Z = np.hstack([np.ones((len(Xf), 1)), (Xf - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2)
    reg[0] = 0.0
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"names": list(deadfeat.NAMES), "median": med.tolist(), "mean": mu.tolist(),
            "std": sd.tolist(), "w": w.tolist()}


def proba(model, X):
    med, mu, sd, w = (np.array(model[k]) for k in ("median", "mean", "std", "w"))
    Z = (np.where(np.isnan(X), med, X) - mu) / sd
    return 1.0 / (1.0 + np.exp(-np.clip(w[0] + Z @ w[1:], -30, 30)))


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def decode(p, blocks, tau, smooth, min_dead, min_live):
    """Probabilities -> dead mask: moving average, threshold, then minimum run lengths."""
    q = deadfeat.window_mean(p, -smooth, smooth) if smooth else p
    dead = np.nan_to_num(q) > tau
    for a, b in _runs(~dead):            # close short live gaps first
        if b - a < min_live and a > 0 and b < len(dead):
            dead[a:b] = True
    for a, b in _runs(dead):             # then drop short stoppages
        if b - a < min_dead:
            dead[a:b] = False
    inside = np.zeros(len(dead), dtype=bool)
    for _h, first, last in blocks:
        inside[first:last + 1] = True
    return dead & inside


def decode_with_whistle(p, blocks, whistle_frames, smooth, min_dead, min_live,
                        tau_high=0.6, tau_low=0.3, window=250):
    """decode(), with the threshold lowered for `window` frames after each whistle.

    Without a whistle a stoppage needs probability tau_high; within `window` frames after a
    whistle onset it needs only tau_low. A stoppage then lasts while the probability stays above
    the threshold that started it (hysteresis), so a whistle-started stoppage is not cut off when
    the window closes. Any stoppage that begins inside a whistle's window is started at the whistle
    itself: the classifier reads the players slowing down, which comes seconds after the whistle.
    IDSSE has no audio, so the low threshold, the window and the anchor cannot be checked on
    external data; the high threshold costs 0.003 dead F1 there (0.953 to 0.950).
    """
    q = deadfeat.window_mean(p, -smooth, smooth) if smooth else p
    q = np.nan_to_num(q)
    enter = np.full(len(q), tau_high)
    anchor = np.full(len(q), -1)
    for w in sorted(whistle_frames):
        enter[w:w + window] = tau_low
        anchor[w:w + window] = w
    dead = np.zeros(len(q), dtype=bool)
    active = None
    for t in range(len(q)):
        if active is None:
            if q[t] > enter[t]:
                active = enter[t]
                dead[t] = True
                if anchor[t] >= 0:
                    dead[anchor[t]:t] = True
        elif q[t] > active:
            dead[t] = True
        else:
            active = None
    for a, b in _runs(~dead):
        if b - a < min_live and a > 0 and b < len(dead):
            dead[a:b] = True
    for a, b in _runs(dead):
        if b - a < min_dead:
            dead[a:b] = False
    inside = np.zeros(len(dead), dtype=bool)
    for _h, first, last in blocks:
        inside[first:last + 1] = True
    return dead & inside


def f1(pred, truth, graded):
    tp = (pred & truth & graded).sum()
    fp = (pred & ~truth & graded).sum()
    fn = (~pred & truth & graded).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec), prec, rec


def train(data, exclude=None):
    keys = [k for k in data if k[0] != exclude]
    X = np.concatenate([data[k][0][data[k][2]][::STRIDE] for k in keys])
    y = np.concatenate([data[k][1][data[k][2]][::STRIDE] for k in keys]).astype(float)
    return fit(X, y)


def rules_dead(features):
    with harness.with_params(**RULE_PARAMS):
        dead, _parts, _restarts = deadball.dead_mask(features, harness.GAME)
    return dead


if __name__ == '__main__':
    started = time.time()
    data = dataset()
    matches = sorted({k[0] for k in data})
    print("dataset: {} match-seeds, {} features, {:.0f}s".format(
        len(data), len(deadfeat.NAMES), time.time() - started))

    # leave-one-match-out probabilities
    held_p = {}
    for m in matches:
        model = train(data, exclude=m)
        for k in data:
            if k[0] == m:
                held_p[k] = proba(model, data[k][0])

    # decode settings: chosen on the held-out probabilities, pooled over all matches
    results = {}
    for combo in itertools.product(*DECODE_GRID.values()):
        params = dict(zip(DECODE_GRID, combo))
        scores = [f1(decode(held_p[k], data[k][3], **params), data[k][1], data[k][2])[0]
                  for k in data]
        results[combo] = float(np.mean(scores))
    best_combo = max(results, key=results.get)
    best = dict(zip(DECODE_GRID, best_combo))
    print("decode settings (held-out dead F1 {:.3f}): {}".format(results[best_combo], best))

    print("\nheld-out match   classifier F1 (prec / rec)    rules F1 (prec / rec)")
    rows = []
    for m in matches:
        c, r = [], []
        for k in data:
            if k[0] != m:
                continue
            X, y, g, blocks, feats = data[k]
            c.append(f1(decode(held_p[k], blocks, **best), y, g))
            r.append(f1(rules_dead(feats), y, g))
        c, r = np.mean(c, axis=0), np.mean(r, axis=0)
        rows.append((c, r))
        print("  {:<8}       {:.3f} ({:.2f} / {:.2f})          {:.3f} ({:.2f} / {:.2f})".format(
            m[-6:], *c, *r))
    c = np.mean([x[0] for x in rows], axis=0)
    r = np.mean([x[1] for x in rows], axis=0)
    print("  mean           {:.3f} ({:.2f} / {:.2f})          {:.3f} ({:.2f} / {:.2f})".format(
        *c, *r))

    final = train(data)
    final["decode"] = best
    final["trained_on"] = "IDSSE (Bassek et al. 2025, CC BY 4.0), {} matches x {} seeds, " \
                          "degraded by exp06 noise.py at {:.1%} team flips".format(
                              len(matches), len(SEEDS), FLIP_RATE)
    final["heldout_dead_f1"] = float(c[0])
    with open(MODEL_PATH, "w") as h:
        json.dump(final, h, indent=1)
    w = np.array(final["w"][1:])
    order = np.argsort(-np.abs(w))
    print("\nlargest weights (standardised):  " + ",  ".join(
        "{} {:+.2f}".format(final["names"][i], w[i]) for i in order[:8]))
    print("wrote {}  ({:.0f}s)".format(os.path.relpath(MODEL_PATH, ROOT), time.time() - started))
