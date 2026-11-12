"""Train and evaluate the touch model on degraded IDSSE, leave one match out.

    python experiments/exp09_touch_model/touchclf.py --lomo      # held-out report vs the rule
    python experiments/exp09_touch_model/touchclf.py --train     # fit on all 7 -> model.json

Rows are candidate (frame, team) closest approaches on degraded tracking (ATD-like noise, team
flip rate 1.5% as in exp08). A row is positive when the clean tracking has a true touch by that
team within +-1 s (labels.py). Only live frames (IDSSE BallStatus 1) are used: the pipeline
never switches possession inside a stoppage, and the dead-ball model decides those frames.

The baseline is exp02's hand-set detector (touch.detect) run on the same degraded features.
Both are scored as touch events: precision = share of claimed touches with a true touch by the
same team within +-1 s; recall = share of true touches claimed. The CLOSE slice is where the
degraded tracking has players of both teams within 1.5 m of the ball -- the case where nearest-
player logic is a coin toss and where our wrong-team errors come from.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp08_deadball_clf"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import labels as labels_mod  # noqa: E402
import touchfeat  # noqa: E402
import harness  # noqa: E402  (registers the IDSSE game)
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import tune  # noqa: E402
import deadclf  # noqa: E402
import possession_metrics as pm  # noqa: E402
from exp02_touch_gated import features as feat2, touch as touch_rule  # noqa: E402

FLIP_RATE = 0.015
SEEDS = (0, 1, 2)
MATCH_FRAMES = 25          # +-1 s: smoothing shifts the closest approach up to ~1 s from the contact
CLOSE_M = 1.5
L2 = 1.0
MODEL_PATH = os.path.join(HERE, "model.json")
CACHE = os.path.join(HERE, "cache", "rows_{}_s{}.npz")


def degraded(match, seed):
    s = tune.cached_scene(match)
    ball, players, team = noise.degrade(s, seed, noise.ATD_LIKE, flip_rate=FLIP_RATE)
    return s, ball, players, team


def truth_hits(frames, teams, true_f, true_t):
    """For each (frame, team): is there a true touch by that team within MATCH_FRAMES?"""
    out = np.zeros(len(frames), dtype=bool)
    for tm in (0, 1):
        tf = np.sort(true_f[true_t == tm])
        sel = np.flatnonzero(teams == tm)
        if not len(tf) or not len(sel):
            continue
        i = np.searchsorted(tf, frames[sel])
        before = np.abs(frames[sel] - tf[np.clip(i - 1, 0, len(tf) - 1)])
        after = np.abs(tf[np.clip(i, 0, len(tf) - 1)] - frames[sel])
        out[sel] = np.minimum(before, after) <= MATCH_FRAMES
    return out


def rows(match, seed):
    """Cached (X, y, frames, teams, close) for one degraded match."""
    path = CACHE.format(match, seed)
    if os.path.isfile(path):
        with np.load(path) as h:
            return {k: h[k] for k in h.files}
    s, ball, players, team = degraded(match, seed)
    live = (s["gt"] == pm.H) | (s["gt"] == pm.A)
    dist, _col, _d = touchfeat.team_distances(ball, players, team)
    frames, teams = touchfeat.candidates(dist, live)
    X = touchfeat.features(ball, players, team, frames, teams)
    t = labels_mod.load(match)
    tlive = live[t["frame"]]
    y = truth_hits(frames, teams, t["frame"][tlive], t["team"][tlive])
    close = (dist[frames, 0] <= CLOSE_M) & (dist[frames, 1] <= CLOSE_M)
    out = {"X": X, "y": y, "frames": frames, "teams": teams, "close": close}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def rule_events(match, seed):
    """exp02's detector on the same degraded tracking: (frames, teams) of claimed touches."""
    s, ball, players, team = degraded(match, seed)
    features = adapter.to_features(ball, players, team)
    kin = feat2.kinematics(features, harness.GAME)
    touches, touch_team = touch_rule.detect(features, kin)
    live = (s["gt"] == pm.H) | (s["gt"] == pm.A)
    f = np.flatnonzero(touches & live & (touch_team >= 0))
    return f, touch_team[f].astype(np.int8)


def event_scores(pred_f, pred_t, match, seed, close_pred=None):
    """Precision / recall of claimed touches against true live touches, overall and close."""
    s, ball, players, team = degraded(match, seed)
    live = (s["gt"] == pm.H) | (s["gt"] == pm.A)
    t = labels_mod.load(match)
    keep = live[t["frame"]]
    tf, tt = t["frame"][keep], t["team"][keep]
    dist, _c, _d = touchfeat.team_distances(ball, players, team)
    hit_p = truth_hits(pred_f, pred_t, tf, tt)
    hit_t = truth_hits(tf, tt, pred_f, pred_t)
    close_p = (dist[pred_f, 0] <= CLOSE_M) & (dist[pred_f, 1] <= CLOSE_M) \
        if close_pred is None else close_pred
    close_t = (dist[tf, 0] <= CLOSE_M) & (dist[tf, 1] <= CLOSE_M)
    safe = lambda a: float(a.mean()) if len(a) else float("nan")  # noqa: E731
    return {"prec": safe(hit_p), "rec": safe(hit_t), "n_pred": len(pred_f), "n_true": len(tf),
            "prec_close": safe(hit_p[close_p]), "rec_close": safe(hit_t[close_t]),
            "n_close_true": int(close_t.sum())}


def fit(X, y):
    model = deadclf.fit(X, y.astype(float), l2=L2)
    model["names"] = list(touchfeat.NAMES)
    return model


def _build(args):
    match, seed = args
    rows(match, seed)
    return match, seed


def build_all(matches, seeds):
    jobs = [(m, s) for m in matches for s in seeds if not os.path.isfile(CACHE.format(m, s))]
    if jobs:
        with Pool(min(8, len(jobs))) as pool:
            for m, s in pool.imap_unordered(_build, jobs):
                print("  rows {} seed {}".format(m[-6:], s), flush=True)


def lomo(matches, taus=(0.3, 0.4, 0.5, 0.6)):
    build_all(matches, SEEDS)
    keys = ("prec", "rec", "prec_close", "rec_close")
    agg = {"rule": [], **{t: [] for t in taus}}
    auc_all = []
    for held in matches:
        train = [rows(m, s) for m in matches if m != held for s in SEEDS]
        X = np.vstack([r["X"] for r in train])
        y = np.concatenate([r["y"] for r in train])
        model = fit(X, y)
        test = rows(held, 0)
        p = deadclf.proba(model, test["X"])
        auc_all.append(auc(p, test["y"]))
        line = "{}  base rate {:.2f}  AUC {:.3f}".format(held[-6:], test["y"].mean(), auc_all[-1])
        rf, rt = rule_events(held, 0)
        r = event_scores(rf, rt, held, 0)
        agg["rule"].append(r)
        line += "  | rule P {:.2f} R {:.2f} closeP {:.2f} closeR {:.2f} (n={})".format(
            r["prec"], r["rec"], r["prec_close"], r["rec_close"], r["n_pred"])
        for tau in taus:
            sel = p >= tau
            r = event_scores(test["frames"][sel], test["teams"][sel], held, 0, test["close"][sel])
            agg[tau].append(r)
            if tau == 0.5:
                line += "  | model@.5 P {:.2f} R {:.2f} closeP {:.2f} closeR {:.2f} (n={})".format(
                    r["prec"], r["rec"], r["prec_close"], r["rec_close"], r["n_pred"])
        print(line, flush=True)
    print("\nmean over held-out matches (seed 0); candidate coverage bounds model recall")
    print("  {:<10} {:>6} {:>6} {:>8} {:>8} {:>8}".format("", "prec", "rec", "closeP", "closeR",
                                                       "claims"))
    for name, rs in agg.items():
        print("  {:<10} ".format(str(name)) + " ".join(
            "{:>6.3f}  ".format(np.nanmean([r[k] for r in rs])) for k in keys)
            + "{:>6.0f}".format(np.mean([r["n_pred"] for r in rs])))
    print("  AUC {:.3f} +- {:.3f}".format(np.mean(auc_all), np.std(auc_all)))


def coverage(matches):
    """Share of true live touches with a candidate of the same team within +-0.5 s."""
    build_all(matches, (0,))
    for m in matches:
        r = rows(m, 0)
        sc = event_scores(r["frames"], r["teams"], m, 0, r["close"])
        print("  {}  candidates {:>6}  cover {:.3f} of true touches (close {:.3f}, n={})  "
              "positive rate {:.2f}".format(m[-6:], len(r["frames"]), sc["rec"], sc["rec_close"],
                                            sc["n_close_true"], r["y"].mean()))


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / max(1, n0 * n1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lomo", action="store_true")
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    matches = idsse.matches()
    started = time.time()
    if args.coverage:
        coverage(matches)
    if args.lomo:
        lomo(matches)
    if args.train:
        build_all(matches, SEEDS)
        data = [rows(m, s) for m in matches for s in SEEDS]
        model = fit(np.vstack([r["X"] for r in data]), np.concatenate([r["y"] for r in data]))
        json.dump(model, open(MODEL_PATH, "w"), indent=1)
        print("wrote", MODEL_PATH)
    print("({:.0f}s)".format(time.time() - started))
