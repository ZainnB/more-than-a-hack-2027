"""Choose exp10's costs on degraded IDSSE, leave-one-match-out, against exp08 on the same inputs.

    python experiments/exp10_viterbi/idsse_eval.py             # grid + report (~minutes)
    python experiments/exp10_viterbi/idsse_eval.py --report    # re-print from results/idsse_grid.csv

Nothing here reads the provided match. The loop:

  1. Each IDSSE match is degraded to look like the ATD (exp06's noise.py, 1.5% team flips as in
     exp08/exp09) with seeds 0, 1, 2 -- the same inputs exp09 was judged on.
  2. Both learned models are held out: the dead-ball classifier and the touch model that score a
     match were trained on the other six (deadclf.train / touchclf.fit with that match excluded).
  3. exp08's pipeline (exp06's possession values, classifier stoppages decoded by exp08's own
     settings) runs on the identical degraded inputs, so every comparison is paired per
     (match, seed).
  4. Every grid configuration (config.GRID) is decoded and scored with the repo's scorer.
  5. Leave-one-match-out: for each match, pick the best configuration on the other six (mean
     spellF1, macroF1 breaks ties) and score it on the seventh. The held-out gains over exp08,
     with their seed-to-seed spread, are what decide whether exp10 is adopted (PLAN.md: change
     only on an IDSSE gain larger than seed noise).
  6. The value that ships is the best on all seven, the same rule exp06 and exp09 used.

Smart Tagging does not exist on IDSSE, so its relief is off here and stays a priori.

Emissions are linear in ha_scale and touch_gain, so each degraded match's evidence is computed
once at scale 1 and only rescaled per configuration; a full grid is one Viterbi per run.
"""

import argparse
import csv
import importlib.util
import itertools
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
for path in (ROOT, EXPERIMENTS, os.path.join(EXPERIMENTS, "exp06_idsse"),
             os.path.join(EXPERIMENTS, "exp08_deadball_clf"),
             os.path.join(EXPERIMENTS, "exp09_touch_model"),
             os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if path not in sys.path:
        sys.path.insert(0, path)

from exp10_viterbi import config, decode, emissions  # noqa: E402
import harness  # noqa: E402  (registers the IDSSE "game"; with_params; score)
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import tune  # noqa: E402
import deadclf  # noqa: E402
import deadfeat  # noqa: E402
import touchclf  # noqa: E402
import hook as touch_hook  # noqa: E402
import possession_metrics as pm  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")        # held-out models, derived from IDSSE only
SEEDS = (0, 1, 2)                          # exp09's seeds
FLIP_RATE = touchclf.FLIP_RATE             # 1.5%, exp08/exp09's realistic team-flip rate
METRICS = ("spellF1", "macroF1", "team_acc", "poss", "poss_gt", "D_prec", "D_rec")


def _exp09_base():
    """exp08's possession values as exp09 recorded them (BASE in exp09's run.py)."""
    spec = importlib.util.spec_from_file_location(
        "run09", os.path.join(EXPERIMENTS, "exp09_touch_model", "run.py"))
    run09 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run09)
    return dict(run09.BASE)


# -- held-out models -----------------------------------------------------------------------------

def dead_models(matches):
    """{match: classifier trained without it}, cached as JSON."""
    paths = {m: os.path.join(CACHE, "dead_without_{}.json".format(m)) for m in matches}
    if not all(os.path.isfile(p) for p in paths.values()):
        data = deadclf.dataset(matches)
        os.makedirs(CACHE, exist_ok=True)
        for m in matches:
            with open(paths[m], "w") as h:
                json.dump(deadclf.train(data, exclude=m), h)
        del data
    out = {}
    for m, p in paths.items():
        with open(p) as h:
            out[m] = json.load(h)
    return out


def touch_models(matches):
    """{match: touch model trained without it}, as exp09's run.prepare builds them."""
    paths = {m: os.path.join(CACHE, "touch_without_{}.json".format(m)) for m in matches}
    if not all(os.path.isfile(p) for p in paths.values()):
        touchclf.build_all(matches, touchclf.SEEDS)
        data = {(m, s): touchclf.rows(m, s) for m in matches for s in touchclf.SEEDS}
        os.makedirs(CACHE, exist_ok=True)
        for held in matches:
            tr = [data[(m, s)] for m in matches if m != held for s in touchclf.SEEDS]
            model = touchclf.fit(np.vstack([r["X"] for r in tr]),
                                 np.concatenate([r["y"] for r in tr]))
            with open(paths[held], "w") as h:
                json.dump(model, h)
    out = {}
    for m, p in paths.items():
        with open(p) as h:
            out[m] = json.load(h)
    return out


# -- inputs --------------------------------------------------------------------------------------

_IN = {}   # (match, seed) -> dict of everything a run needs; filled before the pool forks


def prepare(matches, seeds=SEEDS):
    dmodels, tmodels = dead_models(matches), touch_models(matches)
    smooth = int(emissions.dead_model()["decode"]["smooth"])
    decode08 = emissions.dead_model()["decode"]
    base08 = _exp09_base()
    for m in matches:
        s = tune.cached_scene(m)
        inside = np.zeros(s["n"], dtype=bool)
        for _h, a, b in s["blocks"]:
            inside[a:b + 1] = True
        for seed in seeds:
            ball, players, team = noise.degrade(s, seed, noise.ATD_LIKE, flip_rate=FLIP_RATE)
            features = adapter.to_features(ball, players, team)
            p_dead = deadclf.proba(dmodels[m], deadfeat.compute(ball, players, *deadclf.HALF))
            touches = touch_hook.score_candidates(tmodels[m], ball, players, team, inside)

            # exp08 on exactly these inputs: classifier stoppages, exp06's possession layer.
            dead_mask = deadclf.decode(p_dead, s["blocks"], **decode08)
            with harness.with_params(**base08):
                labels08, _ = run04.labels_from_features(
                    features, s["blocks"], harness.GAME, s["n"], votes=None, mode="none",
                    dead_hook=tune.replace_dead(dead_mask, s["blocks"]))
            r08 = harness.score(s["gt"], pm.label_codes(labels08))

            n = s["n"]
            _IN[(m, seed)] = {
                "gt": s["gt"], "blocks": s["blocks"],
                "team1": emissions.team_evidence(features, harness.GAME, 1.0),
                "touch1": emissions.touch_evidence(n, touches, 1.0),
                "dead": emissions.dead_evidence(p_dead, smooth),
                "exp08": r08,
            }


def run_one(job):
    params, match, seed = job
    x = _IN[(match, seed)]
    team = params["ha_scale"] * x["team1"] + params["touch_gain"] * x["touch1"]
    E = np.empty((len(team), 3))
    E[:, 0] = x["dead"][:, 0] + team[:, 0]
    E[:, 1] = x["dead"][:, 0] + team[:, 1]
    E[:, 2] = x["dead"][:, 1]
    codes = decode.decode_halves(E, x["blocks"], params["c_switch"], params["c_dead"],
                                 params["c_dead"])
    r = harness.score(x["gt"], codes)
    row = dict(params, match=match, seed=seed)
    row.update({k: r[k] for k in METRICS})
    return row


# -- report --------------------------------------------------------------------------------------

def _key(params):
    return tuple(params[k] for k in config.GRID)


def _mean(rows, cfg, matches, metric="spellF1"):
    vals = [r[metric] for r in rows if r["config"] == cfg and r["match"] in matches]
    return float(np.mean(vals))


def _choose(rows, cfgs, matches):
    return max(cfgs, key=lambda c: (round(_mean(rows, c, matches), 4),
                                    _mean(rows, c, matches, "macroF1")))


def _by_seed(rows, cfg, match, metric="spellF1"):
    return np.array([r[metric] for r in sorted(
        (r for r in rows if r["config"] == cfg and r["match"] == match), key=lambda r: r["seed"])])


def report(rows, matches):
    cfgs = sorted({r["config"] for r in rows if r["config"] != "exp08"})
    print("\nleave-one-match-out: choose on 6 matches, score the 7th (3 seeds each, paired with exp08)")
    print("  {:<8} {:<28} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9} {:>11}".format(
        "held", "chosen (sw, ha, dead, touch)", "exp10 sF1", "exp08 sF1", "gain", "seed sd",
        "exp10 mF1", "exp08 mF1", "poss/gt"))
    gains, mgains = [], []
    for held in matches:
        train = [m for m in matches if m != held]
        choice = _choose(rows, cfgs, train)
        diff = _by_seed(rows, choice, held) - _by_seed(rows, "exp08", held)
        mdiff = _by_seed(rows, choice, held, "macroF1") - _by_seed(rows, "exp08", held, "macroF1")
        gains.append(diff.mean())
        mgains.append(mdiff.mean())
        print("  {:<8} {:<28} {:>9.3f} {:>9.3f} {:>+8.3f} {:>8.3f} {:>9.3f} {:>9.3f} {:>5.0f}/{:<5.0f}"
              .format(held[-6:], str(choice), _mean(rows, choice, [held]),
                      _mean(rows, "exp08", [held]), diff.mean(), diff.std(),
                      _mean(rows, choice, [held], "macroF1"),
                      _mean(rows, "exp08", [held], "macroF1"),
                      _mean(rows, choice, [held], "poss"), _mean(rows, choice, [held], "poss_gt")))
    print("  mean held-out spellF1 gain {:+.3f} (improved {}/{} matches), macroF1 gain {:+.3f}".format(
        np.mean(gains), sum(g > 0 for g in gains), len(gains), np.mean(mgains)))

    best = _choose(rows, cfgs, matches)
    notouch = _choose(rows, [c for c in cfgs if c[3] == 0.0], matches)
    apriori = _key(config.A_PRIORI)
    print("\nall 7 matches (21 runs each):")
    print("  {:<34} {:>8} {:>8} {:>8} {:>8} {:>9}".format(
        "", "spellF1", "macroF1", "D prec", "D rec", "poss/gt"))
    for name, c in (("exp08", "exp08"), ("exp10 best " + str(best), best),
                    ("exp10 best, no touch " + str(notouch), notouch),
                    ("exp10 a priori " + str(apriori), apriori)):
        if c not in cfgs and c != "exp08":
            continue
        print("  {:<34} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>4.0f}/{:<4.0f}".format(
            name, _mean(rows, c, matches), _mean(rows, c, matches, "macroF1"),
            _mean(rows, c, matches, "D_prec"), _mean(rows, c, matches, "D_rec"),
            _mean(rows, c, matches, "poss"), _mean(rows, c, matches, "poss_gt")))
    ranked = sorted(cfgs, key=lambda c: -_mean(rows, c, matches))
    print("\ntop 8 configurations (c_switch, ha_scale, c_dead, touch_gain):")
    for c in ranked[:8]:
        print("  {:<28} spellF1 {:.3f}  macroF1 {:.3f}  poss {:.0f}".format(
            str(c), _mean(rows, c, matches), _mean(rows, c, matches, "macroF1"),
            _mean(rows, c, matches, "poss")))
    print("\nchosen for config.CHOSEN: {}".format(dict(zip(config.GRID, best))))
    return dict(zip(config.GRID, best))


def evaluate(workers=8):
    matches = idsse.matches()
    if not matches:
        print("IDSSE data not found in {} -- run.py will use config.A_PRIORI.".format(idsse.DATA_DIR))
        return None
    started = time.time()
    prepare(matches)
    print("prepared {} degraded inputs ({:.0f}s)".format(len(_IN), time.time() - started), flush=True)

    configs = [dict(zip(config.GRID, v)) for v in itertools.product(*config.GRID.values())]
    jobs = [(p, m, s) for p in configs for m in matches for s in SEEDS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=8)
    for m in matches:
        for s in SEEDS:
            row = {k: "" for k in config.GRID}
            row.update(match=m, seed=s, **{k: _IN[(m, s)]["exp08"][k] for k in METRICS})
            row["config"] = "exp08"
            rows.append(row)
    for r in rows:
        if "config" not in r:
            r["config"] = _key(r)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "idsse_grid.csv")
    fields = list(config.GRID) + ["config", "match", "seed"] + list(METRICS)
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r[k] if k != "config" else ("exp08" if r[k] == "exp08" else "grid"))
                        for k in fields})
    print("{} runs in {:.0f}s -> {}".format(len(rows), time.time() - started,
                                            os.path.relpath(path, ROOT)))
    return report(rows, matches)


def read(path):
    rows = tune.read_rows(path)
    for r in rows:
        r["config"] = "exp08" if r["config"] == "exp08" else _key(r)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.report:
        report(read(os.path.join(RESULTS, "idsse_grid.csv")), idsse.matches())
    else:
        evaluate(args.workers)
