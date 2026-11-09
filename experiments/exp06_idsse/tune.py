"""Tune the exp04 pipeline's parameters on degraded IDSSE. Nothing from the provided match.

    python experiments/exp06_idsse/tune.py --stage A              # possession layer
    python experiments/exp06_idsse/tune.py --stage B --base A     # dead-ball layer at A's best
    python experiments/exp06_idsse/tune.py --stage A --base B     # one more round

Every configuration in a stage's grid is run on every downloaded match x noise seed, scored
against IDSSE's own flags, and written to results/stage_<name>.csv. The configuration that ships
is the one with the best mean spellF1 over all of them (macroF1 breaks ties).

How far that choice can be trusted is estimated by leave-one-match-out: for each match, pick the
best configuration on the *other* matches and score it on the held-out one. If the held-out
scores are close to the in-sample ones, the choice is not fitted to any one match.
"""

import argparse
import csv
import itertools
import multiprocessing
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
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import harness  # noqa: E402
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import scene as scene_mod  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402
import possession_metrics as pm  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(idsse.DATA_DIR, "cache")  # git-ignored with the rest of data/
SEEDS = (0, 1)

GRIDS = {
    "A": {"margin_m": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
          "k_fallback": [10, 15, 20, 25, 35, 50, 75],
          "use_touch": [False, True]},
    "B": {"slow_speed": [1.1, 1.2, 1.3, 1.4, 1.5],
          "slow_min_frames": [100, 150, 200, 300],
          "max_dead": [375, 750, 1500],
          "ball_lost_frames": [50, 75, 125]},
    # B's best sat on two grid edges (slow_min_frames 300, max_dead 375), so B2 extends past both.
    "B2": {"slow_speed": [1.3, 1.4, 1.5],
           "slow_min_frames": [250, 300, 400, 500, 750],
           "max_dead": [250, 375, 500],
           "ball_lost_frames": [75, 125]},
}

METRICS = ("spellF1", "macroF1", "D_prec", "D_rec", "team_acc", "poss", "poss_gt", "transF1")

_INPUTS = {}  # (match, seed) -> (features, gt, blocks, n); filled before the pool forks
_DEAD = {}    # (match, seed) -> dead mask from an outside model, used in place of the rules


def cached_scene(match):
    path = os.path.join(CACHE, "scene_{}.npz".format(match))
    if os.path.isfile(path):
        with np.load(path) as h:
            s = {k: h[k] for k in h.files}
        s["blocks"] = [tuple(int(v) for v in row) for row in s["blocks"]]
        s["n"] = int(s["n"])
        s["match"] = match
        return s
    s = scene_mod.build(match)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, n=s["n"], blocks=np.array(s["blocks"]), team=s["team"],
                        ball=s["ball"], ball_z=s["ball_z"], players=s["players"], gt=s["gt"])
    return s


def prepare(matches, seeds=SEEDS, scenario=0, overrides=None, dead_fn=None):
    """Degrade every match once per seed. `overrides` adds degrade() keywords to the scenario's;
    `dead_fn(scene, ball, players, team)` may return a dead mask to replace the dead-ball rules."""
    _name, switches, scenario_overrides = SCENARIOS[scenario]
    kwargs = dict(scenario_overrides, **(overrides or {}))
    _INPUTS.clear()
    _DEAD.clear()
    for match in matches:
        s = cached_scene(match)
        for seed in seeds:
            ball, players, team = noise.degrade(s, seed, switches, **kwargs)
            _INPUTS[(match, seed)] = (adapter.to_features(ball, players, team), s["gt"],
                                      s["blocks"], s["n"])
            if dead_fn is not None:
                _DEAD[(match, seed)] = dead_fn(s, ball, players, team)


def replace_dead(mask, blocks):
    """A dead_hook that swaps in an outside dead mask; restarts are its dead -> live edges."""
    def hook(_dead, _restarts, _features):
        restarts = [int(i) + 1 for i in np.flatnonzero(mask[:-1] & ~mask[1:])
                    if any(first <= i + 1 <= last for _h, first, last in blocks)]
        return mask.copy(), restarts
    return hook


def run_one(job):
    params, match, seed = job
    features, gt, blocks, n = _INPUTS[(match, seed)]
    hook = replace_dead(_DEAD[(match, seed)], blocks) if (match, seed) in _DEAD else None
    with harness.with_params(**params):
        labels, _ = run04.labels_from_features(features, blocks, harness.GAME, n,
                                               votes=None, mode="none", dead_hook=hook)
    r = harness.score(gt, pm.label_codes(labels))
    row = dict(params)
    row.update(match=match, seed=seed)
    row.update({k: r[k] for k in METRICS})
    return row


def configs(stage, base):
    grid = GRIDS[stage]
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(base)
        params.update(zip(keys, values))
        yield params


def key_of(row, keys):
    return tuple(row[k] for k in keys)


def aggregate(rows, keys):
    """{config tuple: {metric: mean}} and per-match means for leave-one-out."""
    by_config, by_config_match = {}, {}
    for row in rows:
        c = key_of(row, keys)
        by_config.setdefault(c, []).append(row)
        by_config_match.setdefault((c, row["match"]), []).append(row)
    mean = {c: {m: float(np.mean([r[m] for r in rs])) for m in METRICS}
            for c, rs in by_config.items()}
    per_match = {cm: float(np.mean([r["spellF1"] for r in rs]))
                 for cm, rs in by_config_match.items()}
    return mean, per_match


def best(mean, configs_allowed=None):
    pool = configs_allowed or list(mean)
    return max(pool, key=lambda c: (round(mean[c]["spellF1"], 4), mean[c]["macroF1"]))


def leave_one_out(rows, keys):
    matches = sorted({r["match"] for r in rows})
    out = []
    for held in matches:
        train = [r for r in rows if r["match"] != held]
        test = [r for r in rows if r["match"] == held]
        mean_train, _ = aggregate(train, keys)
        mean_test, _ = aggregate(test, keys)
        choice = best(mean_train)
        out.append((held, choice, mean_train[choice]["spellF1"], mean_test[choice]["spellF1"],
                    mean_test[best(mean_test)]["spellF1"]))
    return out


def read_rows(path):
    with open(path) as h:
        rows = list(csv.DictReader(h))
    for r in rows:
        for k, v in r.items():
            if k == "match":
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = int(v)
                except ValueError:
                    try:
                        r[k] = float(v)
                    except ValueError:
                        pass
    return rows


def base_from(stage_name, results_dir=None):
    """The best configuration of an earlier stage, read back from its CSV."""
    rows = read_rows(os.path.join(results_dir or RESULTS, "stage_{}.csv".format(stage_name)))
    keys = [k for k in rows[0] if k not in METRICS and k not in ("match", "seed")]
    mean, _ = aggregate(rows, keys)
    return dict(zip(keys, best(mean)))


def run_stage(stage, base=None, name=None, scenario=0, workers=8, results_dir=None,
              overrides=None, dead_fn=None, seeds=SEEDS):
    """Run one stage's grid, write its CSV, print the summary, and return the best config."""
    results_dir = results_dir or RESULTS
    base = base or {}
    name = name or stage
    matches = idsse.matches()
    started = time.time()
    prepare(matches, seeds=seeds, scenario=scenario, overrides=overrides, dead_fn=dead_fn)
    jobs = [(p, m, s) for p in configs(stage, base) for m in matches for s in seeds]
    print("scenario {}: {}  overrides {}  outside dead model: {}".format(
        scenario, SCENARIOS[scenario][0], overrides or {}, dead_fn is not None))
    print("stage {}  base {}  {} configs x {} matches x {} seeds = {} runs  (prep {:.0f}s)".format(
        name, base or "a priori", len(jobs) // (len(matches) * len(seeds)), len(matches),
        len(seeds), len(jobs), time.time() - started))

    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=4)
    print("ran in {:.0f}s".format(time.time() - started))

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "stage_{}.csv".format(name))
    with open(path, "w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    keys = [k for k in rows[0] if k not in METRICS and k not in ("match", "seed")]
    varied = list(GRIDS[stage])
    mean, _ = aggregate(rows, keys)
    ranked = sorted(mean, key=lambda c: -mean[c]["spellF1"])
    print("\ntop 10 by mean spellF1 ({} runs each):".format(len(matches) * len(seeds)))
    print("  " + "  ".join("{:>14}".format(k) for k in varied) +
          "   spellF1  macroF1  D prec  D rec  team   poss")
    for c in ranked[:10]:
        p = dict(zip(keys, c))
        m = mean[c]
        print("  " + "  ".join("{:>14}".format(str(p[k])) for k in varied) +
              "   {:.3f}    {:.3f}   {:.3f}  {:.3f}  {:.3f}  {:.0f}/{:.0f}".format(
                  m["spellF1"], m["macroF1"], m["D_prec"], m["D_rec"], m["team_acc"],
                  m["poss"], m["poss_gt"]))

    print("\nleave-one-match-out (pick on 6, score on the 7th):")
    print("  {:<8} {:>9} {:>10} {:>12}   chosen".format("held out", "train", "held-out",
                                                       "oracle"))
    loo = leave_one_out(rows, keys)
    for held, choice, tr, te, oracle in loo:
        p = dict(zip(keys, choice))
        print("  {:<8} {:>9.3f} {:>10.3f} {:>12.3f}   {}".format(
            held[-6:], tr, te, oracle, {k: p[k] for k in varied}))
    print("  mean held-out spellF1 {:.3f}  (oracle per match {:.3f})".format(
        np.mean([x[3] for x in loo]), np.mean([x[4] for x in loo])))
    print("\nwrote {}".format(os.path.relpath(path, ROOT)))
    return dict(zip(keys, best(mean))), mean[best(mean)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=list(GRIDS), required=True)
    parser.add_argument("--base", help="start from the best of this earlier stage (its CSV name)")
    parser.add_argument("--name", help="output name (default: the stage, plus the base)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--scenario", type=int, default=0,
                        help="index into scenarios.SCENARIOS: tune for that data quality")
    args = parser.parse_args()

    name = args.name or (args.stage + ("_after_" + args.base if args.base else ""))
    if args.scenario:
        name = "s{}_{}".format(args.scenario, name)
    run_stage(args.stage, base_from(args.base) if args.base else {}, name, args.scenario,
              args.workers)
