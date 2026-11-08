"""What would better tracking be worth? Run the tuned pipeline on IDSSE with the noise turned down.

    python experiments/exp06_idsse/whatif.py

The point is to rank the possible upgrades -- a ball detector that sees the ball more often, a
tracker that gets team identity right, a ball track that keeps its sharp kicks -- by what each
would add to the score, before anyone spends weeks building one. Each row degrades the same 7
matches with one thing improved and scores the same tuned pipeline (tuned.py) against IDSSE's
flags. The pipeline's parameters are NOT re-tuned per row, so a row shows what the data alone is
worth; a pipeline built for better data could do better still.

Rows are paired with the baseline on the same (match, seed), and the column "+/-" is the standard
error of that paired difference over all runs, so a gain smaller than about twice it is not
distinguishable from the simulator's own randomness.
"""

import argparse
import csv
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
import tune  # noqa: E402
import possession_metrics as pm  # noqa: E402
from tuned import TUNED  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SEEDS = (0, 1, 2)

from scenarios import SCENARIOS  # noqa: E402

METRICS = ("spellF1", "macroF1", "D_prec", "D_rec", "team_acc", "poss", "poss_gt")
_SCENES = {}


def run_one(job):
    index, match, seed = job
    _name, switches, overrides = SCENARIOS[index]
    s = _SCENES[match]
    ball, players, team = noise.degrade(s, seed, switches, **overrides)
    features = adapter.to_features(ball, players, team)
    with harness.with_params(**TUNED):
        labels, _ = run04.labels_from_features(features, s["blocks"], harness.GAME, s["n"],
                                               votes=None, mode="none")
    r = harness.score(s["gt"], pm.label_codes(labels))
    graded = s["gt"] != pm.X
    seen = ~np.isnan(ball[:, 0]) & graded
    row = {"scenario": index, "match": match, "seed": seed,
           "ball_missing": 1.0 - seen.sum() / graded.sum()}
    row.update({k: r[k] for k in METRICS})
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()

    started = time.time()
    matches = idsse.matches()
    for m in matches:
        _SCENES[m] = tune.cached_scene(m)
    seeds = tuple(range(args.seeds))
    jobs = [(i, m, sd) for i in range(len(SCENARIOS)) for m in matches
            for sd in (seeds if SCENARIOS[i][1] is not noise.CLEAN else seeds[:1])]
    print("{} scenarios x {} matches x {} seeds = {} runs; tuned config {}".format(
        len(SCENARIOS), len(matches), len(seeds), len(jobs), TUNED))
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs, chunksize=2)
    print("ran in {:.0f}s\n".format(time.time() - started))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "whatif.csv"), "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def runs(i):
        return {(r["match"], r["seed"]): r for r in rows if r["scenario"] == i}

    base = runs(0)
    print("{:<38} {:>6} {:>8} {:>7} {:>7} {:>7} {:>7} {:>6} {:>7} {:>8}".format(
        "", "ball", "spellF1", "vs base", "+/-", "macroF1", "D prec", "D rec", "team", "poss"))
    for i, (name, _sw, _ov) in enumerate(SCENARIOS):
        mine = runs(i)
        keys = [k for k in mine if k in base] or [(m, 0) for m in matches]
        spell = np.array([mine[k]["spellF1"] for k in mine])
        if i == 0:
            delta, se = 0.0, 0.0
        else:
            diffs = np.array([mine[k]["spellF1"] - base[k]["spellF1"] for k in keys if k in mine])
            delta, se = diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0
        mean = lambda key: float(np.mean([r[key] for r in mine.values()]))  # noqa: E731
        print("{:<38} {:>5.0f}% {:>8.3f} {:>+7.3f} {:>7.3f} {:>7.3f} {:>7.3f} {:>6.3f} {:>7.3f} "
              "{:>4.0f}/{:<4.0f}".format(
                  name, 100 * mean("ball_missing"), spell.mean(), delta, se, mean("macroF1"),
                  mean("D_prec"), mean("D_rec"), mean("team_acc"), mean("poss"), mean("poss_gt")))
    print("\nwrote {}".format(os.path.relpath(os.path.join(RESULTS, "whatif.csv"), ROOT)))
