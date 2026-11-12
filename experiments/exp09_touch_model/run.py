"""exp08 plus the learned touch model: evaluate on IDSSE, or write the submission.

    python experiments/exp09_touch_model/run.py --idsse     # grid + leave-one-out vs exp08, ~minutes
    python experiments/exp09_touch_model/run.py             # -> evaluation/results/exp09.csv

On IDSSE every piece is held out: the dead-ball classifier and the touch model are both trained
without the match being scored. The incumbent (exp08's possession values, exp02's rule detector)
runs on the identical degraded inputs, so the comparison is paired per (match, seed).

The decode values (tau, fallback length, minimum possession) are picked on six matches and scored
on the seventh. They are adopted only if the held-out gain over exp08 is larger than seed noise
(PLAN.md protocol).
"""

import argparse
import csv
import itertools
import json
import os
import sys
import time
import multiprocessing

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp08_deadball_clf"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import hook as hook_mod  # noqa: E402
import touchclf  # noqa: E402
import harness  # noqa: E402
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import tune  # noqa: E402
import possession_metrics as pm  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402

RESULTS = os.path.join(HERE, "results")
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "exp09.csv")
# Chosen on IDSSE (leave-one-match-out picked the same values in all 7 folds; held-out spellF1
# +0.068 over exp08, 7/7 matches). See README.md.
# Second grid (GRIDS["confirm"]) added the second-touch check: 1 s window chosen in all 7 folds,
# +0.011 spellF1 over (0.85, 250, 1.0, no check), 6/7 matches. Look-ahead confirmation (-0.021)
# and longer minimum possessions (-0.029) both lost on IDSSE.
CHOSEN = {"tau": 0.8, "k_fallback": 250, "min_possession_s": 1.0, "confirm_w": 25,
          "touch_confirm": False}
SEEDS = (0, 1, 2)
BASE = {"slow_speed": 1.5, "slow_min_frames": 500, "max_dead": 250, "ball_lost_frames": 125,
        "use_touch": True, "margin_m": 1.0, "k_fallback": 75}       # exp08 as shipped
# First grid: tau 0.5-0.9, k_fallback 75-500, min_possession_s 0.3/1.0. Every fold chose
# (0.8, 250, 1.0), with min_possession_s on the grid edge, so the grid was extended past it.
GRID = {"tau": [0.7, 0.75, 0.8, 0.85, 0.9],
        "k_fallback": [125, 250, 375, 500],
        "min_possession_s": [0.6, 1.0, 1.5, 2.0, 3.0]}

# Quick-switch suppression (asked for on 2026-11-10: remove 1-3 s flip-and-backs even at the cost
# of real short switches). Three mechanisms, each with tau re-chosen since confirmation changes it:
#   confirm_w       second-touch check in hook.second_touch, window in frames (0 = off)
#   touch_confirm   exp02's look-ahead: the new team must be nearest on most of the next second
#   min_possession_s  merge_short folds a possession shorter than this between the same team
GRIDS = {"base": GRID,
         "confirm": {"tau": [0.7, 0.8, 0.85, 0.9],
                     "k_fallback": [250],
                     "min_possession_s": [1.0, 2.0, 3.0],
                     "confirm_w": [0, 25, 50, 75, 125],
                     "touch_confirm": [False, True]}}
HOOK_ONLY = ("tau", "confirm_w")
METRICS = ("spellF1", "macroF1", "team_acc", "poss", "poss_gt", "transF1")

_IN = {}   # (match, seed) -> (features, gt, blocks, n, dead, frames, teams, p)


def prepare(matches, seeds):
    import retune
    dead_fn = retune.classifier_dead_fn()
    data = {(m, s): touchclf.rows(m, s) for m in matches for s in touchclf.SEEDS}
    models = {}
    for held in matches:
        tr = [data[(m, s)] for m in matches if m != held for s in touchclf.SEEDS]
        models[held] = touchclf.fit(np.vstack([r["X"] for r in tr]),
                                    np.concatenate([r["y"] for r in tr]))
    for m in matches:
        s = tune.cached_scene(m)
        inside = np.zeros(s["n"], dtype=bool)
        for _h, a, b in s["blocks"]:
            inside[a:b + 1] = True
        for seed in seeds:
            ball, players, team = noise.degrade(s, seed, noise.ATD_LIKE,
                                                flip_rate=touchclf.FLIP_RATE)
            frames, teams, p = hook_mod.score_candidates(models[m], ball, players, team, inside)
            _IN[(m, seed)] = (adapter.to_features(ball, players, team), s["gt"], s["blocks"],
                              s["n"], dead_fn(s, ball, players, team), frames, teams, p)


def run_one(job):
    name, params, match, seed = job
    features, gt, blocks, n, dead, frames, teams, p = _IN[(match, seed)]
    knobs = dict(BASE)
    touch_hook = None
    if name != "exp08":
        knobs["touch_confirm"] = False
        knobs.update({k: v for k, v in params.items() if k not in HOOK_ONLY})
        touch_hook = hook_mod.make(frames, teams, p, params["tau"], params.get("confirm_w", 0))
    with harness.with_params(**knobs):
        labels, _ = run04.labels_from_features(
            features, blocks, harness.GAME, n, votes=None, mode="none",
            dead_hook=tune.replace_dead(dead, blocks), touch_hook=touch_hook)
    r = harness.score(gt, pm.label_codes(labels))
    row = {"config": name, "match": match, "seed": seed}
    row.update(params)
    row.update({k: r[k] for k in METRICS})
    return row


def evaluate(grid_name="base", workers=8):
    matches = idsse.matches()
    started = time.time()
    touchclf.build_all(matches, touchclf.SEEDS)
    prepare(matches, SEEDS)
    print("prepared {} inputs ({:.0f}s)".format(len(_IN), time.time() - started), flush=True)
    grid = GRIDS[grid_name]
    keys = list(grid)
    configs = [("exp08", {k: "" for k in keys})]
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        configs.append(("touch:" + ",".join(str(v) for v in values), params))
    jobs = [(c, p, m, s) for c, p in configs for m in matches for s in SEEDS]
    with multiprocessing.get_context("fork").Pool(workers) as pool:  # workers inherit _IN
        rows = pool.map(run_one, jobs, chunksize=4)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "idsse_{}.csv".format(grid_name))
    with open(path, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("{} runs in {:.0f}s -> {}".format(len(rows), time.time() - started, path))
    report(rows, matches)


def report(rows, matches):
    def mean(cfg, ms, key="spellF1"):
        return float(np.mean([r[key] for r in rows if r["config"] == cfg and r["match"] in ms]))

    cfgs = sorted({r["config"] for r in rows})
    touch = [c for c in cfgs if c != "exp08"]
    print("\nleave-one-match-out: choose the touch config on 6 matches, score the 7th (3 seeds)")
    print("  {:<8} {:<26} {:>8} {:>8} {:>7} {:>7} {:>9}".format(
        "held", "chosen", "touch", "exp08", "gain", "seed sd", "poss/gt"))
    gains = []
    for held in matches:
        train = [m for m in matches if m != held]
        choice = max(touch, key=lambda c: (round(mean(c, train), 4), mean(c, train, "macroF1")))
        a = [r["spellF1"] for r in rows if r["config"] == choice and r["match"] == held]
        b = [r["spellF1"] for r in rows if r["config"] == "exp08" and r["match"] == held]
        diffs = np.array(sorted_by_seed(rows, choice, held)) - np.array(
            sorted_by_seed(rows, "exp08", held))
        gains.append(diffs.mean())
        poss = mean(choice, [held], "poss")
        gt = mean(choice, [held], "poss_gt")
        print("  {:<8} {:<26} {:>8.3f} {:>8.3f} {:>+7.3f} {:>7.3f} {:>5.0f}/{:<4.0f}".format(
            held[-6:], choice, np.mean(a), np.mean(b), diffs.mean(), diffs.std(), poss, gt))
    print("  mean held-out gain {:+.3f}  (matches improved {}/{})".format(
        np.mean(gains), sum(g > 0 for g in gains), len(gains)))
    best = max(touch, key=lambda c: (round(mean(c, matches), 4), mean(c, matches, "macroF1")))
    print("\nall 7 matches: best touch config {}  spellF1 {:.3f}  macroF1 {:.3f}  team {:.3f}  |  "
          "exp08 spellF1 {:.3f}  macroF1 {:.3f}  team {:.3f}".format(
              best, mean(best, matches), mean(best, matches, "macroF1"),
              mean(best, matches, "team_acc"), mean("exp08", matches),
              mean("exp08", matches, "macroF1"), mean("exp08", matches, "team_acc")))


def build_labels(game=None, data_dir=None, chosen=None):
    """The provided match: exp08 (classifier + whistle dead ball) with the touch model's touches."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run08", os.path.join(EXPERIMENTS, "exp08_deadball_clf", "run.py"))
    run08 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run08)
    game = game or run08.GAME
    chosen = dict(CHOSEN, **(chosen or {}))
    with open(touchclf.MODEL_PATH) as h:
        model = json.load(h)
    blocks = run08.half_video_blocks(game)
    n = run08.GAMES[game]["n_video_frames"]
    length, width = run08.GAMES[game]["pitch"]
    p = run08.players_mod.load(game, data_dir)
    ball = np.stack([(p["ball_x"] - 0.5) * length, (p["ball_y"] - 0.5) * width], axis=1)
    players = np.stack([(p["x"] - 0.5) * length, (p["y"] - 0.5) * width], axis=2)
    team = np.broadcast_to(p["team"], p["x"].shape)
    inside = np.zeros(len(ball), dtype=bool)
    for _h, a, b in blocks:
        inside[a:b + 1] = True
    frames, teams, prob = hook_mod.score_candidates(
        model, ball.astype(np.float64), players.astype(np.float64), team, inside)
    touch_hook = hook_mod.make(frames, teams, prob, chosen["tau"], chosen["confirm_w"])

    features = run08.feat3.load(game, data_dir)
    votes = run08.exp01_team.tag_team_votes(features, game, data_dir)
    dead_hook = tune.replace_dead(run08.classifier_dead(game, data_dir, True), blocks)
    knobs = dict(BASE, **{k: v for k, v in chosen.items() if k not in HOOK_ONLY})
    with harness.with_params(**knobs):
        labels, detail = run04.labels_from_features(
            features, blocks, game, n, votes=votes, mode="none", data_dir=data_dir,
            dead_hook=dead_hook, touch_hook=touch_hook)
    return labels, detail, run08


def write(out, chosen=None):
    labels, _detail, run08 = build_labels(chosen=chosen)
    path = run08.write_submission(out, labels)
    graded = np.zeros(len(labels), dtype=bool)
    for _h, first, last in run08.half_video_blocks(run08.GAME):
        graded[first:last + 1] = True
    print("wrote {} -> {}".format(len(labels), os.path.relpath(path, ROOT)))
    for name in ("H", "A", "D", "X"):
        print("  {}  {:5.1f}%".format(name, 100.0 * np.mean(labels[graded] == name)))


def sorted_by_seed(rows, cfg, match):
    return [r["spellF1"] for r in sorted(
        (r for r in rows if r["config"] == cfg and r["match"] == match), key=lambda r: r["seed"])]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--idsse", action="store_true")
    parser.add_argument("--report", action="store_true", help="re-print from results/idsse_grid.csv")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--grid", default="base", choices=["base", "confirm"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override CHOSEN for a report-only variant, e.g. confirm_w=75")
    args = parser.parse_args()
    if args.idsse:
        evaluate(args.grid)
    elif args.report:
        rows = tune.read_rows(os.path.join(RESULTS, "idsse_{}.csv".format(args.grid)))
        report(rows, idsse.matches())
    else:
        over = {}
        for item in args.set:
            k, v = item.split("=")
            over[k] = v == "True" if v in ("True", "False") else float(v) if "." in v else int(v)
        write(args.out, over)
