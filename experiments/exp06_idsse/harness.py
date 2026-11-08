"""Run the exp04 pipeline on IDSSE matches, clean or degraded to look like the ATD, and score it.

    python experiments/exp06_idsse/harness.py                       # every downloaded match
    python experiments/exp06_idsse/harness.py --matches J03WMX WN1 --seeds 2
    python experiments/exp06_idsse/harness.py --conditions clean atd

This is the loop all tuning on external data runs through: build a scene, degrade it, extract the
features the pipeline expects, run the pipeline, and score against IDSSE's own flags with the
repo's scorer (evaluation/possession_metrics.py), unchanged.

Parameters are set through `with_params`, which edits the exp01/exp02 config modules for the
duration of a run and restores them. Nothing here reads the provided match or its ground truth.
"""

import argparse
import contextlib
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

import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import scene as scene_mod  # noqa: E402
import possession_metrics as pm  # noqa: E402
import games  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402
from exp02_touch_gated import config as config2  # noqa: E402
from exp01_geometry_hysteresis import config as config1  # noqa: E402

GAME = "IDSSE"

# The pipeline looks up pitch size and frame rate by game name. Registering IDSSE here, at run
# time, leaves games.py (the bundle's own file) untouched.
games.GAMES[GAME] = {"n_video_frames": 0, "fps": adapter.FPS, "pitch": adapter.PITCH}

CONDITIONS = {"clean": noise.CLEAN, "atd": noise.ATD_LIKE}

# Every knob a tuning run may turn, and where it lives.
KNOBS = {
    "margin_m": (config2, "MARGIN_M"),
    "k_fallback": (config2, "K_FALLBACK_FRAMES"),
    "use_touch": (config2, "USE_TOUCH"),
    "min_possession_s": (config2, "MIN_POSSESSION_S"),
    "max_dead": (config1, "MAX_DEAD_FRAMES"),
    "slow_speed": (config1, "SLOW_PITCH_SPEED"),
    "slow_min_frames": (config1, "SLOW_PITCH_MIN_FRAMES"),
    "ball_lost_frames": (config1, "BALL_LOST_DEAD_FRAMES"),
}


@contextlib.contextmanager
def with_params(**params):
    """Set config values for the duration of a block, then put them back."""
    saved = {}
    for name, value in params.items():
        module, attr = KNOBS[name]
        saved[name] = getattr(module, attr)
        setattr(module, attr, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            module, attr = KNOBS[name]
            setattr(module, attr, value)


def predict(scene, seed=0, switches=noise.ATD_LIKE):
    """The pipeline's code array for a scene after the chosen degradation."""
    ball, players, team = noise.degrade(scene, seed, switches)
    features = adapter.to_features(ball, players, team)
    labels, _detail = run04.labels_from_features(
        features, scene["blocks"], GAME, scene["n"], votes=None, mode="none")
    return pm.label_codes(labels)


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    d = r["per_class"]["D"]
    return {"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"], "poss": r["n_pred_spells"],
            "poss_gt": r["n_gt_spells"], "D_prec": d["precision"], "D_rec": d["recall"],
            "team_acc": r["team_acc_both_alive"], "transF1": r["transitions"][25]["f1"]}


def evaluate(matches, conditions=("clean", "atd"), seeds=(0,), params=None, scenes=None,
             verbose=True):
    """{condition: [per-run score dicts]} over matches x seeds. Clean ignores the seed."""
    params = params or {}
    scenes = scenes or {m: scene_mod.build(m) for m in matches}
    results = {}
    for name in conditions:
        runs = []
        for match in matches:
            for seed in (seeds if name != "clean" else seeds[:1]):
                started = time.time()
                with with_params(**params):
                    pred = predict(scenes[match], seed, CONDITIONS[name])
                r = score(scenes[match]["gt"], pred)
                r.update(match=match, seed=seed)
                runs.append(r)
                if verbose:
                    print("  {:<6} {:<8} seed {}  spellF1 {:.3f}  macroF1 {:.3f}  D {:.2f}/{:.2f}  "
                          "team {:.3f}  poss {}/{}   ({:.0f}s)".format(
                              name, match[-6:], seed, r["spellF1"], r["macroF1"], r["D_prec"],
                              r["D_rec"], r["team_acc"], r["poss"], r["poss_gt"],
                              time.time() - started))
        results[name] = runs
    return results


def summarise(results):
    print("\n  {:<8} {:>9} {:>9} {:>8} {:>8} {:>9} {:>9} {:>6}".format(
        "", "spellF1", "macroF1", "D prec", "D rec", "team acc", "poss", "runs"))
    for name, runs in results.items():
        mean = lambda key: float(np.mean([r[key] for r in runs]))  # noqa: E731
        print("  {:<8} {:>9.3f} {:>9.3f} {:>8.3f} {:>8.3f} {:>9.3f} {:>5.0f}/{:<4.0f} {:>3}".format(
            name, mean("spellF1"), mean("macroF1"), mean("D_prec"), mean("D_rec"),
            mean("team_acc"), mean("poss"), mean("poss_gt"), len(runs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="+", help="match ids, with or without DFL-MAT-")
    parser.add_argument("--seeds", type=int, default=1, help="noise seeds per match")
    parser.add_argument("--conditions", nargs="+", default=["clean", "atd"], choices=list(CONDITIONS))
    args = parser.parse_args()

    available = idsse.matches()
    wanted = available if not args.matches else [
        m for m in available if any(m.endswith(w.replace("DFL-MAT-", "")) for w in args.matches)]
    print("matches: {}   conditions: {}   seeds: {}".format(
        [m[-6:] for m in wanted], args.conditions, args.seeds))
    print("config: margin {} m, k {}, touch {}, max dead {}\n".format(
        config2.MARGIN_M, config2.K_FALLBACK_FRAMES, config2.USE_TOUCH, config1.MAX_DEAD_FRAMES))
    summarise(evaluate(wanted, args.conditions, tuple(range(args.seeds))))
