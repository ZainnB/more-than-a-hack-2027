"""Is the noise simulator as hard as the real tracking? A label-free comparison.

    python experiments/exp06_idsse/validity.py

Tuning on degraded IDSSE is only worth anything if the degradation looks like the ATD. This puts
the pipeline through both and compares quantities that need no labels at all: what the inputs
look like (ball gaps, how close players are, how often both teams crowd the ball) and what the
pipeline does with them (dead share, which rule fires, touch rate, possession rhythm).

Two kinds of row, and they mean different things:

  INPUT   should match if the simulator is faithful. A mismatch here is a simulator bug.
  OUTPUT  can differ for honest reasons -- IDSSE's matches are not BAR-ATH, and day 1 showed they
          have about twice the possessions per live minute -- so read these as context.

Smart Tagging votes are off on both sides so the comparison is like for like.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import harness  # noqa: E402  (registers the IDSSE game)
import adapter  # noqa: E402
import idsse  # noqa: E402
import noise  # noqa: E402
import scene as scene_mod  # noqa: E402
import possession_metrics as pm  # noqa: E402
from games import GAME as REAL_GAME, half_video_blocks  # noqa: E402
from exp03_keeper_fix import features as feat3  # noqa: E402
from exp02_touch_gated import config as config2  # noqa: E402
from exp04_boundary_timing import run as run04  # noqa: E402

FPS = 25


def measure(features, blocks, game):
    labels, d = run04.labels_from_features(features, blocks, game, len(features["ball_x"]),
                                           votes=None, mode="none")
    codes = pm.label_codes(labels)
    graded = np.zeros(len(codes), dtype=bool)
    for _h, a, b in blocks:
        graded[a:b + 1] = True
    seen = ~np.isnan(features["ball_x"]) & graded
    dead = d["dead"] & graded
    live = graded & ~dead
    dh, da = features["d_home"], features["d_away"]
    both = seen & ~np.isnan(dh) & ~np.isnan(da)
    margin = np.abs(dh - da)
    nearest = np.fmin(dh, da)
    x, y = features["ball_x"], features["ball_y"]
    outside = seen & ((x < 0) | (x > 1) | (y < 0) | (y > 1))
    runs = [(a, b + 1) for _h, a, b in blocks]
    spells = pm.possession_spells(codes, runs)
    dur = np.array([(b - a) / FPS for a, b, _k in spells])
    minutes = graded.sum() / FPS / 60.0
    live_min = live.sum() / FPS / 60.0
    speed = features["mean_speed"][graded]
    kin_speed = d["kin"]["speed"][seen]
    return {
        # inputs
        "ball missing %": 100 * (1 - seen.sum() / graded.sum()),
        "ball missing in pipeline-live %": 100 * (live & ~seen).sum() / max(1, live.sum()),
        "ball outside pitch % (of seen)": 100 * outside.sum() / max(1, seen.sum()),
        "nearest player m, median": np.nanmedian(nearest[seen]),
        "team margin m, p25": np.nanpercentile(margin[both], 25),
        "team margin m, p50": np.nanpercentile(margin[both], 50),
        "team margin m, p75": np.nanpercentile(margin[both], 75),
        "margin >= 2 m % (of seen)": 100 * np.mean(margin[both] >= config2.MARGIN_M),
        "contested % (of seen)": 100 * d["contested"][seen].mean(),
        "touches / visible min": d["touches"][seen].sum() / (seen.sum() / FPS / 60.0),
        "ball speed m/s, p50": np.nanpercentile(kin_speed, 50),
        "ball speed m/s, p90": np.nanpercentile(kin_speed, 90),
        "player speed m/s, median": np.nanmedian(speed),
        "players tracked, median": np.median(features["n_tracked"][graded]),
        # outputs
        "dead share %": 100 * dead.sum() / graded.sum(),
        "  of which out-of-bounds rule %": 100 * (d["parts"]["oob"] & graded).sum() / graded.sum(),
        "  of which slow-pitch rule %": 100 * (d["parts"]["slow"] & graded).sum() / graded.sum(),
        "  of which ball-lost rule %": 100 * (d["parts"]["lost"] & graded).sum() / graded.sum(),
        "restarts / min": sum(1 for r in d["restarts"] if graded[r]) / minutes,
        "possessions / live min": len(spells) / max(live_min, 1e-9),
        "possession median s": np.median(dur),
        "possessions <= 2 s %": 100 * (dur <= 2).mean(),
    }


INPUT_KEYS = ["ball missing %", "ball missing in pipeline-live %", "ball outside pitch % (of seen)",
              "nearest player m, median", "team margin m, p25", "team margin m, p50",
              "team margin m, p75", "margin >= 2 m % (of seen)", "contested % (of seen)",
              "touches / visible min", "ball speed m/s, p50", "ball speed m/s, p90",
              "player speed m/s, median", "players tracked, median"]


if __name__ == '__main__':
    real = measure(feat3.load(), half_video_blocks(REAL_GAME), REAL_GAME)

    sim, clean = [], []
    for match in idsse.matches():
        s = scene_mod.build(match)
        for switches, bucket in ((noise.ATD_LIKE, sim), (noise.CLEAN, clean)):
            ball, players, team = noise.degrade(s, 0, switches)
            bucket.append(measure(adapter.to_features(ball, players, team), s["blocks"],
                                  harness.GAME))

    print("{:<34} {:>9}   {:>22}   {:>11}".format(
        "", "real ATD", "IDSSE degraded (range)", "IDSSE clean"))
    for section, keys in (("INPUT -- should match", INPUT_KEYS),
                          ("OUTPUT -- context", [k for k in real if k not in INPUT_KEYS])):
        print("\n" + section)
        for key in keys:
            vals = np.array([r[key] for r in sim])
            print("  {:<32} {:>9.2f}   {:>7.2f} ({:>5.2f}-{:>5.2f})   {:>11.2f}".format(
                key, real[key], vals.mean(), vals.min(), vals.max(),
                np.mean([r[key] for r in clean])))
    print("\n{} IDSSE matches, seed 0".format(len(sim)))
