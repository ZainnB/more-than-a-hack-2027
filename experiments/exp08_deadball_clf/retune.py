"""Re-tune exp06's parameters with a realistic team-flip rate in the simulator.

    python experiments/exp08_deadball_clf/retune.py

exp06 was tuned with 7% of player-frames carrying the wrong team, the figure behind the ATD's
"about 93% team attribution". exp07 then checked near-ball labels against shirt colour and found
them wrong about 1% of the time on clearly visible players; the 93% mixed a wrong team label with
the nearest player simply not being the one in possession. 1.5% is taken here, the middle of the
1-2% that finding supports. Same three stages as exp06 (A, B2, A), same 7 matches x 2 seeds.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import tune  # noqa: E402

sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results")
FLIP_RATE = 0.015
OVERRIDES = {"flip_rate": FLIP_RATE}


def run(dead_fn=None, prefix="flip15"):
    a, _ = tune.run_stage("A", {}, prefix + "_A", results_dir=RESULTS, overrides=OVERRIDES,
                          dead_fn=dead_fn)
    if dead_fn is None:
        b, _ = tune.run_stage("B2", a, prefix + "_B2", results_dir=RESULTS, overrides=OVERRIDES)
    else:
        b = a  # the dead-ball rules are replaced, so their parameters are not tuned
    final, score = tune.run_stage("A", b, prefix + "_A2", results_dir=RESULTS,
                                  overrides=OVERRIDES, dead_fn=dead_fn)
    return final, score


def classifier_dead_fn():
    """A dead_fn for tune.prepare: the classifier trained WITHOUT the scene's own match."""
    import deadclf
    import deadfeat
    with open(deadclf.MODEL_PATH) as h:
        decode_params = __import__("json").load(h)["decode"]
    data = deadclf.dataset()
    models = {m: deadclf.train(data, exclude=m) for m in sorted({k[0] for k in data})}

    def dead_fn(scene, ball, players, _team):
        p = deadclf.proba(models[scene["match"]], deadfeat.compute(ball, players, *deadclf.HALF))
        return deadclf.decode(p, scene["blocks"], **decode_params)
    return dead_fn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classifier", action="store_true",
                        help="replace the dead-ball rules with deadclf's model (held out per match)")
    args = parser.parse_args()
    if args.classifier:
        final, score = run(dead_fn=classifier_dead_fn(), prefix="clf")
        print("\nFINAL with the classifier: {}  spellF1 {:.3f}  macroF1 {:.3f}".format(
            final, score["spellF1"], score["macroF1"]))
        raise SystemExit(0)
    final, score = run()
    print("\nFINAL at {:.1%} flips: {}  spellF1 {:.3f}  macroF1 {:.3f}".format(
        FLIP_RATE, final, score["spellF1"], score["macroF1"]))
