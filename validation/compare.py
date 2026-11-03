"""Score every submission against every annotated window, separately and pooled.

    python validation/compare.py
    python validation/compare.py --bootstrap 1000 --csv validation/compare.csv

The scorer ranks against one ground truth at a time. That hides the thing that matters most with
this little data: whether a result holds in every window or comes from one of them. On the first
run of exp02 the two label sets disagreed outright -- the variant that won the 15-minute
validation set by a wide margin was the worst experiment on the public 5 minutes -- and only a
per-window breakdown says why.

Four windows, plus a pooled one:

    public   11185-18685   the organisers' 5 minutes, 1st half
    win1     18685-26185   hand-annotated, 1st half, home-dominated
    win2    143308-150808  hand-annotated, 2nd half, balanced
    win3    178308-185808  hand-annotated, 2nd half, away-leaning, late
    all20    all four      20 minutes, the best single estimate this repository can make

Per-window spellF1 is noisy -- each window holds 7-17 possessions, so one possession moves it by
several points -- and the pooled row has a band of roughly +/-0.1 even at 44.

The hand labels were re-annotated on 2026-11-03 against the organisers' actual convention --
interceptions and clearances count as possessions -- and now run at 2.93 possessions/min against
the official 3.40, from 1.87 on the first pass. On the public window, where the two can be
compared directly, they match 17 possessions to 17 with no unmatched boundaries either way.

So the pooled row is now worth reading on its own terms. Keep checking it against `public`
anyway: that is the only column scored on labels we did not make.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import windows as win  # noqa: E402
import expand as expander  # noqa: E402

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESULTS = os.path.join(ROOT, "evaluation", "results")


def ground_truths():
    """{name: full-length code array with X outside that window}, plus the pooled 'all20'."""
    public = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    hand = pm.label_codes(expander.expand())

    out = {"public": public}
    for name, start, stop in win.all_frames():
        codes = np.full(len(hand), pm.X, dtype=np.int8)
        codes[start:stop] = hand[start:stop]
        out[name] = codes

    pooled = hand.copy()
    graded_public = public != pm.X
    pooled[graded_public] = public[graded_public]
    out["all20"] = pooled
    return out


def summarise(gt, pred, bootstrap_n=0):
    r = pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)
    d = r["per_class"]["D"]
    return {
        "macroF1": r["macro_f1"],
        "spellF1": r["spell_f1_mean"],
        "spell_lo": r["spell_bootstrap_low"],
        "spell_hi": r["spell_bootstrap_high"],
        "poss": r["n_pred_spells"],
        "poss_gt": r["n_gt_spells"],
        "D_prec": d["precision"],
        "D_rec": d["recall"],
        "team_acc": r["team_acc_both_alive"],
        "home_pred": r["pred_home_share"],
        "home_gt": r["gt_home_share"],
        "transF1": r["transitions"][25]["f1"],
    }


def submissions(names=None):
    paths = sorted(glob.glob(os.path.join(RESULTS, "*.csv")))
    out = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if names and name not in names:
            continue
        table = pm.read_label_csv(path)
        if pm.validate_submission(table, len(table)) is None:
            out.append((name, pm.label_codes(table)))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--csv", help="also write every number to this CSV")
    parser.add_argument("--only", nargs="+", help="score only these submissions")
    args = parser.parse_args()

    truths = ground_truths()
    subs = submissions(args.only)
    order = ["public", "win1", "win2", "win3", "all20"]

    rows = []
    for name, pred in subs:
        for window in order:
            s = summarise(truths[window], pred, args.bootstrap if window == "all20" else 0)
            s.update(submission=name, window=window)
            rows.append(s)

    print("READ THE 'public' COLUMN FIRST. It is the only window with the organisers' own labels,")
    print("and validation/calibrate.py measured our hand labels at 35% fewer possessions than")
    print("theirs -- so 'all20' is 15 biased minutes against 5 authoritative ones, and its")
    print("spellF1 rewards merging. macroF1 and the dead-ball columns are far less affected.\n")
    print("spellF1 by window (possessions predicted / true in brackets)\n")
    header = "  {:<20}".format("submission") + "".join(
        "{:>16}".format(w) for w in order)
    print(header)
    by = {(r["submission"], r["window"]): r for r in rows}
    ranked = sorted(subs, key=lambda s: -by[(s[0], "all20")]["spellF1"])
    for name, _pred in ranked:
        line = "  {:<20}".format(name)
        for w in order:
            r = by[(name, w)]
            line += "{:>9.3f} {:>6}".format(r["spellF1"], "{}/{}".format(r["poss"], r["poss_gt"]))
        print(line)

    print("\npooled 20 minutes, everything")
    print("  {:<20} {:>8} {:>8} {:>15} {:>7} {:>7} {:>8} {:>8} {:>11}".format(
        "submission", "macroF1", "spellF1", "spell 5-95%", "D prec", "D rec", "team acc",
        "transF1", "home p/gt"))
    for name, _pred in ranked:
        r = by[(name, "all20")]
        band = "{:.3f}-{:.3f}".format(r["spell_lo"], r["spell_hi"]) if args.bootstrap else "-"
        print("  {:<20} {:>8.4f} {:>8.4f} {:>15} {:>6.1f}% {:>6.1f}% {:>8.4f} {:>8.4f} {:>5.2f}/{:.2f}"
              .format(name, r["macroF1"], r["spellF1"], band, 100 * r["D_prec"],
                      100 * r["D_rec"], r["team_acc"], r["transF1"],
                      r["home_pred"], r["home_gt"]))

    if args.csv:
        keys = ["submission", "window", "macroF1", "spellF1", "spell_lo", "spell_hi", "poss",
                "poss_gt", "D_prec", "D_rec", "team_acc", "transF1", "home_pred", "home_gt"]
        with open(args.csv, "w") as handle:
            handle.write(",".join(keys) + "\n")
            for r in rows:
                handle.write(",".join(str(r[k]) for k in keys) + "\n")
        print("\nwrote {}".format(os.path.relpath(args.csv, ROOT)))
