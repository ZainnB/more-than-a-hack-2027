"""Side-by-side report on the four annotated windows, and where the possessions are lost.

    python experiments/exp10_viterbi/breakdown.py
    python experiments/exp10_viterbi/breakdown.py --only exp08 exp10_viterbi --bootstrap 1000

Report only: nothing here feeds back into a value (PLAN.md section 1).

Two tables.

1. Scores per window -- spellF1 and predicted/true possessions on public, win1, win2, win3 and the
   pooled 20 minutes, with pooled macroF1 and dead-ball precision / recall. Same numbers as
   validation/compare.py, laid out for a few submissions side by side.

2. The short-possession diagnosis. For every true possession, the best IoU any predicted
   possession of the same team reaches (the quantity spellF1 thresholds at 0.3..0.7). Reported as
   the share reaching 0.7 -- the strictest threshold, where a possession is "really" found -- by
   true length, plus the median start and end error of that best match. exp08 matched 0% of
   true possessions under 3 s and 20% of 3-8 s ones; this is the table that says whether a global
   decoder fixed that.

Spells are built exactly as the scorer builds them: X forward-filled inside each window, possessions
as maximal H / A runs, never across a window edge.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
for path in (ROOT, os.path.join(ROOT, "validation"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import compare  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
WINDOWS = ["public", "win1", "win2", "win3", "all20"]
BUCKETS = [("<3 s", 0, 3), ("3-8 s", 3, 8), ("8-20 s", 8, 20), (">20 s", 20, np.inf)]
MATCH_IOU = 0.7


def per_possession(gt, pred):
    """[(true length s, best same-team IoU, start error s, end error s)] for every true possession.

    Errors are predicted minus true, so negative = the predicted possession starts (or ends)
    early. NaN where no predicted possession of that team overlaps at all.
    """
    runs = pm.valid_runs(gt)
    truth = pm.possession_spells(gt, runs)
    preds = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    out = []
    for a, b, team in truth:
        best = (0.0, np.nan, np.nan)
        for c, d, k in preds:
            if k != team or d <= a or b <= c:
                continue
            iou = (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
            if iou > best[0]:
                best = (iou, (c - a) / FPS, (d - b) / FPS)
        out.append(((b - a) / FPS,) + best)
    return out


def main(names, bootstrap):
    truths = compare.ground_truths()
    subs = dict(compare.submissions(names))
    names = [n for n in names if n in subs]

    print("spellF1 by window (predicted / true possessions)\n")
    print("  {:<24}".format("submission") + "".join("{:>16}".format(w) for w in WINDOWS)
          + "{:>9}{:>8}{:>8}".format("macroF1", "D prec", "D rec")
          + ("{:>15}".format("spell 5-95%") if bootstrap else ""))
    for name in names:
        line = "  {:<24}".format(name)
        for w in WINDOWS:
            r = compare.summarise(truths[w], subs[name], bootstrap if w == "all20" else 0)
            line += "{:>9.3f} {:>6}".format(r["spellF1"], "{}/{}".format(r["poss"], r["poss_gt"]))
        line += "{:>9.3f}{:>7.1f}%{:>7.1f}%".format(r["macroF1"], 100 * r["D_prec"],
                                                    100 * r["D_rec"])
        if bootstrap:
            line += "{:>15}".format("{:.3f}-{:.3f}".format(r["spell_lo"], r["spell_hi"]))
        print(line)

    print("\nper true possession, pooled 20 minutes: share matched at IoU >= {} by true length,"
          "\nand median boundary error of the best same-team match (s; negative = early)\n".format(
              MATCH_IOU))
    head = "  {:<24}".format("submission") + "".join(
        "{:>14}".format(label) for label, _lo, _hi in BUCKETS) + "{:>10}{:>16}{:>16}".format(
        "all", "start med|abs|", "end med|abs|")
    print(head)
    counts = None
    for name in names:
        rows = per_possession(truths["all20"], subs[name])
        line = "  {:<24}".format(name)
        counts = []
        for _label, lo, hi in BUCKETS:
            sel = [r for r in rows if lo <= r[0] < hi]
            hit = sum(r[1] >= MATCH_IOU for r in sel)
            counts.append(len(sel))
            line += "{:>14}".format("{:.0f}% ({}/{})".format(
                100.0 * hit / max(1, len(sel)), hit, len(sel)))
        line += "{:>10}".format("{:.0f}%".format(100.0 * np.mean([r[1] >= MATCH_IOU for r in rows])))
        starts = np.array([r[2] for r in rows if np.isfinite(r[2])])
        ends = np.array([r[3] for r in rows if np.isfinite(r[3])])
        line += "{:>16}{:>16}".format(
            "{:+.1f}|{:.1f}".format(np.median(starts), np.median(np.abs(starts))),
            "{:+.1f}|{:.1f}".format(np.median(ends), np.median(np.abs(ends))))
        print(line)
    print("\n  mean best IoU by bucket:")
    for name in names:
        rows = per_possession(truths["all20"], subs[name])
        print("  {:<24}".format(name) + "".join("{:>14.2f}".format(
            np.mean([r[1] for r in rows if lo <= r[0] < hi]) if any(lo <= r[0] < hi for r in rows)
            else np.nan) for _label, lo, hi in BUCKETS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", default=[
        "exp08", "exp10_viterbi", "exp10_viterbi_notouch", "exp10_viterbi_notag",
        "exp10_viterbi_apriori"])
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    main(args.only, args.bootstrap)
