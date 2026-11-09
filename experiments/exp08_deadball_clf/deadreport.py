"""How good are our stoppages? Frame-level D scores, and start / end timing per stoppage.

    python experiments/exp08_deadball_clf/deadreport.py exp06 exp08b
    python experiments/exp08_deadball_clf/deadreport.py exp06 exp08b --detail public

For every true stoppage in a window: was it found (any overlap with a predicted one), and if so
how far the predicted start and end are from the true ones, in seconds (negative = early). A
predicted stoppage that overlaps no true one is a false stoppage. A report, not a target: nothing
here is used to choose a value.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "validation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import possession_metrics as pm  # noqa: E402
import compare  # noqa: E402
from towatch import video_time  # noqa: E402

FPS = 25.0
WINDOWS = ("public", "win1", "win2", "win3")


def stretches(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def window_bounds(gt):
    graded = np.flatnonzero(gt != pm.X)
    return int(graded[0]), int(graded[-1]) + 1


def evaluate(gt, pred):
    a, b = window_bounds(gt)
    truth = gt[a:b] == pm.D
    mine = pred[a:b] == pm.D
    tp, fp, fn = (truth & mine).sum(), (~truth & mine).sum(), (truth & ~mine).sum()
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    t_runs, p_runs = stretches(truth), stretches(mine)
    matched, start_err, end_err = [], [], []
    for ts, te in t_runs:
        hits = [(ps, pe) for ps, pe in p_runs if ps < te and pe > ts]
        if hits:
            ps, pe = hits[0][0], hits[-1][1]
            # a stoppage clipped by the window edge has no real start or end there
            s_err = (ps - ts) / FPS if ts > 0 else None
            e_err = (pe - te) / FPS if te < len(truth) else None
            matched.append((ts + a, te + a, ps + a, pe + a, s_err, e_err))
            if s_err is not None:
                start_err.append(s_err)
            if e_err is not None:
                end_err.append(e_err)
        else:
            matched.append((ts + a, te + a, None, None, None, None))
    false = [(ps + a, pe + a) for ps, pe in p_runs
             if not any(ps < te and pe > ts for ts, te in t_runs)]
    return {"prec": prec, "rec": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "n_true": len(t_runs), "found": sum(1 for m in matched if m[2] is not None),
            "false": false, "start": start_err, "end": end_err, "matched": matched}


def fmt_err(errs):
    if not errs:
        return "      -      "
    e = np.array(errs)
    return "{:+5.1f}s ({:4.1f})".format(np.median(e), np.median(np.abs(e)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="+", help="names in evaluation/results/")
    parser.add_argument("--detail", help="list every stoppage in this window")
    args = parser.parse_args()

    gts = compare.ground_truths()
    preds = {n: pm.label_codes(pm.read_label_csv(os.path.join(compare.RESULTS, n + ".csv")))
             for n in args.submissions}
    print("start / end: median signed error (median absolute), seconds; negative = early\n")
    print("{:<8} {:<9} {:>5} {:>5} {:>5}  {:>9} {:>6}  {:<15} {:<15}".format(
        "window", "model", "prec", "rec", "F1", "found", "false", "start", "end"))
    pooled = {n: {"start": [], "end": [], "found": 0, "n_true": 0, "false": 0} for n in preds}
    for w in WINDOWS:
        for n, pred in preds.items():
            r = evaluate(gts[w], pred)
            for k in ("start", "end"):
                pooled[n][k] += r[k]
            pooled[n]["found"] += r["found"]
            pooled[n]["n_true"] += r["n_true"]
            pooled[n]["false"] += len(r["false"])
            print("{:<8} {:<9} {:>5.2f} {:>5.2f} {:>5.2f}  {:>4}/{:<4} {:>6}  {:<15} {:<15}".format(
                w, n, r["prec"], r["rec"], r["f1"], r["found"], r["n_true"], len(r["false"]),
                fmt_err(r["start"]), fmt_err(r["end"])))
        print()
    print("all four windows")
    for n, p in pooled.items():
        print("  {:<9} found {:>2}/{:<2}  false {:>2}   start {}   end {}".format(
            n, p["found"], p["n_true"], p["false"], fmt_err(p["start"]), fmt_err(p["end"])))

    if args.detail:
        print("\n{} -- every true stoppage".format(args.detail))
        for n, pred in preds.items():
            r = evaluate(gts[args.detail], pred)
            print("  {}".format(n))
            for ts, te, ps, pe, se, ee in r["matched"]:
                if ps is None:
                    print("    true {}-{}   MISSED".format(video_time(ts), video_time(te)))
                else:
                    print("    true {}-{}   ours {}-{}   start {}  end {}".format(
                        video_time(ts), video_time(te), video_time(ps), video_time(pe),
                        "{:+.1f}s".format(se) if se is not None else "edge",
                        "{:+.1f}s".format(ee) if ee is not None else "edge"))
            for ps, pe in r["false"]:
                print("    FALSE  ours {}-{}".format(video_time(ps), video_time(pe)))
