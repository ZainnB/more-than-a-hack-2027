"""Does IDSSE's BallPossession / BallStatus behave like the public ground truth?

    python experiments/exp06_idsse/day1_check.py J03WMX

The go / no-go for tuning on IDSSE. CHALLENGE.md says the ground truth is derived mechanically
from corrected tracking, so the question is whether IDSSE's flags -- also corrected tracking --
share the *behaviour* we found in the public window by inspection:

  1. possession changes while the ball is in the air, not at a touch
  2. interceptions and clearances count as possessions, so spells can be very short
  3. dead ball starts at the whistle / stoppage, not at the instant the ball crosses the line
  4. the rhythm: dead share, possessions per live minute, stoppage lengths

The same measurements are made on the public GT where it has one, so each row is like for like.
Only aggregate numbers are printed.
"""

import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import idsse  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
HALF_X, HALF_Y = 52.5, 34.0

# Wall of the ball being "in the air" for the flight analysis. 1 m is above any foot-level pass.
AIRBORNE_M = 1.0


def codes_for(entry):
    """H / A / D / X code per frame: D where BallStatus is 0, else the possessing team."""
    ball = entry["ball"]["extra"]
    poss, status = ball["BallPossession"], ball["BallStatus"]
    codes = np.full(len(poss), pm.X, dtype=np.int8)
    live = status == 1
    codes[live & (poss == 1)] = pm.H
    codes[live & (poss == 2)] = pm.A
    codes[status == 0] = pm.D
    return codes


def spell_stats(codes, label):
    n = len(codes)
    spells = pm.possession_spells(codes, [(0, n)])
    dead = codes == pm.D
    live_min = (~dead).sum() / FPS / 60.0
    dur = np.array([(b - a) / FPS for a, b, _k in spells])
    # dead stretches
    edges = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    dlen = (stops - starts) / FPS
    print("  {:<22} dead {:>5.1f}%  poss {:>4}  per live-min {:>4.1f}  dur median {:>4.1f}s  "
          "<=1s {:>3.0f}%  <=2s {:>3.0f}%  | stoppages {:>3}  median {:>4.1f}s  min {:>4.1f}s  "
          "<5s {:>3.0f}%".format(
              label, 100.0 * dead.mean(), len(spells), len(spells) / max(live_min, 1e-9),
              np.median(dur), 100.0 * (dur <= 1).mean(), 100.0 * (dur <= 2).mean(),
              len(dlen), np.median(dlen) if len(dlen) else 0, dlen.min() if len(dlen) else 0,
              100.0 * (dlen < 5).mean() if len(dlen) else 0))
    return spells, starts, stops


def nearest_by_team(entry, team):
    """Distance from the ball to the nearest tracked player of `team`, per frame."""
    bx, by = entry["ball"]["x"], entry["ball"]["y"]
    best = np.full(len(bx), np.inf)
    for t in entry["tracks"]:
        if t["team"] != team:
            continue
        d = np.hypot(t["x"] - bx, t["y"] - by)
        best = np.fmin(best, np.where(np.isnan(d), np.inf, d))
    return np.where(np.isinf(best), np.nan, best)


def flip_analysis(entry, codes, label):
    """Where in the ball's flight, and next to whom, does a live H<->A flip happen?"""
    z = entry["ball"]["extra"]["Z"]
    bx, by = entry["ball"]["x"], entry["ball"]["y"]
    d = [nearest_by_team(entry, idsse.HOME), nearest_by_team(entry, idsse.AWAY)]
    flips = [i for i in range(1, len(codes))
             if codes[i] != codes[i - 1] and codes[i] in (pm.H, pm.A)
             and codes[i - 1] in (pm.H, pm.A)]
    if not flips:
        print("  {}: no live flips".format(label))
        return
    flips = np.array(flips)
    incoming = np.where(codes[flips] == pm.H, 0, 1)
    d_in = np.array([d[t][f] for t, f in zip(incoming, flips)])
    airborne = z[flips] > AIRBORNE_M
    speed = np.hypot(np.gradient(bx), np.gradient(by)) * FPS
    print("  {}: {} live flips ({} per 5 live-min)".format(
        label, len(flips), "{:.1f}".format(len(flips) / ((codes != pm.D).sum() / FPS / 60) * 5)))
    print("      ball airborne (Z>{:.0f} m) at flip: {:.0f}%   ball speed at flip: median {:.1f} m/s"
          .format(AIRBORNE_M, 100.0 * airborne.mean(), np.median(speed[flips])))
    print("      incoming team's nearest player at flip: median {:.1f} m   <=1.5 m: {:.0f}%   "
          ">3 m: {:.0f}%".format(np.nanmedian(d_in), 100.0 * np.nanmean(d_in <= 1.5),
                                 100.0 * np.nanmean(d_in > 3)))
    print("      same, by lag (frames) -> median distance, incoming team:")
    row = []
    for lag in (-50, -25, -10, 0, 10, 25, 50):
        vals = np.array([d[t][min(max(f + lag, 0), len(codes) - 1)]
                         for t, f in zip(incoming, flips)])
        row.append("{:+d}: {:.1f}".format(lag, np.nanmedian(vals)))
    print("      " + "   ".join(row))


def dead_start_analysis(entry, codes, events, label):
    """How does a live -> dead transition relate to the ball leaving the pitch?"""
    bx, by = entry["ball"]["x"], entry["ball"]["y"]
    out = (np.abs(bx) > HALF_X) | (np.abs(by) > HALF_Y)
    dead = codes == pm.D
    starts = np.flatnonzero(np.diff(dead.astype(np.int8)) == 1) + 1
    if not len(starts):
        return
    lead, kinds = [], []
    for s in starts:
        # frames since the ball was last inside the pitch, looking back from the dead start
        back = 0
        while s - back - 1 >= 0 and out[s - back - 1] and back < 500:
            back += 1
        lead.append(back)
        kinds.append(out[max(s - 1, 0)] or out[s])
    lead = np.array(lead)
    kinds = np.array(kinds)
    print("  {}: {} live->dead transitions".format(label, len(starts)))
    print("      ball already outside the pitch at dead start: {:.0f}%   (so {:.0f}% start with the "
          "ball in play: whistle / foul / offside)".format(100.0 * kinds.mean(),
                                                           100.0 * (1 - kinds.mean())))
    if kinds.any():
        # for ball-out stoppages: time the ball had already been out when status flipped
        print("      ball-out starts: ball had been outside for median {:.2f}s  p90 {:.2f}s "
              "before BallStatus went 0".format(np.median(lead[kinds]) / FPS,
                                               np.percentile(lead[kinds], 90) / FPS))
    return starts


def event_alignment(entry, starts, events_path):
    """Time from the nearest stoppage-type event to each in-play dead start."""
    text = open(events_path, encoding="utf-8").read()
    wanted = ("Foul", "Offside", "FinalWhistle", "RefereeBall", "Penalty")
    found = []
    for m in re.finditer(r'<Event [^>]*EventTime="([^"]+)"[^>]*>\s*<([A-Za-z]+)', text):
        if m.group(2) in wanted:
            found.append((_epoch(m.group(1)), m.group(2)))
    if not found:
        return
    # frame -> wall-clock via the ball track timestamps is not stored; use frame numbers only
    return found


def _epoch(stamp):
    return datetime.fromisoformat(stamp).timestamp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("match", help="e.g. J03WMX")
    args = parser.parse_args()

    data = idsse.load(args.match)
    print("IDSSE {} -- BallStatus / BallPossession\n".format(args.match))

    print("1. RHYTHM  (poss = maximal run of one team with the ball alive, per CHALLENGE.md)")
    for half, entry in data.items():
        spell_stats(codes_for(entry), "IDSSE " + half)

    sys.path.insert(0, os.path.join(ROOT, "validation"))
    import compare  # noqa: E402
    gt = pm.label_codes(pm.read_label_csv(compare.PUBLIC_GT))
    import windows as win  # noqa: E402
    a, b = win.PUBLIC_WINDOW
    spell_stats(gt[a:b].copy(), "PUBLIC GT (5 min)")

    print("\n2. WHERE DO POSSESSION CHANGES HAPPEN  (live H<->A flips)")
    for half, entry in data.items():
        flip_analysis(entry, codes_for(entry), "IDSSE " + half)
    print("  PUBLIC GT (tracking, no height): incoming team's nearest player median >= 2.0 m at every"
          " lag from -50 to +75;\n      official change falls median 59% of the way through the"
          " ball's flight (12 changes)")

    print("\n3. HOW DOES DEAD BALL START")
    for half, entry in data.items():
        dead_start_analysis(entry, codes_for(entry), None, "IDSSE " + half)
