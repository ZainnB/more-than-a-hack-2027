"""The three validation windows, and the selection that produced them.

    python validation/windows.py              # print the windows
    python validation/windows.py --audit      # re-run the selection over the whole match

Why a second ground truth exists
--------------------------------
The public ground truth is 5 minutes and 17 possessions, and the scorer's own bootstrap says that
at that size **spellF1 gaps below ~0.33 are noise**. exp01 beats baseline_smooth50 by 0.098 there,
which is not a result. The public window is also entirely in the first half, so nothing we have
built has ever been measured on the second.

So these 15 minutes are hand-annotated. RULES.md permits exactly this -- "Small amounts of
hand-labelled data used to train or validate a model are fine and normal. The distinction is
whether the submitted labels come out of a model or out of a person" -- and the two hard lines are
that these labels never enter the submission path, and that the technical document says the
validation set was annotated by hand.

What 15 minutes buys, and what it does not
------------------------------------------
Roughly 50 possessions. Bootstrap width scales about as 1/sqrt(n), so the spellF1 band comes down
from ~0.33 to roughly ~0.19 -- still far too wide to rank a 0.1 difference. Only the full match's
~360 possessions reaches the published 0.04 band.

**This set is not for ranking spellF1.** It is for the quantities that do tighten at this size:
per-frame macro-F1 and per-class precision/recall, dead-ball precision over ~12 stretches instead
of 4, team accuracy over ~50 possessions instead of 17, second-half behaviour, and above all
home-bias: exp01 predicts H on 32.4% of graded frames against A's 20.1%, and nothing we currently
have can say whether that is the match or the model.

How the windows were chosen
---------------------------
On Smart Tagging alone, which is independent of our pipeline. Selecting windows on a feature our
own dead-ball or team layer uses would make the validation partly circular.

They **span** the possession balance rather than clustering at 50/50. This match is genuinely
home-dominated -- the tagging-derived Barcelona share is 0.6-0.88 across most of it -- so a set of
three balanced windows would be measuring a match that was not played. One home-dominated, one
balanced and one away-leaning window tests for bias without pretending the game was even.

None overlaps the public window, so that stays an independent check. None contains a goal:
celebrations are long, trivially detected and would dominate the dead-ball statistics.
"""

import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
# Half-open [start, stop) in video frames, which is also submission row numbers.
WINDOWS = collections.OrderedDict([
    ("win1", {"start": 18685, "stop": 26185, "half": 1, "balance": 0.82,
              "note": "home-dominated; 3 throw-ins and a goal kick"}),
    ("win2", {"start": 143308, "stop": 150808, "half": 2, "balance": 0.62,
              "note": "balanced; 2 corners and 2 goal kicks"}),
    ("win3", {"start": 178308, "stop": 185808, "half": 2, "balance": 0.36,
              "note": "away-leaning, late in the match; 4 throw-ins and 2 goal kicks"}),
])

# The public ground truth's window, excluded from selection so it stays an independent check.
PUBLIC_WINDOW = (11185, 18685)

# Phase codes whose Team tag names the team *in possession*. Used only to estimate the possession
# balance of a candidate window, never as a label. The DEFENDING IN * codes are these same
# intervals mirrored onto the other team, so including them would double-count every phase.
ATTACKING_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
                   "LONG BALL")
HOME_CLUB, AWAY_CLUB = "Barcelona", "Atletic Club"  # as spelled in the Smart Tagging XML

WINDOW_FRAMES = 7500   # 5 minutes at 25 fps
AUDIT_STEP = 2500      # candidate windows every 100 s when auditing


# Windows annotated by hand for CALIBRATION, not validation: they already have an official ground
# truth, and the hand labels are compared against it to measure how closely our annotation
# convention matches the organisers'. Deliberately kept out of WINDOWS so they never enter
# val15.csv, the pooled score, or anything an experiment is tuned on. See calibrate.py.
CALIBRATION = collections.OrderedDict([
    ("public", {"start": PUBLIC_WINDOW[0], "stop": PUBLIC_WINDOW[1], "half": 1, "balance": 0.40,
                "note": "the organisers' public 5 minutes, re-annotated blind"}),
])


def frames(name):
    """(start, stop) for one window, half-open. Validation or calibration."""
    window = WINDOWS[name] if name in WINDOWS else CALIBRATION[name]
    return window["start"], window["stop"]


def all_frames():
    """[(name, start, stop)] for every window, in order."""
    return [(name, w["start"], w["stop"]) for name, w in WINDOWS.items()]


def contains(frame):
    """The window a frame belongs to, or None."""
    for name, start, stop in all_frames():
        if start <= frame < stop:
            return name
    return None


def total_frames():
    return sum(stop - start for _name, start, stop in all_frames())


def audit(game=GAME, data_dir=None):
    """Re-run the window selection over the whole match. Returns every candidate, scored.

    Kept runnable rather than written down as a result, so the choice of windows can be checked
    and so a different trade-off can be made later without reverse-engineering this one.
    """
    import load_smart_tagging as tagging

    events = tagging.load(game, data_dir)
    goals = [(e["start_frame"], e["end_frame"]) for e in events
             if (e["code"] or "").strip().upper() == "GOALS"]

    rows = []
    for half, first, last in half_video_blocks(game):
        for start in range(first, last - WINDOW_FRAMES, AUDIT_STEP):
            stop = start + WINDOW_FRAMES
            set_pieces = [e for e in events
                          if (e["code"] or "").strip().upper() == "SET PIECES"
                          and e["end_frame"] > start and e["start_frame"] < stop]

            held = collections.Counter()
            for event in events:
                if (event["code"] or "").strip().upper() not in ATTACKING_CODES:
                    continue
                if event["end_frame"] <= start or event["start_frame"] >= stop:
                    continue
                club = event["tags"].get("Team")
                if club in (HOME_CLUB, AWAY_CLUB):
                    held[club] += min(stop, event["end_frame"]) - max(start, event["start_frame"])
            total = sum(held.values())

            rows.append({
                "half": half, "start": start, "stop": stop,
                "n_set_pieces": len(set_pieces),
                "balance": (held[HOME_CLUB] / total) if total else 0.5,
                "has_goal": any(gs < stop and ge > start for gs, ge in goals),
                "overlaps_public": stop > PUBLIC_WINDOW[0] and start < PUBLIC_WINDOW[1],
                "types": collections.Counter(e["tags"].get("Type") for e in set_pieces),
            })
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audit", action="store_true",
                        help="re-run the selection over every candidate window")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    print("validation windows -- {} frames, {:.1f} minutes of play".format(
        total_frames(), total_frames() / float(fps) / 60.0))
    for name, window in WINDOWS.items():
        print("  {:<6} {:>7} - {:>7}  half {}  balance {:.2f}  {}".format(
            name, window["start"], window["stop"], window["half"],
            window["balance"], window["note"]))
    print("\n  balance = Barcelona share of tagged attacking-phase time. It is a selection")
    print("  signal from Smart Tagging, not a label, and it is not ground truth.")

    if args.audit:
        print("\ncandidates (** = chosen):")
        chosen = {(w["start"], w["stop"]) for w in WINDOWS.values()}
        for row in audit():
            flag = ""
            if row["overlaps_public"]:
                flag = "public"
            elif row["has_goal"]:
                flag = "goal"
            if (row["start"], row["stop"]) in chosen:
                flag = "** CHOSEN"
            print("   half {}  {:>7}-{:>7}  sp {:>2}  bal {:.2f}  {:<10} {}".format(
                row["half"], row["start"], row["stop"], row["n_set_pieces"],
                row["balance"], flag, dict(row["types"])))
