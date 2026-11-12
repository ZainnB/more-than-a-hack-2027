"""Pick an unused 5-minute window at random and print what the model did there, and why.

    python validation/blind.py                      # pick, report, and save the pick
    python validation/blind.py --seed 7             # a different window, reproducibly
    python validation/blind.py --window win4        # re-print a window already picked
    python validation/blind.py --out notes.txt

Why this exists, and why it is the opposite of annotate.py
----------------------------------------------------------
Every window we have measured on was annotated first and scored second. That order has a known
failure: twice now a decision was tuned against hand labels that turned out to follow a different
convention from the organisers', and twice it had to be reversed. Annotating first also means the
labeller has already decided what the answer is before ever seeing what the model claimed, so the
only question left is "who is right", never "why did it do that".

This reverses the order. The window is chosen by a seeded random draw -- not by balance, not by
set-piece count, not by anything our pipeline computes -- and the model's output is written down
in full **before** anybody watches the video. The report is a list of claims with timestamps. The
video then adjudicates them one at a time.

What that buys: the failure modes come back as *categories* ("it holds through every aerial duel",
"it restarts possession two seconds late on throw-ins") rather than as a frame-level disagreement
rate, and a category is something that can be fixed. What it costs: no score. There is no ground
truth for this window and none is produced here, so nothing in this file can be tuned on.

Everything printed is the model's claim, never a label. `validation/labels/` is not read or
written by this module.
"""

import argparse
import collections
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXPERIMENTS = os.path.join(ROOT, "experiments")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))

import possession_metrics as pm  # noqa: E402
import windows as win  # noqa: E402
from towatch import video_time, match_time  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
WINDOW_FRAMES = 7500   # 5 minutes at 25 fps, the same size as every other window
DEFAULT_SEED = 1

# A window is not drawn if it comes within this many frames of a half boundary: the first and last
# seconds of a half are atypical (players in formation, ball dead) and would waste the draw.
EDGE_MARGIN = 500

# Flags. None of these is an error -- they are the things most worth having an eye open for while
# watching, chosen because each one has caused a real bug before.
LONG_POSSESSION_S = 25.0   # longer than any plausible uninterrupted possession
SHORT_POSSESSION_S = 0.6   # at or below the merge threshold, so it survived on purpose
LONG_DEAD_S = 45.0         # longer than a throw-in or a goal kick should ever take
LONG_X_S = 2.0             # an abstention this long means the half's opening frames went unclaimed

# Where a drawn window is recorded, so the pick cannot be quietly re-rolled after seeing it.
PICKS = os.path.join(HERE, "picks.csv")


def used_spans():
    """Every frame span already annotated or published, which the draw must avoid."""
    spans = [(start, stop) for _name, start, stop in win.all_frames()]
    spans.append(win.PUBLIC_WINDOW)
    spans.extend((row["start"], row["stop"]) for row in read_picks().values())
    return spans


def read_picks():
    """{name: {start, stop, seed}} for windows already drawn."""
    out = collections.OrderedDict()
    if not os.path.exists(PICKS):
        return out
    with open(PICKS) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("name,"):
                continue
            name, start, stop, seed = line.split(",")[:4]
            out[name] = {"start": int(start), "stop": int(stop), "seed": int(seed)}
    return out


def pick(seed=DEFAULT_SEED, game=GAME, size=WINDOW_FRAMES):
    """(start, stop) for a uniformly random window that overlaps nothing already used.

    Uniform over legal start frames, not over a coarse grid, so the draw cannot be steered by the
    choice of grid. The seed is recorded with the pick for the same reason.
    """
    legal = []
    for _half, first, last in half_video_blocks(game):
        lo, hi = first + EDGE_MARGIN, last - EDGE_MARGIN - size
        for start in range(lo, hi + 1):
            if not any(start < stop and start + size > begin for begin, stop in used_spans()):
                legal.append(start)
    if not legal:
        raise SystemExit("no window of {} frames is left unused".format(size))
    start = random.Random(seed).choice(legal)
    return start, start + size


def save_pick(name, start, stop, seed):
    fresh = not os.path.exists(PICKS)
    with open(PICKS, "a") as handle:
        if fresh:
            handle.write("# Blind windows drawn by blind.py. Recorded at the moment of the draw,\n")
            handle.write("# before the report was read, so a pick cannot be re-rolled after\n")
            handle.write("# seeing it. These are NOT labels -- see validation/labels/ for those.\n")
            handle.write("name,start,stop,seed\n")
        handle.write("{},{},{},{}\n".format(name, start, stop, seed))


def tagging_context(start, stop, game=GAME, data_dir=None):
    """What Smart Tagging says is in the window. Context for watching, not a label."""
    try:
        import load_smart_tagging as tagging
        events = tagging.load(game, data_dir)
    except Exception as error:                                   # noqa: BLE001
        return {"error": str(error)}

    inside = [e for e in events if e["end_frame"] > start and e["start_frame"] < stop]
    codes = collections.Counter((e["code"] or "").strip().upper() for e in inside)
    set_pieces = collections.Counter(
        e["tags"].get("Type") for e in inside
        if (e["code"] or "").strip().upper() == "SET PIECES")

    held = collections.Counter()
    for event in inside:
        if (event["code"] or "").strip().upper() not in win.ATTACKING_CODES:
            continue
        club = event["tags"].get("Team")
        if club in (win.HOME_CLUB, win.AWAY_CLUB):
            held[club] += min(stop, event["end_frame"]) - max(start, event["start_frame"])
    total = sum(held.values())
    return {
        "balance": (held[win.HOME_CLUB] / total) if total else None,
        "set_pieces": set_pieces,
        "goals": codes.get("GOALS", 0),
        "events": sum(codes.values()),
    }


def segments(labels, start, stop):
    """[(a, b, label)] runs of one label inside the window, clipped to it."""
    codes = pm.label_codes(labels)
    window = codes[start:stop]
    edges = np.flatnonzero(np.diff(window)) + 1
    bounds = np.concatenate(([0], edges, [len(window)]))
    return [(int(start + bounds[i]), int(start + bounds[i + 1]), pm.LABELS[window[bounds[i]]])
            for i in range(len(bounds) - 1)]


WHY = {1: "touch", 2: "evidence", 3: "restart"}


def reason(detail, frame, radius=40):
    """The pipeline's own recorded reason for the boundary nearest `frame`.

    `why` is stamped before merge_short and the snap moved anything, so the mark for a final
    boundary can sit a few frames away from it. Reported with its offset rather than silently
    matched, so a large offset is visible as what it is.
    """
    why = detail["why"]
    lo, hi = max(0, frame - radius), min(len(why), frame + radius + 1)
    marks = np.flatnonzero(why[lo:hi]) + lo
    moved = {new: (old, anchor) for old, new, anchor in detail["moves"]}
    snap = ""
    for new, (old, anchor) in moved.items():
        if abs(new - frame) <= 2:
            snap = " +snap({},{:+d})".format(anchor, new - old)
            break
    if not marks.size:
        return "held" + snap
    near = int(marks[np.argmin(np.abs(marks - frame))])
    offset = near - frame
    return "{}{}{}".format(WHY.get(int(why[near]), "?"),
                           "" if offset == 0 else "@{:+d}".format(offset), snap)


def dead_reason(detail, a, b):
    """What produced a dead stretch, and whether a restart was detected.

    `parts` holds the geometry rules' masks. When a dead_hook replaced the mask (exp08's
    classifier), those masks no longer describe the stretch, so they are reported as what the
    rules *would* have said, and the whistle that lowered the threshold is named instead.
    """
    fired = [name for name, mask in detail["parts"].items() if mask[a:b].any()]
    restart = [r for r in detail["restarts"] if a - 50 <= r < b + 50]
    if not detail.get("hooked"):
        return "+".join(sorted(fired)) or "none", restart
    note = "clf"
    whistles = detail.get("whistles", [])
    near = [w for w in whistles if a - 250 <= w <= b]
    if near:
        note += "+whistle@{:+.1f}s".format((near[0] - a) / 25.0)
    if fired:
        note += " (rules: {})".format("+".join(sorted(fired)))
    return note, restart


def quality(detail, start, stop):
    """Tracking-quality facts for the window. Reasons to distrust a stretch, not errors."""
    features = detail["features"]
    ball = features["ball_x"][start:stop]
    tracked = features["n_tracked"][start:stop]
    return {
        "ball_missing": float(np.mean(np.isnan(ball))),
        "over_tracked": float(np.mean(tracked > 22)),
        "median_tracked": float(np.median(tracked)),
        "touches": int(detail["touches"][start:stop].sum()),
        "contested": float(np.mean(detail["contested"][start:stop])),
    }


def other_models(start, stop, labels):
    """[(name, frame agreement with the primary, possession count)] for every saved submission."""
    import glob
    out = []
    primary = pm.label_codes(labels)[start:stop]
    for path in sorted(glob.glob(os.path.join(ROOT, "evaluation", "results", "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            codes = pm.label_codes(pm.read_label_csv(path))
        except Exception:                                        # noqa: BLE001
            continue
        if len(codes) != GAMES[GAME]["n_video_frames"]:
            continue
        spells = pm.possession_spells(codes, [(start, stop)])
        out.append((name, float(np.mean(codes[start:stop] == primary)), len(spells)))
    return out


def render(name, start, stop, seed, detail, labels):
    fps = GAMES[GAME]["fps"]
    lines = []
    runs = [(start, stop)]
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, runs)
    segs = segments(labels, start, stop)
    context = tagging_context(start, stop)
    q = quality(detail, start, stop)

    lines.append("BLIND WINDOW {}   frames {}-{}   seed {}".format(name, start, stop, seed))
    lines.append("video {} - {}     match {} - {}     half {}".format(
        video_time(start), video_time(stop), match_time(start), match_time(stop),
        1 if start < 110808 else 2))
    lines.append("")
    lines.append("Nothing below is a label. It is what exp04 claims, written down before the")
    lines.append("video was watched, so watching it can only confirm or refute -- not score.")

    # -- what is in the window, from Smart Tagging ---------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("CONTEXT (Smart Tagging, independent of our pipeline)")
    if "error" in context:
        lines.append("  unavailable: {}".format(context["error"]))
    else:
        lines.append("  tagged attacking-phase balance   {}".format(
            "{:.2f} home".format(context["balance"]) if context["balance"] is not None else "n/a"))
        lines.append("  set pieces   {}".format(dict(context["set_pieces"]) or "none"))
        lines.append("  goals        {}".format(context["goals"]))

    # -- the frame mix ---------------------------------------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("FRAME MIX")
    for label in pm.LABELS:
        count = int(np.sum(codes[start:stop] == pm.LABELS.index(label)))
        lines.append("  {}  {:>6} frames  {:>5.1f}%  {:>6.1f}s".format(
            label, count, 100.0 * count / (stop - start), count / float(fps)))

    # -- the claims, in order --------------------------------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("THE CLAIMS, IN ORDER   ({} possessions, {} dead stretches)".format(
        len(spells), sum(1 for _a, _b, k in segs if k == "D")))
    lines.append("  'why' is the pipeline's own recorded reason, at its offset in frames from the")
    lines.append("  boundary. touch = a detected contact; evidence = {} frames of the other team".format(
        _k_frames()))
    lines.append("  being nearer by the margin, backdated; restart = first claim after a stoppage.")
    lines.append("")
    lines.append("  {:<4} {:<10} {:<9} {:>7}  {:<5} {}".format(
        "#", "video", "match", "dur", "says", "why / rules"))
    index = 0
    for a, b, label in segs:
        seconds = (b - a) / float(fps)
        flag = ""
        if label in ("H", "A"):
            index += 1
            note = reason(detail, a)
            if seconds > LONG_POSSESSION_S:
                flag = "  <-- long for one possession"
            elif seconds <= SHORT_POSSESSION_S:
                flag = "  <-- survived the merge threshold"
            tag = "{:<4}".format(index)
        elif label == "D":
            rules, restart = dead_reason(detail, a, b)
            note = "dead: {}{}".format(rules, "  restart@{}".format(
                video_time(restart[0])) if restart else "  no restart found")
            if seconds > LONG_DEAD_S:
                flag = "  <-- long stoppage"
            tag = "    "
        else:
            note = "abstain"
            if seconds > LONG_X_S:
                flag = "  <-- long abstention"
            tag = "    "
        lines.append("  {} {:<10} {:<9} {:>6.1f}s  {:<5} {}{}".format(
            tag, video_time(a), match_time(a), seconds, label, note, flag))

    # -- shape of the possessions ----------------------------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("SHAPE")
    if spells:
        durations = np.array([(b - a) / float(fps) for a, b, _k in spells])
        home = sum(1 for _a, _b, k in spells if k == pm.H)
        lines.append("  possessions         {}   ({} home, {} away)".format(
            len(spells), home, len(spells) - home))
        lines.append("  per minute          {:.1f}".format(
            len(spells) / ((stop - start) / float(fps) / 60.0)))
        lines.append("  duration  median {:.1f}s   min {:.1f}s   max {:.1f}s   "
                     "p90 {:.1f}s".format(np.median(durations), durations.min(),
                                          durations.max(), np.percentile(durations, 90)))
        lines.append("  model's home share  {:.2f}{}".format(
            np.sum([b - a for a, b, k in spells if k == pm.H]) /
            float(np.sum([b - a for a, b, _k in spells])),
            "   (tagging says {:.2f})".format(context["balance"])
            if context.get("balance") is not None else ""))
    lines.append("  dead stretches      {}".format(sum(1 for _a, _b, k in segs if k == "D")))
    lines.append("  restarts detected   {}".format(
        sum(1 for r in detail["restarts"] if start <= r < stop)))

    # -- reasons to distrust a stretch -----------------------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("TRACKING QUALITY  (why a claim might be wrong through no fault of the logic)")
    lines.append("  ball untracked      {:.1f}% of frames".format(100.0 * q["ball_missing"]))
    lines.append("  >22 players tracked {:.1f}% of frames   median {:.0f}".format(
        100.0 * q["over_tracked"], q["median_tracked"]))
    lines.append("  touches detected    {}   ({:.1f}/min)".format(
        q["touches"], q["touches"] / ((stop - start) / float(fps) / 60.0)))
    lines.append("  contested frames    {:.1f}%".format(100.0 * q["contested"]))

    # -- where the models disagree ---------------------------------------------------------------
    lines.append("\n" + "-" * 76)
    lines.append("THE OTHER MODELS ON THE SAME WINDOW")
    lines.append("  agreement is frame-level against exp04, which is the one described above.")
    lines.append("")
    lines.append("  {:<22} {:>10} {:>12}".format("model", "agreement", "possessions"))
    for model, agreement, count in other_models(start, stop, labels):
        lines.append("  {:<22} {:>9.1f}% {:>12}".format(model, 100.0 * agreement, count))

    lines.append("\n" + "-" * 76)
    lines.append("HOW TO WATCH THIS")
    lines.append("  Seek to the video times above in the tactical mp4; they are frame/25, the same")
    lines.append("  clock validation/annotate.py shows. For each numbered possession ask only:")
    lines.append("    1. did possession really change there, within about half a second?")
    lines.append("    2. is the team right?")
    lines.append("    3. for every 'dead' row -- was the ball actually out of play?")
    lines.append("  Write what you see against the row number. A pattern across several rows is")
    lines.append("  worth more than any single row, and is the thing to bring back.")
    return "\n".join(lines)


def side_by_side(start, stop, detail, labels, truth, truth_name="official"):
    """The truth and the model on one timeline, one row per stretch where neither changes.

    Rows are cut at every boundary of *either* labelling, so a disagreement is a row of its own
    rather than something to be worked out by lining up two lists. '#' numbers each side's
    possessions where they begin; 'why/rules' is the model's reason, given only on rows where
    the model's own label changes.
    """
    fps = GAMES[GAME]["fps"]
    ours = pm.label_codes(labels)[start:stop]
    theirs = truth[start:stop]
    cuts = np.flatnonzero((np.diff(ours) != 0) | (np.diff(theirs) != 0)) + 1
    bounds = np.concatenate(([0], cuts, [stop - start]))

    lines = ["  {:<10} {:<9} {:>6}  {:<3} {:<5} {:<3} {:<5} {:<2} {}".format(
        "video", "match", "dur", "#", "them", "#", "ours", "", "why/rules (ours)")]
    n_theirs = n_ours = 0
    disagree = 0
    for i in range(len(bounds) - 1):
        a, b = start + int(bounds[i]), start + int(bounds[i + 1])
        t, o = pm.LABELS[theirs[bounds[i]]], pm.LABELS[ours[bounds[i]]]
        t_new = i == 0 or theirs[bounds[i]] != theirs[bounds[i] - 1]
        o_new = i == 0 or ours[bounds[i]] != ours[bounds[i] - 1]
        t_num = o_num = ""
        if t_new and t in ("H", "A"):
            n_theirs += 1
            t_num = str(n_theirs)
        if o_new and o in ("H", "A"):
            n_ours += 1
            o_num = str(n_ours)
        why = ""
        if o_new:
            if o in ("H", "A"):
                why = reason(detail, a)
            elif o == "D":
                rules, restart = dead_reason(detail, a, b)
                why = "dead: {}{}".format(rules, "  restart@{}".format(
                    video_time(restart[0])) if restart else "")
        mark = "" if t == o else "≠"
        if mark:
            disagree += b - a
        lines.append("  {:<10} {:<9} {:>5.1f}s  {:<3} {:<5} {:<3} {:<5} {:<2} {}".format(
            video_time(a), match_time(a), (b - a) / float(fps),
            t_num, t if t_new else "  .", o_num, o if o_new else "  .", mark, why))
    header = ["{} possessions {}, ours {}.  frames disagreeing {:.1f}% ({:.1f}s)".format(
        truth_name, n_theirs, n_ours, 100.0 * disagree / (stop - start), disagree / float(fps)),
        "'.' = same label continuing from the row above. '≠' = the two disagree on this row.",
        ""]
    return "\n".join(header + lines)


def gap_summary(start, stop, labels, truth):
    """Where this window's disagreement sits: frames by kind of error, and spells missed."""
    fps = GAMES[GAME]["fps"]
    ours = pm.label_codes(labels)[start:stop]
    theirs = truth[start:stop]
    live = {pm.H, pm.A}
    kinds = [
        ("we say dead, they say live", np.isin(theirs, list(live)) & (ours == pm.D)),
        ("we say live, they say dead", (theirs == pm.D) & np.isin(ours, list(live))),
        ("both live, wrong team", np.isin(theirs, list(live)) & np.isin(ours, list(live))
         & (ours != theirs)),
        ("we abstain", ours == pm.X),
    ]
    total = stop - start
    lines = ["  WHERE THE GAP IS"]
    wrong = sum(int(m.sum()) for _n, m in kinds)
    lines.append("    frames agreeing      {:>6.1f}%".format(100.0 * (total - wrong) / total))
    for name, mask in kinds:
        lines.append("    {:<26} {:>5.1f}%  {:>6.1f}s".format(
            name, 100.0 * mask.sum() / total, mask.sum() / float(fps)))

    runs = [(start, stop)]
    full_truth = np.full(len(truth), pm.X, dtype=np.int8)
    full_truth[start:stop] = theirs
    full_ours = np.full(len(truth), pm.X, dtype=np.int8)
    full_ours[start:stop] = ours
    gt_spells = pm.possession_spells(full_truth, runs)
    pred_spells = pm.possession_spells(full_ours, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)
    lines.append("    possessions: theirs {}, ours {}; matched at IoU 0.5: {}".format(
        len(gt_spells), len(pred_spells), int(np.sum(matched))))
    missed = [(a, b, pm.LABELS[k]) for (a, b, k), hit in zip(gt_spells, matched) if not hit]
    if missed:
        short = sum(1 for a, b, _k in missed if (b - a) / fps <= 2.0)
        lines.append("    unmatched possessions of theirs: {} ({} are 2 s or shorter)".format(
            len(missed), short))
        lines.append("      " + "  ".join("{} {:.1f}s {}".format(video_time(a), (b - a) / fps, k)
                                          for a, b, k in missed[:12]))
    return "\n".join(lines)


def _k_frames():
    from exp02_touch_gated import config as config2
    return config2.K_FALLBACK_FRAMES


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--name", help="name for a new pick (default: the next free winN)")
    parser.add_argument("--window", help="re-print a window already in picks.csv")
    parser.add_argument("--windows", action="store_true",
                        help="every window, labels and exp08 side by side, with a gap summary")
    parser.add_argument("--out", help="write to this file as well as printing")
    args = parser.parse_args()

    if args.windows:
        import compare
        sys.path.insert(0, os.path.join(EXPERIMENTS, "exp08_deadball_clf"))
        from exp08_deadball_clf import run as run08
        import whistle as whistle_mod

        labels, detail = run08.build_labels()
        detail["hooked"] = True
        detail["whistles"] = [int(e[0] * 25) for e in whistle_mod.load()["events"]]
        truths = compare.ground_truths()
        chunks = ["POSSESSION SIDE BY SIDE -- every window, exp08 against the labels",
                  "",
                  "public = the organisers' published labels; win1-3 = our hand annotation.",
                  "ours = exp08 (exp06 possession layer, IDSSE dead-ball classifier, whistle).",
                  "Dead rows name what decided them: clf, the whistle that lowered its threshold,",
                  "and in brackets what the old geometry rules would have said.", ""]
        for name in ("public", "win1", "win2", "win3"):
            start, stop = win.frames(name)
            chunks.append("=" * 100)
            chunks.append("{}   frames {}-{}   video {} - {}   match {} - {}".format(
                name.upper(), start, stop, video_time(start), video_time(stop),
                match_time(start), match_time(stop)))
            chunks.append("")
            chunks.append(side_by_side(start, stop, detail, labels, truths[name], name))
            chunks.append("")
            chunks.append(gap_summary(start, stop, labels, truths[name]))
            chunks.append("")
        text = "\n".join(chunks)
        print(text)
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text + "\n")
            print("\nwrote {}".format(os.path.relpath(args.out, ROOT)))
        raise SystemExit(0)

    picks = read_picks()
    if args.window:
        if args.window not in picks:
            raise SystemExit("{} is not in {}. drawn: {}".format(
                args.window, os.path.relpath(PICKS, ROOT), list(picks) or "none"))
        row = picks[args.window]
        name, start, stop, seed = args.window, row["start"], row["stop"], row["seed"]
    else:
        start, stop = pick(args.seed)
        name = args.name or "blind{}".format(len(picks) + 1)
        save_pick(name, start, stop, args.seed)
        print("drew {}: frames {}-{} (video {} - {}), recorded in {}\n".format(
            name, start, stop, video_time(start), video_time(stop),
            os.path.relpath(PICKS, ROOT)))

    from exp04_boundary_timing import run as run04
    labels, detail = run04.build_labels()

    text = render(name, start, stop, seed if args.window else args.seed, detail, labels)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
        print("\nwrote {}".format(os.path.relpath(args.out, ROOT)))
