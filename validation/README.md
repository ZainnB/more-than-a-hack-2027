# Validation set

Written for: the team, and as the basis for the technical document's statement on how results
were validated.

15 minutes of the match, possession labelled by hand from the tactical video, to measure the
experiments on more than the organisers' 5 public minutes. RULES.md permits hand-labelled data
for validation; these labels never enter the submission path, and the technical document must say
they were annotated by hand.

## The windows

| | frames | video time | half | true home share | dead share | possessions |
|---|---|---|---|---|---|---|
| public (organisers) | 11185–18685 | 07:27–12:27 | 1st | 0.57 | 19.6% | 17 |
| win1 | 18685–26185 | 12:27–17:27 | 1st | 0.77 | 18.6% | 13 |
| win2 | 143308–150808 | 95:32–100:32 | 2nd | 0.56 | 34.2% | 8 |
| win3 | 178308–185808 | 118:52–123:52 | 2nd | 0.44 | 61.1% | 11 |
| **pooled** | | 20 min | | 0.61 | 33.4% | 48 |

Re-annotated 2026-10-23: win1 and win3 were revised after the calibration showed short
possessions were being merged, lifting the rate from 1.87 to **2.13 possessions/min** against the
official convention's 3.40. The gap is narrower but not closed, so the caveat below still
applies. First-pass labels are kept as `*.firstpass.csv`.

Chosen on Smart Tagging alone, so the selection is independent of every experiment, and spanning
the match's possession balance rather than clustering at 50/50 — the match is home-dominated.
`windows.py --audit` re-runs the selection.

## Files

```
windows.py     the window definitions and the selection that produced them
annotate.py    the stamping tool (needs requirements-validation.txt: opencv)
expand.py      labels/win*.csv boundary stamps  ->  gt/val15.csv, a full 199319-row ground truth
compare.py     every submission x every window, plus the pooled 20 minutes, with bootstrap bands
calibrate.py   a blind hand annotation of the PUBLIC window vs the official labels: how closely
               our labelling convention matches the organisers', boundary by boundary, and the
               human ceiling. The public annotation is never merged into val15 or the pool.
towatch.py     the list of moments to go and watch: official possessions our labels miss, and
               the model's near-misses, with video and match-clock timestamps
blind.py       the reverse of annotate.py: draws an unused 5-minute window at random, records
               the draw in picks.csv, and prints every claim the model makes there BEFORE the
               video is watched. Produces no labels and no score, so nothing can be tuned on it
picks.csv      the windows blind.py has drawn, written at the moment of the draw
blind*.txt     the reports, kept so a watched window can be re-read against what was claimed
compare.csv    the last compare.py run, every number
labels/        the hand annotation -- committed; it is the source of truth
gt/            built from labels/ by expand.py -- gitignored
```

```bash
python validation/expand.py                       # after editing any labels
python validation/compare.py --bootstrap 1000     # the scoreboard to decide on
python evaluation/run_evaluation.py --gt validation/gt/val15.csv --bootstrap 1000   # 15 min only
python validation/blind.py --seed 3               # draw and report a fresh blind window
python validation/blind.py --window blind2        # re-print one already drawn
```

### Why blind windows as well as annotated ones

Annotating first and scoring second has a measured failure mode here: twice a decision was tuned
against hand labels that followed a different convention from the organisers', and twice it had to
be reversed. It also means the labeller has settled on the answer before seeing the model's claim,
so the only question left is who is right, never *why the model did that*.

`blind.py` inverts it. The window is a seeded uniform draw over every frame not already used — not
chosen on balance, set-piece count, or anything the pipeline computes — and the model's claims are
written down with timestamps before anyone watches. The video then adjudicates them one by one.
The output is a list of error *categories*, which can be fixed, rather than a disagreement rate,
which cannot. There is no ground truth for these windows and none is produced, so a blind window
can never be tuned on and never enters `val15.csv` or the pooled score.

## Calibration: what these labels do and do not measure

`public.csv` is a blind re-annotation of the window the organisers published labels for, so the
two conventions can be compared directly (`python validation/calibrate.py`). Measured:

| | result |
|---|---|
| frame agreement | 90.4% — macroF1 0.904, the **human ceiling** |
| possessions | **11 against the official 17** — short ones merged, a 35% under-count |
| dead-ball starts | median offset **0 frames** — the conventions agree |
| dead-ball ends | median **−22 frames (0.9 s early)** on the first pass; hand-corrected since |

The official labels come from corrected tracking, mechanically. Watching the four possessions we
had missed showed why: **an interception or a clearance counts as a possession, with no
requirement that the player control the ball**, and possession survives the linesman's flag until
the whistle. Re-annotating on that rule took the public window from 11 possessions to 17 against
their 17, with zero unmatched boundaries. Consequences:

- **Use `public` for possession counts and thresholds.** It is the only window on the convention
  the competition scores against. This reversed two earlier decisions — see
  [../experiments/LEARNINGS.md](../experiments/LEARNINGS.md) lesson 0.
- **Use the pooled 20 minutes for dead ball, team accuracy and macroF1**, where the conventions
  agree and the extra sample is worth having.
  as `*.asannotated.csv` and `--revert` restores them.
- The possession under-count **cannot** be corrected mechanically; those boundaries were never
  marked. Fixing it means re-annotating with brief possessions marked explicitly.

## How to read it

- **Decide on the pooled row**, then check the per-window rows agree. The two label sets have
  already disagreed outright once (see [../experiments/LEARNINGS.md](../experiments/LEARNINGS.md)).
- **Pooled spellF1 still has a ±0.1 band** at 44 possessions. Differences smaller than that are
  not results. macroF1 is much tighter.
- **The set is no longer fully out-of-sample.** exp02's 2 s threshold was chosen on it. Any future
  claim of an out-of-sample score needs a fresh window, annotated without looking at the current
  model's output — the annotation tool deliberately shows only raw tracking for this reason.
