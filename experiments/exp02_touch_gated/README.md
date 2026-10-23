# Experiment 2 — backdated possession hysteresis (touch gating measured and switched off)

> **Status:** superseded by [exp03](../exp03_keeper_fix/README.md), which runs this exact
> pipeline on keeper-corrected tracking and is worth +0.018 on the official window.

Written for: the team, as the record of what exp02 is, what was tried, and why. Cross-experiment
lessons are in [../LEARNINGS.md](../LEARNINGS.md).

The folder is named for what was planned — a touch-gated state machine from the research note.
What it ships is different, because the plan was measured on 20 annotated minutes and half of it
did not survive. Both the planned version and the shipped one are reproducible from here.

## Result

Two columns, because they measure different conventions. `public` is the organisers' own labels
— the convention the competition is scored against. `pooled` adds our 15 hand-annotated minutes,
which [calibrate.py](../../validation/calibrate.py) measured at **35% fewer possessions** than
the official convention, so its spellF1 rewards merging and cannot settle a threshold.

| | public spellF1 | public poss (17) | pooled spellF1 | pooled macroF1 | full-match poss |
|---|---|---|---|---|---|
| baseline_smooth50 (published floor) | 0.326 | 26 | 0.248 | 0.664 | 594 |
| baseline_smooth125 | 0.424 | 16 | 0.322 | 0.588 | 398 |
| exp01 | 0.424 | 17 | 0.323 | 0.784 | 305 |
| **exp02 (1 s, backdated, no touch)** | **0.485** | **16** | 0.389 | 0.810 | **264** |
| exp03 (exp02 + keeper fix) | 0.503 | 14 | 0.442 | 0.826 | 242 |


The official rate implies roughly **360** possessions across the graded match, so even at 1 s
this still under-segments by about a quarter.

Per window (spellF1 · predicted/true possessions):

| | public | win1 | win2 | win3 |
|---|---|---|---|---|
| exp01 | 0.424 · 17/17 | 0.245 · 22/9 | 0.303 · 21/8 | 0.248 · 18/11 |
| **exp02** | **0.485 · 16/17** | 0.533 · 12/9 | 0.400 · 21/8 | 0.311 · 16/11 |
| exp02 at 2 s | 0.462 · 9/17 | 0.753 · 8/9 | 0.517 · 16/8 | 0.383 · 13/11 |

**How to read the disagreement.** 2 s wins every hand-annotated window and loses the official
one. That is the calibration bias, not noise: the official labels contain possessions of 0.8 s,
1.3 s and 3.5 s that a 2 s evidence threshold cannot produce, and that a human annotator does not
mark either. The threshold is set from the official column; see `K_FALLBACK_FRAMES` in
[config.py](config.py) for the full table.

## What it does

The dead-ball layer is exp01's, imported unchanged (`exp01_geometry_hysteresis.deadball`,
revision 1). Only the possession layer is new:

```
features.py   Savitzky-Golay ball kinematics (speed, acceleration, heading change)
touch.py      touch detection                      (built, measured, switched off)
team.py       the possession state machine
run.py        glue + ablation flags
config.py     every threshold, with source and measurement
```

The shipped state machine, per half:

- stoppage → drop the held team; after the restart, the first frame with a team call sets it
- the opposition has been nearer by ≥ 2 m on **50 frames of evidence** (unseen frames pause the
  count, own-team frames reset it) → possession changes, **labelled from where that evidence
  began**
- a ball moving faster than 8 m/s with no touch is in flight and is not evidence for anyone
- ball untracked → hold; after 4 s untracked, consult the reliable Smart Tagging codes
- possessions shorter than 0.3 s between two of the same team are folded back in

## What changed from exp01, and what each change earned

exp01's possession error on the validation set was dominated by **flip-backs**: 14 possessions of
~2 s between two of the other team's, almost all wholly wrong in the hand labels — a presser
within 2 m for a second, credited with the ball, losing it again ~2 s later. exp02 changed three
things. Measured separately (pooled spellF1 unless stated):

| change | measured effect |
|---|---|
| **backdating** the change to where the evidence began | helps at *every* threshold tried, and more the longer the threshold: macroF1 +0.023 at 2 s, +0.049 at 5 s, +0.073 at 7 s. The most robust result in the experiment. |
| **stricter evidence**, 1 s → 2 s | 0.440 → 0.529; 65 → 46 possessions. Beyond 2 s it over-merges the public window. |
| **touch gating** | *negative* — see below |

Backdating is what makes a stricter threshold affordable. Without it, demanding more evidence
makes every change later; with it, the extra certainty costs no latency. That is the "decouple
*when* possession changed from *how sure* we are" idea the plan set out — it just turned out not
to need touches.

Threshold sweep, fallback only, backdated:

| evidence | public | win1 | win2 | win3 | pooled | possessions |
|---|---|---|---|---|---|---|
| 1 s | 0.461 | 0.552 | 0.441 | 0.357 | 0.440 | 65 |
| **2 s** | 0.431 | 0.776 | 0.583 | 0.432 | **0.529** | **46** |
| 3 s | 0.267 | 0.776 | 0.540 | 0.417 | 0.467 | 39 |
| 5 s | 0.278 | 0.776 | 0.600 | 0.417 | 0.485 | 36 |
| 7 s | 0.278 | 0.776 | 0.600 | 0.364 | 0.472 | 34 |

Everything from 2 s to 7 s scores 0.467–0.529 pooled. A plateau, not a spike.

## Touch gating: built, measured, switched off

The research note's strongest documented result is Link & Hoernig's: requiring a change of
possession to coincide with a kick took proximity precision from 0.68 to 0.86. The plan built
exp02 around it. On this data it does not transfer.

**That verdict is now in doubt, and it is the open question for exp03.** It was reached on the
pooled hand labels, which calibration later showed reward merging — and the touch path's whole
effect is to produce *more* possessions, which is what the official convention wants. Re-judged
at the shipped 1 s threshold:

| | official spellF1 | poss (17) | full-match poss | pooled spellF1 |
|---|---|---|---|---|
| no touch (shipped) | 0.485 | 16 | 264 | 0.422 |
| touch, confirm 0.6 / 1 s | 0.469 | 18 | **353** | 0.284 |
| **touch, confirm 0.8 / 2 s** | **0.509** | 16 | 289 | 0.386 |

On the two signals that are *not* convention-biased — the official window and the possession
count against an implied ~360 — the touch path is now slightly ahead. On the biased pooled score
it is behind. It ships **off** anyway, because 0.509 against 0.485 is far inside the official
window's noise band of ~0.33, and choosing on it would be tuning on 17 possessions. That is a
deliberate refusal to decide, not a verdict. `exp02_touch.csv` is kept so the question stays
live, and a fresh annotated window — labelled in the official convention — should settle it.

The original measurement, pooled spellF1 (possessions), at the two older fallback thresholds:

| confirmation required after the touch | with 2 s fallback (shipped) | with 3 s fallback |
|---|---|---|
| no touch path | **0.529 (46)** | 0.467 (39) |
| 60% of the next 1 s — as planned | 0.356 (75) | 0.356 (75) |
| 80% of the next 1 s | 0.376 (71) | 0.372 (71) |
| 80% of the next 2 s | 0.450 (59) | 0.471 (57) |
| 90% of the next 2 s | 0.451 (57) | 0.469 (55) |

By the pooled hand labels, no confirmation setting let touches beat the fallback alone. Note
even then the pattern that should have been the warning: touches helped on the **public** window
and hurt on all three hand-annotated ones. That was read as the public window being unusual. It
was the convention gap.

**Why, as far as measured.** The detector's touches are real ball events — about 2000 over the
match, one every 3.2 s, the right rate for football — but they are not *control* events. A
pressing opponent's tackle, block or deflection is a touch, and the nearest-player evidence used
to confirm it favours the presser for the next second because the presser is, by definition,
close. So the touch path produces extra short possessions. Whether that is an error or the
point depends entirely on the convention: a human annotator calls them noise, and the official
labels contain possessions of 0.8 s and 1.3 s.

Getting there produced three findings about deriving acceleration from this tracking, all in
[features.py](features.py) and [touch.py](touch.py):

1. **A Savitzky-Golay filter of order 2 cannot denoise acceleration.** It reproduces quadratics
   exactly, and constant acceleration is quadratic in position, so smooth-then-difference passes
   the noise straight through. The acceleration distribution was identical at window 7 and window
   31. Derivatives are read off the polynomial fit instead.
2. **Never leave raw values at the edge of a smoothed run.** The first version did, and put a
   step at both ends of every tracked run: ball speeds of 759 m/s.
3. **Published thresholds measure a ball in space; this one is on a plane.** Median ball
   acceleration is 6.7 m/s², so Link & Hoernig's 4 m/s² is true on 64% of frames. A ball in the
   air projects its height onto the pitch as genuine curvature. The threshold only works as a
   *local maximum* with a refractory window.

And one measured limit: the bend detector agrees with only 18% of the dead-ball layer's restarts
within 3 frames, because the ball's track often *begins* at the restart (the smoothing window is
undefined there) and the thrower stands off the pitch. Restarts are unioned in from the dead-ball
layer instead.

**What would be worth trying with touches.** Not as a trigger, but as *timing*: when the fallback
accepts a change, snap its start to the nearest touch inside the evidence window rather than to
the first frame of evidence. That keeps the fallback's precision and could recover the public
window's short exchanges. Not built.

## Known limitations

- **In-sample threshold.** 2 s was chosen on the 44 possessions it is scored on. A fifth window,
  annotated without looking at exp02's output, is the only way to get an out-of-sample number.
- **Short possessions are merged.** Anything under ~2 s that the opposition does not hold is
  folded away. The public window shows this costs real possessions in rapid exchanges.
- **The dead-ball layer now under-calls slightly**: 27.2% predicted vs 31.4% true on the pooled
  20 minutes, `D` recall 79.1%. One public-window foul is still undetected. This is now the
  larger remaining error source, not the possession layer.
- **Dominant-team exaggeration is gone**, which suggests it was a lag artifact rather than a bias:
  home share per window is now 0.80/0.77, 0.57/0.57, 0.42/0.44 (predicted/true), against exp01's
  0.81/0.77, 0.62/0.57, 0.36/0.44.

## Reproduce

```bash
python experiments/exp02_touch_gated/run.py                  # shipped -> evaluation/results/exp02.csv
python evaluation/validate_submission.py evaluation/results/exp02.csv
python validation/compare.py --bootstrap 1000

# ablations, each to its own CSV
python experiments/exp02_touch_gated/run.py --touch          --out evaluation/results/exp02_touch.csv
python experiments/exp02_touch_gated/run.py --k-fallback 75  --out evaluation/results/exp02_k75.csv
python experiments/exp02_touch_gated/run.py --no-backdate    --out evaluation/results/exp02_nobackdate.csv

# the touch detector on its own
python experiments/exp02_touch_gated/touch.py --report
python experiments/exp02_touch_gated/touch.py --sweep
```

Dependencies are numpy and pandas only — the Savitzky-Golay kernels are computed with
`numpy.linalg.pinv` rather than imported from scipy. A run takes under a second once exp01's
feature cache exists, about two seconds from the raw bundle.
