# Experiment 6 — tuning on IDSSE (Bundesliga, corrected tracking)

Written for: the team. Plan and rules: [../PLAN.md](../PLAN.md). Lessons: [../LEARNINGS.md](../LEARNINGS.md).

exp04's match-tuned values were removed because the rules forbid tuning on the provided match.
This experiment re-tunes them on external data instead. IDSSE (Bassek et al. 2025, CC BY 4.0,
7 matches, 25 fps) is the candidate: it ships `BallPossession` and `BallStatus` per frame from
DFL's corrected tracking, which is the same kind of source CHALLENGE.md says the ground truth
comes from.

## Day 1 — do IDSSE's flags behave like the public ground truth?

`python experiments/exp06_idsse/day1_check.py J03WMX` (also `WN1`). Two matches so far, four halves.

| | IDSSE (4 halves) | public GT (5 min, 17 poss.) |
|---|---|---|
| dead share | 33–59% | 19.6% |
| possessions per live minute | 7.9–13.8 | 4.2 |
| median possession | 3.0–3.4 s | 7.6 s |
| possessions of 2 s or less | 33–38% | 12% (2 of 17) |
| median stoppage | 14.6–25.4 s | 16.4 s (4 stoppages) |
| dead starts with the ball already out | 67–79% | — |
| delay from ball crossing the line to dead | median 0.12–0.16 s | — |
| flips at which the incoming team's nearest player is within 1.5 m | **96–99%**, median 0.6 m | median 2.0 m or more at every lag, on the provided tracking |
| flips with the ball airborne (Z over 1 m) | 19–36% | not measurable, no height |

### Reading it

- **Dead ball: consistent.** Ball-out stoppages go dead within about 0.15 s of the ball crossing
  the line, which is what exp01's out-of-bounds rule assumes. A quarter to a third of stoppages
  start with the ball in play, which is the whistle case the provided tracking cannot see.
- **Possession flips happen at a touch.** With corrected tracking, the gaining team's nearest
  player is within 1.5 m on 96–99% of flips and never beyond 3 m. The earlier finding that
  flips happen "in the air" on the public window was a property of the provided tracking, not
  of the labels. Only 19–36% of IDSSE flips have the ball above 1 m.
- **The rhythm differs.** IDSSE has roughly twice the possessions per live minute, and a third
  of its possessions last 2 s or less against 2 of 17 on the public window. If the public window
  followed IDSSE's rhythm, 2 or fewer of 17 would be about a 3% event. That is moderate evidence
  that the organisers' labels apply a calmer convention than DFL's, or that those 5 minutes were
  unusual. It cannot be settled from 17 possessions.

### Consequence for tuning

Values that govern possession length (`K_FALLBACK_FRAMES`, the margin, the minimum possession)
will be set by IDSSE's rhythm. If the organisers' convention is calmer, tuning there over-splits.
The public GT can be used to *measure* that gap, never to close it.

## Day 2 — the harness

```bash
python experiments/exp06_idsse/harness.py --matches J03WMX WN1 --seeds 2
```

| file | does |
|---|---|
| `idsse.py` | parses the DFL XML (stdlib + numpy only) |
| `scene.py` | one match as a scene laid out like the provided one: each half trimmed to its live span, a 200-frame X gap between halves, IDSSE flags as the ground truth |
| `adapter.py` | scene -> the feature dict exp01..exp04 consume, key for key |
| `noise.py` | degrades a scene to look like the ATD (below) |
| `harness.py` | scene -> degrade -> exp04 pipeline -> the repo's own scorer; `with_params(...)` sets config values for one run |

The pipeline is exp04's `labels_from_features`, split out of `build_labels` with the output
checked byte-identical. It runs at the a priori values in exp04's `config.py`. Smart Tagging
votes are `None`: IDSSE has no tagging.

### The noise simulator

Every number comes from the provided *tracking* (input data), none from the ground truth:

- **Ball smoothing.** The ATD ball is nearly an exact smooth curve even through kicks (1.3 mm p99
  residual against a 7-frame fit; IDSSE 300 mm). A 51-frame Savitzky-Golay filter is used, chosen
  so exp02's touch detector fires at the ATD's rate: 28.1 touches per visible-ball minute (IDSSE
  gives 32.0 and 24.0 on the two matches measured, mean 28.0).
- **Ball gaps.** 40.7% of frames, in long runs (ATD: 150 gaps, median 268 frames, none under 75),
  and concentrated in stoppages. A two-state chain, calibrated by simulation to 14.7% missing in
  play and 74.5% in stoppages. The live-state gap length (4 s) is the one free choice, forced by
  the fact that long gaps would otherwise bleed from a stoppage into play.
- **Team flips** on 7% of player-frames, in 4 s segments. **Player dropout** 4.5%, 2 s segments.
- **Height** dropped; only x and y are used.

Not simulated, and each could matter: goalkeeper identity switches, more than 11 players per
team, player position jitter, and ball gaps that cluster in crowded moments rather than only in
stoppages.

### First result (a priori config, 2 matches, 2 seeds)

| | spellF1 | macroF1 | dead precision | dead recall | team acc | possessions (true 506) |
|---|---|---|---|---|---|---|
| clean IDSSE | 0.511 | 0.869 | 0.962 | 0.948 | 0.861 | 359 |
| ATD-like | 0.286 | 0.750 | 0.827 | 0.959 | 0.772 | 302 |

- The dead-ball layer, designed on the ATD, transfers: 0.96 / 0.95 on clean data.
- Degradation costs 0.23 spellF1 and 0.12 macroF1, mostly through team accuracy (0.86 to 0.77)
  and dead precision (0.96 to 0.83, the ball-lost rule firing when the ball is hidden).
- Even on clean data the pipeline finds 359 of 506 possessions: it under-segments relative to
  DFL's convention, which is the gap tuning on this data will try to close.
- **Not yet validated:** that this noise is as hard as the real thing. The real match's team
  accuracy is higher than the simulation's, which may mean the simulator is too harsh, or that
  IDSSE is simply the harder match.

## Day 2b — simulator check, then tuning

### Is the simulator as hard as the real thing? (`validity.py`, no labels involved)

The first version failed one input check. IDSSE parks the ball just past the line for a whole
stoppage, while the ATD loses it after a median of 21 frames, so 13.8% of the simulated ball
frames were outside the pitch against 4.2% for real, and the out-of-bounds rule fired on 35% of
frames against 20%. With each exit now hidden after a lognormal keep time matched to the ATD:

| input (should match) | real ATD | IDSSE degraded, 7 matches |
|---|---|---|
| ball missing | 40.7% | 40.5% (37.7–42.7) |
| ball outside the pitch, of seen | 4.2% | 3.0% (1.3–4.2) |
| nearest player, median | 1.91 m | 1.96 m |
| team margin p25 / p50 / p75 | 1.51 / 3.63 / 7.59 m | 1.52 / 3.70 / 7.81 m |
| contested, of seen | 5.3% | 6.7% |
| touches per visible minute | 32.8 | 37.1 |
| restarts per minute | 0.52 | 0.52 |

Remaining gaps: the ball is missing in play 12% of the time against 16%, and slightly fewer
players are tracked (21 against 22).

### Tuning (`tune.py`, results in `results/`)

7 matches x 2 noise seeds per configuration, mean spellF1 against IDSSE's flags.

| config | IDSSE spellF1 |
|---|---|
| tuned on the public window (margin 1.25, k 35, touch off) | 0.309 |
| a priori (exp04 now) | 0.330 |
| **tuned on IDSSE** | **0.355**, held-out match 0.353 |

Stages: possession layer (A), dead-ball layer (B, then B2 past B's grid edges), possession layer
again. The touch path is on in every top configuration. The top of each stage is a flat plateau
(top ten within 0.002), so ties go to the plateau's centre. Final values: `tuned.py`.

### Result on the provided match (a report, never used to choose anything)

`python experiments/exp06_idsse/run.py` -> `evaluation/results/exp06.csv`

| | public | win1 | win2 | win3 | pooled spellF1 | pooled macroF1 | possessions |
|---|---|---|---|---|---|---|---|
| baseline_smooth50 | 0.326 | 0.297 | 0.230 | 0.104 | 0.234 | 0.648 | 594 |
| exp04 tuned on the public window | **0.629** | 0.533 | 0.267 | 0.267 | 0.396 | **0.821** | 243 |
| exp04 a priori | 0.485 | 0.538 | 0.367 | 0.303 | 0.410 | 0.786 | 314 |
| **exp06 tuned on IDSSE** | 0.507 | **0.571** | **0.471** | **0.327** | **0.462** | 0.818 | 307 |

- **Better than a priori on all four windows**, including the organisers' own (+0.022). A
  consistent sign over four independent windows is the evidence. The pooled bootstrap band
  (0.384–0.550) is too wide to rank on alone.
- **The public window is the only place the public-tuned version wins**, which is where it was
  fitted.
- **One clear cost: dead-ball recall on the public window drops to 0.38** (macroF1 0.738 there,
  against 0.818 pooled). IDSSE chose a slow-pitch rule that needs 20 s of walking pace. The public
  window's whistle stoppages are shorter, so they are missed. That is not something to tune back
  on the public window. It is the whistle problem exp05 targets.

## What-if: what would better tracking be worth? (`whatif.py`, `tune.py --scenario N`)

Fixed pipeline (tuned.py), 7 matches x 3 seeds, paired against the baseline:

| what gets better | ball missing | spellF1 | vs baseline (+/- SE) |
|---|---|---|---|
| baseline (ATD-like) | 41% | 0.350 | — |
| ball missing, target 30% | 32% | 0.363 | +0.013 (0.002) |
| ball missing, target 15% | 28% | 0.374 | +0.023 (0.002) |
| team flips halved | 41% | 0.378 | +0.027 (0.004) |
| team flips 0% | 41% | 0.405 | +0.055 (0.005) |
| team flips fixed within 1.5 m of the ball | 41% | 0.408 | +0.057 (0.003) |
| ball not smoothed (sharp kicks) | 41% | 0.361 | +0.011 (0.006) |
| ball 30% + flips fixed within 1.5 m | 32% | 0.423 | +0.073 (0.004) |
| ball 30% + flips 0% + sharp kicks | 33% | 0.448 | +0.098 (0.004) |
| perfect tracking | 0% | 0.450 | +0.099 (0.011) |

Re-tuned for the data quality instead (stages A, B2, A):

| data | re-tuned spellF1 | best settings |
|---|---|---|
| ball 30% + flips fixed within 1.5 m | 0.429 | touch on, margin 1.0, k 75 |
| ball 30% + flips 0% + sharp kicks | 0.455 | touch on, margin 1.0, k 75 |
| perfect tracking | **0.512** | touch **off**, margin 0.5, k 10 (stage A; B2's grid had no better dead-ball cell) |

- Ball-missing targets of 20% and 15% realise as 29% and 28%: every ball past the line for more
  than about a second is hidden, as on the ATD, and that alone is about 27% of frames.
- Only the nearest players decide the team call, so fixing identity within 1.5 m of the ball is
  worth all of fixing it everywhere.
- Perfect tracking caps this decoder at 0.512. Tracking is worth up to +0.16 and a realistic
  upgrade about +0.08; the rest is decoder logic.

The plan built on this is [../TRACKING_PLAN.md](../TRACKING_PLAN.md).

## Data

`data/external/idsse/` (git-ignored), all 7 matches. The loader is `idsse.py`.
