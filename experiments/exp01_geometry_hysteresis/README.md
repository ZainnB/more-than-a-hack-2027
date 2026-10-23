# Experiment 1 — geometric dead-ball detection + possession hysteresis

Written for: the team, as the record of what exp01 is, what was tried, and why each decision was
made. It is also the raw material for the technical document's "approach" and "limitations"
sections.

> **Status:** superseded by [exp02](../exp02_touch_gated/README.md) and then
> [exp03](../exp03_keeper_fix/README.md), both of which keep this experiment's dead-ball layer
> and replace its possession layer. Revised once after the
> hand-annotated validation set existed — see "Revision 1" below. Cross-experiment lessons are
> collected in [../LEARNINGS.md](../LEARNINGS.md).

## Result on all 20 annotated minutes

The public 5 minutes plus the 15 hand-annotated minutes in [validation/](../../validation/),
44 possessions, scored with `python validation/compare.py --bootstrap 1000`:

| | macroF1 | spellF1 | spellF1 5–95% | possessions (true 48) | `D` prec / rec |
|---|---|---|---|---|---|
| baseline_smooth50 (published floor) | 0.6638 | 0.2484 | 0.194–0.306 | 113 | 91.1% / 42.8% |
| baseline_smooth125 | 0.5876 | 0.3221 | 0.244–0.400 | 65 | 100.0% / 20.2% |
| **exp01 (current)** | **0.7841** | 0.3232 | 0.256–0.390 | 77 | **86.6% / 79.3%** |
| exp02 | 0.8095 | 0.3893 | 0.307–0.468 | 64 | 86.6% / 79.3% |
| exp03 | 0.8261 | 0.4423 | 0.350–0.531 | 56 | 86.6% / 79.3% |

exp01 clearly beats the published floor on macroF1 and on dead-ball precision. On spellF1 it does
**not** separate from `baseline_smooth125` — and note that the pooled column is 15 minutes of
hand labels against 5 official ones, so it inherits their 35% possession under-count. On the
organisers' own window exp01 scores 0.424 with 17 possessions against a true 17.

## Result on the public window alone (as first reported)

Scored on the 5-minute public ground truth (17 possessions). Kept because it is what the decisions
below were originally made against, and because the gap between this table and the one above is
itself the main lesson.

| | baseline_smooth50 | **exp01** |
|---|---|---|
| macroF1 | 0.6207 | **0.8369** |
| spellF1 | 0.3256 | **0.4235** |
| possessions (truth 17) | 26 | **17** |
| mean possession (truth 14.2 s) | 10.8 s | **15.0 s** |
| `D` precision / recall | 0.657 / 0.214 | **0.975 / 0.745** |
| transF1 @ ±25 frames | 0.0816 | **0.3500** |
| possession-share error | 0.048 | **0.008** |
| fragmentation | 1.43 | **1.00** |

**How much to believe this.** The scorer's own bootstrap says that on 17 possessions, spellF1 gaps
below ~0.33 are noise. The +0.098 spellF1 gain is therefore *not* evidence. macroF1 is the
trustworthy number (noise band ~0.01, gain +0.216), plus the structural ones: the possession
count matches, fragmentation is exactly 1.00, and `D` precision is 0.975. The success criteria
were set on those, not on spellF1.

**In hindsight, even the "structural" numbers were luck.** On the validation set the same model
predicted 49 possessions against 28, and `D` precision was 0.706. A 17-possession window cannot
validate a possession count any more than it can rank spellF1.

## What it does

The baseline answers "who has the ball?" once per frame (nearest player, `D` if the ball is
untracked) and smooths with a majority vote. exp01 replaces both halves:

```
features.py   one 2-second pass over the raw ATD  ->  cached per-frame arrays
deadball.py   which frames are stoppages          ->  boolean mask
team.py       who has the ball on the rest        ->  held state with hysteresis
run.py        glue: writes evaluation/results/exp01.csv
config.py     every threshold, with the reason for its value
```

Frames outside the two graded halves are written as `X`. They are never scored, so it is free.

## Dead-ball layer (`deadball.py`)

The baseline's only rule — "tracker lost the ball, therefore `D`" — is 55.8% precise, because the
tracker loses the ball constantly in open play. It detects occlusion, not the referee. Three rules
replace it, in descending order of trust:

| rule | what it detects | public-window precision | recall |
|---|---|---|---|
| `oob` | ball tracked outside the unit square, until a restart is found | 99.4% | 48.3% |
| `slow` | all players' mean speed < 1.3 m/s for ≥ 8 s (the foul detector) | 92.0% | 48.3% |
| `lost` | ball untracked ≥ 3 s *and* a slow pitch | 99.5% | 24.9% |
| **union** | | **97.5%** | **74.5%** |

### Decisions and why

**Out of bounds uses a hard `[0, 1]` boundary, no tolerance.** A band of 0.01 drops precision from
98.1% to 45.8%, because a live ball runs along the touchline constantly.

**A restart is found, not assumed.** Stoppages in the public window run from 5.5 s to 20.4 s, so
any fixed duration is wrong by ten seconds somewhere. A restart is *ball back inside, settled for
~1 s, then struck at ≥ 6 m/s*. Each clause was earned by a failure:
- *Struck* alone ended a stoppage 262 frames early at the goal kick at frame 18389: the ball is
  thrown back in at up to 6.5 m/s, which clears the strike threshold by itself.
- *Settled first* fixes that. Every real restart is taken from a stationary ball.
- If no struck ball is found, the stretch ends where the ball re-entered the pitch. That costs
  recall, not precision, which is the cheaper error.

**Fouls are caught by player speed, not ball geometry.** Both public-window dead stretches with
no out-of-bounds frame are fouls, and in one of them the ball is tracked on 100% of frames and
never leaves the pitch, so no geometric rule can see it. All 22 players walking at once can: mean
speed there is 1.13 m/s against a match median of 1.52. The rule smooths over 50 frames and then
requires 200 frames (8 s) of slowness, because without the duration it fires on every lull.

**A detected restart vetoes the dead-ball rules for the next 2 s.** Players stand still in
formation before a kickoff, so the slow-pitch rule reads the first second of each half as dead.

**Where it is deliberately silent.** One of the two public fouls is still undetected (ball lost on
93% of it, pitch not slow enough). The obvious candidate signal was measured and rejected — see
"Measured dead ends".

## Possession layer (`team.py`)

A possession is a *state*, not a per-frame classification. The state machine holds the current
team and concedes only on evidence:

> concede possession when the opposing team has been nearer to the ball by at least **2.0 m**
> for **25 consecutive frames**

Both numbers come from CHALLENGE.md, not a sweep. The margin exists because "a simple touch of the
ball without control" is explicitly not a change of possession, and a player half a metre nearer
than an opponent has not controlled it. The duration is CHALLENGE.md's own proxy for control —
"a player keeps the ball for a few seconds" — at its conservative end (25 frames = 1 s).

The margin is a real confidence signal, not a knob: on frames where nearest-player gets the team
wrong, the median `|d_home − d_away|` is 1.78 m; where it is right, 4.06 m.

| rule (public window) | team accuracy | possessions (truth 17) |
|---|---|---|
| nearest-player, raw | 0.7784 | 128 |
| margin ≥ 2.0 m | 0.8189 | 66 |
| margin ≥ 2.0 m, k = 25 | **0.8621** | **17** |
| margin ≥ 1.0 m, k = 50 | 0.8981 | 9 (over-merged) |

### Decisions and why

**Ball missing → hold the current team, do not emit `D`.** This is the single biggest difference
from the baseline, which emits `D` on all 43% of ball-less frames. It is nearly free on the
ranking metric: spellF1 never sees an abstention (the scorer forward-fills it), so going quiet
inside a possession costs nothing, while a wrong `D` splits one real possession in two and loses
it outright.

**State resets at every stoppage.** The team that restarts is read from the geometry at the
restart rather than inherited from before the whistle — usually the ball goes to the other team.

**Smart Tagging votes on the team only where the ball is missing, and only from reliable codes.**
Per-code accuracy of "the tagged team is in possession", on GT-alive frames:

| code | accuracy | instances in public window | used? |
|---|---|---|---|
| `BUILD UP` | 100% | **1** | yes |
| `CREATING CHANCES` | 100% | **1** | yes |
| `SET PIECES` | 96.2% | 3 | yes |
| `LONG BALL` | 90.2% | **1** | yes |
| `PROGRESSION` | 72.6% | 5 | no — below the nearest-player call's 85.8% |
| `BALL IN FINAL THIRD` | 53.7% | 4 | no — coin flip |
| `PLAYERS IN THE BOX` | 45.9% | 3 | no — worse than a coin flip |
| `BALL IN THE BOX` | 44.0% | 3 | no — worse than a coin flip |
| `DEFENDING IN *` | — | — | no — the attacking codes mirrored; each sums to exactly 100% with its twin, so no independent information |

Three of the four trusted codes rest on **one** tagged interval each, which is why they are used
only as a tiebreak on ball-less frames and never to override a tracked ball.

## Measured dead ends

Recorded so they are not retried. All on the public window (dead-ball base rate 19.6%).

- **A duel is not a dead ball.** Both teams within 3 m of the ball: P(dead) = **1.6%**, far below
  base rate. Sustained duels: 0.0%. Dilated forward 200 frames: 17.5%, still below base rate.
  Duels are contested *live* play. This killed the "ball and both teams close together ⇒ foul"
  idea.
- **Resolving a contested frame from the future makes it worse.** On duel frames nearest-player
  scores 0.708; giving the ball to whoever holds it 1 s later scores 0.582, 2 s later 0.580.
  CHALLENGE.md says possession "stays with the last team that had it" — the rule must look
  *backward*, which is what the hysteresis does.
- **Smart Tagging set pieces are a weak dead-ball detector.** 59 instances against ~84 estimated
  real stoppages, covering only 2 of 4 public stretches, as fixed 875-frame (35 s) windows rather
  than restart instants, and never firing for fouls. Out-of-bounds found 99 restart events at 99%
  precision. Tagging therefore does not create dead stretches here.
- **The kickoff rule was removed.** It used Smart Tagging "Kick Off" tags to mark the stoppage
  before each restart. Measured: it covered **0.0%** of the 2,934 frames FAQ.md says are all dead
  before the second-half kickoff (oob + slow already cover 79.1%), and on the four post-goal
  celebrations it made coverage *worse* — 97.4% with it, 99.4% without — because its restart veto
  cleared 50 genuinely dead frames before each restart. Scores are identical without it. The
  code stays in `deadball.py` (not in `ALL_RULES`) so the measurement is not lost.

## Bugs found along the way

Worth knowing, because both produced plausible-looking output.

**The ball's speed column is NaN on every frame** in this ATD export (all 113,488 tracked
frames). DATA.md documents a speed in m/s and does not mention it. Ball speed is derived by
differencing positions (`ball_speed_mps` in `features.py`). Player speeds are populated.

**`float("NaN")` parses without raising.** The first version guarded speed parsing with
`try/except ValueError`, which silently averaged NaN into the mean-player-speed feature and
emptied it on 94.6% of frames. It looked plausible and calibrated cleanly against the broken
feature; it was caught only because a threshold justified as "below walking pace" turned out to
select half the match. The missing-value test is now on the text (`field[:1] != "N"`).
`SLOW_PITCH_SPEED` was re-calibrated after the fix.

## Revision 1 — after the hand-annotated validation set

The 15-minute validation set (three windows, 28 possessions, two of them in the second half)
was built specifically because the public window could not rank anything. Its first run against
exp01 contradicted the public window in two places.

### The rule ranking inverted

| rule | public window precision | validation precision |
|---|---|---|
| `oob` — described above as "the primary rule" | 99.4% | **51.9%** |
| `slow` — the one flagged above to "treat with suspicion" | 92.0% | **99.2%** |
| `lost` | 99.5% | 97.6% |

`oob` alone caused **2912 of the 2955** false dead-ball frames on the validation set, and 1698 of
those fell inside Athletic possessions — a third of all their possession time called dead, which
is why `A` recall was 0.557.

The out-of-bounds detection itself was fine; every bad stretch had found a restart. The stretches
were far too long (24 s, 34 s, 44 s) because `find_restart` waits for a *settled-then-struck*
ball. When a throw-in is taken quickly without the ball visibly settling, the search scans past
the real restart and locks onto a later one; several separate out-of-bounds events were
resolving to the same distant restart. The false dead ball landed inside live possessions and
split them.

### The fix: cap how far the restart search may run

A first attempt — "end the stoppage at the earliest evidence play resumed" — was built, measured
and **removed**. It scored well on the hand labels because it matched a bias in them:
`validation/calibrate.py` later showed those labels end dead balls ~0.9 s early. Against the
organisers' own labels it cost 16 points of dead-ball recall (74.5% → 58.1%). The code is gone;
this is the record of it.

What replaced it is duller and better: bound how far past the ball leaving the pitch the restart
may be searched for, 60 s → **15 s** (`config.MAX_DEAD_FRAMES = 375`). It fixes the same
over-extension without asserting *when* a stoppage ends, so it takes no side on the convention.

| | official public | hand-annotated 15 min |
|---|---|---|
| 60 s cap (original) | 97.5% / 74.5% | 70.6% / 89.0% |
| **15 s cap (current)** | **97.5% / 74.5%** | **83.7% / 84.5%** |
| play-resumed (removed) | 96.8% / 58.1% | 90.7% / 83.0% |

Dead-ball precision / recall. The cap costs the official labels nothing and fixes most of what
the hand labels objected to.

<details>
<summary>What the removed rule did, for the record</summary>

`find_play_resumed`: the ball inside the pitch and moving faster than 1 m/s for 1 continuous
second. The stoppage ends at whichever comes first, this or the settled-then-struck restart.
The reasoning is the asymmetry of the two errors on spellF1: a stoppage that ends a little early
costs a few frames of recall, while one that ends late puts dead ball inside a live possession
and costs that whole possession.

| | `D` precision | `D` recall | false `D` frames | of which inside true `A` |
|---|---|---|---|---|
| before (validation set) | 70.6% | 90.2% | 2995 | 1730 |
| **after** | **90.5%** | 84.2% | **702** | **480** |

Pooled over all 20 minutes: macroF1 0.7655 → 0.8044, `D` precision 73.3% → 91.3%, `A` recall
0.655 → 0.756, while spellF1 barely moves (0.3193 → 0.3279).

**It was a regression on the public window** (macroF1 0.8369 → 0.7884, spellF1 0.4235 → 0.4000),
and that was the clue. The goal kick at frame 18389 is exactly the case it gives up — the ball is
carried back to the spot at walking pace for ten seconds, and the rule calls that live. It was
kept at the time because the pooled 20 minutes favoured it; calibration later showed those
minutes were the biased evidence and the public window was right. Removed.

</details>

### The full-match dead-ball worry, resolved

The limitation below warned that exp01 called 47.1% of the match dead against the public
window's 19.6%. The truth is in between: the validation windows are **35.4%** dead (13.9%,
32.7%, 59.5% by window — win3 is late in the match and full of stoppages), so the public window
was simply unusually live. After revision 1, exp01 calls 40.7% of the full match dead, and on the
pooled 20 minutes it now slightly *under*-calls (27.2% predicted vs 31.4% true).

## Known limitations and open risks

As originally written, with what the validation set later showed added in **bold**.


- **The full-match `D` share is unverified and possibly too high.** exp01 labels 47.1% of graded
  frames dead; the public window is 19.6%. Some of that is real (the 2,934 frames before the
  second-half kickoff are all `D`, and dead time is typically a third or more of a match), but
  the `slow` rule alone claims 51,473 frames and nothing has checked it outside the first
  5 minutes. This is the first thing the 15-minute validation set should test.
  **Answered: the validation windows are 35.4% dead and the slow rule was 99.2% precise there. The
  over-calling came from `oob`, not `slow` — see Revision 1.**
- **Away possession looks under-predicted:** 20.1% of graded frames against home's 32.4%. Could be
  the match (it is home-dominated) or a bias — also unresolved until the validation set exists.
  **Answered: both. The match is home-dominated (true home share 0.62 over 20 minutes) and exp01
  matched that overall, but it exaggerated whichever team dominated each window (win3: 0.29
  predicted vs 0.44 true) and swallowed Athletic possession into false dead ball. Revision 1 fixed
  the second; exp02 fixed the first.**
- **One public-window foul is undetected**, and the slow-pitch threshold sits on a plateau
  (78–93% precision across windows 25/50/75 and thresholds 1.1–1.3) rather than a spike, which is
  reassuring but rests on 4 dead stretches.
- **Transitions are late by construction.** `K_FRAMES = 25` concedes possession a full second
  after the evidence appears: 20 predicted transitions against 20 real, only 7 matched at ±1 s.
  This is what exp02 (touch-gated, backdated) targets.
  **On the validation set the bigger problem turned out to be flip-backs, not lateness: 14
  possessions of ~2 s between two of the other team's, almost all wholly wrong — a presser
  within 2 m for a second, credited with the ball, losing it again. 1 s of evidence is too
  little. exp02 raises it to 2 s and backdates.**
- **Every threshold was set from physical reasoning and checked on 17 possessions.** None was
  grid-searched, deliberately: EVALUATION.md shows that tuning one smoothing window on the public
  window picks the wrong value.
- **Human ceiling:** the research note cites κ = 0.72 between two humans for "control", and
  10–15% disagreement in event tagging. Chasing the last few points past that is not meaningful.

## Reproduce

```bash
uv venv && uv pip install -r requirements.txt          # numpy + pandas only
python experiments/exp01_geometry_hysteresis/run.py     # ~2 s; writes evaluation/results/exp01.csv
python evaluation/validate_submission.py evaluation/results/exp01.csv
python evaluation/run_evaluation.py --bootstrap 1000
```

Each layer is separately measurable:

```bash
python experiments/exp01_geometry_hysteresis/features.py --report   # what was parsed
python experiments/exp01_geometry_hysteresis/deadball.py            # per-rule precision/recall
python experiments/exp01_geometry_hysteresis/team.py --sweep        # margin/k grid (a cliff check,
                                                                    # not a leaderboard)
```

The run is deterministic: deleting `cache/` and rebuilding from the data bundle gives a
byte-identical CSV. `cache/` holds derived tracking data and is gitignored.
