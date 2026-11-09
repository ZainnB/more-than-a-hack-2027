# Experiment 8 — realistic team flips, and a dead-ball classifier trained on IDSSE

Written for: the team. Plan: [../TRACKING_PLAN.md](../TRACKING_PLAN.md) step 2a. Builds on the
IDSSE harness in [../exp06_idsse/](../exp06_idsse/).

## Part 1 — re-tune at a realistic flip rate (`retune.py`)

exp06 was tuned with 7% of player-frames carrying the wrong team. exp07 found near-ball labels
wrong about 1% of the time on clearly visible players, so the simulator now uses **1.5%**. Same
stages (A, B2, A), 7 matches × 2 seeds.

| | IDSSE spellF1 | held-out |
|---|---|---|
| exp06 values, 7% flips | 0.355 | 0.353 |
| re-tuned, 1.5% flips | 0.405 | 0.398 |

The dead-ball values did not move (1.5 m/s, 500, 250, 125). The possession layer produced a tie
between two very different regimes, and the tie held across flip rates of 1–7% (mean 0.388
against 0.387):

| | touch | margin | k | IDSSE | possessions (true 437) |
|---|---|---|---|---|---|
| chosen by the tuner's macroF1 tie-break | off | 0.5 | 15 | 0.405 | 358 |
| the other | on | 0.75 | 50 | 0.405 | 290 |

## Part 2 — dead-ball classifier (`deadfeat.py`, `deadclf.py`, `model.json`)

- **Features:** 23 per-frame features, tracking only, computed identically on IDSSE and the ATD.
  Player speed now, over the last 3 and 8 s and the next 3 and 8 s, the slowdown, the share walking
  or sprinting, team spread, how often the ball is seen, ball outside the pitch, ball speed,
  distance to the nearest line, and the nearest player's distance to the ball. Player speed is
  differenced from positions on both datasets.
- **Model:** L2 logistic regression fitted by Newton's method in numpy, so the submission needs no
  new dependency. Labels are IDSSE's BallStatus; tracking is degraded by exp06's simulator at 1.5%
  flips.
- **Decode:** smoothing ±12 frames, threshold 0.5, stoppages of at least 100 frames, live gaps of
  at least 25. Chosen on IDSSE held-out probabilities.

Leave-one-match-out on IDSSE, dead-ball F1:

| | F1 | precision | recall |
|---|---|---|---|
| **classifier** | **0.953** | 0.95 | 0.95 |
| tuned rules | 0.896 | 0.92 | 0.87 |

It wins on all 7 held-out matches (worst case 0.941 against 0.832). Largest weights: slow players
now, slow ball, ball unseen, slow players over the next 8 s, ball out in the last 5 s.

Downstream on IDSSE at 1.5% flips, the possession layer re-tuned around it:

| | spellF1 | held-out | macroF1 |
|---|---|---|---|
| tuned rules | 0.405 | 0.398 | 0.812 |
| **classifier** | **0.422** | **0.416** | **0.845** |

## Result on the provided match (report only)

| | public | win1 | win2 | win3 | pooled | your 3 windows | pooled macroF1 | possessions |
|---|---|---|---|---|---|---|---|---|
| exp06 | **0.507** | 0.571 | 0.471 | 0.327 | 0.462 | 0.451 | 0.818 | 307 |
| exp08: re-tuned rules | 0.390 | 0.371 | 0.283 | 0.275 | 0.325 | 0.305 | 0.813 | 379 |
| exp08: re-tuned + classifier | 0.410 | 0.415 | 0.263 | 0.287 | 0.337 | 0.312 | 0.823 | 412 |
| **exp08b: exp06 possession + classifier** | 0.487 | **0.622** | **0.473** | **0.362** | **0.479** | **0.478** | 0.822 | 331 |

- **The classifier helps on the match** with the possession values held fixed: pooled spellF1
  +0.012, macroF1 +0.010, public dead recall 0.38 → 0.44. That agrees with IDSSE.
- **The 1.5% re-tune hurts on the match.** It chose touch off with k 15, which splits the match into
  far more possessions (307 → 412). This is the convention gap from exp06's day 1: IDSSE's
  possession flag changes hands about twice as often per live minute as the organisers' labels,
  and with less noise in the simulator the tuner followed IDSSE's faster rhythm.

## The exp08b caveat — read before using it

exp08b keeps exp06's possession values and changes only the dead-ball layer. On IDSSE those
possession values score 0.420 with the classifier against 0.422 for the re-tuned ones, a difference
inside seed noise. So IDSSE does not prefer the change. The rule behind exp08b is:

> **Change a value only if IDSSE shows an improvement larger than seed noise.**

By that rule the possession values stay and the classifier goes in (its IDSSE gain, +0.017
spellF1 and +0.033 macroF1, is far beyond noise).

**That rule was stated after seeing the provided match's scores.** It is a sound rule and could
have been written down in advance, but it was not, and the two candidate configurations were
compared on the match first. Whether to adopt exp08b is the team's decision. If adopted, the rule
should go into PLAN.md as the standing protocol, so every later change is judged by it in advance,
and the technical document should say when it was introduced.

The whistle anchor (`--whistle`) changes little here: pooled 0.334 against 0.337 on the re-tuned
version. It stays off; IDSSE has no audio to validate it.

## Reproduce

```bash
python experiments/exp08_deadball_clf/retune.py                    # part 1
python experiments/exp08_deadball_clf/deadclf.py                   # part 2, writes model.json
python experiments/exp08_deadball_clf/retune.py --classifier       # possession layer around it
python experiments/exp08_deadball_clf/run.py                       # -> exp08.csv
python experiments/exp08_deadball_clf/run.py --possession exp06 --out evaluation/results/exp08b.csv
```
