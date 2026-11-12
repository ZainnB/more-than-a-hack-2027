# Experiment 10 — a Viterbi decoder over H / A / D (not adopted)

Written for: the team. Builds on exp08 (dead-ball classifier) and exp09 (touch model); tuned with
the IDSSE harness from exp06. Lessons: [../LEARNINGS.md](../LEARNINGS.md).

```bash
python experiments/exp10_viterbi/decode.py          # unit test of the decoder
python experiments/exp10_viterbi/idsse_eval.py      # choose values on IDSSE (~10 min first run)
python experiments/exp10_viterbi/run.py             # -> evaluation/results/exp10_viterbi.csv
python experiments/exp10_viterbi/run.py --no-touch  # ablations: --no-touch, --no-tagging, --a-priori
python experiments/exp10_viterbi/breakdown.py       # side-by-side + short-possession diagnosis
```

## Why

On the four annotated windows, exp08 predicted about the right number of possessions (62 against
60), but it matched 0% of true possessions shorter than 3 s and 20% of 3–8 s ones. Every
experiment so far switches teams locally: the challenger has to lead by a margin for k frames. A
strict setting merges real interceptions, and a loose one makes flicker. A global decoder can in
principle keep a short possession when its evidence is strong and drop it when it is weak. A
switch pays off when the evidence over its frames exceeds two switch costs (`config.py`
explains this).

## What it is

| file | role |
|---|---|
| `emissions.py` | Per-frame log-scores for H, A and D, built only from existing evidence:<br>• exp08 classifier log-odds (smoothed ±12 frames, as exp08 does)<br>• the signed nearest-player margin through a logistic, clipped at 2 m<br>• neutral (0) when the ball is unseen, in flight, or more than 5 m from every player<br>• exp09 touch candidates as a bonus for the touching team<br>• Smart Tagging transitions, which lower the switch cost into the team gaining the ball |
| `decode.py` | Log-space Viterbi with back-pointers, O(n×9), run on each half separately. Includes a unit test |
| `config.py` | Every parameter with its reason, the a priori fallback, and the IDSSE choice |
| `idsse_eval.py` | Grid with leave-one-match-out on 7 degraded IDSSE matches × 3 seeds, paired against exp08 on identical inputs. Dead-ball and touch models are held out too |
| `run.py` | Builds the submission. It is deterministic: rerunning gives a byte-identical CSV |
| `breakdown.py` | Report only: per-window table and per-possession IoU by length |

## Values and how they were chosen (IDSSE only)

The first grid's best cell sat on three edges, so the grid was extended, as in exp06 and exp09.
The extended surface is a flat ridge: the top 8 cells score 0.423–0.426.

Chosen: **c_switch 8, ha_scale 0.05, c_dead 60, touch_gain 8**. Smart Tagging relief (3 s window,
half the switch cost) is set a priori, because IDSSE has no tagging.

| IDSSE, 7 matches × 3 seeds | spellF1 | macroF1 | possessions (true 437) |
|---|---|---|---|
| exp08 | 0.418 | 0.841 | 277 |
| **exp10 (chosen)** | 0.426 | 0.840 | 288 |
| exp10, best without touch | 0.387 | 0.826 | 321 |
| exp10 a priori | 0.411 | 0.837 | 298 |

Held out (choose on 6, score the 7th), the gain over exp08 is **+0.004 spellF1, 4/7 matches**,
with a seed sd of about 0.01. That is inside seed noise, so under PLAN.md's rule **exp08 stays the
incumbent**.

## Result on the provided match (report only, nothing chosen from it)

| | public | win1 | win2 | win3 | pooled | 5–95% | macroF1 | D prec / rec | full-match poss |
|---|---|---|---|---|---|---|---|---|---|
| exp08 (`--no-whistle`) | **0.487** 15/17 | **0.622** 14/13 | **0.473** 18/15 | **0.362** 16/16 | **0.479** 62/60 | 0.400–0.567 | **0.822** | 88.9 / 79.4 | 331 |
| exp10 | 0.456 19/17 | 0.427 17/13 | 0.373 15/15 | 0.257 26/16 | 0.368 76/60 | 0.294–0.447 | 0.799 | 89.8 / 76.8 | 348 |
| exp10 `--no-touch` | 0.352 8/17 | 0.505 6/13 | 0.435 8/15 | 0.304 9/16 | 0.387 30/60 | 0.280–0.510 | 0.822 | 89.7 / 77.1 | 151 |
| exp10 `--no-tagging` | identical to exp10 in all four windows | | | | | | | | 348 |
| exp10 `--a-priori` | 0.411 18/17 | 0.426 18/13 | 0.388 17/15 | 0.322 20/16 | 0.382 72/60 | 0.312–0.467 | 0.819 | 90.1 / 79.3 | 350 |

Per true possession, pooled (share with a same-team prediction at IoU ≥ 0.7):

| | <3 s (15) | 3–8 s (20) | 8–20 s (7) | >20 s (18) | start err med / abs | end err med / abs |
|---|---|---|---|---|---|---|
| exp08 | 0% | 20% | 29% | 56% | −0.6 / 1.3 s | +0.0 / 1.4 s |
| exp10 | 0% | 10% | 29% | 56% | −1.1 / 2.3 s | −1.1 / 2.1 s |

**The decoder did not fix short possessions.** It predicts more possessions (76 against 60), but
they land in the wrong places. Boundaries are about 1 s early and twice as spread. The mean best
IoU for <3 s possessions fell from 0.24 to 0.15.

## Limitations and what this suggests

- **The emissions, not the decoder, are the bottleneck.** Without touches, proximity alone needs
  so much evidence per switch that the chosen costs merge play into 151 possessions. With
  touches, the touch bonus drives most switches. Its ±0.5 s spread and the logistic margin both
  move evidence earlier than the true contact, which explains the early boundaries. A global
  decoder only helps if the evidence for a 1 s interception is stronger than that of a 1 s press,
  and these features do not separate the two.
- **The `--no-touch` ablation keeps the touch-on costs.** IDSSE's own best no-touch configuration
  is different (c_switch 2, ha_scale 0.1). The row isolates removing the evidence, not a retuned
  no-touch decoder.
- **IDSSE's convention (lesson 0e)** has more and shorter possessions than the organisers'
  labels. A tie on IDSSE followed by a loss on the match is the same pattern as exp08's re-tune.
- **The tagging relief touches 604 frames in the windows but changes none.** It changes 162
  frames across the full match, so its effect cannot be measured here.
- **First-order HMMs have geometric durations.** A semi-Markov decoder with an explicit
  possession-length prior, built from the official convention's lengths, is the obvious next
  step. So is placing touch evidence at its candidate frame instead of spreading it.
