# Experiment 4 — boundary timing, and the threshold interaction it uncovered

Written for: the team. Cross-experiment lessons are in [../LEARNINGS.md](../LEARNINGS.md).

Everything up to the boundary refinement is exp03's, imported rather than copied, so
`run.py --mode none` with exp03's thresholds reproduces exp03 byte for byte.

## Why it was attempted

exp03's errors looked like bad placement rather than invention. Of its spurious possessions,
**60–80% overlap a real possession of the same team** and simply miss the IoU 0.5 bar, so one
badly-placed boundary is charged twice: once as a miss, once as a spurious. And the error is
spread, not lag — matched against the hand labels, exp03's transitions sit at a median of **−4
frames** with a 10th–90th range of **−72 to +33**. Nothing a constant offset could fix.

The hypothesis was that moving boundaries onto the event that caused them would lift both counts
at once, and that tightening a threshold would not, because it would delete the half-credited
overlapping spells along with the junk.

**The second half of that hypothesis was wrong**, and that is where the gain came from.

## Result

| | public spellF1 | public poss (17) | pooled spellF1 | 5–95% | pooled macroF1 | team acc | full-match poss |
|---|---|---|---|---|---|---|---|
| baseline_smooth50 (floor) | 0.326 | 26 | 0.234 | 0.180–0.294 | 0.648 | 0.815 | 594 |
| exp01 | 0.424 | 17 | 0.275 | 0.207–0.347 | 0.778 | 0.819 | 305 |
| exp02 | 0.560 | 18 | 0.326 | 0.251–0.409 | 0.805 | 0.861 | 290 |
| exp03 | 0.542 | 14 | 0.353 | 0.269–0.440 | 0.819 | 0.883 | 266 |
| **exp04** | **0.629** | 11 | **0.396** | 0.305–0.491 | **0.821** | **0.887** | 243 |

Best on every pooled column and on the organisers' own window. Every window improves or holds
against exp03 except win3, which is flat.

## What actually earned it: the thresholds interact

exp03 ran at margin 1.5 m / 25 frames. Those two were chosen at different times, each with the
other held fixed, and **neither sweep could see that they interact**. A looser margin admits
weaker evidence, so it needs more of that evidence before it should be trusted.

Swept jointly, `k` is the parameter that matters:

| k | public spellF1 | pooled spellF1 | spurious / missed | full-match poss |
|---|---|---|---|---|
| 25 (exp03) | 0.555 | 0.347 | 41 / 40 | 287 |
| 30 | 0.621 | 0.393 | 32 / 40 | 258 |
| **35** | **0.629** | **0.396** | **30 / 40** | 247 |
| 40 | 0.629 | 0.389 | 28 / 40 | 232 |
| 50 | 0.600 | 0.389 | 28 / 41 | 208 |
| 60 | 0.462 | 0.343 | 28 / 43 | 199 |

30 to 50 is a broad plateau and both label sets agree on it. The margin is nearly flat across
0.75–1.25 m once `k` is right, so 1.25 was taken as the most conservative of the tied settings.

**Look at the spurious/missed columns.** Going from k=25 to k=35 removes 11 spurious possessions
while `missed` stays at exactly 40. The suppressed spells were genuinely junk, not the
half-credited overlapping ones. My prediction that tightening would cost us matched possessions
was simply wrong, and the ledger says so plainly.

## What barely earned it: the refinement the experiment is named for

Snapping each boundary onto the nearest touch by the incoming team gains **+0.013 on the public
window and nothing pooled**. The search radius is the load-bearing part:

| mode | radius | public | pooled | macroF1 |
|---|---|---|---|---|
| none | — | 0.542 | 0.353 | 0.819 |
| touch | ±8 | **0.555** | 0.353 | 0.819 |
| touch | ±25 | 0.542 | 0.346 | 0.820 |
| touch | ±50 | 0.516 | 0.340 | 0.814 |
| crossing | ±8 | 0.542 | 0.350 | 0.818 |
| crossing | ±50 | 0.529 | 0.343 | 0.816 |

Every setting wider than ±8 frames makes things worse: a boundary allowed to travel two seconds
finds the wrong touch. The "crossing" anchor — the frame the incoming team actually became
nearest, rather than nearer by a full margin — never beat doing nothing.

It ships at ±8 because the sign of the gain is consistent across every `k` in the plateau and it
**cannot change the possession count by construction**, so it carries none of the risk the
threshold does. But it is not what made exp04 better, and the folder name oversells it.

## Limitations

- **This was a grid search.** Roughly 40 cells over 17 public and 60 pooled possessions. The
  defence is that the plateau is broad and smooth, both independent label sets agree on it, and
  the mechanism is interpretable — not that any single cell is trustworthy.
- **The possession count moved further from the target**, 266 → 243 against an implied ~360, and
  the score went *up*. Either the ~360 extrapolation is too high (it comes from the public
  window, which is unusually fast), or spellF1 at 60 possessions is not sensitive to the count.
  Both are possible and neither is settled.
- **`missed` has not moved all experiment**: 40 at k=25, 40 at k=35, 40 at k=50. Roughly two
  thirds of the true possessions are not being found at all at IoU 0.5, and no threshold change
  touches that. This is the wall.
- **win2 and win3 remain weak** at 0.267 each, against 0.629 on the public window.

## Reproduce

```bash
python experiments/exp04_boundary_timing/run.py                       # -> evaluation/results/exp04.csv
python experiments/exp04_boundary_timing/refine.py --report           # the three snap modes
python experiments/exp04_boundary_timing/run.py --mode none --k 25 --margin 1.5   # = exp03
python validation/compare.py --bootstrap 1000
```
