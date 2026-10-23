# Experiment 3 — correcting the tracker's keeper identity switches

Written for: the team. Cross-experiment lessons are in [../LEARNINGS.md](../LEARNINGS.md).

The only change from exp02 is the tracking it is handed. Every rule, threshold and layer is
exp02's, imported rather than copied, so the difference in score is attributable to the
correction and to nothing else — `run.py --no-correction` reproduces exp02 byte for byte.

## Where it came from

Not from a metric. While annotating, Zain noticed the home keeper drawn in the away team's
colour, including once where the ball was passed back to him — a naive reading scores that as a
change of possession, and he correctly did not label it as one. Five or six instances across the
three windows.

## What was actually wrong

**Not the metadata.** It assigns each of the 46 track slots a team for the whole match, and
averaged over the match both keepers are right:

| slot | team | 1st-half mean x | 2nd-half mean x | in a penalty area |
|---|---|---|---|---|
| 23 | HOME | 0.102 | 0.887 | 82% |
| 45 | AWAY | 0.847 | 0.176 | 88% |

Home defends the left goal in the first half, away the right. Both match.

**The fault is transient.** Slot 45 is re-used for other players constantly — it appears on the
wrong side of halfway on **9.6%** of its first-half sightings and **12.6%** of its second-half
ones. Slot 23 is far more stable at 0.4% and 1.8%. When that happens near the ball, a home
player is wearing the away label.

It matters on **1680 frames** of the graded match (1.06%) where a wrong-side keeper slot is
within 5 m of the ball. Small in frames, large in consequence: those frames sit next to the ball,
which is exactly where possession is decided.

## The fix

Slot identity is unreliable frame by frame but perfectly reliable in aggregate. A slot that
spends 80%+ of the match inside a penalty area is a keeper slot, whatever it happens to be
tracking right now. So:

> a keeper slot found on the wrong side of halfway is not its keeper, and whoever it is tracking
> there plays for the team defending that end

Which team defends which end is read off the kickoff frame and is a fact about the match, not a
parameter. The slot classification is unambiguous — the two keepers sit at 82% and 88%, no
outfield slot exceeds 50% — so any threshold from 0.5 to 0.8 picks the same two slots.

**A purely positional rule was tried first and rejected.** "The deepest player in the box is that
end's keeper" fired on 6424 frames but reached only **73** of the 1680 that matter: a keeper
10.7 m off his line is outside any depth threshold tight enough to be safe at corners, and an
away player standing near the home goal is perfectly ordinary. Position alone cannot say a label
is wrong. Position *combined with a slot that is supposed to be a keeper* can. The slot-based
rule reaches all 1680.

Effect on the features: `ball_x`, `n_tracked` and `mean_speed` are byte-identical (all are
team-agnostic); `d_home` differs on 1796 frames, `d_away` on 2656, and **the nearest-player team
flips on 1508 frames** — 0.76% of the match, and the frames where possession can change hands
differently.

## Result

`public` is the organisers' own labels — the convention the competition scores against. `pooled`
adds the 15 hand-annotated minutes, which under-count possessions (see LEARNINGS lesson 0), so
read it for macroF1 and dead ball rather than for possession counts.

| | public spellF1 | public poss (17) | pooled spellF1 | pooled macroF1 | team acc | full-match poss |
|---|---|---|---|---|---|---|
| baseline_smooth50 (floor) | 0.326 | 26 | 0.248 | 0.664 | 0.825 | 594 |
| baseline_smooth125 | 0.424 | 16 | 0.322 | 0.588 | 0.862 | 398 |
| exp01 | 0.424 | 17 | 0.323 | 0.784 | 0.839 | 305 |
| exp02 | 0.485 | 16 | 0.389 | 0.810 | 0.880 | 264 |
| **exp03** | **0.503** | 14 | **0.442** | **0.826** | **0.906** | 242 |

The correction alone is worth **+0.018 on the official window, +0.053 pooled, +0.017 macroF1**
and +0.026 team accuracy. Every window improves or holds; none regresses.

**The touch path was re-tested on corrected tracking and still does not help** — 0.485 official
and 0.397 pooled with it on, against 0.503 and 0.442 without. Some of what touch gating appeared
to fix earlier was keeper noise, and with the noise gone it has less to do.

## Limitations

- **The possession count went the wrong way.** 242 against the ~360 the official convention
  implies. The correction removes spurious changes, which is right, but exp03 under-segments more
  than exp02 did. Under-segmentation is now the largest single error in the pipeline.
- **One match, two keeper slots.** The rule generalises in principle — it needs only a keeper
  slot and which end a team defends — but it has only been tested here.
- **A keeper genuinely upfield** (a corner in the 95th minute) is mislabelled. Rare, brief, and
  in those frames the ball is at the other end.
- **It does not fix outfield identity switches**, which are certainly present: a team is tracked
  with more than 11 players on 42.5% of win3's frames. Nothing yet addresses that.

## Reproduce

```bash
python experiments/exp03_keeper_fix/keeper.py --report     # the diagnosis
python experiments/exp03_keeper_fix/features.py --report   # what the correction changes
python experiments/exp03_keeper_fix/run.py                 # -> evaluation/results/exp03.csv
python experiments/exp03_keeper_fix/run.py --no-correction # ablation: reproduces exp02 exactly
python validation/compare.py --bootstrap 1000
```
