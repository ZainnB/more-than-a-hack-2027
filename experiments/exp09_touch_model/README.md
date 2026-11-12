# exp09 — a learned touch model, trained on IDSSE

Replaces exp02's hand-set touch detector with a logistic regression that answers "did team T
touch the ball at this approach?". It is trained on degraded IDSSE, where the clean 3D tracking
gives the answer, and decides every possession switch through a `touch_hook` in exp04.

```bash
python experiments/exp09_touch_model/labels.py --report    # true touches vs events and flips
python experiments/exp09_touch_model/touchclf.py --lomo    # touch-level, held out, vs the rule
python experiments/exp09_touch_model/run.py --idsse        # pipeline on IDSSE vs exp08 (~1 min)
python experiments/exp09_touch_model/touchclf.py --train   # final model.json, all 7 matches
python experiments/exp09_touch_model/run.py                # -> evaluation/results/exp09.csv
```

## Pieces

- `events.py` aligns the IDSSE event log to scene frames and recovers the person behind each column.
- `labels.py` finds true touches on the clean ball. A touch is a velocity change that free flight
  cannot make (gravity is removed when airborne, so a bounce does not count but a header does),
  with a player within 1.5 m. Every team in reach is credited, and the touch is flagged
  `contested` when both teams are.
- `touchfeat.py` builds candidates (each team's closest approaches, within 4 m) and 40 features
  from ATD-like signals.
- `touchclf.py` handles the dataset, leave-one-match-out evaluation and training (deadclf's Newton
  solver).
- `hook.py` turns scored candidates into a touch_hook. `run.py` runs the IDSSE grid and writes the
  submission.

## What was measured

**Labels (clean IDSSE, 7 matches).** 94–96% of BallPossession flips have a touch by the new team
within ±0.5 s. 90–94% of events have a touch by the same team within ±2 s; the event log is
hand-timed, with the player a median 3.7 m from the ball at its stamp. Crediting only the nearest
player covered just 84–86% of flips, because duels are ambiguous. 22–28% of touches are contested.

**Candidates.** Smoothing the ball over 2 s shifts the closest approach up to about 1 s from the
contact, so labels match at ±1 s. At that window, 96% of visible team-change touches have a
candidate. About half of all missed touches happen while the ball is unseen; the fallback covers those.

**Touch level, held out.** On challenger approaches (the team not in possession), at the rule's
precision of 0.845, the model finds 0.72 of real team-change touches against the rule's 0.49.
AUC is 0.84 overall and 0.77 when both teams are within 1.5 m.

**Pipeline on degraded IDSSE (3 seeds, every piece held out).** Leave-one-match-out picked
tau 0.85, fallback 250 frames, minimum possession 1.0 s in all 7 folds, an interior optimum after
extending the first grid. Held-out spellF1 0.418 → 0.487 (+0.068), 7/7 matches, each gain 3–9×
seed sd. macroF1 0.841 → 0.858, team accuracy 0.834 → 0.859.

**Label-free check on the real ATD.** 33 claims per live minute against 37.5 on IDSSE; mean p
0.71 against 0.73; no feature median shifts more than 0.23 sd. The simulator is faithful for these
features.

## The open question: the match windows disagree (report only, nothing tuned on them)

| | public (official) | win1 | win2 | win3 | pooled | macroF1 | possessions |
|---|---|---|---|---|---|---|---|
| exp08 | 0.503 | 0.637 | 0.463 | 0.375 | 0.483 | 0.837 | 325 |
| exp09 | 0.585 | 0.400 | 0.250 | 0.382 | 0.405 | 0.812 | 418 |

exp09 finds many more real team changes within ±1 s: 17/36 across the windows against 9/36. It
also adds 35 switches with no true change nearby, mostly 1–3 s flip-and-backs. Whether those are
real one-touch interceptions (the official convention counts them, lesson 0b) or phantom touches
is a convention question. IDSSE cannot settle it (lesson 0e: IDSSE has about twice the official
possession rate), and the hand labels are the ones we already suspect of bias. The way to settle
it is to watch the listed switches, not to tune on the windows.

## Removing the quick switches (2026-11-10)

Three mechanisms, each with tau re-chosen on IDSSE (`run.py --idsse --grid confirm`):

| mechanism (best on IDSSE) | IDSSE spellF1 | vs exp09 above |
|---|---|---|
| second-touch check, 1 s window (`hook.second_touch`) | 0.498 | +0.011, 6/7 matches: **adopted** |
| second-touch, 2 / 3 / 5 s windows | 0.484 / 0.475 / 0.459 | -0.002 / -0.012 / -0.028 |
| look-ahead confirmation (exp02's `_confirmed`) | 0.466 | -0.021 |
| minimum possession 2 s | 0.458 | -0.029 |

IDSSE prefers short possessions (its convention), so it stops at a 1 s window. On the match the
1 s check barely changes anything (96 possessions against 94): touch timing on the smoothed ball
is uncertain by about 1 s, so a 1 s window rarely sees the recovery touch. Stricter variants, as
a report only:

| | public | win1 | win2 | win3 | pooled | macroF1 | poss (windows) |
|---|---|---|---|---|---|---|---|
| exp08 | 0.503 | 0.637 | 0.463 | 0.375 | 0.483 | 0.837 | 60/60 |
| exp09 (1 s, adopted on IDSSE) | 0.533 | 0.340 | 0.242 | 0.400 | 0.382 | 0.810 | 96/60 |
| 3 s window | 0.541 | 0.414 | 0.250 | 0.469 | 0.416 | 0.827 | 67/60 |
| 3 s window + 3 s min possession | 0.613 | 0.600 | 0.250 | 0.497 | 0.473 | 0.830 | 55/60 |

Choosing among those rows by these scores would be selection on the match (lesson 0f). A
stricter setting has to be justified by the convention (how short an official possession can be),
not by the table.
