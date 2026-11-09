# Learnings

Written for: the team. A running log of what has been learned across experiments — the things
worth knowing before starting the next one. Per-experiment detail is in each folder's README;
this file holds only what generalises. Newest section first.

## Scoreboard so far

Two columns, because they measure different conventions and only one decides the competition.
`public` is the organisers' own 5 minutes; `pooled` adds our 15 hand-annotated ones (re-annotated
2026-10-23, 32 possessions).

Scored against the **third-pass** hand labels (2026-11-03), which finally match the organisers'
convention — see lesson 0b. All numbers below fell when the labels improved; that is the labels
getting harder and more correct, not the models getting worse.

| | public spellF1 | public poss (true 17) | pooled spellF1 | 5–95% | pooled macroF1 | full-match poss |
|---|---|---|---|---|---|---|
| baseline_smooth50 (published floor) | 0.326 | 26 | 0.234 | 0.180–0.294 | 0.648 | 594 |
| baseline_smooth125 | 0.424 | 16 | 0.250 | 0.179–0.333 | 0.572 | 398 |
| exp01 | 0.424 | 17 | 0.275 | 0.207–0.347 | 0.778 | 305 |
| exp02 | 0.560 | 18 | 0.326 | 0.251–0.409 | 0.805 | 290 |
| exp03 | 0.542 | 14 | 0.353 | 0.269–0.440 | 0.819 | 266 |
| exp04 as first shipped, tuned on the public window | 0.629 | 11 | 0.396 | 0.305–0.491 | 0.821 | 243 |
| exp04 a priori (match-tuned values removed) | 0.485 | 16 | 0.410 | 0.328–0.498 | 0.786 | 314 |
| exp06: tuned on IDSSE | 0.507 | 13 | 0.462 | 0.384–0.550 | 0.818 | 307 |
| **exp08: + dead-ball classifier + whistle** | 0.503 | 14 | **0.483** | 0.403–0.573 | **0.837** | 325 |

From exp06 on, every value is chosen on external data (IDSSE) or a priori, never by scoring the
match; the match is scored only to report. Two exceptions are flagged where they happen: exp08's
"keep the incumbent" rule and its 36 dB whistle cut (lesson 0f).

**The human ceiling is 0.894 spellF1 / 0.956 macroF1** (our own blind labels scored against the
organisers' on the public window). The bands still overlap at 60 possessions, so the ordering is
provisional; the evidence that exp06 and exp08 are real gains is that they hold on IDSSE matches
they never saw, not these columns.

---

## 0g. Simulate the data you have, then check the simulation against it

exp06 tunes on 7 Bundesliga matches degraded to look like our tracking. The first version of the
simulator kept the ball tracked outside the pitch for whole stoppages, as IDSSE does; our tracking
loses it about a second after it crosses the line. The out-of-bounds rule then fired on 35% of
simulated frames against 20% on the real match, which would have biased every dead-ball value tuned
there. It was caught by `validity.py`, which compares label-free statistics of the two before any
tuning. Check the simulator first, every time it changes.

## 0f. Rules written after seeing the scores are still selection

Twice in exp08 a choice was made after the match had been scored: keeping exp06's possession values
(IDSSE could not tell them from the re-tune, 0.420 against 0.422, but the match could), and raising
the whistle cut from 35 to 36 dB to remove one false whistle 0.24 dB below a real one. Both are
recorded as such. The rule now in PLAN.md — change a value only if IDSSE improves beyond seed noise
— is written down in advance so the next choice does not depend on the match.

## 0e. The external labels have their own convention

IDSSE's possession flag changes hands about twice as often per live minute as the organisers'
labels. With heavy simulated noise this did not matter; when exp08 lowered the team-flip rate
from 7% to 1.5%, the tuner followed IDSSE's faster rhythm, chose touch-off with k 15, and split our
match into 412 possessions against 307 (pooled spellF1 0.462 to 0.337). Tuning on external data
transfers the external convention along with everything else.

## 0d. A measured error rate can be two errors added together

The "93% team attribution" that set exp06's 7% flip rate was nearest-player accuracy, which mixes a
wrong team label with the nearest player not being the possessor. exp07 checked near-ball labels
against shirt colour and found the label wrong only about 1% of the time on clearly visible
players; the rest sits in duels, where players overlap and colour cannot help. Fixing team
identity was therefore worth much less than the simulator promised, and colour changed the score
by nothing.

## 0c. A learned stoppage model beats hand-written rules, and the whistle adds what tracking can't

exp08's dead-ball classifier (23 tracking features, logistic regression, trained on IDSSE
BallStatus) scores 0.953 dead F1 on held-out IDSSE matches against 0.896 for the tuned rules, and
wins on all seven. On the match it finds 20 of 22 stoppages against 18, with 2 false against 7.
The referee's whistle, found in the tactical audio, then catches the foul stoppages the players'
movement shows too late: the public window's 10:35 stoppage went from missed to starting 0.9 s
early. The audio was hard: a continuous tonal background, a steady 3.3 kHz tone, and a whistling
crowd each fooled a version of the detector (`exp08_deadball_clf/WHISTLE_HISTORY.md`).

---

## 0a. Thresholds chosen one at a time hide their interaction

exp01 set the evidence margin at 2 m and never swept it again. exp02, exp03 and the margin fix
all swept the *frame* threshold with the margin held fixed, or the margin with frames held fixed.
Each sweep found a clean plateau and each was locally right.

Swept **jointly** in exp04 they turned out to interact in an obvious-in-hindsight way: a looser
margin admits weaker evidence, so it needs more frames of it. Moving from (1.5 m, 25 frames) to
(1.25 m, 35 frames) took pooled spellF1 from 0.353 to 0.396 and the official window from 0.542 to
0.629 — a bigger gain than any single-axis sweep had produced, from parameters that had both
already been "tuned".

**When two parameters trade off against each other, a one-at-a-time sweep is not a sweep.**

---

## 0b. The convention gap closed by watching four clips

Lesson 0 found that our labels had 35% fewer possessions than the organisers'. What it could not
say was *why*. Watching the four specific possessions we had missed answered it in one sitting:

- **An interception counts as a possession, with no requirement of control.** Three of the four
  were a defender getting a touch on the ball and immediately losing it. The official labels
  start a new possession there; a human watching does not.
- **A clearance counts the same way.**
- **Possession survives the linesman's flag.** It ends at the whistle or the referee's raised
  hand, not when the flag goes up.

Note this **contradicts a plain reading of CHALLENGE.md**, which says "a simple touch of the ball
without control" is not a change of possession. The ground truth is the authority — CHALLENGE.md
says so itself — and the ground truth counts them. Where prose and labels disagree, go and watch
the labels.

Re-annotating on that rule took the hand labels from 11 possessions to **17 against the
organisers' 17**, with zero unmatched boundaries in either direction, frame agreement 90.4% →
**95.6%**, and spellF1 as a submission 0.714 → **0.894**. Possession-change timing came out at a
median of −1.5 frames, which is exact.

It also explained a tuning result we had measured but not understood: a looser evidence margin
(2.0 m → 1.5 m) helps precisely because interceptions and clearances are brief contacts that a
2 m margin suppresses.

**Method worth repeating: when two label sets disagree, do not reason about the definition — get
the list of specific disagreements and watch them.** `validation/towatch.py` generates it.

---

## -1. The best bug report came from a human watching the video

exp03 exists because Zain noticed, while annotating, that the home keeper was sometimes drawn in
the away team's colour. No metric surfaced it. It turned out to affect 1680 frames right next to
the ball, and fixing it was worth more than any threshold change made so far (+0.018 official,
+0.053 pooled, +0.017 macroF1, +0.026 team accuracy).

The specific lesson about tracking data: **slot identity is unreliable frame by frame but
reliable in aggregate.** The metadata's team assignment is correct on average for both keepers;
slot 45 is simply re-used for other players on 10-13% of its sightings. So do not trust a slot's
label on a given frame, but *do* trust what a slot does over a match — that is what identifies it
as a keeper slot in the first place, and what makes the correction possible.

Corollary worth checking next: a team is tracked with **more than 11 players on 42.5% of win3's
frames**, so outfield identity switches are certainly present too, and nothing addresses them.

---

## 0. Our labels and the organisers' labels are not the same convention

The single most important thing learned so far, and it invalidated two earlier decisions.

A blind re-annotation of the public window (`validation/calibrate.py`) against the organisers'
own labels for it:

- **frame agreement 90.4%**, macroF1 0.904 — that is the *human ceiling*, the best a system
  should expect
- **11 possessions where the official labels have 17** — a 35% under-count. Every official
  possession overlapped one of ours, so nothing was missed; short ones were **merged**. The
  official labels contain possessions of 0.8 s, 1.3 s and 3.5 s. A human watching video does not
  register a 0.8 s touch as a possession; corrected tracking does, and CHALLENGE.md says the
  ground truth is derived from it *mechanically*, with nothing smoothed afterwards.
- **dead-ball starts matched exactly** (median offset 0 frames) but **dead-ball ends were ~0.9 s
  early** (−12 to −29 frames). A human ends a stoppage when play visibly resumes; the official
  labels end it when the ball is struck.

Both earlier decisions that were made on the hand labels alone reversed:

| decision | chosen on hand labels | after calibration |
|---|---|---|
| exp01's dead-ball over-extension fix | end the stoppage at the first sign of movement | **wrong** — it fit the 0.9 s bias and cost the official labels 16 points of dead-ball recall (74.5% → 58.1%). Replaced by a 15 s cap on the restart search, which costs the official labels nothing and fixes the same problem. |
| exp02's evidence threshold | 2 s | **1 s** — the official labels prefer it 0.485 to 0.462 and it gives 16/17 possessions against 9/17 |

**What to take from it.** A hand-built validation set measures your convention, not the target's.
Calibrate it against whatever official labels exist *before* tuning anything on it. A mechanical
correction can hold the line in the meantime — a +22-frame shift on the dead-ball ends did, and
was dropped once the labels were redone by hand, with a measured effect of 0.0006 macroF1 — but
the real fix is to find out *why* the conventions differ and re-annotate. See lesson 0b.

**Possession rate is the cheap sanity check.** The official convention runs at 3.4 possessions
per minute, implying ~360 across the graded match. Any model far off that is wrong whatever
spellF1 says: at 2 s exp02 ships 203, at 1 s it ships 264.

---

## From the validation set and exp02

**1. The public window could not validate anything — not even the structural numbers.**
*(Still true for sample size. Lesson 0 adds the other half: it is nevertheless the only
convention authority, so small and authoritative beats large and skewed for anything about
possession counts.)*
On it, exp01 hit the possession count exactly (17/17), fragmentation 1.00, `D` precision 0.975.
On the 15-minute validation set the same model predicted 49 possessions against 28 at `D`
precision 0.706. The scorer had already said spellF1 was unrankable at 17 possessions; it turns
out *everything* was. Decide only on pooled annotated time, and look per window before trusting
a pooled number.

**2. Two label sets can disagree outright, and the per-window view is what explains it.**
The variant that won the validation set by a wide margin was the worst experiment on the public
window. Not noise: the public window has a cluster of rapid short possessions (0.8–3.5 s) that
any evidence threshold above ~1 s merges. `validation/compare.py` exists because of this.

**3. The rule you trust and the rule you doubt can swap.** exp01's out-of-bounds rule was 99.4%
precise on the public window and 51.9% on validation; the slow-pitch rule went 92.0% → 99.2%.
The flagged-as-risky rule was fine. The confident one was wrong — and it was wrong in its
*duration* logic, not its detection, which no precision number on 4 stretches could have shown.

**4. Asymmetric errors should decide the default — but bound the damage, do not take a side on
the convention.** A stoppage that ends late puts dead ball inside a live possession, splits it,
and costs a whole possession on spellF1; one that ends early costs a few frames of recall. The
first fix for that was "end the stoppage at the earliest sign of movement", which was measured
on the hand labels and turned out to be fitting their 0.9 s bias (lesson 0). The fix that
survived calibration was duller and better: **cap how far the restart search may run** (60 s →
15 s). It bounds the worst case without asserting when a stoppage ends, and costs the official
labels nothing — 97.5% precision / 74.5% recall either way.

**5. Backdating is the most robust single improvement found so far.** Labelling a possession
change from where its evidence *began*, rather than where it became conclusive, helped at every
threshold tried, and helped more the stricter the threshold. It is what lets you demand more
evidence without paying for it in latency. Any future rule that waits for confirmation should
backdate.

**6. Most possession errors looked like flip-backs — but the cure was worse than the disease.**
The hand labels showed exp01 handing ~2 s possessions to a presser who never controlled the ball,
which argued for demanding 2 s of evidence. Calibration then showed those labels under-count
possessions by 35%, and that the official convention *does* contain possessions that short. The
flip-backs are real; raising the threshold to suppress them destroys more real possessions than
it removes false ones. Backdating survived; the longer threshold did not.

**7. A published method can fail to transfer — but check which convention judged it.** Touch
gating, the research note's strongest result (0.68 → 0.86 precision), was rejected on the pooled
hand labels. After calibration that verdict is in doubt: the touch path's whole effect is to
produce more, shorter possessions, which is what the official convention wants and what a human
annotator strips out. On the official window it is now marginally *ahead* (0.509 vs 0.485) and
its possession count is closer to the implied ~360. It ships off because 0.509 vs 0.485 is far
inside the official window's noise band — refusing to decide on 17 possessions, not a verdict.
The warning sign was visible before calibration and was misread: touches helped on the public
window and hurt on all three hand-annotated ones. The detector
finds the right *number* of touches but they are ball events, not control events: a presser's
block is a touch, and confirming it with nearest-player evidence favours the presser by
construction. Measure the published idea before building on it; keep the code and switch it off
rather than deleting the record.

**8. Apparent "home bias" was a lag artifact.** exp01 exaggerated whichever team dominated each
window (0.29 vs 0.44 true home share in the away-leaning window). exp02 fixed it without touching
anything team-specific (0.42 vs 0.44). Check whether a bias is in the data, in the model's
dynamics, or real before correcting for it.

**9. Smart Tagging predicted the true possession balance well.** Windows chosen on its balance
(0.82 / 0.62 / 0.36) came out at 0.77 / 0.57 / 0.44 true home share. Good as a selection signal
for annotation even though it is poor as a per-frame label.

**10. The next lever is under-segmentation.** Every variant produces fewer possessions than the
official convention implies: ~360 expected, exp03 ships 242. The dead-ball layer is close to its
ceiling (86.6% precision, 79.3% recall pooled); the possession layer is not. Settling the touch
question on a correctly-labelled window is the obvious next step.

**10b. Re-tested on corrected tracking, touch gating still does not help** (0.485 official /
0.397 pooled with it, against 0.503 / 0.442 without). Some of what it appeared to fix earlier was
keeper noise; with that gone it has less to do. The question is not closed, but it is now the
*second* thing to try for under-segmentation rather than the first.

## From building the features

**11. Measure the data before trusting its documentation.** The ATD's ball speed column is NaN on
every frame, though DATA.md documents it. Player speeds are fine.

**12. `float("NaN")` does not raise.** A `try/except ValueError` around parsing silently averaged
NaN into a feature and emptied it on 94.6% of frames. It looked plausible and calibrated cleanly
— against a broken feature. Test the text, not the exception.

**13. Savitzky-Golay of order 2 cannot denoise acceleration**, because it reproduces quadratics
exactly. Read derivatives off the fit. Never leave raw values at the edges of smoothed runs.

**14. 2D-projected balls accelerate.** A ball in the air projects its height onto the pitch plane
as real curvature: median acceleration 6.7 m/s². Published thresholds measured on 3D-ish optical
tracking need checking against a rate prior (≈ 2000 touches per match).

## From exp01

**15. Duels are not dead balls.** Both teams within 3 m: P(dead) = 1.6%, below the base rate.
**16. Never resolve contested possession from the future.** Taking the team that has the ball
1–2 s later made team accuracy worse (0.708 → 0.58). Possession stays with the last team that
had it.
**17. Smart Tagging set pieces are a weak dead-ball detector** (35 s windows, half the stoppages
missed, never fouls), and its kickoff rule earned nothing.
**18. Holding the team through ball-less frames** instead of emitting `D` is nearly free on
spellF1 and was the biggest single departure from the baseline.

## Process

- **Calibrate a hand-built label set against official labels before tuning on it.** This cost two
  wrong decisions. `validation/calibrate.py` takes one afternoon of annotation and would have
  caught both.
- Pre-register a threshold from reasoning, then measure. Report both the pre-registered and the
  tuned result when they differ.
- When two label sets disagree, do not average them — work out *which convention each measures*
  and which one the score uses.
- Keep every ablation reproducible from a flag. Every claim above can be re-run.
- Look for plateaus, not peaks. A threshold whose neighbours score the same is safe to keep.
- Annotate a fresh window, without looking at the current model's output, before claiming an
  out-of-sample number.
