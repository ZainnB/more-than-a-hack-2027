# Learnings

Written for: the team. A running log of what has been learned across experiments — the things
worth knowing before starting the next one. Per-experiment detail is in each folder's README;
this file holds only what generalises. Newest section first.

## Scoreboard so far

Two columns, because they measure different conventions and only one decides the competition.
`public` is the organisers' own 5 minutes; `pooled` adds our 15 hand-annotated ones (re-annotated
2026-10-23, 32 possessions).

| | public spellF1 | public poss (true 17) | pooled spellF1 | pooled macroF1 | full-match poss |
|---|---|---|---|---|---|
| baseline_smooth50 (published floor) | 0.326 | 26 | 0.248 | 0.664 | 594 |
| baseline_smooth125 | 0.424 | 16 | 0.322 | 0.588 | 398 |
| exp01 | 0.424 | 17 | 0.323 | 0.784 | 305 |
| exp02 | 0.485 | 16 | 0.389 | 0.810 | 264 |
| **exp03** | **0.503** | 14 | **0.442** | **0.826** | 242 |

Each experiment beats the last on both columns. The official convention implies **~360**
possessions across the match, so every one of them under-segments, and exp03 most of all — that
is now the biggest single error in the pipeline.

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
Calibrate it against whatever official labels exist *before* tuning anything on it. Where a
convention gap is systematic and measurable, correct the labels (`recalibrate.py` shifted the 16
dead-ball ends) — and where it cannot be corrected, as with the missing short possessions, stop
using those labels for that decision. The hand labels remain the authority for dead-ball
detection, team accuracy and second-half coverage; they are not an authority on possession counts.

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
