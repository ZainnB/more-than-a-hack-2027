# Tracking plan — better inputs for the possession pipeline

Written for: the team. Builds on [PLAN.md](PLAN.md) (rules, working policy) and the research in
[research/](research/). The numbers below come from the IDSSE harness in
[exp06_idsse/](exp06_idsse/) (`whatif.py`, `tune.py --scenario`).

Deadline **November 17, 2026**. Steps are ordered by measured value per day of work.

---

## 1. Why tracking, and how much it can be worth

The harness degrades 7 Bundesliga matches to look like our tracking. Two runs settle what better
inputs are worth: one turns each kind of noise down with the pipeline held fixed, the other re-tunes
the pipeline for each data-quality level.

| data quality | fixed pipeline | re-tuned for that data |
|---|---|---|
| ATD-like (today) | 0.350 | 0.355 |
| team identity fixed within 1.5 m of the ball | 0.408 | — |
| ball missing 41% → 32% | 0.363 | — |
| ball 32% + team identity fixed near the ball | 0.423 | 0.429 |
| ball 32% + no team flips + sharp kicks | 0.448 | 0.455 |
| perfect tracking | 0.450 | **0.512** |

What that means:

- **Team identity near the ball is the biggest single lever: +0.057.** Fixing identity only within
  1.5 m of the ball is worth as much as fixing it everywhere, because only the nearest players
  ever decide the team call. That makes the job small: a few players per frame, not 22.
- **Seeing the ball more often is worth about +0.013–0.023.** The ceiling is low because a ball that
  has gone past the line is lost by any camera; only the in-play gaps can shrink.
- **Sharp kicks (an unsmoothed ball) add +0.011 alone, +0.027 combined** with good team identity,
  because the touch path needs both.
- **Perfect tracking with this decoder tops out at 0.512.** Everything above that is the decoder's
  logic, not the data. Tracking is worth up to +0.16; a realistic upgrade is worth about +0.08.
- **Tuning does not go away.** The best settings change completely with data quality: on perfect
  data the touch path is *off*, margin 0.5 m, k 10 frames; on ATD-like data it is *on*, margin
  1.0 m, k 75. What better inputs remove is the brittle rules (ball-lost, a 20 s slow-pitch window)
  that exist only to paper over bad data. Re-tuning is a 3-minute step on the harness, so it stays.

Caveat: the simulator spreads team flips evenly in time. Real identity switches probably cluster in
duels, which would make the near-ball fix worth *more* than modelled.

## 2. Rules that shape the design

- **No training, fitting or calibrating on the provided video** (kept hard). Anything learned is
  trained on external data. Per-match colour clustering on our video is a grey zone (PLAN.md D10), so
  the team step below fits nothing: it compares each player with the other players in *the same
  frame*.
- **No kit colours are given.** The ATD metadata has none, so team colour must come from the video
  frame itself, anchored to the ATD's own team labels (about 93% right), never from a fitted model.
- **Derived video and audio features are never committed.** They live in git-ignored caches and the
  pipeline regenerates them.
- **Dependencies.** Decoding video needs opencv (already in `requirements-validation.txt`) and
  decoding the tactical audio needs PyAV. Both move into the submission's requirements, which ends
  the numpy + pandas convention. The rules allow it; the technical document says so.

---

## 3. Steps

### Step 0 — feasibility spikes (half a day, Nov 9) — gate for everything below

| | spike | pass if |
|---|---|---|
| 0a | decode the tactical mp4 end to end with opencv, time it | a full pass under ~30 min on the M5 |
| 0b | project ATD players onto 20 frames with the provided homography, draw them | feet land on the players; note player height in pixels across the pitch, which sizes the crops |
| 0c | decode 10 s of tactical audio around the public window's three whistle-only stoppages (08:34, 10:35, 11:35), plot the spectrogram | a narrow tone at 2–4.5 kHz is visible: whistle **GO**, else **NO-GO** |
| 0d | label-free size of the identity problem on our match: near-ball player from an over-subscribed team (more than 11 tracked), keeper on the wrong half, a track whose team changes | a baseline to measure step 1 against (6.24% of frames already measured for the first) |

### Step 1 — team identity near the ball (Nov 9–11) — the +0.057 lever

**Method: same-frame colour consensus.** Fits nothing.

1. On frames where the ball is seen, take each ATD player within 3 m of it. These are the only players
   that can decide the team call.
2. Project the player's feet into the image (homography), crop a torso box sized from 0b, mask the
   grass (green hue), and describe the remaining pixels: a hue–saturation histogram and a robust
   mean colour.
3. Build each team's reference **from the same frame**: the median descriptor of the other players
   the ATD labels as that team, excluding near-ball players and goalkeepers (different kits; exp03's
   keeper slots identify them).
4. If the candidate is clearly closer to the other team's reference, flip its team for that frame.
   Then take a majority over about ±1 s along the track so one bad crop cannot flicker it.
5. Recompute `d_home`, `d_away` and the near-ball counts from the corrected teams. The pipeline is
   unchanged downstream.

**Thresholds** (how much closer counts as "clearly", the smoothing window) are set a priori and
conservatively. They cannot be tuned on our video. If time allows, they are tuned on external
footage. SoccerTrack v2 is the licence-clean option, but it is 145 GB and gated, so that is
probably after the deadline.

**Measure:**
- label-free: the share of near-ball frames re-labelled (expect a few percent; above ~20% means
  the kits are too similar for colour, so stop), and the over-subscribed rate before and after
- report only: public GT and hand windows

**Risk:** similar kits, shadows, occlusion in exactly the duels that matter. The fallback when
colour is ambiguous is to keep the ATD label, so the step can only abstain, never invent.

### Step 2 — stoppages (Nov 11–13) — the dead-ball recall problem

exp06 misses short whistle stoppages: dead recall is 0.38 on the public window, because the tuned
slow-pitch rule needs 20 s of walking pace.

**2a — a dead-ball classifier trained on IDSSE (tracking only, legal, measurable).**
- Per-frame features after Lang et al. 2022 and Vidal-Codina et al. 2022: the distribution of
  team speeds, collective deceleration (the onset of everyone slowing at once), team spread,
  players stationary near a line or the corner arc, ball out of bounds, ball missing.
- A small logistic or boosted model in numpy, trained on degraded IDSSE BallStatus, scored with
  leave-one-match-out.
- It **replaces** the hand-tuned slow-pitch and ball-lost rules. That is the "fewer parameters" win.

**2b — the referee's whistle on the tactical audio (only if 0c is GO).**
- The tactical mp4 carries 48 kHz audio on the submission timeline: no alignment, no replays.
- Detector designed a priori: band energy 2–4.5 kHz relative to the whole spectrum, a sharp
  spectral peak (low flatness), duration at least 0.15 s. Nothing is fitted to our audio.
  Optionally checked on external CC0 whistle recordings.
- Used as an anchor: when 2a finds a stoppage, its start moves back to the last whistle in the
  preceding few seconds.
- IDSSE has no audio, so 2b can only be measured on our match, as a report.

**2c — collective deceleration** is the no-audio fallback, and is already a feature of 2a.

### Step 3 — re-tune, freeze, submit (Nov 14–16)

- Update the simulator to match what steps 1–2 achieved, for example near-ball identity fixed at
  the rate step 1 actually re-labels. Re-run `tune.py` stages A and B2, freeze the values with
  their provenance, and run on the match.
- `validate_submission.py`, write the technical document, upload **before** the 17th. It needs a
  Google account.

### Step 4 — ball detector (after the deadline, unless steps 1–3 finish early)

- Lower priority because it is worth less: +0.013–0.023 for visibility, +0.011–0.027 for sharp
  kicks.
- The full version (a WASB-style heatmap network retrained on SoccerTrack v2 with our own labels on
  that footage) is days of work.
- A cheaper interim, no training: a small-blob detector searched only near the ATD ball's
  predicted path, to restore sharp kicks and fill short in-play gaps. Decide after step 1.

### Step 5 — the decoder (the bigger ceiling, Phase 2)

Perfect tracking caps this decoder at 0.512. Beyond that is a learned possession decoder: an
HMM or CRF with learned emissions and duration constraints, trained on IDSSE, as in research
PLAN.md D5 option 2. Not before the deadline.

---

## 4. Timeline

| date | work | gate |
|---|---|---|
| Nov 9 | Step 0 spikes; start step 1 crops | 0a–0d pass; whistle GO / NO-GO |
| Nov 10–11 | Step 1 team identity | re-label share sane; report scores |
| Nov 11–13 | Step 2a classifier on IDSSE; 2b whistle if GO | IDSSE leave-one-out dead F1 beats the rules |
| Nov 14–15 | Step 3 re-tune and freeze | held-out IDSSE holds |
| Nov 16 | final run, validation, technical document | file valid |
| Nov 17 | submit (early in the day) | |
