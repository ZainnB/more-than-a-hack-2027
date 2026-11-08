# plan.md — Possession labelling challenge (Phase 1)

Working plan for Claude Code. It merges two research passes (a quick survey and a deeper review) into
**decisions to make in order**, **things to verify before deciding**, and **build phases with gates**.

Anything marked **[VERIFY]** came from secondary sources or a research summary and has not been checked
against the primary text (licence file, paper, dataset card). Do not rely on it until it is checked.

---

## 0. Read this first (instructions for Claude Code)

1. Read `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `RULES.md`, `SUBMISSION.md` before doing anything.
   If they disagree with this plan, they win.
2. Audit the existing repo (earlier experiments such as `exp05`) and summarise what already exists before
   writing new code. Reuse; don't rebuild.
3. You may open and inspect the provided match data to gain insight. The line that matters is in §1:
   nothing provided, raw or derived, is committed or shipped, and no parameter is chosen by scoring against
   the match.
4. Work phase by phase (§4). Each phase ends at a gate. Stop at every gate marked **USER DECISION** and ask.
5. Keep `PROVENANCE.md` and `LICENSES.md` (§6) up to date on every commit that adds a dataset, model,
   weight file, or dependency.

---

## 1. Hard rules (from the challenge) and the guardrails they imply

| Rule | Guardrail in this repo |
|---|---|
| No training, fitting or calibrating on the GT possession data or on the provided video | Every learned parameter, threshold and hyperparameter is either set a priori (physics, the laws of the game, a published value) or fitted on **external** data. Each one is logged in `PROVENANCE.md` with its source. |
| Labels must come from an algorithm | The final file is produced by one reproducible command. |
| Derived outputs of the provided data may not be committable (check `RULES.md` / data terms) | `data/provided/` and `outputs/provided/` are git-ignored. The pipeline regenerates them. |

**Team working policy:**
- **Agent access.** Claude Code may open, inspect and process the provided match data during development to
  understand it and to debug the pipeline. Nothing that contains provided data, raw or derived frame-level
  exports, is committed or shipped.
- **Evaluation is allowed; tuning is not.** The public 5-minute GT and our own hand-annotated windows may be
  used to *measure* the pipeline and to find its failure modes. No parameter value, threshold or on/off
  switch is ever chosen because it scored better on them. If a failure found this way motivates a change,
  the new value must come from a priori reasoning or external data, never from a sweep on the match.
- **Hand labels.** Our hand annotation of the provided match is an internal test of our pipeline only. It is
  never training or tuning data and never part of the submission.
- **Provided tracking statistics** (for example the ball-missing rate or team-attribution noise) are input
  data, not GT or video, and may be measured and used, for example to configure the noise simulator (D4).

---

## 2. Decisions, in order

Each decision lists the options, a recommendation, what to check first (§3 IDs), and who decides.
Decisions are ordered so that each one unblocks the next. Agent data access, use of the public GT, and the
noise-model source are settled by the team working policy in §1, so they are no longer decisions.

### D2 — Repository licence and dependency licence policy
- **Options:** (a) permissive repo (MIT/Apache-2.0): no GPL/AGPL code or weights; (b) GPL-3.0 repo: allows
  sn-gamestate / GPL data-derived models; AGPL (Ultralytics) still adds network-use obligations.
- **Recommendation:** (a). Use RF-DETR N/S/M/L (Apache-2.0) instead of Ultralytics YOLO.
- **Check first:** C1–C3.
- **Decider:** user.

### D3 — Primary tuning dataset for possession and dead-ball logic
- **Candidate:** IDSSE / DFL (Bassek et al. 2025): 7 Bundesliga matches, 25 fps TRACAB tracking,
  per-frame **BallPossession** and **BallStatus**, CC BY 4.0, loadable with `floodlight`.
- **Why:** the challenge GT is "dead where corrected tracking marks the ball dead, otherwise the owning
  team". The DFL flags are very likely the same kind of signal.
- **Check first:** C10 (what the flags mean), C11 (does dead ball start at the whistle or at ball-out?
  Compare the DFL definition with what the public GT shows).
- **Secondary sets:** SkillCorner open data (10 matches, 10 fps broadcast tracking, per-frame possession
  group, MIT) for robustness on noisy automatic tracking. Metrica samples (no licence) only for prototyping.
- **Decider:** user, after C10/C11.

### D4 — Noise model used to corrupt clean external data *(settled)*
- The decoder must be robust to the provided tracking's noise: ball missing on 41% of graded frames, about
  93% team attribution, fragmented tracks.
- **Settled:** measure these rates from the provided tracking (it is input data, see §1) and randomise
  around them (for example ball-missing 30–50%) so the decoder is not fitted to a single value.

### D5 — Possession decoder family
- **Options:**
  1. Rules (nearest player + "control" heuristics à la Link & Hoernig 2017 / Vidal-Codina et al. 2022) +
     HMM smoothing over states H/A/D.
  2. Learned emissions (gradient boosting on per-frame features) + hidden semi-Markov model / CRF with
     minimum-duration and "controlled possession" transition constraints.
  3. Ball-free trajectory models (BallRadar, PathCRF) for frames without a ball **[VERIFY C15]**.
- **Recommendation:** start with 1 as the baseline, move to 2, and use 3 only if Phase 5 shows that
  ball-missing frames dominate the error.
- **Decider:** user at Gate G3.

### D6 — Abstention (`X`) policy
- **Facts:** spellF1 fills an `X` with your last label in that half. Cheap inside a spell, expensive
  across a hand-over. macroF1 counts every `X` as a miss.
- **Options:** never abstain on graded frames; or abstain only on long low-confidence stretches.
- **Recommendation:** decide by simulation on corrupted IDSSE (Phase 5), not by intuition.
- **Decider:** user at Gate G3.

### D7 — Go / no-go on building our own vision *(the big one)*
- **Question:** How much would better ball and player data actually move the score?
- **Method:** value-of-information experiment in Phase 5. Run the decoder on IDSSE with the ball missing on
  41% of frames versus 20% and 10%, and with team-flip noise at 7% versus 2%. The size of the gains tells
  you whether a ball detector or a player/team detector is worth weeks of work.
- **Decider:** user at Gate G4.

### D8 — Ball detector: architecture and training data *(only if D7 = go)*
- **Architecture:** WASB-style multi-frame HRNet heatmap network (code MIT). **Retrain it; don't ship the
  released soccer weights**, which were trained on ISSIA-CNR **[VERIFY C4, C5]**.
- **Training data options:**
  (a) SoccerTrack v2 panoramic footage (CC BY 4.0). Its ball track is reportedly *interpolated between
      events*, not per-frame truth **[VERIFY C7]**, so it needs extra labels.
  (b) TeamTrack / SoccerTrack v1 wide-view soccer with ball boxes **[VERIFY licence C8]**.
  (c) Your own labels on **external** footage (allowed), possibly bootstrapped from (a)'s event anchors.
- **Check first:** C4–C8, T1, T2.
- **Decider:** user.

### D9 — Player and referee detection *(only if Phase 5/G4 shows player tracking is a bottleneck)*
- **Recommendation:** RF-DETR-S (Apache-2.0) with tiling on the wide view + ByteTrack or OC-SORT, fine-tuned
  on the SoccerTrack v2 bounding-box subset. Short tracklets only; no identity work.
- **Check first:** C2, C9, T3.

### D10 — Team assignment approach *(only if building D9)*
- **Issue:** per-match k-means / SigLIP clustering on the provided video may count as "fitting on the
  provided video".
- **Options:** (a) pairwise "same team?" embedding trained on external data, groups named by majority vote
  against the provided tracking's team labels; (b) fixed colour rule tuned on external data;
  (c) per-match clustering (ask organisers first).
- **Recommendation:** (a), or keep the provided team attribution if it is good enough.

### D11 — Restart / event spotter *(optional)*
- **Options:** (a) none: rely on the tracking-based dead-ball classifier; (b) small temporal model trained on
  SoccerTrack v2 ball-action labels (Out, Throw In, Free Kick, Goal…); (c) SoccerNet-trained T-DEED
  (**not recommended**: NDA data, broadcast domain).
- **Recommendation:** (a) first; (b) only if dead-ball boundaries are the main error source.

### D12 — Which video feed(s)
- **Options:** tactical only; tactical + broadcast (shot-type / replay cues for stoppages, but needs
  alignment and replay handling).
- **Recommendation:** tactical only unless D11 (b) is chosen.

---

## 3. Independent checks (do these before the matching decision)

Record the result of each check (date, source, verdict) in `LICENSES.md` or `PROVENANCE.md`.

### Licences and terms
- **C1** RF-DETR: confirm which sizes are Apache-2.0 (N/S/M/L) and which are Roboflow PML; confirm the
  COCO weights you download are the Apache ones.
- **C2** Trackers: read the LICENSE files of ByteTrack, OC-SORT, BoT-SORT.
- **C3** If considering them: sn-gamestate (reported GPL-3.0), TrackLab (reported MIT), SoccerNet-GSR data
  card (reported gpl-3.0).
- **C4** ISSIA-CNR: find the official terms. Reported as non-commercial research only.
- **C5** WASB-SBDT: code is MIT; check GitHub issue #12 (licence of weights/annotations) and whether the
  soccer annotations can be redistributed.
- **C6** SoccerTrack v2: read the Hugging Face gated-access terms and the dataset card; confirm CC BY 4.0,
  download size (reported about 145 GB), and which matches are train/val/test.
- **C7** SoccerTrack v2 ball track: confirm whether it is per-frame or interpolated between events, and
  which matches are clamped. Decides D8.
- **C8** TeamTrack / SoccerTrack v1: confirm licence on the Kaggle page and whether soccer clips have ball
  boxes.
- **C9** SoccerTrack v2 bounding-box subset (MMSports 2025 challenge): availability and terms.
- **C12** SoccerNet NDA text: confirm whether NDA data or weights trained on it can be used in an external
  competition with a public repo. Default assumption: **no**.
- **C13** DFL Bundesliga Data Shootout (Kaggle) rules: confirm "competition use only". If true, avoid every
  Roboflow Universe football dataset or weight derived from it.
- **C14** PFF FC World Cup 2022 data: read the sign-up terms before using it for anything shipped.
- **C16** SkillCorner open data: confirm the MIT licence covers the data, not only the code.

### Facts that decisions depend on
- **C10** IDSSE: read the DFL/Sportec specification and the Bassek et al. 2025 paper for the exact meaning
  of BallPossession and BallStatus (who owns a loose ball? when does "inactive" start and end?).
- **C11** Compare those definitions with the public GT, the challenge docs and the video. Note differences
  such as dead ball starting at the whistle versus when the ball crosses the line.
- **C15** BallRadar (KDD 2023) and PathCRF **[VERIFY: venue, results, code licence, trained on IDSSE?]**.
  Also Peral et al., arXiv 2512.19528 **[VERIFY]**.
- **C17** Lang et al. 2022 (Sci. Rep.): extract the feature list and model settings (reported: up to 92%
  frame accuracy for in-play status from player positions only; no public code found).
- **C18** Vidal-Codina et al. 2022 (Sports Engineering, CC BY): extract possession and set-piece rules and
  default parameters.

### Technical spikes (small, time-boxed)
- **T1** WASB code on Apple Silicon: it ships a CUDA Docker setup. Can it run on PyTorch MPS? Frames/s at
  512–768 px input on 1080p video?
- **T2** Throughput budget: estimate total hours for 199,319 frames per model on the M5. Reject any module
  that needs more than about 6 h per full pass.
- **T3** RF-DETR-S with tiling on MPS: frames/s at 1080p.
- **T4** `floodlight` (and `kloppy`) install and load IDSSE; confirm the possession and ballstatus arrays
  line up with frames at 25 fps.

---

## 4. Build phases and gates

### Phase 0 — Compliance scaffolding (no provided data needed)
- Create `LICENSES.md`, `PROVENANCE.md`, `.gitignore` entries for `data/provided/`, `outputs/provided/`.
- Write `docs/agent_policy.md` restating §1.
- Draft the organiser questions (§7) for the user to send.
- **Gate G0 — USER DECISION:** send §7 questions; record the D2 answer.

### Phase 1 — External data ingestion
- Loaders: IDSSE (floodlight), SkillCorner (kloppy), optionally Metrica (prototyping only).
- Common frame schema at 25 fps: `period, frame, t, players[x,y,team,role], ball[x,y,z|missing],
  possession{H,A}, ball_status{alive,dead}`. Resample SkillCorner 10 fps → 25 fps.
- Unit tests on schema and frame alignment (T4).
- **Deliverable:** `data/external/` cache (git-ignored if licences require) + loaders + tests.

### Phase 2 — Metrics and noise simulator
- Reimplement `spellF1` and `macroF1` exactly as `EVALUATION.md` describes, including filling `X` with the
  previous label within a half. Test against the baseline in `EVALUATION.md` if one is provided as code.
- Noise simulator for clean external tracking: ball dropout (long gaps, per D4 rates), team-label flips,
  track fragmentation and ID switches, position jitter, missing players.
- **Deliverable:** `eval/metrics.py`, `sim/noise.py`, tests.

### Phase 3 — Baseline decoder (D5 option 1)
- Per-frame features: nearest player per team to the ball, distance, relative speed, ball speed and
  direction change ("kink" = kick/touch), time since last touch.
- Rules for controlled possession plus an HMM (H/A/D) with minimum-duration constraints.
- Evaluate on IDSSE with leave-one-match-out, clean and corrupted.
- **Gate G1:** report spellF1/macroF1 on clean versus corrupted IDSSE. Baseline locked.

### Phase 4 — Dead-ball classifier (tracking only)
- Implement Lang-style features (C17): team speed distributions, spread, players stationary near
  boundaries, collective deceleration, distances to lines and spots; plus Vidal-Codina restart-formation
  rules (C18): throw-in taker on the touchline, corner arc, walls, kick-off shape.
- Train gradient boosting on IDSSE BallStatus (leave-one-match-out). Optionally add SkillCorner phases of
  play as a weak in-play signal.
- **Gate G2:** dead-ball F1 and boundary error (frames) on held-out IDSSE.

### Phase 5 — Integrated decoder, abstention and value-of-information
- Move to D5 option 2: learned emissions (possession and dead-ball probabilities) + semi-Markov/CRF decoding.
- Simulate the `X` policies (D6).
- **Value-of-information runs (D7):** ball-missing 41% / 20% / 10%; team-flip 7% / 2%; with and without
  player-track fragmentation. Report the spellF1 change for each.
- **Gate G3 — USER DECISION:** choose decoder variant (D5) and `X` policy (D6).
- **Gate G4 — USER DECISION:** go/no-go on vision (D7), and which module first (ball versus players/team
  versus restart spotter), ranked by measured gain per unit of effort.

### Phase 6 — First run on the provided match (no tuning)
- Run the locked pipeline on provided tracking + tagging + homography. Write the submission file.
- Diagnostics: label distribution, spell count, mean spell length, share of `X`, and the score against the
  public GT and our hand-annotated windows. Scores are for finding failure modes, never for picking values
  (§1).
- **Deliverable:** `outputs/provided/possession.csv` (git-ignored), reproducible via one command.

### Phase 7 — Ball detector (only if G4 = go on ball)
- Resolve D8 with checks C4–C8, T1, T2.
- Retrain a WASB-style heatmap network on external wide/panoramic footage. Augment to match the target:
  downscale so the ball is 3–6 px and players 25–50 px, motion blur, compression, colour jitter.
- Add trajectory post-processing: gap bridging, kink (touch) detection.
- Validate on held-out external matches only. Feed detections to the decoder as emissions.

### Phase 8 — Players / team (only if G4 says so)
- RF-DETR-S + tiling + ByteTrack/OC-SORT, fine-tuned on SoccerTrack v2 box subset (D9).
- Team assignment per D10. Use to fill gaps in the provided tracking, not to replace it wholesale.

### Phase 9 — Restart / event spotter (only if dead-ball boundaries dominate the error)
- D11 option (b): small temporal model on SoccerTrack v2 ball-action labels, tracking features first,
  low-resolution frame features second.

### Phase 10 — Submission and technical document
- Final run from a clean checkout, one command, pinned environment, model download script (no derived
  outputs committed if terms forbid it).
- Technical document: pipeline diagram; every dataset and model with licence and what it was used for;
  every tuned parameter and the external data it was tuned on; the D2–D12 decisions and reasons;
  noise-simulation and value-of-information results; limitations.

---

## 5. Resource register

Status: 🟢 use · 🟡 use after the listed check · 🔴 do not use for anything shipped.

### Data
| Resource | What it gives | Licence (as reported) | Status |
|---|---|---|---|
| IDSSE / DFL (Bassek et al. 2025) | 7 matches, 25 fps, BallPossession + BallStatus + events | CC BY 4.0, credit DFL + cite paper | 🟡 C10 |
| SkillCorner open data | 10 A-League matches, 10 fps broadcast tracking, possession group, dynamic events, phases of play | MIT | 🟡 C16 |
| SoccerTrack v2 | 10 full matches, panoramic 4K, game-state + 12-class ball actions; box subset | CC BY 4.0 (code MIT) | 🟡 C6, C7, C9 |
| TeamTrack / SoccerTrack v1 | Wide fisheye 8K soccer clips with boxes | unverified | 🟡 C8 |
| Metrica sample data | 3 anonymised games, tracking + events | no licence | 🟡 prototyping only |
| PFF FC World Cup 2022 | 64 matches, broadcast tracking + events | sign-up terms | 🟡 C14 |
| ISSIA-CNR | Fixed multi-camera soccer, ball annotations | reported non-commercial research only | 🔴 unless C4 says otherwise |
| SoccerNet (BAS, Team-BAS, Tracking video) | Broadcast video + labels | NDA | 🔴 unless C12 says otherwise |
| DFL Bundesliga Data Shootout (Kaggle) + Roboflow football sets derived from it | Clips, event labels, YOLO data | reported competition-use only | 🔴 |

### Models and code
| Resource | Role | Licence (as reported) | Status |
|---|---|---|---|
| WASB-SBDT code | Ball heatmap detector architecture | MIT | 🟢 code / 🔴 released soccer weights (ISSIA) |
| TrackNetV3 / TrackNetV4 | Alternative ball heatmap architectures | MIT (V3); V4 unverified | 🟡 |
| FootAndBall | Light ball+player detector | MIT code; ISSIA-trained weights | 🟢 code / 🔴 weights |
| RF-DETR N/S/M/L | Player/referee detector | Apache-2.0 | 🟡 C1 |
| RF-DETR XL/2XL/Atto/Femto/Pico | — | Roboflow PML | 🔴 |
| Ultralytics YOLO | — | AGPL-3.0 | 🔴 under D2 (a) |
| ByteTrack / OC-SORT | Short-term association | check | 🟡 C2 |
| floodlight, kloppy | Loaders (IDSSE, SkillCorner, Metrica, PFF) | check | 🟡 |
| BallRadar / PathCRF | Ball-free possession inference | unverified | 🟡 C15 |
| T-DEED (SoccerNet-trained) | Action spotting | code licence check; NDA training data | 🔴 weights |
| sn-gamestate / PRTreID | Game-state pipeline / team+reID | reported GPL-3.0 | 🔴 under D2 (a) |
| Pose models (RTMPose, ViTPose) | — | — | skip (players too small) |

### Papers to implement from
- Link & Hoernig 2017, *Individual ball possession in soccer* (PLOS ONE).
- Vidal-Codina et al. 2022, *Automatic event detection in football using tracking data* (Sports Engineering, CC BY).
- Lang, Wild, Isenko, Link 2022, *Predicting the in-game status in soccer…* (Scientific Reports).
- Bassek et al. 2025, *An integrated dataset of spatiotemporal and event data in elite soccer* (Scientific Data).
- Tarashima et al. 2023, WASB (BMVC).

---

## 6. Record-keeping files

- `LICENSES.md`: one row per dataset/model/dependency: name, version/commit, licence, link, date checked,
  check ID (C1…), allowed use, attribution text.
- `PROVENANCE.md`: one row per learned or tuned thing: name, dataset fitted on, split, script, commit.
  Must never list the provided match or its GT.
- `docs/decisions.md`: D2–D12 with the option chosen, date, and reason (feeds the technical document).

---

## 7. Questions to send the organisers (Gate G0)

1. Does per-match unsupervised clustering (for example jersey colours) on the provided video count as
   "fitting on the provided video"?
2. May models trained on NDA-gated research datasets (for example SoccerNet) be used if the weights are
   downloaded by the pipeline rather than committed?