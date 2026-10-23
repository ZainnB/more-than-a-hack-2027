# Ball Possession Detection from Broadcast-Derived, Team-Labelled Tracking Data: Methods, Code, Baselines and a Recommendation Ladder

For your data (2D ball and on-screen people, labelled only home/away, no persistent IDs), the best design is a **team-level, touch-gated possession state machine**. Per team, take the nearest visible player to the ball. Confirm a change of possession only at a ball "touch" event (an acceleration or direction-change spike) near a player of the other team. Carry possession through passes and loose-ball phases. Exclude dead-ball time. Then smooth with hysteresis or an HMM/Viterbi decoder. Almost every published individual-possession method reduces cleanly to this team-level form, because you only ever need the *team* of the nearest player, never their identity.

## TL;DR
- **No single possession definition exists.** Opta's headline possession % is pass-based. Wyscout counts pure time with the ball out of play excluded. FIFA adds an explicit "in contest" state. Decide up front which one you are reproducing, because the gap between definitions (11%/16%/28% for the same match) is larger than any algorithmic error you will fight.
- **Build in this order:** (1) proximity per team with a ~1–2 m possession zone and a duel zone; (2) touch gating from ball acceleration and direction change; (3) "last controlling team keeps possession until the opponent controls"; (4) HMM/Viterbi or hysteresis smoothing. Link & Hoernig (TRACAB, 25 Hz) reached timeline agreement of 0.87–0.92 with this family of rules. Expect less on broadcast data, where ball position error can reach several metres.
- **None of the strongest learned methods needs player IDs.** PathCRF, TranSPORTmer and set/graph networks all work if you treat players as an unordered, team-tagged set. The fastest route to labels is to *simulate your broadcast conditions*: take SkillCorner open data or the DFL/IDSSE data, which include provider possession flags, and mask out undetected or off-screen players.

## Key Findings

### 1. Definitions of possession (decide this first)

| Source | Definition | Dead ball | Contested state |
|---|---|---|---|
| **Opta / Stats Perform** (headline %) | Pass share: team passes ÷ total passes. Opta dropped its earlier "chess clock" logging because missed button presses skewed results.\[1\] Separately, Opta defines *a possession* as starting "upon a player controlling the ball"; winning a tackle or header does not by itself start one.\[2\] | Not time-based, so not applicable | Implicit |
| **Wyscout** | "Pure time spent in possession". A *Possession* is a sequence of events of one team. Lost duels, missed balls and rebounds that are immediately picked up do not break a possession. Open-play possessions are bucketed 0–10 s, 10–20 s, 20–45 s and 45+ s. | Out-of-play time (ball out, setting up a set piece) counts for neither team\[3\] | Absorbed into the possession |
| **StatsBomb** | Each event carries a `possession` counter and `possession_team_id`: "the team that started this possession in control of the ball". Possession chains are the native unit. StatsBomb's own blog notes that media possession % is "really just a pass ratio".\[4\]\[5\] | Chain boundaries | Events inside a chain keep the chain's team |
| **FIFA (Football Language / World Cup stats)** | Team possession plus an explicit **"in contest"** state: "when neither team has controlled possession of the ball" (duels, aerial duels, loose balls after clearances, rebounds, misplaced passes). At the 2022 World Cup the ball was in contest on average 11% of each game. Spain averaged 9%.\[6\]\[7\] | Excluded | **3-state** |
| **Link & Hoernig (academic, 2017)** | Team Ball Possession (TBP) runs from a team's first individual possession until the opponent's first. While the ball is in play, one team *always* has TBP. They also define narrower Team Ball Control (TBC) and Team Playmaking (TPM). Means per match: TBP 56:04 min, TPM 50:01, TBC only 17:49. | Not assigned to anyone\[8\] | 2-state for TBP, with finer states underneath |

**Why providers disagree:** pass-count and time-based possession differ, and the choice of what counts as "control", how contested time and dead time are allotted, and live human logging all add more variance. The canonical example is Celtic v Barcelona (2012): Celtic were reported at 11%, 16% or 28% depending on the source. The OptaPro blog post "Possession(s)" states that Celtic "possessed the ball for around 18% of the time… Opta's in-game measurement using the simplified pass completion method was 16.4%, whereas measurements from other data suppliers that used the 'chess clock' approach were around the 30% mark." Even manual event data carries 10–15% tagging differences between analysts, according to Vidal-Codina et al. Treat any provider % as one convention, not ground truth.

**Recommendation for your output:** report three numbers:
1. Time-based possession % over ball-in-play frames with contested and unknown time excluded (Wyscout-like).
2. The contested % (FIFA-like).
3. Unknown/unobservable % (unique to broadcast data).

You can always recompute a 2-state "carry-through" version by giving contested time to the last controlling team (Link & Hoernig TBP).

### 2. Geometric / proximity baselines

**a) Nearest-player + threshold.** This is the default in nearly every YOLO/ByteTrack hobby repo.
- The widespread `PlayerBallAssigner` pattern measures the distance from the ball centre to both bottom corners of each player's bounding box (as "feet"), takes the minimum, and assigns the ball to the closest player under `max_player_ball_distance = 70` **pixels**.\[9\] Team ball-control % is the cumulative share of assigned frames.\[10\]\[11\]
- One fork has an open issue proposing a switch to metric pitch distance with a 2.0 m threshold.\[12\] That is the correct fix: pixel thresholds change with zoom.
- **Tryolabs** (soccer-video-analytics, YOLOv5 + Norfair) uses the same closest-foot distance, a threshold, and a possession **"inertia"** counter. The other team must hold the ball for N *consecutive* frames before possession flips, and "if there is an interruption, the inertia restarts". The authors say this is meant to suppress flips on rebounds or when the ball passes near a player. Possession defaults of 45 px and 20 frames come only from an auto-generated DeepWiki summary of `soccer/match.py`. The verifiable `inertia=20` in `run.py` (`InertiaClassifier(classifier=hsv_classifier, inertia=20)`; blog: "HSV classifier with inertia 20 on a 25 FPS video") belongs to the team-jersey *classifier*, not the possession counter. The authors also say the demo is not fit for real matches.
- **Failure modes:**
  - Lofted balls passing "over" players, because there is no z.
  - Fast passes rolling past a player who does not touch them.
  - A player running beside a loose ball.
  - Duels where two opponents are equidistant.
  - The true nearest player being off-screen.
  - Single-frame team-label flips.

**b) Possession zone + duel zone (Vidal-Codina et al., MIT/FIFA, 2022).**
- Possession zone (PZ): a circle of radius R_pz around every player. Duel zone (DZ): a circle of radius R_dz ≥ R_pz around the ball. If at least two opponents are inside the DZ, the frame is a duel, and **duel prevails** over possession.\[13\]
- The ball-control state per frame takes one of four values: dead ball, no possession, possession or duel.\[13\] That is effectively a 3-state team model plus dead ball.
- Chosen radii: R_pz = 0.5 m for Track160 and 1.0 m for Tracab/Hawk-Eye (searched over 0.5–1.5 m, a range recommended by FIFA experts); R_dz = 1.0 m.\[13\]\[14\]
- Rationale: 99% of Tracab passes show a minimum passer–ball distance under 1 m, but only 80% under 0.5 m.\[14\]
- Touch/trajectory change thresholds:\[14\]
  - direction change ε_θ = cos 10°\[14\]
  - speed change ε_v = 5 m/s (1 m/s for Track160)\[14\]
  - minimum ball displacement ε_s = 10 cm\[14\]
- The ball is smoothed with a Savitzky–Golay filter (2nd order, 7-frame window).\[13\]
- For lower-quality data such as broadcast, the authors advise enlarging the possession zone and retuning.\[13\]

**c) Kick-detection + ball-prediction (Link & Hoernig, PLOS ONE 2017).** This is the most directly reusable rule set.
- Preprocessing: remove ball jumps, drop sections shorter than 1 s, then apply a Rauch–Tung–Striebel smoother before computing velocity and acceleration (25 Hz).\[8\]
- **Start** of possession: the player is the nearest to the ball AND within distance T_K AND the ball's acceleration shows a **local maximum ≥ 4 m/s²** ("kick detection"). This suppresses balls passing over players.\[8\]
- **End** of the action: project the ball 1 s ahead at its current velocity. The action ends when that predicted position is more than T_A from the player. Adding player prediction did not help.\[8\]
- Thresholds were grid-searched against manual ground truth.\[8\]

| Method | Recall | Precision | F | Timeline |
|---|---|---|---|---|
| Naive proximity | .90 | .68 | .77 | .82 |
| + kick detection | .80 | .86 | .83 | .87 |
| + kick det. + ball prediction (IBA) | .86 | .91 | .88 | .92 |

Matching used a 0.6 s tolerance.\[8\] The key takeaway: **touch gating buys +15–18 points of precision** over pure proximity. The authors note that real-time human loggers reached only 52% under the same tolerance. They also found that the IBC "control" construct reaches only κ = .72 even between two humans.\[8\]

**d) Temporal smoothing options** (your choice matters more than the threshold):
- *Hysteresis / inertia:* require K consecutive frames for a switch. Tryolabs uses about 0.8 s. Set K in **seconds** so it works at both 10 Hz and 25 Hz.
- *Mode filter:* a sliding-window majority vote over 0.3–0.5 s, applied to team labels before possession logic.
- *Minimum segment duration:* merge possession segments shorter than X s into neighbours unless bounded by a touch event.
- *HMM/Viterbi:* states {home, away, contested, dead/unknown}. Emissions come from distance features; a transition matrix penalises switches. This is the principled version of all of the above, and the team-level analogue of PathCRF's masked CRF (see §4).

**e) Velocity/direction awareness.**
- Gate assignments by ball speed. A ball moving fast and showing no deceleration or direction change near a player is "in flight", not possessed (Link & Hoernig's 1 s ball projection; Vidal-Codina's ε_v and ε_θ).
- A ball moving toward a player with decreasing relative distance signals a likely reception. One moving away after an acceleration spike is a pass.

**f) Dominant region / pitch control.** These answer "who would control the ball *if* it were at x", not "who has it". Lineage: Voronoi dominant regions (Taki & Hasegawa; Fujimura & Sugihara's motion-model Voronoi), Spearman's physics-based pitch control, and Fernández & Bornn's space-occupation model.\[15\]\[16\] They are useful as a *soft prior* for loose-ball and contested frames (§3), not as a primary possession detector.

### 3. Pitch control as a team-level fallback

- **Spearman / LaurieOnTracking implementation.** Repo: github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, file `Metrica_PitchControl.py`.\[17\]\[18\]\[19\]\[20\]
- **Time to intercept:** the player keeps his current velocity for `reaction_time` = 0.7 s, then runs at `max_player_speed` = 5 m/s. Arrival uncertainty is a sigmoid with `tti_sigma` = 0.45.\[18\]
- **Control rates:** `lambda_att` = 4.3, `kappa_def` = 1. Shaw sets κ = 1 so home and away are symmetric; Spearman's paper uses 1.72. Goalkeepers get λ × 3.\[18\]
- **Ball and integration:** `average_ball_speed` = 15 m/s, integration step 0.04 s.\[18\]
- **Runs on team-labelled-only data?** Yes, with caveats:
  - It needs per-player positions and **velocities**, but no IDs. You can estimate velocity per frame by short-horizon nearest-neighbour association within a team; the identities only need to survive ~0.2–0.5 s.
  - Off-screen players are simply missing. That biases control toward the team with more visible players near the ball, so restrict use to a radius around the ball, where broadcast coverage is best.
  - Use it as the emission feature for "contested/loose ball" frames: P(home controls the ball's landing point) breaks ties when no player is inside the PZ.
- Related: Metrica Sports sample data (github.com/metrica-sports/sample-data) provides 2 anonymised matches of full tracking plus events, the dataset these tutorials use.\[21\]

### 4. Machine learning / deep learning

**Directly relevant, and ID-free once adapted:**
- **PathCRF** (Kim et al., KDD 2026; github.com/hyunsungkim-ds/pathcrf).
  - Detects on-ball events from *player trajectories only* (no ball). Each frame is a fully connected graph, and the model selects exactly one edge per frame: a self-loop for a dribble, or sender→receiver for a ball in flight.\[22\]
  - Backbone: a socio-temporal encoder (Set Attention Blocks, TranSPORTmer-style). A **dynamic masked CRF** forbids impossible transitions, and Viterbi decoding produces the path.\[22\]
  - Results: 69.64% correct edge among 26² = 676 candidates; event F1 75.69%. The authors report that team possession statistics are "closely approximated".\[22\]
  - **Adaptation for you:** collapse nodes to {home-set, away-set} and the states to {home, away, in-flight-home→?, loose, dead}. Feed in the ball as well, which you have and they don't. The CRF transition mask is exactly the "possession can only change via a touch" constraint.
- **TranSPORTmer** (Capellera et al., ACCV 2024).
  - A Set-Attention transformer over agents and time, with an extra "CLS agent" that classifies per-frame state: **pass, possession, uncontrolled, out-of-play**.\[23\] This is close to your target label space.
  - Permutation-equivariant by construction. Team identity is an input type token (attacking/defending/ball), not an ID.\[23\]\[24\]
  - It also does ball imputation from player trajectories, which is useful for your ball-dropout gaps.\[23\]
- **Peral et al. 2025** ("Temporally accurate events detection through ball possessor recognition"; plus a multi-modal masked pre-training follow-up).
  - Jointly infers ball trajectory, ball state and possessor from player trajectories and player image crops.\[25\]
  - The VISAPP 2025 paper's headline result is pass F1 67.28% at a 0.6 s window (74.34% at 1 s; mAP@1 44.15). An 85.85% figure circulating in secondary summaries is, in that paper, the *precision* of the ResNet50 video-tube possession baseline, not a ball-state accuracy, and the paper makes no comparison with TranSPORTmer.
  - It relies on tactical-camera footage that always shows the whole pitch, which your broadcast data lacks: "25Hz tactical camera videos of 36 matches" from LaLiga and Serie A 2022-23.
- **Set / graph architectures for ID-free inputs.** Per-frame visible players form a variable-size set: use DeepSets/Set Transformer pooling with team as a node feature and padding masks.
  - *unravelsports* (github.com/UnravelSports/unravelsports) converts tracking data (including SkillCorner broadcast data, via kloppy) into graphs, with options such as `adjacency_matrix_type="split_by_team"` and `adjacency_matrix_connect_type="ball"`. It ships PyTorch Geometric/Spektral classifiers.\[26\]\[27\]
  - It grew out of the U.S. Soccer Federation counterattack GNN, which was trained on SkillCorner *broadcast* tracking (github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn).\[28\] Node-level graph models are permutation-equivariant, so no IDs are needed.\[29\]

**Event detection = turnover detection:**
- **Vidal-Codina et al.** possession losses and gains → passes, receptions, interceptions, set pieces. Over 90% detection rate in most categories.\[13\]
- **Mills et al. (Sports Engineering 2026)** ran the Vidal-Codina algorithm on *broadcast-derived* tracking from three commercial providers (2022 World Cup match):\[30\]
  - Ball position RMSE ranged from 3.5 m (best: Provider 1, programme feed) up to 16.2 m.\[30\]
  - Pass/cross auto-event F1 still reached up to 0.93 (Camera 1), vs 0.93 for TRACAB multi-camera. Mean F1 across providers: Camera 1 0.81, programme feed 0.78, tactical 0.69.\[30\]
  - Shots were weak (best 0.24 vs 0.48 multi-camera). Saves were not detected at all, and goals were sometimes misattributed as own goals because last touch went to the wrong team.\[30\]
  - Replays in the programme feed broke kickoff and goal detection.\[30\]
  - This is the single best external estimate of how a possession-zone method degrades on broadcast data.
- **Bauer et al. (Scientific Reports 2022)** predicted in-game status (ball in/out of play) from player tracking with ML.\[31\] Useful for building your dead-ball flag.

**Video-side (complementary signals, not tracking-based):**
- **SoccerNet Ball Action Spotting.**
  - 2023 edition: *pass* and *drive*, 7 games, 11,041 timestamps.\[32\]
  - 2025 *Team* Ball Action Spotting: 12 classes, with each action also attributed to team left/right relative to the camera. Metric: Team-mAP@1. Baseline: T-DEED (github.com/SoccerNet/sn-teamspotting). The winning entry merged team and action into a single joint head.\[33\]\[34\]\[35\]
  - A team-attributed pass/drive spotter is effectively a **touch detector with team label**, ideal for touch-gating in Rung 3.
- **SoccerNet Game State Reconstruction** (github.com/SoccerNet/sn-gamestate) produces minimap positions with role, team (left/right) and jersey. GS-HOTA scores localisation and attributes; the 2024 winner reached 63.81.\[36\]\[37\] Your data looks like GSR output, and GSR has no possession task.
- **DFL Bundesliga Data Shootout** (Kaggle, 2022): video event detection of *play*, *challenge* and *throw-in*. Average precision was computed over class-specific time tolerances (play 0.15–0.35 s; challenge 0.30–0.70 s). Kaggle's competition page lists 530 teams (610 participants, 846 submissions), and per DFL the top three teams each received USD 25,000, co-funded by AWS. The clips are the source of roboflow/sports' training data.
- **Sarkar et al.** (Pattern Recognition 2022) compute possession from broadcast video by detecting passes with a min-cost flow network over player/ball split–merge events.\[38\]\[39\] They had to use Google Research Football simulations because "there are no openly available datasets on ball possession statistics".\[40\]\[41\] The earlier interaction-energy version reached about 83%.\[39\]

**Weak/self-supervision with team-only labels:**
- Use provider possession flags as labels on public data, then **degrade the inputs to match yours**:
  - drop player IDs
  - drop players flagged `is_detected = false`
  - add homography noise
  - randomly drop ball frames
- Self-supervised pretext tasks (masked ball imputation, à la TranSPORTmer/Peral) on your own unlabelled broadcast tracks, then fine-tune a small head on a few hand-labelled minutes.

### 5. Open-source code, data and libraries

| Resource | What it gives you for possession |
|---|---|
| **roboflow/sports** (github.com/roboflow/sports) | Player/ball/pitch-keypoint YOLO models, ball tracking utilities, and a team classifier (SigLIP embeddings → UMAP → KMeans). No mature possession module. Community add-ons (e.g., github.com/TyBM223/soccer-possession) layer possession % plus a "coverage" stat on top.\[42\]\[43\] |
| **tryolabs/soccer-video-analytics** | Closest-foot distance + pixel threshold + consecutive-frame inertia (see §2a). Demo quality by the authors' own statement.\[44\] |
| **YOLO "football_analysis" family** (many forks with `player_ball_assigner/`) | Nearest bounding-box foot corner < 70 px.\[9\] Ball interpolation for missing detections. Cumulative team-control %.\[10\]\[11\] No hysteresis in the base version. |
| **Vidal-Codina possession/event model** | Python implementation used by Mills et al.; not found as a public repo.\[45\] Re-implement from the arXiv paper (2202.00804). |
| **PathCRF** (github.com/hyunsungkim-ds/pathcrf) | Masked-CRF possession-path inference; reproducible on public tracking data.\[22\] |
| **LaurieOnTracking** | Pitch control, EPV grid, Metrica data loaders.\[17\]\[46\] |
| **SkillCorner/opendata** | 10 A-League 2024/25 matches of **broadcast** tracking at 10 Hz. Per-frame `possession` dict (`player_id`, `group` = home/away). `is_detected` flag per player and for the ball. In match 1886347, 40.7% of player records and 17.7% of ball positions are extrapolated.\[47\]\[48\] **This is the closest public analogue to your data, with possession labels.** |
| **kloppy** (github.com/PySport/kloppy) | Standardises SkillCorner, Metrica, Tracab, StatsBomb and others. Note: an open issue (#639) asks to preserve SkillCorner's `is_detected` flags, which kloppy currently drops.\[47\] |
| **floodlight** (github.com/floodlight-sports/floodlight) | `Code` objects for possession and ball status. Butterworth/Savitzky–Golay filters. Velocity/acceleration models. Loaders for DFL, Tracab, Second Spectrum, SkillCorner, and the **IDSSE** public DFL dataset (tracking + events + possession + ball status).\[49\]\[50\] |
| **socceraction / VAEP** (github.com/ML-KULeuven/socceraction) | SPADL event conversion; StatsBomb possession schema.\[4\] Useful for evaluating against event-derived possession chains. |
| **StatsBomb open-data** (github.com/statsbomb/open-data) incl. 360 | Possession chains plus freeze-frames of *visible* players around each event. 360 is "broadcast-visible area" data, the event-level analogue of your view. |
| **mplsoccer** | Pitch plotting and StatsBomb loaders. |
| **unravelsports** | Graph conversion and GNN training from kloppy tracking data.\[26\] |
| **SoccerNet** sn-gamestate, sn-teamspotting | GSR baseline (detection → team → minimap) and team-attributed ball-action spotting. |
| **Football-Tracking-Interpolation** (github.com/mpenn114/Football-Tracking-Interpolation) | Imputes off-screen players from discrete broadcast data (arXiv 2311.14642).\[51\] Relevant to the "true nearest player is off-screen" problem. |

### 6. Evaluation

- **Ground truth sources:**
  - Manual frame-level annotation from video (Link & Hoernig used a panoramic video with a position overlay).\[8\]
  - Provider possession flags: SkillCorner `possession.group`, DFL/IDSSE possession codes, Tracab flags.
  - Event-derived possession chains (StatsBomb/Opta).
- **Metrics to report:**
  1. **Frame-level team accuracy / macro-F1** over ball-in-play frames, with "unknown" frames counted separately.
  2. **Timeline agreement with a tolerance window.** Link & Hoernig count a frame as correct if the same possession appears in ground truth within ±0.6 s.\[8\]
  3. **Transition (turnover) P/R/F1** with a temporal tolerance: 0.6 s in Link & Hoernig, 3 s in Vidal-Codina/Mills event matching, 0.15–0.7 s in DFL Shootout AP.
  4. **Segment IoU** between predicted and true possession intervals.
  5. **Match-level possession % absolute error** against a provider, *after aligning definitions*. Contested and dead time must be handled identically.
- **Reference numbers:**
  - Link & Hoernig, TRACAB 25 Hz: IBP F = .83 and timeline .87; IBA F = .88 and timeline .92.\[8\]
  - Vidal-Codina, optical data: over 90% detection for most event types.\[13\]
  - Mills et al., broadcast data: pass F1 up to 0.93, mean 0.69–0.81 by feed.\[30\]
  - PathCRF, no ball: 75.69% event F1.\[22\]
  - Human reliability ceiling: κ = .72 for "control", and 10–15% event tagging disagreement.\[13\]

### 7. Engineering issues specific to broadcast-derived tracking

- **Homography errors** inflate ball–player distances unevenly: errors are worst at frame edges and in the far attacking areas (Mills et al. show tactical-feed errors concentrated at the extremities).\[30\] Use metric thresholds of 1.5–2 m rather than optical-data values of 0.5–1 m, and enlarge them with distance from the camera/frame centre.
- **Ball dropouts:**
  - Interpolate only short gaps (≲0.5 s), and only when the endpoints are consistent with ballistic motion.
  - Never let interpolated ball positions *create* a possession change. Flag them and only allow carry-through.
  - Longer gaps → "unknown" unless both sides of the gap show the same team in control.
- **Off-screen nearest player:** if the ball is within ~5–10 m of the image border, a nearer opponent may be invisible. Downgrade confidence or mark "unknown" rather than assigning the visible team. Requiring touch evidence for any switch also limits damage.
- **Camera cuts and replays:** detect shot boundaries (homography discontinuities) and replays, and treat them as hard resets to "unknown". Mills et al. found replays in the programme feed broke kickoff and goal detection.\[30\]
- **Team misclassification:**
  - One mislabelled player next to the ball flips possession.
  - Mitigations:
    - temporal mode filter on each short-lived local track's team label
    - require K frames of consistent team evidence for a switch
    - exclude referees and goalkeepers from KMeans colour clusters; handle goalkeepers by position near their own goal
  - Audit: possession flips where the "new" player stands inside a cluster of the old team are suspicious.
- **Ball height:** with 2D only, lobbed passes and crosses pass "through" possession zones. Gate on ball speed plus absence of deceleration or direction change, as in Link & Hoernig's kick detection.\[8\] Never assign a player when the ball keeps a constant velocity through their zone.
- **Frame rate:** define all windows in seconds. At 10 Hz, acceleration peaks from touches are smeared or missed, so rely more on direction change across ±2 frames and less on the 4 m/s² peak. At 25 Hz broadcast, smooth first (Savitzky–Golay or RTS); raw second differences are noise-dominated.
- **Dead ball:** derive it from the ball leaving pitch bounds, long ball absence, players stationary/clustered (set-piece configurations), or an ML in-play classifier (Bauer et al.). Exclude it from all percentages.\[13\]

## Recommendations — the ladder (simplest → most sophisticated)

**Rung 0 — Preprocessing (mandatory).**
- Project everything to metres.
- Smooth the ball (Savitzky–Golay, 2nd order, ~7 frames at 25 Hz), then compute velocity, acceleration and heading change.
- Interpolate ball gaps up to 0.5 s and flag them.
- Mode-filter team labels on short local tracks (associate within a team by nearest neighbour across consecutive frames; no global ID needed).
- Build `in_play` and `camera_cut` flags.

**Rung 1 — Per-team proximity, 3-state (hours to build).** For each frame compute `d_H` = min distance from the ball to visible home players and `d_A` likewise. Then:
```
if not in_play: state = DEAD
elif d_H < R_pz and d_A < R_dz: state = CONTESTED   # duel prevails
elif min(d_H,d_A) < R_pz: state = team of the min
else: state = LOOSE
```
Start with R_pz = 1.5 m and R_dz = 2.0 m for broadcast (Vidal-Codina optical values are 0.5–1.0 m); tune on labelled data. Team possession = last HOME/AWAY state carried through LOOSE. Expect frequent spurious flips on lofted passes and rebounds.

**Rung 2 — Add hysteresis.** Switch the carried team only after the new team holds HOME/AWAY for ≥0.3–0.5 s, or ≥0.8 s as in the Tryolabs demo. Reset the counter on interruption.\[44\] Merge segments shorter than 0.3 s. This removes most of the flicker cheaply but delays transitions.

**Rung 3 — Touch-gated state machine (recommended production baseline).**
- A touch event = smoothed ball acceleration local max ≥ ~4 m/s² (Link & Hoernig) OR heading change > 10° with speed change, exceeding noise (Vidal-Codina).\[13\]
- Possession may switch to team T **only** at a touch event where the nearest player within R_pz belongs to T. Between touches, possession is carried.
- A ball in flight (high speed, no touch) is never assigned.
- Two opponents within R_dz at the touch → CONTESTED until the next clean touch.
- On optical data this rule family gave F ≈ .83–.88 and timeline ≈ .87–.92. On broadcast data, plan for a meaningful drop, driven by ball error and visibility.
- If you have video, a SoccerNet team-ball-action spotter (pass/drive with team left/right) is an alternative touch detector.

**Rung 4 — Probabilistic decoding (HMM/CRF).**
- States: {HOME, AWAY, CONTESTED, DEAD/UNKNOWN}.
- Emissions: logistic functions of `d_H`, `d_A`, touch score, ball speed, distance to the frame edge, interpolation flag, and optionally local pitch control at the ball.
- Transitions: forbid HOME↔AWAY without an intermediate touch frame (PathCRF-style masking). Decode with Viterbi.
- This gives calibrated confidence and handles "unknown" off-screen periods gracefully.

**Rung 5 — Learned set/sequence model.**
- Input per frame: the ball (x, y, v, a) plus an unordered set of visible players (x, y, v, team, is_edge), padded and masked.
- Architecture: a Set Transformer/DeepSets encoder per frame → temporal Transformer/TCN (or TranSPORTmer-style joint agent-time attention) → per-frame logits over the Rung-4 states → CRF decoding.
- Training data: SkillCorner open data (broadcast, possession group labels) and IDSSE/DFL (floodlight), degraded to match your inputs: drop IDs, drop non-detected/off-frame players, jitter positions by 1–3 m, drop ball frames. Fine-tune on a few annotated minutes of your own footage.
- Expected gain over Rung 4 is in contested/loose phases and around frame edges. There is no public benchmark for exactly your setting, so measure it.

**ID-free signals worth engineering:**
- Ball speed, acceleration magnitude and heading change (touch proxies).
- Ball deceleration on approach (reception).
- Relative velocity between the ball and the nearest player of each team (dribble = co-moving).
- Count of each team's visible players within 3/5/10 m.
- Nearest-player distance ratio d_H/d_A.
- Distance from the ball to the image border (visibility risk).
- Time since the last touch.
- Local pitch-control value at the ball.
- Interpolation/extrapolation flags.

## Caveats
- The Tryolabs possession threshold and inertia (45 px, 20 frames) come only from an auto-generated DeepWiki summary of `soccer/match.py`, not the source file. The only verifiable `inertia=20` in the repo is the team-jersey classifier's. The blog confirms the possession logic but gives no numbers.
- Peral et al.'s VISAPP 2025 paper reports pass F1 67.28% (0.6 s window), not an 85.85% ball-state accuracy. If that figure comes from the multi-modal follow-up, it is unverified. Both works use tactical-camera crops you do not have.
- Accuracy figures for Link & Hoernig and Vidal-Codina come from multi-camera optical data. Mills et al. is a single broadcast match from three unnamed providers. No public benchmark exists for team-labelled, ID-free broadcast possession, so you must build a small labelled validation set: an hour of annotated play spread across several matches is a reasonable start.
- Thresholds suggested here for broadcast data (R_pz 1.5 m, R_dz 2.0 m, 0.3–0.5 s hysteresis, 0.5 s interpolation cap) are engineering starting points derived from the cited work, not published optima.
- Several repository links (e.g., socceraction, kloppy, mplsoccer, Metrica sample data) are standard community locations not re-verified in this research pass.

## Sources

1. [The Least Thing: How "Ball Possession" is Measured in Football](http://leastthing.blogspot.com/2012/02/how-ball-possession-is-measured-in.html)
2. [BLOG: Possession(s)](https://www.optasportspro.com/news-analysis/blog-possessions/)
3. [Ball possession - Wyscout Glossary](https://dataglossary.wyscout.com/ball_possession/)
4. [socceraction.data.statsbomb.StatsBombEventSchema - socceraction 1.5.3 documentation](https://socceraction.readthedocs.io/en/latest/api/generated/socceraction.data.statsbomb.StatsBombEventSchema.html)
5. [Splitting Possession into Offense/Defense • Statsbomb Blog Archive](https://blogarchive.statsbomb.com/articles/soccer/splitting-possession-into-offensedefense/)
6. [In contest - FIFA Training Centre](https://www.fifatrainingcentre.com/en/resources-tools/football-language/in-contest/in-contest.php)
7. [FIFA's in contest metric - The fight for possession - Insight and analysis - England Football Learning - The England Football Community](https://community.thefa.com/coaching/b/insights-analysis-blogs/posts/fifa-s-in-contest-metric---the-fight-for-possession)
8. [Individual ball possession in soccer - Research journals - PLOS](https://journals.plos.org/plosone/article/file?type=printable&id=10.1371%2Fjournal.pone.0179953)
9. [huggingface.co](https://huggingface.co/spaces/ixxan/football-vision-analytics/commit/443e9fb3694491f0ce37ebddbe7f14e1ad31677b)
10. [GitHub - Seronic2001/football\_analysis · GitHub](https://github.com/Seronic2001/football_analysis)
11. [GitHub - kaustubh285/football\_analysis\_yolo · GitHub](https://github.com/kaustubh285/football_analysis_yolo)
12. [\[Feat\]: Domain Analytics & Metric Space Decoupling (Metric Possession, Robust Team Clustering, Smooth Kinematics) · Issue #35 · donaldng05/football-cv · GitHub](https://github.com/donaldng05/football-cv/issues/35)
13. <https://arxiv.org/pdf/2202.00804>
14. <https://arxiv.org/pdf/2202.00804v1>
15. [(PDF) Quantifying Pitch Control](https://www.researchgate.net/publication/334849056_Quantifying_Pitch_Control)
16. [A Neighbor-based Approach to Pitch Ownership Models in Soccer](https://arxiv.org/pdf/2501.05870)
17. [GitHub - Friends-of-Tracking-Data-FoTD/LaurieOnTracking: Laurie's code for Metrica tracking data. · GitHub](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
18. [LaurieOnTracking/Metrica\_PitchControl.py at master · Friends-of-Tracking-Data-FoTD/LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_PitchControl.py)
19. [Laurie Shaw on X: "Here's a link to the Friends of Tracking lectures that cover pitch control: https://t.co/DyGuhGg64A Methodology is based on research by @the\_spearman, with some bells and whistles." / X](https://x.com/EightyFivePoint/status/1301901965669085187)
20. [Pitch-control model and player velocities by RedHot099 · Pull Request #31 · RedHot099/TactiFootVision · GitHub](https://github.com/RedHot099/TactiFootVision/pull/31)
21. [LaurieOnTracking/Tutorial3\_PitchControl.py at master · Friends-of-Tracking-Data-FoTD/LaurieOnTracking · GitHub](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Tutorial3_PitchControl.py)
22. [PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories](https://arxiv.org/html/2602.12080v2)
23. [TranSPORTmer - ACCV 2024 Open Access Repository](https://openaccess.thecvf.com/content/ACCV2024/html/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.html)
24. [TranSPORTmer: A Holistic Approach to Trajectory Understanding in Multi-Agent](http://www.iri.upc.edu/people/aagudo/Papers/ACCV/ACCV2024/gcapellera_accv24.pdf)
25. [Multi-Modal Soccer Scene Analysis with Masked Pre-Training — Lacuna](https://lacuna.tiptreesystems.com/work/multi-modal-soccer-scene-analysis-with-masked-pre-training/wrk_de6bad900ee38ac3112941602ac8b45f)
26. [unravelsports Documentation — unravelsports 1.2.0 documentation](https://unravelsports.readthedocs.io/en/latest/)
27. [unravelsports 1.2.1 on PyPI - Libraries.io - security & maintenance data for open source software](https://libraries.io/pypi/unravelsports)
28. [GitHub - USSoccerFederation/ussf\_ssac\_23\_soccer\_gnn: U.S. Soccer Federation 2023 MIT Sloan Sports Analytics Conference Graph Neural Network soccer research repository · GitHub](https://github.com/USSoccerFederation/ussf_ssac_23_soccer_gnn)
29. [Graph neural network](https://en.wikipedia.org/wiki/Graph_neural_network)
30. [Automatic event detection in association football using broadcast-derived tracking data](https://link.springer.com/article/10.1007/s12283-026-00549-4)
31. [Predicting the in-game status in soccer with machine learning using spatiotemporal player tracking data](https://www.nature.com/articles/s41598-022-19948-1)
32. [SoccerNet 2023 Challenges Results](https://arxiv.org/pdf/2309.06006)
33. [SoccerNet 2025 Challenges Results](https://arxiv.org/pdf/2508.19182)
34. [GitHub - SoccerNet/sn-teamspotting: DevKit for SoccerNet Team Action Spotting Challenge 2025 · GitHub](https://github.com/SoccerNet/sn-teamspotting)
35. [SoccerNet 2025 Challenges](https://www.emergentmind.com/topics/soccernet-2025-challenges)
36. [From Broadcast to Minimap: Achieving State-of-the-Art SoccerNet Game State Reconstruction](https://arxiv.org/pdf/2504.06357)
37. [SoccerNet - Game State Reconstruction](https://www.soccer-net.org/tasks/game-state-reconstruction)
38. [From Soccer Video to Ball Possession Statistics](https://www.researchgate.net/publication/354713260_From_Soccer_Video_to_Ball_Possession_Statistics)
39. [Estimation of Ball Possession Statistics in Soccer Video](https://dl.acm.org/doi/10.1145/3293353.3293403)
40. [From soccer video to ball possession statistics - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005185)
41. [From soccer video to ball possession statistics](https://dl.acm.org/doi/abs/10.1016/j.patcog.2021.108338)
42. [GitHub - TyBM223/soccer-possession · GitHub](https://github.com/TyBM223/soccer-possession)
43. [GitHub - roboflow/sports: computer vision and sports · GitHub](https://github.com/roboflow/sports)
44. [Automatically measuring soccer ball possession with AI and video analytics](https://tryolabs.com/blog/2022/10/17/measuring-soccer-ball-possession-ai-video-analytics)
45. [ORIGINAL PAPER Sports Engineering (2026) 29:15](https://link.springer.com/content/pdf/10.1007/s12283-026-00549-4.pdf)
46. [Intent vs. Outcome: adding another layer to expected possession value](https://thecomeonman.github.io/IntentVsOutcome/)
47. [SkillCorner tracking: keep the is\_detected flag for players and the ball · Issue #639 · PySport/kloppy · GitHub](https://github.com/PySport/kloppy/issues/639)
48. [GitHub - SkillCorner/opendata: SkillCorner Open Data with 10 matches of broadcast tracking data. · GitHub](https://github.com/SkillCorner/opendata)
49. [floodlight - A high-level, data-driven sports analytics framework Dominik Raabe](https://www.theoj.org/joss-papers/joss.04588/10.21105.joss.04588.pdf)
50. [pypi.org](https://pypi.org/project/floodlight/)
51. [Continuous football player tracking from discrete broadcast data](https://arxiv.org/pdf/2311.14642)
