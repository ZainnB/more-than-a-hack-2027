# Building a legal, lightweight per-frame possession and dead-ball pipeline for full-match football video (2023–2026 review)

The best approach is not a better ball detector. It is a sequence model (CRF/HMM-style) for possession and dead-ball state, trained on the Bundesliga IDSSE dataset, whose per-frame BallPossession and BallStatus flags almost certainly use the same TRACAB/DFL definitions as your ground truth. Each frame's evidence comes from several small, licence-clean vision modules: an RF-DETR (Apache-2.0) player detector, a heatmap ball detector, a team-colour classifier, and a restart detector. Video-side fine-tuning should use only SoccerTrack v2, which is CC BY 4.0, has full-pitch panoramic video and allows commercial use. Most other popular football vision assets (SoccerNet BAS video, DFL Kaggle data, Roboflow football datasets, ISSIA-derived ball weights, Ultralytics YOLO) carry NDA, competition-only, non-commercial or AGPL restrictions that conflict with a public repository.

## TL;DR

- **Possession logic is where most of the score is.** The ball is missing on 41% of frames, so per-frame ball detection cannot decide spells on its own. Train a constrained sequence decoder (H/A/D states, Viterbi, minimum-duration and "controlled possession" transitions) on IDSSE (7 Bundesliga matches, 25 fps, CC BY 4.0). IDSSE gives "BallPossession; 1 = home team in possession, 2 = away team in possession" and "BallStatus; 0 = ball inactive, 1 = ball active" per frame, which is essentially your label space. Make the decoder robust by corrupting IDSSE to look like your noisy input (drop ball positions on 41% of frames, flip about 7% of team labels). This tunes everything on external data only.
- **Vision stack (all permissive):** RF-DETR Nano/Small/Medium (Apache 2.0) for players and referees, with SAHI-style tiling on the wide camera, and ByteTrack/OC-SORT-style association. A WASB-style HRNet heatmap ball detector, retrained on SoccerTrack v2 plus your own small labelled set from other footage. Team colour handled by clustering anchored to the provided tracking's team labels. A tracking-only in-play/dead-ball classifier in the style of Lang et al. 2022, whose AdaBoost model reached frame-level accuracy of 0.92 and F1 of 0.93 from player positions alone (trained on manually annotated in-game status from 102 Bundesliga matches with TRACAB tracking), plus restart-formation rules after Vidal-Codina et al. 2022.
- **Licence red flags:** the SoccerNet BAS/Team-BAS video is NDA-gated. The DFL Kaggle data is "Competition-Use Only", so Roboflow football assets derived from it are risky. ISSIA-CNR is "available for non-commercial research in video signal processing only", which taints the WASB and FootAndBall soccer weights. Ultralytics YOLO is AGPL-3.0. sn-gamestate is GPL-3.0. Metrica sample data has no licence at all.

## Key Findings

1. **Your ground truth almost certainly mirrors DFL/TRACAB semantics, and IDSSE ships them.** In Bassek et al. 2025 (Scientific Data), "the ball possession is defined by the player in control of the ball" and "the match state is inactive when the match is interrupted by the referee, for example after a foul or during a substitution." Two consequences follow:
   - Dead-ball boundaries follow the referee's interruption and the restart, not the instant the ball crosses the line.
   - The possession label behaves like the challenge's "controlled possession" rule.

   IDSSE is the only open dataset with both flags at 25 fps from a validated multi-camera system. Use it to fit every transition parameter. floodlight parses both flags directly.
2. **SoccerTrack v2 is the only video dataset that is clean to use for training.** It has 10 university matches, 932 minutes of fixed 4K panoramic video, per-frame game state, and 21,428 events in 12 classes (Pass, Drive, Header, High Pass, Out, Cross, Throw In, Shot, Ball Player Block, Player Successful Tackle, Free Kick, Goal). The data is CC BY 4.0, "which permits academic and commercial use with attribution", and the code is MIT. Two caveats:
   - The `ball/` track "is a reconstruction rather than a per-frame measurement: it is anchored at the annotated events and interpolated in straight constant-speed lines". In two matches (132831 and 132877, i.e. M9 and M10) it is clamped to the pitch. **Do not use it as pixel-level ball ground truth.**
   - Bounding boxes exist only in the MMSports 2025 subset: "one four-minute clip per match with every player boxed in every frame (59,842 frames and 1.3 million boxes in total)".
3. **Ball-free possession inference is now a mature research line, and it is your fallback for the 41% of frames without a ball:**
   - BallRadar (KDD 2023): ball trajectory and possessor inferred from player trajectories.
   - PathCRF (KDD 2026): a CRF over possession edges with Viterbi decoding, trained on IDSSE. It reports 69.64% accuracy in choosing the correct possession edge among 676 candidates and 75.69% event-detection F1, from player trajectories alone.
   - Peral et al. (arXiv 2512.19528): joint ball trajectory, ball state and possessor inference on tactical-camera footage.
4. **Video event spotting is useful but secondary.** In the SoccerNet 2025 Team-BAS challenge, 16 teams made 67 submissions, and the winning team ("dudek") scored 60.03 Team-mAP@1, 8.31 points above the T-DEED baseline of 51.72 (SoccerNet 2025 Challenges Results, arXiv 2508.19182). The training video is NDA-gated, and the 7-match training set is broadcast footage, so it would face strong domain shift on your tactical camera. On SoccerTrack v2, a trajectory-only spotter reaches mAP 0.222 at 1 s and 0.524 at 5 s. "Out" (a synthesised restart marker) is a weak class at 0.364 AP even with the ball.
5. **Full game-state pipelines are too heavy and too fragile for this job.** The SoccerTrack v2 GSR baseline needs about 37 GPU-hours per half. Its full-half GS-HOTA averages 12.25, against 31.35 on 30-second clips. On the best test half it produced 345 tracklets for 23 identities, and team accuracy on 36,665 matched pairs was 76.17%. Don't chase identity. Detect per frame, associate briefly, and reason at team level.

## Details by module

### 1. Ball detection and tracking

| Option | What it gives | Licence / terms | Trained on | Notes for your use |
|---|---|---|---|---|
| **WASB-SBDT** (nttcom, BMVC 2023) | HRNet-style multi-frame heatmap detector plus online tracker; strongest published soccer SBDT baseline | Code MIT. **Weights and annotations have no stated licence**: open issue #12 asks exactly this and has no reply | Soccer data is ISSIA-CNR (GET_STARTED links the ISSIA download) | Use the **architecture/code**, retrain yourself. Don't ship the soccer checkpoint: ISSIA is "available for non-commercial research in video signal processing only" |
| **FootAndBall** (jac99) | Single light FCN for players and ball | Code MIT | "ISSIA-CNR dataset (cameras 1,2,3,4) and SoccerPlayerDetection dataset (set 1)" | Same ISSIA taint. Players in ISSIA are 63–144 px tall, versus 25–50 px in your footage, so the scale gap is large |
| **TrackNetV2/V3, DeepBall, TOTNet (2025)** | Heatmap trackers, some occlusion-aware | Check each repo's licence (not verified here) | Mostly tennis/badminton | Architectures only. No usable soccer weights |
| **RF-DETR + SAHI tiling** | Box detector for ball as a tiny object | Apache 2.0 (N/S/M/L) | COCO | A 3–6 px ball is below the reliable range of DETR-style detectors even with tiling; one 2026 study notes accuracy "drops significantly" under 10 px. Use RF-DETR for players, not the ball |
| **Soccana / Roboflow Universe sets** | YOLO ball/player data | Mixed. Soccana states it includes SoccerNet-derived imagery | — | **Avoid**: provenance from SoccerNet/NDA or Kaggle data |

**Recommendation.** Retrain a WASB-style 3-frame heatmap network (code MIT) from scratch or from ImageNet-free initialisation. Use crops from SoccerTrack v2 panoramic video, with ball labels that you produce yourself on that footage. Labelling external data is allowed; labelling the provided game is not. The annotated SoccerTrack v2 events give approximate ball anchors to bootstrap labelling. Run it on a downscaled tactical frame plus a high-resolution ROI tracker around the last known position. A WASB-sized network at 512–768 px input should process 199k frames in a few hours on MPS, but this is an estimate: benchmark it first. Add a kick/touch detector from trajectory kinks (speed and direction change near a player) as a feature for the possession decoder.

### 2. Player, goalkeeper and referee detection plus MOT

- **RF-DETR Nano/Small/Medium** (Roboflow, ICLR 2026). "RF-DETR Nano, Small, Medium and Large detection … are released under the Apache 2.0 license." Atto, Femto, Pico, XL and 2XL are under PML 1.0, so avoid those. COCO-pretrained; fine-tune on the SoccerTrack v2 MOT subset (players/referees, panoramic). Tile the 1080p wide view into about 4 overlapping 640-px tiles, because 25–50 px players are small.
- **Trackers.** ByteTrack and OC-SORT are motion-only, light and CPU-cheap; check that each repo's licence is permissive. You only need short tracklets, since possession is team-level.
- **Avoid:** Ultralytics YOLOv8/11 (AGPL-3.0), which is also the default `yolo_ultralytics` detector in sn-gamestate. sn-gamestate itself is GPL-3.0; TrackLab is MIT. GPL code in a public repo is allowed but makes your repo GPL. AGPL weights add network-use obligations.

### 3. Team affiliation and role

- **Options:** jersey colour histograms with k-means; SigLIP/CLIP embeddings with UMAP and k-means (the Roboflow `sports` code is MIT, but its data comes from DFL Kaggle); PRTreID (trained on SoccerNet-GSR, whose 2025 HF card says gpl-3.0).
- **Rules risk.** Per-match unsupervised clustering on the provided video is *test-time adaptation without labels*. It plausibly counts as "using the video as input". But "fitting … on the provided video" is ambiguous, and a strict reader could object.
- **Safer designs**, in order:
  1. Have an external-data-trained Siamese "same-team?" embedding make pairwise decisions. Name the two groups by majority vote against the provided tracking's team attribution (about 93% correct), which is an allowed input. No parameters are fit on the video.
  2. Use a fixed, deterministic colour rule with hyperparameters tuned only on SoccerTrack v2.

  Document the choice and ask the organisers. Referees and goalkeepers: train a role classifier on SoccerTrack v2 roles.

### 4. Event and ball-action spotting from video

- **T-DEED, E2E-Spot, and the 2025 Team-BAS winner** (int8.io re-implementation in the "dude.k" framework, using a joint team-action head). Training needs SN-BAS-2024/2025, which are "NDA-gated". The SoccerNet README says "Access is granted separately for each repository". A Kaggle host ruled in a related competition that "We define the SoccerNet dataset as not being fully public data, as an NDA is required". **Treat SoccerNet BAS video, and any weights trained on it, as unsafe for a public, rules-audited repo** unless the organisers explicitly allow it.
- **Clean alternative:** train a small temporal CNN spotter on SoccerTrack v2 BAS labels (Out, Throw In, Free Kick, Goal and so on), using player-tracking features plus low-resolution frame features. The SoccerTrack v2 baseline already shows trajectory-only spotting is feasible.
- **Broadcast feed:** a cheap shot-type/replay classifier (wide versus close-up versus replay) helps flag stoppages, because close-ups and replays cluster in dead time. SoccerNet camera-shot labels are NDA-adjacent, so build a light classifier from features you derive yourself, or use the shot-change heuristics as features only.

### 5. Dead-ball / in-play detection

- **Lang, Wild, Isenko, Link 2022** (Sci. Rep. 12:16291): AdaBoost reached frame-level accuracy of 0.92 and F1 of 0.93 on previously unseen matches, using models trained on manually annotated in-game status from 102 Bundesliga matches with TRACAB tracking. It also reaches 81% precision on stoppages longer than 2 s, using **player positions only**. No public code was found. Reimplement the features (team speeds, spread, distance to lines, stationary players near the boundary) and train on IDSSE BallStatus.
- **Vidal-Codina et al. 2022** (Sports Eng. 25:18) uses "the distinct player configurations during the time intervals where the ball is not in play to inform set piece detection". It is deterministic and has few parameters, so you can tune them on IDSSE. Mills et al. 2026 ("Automatic event detection in association football using broadcast-derived tracking data", Sports Engineering 29:15) applied it to broadcast-derived tracking of one 2022 World Cup group-stage match and "employed default hyperparameters governing indicators of possession, and set pieces used by Vidal-Codina et al."
- **Fusion.** Combine the tracking-only P(dead), the ball-out-of-bounds evidence from homography, the restart detectors (a player standing on the touchline gives a throw-in; players frozen at the corner arc or in a wall give a set piece), Smart Tagging, and the broadcast shot type. Feed all of them to the decoder as emissions.

### 6. Possession logic (the core)

- **State machine.** States are H, A and D. A switch H→A requires sustained control evidence for team A, such as an A player nearest the ball for k frames with low relative speed, or a pass reception. Loose or contested ball keeps the last team. D→H/A is decided by the restart taker's team.
- **Learning.** Fit emission models (gradient boosting or a small GNN/Set-Transformer) and transition and minimum-duration parameters on IDSSE, which is the closest match to your label semantics. Validate on held-out IDSSE matches, and on SoccerTrack v2 for video-derived inputs. PathCRF (IDSSE-trained, open code; check its licence) and BallRadar are ready architectures for frames without a ball.
- **Tools:** floodlight (reads DFL/TRACAB possession and ball-status codes, IDSSE loader), kloppy (PFF, SkillCorner, Metrica loaders), databallpy and ETSY (event–tracking synchronisation).
- **Output policy.** Use X sparingly. spellF1 rewards spell structure, so abstaining inside a spell may split it. Abstain only on long low-confidence stretches. Test this with a simulation on IDSSE.

### 7. Tracking and event datasets with ball status and possession

| Dataset | Content | Licence / terms | Use |
|---|---|---|---|
| **IDSSE / Bassek et al. 2025** | 7 matches (2 Bundesliga, 5 2. Bundesliga, 2022/23), TRACAB 25 Hz, BallPossession and BallStatus, events | CC BY 4.0 "with the authorization from the DFL"; no video | **Primary training and tuning set** |
| **SkillCorner opendata** | A-League 2024/25 broadcast tracking at 10 fps, `possession` {player_id, group} | MIT | Possession-dynamics pretraining. The SkillCorner/opendata GitHub README says the repo "contains data on 10 matches of broadcast tracking data… a sample of 2024/2025 league matches in the Australian A-League"; the "20 matches" figure seen elsewhere comes from a third-party repo, not SkillCorner |
| **PFF FC WC 2022** | 64 matches, broadcast tracking (29.97 fps) plus events | Free on sign-up: the PFF FC blog says it "integrates broadcast tracking data with event data for all 64 games", and PFF FC (now rebranded Gradient Sports) offers an "enhanced" version; formal licence terms not verified | Usable only if the request terms allow public-repo derivative models; read them before use |
| **Metrica sample data** | 3 anonymised matches | **No licence**: "Please be responsible with the use of this data… acknowledge the source" | Prototyping only; don't train shipped components |
| **StatsBomb open data / 360** | Events plus freeze-frames | Custom attribution licence | Restart and event priors only |

### 8. Video datasets

| Dataset | Verdict |
|---|---|
| **SoccerTrack v2** (HF `atomscott/soccertrack-v2`, gated click-through, 145 GB, CC BY 4.0) | **Use.** Panoramic, closest to your tactical camera. Split: train M1, M3–M6, M8; val M2, M10; test M7, M9 |
| **TeamTrack / SoccerTrack v1** | Full-pitch fisheye (8K, 30 fps) and drone soccer clips with boxes. Probably usable (same group), but the licence was not confirmed here; check the Kaggle page |
| **SoccerNet-GSR 2024/2025** | Listed as public; the SN-GSR-2025 HF card says gpl-3.0. Broadcast 30 s clips. Usable with care; GPL on data is unusual, so get confirmation |
| **SoccerNet-Tracking 2023** | Labels "public", but raw video is in a separate gated repo with licence "other". Treat as restricted |
| **SoccerNet BAS/Team-BAS/PCBAS, SoccerNet v2 videos** | **Do not use** (NDA) |
| **DFL Bundesliga Data Shootout (Kaggle)** | **Do not use**. Rules: "DATA ACCESS AND USE: Competition-Use Only… only for participating in the Competition" |
| **ISSIA-CNR** | **Do not use** for anything shipped (non-commercial research only) |

### 9. Pose estimation

At 25–50 px player height, RTMPose/ViTPose keypoints are unreliable and expensive across 22 players × 199k frames. Skip pose. Box geometry, motion and colour carry most of the signal. If you need it, use only box aspect ratio and lower-body motion as kick cues.

## Recommendations: concrete stack and order of work

1. **Decoder first (largest gain).** Build the H/A/D CRF/HMM on IDSSE with noise augmentation that mimics your input. Inputs are the provided tracking, Smart Tagging and homography. This alone should be competitive and is entirely legal.
2. **Dead-ball classifier.** Lang-style tracking features plus Vidal-Codina restart rules, trained on IDSSE BallStatus.
3. **Ball module.** A WASB-architecture heatmap net retrained on SoccerTrack v2 panoramic crops, plus kink-based touch detection, used as decoder emissions where the ball is visible.
4. **Players and team.** RF-DETR-S with tiling and ByteTrack/OC-SORT, to cover gaps in the provided tracking. The team step uses an externally trained same-team embedding and is named via the provided team labels.
5. **Optional.** A SoccerTrack v2-trained restart spotter (Out/Throw In/Free Kick/Goal) on the tactical feed, and a broadcast shot-type cue.
6. **Domain-shift mitigation.** Train vision modules on panoramic SoccerTrack v2, not broadcast. Augment with downscaling (target 25–50 px players, 3–6 px ball), blur, compression and colour jitter. Tune every threshold on SoccerTrack v2 or IDSSE only.

## Licence/rules red flags and "do not use" list

- **Do not use:** SoccerNet NDA videos and any T-DEED or Team-BAS checkpoints trained on them; DFL Kaggle data and derived Roboflow football datasets and weights; ISSIA-trained WASB and FootAndBall weights; Ultralytics YOLO (AGPL) for anything shipped; RF-DETR XL/2XL/Atto/Femto/Pico (PML); Metrica for shipped models; Soccana/Universe sets with SoccerNet provenance.
- **Use with care:** sn-gamestate (GPL-3.0, so your repo becomes GPL); SoccerNet-GSR (gpl-3.0 label on data); PFF (read the request terms); WASB/PathCRF/BallRadar code (confirm repo licences).
- **Rules:** never label, tune, or calibrate on the provided match. Per-match clustering is a grey zone, so anchor it to the provided labels or use external-trained pairwise models, and record the decision in the README.

## Caveats

- Throughput figures are estimates, not benchmarks. Profile RF-DETR-S tiled and a WASB-sized net on an M5 with MPS before committing.
- The SoccerTrack v2 GSR scores were produced with configuration "set by us on released footage" that includes a test half, so they are not clean transfer numbers.
- IDSSE has only 7 matches; use match-level cross-validation. Elite Bundesliga dynamics may differ from your match.
- Several licences (TeamTrack, TrackNetV3, ByteTrack/OC-SORT, PathCRF, BallRadar, PFF terms, the SoccerNet NDA text) were not verified from primary text in this review. Check each LICENSE file before committing.