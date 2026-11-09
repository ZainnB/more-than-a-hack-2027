# Experiment 7 — checking near-ball team labels against shirt colour

Written for: the team. Plan: [../TRACKING_PLAN.md](../TRACKING_PLAN.md) step 1.

## Why

On degraded IDSSE, fixing team identity within 1.5 m of the ball was worth +0.057 spellF1, as
much as fixing it everywhere ([../exp06_idsse/README.md](../exp06_idsse/README.md)). The kits are
very different (Barcelona dark and saturated, Athletic white or light lilac), and the provided
homography maps the ATD's 0-1 coordinates straight to pixels, so each tracked player can be found
in the frame.

## How (nothing fitted to the video)

| file | does |
|---|---|
| `players.py` | every ATD slot's position on every frame, cached |
| `colour.py` | one pass over the video (20 min): for every tracked player on the 61,555 frames where someone is within 3 m of the ball, a torso crop sized from the local pixels-per-metre, grass masked, described by chroma only (saturation, Lab a/b, share of white). Brightness is left out because half the pitch is in shadow |
| `decide.py` | same-frame colour consensus, below (removed 2026-11-09; in git at commit bc6de5d) |
| `run.py` | exp06's pipeline on features with the corrected labels (removed with it) |

`players.py` stays: exp08's dead-ball classifier reads player positions from it. `decide.py` and
`run.py` were removed in the clean-up after exp08 because the step changed no score; the finding
below is what it is kept for.

`decide.py`, per frame:
- The other players form two colour groups (a 2-group split of that frame only). Each group is
  named by which team label is more common in it.
- A near-ball player is flipped only if at least twice as close to the other group **and**
  within 0.35 of the separation from it: a typical member, not a mix of two shirts.
- Players whose torso overlaps another player's box in the image are not judged.
- Per-track majority over ±1 s.

Three versions, each fixed after looking at random samples of its decisions:
1. **Per-team medians** as references broke where the ATD labels 15 players home and 6 away (early
   second half). Real Barcelona players looked like neither team and were dropped.
2. **Dropping ghosts and non-players**: 3 of 4 sampled drops were real players (a close camera
   zoom, a player lying down). Drops are now counted, never applied.
3. **Duels**: overlapping players made the crop catch the neighbour. Now abstained.

## Result

| | near-ball checks | overlap, abstained | flip | after smoothing |
|---|---|---|---|---|
| final | 71,870 | 15.5% | 0.87% | 423 player-frames (0.6%) |

On the match: 483 frames (0.3%) differ from exp06, one possession fewer, **every window's score
identical**. A control run with the colour patches switched off reproduces exp06 exactly.

## What it actually found

When a near-ball player can be seen clearly, colour disagrees with the ATD's team label about
**1%** of the time. The simulator assumed 7%. That figure was nearest-player accuracy against the
labels, which mixes two errors: the nearest player not being the possessor, and a wrong team label.
The wrong-label part is small. The rest of the near-ball difficulty sits in the 15.5% of checks
where players overlap in duels, and tight shirt crops cannot settle those.

So:
- The +0.057 "team identity" lever was overestimated. `noise.TEAM_FLIP_RATE` should come down to
  about 1-2% and exp06 should be re-tuned at that rate.
- What remains near the ball is deciding who has it in a duel, which is a possession question, not
  an identity one.

Derived video data stays in `cache/` (git-ignored). opencv and PyAV bundle different FFmpeg
builds, so this experiment and exp05 must not import both in one process.
