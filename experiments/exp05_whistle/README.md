# Experiment 5 — the referee's whistle (plan, not yet built)

Written for: the team. Cross-experiment lessons are in [../LEARNINGS.md](../LEARNINGS.md).

## The problem, as the video showed it

Watching blind window 2 against exp04's claims, the biggest gap was the referee. Play stops
when he whistles and signals with his hand, and nothing in our inputs records either one. The
referee is not in the tracking: the ATD has only the two teams' slots (24 + 22 in tactical).

The official labels on the public window show what this costs
([validation/public_sidebyside.txt](../../validation/public_sidebyside.txt)). Wherever the ball goes
out, our dead ball is on time. Wherever there is nothing but a whistle, we are late or miss the
stoppage entirely:

| official D starts | exp04 goes dead | gap | cause |
|---|---|---|---|
| 08:34.04 | never | **missed, 5.5 s** | whistle only |
| 10:35.56 | 10:40.64 | **+5.1 s** | whistle only |
| 11:35.20 | 11:38.24 | **+3.0 s** | whistle only |
| 12:01.08 | 12:01.00 | on time | ball out of bounds |

The mechanism: the `slow` rule reads the players' reaction to the whistle, and it needs
`SLOW_PITCH_MIN_FRAMES = 200` (8 s) of a slow pitch before it fires. So it is late by
construction. The convention point from earlier video review matters here too: a linesman's flag
does not stop play. Possession continues until the whistle, so the whistle is the exact event we
need, and a hand signal is only its visual echo.

## Options, in the order to try them

### 1. The whistle on the tactical video's own audio (first)

`BAR-ATH_tactical-001.mp4` carries an audio track, checked by parsing the mp4 boxes:

```
soun  mp4a  48000 Hz  7972.7 s   373,720 samples
vide  avc1  25 fps    7972.8 s   199,319 frames   <- the submission timeline
```

That track is on the same timeline as the submission, which removes the two hard problems a
broadcast route would have:

- **no alignment.** Frame *i* of the submission is time *i*/25 s of this audio.
- **no replays.** The tactical feed is one continuous camera. The broadcast has cuts, close-ups and
  replays.

A referee's whistle is a loud, narrow tone at about 2–4 kHz with harmonics, in bursts of
0.2–1 s. Once the audio is decoded, the detector is numpy only:

1. a short-time FFT, with hops of 40 ms so there is one value per video frame
2. energy in the 2–4.5 kHz band relative to the whole spectrum
3. tonality: a sharp spectral peak with low spectral flatness inside that band
4. a whistle is a run of frames that passes both tests and lasts at least about 0.15 s

How it would be used:

- **dead-ball start.** Backdate a `slow` or `lost` stoppage to the last whistle in the preceding
  few seconds. Promote a stretch the model holds as live to dead when a whistle is followed by the
  players slowing down.
- **restarts.** A whistle often comes before a free kick or a kickoff. This would be weak
  evidence, used for timing only.

Calibration is free and honest. The official labels' D starts are presumably dated to the
whistle, so the public window's three whistle-only stoppages are a ground truth for the
detector's timing that involves no hand annotation of ours.

**Go / no-go, before building anything:** decode about 10 s of audio around 08:34, 10:35 and
11:35, and look at the spectrogram. If the tactical camera's microphone does not pick up the
whistle over the crowd, option 1 is dead and we move to option 2.

Risks:

- **Crowd whistling.** Spanish crowds whistle in protest. That sound comes from thousands of
  sources at once, so it is broad and noisy rather than one clean tone. It should separate on
  tonality, but this has to be measured, not assumed.
- **Stadium music or the PA during stoppages.** This only matters inside dead stretches, so it is
  mostly harmless.
- **Dependency.** Decoding AAC needs PyAV (`pip install av`, which bundles ffmpeg). RULES.md allows
  any public tool as long as it is reproducible and documented. The "numpy + pandas only" rule for
  the submission is *our own* convention, so adopting this means consciously relaxing it and
  saying so in the technical document. Derived audio features must never be committed (same terms
  as the match data), so the pipeline has to decode at run time.

### 2. The players' reaction to the whistle (fallback, no new data)

When the whistle goes, many players slow down sharply within about a second. Detect the "knee",
the onset of a sharp drop in mean player speed across the pitch, and backdate each `slow`
stoppage to it minus a reaction time of about 0.5 s, instead of waiting 8 s. Tracking only, no
dependencies, and it can be measured right now against the same three stoppages.

This is the fallback if the audio is silent, and probably worth having as well even if it is
not. The two signals fail in different situations.

### 3. Referee gestures from the broadcast (parked)

This needs referee detection, pose estimation, and aligning the broadcast to the tactical
timeline (164,487 frames against 199,319, with cuts and replays). Aligning would be feasible by
matching the broadcast's own ATD against the tactical ATD, since both are provided, and replays
would show up as stretches that fail to match. That is weeks of work, and the whistle already
carries the timing. The one thing a gesture adds is the advantage signal, where play visibly
continues after a foul, and that is rare. Revisit only if 1 and 2 are both exhausted.

## Measurement

- **Primary:** the start-time error of D on the public window's whistle-only stoppages, against
  the official labels. Today: missed, +5.1 s, +3.0 s.
- **Then:** public and pooled spellF1 and macroF1, with the usual caveat that the hand-labelled
  windows may encode our own convention bias. Use the official public window and blind windows
  (`validation/blind.py`) as the unbiased checks.
- **Full-match sanity:** how many whistles are detected (a match has roughly 80–120 stoppages that
  start with a whistle, counting fouls, offsides and the ends of halves), and the fraction of
  existing `slow` stoppages with a whistle in the 5 s before them.
