# Getting this repo running

Written for: teammates setting the project up for the first time. This archive is the code,
documentation and hand annotation. The bulky inputs are not in it and have to be fetched; none of
them may be committed or redistributed (see [RULES.md](RULES.md)).

## 1. What is missing and where it goes

### The match bundle (gated — get it from the organisers, do not re-share)

```
data/tactical/BAR-ATH_tactical-001.mp4          5.2 GB   the submission's frame index
data/tactical/BAR-ATH_tactical_atd.txt          196 MB   the tracking
data/tactical/BAR-ATH_tactical_atd.xml           59 KB   track slots -> teams
data/tactical/BAR-ATH_tactical_homography.csv    23 MB   pitch <-> pixels, per frame
data/tactical/BAR-ATH_tactical_smart_tagging.xml 261 KB
data/broadcast/...                                       optional; nothing here uses it yet
```

The pipeline needs only the two `atd` files plus the homography. The video is needed for the
shirt-colour and whistle work (exp07, exp08) and for watching clips.

If you keep the bundle somewhere else, point `MTAH_DATA_DIR` at it instead.

### IDSSE, the external tuning set (CC BY 4.0, free)

Everything from exp06 on is tuned on this, never on the match. 7 Bundesliga matches, 2.6 GB:

```
https://doi.org/10.6084/m9.figshare.28196177
```

Download all 21 files into `data/external/idsse/`. The positions files are ~380 MB each; the
figshare mirror `ndownloader.figshare.com` dropped out mid-download for us, so verify sizes
against the API listing and retry rather than trusting a finished-looking run.

## 2. Setting up

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt              # numpy, pandas: the scorer and the submission
pip install -r requirements-validation.txt   # opencv: anything that opens the video
pip install av                               # PyAV: the whistle detector only
```

**Never import opencv and PyAV in the same process.** They bundle different FFmpeg builds and
macOS warns that mixing them can crash. exp07 (video) and exp08's `whistle.py` (audio) are
deliberately separate scripts for this reason.

## 3. Checking it works

```bash
python experiments/exp08_deadball_clf/run.py          # the current best -> evaluation/results/exp08.csv
python evaluation/validate_submission.py evaluation/results/exp08.csv
python validation/compare.py                          # every submission on every window
```

First runs build caches under `experiments/*/cache/` (git-ignored, derived from the match data, so
never commit them). The colour pass in exp07 reads the whole video and takes about 20 minutes;
everything else is seconds to a couple of minutes.

## 4. Where to start reading

1. [README.md](README.md), [CHALLENGE.md](CHALLENGE.md), [RULES.md](RULES.md) — the task and its limits.
2. [experiments/LEARNINGS.md](experiments/LEARNINGS.md) — the scoreboard and every lesson so far.
   The fastest way in.
3. [experiments/PLAN.md](experiments/PLAN.md) — the working policy. The one hard rule: **no value
   is ever chosen because it scored better on the provided match.** Tuning happens on IDSSE; the
   match is scored only to report.
4. [experiments/exp08_deadball_clf/README.md](experiments/exp08_deadball_clf/README.md) — the
   current best submission.
5. [validation/windows_sidebyside.txt](validation/windows_sidebyside.txt) — our labels against the
   model, every window, with a summary of where the remaining gap is.

## 5. A note on `.git`

At the time this archive was made the local `.git` held an 18 GB leftover temp pack from an
interrupted operation (`git count-objects -vH` reported it as garbage). The real history is about
1.5 MB. It is excluded from this archive; if you see the same thing locally, `git gc --prune=now`
clears it.
