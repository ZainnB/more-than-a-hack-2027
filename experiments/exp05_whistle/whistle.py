"""Find the referee's whistle in the tactical video's own audio track.

    python experiments/exp05_whistle/whistle.py            # decode, detect, cache, summarise

Why the tactical audio
----------------------
The tactical mp4 carries 48 kHz stereo AAC starting at t = 0, the same timeline as the video and
so as the submission: frame i is t = i / 25 s. No alignment, and no replays, since the tactical
feed is one continuous camera. Checked by eye on spectrograms of the public window: the kickoff
whistle sits about 1.1 s before the ball is struck, and the two whistle-only stoppages there have
their whistle within half a second of the official dead-ball start. The two stoppages without a
whistle are a corner and a ball out of play, which referees do not whistle.

What a whistle is
-----------------
A pea whistle is a loud, narrow tone around 2-5 kHz with harmonics, held for a fraction of a
second to a couple of seconds. On a spectrogram it is a stack of flat parallel lines, where crowd
noise and chants are broadband or curved. So a whistle frame is one where, in the 2-6 kHz band:

  - the strongest bin stands at least CREST_DB above the band's median (it is a tone, not noise)
  - its frequency moves less than MAX_GLIDE_HZ from the previous frame (flat, not a chant)

and a whistle is a run of such frames lasting at least MIN_DURATION_S, with breaks up to
MAX_BREAK_S bridged (the pea makes the tone warble). Every threshold is set from that physical
description, not fitted: no labels and no hand-marked whistles were used to choose them.

Imports PyAV only. Do not import opencv in the same process: both bundle their own FFmpeg
libraries and macOS warns that mixing them can crash.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)

# Params
SAMPLE_RATE = 48000
FFT_SIZE = 2048            # 42.7 ms, about 23 Hz per bin
HOP = 480                  # 10 ms, so 100 audio frames per second
# Searched band. Pea whistles sound around 3-4.5 kHz. A first pass over 2-6 kHz found 5957
# "whistles", 90% of them at 2.0-2.5 kHz and seconds long: this recording carries a continuous
# tonal background (chants, music) below 3 kHz that out-shouts the whistle in a wider band. That
# was read off the detections themselves, with no labels involved.
BAND_HZ = (3000.0, 6000.0)
LOW_BAND_HZ = (2000.0, 3000.0)   # kept in the cache for diagnosis only
CREST_DB = 20.0            # peak at least 10x the band median
MAX_GLIDE_HZ = 120.0       # about 5 bins between frames
MIN_DURATION_S = 0.15
MAX_DURATION_S = 3.0       # the long whistle at the end of a half is about this; a tone that
                           # lasts longer is background, not a whistle
MAX_BREAK_S = 0.05
ONSET_DB = 6.0             # a whistle starts abruptly: the band is this much louder than its
ONSET_LOOKBACK_S = 2.0     # median over the preceding two seconds
# Each frequency bin's own background, tracked as an exponential average of its level in dB over
# about BACKGROUND_S seconds. The second pass found the 3-6 kHz band carries a steady tone near
# 3.3 kHz (95% of frames around the 08:34 stoppage were "tonal" against the band median, and 3.3
# kHz was the median pitch of all detections), so a whistle has to be *new* narrowband energy, not
# merely a peak. A constant tone becomes its own background and vanishes.
BACKGROUND_S = 5.0
EXCESS_DB = 12.0           # a whistle stands this far above its bin's recent background
# A pea whistle's warble puts sidebands either side of the tone, so on a spectrogram it is a stack
# of two or three parallel lines. Crowd whistling is one line, usually curving. So a whistle frame
# needs at least two separate peaks above background, STACK_GAP_HZ apart.
STACK_GAP_HZ = (150.0, 700.0)
# A referee's whistle is one deliberate blast of half a second or more; the crowd's are chirps,
# often in rapid bursts. Both rules came from inspecting the detections on the public window,
# where the four referee blasts ran 1.0-2.5 s and the false ones were 0.3-0.4 s chirps or came in
# bursts. That makes the public window a biased check of this detector from here on.
MIN_EVENT_S = 0.5
# The referee's whistle is the loudest tonal sound in the stadium. The peak excess of the 137
# events that pass every test above runs 18-48 dB, bulk 22-34, and the count falls from 9 in the
# 34-36 dB bin to 3 in the next, leaving a loud tail of about 20. The cut was first set at that
# knee, 35 dB, read off the distribution with no labels.
#
# It was then raised to 36 dB, and that step WAS fitted to the match: a 35.46 dB event at 13:47 in
# hand window win1 started a false stoppage, and the team confirmed on video it is not a whistle.
# The margin is thin -- the real second-half kickoff whistle is 35.70 dB and is dropped too, which
# costs nothing here because kickoffs are found from the ball. Treat 36 as tuned on win1.
LOUD_DB = 36.0
CHORUS_S = 4.0             # a blast counts only if no other tone of MIN_EVENT_S or more began
                           # in the CHORUS_S before it. Both kinds of sound are surrounded by
                           # tiny fragments, so counting events of any length removed the real
                           # blasts too; what differs is that a referee's blast follows seconds
                           # with no sustained tone, while a crowd's comes inside a chorus.
VIDEO_FPS = 25

CACHE = os.path.join(HERE, "cache", "whistle_{}.npz")


def tactical_video(game="BAR-ATH", data_dir=None):
    import glob
    folder = data_dir or os.path.join(ROOT, "data", "tactical")
    return sorted(glob.glob(os.path.join(folder, "{}_tactical*.mp4".format(game))))[0]


def frame_features(path, chunk_s=60.0):
    """Per 10 ms audio frame: peak frequency, crest and energy in the whistle band, plus the
    same for the band below it (diagnosis), over the whole track.

    Decoded audio is collected in a list and joined once per chunk. Growing one array frame by
    frame copies it every time, and made the first full pass take 19 minutes.
    """
    import av
    container = av.open(path)
    stream = container.streams.audio[0]
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)
    bands = {"": (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1]),
             "low_": (freqs >= LOW_BAND_HZ[0]) & (freqs < LOW_BAND_HZ[1])}
    window = np.hanning(FFT_SIZE).astype(np.float32)
    out = {k + f: [] for k in bands for f in ("peak", "crest", "energy")}
    out["excess"], out["excess_hz"], out["stack"] = [], [], []
    alpha = 1.0 / (BACKGROUND_S * 100.0)
    state = {"bg": None}
    band_freqs = freqs[bands[""]]

    def process(samples):
        n = (len(samples) - FFT_SIZE) // HOP + 1
        if n <= 0:
            return samples
        idx = np.arange(FFT_SIZE)[None, :] + HOP * np.arange(n)[:, None]
        spectra = np.abs(np.fft.rfft(samples[idx] * window, axis=1))
        for key, mask in bands.items():
            sub = spectra[:, mask]
            top = sub.argmax(axis=1)
            p = sub[np.arange(n), top]
            m = np.median(sub, axis=1) + 1e-9
            out[key + "peak"].append(freqs[mask][top].astype(np.float32))
            out[key + "crest"].append((20.0 * np.log10(p / m + 1e-9)).astype(np.float32))
            out[key + "energy"].append((10.0 * np.log10((sub ** 2).sum(axis=1) + 1e-12))
                                       .astype(np.float32))
        level = 20.0 * np.log10(spectra[:, bands[""]] + 1e-9)
        excess = np.empty(n, dtype=np.float32)
        where = np.empty(n, dtype=np.float32)
        stack = np.zeros(n, dtype=bool)
        bin_hz = band_freqs[1] - band_freqs[0]
        lo_gap, hi_gap = int(STACK_GAP_HZ[0] / bin_hz), int(STACK_GAP_HZ[1] / bin_hz)
        bg = level[0].copy() if state["bg"] is None else state["bg"]
        for i in range(n):  # causal: each frame is compared with the background before it
            e = level[i] - bg
            k = int(e.argmax())
            excess[i], where[i] = e[k], band_freqs[k]
            if e[k] >= EXCESS_DB:
                # a second peak above background, a sideband's distance from the first
                side = np.concatenate([e[max(0, k - hi_gap):max(0, k - lo_gap + 1)],
                                       e[k + lo_gap:k + hi_gap + 1]])
                stack[i] = side.size > 0 and side.max() >= EXCESS_DB
            bg += alpha * (level[i] - bg)
        state["bg"] = bg
        out["excess"].append(excess)
        out["stack"].append(stack)
        out["excess_hz"].append(where)
        return samples[n * HOP:]

    pending, total, carry = [], 0, np.zeros(0, dtype=np.float32)
    chunk = int(chunk_s * SAMPLE_RATE)
    for frame in container.decode(stream):
        x = frame.to_ndarray()
        x = x.mean(axis=0) if x.ndim == 2 else x
        pending.append(x.astype(np.float32))
        total += len(x)
        if total >= chunk:
            carry = process(np.concatenate([carry] + pending))
            pending, total = [], 0
    process(np.concatenate([carry] + pending))
    return {k: np.concatenate(v) for k, v in out.items()}


def detect(f):
    """[(start_s, end_s, median_hz, max_excess_db)] whistle events from frame_features().

    The band-energy onset test of the second pass is gone: the per-bin background does that job
    better, since a whistle over a loud steady tone raises the band energy very little.
    """
    crest, peak, energy = f["crest"], f["excess_hz"], f["energy"]
    tonal = (f["excess"] >= EXCESS_DB) & (crest >= CREST_DB) & f["stack"].astype(bool)
    steady = np.zeros_like(tonal)
    steady[1:] = np.abs(np.diff(peak)) <= MAX_GLIDE_HZ
    hit = tonal & steady
    gap = int(round(MAX_BREAK_S * 100))
    filled = hit.copy()
    idx = np.flatnonzero(hit)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            filled[a:b] = True
    edges = np.diff(np.concatenate(([0], filled.astype(np.int8), [0])))
    look = int(ONSET_LOOKBACK_S * 100)
    events = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = (b - a) / 100.0
        if not MIN_EVENT_S <= duration <= MAX_DURATION_S:
            continue
        events.append((a / 100.0, b / 100.0, float(np.median(peak[a:b])),
                       float(f["excess"][a:b].max())))
    # Judged against every sustained tone, not only the ones that became events, so a chorus
    # that the background test let through partly still counts against what follows it.
    sustained = np.array([a / 100.0 for a, b in _runs(filled) if (b - a) / 100.0 >= MIN_EVENT_S])
    kept = []
    for e in events:
        earlier = sustained[(sustained < e[0] - 1e-6) & (sustained >= e[0] - CHORUS_S)]
        if not earlier.size and e[3] >= LOUD_DB:
            kept.append(e)
    return kept


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def to_video_frames(events, n_frames):
    """Boolean per video frame: True while a whistle sounds."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, end, _hz, _db in events:
        mask[int(start * VIDEO_FPS):int(np.ceil(end * VIDEO_FPS)) + 1] = True
    return mask


def load(game="BAR-ATH", data_dir=None, rebuild=False):
    path = CACHE.format(game)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as h:
            features = {k: h[k] for k in h.files}
        # detection is cheap and its thresholds may change, so it is never cached
        features["events"] = np.array(detect(features), dtype=np.float64).reshape(-1, 4)
        return features
    features = frame_features(tactical_video(game, data_dir))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **features)
    features["events"] = np.array(detect(features), dtype=np.float64).reshape(-1, 4)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    started = time.time()
    w = load(rebuild=args.rebuild)
    ev = w["events"]
    print("{} audio frames ({:.0f} min) in {:.0f}s; {} whistles".format(
        len(w["crest"]), len(w["crest"]) / 6000.0, time.time() - started, len(ev)))
    if len(ev):
        dur = ev[:, 1] - ev[:, 0]
        print("  duration median {:.2f}s  p90 {:.2f}s   pitch median {:.0f} Hz".format(
            np.median(dur), np.percentile(dur, 90), np.median(ev[:, 2])))
