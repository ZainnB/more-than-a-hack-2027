"""Shirt-colour descriptors for every tracked player, on the frames where the team call matters.

    python experiments/exp07_team_colour/colour.py          # one pass over the video, ~10 min

Why
---
On degraded IDSSE, fixing team identity within 1.5 m of the ball is worth +0.057 spellF1 -- as
much as fixing it everywhere, because only the nearest players decide the team call. The ATD's
team labels are about 93% right, and the video shows the kits plainly: Barcelona dark and
saturated (blue and garnet), Athletic white or light lilac, Athletic's keeper green, the referee
yellow. This extracts what decide.py needs to check the label of each player near the ball
against the shirts in the same frame.

What is measured
----------------
For each tracked slot on each selected frame: project the feet into the image with the provided
homography, size a box from the local pixels-per-metre (0.7 m wide, 1.9 m tall), and take the
torso band of it (15-55% down from the top). Grass pixels are masked by hue. On what remains:

  sat       mean saturation, 0-1           Barcelona high, Athletic low
  a, b      mean Lab chroma, centred on 0  garnet is +a, blue is -b, yellow is +b, green is -a
  white     fraction of pixels with low saturation
  fill      fraction of the torso band that is not grass   (a ghost track is nearly all grass)

Brightness is deliberately left out. Half the pitch is in shadow and half in sun, and a white
shirt in shadow is darker than a garnet one in sun; its chroma does not change.

Frames selected: inside a half, ball tracked, and at least one player within CANDIDATE_M of it.
Nothing here is fitted to the video. The output is derived match data and lives in cache/.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import cv2  # noqa: E402
import players as players_mod  # noqa: E402
import load_homography as homography  # noqa: E402
from games import GAME, GAMES, half_video_blocks, video_path  # noqa: E402

# Params
CANDIDATE_M = 3.0             # players this close to the ball can decide the team call
BOX_WIDTH_M, BOX_HEIGHT_M = 0.7, 1.9
TORSO = (0.15, 0.55)          # fraction of box height, from the top
GRASS_HUE = (30, 90)          # OpenCV hue scale 0-179; pitch green sits inside it
GRASS_MIN_SAT = 50
WHITE_MAX_SAT = 60
MIN_BOX_PX = (4, 10)          # smaller than this and there is nothing to read

FIELDS = ("sat", "a", "b", "white", "fill")
CACHE = os.path.join(HERE, "cache", "colour_{}.npz")


def selected_frames(p, game=GAME):
    length, width = GAMES[game]["pitch"]
    keep = np.zeros(len(p["ball_x"]), dtype=bool)
    for _h, first, last in half_video_blocks(game):
        keep[first:last + 1] = True
    seen = ~np.isnan(p["ball_x"])
    dx = (p["x"] - p["ball_x"][:, None]) * length
    dy = (p["y"] - p["ball_y"][:, None]) * width
    near = np.nanmin(np.where(np.isnan(dx), np.inf, np.hypot(dx, dy)), axis=1) <= CANDIDATE_M
    return np.flatnonzero(keep & seen & near)


def describe(image, u, v, ppm):
    """Descriptor of one player's torso, or None if the box is too small or off the image."""
    w, h = BOX_WIDTH_M * ppm, BOX_HEIGHT_M * ppm
    if w < MIN_BOX_PX[0] or h < MIN_BOX_PX[1]:
        return None
    top = v - h
    y0, y1 = int(top + TORSO[0] * h), int(top + TORSO[1] * h)
    x0, x1 = int(u - w / 2), int(u + w / 2)
    rows, cols = image.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > cols or y1 > rows or x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] >= GRASS_HUE[0]) & (hsv[:, 0] <= GRASS_HUE[1]) & (hsv[:, 1] >= GRASS_MIN_SAT)
    body = ~grass
    fill = body.mean()
    if body.sum() < 4:
        return (np.nan, np.nan, np.nan, np.nan, fill)
    s = hsv[body, 1]
    return (s.mean() / 255.0, lab[body, 1].mean() - 128.0, lab[body, 2].mean() - 128.0,
            (s < WHITE_MAX_SAT).mean(), fill)


def extract(game=GAME, data_dir=None, limit=None):
    p = players_mod.load(game, data_dir)
    matrices, _err, valid = homography.load(game, data_dir)
    frames = selected_frames(p, game)
    if limit:
        frames = frames[:limit]
    length = GAMES[game]["pitch"][0]
    out = np.full((len(frames), p["x"].shape[1], len(FIELDS)), np.nan, dtype=np.float32)

    cap = cv2.VideoCapture(video_path(game, data_dir))
    if not cap.isOpened():  # the bundle names the file BAR-ATH_tactical-001.mp4
        import glob
        cap = cv2.VideoCapture(sorted(glob.glob(os.path.join(
            os.path.dirname(video_path(game, data_dir)), "*tactical*.mp4")))[0])
    position = 0
    started = time.time()
    for i, frame in enumerate(frames):
        while position < frame:  # grab without decoding the frames we skip
            cap.grab()
            position += 1
        ok, image = cap.read()
        position += 1
        if not ok or not valid[frame]:
            continue
        H = matrices[frame]
        for slot in np.flatnonzero(~np.isnan(p["x"][frame])):
            x, y = p["x"][frame, slot], p["y"][frame, slot]
            u, v = homography.project(H, x, y)
            u2, _ = homography.project(H, x + 1.0 / length, y)
            d = describe(image, float(u), float(v), abs(float(u2 - u)))
            if d is not None:
                out[i, slot] = d
        if i and i % 20000 == 0:
            print("  {} / {} frames  ({:.0f}s)".format(i, len(frames), time.time() - started),
                  flush=True)
    return {"frames": frames, "desc": out, "fields": np.array(FIELDS)}


def load(game=GAME, data_dir=None, rebuild=False):
    path = CACHE.format(game)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as h:
            return {k: h[k] for k in h.files}
    out = extract(game, data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)
    return out


if __name__ == '__main__':
    started = time.time()
    c = load(rebuild=True)
    d = c["desc"]
    print("{} frames described in {:.0f}s".format(len(c["frames"]), time.time() - started))
    seen = ~np.isnan(d[..., 0])
    print("player crops with a reading: {}".format(int(seen.sum())))
