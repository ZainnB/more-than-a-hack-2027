"""Stamp possession boundaries on the tactical video, by hand, one window at a time.

    pip install -r requirements-validation.txt
    python validation/annotate.py --window win1

Controls
--------
    h / a / d     stamp a boundary here: Home, Away, Dead ball
    , / .         step one frame back / forward
    [ / ]         step one second back / forward        (25 frames)
    { / }         step ten seconds back / forward       (250 frames; shift of [ and ])
    space         play / pause
    n / p         jump to the next / previous stamp
    u             undo the last stamp
    w             write now (it also writes on quit)
    q / Esc       quit

What you are annotating
-----------------------
One label per frame: which team is in possession with the ball alive, or D for a dead ball. You
stamp only the frames where the answer *changes*; expand.py fills in the rest.

CHALLENGE.md is the specification and it is worth rereading before starting. The two rules that
decide most of the hard frames:

  - While the ball is in play, one of the two teams always has it. There is no contested label.
    "In phases where the ball is contested or loose, possession stays with the last team that had
    it", so a 50/50 in midfield is not a boundary -- the boundary is where someone comes out of it
    in control.
  - A touch without control is not a change of possession. Neither is a deflection, nor an aerial
    duel won without the ball being brought under control afterwards.

Dead ball means the ball is not in play by the referee's reckoning: between the ball leaving the
pitch and the restart being taken, and from a whistle to the free kick. Do not try to say who
"owns" a dead ball -- the ground truth does not, and neither should this.

On not biasing the labels
-------------------------
The overlay draws the raw tracking -- players coloured by team, and the ball -- and **nothing
else**. It deliberately does not show the nearest player, any possession guess, or any output of
exp01. A human who is shown the model's answer tends to agree with it, and a validation set that
agrees with the model by construction measures nothing.

Judge from the video. The overlay is there because the ball is small at this zoom and the team
colours are hard to separate in shadow, not to tell you the answer.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

import windows as win  # noqa: E402
import expand as expander  # noqa: E402
from games import GAME, GAMES, DATA_DIR, require  # noqa: E402
from baseline import HOME, team_of_each_track  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402

# Params
# BGR, because OpenCV. Red for home, cyan for away -- far apart in both hue and brightness, which
# matters on a sunlit pitch with half of it in shadow.
COLOUR_HOME = (60, 60, 235)
COLOUR_AWAY = (235, 235, 60)
COLOUR_BALL = (255, 255, 255)
COLOUR_TEXT = (60, 255, 60)
LABEL_COLOURS = {"H": COLOUR_HOME, "A": COLOUR_AWAY, "D": (200, 200, 200)}

STEPS = {",": -1, ".": 1, "[": -25, "]": 25, "{": -250, "}": 250}
WINDOW_NAME = "annotate -- h/a/d stamp, ,/. frame, [/] second, u undo, w write, q quit"


def find_video(data_dir=None):
    """The tactical mp4, tolerating the bundle's actual naming.

    games.asset_paths() builds 'BAR-ATH_tactical.mp4' from games.SUFFIXES, but the delivered
    bundle names it 'BAR-ATH_tactical-001.mp4'. Rather than encode one download's quirk into
    games.py, glob for it here -- this tool is the only thing in the repository that opens video.
    """
    folder = os.path.join(data_dir or DATA_DIR, "tactical")
    matches = sorted(glob.glob(os.path.join(folder, "BAR-ATH_tactical*.mp4")))
    if not matches:
        raise SystemExit("no tactical mp4 under {}\nPass --video explicitly.".format(folder))
    return matches[0]


def read_atd_frames(start, stop, game=GAME, data_dir=None):
    """{video_frame: (players, ball)} for one window, read in a single pass.

    players is [(x, y, team)] and ball is (x, y) or None, both in normalised 0-1 coordinates so
    they go straight into the homography. Only the window is kept: the whole file is 195 MB and
    there is no reason to hold it.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team_of_track = team_of_each_track(meta_path)

    out = {}
    with open(raw_path) as handle:
        for line in handle:
            head, _sep, _rest = line.partition(":")
            try:
                frame = int(head) - 1  # the ATD counts from 1, submissions from 0
            except ValueError:
                continue
            if frame < start:
                continue
            if frame >= stop:
                break
            parts = line.split(":")
            if len(parts) != 3:
                continue

            players = []
            for slot, group in enumerate(parts[1].split(";")):
                if not group or group[0] == "N":
                    continue
                fields = group.split(",")
                team = team_of_track[slot] if slot < len(team_of_track) else HOME
                players.append((float(fields[0]), float(fields[1]), team))

            ball_group = parts[2].strip()
            ball = None
            if ball_group and ball_group[0] != "N":
                fields = ball_group.split(",")
                ball = (float(fields[0]), float(fields[1]))
            out[frame] = (players, ball)
    return out


def label_at(stamps, frame):
    """The label in force at a frame, given the stamps so far."""
    current = "?"
    for stamped, label in stamps:
        if stamped <= frame:
            current = label
        else:
            break
    return current


def draw(image, frame, atd, homography, stamps, name, start, stop, playing, cv2):
    """The video frame with the tracking drawn on it and the annotation state in the corner."""
    players, ball = atd.get(frame, ([], None))
    matrix = homography[frame]

    if not np.isnan(matrix).any():
        for x, y, team in players:
            px, py = project(matrix, x, y)
            colour = COLOUR_HOME if team == HOME else COLOUR_AWAY
            cv2.circle(image, (int(px), int(py)), 11, colour, 3)
        if ball is not None:
            px, py = project(matrix, ball[0], ball[1])
            cv2.circle(image, (int(px), int(py)), 9, COLOUR_BALL, -1)
            cv2.circle(image, (int(px), int(py)), 14, (0, 0, 0), 2)

    fps = GAMES[GAME]["fps"]
    current = label_at(stamps, frame)
    done = frame - start
    total = stop - start

    # Drawn along the bottom, deliberately. The broadcast's burnt-in match clock sits top-left,
    # and that is the thing an annotator cross-references against -- covering it would be the one
    # placement that costs something.
    lines = [
        "{}  frame {}  video {:02d}:{:05.2f}  {}/{} ({:.0f}%){}".format(
            name, frame, int(frame / fps) // 60, frame / float(fps) % 60,
            done, total, 100.0 * done / total, "   PLAYING" if playing else ""),
        "label here: {}        stamps: {}".format(current, len(stamps)),
    ]
    for index, text in enumerate(lines):
        origin = (30, image.shape[0] - 150 + index * 42)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    LABEL_COLOURS.get(current, COLOUR_TEXT) if index == 1 else COLOUR_TEXT, 2)

    # A timeline of what has been stamped so far, so gaps are obvious at a glance.
    bar_y, bar_h = image.shape[0] - 60, 26
    cv2.rectangle(image, (30, bar_y), (image.shape[1] - 30, bar_y + bar_h), (40, 40, 40), -1)
    width = image.shape[1] - 60
    edges = [f for f, _ in stamps] + [stop]
    for (stamped, label), end in zip(stamps, edges[1:]):
        x0 = 30 + int(width * (stamped - start) / float(total))
        x1 = 30 + int(width * (min(end, stop) - start) / float(total))
        cv2.rectangle(image, (x0, bar_y), (max(x1, x0 + 1), bar_y + bar_h),
                      LABEL_COLOURS[label], -1)
    cursor = 30 + int(width * (frame - start) / float(total))
    cv2.line(image, (cursor, bar_y - 8), (cursor, bar_y + bar_h + 8), (0, 255, 255), 3)
    return image


def write_stamps(name, stamps):
    path = expander.label_path(name)
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in stamps)
    return path


def load_stamps(name):
    """Whatever was stamped in an earlier session, so annotating can be done in sittings."""
    path = expander.label_path(name)
    if not os.path.isfile(path):
        return []
    stamps = []
    with open(path) as handle:
        for line in handle:
            text = line.strip()
            if not text or text.lower().startswith("frame"):
                continue
            frame, label = text.split(",")[:2]
            stamps.append((int(frame), label.strip().upper()))
    return sorted(stamps)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", required=True,
                        choices=list(win.WINDOWS) + list(win.CALIBRATION))
    parser.add_argument("--video", help="tactical mp4 (default: found under the data directory)")
    parser.add_argument("--start-at", type=int, help="open at this frame instead of the first")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv is not installed.\n"
                         "    pip install -r requirements-validation.txt")

    name = args.window
    start, stop = win.frames(name)
    video = args.video or find_video()

    print("loading tracking for {} [{}, {}) ...".format(name, start, stop))
    atd = read_atd_frames(start, stop)
    homography, _error, _valid = load_homography()

    capture = cv2.VideoCapture(video)
    if not capture.isOpened():
        raise SystemExit("could not open {}".format(video))

    stamps = load_stamps(name)
    if stamps:
        print("resuming with {} stamps already in {}".format(name, expander.label_path(name)))
    frame = args.start_at if args.start_at is not None else start
    playing = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    position = None

    try:
        while True:
            frame = max(start, min(stop - 1, frame))
            if position != frame:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                position = frame
            ok, image = capture.read()
            position = frame + 1
            if not ok:
                print("could not read frame {}".format(frame))
                break

            cv2.imshow(WINDOW_NAME, draw(image, frame, atd, homography, stamps,
                                         name, start, stop, playing, cv2))
            key = cv2.waitKey(40 if playing else 0) & 0xFF
            if key == 255:
                if playing:
                    frame += 1
                continue
            character = chr(key)

            if key in (27, ord("q")):
                break
            elif character == " ":
                playing = not playing
            elif character in STEPS:
                playing = False
                frame += STEPS[character]
            elif character in ("h", "a", "d"):
                stamps = sorted([s for s in stamps if s[0] != frame]
                                + [(frame, character.upper())])
                write_stamps(name, stamps)
            elif character == "u" and stamps:
                removed = max((s for s in stamps if s[0] <= frame), default=stamps[-1])
                stamps = [s for s in stamps if s != removed]
                write_stamps(name, stamps)
                frame = removed[0]
            elif character == "n":
                later = [f for f, _ in stamps if f > frame]
                if later:
                    frame = later[0]
            elif character == "p":
                earlier = [f for f, _ in stamps if f < frame]
                if earlier:
                    frame = earlier[-1]
            elif character == "w":
                print("wrote {}".format(write_stamps(name, stamps)))
            elif playing:
                frame += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()
        path = write_stamps(name, stamps)

    print("\n{} stamps -> {}".format(len(stamps), os.path.relpath(path, ROOT)))
    covered = stamps and stamps[0][0] == start
    if not covered:
        print("NOTE: the first stamp is not at {}, so expand.py will reject this window.".format(
            start))
        print("      Go to the first frame and stamp the label play opens with.")
    print("\nWhen every window is done:")
    print("  python validation/expand.py")


if __name__ == '__main__':
    main()
