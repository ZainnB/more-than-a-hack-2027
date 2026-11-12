"""IDSSE event log aligned to scene frames: who played the ball, when, and how.

    python experiments/exp09_touch_model/events.py J03WMX

The event log names the player behind every pass, shot, tackle, claim and other ball action, with
a wall-clock EventTime. The positions file carries a wall-clock T on every frame. This maps one
onto the other, into the frame index of `exp06_idsse/scene.build` (halves trimmed to their live
span, 200-frame gap between them), and maps each player to the scene's column.

It also recovers what scene.build does not keep: the person id behind each column (scene.build
orders them by sorted person id, so the order is reproducible from the FrameSet headers alone).

Everything here reads external data only. Output is cached under cache/ (git-ignored).
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, os.path.join(EXPERIMENTS, "exp06_idsse"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import idsse  # noqa: E402
import scene as scene_mod  # noqa: E402

CACHE = os.path.join(HERE, "cache")
FPS = 25.0

# Event elements that mean "this player touched the ball", and the attribute naming the player.
TOUCH_TYPES = {"Play": "Player", "ShotAtGoal": "Player", "OtherBallAction": "Player",
               "BallClaiming": "Player", "TacklingGame": "Winner", "ThrowIn": None,
               "GoalKick": None, "CornerKick": None, "FreeKick": None, "KickOff": None}

FRAME_N = re.compile(r'N="(\d+)"')
FRAME_T = re.compile(r'T="([^"]+)"')
STATUS = re.compile(r'BallStatus="(\d+)"')


def _seconds(stamp):
    return datetime.fromisoformat(stamp).timestamp()


def scan_positions(match):
    """Person order (as scene.build makes it) and, per half, the wall-clock time of every frame."""
    side = idsse.teams(match)
    team_code = {side["home"]: idsse.HOME, side["away"]: idsse.AWAY}
    persons = {}
    halves = {}
    in_ball, half = False, None
    n_list, t_list, s_list = [], [], []
    with open(idsse.path(match, "positions"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                attrs = dict(idsse.ATTR.findall(line))
                half = attrs["GameSection"]
                in_ball = attrs["TeamId"] == "BALL"
                if attrs["TeamId"] in team_code:
                    persons.setdefault(attrs["PersonId"], team_code[attrs["TeamId"]])
                n_list, t_list, s_list = [], [], []
            elif in_ball and line.startswith("<Frame "):
                n_list.append(int(FRAME_N.search(line).group(1)))
                t_list.append(_seconds(FRAME_T.search(line).group(1)))
                s_list.append(int(STATUS.search(line).group(1)))
            elif in_ball and line.startswith("</FrameSet"):
                halves[half] = (np.array(n_list), np.array(t_list), np.array(s_list))
                in_ball = False
    order = sorted(persons)
    return order, np.array([persons[p] for p in order], dtype=np.int8), halves


def frame_clock(match):
    """Cached: person ids per column, and wall-clock seconds for every scene frame (NaN in gap)."""
    path = os.path.join(CACHE, "clock_{}.npz".format(match))
    if os.path.isfile(path):
        with np.load(path) as h:
            return list(h["persons"]), h["team"], h["clock"]
    order, team, halves = scan_positions(match)
    parts = []
    for number, half in enumerate(scene_mod.HALF_ORDER, start=1):
        _n, t, status = halves[half]
        live = np.flatnonzero(status == 1)
        parts.append(t[live[0]:live[-1] + 1])
        if number == 1:
            parts.append(np.full(scene_mod.GAP_FRAMES, np.nan))
    clock = np.concatenate(parts)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, persons=np.array(order), team=team, clock=clock)
    return order, team, clock


def load(match):
    """List of dicts {frame, column, team, type, sub, header_hint} for every ball event that names
    a player and falls inside a live span. `sub` is the child element (Pass, Cross, ...);
    `header_hint` is True for aerial duels and headed shots."""
    order, team, clock = frame_clock(match)
    column = {p: i for i, p in enumerate(order)}
    valid = np.flatnonzero(~np.isnan(clock))
    times = clock[valid]
    out = []
    root = ET.parse(idsse.path(match, "events")).getroot()
    for event in root.iter("Event"):
        t = _seconds(event.get("EventTime"))
        for el in event.iter():
            kind = el.tag
            if kind not in TOUCH_TYPES:
                continue
            attr = TOUCH_TYPES[kind]
            if attr is None:  # set pieces carry the taker on their child Play
                continue
            player = el.get(attr)
            if player not in column:
                continue
            i = int(np.searchsorted(times, t))
            if i <= 0 or i >= len(times):
                continue
            i = i if abs(times[i] - t) < abs(times[i - 1] - t) else i - 1
            if abs(times[i] - t) > 1.0 / FPS:
                continue  # outside the live span
            subs = [c.tag for c in el]
            out.append({"frame": int(valid[i]), "column": column[player],
                        "team": int(team[column[player]]), "type": kind,
                        "sub": subs[0] if subs else "",
                        "header_hint": (kind == "TacklingGame" and el.get("Type") == "air")
                        or el.get("TypeOfShot") == "head"})
            break  # one touch per event: the outermost element naming a player
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("match")
    args = parser.parse_args()
    match = [m for m in idsse.matches() if m.endswith(args.match)][0]
    ev = load(match)
    print("{}: {} ball events in live play".format(match, len(ev)))
    kinds = {}
    for e in ev:
        kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    print("  " + "  ".join("{} {}".format(k, v) for k, v in sorted(kinds.items())))
    print("  header hints: {}".format(sum(e["header_hint"] for e in ev)))
