"""One pass over the raw ATD, turning 199319 lines of tracking into a handful of numpy arrays.

    python experiments/exp01_geometry_hysteresis/features.py --report

Everything downstream -- the dead-ball layer and the possession state machine -- reads these
arrays and never the text file, so the expensive parse happens once and the parts that are
actually being iterated on are instant.

The whole pass takes about 1.5 seconds, which is worth knowing because it means the cache is a
convenience rather than a necessity. Delete it whenever you suspect it.

Why these features
------------------
Each one answers a question the raw tracking does not answer directly:

  ball_x, ball_y    is the ball on the pitch at all? The hard [0, 1] boundary is the single
                    highest-precision dead-ball signal in the data (98.1%).
  d_home, d_away    not "who is nearest" but *by how much*. The margin between them is the
                    confidence of the team call, and it is what the hysteresis gates on.
  n_home_r, n_away_r  how contested the ball is. Measured as a dead-ball signal and rejected --
                    see deadball.py -- but kept because it is the natural place to look next.
  n_tracked         how much of the pitch the tracker can see on this frame.
  mean_speed        the whole pitch slowing down at once, which happens during a stoppage and at
                    no other time.

Coordinates are scaled to metres before any distance is computed, exactly as starter/baseline.py
does it. The pitch is 105.3 x 68, so skipping that would stretch every distance along the short
axis and silently pick the wrong player near the touchlines.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import HOME, team_of_each_track  # noqa: E402

# Params
# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed", NaN for a missing
# observation. Splitting on ':' gives exactly three parts. Same contract baseline.py relies on.
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"  # first character of "NaN", which is all there is to test for

# Radii, in metres, at which to count how many players of each team are near the ball. Two rather
# than one because they answer different questions: 3 m is a challenge, 5 m is a crowd.
NEAR_RADII = (3.0, 5.0)

CACHE_NAME = "features_{}_{}.npz"


def cache_path(game=GAME, source="tactical", folder=None):
    """Where the cached arrays live. Derived data from the gated bundle, so never commit it."""
    return os.path.join(folder or os.path.join(HERE, "cache"),
                        CACHE_NAME.format(game, source))


def extract(game=GAME, data_dir=None):
    """Parse the raw ATD into per-frame feature arrays. One streaming pass, no kloppy.

    Returns a dict of arrays, each indexed by *video* frame, so index i lines up with row i of a
    submission. The ATD's own frame_id counts from 1, which is the single subtraction below.

    Missing observations stay NaN rather than becoming zero. A zero would be a position on the
    corner flag and would quietly poison every distance computed from it.
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)

    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    d_home = np.full(n_frames, np.nan, dtype=np.float32)
    d_away = np.full(n_frames, np.nan, dtype=np.float32)
    near = {radius: (np.zeros(n_frames, dtype=np.int8), np.zeros(n_frames, dtype=np.int8))
            for radius in NEAR_RADII}
    n_tracked = np.zeros(n_frames, dtype=np.int16)
    mean_speed = np.full(n_frames, np.nan, dtype=np.float32)

    for line in open(raw_path):
        parts = line.split(FIELD_SEP)
        if len(parts) != 3:
            continue
        frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0

        # Players first: n_tracked and mean_speed are meaningful whether or not the ball is seen,
        # and the ball-lost dead-ball rule needs exactly that combination.
        positions, speeds = [], []
        for slot, group in enumerate(parts[1].split(GROUP_SEP)):
            if not group or group[0] == MISSING:
                continue
            fields = group.split(",")
            positions.append((float(fields[0]) * length, float(fields[1]) * width,
                              team_of_track[slot] if slot < len(team_of_track) else HOME))
            # float("NaN") parses, it does not raise, so the missing-value test has to be on the
            # text. Averaging a list that contains one NaN gives NaN, and a tracked player with
            # an unknown speed is common enough that doing this with try/except silently emptied
            # this feature on 94.6% of frames.
            if len(fields) > 2 and fields[2][:1] not in ("", MISSING):
                speeds.append(float(fields[2]))
        n_tracked[frame] = len(positions)
        if speeds:
            mean_speed[frame] = sum(speeds) / len(speeds)

        ball_group = parts[2].strip()
        if not ball_group or ball_group[0] == MISSING:
            continue  # 43% of frames. Everything below is a distance to the ball.
        ball_fields = ball_group.split(",")
        ball_x, ball_y = float(ball_fields[0]), float(ball_fields[1])
        ball[frame, 0] = ball_x
        ball[frame, 1] = ball_y
        # Read for completeness, but do not build anything on it: in this export the ball's third
        # field is NaN on every single frame where the ball is seen. Ball motion has to come from
        # differencing positions instead -- see ball_speed_mps().
        if len(ball_fields) > 2 and ball_fields[2][:1] not in ("", MISSING):
            ball[frame, 2] = float(ball_fields[2])

        ball_mx, ball_my = ball_x * length, ball_y * width
        best = [np.inf, np.inf]
        counts = {radius: [0, 0] for radius in NEAR_RADII}
        for player_x, player_y, team in positions:
            dx, dy = player_x - ball_mx, player_y - ball_my
            distance = (dx * dx + dy * dy) ** 0.5
            side = 0 if team == HOME else 1
            if distance < best[side]:
                best[side] = distance
            for radius in NEAR_RADII:
                if distance < radius:
                    counts[radius][side] += 1
        if best[0] < np.inf:
            d_home[frame] = best[0]
        if best[1] < np.inf:
            d_away[frame] = best[1]
        for radius in NEAR_RADII:
            near[radius][0][frame] = counts[radius][0]
            near[radius][1][frame] = counts[radius][1]

    out = {
        "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_speed": ball[:, 2],
        "d_home": d_home, "d_away": d_away,
        "n_tracked": n_tracked, "mean_speed": mean_speed,
    }
    for radius in NEAR_RADII:
        key = str(int(radius))
        out["n_home_" + key] = near[radius][0]
        out["n_away_" + key] = near[radius][1]
    return out


def load(game=GAME, data_dir=None, rebuild=False, folder=None):
    """Cached extract(). Pass rebuild=True after changing anything above."""
    path = cache_path(game, "tactical", folder)
    if not rebuild and os.path.isfile(path):
        with np.load(path) as handle:
            return {key: handle[key] for key in handle.files}

    features = extract(game, data_dir)
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    np.savez_compressed(path, **features)
    return features


# -- derived views -----------------------------------------------------------------------------
# Kept here rather than in the consumers so that "what counts as the ball being seen" has exactly
# one definition in this experiment.

def ball_seen(features):
    """True where the tracker reported a ball position on this frame."""
    return ~np.isnan(features["ball_x"])


def ball_inside(features):
    """True where the tracked ball is on the pitch. The complement is the dead-ball signal."""
    x, y = features["ball_x"], features["ball_y"]
    return ball_seen(features) & (x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)


def ball_speed_mps(features, game=GAME):
    """Ball speed in m/s, differenced from consecutive tracked positions. NaN where unknown.

    The ATD carries a third field for the ball, and in this export it is NaN on all 113488 frames
    where the ball is seen -- the column exists and is empty. Player speeds *are* populated, so it
    is the ball specifically that has none.

    That matters because "is the ball moving?" is what separates a restart from a ball merely
    placed on the spot, and it is the test the dead-ball layer closes its stretches on. Here it is
    recovered geometrically: scale to metres, difference against the previous frame, multiply by
    fps. Defined only where two consecutive frames both have a position, so a gap in the track
    produces NaN rather than a fictitious jump across it.
    """
    length, width = GAMES[game]["pitch"]
    x = features["ball_x"] * length
    y = features["ball_y"] * width

    speed = np.full(len(x), np.nan, dtype=np.float64)
    dx = np.diff(x)
    dy = np.diff(y)
    speed[1:] = np.sqrt(dx * dx + dy * dy) * GAMES[game]["fps"]
    return speed


def team_margin(features):
    """(nearest_team, margin_m, has_answer) from the two per-team distances.

    nearest_team is 0 for home and 1 for away, following baseline.py's HOME/AWAY codes.
    margin_m is |d_home - d_away|: how much nearer the nearer team is, which is the confidence of
    the call rather than the call itself. has_answer is False where neither team was tracked near
    a visible ball, and the margin there is meaningless rather than zero.
    """
    home = features["d_home"]
    away = features["d_away"]
    has_home, has_away = ~np.isnan(home), ~np.isnan(away)
    has_answer = has_home | has_away

    # A team with no tracked player is infinitely far, so the other team wins by default. Using
    # inf rather than NaN keeps the comparison below total.
    home_filled = np.where(has_home, home, np.inf)
    away_filled = np.where(has_away, away, np.inf)
    nearest = np.where(home_filled <= away_filled, 0, 1).astype(np.int8)

    # Three cases, kept explicit because inf - inf is NaN and would silently become a margin of
    # zero -- i.e. maximum ambiguity -- on exactly the frames where there is no answer at all.
    both = has_home & has_away
    margin = np.zeros(len(home), dtype=np.float64)
    margin[both] = np.abs(home[both] - away[both])
    margin[has_answer & ~both] = np.inf  # one team tracked, the other not: nothing to weigh
    return nearest, margin, has_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="ignore the cache")
    parser.add_argument("--report", action="store_true", help="print a summary of what was read")
    args = parser.parse_args()

    features = load(rebuild=args.rebuild)
    n_frames = len(features["ball_x"])
    print("features for {} frames -> {}".format(n_frames, os.path.relpath(cache_path(), ROOT)))

    if args.report:
        seen = ball_seen(features)
        inside = seen & (features["ball_x"] >= 0) & (features["ball_x"] <= 1) \
            & (features["ball_y"] >= 0) & (features["ball_y"] <= 1)
        nearest, margin, has_answer = team_margin(features)
        finite = margin[np.isfinite(margin) & has_answer]

        print("\nball")
        print("  seen           {:>7} frames  {:5.1f}%".format(
            int(seen.sum()), 100.0 * seen.mean()))
        print("  inside pitch   {:>7} frames  {:5.1f}%".format(
            int(inside.sum()), 100.0 * inside.mean()))
        print("  outside pitch  {:>7} frames  {:5.1f}%   <- the dead-ball signal".format(
            int((seen & ~inside).sum()), 100.0 * (seen & ~inside).mean()))
        print("  x range  {:+.4f} .. {:+.4f}".format(
            np.nanmin(features["ball_x"]), np.nanmax(features["ball_x"])))
        print("  y range  {:+.4f} .. {:+.4f}".format(
            np.nanmin(features["ball_y"]), np.nanmax(features["ball_y"])))

        print("\nplayers")
        print("  mean tracked per frame     {:.1f}".format(features["n_tracked"].mean()))
        print("  mean player speed  {:.2f} .. {:.2f}".format(
            np.nanmin(features["mean_speed"]), np.nanmax(features["mean_speed"])))

        print("\nteam margin |d_home - d_away|, where the ball is seen")
        for percentile in (10, 25, 50, 75, 90):
            print("  p{:<2}  {:5.2f} m".format(percentile, np.percentile(finite, percentile)))
        print("  (the hysteresis in team.py gates on this -- see config.MARGIN_M)")
