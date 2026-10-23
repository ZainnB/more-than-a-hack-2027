"""Smoothed ball kinematics: the features touch detection needs and exp01 did not have.

    python experiments/exp02_touch_gated/features.py --report

exp01's `ball_speed_mps()` is a raw first difference of tracked positions. That is fine for
asking "is the ball moving at all", which is all exp01 used it for, and useless for asking "did
the ball just change direction", which is what a touch is. The research note is explicit: at
25 Hz, raw second differences are noise-dominated, so smooth first.

So this module does the same thing properly. Savitzky-Golay over 7 frames, 2nd order, then
central differences for velocity and again for acceleration.

Two things it is careful about
------------------------------
**Never smooth across a gap.** The ball is untracked on 43% of frames, in runs. A filter run over
a NaN hole invents a trajectory through it, and that invented trajectory bends, which is exactly
the thing being detected. Every run of consecutive tracked frames is filtered on its own and runs
shorter than the window are left alone.

**No scipy.** `scipy.signal.savgol_filter` would do this in a line, but the whole repository runs
on numpy and pandas, and the organisers reproduce a submission from a pinned environment. A
Savitzky-Golay filter of fixed order and width is a fixed convolution kernel -- for the quadratic
7-point smoother, [-2, 3, 6, 7, 6, 3, -2] / 21 -- so the dependency buys nothing here.
"""

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXPERIMENTS)
sys.path.insert(0, HERE)

from games import GAME, GAMES  # noqa: E402

try:
    from . import config
except ImportError:
    import config  # noqa: E402

# exp01's ATD parse and cache are reused wholesale rather than copied. It is imported through its
# package so that its own `config` is exp01's, not this experiment's -- both files are called
# config.py, and a bare sys.path import would hand one of them the other's constants.
from exp01_geometry_hysteresis import features as exp01_features  # noqa: E402

load = exp01_features.load
ball_seen = exp01_features.ball_seen
ball_inside = exp01_features.ball_inside
team_margin = exp01_features.team_margin

def sg_kernel(window, order, deriv=0):
    """Savitzky-Golay weights for any window, polynomial order and derivative, in six lines.

    A Savitzky-Golay filter is a least-squares polynomial fit over a sliding window, evaluated at
    the window's centre. Fitting `y = c0 + c1*t + c2*t^2 + ...` gives every derivative at once:
    the value is c0, the slope is c1, the acceleration is 2*c2. Each is a fixed convolution
    kernel, `pinv(A)[deriv] * deriv!`, so scipy is not needed for any of it.

    Checks out against the textbook smoother: window 7, order 2, deriv 0 gives
    [-2, 3, 6, 7, 6, 3, -2] / 21.

    **Take the derivative from the fit, do not difference the smoothed values.** An order-2
    filter reproduces a quadratic exactly, and constant acceleration *is* quadratic in position,
    so smoothing then differencing twice passes the noise straight through -- measured here, the
    acceleration distribution was identical at window 7 and window 31, which is what sent this
    looking. Reading 2*c2 off the fit instead averages the curvature over the whole window, and
    widening the window then does what widening a window should.
    """
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    powers = np.vstack([offsets ** p for p in range(order + 1)]).T
    return np.linalg.pinv(powers)[deriv] * float(math.factorial(deriv))


def _runs_of(mask):
    """[(start, stop)] half-open spans where a boolean array is True."""
    flagged = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], flagged, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def sg_apply(values, window=None, order=None, deriv=0, fps=None):
    """Savitzky-Golay smoothing applied inside each run of finite values.

    The output is NaN wherever the filter could not be applied honestly: inside gaps, across runs
    shorter than the window, and in the half-window at each run's edge.

    That last one is not fussiness, it is the difference between this working and not. Leaving
    raw values at the edges and smoothed values in the middle puts a step discontinuity at both
    ends of all 113k tracked runs, and a central difference across a step is an enormous spike --
    the first version of this function did exactly that and produced ball speeds of 759 m/s and a
    signal *noisier* than the raw one it was meant to clean. Padding instead would fabricate
    curvature precisely where the ball reappears after an occlusion, which is where a spurious
    touch does the most damage. Three frames at each end are cheap; a fake touch is not.
    """
    window = window or config.SG_WINDOW
    order = order or config.SG_ORDER
    fps = float(fps or GAMES[GAME]["fps"])
    # np.convolve reverses its kernel; the smoother is symmetric so it never noticed, but the
    # derivative kernels are antisymmetric and would come out with the wrong sign.
    kernel = sg_kernel(window, order, deriv)[::-1] * (fps ** deriv)

    out = np.full(len(values), np.nan, dtype=np.float64)
    half = window // 2
    for start, stop in _runs_of(np.isfinite(values)):
        if stop - start < window:
            continue  # too short to filter; left NaN rather than part-filtered
        # 'valid' then reinsert, so the kernel never reaches outside the run.
        out[start + half:stop - half] = np.convolve(values[start:stop], kernel, mode="valid")
    return out


def sg_smooth(values, window=None, order=None):
    """The plain smoother: the Savitzky-Golay fit's value at each frame."""
    return sg_apply(values, window, order, deriv=0)


def ball_metres(features, game=GAME):
    """Ball position in metres. NaN where untracked."""
    length, width = GAMES[game]["pitch"]
    return features["ball_x"] * length, features["ball_y"] * width


def kinematics(features, game=GAME):
    """Smoothed ball motion: dict of speed, acceleration, heading change and displacement.

    Everything is derived from the *smoothed* trajectory, and everything is in SI units -- m,
    m/s, m/s^2, degrees -- so the thresholds in config.py are the published ones and need no
    conversion to read.

    Velocity and acceleration are read off the polynomial fit rather than differenced from the
    smoothed positions -- see sg_kernel for why differencing an order-2 fit denoises nothing. The
    kernels are symmetric about the centre frame, so an acceleration peak lands *on* the frame of
    the touch rather than after it. A forward difference would make every touch systematically
    late, and not being late is the entire point of this experiment.
    """
    fps = float(GAMES[game]["fps"])
    raw_x, raw_y = ball_metres(features, game)
    x = sg_smooth(raw_x)
    y = sg_smooth(raw_y)

    vx = sg_apply(raw_x, deriv=1, fps=fps)
    vy = sg_apply(raw_y, deriv=1, fps=fps)
    ax = sg_apply(raw_x, deriv=2, fps=fps)
    ay = sg_apply(raw_y, deriv=2, fps=fps)

    speed = np.sqrt(vx * vx + vy * vy)
    accel = np.sqrt(ax * ax + ay * ay)

    # Heading change across +/- span frames, in degrees. Computed from the velocity vectors
    # rather than from positions, so it is a change of *direction of travel* and is unaffected by
    # how fast the ball happens to be going.
    span = config.TOUCH_HEADING_SPAN
    heading = np.full(len(x), np.nan, dtype=np.float64)
    before_x, before_y = vx[:-2 * span], vy[:-2 * span]
    after_x, after_y = vx[2 * span:], vy[2 * span:]
    dot = before_x * after_x + before_y * after_y
    norms = np.sqrt(before_x ** 2 + before_y ** 2) * np.sqrt(after_x ** 2 + after_y ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.clip(np.where(norms > 0, dot / norms, np.nan), -1.0, 1.0)
    heading[span:-span] = np.degrees(np.arccos(cosine))

    # Straight-line distance covered across the same span, Vidal-Codina's epsilon_s. A heading
    # change is only meaningful if the ball actually went somewhere.
    displacement = np.full(len(x), np.nan, dtype=np.float64)
    displacement[span:-span] = np.sqrt((x[2 * span:] - x[:-2 * span]) ** 2
                                       + (y[2 * span:] - y[:-2 * span]) ** 2)

    return {"x": x, "y": y, "vx": vx, "vy": vy,
            "speed": speed, "accel": accel,
            "heading_change": heading, "displacement": displacement}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    features = load(rebuild=args.rebuild)
    kin = kinematics(features)
    raw_speed = exp01_features.ball_speed_mps(features)

    print("smoothed ball kinematics over {} frames".format(len(kin["x"])))
    print("\n  {:<22} {:>9} {:>9} {:>9} {:>9}".format("", "p50", "p90", "p99", "max"))
    for name, values in (("speed m/s (smoothed)", kin["speed"]),
                         ("speed m/s (exp01 raw)", raw_speed),
                         ("accel m/s^2", kin["accel"]),
                         ("heading change deg", kin["heading_change"]),
                         ("displacement m", kin["displacement"])):
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        print("  {:<22} {:>9.2f} {:>9.2f} {:>9.2f} {:>9.2f}".format(
            name, *np.percentile(finite, [50, 90, 99]), finite.max()))

    # Jitter measured only where both signals are defined on *consecutive* frames, so the
    # comparison is not contaminated by differences taken across a gap in the ball track.
    both = np.isfinite(raw_speed) & np.isfinite(kin["speed"])
    pairs = both[:-1] & both[1:]
    noise_raw = np.std(np.diff(raw_speed)[pairs])
    noise_smooth = np.std(np.diff(kin["speed"])[pairs])
    print("\n  frame-to-frame jitter in speed, over {} consecutive tracked pairs:".format(
        int(pairs.sum())))
    print("    raw {:.3f} -> smoothed {:.3f} m/s  ({:.1f}x quieter)".format(
        noise_raw, noise_smooth, noise_raw / max(noise_smooth, 1e-9)))
    print("  this is why acceleration is only usable after smoothing -- see config.SG_WINDOW")
