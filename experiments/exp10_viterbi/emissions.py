"""Per-frame log-scores for H, A and D, built only from evidence earlier experiments already compute.

Nothing here is a new feature. Each term reuses existing code by import:

  dead vs live   exp08's dead-ball classifier (deadfeat.compute -> deadclf.proba), the same
                 model.json exp08 ships, smoothed exactly as exp08's own decode smooths it.
  home vs away   the per-team nearest-player distances (exp03's keeper-corrected features on the
                 match, exp06's adapter on IDSSE) and exp02's in-flight test.
  touches        exp09's touch model (hook.score_candidates), the same model.json exp09 ships.
  tagging        Smart Tagging transition phases, read with the starter loader exp01 uses.

What an emission means here
---------------------------
E[t, k] is "how much frame t looks like state k", in nats (natural-log units). Only differences
matter to the decoder, so each term is built so that *no evidence is exactly 0*: a frame where
the ball is unseen adds nothing to H or A, and the decoder simply carries whichever team it was
already holding. That is the point of a global decoder over the 41% of frames with no ball: no
smoothing rule has to decide what happens in the gap, the costs do.

The same `build` runs on the provided match and on degraded IDSSE, so what idsse_eval.py tunes is
exactly what run.py ships.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
ROOT = os.path.dirname(EXPERIMENTS)
for path in (ROOT, EXPERIMENTS, HERE, os.path.join(EXPERIMENTS, "exp08_deadball_clf"),
             os.path.join(EXPERIMENTS, "exp07_team_colour"), os.path.join(EXPERIMENTS, "exp09_touch_model"),
             os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

from exp10_viterbi import config  # noqa: E402  (exp10's own)
import deadclf  # noqa: E402
import deadfeat  # noqa: E402
from exp02_touch_gated import config as config2, features as feat2  # noqa: E402

H, A, D = 0, 1, 2
LOG2 = np.log(2.0)


def _log_sigmoid(x):
    """log(1 / (1 + e^-x)), written so it neither overflows nor loses precision at either end."""
    return -np.logaddexp(0.0, -x)


def team_evidence(features, game, ha_scale):
    """(n, 2) home/away log-scores from the signed nearest-player margin.

    z = d_away - d_home: positive when home is nearer. Passed through a logistic, so

        E_H - E_A = ha_scale * z / Z_SOFT,   clipped at |z| = Z_MAX  (so at most 4 * ha_scale)

    and a near tie is close to 0 on both. Each side is shifted by log 2 so that z = 0 scores
    exactly 0, the same as an unseen ball: "can't tell" and "can't see" are the same to the
    decoder.

    Neutral (0, 0) wherever the margin is not evidence of control:
      - ball unseen, or no player tracked near it
      - the nearest player further than FAR_M (the ball is travelling between players)
      - the ball in flight above exp02's IN_FLIGHT_SPEED_MPS
    One team tracked and the other not counts as a clear margin for the tracked team, but only if
    it is within FAR_M: the other team is somewhere, just not seen.
    """
    home = np.asarray(features["d_home"], dtype=np.float64)
    away = np.asarray(features["d_away"], dtype=np.float64)
    has_h, has_a = np.isfinite(home), np.isfinite(away)
    nearest = np.fmin(np.where(has_h, home, np.inf), np.where(has_a, away, np.inf))

    z = np.zeros(len(home))
    both = has_h & has_a
    z[both] = away[both] - home[both]
    z[has_h & ~has_a] = config.Z_MAX_M
    z[has_a & ~has_h] = -config.Z_MAX_M
    z = np.clip(z, -config.Z_MAX_M, config.Z_MAX_M)

    speed = np.nan_to_num(feat2.kinematics(features, game)["speed"], nan=0.0)
    usable = (has_h | has_a) & (nearest <= config.FAR_M) & (speed <= config2.IN_FLIGHT_SPEED_MPS)

    out = np.zeros((len(home), 2))
    x = z[usable] / config.Z_SOFT_M
    out[usable, H] = ha_scale * (_log_sigmoid(x) + LOG2)
    out[usable, A] = ha_scale * (_log_sigmoid(-x) + LOG2)
    return out


def dead_evidence(p_dead, smooth):
    """(n, 2) log-scores (live, dead) from the classifier's per-frame p(dead).

    Smoothed over +-`smooth` frames first (exp08's decode setting, chosen on IDSSE), because the
    raw probability wobbles frame to frame and the decoder would otherwise pay for each wobble.
    """
    q = deadfeat.window_mean(np.asarray(p_dead, dtype=np.float64), -smooth, smooth) \
        if smooth else np.asarray(p_dead, dtype=np.float64)
    q = np.clip(np.nan_to_num(q, nan=0.5), config.P_CLIP, 1.0 - config.P_CLIP)
    return np.stack([config.W_DEAD * np.log1p(-q), config.W_DEAD * np.log(q)], axis=1)


def touch_evidence(n, touches, touch_gain):
    """(n, 2) bonus for the touching team around each confident exp09 touch candidate.

    A candidate with p > TOUCH_MIN_P adds touch_gain * logit(p) nats in total to its own team,
    spread evenly over +-TOUCH_SPREAD frames because its timing is only known to about a second.
    A weak candidate adds nothing (it is not evidence *against* the team).
    """
    out = np.zeros((n, 2))
    if touches is None or not touch_gain:
        return out
    frames, teams, p = touches
    keep = p > config.TOUCH_MIN_P
    w = 2 * config.TOUCH_SPREAD + 1
    for f, tm, q in zip(frames[keep], teams[keep], p[keep]):
        q = min(float(q), 1.0 - config.P_CLIP)
        lo, hi = max(0, int(f) - config.TOUCH_SPREAD), min(n, int(f) + config.TOUCH_SPREAD + 1)
        out[lo:hi, int(tm)] += touch_gain * np.log(q / (1.0 - q)) / w
    return out


def build(features, game, p_dead, dead_smooth, ha_scale, touch_gain=0.0, touches=None):
    """(n, 3) emissions E[:, (H, A, D)]: live evidence on H and A, dead evidence on D."""
    n = len(features["d_home"])
    team = team_evidence(features, game, ha_scale) + touch_evidence(n, touches, touch_gain)
    live_dead = dead_evidence(p_dead, dead_smooth)
    E = np.empty((n, 3))
    E[:, H] = live_dead[:, 0] + team[:, H]
    E[:, A] = live_dead[:, 0] + team[:, A]
    E[:, D] = live_dead[:, 1]
    return E


# -- the provided match --------------------------------------------------------------------------

def dead_model():
    with open(deadclf.MODEL_PATH) as h:
        return json.load(h)


def match_inputs(game, data_dir=None, with_touch=True):
    """(features, p_dead, touches, dead_smooth) for the provided match, from cached features.

    p_dead is computed exactly as exp08_deadball_clf/run.py's classifier_dead does, stopping
    before its threshold-and-minimum-length decode: the decoder needs the probability, not the
    mask. touches are exp09's scored candidates, before its tau threshold, for the same reason.
    """
    import players as players_mod      # exp07's per-slot position cache
    from exp03_keeper_fix import features as feat3
    from games import GAMES, half_video_blocks

    model = dead_model()
    p = players_mod.load(game, data_dir)
    length, width = GAMES[game]["pitch"]
    ball = np.stack([(p["ball_x"] - 0.5) * length, (p["ball_y"] - 0.5) * width],
                    axis=1).astype(np.float64)
    players = np.stack([(p["x"] - 0.5) * length, (p["y"] - 0.5) * width],
                       axis=2).astype(np.float64)
    p_dead = deadclf.proba(model, deadfeat.compute(ball, players, length / 2.0, width / 2.0))

    touches = None
    if with_touch:
        import hook as touch_hook      # exp09
        import touchclf
        with open(touchclf.MODEL_PATH) as h:
            tmodel = json.load(h)
        team = np.broadcast_to(p["team"], p["x"].shape)
        inside = np.zeros(len(ball), dtype=bool)
        for _h, a, b in half_video_blocks(game):
            inside[a:b + 1] = True
        touches = touch_hook.score_candidates(tmodel, ball, players, team, inside)

    features = feat3.load(game, data_dir)
    return features, p_dead, touches, int(model["decode"]["smooth"])


def tag_relief(game, n, fps, data_dir=None):
    """(relief_h, relief_a): fraction of the switch cost forgiven per frame, from Smart Tagging.

    ATTACKING TRANSITION tagged for team T means T has just won the ball; DEFENSIVE TRANSITION
    tagged for T means T has just lost it, so the *other* team won it. Within TAG_WINDOW_S of
    either's start, switching to the winning team costs (1 - TAG_RELIEF) of the usual price. Tags
    whose Team is N/A carry no side and are skipped. Club names are resolved against the ATD
    metadata, as exp01's tag_team_votes does.
    """
    import load_smart_tagging as tagging
    from baseline import team_names
    from games import require

    events = tagging.load(game, data_dir)
    home_name, away_name = team_names(require(["atd_meta"], game, data_dir)[0])
    side_of = {str(home_name).strip().lower(): H, str(away_name).strip().lower(): A}
    relief = np.zeros((n, 2))
    half = int(round(config.TAG_WINDOW_S * fps))
    for event in events:
        code = (event["code"] or "").strip().upper()
        if code not in ("ATTACKING TRANSITION", "DEFENSIVE TRANSITION"):
            continue
        side = side_of.get(str(event["tags"].get("Team") or "").strip().lower())
        if side is None:
            continue
        gainer = side if code == "ATTACKING TRANSITION" else 1 - side
        start = int(event["start_frame"])
        lo, hi = max(0, start - half), min(n, start + half + 1)
        relief[lo:hi, gainer] = config.TAG_RELIEF
    return relief[:, H], relief[:, A]
