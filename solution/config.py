"""Every tunable number of the pipeline, in one place, with where it comes from.

NONE of these values was fitted to the public 5-minute ground truth, nor to anything about the
provided game's labels (forbidden, see RULES.md).

- Feature-level values (cleaning, smoothing, set-piece window geometry) come from football priors
  and from label-free statistics of the provided tracking / Smart Tagging (diag_step1/2/3).
- Evidence weights, thresholds and decoder costs were then TUNED on the open-data bench
  (extval/): Metrica's three public sample games (other matches, event data -> H/A/D labels with
  the CHALLENGE.md "controlled possession" rule), with their tracking degraded to match the
  label-free statistics of the provided ATD. Command: python -m extval.tune
  Leave-one-game-out spellF1 on that bench: 0.633 tuned vs 0.575 untuned (baseline ~0.25).

Units: metres, seconds, frames at 25 fps. Evidence values are per-frame log-odds; the decoder
sums them over time, so a value of 1.0 held for 1 s weighs 25.
"""

FPS = 25

# --------------------------------------------------------------------------- ball cleaning
# Positions further than this beyond a line are not the match ball (spare balls, tracker
# confusion: the provided tracking reaches 23 m past the touchline). Treated as "no ball".
BALL_FAR_OUT_M = 5.0
# Ball jumps faster than this between two frames are tracker errors (a pro shot is ~30-35 m/s).
BALL_MAX_SPEED = 40.0
# Light centred smoothing of the ball track before differentiating (positions jitter by ~1 m/s
# equivalent even when the ball is still).
BALL_SMOOTH_FRAMES = 5
# Collective speed is averaged over this centred window.
COLLECTIVE_SMOOTH_FRAMES = 25

# --------------------------------------------------------------------------- dead-ball evidence
DEAD_BIAS = -0.5            # prior: play is alive more often than not
GAP_MIN_S = 0.5             # ball missing for >= 3 s ...
GAP_W = 1.0                 # ... adds this
GAP_LONG_S = 20.0           # and for >= 10 s ...
GAP_LONG_W = 0.25            # ... adds this on top
SPEED_REF = 1.8             # m/s. Match median ~1.5, around restarts ~0.95 (diag_step2)
SPEED_SCALE = 0.5           # m/s per unit of evidence
SPEED_W = 0.5               # weight of the collective-speed term (clipped to +/-2 units)
OOB_MARGIN_M = 0.5          # ball this far past a line ...
OOB_MIN_FRAMES = 10         # ... for at least 0.4 s => out of play
OOB_W = 2.0
SP_BEFORE_S = 3.0           # Smart Tagging set piece: tag opens ~3.5 s before the restart
SP_SEARCH_S = 15.0          # look for the restart kick up to this long after the tag start
SP_FALLBACK_S = 5.0         # median ball re-appearance after a tag start (diag_step2)
SP_KICK_SPEED = 3.0         # m/s: the ball starts moving => restart
SP_W = 1.0
FAST_BALL_SPEED = 6.0       # a ball moving faster than this is usually in play
FAST_BALL_W = 2.5

# --------------------------------------------------------------------------- ownership evidence
CONTROL_DIST_M = 1.5        # nearest player within 1.5 m: plausible control
NEAR_DIST_M = 3.0
STRENGTH_CONTROL = 1.0
STRENGTH_NEAR = 0.4
STRENGTH_FAR = 0.0
DIFF_SCALE_M = 0.5          # distance advantage (m) per unit of evidence, clipped to +/-2
FLIGHT_SPEED = 8.0          # ball faster than this is passing by, not controlled ...
FLIGHT_FACTOR = 0.0         # ... so its proximity evidence is damped
TAG_W = 0.2                 # Smart Tagging phase team, per frame, weak
TERR_W = 0.1                # where the play is (+1 deep in away half), seen from home's attack
TERRV_W = 0.1               # how fast the play moves towards the away goal (m/s)

# Smart Tagging codes whose Team tag is the team IN possession / the team NOT in possession.
TAG_OWN = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
           "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
TAG_OTHER = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")

# --------------------------------------------------------------------------- decoder (HMM)
# Cost of each state change, in the same units as the evidence. They set how many possessions
# come out: a switch needs ~SWITCH_COST of accumulated contrary evidence.
SWITCH_COST = 28.0          # H <-> A
TO_DEAD_COST = 24.0         # H/A -> D
FROM_DEAD_COST = 16.0       # D -> H/A


def params(**overrides):
    """All UPPER_CASE values above as a namespace, optionally overridden (used for validation)."""
    from types import SimpleNamespace
    values = {k: v for k, v in globals().items() if k.isupper()}
    unknown = set(overrides) - set(values)
    if unknown:
        raise KeyError("unknown parameter(s): {}".format(sorted(unknown)))
    values.update(overrides)
    return SimpleNamespace(**values)
