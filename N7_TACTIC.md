# N7_TACTIC — Technical document, More than a Hack 2027, Phase 1

## 1. Approach in one paragraph

Possession is treated as a **segmentation** problem, because the ranking metric (spellF1) scores
whole possessions: one spurious flicker splits a true possession and adds a false one. For every
frame we compute two pieces of evidence from the automatic tracking and the Smart Tagging — *does
this look like a stoppage?* and *which team looks in control?* — and decode the H/A/D sequence with
a three-state hidden Markov model (Viterbi): the output maximises total evidence minus a cost for
every change of state. All weights and costs were tuned on **other matches** (Metrica's open sample
games), with their tracking degraded to look like the provided one, never on the provided game.

## 2. What the data does and does not contain (label-free diagnostics)

- The automatic tracking has **no ball-alive/dead channel and no owning-team channel**
  (`ball_state`, `ball_owning_team` empty), so both must be inferred.
- Ball missing on 41% of graded frames, almost entirely in **long gaps** (85% of missing frames in
  gaps >10 s, none of 1–3 s: short occlusions appear to be interpolated by the tracker).
- The ball-speed field is filled on <15% of ball frames and implausibly low; we recompute speed
  from smoothed positions. Player speeds are in m/s (despite the `speed_kph` channel name).
- Ball-to-nearest-player distance: median 1.9 m, only 29% of frames under 1 m — the ball is rarely
  "at the feet" in this tracking.
- Around tagged set pieces the ball is missing 69% of the time (42% at random instants) and
  collective speed drops from ~1.4 to ~0.95 m/s; tags open ~3.5 s before the restart.
- Smart Tagging team tags agree with the team in clear control (ball < 1.5 m from a player, the
  nearest opponent > 3 m away) 71% (possession phases) and 81% (defending phases) of the time.

## 3. Evidence

**Dead ball** (per-frame log-odds, summed by the decoder):
prior towards live play; + ball missing (more if the gap is long); + low collective speed; + ball
persistently (≥0.4 s) just past a line (far-out detections, up to 23 m past the touchline, are
treated as "no ball": spare balls / tracker confusion); + inside a set-piece window, from 3 s
before the tag start to the detected restart kick; − ball moving fast.

**Ownership** (per-frame log-odds, home vs away):
- distance advantage between the nearest home and nearest away player, counted only when control
  is plausible (nearest player within 1.5 m, weakly up to 3 m) and **ignored when the ball travels
  faster than 8 m/s** (a ball passing a player is not a change of possession — CHALLENGE.md);
- **direction of play**: how fast the centroid of all players moves towards one goal (attack
  direction of each team inferred per half from team centroids). When a team wins the ball the
  whole block turns towards the other goal; this is the strongest signal during the long ball
  gaps, where proximity says nothing;
- weak priors from where the play is and from Smart Tagging phases whose Team tag names the team
  in possession (Build Up, Progression, …) or out of it (Defending in …, Defensive Transition).

## 4. Possession rules implemented

- Loose / contested ball: no evidence → the decoder keeps the previous owner.
- Touches, deflections, aerial duels: too short to pay the switching cost → no change.
- After a restart: dead until the restart kick, then whichever team controls the ball.
- Dead ball is always `D`. No `X` on graded frames.

## 5. Validation without touching the provided game's labels

The public 5 minutes are only used to check alignment and format (RULES.md). Tuning uses a bench
built from **Metrica's three open sample games** (other matches; allowed by RULES.md):

1. **Labels** from their event data: `D` from the moment play stops (ball out, foul, offside…)
   to the next on-ball event; otherwise the team of the last possession-defining event. Then the
   CHALLENGE.md notion of *controlled* possession: a turnover counts only if the new team keeps
   the ball ≥3 s or completes ≥3 passes. This turns ~380 event possessions per match (mean 9 s) into
   ~230 (mean ~15 s), in line with the organisers' description of their ground truth.
2. **Degradation** of their clean tracking to match the label-free statistics of section 2:
   long ball gaps during play and stoppages, tracker interpolation of short occlusions, a slowly
   varying 0.8 m ball position error (fitted to the ball-to-player distance distribution), player
   dropouts (~20.5 of 22 visible), spare-ball detections, and simulated Smart Tagging (set pieces
   with realistic recall and offset, phase tags with ~75% team accuracy).
3. **Four scenarios** bracket what we cannot measure (how strongly ball loss is tied to stoppages,
   tag noise); parameters are chosen for the average over scenarios.
4. Scored with the **official** `possession_metrics.py`.

Sanity check of the bench: the organisers' baseline scores **0.247** spellF1 on a held-out bench
game versus **0.2433** on the real full match.

| leave-one-game-out (spellF1) | game 1 | game 2 | game 3 | mean |
|---|---|---|---|---|
| baseline smooth50 | — | — | 0.247 | ~0.25 |
| our pipeline, a-priori parameters | 0.573 | 0.571 | 0.581 | 0.575 |
| our pipeline, tuned on the two other games | **0.659** | **0.614** | **0.626** | **0.633** |

Final parameters (`solution/config.py`) are tuned on all three games (bench spellF1 0.655).

**Tried and not kept:** logistic models learned on ~60 binned features as HMM emissions
(`solution/model.py`, `extval/train.py`): 0.561 vs 0.581 for the hand-made evidence on a held-out
game (both before the direction-of-play cue was added). With labels coming from only three games
and a simulated degradation, more free parameters meant more room to learn the simulator rather
than the football, so the simpler, more interpretable evidence was kept.

## 6. Training data

Every labelled dataset used, as SUBMISSION.md asks. None of it comes from the provided game.

| dataset | source | who labelled it | how much | used for |
|---|---|---|---|---|
| Metrica Sports open sample games 1, 2, 3 | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | event data produced by Metrica Sports (human-coded events); H/A/D frame labels derived from it by our code (`extval/opendata.py`) | 3 matches, 7,300 events, ~283 graded minutes | tuning the evidence weights and decoder costs; validation |

No hand labels were produced by us, on any game.

## 7. Data used, and dropped

| asset | used |
|---|---|
| tactical ATD (players, ball) | yes |
| Smart Tagging | yes (set-piece restarts, phase team tags) |
| homography, videos, broadcast feed | no |
| Metrica open sample games (tracking + events) | yes, for validation and tuning only |

No hand labels anywhere. No data about the provided game other than what was supplied.

## 8. Limitations

- The bench is a simulation: properties of the automatic tracking that cannot be measured without
  labels (e.g. how often the ball is lost during live play) are bracketed by scenarios, not known.
- Event-derived labels approximate, but are not, the corrected-tracking definition of the
  ground truth.
- Boundaries are not re-aligned to the exact contact frame.
- We still under-segment slightly (≈200 predicted vs ≈230 true possessions per bench match).

## 9. Reproduction

See `README.md`. `python make_submission.py` regenerates `N7_TACTIC.csv` exactly;
`python -m extval.tune` reproduces the tuning on the open games.
