# Phase 1 exit — the world (the limiter): three tables, the pre-registered baselines beside them (2026-09-05)

*The plan of record, Phase 1: "**Exit:** `evidence/w-exit.md`, three tables, the pre-registered baselines
beside them."* Phase 1 held three tasks. W3 landed (the instrument and the baselines). W1 was built,
measured, and retired by owner decision. W2 was probed and refused on measurement before a build. The
phase closes with a FINDING rather than a wire, which is the outcome the plan's Part C ("the world before
the wire") exists to produce cheaply. Every number below is copied from the task record it names;
re-derive from those files, never from this one.

## Table 1 — W3: the pre-registered baselines of today's world (`evidence/w3-world-baselines.md`)

Fixed pack, `throwaway_home()`, `persist=False`, per seed, never averaged; head `43d06f7e`. Reproduced
digit-for-digit at `f546e0a` by W2's instrument control (W2 record §4.1).

| seed | `curiosity_mag` last-200 @3000: distinct@4dp, mean | @6000 | `entropy_first` (first 1000 wake rows) | last-1000 @6000 |
|---|---|---|---|---|
| 0 | 1, 0.00000 — AT FLOOR | 1, 0.00000 | 1.8801 bits / 7 | 0.5974 / 4 |
| 1 | 127, 0.02001 — alive | 5, 0.00026 — collapsing | 1.0369 / 4 | 2.6046 / 8 |
| 2 | 1, 0.00000 — AT FLOOR | 1, 0.00000 | 2.0415 / 6 | 0.0000 / 1 |

The Chain B facade (`state()["learning_progress"]["cortex"]`) was alive on every seed at every
checkpoint (181–190 distinct at 4 dp). Chain A (the arm's curiosity) is the limiter.

## Table 2 — W1: the goal curriculum, built and RETIRED (`evidence/w1-goal-curriculum.md`)

| measurement | result | disposition |
|---|---|---|
| W-R3 (T076's red): `propose()`'s goal cell moves when `coverage_reward` is neutered | the mechanism was green (9 wiring criteria, 14/14 mutants red) | — |
| Part F unrelated-scalar specificity control | **RED**: substituting `_rpe` / `_plan_w` / `_plan_base` moves 0.863–0.906 of proposals against a SHUFFLE BOUND of 0.879 | no objective-specificity |
| the actuation half, scored on coverage (400 ticks × 8 seeds) | goal vs incumbent paired t **−0.34**; vs magnitude-preserving shuffle **−0.52**; FRONTIER arm **−0.50** — none resolvable at 1.895 (incumbent alone spans 50–107 cells; a 10 % effect needs ~96 seeds) | no resolvable effect either way |
| the invented floor `1/(1+snr)` | not inert (8/8, t 21.44) but never better than the literature constant; explores 2.5× more | not ported |
| owner decision 2026-09-04 | **RETIRED** (P8 deletion-on-a-null), `17e4613`; bandit and buffer untouched | T076's mechanism WITHDRAWN |

The structural line that carries: a curriculum whose candidate set is its own history cannot serve a
novelty objective, and proposing unreached cells (the FRONTIER arm) does not rescue it.

## Table 3 — W2: the real latents into the context, REFUSED ON MEASUREMENT (`evidence/w2-context-latents.md`)

Nine in-process cells, no tree change, seeds 0–2, 3000 ticks, head `f546e0a`; gates pre-registered
with their predicted outcomes before the run.

| gate | what it asked | result | predicted |
|---|---|---|---|
| G5 instrument | the incumbent reproduces Table 1 with the read-only wraps on | 3/3 digit-for-digit | pass |
| G-ORDER | the world model's error is lower under the natural order than under a per-cycle permutation | last-block perr 0.00139/0.00134, 0.00135/0.00178, 0.00191/0.00179 — 0/3 | fail |
| G1 (W-R1's premise) | `curiosity_mag`(W2) > `curiosity_mag`(W2, context time-shuffled) at 3000 on 3/3 | 0.132/0.013, **0.000/0.039**, 0.092/0.053 — 2/3 | fail |
| G-ID (W-R2a's twin) | first-1000 entropy above Table 1 | 5.39/5.18/6.08 bits vs 1.88/1.04/2.04 — 3/3, by 51–102 arm units | pass by recruitment |
| G-W-R2b | grower's held-out next-category accuracy above its persistence floor and above the shuffled arm | 0.017/0.013/0.017 vs floors 0.051/0.062/0.039 — 0/3; vs shuffled 1/3 | fail |

Beside them, the instrument finding that decides the phase: the arm's `JepaWorldModel` latent collapses
on every context this body sees (trained/fresh target-latent spread ×0.005–0.11 on nine of nine models),
so its prediction error cannot see succession; and the tracked packs carry no first-order succession
(ordered next-label MI at the median of 200 shuffles).

## The phase's finding, and what carries to Phase 2

**The limiter is not the world's richness and not the body.** It is (i) a world model whose latent
collapses — under the arm's own defaults, on the incumbent context and on a three-times-richer one alike —
and (ii) a stimulus with no first-order succession to model. Every "curiosity leaves the floor" reading
Phase 1 produced (W3 seed 1 at 3000, W2's 30–190×) is the learning-progress transient of newly recruited
arm prototypes, and it returns to exactly 0.0 when recruitment stops (W2 §4.4, seed 1).

**Carried to Phase 2, in the plan's order:** L1a, L3 and the oracle are stream-independent and start now.
**Held:** the null disposition "L2 re-targeted at prediction error" — the prediction error it names is the
collapsed one; it waits on Q1 (`w2-context-latents.md` §5, plan Part H). **Filed as their own arcs:** the
JEPA collapse (its red is specified: a planted 3-bit successor world must give perr(ordered) <
perr(shuffled); `latent_std` must hold a pre-registered fraction of the fresh model's spread; the
hinge/invariance gradient ratio counted before the rule) and `AutonomousArm.step`'s `_cur_mag` held gauge.
**Nothing from Phase 1 is on the live tick that was not there before it**: W3's instrument is
dormancy-exempt as a measurement tool; W1 is deleted; W2 was never written.
