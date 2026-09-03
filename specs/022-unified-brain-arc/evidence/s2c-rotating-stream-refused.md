# Rank 2c: a rotating real N-MNIST stream — REFUSED ON MEASUREMENT, not built (2026-09-02)

Audit §8 rank 2, commit 3 of 3: *"move `_perceive_real` off the fixed 256-frame N-MNIST pack the |LP|
facade has already saturated on (a rotating or streamed real dataset — P7 accepts the dependency). Red:
the |LP| facade must leave its floor (1 distinct value → >1) with the same seeds; if it does not, the
stimulus was not the limit and the finding is recorded."* Owner directive: "Do rank 2."

This is the third outcome that rule allows, reached BEFORE a build: the premise does not reproduce on the
signal the audit named, the floor it described lives on a different chain, and a rotating stream —
probed in-process without touching the tree — does not move that chain either. Verify-first says a
mechanism is built after its red is shown reachable and before its result is claimed; here the probe
stage returned the finding, so nothing lands but the record. (P0's honesty guard: a capability reported
without measurement is fake; P3: a stimulus change that charges nothing is a wire that does not charge.)

## 1. The premise, re-measured: the facade the growth stop reads is NOT at a floor on the pack

The audit's §3 item 1 and its rank-2 text say the basal-ganglia `LearningProgress` facade (Chain B: percept
surprise from `_perceive_real` over the 256-frame pack → `consume_surprise` → `lp("cortex")`, the signal
`GrowthStop` reads through `_LPRoute`) "converges … 1 distinct value over the last 200 ticks". Fresh
brains, `throwaway_home()`, `persist=False`, `state()["learning_progress"]["cortex"]` per tick:

| run | last 200 ticks: distinct values (raw / 4 dp) | range | mean of last 200 | per-256-tick means |
|---|---|---|---|---|
| seed 0, 1200 ticks | 100 / 98 | 0.209 | 0.056 | 0.004, 0.051, 0.057, 0.063 |
| seed 0, 3000 ticks | 100 / 95 | 0.197 | 0.059 | … 0.055, 0.063, 0.058, 0.057, 0.067, 0.057, 0.073 |
| seeds 1, 2 at 1200 | 98, 100 / — | — | — | — |

The facade is alive on the pack: ~100 distinct values per 200 ticks, a range of 0.2, a per-cycle mean of
0.05–0.07 that is not descending toward the audit's 0.016. It is also identical across brain seeds (the
pack order and the surprise instrument are seed-free), so "with the same seeds" was moot for it. **The
rank-2 red as written has no reachable baseline on this tree** — a criterion with no reachable red is
deleted, not weakened (P8 clause 1).

## 2. Where the audit's floor actually is: the arm's curiosity chain, and it reproduces

The audit's own gap-round refuter recorded the two chains (`GAP_A2`): Chain A = the arm loop's
spiking-curiosity RECTIFIED `learning_progress(unit)` over its JEPA world-model error on the `_safe_ctx`
context stream (the 3-stimulus synthetic rotation + real proprioception + the attention-pooled real
channels), read as `_lp_mean`, `curiosity_mag` and ACh; Chain B = the facade above. The "1 distinct
value" was Chain A. Re-measured here (seed 0, wake ticks only, last 200 wake rows):

| signal | 1200 ticks: distinct raw / 4 dp, mean | 3000 ticks: distinct raw / 4 dp, mean |
|---|---|---|
| `_lp_mean` (Chain A) | 200 / 27, 0.0043 | 200 / 2, 0.0016 |
| `curiosity_mag` (Chain A) | 77 / 77, 0.0036 | **1 / 1, 0.0000** |
| `neuromod_acetylcholine` (Chain A) | 34 / 34, 0.0044 | 2 / 2, 0.0016 |
| facade `cortex` (Chain B) | 200 / 186, 0.065 | 200 / 189, 0.068 |

Chain A converges to its floor by 3000 ticks exactly as the audit measured — on the world-model error
over the arm's context, not on the N-MNIST pack. Chain B, the signal the growth stop consumes, does not.

## 3. The probe the build would have had to pass: rotating windows do not move Chain A

Cached test frames (`data/nmnist_cache/test_T15/*.hdf5`, 10,000 files, tonic `ToFrame` at T=15 — the
pack's own binning: integer counts, max 8, shape 15×2×34×34; a 256-frame window loads in 2.5 s; the
encoder classifies a cached window at 0.97 against 0.93 on the pack; 10 of 256 window frames overlap the
pack). In-process probe, no tree change: swap `_real_frames/_real_labels` to a fresh seeded 256-frame
window every time the pointer completes a cycle, 3000 ticks, against the fixed pack, same seed:

| seed 0, last 200 wake rows | FIXED PACK | ROTATING (5 windows) |
|---|---|---|
| `curiosity_mag` distinct (raw / 4 dp), mean | 1 / 1, 0.0000 | **1 / 1, 0.0000** |
| `_lp_mean` distinct 4 dp, mean | 2, 0.0016 | 1, 0.00003 |
| ACh distinct 4 dp, mean | 2, 0.0016 | 1, 0.0000 |
| facade `cortex` distinct 4 dp, mean | 189, 0.068 | 183, 0.063 |
| novelty_live distinct, mean | 98, 0.048 | 107, 0.053 |
| vision prototypes (`real_perception_units`) / encoder acc | 24 / 0.93 | 26 / 0.95 |

| seed 1, last 200 wake rows | FIXED PACK | ROTATING (5 windows) |
|---|---|---|
| `curiosity_mag` distinct (raw / 4 dp), mean | 127 / 127, 0.020 (not yet at the floor) | **1 / 1, 0.0000** |
| `_lp_mean` distinct 4 dp, mean | 58, 0.012 | 1, 0.047 |
| ACh distinct 4 dp, mean | 117, 0.019 | 1, 0.047 |
| facade `cortex` distinct 4 dp, mean | 190, 0.068 | 184, 0.063 |
| vision prototypes / encoder acc | 24 / 0.93 | 26 / 0.95 |

| seed 2, last 200 wake rows | FIXED PACK | ROTATING (5 windows) |
|---|---|---|
| `curiosity_mag` distinct (raw / 4 dp), mean | 1 / 1, 0.0000 | **1 / 1, 0.0000** |
| `_lp_mean` distinct 4 dp, mean | 1, 0.0055 | 1, 0.0017 |
| ACh distinct 4 dp, mean | 1, 0.0055 | 1, 0.0017 |
| facade `cortex` distinct 4 dp, mean | 190, 0.069 | 182, 0.063 |
| vision prototypes / encoder acc | 24 / 0.93 | 26 / 0.95 |

Three seeds, one shape: a rotating real stream leaves Chain A at its floor on every seed — and on seed 1,
where the fixed pack had NOT yet converged at 3000 ticks, rotation is what converged it — leaves Chain B
alive as before, and buys two more vision prototypes and two points of encoder accuracy. **The N-MNIST pack is not
the limit of the chain that saturates; the arm's context is** — the synthetic 3-stimulus rotation and the
proprioception of an arm that, on the live daemon, has been E-STOP-latched since 2026-08-31 (rank 2b's evidence records the likely consequence as an INFERENCE with no latch-cleared
control: 3,539 wake ticks without a refit, `n_modes` 2, floors None). The pooled real channels
enter that context as scalars a world model masters in a few hundred ticks whatever frames feed them.

## Disposition

**Not built.** No rotating loader lands: it would be a stimulus change with no consumer-side red it can
fail (P3's shape 4 — a faculty fed every tick, never read differently), and the criterion the audit
attached to it is deleted per P8 clause 1 (no reachable red on the named signal; the corrected red on
Chain A is reachable and the probe FAILS it). The dependency P7 would have accepted (`h5py` is already
installed; no `tonic` at runtime) is not the question.

What the measurement changes downstream:

- **The lever for Chain A is the arm's context, not the pack**: the E-STOP clear (audit §8 rank 8, an
  owner action) restores proprioceptive motion; beyond that, the `_safe_ctx` synthetic rotation is the
  bounded stimulus the world model masters — the audit's original `_safe_ctx` comment named it. A
  richer CONTEXT (not richer frames) is the pre-registration worth writing: its red is Chain A's
  `curiosity_mag` leaving 1 distinct value at 3000 ticks under the same seeds — the instrument built here.
- **Rank 2 closes at 2 of 3 built + 1 refused on measurement**: 2a (the grower fed the brain's own
  stream; NULL against the stream's floors), 2b (the prototype learners persist; the restart red green;
  attribution NULL), 2c (refused). Ranks 3 and 4 of the audit stand as ordered.

Reproduce: the three probes (facade floor at 1200/3000; Chain A vs B; fixed vs rotating, seeds 0–2) are
archived with the session's audit record beside the memory store (`archive/audit-2026-09-01/rank2c/`);
each is an in-process script over `make_brain(seed, persist=False)` under `throwaway_home()`.
