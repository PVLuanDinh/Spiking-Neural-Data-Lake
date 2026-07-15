# Brain Review -- cohesion audit of the unified Brain

Scope: `src/snn_data_lake/cortex/brain.py` (the unified `Brain`) and
`src/snn_data_lake/cortex/self_improvement_suite.py` (the new 8-rung census).
READ-ONLY review -- suggestions only, no code changed.

## Cohesion verdict: MOSTLY one program

`Brain` is genuinely ONE class with ONE driving loop: `tick()` advances a
`CircadianClock`, then runs `_wake_tick()` or `_sleep_tick()`, and every tick
folds the REAL signals into a single evolving `SelfModel`. `state()` and
`respond()` both read from that one self-model. That spine is cohesive, not a
pile of modules.

Where it is genuinely unified (live in the loop, feeding the self-model or
actually gating behavior):

- LiveBrain (perception + GWT + amygdala + BG gate + interoception + PFC + lake
  WM + spiking curiosity + JEPA + interaction learner) -- ticked every wake.
- Ensemble modulators (ACh, LC, raphe, astrocyte, homeostat, ring) -- ticked
  every wake, fold into arousal / homeostatic load / telemetry.
- Metacognitive monitor (L10) -- REAL control, not telemetry: gates `respond()`
  (HOLD/abstain) and gates lifelong learning (`learn_gate()` in `_enter_sleep`).
- Access-consciousness (GNW ignition) -- updated every wake, feeds self-model
  and `regions_live`.
- Neurogenesis (torch, sleep-gated), attractor sleep-replay, SWR buffer replay.
- Reasoning (transitive inference), lifelong consolidation (L6+L11), criticality
  (L7) -- all trained/maintained LIVE every night in `_enter_sleep()`.

Where it is honestly-labeled but still BOLTED-ON (constructed and run ONCE at
birth in `__init__`, reported in `state()`, but never fed back into the loop):

- Wholistic integration capstone (L13) -- `WholisticLoop(...).run()` at birth;
  emits `integration_competence/gain/retention/ok` but never participates in a
  subsequent tick.
- Self-improvement suite census (L5, L8, L9x2, L6, L11x2, L12) -- `census(seed)`
  at birth; emits `state()["rungs"]` but never drives a later tick.

So: cohesive core loop + two birth-time composition readouts hanging off it. The
census is real composition (all 8 rungs load and run in-process under one seed),
but it is composition-AS-SNAPSHOT, the same wiring status as the older
`wholistic_loop` capstone -- not the live-integration status of reasoning /
lifelong / criticality / metacog.

## Redundancy / double-wiring / conflict

No runtime conflict and no dead path: the census runs on its own seeded
instances, is fully guarded, and only writes `self._rungs` (a pure telemetry
sink). But there is real CONCEPTUAL duplication across the two layers:

1. TWO "unifiers." Both `wholistic_loop` (L13, "composes its OWN primitives into
   ONE arbitrated loop") and `self_improvement_suite` ("all rungs into ONE
   program") are birth-time composition readouts, over DIFFERENT rung sets, each
   framed as the capstone. That is the main cohesion smell -- two things both
   claiming to be THE composition layer.

2. Consolidation appears FOUR times across four distinct modules:
   - LIVE: `hippocampus/sleep_replay.py` (AttractorNet, in `_sleep_tick`)
   - LIVE: `plasticity/lifelong_consolidation.py` (L6+L11, in `_enter_sleep`)
   - CENSUS: `hippocampus/sleep_consolidation.py` (L11 sleep-retention gap)
   - CENSUS: `plasticity/benna_fusi.py` (L11 cascade retention)
   The census versions are birth snapshots of a capability the live loop already
   exercises -- they demonstrate consolidation in parallel rather than as part of
   the one nightly consolidation pass.

3. Growth appears THREE times: census `function_preserving_growth` (L6, exact
   0.0 invariant), live `lifelong_consolidation` grow<->prune (L6), and live
   neurogenesis (`neuro.tick`). Only the census one is a snapshot.

None of these is a bug; all are honestly labeled. They are the cost of bolting a
birth-census onto a loop that already integrates the same concepts live.

## Liveness vs demo: which rungs to promote

The census is honestly labeled -- the module docstring and the `state()["rungs"]`
comment both say "reduced-scale liveness census; full proof = each module's
self-check." That claim is defensible and should stay. But several rungs have a
natural LIVE home and would be genuinely wired (not a snapshot) there:

PROMOTE to the live loop (worth it):
- L11 `sleep_consolidation` + L11 `benna_fusi` -> `_enter_sleep()`. Offline
  consolidation already runs there (reasoning, lifelong, criticality, attractor
  sleep). These two belong in that nightly pass, not as a separate birth demo.
- L5 `generalizing_novelty` -> `_wake_tick()` feeding the ensemble novelty
  signal. `_tick_ensemble` already computes `novelty` from `lp_mean`; a SimHash
  pseudo-count is exactly what an intrinsic-curiosity drive consumes, so this
  rung would do real work every wake tick.

PROMOTE cautiously, VALUE/SELF-MODEL ONLY (never the actuator):
- L9 `drive_reduction` reward -> fold into the self-model valence / homeostatic
  load, NOT the motor value path. Safe only if it never reaches PFC goal / arm.
- L9 `autotelic_goals` -> at most the self-model's goal signal; do NOT let a
  babbled goal set the PFC motor intent (that borders the actuator route).

KEEP as a birth readout (fine as-is):
- L6 `function_preserving_growth` -- it is an exact-invariant PROOF (diff==0.0),
  not a driver; live growth is already covered by neurogenesis + lifelong.
- L8 `meta_plasticity` -- wiring learning-rate gating into the live learners is
  invasive; the birth readout is the honest place for it.
- L12 `quality_diversity` -- illumination/behavioral coverage is a diagnostic;
  no natural per-tick home.

## Safety: CONFIRMED value/representation-only

- The census imports NOTHING from `brainstem/` or the arm. Its import block pulls
  only from `cortex/`, `hippocampus/`, `neuromodulation/`, `plasticity/`, and its
  own `__main__` asserts (un-riggably, by scanning its import lines) that
  `brainstem`, `serial_arm`, `arm_sim`, `arm_config` never appear.
- `brain.py` imports `brainstem.arm_config.CONTACT` only as a scalar constant to
  compute `SAFE_DISTANCE`; it is not an actuator handle. The census does not even
  do that.
- The new suite adds NO actuator route. All motor safety still lives inside the
  COMPOSED, untouched LiveBrain: the reflex E-STOP fires first and the gated
  `ArmSim.apply()` is the only path to motion. `respond()` still routes any
  structured arm/lake command through the deterministic NL firewall, never the
  language model. The `__main__` self-check (claim 3) re-verifies E-STOP fire +
  freeze-persist and that production never trips.

## Determinism + guarding: CONFIRMED

- `census(seed)` uses per-instance `random.Random(seed)` throughout; each rung is
  wrapped `try/except -> SENTINEL`, so one rung can never take down the brain.
  `__main__` asserts `census(0) == census(0)` byte-identical.
- The `Brain.__init__` call site wraps the census in `try/except -> self._rungs =
  {}`. Every other composed piece is likewise guarded: criticality
  (`try/except -> None`), wholistic (`try/except`), and each of reasoning /
  lifelong / criticality inside `_enter_sleep` has its own `try/except`. A hiccup
  in any composed piece degrades to a default and cannot perturb the loop or the
  safety envelope.
- The `Brain` self-check (claim 4) constructs two seed=0 brains and asserts
  byte-identical state including `self_state` and the Q-table checksum.

## The one honest caveat: metaplast_gain = -0.25

At reduced scale the L8 rung legitimately returns a NEGATIVE `metaplast_gain`
(post-flip full minus both-off), and the suite CORRECTLY does not assert it > 0
(the positive result is proven only in `meta_plasticity.py`'s own full-scale
self-check). This is honest inside the suite. The risk is at the CONSUMER: in
`state()["rungs"]` the value surfaces RAW as `metaplast_gain: -0.25` under a bare
key. The only "reduced-scale" caveat is a single collective code-comment on the
`rungs` key -- it is not attached to the individual number a reader sees, so
`-0.25` can be misread as "meta-plasticity HURTS / a regression."

Recommendation: keep the raw value (honesty), but make the scale un-missable at
the point of consumption -- wrap the census output, e.g.
`{"census": {...}, "_scale": "reduced", "_note": "liveness/composition proof, not
performance; values may be negative at this scale -- full proof in each module's
self-check"}`, or tag each metric. Cheapest honest fix; nothing else about the
number needs to change.

## Ranked concrete changes

1. (HIGH) Promote the two L11 consolidation rungs (`sleep_consolidation`,
   `benna_fusi`) into `_enter_sleep()` and L5 `generalizing_novelty` into
   `_wake_tick()`'s ensemble-novelty signal. This turns 3 of 8 rungs from a birth
   snapshot into genuine live integration, next to reasoning/lifelong/criticality
   where consolidation already lives. Keep them guarded + value-only.

2. (HIGH) Resolve the "two unifiers" redundancy. `wholistic_loop` (L13) and
   `self_improvement_suite` are both birth-time composition readouts over
   different rung sets. Pick ONE named composition layer: either fold the census
   under the wholistic capstone, or explicitly demote the census to a "birth
   diagnostic" in its label so only one thing claims to be THE ladder composition.

3. (MEDIUM) Fix the misleading raw `metaplast_gain: -0.25` in `state()["rungs"]`
   by wrapping the census with an explicit `_scale`/`_note` tag (see caveat
   above) so a negative reduced-scale value cannot read as a regression.

4. (LOW) Promote `drive_reduction` reward into the self-model valence /
   homeostatic-load path -- but ONLY there, never the motor value path -- and
   leave `autotelic_goals` at self-model-goal at most; both border the actuator
   and should not touch PFC/arm.

5. (LOW / optional cleanup) `census(seed)` runs (~0.3s) on EVERY `Brain()`
   construction, including the 4-5 throwaway brains built inside the self-check.
   Consider lazy-computing `_rungs` on first `state()` read, or gating it behind a
   flag, to keep repeated construction cheap. Purely a cost nicety; not required.

## What is genuinely unified vs still bolted-on

GENUINELY UNIFIED (live, feeds the one self-model or actually gates behavior):
LiveBrain, ensemble modulators, access-consciousness, neurogenesis + attractor
sleep-replay + SWR replay, reasoning, lifelong consolidation, criticality, and
-- importantly -- the metacognitive monitor, which is real control (respond HOLD
+ lifelong learn-gate), not telemetry.

STILL BOLTED-ON (honestly-labeled birth-time readouts that report but do not
drive a later tick): the wholistic integration capstone (L13) and the
self-improvement suite census (L5/L8/L9x2/L6/L11x2/L12). They prove in-process
COMPOSITION under one seed -- a real and defensible claim -- but they are
snapshots, not live participants, and they duplicate (in miniature) consolidation
and growth machinery the live loop already runs. Promoting the 3 rungs in change
(1) and collapsing the two capstones in change (2) is what would move this from
"mostly one program" to "one program."
