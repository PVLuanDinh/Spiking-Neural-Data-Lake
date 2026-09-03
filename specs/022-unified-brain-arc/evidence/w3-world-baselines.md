# W3: the world-red instrument, and Phase 1's pre-registered baselines (2026-09-03)

*The plan of record, Phase 1: "**W3 — the world-red instrument.** Promote the rank 2c probes
(`archive/audit-2026-09-01/rank2c/probe_chains.py` and `probe_fixed_vs_rotating.py`) into `src` as one
instrument with a self-check, registered in `REGIONS` and `run_selfchecks`: per seed, mean and distinct
values over the last 200 wake rows at 3000 AND 6000 ticks of `state()["curiosity_mag"]`, `Brain._lp_mean`,
`state()["neuromod_acetylcholine"]` (Chain A) and `state()["learning_progress"]["cortex"]` (Chain B); the arm
learner's category entropy over 1000 wake ticks; refit count. … **Pre-register the baselines from a measured
run of today's world — s2c's FIXED PACK arm under `throwaway_home()`, `persist=False`, 3 seeds — BEFORE
W1** (the record has no category-entropy baseline)."*

Ratified by the owner on 2026-09-02 under O4. W3 may start before O4/O5 by the plan's own words ("an
instrument with a self-check, no reorder"); both are in any case ratified.

## What was built

`substrate/world_red_probe.py` (commit `20c26117`), registered in `REGIONS`, in `run_selfchecks`'s CORE
list, in `dormancy_gate.EXEMPT` (permanent: a measurement instrument, not a tick faculty — the same
disposition `perf_baseline` carries), and in `docs/INVENTION_REGISTER.md` as invention #9. Bare invocation
is the `run_selfchecks` path: a 0.07 s wiring check over synthetic series that constructs no brain. Its 7
criteria were mutation-tested at landing — 8 mutants, 8 red, 0 survived, with the unmutated check as the
sensitivity control run before and after each mutation.

## Protocol — PRE-REGISTERED before the first run (this section is written first; results follow)

**This run establishes BASELINES. It is not a hypothesis test and it has no bar to pass.** The bar belongs
to W-R1 and W-R2, which W1 and W2 are scored against later. Stating that plainly matters, because a
baseline reported with a bar reads as a result that cleared one.

**Arm (one, deliberately).** The FIXED PACK — today's world, unchanged: `make_brain(seed, persist=False)`
inside `substrate.home.throwaway_home()`, one fresh home per seed so no seed inherits another's lake. No
rotation, no scene change, no goal drive. rank 2c already refused the rotating arm on measurement; W1 is
what changes the world, and this is the "before".

**Seeds.** 0, 1 and 2 — the three s2c measured, so the direction of each is comparable. **Per seed,
never averaged.** s2c measured seed 1 as the one fixed-pack seed whose curiosity had NOT converged at 3000
ticks; a three-seed average hides exactly the seed a later treatment could pass for free on.

**Checkpoints.** 3000 and 6000 `tick()` calls (not wake rows — the plan's W-R1 wording). At each: the last
**200 wake rows**, matching s2c's window. Sampling is wake-only, because `Brain._lp_mean` is written only in
`_wake_tick` and holds its last value through a whole night — sampling every tick would manufacture
plateaus that are an artifact of the circadian cadence, not of the world.

**Statistic, per channel.** `distinct_raw / distinct_round` at 4 dp, adjacent `changes`, `missing`, and the
mean over non-missing values. **The liveness verdict keys on the ROUNDED count**, never the raw one: s2c
measured real channels at 200 / 2, meaning all 200 rows differed at full float precision and collapsed to
two values at 4 dp. Reporting only the raw count would call a dead chain alive.

**Channels.** Chain A — `curiosity_mag`, `_lp_mean`, `neuromod_acetylcholine`. Chain B — the `cortex`
facade `GrowthStop` reads. **The two chains are never averaged or differenced**: `learning_progress.py`'s
own docstring records that the spiking drive was MEASURED INVALID as an |LP| basis, so a merged statistic
would lie in both directions.

**Category entropy.** Shannon bits over `_last_res["unit"]` for the **first 1000 wake rows** — a fixed
window, chosen in advance so no later treatment can pick a friendlier slice. The last-1000 window is
reported beside it and is explicitly NOT a bar. `None` categories (the reflex / E-STOP branch) are counted
in their own field, never dropped: an E-STOP-dominated run must not be able to read as a low-entropy world.

**Refit count.** Reported as three numbers — `scheduled` (`engine.ticks // engine.refit_every`), `effective`
(`len(engine.trajectory)`), and their difference, which counts refits that early-returned on the data
guard. No refit counter exists in the engine; this is arithmetic over two counters it already keeps, not
the inference rank 2b was forced into on the latched daemon.

**Predictions, pinned here so agreement or disagreement is informative either way.** From s2c's §2/§3
tables (its "last 200 WAKE rows" sampling, NOT its §1 "last 200 ticks" tables, which report the same
facade with different numbers), at 3000 ticks:

| seed | `curiosity_mag` raw/4dp, mean | `_lp_mean` 4dp, mean | ACh 4dp, mean | facade 4dp, mean |
|---|---|---|---|---|
| 0 | 1 / 1, 0.0000 | 2, 0.0016 | 2, 0.0016 | 189, 0.068 |
| 1 | 127 / 127, 0.020 (not yet at the floor) | 58, 0.012 | 117, 0.019 | 190, 0.068 |
| 2 | 1 / 1, 0.0000 | 1, 0.0055 | 1, 0.0055 | 190, 0.069 |

**Exact equality is NOT expected and is NOT the bar** — s2c measured on an earlier head, and the tree has
since taken H1 and the O1 latch clearance. What is pre-registered is the DIRECTION: seeds 0 and 2 at the
Chain A floor, seed 1 still alive, and the Chain B facade alive on all three.

**No prior measurement exists for three of the five things this run records**, and they are measured here
rather than copied: the entire 6000-tick half (the string `6000` does not occur in s2c), the category
entropy, and the refit count. One orientation reading taken on this head before the protocol was written is
pinned as the seed-0 entropy expectation: **7 distinct categories, 1.8801 bits** over the first 1000 wake
rows, distribution `{3: 479, 1: 254, 4: 174, 2: 48, 5: 42, 6: 2, 0: 1}`.

**Controls, and the red for THIS run.** The instrument's own controls travel in the same command as every
measurement: the bare wiring check runs first in every mode, so a pinned constant is proven to read AT
FLOOR and a ramp to read LIVE before any channel's verdict is believed. Beyond that:

- **This run's red is a CONTRADICTION, not a floor.** If seed 1's `curiosity_mag` reads at the floor
  (`distinct_round` 1) at 3000 ticks, that contradicts s2c's 127/127 under the same seed and arm, and the
  baseline is NOT trusted until the disagreement is explained — the candidates being the head change, a
  behavioural change from clearing the E-STOP, or the instrument. A number that merely differs is data; a
  reversal on the one seed the earlier record singled out is a finding.
- **All-channels-at-floor on all three seeds at both checkpoints** would mean either a saturated world or a
  dead instrument, and those are told apart by the wiring check's pinned-constant and ramp arms plus the
  facade, which s2c measured alive at ~190 distinct values.
- The `missing` field is its own control: a facade that is absent (never fed) is reported as missing, never
  as 0.0, so an unfed channel cannot masquerade as a silent one.

**Disposition on a null.** A world that reads at its floor on every Chain A channel is a FINDING about the
world, not a failure of W3 — it is precisely the limiter Phase 1 exists to move, and it stands as the
recorded "before" for W-R1. The instrument is deleted only if its OWN reds stop being reachable (P8 clause
5) — that is, if no constructed input can make a criterion fail.

**Conditions held fixed.** The live daemon is left RUNNING during the sweep. It contends for CPU (measured
at about 20 % run-to-run timing variance) but the recorded statistics are distinct-counts, means and
entropies, not timings, so contention cannot move them; stopping the brain to measure a baseline about the
brain would be the larger intervention. Nothing here writes to the live lake: every mode runs
`persist=False` under `throwaway_home()`, and `_assert_scratch_home` refuses the live home outright.

**Command.**

```
PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.world_red_probe \
  --record --seeds 0,1,2 --checkpoints 3000,6000
```

Expected cost, extrapolated from a measured 0.131-0.156 s per tick on this host: about 39-47 minutes for
the three seeds, plus about 4.5 s of brain construction each. Reported here because a sweep whose cost is
not stated in advance tends to get quietly truncated.

## Result

*(To be appended after the run. This section is deliberately empty in the commit that pre-registers the
protocol above; the run had not been executed when the protocol was written.)*
