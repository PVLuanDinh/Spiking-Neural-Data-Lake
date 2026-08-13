# Coverage evidence -- SpatialTemporal

Scored against `docs/COVERAGE_RUBRIC.md`. One row per faculty; the category score is the mean of the
`band` column. A row is never deleted to raise a score: a faculty that cannot be measured stays here
as `UNSCORED`, which counts as 0. Bands above 40 require an A/B artifact; above 80, held-out evidence.

**Category score: 23.9** (mean of 11 bands, 0 UNSCORED, 0 above 40).

| faculty | band | citation | artifact | measured_by | measured_at |
|---|---|---|---|---|---|
| path_integration_dead_reckoning | 35 | src/snn_data_lake/brainstem/autonomous_loop.py:1958#step; src/snn_data_lake/cortex/brain.py:2189#_c1_tick | - | opus-5 auditor / live tick instrumentation, 60 ticks seed 0, daemon sys.path | 2026-07-30 |
| interval_timing_time_cells | 35 | src/snn_data_lake/cortex/time_cells.py:168#tick; src/snn_data_lake/cortex/brain.py:1742#_route_percept_to_action | st_timing_60.txt | opus-5 auditor / spatiotemporal_probe.py timing 60 (charge_probe verdict) | 2026-07-30 |
| sequence_binding_synfire | 35 | src/snn_data_lake/cerebellum/synfire_chain.py:172#survives; src/snn_data_lake/cortex/brain.py:2334#_c1_tick | - | opus-5 auditor / live tick instrumentation, 60 ticks seed 0, daemon sys.path | 2026-07-30 |
| path_integration_spiking_can | 30 | src/snn_data_lake/hippocampus/spiking_grid_can.py:217#step; src/snn_data_lake/cortex/brain.py:2190#_c1_tick | - | opus-5 auditor / live tick instrumentation, 60 ticks seed 0, daemon sys.path | 2026-07-30 |
| interval_timing_wake_counter | 30 | src/snn_data_lake/cortex/brain.py:2043#_wake_tick; src/snn_data_lake/cortex/brain.py:3433#state | - | opus-5 auditor / live tick instrumentation, 60 ticks seed 0, daemon sys.path | 2026-07-30 |
| place_coding | 21 | src/snn_data_lake/hippocampus/grid_place_map.py:184#observe; src/snn_data_lake/cortex/brain.py:1516#_consume_perception | st_place_60.txt | opus-5 auditor / call instrumentation + spatiotemporal_probe.py place 60; band UNCHANGED but its rationale is superseded, see "The freeze is fixed" | 2026-07-30 |
| grid_coding | 21 | src/snn_data_lake/hippocampus/grid_place_map.py:124#grid_code; src/snn_data_lake/hippocampus/grid_cells.py:71 | - | opus-5 auditor / call instrumentation, 60 ticks seed 0, daemon sys.path; band UNCHANGED, see "The freeze is fixed" | 2026-07-30 |
| landmark_drift_correction | 21 | src/snn_data_lake/hippocampus/spiking_grid_can.py:250#observe_landmark; src/snn_data_lake/cortex/brain.py:1458#_consume_perception | - | opus-5 auditor / return-value instrumentation, 60 ticks seed 0 | 2026-07-30 |
| sequence_memory_polychronous | 15 | src/snn_data_lake/cerebellum/polychronization.py:23#module; src/snn_data_lake/cortex/rose_language.py:358#dependency_order_ok | - | opus-5 auditor / static importer trace (STATIC ONLY, not runtime; paths+line re-derived at the 2026-08-12 promotion) | 2026-07-30 |
| sequence_prediction_sorn | 10 | src/snn_data_lake/reasoning/sorn.py:201#self_organize | - | opus-5 auditor / static importer trace (STATIC ONLY, not runtime) | 2026-07-30 |
| heading_integration_ring_attractor | 10 | src/snn_data_lake/hippocampus/ring_attractor.py:107#ProprioceptiveEstimator; src/snn_data_lake/brainstem/live_arm.py:67#_make_observer | - | opus-5 auditor / static importer trace (STATIC ONLY, not runtime) | 2026-07-30 |

## The one finding that changes how every single-source verdict in this ledger should be read

> **CLOSED 2026-08-13 by specs/013, and this section states the OPPOSITE of what it once did.**
> It is kept because the reasoning is what generalises, not the verdict. What follows describes the
> defect; the measurement that closed it is at the end.

`tools/charge_probe.py:39` puts exactly one directory on `sys.path`: `<root>/src`. `coverage_score.py
record` pins `PYTHONPATH=<root>/src` too. Neither adds the repo ROOT, so the top-level `training/`
package was unimportable and the perception tier's bare-root imports raised, whereupon a single
shared `except Exception` in `cortex/brain.py::Brain.__init__` set `_perception_tier="stdlib"` and
`_perceiving_real=False`. **Under the sanctioned probe harness this brain was blind**, even though
`models/eprop_nmnist.pt` and `models/nmnist_pack.npz` were both present. `learn_daemon.py` sits at
the repo root, so the always-on daemon -- the live path the coverage question is about -- had the
root importable and perceived real N-MNIST frames; the ledger and the daemon disagreed.

**RE-MEASURED 2026-08-13 at the tip, under that same sanctioned harness (`PYTHONPATH=<root>/src`,
repo root NOT on `sys.path`): `perception_tier = "eprop-snn"`, `_perceiving_real = True`,
`_perceiving_audio = True`.** The caveat is discharged, by two independent halves of specs/013:
the last bare-root perception import went in-package (`UNGOVERNED_ROOT` is now `()`), so nothing
raises; and the shared handler was SPLIT, so vision and audio no longer share a fate even if one
day something does. A single-source verdict recorded under this harness before 2026-08-13 was
measuring the harness; one recorded after is measuring the brain.

*(A prior revision of this paragraph, written mid-lane at C1, claimed the observable was UNCHANGED
because "that surviving audio import still raises in the same try". That was true when written and
was falsified two commits later by C2 and C3 -- the lane's own later work invalidated its own
coverage prose, and no gate saw it. Caught by the specs/013 pre-push review.)*

Measured, same seed, 60 ticks, `.venv-lake`:

- `PYTHONPATH=src` -- tier `stdlib`, `_perceived_digit` None on 60/60 ticks, `_place_here` None on
  60/60, `_place_value` 0 pairs.
- root on `sys.path` -- tier `eprop-snn`, `_perceived_digit` set, `_place_here` set, `_place_value`
  7 pairs with values to 0.6427.

So `place_60.txt`'s `NOT-ELIGIBLE` (exit 2, "the source never activated in this run") is a property of
the PROBE, not of the brain. Had I scored from it I would have recorded a rule-4 zero against a
faculty that does fire in production. The fix is one line in `tools/charge_probe.py`; that file was
being edited by other agents during this audit, so the correction is quarantined in
`docs/coverage/artifacts/spatiotemporal_probe.py`, which inserts root+src and then calls
`charge_probe.main()` -- the neuter (`_PinnedWeights`), the eligibility test, the observables and the
exit-code contract are all the tracked tool's, unchanged.

**Then the known-good control failed, and that is the second finding.** `drive 60` is the ledger's
non-degeneracy control: 22/60 trajectory flips, exit 0 (`drive_60.txt`, cited at
`COVERAGE_BASELINE.md:90`). Re-run on the corrected harness it returns **0/60, exit 1**
(`st_drive_control_60.txt`). Two readings were available -- my harness is degenerate, or the drive
vector is redundant -- and `st_pathway_control_60.txt` decides it: neutering **all eleven** drive
sources at once yields **47/60 trajectory flips**, so the observable is sensitive and the top-down
pathway is decisive. What is not decisive is any single source once real perception is live: ten other
voices carry the sum. `charge_probe`'s leave-one-out neuter therefore **systematically under-detects
charge in the regime the daemon actually runs in**, and every 0/60 recorded on it -- in this file and
in the other category files -- is a *masked* negative, not evidence of inertness. That cuts both ways
and I have not claimed a band from it: the same control also shows the leave-one-in variant (place the
only wired source, everything else pinned) at 0/60, so place is not even sufficient on its own.

## Rule-4 analysis, per faculty, from runtime instrumentation

Static reading has been wrong in both directions here, so every claim below except the last three rows
comes from instrumenting the live tick, not from grep.

**place_coding is a live silent-degradation bug, and this is the largest thing I found.** Over 60
ticks the wake path calls `GridPlaceMap.observe` **31** times but `place_at` only **3** times -- from
the same straight-line pair of statements at `brain.py:1310-1311`. The only path that skips `:1311` is
`observe` raising into the bare `except Exception: pass` at `:1335-1336`. It does, 28 times out of 31:

    AttributeError: 'int' object has no attribute 'startswith'
      grid_place_map.py:166  variants = [p for p in self._places if p == percept or p.startswith(...)]

`_places` keys are whatever the caller passes for the first variant and `f"{percept}@{k}"` (a str) for
later ones. `brain.py:1310` passes `self._perceived_digit`, an **int**. Once the bare int key `6` is
stored, any later percept that is not 6 makes `p == percept` false and `p.startswith` raise. The
module's own self-check and `tests/test_grid_place_map.py` pass str percepts, so nothing caught it.

Consequence, measured: `_place_here` is assigned on ticks 3, 4 and 9 and then holds `'6@1'` frozen for
the remaining 51 ticks; `_place_from_spiking_bump` is likewise frozen True on exactly 60-9=51 ticks.
No new place is ever bound. The R-STDP update at `:1390-1394` and the source eligibility at `:1397`
keep learning and reading a **stale** place identity, and the drive term at `:1572` consumes a
constant. That is the "constant where a computed value belongs" tell, and it is why this scores 21
rather than the 40 its call graph suggests: the faculty runs three times and then stops, silently. It
is not 0, because rule 4 is about a precondition that never fires and this one fires three times.

**grid_coding** is live and correct as far as it goes: `place_activation` -> `grid_code` ->
`GridCell.rate` runs on all 31 `observe` calls (`:164-165` execute before `:166` throws), so the
hexagonal population code is computed every perceiving tick. But its output reaches nothing, because
its consumer throws one line later, and the two capabilities that make `grid_cells.py` a *grid*
module rather than a static positional basis -- velocity path integration (`GridCell.step`,
`grid_cells.py:71`) and multi-scale disambiguation (`GridModulePopulation`, `:89`) -- have no caller
anywhere outside `run_selfchecks.py`. Scored 21, not at place's level and not at a fresh 40: it is the
same serial chain, and crediting it twice is the defect `coverage_score.py:126-130` warns about.

### The freeze is fixed (`3902f018`) — and the bands deliberately did NOT move

The int-percept defect described above is repaired. `GridPlaceMap._key` normalizes at the class boundary,
so `_places` is uniformly str-keyed and the variant scan no longer raises. Measured on the same harness,
60 ticks seed 0:

- `observe` calls / raises: 31 / **28** -> 31 / **0**
- `place_at` calls: **3** -> **31**
- distinct `_place_here`: **2** (`'6@1'` frozen for 51 ticks) -> **24**, tracking percepts to tick 59
- learned `(place, action)` pairs: 7 -> **27**

(Deliberately not a markdown table: `coverage_score.py`'s row parser reads any pipe-delimited line in this
file as a faculty row, and a second table here is refused as a malformed one. F3 caught exactly that on the
first draft of this section.)

`st_place_60.txt` was **re-recorded** at `90e27f57`. That is a deliberate exception to this ledger's
no-re-recording rule and it is not the forbidden kind: the two unreproducible artifacts stay un-re-recorded
because their numbers MOVE ON THEIR OWN, whereas this one changed because committed code changed, and the
old transcript documented a faculty that no longer behaves that way. The probe's verdict and exit code are
unchanged (`CONSUMED-BUT-NOT-DECISIVE`, exit 1).

**Both bands stay at 21, and the reason is rule 2.** The rationale that justified place's 21 — "the faculty
runs three times and then stops, silently" — is now FALSE, and a stale justification under an unchanged
number is its own defect, so it is corrected here rather than left to read as still true. But correcting a
rationale is not evidence for a band. Moving into 41–60 ("changes an internal read") requires a runtime
neuter A/B showing that removing place identity alters something the brain later reads. What exists is a
before-fix/after-fix comparison, which is a different experiment: it compares two versions of the code, not
a neutered signal against an intact one. The neuter that HAS been run still changes no arm decision, which
is the 61–80 test failing, not the 41–60 test passing. **The measurement that would settle it:** neuter
`_place_here` (pin it to a constant) on the fixed code and measure whether the learned `(place, action)`
table and the drive term differ. Until someone runs that, 21 is what the evidence supports.

**The `place_coding` / `grid_coding` duplicate is NOT collapsed, and the arithmetic is why.** These bands
are 35,35,35,30,30,21,21,21,15,10,10 — sum 263 over 11 rows, mean 23.9. Dropping one 21 as a duplicate
gives 242 over 10 = **24.2**. Removing a duplicated row RAISES this category's score, because the removed
band sits below the mean. That makes de-duplication a way to score higher, which inverts the purpose of the
W3/W5 checks entirely: they exist to stop a serial chain being credited twice, not to hand out points for
admitting it was. Collapsing here would also mean re-banding the merged faculty on the post-fix behaviour,
which is exactly the measurement that has not been run. So the duplicate stands, documented, until a
category re-audit resolves the roster and the band together. **This is a known inflation, disclosed, and it
inflates by at most the 0.3 points shown above.**

**One more limit this exposed, worth stating because the tool is silent about it.** After the fix,
`--rerun st_place_60` reported `reproduces` — the fingerprint compares the two neuter flip counts, and those
did not move. Two un-fingerprinted lines did: learned pairs 7 → 27 and source weight 0.107 → 0.1136. So a
probe whose measured substance changed materially passed the forgery check unchanged. This is the same hole
the widened `_fingerprint` regex closed one layer up, surviving one layer down: `rerun` compares the
observables a probe happens to print in `N/M` form and nothing else. **`reproduces` means "the flip counts
match", not "the measurement is unchanged."**

**path_integration_dead_reckoning** is the healthiest faculty in the category. `AllocentricMap.move`
runs on 57/60 ticks with **31 distinct** positions, dead-reckoned from the arm's own end-effector
forward kinematics (`autonomous_loop.py:503-531`, proprioceptive reafference rather than efference
copy, so a safety-refused command contributes zero). It is genuinely consumed by another module's
state -- `brain.py:1917-1924` differences it into a velocity and steps the spiking CAN with it -- which
is why it is at the top of the unproven band rather than in telemetry-only territory. Two honest
caveats. First, `AllocentricMap.observe` is called with `str(action)` as the landmark id and
`(intero, threat)` as the coordinates (`autonomous_loop.py:1596-1598`), so `map_nearest` returns
`'JOINT_B_ROTATE(+6deg)'`: the landmark *map* is not a map of space, only the dead-reckoned *position*
is spatial. Second, and this is a measurement obstacle rather than a defect: `charge_probe`'s
trajectory observable **is** this faculty's output (`brain.py:2785-2787`, `map_position` =
`arm._allo.position`), so no A/B built on that observable can test it -- the instrument and the
subject are the same variable. Raising this above 40 needs an A/B on a different observable.

**path_integration_spiking_can** is live and varying: `step`+`decode` produce `_grid_can_xy` on 56/60
ticks with **29 distinct** decoded positions, integrating the allocentric velocity in a recurrent bump.
The prior sweep's note that "decode has no established consumer" is **half refuted**: `brain.py:1305`
does read it, and on the three ticks the place binding survived, the place code was grounded on the
bump (`_place_from_spiking_bump` True). But that consumer is the same one the `observe` crash disables,
so from tick 9 onward the decode reaches only `state()["grid_can_xy"]`. Live, varying, telemetry in
effect: 30.

**landmark_drift_correction** runs and corrects nothing. `observe_landmark` is called 3 times in 60
ticks (once per surviving `place_at` hit), raises no exception, and returns **exactly 0.0 every time**
(no rounding involved -- the raw returns are 0.0, and `last_capture` was False on all three). Two of
the three were first sightings, which imprint and return 0.0 by contract
(`spiking_grid_can.py:253-265`); the single re-sighting mixed a landmark anchor imprinted from the
current bump into that same bump, so the correction was genuinely nil. `state()["place_correction"]`
is therefore a constant 0.0 in every live run, indistinguishable from the value it is initialised to.
The drift-correction mechanism has never applied a nonzero correction on a live tick: 21.

**interval_timing_time_cells** is live and varying -- `TimeCellClock.tick` gives `_elapsed_time` on
57/60 ticks with **27 distinct** values (max 5.17), `learn_interval` fires on rewarded onsets and
learned latencies in {1, 2}, and `elig["timing"]` at `brain.py:1526` enters the undirected drive sum.
`st_timing_60.txt` is exit 2 `UNKNOWN`, not a negative: an undirected source has no value table, so
`charge_probe` correctly refuses to say whether it fired. What pushes it to 35 rather than 40 is the
brain's own verdict: after 60 ticks the three-factor rule has driven `_drive_w["timing"]` to **0.010**,
the lowest of all eleven weights (novelty 0.178, ignition 0.190, place 0.107) -- roughly 1/19 of
novelty's influence. The clock also reaches a chat reply at `brain.py:3134-3136`, but that append is
gated on the user asking "how long", so it never fires on an autonomous tick, and an f-string
interpolating a value into a reply is the prose form of a telemetry field, not the changed-decision the
41-60 band describes. I did not claim it.

**interval_timing_wake_counter** is a *second*, redundant interval-timing mechanism in the same class:
a plain integer incremented at `brain.py:1801` and reset per wake episode at `:2590`, live and varying
(13 distinct values, max 12), reaching only `state()`. Its companion field `interval_weber_sd` is
`0.15 * self._interval_timer` (`:2793`) -- a hardcoded coefficient presented as a scalar-timing Weber
SD, next to a module (`time_cells.py`) that computes an emergent one. This is a P1 hardcode and a P2
duplicate; I score it 30 (live, varying, telemetry-only) rather than discounting it, and flag the
disposition rather than deciding it.

**sequence_binding_synfire** is live and consumed: `_bound` is True on **37/60** ticks under real
perception (4/60 with perception off -- another number the src-only harness would have understated),
and `brain.py:2060-2070` uses it to gate whether the Kuramoto coherence lowers
`_reentrant.ignition_threshold`, which the workspace then reads. That is another module's internal
state, so 35. Honest scope, and it is the reason this is not the category's "sequence" answer: the
chain is used as an all-or-none synchrony *reliability test* (`survives(a0, sigma0)`); its temporal
ORDER is never read, so it binds, it does not represent a sequence.

The last three rows are **static evidence only** and are labelled as such in `measured_by`, because the
honest band for a static-only claim is low and saying which evidence it rests on is part of the claim:

- **sequence_memory_polychronous** -- `polychronization.py`'s `DelayNetwork` stores memory in spike
  timing (Izhikevich PGs), the closest thing in the repo to a real sequence code. Its only importer in
  `src/` is `rose_language.py:358` (cortex/ since 2026-08-12), reached through `brain._rose_serve()` (`brain.py:4070-4082`), a
  lazily-built chat-path accessor. Live path, but not the wake tick: 15.
- **sequence_prediction_sorn** -- `sorn.py` self-organises a recurrent net that beats a static
  reservoir on sequence prediction. No importer anywhere in `src/`; referenced only by
  `run_selfchecks.py:286`. Built, not live: 10.
- **heading_integration_ring_attractor** -- the repo's 1-D CAN velocity integrator and its
  `ProprioceptiveEstimator`. Imported only by `brainstem/live_arm.py:57`, inside `_make_observer()`,
  behind an off-by-default `--proprioception` flag on a separate CLI entrypoint. It is not on the
  `make_brain -> Brain -> LiveBrain -> AutonomousArm` path this category's other faculties run on: 10.

## Capability inventory (Policy 2: what each file uniquely provides)

- `hippocampus/grid_cells.py` -- hexagonal grid firing fields (`rate`), velocity path integration
  (`step`), multi-scale population (`GridModulePopulation`). Only `rate` has a live caller.
- `hippocampus/grid_place_map.py` -- grid -> place read-out: one-shot Hebbian place templates,
  population-competition firing bar derived rather than hand-set, `place_at` (where -> what) and
  `recall_place` (what -> where). `recall_place` has no live caller.
- `hippocampus/spiking_grid_can.py` -- the only *network* path integrator: recurrent 2-D CAN, bump
  dynamics, landmark phase re-anchoring with cue capture, drift measurement.
- `hippocampus/allocentric_map.py` -- dead-reckoned position + valence-weighted landmark store with a
  `nearest` read-back.
- `hippocampus/ring_attractor.py` -- 1-D CAN heading/joint integrator + encoder-free proprioception
  and slip detection. Unique capability; not on the brain's tick.
- `cortex/time_cells.py` -- ramp-to-threshold population clock with emergent Weber scaling and
  reward-latency learning.
- `cerebellum/synfire_chain.py` -- all-or-none synchronous-volley propagation with a reliability
  threshold derived from its own width/window.
- `reasoning/polychronization.py` -- combinatorial memory keyed on relative spike TIMING.
- `reasoning/sorn.py` -- plasticity-driven sequence prediction above a static-reservoir control.
- `cerebellum/wavefront_planner.py` -- configuration-space collision-free route planning. Considered
  and **left out of the table**: spatial route planning is not one of the four sub-areas the rubric
  names for this category (`COVERAGE_RUBRIC.md:32`), and I did not want to widen the roster on my own
  authority. It is imported only by `live_arm.py:194` behind an off-by-default flag, so it would score
  10 and lower this mean; a later auditor adding it with a stated reason is welcome.
  `reasoning/delay_learning.py` was excluded on the same basis: it learns a delay *parameter*, it does
  not represent a sequence.

## Gaps, ranked by distance_to_100 x feasibility

1. **Fix `grid_place_map.py:166` to compare keys as strings** (or make `brain.py:1310` pass
   `str(digit)`). One line. It unblocks place binding on ~28 of 31 perceiving ticks, un-freezes
   `_place_here`, gives the R-STDP place-value a real place to key on, restores the CAN's only live
   consumer, and gives `observe_landmark` re-sightings that can actually correct drift. Four rows in
   this table are held down by it. Highest value per line in the category by a wide margin.
2. **Add the repo root to `tools/charge_probe.py`'s `sys.path`.** One line, and it makes every future
   charge probe measure the daemon's brain instead of a blind one. Until then the ledger's negatives
   are ambiguous.
3. **Replace `charge_probe`'s leave-one-out neuter with a graded or grouped intervention.** The 47/60
   vs 0/60 gap shows a redundant sum defeats single-source ablation. Without this, no drive-routed
   faculty in this brain can earn band 61 again, whatever it does.
4. **Give the place code more than two places.** Even repaired, the live environment bound exactly two
   identities (`'6'`, `'6@1'`) over 60 ticks, so band 81 (generalisation) is not reachable by wiring --
   it needs a stimulus stream that visits distinguishable locations.
5. **Wire one real sequence faculty.** `polychronization` and `sorn` are the two genuine sequence
   mechanisms and neither is on the tick; `synfire` is on the tick but is used as a synchrony gate. The
   category's "sequence" sub-area is currently answered by nothing that represents temporal order.

## What I could not measure, and what this score is not

- **No band above 40 is claimed, and none is available.** The only A/B route the instrument accepts
  for this category is a `_drive_w` source; the two spatial/temporal ones measured exit 1 and exit 2
  respectively, and the pathway control shows a single-source ablation cannot resolve charge in this
  regime at all. This is an honest null for the 41+ range, not an unfinished measurement.
- **The three static-only rows are labelled static-only.** I did not run `polychronization`, `sorn` or
  `ring_attractor` inside a live brain, so their bands rest on importer traces. Given how badly static
  reading has performed in this repo, treat 15/10/10 as upper-bounded by evidence quality, not as
  refuted claims.
- **The artifacts do not record which venv ran them.** `coverage_score.py record` writes `python:
  3.11.15` but not the interpreter path, and this category's whole finding is that the *environment*
  decides whether the faculty fires. Mine were all `.venv-lake` (torch 2.13.0+cpu, numpy 2.4.6) at
  commit `8b6aff77`.
- **`st_place_60.txt` and `st_timing_60.txt` share the observable fingerprint `(0/60, 0/60)`**, so the
  scorer will raise W3 ("probably one serial chain scored as 2 faculties"). It is a false positive
  here: the shared fingerprint is the *null* fingerprint, which two unrelated faculties reach whenever
  neither moves the arm. W3's heuristic is aimed at shared *positive* fingerprints. I left both
  artifacts attached rather than changing a tick count to dodge the warning.

  **Adjudicated 2026-07-30, independently: FALSE POSITIVE CONFIRMED, both bands stand.** The claim
  above was checked rather than taken, by comparing the two artifacts' **full stdout** instead of their
  fingerprints -- which is the whole point, since the fingerprint drops every line that distinguishes
  them. They pin different `_drive_w` keys (`place` at `brain.py:1575-1576`, action-specific through
  `_place_value.get((place, a))`; `timing` at `brain.py:1526`, entering the undirected sum with no
  value table), they live in different modules (`hippocampus/grid_place_map.py` vs
  `cortex/time_cells.py`), and the recorded stdout differs on three lines the fingerprint cannot see:
  `st_place_60.txt` carries `(place, action) values learned from dopamine : 7 pairs` and weight
  `0.107` and **exit 1 CONSUMED-BUT-NOT-DECISIVE**; `st_timing_60.txt` has no value-table line at all,
  weight `0.01`, and **exit 2 UNKNOWN**. The two bands also rest on separate runtime instrumentation
  recorded above -- place's 31 `observe` / 3 `place_at` asymmetry versus timing's `_elapsed_time` on
  57/60 ticks with 27 distinct values -- so neither is a second reading of the other's measurement.
  Both rows are at or below 40 (35 and 21) and F8 only demands an exit-0 artifact above 40, so no band
  in this pair draws anything from the colliding fingerprint. RUNTIME for the artifact contents (both
  are recorded probe runs), STATIC for the two key traces.

  Separately, and **not** what W3 fired on: the genuine same-chain pair in this file is
  `place_coding` and `grid_coding`, declared as such in the rule-4 analysis above ("it is the same
  serial chain, and crediting it twice is the defect `coverage_score.py:126-130` warns about").
  W3 cannot see it because `grid_coding` has no artifact. It is left as scored -- the two rows were
  banded on different measured facts (place's binding freeze versus the grid population code being
  computed on all 31 perceiving ticks) -- and flagged here so a later sweep resolves it on evidence
  rather than on the tool's silence. Absence of W3 is not evidence of absence.
  **Revisited 2026-07-30 and still left as scored, for a NEW reason:** collapsing the pair would raise
  this category from 23.9 to 24.2, because the removed band is below the mean. See "The freeze is fixed"
  above -- de-duplication must not be a route to a higher score, and the post-fix re-band it would
  require has not been measured.
- **One tick count, one seed.** 60 ticks at seed 0 throughout, chosen to match the control the ledger
  already cites. `coverage_score.py`'s own README names tick-count shopping as a live hole
  (`drive 4` exits 1, `drive 60` exits 0); I did not go looking for a longer run that would flip a
  verdict, and I also did not run long enough to rule one out.
- **The withdrawn baseline scored this category 36** (`COVERAGE_BASELINE.md:34`, marked "not
  verified"). 23.9 is lower. Per rubric rule 5 that is the instrument working, not a regression: the
  36 had no committed operands, so the two numbers are not on the same scale and the delta is not a
  measurement. Note also that the row this file replaces cited `COVERAGE_BASELINE.md:18`, which is now
  a blank line -- the citation drifted when the withdrawal banner was added, exactly the failure
  `coverage_score.py:133-148` says it cannot catch.

## Files this audit wrote

- `docs/coverage/SpatialTemporal.md` (this file).
- `docs/coverage/artifacts/spatiotemporal_probe.py` -- `charge_probe`'s verdict on the daemon
  `sys.path`. Deviation from "artifacts only": a `.py` under `artifacts/`, because the harness fix
  belongs in `tools/charge_probe.py` and that file was owned by other agents this session.
- `docs/coverage/artifacts/spatiotemporal_pathway_ab.py` -- the all-source / leave-one-in harness
  control. Its exit code is about harness sensitivity and must never be cited as a faculty charge.
- Recorded artifacts: `st_drive_control_60.txt` (exit 1), `st_place_60.txt` (exit 1),
  `st_timing_60.txt` (exit 2), `st_pathway_control_60.txt` (exit 0, harness control).
