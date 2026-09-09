# Tasks: Vectorize the scalar paths

**Input**: spec.md (FR-004/FR-005, SC-009-SC-012 and its amendments), plan.md, research.md.
**Prerequisites**: pinned measurement host; `.venv-lake` cp314t; `substrate/perf_baseline.py` recorded.
**Why this file exists**: plan.md declares the omission -- it orders Gate Zero FIRST and this spec had no
`tasks.md` to land it in. Created 2026-09-09 by specs/022 S6, which built Gate Zero and read it.
**Discipline (every task)**: one commit, registration-complete (REGIONS + `run_selfchecks` where
module-shaped), full gate green; every conversion carries FR-004's measurement at the site's LIVE width and
every refusal FR-005's; the `tools/` locations plan.md names are refused by the p4c ratchet since specs/022
T001, so instruments land in `src/snn_data_lake/substrate/`.
**Mapping to plan.md's Implementation Sequence**: T001-T002 = P1 item 1; T003 = P2 item 7; T004 = P1 items
2-4 plus P2 items 5-6; T005 = P3 items 8-10.

- [X] T001 Gate Zero -- `substrate/amdahl_gate.py` LANDED 2026-09-09 (`b8413c1`; its reds RUN RED first at
  `2c3c97e`, `1 error in 0.10s`, a collection ImportError). Neuter-and-measure the Amdahl ceiling of a
  delivery-path conversion over a live `make_brain(seed, persist=False)` in a scratch home: paired warm wake
  ticks, W = 150 per arm, p50 with a 1,000-resample bootstrap band, raw AND reference-corrected (raw decides
  when they disagree), plus N* derived two ways. THREE CONTROLS HELD in the same command (a padded-share
  brain must read a LARGE ceiling -- the positive control, 0.87; a repeatability pair; a sham neuter on a
  zero-share path must read within noise of zero); `run_selfchecks` 353/353, every gate rc 0. MEASURED at
  the deployed width (177-189 live LIF units): ceiling 0.005-0.017 raw, 95% bands inside 0.04 on seeds 0
  and 1; seed 2 INDETERMINATE twice (its band, +/-0.2, straddles the 0.15 bar).
  **CORRECTED 2026-09-09 (panel fold, C10):** this row said "Three self-check reds fire their broken arms
  in the same command". They did not. 2.2(iv) constructs exactly ONE perturbation and only padded-share is
  built from it -- repeatability is a self-consistency pair with nothing perturbed and the sham is a
  negative control -- so NO criterion was shown FAILING in that command, which now says "3 controls HELD"
  and says so. The padded-share criterion IS seen refusing at
  `tests/test_amdahl_gate.py::test_the_instrument_blind_disposition_refuses_the_reading` (`pad_iters=0`),
  and the repeatability criterion now has the one broken arm it lacked: `red_repeatability_broken_arm`
  re-reads it with its second reading on the PADDED rig, prints it as a fourth `--self-check` line and
  asserts DISJOINT bands -- measured control 0.0416 band (0.03318, 0.04845) against broken 0.8731 band
  (0.87196, 0.87406). Its outcome is kept out of the exit code: it demonstrates the criterion can fail, it
  is not a control on a reading.
  **CORRECTED 2026-09-09 (panel fold, C8 + C12):** the sentence "The instrument's floor is ~0.008, so the
  honest reading is 'the removable cost is BELOW WHAT THIS INSTRUMENT RESOLVES'" that stood here is CUT.
  The sham arm's magnitude is a PER-RUN quantity by the instrument's own definition ("the sham arm removes
  NO work, so whatever it reads is the instrument's own floor"), not a property of the instrument: over
  the five recorded readings it runs -0.0009 / +0.0005 / -0.0078 / +0.0185 / -0.0140, and the gate's own
  "ceiling exceeds 2x the floor" flag prints True on 3 of those 5 -- so the licensed conclusion fails on
  the majority of the readings it was written over, and ~0.008 is only seed 1's own draw. "95% bands
  inside 0.04" is likewise WITHDRAWN: it is a property of ONE unlogged reading (the seed-0 quiet extra,
  band upper 0.0383), and five of six fresh determinate bands breach it. What reproduces, and what the
  DECLINE rests on: at this width the ceiling is BELOW WHAT THE INSTRUMENT RESOLVES -- the point estimate
  is not stable across readings (six panel re-readings of the same three seeds read -0.03 to +0.03, none
  of them inside 0.005-0.017) and is not quotable as a number, while every determinate 95% band's upper
  bound sits far under the 0.15 bar. No replacement constant is published. Per-reading table:
  `../022-unified-brain-arc/evidence/s6-kernel-wire.md` 3.2; the per-finding entries are its 3.7
- [X] T002 The kernel-wire conversion (`KernelTick._deliver_and_rotate` as the live tick's delivery)
  DECLINED AT THE DEPLOYED WIDTH under FR-005, 2026-09-09, with the measurement recorded
  (`../022-unified-brain-arc/evidence/s6-kernel-wire.md`). Two determinate seeds agree; the bar is
  `perf_baseline.TOLERANCE` = 0.15, and a ceiling of 0.005-0.017 is an order of magnitude under it, so no
  conversion here could be asserted as a gain by the tree's one performance instrument. RE-ARMED at the
  measured trigger, not abandoned: `dormancy_gate`'s `kernel_tick`/`kernel_state` `wire` rows are re-dated
  to specs/022 T062 -- the row that first places units at scale, ~16k signed neurons -- or to growth
  reaching N*, a LOWER BOUND of 2,504-2,828 units by the profile derivation and 4,018-6,316 by the neuter
  derivation. The two disagree 1.5-2.5x (cProfile attributes more to the object path than the neuter
  removes); that disagreement is reported as the instrument's own finding, NOT resolved into one number.
  **CORRECTED 2026-09-09 (panel fold, C1 + C4 + C6 + C8 + C11).** (a) N* over ALL FIVE recorded readings
  is profile 2,504-2,828 (2,565 on the seed-0 quiet extra) and neuter 642-6,316 (1,788 on that same
  reading), the two 1.43-4.40x apart. The ranges above dropped seed 2 boot's 642 and 4.40 and the quiet
  extra's 1,788 and 1.43 while keeping 2,828 from that same boot reading; no single reading rule produced
  that set, and 2.9 -- written BEFORE the re-read, while 4.40 was still the adverse number -- had stated
  the disagreement as 2.4-4.4x. (b) Neither N* is a second instrument. N*_profile = K*((1-s)/s)*N and
  N*_neuter = K*((1-c)/c)*N with K = tol/(1-tol), exactly, so the neuter's N* is that reading's ceiling
  restated in width units and INHERITS its 95% band -- unbounded above on every determinate seed (seed 0
  boot maps to [2,049, +inf), seed 1 to [2,045, +inf)) -- while the profile's N* has no interval at all,
  because no bootstrap exists over profiler tottime. Seed 2 boot's 642 is the same formula at its
  INDETERMINATE ceiling 0.049, not a different regime. (c) The stated cause is withdrawn without a
  replacement: "cProfile attributes more to the object path than the neuter removes" is contradicted by
  the record's own quiet reading, where cProfile inflated R by 1.65x (170.82/103.71) against c_u by 1.15x,
  and on that reading the neuter's N* sits BELOW the profile's, inverting the stated direction. Part of
  the gap is real removable cost the neuter leaves behind rather than profiler overhead (see C5). (d) "a
  LOWER BOUND" becomes a POINT ESTIMATE at one measured width, its bound direction UNTESTED: c_u is
  measured at ONE width and its width-invariance is untested, so the lower-bound direction holds only
  while the object path grows at most linearly in N. (e) "a ceiling of 0.005-0.017 is an order of
  magnitude under it" -- the ORDER-OF-MAGNITUDE margin reproduces on every reading ever taken and the
  DECLINE with it (6/6 fresh readings read DECLINE); the POINT RANGE does not, so the reproducible
  statement is "below what the instrument resolves at this width, every determinate band upper far under
  the 0.15 bar". (f) T062 PLACES NO UNITS: PLAN-FINAL scopes its H01 ingest to the c3 SYNAPSE GRAPH
  composed onto `cortical_prior.py`, its decision table reads "placement has no reader", and its own
  same-arc dated note names GROWTH as the placer; ~16k is a measured ingest milestone only. T062's own
  pre-registration does define an ingested population, but OFF the live tick -- the re-arm condition is
  THE FIRST TICK POPULATION AT OR ABOVE N*, which today only growth can produce. The DECLINE, the bar and
  the disposition are unchanged
- [ ] T003 Measure the connectivity law: nonzero synapses per neuron at THREE OR MORE population widths,
  the functional form fitted rather than assumed. OWED BEFORE any ceiling in this spec's amendments is
  quoted -- constant fanout and fraction-of-width differ by orders of magnitude, and plan.md's Complexity
  Tracking table refuses to commit the tiering design until this number exists
- [ ] T004 The T0-T3 ladder over the remaining hot sites the profile names, each CONVERTED or DECLINED with
  its own measurement attached (FR-004 / FR-005, never judgement): `prefrontal_executive._decide_spiking`
  17.9 %, `nt.fsync` 17.1 %, `eprop_kernel.forward` 10.4 %, `builtins.sum` 9.7 % (904,324 calls) -- shares
  from cProfile over 150 wake ticks, 17.816 s tottime, on the pinned host at wake p50 108.663 ms. T0
  classification first: integer-reduction and elementwise sites need no A/B, transcendental and
  float-reduction sites take the full ladder. `nt.fsync` is the lake, not arithmetic, and takes a
  disposition of that shape rather than a vectorization
- [ ] T005 [P3, GATED ON N* REACHED -- not startable at the deployed width] The GPU tier: a CSR int32
  synaptic matrix + dense f32 spike vector, post-major, one SpMV, chunked construction at 2e7-2e8 nnz;
  accumulation that avoids atomics so the path stays BIT-STABLE; and its OWN golden-output reproducibility
  gate, separate from the CPU gate -- neither proves the other and no gate may be written as if it does
  (SC-009-SC-012; the CPU gate stays authoritative and the suite stays runnable with no GPU). No CUDA arm
  exists to gate today: `substrate/device.py::select` takes a free-form workload string and answers "cpu"
  at every live width, and `KernelTick` is pure numpy

**Dependencies**: T001 precedes everything (plan.md: "Without it, no conversion has evidence"). T002 is
T001's recorded outcome. T003 gates every quoted ceiling, T005's VRAM claim included. T005 gates on N*.
