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
  when they disagree), plus N* derived two ways. Three self-check reds fire their broken arms in the same
  command (a padded-share brain must read a LARGE ceiling -- the positive control, 0.87; repeatability; a
  sham neuter on a zero-share path must read within noise of zero); `run_selfchecks` 353/353, every gate
  rc 0. MEASURED at the deployed width (177-189 live LIF units): ceiling 0.005-0.017 raw, 95% bands inside
  0.04 on seeds 0 and 1; seed 2 INDETERMINATE twice (its band, +/-0.2, straddles the 0.15 bar). The
  instrument's floor is ~0.008, so the honest reading is "the removable cost is BELOW WHAT THIS INSTRUMENT
  RESOLVES" -- which supports the decline more strongly than the point estimate does
- [X] T002 The kernel-wire conversion (`KernelTick._deliver_and_rotate` as the live tick's delivery)
  DECLINED AT THE DEPLOYED WIDTH under FR-005, 2026-09-09, with the measurement recorded
  (`../022-unified-brain-arc/evidence/s6-kernel-wire.md`). Two determinate seeds agree; the bar is
  `perf_baseline.TOLERANCE` = 0.15, and a ceiling of 0.005-0.017 is an order of magnitude under it, so no
  conversion here could be asserted as a gain by the tree's one performance instrument. RE-ARMED at the
  measured trigger, not abandoned: `dormancy_gate`'s `kernel_tick`/`kernel_state` `wire` rows are re-dated
  to specs/022 T062 -- the row that first places units at scale, ~16k signed neurons -- or to growth
  reaching N*, a LOWER BOUND of 2,504-2,828 units by the profile derivation and 4,018-6,316 by the neuter
  derivation. The two disagree 1.5-2.5x (cProfile attributes more to the object path than the neuter
  removes); that disagreement is reported as the instrument's own finding, NOT resolved into one number
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
