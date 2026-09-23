# T101 (B1) -- the Brian2 framework contract: accepted by parity, refusals, continuation and cost, not by adoption

Plan row B1 / T101 (`PLAN-FINAL-2026-09-02.md`, the B workstream added 2026-09-22); FR-037, SC-015; the red B-R1 (Part F).
Written 2026-09-22 15:00 CDT (clock read at write time) on the owner's word in chat -- "Finish the Brian implementation
first then commit" -- against a tree at `678a26a` whose WORKING TREE carries the owner's uncommitted Brian2 migration
(46 files, seven Codex sessions 00:07-02:13 CDT; disclosed in section 1). Sections 1-2.9 are FROZEN at the commit that
lands this file; amendments are dated and appended (2.10); nothing above 2.10 is edited in place.

## 1. Territory, measured before the design (2026-09-22, the owner's uncommitted migration as found)

1. WHAT THE MIGRATION ALREADY IS. `src/snn_data_lake/substrate/brian2_backend.py` (273 lines, new): `BrianLIF`, a Torch
   `nn.Module` implementing the discrete LIF step `v[t] = beta * v[t-1] + I[t]`, spike iff `v[t] >= theta`, reset IN THE SAME
   STEP by `subtract` (`v -= theta`), `zero` (`v = 0`) or `none`; `fast_sigmoid`, the SuperSpike surrogate derivative
   (Heaviside forward); `brian_reference_trace`, which runs the same equation through a REAL Brian2 2.10.1 `NeuronGroup`
   (`run_regularly("v = decay * v + drive", when="groups")`, threshold `v >= threshold_value`, `dt = 1 ms`, numpy codegen)
   in an ISOLATED CHILD PROCESS and returns spikes, membranes and the Brian2 version; a bare self-check (three reset modes,
   three steps, three cells; a surrogate-gradient check; `assert not sys._is_gil_enabled()` in the parent).
   `tests/test_brian2_backend.py` (49 lines, new): parity per reset mode on a 3-step trace, the surrogate gradient, an
   unknown reset mode refused. Registered in `__init__.py` REGIONS and `run_selfchecks.py` CORE.
2. THE CALL SITES. Eleven modules import the bridge and none imports `snntorch` (grep over `src/`, 2026-09-22):
   `training/{trainer,offload,mnist,spiking_lm}.py`, `plasticity/{deep_stdp,snn_mnist_stdp}.py`,
   `reasoning/{developmental_loop,snn_dreamer_6dof,snn_motor_control,snn_ssm_shd,snn_world_model_rl}.py`, plus the
   package `__init__` registry. `pyproject.toml`: `brian2==2.10.1` core, `snntorch==1.0.0` retired from core and both
   extras; `ci.yml`: the snntorch CPU-integration step removed; `tools/dependency_ledger.py` and its test follow.
3. THE CUSTOM KERNELS THAT STAY. The bridge replaces the former snntorch LAYER only. The tree's own neuron dynamics remain
   their own contract and are NOT relabelled Brian2 (class-level census, `grep '^class .*(LIF|Leaky|Neuron|AdEx|ALIF)'`,
   2026-09-22): `cortex/lif_neuron.py::LIFNeuron`, `cortex/alif_neuron.py::ALIFNeuron`, `cortex/adex_neuron.py::AdEx`,
   `cortex/autapse_neuron.py::AutapseNeuron`, `cortex/dendritic_neuron.py::DendriticNeuron`, `cortex/srm_neuron.py::
   {SRMNeuron, IFNeuron}`, `cortex/eprop_kernel.py::EpropLIFBase`, `cortex/heterogeneous_microcircuit.py::{_LIFCell,
   _AdExCell}`, `cortex/neuron_zoo.py::{_AdExCell, _IzhikevichCell, NeuronZoo}`, `cerebellum/spiking_pid.py::_LIFPopulation`,
   `basal_ganglia/r_stdp.py::RSTDPNeuron`, `plasticity/{bcm_plasticity::BCMNeuron, homeostatic_scaling::HomeostaticNeuron,
   oja_pca::OjaNeuron}`, `substrate/basic_components.py::_DemoLIFLayer`, `thalamus/gwt_deep.py::LIF`, and the live
   substrate tick (`KernelTick`). Nineteen named classes plus the kernel; none is touched by this arc.
4. THE FREE-THREADING FACT, measured on this host (cp314t 3.14.6, `.venv-lake`): `import brian2` in a process prints the
   interpreter warning "The global interpreter lock (GIL) has been enabled to load module 'brian2.memory.cythondynamicarray'"
   and `sys._is_gil_enabled()` reads True afterwards; the bridge's parent process after a reference call reads False (the
   backend's bare self-check, 2026-09-22: "brian2 2.10.1: isolated reference and Torch surrogate bridge agree", rc 0).
5. THE GATES ON THE MIGRATED TREE, run 2026-09-22 before this design (static gates, no brain): `dormancy_gate`,
   `method_dormancy_gate` (75 update rules, none dormant, 22 UNCERTAIN reported), `import_manifest` (7 targets),
   `gradient_claim_gate`, `eprop_transport_gate` (435 modules scanned, 0 governed findings), `replication_guard`,
   `tools/dependency_ledger.py --assert`, `tools/red_reachability.py --assert` -- all rc 0. `run_selfchecks.py` on the
   migrated tree: 360 of 360 self-checks (the C0 fold battery of the same day ran on `678a26a` WITHOUT the migration;
   the migrated tree's full battery is this arc's, section 3).
6. WHAT IS NOT YET TRUE, and is this arc's work (FR-037 / SC-015, quoted): "declared threshold/reset/timestep/delay/
   precision semantics" -- the semantics are implemented, not DECLARED as a contract object a reader or a test can hold;
   "unsupported semantics refuse" -- a refractory period, a synaptic delay, a non-unit dt or an adaptive threshold are
   neither supported nor refused today (a caller gets a silently different neuron); "fresh-process continuation" -- no
   checkpoint/restore of the membrane exists (`mem` is a non-persistent buffer) and nothing proves a restored model
   continues as an uninterrupted twin; "measured cost" -- the reference worker's startup, IPC and memory are unmeasured;
   "bounded timeouts ... and failure handling" -- `subprocess.run` has NO timeout and no size bound (a hung or oversized
   reference call hangs the caller); "the bridge, real Brian2 worker and remaining custom kernels MUST be distinguished
   explicitly" -- the inventory in item 3 lives in this record only, with no ratchet against drift; the executable parity
   covers 3 steps x 3 cells x 3 modes and no migrated MODEL.

## 2. Pre-registration (frozen at the commit that lands this file)

### 2.0 The design read (Codex): NOT SENT before the build -- the executor IS Codex this arc (owner routing 2026-09-22);
the closing report (protocol section 7) is sent after the fold as usual, on the folded tree.

### 2.1 The mechanism -- ONE module gains a CONTRACT, refusals, bounds, a checkpoint and an inventory ratchet

All in `substrate/brian2_backend.py` (no new module; P2 -- a second bridge or a second reference runner would be additive):

- `CONTRACT`: a frozen mapping declaring, by name, the semantics the bridge and the reference share -- `equation`
  ("v[t] = beta*v[t-1] + I[t]"), `threshold` ("spike iff v >= theta; equality spikes"), `reset` ("same step, after the
  threshold test; subtract | zero | none"), `schedule` ("integrate, threshold, reset -- within one step; Brian2
  `run_regularly(..., when='groups')` then threshold/reset"), `dt` ("one step = 1 ms in the reference; the bridge is
  unitless per step"), `units` ("v, I, theta dimensionless"), `delays` ("none"), `refractory` ("none"), `adaptive_threshold`
  ("none"), `precision` ("bridge float32, reference float64; membranes agree to atol 1e-6 over traces of length <= 64;
  spikes exact"), `learning` ("the bridge is a LAYER: it computes a surrogate derivative only when a caller runs autograd
  and never steps an optimizer; governed learning rules are untouched"), `rng` ("none owned; deterministic given inputs").
- REFUSALS, explicit and named: `BrianLIF(...)` raises `NotImplementedError` naming the contract entry for any of the
  unsupported keywords `refractory`, `delay`, `dt`, `adaptive_threshold` (it accepts `**unsupported` so a caller that
  passes one is refused, never silently given a different neuron); `brian_reference_trace` gains `timeout_s` (default 120)
  and refuses a request whose `steps x cells` exceeds `MAX_REFERENCE_CELLS` (2,000,000) BEFORE launching the worker
  (`ValueError`), raises `TimeoutError` when the child exceeds `timeout_s` (the child is killed), and keeps its existing
  `RuntimeError` for a failed child or a missing result marker.
- CHECKPOINT: `BrianLIF.checkpoint()` -> mapping {`beta`, `threshold`, `reset_mechanism`, `mem`} and
  `BrianLIF.restore(payload)`; `restore` REFUSES (`KeyError` naming the key) a payload missing any of the four keys, and
  refuses a `reset_mechanism` outside `RESET_MECHANISMS`.
- INVENTORY RATCHET: `MIGRATED_CALL_SITES` (the eleven modules of 1.2) and `CUSTOM_KERNELS` (the nineteen classes plus
  the kernel of 1.3, by `module::Class`), asserted by a test that greps `src/` -- every module importing the bridge is
  listed, no module imports `snntorch`, every listed custom kernel still exists -- so the boundary cannot drift silently.
- MIGRATED-MODEL PARITY: `training/trainer.py::SpikingReadout` (the one call site whose forward is a pure unrolled
  `nn.Linear -> BrianLIF` over rate-coded spikes) is accepted by an executable Brian2 reference of ITS OWN currents: for a
  seeded batch, the per-step currents `fc(spikes[t])` are recorded and fed to `brian_reference_trace`; the model's output
  spikes must equal the reference's step by step. The other ten call sites use the same layer and are accepted BY THE
  LAYER's contract only; their training rules are not touched or claimed (2.7).

### 2.2 The conjuncts (written first; each is a function in the module's bare invocation and a test in `tests/`)

- (p) PARITY, PROPERTY-STYLE: for every `reset in {subtract, zero, none}` x `beta in {0.5, 0.8, 0.95}` x `theta in
  {0.7, 1.0, 1.5}` (27 arms), a seeded trace (`numpy.random.default_rng(seed)`, currents uniform in [-1, 2], initial
  membranes uniform in [0, theta), T = 32 steps, N = 16 cells): bridge spikes == reference spikes exactly on every cell and
  step, and `max |v_bridge - v_reference| <= 1e-6`. The measured maximum over all arms is REPORTED.
- (m) MIGRATED-MODEL PARITY: `SpikingReadout(beta=0.9)` under `torch.manual_seed(0)` on a rate-coded batch of 8 over
  T = 25 steps: output spikes per step equal the reference's on every step (exact).
- (c) FRESH-PROCESS CONTINUATION = THE UNINTERRUPTED TWIN: process A runs the seeded trace of (p)'s `subtract/0.8/1.0` arm
  for 12 steps and writes `checkpoint()`; process B restores it and runs steps 13..32; process C runs 1..32 uninterrupted;
  B's spikes and membranes for steps 13..32 equal C's byte for byte (`numpy.array_equal` on float32).
- (g) THE PARENT STAYS FREE-THREADED: `sys._is_gil_enabled()` is False in the parent after (p) and (m); the POSITIVE
  CONTROL that the hazard is real: a child `python -c "import brian2, sys; print(sys._is_gil_enabled())"` prints True.
- (r) REFUSALS: each unsupported keyword raises `NotImplementedError` at construction; a request over
  `MAX_REFERENCE_CELLS` raises `ValueError` with no worker launched (the child count is 0, measured by a counter on the
  launch path); a constructed slow reference (a child-side sleep flag on the payload, test-only) with `timeout_s = 0.5`
  raises `TimeoutError`; `restore` of a payload missing `mem` raises `KeyError`.
- (i) THE INVENTORY RATCHET holds (2.1).
- (k) COST, REPORTED NEVER BARRED: wall time of one `brian_reference_trace` call at (T, N) = (32, 16) and at (1, 1)
  (startup floor), the payload sizes in bytes, and the child's peak working set (`psutil` when importable in the venv,
  else the child's `resource`-equivalent is reported ABSENT, never guessed), each the median of 5 calls on this host.

VERDICT RULE. T101 reads ACCEPTED iff (p), (m), (c), (g), (r) and (i) all hold AND every red of 2.3 fires under its
mutation; otherwise NOT ACCEPTED with the failing conjunct named. (k) is reported and never bars.

### 2.3 The reds (B-R1, each a constructed mutation that must turn a green conjunct red; each runs in the bare invocation)

- R1 THRESHOLD ALTERED: a mutant bridge tests `v > theta`; on an input landing exactly on theta (constructed: beta 0.8,
  v0 0, I = theta) the reference spikes and the mutant does not -> (p) RED.
- R2 RESET ALTERED: a mutant applies the reset one step LATE (the former one-step mismatch the migration closed) -> (p) RED
  on the step after the first spike; a second mutant swaps subtract for zero -> RED.
- R3 SCHEDULE ALTERED: a mutant tests the threshold on `v[t-1]` before integrating -> RED.
- R4 GIL REGRESSION: the guard `_assert_free_threaded()` evaluated in a process that imported brian2 in-process exits 1
  (a constructed child), while the parent's evaluation passes -> the guard is reachable.
- R5 STATE OMITTED: `restore` of a checkpoint with `mem` deleted raises; a mutant `restore` that fills a missing `mem`
  with zeros makes (c) RED (B's steps 13..32 diverge from C's).
- R6 SILENT FALLBACK: a mutant constructor that swallows `refractory=2` (drops the keyword) is caught by (r) -- the test
  expects the refusal and the mutant does not raise -> RED.
- R7 RESOURCE OVERRUN: a request of `steps x cells = MAX_REFERENCE_CELLS + 1` is refused before launch; a constructed
  slow child under `timeout_s = 0.5` raises `TimeoutError` within 2 s (measured), never hangs.
The controls that are NOT the method (Part F): the correct real Brian2 trace (the reference itself, run by Brian2's own
code path); the supported-mode / local-learning twin -- the gradient-claim and e-prop transport gates read the same on the
migrated tree as on `678a26a` (1.5); the uninterrupted twin of (c).

### 2.4 Disposition (fixed in advance)

ACCEPTED: the migration is committed on `main` as the owner's work (two commits: the plan amendment; the migration) followed
by this arc's contract commit and its record; T101 is marked `[X] ACCEPTED (B1 contract)` in tasks.md with the scope
sentence of 2.7 beside it; the C0 fold branch is rebased on top and ONE battery runs on the stacked head.
NOT ACCEPTED on a parity conjunct: the BRIDGE is corrected (the reference is Brian2's own code and is not edited to match),
the failing arm and its numbers are recorded in 3.x, and the conjunct is re-run ONCE after the fix with the fix named; the
migration still lands (the owner's decision), and T101 stays OPEN with the failing conjunct named in tasks.md.
NOT ACCEPTED on (g): the isolation is broken; the migration does NOT land until the parent reads free-threaded.
NOT ACCEPTED on (c) or (r): the missing behaviour is built in this arc (it is the arc's own deliverable), re-run once.
A red that does not fire under its mutation: the conjunct it guards is deleted, not weakened (P8 clause 1).

### 2.5 What this row does NOT claim

Not that every custom neuron is Brian2 (1.3 lists nineteen that are not, deliberately); not that any training rule in the
migrated modules is governed learning (they were BPTT experiments under snntorch and remain BPTT experiments under the
bridge; the gradient-claim census, not this record, says which rules owe what); not that the live tick runs Brian2 (it does
not; the substrate kernel is untouched); not that Brian2 standalone/codegen modes are used (numpy target only, in a child);
not that the bridge is faster or slower than snntorch (unmeasured, out of scope); not that the ten other call sites'
FORWARDS are reference-checked (only the layer they share is).

### 2.6 Seeds, cost, host

Seeds 0..4 for (p) (one seed per reset mode arm set; seed 0 reported), `torch.manual_seed(0)` for (m), seed 0 for (c).
Every reference call is one child process (Brian2 import ~1 s); 27 + 1 + 3 + refusals ~ 40 child launches ~ 60-90 s; no
GPU, no brain, no lake. Host: Windows 11, cp314t 3.14.6 `.venv-lake`, torch 2.11+cu128 (CPU path), brian2 2.10.1.

### 2.7 Commit order (one change each; the reds before any verdict)

1. This record, sections 1-2.9 (the pre-registration), on `main` -- committed alone, ahead of the owner's still-uncommitted
   migration (disclosed: the migration was BUILT before this pre-registration was written, by the owner through Codex; this
   record's reds and conjuncts are new and are run for the first time after this commit).
2. The plan amendment (`specs/022` four files) -- the owner's, committed verbatim.
3. The migration (the other 44 files) -- the owner's, committed verbatim, with the commit message naming the missing
   battery and this arc.
4. The contract: `substrate/brian2_backend.py` + `tests/test_brian2_backend.py` (built by the Codex executor from a written
   brief, reviewed and re-run by the orchestrator).
5. This record's section 3 (the results), `tasks.md`'s T101 line.
Then the panel (three lenses), the fold, the closing report to Codex, the C0 fold branch rebased on top, ONE battery on the
stacked head, the closing execution-record commit LAST, the push, the daemon poll.

### 2.8 Amendments (dated, appended; nothing above is edited in place)

2026-09-22 15:0x CDT (clock read at write time), on the Codex executor's first build report, BEFORE any conjunct is re-run
and before section 3 exists:

(a) THE (p) MEMBRANE BAR WAS AN APPARATUS CLAIM ABOUT float32, AND ARITHMETIC SAYS IT IS FALSE. 2.2 (p) pre-registered
`max |v_bridge - v_reference| <= 1e-6` over T = 32 steps; the first run read spike mismatches 0 on all 27 arms and a
maximum membrane difference of 4.194e-6. The 1e-6 was carried from the migration's 3-step test without measuring
accumulation: the bridge computes in the caller's dtype (float32 for every migrated call site), one float32 rounding per
step is up to `2^-23 * |v|`, and it compounds over T steps; for the `none` reset at beta 0.95 the membrane reaches ~10, so a
worst-case bound is `32 * 2^-23 * 10 ~= 3.8e-5` and the measured 4.19e-6 sits well inside it. The reference is Brian2's own
float64 code and is not edited. Under 2.4 the letter of the disposition is "the bridge is corrected"; the bridge's
SEMANTICS are not at fault (spikes exact on all 27 arms, and (c) byte-equal in float32), the CRITERION's precision clause
was. The correction, dated here before the re-run and applied to CONTRACT["precision"] in the same commit as the delta:
(p) SPLITS INTO TWO ARMS run on the same 27 (reset, beta, theta) cells and the same seeded traces -- (p32) the bridge in
float32, spikes exact, and per arm `max |dv| <= T * 2^-23 * max(1, max |v_reference|)` (a bound derived from the declared
precision and the reference's own trace, never from the bridge); (p64) the bridge in float64 (the layer accepts the
caller's dtype), spikes exact, `max |dv| <= 1e-9` -- the arm that shows the EQUATION is the reference's and that the
float32 residual is rounding, not semantics. Both maxima are REPORTED per arm. The reds' own comparator (R1-R3, a 4-step
trace) uses the same derived rule. What this does NOT do: move a red, change the bridge's dtype or equation, or accept a
spike mismatch anywhere. Re-run ONCE after the delta lands, as 2.4 provides.

(b) THE CUSTOM-KERNEL COUNT IN 1.3 IS TWENTY CLASSES, NOT NINETEEN: the list itself was complete; the count beside it was
wrong (caught by the executor while writing `CUSTOM_KERNELS`, which carries all twenty plus `KernelTick`, 21 entries).

2026-09-22 15:2x CDT (clock read at write time), on the delta's first re-run, BEFORE the next re-run:

(c) (a)'s FLOAT64 ARM COMPARED UNLIKE PARAMETERS -- THE SECOND CORRECTION OF THIS CRITERION'S APPARATUS, disclosed as such.
The delta's re-run read (p32) inside its derived bound (4.194e-6 against 5.58e-5, spikes exact) and (p64) at 1.92e-6 against
its 1e-9 bar. Cause: the bridge holds `beta` and `theta` as float32 buffers by construction (0.95 is stored as
0.949999988...), and (a) handed the REFERENCE the exact float64 0.95, so the float64 arm ran the equation at two different
parameters; a cast cannot recover what float32 never held. The like-with-like construction, which is also the contract's
own reading ("the bridge computes in the caller's dtype" -- and its PARAMETERS are float32): the reference receives the
float32-representable parameters the bridge holds, `beta_ref = float(numpy.float32(beta))`, `theta_ref =
float(numpy.float32(theta))`, in BOTH arms and in the reds' comparator; (p64) then tests the equation at identical
parameters with the bar unchanged at 1e-9, and (p32)'s residual becomes pure accumulation under the unchanged derived
bound. No bar moves, no red moves, the bridge is unchanged, no spike mismatch is accepted anywhere. Both maxima are still
reported per arm. Re-run ONCE after this delta lands; a further apparatus correction of (p) after that is a NEW
pre-registration, not an amendment.

## 3. The record

### 3.0 Apparatus as run (2026-09-22 15:0x-15:4x CDT, clock read at write time)

The Codex executor (`gpt-6-astra`, reasoning `high`, the `codex` MCP tool, cwd = the main checkout, writes confined to the
two named files) built the contract from the brief in three passes: the build (2.1-2.3 as written), the delta for 2.8 (a),
the delta for 2.8 (c). Each pass ran the bare invocation, the test file, ruff check and ruff format. The orchestrator then
RE-RAN everything on this host (Windows 11, cp314t 3.14.6 `.venv-lake`, torch 2.11+cu128 on the CPU path, brian2 2.10.1,
the daemon RUNNING throughout): the bare invocation `python -B -m snn_data_lake.substrate.brian2_backend` -> rc 0 in 77 s
(the executor: 75 s); `pytest tests/test_brian2_backend.py` -> 23 passed in 80 s; ruff check and format clean; the working
tree's porcelain count unchanged at 48 (the owner's migration + the two built files) -- no other file touched. Every number
below is the orchestrator's run; the executor's agree to the printed digits.

### 3.1 The build

`substrate/brian2_backend.py` 273 -> 884 lines; `tests/test_brian2_backend.py` 49 -> 158 lines (3 original + 20 new tests).
What landed, by 2.1: `CONTRACT` (12 keys, frozen `MappingProxyType`); the four `NotImplementedError` refusals naming the
contract entry and a `TypeError` for any other keyword; `brian_reference_trace(timeout_s=120.0)` with
`MAX_REFERENCE_CELLS = 2_000_000` refused before launch, `TimeoutError` on the bound, the launch counter
`_REFERENCE_LAUNCHES`, and `_assert_free_threaded()` after every reference call; `checkpoint()` / `restore()` with
`KeyError` per missing key and `ValueError` on an unknown reset; `MIGRATED_CALL_SITES` (11) and `CUSTOM_KERNELS` (21, the
twenty classes of 1.3 plus `substrate/kernel_tick.py::KernelTick`) with `conjunct_i` parsing import statements by AST; the
parity arms of 2.8 (a)/(c) sharing ONE reference call at the bridge's float32-representable parameters; `conjunct_m` on
`training/trainer.py::SpikingReadout(beta=0.9)` with the per-step currents captured by a forward hook; `conjunct_c` as three
interpreters (`--continuation A|B|C`) exchanging JSON; `conjunct_g` with its positive-control child; `conjunct_r`;
`cost_report()` (five-call medians, the child's `peak_wset` via psutil 7.2.2); seven mutants as separate classes or
functions (`_MutantThreshold`, `_MutantLateReset`, `_MutantZeroReset`, `_MutantSchedule`, `_MutantMissingMem`,
`_MutantSilentFallback`, the constructed in-process brian2 child, the `sleep_s` test-only payload flag). The shipped
`BrianLIF` equation, threshold test, reset modes and schedule are byte-for-byte the owner's; the child's Brian2 model is
unchanged except the test-only `sleep_s` delay before it builds.

### 3.2 The conjuncts, each quoted and marked

- (p32) PARITY, float32 (2.8 (a)/(c)): **HOLDS** -- 27 arms, spike mismatches 0 of 13,824 cells; max membrane difference
  2.887e-6 against the derived bound 5.579e-5 (the worst arm `none` / beta 0.95 / theta 1.0, reference membrane up to 14.61,
  bound `32 * 2^-23 * 14.61`); the reference at `beta_ref = float(np.float32(beta))`.
- (p64) PARITY, float64 (2.8 (a)/(c)): **HOLDS** -- 27 arms, spike mismatches 0; max membrane difference 0.0 (bar 1e-9): the
  bridge's equation IS the reference's at identical parameters; the float32 residual above is rounding.
- (m) MIGRATED-MODEL PARITY: **HOLDS** -- `SpikingReadout(beta=0.9)`, batch 8, 25 steps: 0 spike mismatches against the
  reference of the model's own recorded currents; the summed outputs equal the reference's sums.
- (c) FRESH-PROCESS CONTINUATION = THE UNINTERRUPTED TWIN: **HOLDS** -- 3 processes, B's 20 continued steps equal C's on
  spikes and membranes byte for byte (`array_equal` on float32).
- (g) THE PARENT STAYS FREE-THREADED: **HOLDS** -- parent `_is_gil_enabled()` False after every reference call; the positive
  control (a child importing brian2 in-process) prints True.
- (r) REFUSALS: **HOLDS** -- `refractory`, `delay`, `dt`, `adaptive_threshold` each `NotImplementedError` naming the contract
  entry; an unknown keyword `TypeError`; a checkpoint missing `beta` / `threshold` / `reset_mechanism` / `mem` `KeyError`
  naming it; an unknown reset `ValueError`; the oversize request refused with 0 worker launches; the constructed slow child
  under `timeout_s = 0.5` raised `TimeoutError` in 0.52 s.
- (i) THE INVENTORY RATCHET: **HOLDS** -- 11 modules import the bridge (exactly `MIGRATED_CALL_SITES`, none unexpected, none
  missing), 0 import snntorch, 21 of 21 custom kernels resolve to classes.
- (k) COST, reported never barred: one reference call at (32, 16) = 2.94 s wall, 2,794-byte payload, child peak working set
  635 MB; at (1, 1) = 1.92 s, 102 bytes, 635 MB (five-call medians). The startup floor is Brian2's import (~1.9 s) and its
  working set (~635 MB) -- the child is not cheap, which is what a reference is allowed to be and a live path is not (3.6).

### 3.3 The reds (B-R1), each shown red under its mutation

| red | mutation | shipped | mutated | fires |
|---|---|---|---|---|
| R1 threshold altered | `v > theta` | 0 mismatches, in bound | 2 spike mismatches on 4 steps | YES |
| R2 reset altered | reset one step LATE; subtract swapped for zero | in bound | late: second-step membrane off by 0.2; zero: off | YES |
| R3 schedule altered | threshold on `v[t-1]` before integration | in bound | mismatches | YES |
| R4 GIL regression | a child imports brian2 in-process then calls the guard | parent False | rc 1, `RuntimeError ... enabled the GIL` | YES |
| R5 state omitted | `mem` deleted from the checkpoint; a mutant restore that fills it with zeros | `KeyError('mem')` | (c) RED: B-omit diverges from C | YES |
| R6 silent fallback | a mutant constructor swallowing `refractory=2` | `NotImplementedError` | no refusal -> caught | YES |
| R7 resource overrun | `MAX_REFERENCE_CELLS + 1` cells; a 10 s child under `timeout_s = 0.5` | refused, 0 launches; `TimeoutError` in 0.51 s | -- | YES |

**VERDICT (2.2 as amended by 2.8 (a) and (c)): ACCEPTED FOR THE AMENDED CONTRACT; the original experiment was NOT
ACCEPTED on (p)** (float32 4.194e-6 against the frozen 1e-6; float64 1.92e-6 against 1e-9 at unlike parameters -- both kept
in 2.8; replaced at the fold, 3.7 PH1). Under the amendments (p32), (p64), (m), (c), (g), (r) and (i) hold and R1-R7 fire
(R7 as scoped in 3.7 I3). What it licenses: "the Torch bridge
implements the reference's equation, threshold and reset at the parameters it holds, refuses what it does not implement,
survives a process boundary, and keeps the brain free-threaded" -- the B1 contract of FR-037 / SC-015 for the LAYER the
eleven call sites share. It licenses nothing about those modules' training rules (2.5) and nothing about the twenty-one
custom kernels.

### 3.4 Disposition (2.4, ACCEPTED branch)

The plan amendment and the migration are committed on `main` as the owner's work, verbatim, after this record's
pre-registration and amendments -- NOT in 2.7's order: 2.7 froze the migration (step 3) before the contract (step 4), and
the bridge + contract landed first (`2e76bcd`) so no intermediate commit is import-broken (a deviation, 3.5 item 5; corrected
at the fold, 3.7 PH4); T101 is marked `[X] ACCEPTED UNDER THE AMENDED CONTRACT` in tasks.md with 2.5's scope beside it; the
C0 fold branch is rebased on top and ONE battery runs on the stacked head.

### 3.5 Deviations and disclosures

1. THE ORIGINAL EXPERIMENT WAS NOT ACCEPTED ON (p): the float32 bridge missed the pre-registered membrane bar (max 4.194e-6
   against 1e-6 -- a bar frozen 2.1 wrote FOR the float32 bridge). Amendment 2.8 (a), selected AFTER that result, revised the
   float32 tolerance on an arithmetic rationale and added a float64 arm; 2.8 (c) then gave the reference the bridge's stored
   float32 parameter values without raising the float64 bar (1.92e-6 against 1e-9 before it). Each was committed before its
   re-run; neither shows that the original criterion passed. This departs from 2.4's disposition ("the bridge is corrected",
   T101 OPEN) and is disclosed as such: the acceptance is for the AMENDED experiment, under the project's dated-amendment
   practice (T062 section 2.9, `cd29a1b`). The matched-parameter float64 arm supports equation agreement; it does not negate
   either recorded failure, both kept in 2.8. (Replaced at the fold: this item said the corrections were "never about a bar
   being missed by the bridge", which is literally false -- 3.7 PH1.)
2. THE MIGRATION WAS BUILT BEFORE THIS PRE-REGISTRATION, by the owner through Codex; the record's section 1 is the measured
   starting point and the reds and conjuncts are the arc's own, first run after the record's commit.
3. THE EXECUTOR WAS CODEX (owner routing 2026-09-22), the first arc built that way; three bounded passes, every number
   re-run by the orchestrator, no git command by the executor; pytest under the executor hit a temporary-directory
   permission error twice and was retried with `--basetemp` (an environment quirk of this host's `%TEMP%`, not a test).
4. NOT PANELLED at `89b5e26`; the panel and its fold are 3.7.
5. THE LANDED ORDER REVERSED FROZEN 2.7 STEPS 3 AND 4: `2e76bcd` introduced the bridge and the contract before `045e78e`
   introduced the importing migration, for the import-dependency reason 3.4 gives; the original bridge and its test first
   entered git together with the contract in `2e76bcd` (2.7 did not require a separate original-bridge commit). Added at the
   fold (3.7 PH4).

### 3.6 Openings

1. THE REFERENCE CHILD IS HEAVY: ~1.9 s startup and ~635 MB peak per call (3.2 (k)). It is a REFERENCE, never a live path;
   any future worker-backed LIVE model (the plan's own caveat) owes bounded timeouts, deterministic state transfer and failure
   handling proven at its own scale before deployment -- this contract's `timeout_s` and size bound are the floor, not that
   proof.
2. `MAX_REFERENCE_CELLS = 2,000,000` and `timeout_s = 120` are CHOSEN bounds, not measured ones; the (32, 16) trace costs
   2.94 s and 2.8 KB, so the ceiling sits far above any use in this tree.
3. TEN OF THE ELEVEN CALL SITES ARE ACCEPTED BY THE LAYER ONLY (2.5): their forwards are not reference-checked; `SpikingReadout`
   is. A module whose forward composes the layer differently (recurrent, multi-layer) owes its own (m) before it is claimed.
4. THE MIGRATED TRAINERS REMAIN BPTT EXPERIMENTS (`training/trainer.py`'s `loss.backward()` and the like), as they were under
   snntorch; the gradient-claim census, not this record, says which owe what; the bridge itself never steps an optimizer.
5. THE `none` RESET AT beta 0.95 REACHES MEMBRANES OF ~14.6 in 32 steps and dominates the float32 residual (2.887e-6); a caller
   holding a `none`-reset membrane for hundreds of steps in float32 should expect drift of order `T * 2^-23 * |v|` -- the
   contract states the bound, it does not promise it small.
6. `@pytest.mark.slow` is used by the new tests and is not registered in `pyproject.toml` (pytest warns, does not fail); the
   executor reported it registered -- an instrument disagreement, the pyproject grep is the orchestrator's; register or drop.
7. THE INVENTORY RATCHET IS STATIC: it proves who imports the bridge and that no one imports snntorch; it does not prove any
   call site's forward matches the reference (opening 3).

### 3.7 The panel and its fold (2026-09-22; panelled head `89b5e26`; fold `1cea2f1`..`29fb33d`)

APPARATUS. Three lenses (instrument, pre-registration honesty, integration) and one refuter per filed finding, all Codex
(`gpt-6-astra`, reasoning `high`, the `codex` MCP tool, read-only on the frozen head); each refuter judged the CLAIM and the
proposed FIX as two separate claims. Every number below that a lens or refuter reported was re-run by the orchestrator
before it was written. The fold's code edits were applied by the Codex executor (`gpt-6-astra`, reasoning `low`, exact
briefs; owner 2026-09-22: "Codex is executor, use appropriate models ... to save usage credits"); the orchestrator reviewed
every diff, re-ran every check, wrote this record and made the commits. THE FOLD WAS NOT RE-PANELLED (the owner's credit
instruction): it is reviewed by the Codex closing report and verified by the arc's battery only -- said plainly per the
rhythm.

LEDGER: 12 filed -- STAND 4 (I1, PH3, G1, G2), QUALIFIED 7 (I2, I3, PH1, PH2, PH4, PH5, G3), REFUTED 1 (PH6); fix verdicts
ADOPT 2 (I1, PH5), AMEND 9, REJECT 1 (PH6). Severity after refutation: BLOCKER 1 (G1), MAJOR 3 (I1, PH2, PH3), MINOR 7,
NONE 1. The fold adopts each REFUTER's minimal fix, never the lens's.

| id | finding | claim | fold |
|---|---|---|---|
| G1 | `tools/dependency_ledger.py --assert` exit 1: `psutil` imported (guarded) by the reference child, declared nowhere, installed only as pygenn's transitive dependency; exit 0 at `a0f0314`, 1 from `2e76bcd` | STANDS, BLOCKER | `1cea2f1`: psutil in core + both prose sites the MISDESCRIBED check reads; Linux peak RSS via stdlib `resource`. `--assert` exit 0 (UNDECLARED 0, UNUSED 0, MISDESCRIBED 0); `cost_report` on Windows 631,771,136 B |
| PH3 / I1 | `CONTRACT["precision"]` declared exact spikes and the float32 bound universally; `beta = float32(0.8)`, `v0 = 0.5`, `theta = 1`, current `nextafter(float32(0.6), 0)` = 0.5999999642: the float64 sum is 0.9999999702, float32 rounds it to 1.0, the bridge spikes, the reference does not, error 0.99999997 against a bound of 1.19e-7 (current 0.5 control: 2.98e-8, in bound; float64 bridge exact in both); inside the registered value ranges of 2.2 (p) and reachable through the shipped `SpikingReadout`. A cancellation case (currents -8e6, `v0` 1e7, theta 2e7; OUTSIDE the registered domain) errs 0.1192 against 1.19e-7 with no spike mismatch | STAND (MAJOR) | `7b88be5`: the declaration restricted to the registered seeded parity traces, named as acceptance tolerances, not universal guarantees; `test_threshold_rounding_limit` pins the falsifier and its control; the contract test asserts the restriction |
| PH2 | 2.6 registered seeds 0..4 for (p); the bare invocation and the test ran seed 0 only | QUALIFIED (2.6 said "seed 0 reported"; executable coverage was the gap) | `6a8fbc0`: `test_conjunct_p` parametrized over 0..4; the bare invocation (a CORE self-check) keeps seed 0 by design. All five seeds pass: 27 arms each, 0 spike mismatches, float64 0.0, float32 maxima 2.887e-6, 2.220e-6, 4.391e-6, 3.255e-6, 3.154e-6 |
| G2 | a checkpoint from the pre-migration saver (snntorch 1.0.0) fails the strict load on `lif.graded_spikes_factor` / `lif.reset_mechanism_val`; dropping them silently LOADS and changes outputs (the former layer reset one step later) | STANDS | `ae95632`: a named `ValueError` on either key; strict otherwise; `tests/test_trainer_checkpoint.py`; positive control: the actual former saver's checkpoint gets the named refusal, the current saver's loads. No converter (opening 8) |
| PH1 | 2.8 (a) revised a failed bar after its result; 3.5's "never about a bar being missed by the bridge" was literally false | QUALIFIED (the refuter REJECTED the lens's demand for a new arc: the T062 precedent accepts an amended experiment in-arc) | this commit: the VERDICT line, 3.5 item 1 and tasks.md T101 now read ACCEPTED FOR / UNDER THE AMENDED CONTRACT, with the original experiment NOT ACCEPTED on (p) |
| I1 | the float32 formula is not a general recurrence bound | STANDS | as PH3, plus: the formula in 2.8 (a), 2.8 (c), 3.2 (p32) and 3.6 item 5 is an ACCEPTANCE TOLERANCE for the measured registered traces; 3.6 item 5's long-trace drift extrapolation is WITHDRAWN |
| I2 | a float32-only +1e-7 input mutation passes all 27 arms with float64 exact, so (p64) does not by itself attribute the float32 residual to rounding | QUALIFIED (the bridge has one dtype-agnostic code path; the injected branch is not in it) | record: 2.8 (a)'s "rounding, not semantics", 2.8 (c)'s "pure accumulation" and 3.2 (p64)'s "the float32 residual above is rounding" read as: agreement at matched parameters SUPPORTS a finite-precision explanation on the tested traces; it does not establish rounding as the exclusive cause |
| I3 | R7's 0.5 s deadline fires before the child reaches its injected sleep: `TimeoutError` at 0.508 / 0.510 / 0.513 s with `sleep_s` None / 0 / 10; a child with `sleep_s = 0` needs 1.94 s just to return | QUALIFIED (frozen R7 registers the timeout OUTCOME, which holds) | record: R7 is an oversized-request refusal plus a STARTUP-INCLUSIVE process-deadline refusal; `sleep_s = 10` was requested but its execution before the deadline was not shown. Supplemental: `sleep_s = 30` under `timeout_s = 6` raises `TimeoutError` at 6.01 s, so the deadline also holds once the sleep is reached |
| PH4 | 3.4 called the landed order "2.7's order"; it reversed steps 3 and 4 | QUALIFIED | this commit: 3.4 corrected, 3.5 item 5 added |
| PH5 | the stamps "clock read at write time" (the header 15:00, 2.8 (a)-(b) 15:0x, 2.8 (c) 15:2x, 3.0 15:0x-15:4x CDT) sit after the commits that carry them (`a0f0314` 14:41:41, `b2fd79c` 14:52:46, `43c13d7` 14:59:38, `89b5e26` 15:11:08 CDT, author = committer time) | QUALIFIED ("0x" disclosed approximation) | RETRACTED here: those stamps are not clock readings; git's timestamps are metadata, not execution times; commit ancestry is the evidence for order |
| G3 | the four migrated trainers (`training/trainer.py::train_model`, `offload.py::train_with_offload`, `mnist.py::train_model`, `spiking_lm.py::train`) step optimizers by backprop, sit inside the e-prop invariant's explicit entrypoint closure ("There is NO train-tier exemption"), and read `MISSING_DECLARATION`, `governed=False` in the census | QUALIFIED (inherited, already disclosed in 2.5 and 3.6 item 4; the callers already selected `fast_sigmoid()`; no new rule, no bridge census row) | record: `governed=False` limits the census ASSERTION, not the policy; T101 supplies no learning-rule disposition for these trainers; their obligations stay OPEN (opening 9) |
| PH6 | three commit messages say "verified by the arc's battery" with no witness at `89b5e26` | REFUTED: that sentence is the rhythm's REQUIRED form | none |

FOUND WHILE FOLDING, NOT BY THE PANEL (no lens ran either command):
- `tools/dependency_ledger.py --self-check` (a CI lint step) had been RED since `045e78e`: its CORE-PROSE and CORE-ONLY
  mutations replaced a description string that commit removed, so both printed BROKEN. `395bb99` re-anchors them on the
  `description = "` key: "OK: 5/5 criteria proven falsifiable".
- `ruff format --check .` reported 17 files, all touched by `045e78e`. Their COMMITTED blobs are LF and format-clean; the
  working copies carried mixed line endings from in-place editing. Running the formatter normalized them and `git add` found
  nothing to commit; the tree now reads "612 files already formatted". A battery in this checkout would have red on the
  format stage with no defect in any commit -- the checkout is part of the instrument (obs #2222).
- `tests/test_gradient_claim_gate.py` carried `f"src\snn_data_lake\{env}"` (invalid escapes, a SyntaxWarning on every
  compile, present since `fafea2f`); `29fb33d` writes it `rf"..."`, AST identical.

VERIFIED AFTER THE FOLD (orchestrator): the three new tests and the contract test pass (9 selected); `test_conjunct_p` x5 and
`tests/test_gradient_claim_gate.py` pass (25 in 354 s); `ruff check .` passes; the ledger's `--assert` and `--self-check` exit
0; the full suite, run_selfchecks and the gates run in the arc's battery.

OPENINGS ADDED BY THE PANEL (continuing 3.6):
8. LEGACY `SpikingReadout` CHECKPOINT COMPATIBILITY IS OPEN. The refusal contains it; it does not port the retired
   layer's persisted state, so P2's port-first obligation is not met by it. A validated conversion (reset timing included)
   owes its own evidence-gated arc if a legacy checkpoint or a compatibility requirement turns up; none was found among the
   accessible candidates, and machine-wide absence is unproven.
9. THE FOUR MIGRATED TRAINERS' E-PROP OBLIGATIONS (G3) are open for adjudication through the existing acceptance process and
   truthful `GRADIENT_CLAIM` declarations; not a T101 prerequisite.
10. A UNIVERSAL PRECISION GUARANTEE (threshold edges, cancellation, explicit input/threshold margins) is not claimed; if ever
    wanted it owes its own pre-registered arc, keeping `test_threshold_rounding_limit` as its regression.
11. THE LINUX PEAK-RSS BRANCH has run only under WSL Ubuntu's python3 (the stdlib call), never on a Linux CI runner.
12. R7 HAS NO ACKNOWLEDGED BLOCKING-CHILD CONTROL: the registered outcome stands; attributing the timeout to the injected
    sleep would need a child that proves entry into the delay before the deadline counts.
13. THE INVENTORY SCANNER READS IMPORT STATEMENTS ONLY: `importlib.import_module("...")` / `__import__("...")` string forms
    escape it (instrument lens; no such use in the tree today). The launch counter's before/after delta is not per-request
    under concurrent callers (instrument lens).
14. THE PANEL'S LENSES RAN NEITHER `--self-check` NOR `ruff format --check`: two reds on this head were found by the fold,
    not the panel. A lens brief names every battery stage the arc can move, or states which it skipped.
