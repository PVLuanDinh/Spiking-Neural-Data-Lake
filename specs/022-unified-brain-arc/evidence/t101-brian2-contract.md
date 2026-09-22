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

## 3. The record

(written after the reds run; nothing here is written before 2.7 step 4 lands)
