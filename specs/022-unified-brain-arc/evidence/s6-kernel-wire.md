# S6 — the sparse kernel onto the live tick, pre-registered as the measurement its own plan orders first (2026-09-09)

> Plan row (PLAN-FINAL-2026-09-02.md, Phase 3, S6): "wire the sparse kernel onto the live tick. `KernelTick` is
> dormancy-exempt `wire` today; S6 makes the flat (post, weight) scatter in `KernelTick._deliver_and_rotate`
> the live tick's delivery (FR-035 is already met by `substrate/kernel_state.py`, T013, SoA), specs/008 gets a
> `tasks.md`, then the resident-CUDA arm gated on `substrate/device.py::select` with a new workload name
> (`"substrate_tick"`, introduced by S6) at the crossover re-derived on this host. Red: tick time at the
> measured width; a neuter of the kernel changes the tick's output." Its 2026-09-08 amendment (the pager
> wire, S-R5) is RETRACTED (T068 LOSS); S6 keeps its original scope. Part F **S-R4**: "the wired kernel does
> not change the tick's output when neutered, or tick time does not fall | the object-tick twin | S6".
>
> Governing text this row obeys where the plan bullet is silent: `specs/008-vectorize-scalar-paths/plan.md`
> ("Build `tools/convert_gate.py`: Gate Zero (neuter-and-measure the Amdahl ceiling) plus the T0–T3 ladder.
> Without it, no conversion has evidence") and its `spec.md` FR-004 (each converted site measurably faster at
> its real width) and **FR-005 (a slower site is DECLINED with the measurement recorded)**; POLICIES.md P8
> (a criterion with no reachable red is deleted, not weakened), P2 (exactly one mechanism), P3 (an exemption
> is a claim about a consumer), the cross-cutting rule that a roadmap's number is a claim about an instrument.

## 1. Territory, measured before the build (HEAD `6268a39`; the orientation's 60 facts are in the session archive)

- **The live tick has no delivery structure the kernel could replace.** `Brain.tick → Brain._wake_tick →
  LiveBrain.tick → AutonomousArm.step` runs a heterogeneous federation of faculty modules, each owning its
  own neurons and drive arithmetic: 165 live `cortex/lif_neuron.py::LIFNeuron` objects and 408,998 cells of
  dense 2-D float ndarray (`basic_components.Connection.propagate` is the generic dense propagate); zero
  live `Synapses`, `Population`, `Connection` objects; **no sparse (pre, post, weight, delay) edge list
  anywhere on the live tick**. The kernel is never imported by a live brain process (`sys.modules` holds no
  `kernel_tick`/`kernel_state` after 420 ticks; 0 `KernelTick`/`KernelState`/`Synapses` objects); its only
  real importers are `determinism_gate` and `replay_gate`, which build synthetic nets.
- **The Amdahl fact.** Measured this host, `make_brain(seed=0, persist=False)`, scratch home, 400 ticks:
  wake p50 **108.663 ms** (n = 194), asleep p50 9.438 ms. cProfile over 150 wake ticks (17.816 s tottime):
  `prefrontal_executive._decide_spiking` 17.9 %, `nt.fsync` 17.1 %, `eprop_kernel.forward` 10.4 %,
  `builtins.sum` 9.7 % (904,324 calls), … and **`lif_neuron.step` 0.57 %, the whole `lif_neuron` bucket
  0.78 %**. The mechanism the kernel's `_propagate` "owns" (its docstring: "the vectorized form of
  `cortex/lif_neuron.py`'s per-instance equation") is under 1 % of the deployed tick. A delivery path made
  infinitely fast cannot move the wake p50 by more than ~1 %; `perf_baseline`'s ratchet tolerance is 15 %.
  **S-R4's timing conjunct ("tick time does not fall") has no reachable red at the deployed width** — not
  because the kernel is slow (a bare `KernelTick` over 16,384 units ticks in 0.55 ms, T068 R2's twin) but
  because there is nothing at 165 units for it to make faster.
- **"The measured width" is unnamed and the candidates differ by four orders**: the deployed 165 units;
  T066/T068's comparison width 16,384; T065's binding widths (222,193 / 888,825 / 3,555,352 — withdrawn by
  T068 §1 as the wrong quantity). The plan bullet's "FR-035 is already met by `kernel_state.py`" is true of
  the artifact and false of the hot path: `KernelState` is a parallel structure whose own header says
  "Nothing is retired in THIS commit: the kernel is not yet the live path"; its FIELDS are LIF/ALIF-only.
- **The kernel carries no plasticity.** `compile_bundles` "quantizes a static edge list ONCE and never moves
  a weight"; `Synapses` exposes no update path. The live brain's learning (`eprop_kernel.forward`, 10.4 % of
  the tick) holds its own membranes and weights in torch tensors. A wire therefore either leaves two
  membranes for the same units (P2's "exactly one mechanism" violated by construction) or adds a
  weight-update/recompile path the S6 bullet does not authorize. Growth (neurogenesis every sleep night)
  cannot live in a fixed `Synapses` table (`p_parts` and `max_delay` fixed at construction).
- **No CUDA kernel path for the substrate tick exists.** `KernelTick` is pure numpy; the three GeNN files
  are training-tier MNIST scaffolds, one self-declared never executed. `device.select` takes a free-form
  workload string (no registry) and "answers 'cpu' everywhere it is asked" at every live width; its
  crossover is an in-process numpy-vs-torch bench, not the tick. "The resident-CUDA arm gated on `select`"
  is a gate over an implementation that does not exist — P3 shape 5 (a declared-empty tier).
- **Both kernel dormancy exemptions name consumers that no longer carry the wire** (`T064/T066/T070`: T064's
  deliverable void, T066 deleted, T070 an instrument); `dormancy_gate --assert` stays green because the
  justification is prose the gate prints, not a reference it resolves (the orientation's obs #2155). S6 is
  the only surviving named consumer, and S6 has no `tasks.md` id, no SC in `spec.md`, no quickstart line;
  T071 (US3 exit) names prior separation, the paging verdict and the region charge gate — not the wire.
- **`specs/008` exists with no `tasks.md`** (its plan declares the omission) and orders Gate Zero FIRST;
  its own opening measurement (74.26 ms) is stale against today's 108.663 ms wake p50. Its late amendments
  make the live tick a GPU target at 1M+ (SC-009–SC-012) via a CSR matrix and one SpMV — the shape the CUDA
  arm would take, owned by specs/008's P3 items, not by S6.
- **Prior art:** none — `cortex/brain.py` has zero occurrences of `KernelTick`/`kernel_state`; the one
  promised consumer (T033+ `episode_driver`) landed without importing the module. The kernel's port is
  `tick(ext_q)` (fixed-point int64 external drive over the index space).

## 2. Pre-registration (written before the instrument exists)

### 2.1 What S6 is, restated from its governing text

S6 is the measurement `specs/008` orders before any conversion, applied to the conversion the plan bullet
names, with the decision rule written first. It builds **Gate Zero** as a substrate instrument — the Amdahl
ceiling of a delivery-path conversion on the live tick — reads it at the deployed width, derives the width
at which the conversion could clear `perf_baseline`'s tolerance, and applies FR-005: the wire is DECLINED at
any width where the ceiling is below the tolerance, and re-armed as the substrate for the row that first
places units at scale (T062's H01 ingest, ~16k signed neurons). It does NOT wire the kernel onto the live tick
at 165 units, does NOT build a CUDA kernel, and does NOT add plasticity to `Synapses`.

### 2.2 The instrument (Gate Zero), and where it lives

`substrate/amdahl_gate.py` (the `tools/` path `specs/008` names is refused by the p4c ratchet since T001):
over a live `make_brain(seed, persist=False)` in a scratch home, (i) profile W warm wake ticks and attribute
tottime to the per-object neuron path (`cortex/lif_neuron.py` + `neuron_zoo` + `basic_components.Connection`
— the set `KernelTick` would replace) versus the rest; (ii) NEUTER the path — replace `LIFNeuron.step` with
its cheapest correct-shape stand-in (a no-op that returns the previous output) — and re-time the wake p50 in
the same command, both arms paired with the crc32 reference sample and reported RAW and corrected (T068's
lesson: the reference is not arm-independent for a light process; both are printed, raw decides if they
disagree); the **Amdahl ceiling** = 1 − p50(neutered) / p50(live), reported with its 95 % bootstrap band over
the paired ticks; (iii) derive the **width trigger**: from the measured per-unit cost of the object path
(tottime ÷ unit-ticks) and the measured rest-of-tick cost, the unit count N* at which the object path's
share reaches the tolerance (15 %), printed with the assumption stated (the rest of the tick held at its
measured cost — a lower bound on N*, since the rest also grows); (iv) the self-check's reds: a constructed
brain whose object path is padded to a large share must read a large ceiling (the instrument can see a
ceiling); the neutered arm on the deployed brain must read within the band of the live arm's ceiling twice
(repeatability); a neuter that is NOT installed (the control) must read a ceiling within noise of zero.

### 2.3 The decision rule and the reds (written first)

- **DECLINE** (FR-005) iff the Amdahl ceiling at the deployed width is below 0.15 on its upper 95 % bound.
  Disposition: the kernel wire is recorded DECLINED-AT-WIDTH with the ceiling and N*; the `kernel_tick` and
  `kernel_state` exemptions are rewritten to name the measured trigger (units at scale from T062's ingest,
  or N* reached by growth, whichever first) and dated to T062; `Synapses.empty`'s stale pager docstring is
  corrected; specs/008 gets its `tasks.md` with Gate Zero landed as T1, this row's DECLINE as T2's recorded
  outcome, and the CUDA arm (CSR + SpMV, bit-stable, its own GPU gate) as P3 items that gate on N*; S-R4 is
  re-registered with its widths named (conjunct A at the trigger width where the kernel first produces a
  live output; conjunct B at N*; the object-tick twin as a separate process); the plan's S6 bullet gets a
  dated annotation, never an edit.
- **PROCEED** iff the ceiling's lower bound is ≥ 0.15 at the deployed width — then the wire is built under
  the bullet as written, with S-R4 at the deployed width. (Pre-stated: the territory makes this branch
  unlikely; it is written so the rule is a rule.)
- **Reds:** the instrument's three self-check reds (2.2 iv) must fire their broken arms in the same command
  before the gate reads the live brain; a ceiling read on a tick that includes `fsync` (17.1 %) is a
  measurement of the lake, not the neuron path — the gate reports the ceiling with AND without the I/O
  share (persist is False in the scratch brain, so the fsync share is the daemon's, disclosed).

### 2.4 Statistic and bar

Paired ticks, W = 150 warm wake ticks per arm, the p50 with a 1,000-resample bootstrap band; the bar is
`perf_baseline.TOLERANCE` = 0.15, chosen because a conversion below the ratchet's own tolerance can never be
asserted as a gain by the tree's one performance instrument.

### 2.5 The control that is not the method

The un-neutered arm run twice (repeatability) and a sham neuter (a stand-in installed on a path with zero
share) must read a ceiling within noise of zero; the padded-share constructed brain is the positive control.

### 2.6 Disposition on a null (fixed in advance)

| outcome | disposition |
|---|---|
| ceiling upper bound < 0.15 | DECLINE the wire (2.3); the exemptions re-dated to the measured trigger; S-R4 re-registered at named widths |
| ceiling lower bound ≥ 0.15 | PROCEED under the bullet as written |
| the padded-share red does not fire | the instrument is blind — no reading; fix it first |
| raw and corrected readings disagree on the side of 0.15 | raw decides (T068 §3.6 item 3), both printed |

### 2.7 What this row does NOT claim

No claim that the kernel is slow or fast on the live tick (it is not on it); no CUDA claim; no tick-time gain;
no discharge of T071. It claims: the Amdahl ceiling of the named conversion at the deployed width, measured
with its reds; the width at which the conversion could matter, derived; and the plan's S6 disposition applied
by a rule written before the number.

### 2.8 Commit order

1. this file — 2. the instrument's reds, RUN RED — 3. `substrate/amdahl_gate.py` + its registration (REGIONS,
CORE, a self-check that fires the reds) — 4. the reading on the live brain and the disposition applied
(exemptions, `Synapses.empty` docstring, specs/008 `tasks.md`, S-R4 re-registered, the S6 annotation) — panel
— fold — ONE battery — closing record — push.

### 2.9 Amendment BEFORE the re-read (2026-09-09; the three readings already taken are quoted, and none is discarded)

The instrument landed (`2c3c97e` reds RUN RED: `1 error in 0.10s`, a collection ImportError; `b8413c1` the
gate, 787 lines, three reds firing their broken arms in one command, `run_selfchecks` 353/353, every gate
rc 0) and was read three times at W = 150 paired wake ticks. The readings, verbatim from the builder's
report: seed 0 ceiling RAW 0.00492, band (−0.00824, 0.01501), corrected 0.00243 (−0.03726, 0.05063), sham
floor −0.00089, N* 2,504 (profile) / 6,316 (neuter) → **DECLINE**; seed 1 RAW 0.00510 (−0.03397, 0.01504),
corrected −0.07472 (−0.14971, −0.00529), the ceiling NOT above 2× the sham floor (0.0078) → **DECLINE**
(raw and corrected agree on the verdict while disagreeing in sign); **seed 2 RAW 0.04935 with a band
(−0.24872, 0.19646) that STRADDLES the 0.15 bar** — wake p50 82.3 ms against 104–105 on the other seeds,
n_LIF 189 against 177, the band ~15× wider, N* 2,828 / 642 (agree-ratio 4.40) → the gate printed
INDETERMINATE, a word 2.6 has no row for. Every control HELD on every seed (padded-share 0.87, repeatability
overlap, sham within |0.05|).

**What 2.6 did not say, and says now.** (a) A band that straddles the bar is not a verdict for that seed;
the seed is INDETERMINATE and is re-read ONCE under the host condition 2.2 assumed and did not write down:
the daemon UP and its cockpit LISTENING (a booting daemon runs its refit burst — `perf_baseline`'s own
header names it the pinned host's dominant timing variable), nothing else running, the readings taken in
one session. The builder's readings were taken while the daemon was RESTARTING on the instrument's own
commit (its cockpit was not listening; the auto-restart-on-commit fired at 05:53 local) — the regime seed 2
fell into is the daemon's boot, disclosed here as the confound before the re-read. (b) If the re-read
straddles again, the seed stays INDETERMINATE and no verdict is read from it. (c) The row's verdict is read
from the determinate seeds and requires at least two of them to agree; with fewer than two determinate
seeds there is no verdict and the instrument's precision is the finding. Seeds 0 and 1 are DECLINE already,
so the re-read can only add a third DECLINE, a PROCEED that makes the seeds disagree (then: no verdict; the
disagreement is recorded and a fourth seed is read under the same condition), or a second INDETERMINATE.

**The claim the deployed width supports, stated at the strength the instrument allows.** On seed 1 the
ceiling did not exceed twice the sham arm's magnitude: the removable cost at 177 live LIF units is BELOW
WHAT THIS INSTRUMENT RESOLVES (its floor is ~0.008). That is the honest reading — not "the ceiling is
0.005" but "the ceiling is under the floor, which is under the bar by an order of magnitude" — and it
supports DECLINE more strongly than the point estimate. N* stays a lower bound and the two derivations'
disagreement (2.4–4.4×: cProfile attributes more to the object path than the neuter removes) is reported
as the instrument's own finding, not resolved into one number.

**Not changed.** The bar (0.15), the rule's direction, the reds, the controls, the dispositions of 2.6 as
written; the raw-decides rule (2.6 row 4) was applied on seed 1 as written.
