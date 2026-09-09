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

## 3. Results (on the stacked head, before the panel)

### 3.1 The reds, RUN RED (`2c3c97e`), and the instrument (`b8413c1`)

`tests/test_amdahl_gate.py` was committed before the module existed: `ERROR tests/test_amdahl_gate.py …
ImportError: cannot import name 'amdahl_gate' … 1 error in 0.10s`. Six tests: the three 2.2(iv) reds (the
padded-share positive control with its install and call counts asserted; repeatability by band overlap; the
sham neuter with `sham_calls == 0` asserted), the 2.6-row-3 "instrument blind" refusal (exit 1, no ceiling,
no verdict, refused BEFORE constructing a brain), the red roster asserted BY NAME (an arm dropped at build
time reds here instead of being reported as measured — the T066 lesson), and a non-positive live arm
refused by the ceiling math. After the module: 7 passed.

`substrate/amdahl_gate.py` (787 lines; registered in REGIONS, CORE, and an EXEMPT row of the
`determinism_gate` shape — measured necessary: without it `dormancy_gate` reported `dormant 1`). Its shape,
as built: THREE seed-identical scratch brains interleaved tick by tick — `live` (unpatched), `neut`
(`LIFNeuron.step` → a stand-in returning a class-level constant 0, one attribute read, no arithmetic: the
floor on any converted arm, so the ceiling is an UPPER bound on any achievable gain), `sham`
(`Connection.propagate` → the stand-in; a path with ZERO live instances) — with class-level patches
installed and restored around each individual tick outside the timed region, phase alignment asserted per
index, arm order counterbalanced (a rotating three-arm Latin square; a fixed order had given the second arm
a systematic ≈ 0.6 % advantage — removed by design, not disclosed in prose), per-tick `cProfile` merged
only for awake ticks, the install and call counts probed on a separate UNTIMED tick (an instrumented arm
would read slower and make a DECLINE easier — the wrong direction), the width read by two instruments that
must agree (a process-wide gc census and a bounded object-graph walk: 141 == 141 on every seed), N* from two
derivations (the profiler's attribution; the neuter's profiler-free wall clock), and both disclosed
attribution biases pushing the SAME way (the ceiling over-stated: `basic_components` charged whole to the
object path; the neuter also removes the downstream work suppressed spikes would have caused) so a DECLINE
cannot be manufactured by them. `graph-exists` returned three near-matches, each read and DISTINCT (a
perceiver, a circuit driver, the tick itself); the module REUSES `perf_baseline._ref_ms` and `TOLERANCE`
rather than re-implementing them.

**Two facts the build found and fixed before any reading.** (1) The first sham path was wrong and the
control caught it: `LIFNeuron.run` read as zero-share on the constructed harness but logged 11 calls in one
tick on the live brain — its ceiling still read ≈ 0, so only the asserted CALL COUNT exposed it; the sham is
now `Connection.propagate`, with `calls == 0` measured every run. (2) The sham's pass condition binds on
MAGNITUDE (|c| ≤ 0.05, as the reds registered); a band-contains-zero conjunct added during implementation
was removed FOR CAUSE across three designs (it fired on ~1 clean run in 5 with every failure 17× below the
bar — a test ~30× finer than the decision it supports); the band is still printed as `band_excludes_zero`.

### 3.2 The reading (W = 150 paired wake ticks per arm; `--seed 0|1|2`)

The builder's three readings were taken while the daemon was RESTARTING on the instrument's own commit (its
cockpit not listening; the auto-restart fired at 05:53 local) — disclosed in 2.9 before the re-read; the
re-read and one extra reading were taken with the daemon up and listening, nothing else running.

| | seed 0 (boot) | seed 0 (quiet, extra) | seed 1 (boot) | seed 2 (boot) | **seed 2 re-read (quiet)** |
|---|---|---|---|---|---|
| controls (padded-share / repeatability / sham) | 0.872 / overlap / −0.007 HELD | HELD | 0.873 / overlap / +0.003 HELD | 0.873 / overlap / −0.004 HELD | 0.873 / overlap / −0.005 HELD |
| live LIF units (build → profile) | 141 → 177 | 141 → 177 | 141 → 177 | 141 → 189 | 141 → 189 |
| object-path share / I/O share | 0.0123 / 0.0144 | — | 0.0121 / 0.0049 | 0.0117 / 0.0051 | 0.0121 / 0.0050 |
| RAW p50 live / neutered (ms) | 104.84 / 104.32 | 105.52 / 103.71 | 103.23 / 102.70 | 82.31 / 78.25 | 80.23 / 79.57 |
| **ceiling RAW (95 % band)** | **0.0049 (−0.0082, 0.0150)** | **0.0172 (0.0043, 0.0383)** | **0.0051 (−0.0340, 0.0150)** | 0.0494 (−0.2487, 0.1965) | **0.0082 (−0.2863, 0.1804)** |
| ceiling CORRECTED (band) | 0.0024 (−0.0373, 0.0506) | 0.0258 (−0.0076, 0.0790) | −0.0747 (−0.1497, −0.0053) | 0.0094 (−0.1093, 0.1567) | −0.0170 (−0.2149, 0.0642) |
| sham floor / ceiling > 2× floor | −0.0009 / yes | +0.0005 / yes | −0.0078 / **no** | +0.0185 / yes | −0.0140 / **no** |
| N* profile / neuter | 2,504 / 6,316 | — | 2,550 / 6,091 | 2,828 / 642 | 2,731 / 4,018 |
| verdict word (raw / corrected) | DECLINE / DECLINE | DECLINE / DECLINE | DECLINE / DECLINE | INDETERMINATE | **INDETERMINATE** / DECLINE |

### 3.3 The verdict, in the pre-registration's words

2.3: *"DECLINE (FR-005) iff the Amdahl ceiling at the deployed width is below 0.15 on its upper 95 % bound."*
2.9: *"a band that straddles the bar is not a verdict for that seed; the seed is INDETERMINATE and is re-read
ONCE … the row's verdict is read from the determinate seeds and requires at least two of them to agree."*

| seed | upper bound (raw) | determinate | verdict |
|---|---|---|---|
| 0 | 0.0150 (quiet extra 0.0383) | yes | DECLINE |
| 1 | 0.0150 | yes | DECLINE |
| 2 | 0.1965 → re-read 0.1804 | no, twice | INDETERMINATE (no verdict read from it) |

**Two determinate seeds agree: the kernel wire is DECLINED at the deployed width** — the ceiling of any
delivery-path conversion on this brain is under 0.04 on every determinate band, and on seed 1 (and the seed-2
re-read) the removable cost is below twice the sham floor: BELOW WHAT THE INSTRUMENT RESOLVES, an order of
magnitude under the bar. The width trigger N* is a LOWER bound of 2,500–2,800 units by the profile and
4,000–6,300 by the neuter; the two derivations disagree 1.5–2.5× because `cProfile` attributes more to the
object path than the neuter removes — reported as the instrument's finding, not resolved into one number.
Seed 2 is a genuinely different regime (wake p50 80 ms against 104; 189 units against 177; paired per-tick
differences ±20 % of the tick) that 150 paired ticks cannot resolve to a 0.15 bar under any host condition
tried; it is recorded, not explained.

### 3.4 The disposition applied

Per 2.3's DECLINE branch, in the fold's disposition commit (§4 names it): the `kernel_tick` and
`kernel_state` exemptions rewritten as a dated succession naming the measured trigger (T062's ingested
population, or growth reaching N*), dated to T062; `Synapses.empty`'s stale pager docstring dated; specs/008
gets its `tasks.md` (Gate Zero landed, the conversion DECLINED with its numbers, the connectivity law, the
T0–T3 ladder over the profile's real hot sites, the CSR+SpMV CUDA arm gated on N*); S-R4 re-registered with
its widths named; the S6 bullet and the ordering line annotated, never edited; a tasks note beside T071
(whose three named items are unchanged). No CUDA arm is built: none exists to gate. T062 is next.

### 3.5 Deviations from the pre-registration, each with its reason

1. Bare `main()` constructs three brains (95 s inside `run_selfchecks`), against the house pattern that a
   bare invocation is a cheap wiring check and "anything that observes the brain must be cheap enough not to
   become the load" — built as 2.2(iv) specified; flagged for the panel as an orchestrator/owner call.
2. 2.2(iii) asked for one N*; two were built and they disagree — one number would have looked precise and
   been unaudited.
3. 2.5's "a path with zero share" named no path; the first candidate was wrong on the live brain and the
   control caught it (3.1); the chosen path's zero share is COUNTED every run.
4. Registering the module made it a runnable gate, so `policy_manifest` reds until `docs/POLICIES.md` claims
   it by name (its unclaimed baseline is ratcheted to empty); the builder added one paragraph under P2's
   "'better' requires a measurement that exists" guard naming Gate Zero as that measurement for a
   delivery-path swap, in the same commit (12,723 of 12,800 words) — a POLICIES.md edit 2.8 did not
   anticipate, put to the panel.
5. The readings ran under the daemon's boot burst (2.9); the re-read and the extra under the quiet condition.
6. The "with and without the I/O share" report is a no-op at `persist=False` (I/O share 0.5–1.4 %), as 2.3
   anticipated; the fsync entries are the scratch handle's flushes plus ambient.
7. 787 lines against the "~600" target; the excess is docstring carrying 3.1's measured facts.

### 3.6 What this row does NOT claim

No claim that the kernel is slow or fast on the live tick (it is not on it); no CUDA claim; no tick-time gain;
no discharge of T071. It claims: the Amdahl ceiling of the named conversion at the deployed width, measured
with its reds and its controls and read by a rule written before the number; a lower bound on the width at
which the conversion could matter; and the plan's S6 disposition applied by that rule.
