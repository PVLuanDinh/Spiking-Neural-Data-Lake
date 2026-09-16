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

**PANEL-FOLD CORRECTIONS to §3.1 (appended 2026-09-09; nothing above is edited).**

- **The neutered arm is NOT "the floor on any converted arm", and the ceiling is not strictly an upper
  bound (C5).** The stand-in removes the step BODY and KEEPS the per-call Python dispatch, and the live
  brain does not call it once per unit: an awake `make_brain(seed=0, persist=False)` tick makes ~5,100
  `LIFNeuron.step` calls over ~74 DISTINCT stepped units (~69 sub-steps each), 1,264 asleep. The residual
  is calls × dispatch — 5,124 × 50–180 ns = 0.26–0.92 ms on a 104.84 ms tick, i.e. +0.002..+0.009 of
  ceiling — which a flat (post, weight) scatter removes along with the caller's loop it also replaces. A
  numpy LIF-shaped scatter over the same 165 units costs 5.41 µs/tick against the neutered arm's 15.21 µs,
  so a vectorized delivery produces one output per unit for LESS than a Python attribute read, which is
  what the retired sentence said was impossible. Second instrument, failing differently: cProfile
  attributes 1.290 ms to the object path where the neuter removes 0.520 ms, and the 0.770 ms gap sits
  inside that band — so part of the profile-vs-neuter N* disagreement is real removable cost rather than
  profiler overhead alone. Corrected at `substrate/amdahl_gate.py`'s module header and
  `_neutered_step`'s docstring.
- **"Both disclosed attribution biases push the SAME way" is corrected to THREE, and the third runs the
  OTHER way (C5).** The kept dispatch UNDER-states the ceiling by +0.002..+0.009 — 0.5–1.8× the measured
  seed-0 ceiling itself — so "a DECLINE cannot be manufactured by them" is true only of the two biases it
  names. The DECLINE survives it rather than being immune to it by construction: the corrected seed-0
  upper band is ≤ 0.024 against the 0.15 bar, a ~6× margin. Stated as a BOUND and not a point, because
  two instruments read the residual at 8–30 µs per 165 calls and do not agree to two significant figures.
- **A THIRD bias in the WIDTH — not a fourth (C13).** The gate walked only the LIVE brain and only AFTER
  the profile block, so the width it printed is not the width the ceiling was timed at, and the neuter's
  c_u divided the paired delta by a width the neutered arm did not carry. Measured per arm at the END of
  the paired block: seed 0 165/165/165, seed 1 live 177 / neut 165 / sham 177, seed 2 189/177/189. The gap
  is real on two seeds, one of them determinate (seed 1), and ABSENT on seed 0 — where the whole 12-unit
  spread is created by the live-only profile block AFTER the timing is over (177/165/165 post-profile
  against 165/165/165 at block end, two independent runs). The neutered brain grows LESS, not not at all
  (141 → 165 on seeds 0/1, 141 → 177 on seed 2). DIRECTION: the missing units make the neutered arm faster
  for a reason other than the neutering, so the ceiling is OVER-stated and N* UNDER-stated — the same
  conservative direction as the two biases above, and no DECLINE can be manufactured by it. On seed 1's
  recorded numbers the width component is ~12 × 0.012–0.015 ms ≈ 0.15–0.18 ms of a 0.53 ms delta, ~0.0015
  of a 0.0051 ceiling whose band is (−0.034, 0.015). The gate now walks all three arms before
  `_profile_shares` and reports `n_lif_paired_block`; `n_lif_per_brain_profiled` keeps its key and is
  relabelled the POST-PROFILE LIVE width; no reading is re-taken and no published number is overwritten.

### 3.2 The reading (W = 150 paired wake ticks per arm; `--seed 0|1|2`)

The builder's three readings were taken while the daemon was RESTARTING on the instrument's own commit (its
cockpit not listening; the auto-restart fired at 05:53 local) — disclosed in 2.9 before the re-read; the
re-read and one extra reading were taken with the daemon up and listening, nothing else running.

| | seed 0 (boot) | seed 0 (quiet, extra) | seed 1 (boot) | seed 2 (boot) | **seed 2 re-read (quiet)** |
|---|---|---|---|---|---|
| controls (padded-share / repeatability / sham) | 0.872 / overlap / −0.007 HELD | HELD | 0.873 / overlap / +0.003 HELD | 0.873 / overlap / −0.004 HELD | 0.873 / overlap / −0.005 HELD |
| live LIF units (build → profile) | 141 → 177 | 141 → 177 | 141 → 177 | 141 → 189 | 141 → 189 |
| object-path share / I/O share | 0.0123 / 0.0144 | **0.0120 / 0.0347** | 0.0121 / 0.0049 | 0.0117 / 0.0051 | 0.0121 / 0.0050 |
| RAW p50 live / neutered (ms) | 104.84 / 104.32 | 105.52 / 103.71 | 103.23 / 102.70 | 82.31 / 78.25 | 80.23 / 79.57 |
| **ceiling RAW (95 % band)** | **0.0049 (−0.0082, 0.0150)** | **0.0172 (0.0043, 0.0383)** | **0.0051 (−0.0340, 0.0150)** | 0.0494 (−0.2487, 0.1965) | **0.0082 (−0.2863, 0.1804)** |
| ceiling CORRECTED (band) | 0.0024 (−0.0373, 0.0506) | 0.0258 (−0.0076, 0.0790) | −0.0747 (−0.1497, −0.0053) | 0.0094 (−0.1093, 0.1567) | −0.0170 (−0.2149, 0.0642) |
| sham floor / ceiling > 2× floor | −0.0009 / yes | +0.0005 / yes | −0.0078 / **no** | +0.0185 / yes | −0.0140 / **no** |
| N* profile / neuter | 2,504 / 6,316 | **2,565 / 1,788** | 2,550 / 6,091 | 2,828 / 642 | 2,731 / 4,018 |
| verdict word (raw / corrected) | DECLINE / DECLINE | DECLINE / DECLINE | DECLINE / DECLINE | INDETERMINATE | **INDETERMINATE** / DECLINE |

**PANEL-FOLD CORRECTIONS to §3.2 (appended 2026-09-09). Every number above is unchanged except the two
bolded cells, which were BLANK in the pushed version and are now filled with what that reading printed.**

- **The two "—" cells of the "seed 0 (quiet, extra)" column are FILLED (C9):** object-path share 0.0120 /
  I/O share 0.0347, and N* 2,565 (profile) / 1,788 (neuter), agree-ratio 1.43. They were blank although
  the gate prints both on EVERY run — verified in all four gate logs — and no policy anywhere makes an
  "extra" reading print less; the ceiling, p50s, sham floor, unit count and verdict for that column were
  not blanked. Two facts they were hiding. (a) The neuter's N* (1,788) is BELOW the profile's (2,565),
  inverting the direction §3.3 states as the cause, and 1,788 is outside the neuter range §3.3 quotes and
  below the profile floor it quotes. Arithmetic checks against `width_trigger`:
  0.15·170.8187/(0.85·0.01175141) = 2565.2 and 0.15·103.7124/(0.85·0.01023333) = 1788.5. (b) That column's
  I/O share is **7× every other quiet reading** (0.0347 against 0.0049–0.0051), an anomaly the blank cell
  suppressed. The direction claim fails on the table's own face without any re-run: the seed-2 boot cell
  prints 2,828 / 642, neuter far below profile, with 642 likewise outside the quoted neuter range while
  2,828 from the SAME reading sits inside the quoted profile range — so determinacy cannot have been the
  exclusion rule.
- **The band is host jitter, not a property of the brain (C3).** `_bootstrap_ceiling`
  (`substrate/amdahl_gate.py`) is a 1,000-resample bootstrap over PAIRED per-tick differences, so the
  band's width IS the host's per-tick dispersion — the module's own red-(b) docstring already says "a
  noisy block widens its own band". Three panel re-reads of seed 0 under the same declared quiet condition
  but a loaded host (wake p50 189.4 / 243.2 / 342.6 ms against the recorded 104.8) read band uppers
  0.0781 / 0.0486 / 0.0521, while the cProfile object-path share held at 0.0117–0.0128 across all of them.
  The verdict is unchanged and no band-upper figure is quotable as a bound; the SHARE is the quantity that
  survived a 3.3× host-load swing at ±5%.
- **The host condition asserted at "the re-read and one extra reading were taken with the daemon up and
  listening, nothing else running" is an ASSERTION, never an observation (C14).** No clock time and no
  port-8137 state was captured for either quiet reading; the gate emitted only `elapsed_s`, a duration.
  What the record DOES support is the bracket: both fall between `eb6496c` (06:05:17) and `0370c1c`
  (06:15:53), inside the daemon epoch relaunched at ≈06:05:29 — chat_learn.log's timestamped launcher
  lines plus its own 8 s relaunch delay, corroborated to the second by PID 39200's 06:19:06 creation after
  the 06:18:58.04 exit — and the sibling epoch (06:16:03 → 06:18:58) shows boot through `serve()`'s bind
  and a full cycle completing in under 175 s, leaving ≈450 s of listening host for two ~85 s readings. So
  the honest word is **PLAUSIBLE BUT UNVERIFIED**, not "were taken with"; the timeline is not tight. One
  free observation that bears on 2.9's own premise: seed 0's boot p50 (104.84) and its quiet p50 (105.52)
  differ by under 1%, so whatever the host was, the boot burst moved seed 0's wake p50 by less than a
  percent. The gate now stamps `started_utc` beside `elapsed_s` so a future reading is locatable on the
  commit timeline; it does NOT probe port 8137 — that would couple a scratch-home instrument to live
  production state and make a registered gate's output host-dependent. Neither reading is re-taken:
  2.9(a) fixed the re-read at ONCE, 2.9(b) already disposes of the second straddle, and a reading taken
  today measures today's host (a panel `--seed 2` run read live p50 193.3 ms against the recorded 80.2).

### 3.3 The verdict, by the rule of record

2.3: *"DECLINE (FR-005) iff the Amdahl ceiling at the deployed width is below 0.15 on its upper 95 % bound."*
2.9: *"a band that straddles the bar is not a verdict for that seed; the seed is INDETERMINATE and is re-read
ONCE … the row's verdict is read from the determinate seeds and requires at least two of them to agree."*

**Provenance, per rule (added 2026-09-09, C18 — the heading above read "in the pre-registration's words",
which quoted 2.9 under that banner).** 2.3 was pre-registered at `8ed5f0d`, before every number. 2.9's
INDETERMINATE category and its two-determinate-seed quorum were amended at `eb6496c` — AFTER the three
boot readings it quotes and BEFORE the seed-2 re-read, exactly as its own heading states. The verdict does
not depend on 2.9 existing: seed 0, the brain 2.2 and 2.3 themselves name, is DECLINE under 2.3 alone on
both its readings (upper bounds 0.0150 and 0.0383 against a 0.15 bar), and PROCEED requires a lower bound
≥ 0.15 while the largest lower bound across all five recorded readings is +0.0043 — so PROCEED is
unreachable under every construction of the recorded data. (The citation "FR-005" is itself corrected in
§3.5 item 8; the operative pre-registered condition is `contracts/conversion.md` C1.3.)

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

**PANEL-FOLD CORRECTIONS to §3.3 (appended 2026-09-09; the paragraph above is not edited).**

- **"under 0.04 on every determinate band" is withdrawn and replaced by the readings and the bar (C3).**
  The three determinate 95 % band uppers read **0.0150, 0.0383 and 0.0150, every one of them under the
  0.15 bar.** 0.04 was set by the quiet extra's 0.0383 alone — strike that column and the same sentence
  would read "under 0.02" — and it does not reproduce: three panel re-reads of seed 0 read band uppers
  0.0781 / 0.0486 / 0.0521. No replacement bound is published: a "widest band actually observed" is a
  running maximum of host noise, unfalsifiable by construction and stale on the day it lands.
- **N*, restated over ALL FIVE recorded readings (C1, C4, C9, C23).** Profile **2,504–2,828** (2,565 on
  the quiet extra), neuter **642–6,316** (1,788 on that same reading), the two **1.43–4.40× apart** — a
  panel re-read on a loaded host adds 1.34. The committed "2,500–2,800 / 4,000–6,300 … 1.5–2.5×" matches
  no coherent subset: it dropped seed 2 boot's 642 and 4.40 and the quiet extra's 1,788 and 1.43 while
  KEEPING 2,828 from that same boot reading. The one reading rule that would explain most of it —
  "seed 2's re-read supersedes its boot reading" — gives neuter 4,018–6,316, ratio 1.47–2.52 and width
  177–189 correctly, but then the profile upper is 2,731, not 2,828; one number, one rule, one mismatch.
  Nor does determinacy license the drop: 2.2(iii) asked for N* with no determinacy gate at all, 2.9(b)(c)'s
  rule governs THE VERDICT, and 2.9 — written before the re-read, while 4.40 was still the adverse
  number — states the disagreement as **2.4–4.4×**. §3.3 then took the new low (1.47) and dropped the old
  high (4.40); the arc's own earlier text is the control on its later text. 2.9 is NOT edited: it was
  correct for the three readings it quotes.
- **Neither N* is a second instrument (C4).** N*_profile ≡ K·((1−s)/s)·N and N*_neuter ≡ K·((1−c)/c)·N
  with K = tol/(1−tol), exact to max abs diff 0.0 over 2,000 random inputs, so each is its upstream
  statistic restated in width units and `agree_ratio` is the share-vs-ceiling disagreement re-expressed.
  The neuter's N* therefore INHERITS the ceiling's 95 % band: mapped through the same identity with
  endpoints swapped, seed 0 boot → [2,049, unbounded), seed 1 → [2,045, unbounded), seed 2 boot →
  [136, unbounded), seed 2 re-read → [151, unbounded), the quiet extra → [785, 7,300] — the only reading
  whose band excludes zero. So every determinate seed admits an N* BELOW the 2,500 the record quotes as a
  floor, and four of five admit no finite upper bound at all. The profile arm has no interval whatever:
  no bootstrap exists over profiler tottime. Seed 2 boot's 642 is not an unexplained regime; it is that
  formula at its INDETERMINATE ceiling 0.049.
- **"a LOWER bound" becomes a POINT ESTIMATE at one measured width, its bound direction UNTESTED (C6).**
  The model share(N) = c_u·N/(c_u·N + R) additionally requires the object path's cost to be proportional
  to N, a condition 2.2(iii) never names and nothing tests. c_u is measured at ONE width, and the live
  path's composition is not per-unit: an awake tick enters it ~5,124 times over 71–74 DISTINCT stepped
  units of a 165-unit census, per-neuron counts running 10..1,540 with a single unit carrying ~30 % of the
  path. The LOWER-bound direction holds only while the path grows at most linearly in N. No harness test
  is added — `_TickHarness` is one call per unit BY CONSTRUCTION (c_u 0.480/0.454/0.431/0.462 µs/unit at
  n_units 165/330/660/1320, calls-per-unit exactly 1.00 at every width), so it can only ever print a PASS
  and would convert an untested assumption into a falsely verified one.
- **The stated CAUSE of the disagreement is withdrawn without a replacement (C9).** "cProfile attributes
  more to the object path than the neuter removes" is contradicted by the record's own quiet reading,
  where cProfile inflated R by 1.65× (170.8187/103.7124) against c_u by 1.15× (0.011751/0.010233) —
  1.647/1.148 = 1.435, exactly the printed 1.43 — and the neuter's N* landed BELOW the profile's. The
  competing story ("the neuter's R moves with host load while cProfile's does not") is also false:
  cProfile's R moved 170.8187 → 415.0933 ms/tick between the quiet and a loaded reading. No cause is
  asserted here in its place; part of the gap is the dispatch residual §3.1's appended correction bounds
  at +0.002..+0.009 (C5). The honest statement is that the neuter derivation does not RESOLVE N* at this
  width and the two derivations reverse direction between runs.
- **The host-condition adjective (C14).** "under any host condition tried" presumes two host conditions
  were established; §3.2's appended correction records that neither quiet reading carries a clock time or
  a port observation, so the honest form is "under both readings taken, whose host conditions were not
  recorded". `0370c1c`'s pushed message calls the seed-0 extra "a steady-host re-read" — an unrecorded
  host claim now permanent in the git record, corrected here rather than by rewriting the message.

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
   *APPENDED 2026-09-09 (C17): the 95 s carries no host qualifier and is not a constant. The cost is
   ≈ 3 arms × W(150) × the wake p50 this same reading prints, plus three `make_brain` builds, the warmup,
   the reds (measured 2.3–4.4 s) and cProfile's overhead — so it tracks the host's ambient load. Measured
   95 s on the builder's host at wake p50 ~104 ms, and 308 s on a contended host at p50 313 ms (100 % CPU,
   < 1 GB free, 12 competing `python3.14t` processes). The driver is AMBIENT CONTENTION, which is exactly
   what 2.9's pre-registered "nothing else running" condition exists to hold fixed — not daemon boot
   state. On such a host the instrument's PRECISION degrades too, not only its clock: the sham noise floor
   read −0.04854 against the builder's −0.00089, sitting on the |0.05| control bar, and the neuter's N*
   derivation refused outright ("live-neutered p50 delta −14.7961 ms not > 0"); controls still HELD and
   the verdict still read DECLINE, so no reading is retracted — but the cost is not "wall time only", and
   2.9's host condition binds on the READING's precision, not just on its duration. The
   cheap-bare-invocation route stays an OPEN orchestrator/owner call and is deliberately NOT taken in this
   fold: `run_reading`'s exit code carries seven LIVE-brain controls the three harness reds neither check
   nor can check (`neuter_installed`, `neuter_calls_one_tick > 0`, `sham_installed`, `sham_calls == 0`,
   the sham magnitude bar, `paired_wake_ticks == W`, `census_cross_check.agree`), no test charges any of
   them, and `dormancy_gate` exempts this module at MODULE level — so moving the reading behind a flag
   would make `live_reading` reachable-but-uncharged, the shape P3 forbids, and would delete the only
   apparatus that has ever caught a real defect here (§3.1's first sham path, exposed by its CALL COUNT
   alone). If it is later taken it owes, in order: a test that charges those live controls, and a
   `perf_baseline`-style sentence stating they are no longer in the battery path.*
   *APPENDED 2026-09-09 (C22): `run_selfchecks` runs CORE modules with `capture_output=True` and prints a
   child's stdout only on a NON-ZERO rc, so every GREEN battery TAKES a complete seed-0 reading — reds,
   width, ceiling, band, sham floor, both N* — and DISCARDS it; nothing but the PASS line's duration
   records that it happened, and the carry cache stores only `secs`. §3.2's "five" is the count of
   readings taken under 2.9's stated host condition and KEPT, not the count this instrument has produced.
   A battery by-product is inadmissible to §3.2 in any case, because 2.9's host condition (daemon up,
   cockpit listening, nothing else running) is not established for a battery child. Stated because it sits
   in tension with 2.9's own heading pledge, "the three readings already taken are quoted, and none is
   discarded".*
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

**Deviations 8–13 added by the panel fold, 2026-09-09. Items 1–7 above are not edited.**

8. **The DECLINE branch is tagged FR-005 throughout** — §2.1, §2.3, §3.3, specs/008 `tasks.md`,
   PLAN-FINAL:646, `dormancy_gate`'s exemption prose and `0370c1c`'s message. 2.3 labelled the rule
   "FR-005", but FR-005's own sentence covers a site measured SLOWER AFTER vectorizing, and its antecedent
   was never evaluated because no array version was built — so the Conversion-site entity's "measured
   array cost" and "equivalence result" do not exist for this site. The operative pre-registered condition
   (ceiling below 0.15 on its upper 95 % bound) is `contracts/conversion.md` **C1.3**: *"a site whose
   ceiling is below the noise floor of the tick is DECLINED, and the decline carries its number"* — the
   exact on-point rule S6 satisfied. The disposition is additionally carried by **FR-010** and **SC-006**
   ("declined with the measurement that justified declining"), **FR-004 read contrapositively**, and
   `plan.md` **Standing Invariants 1 and 3** — all pre-registered long before S6, so the verdict, the bar
   and every number are unchanged and only the citation was narrow. The record is UNDER-citing, not
   over-reaching. Corrected in the living text (`dormancy_gate`'s two `wire` rows, the sites re-read every
   run). `specs/008/spec.md` is NOT amended: widening a governing criterion after its number is read would
   retroactively make the paraphrase correct and erase the evidence that it was imprecise — the failure
   those same exemption rows refuse for themselves. `0370c1c`'s message cannot be corrected forward and is
   recorded here instead. (Panel C20 + C27.)
9. **2.9(c) introduces a seed-aggregation rule and Part F forbids one.** PLAN-FINAL-2026-09-02.md:784
   closes Part F's preamble with *"no majority-of-seeds rule anywhere"*, eighteen lines above the S-R4 row
   this arc annotated at :802; the same preamble fixes *"S = 8, paired one-sided t > 1.895 (7 d.f.,
   α = 0.05)"*. 2.9(c)'s "the row's verdict is read from the determinate seeds and requires at least two
   of them to agree" was never checked against it, and §3.5's seven items did not carry the conflict.
   **The scope question is STATED, NOT RESOLVED**: Part F's preamble governs "Every seed-paired red", and
   S6's ceiling reading is a disposition gate rather than a red in the register — S-R4 itself was
   re-registered UNRUN — but the word is "anywhere", and `w1-goal-curriculum.md` §5 already recorded this
   same unresolved scope question rather than settling it ("arguably scoped to reds rather than probes,
   but the conflict was never noted"; `l-parallel-arms.md` folded the other instance). Two facts that cap
   it. (a) S6's pairing unit is the TICK — 150 paired ticks WITHIN a seed — so a paired t over 7 d.f. of
   seeds is not the statistic this measurement has, and 2.4 fixed its own statistic before any number
   existed. (b) 2.9(c) is UNANIMITY-among-determinate with a quorum floor of two, routing disagreement to
   "no verdict … a fourth seed" — it REFUSES rather than resolving by plurality, strictly stronger than
   the 6-of-8 form both prior findings flagged. **The rule did no work**: each determinate reading
   independently clears 2.3's per-seed bar at upper bounds 0.0150, 0.0383 and 0.0150, no reading ever
   reached 0.15 except the straddling seed whose conservative branch was taken, and the verdict is
   invariant to deleting 2.9(c) entirely. Residue, disclosed: the rule was written with three readings in
   hand (2.9's own heading says so) and carries no α; a reader wanting Part F's statistic has the per-seed
   bands in §3.2. 2.9 and Part F are not edited, and the row is NOT recomputed from any pooled statistic.
   (Panel C19.)
10. **The "seed 0 (quiet, extra)" reading is UNREGISTERED and unlogged.** 2.9 enumerated exactly ONE
    re-read of the INDETERMINATE seed plus a conditional fourth seed; an extra reading of an
    already-DETERMINATE seed is not among them, and item 5 above disclosed only its host condition. Its
    numbers are transcribed from the run's console and no gate output was saved — which is why its
    object-path-share and both N* cells were the only dashes in §3.2's table, fields the instrument emits
    on every run. It is disclosed by name in three committed places (the §3.2 column header, §3.3's "quiet
    extra 0.0383", `0370c1c`'s body) and NO verdict is read from it: §3.3 reads seed 0's upper bound as
    0.0150 from the boot reading. Its inclusion moved the reported bound the CONSERVATIVE way
    (0.0150 → 0.0383) — the orchestrator widened its own bound rather than narrowing it. It is NOT
    re-taken and NOT deleted: a run today is a NEW reading under a different host (four seed-0 point
    ceilings are now on record — 0.0049 / 0.0172 / 0.0068 / 0.0357), so filing one beside the 05:54 logs
    would fabricate provenance for a reading taken at ≈06:08, and deleting the highest reading would
    TIGHTEN the committed claim on a bookkeeping ground. (Panel C21.)
11. **The "0.04" bound is a property of one unlogged reading, not of the brain.** §3.3 asserted "the
    ceiling of any delivery-path conversion on this brain is under 0.04 on every determinate band"; 0.04
    is set by the quiet extra's band upper 0.0383 alone, and striking that column would make the same
    sentence read "under 0.02". It does not reproduce: three panel re-reads of seed 0 under the declared
    condition read band uppers 0.0781 / 0.0486 / 0.0521, while the cProfile object-path share held at
    0.0117–0.0128. The clause is withdrawn everywhere it was quoted as a bound (§3.3, both `dormancy_gate`
    rows, specs/008 T001/T002, PLAN-FINAL:645 and :802) and NO replacement bound is published — a "widest
    band actually observed" is a running maximum of host noise, unfalsifiable by construction and stale on
    the day it lands. What is published instead is the readings and the bar. (Panel C3, with C8.)
12. **"3 reds fired their broken arms" was printed on every run, and no arm was constructed to fail.**
    2.2(iv) defines exactly ONE constructed perturbation and only `red_padded_share` is built from it;
    2.5 correctly names the other two "the control that is not the method" and "the positive control", and
    §3.2 uses the accurate word ("controls … HELD"). The loose phrase is 2.3's (line 112) and 2.9's
    (line 154), and it propagated into the gate's closing print, its module header, `self_check_reds`'
    docstring, `b8413c1`'s body and specs/008 T001 — none of which was disclosed as a deviation. The print
    and the two docstrings are corrected to "3 controls HELD", with the clause that no criterion is shown
    FAILING in that command and a pointer to
    `tests/test_amdahl_gate.py::test_the_instrument_blind_disposition_refuses_the_reading` (`pad_iters=0`),
    where the padded-share criterion IS seen refusing. 2.3 and 2.9 are NOT edited and `b8413c1`'s message
    cannot be; both are superseded by this entry. **ONE arm was added** — the only criterion in the file
    with no perturbed arm anywhere in the tree: `red_repeatability_broken_arm` re-reads the repeatability
    criterion with its SECOND reading on the PADDED rig and asserts the bands read DISJOINT. Measured at
    the fold: control 0.0416 band (0.03318, 0.04845) against broken 0.8731 band (0.87196, 0.87406). It
    prints as a fourth `--self-check` line and its outcome is kept out of `bad`, so it cannot flip the exit
    code: it demonstrates the criterion CAN fail, it is not a control on a reading. Two other broken arms
    were REFUSED for cause: a stand-in on a non-zero-share path for the sham would BE the neuter (the only
    such path on this harness) and reads on both sides of the 0.05 bar with ambient (0.0478/0.0535 calm
    against +0.063..+0.102 loaded) — the flakiness `red_sham_neuter`'s docstring records removing for
    cause; and a zero-pad arm is a P2 duplicate of the blind refusal already in the suite. (Panel C10.)
13. **Panel-fold instrument changes, all REPORTED and none asserted on.** `substrate/amdahl_gate.py`
    gained: a controls ROSTER built once where the criterion lives, with the false terms named on FAILED
    (C26); the MEASUREMENT arm's install/call counts printed with the probe tick's PHASE, because the
    count varies ~4× with it (4,324–5,124 awake against 1,264–5,872 asleep) while every paired sample is
    awake (C16); per-arm widths walked BEFORE the profile (C13); the two N* identities and the neuter
    arm's mapped band (C4); the linearity caveat (C6); a `types.ModuleType` skip in the width walk,
    measured number-neutral — 141 / 14,976 at one brain and at three, against a 423 process census, zero
    modules reached (C15); a wall-clock `started_utc` with NO port probe (C14); and the one broken arm of
    item 12. **No bar, no conjunct, no verdict word and no exit-code criterion moved, and no reading was
    re-taken.** One consequence to state: the module's stdout no longer byte-matches the five recorded
    transcripts. The record quotes no raw gate stdout anywhere, so nothing in §3.2 or §3.3 changes, but a
    reader re-running the gate should expect extra lines. `--self-check` remains cheap: 2.263 s measured
    at the fold. (Panel C4, C5, C6, C10, C13, C14, C15, C16, C26.)

### 3.6 What this row does NOT claim

No claim that the kernel is slow or fast on the live tick (it is not on it); no CUDA claim; no tick-time gain;
no discharge of T071. It claims: the Amdahl ceiling of the named conversion at the deployed width, measured
with its reds and its controls and read by 2.3, written before every number, with 2.9's aggregation added
after the three boot readings and before the re-read; a lower bound on the width at
which the conversion could matter; and the plan's S6 disposition applied by that rule.

*CORRECTED 2026-09-09 (panel fold).* Two of the words above are now weaker than the record supports. "read
by a rule written before the number" was true of 2.3 and not of 2.9(c), which is why the clause is
rewritten in place rather than left to imply both (C18). And "a lower bound on the width" is a POINT
ESTIMATE at one measured width with its bound direction UNTESTED (C4, C6): each N* is its upstream
statistic restated in width units, the neuter's inherits a 95 % band that is unbounded above on every
determinate seed, and the model's linearity-in-N condition is unnamed in 2.2(iii) and untested. The
DECLINE, the 0.15 bar and the disposition are unchanged by both corrections.

## 3.7 Panel fold (2026-09-09)

Three lenses (instrument, pre-registration-honesty, disposition) filed 40 findings against head `2cc50ef`;
27 STOOD after refutation and **all 27 of the lenses' proposed fixes were REFUTED**. What lands below is
the REFUTER's minimal fix in every case, not the lens's. Nothing pre-registered (§1, §2.1–2.9) is edited;
Results text is corrected by appended dated lines; the two `dormancy_gate` rows keep every earlier word
verbatim and take a dated succession, the discipline they already apply to their own superseded reading.
**No verdict moves.** The DECLINE, the 0.15 bar, the reds, the controls and both dispositions stand, and
six fresh readings of the same three seeds taken during the panel all read DECLINE.

1. **C1 — the N* range in committed source is contradicted by the gate's own printed output.** The
   exemptions and §3.3 said "2,500–6,300 units … 1.5–2.5×"; the gate printed neuter N* [6316, 6091, 642,
   4018, 4561] and ratios [2.52, 2.39, 4.40, 1.47, 1.68]. *Refuter: REPRODUCED EXACTLY, and 2.9's own
   "2.4–4.4×" — written before the re-read — is the record that the arc did not read N* that way while
   4.40 was still adverse.* CHANGED: the ALL-READINGS presentation at `dormancy_gate.py`'s `kernel_tick`
   and `kernel_state` rows (appended), §3.3 (appended), specs/008 `tasks.md` T002 (appended) and
   PLAN-FINAL's S6 bullet (further dated annotation). NOT CHANGED: the "determinate-seed range" branch the
   lens offered was REFUSED — adopting it would drop 642 and 4.40 by retro-applying the VERDICT's
   determinacy rule to a quantity 2.2(iii) never gated on determinacy. 2.9's "2.4–4.4×" is untouched.
2. **C2 — the stated ceiling sits inside the instrument's own matched A/A null.** The sham is installed on
   `Connection.propagate`, a class with ZERO live instances; the matched A/A (`LIFNeuron.step` →
   `_REAL_STEP`, zero work removed) read +0.0118 / −0.0296 / +0.0319 on three live readings, all larger in
   magnitude than every determinate ceiling on record. *Refuter: STANDS on 3 of 3 independent live
   readings, though "+0.0118" is one noise draw, not a constant; an interleaved harness run localizes a
   ~0.007 systematic bias to patching the class resolved 5,124× per tick (hot-class identity median
   −0.00685 against the sham's +0.00031 and a COLD-class identity's +0.00125), and its sign slows the
   NEUTER arm, so it UNDER-states the ceiling.* CHANGED: both `dormancy_gate` rows now carry "AT OR BELOW
   THE INSTRUMENT'S OWN NULL, which reads 0.01–0.03 run to run" inside the appended correction; §3.1's
   appended note records that `amdahl_gate.py`'s own comment already recorded a no-work arm reading +0.018
   against the neuter's +0.021 under a fixed order, and that this fact never reached the exemption rows;
   specs/008 `plan.md` invariant 5 ("Every A/B ships a matched A/A control") is recorded UNMET as built.
   NOT CHANGED, and this is a judgement recorded rather than a fix deferred: **the reported matched-A/A
   arm is NOT added.** The refuter's own minimal fix requires it be taken "as its own run rather than as a
   fourth interleaved brain", i.e. a full three-brain live reading (95–308 s) plus a fourth brain, and
   requires that nothing be asserted or redded on it "until its stationarity has been measured across at
   least five runs on a quiet host". This fold takes no new live reading, so the arm would land as an
   uninterpretable printed number with five owed readings behind it; the cost is judged prohibitive here
   and the arm is recorded as owed rather than half-built. The pre-registered decision rule is NOT
   changed: promoting a null comparison to a pass condition after the numbers are in would relabel a
   pre-registered condition, and the null draws both signs, so the gate would pass or fail on host load.
3. **C3 — "95 % bands inside 0.04" does not reproduce, and the one reading that sets it has no gate log.**
   *Refuter: CONFIRMED and stronger than filed — strike that column and the sentence would read "under
   0.02"; three of four re-readings breach 0.04, one twofold; the stable quantity is the cProfile
   object-path share at 0.0117–0.0128 across a 3.3× host-load swing.* CHANGED: §3.3's appended correction
   states the three determinate band uppers (0.0150 / 0.0383 / 0.0150) and the bar instead of a derived
   bound; §3.2 carries the bootstrap-is-host-jitter sentence with the three re-read uppers; §3.5 item 11;
   the clause is withdrawn at both `dormancy_gate` rows, specs/008 T001/T002 and PLAN-FINAL:645/:802. NOT
   CHANGED: no replacement band figure is published anywhere — the lens's "state the bound as 0.049" was
   already false when filed (a second lens read 0.0781 the same hour).
4. **C4 — N*_neuter is not a second instrument; it is the ceiling restated, so it inherits a band that is
   unbounded above.** *Refuter: reproduced and stronger — the identity is exact (max abs diff 0.0 over
   2,000 random inputs) and holds for the PROFILE arm too, so `agree_ratio` is just the share-vs-ceiling
   disagreement re-expressed.* CHANGED: `amdahl_gate.py::width_trigger` states both identities in its
   docstring and returned dict and prints them; `n_star_band` maps `band_raw` through the same identity
   with endpoints SWAPPED and "unbounded" wherever c ≤ 0; the profile arm's `n_star_band` is `None` with
   its reason printed. Verified against all five recorded readings: [2049, unbounded), [2045, unbounded),
   [136, unbounded), [151, unbounded), [785, 7300]. §3.3 and both exemption rows carry it. NOT CHANGED:
   the neuter derivation is NOT deleted — dropping a built, printed, disclosed arm after reading its
   number is the move §3.1 registers a red against, and it would leave the WORSE arm standing (the profile
   N* has no interval at all). No microbenchmark third instrument is built: that is new measurement work
   owing its own pre-registration, recorded here as the shape a genuinely independent instrument would
   take.
5. **C5 — the stand-in is not the floor of a converted arm; it keeps the per-unit Python call.** *Refuter:
   STANDS and is understated — a numpy scatter over the same 165 units costs 5.41 µs/tick against the
   neutered arm's 15.21 µs, so the premise is false on its face; and the lens's magnitude was 24–88×
   too small because it was measured on the constructed harness, which makes one call per unit.* CHANGED:
   `amdahl_gate.py`'s module header and `_neutered_step`'s docstring lose "cannot do it for less than an
   attribute read", "a floor on any converted arm" and "an UPPER bound on any achievable gain", and carry
   the live-brain shape instead; the "two disclosures push the SAME way" sentence becomes THREE with the
   third the other way, bounded at +0.002..+0.009; §3.1 takes the appended correction. NOT CHANGED: §3.1's
   prose is not rewritten in place, and "~0.01 % of the tick" is not recorded anywhere — that is the
   harness's number, not the brain's.
6. **C6 — N*'s model assumes the object path's cost is linear in unit count, and the assumption is never
   stated.** *Refuter: STANDS and the re-run makes it larger — only 71–73 of 165 censused units are
   stepped per awake tick, per-neuron counts run 10..1,540, and one unit carries ~30 % of the path, so
   "cost per unit" is not a mechanism-level quantity.* CHANGED: the condition is added to
   `width_trigger`'s CAVEAT docstring and to the printed CAVEAT line; the committed direction word is
   weakened from "a LOWER BOUND" to a point estimate with its bound direction untested at both
   `dormancy_gate` rows, specs/008 T002, §3.3 and PLAN-FINAL. NOT CHANGED: 2.2(iii) is left UNEDITED, and
   the proposed harness linearity test is NOT built — `_TickHarness` is linear by construction (measured
   c_u 0.480/0.454/0.431/0.462 µs/unit at 165/330/660/1320, calls-per-unit exactly 1.00), so it could only
   ever print a PASS and would convert an untested assumption into a falsely verified one, a control that
   is the method itself.
7. **C7 — "the instrument's floor is ~0.008" is one reading's sham point estimate, published in three
   files, and it contradicts the range it is attached to.** *Refuter: every element reproduces — 0.017 is
   2.1× the number it is said to sit under, and the reading that produced 0.017 is the one whose own gate
   line reports "ceiling exceeds 2x the floor: True".* CHANGED: PLAN-FINAL:802 loses exactly the words
   "under the instrument's own floor (~0.008) and", leaving "sits an order of magnitude under the 0.15
   bar", with a dated annotation recording what was deleted and why; §3.5 item 11 and this entry carry the
   rest. NOT CHANGED: evidence 2.9's own use of ~0.008 is CORRECT as written — it is scoped to seed 1 and
   0.00779 is seed 1's own floor — so 2.9 is not edited (C12's refuter over C7's part (2)); and no
   replacement resolution constant ("of order 0.02–0.07") is published, since that is a new false constant
   from a handful of draws with no stationarity check.
8. **C8 — "ceiling 0.005–0.017 at the deployed width" is published as THE measurement in four places and
   does not reproduce.** *Refuter: CONFIRMED by an independent reproduction of the COMMITTED instrument —
   raw ceilings +0.03246 / −0.03168 / −0.01024 with controls HELD on all three; combined with the panel's
   +0.02581 / +0.00271 / −0.01910, six fresh readings of the same three seeds produce ZERO inside the
   published range, while all six read DECLINE.* CHANGED: the two sites that quoted the range AS the
   measurement without a caveat — both `dormancy_gate` rows and specs/008 T002 — now carry the
   reproducible statement ("BELOW WHAT THE INSTRUMENT RESOLVES … not quotable as a number … every
   determinate 95 % band's upper bound far under the 0.15 bar") with a pointer to §3.2's per-reading
   table; "bands inside 0.04" is dropped at all four sites. NOT CHANGED: no pooled cross-run statistic is
   computed (it would pool ticks across three host ambients and charge between-condition variance to
   within-run noise, and no committed code produces it), and no reading COUNT is baked into any source or
   spec file, because a baked-in count is a stale constant by construction.
9. **C9 — a determinate reading the record took was blanked out of its own table.** The "seed 0 (quiet,
   extra)" column's object-path/I-O and both N* cells read "—" although the gate prints them every run.
   *Refuter: reproduced on file — that column printed N* 2,565 / 1,788 with the NEUTER BELOW the profile,
   inverting the stated cause, and its I/O share is 7× every other quiet reading; and the record
   contradicts itself independently of the blank cell, since the seed-2 boot cell prints 2,828 / 642 with
   642 outside the quoted range while 2,828 from the same reading is inside it.* CHANGED: both cells are
   FILLED in §3.2 with a dated one-liner beneath the table naming the blanking and the 7× I/O share; the
   spread is stated over all five readings; the causal sentence is withdrawn WITHOUT a replacement cause.
   NOT CHANGED: "1,788–6,316" is not adopted (it still excludes 642) and neither is "6,091–6,316" (it
   excludes a determinate seed); no new cause is asserted, because both candidate mechanisms are
   contradicted by the record's own readings.
10. **C10 — "3 reds fired their broken arms" is printed on every run and is false.** *Refuter: REPRODUCED
    on three runs, all rc 0, three PASS lines and no RED or FAIL; the phrase is 2.3's own loose summary of
    2.2(iv), which constructs exactly one perturbation.* CHANGED: the closing print, the module header and
    `self_check_reds`' docstring say "3 controls HELD" with the no-criterion-shown-failing clause and a
    pointer to the blind-disposition test; ONE arm is added
    (`amdahl_gate.py::red_repeatability_broken_arm`) with its test
    (`test_the_repeatability_criterion_fires_on_a_deliberately_broken_arm`, the fired arm and the
    as-shipped control in the same test); specs/008 T001 and §3.5 item 12 record it. NOT CHANGED: 2.3 and
    2.9 are not edited, `b8413c1`'s message cannot be, and the two other proposed arms are refused for
    cause (a non-zero-share sham arm straddles its own 0.05 bar with ambient; a zero-pad arm duplicates
    the existing blind refusal).
11. **C11 — the re-armed exemption names T062 as "the row that FIRST PLACES UNITS AT SCALE"; the plan says
    T062 places no units.** *Refuter: reproduced at every site — the number (~16k) is right and the ROLE is
    wrong; the clause lives at seven passages in four files, and §2.3 PRE-REGISTERED the wrong premise, so
    it predates the measurement.* CHANGED: a dated SUCCESSION appended at both `dormancy_gate` rows
    (keeping the 2026-09-09 CURRENT READING verbatim), specs/008 `tasks.md` T002, specs/022 `tasks.md`'s
    T071 note and PLAN-FINAL's S6 bullet. The live consumer is restated as **THE FIRST TICK POPULATION AT
    OR ABOVE N***. Recorded for the next arc: T062's own pre-registration DOES define an "ingested
    population" (a `KernelState` + `Synapses` built from the H01 graph, consumed by T063) — but OFF the
    live tick, so the exemption's consumer is still the first population ON the tick, which T062 does not
    put there. NOT CHANGED: the clause is NOT dropped or silently demoted — that is the quiet swap these
    rows exist to forbid, and it would be the second in-place rewrite of this clause in two days; and
    §2.3's pre-registration text takes no edit, because its being wrong BEFORE the measurement is itself
    the evidence.
12. **C12 — "the instrument's floor is ~0.008" states a per-run scatter statistic as a property of the
    instrument, for a quantity measured to span 110×.** *Refuter: STANDS on its two landed sites and is
    worse than argued — two independent re-reads at the deployed width (seed 0 +0.00682, seed 1 −0.00922)
    BOTH printed "ceiling exceeds 2x the floor: True", and seed 1 is the one seed the licensed sentence
    was scoped to; across nine readings the licensed conclusion is False on five.* CHANGED: the sentence
    is CUT from specs/008 T001; PLAN-FINAL S-R4 loses the floor clause (see C7); this entry and the §3.2
    appended note record that the sham's magnitude is per-run BY THE INSTRUMENT'S OWN DEFINITION
    (`amdahl_gate.py`: "the sham arm removes NO work, so whatever it reads is the instrument's own
    floor"), that over the five recorded control-held readings it runs −0.0009 / +0.0005 / −0.0078 /
    +0.0185 / −0.0140 and flips sign, and that the "exceeds 2x floor" flag is True on 3 of the 5. NOT
    CHANGED: evidence 2.9 and the §3.2 table are untouched — 2.9's use is correctly scoped to seed 1 — and
    the two REFUSED readings (−0.01883 and −0.05562, both "controls FAILED", the second redding
    `test_selfchecks.py::test_core_selfcheck_in_process[substrate.amdahl_gate]`) are NOT laundered into a
    stated precision span: that the instrument catches its own bad draws is the opposite of imprecision.
13. **C13 — the neutered arm ends the paired block narrower than the live arm, so the ceiling compares
    brains of different size.** *Refuter: STANDS IN SUBSTANCE, WRONG IN ITS NUMBERS AND ITS CAUSE — the
    cited "177 vs 189" is a cross-seed range of the LIVE width, not a live-vs-neuter pair; measured per
    arm the gap is seed 0 165/165/165, seed 1 177/165/177, seed 2 189/177/189; and the neuter GROWS LESS
    rather than not at all. A larger defect the finding missed: on seed 0 the whole 12-unit gap is created
    by the live-only PROFILE block after the timing is over.* CHANGED: `amdahl_gate.py::live_reading`
    walks all three arms immediately after the paired loop and BEFORE `_profile_shares`, reports
    `n_lif_paired_block`, gives the neuter derivation a `n_star_paired_block` on the paired block's LIVE
    width printed BESIDE the published number, prints the neuter arm's own width and the profile block's
    start and end widths; §3.1 takes the dated line naming this as the THIRD bias with the measured
    widths. NOT CHANGED: growth is NOT held fixed (a post-hoc third class-level patch with its own
    confound that would invalidate the five recorded readings and stop measuring the deployed tick); no
    reading is re-taken; the exemption rows keep "177–189 live LIF units", with the timed widths recorded
    in §3.1 rather than swapped into the rows — a single live-vs-neuter pair there would be seed 2's, the
    seed no verdict is read from.
14. **C14 — the host condition the amendment makes decisive is asserted for both re-readings with no clock
    time.** *Refuter: PARTIALLY — the load-bearing half is confirmed (the condition is an assertion, never
    an observation; the file carries two clock times and both belong to the boot readings), and the
    timeline half is REFUTED: chat_learn.log's launcher timestamps bracket the epoch and leave ≈450 s for
    two ~85 s readings.* CHANGED: §3.2 records the bracket and the honest word, PLAUSIBLE BUT UNVERIFIED;
    §3.3's "under any host condition tried" is qualified; the gate stamps `started_utc` beside `elapsed_s`
    so future readings are locatable. NOT CHANGED: no live-port probe is added — it would couple a
    scratch-home instrument to live production state and make a registered gate's output host-dependent;
    and neither reading is re-taken, because 2.9(a) fixed the re-read at ONCE, 2.9(b) already disposes of
    the second straddle, and a reading today measures today's host.
15. **C15 — `_count_lif_in` follows module globals, contrary to the docstring that is the second width
    instrument's whole safety argument.** *Refuter: REPRODUCED VERBATIM, and the leak is transitive
    (module → dict → list → 3 found); classes ARE skipped, so the docstring is half-true.* CHANGED:
    `types.ModuleType` is added to the skip list, with neutrality MEASURED not assumed — 141 / 14,976 at
    one brain and at three against a 423 process census, zero modules reached, so the leak was LATENT and
    no recorded number moved; the docstring's second sentence now says the cross-check runs with exactly
    one brain alive and CANNOT by construction detect a sibling walk, and states the fact that does
    discriminate. NOT CHANGED: `census_cross_check` is NOT re-sited — it is a hard conjunct of
    `controls_ok` whose number is already recorded, and the "process == N × per-brain" equality it would
    move to holds only before any arm is ticked.
16. **C16 — the gate prints the control arm's liveness count and hides the measurement arm's.** *Refuter:
    reproduced at the site; the two keys appear only at their write and assert sites, never in a print,
    and `_probe_install` discarded the tick's phase.* CHANGED: `_probe_install` returns
    `(installed, calls, phase)`, the four unpack sites are updated, the reds carry `pad_phase` /
    `sham_phase`, the report carries `neuter_probe_phase` / `sham_probe_phase`, and `_print_reading`
    prints the measurement arm's liveness WITH its probe phase on one line before the sham's — because the
    count varies ~4× with phase (4,324–5,124 awake against 1,264–5,872 asleep) while every paired sample
    is awake, so an unlabelled count would be a worse log line than none. NOT CHANGED: the probe does not
    SELECT an awake tick — that would redefine a control after its readings are on file, and it buys
    nothing, since no tick of either phase reached zero calls.
17. **C17 — bare `main()`'s cost is published as one flat number with no host qualifier.** *Refuter: STANDS
    in substance and reproduces HARDER (308.5 s, not the filed 216.6 s), but the finding's cause is
    misattributed and one of its citations does not exist — the driver is ambient contention (100 % CPU,
    987 MB free, 12 competing processes), not daemon boot state, and `s6_build_report.md` is not in the
    tree.* CHANGED: §3.5 item 1's appended clause qualifies the 95 s by MECHANISM (≈ 3 arms × W × the wake
    p50 the reading itself prints, plus fixed cost) with both measured endpoints, and records that on a
    contended host the instrument's PRECISION degrades too. NOT CHANGED: `main()` is NOT restructured and
    the live reading does NOT move behind a flag — the exit code carries seven live-brain controls the
    three harness reds cannot check, no test charges them, and the module is dormancy-exempt at module
    level, so the route would make `live_reading` reachable-but-uncharged. It stays an open
    orchestrator/owner call with its two prerequisites named. No "95–217 s" range is written.
18. **C18 — the disposition commit calls 2.9's aggregation rule pre-registered.** *Refuter: the ordering
    fact reproduces and is not in dispute, but §2.9 IS the disclosure — its own heading says so — and the
    genuinely misdescriptive instances are elsewhere: §3.3's heading, §3.6's wording, and the record
    commit's subject.* CHANGED: §3.3 is retitled "The verdict, by the rule of record" and carries a
    provenance line naming which rule was fixed when; §3.6's clause is rewritten to name 2.3 and 2.9
    separately. NOT CHANGED: no pooled or per-tick-ratio robustness figures are imported — they come from
    a 06:36 capture that post-dates head and whose seed 2 reads DETERMINATE DECLINE, contradicting §3.2's
    table. The robustness is discharged from the FIVE recorded readings alone: seed 0 is DECLINE under 2.3
    alone on both its readings, and PROCEED is unreachable at a maximum lower bound of +0.0043.
19. **C19 — 2.9(c)'s quorum rule is the aggregation Part F's preamble forbids "anywhere".** *Refuter:
    reproduced, and the non-disclosure is complete — but three corrections CAP it: 2.9(c) is
    unanimity-among-determinate with a quorum floor and refuses rather than resolving; the rule did no
    work; and the post-reading timing is disclosed in 2.9's own heading.* CHANGED: §3.5 item 9, following
    the disposition `w1-goal-curriculum.md` §5 already set for this same prohibition. NOT CHANGED: the
    scope question is STATED, NOT RESOLVED, exactly as w1 left it; 2.9 and Part F are not edited; and the
    row is not recomputed from a pooled statistic, which would be a second unregistered post-hoc rule
    answering a complaint about an unregistered post-hoc rule.
20. **C20 — FR-005 governs a site measured slower AFTER conversion, and no conversion was built.**
    *Refuter: reproduced at every site; the finding named the WRONG replacement authority and missed
    `contracts/conversion.md` C1.3, the on-point rule S6 actually satisfied.* CHANGED: §3.5 deviation 8;
    the living prose in `dormancy_gate.py`'s two rows. NOT CHANGED: evidence §2.1/§2.3, §3.3's verbatim
    quote, PLAN-FINAL's dated annotation and `0370c1c`'s body are left exactly as they are — §3.5 item 8
    is the record for all of them.
21. **C21 — the "seed 0 (quiet, extra)" reading has no artifact, was not authorized by the amendment that
    preceded it, and its number is load-bearing in source.** *Refuter: every sub-assertion reproduces, but
    three corrections favour the record — the reading is disclosed by name in three committed places, no
    verdict rests on it, and its ceiling identity checks to 4 dp; the residual defect is narrower, that
    three readings are collapsed into "two determinate seeds".* CHANGED: §3.5 item 10 names it UNREGISTERED
    and transcribed-from-console; the attribution is separated at both `dormancy_gate` rows and specs/008
    T001 ("0.005 raw on the two determinate recorded seeds (bands to 0.015), 0.017 on an UNREGISTERED
    quiet re-read of seed 0 (band to 0.038)"). NOT CHANGED: the reading is neither re-run to backfill an
    artifact (that would fabricate provenance — four seed-0 point ceilings now differ 7×) nor deleted
    (that would tighten the committed claim by removing its least favourable datum).
22. **C22 — every green battery silently takes and discards a complete seed-0 reading.** *Refuter:
    reproduced end to end and measured — the exact battery invocation returns rc 0 in 181 s with a full
    reading on stdout, and `run_selfchecks` writes a child's stdout only when rc ≠ 0.* CHANGED: §3.5 item 1
    carries the appended clause, including that "five" is the KEPT count under 2.9's host condition, not
    the count this instrument has produced. NOT CHANGED: no code — the cheap-bare-invocation route is the
    same open owner call as C17 and owes a test that charges `live_reading` first.
23. **C23 — numbers from the INDETERMINATE seed are selected inconsistently between the ranges.**
    *Refuter: reproduced, but in a much narrower form — exactly ONE landed number is unsupported by any
    consistent reading rule, the profile upper 2,828 (seed 2's BOOT reading) sitting beside a neuter floor
    4,018 and a ratio end 1.47 that are seed 2's RE-READ. 2,828 and 4,018 cannot both be right.* CHANGED:
    §3.3's appended correction names the reading rule that would explain the rest ("seed 2's re-read
    supersedes its boot reading") and the one number it does not support, and records why the disagreement
    range moved from 2.9's "2.4–4.4×" to "1.5–2.5×". The ranges are restated over ALL FIVE readings, which
    dissolves the selection rather than repairing it. NOT CHANGED: the ceiling range 0.005–0.017 is not
    narrowed — 0.0172 is seed 0's OWN determinate quiet reading and is the honest wide end — 177–189 stays
    as the description of the built width, and 1,788 is not retrofitted into any landed range as if it had
    always been there.
24. **C24 — specs/008's quickstart still tells a reader to BUILD Gate Zero.** *Refuter: the narrow
    observable reproduces, but all three supporting assertions are refuted — `plan.md`:131 and :141 carry
    the identical superseded assertion, the disposition's pre-registered scope was "specs/008 gets its
    `tasks.md`" (creation of one file, not currency of the directory), and the stale-baseline half is
    already disclosed at §1. Decisively, the fix's own prescribed command DOES NOT RUN: bare `python` on
    this host is 3.11 and the free-threading import guard refuses it.* CHANGED: ONE dated succession
    section appended to BOTH `quickstart.md` and `plan.md` — either both or neither — recording where Gate
    Zero landed, that `tools/convert_gate.py` will not be built (p4c ratchet), that T004 still owes the
    T0–T3 ladder, that the 74.26 ms baseline is stale while SC-001 still pins it, and the runnable command
    with the pinned `.venv-lake` interpreter. NOT CHANGED: `:7`, `:56`, `:131`, `:141` and the 74.26 blocks
    are the record of what was believed in 2026-08 and take no edit; `spec.md` SC-001 is NOT re-recorded —
    that is its own task with its own measurement.
25. **C25 — S-R4's re-registration glosses its conjuncts in PASS polarity under a column headed "red".**
    *Refuter: the textual observation reproduces and the polarity clash is real within the arc (§1 calls
    the RED-form clause "S-R4's timing conjunct"), but the cell is self-disambiguating — "conjunct B has
    no reachable red" is only meaningful if conjunct B is a criterion — and `plan.md` already uses
    "conjunct" in pass polarity at L-R2.* CHANGED: one appended reading note at the end of the S-R4 cell:
    "conjunct A/B" name the PASS conditions; the red is the negation of either. NOT CHANGED: the
    parentheticals are not restated in red polarity (an append would give the cell two glosses in opposite
    polarities and contradict its own next clause; a replace would edit dated re-registration text and
    redefine what the conjuncts denote after the reading), and evidence §1's opposite usage is left as the
    record.
26. **C26 — when the gate's controls fail, no term is named.** *Refuter: STANDS on every load-bearing fact
    — `controls_ok` is exactly eight conjuncts by AST, the neuter's two keys reach no print site, and a
    CONSTRUCTED failure driven through the real printer shows every printed conjunct HOLDING beside
    `controls FAILED`; the real 198.3 s failure in the log is undeterminable for exactly this reason.*
    CHANGED: the roster is built ONCE in `run_reading` as `rep["controls"]` and the printer consumes it,
    so it cannot drift from the exit code; `main()` prints the roster on every run and names the FALSE
    terms on FAILED, placed last to survive `run_selfchecks`' 2000-char tail truncation. NOT CHANGED: the
    eight predicates are NOT re-expressed in the printer — conjunct 7 compares against the REQUESTED tick
    count and is not reconstructible from `rep`, so a printer-side copy would misreport it under
    `--ticks N`. C14's "good half" landed with it: a wall-clock `started_utc`, no port probe.
27. **C27 — FR-005 is quoted for a disposition its own sentence does not cover.** *Refuter: STANDS on the
    text, with the finding's quote misattributed (it belongs to `0370c1c`'s message, not T002) and its
    site list short by the most consequential one, `dormancy_gate.py`'s live printed prose; and the record
    is UNDER-citing — FR-010 and SC-006 literally say "declined with the measurement that justified
    declining".* CHANGED: the same correction as C20, merged — the `dormancy_gate` rows now cite FR-010 /
    SC-006, FR-004 contrapositively, `plan.md` Standing Invariants 1 and 3, and `contracts/conversion.md`
    C1.3. NOT CHANGED: `spec.md` is NOT amended with a dated FR-005 clause (that would post-hoc authorize
    a verdict already taken and erase the evidence the paraphrase was imprecise), `0370c1c` is NOT
    rebased (it sits under the closing record commit the arc's battery ran on), and specs/008 `tasks.md`'s
    FR-005 tags at lines 9/26/40 are left as they are — the refuter marked that widening optional and item
    8 discharges the finding.

## 3.8 Post-push amendment (2026-09-15): the sham controls read "within noise of zero" as pre-registered

**What happened, numbers first.** The T063+T071+T070 stack's battery on head `5b6cb2a` (68 commits over this
row's push) read pytest RED twice, on ONE head, on this instrument's sham controls and nothing else: run 2
(00:21-01:51 local) on the constructed red `red_sham_neuter`, ceiling **-0.1025** against |c| <= 0.05 with band
**(-0.28571, +0.00911)**; run 3 (01:54-03:27) on the live reading's `sham_within_noise_band` control inside
`tests/test_selfchecks.py::test_core_selfcheck_in_process[substrate.amdahl_gate]`, sham arm **+0.05844** with band
**(-0.0802, +0.10295)** (RAW p50 live 64.22 / neutered 64.92 ms at width 189, the reading itself INDETERMINATE by
its own floor rule). Diagnosed before any change: on the idle host minutes after run 2 the same test PASSED and
five direct `red_sham_neuter()` readings were +0.0046, +0.0001, +0.0018, -0.0087, -0.0003 with bands about
+/-0.01 -- the battery-time bands were 20x wider; the host carries the always-on brain daemon (6.8 GB, 5,163
CPU-s at 01:52). The stack's one change on the live tick, T070's suffix-array LZ76 in `integration_measure`
(reached every tick through `region_coherence`), was probed as a cause and is NOT one: on tick-size raster strings
(13 regions x 16..256 samples) it reads 0.524 vs 0.269 ms at 208 chars and is FASTER above 300 chars (0.551 vs
1.090 ms at 416, 0.920 vs 4.350 at 832, 1.352 vs 16.692 at 1,664), jitter comparable. This row had already
measured the loaded sham at +0.063..+0.102 (3.7 item 12's "0.0478/0.0535 calm against +0.063..+0.102 loaded")
and recorded two REFUSED readings (-0.01883, -0.05562; 3.7 C12) without changing the bar.

**What changed, and what did not.** `amdahl_gate.within_noise_of_zero(ceiling, band)` now decides BOTH sham
controls: the magnitude bar |c| <= 0.05 as built, OR the sham's own paired bootstrap band containing zero.
The pre-registration's words (2.2 iv / 2.5) are "within noise of zero"; the 0.05 was the build's
operationalization, recorded in `red_sham_neuter`'s docstring as "a criterion choice made AFTER seeing data".
A band that EXCLUDES zero never holds through the new clause -- the instrument's own bad draw (C12) stays a
FAIL -- and the mirror rule (band excluding zero as a FAIL) is NOT reinstated: the build removed it for cause.
The bar is not moved; no new number enters; the band the instrument already printed "reported, not asserted"
is allowed to decide. Reds run first: `tests/test_amdahl_gate.py::test_within_noise_of_zero_holds_by_magnitude_or_by_its_own_band_and_fails_on_a_band_excluding_zero`
constructs both failing cases and both battery readings. NOT changed: `W_TICKS`, `RED_TICKS`, the reading's
verdict rule, its floor rule, the controls roster's other seven terms, and 2.2/2.5's text. This amendment is
battery-verified only (obs #35) and is the stack's stacked commit for a red on an earlier arc's artifact
(owner rule 2026-09-10); obs #2175 records the transferable rule (a timing control with a fixed epsilon on a
loaded host owes a refusal or its own band, never a re-run until green).

## 3.9 Post-push amendment (2026-09-16): a sham band wider than the decision's resolution is a NOISE REFUSAL, not a verdict

**What happened, numbers first.** The T077+T078+T079 stack's battery run 2 on head `0bfb3d7` (00:40-02:19 local,
`scratchpad/battery-stack6`; run 1 died at the session usage limit at pytest 48 % after 13 host-load F marks
that pass alone, obs #2188) read every stage green -- record assert, certify, ruff, format, 355/355 self-checks,
16/16 gates, coverage 89.57 % -- and pytest RED on exactly one test, again this instrument's live sham control
inside `tests/test_selfchecks.py::test_core_selfcheck_in_process[substrate.amdahl_gate]`: sham arm **-0.05595**
with band **(-0.3138, -0.01658)**, 0.297 wide against the 0.15 resolution the gate's decision needs (`TOLERANCE`);
RAW p50 live 66.39 / neutered 68.11 ms at width 189, ceiling -0.0259 with band (-0.36596, +0.03158), the reading
itself INDETERMINATE by its own floor rule. In the SAME run the constructed red `red_sham_neuter` read **-0.0060**
with band (-0.00995, +0.00633) and held by magnitude, and the run_selfchecks stage had read this same no-argument
command PASS (99.7 s) about seventy minutes earlier on the same tree; the reading started 02:09:48 local and took
69.4 s while the always-on brain daemon was in its sleep phase (circadian `asleep`, 7.3 GB, 4,564 CPU-s since
00:27). Nothing else ran on the host (the panel and the workflow were finished before the battery; obs #2188).
3.7 C12 had refused a sham of -0.05562 on 2026-09-09 -- the same magnitude. The stack's three arcs did not touch
this module (last change 53172a8, 2026-09-15); the tree is not the cause, the draw is.

**What changed, and what did not.** 3.8 landed one half of obs #2175's prescription (the band containing zero
holds) and left a band that EXCLUDES zero a FAIL. The other half -- "if the bootstrap band's width exceeds the
resolution the gate's decision needs (0.15), the red reports NOISE, unresolvable at this load; re-run idle,
rather than PASS or FAIL" -- is now the third outcome of `within_noise_of_zero`: `(False, "unresolvable")` when
the band excludes zero AND its width exceeds `TOLERANCE`. It is not a pass: the roster term is still False and the
exit code stays 1, so a battery still reads the stage red and the remedy is a re-run on a quiet host with the
cause named -- never a stitched stage, never "re-run until green". It is not a bar move: no number changes, a
NARROW band excluding zero stays `(False, None)`, and the width test runs AFTER the two holding clauses, so no draw
that held before reads differently. `run_reading` records the clause (`sham_noise_clause`) and `main()` prints
one `NOISE REFUSAL (3.9)` line naming the width and the remedy; `red_sham_neuter`'s message says "NOT held --
NOISE REFUSAL" instead of "held by: None". Reds run first: `tests/test_amdahl_gate.py::test_within_noise_of_zero_clauses_and_their_reds`
now constructs this battery's draw and asserts it is named unresolvable, and that a narrow band excluding zero is
still a FAIL. NOT changed: `SHAM_NOISE_BAND`, `TOLERANCE`, `W_TICKS`, `RED_TICKS`, the reading's verdict rule,
its floor rule, the roster's other seven terms, and 2.2/2.5's text. This amendment is battery-verified only
(obs #35) and is the stack's stacked commit for a red on an earlier arc's artifact (owner rule 2026-09-10).
Standing disposition for the battery: a NOISE REFUSAL on this control is re-run on a quiet host; three of the
last five batteries have read this control red under the daemon's sleep-phase load with no tree defect, so a
refusal that recurs is an operational question for the owner (the daemon's load during a battery), not a bar.
