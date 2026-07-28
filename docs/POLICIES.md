# Development Policies (P1–P5)

Five standing policies govern how this brain is built. They are **mechanized** where possible (a CI
self-check that fails on violation) and otherwise enforced by review. Each has an **honesty guard** —
the rule that stops the policy from being satisfied by theater.

**Enforcement status, stated plainly, because it is easy to assume a gate exists:**

| | mechanized in CI | enforced by review only |
|---|---|---|
| P1 no-hardcode | — | yes |
| P2 adopt-or-merge | — | yes |
| P3 no dormancy | `substrate/dormancy_gate.py` (per-MODULE) | the per-METHOD amendment; the charge tier |
| P4 one brain | `substrate/replication_guard.py` | — |
| P5 use the GPU where it helps | — | yes |
| e-prop not backprop | — | yes |

**Provenance of the 2026-07 amendments.** Three independent audits mapped this codebase against
external frameworks — predictive coding, the Thousand Brains Theory, and quantum-consciousness
theories. They proposed five policies between them; **one** survived adversarial review, and the
rejections were as instructive as the survivor. Two proposed gates were *implemented before adoption*
and killed by their own output: one scored 0 failures / 47 UNCERTAIN (it could not catch its own
motivating case), the other produced 58 failures where 1 was claimed (adoption would have meant a
58-entry rubber-stamp allowlist). That bar is now itself a rule — see *Adding a policy* below.

What the audits mostly returned was not governance but **defects**, and they clustered into one
recognisable family, recorded under P3.

---

## P1 — Brain-native / no-hardcode

Every behaviour-arbitrating value must be a **learned weight** or a **running-baseline-relative** form
(three-factor R-STDP / e-prop; running EMA or `SignalCalibrator` quantile), not a
magic constant. Corollary: **never keep a constant fixed** — when a hardcode resists replacement, the
blocking constraint is usually the bug, not the number.

**Honesty guard (defeasible only by gate evidence):** a constant may stay fixed in exactly three proven
cases — (a) it is a fixed point on a **normalized** output (e.g. a threshold on a `tanh` confidence in
[-1,1] is dimensionless, like `p ≥ 0.3`); (b) a running-baseline replacement is **not reproducible from
the persisted policy** (e.g. a durable-reload contract persists weights but not the running baseline);
(c) the constant is **inert** (default-off, zero live callers) so a learned form would never charge.
Reuse anchors: `basal_ganglia/r_stdp.py` (`w += lr*d*e`), `substrate/signal_calibrator.py`, the
`_*_base` running-EMA idiom, `cortex/brain.py::_drive_w`.

## P2 — Adopt-better **or merge**, then retire the inferior (non-additive)

When two mechanisms do the same job, exactly **one** survives. There are two legitimate routes to that,
and they are co-equal — the choice between them is a judgement about the mechanisms, not a fallback:

1. **ADOPT** — one is better on evidence; it becomes THE path and the other is **deleted**.
2. **MERGE** — neither dominates, or each holds something the other needs; build the **one** mechanism
   that supersedes both, then retire both originals. A merge is not a compromise or a coexistence; it
   ends with fewer moving parts than it started with.

What is forbidden is the third option: leaving both live "additively". That dilutes the signal, grows
bloat, and leaves every later reader guessing which one is authoritative.

**Worked merge:** `cortex/neuron_zoo.py` — the repo's neuron models each had their own constructor and
step signature, so a population was always homogeneous. The zoo is one heterogeneous population behind
a single contract, and it earns the name because it *"does NOT re-implement any dynamics -- it IMPORTS
the existing neuron modules and puts a thin adapter in front of each"*. One surviving interface, zero
duplicated dynamics. Smaller-scale: `PredictiveCodingNetwork.learn()` now calls `learn_step()` (v4.71),
so the offline trainer and the live tick apply the identical rule — one definition, not two.

**Honesty guard:** prove the adopt-or-merge is behaviour-equivalent (or superior) **before** deleting —
an A/B on identical inputs, gate-verified no regression, and any unique capability of the retired
mechanism ported first. Reject it with evidence if it regresses (verify-first works when it says *stop*).

**Merge-specific guard — a merge must actually reduce.** "Merging" that keeps both code paths behind a
flag, a mode switch, or an `if` is additive with extra steps, and is the failure this policy exists to
prevent wearing a more respectable name. Count the survivors: if two implementations of the same job are
still reachable when you are done, it was not a merge.

**"Better" requires a measurement that exists.** If the two mechanisms cannot be compared by any
benchmark in this repo, the swap is not P2-eligible and the answer is *build the benchmark first*, not
*pick the more fashionable one*. Worked case: the Thousand Brains audit proposed replacing the Global
Workspace (`thalamus/global_workspace.py`, one broadcast slot, winner-take-all) with peer-to-peer
columnar voting. Both are serious theories of the same phenomenon; SNDL has **no consensus benchmark**
— no ambiguous multi-source evidence with a known right answer — so neither could be shown better, and
the swap was refused. An external framework being newer, or more elegant, or better publicised is not
evidence. This guard is what stops P2 from becoming a fashion treadmill.

**Corollary — a research direction is not a policy.** "Adopt architecture X" governs *what* we build;
policies govern *how*. Proposals of that shape belong in a design note, not in this file.

## P3 — No dormancy

**Every part of the brain must never remain dormant; all present AND future builds must be wired and
utilized appropriately.** A faculty is not "done" until it is one of three dispositions:

1. **wired + charged** — imported by a live path AND its output routes into behaviour (decision / value
   / learning / reply), proven to change an outcome;
2. **retired** — superseded / no honest consumer → deleted (this is P2 applied to dormancy);
3. **exempt** — legitimately not on the live tick, listed with a reason (train-tier trainer, benchmark
   harness, entrypoint-only tool, or awaiting a prerequisite such as a real sensor).

**Mechanized:** `substrate/dormancy_gate.py` (in `run_selfchecks` CORE) builds the static import graph
and asserts every REGIONS/TORCH_TIER module is **reachable from a live root/entrypoint OR in the EXEMPT
allowlist** — a new build that no live path imports and that is not exempted FAILS CI.

**Honesty guard (the crux):** reachability proves a module is *imported*, NOT that its output changes
behaviour. "Utilized **appropriately**" (does the wire actually *charge*) is a second, per-faculty tier,
proven by the runtime **neuter-A/B probe** (`scratchpad/smoke_drive_charge.py` → "BEHAVIOURAL CHARGE OK"
vs "CONSUMED-BUT-NOT-DECISIVE") during a wiring wave — too expensive for CI. So "never dormant" must NOT
become "force-wire everything": a wire that does not charge is **reverted and retired/exempted**, never
faked with a dead consumer. (History: a coverage-audit once flagged PFC / amygdala / cortical-column /
synfire as dormant; every one was REFUTED on runtime A/B. Static grep proves import-absence, not
behavioural inertness — mark unproven cases UNCERTAIN, not dormant.)

### The dormancy family — six observed shapes of "live but inert"

The module gate asks one question: *is this module imported from a live root?* Every defect below
answered **yes** and was still dead. They are the same failure wearing different clothes, and naming
them is most of what makes the next one findable. All six are real, all were found by audit rather
than CI, and all are cited from this tree:

| # | shape | worked example |
|---|---|---|
| 1 | **method never called** — the module is live, one of its methods is not | `PredictiveCodingNetwork.learn()` — inference ran every tick, learning had one caller: its own `__main__`. Fixed v4.71 |
| 2 | **return value discarded** — a live call whose result goes nowhere | `connectome_router.step()` at `cortex/brain.py:2000`; its only export reaches telemetry alone, measured constant 0.250 over 122 ticks |
| 3 | **key-space mismatch** — a learned path structurally unreachable | `thalamus/reentrant_workspace.py` potentiates on `(source, winner)` at `:271`, looks up `(subscriber, winner)` at `:289`; disjoint live, so the learned re-route took 2 values in 122 ticks |
| 4 | **write-only consumer** — a faculty fed every tick, never read | `cortex/agency_monitor.py` — `observe()` runs live; nothing reads `agency()` / `controllability()` / `status()`, and no key reaches telemetry |
| 5 | **constant wearing a signal's clothes** — present, plausible, never varies | `spikes.midbrain` (its novelty term is never enabled); `integration_competence` (assigned only in `Brain.__init__`); the `gated` trio (never *computed*, not merely never applied) |
| 6 | **silent default** — a lookup table missing the cases that matter | `_ALLO_MOVE` had no entry for the four `JOINT_*_ROTATE` tokens, so `.get(action, (0.0, 0.0))` meant the commands that actually move the arm contributed **zero** to path integration. Fixed v4.72 |

**The transferable rule:** *presence is not participation.* Being imported, being called, being in the
snapshot, and having a value are four different things, and none of them is *changing an outcome*. When
auditing a faculty, name which of the four you verified — and shapes 5 and 6 are why
`substrate/brain_telemetry.py::PROVENANCE` exists: a signal that cannot vary must say so beside its own
number, or a dashboard turns it into a claim.

### P3 amendment — dormancy is per-METHOD, not per-module

**If a module on a live path computes a prediction error, free energy, or any other learning signal,
the update rule that consumes it must also be called from a live path.** A learning-verb method on a
reachable module with no live-path call site is dormant, and must be wired, retired, or listed in
`EXEMPT_METHOD` with a disposition and a reason.

**Why module granularity is not enough — a worked case, since fixed.** `dormancy_gate.py:71` keys on
the module basename, so a module passes P3 clean with half its capability dead.

`cortex/predictive_coding.py` was exactly that. The brain settled its 3-level Rao-Ballard net on every
wake tick (`cortex/live_brain.py:161`) with the error-neuron architecture implemented faithfully — while
`PredictiveCodingNetwork.learn()` had **one** call site in the tree, inside the module's own `__main__`
self-check. The live path called `reset()` + `infer()` only, against weights seeded random at
construction that never changed (verified: `w1` was `array_equal` before and after 40 further wake
ticks). Inference wired, learning never called, and the module-level gate saw a live, imported,
fully-compliant module.

**That case is now closed** (v4.71): the per-sample rule was extracted as `learn_step()`
(`predictive_coding.py:244`) and is called every tick at `cortex/live_brain.py:185`. Measured over 400
ticks — prediction-error energy −67.8% with the call, +9.1% without it (weights frozen).

Fixing it does not retract the argument; it completes it. The blind spot is no longer hypothetical: a
half-dead module demonstrably passed the module gate, for as long as that feature had existed, and was
found by an audit rather than by CI. That is the evidence this amendment rests on. The candidate set
below was produced by the import-scoped pass described next and has NOT been re-adjudicated since the
predictive-coding fix — treat it as a starting list to verify, not a verdict.

**Mechanization: SPECIFIED, NOT YET BUILT.** A pure static AST pass (~1 s, no runtime cost). Two
details are load-bearing, both found by implementing the naive version first and watching it fail:

1. **Import-scoped resolution.** Matching bare `ast.Attribute` call names across all reachable modules
   scores 0 failures / 47 UNCERTAIN — the name `learn` resolves in 20 unrelated modules, so it cannot
   even catch its own motivating case. A call site counts only if its module imports the defining
   module.
2. **Same-module callers COUNT, except under `if __name__ == "__main__"`.** Excluding the defining
   module produces a real false positive (`plasticity/meta_plasticity.py:261` legitimately drives
   `_Unit.learn` at `:131`). The `__main__` guard is precisely what isolates the genuine case.

Under that rule the candidate set was 5. One — `cortex/predictive_coding.py` — has since been wired
and charge-proven (above). The remaining four are unverified since: `hippocampus/affective_memory.py:173`,
`neuromodulation/drive_reduction.py:196`, `neuromodulation/empowerment.py:86`,
`plasticity/meta_plasticity.py:131`. Each still needs the three-way P3 disposition applied to it —
wire-and-prove-charge, retire, or `EXEMPT_METHOD` with a reason — and none of that is done.

**Honesty guard.** (a) P3's neuter-A/B tier applies unchanged — a call site proves a *caller*, not a
*charge*; adding a dead consumer to silence the gate is the failure this policy already forbids.
(b) An unresolved name is UNCERTAIN and hand-adjudicated, never auto-dormant. (c) `EXEMPT_METHOD`
entries tagged `wire` that outlive their wave are reported as a standing TODO, so exempting everything
is visible rather than silent.

**Stated honestly: this amendment would have caught no *earlier* incident.** The audited history before
it is *consumption* gaps (a producer with callers whose output nothing reads) and an unsupplied
constructor argument — a call-site check passes all of them. It earns its place on the predictive-coding
case above: found by audit, confirmed by measurement, and invisible to the module gate throughout.

**Standing caution for this section.** It was written while its motivating case was still open, and went
stale within two commits when that case was fixed — the text asserted dead code that had become live.
An amendment justified by a live example has to be re-read whenever that example changes, or the policy
file starts describing a codebase that no longer exists.

## P4 — One brain / no replication

**There is only 1 brain.** A build must **utilize** the canonical brain functionality, never
**re-implement** it — that causes bloat. The canonical brain is a clean 3-layer **composition**:
`cortex.brain.Brain` (cognition + circadian clock) → composes `cortex.live_brain.LiveBrain` (perception
+ workspace) → composes `brainstem.autonomous_loop.AutonomousArm` (action + reward + arm + SAFETY).

**Composition is NOT replication** (important): `Brain` owning an inner `LiveBrain` + wake/sleep loop is
the *correct* design — it delegates, it does not re-implement. The bloat P4 targets is a **second**
definition of a stack class (a parallel "brain"), a consumer standing up its **own** brain + drive-loop
+ reply engine, or duplicate world-model / Q-learner families.

**Mechanized:** `substrate/replication_guard.py` (in `run_selfchecks` CORE) asserts each canonical stack
class (`Brain` / `LiveBrain` / `AutonomousArm`) is defined **exactly once**, in its canonical module,
unless a known duplicate is on the `PENDING_RETIREMENT` allowlist (with a reason). The single sanctioned
construction path is **`cortex.brain.make_brain()`** — consumers obtain the brain through it rather than
calling `Brain(...)` directly. "One brain" = one canonical class + one construction path, not a
process-wide singleton (the daemon, a chat session, and a test each legitimately own an instance).

**Honesty guard:** before retiring a replica, port its unique capability into the canonical path and
verify nothing is lost (P2 discipline).

## P5 — Use the GPU for everything it actually speeds up

**Standing owner authorization (2026-07-27): the RTX 5070 is available to the whole project, and any
work that the GPU genuinely accelerates SHOULD use it.** Default to GPU for that work; do not leave it
idle out of habit, and do not treat CUDA as a special-occasion tier. Hardware: RTX 5070, 12 GB, torch
`2.11.0+cu128`, verified `cuda.is_available() == True`. Env `NORD_GPU=1`; the verified environment is
`.venv-cu128` (the CPU gate env `.venv-lake` stays for deterministic `run_selfchecks`/`pytest`).

**"That it actually speeds up" is the operative clause, and it is decided by measurement, not by
assumption.** Ship the GPU path when a benchmark on the real workload shows it faster; do not ship it
because CUDA is available. This is P2's "better requires a measurement that exists" applied to hardware.

**MEASURED CROSSOVER — do not re-litigate this without new numbers.** For SNDL's elementwise
spiking-update kernels, eager torch-CUDA on this card beats numpy-on-CPU only between
**N ≈ 24,576 and 32,768** elements. Every population on the live tick is **24–240** wide — 100× to
1300× short of that. Consequences, all measured:

- Vectorizing `cortex/prefrontal_executive.py::_decide_spiking` at its real width of 38 **regresses**:
  0.36× on numpy, 0.05× on torch-CPU, **0.01× on torch-CUDA**. Kernel-launch latency alone exceeds the
  whole per-call budget.
- `cortex/lif_neuron.py::step` looks like the hot path (244,768 calls per 10 ticks) and is not: neutering
  it entirely moves an awake tick 43.6 → 41.2 ms, a **6% ceiling**. Moving it to CUDA cannot beat that
  ceiling and would pay launch latency for every one of ~24.5k tiny calls.
- The real win on that hot path was **CPU-side batching**, not hardware: `AttractorNet.accuracy` folded
  36 independent recall trials into one array pass for a measured **6.93×**, bit-exact (v4.68).

**Where the GPU therefore pays today, and should be used without hesitation:** the `[train]` / `[gpu]`
tiers — surrogate-gradient training, `hippocampus/ng_snn.py`, `cortex/neurogenesis_engine.py`, the
torch readout studies under `reasoning/brain_adapters/`, and any offline sweep over a real dataset.
These are wide, batched, and repeated, which is exactly the regime the crossover favours. The daemon
already sets `NORD_GPU=1` and runs under `.venv-cu128` so this tier lands on the card by default.

**Where it does not pay today:** the live wake/sleep tick, at current population widths. That is a
statement about widths, not a prohibition — **the moment a live population crosses the measured
threshold, this policy says put it on the GPU.** Re-measure when populations grow; the crossover is a
number, not a doctrine.

**Honesty guard.** (a) A GPU path ships with its benchmark, on the real workload, against the CPU path
it replaces — a speedup asserted from `cuda.is_available()` is not a speedup. (b) Report regressions:
"we tried CUDA here and it was 0.01×" is a result worth keeping, and this policy exists partly to stop
that experiment being repeated. (c) The deterministic CPU gate stays authoritative — `run_selfchecks`
and `pytest` run on `.venv-lake` so results are reproducible on machines without this card, and a GPU
path must not become the only way to verify the project. (d) Correctness first: a GPU rewrite carries
the same bit-exactness burden as any other P2 swap.

---

## Cross-cutting invariants (never relaxed by any policy)

- **Safety envelope untouched:** the five brainstem files — `reflex.py`, `serial_arm.py`, `arm_sim.py`,
  `arm_config.py`, `interpreter.py` — are modified only with explicit approval; the reflex E-STOP and
  cortisol/safety paths are never relaxed. Every wired drive enters selection **after** the E-STOP
  early-return, so it is safe by construction (proven: max drive + distance 0 → EMERGENCY_STOP).
- **E-prop, not backprop (review-enforced, NO GATE).** Any backprop consideration pivots to **e-prop**
  (Bellec 2020) — a local eligibility trace times a broadcast learning signal; no BPTT, no weight
  transport — verified **cosine > 0.5** against an independent reverse-mode BPTT gradient. Verification
  anchor: `reasoning/eprop.py:38-40`, with the independent gradient at `:191-192`.
  This rule governed the project before it was written down (it is why v3.79 pivoted all learning to
  e-prop on measured evidence: 73.14% vs the repo's backprop SSM at ~32%), which was the gap — a rule
  enforced only by whoever remembered it.
  **It is NOT mechanized, and a `.backward()` tripwire would be theatre:** it passes green today while
  `cerebellum/jepa_world_model.py:272` performs explicit weight transport on every wake tick (the same
  `Wz` used forward and transposed backward, in hand-written numpy) and `hippocampus/ng_snn.py:137`
  runs a closed-form batch solve over its whole buffer each sleep tick. Neither uses autograd; both are
  non-local. Absence of `.backward()` is not evidence of locality.
- **Verify-first + honest framing:** probe → act → prove (A/B / gate) → commit; report the true number
  including a regression; reject theater with evidence; never fake a capability, a metric, or a consumer.
- **Runtime evidence outranks static reading.** Every decisive finding in the 2026-07 audits came from
  *running* the system, and the static reads were wrong in both directions. Monkeypatching
  `torch.Tensor.backward` over 80 live ticks proved no backprop on the tick (0 hits) where grep could
  only have found call sites. A neuter-A/B on `pc_surprise` proved it decisive (59/60 ticks changed
  `reentrant_strength`). A matched **control** proved predictive-coding learning real — error −67.8%
  with the update, +9.1% *without* it. Conversely a `cProfile` ranking sent a whole optimisation plan
  after the wrong function: 244,768 calls to `lif_neuron.step` looked like the hot path, and neutering
  it entirely moved the tick 43.6 → 41.2 ms (a 6% ceiling — per-call profiler overhead attributed as
  time). **Before optimising a hotspot, neuter it and measure the delta; before believing a signal is
  live, neuter it and watch what changes.**
- **A performance regression can be correctness-green.** v4.64 put a ~1 s static-import-graph census
  inside `snapshot()`, which the cockpit polls at 1 Hz — so the dashboard made the daemon re-scan its
  own source every second, starving the tick loop it existed to display. Every suite stayed green
  throughout, because self-checks call `snapshot()` a handful of times where a second is invisible; the
  cost only bites at rate against a live loop. It was found by watching the running system. **Anything
  that observes the brain must be cheap enough not to become the load** — and no gate here can tell you
  that.
- **Adding a policy: implement the gate before adopting the rule.** A proposed mechanization must be
  built and run against this tree, and its output counted, *before* the policy lands. Two proposals died
  exactly there — 0 failures / 47 UNCERTAIN, and 58 failures where 1 was claimed. Also required: quote
  the source line the rule derives from (a general good-practice rule in a costume is a reject), and
  name the real past incident it would have caught, or state plainly that there is none and say what it
  earns its place on instead. Fewer, sharper policies win; "no new policy" is a complete result.
- **Gated + reversible:** one change per commit, full gate green (`run_selfchecks` + `pytest` + `ruff`),
  commit + tag locally; a regression is isolated to its own change.
