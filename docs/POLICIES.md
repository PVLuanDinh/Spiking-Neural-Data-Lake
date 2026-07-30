# Development Policies (P0–P6)

**P0 is the purpose; P1–P6 are how it is pursued without lying to ourselves.**
Seven standing policies govern this brain. They are **mechanized** where possible (a CI
self-check that fails on violation) and otherwise enforced by review. Each has an **honesty guard** —
the rule that stops the policy from being satisfied by theater.

**Enforcement status, stated plainly, because it is easy to assume a gate exists:**

| | mechanized in CI | enforced by review only |
|---|---|---|
| **P0 autonomous self-improvement** | — | yes (its guards are P2/P3/verify-first) |
| P1 no-hardcode | — | yes |
| P2 adopt-or-merge | — | yes |
| P3 no dormancy | `substrate/dormancy_gate.py` (per-MODULE), `substrate/method_dormancy_gate.py` (per-METHOD), `substrate/import_manifest.py` (cross-tier) | the charge tier |
| P4 one brain | `substrate/replication_guard.py` | — |
| P5 use the GPU where it helps | — | yes |
| **P6 mutual reach + use** | — | yes (provisional; its gate is specified but NOT built, see P6) |
| e-prop not backprop | — | yes |
| *(not a policy — repo hygiene)* citations resolve | `tools/citation_guard.py` (step in the `lint` CI job, asserting since 2026-07-30) | — |

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

## P0 — The brain must autonomously improve itself, by every available avenue

**This is the project's purpose, and it outranks the rest.** P1–P6 describe how to build well; P0
describes what is being built. The brain must not be a fixed artifact that a human upgrades between
releases — it must get better on its own, continuously, through every channel it can reach: **writing
its own code, acquiring knowledge, and growing its own neural substrate.** Where an avenue of
self-improvement is missing, that is a gap to close, not a boundary to respect.

**The avenues, and their honest status today.** "By every available avenue" is only meaningful if we
say which are open:

| avenue | status |
|---|---|
| **self-building code** | **What fires:** `brain_native_selfbuild.py` (1088) at `cortex/brain.py:1708` and `self_build_loop.py` at `:2520`, every 8 nights, **only** under `NORD_SELFMOD=1` (enabled in `learn_daemon.bat:14`); `self_mod_harness.py` (648) is reached through `Brain.self_improve` at `:3485`. **What does NOT fire, corrected 2026-07-29:** `godel_machine.py` — its `attempt_on_repo` (`:159`) has **zero call sites** tree-wide, and the only reference to the module anywhere outside itself is its *filename as a string* in the self-build's protected-paths list (`brain_native_selfbuild.py:73`). This row previously listed it among the modules that fire every 8 nights. Wiring it is open P0 work; until then it is an unwired real-repo adapter, and saying otherwise was the census-counted-as-capability failure P0's own honesty guard names |
| **knowledge acquisition** | live — chat ingest → the one durable lake (`substrate/chat_memory.py`), `substrate/web_ingest.py`, reasoning-curriculum growth during sleep |
| **neural growth** | **Split into "grows" and "keeps what it grew", corrected 2026-07-30, because this row said "live" and meant only the first.** Growth runs **only during sleep** (asserted: `cortex/brain.py:3780`). What is KEPT differs per module and had never been checked: **`cortex/neurogenesis_engine.py` retains NOTHING** — its `NGSNN` is built at `:171`, fit at `:174`, scored at `:176`, has only `n_units`/`acc` read off it at `:177-178`, and is **never assigned to `self`** (verified: zero matches for `self.*model*=`), so it is collected at the end of **every refit** — not merely on restart. Its 8..120 `grown_units` range across nights 41–8004 is the size of the most recent throwaway fit, not an accumulating population. **`substrate/similarity_sized_growth.py:78` DOES retain** (`self.W`, an `(n_classes, dim)` array). `hippocampus/{growing_learner,function_preserving_growth}.py` **UNVERIFIED** — a narrow attribute scan found no retained weight array, which is not the same as proving there is none. And **nothing here is persisted across a restart**: the engine has no `save`/`load` at all, and the daemon relaunches on every new commit (~20 times in a working session; the night counter resetting `364 → 41 → 81 → 122` is that happening) |
| **cortical column construction** | **NOT live.** `cortex/cortical_microcircuit.py` is a faithful 8-population Potjans-Diesmann circuit that is EXEMPT-permanent in `substrate/dormancy_gate.py:83` — *it has no input port*. This is the clearest open avenue named in this policy, and closing it is P0 work |
| **self-recalibration** | live — `substrate/signal_calibrator.py`, running-baseline forms per P1 |

**Honesty guard — growth is not improvement, and this is where P0 would otherwise rot.** Adding
neurons, adding code, adding knowledge records: none of these is improvement until something
*measurably gets better*. A self-improving system that reports its own progress is the single easiest
place in this project to generate theater, and every other policy exists partly to stop that:

- a grown unit that changes no outcome is **dormant** (P3), not progress;
- **a grown unit that is not KEPT is not even growth** — added 2026-07-30, because the avenue table
  above claimed neural growth was "live" while its main engine discarded every model it fit. This guard
  had a gap the whole time: it tested whether growth *changes an outcome*, and never whether growth
  *accumulates*. A measurement loop that reports rising accuracy as units grow is a legitimate and
  useful thing, but it is an experiment about growth, not the growth itself. Note the honest
  possibility, which is open work rather than a defect: retaining nothing may be CORRECT for a
  measurement loop — in which case what needs fixing is the claim, not the engine, and this row is that
  fix. Deciding whether the engine *should* keep its model is a design question about what growing
  means here, and `hippocampus/ng_snn.py`'s earn-your-place stop (capacity monotone in data, 4,512
  units at 24,000 rows) is the template for a grower whose result is worth keeping;
- a self-written module that duplicates a mechanism is **bloat** (P2), not capability;
- a "capability" reported without a measurement is a **fake capability** (verify-first), not a gain;
- the census of what the brain has is **not** a measure of what it can do — count improvements by
  their effect, never by their number.

Real example of the failure mode: predictive-coding *inference* ran on every tick for the life of the
feature while its *learning* method was never called (v4.71). The brain had the machinery, reported the
signal, and improved at nothing.

**Safety is what makes aggressive self-improvement possible, not what limits it.** The self-modification
path is sandbox-verified, arm-safety firewalled, and rolled back byte-exact on any gate failure —
which is precisely why it can be left ON. A self-improving system that cannot be rolled back is one
that gets switched off after its first bad night, and then it improves at zero. The safety envelope
(five brainstem files, E-STOP never relaxed) is **not** an avenue of self-improvement and is never a
target of it: the brain may rewrite what it thinks, never what stops it. That boundary is what buys
the freedom in the rest of this policy.

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

**Fourth case, stated because the tree contains it and the three above do not cover it: raw physical
limits inside the safety envelope stay fixed, by design.** `brainstem/arm_config.py:22`
(`CONTACT = 0.04`) and `brainstem/reflex.py:22` (`DEFAULT_RULES`) arbitrate behaviour and are constants
— which the three exceptions would flag. They are correct anyway: a reflex threshold that drifts with a
running baseline is a reflex that can be trained away, and the whole point of the envelope is that it
cannot. The resolution currently lives only in code (`arm_config.py:29` declares these sim placeholders;
`serial_arm.py:46-52` refuses real hardware on every construction while `SAFETY_CALIBRATED = False`),
so it is written here too. Safety constants are exempt from P1 *because* they are safety constants, not
because nobody noticed them.

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

**OPEN P2 DEBT, measured 2026-07-28: two workspaces run additively.** `thalamus/global_workspace.py`
and `thalamus/reentrant_workspace.py` are both live and both fed every tick. Measured over 40 wake
ticks on a real brain: **72 submits each**, with 3 content values reaching both. Their public surfaces
are near-identical — `submit(source, content, salience)`, `subscribe`, `current`, `step` on both — and
`ReentrantWorkspace` adds `broadcast_to`, `association`, `access_strength`, `inbox`. On interface
alone it is a superset. That is exactly the shape this policy says must resolve to one survivor.

It is **not** resolved here, for the reason this policy already gives: *"better" requires a measurement
that exists*, and SNDL still has no consensus benchmark — no ambiguous multi-source evidence with a
known right answer — so neither can be shown better. Retiring `GlobalWorkspace` on interface elegance
alone would be the fashion-treadmill move this guard exists to stop, and it has **8 importers**
(`access_consciousness`, `executive_control`, `consciousness_probe`, `concept_reflection`,
`autonomous_loop`, …) against ReentrantWorkspace's 3.

What DID get fixed is a defect that had to be cleared before any comparison could be fair: the
`AccessConsciousness` wrapper was **constructed inside the wake tick**, so a fresh `GlobalWorkspace`
was built every tick — measured **25 constructions over 40 ticks** against ReentrantWorkspace's **1**.
Every instance took exactly one step, making incumbent defence, hysteresis and decay structurally
unreachable, while a comment eighteen lines below promised "ignited content is HELD across ticks, not
rebuilt from scratch each tick". Now persisted: **25 → 2** constructions, `access_index` still varying
across 25 distinct values. Any future A/B between the two workspaces would previously have compared a
persistent mechanism against one that was destroyed before it could express itself.

**The benchmark that would settle it**, stated so the debt is actionable rather than perpetual: a set
of ambiguous multi-source inputs with a known correct winner, scored on which workspace's ignited
content matches it. Until that exists, both stay and this entry is the honest record of why.

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

**Also mechanized: the CROSS-TIER edge, `substrate/import_manifest.py`.** `BRAIN_MODULES` is built from
REGIONS + TORCH_TIER, so `reasoning/` (172 modules) and `training/` are outside the gate entirely — and
they are also ruff-excluded and omitted from the coverage denominator. The governed tiers nonetheless
reach into them through **lazy imports inside bare `except Exception:` swallows**. **Re-counted
2026-07-29, with the definition stated, because the old figure was unreproducible and so was its
correction:** `cortex/brain.py` has **111** bare `except Exception:` handlers (no `as name`), **39** of
which wrap a `try` body containing an import; across the 245 governed-tier files it is **142** bare and
**49** import-wrapping. This text said "67 such swallows in `cortex/brain.py` alone" and the audit that
flagged it said "112 / 203" — neither states what it counted, which is why neither reproduces. The
counter is `ast`-based and stated here so the next reader can re-derive it rather than inherit it. Delete or break one of those targets and every gate stays green while the
brain sets the faculty to `None` and ticks on without it — including `self_build_loop` and
`self_mod_harness`, which *are* P0's self-building-code avenue, and `rose_language`, which is language.
The canary derives the manifest by AST walk (never hand-maintained — the `_external_roots()` lesson) and
asserts each target still imports. Current output: **15 ungoverned modules load-bearing for the governed
tiers across 17 call sites.** It proves import-ability, not charge; that is deliberate, because
import-ability is precisely what a deletion breaks and what the swallows hide.

**Honesty guard (the crux):** reachability proves a module is *imported*, NOT that its output changes
behaviour. "Utilized **appropriately**" (does the wire actually *charge*) is a second, per-faculty tier,
proven by the runtime **neuter-A/B probe** (`tools/charge_probe.py` → "BEHAVIOURAL CHARGE OK"
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
| 2 | **return value discarded** — a live call whose result goes nowhere | `connectome_router.step()`; its only export reached telemetry alone, measured constant 0.250 over 122 ticks. **SINCE FIXED** (`51bf5253`, v4.80): the return is consumed at `cortex/brain.py:2010`. The old cite `:2001` had also drifted onto an unrelated line — corrected 2026-07-29, because this table is the reference readers copy from |
| 3 | **key-space mismatch** — a learned path structurally unreachable | `thalamus/reentrant_workspace.py` potentiates on `(source, winner)` at `:271`, looks up `(subscriber, winner)` at `:289`; disjoint live, so the learned re-route took 2 values in 122 ticks |
| 4 | **write-only consumer** — a faculty fed every tick, never read | `cortex/agency_monitor.py` — `observe()` runs live; nothing reads `agency()` / `controllability()` / `status()`, and no key reaches telemetry |
| 5 | **constant wearing a signal's clothes** — present, plausible, never varies | `spikes.midbrain` (its novelty term is never enabled); `integration_competence` (assigned only in `Brain.__init__`); the `gated` trio (never *computed*, not merely never applied) |
| 6 | **silent default** — a lookup table missing the cases that matter | `_ALLO_MOVE` had no entry for the four `JOINT_*_ROTATE` tokens, so `.get(action, (0.0, 0.0))` meant the commands that actually move the arm contributed **zero** to path integration. Fixed v4.72 |
| 7 | **the registry nobody registered with** — a plug point, called every cycle, with zero plugs | `substrate/region_wiring.py` is *the* documented per-cycle faculty-charge registry, and its docstring asserted that "the daemon imports `wired_faculties` (which registers every adapter) at boot". **There is no `wired_faculties` module**, `learn_daemon.py:291` imports only `region_wiring`, and nothing anywhere calls `wire()` outside the self-check — which clears `_WIRED` first, so it proves the mechanism and says nothing about production. `charge_all()` has iterated an empty list every cycle for the life of the feature and the daemon logs `adapters=0`. Docstring corrected; the registry is still empty, and filling it is open P3 work |

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

**Mechanization: BUILT — `substrate/method_dormancy_gate.py`, in `run_selfchecks` CORE.** A pure
static AST pass (~1 s, no runtime cost). Current output on this tree: **56 update rules scanned,
0 dormant, 16 UNCERTAIN, 5 `EXEMPT_METHOD` entries** (one of them an open `wire` reported as a
standing TODO). That count is what makes it adoptable — an earlier draft of this same gate produced
59 failures, which is the rubber-stamp shape that killed a different proposal above.

**Three details are load-bearing, each found by implementing the naive version first and watching it
fail.** The first two were predicted when this section was written; the third and the correction to
the second were found by building it:

1. **Import-scoped resolution.** Matching bare `ast.Attribute` call names across all reachable modules
   scores 0 failures / 47 UNCERTAIN — the name `learn` resolves in 20 unrelated modules, so it cannot
   even catch its own motivating case. A call site counts only if its module imports the defining
   module.
2. **Same-module callers COUNT, except under `if __name__ == "__main__"`.** Excluding the defining
   module produces a real false positive (`plasticity/meta_plasticity.py:261` legitimately drives
   `_Unit.learn` at `:131`). The `__main__` guard is precisely what isolates the genuine case.
   **Correction found by building it:** that guard block is almost always a single line — `main()`.
   The self-check's real body lives in a top-level `def main()`, so marking only the literal `if`
   counts every assertion inside `main()` as a live caller. The first run scored 0 dormant *while its
   own motivating case sat there dead*. The functions the guard calls must be marked too.
3. **A function used as a VALUE is used.** `reasoning/arc_mint.py:224-228` registers its fitters in a
   dispatch table — `("affine", 3, _fit_affine)` — so a pass inspecting only `ast.Call` reported all
   five as dead: 5 false positives out of 12. Bare `Name`/`Attribute` references count as use.

**Scope, and it is the difference between adoptable and useless.** The gate audits a learning-verb
**method** (inside a class) on a module the **module gate already proves reachable**. Widening either
half re-introduces the 59-failure list — mostly module-level helpers inside `reasoning/` and
`training/` files that the module gate deliberately scopes out.

**Adjudication of the candidate set, now done.** The five-candidate list this section used to carry
was stale, as it warned. Re-run under the built gate: `cortex/predictive_coding.py` wired and
charge-proven (v4.71); `plasticity/meta_plasticity.py:131` and `neuromodulation/drive_reduction.py:196`
confirmed **non-findings** — both have real non-`__main__` callers and reach the live path through the
birth-time `census()` chain (`cortex/brain.py:379-385`), so they belong to the charge tier, not the
call-site tier. `neuromodulation/empowerment.py:86` `reward()` is genuinely dead — its only call site
is inside its own `main()`, and the live path calls the sibling `step_reward()` (`:91`), which
recomputes rather than delegating; it is a **retire** candidate, and it is out of this gate's scope
because it computes a signal rather than applying an update. `hippocampus/affective_memory.py:173`
`consolidate()` is **confirmed dormant by the gate itself** and held as an open `wire` exemption.
The gate also found one the list never had: `neuromodulation/drive_reduction.py:238 learned_map()`,
a readout the verb match catches on the word "learn" — exempted as a named false positive rather than
narrowed away, because narrowing to dodge it would also hide accessors that genuinely should be read.

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

**CROSSOVER RE-MEASURED 2026-07-28, and it MOVED — this policy asked for exactly that ("re-measure
when populations grow; the crossover is a number, not a doctrine").** The range recorded below was
optimistic. Measured again on the RTX 5070, torch 2.11.0+cu128, 300 iterations per point:

| N | numpy | torch-CUDA | winner |
|---:|---:|---:|---|
| 240 | 0.0022 ms | 0.0627 ms | numpy, 34x |
| 4,096 | 0.0028 ms | 0.0139 ms | numpy, 5x |
| **24,576** | 0.0069 ms | 0.0136 ms | **numpy, 2x** |
| **32,768** | 0.0075 ms | 0.0133 ms | **numpy, 1.8x** |
| 262,144 | 0.5499 ms | 0.0133 ms | CUDA, 41x |

numpy still wins at BOTH ends of the recorded 24,576-32,768 band, so the true crossover is somewhere
between 32,768 and 262,144. The run was under machine load, which penalises numpy more than CUDA, so
this errs in the conservative direction. The practical consequence is unchanged and now has more
headroom: live populations of 24-240 are nowhere near it.

**CUDA GRAPHS TESTED 2026-07-28 — they move the crossover 6-8x, and the conclusion still holds.**
The measurements above are EAGER torch, where each tiny op pays a separate kernel launch. CUDA graphs
capture a whole op sequence and replay it with a single `cudaGraphLaunch`, which is aimed exactly at
the CPU-dispatch overhead that made the width-38 test read 0.01x. Measured on the RTX 5070, 200
sequential LIF-shaped ops (`v = v*decay + in; s = v >= theta; v -= theta*s`), best-of-20 after warmup:

| N | numpy | CUDA graph | winner |
|---:|---:|---:|---|
| 24 | 0.578 ms | 1.118 ms | numpy 0.52x |
| 240 | 0.613 ms | 1.127 ms | numpy 0.54x |
| 1,024 | 0.788 ms | 1.129 ms | numpy 0.70x |
| 3,072 | 1.088 ms | 1.224 ms | numpy 0.89x |
| **4,096** | 1.233 ms | 1.225 ms | **graph 1.01x** |
| 24,576 | 4.010 ms | 1.223 ms | graph 3.28x |
| 65,536 | 8.975 ms | 1.425 ms | graph 6.30x |

Two results, both worth keeping. **Graphs beat EAGER CUDA everywhere** (2.2 ms vs 7.4-26.6 ms on the
same sequence) — the launch-overhead hypothesis is confirmed, and any future GPU path here should
capture rather than dispatch. And **the crossover drops from ~24,576-32,768 to ~4,096**, because
replay cost is flat in N while numpy's is linear.

But live populations are **24-240**, still ~17x below even the improved crossover, so **the
conclusion is unchanged: the live tick does not belong on the GPU.** This is the strongest technique
available tested against the real shape, and it loses by 2x at our widths. Recording it under guard
(b): "we tried CUDA graphs here and it was 0.54x" is a result, and it should stop this experiment
being repeated a third time. The number that would change the answer is population WIDTH — at 4,096
the GPU starts paying, so a genuinely wide population is the thing to look for, not a faster kernel.

**Update (2026-07-28): that width now exists, on the growth tier.** With the NG-SNN earn-your-place
stop corrected (it had been scoring rows the solve already fit — see
[PERFORMANCE.md](PERFORMANCE.md)), grown capacity is monotone in data and reaches **4,512 units at
24,000 rows**, past the 4,096 crossover, on plain MNIST at 8x8. The earlier reading — that capacity
saturates near 2,000-2,600 and that GPU-relevant width would need a harder problem — was an artifact
of the broken criterion, not a property of the task.

This does **not** move the live tick: those populations are still 24-240 and the conclusion above
stands unchanged. What it changes is the `[train]`/`[gpu]` tier, which now belongs on the card by
WIDTH and not merely by fit cost. It also relocates the thing worth re-measuring as the brain grows:
this capacity curve, not the kernel. And it is a second instance of the guard-(b) lesson — the first
saturation claim was refuted by fixing the instrument, exactly as the first two "the live tick should
be on CUDA" claims were refuted by measuring it.

**The live-tick CUDA perception path is measured at last, and it does not pay.** `cortex/brain.py:170`
defaults `device="cuda"` and `_perceive_real()` runs a per-frame batch-of-1 forward pass every wake
tick -- a GPU path that shipped without the benchmark honesty guard (a) demands. On the real live shape
(1 frame x 15 timesteps x 2x34x34 -> 128 -> 64): **torch-CPU 0.073 ms, torch-CUDA 0.073 ms, 1.00x.**
Not a regression like the 0.01x at width 38, but not a gain either -- it holds VRAM and adds a
host-device transfer dependency for nothing. The default is LEFT AS IS pending a bit-exactness check,
because guard (d) cuts both ways: moving a live numeric path OFF CUDA carries the same burden as moving
one onto it. Flipping it is a follow-up with a controlled A/B, not a free win.

**The original measurement, still valid.** For SNDL's elementwise
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
- **And it happened again, larger, 2026-07-29 — with the widening plan refuted on the way.** This
  policy's standing invitation ("the moment a live population crosses the measured threshold, put it on
  the GPU") was taken up: the plan was to widen the live populations past the ~4,096 CUDA-graph
  crossover and move the tick. A runtime width inventory killed it. The whole live brain holds **132**
  per-object LIF cells, and *all* live per-tick neuron state fused into one flat vector is
  **~1,700–3,500** elements — inside the crossover band, not past it. `py-spy` on the live daemon then
  showed why widening would have optimised nothing: **`AutonomousArm.step()` is 0.00% of the profile**
  (`_c1_tick` 0.10%, `_tick_ensemble` 0.01%), while **74.5%** of samples sit in
  `cortex/vsa_reasoning.py::coincidence` — a Python loop run once per stored episode per tick, entered
  via `hippocampus/episodic_recall.py`. Fusing that as one GEMV, using the exact integer identity
  `coincidence(a,b) == (D + a·b)/2` for ±1 spike codes, measured **`complete()` 59.946 → 0.909 ms at
  N=2,000 (66×)**, bit-exact and tie-identical. The load-bearing part is not the factor: the scalar
  cost was **linear in the episode count** (fitted `ms/tick = 56.60 + 0.0804·N_episodes`) and the fused
  cost is **flat** (0.61–0.91 ms across N=60–2,000), so the growth term is gone rather than reduced.
  Two further results worth keeping so they are not re-litigated: for this GEMV the measured CPU→GPU
  crossover is **~1,600–2,000 rows**, so it ships on the **CPU** and a captured CUDA graph is a *loss*
  below that; and for fused LIF at our real widths **numba beats a CUDA graph by ~16×**. `spikingjelly`
  and `cupy` were rejected on a structural reason rather than a benchmark — their fused multi-step
  kernels collapse the **time** axis `T`, and the live tick has `T=1` by construction, so the axis
  those kernels exist to collapse is absent. Net new dependencies for all of the above: **none**.

**Where the GPU therefore pays today, and should be used without hesitation:** the `[train]` / `[gpu]`
tiers — surrogate-gradient training, `hippocampus/ng_snn.py`, `cortex/neurogenesis_engine.py`, the
torch readout studies under `reasoning/brain_adapters/`, and any offline sweep over a real dataset.
These are wide, batched, and repeated, which is exactly the regime the crossover favours. The daemon
already sets `NORD_GPU=1` and runs under `.venv-cu128` so this tier lands on the card by default.

**Where it does not pay today:** the live wake/sleep tick, at current population widths. That is a
statement about widths, not a prohibition — **the moment a live population crosses the measured
threshold, this policy says put it on the GPU.** Re-measure when populations grow; the crossover is a
number, not a doctrine.

**The two gate environments, measured against each other 2026-07-29 — and why neither is "retired".**
The instruction was to consolidate onto one environment and drop the CPU gate's veto. Measured, that
turns out to be half right, and the half that is wrong matters: **CI has no GPU.** Every job runs on a
GitHub runner, so a gate that requires this card cannot be the gate. What guard (c) below was really
protecting is not the CPU *environment*, it is the property that the suite is runnable without one.

So both are now verified equivalent rather than one being privileged. On a clean checkout at
`598b07dc`, `.venv-cu128` (torch 2.11.0+cu128, numpy 1.26.4): **278/278** self-checks, **473 passed**,
coverage **90.80%**. `.venv-lake` (torch 2.13.0+cpu, numpy 2.4.6): **277/277**, **473 passed**,
**91.11%**. The 278-vs-277 is nengo being installed in the CUDA env, so `NENGO_GATED` runs instead of
skipping. The coverage gap is the worktree having no `models/` (gitignored), so perception falls to the
stdlib tier and less code executes — not a behavioural difference.

**Two environment defects had to be fixed before that comparison was even possible, and both are worth
knowing because neither is visible from reading `pyproject`:**

1. **`nengo` ships a pytest plugin whose hook signature is invalid under pytest 9.1.1** —
   `pytest_report_collectionfinish(config, startdir, items)`. Autoload aborts collection with
   `PluginValidationError` before a single test runs, in any environment where nengo is installed.
   It presents as a total test failure with nothing to do with this code. `-p no:nengo` is now in
   `addopts`; it is inert where nengo is absent. **This is the actual reason the CPU env had been the
   only place pytest could start** — that was never a policy preference, it was a broken plugin.
2. **`ruff` was FLOORED at `>=0.15.20`, so installing the dev extra elsewhere resolved to 0.16.0** and
   created a third formatting authority disagreeing with both `.pre-commit-config.yaml`'s pinned rev
   and the 422 files formatted in-tree — arriving as a mysterious `ruff format --check` failure. Now
   pinned. A formatter's version is part of the format.

**Honesty guard.** (a) A GPU path ships with its benchmark, on the real workload, against the CPU path
it replaces — a speedup asserted from `cuda.is_available()` is not a speedup. (b) Report regressions:
"we tried CUDA here and it was 0.01×" is a result worth keeping, and this policy exists partly to stop
that experiment being repeated. (c) **The suite must stay runnable without a GPU** — restated 2026-07-29 from "the deterministic CPU
gate stays authoritative", which named an *environment* when the property that matters is
GPU-independence. CI has no card, so this is a hard constraint, not a preference; a GPU path must never
become the only way to verify the project. Both envs are now verified equivalent (above), so either may
be used locally. Determinism itself comes from seeding — `random.Random(seed)` per instance,
`zlib.crc32` never builtin `hash()` — not from the absence of hardware. (d) Correctness first: a GPU rewrite carries
the same bit-exactness burden as any other P2 swap.

---

## P6 — Every build must be able to reach every other, and actually use it

**Owner policy, 2026-07-29.** A faculty must not be an island. Every build in this brain must be able to
**communicate with** the others and **utilize** them, the way regions of a real brain are mutually
reachable and mutually consumed — not a shelf of correct components that never speak.

**Stated as a checkable property, because "be like a real brain" is a research direction and this file
rejects those** (see the corollary under P2): a faculty satisfies P6 when (a) it is **reachable** from at
least one other faculty through a typed, declared route rather than only from an entrypoint or a chat
handler, and (b) at least one other faculty **consumes its output** in a way that changes that consumer's
own output. One-way is allowed — brains are full of one-way projections — but *zero*-way is not.

**How this is NOT P3, since the two are easy to confuse and the distinction is the whole point.** P3 asks
whether a faculty changes **behaviour** — the arm, a value, a weight, a reply. P6 asks whether a faculty
changes **another faculty**. They come apart in both directions, and both directions are real here:

- **P3-clean, P6-dead:** a faculty wired straight into `_route_percept_to_action` charges the arm decision
  and talks to nothing. Most of this brain's wiring waves have this shape. The rubric's own pre-declared
  ceiling says so — wiring into that one site "cannot raise Language, Imagination, Gf, Consciousness or
  WholeBrain" — and a brain whose faculties all report to one action gate and never to each other is
  precisely why that ceiling exists.
- **P6-clean, P3-dead:** `MetacognitiveMonitor` feeds and is fed while its `hold()` output reaches only a
  `respond()` branch the Gc branches preempt — reachable, conversational, and behaviourally inert. Its
  `explore()` looks the same on a static read (`brain.py:1859` assigns `_metacog_explore`; the only other
  appearances are the `__init__` default and a telemetry key) — but that is **UNCERTAIN and deliberately
  not asserted**: this file records four dormancy claims of exactly that shape REFUTED on runtime A/B,
  and a policy example resting on an unconfirmed read would be the failure P3's honesty guard names.
  The `hold()` half is measured; the `explore()` half awaits a neuter-A/B.

**Source lines it derives from, quoted so it is not a good-practice rule in a costume.** P3's own
transferable rule: *"presence is not participation."* And `substrate/region_wiring.py`, whose docstring
asserted that "the daemon imports `wired_faculties` (which registers every adapter) at boot" — **there is
no `wired_faculties` module**, `charge_all()` has iterated an empty list for the life of the feature, and
the daemon logs `adapters=0`. That registry is the plug point P6 names, and it has never had a plug.

**Real past incidents it would have caught** — required by the bar below, and there is no shortage. Each
row was **re-verified against the tree on 2026-07-29**, because several were fixed or reverted in the same
session that wrote this policy, and a rule citing an already-closed defect as live is the drift this file
keeps suffering. Rows marked FIXED are kept deliberately: the bar asks what the policy *would have caught*,
and a closed incident is the better evidence precisely because its cause is known.

| incident | shape |
|---|---|
| `region_wiring` registry: 0 registrations, `adapters=0` live | the plug point nobody plugged into |
| `thalamus/connectome_router.step()` return discarded, export a constant 0.250 — **since FIXED** (`51bf5253`, v4.80); the return is now consumed at `brain.py:2010` | computed, reached no faculty |
| `cortex/agency_monitor.py` `observe()` every tick; `agency()`/`controllability()`/`status()` zero readers | write-only organ |
| `claustrum/claustrum.py` `broadcast()` (defined `:118`) zero live call sites — its only caller is `:193`, inside its own `__main__` | a broadcast hub that broadcasts to nobody |
| `reasoning/pragmatics.py` reachable only from `spike_chat_lake`, no `brain.py` caller | wired to chat, not to the brain |
| `reasoning/arc_solver.py` — "imports stdlib only, with zero wiring to any perceptual machinery" | Gf isolated from Perception |
| `reasoning/nightly_skills.py` forms and persists a skill every night; `route()` called only from its own `main()` | the brain never uses a skill it formed |
| `cortex/neurogenesis_engine.py` has no persistence path, so grown units die on every restart | a faculty that cannot even reach its own future self |

The last one is the sharpest: the brain has **one durable spiking lake** and the grown substrate is not in
it, so P0's neural-growth avenue resets ~20 times in a working session and never compounds.

**Honesty guard — a route is not a conversation.** The failure mode P6 will attract is a telemetry bus:
wire every faculty into a shared dict, watch `adapters` climb, and declare harmony. `region_wiring`'s own
source already warns that registering telemetry adapters would fake liveness. So a P6 claim needs the same
evidence P3 demands, aimed one level over: neuter faculty A and show faculty B's **own output** changes.
Adding a reader that stores a value is not consumption. And **do not force-wire** — a route that carries
nothing is reverted and the faculty is exempted with a reason, exactly as under P3. Two faculties were
reverted on that rule this session (PFC working memory, crossmodal binding), both at 0/200.

**Mechanization: NOT BUILT, and this file's own bar says that matters.** *"Adding a policy: implement the
gate before adopting the rule… its output counted, before the policy lands."* Two prior proposals died
there. This policy lands **review-enforced and provisional** on the owner's instruction, with the debt
stated rather than hidden: P6 is not mechanized, and until it is, it is enforced the way P0, P1, P2 and P5
are — by review.

**The gate that would settle it, specified so the debt is actionable.** `connectome.py` already declares
the region→region wiring as typed, directed, signed loops; `substrate/import_manifest.py` already
AST-derives a cross-tier edge set; `substrate/dormancy_gate.py` already builds the static import graph.
So the instrument is: derive the actual faculty→faculty consumption graph, diff it against
`connectome.py`'s declared edges, and report faculties whose in-degree **and** out-degree are both zero
outside an entrypoint. **Build it and count its output before promoting P6 to mechanized** — and if it
reports a rubber-stamp-sized list, that result kills the gate, not the tree. "No gate" is a complete
result; the rule stands on the eight incidents above either way.

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
  `cerebellum/jepa_world_model.py:277` performs explicit weight transport on every wake tick (the same
  `Wz` used forward and transposed backward, in hand-written numpy) and `hippocampus/ng_snn.py:137`
  runs a closed-form batch solve over its whole buffer each sleep tick. Neither uses autograd; both are
  non-local. Absence of `.backward()` is not evidence of locality.

  **Verdict on both, since "review-enforced" had so far meant "unadjudicated" (2026-07-28):**

  - `cerebellum/jepa_world_model.py:275-278` is a **real violation of the locality principle, on the
    live wake tick.** `s += gp[i] * self.Wz[i][k]` reads the forward weight matrix transposed to carry
    error backward — weight transport is precisely the biological implausibility e-prop exists to
    avoid, and hand-writing it in numpy rather than calling autograd changes nothing about that. The
    e-prop-compatible fix is **feedback alignment** (Lillicrap 2016): replace the transpose with a
    *fixed random* feedback matrix `Bz`, which is local, needs no weight symmetry, and is known to
    work on problems this shape. It is cheap and it is measurable — prediction error before/after on
    the real tick, plus cosine against the transport gradient — so it is P2-eligible in a way most
    swaps here are not. **Open work, with its benchmark already specified.**
  - `hippocampus/ng_snn.py:137` `fit()` is **not** a violation. Greedy matching pursuit with an
    analytical beta per block is a direct solve: no gradient is propagated through a network and no
    weight is transported. It IS non-local, and the policy should say so distinctly rather than
    lumping it in — "non-local" and "backprop" are different claims, and only the second is what this
    rule forbids. Disposition: keep, and stop citing it as a pending violation.
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
  **Amended 2026-07-29, because P6 broke this and leaving the invariant unqualified would have made this
  file contradict itself.** The owner may adopt a policy with its gate UNBUILT. Such a policy lands
  marked **provisional** in the enforcement table, must still meet every other requirement above
  (quoted source line, named real incidents, each re-verified against the tree), and must carry the
  gate's specification so the debt is actionable rather than perpetual. P6 is the only entry that has
  used this, and it says so in its own text. The gate-first rule still binds anyone proposing a
  *mechanization*: the two proposals that died at this bar died on their own output, and that outcome —
  "the gate we built kills the rule" — remains the point of building it first.
- **A third proposal died at this bar, 2026-07-30: "a tracked file may not cite an untracked `.py`
  instrument."** It had the best motivation of any proposal here — three real incidents, two of them in
  one session: `POLICIES.md` citing a `scratchpad/smoke_drive_charge.py` that never existed, the same
  dead citation in `dormancy_gate.py`, and `DEPTH_CEILINGS.md`'s AR appendix citing nine gitignored
  `scratchpad/ar_lm_*.py` scripts as its evidence *hours after* the first two were corrected. It was
  built anyway, per this rule, and counted: **474 violations across 126 files, 195 in `CHANGELOG.md`
  alone**, from 1,431 citations. Allowlist-shaped, eight times worse than the 58-failure proposal above,
  so it is **not adopted**. The count also explained itself: a changelog citing a since-retired file is
  correct, because the entry describes a tree that existed. The instrument is kept as a report at
  `tools/citation_guard.py`, and the surviving narrower hypothesis is recorded there — the real defect is
  an untracked script presented as a *runnable reproduction path*, not a filename in prose. Building the
  naive version first is what distinguished those two, and it cost one file instead of a policy.
  **The narrow version was then built and counted too, and it SURVIVED at 21 violations across 5 files**
  (`tools/citation_guard.py`, STRICT scope — a `.py` cited inside a shell-fenced block or under a
  Reproduce/Usage heading). 21 was a work list, not an allowlist, so the rule was adoptable in that
  scope. What it found was worse than the incident that motivated it: **`docs/RUNNING.md`, the project's
  how-to-run page, had five commands pointing at files absent from a clone.**
  **FINAL STATE, same day: broad REJECTED at 475 across 127 files (report only, still non-asserting);
  strict ADOPTED and ASSERTING at 0 unexempted violations with exactly 1 exemption.** The 21 were
  cleared, not exempted, and the dispositions differed as predicted:
  - **10 were stale paths.** `research/` became `src/snn_data_lake/reasoning/` in v3.78 (`627b2e2b`), and
    `RUNNING.md` (×5), `JEPA.md` (×3), `PERFORMANCE.md` (×1) plus `HARDWARE_SETUP.md`'s sketch path were
    never updated — the files had *moved*, not gone. Corrected to `python -m snn_data_lake.reasoning.<mod>`.
    One was a genuine retirement: `spike_knowledge_graph.py` was merged into `spike_kg.py` (`17067032`).
  - **8 were promoted into `tools/`** from gitignored `scratchpad/` — real evidence for real published
    numbers, now tracked, with inputs taken as explicit paths and a clean SKIP when a dataset is absent.
    Each was run; `lang_scale.py`, `lang_diag.py` and `lang_probe.py` reproduce their doc tables exactly
    (53.4→53.1 decay ablation, 0/1675 fold collisions, 0.65% over 117,859 contexts, 2,992 params).
  - **1 was deleted, not promoted:** `ar_lm_untrained_control.py` no longer exists on disk, and the
    tracked bench builds the untrained twin anyway. Its command is gone and the doc now says the 49.5% is
    a recorded result, not reproducible from that file.
  - **1 exemption, `docs/HARDWARE_SETUP.md:144`.** The cited file IS tracked, at
    `integrations/robot arm/sensor_reading.py`; the SPACE in the directory name truncates the guard's
    path token, so it cannot resolve. Fixing that in the resolver was rejected because it would equally
    resolve the stale `robot arm/...` form the line carried until it was corrected — the tokenizer fix
    would hide a live defect class, so the false positive is exempted where the reason is readable.
    `EXEMPT` follows `dormancy_gate.EXEMPT`'s pattern and, because line-keyed entries rot, the gate also
    asserts every entry still matches a real violation: an exemption that blesses nothing fails loudly
    instead of reading as coverage. Both asserts were tested by making them fire.
  Registered as a step in `.github/workflows/ci.yml`'s lint job, not in `run_selfchecks.py`: the latter
  only executes `python -m snn_data_lake.<mod>` package modules, so qualifying would mean moving it into
  the package and churning the canonical 277 count quoted in five documents — to gate repo hygiene, which
  is not a brain-faculty self-check. Recorded here because "the broad rule died at 475 and the narrow one
  now holds the line at 0+1" is the result, and the only reason either number exists is that this bar
  demands counting before adopting.
- **Gated + reversible:** one change per commit, full gate green (`run_selfchecks` + `pytest` + `ruff`),
  commit + tag locally; a regression is isolated to its own change.
