# Development Policies (P0–P8)

**P0 is the purpose; P1–P8 are how it is pursued without lying to ourselves.**
Nine standing policies govern this brain. They are **mechanized** where possible (a CI
self-check that fails on violation) and otherwise enforced by review. Each has an **honesty guard** —
the rule that stops the policy from being satisfied by theater.

**WHAT THIS FILE IS, AND IS NOT (owner directive, 2026-08-30).** This file is the project's
CONSTITUTION and BUILD GUIDE: rules, definitions, honesty guards, standing verdicts, and the
commands that re-derive any number a rule depends on. It deliberately carries **no status, no
counts, no censuses, and no correction narratives** — everything that can go stale lives elsewhere,
because a constitution that describes the tree rots the day the tree moves, and this file rotted
that way for weeks before the split. Where a claim about the CURRENT tree is needed, this file
names the instrument that derives it; **re-derive, never quote**. The routing rule for any new
content:

- **A rule, guard, definition, or standing verdict** → here, stated once.
- **A live value** (a count, a census, a gate reading) → no document at all; the named gate
  derives it on demand.
- **A dated measurement, incident, or correction narrative** → `docs/POLICIES_EVIDENCE.md`
  (tracked, public, gated) — relocated byte-exact with a pointer left here.
- **Per-task status** → `specs/*/tasks.md` and `specs/*/evidence/`.
- **Local, disclosure-gated state** → the repo-local `CLAUDE.md` (the one named P4c exception,
  below) or the operator's own memory outside the repo — never this file, never public docs.

The enforcement is the doc-budget ceiling: this file's ceiling was LOWERED to its
post-split size (`gates/doc_budget_gate.py`), so status re-accumulating here reds the build
instead of rotting quietly. The evidence that used to interleave these sections is in
`docs/POLICIES_EVIDENCE.md` under "POLICIES.md constitution split"; git history holds every
compressed clause.

**Enforcement status, stated plainly, because it is easy to assume a gate exists:**

| | mechanized in CI | enforced by review only |
|---|---|---|
| **P0 autonomous self-improvement** | — | yes (its guards are P2/P3/verify-first) |
| P1 no-hardcode | — | yes |
| P2 adopt-or-merge | — | yes |
| P3 no dormancy | `gates/dormancy_gate.py` (per-MODULE), `gates/method_dormancy_gate.py` (per-METHOD), `gates/import_manifest.py` (cross-tier) | the charge tier |
| P4 one brain (+ P4b partial; P4d provisional) | `gates/replication_guard.py`; `gates/lake_rebuild_gate.py` mechanizes P4b's index-is-derived clause (delete → rebuild from a lake scan → logical-dump equality, planted-divergence red + lake-tracking control); `gates/unit_label_gate.py` mechanizes P4d's provenance check (owner decision 16) | P4b's remaining prose (one-lake, restart-is-not-amnesia); P4d in full (provisional; its gate is specified in P4d) |
| P5 use the GPU where it helps (+ P5b provisional) | — | yes (P5b: the six-criterion record in each Rust build's pre-registration) |
| **P6 mutual reach + use** | **BUILT AND ASSERTING** — `tools/p6_gate.py --assert` and `--self-check` (lint CI job; the connectome-diff half is built). Ships as a **RATCHET**: baselines assert the structure never gets WORSE. Re-derive every count from the gate; never edit this row from memory | yes (provisional; a ratchet says the count did not worsen, never that P6 is satisfied) |
| **P7 any dependency that helps** | **BUILT AND ASSERTING** — `tools/dependency_ledger.py --assert` and `--self-check` (lint CI job) | yes (the judgement half; see P7 guard (b)) |
| **P8 invent the method** | **BUILT AND ASSERTING** — `tools/red_reachability.py --assert` (lint CI job); the null-scale amendment (2026-10-23): `gates/null_scale_gate.py` (`--assert` + `--self-check`, lint CI job) | yes (whether a criterion measures the RIGHT thing; whether a STRUCTURAL null's argument holds and the named scaling axis is the right one) |
| e-prop not backprop | **PARTIALLY — TWO gates: `gates/eprop_transport_gate.py`** (CORE) asserts no GOVERNED module reads a weight transposed into a weight update, and reports the ungoverned tier without asserting on it — it is NOT a `.backward()` tripwire, and `mlp_q`'s single-row-indexing spelling evades both its rules, printed on every run; **and `gates/gradient_claim_gate.py`** (scope/census) asserts every governed learning rule is classified as owing a cosine or not, ratcheting MISSING_DECLARATION and UNDETERMINABLE, plus the ALL-scope PHANTOM_DECLARATION bound (`--assert` + `--self-check`, lint CI job) | yes (everything the static rule cannot reach: whether a second matrix is transported or independently maintained is semantic, and the cosine bar itself) |
| *(not a policy — the safety envelope's own gate)* ONE envelope definition, five brainstem files | `gates/safety_envelope.py` (bare = assert, in run_selfchecks; envelope EDITS stay approval-gated) | — |
| *(not a policy — repo hygiene)* standing docs keep their word ceilings | `gates/doc_budget_gate.py` (bare = assert, in run_selfchecks) | — |
| *(not a policy — repo hygiene)* outside data passes license + injection gates before the brain keeps it | `sensorimotor/ingest_guard.py` (bare = assert, in run_selfchecks) | — |
| *(not a policy — repo hygiene)* citations resolve | `tools/citation_guard.py` (step in the `lint` CI job) | — |
| *(not a policy — a spec instrument, invention register row 7)* growth coverage counted before any rule lands. *DATED 2026-09-25 (P4d, owner decision C; the vocabulary a session call under the 2026-09-05 authorization):* all sixteen `REGION_COORDS` names are the LABEL vocabulary of the anatomical map, read from a grown neuron's position; only the ten owner-ratified growable names (2026-08-19) are admitted, and a SCAFFOLD name, neuromodulation, or a position the map cannot tell apart from theirs is refused; growth INTO the reservoir awaits an owner-ratified target | `gates/growth_coverage_gate.py` (`--assert` + `--self-check`, `lint` CI job). The coverage red is REPORTED every run; asserted: the identity-pinned ratchet, scaffold/undeclared refusals, unresolved-site reds. Green never means coverage is adequate | — |
| *(not a policy — repo hygiene)* documented `python -m` commands run | `tools/doc_command_guard.py --assert` — in `run_selfchecks`'s tools stage and the `lint` CI job; `policy_manifest`'s census counts both | — |
| *(not a policy — P0's honesty guard, owner-adopted 2026-09-02, specs/022 OD-5)* a unit COUNT is reported as **admitted**, never as growth, until SC-012a fires ("provisional" = time-bound) | `tests/test_units_admitted_wording.py` (every battery): label bound to the value, PROVENANCE key on the row; red counted pre-change (3 of 3) | — |

**The rule this table has repeatedly broken, kept beside it:** this file is edited far more often
than it is re-derived, and a "NOT BUILT" is the single cheapest sentence in it to leave standing.
**Re-run the gate before writing a sentence about the gate.** The counted outputs that used to sit
here, and the corrections that earned this rule, are relocated to `docs/POLICIES_EVIDENCE.md`.

| gate | output | in CI? |
|---|---|---|
| `tools/red_reachability.py --assert` | re-run it; historical counted output → POLICIES_EVIDENCE.md | yes, `lint`, `--assert`; and `run_selfchecks`'s tools stage |
| `tools/dependency_ledger.py --assert` | re-run it; quote a gate by re-running it, or do not quote it | yes, `lint`, `--assert`; and `run_selfchecks`'s tools stage |
| `tools/dependency_ledger.py --self-check` | re-run it; every criterion proven falsifiable or the run fails | yes, `lint` |
| `tools/doc_command_guard.py --assert` | re-run it; scans tracked docs for `python -m` commands that do not run | `run_selfchecks`'s tools stage; re-derive from `policy_manifest`'s census, which counts that stage |
| `tools/p6_gate.py` | re-run it; a RATCHET — green means "no worse", never "P6 satisfied" | yes; and `run_selfchecks`'s tools stage |
| `tools/p4c_ratchet.py` | re-run it with `--assert`; per-ZONE ratchet, machine-generated baseline that refuses to write a rise | yes, and locally in `run_selfchecks`'s tools stage |

---

## P0 — The brain must autonomously improve itself, by every available avenue

**This is the project's purpose, and it outranks the rest.** P1–P8 describe how to build well; P0
describes what is being built. The brain must not be a fixed artifact that a human upgrades between
releases — it must get better on its own, continuously, through every channel it can reach: **writing
its own code, acquiring knowledge, and growing its own neural substrate.** Where an avenue of
self-improvement is missing, that is a gap to close, not a boundary to respect.

**Owner directive, 2026-08-04 — the avenue list is a BUILD LIST, not a status ledger.** Every
avenue named, **current and future, will be built.** An avenue that does not fire yet is open P0
work; it is never a settled state, never a boundary to respect, and never a reason to narrow this
policy to whatever already works. "Will be built" is a commitment about *work*; it grants no credit
for the capability in advance — the ordering stays exactly as the honesty guard states it: build the
channel, fill it, measure it, and credit it only at the third step.

**The avenues.** Their current firing state is a measurement, not a constitution entry — re-derive
it (the routes and instruments are named per avenue below); the dated status table this section
used to carry is in `docs/POLICIES_EVIDENCE.md`:

| avenue | what it is, and where its state is derived |
|---|---|
| **self-building code** | `engine/brain.py::Brain._maybe_autonomous_self_build` → `infra/self_build_loop.SelfBuildLoop.run`, night-cadenced under `NORD_SELFMOD=1`; pinned by `tests/test_selfbuild_route.py` (route + opt-in gate as its control). Probe with `Brain._selfbuild_last` on a scratch `NORD_HOME` — a run that appends to the live lake is the probe contaminating the subject |
| **knowledge acquisition** | chat ingest → the one durable lake (`substrate/chat_memory.py`), `sensorimotor/web_ingest.py`, reasoning-curriculum growth during sleep |
| **neural growth** | problem-driven growth (P4d): grow column-sized batches on the anatomical map while a problem's solved predicate fails, attempt, repeat, and pause at a budget (UNSOLVED, kept open); sleep prunes and saves after the solve. ONE grower (P2) — the incumbent is sleep-tick growth with retention: `growth/neurogenesis_engine.py` (earn-your-place prune, `growth_state_bytes`/`restore_growth_state`/`substrate_is_retained`), persisted into the ONE lake on two live paths (sleep tick and construction; `Brain._flush_growth` — ONE mechanism, TWO callers), which the problem-driven loop merges with or replaces; ONE stop rule likewise, the incumbent `growth/growth_stop.py` (FR-023). Restart-with-grown-substrate is pinned by `tests/test_brain_growth_restart.py`, whose immediate-exit case carries the reachable red |
| **cortical column construction** | under P4d (owner decision 2026-09-25) a column is the GROWTH STEP, a small layered batch placed on the anatomical map where error builds up; this microcircuit is a candidate source of its layer targets. `reservoir/cortical_microcircuit.py::thalamic_drive` IS the input port (anatomically targeted, L4/L6 only; exactly 0.0 at 0 Hz; refuses a negative rate). Wiring a live caller is the open work; a spatial-PATTERN input (the port carries a scalar rate) is the genuinely absent half |
| **self-recalibration** | `substrate/signal_calibrator.py`, running-baseline forms per P1 |
| **future avenues** | open by construction — a newly identified channel joins this list and is built, on the same terms: named blocker, then channel, then fill, then measurement. Treating the list as complete is the failure this row exists to prevent |

**Honesty guard — growth is not improvement, and this is where P0 would otherwise rot.** Adding
neurons, adding code, adding knowledge records: none of these is improvement until something
*measurably gets better*. A self-improving system that reports its own progress is the single easiest
place in this project to generate theater, and every other policy exists partly to stop that:

- a grown unit that changes no outcome is **dormant** (P3), not progress;
- **Scaffolding a capability the project does not have yet is PERMITTED, and is not a fake capability
  — owner decision, 2026-07-30.** This guard forbids *reporting* a capability without a measurement. It
  does not forbid *building the channel* one will need. The conditions that keep it honest: the
  infrastructure is labelled as infrastructure, no coverage band or P0 avenue may claim the capability
  until it is measured, and the gap it exists to fill is named at the site (as
  `neurogenesis_engine.substrate_is_retained()` names it). Build the channel, then fill it, then
  measure it — and do not credit it until the third step;
- **a grown unit that is not KEPT is not even growth** — added 2026-07-30, because the avenue list
  once claimed neural growth was "live" while its main engine discarded every model it fit. This guard
  had tested whether growth *changes an outcome*, never whether growth *accumulates*. A measurement
  loop that reports rising accuracy as units grow is a legitimate thing, but it is an experiment
  ABOUT growth, not the growth itself. The honest possibility stays open: retaining nothing may be
  CORRECT for a measurement loop — then what needs fixing is the claim, not the engine.
  `growth/ng_snn.py`'s earn-your-place stop (capacity monotone in data) is the template for a
  grower whose result is worth keeping;
- a self-written module that duplicates a mechanism is **bloat** (P2), not capability;
- a "capability" reported without a measurement is a **fake capability** (verify-first), not a gain;
- the census of what the brain has is **not** a measure of what it can do — count improvements by
  their effect, never by their number;
- **and a measurement that cannot fail is not a measurement** — added 2026-08-04 with P8, because
  this guard never said what makes something a measurement. See P8 for the burden and
  `PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:57-58` for the line it derives from.

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
Reuse anchors: `plasticity/r_stdp.py` (`w += lr*d*e`), `substrate/signal_calibrator.py`, the
`_*_base` running-EMA idiom, `engine/brain.py::_drive_w`.

**Fourth case: raw physical limits inside the safety envelope stay fixed, by design.**
`brainstem/arm_config.py`'s `CONTACT` and `brainstem/reflex.py`'s `DEFAULT_RULES` arbitrate behaviour
and are constants — which the three exceptions would flag. They are correct anyway: a reflex threshold
that drifts with a running baseline is a reflex that can be trained away, and the whole point of the
envelope is that it cannot. (`arm_config.SAFETY_CALIBRATED` is `False` and
`serial_arm.assert_calibrated()` raises on every real-servo entry point while it stays False.)
Safety constants are exempt from P1 *because* they are safety constants, not because nobody noticed
them.

The worked case behind the corollary (a hardcode that was already a derivation, and a defence no
value could fix — both in `faculties/global_workspace.py`) is relocated to
`docs/POLICIES_EVIDENCE.md`. Its two transferable tells stay: **a constant that only ever appears
divided is a round trip written down as a number**, and **a constant that resists every value in a
sweep is not a tuning problem — the blocking constraint is doing two jobs.**

## P2 — Adopt-better **or merge**, then retire the inferior (non-additive)

When two mechanisms do the same job, exactly **one** survives. There are two legitimate routes to that,
and they are co-equal — the choice between them is a judgement about the mechanisms, not a fallback:

1. **ADOPT** — one is better on evidence; it becomes THE path and the other is **deleted**.
2. **MERGE** — neither dominates, or each holds something the other needs; build the **one** mechanism
   that supersedes both, then retire both originals. A merge is not a compromise or a coexistence; it
   ends with fewer moving parts than it started with.

What is forbidden is the third option: leaving both live "additively". That dilutes the signal, grows
bloat, and leaves every later reader guessing which one is authoritative.

**Worked merge:** `growth/neuron_zoo.py` — one heterogeneous population behind a single contract,
earning the name because it *"does NOT re-implement any dynamics -- it IMPORTS the existing neuron
modules and puts a thin adapter in front of each"*. One surviving interface, zero duplicated dynamics.

**Honesty guard:** prove the adopt-or-merge is behaviour-equivalent (or superior) **before** deleting —
an A/B on identical inputs, gate-verified no regression, and any unique capability of the retired
mechanism ported first. Reject it with evidence if it regresses (verify-first works when it says *stop*).

**Merge-specific guard — a merge must actually reduce.** "Merging" that keeps both code paths behind a
flag, a mode switch, or an `if` is additive with extra steps. Count the survivors: if two
implementations of the same job are still reachable when you are done, it was not a merge.

**"Better" requires a measurement that exists.** If the two mechanisms cannot be compared by any
benchmark in this repo, the swap is not P2-eligible and the answer is *build the benchmark first*, not
*pick the more fashionable one*. An external framework being newer, more elegant, or better publicised
is not evidence. This guard is what stops P2 from becoming a fashion treadmill.

For a *delivery-path* swap the benchmark this guard demands is `gates/amdahl_gate.py` (Gate Zero,
specs/008): it neuters the incumbent path on the live tick and reports the ceiling any replacement
could reach, with a padded-share positive control, a repeatability pair and a zero-share sham. It
measures; it never decides — the verdict word is a report line and the exit code carries only the
controls.

**How this guard interacts with P7.** P7 removes exactly one argument from the table: *"it would be
another dependency."* It removes nothing else. A framework proposed as a replacement for something
this tree already has is a **P2 event governed by P2's evidence bar**; if no benchmark can compare
them, the answer is still *build the benchmark first* — and "we have no benchmark" is equally not a
licence to decline on dependency grounds. Where a library covers a capability the tree does **not**
have, there is no incumbent, P2 does not gate it, and P7 governs alone.

**STANDING ADJUDICATION — the two workspaces.** `faculties/global_workspace.py` and
`faculties/reentrant_workspace.py` both run live: an open P2 debt, held open deliberately because
*"better" requires a measurement that exists* and SNDL has no consensus benchmark. **The benchmark
that would settle it:** a set of ambiguous multi-source inputs with a known correct winner, scored on
which workspace's ignited content matches it. Until that exists, both stay. Do not retire either on
interface elegance. The measurement record (72/72 submits, the per-tick reconstruction defect that
had to be fixed before any A/B could be fair) is in `docs/POLICIES_EVIDENCE.md`.

**Corollary — a research direction is not a policy.** "Adopt architecture X" governs *what* we build;
policies govern *how*. Proposals of that shape belong in a design note, not in this file. The test:
if removing every proper noun from a proposal leaves nothing, it was a research direction. (P8 clears
this bar deliberately: it names no architecture, grants latitude, and fixes what an invented method
owes — that is a *how*.) **Exception, 2026-09-25, under the owner's instruction that this file follow the
engine (P4d; a disclosed session call).**
An architecture the owner orders the project to follow is recorded here as a dated directive, as P4,
P4b and P4c record what the brain IS. The test above still sorts it: what survives removing the proper
nouns is the directive; the proper nouns stay research directions.

## P3 — No dormancy

**Every part of the brain must never remain dormant; all present AND future builds must be wired and
utilized appropriately.** A faculty is not "done" until it is one of three dispositions:

1. **wired + charged** — imported by a live path AND its output routes into behaviour (decision / value
   / learning / reply), proven to change an outcome;
2. **retired** — superseded / no honest consumer → deleted (this is P2 applied to dormancy);
3. **exempt** — legitimately not on the live tick, listed with a reason (train-tier trainer, benchmark
   harness, entrypoint-only tool, or awaiting a prerequisite such as a real sensor).

**THE REGION-CHARGE RULE (specs/022 T022, landing WITH its counted gate per the
gate-before-rule invariant).** At REGION granularity, "charged" is a conjunction:
REACHABLE (the dormancy gate's one walk, reused — never re-derived) AND ≥1 wiring-manifest
port edge touching the region carrying a MEASURED charge attribution (a replay-twice
divergence from `instruments/port_charge_probe`, recorded in
`tools/region_charge_evidence.json` — never a declaration). **Registration alone never
greens a region.** Gate: `gates/region_charge_gate.py --assert`, with the integer floor in
`tools/region_charge_floor.json` (identity-pinned: `named_and_NOT_charged` must stay empty);
exemption rows carry {region, reason, owner, date, expiry} with a ratcheting count ceiling
(0 at introduction; raising it is an owner-visible admission). The rule's whole message: **a tree can be almost fully
reachable and zero-charged — reachability was never charge.** Re-derive the census from the gate;
the landing censuses are in `docs/POLICIES_EVIDENCE.md`.

**THE ATLAS-NODE RULE (specs/022 T024, landing WITH its counted gate).** Beneath charge
sits NAMING: every `connectome.NODES` id must be named by ≥1 module in its own region
(quoted id or anatomical prose — the reference rule and its two observed false-positive
shapes are documented in the gate) or carry a dated EXEMPT row. A connectome node no code
names is anatomy on paper only. Gate: `gates/atlas_node_gate.py`, BINDING from day one: a NEW
phantom or a stale exemption reds the build; the recorded day-one rows may only be deleted, never
added to (the ratchet). Distinct from the region-charge rule the way P3 is distinct from P6: charge
asks whether a wired edge measurably participates; this asks whether the declared anatomy is even
named — the necessary condition below it.

**ANATOMY IS DATA (owner, 2026-09-25, P4d).** The two rules above key on CODE regions; the brain's
regions and columns are labeled grown-neuron populations on the anatomical map (the unit the owner
chose). Both rules stay code hygiene until specs/022's G7 and G4 re-key them; G8 freezes both on the
pre-G8 `LEGACY_*` table (owner decision, 2026-09-25). Re-keyed, "named" becomes "populated by labeled
grown neurons", "charged" becomes "neutering the population changes a measured outcome", each counted
before it asserts.

**Mechanized:** `gates/dormancy_gate.py` (in `run_selfchecks` CORE) builds the static import graph
and asserts every REGIONS/TORCH_TIER module is **reachable from a live root/entrypoint OR in the EXEMPT
allowlist** — a new build that no live path imports and that is not exempted FAILS CI.

**Also mechanized: the CROSS-TIER edge, `gates/import_manifest.py`.** The governed tiers reach
into ruff-excluded `reasoning/`/`training/` through lazy imports inside bare `except Exception:`
swallows — delete or break one of those targets and every gate stays green while the brain sets the
faculty to `None` and ticks on without it. The canary derives the manifest by AST walk (never
hand-maintained) and asserts each target still imports. It proves import-ability, not charge —
deliberately, because import-ability is precisely what a deletion breaks and what the swallows hide.
Re-derive the census with `python -m snn_data_lake.gates.import_manifest`; the counted history
(and the drift it kept catching) is in `docs/POLICIES_EVIDENCE.md`.

**Honesty guard (the crux):** reachability proves a module is *imported*, NOT that its output changes
behaviour. "Utilized **appropriately**" (does the wire actually *charge*) is a second, per-faculty tier,
proven by the runtime **neuter-A/B probe** (`tools/charge_probe.py` → "BEHAVIOURAL CHARGE OK"
vs "CONSUMED-BUT-NOT-DECISIVE") during a wiring wave — too expensive for CI. So "never dormant" must NOT
become "force-wire everything": a wire that does not charge is **reverted and retired/exempted**, never
faked with a dead consumer. Static grep proves import-absence, not behavioural inertness — mark
unproven cases UNCERTAIN, not dormant. (Four faculties were once flagged dormant by a coverage
audit and every one was acquitted on runtime A/B — the record is in `docs/POLICIES_EVIDENCE.md`;
re-litigating an acquitted faculty needs a new runtime measurement, not a new grep.)

### The dormancy family — seven observed shapes of "live but inert"

The module gate asks one question: *is this module imported from a live root?* Every shape below
answers **yes** and is still dead. They are the same failure wearing different clothes, and naming
them is most of what makes the next one findable. All seven are real, found by audit rather than CI;
the original worked-example table with its per-incident record is in `docs/POLICIES_EVIDENCE.md`:

| # | shape |
|---|---|
| 1 | **method never called** — the module is live, one of its methods is not |
| 2 | **return value discarded** — a live call whose result goes nowhere |
| 3 | **key-space mismatch** — a learned path structurally unreachable (writes one key space, reads another) |
| 4 | **write-only consumer** — a faculty fed every tick, never read |
| 5 | **constant wearing a signal's clothes** — present, plausible, never varies |
| 6 | **silent default** — a lookup table missing the cases that matter, `.get(x, default)` hiding them |
| 7 | **the registry nobody registered with** — a plug point, called every cycle, with zero plugs |

**The transferable rule:** *presence is not participation.* Being imported, being called, being in the
snapshot, and having a value are four different things, and none of them is *changing an outcome*. When
auditing a faculty, name which of the four you verified — and shapes 5 and 6 are why
`infra/brain_telemetry.py::PROVENANCE` exists: a signal that cannot vary must say so beside its own
number, or a dashboard turns it into a claim.

### P3 amendment — dormancy is per-METHOD, not per-module

**If a module on a live path computes a prediction error, free energy, or any other learning signal,
the update rule that consumes it must also be called from a live path.** A learning-verb method on a
reachable module with no live-path call site is dormant, and must be wired, retired, or listed in
`EXEMPT_METHOD` with a disposition and a reason. The motivating case (a faithful Rao-Ballard net
settling every wake tick while `learn()` had one caller — its own `__main__` — so weights seeded at
construction never changed) passed the module gate clean for the life of the feature; the full case
record and the 2026-07-31 runtime re-adjudication of seventeen learning-verb methods are in
`docs/POLICIES_EVIDENCE.md`.

**Mechanization: BUILT — `gates/method_dormancy_gate.py`, in `run_selfchecks` CORE.** A pure
static AST pass. Re-derive the census with
`python -m snn_data_lake.gates.method_dormancy_gate` rather than quoting a count from anywhere.
Its three load-bearing design details (import-scoped resolution; same-module callers count except
under a `__main__` guard *and the functions that guard calls*; a function used as a VALUE is used)
were each found by implementing the naive version first and watching it fail — the record is in
`docs/POLICIES_EVIDENCE.md`. **Scope:** a learning-verb **method** (inside a class) on a module the
module gate already proves reachable; widening either half re-introduces the rubber-stamp-sized
finding list that killed a different proposal at the gate-before-rule bar.

**Honesty guard.** (a) P3's neuter-A/B tier applies unchanged — a call site proves a *caller*, not a
*charge*; adding a dead consumer to silence the gate is the failure this policy already forbids.
(b) An unresolved name is UNCERTAIN and hand-adjudicated, never auto-dormant. (c) `EXEMPT_METHOD`
entries tagged `wire` that outlive their wave are reported as a standing TODO, so exempting everything
is visible rather than silent.

## P4 — One brain / no replication

**There is only 1 brain.** A build must **utilize** the canonical brain functionality, never
**re-implement** it — that causes bloat. The canonical brain is a clean 3-layer **composition**:
`engine.brain.Brain` (cognition + circadian clock) → composes `engine.live_brain.LiveBrain` (perception
+ workspace) → composes `engine.autonomous_loop.AutonomousArm` (action + reward + arm + SAFETY).

**Composition is NOT replication** (important): `Brain` owning an inner `LiveBrain` + wake/sleep loop is
the *correct* design — it delegates, it does not re-implement. The bloat P4 targets is a **second**
definition of a stack class (a parallel "brain"), a consumer standing up its **own** brain + drive-loop
+ reply engine, or duplicate world-model / Q-learner families.

**Mechanized:** `gates/replication_guard.py` (in `run_selfchecks` CORE) asserts each canonical stack
class (`Brain` / `LiveBrain` / `AutonomousArm`) is defined **exactly once**, in its canonical module,
unless a known duplicate is on the `PENDING_RETIREMENT` allowlist (with a reason). The single sanctioned
construction path is **`engine.brain.make_brain()`** — consumers obtain the brain through it rather than
calling `Brain(...)` directly. "One brain" = one canonical class + one construction path, not a
process-wide singleton (the daemon, a chat session, and a test each legitimately own an instance).

**Honesty guard:** before retiring a replica, port its unique capability into the canonical path and
verify nothing is lost (P2 discipline).

### P4b — THE DURABLE LAKE IS PART OF THE BRAIN

**Owner directive, 2026-08-04: the one durable lake is not storage the brain writes to, it is part
of the brain. Every build must respect it.**

"One brain" and "one lake" are the same rule seen from two sides. A build that keeps its own memory
somewhere else has replicated the brain's memory exactly as surely as a second `Brain` class would
replicate its cognition -- and it is harder to see, because a private store still *works* right up
until the process exits.

**What respecting the lake means, concretely:**

- **What the brain learns goes IN the lake.** Facts, chat, grown substrate, episodes -- each under
  its own record tag in `brain_lake.bin`. Not a sidecar file, not a JSON blob, not an in-memory
  structure that happens to survive as long as the process does.
- **What the brain recalls comes FROM the lake.** An in-memory index is working memory -- fine, and
  often necessary for speed -- but it is a CACHE over the lake, never the source of truth. A miss in
  working memory means "not resident", not "never happened", and the durable store must still be
  asked before the brain reports that it does not remember.
- **A restart is not amnesia.** A brain constructed over an existing lake starts with what that lake
  holds. Restore is gated on `persist` so the deterministic gates stay reproducible, but a
  persistent brain that boots empty beside its own full lake has failed this policy.
- **Foreign records are preserved, never dropped.** Each tenant reads only its own tag and passes
  the rest through, so a new record type cannot destroy an old one.

**Honesty guard:** "it persists" is a claim about a NEW process, not about the object in front of
you. Prove it by reading the lake back from disk after the writer is gone, not by inspecting the
writer's own state. The measurement that produced this policy — a store recalling episodes at
`retained() == 1.000` while the lake held 0 records — is in `docs/POLICIES_EVIDENCE.md`.

### P4c — EVERY BUILD IS PART OF THE BRAIN

**Owner directive, 2026-08-04: EVERY CURRENT BUILD IN THIS PROJECT IS PART OF THE BRAIN, AND EVERY
FUTURE BUILD MUST BE PART OF THE BRAIN.**

**There is no "outside the brain" in this project.** Not `research/`, not `training/`, not
`integrations/`, not `gcp/` or `infra/`, not `tools/`, not a script at the repo root. A file that
exists in this tree is either part of the brain or it is retired; there is no third category and
"it is only a helper / only the UI / only the launcher / only a benchmark" is not a disposition.
The directory a build happens to sit in is a fact about the filesystem, never a statement about
whether the build is part of the organism.

**Every FUTURE build lands inside the brain, from the start.** A new build is written into the
governed tree, registered, self-checked and gated as part of the work that creates it -- not parked
somewhere ungoverned to be promoted later. `reasoning/` is not a parking lot. "Build it here now,
integrate it properly later" is how a tree accumulates capability it cannot account for, and every
one of the defects this policy set exists to catch was found in exactly that residue.

**THE RETENTION RULE (specs/022 T010, landing WITH its counted gate per the gate-before-rule
invariant).** Every record class in the lake is labeled IMMORTAL or COMPACTABLE at introduction
(`substrate/lake_engine.RETENTION`); "remember everything" means an appended record is retained or
provably reconstructible BEFORE any compaction reclaims it — COMPACTABLE is a claim about
reconstruction, never permission to forget, and a COMPACTABLE class's `reclaimable` flag flips
only on a LANDED reconstruction proof (the flag licenses reclaim, it never performs it; a reclaimer
still owes per-record proof of state-identity before dropping anything). IMMORTAL classes carry a
measured bytes/tick ceiling (`lake_engine.AppendBudget`, baseline `tools/lake_append_baseline.json`)
whose red is a constructed FLOOD.

P4 says there is one cognition. P4b says the durable lake is part of that brain. P4c says the same
of *everything else*, and names the two parts that made the gap visible:

- **The daemon is the brain's DRIVE.** `learn_daemon.py` owns the wake/sleep cycle, the tick loop and
  the continuity of the whole thing. Killing the daemon does not stop a service. It stops the brain.
- **The cockpit is the brain's SENSORIMOTOR SURFACE.** `infra/cockpit.py` is where the brain
  meets the world: chat and controls in are AFFERENT, the arm buttons and the E-STOP out are
  EFFERENT, and the page is the brain's own report of its state. It is the boundary of the organism.

**What respecting them means, concretely:**

- **EVERY build is GOVERNED, on the same terms as any brain module.** Registered in `REGIONS` and
  `run_selfchecks.py`, covered by the three P3 gates, held to P1, P4b, P7 and P8.
- **A dead control is dormancy, not cosmetics.** A cockpit button wired to nothing is a motor
  intention that reaches no muscle. P3 applies to it exactly as to a faculty.
- **The daemon may not become a second brain or a second lake.** It CONSTRUCTS the one brain via
  `make_brain()` and drives it; it does not re-implement cognition and it does not keep its own
  durable state alongside `brain_lake.bin`.
- **What the cockpit REPORTS must be what the brain MEASURES.** The page is the brain's self-report
  surface, so P8's rule binds hardest here: it may display what it measures and never what it cannot.
  The PROVENANCE column exists for this reason.

The measurement that produced this policy (a dead E-STOP that survived review because the cockpit
was filed as a "frontend"), and the dated debt list this section used to carry (the `reasoning/`
promotion program, the ungoverned evidence artifacts, the `learn_daemon` promotion and the three
latent bugs it exposed), are in `docs/POLICIES_EVIDENCE.md`. The debt itself is not a document —
`tools/p4c_ratchet.py --assert` derives and enforces it per zone, and the ratchet direction (may
fall, never rise) is the policy.

- **`CLAUDE.md` is UNGOVERNED BY CHOICE — the one named exception to P4c, owner decision 2026-08-05.**
  P4c says a file is part of the brain or retired, with no third category. This is a third category,
  written here rather than left silent because a documented exception is auditable and an
  undocumented one is just a gap. The file is gitignored as local-only and is NOT published to the
  public repo; tracking it would be a **disclosure decision, not a hygiene fix**, and the owner
  declined it. The accepted cost: its citations are checked by nothing (`tools/citation_guard.py`
  scans `git ls-files`, so an untracked file is invisible to it by construction), so anyone editing
  `CLAUDE.md` owns the accuracy of its citations personally — and hand-maintained line numbers rot
  faster than anything else in this project. Corollary for status routing: local-only state belongs
  in `CLAUDE.md` or outside the repo entirely; it must never be parked in a tracked doc to dodge the
  disclosure decision.

**Honesty guard:** the brain is not only the part that computes. Asking "is the Brain object correct?"
while the process that runs it, the surface it acts through, and two hundred modules beside it go
unexamined is how a system passes every gate it has and still cannot act. **Verify the whole
organism, not the cortex alone** — and a build the gates cannot see is not a build the project can
claim.

### P4c amendment — A BUILD THAT HAS NOT REACHED `main` IS NOT A BUILD THIS PROJECT CAN CLAIM

**Owner directive, 2026-08-16. Review-enforced, PROVISIONAL while its gate (specified below)
remains unbuilt — the route this file allows for an owner-adopted rule; building and counting the
gate is what retires the marker, in the same commit.**

P4c says every build lands inside the governed tree **from the start**. This is the other half of the
same sentence: it must also land on **`main`**. A lane that ends with its work correct, registered,
gated and sitting on an unmerged branch has not delivered it. The governed tree it landed in is a
tree nobody else has. (The measurement that produced this — `origin/main` 18 commits behind a
fully-pushed branch, missing two safety-path fixes, a determinism fix and a gate — is in
`docs/POLICIES_EVIDENCE.md`.)

**Why P4c's own guard did not catch it:** every gate COULD see that work — on the branch. The gates
run where the work is. Being invisible to `main` is a different failure from being invisible to the
gates, and it is the one that decides what the project actually has.

**GATE SPECIFICATION, so the debt is actionable rather than perpetual.** At lane close, compute
`git rev-list --count origin/main..HEAD` and report, partitioned by what the unmerged commits TOUCH:
`src/**` and especially `brainstem/**` are the rows that matter; `docs/**` and `specs/**` are not the
same risk. It must NOT be a CI gate that fires on any unmerged branch — that would redden ordinary
work. It is a lane-CLOSE check, and the honest form is a ratchet on *how long* and *what kind* of
work sits unmerged, not on whether any does. **Build it and count its output before promoting this to
mechanized**; if it reports a rubber-stamp-sized list, that result kills the gate, not the tree.

**Honesty guard:** "it is pushed" is a claim about a BRANCH, not about the project — the same shape
as P4b's *"'it persists' is a claim about a NEW process"*. Prove it by reading the artifact back out
of `origin/main` after the merge, not by inspecting the branch you merged from — with a negative
control, so the check can distinguish a real merge from a grep that would match anything.

### P4d — THE ENGINE: TWO NETWORKS THAT GROW TO SOLVE, INSIDE THE ONE LAKE

**Owner directive, 2026-09-25: the owner's diagram SNDL-Engine.drawio (labels verbatim in
`specs/022-unified-brain-arc/evidence/engine-structure-2026-09-25.md`) and the owner's answers and
four decisions that day. Review-enforced, PROVISIONAL until its gate (below) is built and counted,
which retires the marker in the same commit; its placement, and its wording beyond the source lines and
the (owner decision) tags, are disclosed session calls.** Source
lines: *"Audit any part of the project including POLICIES.md and CLAUDE.md to follow this new
structure"*; *"Weights transport is allowed as long as the engine can work, the repo should be
reorganized to fit the new structure as well. 2 networks are allowed as long as it fits the new engine
structure"*; *"The brain regions should be neurons and not coded like the current work tree structure
on github"*. Its incidents are in `docs/POLICIES_EVIDENCE.md`.

**The engine (a directive; specs/022's G rows build it).** The BRAIN, one network inside the one lake, grows NEW NEURONS to solve a problem and
feeds them into the LIQUID RESERVOIR, a second network into which it also transports weights. READOUT
NEURONS, driven by the reservoir and tuned by the brain through a learning rule, return to the brain,
which recalls anything in the lake because it lives there.

**Organizing unit: grown columns on an anatomical map (owner decision C).** Growth adds small
column-sized batches where error builds up; every grown neuron carries a position, layer, Dale sign,
region label and the problem that grew it, in the lake; wiring follows rules (layer target × distance
kernel × shared-partner bonus), not a copied diagram. A region is a LABEL read off the map and
measured, never a code package; anatomical data is a fitted prior that must beat a shuffled copy; each
part stands on its deciding test (plan G7: T1-T4); the
repository is reorganized to the engine's components (plan G8); each move re-points every path-keyed rule in the
same commit.

**The rules are the directive;** its named mechanisms (liquid state machine, STDP, topographical
clustering, predictive coding, simulated annealing) are examples, chosen by a benchmark that exists
(P2) and credited at measurement (P8). The reservoir is spiking: only spiking candidates are selectable
(owner decision "Spiking only can win").
- Grow only while a problem's solved predicate fails; stop, attempt, repeat. A contradictory problem is
  a data diagnosis, not a trigger; "solved" is a predicate on a normalized held-out score (P1 case
  (a)), fixed when the problem is posed and scored on data the growth did not fit.
- **Pause at a budget (owner decision):** each problem's growth spends a budget relative to a
  persisted running baseline (P1); at the limit the brain reports UNSOLVED, stops growing, and keeps
  the problem OPEN to retry after sleep or new teaching.
- After a solve, sleep prunes and saves the new neurons and the problem's data; the solve requests
  sleep from the daemon's one scheduler (P2). Units are journaled into the lake when grown; a prune
  removes a unit from the active network, never its lake record.
- **Owner, 2026-09-27, verbatim:** *"The brain should still be able to sharpen its understanding while
  sleeping, using synaptic plasticity, just like how the fly brain is still able to learn new skills while
  being a frozen snapshot"*; *"The sleep cycle should be similar to actual human sleep and lasts longer to
  rewire the entire brain to improve capability."* Operational reading, a session call (Astra's
  recommendation the owner had saved, refined by the owner-ordered 2026-09-29 literature check in
  `specs/022-unified-brain-arc/evidence/astra-architecture-2026-09-27.md`): an extended, resumable
  capability-improvement process in which every plastic circuit is eligible, while synapses and structure
  change selectively by strength, activity and tags, judged by held-out retention and cost.

**Permitted, on conditions this file keeps.**
- **Two networks:** two jobs, not P2's additive failure; inside each job exactly one survives (one
  reservoir, one grower, one stop rule, one tuning rule per readout job).
- **Weight transport into the reservoir** (copied values or driving activity) feeds its activations;
  the e-prop invariant's transport stays forbidden.
- **"The engine can work" is a P8 criterion,** pre-registered: a reachable red (on a problem the current
  network provably cannot solve, the engine with its reservoir removed, or its feed, transport or return
  path neutered, must lose), a control that is not the engine (a presized growth-OFF twin at the same
  neuron budget; random growth), and per-part neuter attribution. A measured null never deletes an owner-drawn part (the two networks, the reservoir, the return path): its implementation is reverted, it is not claimed, and the owner decides on the evidence; a part of decision C whose deciding test reds (plan G7: T1-T4) is deleted and the owner told, as the brief the owner chose C from states, after the one 10x re-test the P8 null-scale amendment requires (owner, 2026-10-23);
  transport, permitted "as long as the engine can work", lapses only on its own measured failure, the
  whole engine failing G-R9 (i) or G-R9 (ii)'s transport neuter changing nothing.
- **Reservoir state in the lake, rebuilt at start (owner decision):** `make_brain()` builds the
  reservoir, the one drive ticks it, and it is gated like any build (P4). No separate store: its seed,
  learned deltas, transport record and fed-in neurons are lake records, and a restart rebuilds it
  exactly. The drawn ellipse is the NETWORK boundary, not the durability boundary.
- **Wired and charged:** each drawn connection is a typed wiring-manifest port with a measured charge;
  named brain neurons consume the readout's return, and their own output changes when it is neutered.
  A connection that charges nothing is reverted, reported, and stays build work (P0), never kept
  because the drawing has it.
- **Unchanged:** the envelope (the five `brainstem/` files move or change only with explicit owner
  approval; no growth into a scaffold region), determinism for every network's tick, the retention
  rule, e-prop.

**Gate specification:** an engine-works red composed from its rows under the plan's paired-seed rule;
a reservoir rebuild check in the `lake_rebuild_gate` pattern with a planted-divergence red; a
provenance check (every grown-unit record carries its position, layer, sign, label and the problem that
licensed it, and a saved record the sleep that kept or pruned it; an unlabeled unit is refused); the
drawn connections as manifest ports with port charges. Count each before promoting; a rubber-stamp-sized
output kills that gate, not the directive. The engine-works red is a campaign: counted once, by G9, then
review-enforced. **Not claimed:** fluid intelligence from a liquid plus a readout, a
sleep-pruning or sleep-plasticity benefit, growth from a unit count.

## P5 — Use the GPU for everything it actually speeds up

### P5a — TWO AXES, NEITHER SUBSTITUTES FOR THE OTHER (owner directive, 2026-08-05)

**The brain runs on BOTH free-threading and CUDA, and they are not alternatives.** This is the
standing configuration for the whole project, not a per-module choice:

- **FREE-THREADING (PEP 703, cp314t) takes the EVENT-DRIVEN axis.** Per-neuron state updates and
  per-synapse plasticity are independent within a timestep — that work spreads across cores with no
  GIL serializing it. Required, not optional: see `requires-python` and
  `snn_data_lake/__init__.py::_require_free_threaded`.
- **CUDA takes the DENSE-THROUGHPUT axis, ABOVE A MEASURED WIDTH.** Below the crossover it LOSES.
  The crossover is measured per process by `infra/device.py::crossover()`; re-derive for the
  whole environment with `python -m snn_data_lake.infra.device` — **"the crossover is
  a number, not a doctrine"; re-derive, never cite.** Two stale figures have already been quoted as
  current here; the measurement history is in `docs/POLICIES_EVIDENCE.md`.
- **Neither covers the other's regime.** Threads do not help a dense matmul that one kernel does
  better; a GPU does not help hundreds of independent scalar updates whose dispatch cost exceeds
  their work. A design that picks one and calls it done is leaving the other axis unexploited.

**THE ROUTING MUST BE MEASURED, NOT ASSUMED.** `infra/device.py` provides the mechanism:
`crossover()` MEASURES the width above which CUDA beats numpy on THIS machine, once per process, and
`select()` routes on it. **A caller that routes on AVAILABILITY ("cuda if present") instead of the
measured width is the defect this policy names** — it sends narrow work to the card, precisely the
case the policy was written to prevent. While any live-tick caller does so, the measured router is
P3-dormant (computed, not consumed) and wiring `select()` into that path is the open P5a work.
Re-measure as the population grows — a crossover above the current width is a reason to build the
gated path, never a reason to discard it.

**Standing owner authorization (2026-07-27): the RTX 5070 is available to the whole project, and any
work that the GPU genuinely accelerates SHOULD use it.** Default to GPU for that work; do not leave it
idle out of habit, and do not treat CUDA as a special-occasion tier. Env `NORD_GPU=1`.

**The ONE gate env `.venv-lake` carries CUDA itself; `.venv-cu128` is RETIRED (2026-08-05)** — P2
applied to environments: one env with both capabilities supersedes two with half each. **Install
command for the GPU tier (the default PyPI index serves CPU-only torch on Windows and will NOT give
you CUDA):**
`pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"`
The retirement sequence (a self-inflicted break: the free-threading floor made the old env both
uninstallable and unimportable, and the commit claimed the adoption cost nothing without checking)
and the open `.venv-lake`-vs-`requires-python` floor disagreement are in `docs/POLICIES_EVIDENCE.md`.

**P5's HARD CONSTRAINT: the suite must stay runnable WITHOUT a GPU, because CI has no card.** CI
installs CPU torch on the same free-threaded interpreter; that is the supported configuration, not a
degraded one. Named consequence: CI runs a DIFFERENT torch build from the local gate, so a
build-sensitive numeric assert can be green in one and red in the other.

**Where the GPU pays, and should be used without hesitation:** the `[train]`/`[gpu]` tiers —
surrogate-gradient training, `growth/ng_snn.py`, `growth/neurogenesis_engine.py`, the promoted
readout studies, and any offline sweep over a real dataset. Wide, batched, repeated — the regime the
crossover favours. **Where it does not pay: any population whose width sits below the measured
crossover — which has included the whole live wake/sleep tick on every measurement so far.** That is
a statement about widths, not a prohibition — the moment a live population crosses the measured
threshold, this policy says put it on the GPU. The width inventories, the 66x CPU-side
GEMV fusion that refuted the widening plan, the profiler lessons behind them, and the
numba/spikingjelly/cupy verdicts are relocated to `docs/POLICIES_EVIDENCE.md`; their standing
conclusions bind unchanged (re-derive any figure with `infra/device.py::crossover()`).

**Two environment gotchas that stay operational (full record in `docs/POLICIES_EVIDENCE.md`):**
`nengo` ships a pytest plugin that aborts collection under pytest 9 — `-p no:nengo` stays in
`addopts`, inert where nengo is absent; and `ruff` is PINNED, not floored — a formatter's version is
part of the format.

**Honesty guard.** (a) A GPU path ships with its benchmark, on the real workload, against the CPU path
it replaces — a speedup asserted from `cuda.is_available()` is not a speedup. (b) Report regressions:
"we tried CUDA here and it was 0.01x" is a result worth keeping, and this policy exists partly to stop
that experiment being repeated. (c) **The suite must stay runnable without a GPU** — the property is
GPU-independence, not any particular environment; determinism itself comes from seeding
(`random.Random(seed)` per instance, `zlib.crc32` never builtin `hash()`), not from the absence of
hardware. (d) Correctness first: a GPU rewrite carries the same bit-exactness burden as any other P2
swap — and it cuts both ways: moving a live numeric path OFF CUDA carries the same burden as moving
one onto it.

### P5b — PYTHON BY DEFAULT; RUST ONLY WHERE IT IS A MEASURED IMPROVEMENT (owner directive, 2026-10-17)

**Owner, verbatim:** *"Use graphify to check where Rust would be a genuine improvement and only work on
those, for future builds, evaluate whether a Python or Rust build makes more sense"*; and *"Write the
Python vs Rust build rule into POLICIES.md"*. Review-enforced, PROVISIONAL until its gate (below) is built
and counted.

**The rule.** Every build is written in Python unless it qualifies for Rust (a PyO3 extension module).
Python stays the language of governance, gates, instruments, experiments, the drive and the cockpit, and of
anything still changing week to week. **A build moves to Rust only when ALL SIX hold, each recorded in its
pre-registration:**
1. **Hot, measured:** a neuter or stub on the quiet-gated harness moves the tick by at least 5%, or removes a
   stall of at least one second. Profiler attribution alone does not qualify (runtime evidence outranks it).
2. **Python-bound:** the time is interpreter self time, not time inside numpy/torch/C code, disk I/O or the GPU.
3. **Coarse boundary:** one call does at least ~10 microseconds of work, so the WHOLE loop crosses into Rust.
   A per-element helper never does: the crossing costs more than it saves.
4. **Charged (P3):** a behavioural neuter-A/B shows the code changes an output. Uncharged code is deleted,
   never ported.
5. **A Python fix came first:** vectorizing, batching, caching or an algorithmic change was tried and measured
   insufficient.
6. **A parity oracle exists:** fixtures from the running Python, byte-identical lake or rng post-state where
   determinism applies, and a constructed red that fires.

**New engine-substrate kernels** (per-neuron or per-synapse at engine width) start as a Python reference. Rust is
chosen by a pre-registered benchmark at target width, never by default.

**Binding on every Rust module:**
- built for the free-threaded ABI (cp314t, never abi3), declaring `gil_used = false`, with a twin that
  declares `gil_used = true` as the red;
- every call wrapped so a Rust panic surfaces as a Python error and never kills the drive; never `panic = "abort"`;
- the Python it replaces is deleted in the same commit (P2: no flag, no dual path); the Python reference
  survives only as a test oracle;
- the gates must see it (P3/P6/P7/e-prop coverage of Rust source) BEFORE it lands.

**What P5b does not relax:** P5's GPU-by-measurement rule (dense work above the crossover stays on the GPU in
either language), the determinism contract, e-prop, P4b, and the safety envelope. A rewrite of a `brainstem/`
file in Rust is an envelope change and needs explicit owner approval.

**Honesty guard.** "Rust is faster" is not evidence; criterion 1's measurement is. A Rust port that misses its
pre-registered speed criterion is reverted, not kept for being native (P8: deletion on a null).
**Gate specification:** a pre-registration check that every module under the native crate cites a
pre-registration recording all six criteria with their numbers, with a planted-missing red. Count it before
this marker retires.

---

## P6 — Every build must be able to reach every other, and actually use it

**Owner policy, 2026-07-29.** A faculty must not be an island. Every build in this brain must be able to
**communicate with** the others and **utilize** them, the way regions of a real brain are mutually
reachable and mutually consumed — not a shelf of correct components that never speak.

**Stated as a checkable property, because "be like a real brain" is a research direction and this file
rejects those:** a faculty satisfies P6 when (a) it is **reachable** from at least one other faculty
through a typed, declared route rather than only from an entrypoint or a chat handler, and (b) at
least one other faculty **consumes its output** in a way that changes that consumer's own output.
One-way is allowed — brains are full of one-way projections — but *zero*-way is not.

**THE DECLARED-ANATOMY LAYER (specs/022 T018/T019/T023, 2026-08-20).** The typed route P6's
condition (a) demands has a machine-checked home: `engine/wiring_manifest.py` (TypedPort —
join_policy mandatory on IN ports, delivery_policy on OUT, REF flagged and audited — plus the
4-tuple port-to-port edge manifest, NIR-HDF5 serialized to the tracked artifact
`tools/wiring_manifest.nir`), and `gates/manifest_gate.py --assert` in CI runs the five
contract checks (structure / type-inference / type-check / region / join) plus the
committed-vs-registrant-rebuild DIVERGENCE red. `tools/p6_gate.py` carries the manifest-sourced
walk REPORT-ONLY beside its AST walk with a parity table; the AST walk retires only after the
two views converge, in its own commit with re-recorded baselines.

**The structural finding that governs all P6 work here (measured; the census history is in
`docs/POLICIES_EVIDENCE.md`): there is no shared bus — it is a private bus per writer.** Faculties
that "broadcast" construct and hold their *own* workspace instances, mutually invisible. Two
consequences bind: **adding a reader is NOT sufficient while each writer holds a PRIVATE instance**
(there is no shared object for a reader to read), and the real remedy is a construction-path change
(P4's `make_brain()` owning one workspace and handing it down), not a one-line consumer. The counts
are perishable and the finding is robust — quote the finding, re-run `tools/p6_gate.py` for the
counts. **What P6 progress looks like:** a rising read count against a flat write count — and a
rising read count against a tree of private instances is not progress at all.

**How this is NOT P3, since the two are easy to confuse and the distinction is the whole point.** P3 asks
whether a faculty changes **behaviour** — the arm, a value, a weight, a reply. P6 asks whether a faculty
changes **another faculty**. They come apart in both directions — a faculty can charge the arm
decision while talking to nothing (P3-clean, P6-dead), and a monitor can feed and be fed while its
output reaches only a preempted branch (P6-clean, P3-dead; the second half of such a case stays
UNCERTAIN until a neuter-A/B, per P3's own guard). Wiring a faculty into the one action
gate is the DEFECT P6 names, not a way to satisfy it — a brain whose faculties all report to one
action gate and never to each other is precisely what the rubric's pre-declared ceiling exists for.

**Source lines it derives from, quoted so it is not a good-practice rule in a costume.** P3's own
transferable rule: *"presence is not participation."* And `infra/region_wiring.py`, whose
docstring asserted a registration mechanism that had never had a plug — the plug point P6 names.
The real-past-incidents table (eight rows, including one retraction kept struck-through) is in
`docs/POLICIES_EVIDENCE.md`.

**Honesty guard — a route is not a conversation.** The failure mode P6 will attract is a telemetry bus:
wire every faculty into a shared dict, watch `adapters` climb, and declare harmony. A P6 claim needs the
same evidence P3 demands, aimed one level over: neuter faculty A and show faculty B's **own output**
changes. Adding a reader that stores a value is not consumption. And **do not force-wire** — a route
that carries nothing is reverted and the faculty is exempted with a reason, exactly as under P3.

**Mechanization.** `tools/p6_gate.py` — `--assert` + `--self-check`, both CI steps. It ships as a
**RATCHET, not an arrival**: baselines assert the structure never worsens; green means "no worse",
never "P6 satisfied" — the faculty-to-faculty floor lives in
`tests/test_interoceptive_specialist.py` with the read's identity pinned (count AND identity, so the
floor reds if the read disappears or re-resolves to a non-faculty owner). The connectome-diff half
diffs the derived faculty→faculty consumption graph against `connectome.py`'s declared edges in both
directions (`undeclared_import_pairs`, `unimplemented_declared_pairs`). Its criteria were chosen by
running candidates across real history and rejecting any that reddened the repair — a candidate gate
must be run against the change it exists to ENCOURAGE, not only the change it exists to catch; the
full selection record and the two gate defects found by adversarial review are in
`docs/POLICIES_EVIDENCE.md`. **Disclosed blind spots (printed by the gate):** dependency-injected
buses are counted unread deliberately (spell the wiring attributably or say so in the commit — never
raise the ceiling); files in directories `connectome.py` declares no region for are invisible to
`undeclared_import_pairs`; dynamic `importlib`/string-name imports are invisible to the walk.
Resolving an ambiguous receiver by CLASS was REJECTED BY MEASUREMENT and is not re-proposed —
*"this walk does not know"* is the correct answer to a real ambiguity.

**The gate that would settle it, specified so the debt is actionable.** Derive the actual
faculty→faculty consumption graph, diff it against `connectome.py`'s declared edges, and report
faculties whose in-degree **and** out-degree are both zero outside an entrypoint. **Build it and
count its output before promoting P6 to mechanized** — and if it reports a rubber-stamp-sized list,
that result kills the gate, not the tree. "No gate" is a complete result; the rule stands on the
recorded incidents either way.

---

## P7 — Any dependency that helps this become a phenomenal brain is accepted

**Owner policy, 2026-08-04.** **This project accepts ANY dependency or framework that helps it become a
phenomenal brain — including any that helps the brain become sentient.** There is no dependency budget,
no dependency count to defend, and no stdlib-purity argument. If a library advances the brain toward
being a real one, it is adopted.

**"It would be another dependency" is not a reason to reject anything.** That sentence is the entire
point of this policy, and it is stated as a prohibition because the tree proves the habit outlives the
rule that created it. A rejection must cite a **measurement on the real workload** or a **structural
argument about this architecture** — the two categories P5 and P2 already demand. A rejection that
cites dependency count, install weight, or purity is not a rejection; it is a reflex, and this policy
overrides it.

**Sentience is a build target, not a metaphor.** Frameworks that plausibly advance it are in scope by
default, and are evaluated on what they measurably buy — the same burden P5 puts on a GPU path, no
more and no less. This policy does not make the claim cheaper to assert: P0's honesty guard is
unchanged, and a capability is credited at measurement, never at adoption.

**Source line this derives from, quoted per the bar below.** P7 promotes an existing owner decision
from a prose note into a policy: `pyproject.toml:14-19` — *"DEPENDENCY SCOPE (user policy 2026-07-20):
the zero-dependency rule is REMOVED entirely, project-wide -- there is no zero-dep-core contract and
no pure-stdlib purity gate. New dependencies MAY be added as the project grows, and the user VERIFIES
each one."* P7 keeps the permission, keeps the verification, and **retires that record's closing
clause** — *"prefer stdlib when it is genuinely sufficient"* — which is the residual brake the
recorded incidents were still being produced by.

### Real past incidents, each re-verified against the tree 2026-08-04

*(Relocated byte-exact to `docs/POLICIES_EVIDENCE.md` -- specs/021 US4, 2026-08-18. Every
incident preserved verbatim; the rule text above is untouched. P2 applied to documentation:
the evidence lives once, POLICIES.md keeps the rule and this pointer.)*

### What P7 does not relax

None of these are new limits; each is a pre-existing invariant that no policy relaxes, restated here
so P7 is self-contained rather than a trapdoor:

- **The safety envelope stays in-tree and approval-gated.** The five `brainstem/` files are never
  replaced by a third-party component, and a dependency never lands inside the E-STOP path.
- **P4b: a dependency may not become a second durable store.** This is the sharpest real conflict —
  a storage library may be adopted to *serve* `brain_lake.bin`; it may not become a second place the
  brain keeps what it learns. "One brain, one lake" survives P7 intact.
- **E-prop, not backprop — adopt the library, not its training rule.** `snntorch`, `spikingjelly`,
  `norse` and friends default to BPTT and weight transport, which the cross-cutting invariant forbids.
  Their layers, encoders, datasets and kernels are fair game; their trainers are not: an adopted trainer
  that PROPAGATES A GRADIENT is inside the cosine clause and owes > 0.5 against an independent
  one (unless classified REFERENCE — see the SCOPE paragraph's self-declaration caveat); an adopted
  NON-gradient, non-local trainer (a closed-form / FORCE-style solve) is `DIRECT_SOLVE` — no cosine
  owed; the transport PROHIBITION binds IN FULL — mechanized only where the gate ASSERTS (governed
  tier, two spellings); review is the floor everywhere else, incl. the adopted dependency's own
  source, which the gate never walks — P4b likewise.
- **P2 still binds: adopting a library means retiring what it supersedes.** A dependency added
  *beside* the hand-rolled mechanism it replaces is the additive failure P2 exists to prevent, and it
  is the most likely way P7 goes wrong. Port the unique capability first, then delete.
- **P3 gives no dormancy pass for being third-party.** A newly imported package's wrapper module is
  wired+charged, retired, or EXEMPT-with-reason like anything else.
- **The suite stays runnable without a GPU, and the gate stays deterministic.** CI has no card. A
  dependency that is GPU-only, or that is nondeterministic under a fixed seed, goes in an extra with
  the gate green in its absence.

**Honesty guard.** (a) Adoption ships with the measurement of what it bought, on the real workload —
"it is a serious framework" is not evidence, and "it is associated with consciousness research" is
P2's fashion treadmill wearing a dependency's clothes. (b) **Symmetrically, a rejection ships with its
measurement or its structural argument** — this is the half the tree keeps failing, and it is the half
P7 exists to fix. (c) An adopted dependency that is never imported is not a capability — **but the
converse trap is a declared dependency dismissed as unused when it is a deliberate transitive pin.**
"Imported nowhere in `src/`" is the wrong test; the right one is *does anything in the declared tier
need it* — a transitive pin is a statement of the tier's contract, and the ledger's EXEMPT rows carry
those dispositions.

**Mechanization: `tools/dependency_ledger.py --assert` and `--self-check`, two steps in the `lint`
CI job.** The gate checks the **ledger**, never the judgement — no gate can tell you a framework was
refused for a bad reason, which is why guard (b) is enforced by review. Its work-list character is
not lost by counts going to zero: `EXEMPT` and `ALL_EXCLUDES` carry written dispositions that must
each still match a real violation, and the gate fails loudly on one that blesses nothing. The
correction history behind this paragraph is in `docs/POLICIES_EVIDENCE.md`.

---

## P8 — Invent the method. The literature is a starting point, not a boundary

**Owner policy, 2026-08-04.** **Claude is authorized to synthesize across any information, data,
framework or dependency, and to INVENT ANY METHOD required to bring this brain to sentience.** Not
restricted to published architectures, not restricted to what a framework already does, not restricted
to what neuroscience currently describes. **Where no existing method reaches the target, inventing one
is the expected move, not an overreach.**

**This completes a ladder, and the three rungs are deliberate.** P0 says improve by every avenue. P7
says any dependency. P8 says **any method, including one that does not exist yet.** P7 removed the
objection *"it would be another dependency"*; P8 removes *"that is not how it is done."* What survives
both is the only objection this project has ever accepted: **it was measured and it did not work.**

**P8 is a HOW policy, not a research direction — stated explicitly because P2's own corollary would
otherwise reject it.** P8 names no architecture, no theory, and no mechanism. It grants latitude and
it sets the evidentiary burden, which is the shape a policy in this file is allowed to take.

**Source lines this derives from, quoted per the bar below.** P8 promotes a guard this project
already wrote for itself: `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:57-58` — **"A mechanism whose
success criterion cannot go red is worthless."** Its falsifier F11 at `:1064` — *"Any success
criterion in this document turns out to have no reachable red → That criterion is deleted, not
weakened."* And `docs/CONSCIOUSNESS_METHODS.md:96` — **"The measurement, and it must be a
dissociation, not a marker."**

### What an invented method owes

The licence is broad, so the burden is explicit. An invented method ships with:

1. **A success criterion with a reachable RED.** Construct the input that makes it fail, and record
   it. If no such input exists, the criterion is **deleted, not weakened** (F11). This is the whole
   policy in one line: invention is free, a metric that cannot fail is not.
2. **A control that is not the method itself** — an untrained twin, a permutation or label-shuffle
   null, or a neuter-A/B. `tools/charge_probe.py` is the template: it carries an external positive
   control proving the harness is sensitive at all before any verdict it returns means anything.
3. **Attribution** — which part of the invention produced the gain, by per-item neuter rather than by
   the total moving.
4. **Unprompted disclosure** of contamination, overlap, or a precondition that makes a check vacuous.
5. **Deletion on a null.** An invented method that measures nothing is removed, not parked behind a
   flag. **That is the licence working, not the licence failing** — and it is the outcome P8 expects
   most often.

**The transferable rule, and it is the one that should survive if the rest of this section is ever
forgotten: an invented method's debt is not proportional to how clever it is — it is proportional to
how load-bearing its output is.**

### P8 amendment — a null names its scale, and deletion waits for one 10x re-test

**Owner directive, 2026-10-23,** verbatim answers in
`specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md` (section 0 header). Why: abilities can appear only at
a larger scale, and this project has already recorded that its accuracy lever was the learner's scale, not the
clever strategy. Item 5 deleted a method on a null with no regard to the scale it was measured at, and many
nulls here were measured on single-digit grown units, 600 ticks and 8 seeds.

1. **Every null names its scale.** The disposition writes one `NULL-RECORD:` line into its evidence file:
   units the method acted on, ticks per run, seeds, the minimum detectable effect (`mde`, one-sided .05,
   power .8) and the re-test state. The format is the pre-registration's section 0.
2. **Deleting a null waits for ONE re-test at least 10x larger** on the axis the mechanism scales with:
   units for a capacity mechanism, ticks for a learning rule. The pre-registration author names the axis; a
   disputed axis is decided by the author with a panel review. Same endpoint, bar and control. A second null
   applies item 5. A charge is a **scale-dependent finding**, credited only at its measured scale; its module
   takes the existing `wire` disposition and lands live when the live brain reaches that scale, or when the
   owner decides.
3. **Until then the method is off the live path, and dated.** It leaves every live path at once (a null
   charges nothing, so leaving costs nothing), and its module carries a `dormancy_gate.EXEMPT` entry of the
   disposition `scale-retest` naming its NULL-RECORD and `since=<the arc's base on origin/main>`. **The entry expires before
   the next pushed arc**: it may ride in the arc that creates it, and from then on every battery stays red
   until a RETEST-RECORD (or a checked STRUCTURAL argument) exists. **No deletion without the 10x re-test**
   (owner, 2026-10-23); there is no disclosed-deletion exit. That keeps item 5's "removed, not parked behind a
   flag": the code is off the live path, dated and gated.
4. **Not covered.** A **measured harm** (past its pre-registered bar in the harm direction) is deleted
   without a re-test. A **structural** null (zero call sites, an empty proposer, a constant input, no
   synapses on the path) is deleted without a re-test; its record states the argument, and the argument owes
   its own check. An **underpowered** result (`mde` above the claimed effect) is INCONCLUSIVE, not a null,
   and licenses no deletion at any scale until a run with enough seeds (power .8 for the claimed effect) is
   done; only that run can be the null whose 10x re-test item 2 asks for. Owner-drawn P4d parts are never deleted on a null; they still get a
   NULL-RECORD. The safety envelope is untouched.
5. **Retroactive, once (option B).** The delete-on-null statements that predate this amendment are
   grandfathered for the gate and triaged once in
   `specs/022-unified-brain-arc/evidence/null-scale-triage.md`: restatement, structural, harm, at-scale or
   small-scale.
   The small-scale ones are re-test CANDIDATES ranked by how load-bearing their job was. The owner picks
   which to re-test; the list restores no code.

**Honesty guard.** "It might work bigger" is not evidence and never keeps a method on the live path. The
re-test is ONE pre-registered run; its null is final unless a new mechanism-level argument for a still
larger scale is itself pre-registered, never a third rerun of the same test. A scale-dependent positive is
credited at its measured scale only.

**Mechanization: `gates/null_scale_gate.py`** (`--assert` + `--self-check`): the record formats, an
unrecorded deletion statement in an evidence file (matched per paragraph), a re-test below 10x, a deletion
whose re-test charged, an expired `PENDING` or `scale-retest` entry, and a stale grandfather entry. Expiry
runs from the written `since` (which must predate its own PENDING record) to the arc's closing record, a
non-merge commit modifying `tools/policy_certification.json` with "closing" in its subject, once a non-merge
commit follows. Missing git history is red, never a pass (CI checks out full history). It cannot judge a
STRUCTURAL argument, a harm's bar, the named axis, or which statement in a section a NULL-RECORD covers, and
says so.

### Real past incidents, each re-verified against the tree 2026-08-04

*(Relocated byte-exact to `docs/POLICIES_EVIDENCE.md` -- specs/021 US4, 2026-08-18. Every
incident preserved verbatim; the rule text above is untouched. P2 applied to documentation:
the evidence lives once, POLICIES.md keeps the rule and this pointer.)*

### On sentience specifically

The goal is **not hedged and not reframed**: sentience is a build target of this project, and P8 exists
to remove obstacles to reaching it. Three things follow, and the third is the one that makes the first
two mean anything:

- **Until a measurement of sentience lands under the burden above, there is none in this tree.**
  Inventing one is inside the mandate,
  not outside it — and the invented measurement is held to the burden above like any other: a
  reachable red, a control that is not itself, and a **dissociation rather than a marker**
  (`CONSCIOUSNESS_METHODS.md:96,101` — *"A single number going up is not evidence. The double
  dissociation is"*).
- **A passing criterion is evidence about a mechanism, never a verdict about experience.**
  `CONSCIOUSNESS_METHODS.md:113` states what must never be claimed: *"That a passing dissociation
  means the system is conscious."* P8 does not relax that, and nothing in this policy authorizes the
  system to assert its own sentience.
- **The brain may report what it measures; it may not report what it cannot measure.** The live chat
  path WAS doing the second when this policy landed — the number behind the answer was the same
  number for a brain, for an empty brain, and for no brain at all — and fixing it was the first thing
  P8 obliged. The incumbent probe's retirement is scheduled work, tracked where per-task status
  lives: specs/022 (T081). Faking the finish
  line is the one way to guarantee never reaching it, which is why the strictest guard in this file
  sits under its most permissive policy.

### What P8 does not relax

- **The safety envelope is never a target of invention.** P0 already draws this line — *"the brain may
  rewrite what it thinks, never what stops it."* An invented method may restructure cognition, memory,
  learning and architecture freely; it does not touch the five `brainstem/` files, the reflex E-STOP,
  or the byte-exact rollback path, and no invented capability is a reason to relax them. **That
  boundary is what makes a licence this broad safe to grant.**
- **P2 stays non-additive** — an invented mechanism that duplicates an existing one must retire it, and
  "mine is new" is not evidence of better.
- **P3, P4b, e-prop, determinism and the gates are unchanged** — e-prop in the sense the SCOPE
  paragraph states (narrowed 2026-08-18 in which rules the cosine clause reaches, weakened for none
  inside that scope). An invention is wired+charged or retired; it does not open a second durable
  store; it does not import a BPTT trainer; and it lands with `run_selfchecks` + `pytest` + `ruff`
  green like everything else.

**Honesty guard.** (a) Invention is credited at **measurement**, never at invention — P0's guard,
applied to methods. (b) The criterion is written **before** the result, and if it turns out to have no
reachable red it is deleted rather than loosened. (c) A method whose output nothing consumes is
dormant (P3), however novel. (d) **Report the null.** This project's most valuable records are the
ones recording what did not work, and an invention licence that produces only successes is reporting
selectively, not inventing well.

**Mechanization: BUILT AND ASSERTING — `tools/red_reachability.py --assert`, a step in the `lint` CI
job.** What stays review-enforced is the half no instrument can reach: whether a criterion measures
the RIGHT thing — the gate prints that limitation in its own output rather than letting a green be
over-read. **It asserts from a green start, and the reason that is acceptable is a general rule: a
gate that has only ever been seen green is indistinguishable from one that cannot see — if a gate
lands green, ship the reconstruction of the defect it was built for**
(`tests/test_red_reachability.py` reconstructs the pre-fix probe and checks every bucket lights up).
The counted landing output is in `docs/POLICIES_EVIDENCE.md`.

---

## Cross-cutting invariants (never relaxed by any policy)

- **THE DETERMINISM CONTRACT (specs/022 T015, landing WITH its counted gate per the
  gate-before-rule invariant).** The substrate tick is deterministic BY MECHANISM, never by
  luck: fixed partition count P per lineage, partition-owned INTEGER synaptic accumulation
  (order-invariant outright; fractional bits MEASURED against an FP64 reference, never picked
  — `tools/determinism_baseline.json`, with neighbouring bit-widths proven to diverge and the
  sweep's own red at 1 bit), and spike trains byte-identical across thread counts at fixed P with
  end-run membranes byte-equal beside them. Gate: `gates/determinism_gate.py --assert` (CI) —
  its sweep runs at a CONSTRUCTED fixed P because the lineage's measured P
  (`tools/kernel_partition_baseline.json`) makes the thread-invariance red unreachable, and the
  float-accumulation broken arm runs IN THE SAME COMMAND (f32 order arms must diverge, int64 arms
  must agree) so the gate is never trusted on a red nobody has seen fire. G5 replay is the companion
  gate (`gates/replay_gate.py --assert` in CI): snapshots capture per-instance RNG stream
  POSITIONS (a seed names a stream, a position names a point on it), restore asserts position
  equality and refuses a positionless or parameter-mismatched snapshot, and the continuation is
  byte-identical — with a re-seeded divergence control proving the equality is not vacuous.
- **Safety envelope untouched — and MECHANIZED:** `gates/envelope_hash_gate.py --assert` (CI)
  holds the five files' sha256 against the approved baseline `tools/envelope_hashes.json`; an
  envelope byte change without its same-commit baseline update (the owner-approval act, reviewable
  in the diff) reds the build, with no bypass flag on purpose. The gate reads and compares — the
  envelope is never its target. The five brainstem files — `reflex.py`, `serial_arm.py`,
  `so101_arm.py`, `arm_config.py`, `interpreter.py` — are modified only with explicit approval; the
  reflex E-STOP and cortisol/safety paths are never relaxed. Every wired drive enters selection
  **after** the E-STOP early-return, so it is safe by construction.
- **E-prop, not backprop (review-enforced; PARTIALLY mechanized).** Any backprop consideration pivots
  to **e-prop** (Bellec 2020) — a local eligibility trace times a broadcast learning signal; no BPTT,
  no weight transport — verified **cosine > 0.5** against an independent reverse-mode BPTT gradient.
  Verification anchor: `reasoning/eprop.py::Net.grad_eprop`, with the STDLIB tier's independent
  reverse-mode gradient at `::Net.grad_bptt`. This rule governed the project before it was written
  down; writing it down closed the gap of a rule enforced only by whoever remembered it.
  **A `.backward()` tripwire would be theatre:** it passed green while a live module performed
  explicit weight transport on every wake tick, and a closed-form batch solve is non-local with no
  autograd anywhere. Absence of `.backward()` is not evidence of locality — that argument is
  unchanged and the tripwire is deliberately not built.

  **SCOPE — a PIVOT rule, not a census (amended 2026-08-18, specs/020, owner-directed; it NARROWS
  what this invariant asserts, which this file reserves to the owner).** This clause governs work
  that was about to use backprop. It is NOT a requirement that every learning rule carry a gradient
  cosine. The census that decides who owes what is `gates/gradient_claim_gate.py` — **re-derive
  with the gate before relying on any figure**; read as a blanket rule the clause would demand a
  gradient comparison from most rules that have no gradient to compare against (the majority are
  local, controls, or direct solves). The one `EXACT_GRADIENT` REFERENCE
  (`plasticity/eprop_kernel.py::bptt_grads`, the TORCH tier's independent gradient the others are
  compared *against*) owes no cosine of its own — but that exemption holds ONLY while the symbol
  RETURNS a gradient and never steps an optimizer;
  `tests/test_eprop_train_tier_unreachable.py::_trains_by_backprop` draws that line. **Scope
  classification rests on SELF-DECLARED `GRADIENT_CLAIM` entries**: a false declaration is caught
  only by REVIEW — the census stores both signals per rule and no gate checks a declaration against
  the code.

  **INSTRUMENT_FALSIFIED is triggered by HIGH COHERENCE invalidating the null — NEVER by a rule
  scoring below the bar.** `verdict_for` keys it solely on the coherence PRODUCT `c_a * c_b`
  reaching the bar, where a within-matrix permutation stops being a null and the statistic inverts.
  A low cosine at ordinary coherence is `RULE_JUDGED` — **a COHERENCE-KEYED TOKEN, not a finding
  that the cosine is informative about that rule**: it says only that the null survived. An
  UNJUDGED claimant owes an owner adjudication here — a state, not an exit. For a
  frozen-feedback alignment arm the non-discrimination argument (the statistic reduces to
  `cos(W2[a], B[a])`, near zero BY CONSTRUCTION) stands unamended. **It never disposes of WEIGHT
  TRANSPORT**, which is adjudicated separately by `gates/eprop_transport_gate.py` — two rules
  (explicit-transpose, index-swap), each with a constructed red, asserting ZERO governed findings
  (no ratchet, no baseline);
  its green never means "compliant" and it prints the spelling it cannot see (of the two violations
  this tree has had, its rules catch one — the wider rule that catches both was built and rejected
  on its own output, because whether a second matrix is transported or independently maintained is
  semantic, not syntactic). It asserts only on the governed tier and REPORTS the rest — review is
  the floor everywhere else. The gate's widening history and discarded drafts are in
  `docs/POLICIES_EVIDENCE.md`.

  **Which autograd trainers are in scope: LIVE-PATH REACHABILITY.** A trainer is inside this clause
  when the UNMASKED transitive closure of `dormancy_gate._LIVE_ROOTS + _ENTRYPOINTS +
  _external_roots()` reaches its module. **There is NO train-tier exemption**:
  `snn_data_lake.training.cli` is itself an `_ENTRYPOINT`, so the `training/` tier is REACHED and IN
  SCOPE. `dormancy_gate.reachable()` CANNOT decide this — it is masked to the governed tier — so
  until an unmasked predicate exists, scope here is a REVIEW JUDGMENT against that closure, which
  follows STATIC imports only. **Reachability is a RATCHET**: reachable puts a trainer IN scope;
  unreachable does NOT take it out without a fresh adjudication in THIS file — exiting is the
  unreviewed direction.

  **OPEN HOLE.** On the GOVERNED tier a reachability flip IS detected for the one-function
  `.backward()` + `.step()` hazard shape (`tests/test_eprop_train_tier_unreachable.py`); a trainer
  split across functions is outside that detector's shape. While the closure lane — an unmasked
  predicate in `dormancy_gate` plus a wider hazard shape in that test — remains unbuilt, **nothing
  detects a flip on the UNGOVERNED tiers**, where most autograd trainers live: a wiring edit can
  move a trainer into scope with no gate, test or report firing. No measurement says that risk is
  tolerable; the standing trigger for building the closure is the next promotion of a
  hazard-holding `reasoning/`/`training/` module.

  **Provenance of `0.5`.** The null is a within-matrix permutation of the candidate gradient,
  cosined against the unchanged reference; on the stdlib anchor the bar sits inside the separating
  region (P1 case (a): a fixed point on a normalized dimensionless output). **Two disclosures:** the
  reference recomputed against shuffled labels can score ABOVE the bar — the null chosen decides
  whether `0.5` discriminates at all; and the permutation is a null only near zero mean — once the
  coherence product reaches the bar, the statistic INVERTS (see INSTRUMENT_FALSIFIED above). The
  bar is UNCHANGED.

  **THE DISTINGUISHING TEST IS WHAT THE TRANSPOSE FEEDS.** A transposed forward weight feeding a
  **weight update** is credit assignment through a transported weight — exactly what this invariant
  forbids. A transposed weight feeding an **activation** (settling a transient latent state) is
  inference, with the weight update itself a local rule — not a violation, however symmetric the
  read. This test decides every verdict below.

  **THE ENGINE'S TRANSPORT IS NOT THIS INVARIANT'S TRANSPORT (owner permission 2026-09-25, P4d; the
  distinction a disclosed session call).**
  Copying the brain's weight values into the reservoir, or driving it with the brain's activity, feeds
  an ACTIVATION and is permitted. A measured null never deletes an owner-drawn part (the two networks, the reservoir, the return path): its implementation is reverted, it is not claimed, and the owner decides on the evidence; a part of decision C whose deciding test reds (plan G7: T1-T4) is deleted and the owner told, as the brief the owner chose C from states, after the one 10x re-test the P8 null-scale amendment requires (owner, 2026-10-23); transport, permitted "as long as the engine
  can work", lapses only on its own measured failure, the whole engine failing G-R9 (i) or G-R9 (ii)'s
  transport neuter changing nothing. A transposed forward weight
  feeding a WEIGHT UPDATE stays forbidden, including through a transported copy; extending the
  permission to credit assignment would narrow this invariant, which is the owner's call in words.

  **THE STANDING VERDICT REGISTER** — decided sites; the full adjudication narratives are relocated
  byte-exact to `docs/POLICIES_EVIDENCE.md`. A verdict that enumerates is read as a verdict that
  surveys, so this register is explicit that it covers ONLY the sites it lists:

  - `faculties/jepa_world_model.py` — WAS a violation on the live tick; **CLOSED 2026-08-04**
    (one backward definition; the broadcast error half alone). The policy-prescribed remedy
    (feedback alignment) was built first and **REFUTED by its own null** — the entire gain was the
    removal of the transport term. A policy prescribing a fix does not make the fix correct; the
    measurement outranks the prescription.
  - `growth/ng_snn.py::fit` — **NOT a violation.** A direct solve: no gradient propagated, no
    weight transported. It IS non-local — "non-local" and "backprop" are different claims, and only
    the second is forbidden.
  - `faculties/mlp_q.py::MlpQ.update` — transport **RETIRED 2026-08-14**
    (specs/015): `dh` reads a separate feedback matrix `B` receiving the same increment as the
    forward read-out (Kolen-Pollack), so `B` TRACKS `W2` without ever reading it. Neither of the two
    resolutions the module had proposed was needed.
  - `readout/predictive_coding.py` — **ADJUDICATED 2026-08-14, owner: NOT a violation.** Its
    transpose feeds an ACTIVATION (inference); its weight update is a local Hebbian outer product —
    no transposed weight appears in the learning rule at all. **OPEN SCOPE, distinct and recorded:**
    the transpose IS exact weight symmetry on every wake tick; if this invariant is ever extended
    from learning rules to inference pathways — an owner decision — this site is the first thing it
    catches.
  - `reasoning/gf.py::MLP.train` — **ADJUDICATED 2026-08-15, owner: real transport, NOT a
    violation — it is a CONTROL** that exists to LOSE, and porting it to e-prop would VOID the
    experiment (P8's contaminated control). The exemption lives in its own
    `GRADIENT_CLAIM["MLP.train"]` declaration and **converts to a governed violation the day
    `gf.py` is promoted**. Disclosed: declaration-carried exemptions DECENTRALIZE the channel — a
    stale grant is caught by the gate, a grant blessing a REAL finding only by review; the owner
    grant remains the rule. The bar for a future entry: a CONTROL is the instrument this
    invariant's own verification clause requires; **a faculty is not, however far off the tick it
    sits** — `mlp_q` was one constructor argument from live and its transport was RETIRED, not
    exempted.
  - `training/offload` (`train_with_offload`, file deleted) — BPTT with gradient checkpointing and a CPU-resident optimizer,
    **RETIRED 2026-10-02** (owner: port-then-retire): ungoverned and never on the live tick; its learning
    capability is `trainer`'s readout, which the core's LOCAL rule already covers (its entry below); its memory
    technique has no e-prop analogue.
  - `training/mnist` (`MnistBrain`, `train_model`, file deleted) — BPTT trainer of `trained_brain`'s 784-H-10 MNIST
    network, **RETIRED 2026-10-02** (owner: port-then-retire). Its e-prop port (Kolen-Pollack feedback, hidden
    cosine to BPTT 0.993) was NOT landed: it lost the pre-registered A/B, 0.942 against 0.957 (S = 8, t = 9.79).
    Capability lost: the validated MNIST trainer for `trained_brain`; the kept `mnist_genn` scored chance in a one-epoch run.
  - `training/spiking_lm` (`SpikingLM`, `train`, file deleted) with its serve core `sensorimotor/spiking_lm_serve` —
    a char-level spiking LM trained by BPTT, **RETIRED 2026-10-02** (owner: port-then-retire): ungoverned, never on
    the live tick; at its defaults it scored in-sample perplexity 14.4-15.5 (S = 3) against a unigram 17.1 and a
    bigram 7.2 on its own 512-character corpus. Its governed e-prop re-implementation, which the owner's decision
    asks for, is OWED, not done (the root e-prop char-LM is ungoverned).
  - `training/trainer` (`SpikingReadout`, `train_model`, file deleted) — BPTT trainer of the core 8x8 readout,
    **RETIRED 2026-10-02** (owner: port-then-retire): nothing unique to port, since the core's LOCAL rule
    (`readout/snn_classifier`) beat it on identical data, 0.998 against 0.714 (S = 8). Its parity fixture moved
    into `growth/brian2_backend` byte-identically.

- **Verify-first + honest framing:** probe → act → prove (A/B / gate) → commit; report the true number
  including a regression; reject theater with evidence; never fake a capability, a metric, or a consumer.
- **Runtime evidence outranks static reading.** Every decisive finding in the 2026-07 audits came from
  *running* the system, and the static reads were wrong in both directions: a runtime monkeypatch
  proved no backprop on the tick where grep could only find call sites; neuter-A/Bs proved signals
  decisive and controls proved learning real; and a profiler ranking sent a whole optimisation plan
  after the wrong function (per-call overhead attributed as time — the neutered "hot path" was a 6%
  ceiling). **Before optimising a hotspot, neuter it and measure the delta; before believing a signal
  is live, neuter it and watch what changes.**
- **A performance regression can be correctness-green.** A ~1 s census inside a snapshot polled at
  1 Hz starved the tick loop it existed to display, with every suite green throughout — the cost only
  bites at rate against a live loop. **Anything that observes the brain must be cheap enough not to
  become the load** — and no gate here can tell you that; it is found by watching the running system.
- **Adding a policy: implement the gate before adopting the rule.** A proposed mechanization must be
  built and run against this tree, and its output counted, *before* the policy lands. Two proposals
  died exactly there — one could not catch its own motivating case, the other produced a
  rubber-stamp-sized allowlist. Also required: quote the source line the rule derives from (a general
  good-practice rule in a costume is a reject), and name the real past incident it would have caught,
  or state plainly that there is none and say what it earns its place on instead. Fewer, sharper
  policies win; **"no new policy" is a complete result.**
  **Amended 2026-07-29: the owner may adopt a policy with its gate UNBUILT.** Such a policy lands
  marked **provisional** in the enforcement table, must still meet every other requirement above,
  and must carry the gate's specification so the debt is actionable rather than perpetual. The
  gate-first rule still binds anyone proposing a *mechanization* — "the gate we built kills the
  rule" remains the point of building it first. "Provisional" is a debt, not a category: a count
  worth watching rather than a precedent worth spending.
- **The citation-guard outcome, kept as precedent for that bar:** the broad rule ("a tracked file may
  not cite an untracked `.py`") was built, counted, and **REJECTED on its own output**
  (allowlist-shaped); the narrow rule (an untracked script presented as a *runnable reproduction
  path* — STRICT scope) was built, counted, **ADOPTED and ASSERTING** (`tools/citation_guard.py`,
  lint CI job), with its exemptions themselves asserted against real violations so an exemption that
  blesses nothing fails loudly. The counted history is in `docs/POLICIES_EVIDENCE.md`. Building the
  naive version first is what distinguished the two, and it cost one file instead of a policy.
- **Gated + reversible:** one change per commit, full gate green (`run_selfchecks` + `pytest` + `ruff`),
  commit + tag locally; a regression is isolated to its own change.

  **AMENDED 2026-08-16, owner: the local full-suite requirement is DELEGABLE TO CI — but only while
  CI is DEMONSTRABLY EXECUTING, and that is a measurement, not an assumption.** Adopted from
  DeepSeek Harness's *"never default to the full suite locally; CI owns exhaustive coverage and the
  platform matrix"* — quoted per this file's own adding-a-policy bar. The precondition was
  FALSE on the day this was adopted (a billing block had CI starting zero jobs while reporting
  `completed / failure`). **How to check it, because a status badge cannot tell you:** a
  billing-blocked run reports failure exactly like a real one; the distinguishing signal is DURATION
  and JOB COUNT — `gh run list` for the duration, `gh run view <id>` for jobs that actually ran,
  never the pass/fail word alone. **What does NOT change:** the full local gate is still required
  whenever CI is not demonstrably green on the change — before a merge to `main`, on any change to
  the safety envelope, and any time the CI/local torch-build skew could bite. "CI will catch it" is
  a claim about a run that has EXECUTED, the same way *"it persists"* is a claim about a new process
  and *"it is pushed"* is a claim about a branch.

  **Mechanized 2026-09-02 (owner decision O3, specs/022 plan of record):** `policy_manifest --certify`
  executes the policy gate commands and writes the tree it ran on — commit and dirty flag — to
  `tools/policy_certification.json`; `policy_manifest --assert` (also bare, in `run_selfchecks`) refuses a
  record that is not certified, was written on a dirty tree, or names none of the accepted heads (the
  batched-battery amendment below); only ENABLED `ci.yml` lines count as CI-reachable, and the census
  counts the `tools/` gates `run_selfchecks` runs. A job disabled with `if: false` is not a comment and the
  line filter does not see it — disclosed, not handled.

  **AMENDED 2026-09-02, owner: THE FULL GATE IS PER PUSHED ARC, NOT PER COMMIT** — the instruction,
  verbatim: stack an arc's commits and run one battery for all of them, and record the change here. Under
  it, one change per commit stays and the full gate runs once, on the stacked head of an arc, before that
  head is pushed. Practice adopted with the rule (the session's, recorded in specs/022's plan of record, not
  owner text): every commit names the battery that verified the head it was pushed under, so an unverified
  intermediate commit is disclosed rather than implied green; a red on the stacked head is attributed by
  re-running the failing STAGE per commit, never whole batteries, and an unattributable red does not push;
  the review panel is per arc. The gate for the rule is the execution record: the battery opens with
  `policy_manifest --assert` — the previous record must have been written on a clean tree and must name
  the pushed head or its parent, or the stacked head or its parent — and HALTS on its red; then
  `--certify` writes the record for the head under test; a green battery commits that record as the arc's
  last commit and pushes with it. A shallow checkout cannot resolve the parent refs; the assert says so
  instead of failing.

  **AMENDED 2026-10-02, owner: A DOCS-ONLY ARC MAY PUSH ON THE DOCS-ONLY BATTERY** ("add the docs-only
  battery rule"; the owner chose a runtime reader map). For a diff against `origin/main` touching only `*.md`,
  `python -m snn_data_lake.gates.docs_only_battery --run <out>` replaces the full battery: it refuses any other
  diff or a map not measured on this code, runs every other stage in full, and runs pytest only on the files
  the map shows reading the changed docs. Every full battery rebuilds the map. The gate and its count:
  `specs/022-unified-brain-arc/evidence/docs-only-battery.md`.
- **A `file.py:NNN` citation is a claim, and it is the least durable claim this project makes.** The
  docs-wide audit that measured it (thousands resolved; the worst misses in this file's own most
  safety-relevant paragraphs; a file whose line numbers moved *between two reads minutes apart*) is
  in `docs/POLICIES_EVIDENCE.md`. **The disposition is NOT another gate** — a line-resolution gate
  reports allowlist-shaped findings on historical records, the shape that killed the broad citation
  rule. **The rule is editorial: cite the SYMBOL, not the line.**
  `neurogenesis_engine.substrate_is_retained()` survives a refactor; a `:NNN` survives until the
  next insertion above it. Where a line is genuinely needed, expect it to rot and re-derive it
  rather than copy it. **A line citation into an actively-developed file is stale on a timescale of
  minutes, not releases.**
