# Tasks: Order neurogenesis by cortical architecture, from a prior that must be beatable

**Feature**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md)

## How to read this file

Phases are **A / B / C / D from plan.md**, and their order is **forced by dependency, not chosen**.
Every task names the FR or SC it closes and the command that proves it.

**Every task must leave all gates green before it is committed** (one change per commit):

```bash
.venv-lake/Scripts/python.exe -m ruff check . && .venv-lake/Scripts/python.exe run_selfchecks.py
```

```bash
.venv-lake/Scripts/python.exe -m pytest -q
```

```bash
.venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert && .venv-lake/Scripts/python.exe tools/p6_gate.py --assert && .venv-lake/Scripts/python.exe tools/dependency_ledger.py --assert && .venv-lake/Scripts/python.exe tools/red_reachability.py --assert && .venv-lake/Scripts/python.exe tools/citation_guard.py && .venv-lake/Scripts/python.exe tools/doc_command_guard.py --assert
```

Baselines to hold: **280/280** self-checks, **≥70%** coverage (currently 93.05%), **248** ungoverned
builds (may fall, never rise).

**[P] means genuinely parallel** — different files, no shared state. `cortex/brain.py`,
`src/snn_data_lake/__init__.py` and `run_selfchecks.py` are **shared config**: no task touching any of
them is ever [P] with another that does. This project has been bitten by concurrent lanes on exactly
those files.

**REPORT-ONLY, never modified by any task here**: `brainstem/reflex.py`, `serial_arm.py`,
`so101_arm.py`, `arm_config.py`, `interpreter.py`.

**Delays are out of scope in every phase.** No task may introduce one. Seven independent readers found
zero conduction velocity in the source; a delay derived from it would be literature-sourced, not
H01-derived, and would carry the authority of a measurement it does not have.

---

## Phase 0: BLOCKING PRE-FLIGHT — re-check this plan against POLICIES.md before writing any code

**Do not start T001 until T000 passes.** This spec was written 2026-08-06; `docs/POLICIES.md` and
`CLAUDE.md` are living documents and the tree moves under them. Every measured figure below is a
claim about a tree that has since changed, and this project's own rule is **re-derive, never quote**.

- [ ] T000 Re-read `docs/POLICIES.md` and `CLAUDE.md` IN FULL, then re-derive every gate baseline and
  confirm each of the following against the CURRENT tree, recording the result in
  `specs/011-h01-neurogenesis-prior/preflight.md`. **Any mismatch means this plan is re-opened, not
  worked around.**
  - **P0** — is "cortical column construction" still listed as OPEN? If Phase A already shipped in
    some other form, US4 is done and must not be rebuilt.
  - **P1** — the movability mechanism (prior read ONCE at construction, never on a live path) is the
    load-bearing claim of this whole feature. Confirm nothing has introduced a live-path re-read.
  - **P2** — confirm `cortex/cortical_microcircuit.py` is still the ONLY layered-population
    mechanism, and that none of the ten existing gates (dormancy_gate, method_dormancy_gate,
    import_manifest, replication_guard, citation_guard, red_reachability, dependency_ledger,
    doc_command_guard, p6_gate, p4c_ratchet) already does what this feature proposes to add.
  - **P3** — confirm the neuter-A/B (SC-006) is still the bar, and that placement-that-changes-no-
    outcome would still be reported as inert.
  - **P4b** — confirm the reference-data-vs-learned-state split still holds: the prior lives with the
    code, grown state goes to the ONE lake, and "it persists" still means a NEW process.
  - **P4c** — **re-run `tools/p4c_ratchet.py` and record the CURRENT count.** The spec assumes 248
    total / 150 in `reasoning/`. If it has fallen, T015's registration requirement is unchanged but
    the baseline in this file is stale; if it has RISEN, something regressed and that is a finding.
  - **P5** — confirm the suite still runs without a GPU (CI has no card) and that nothing here needs
    one.
  - **P7** — confirm no dependency is needed. If one now WOULD help, say what it buys; do not carry a
    "no new dependency" certification either way.
  - **P8** — **re-check every one of the eleven success criteria for a reachable red**, and
    specifically re-check that none of them measures a defect and scores it as the feature. This tree
    caught **seven** instances of that shape in one session, including one written by the author of
    this spec. SC-005 in particular must still fail in BOTH directions.
  - **The five brainstem safety files** are still report-only, and still the same five.
  - **`.venv-lake`** is still the one gate env and still free-threaded with CUDA.

- [ ] T000b Re-derive the four H01 figures this spec leans on, against the source files rather than
  against this document: the 100.0% join coverage on `agglo_seg`, the seven-value `cortical layer`
  column, layer 1's zero excitatory somata, and the E/I column being INCOMING SYNAPSES rather than
  cell counts. **A figure quoted from a spec is a figure nobody has checked.**

---

## Phase 1: Setup

- [ ] T001 Verify the environment and record the pre-feature baseline in `specs/011-h01-neurogenesis-prior/baseline.md`: interpreter version and `Py_GIL_DISABLED`, the four gate numbers (self-check count, coverage %, ungoverned count total and per zone, p6_gate census), so every later claim of "unchanged" has something to be unchanged against.

- [ ] T002 [P] Fetch the four source files to a local working directory **outside the repo** and record each one's byte count and SHA-256 in `specs/011-h01-neurogenesis-prior/baseline.md`. Files: `agglo_20200916c3_cell_data.json` (11.2 MB), `goog14r0s5c3_e_i_ratios_feb_2023.csv` (2.0 MB), `excitatory_axons_summary_df.csv`, `inhibitory_axons_summary_df.csv`, `cortical_bounds_circles.json` (496 B). **They are not vendored** — only derived constants enter the repo.

---

## Phase 2: Foundational — the three decisions that gate everything

These are **decisions, not implementations**. Each is carried from plan.md as explicitly unresolved,
and leaving any to an implementer mid-task is how a criterion silently becomes vacuous. Each records
its answer and rationale in `specs/011-h01-neurogenesis-prior/decisions.md`.

- [X] T003 **DECIDE which task the three-arm comparison is scored on**, and record it with the measurement supporting the choice in `specs/011-h01-neurogenesis-prior/decisions.md`. **DATED NOTE 2026-09-09 (specs/022 T063): DECIDED in `decisions.md` — K-way drive-class discrimination read from the NON-DRIVEN units with a least-squares readout, on the T062 kernel population; four candidates rejected with the side each failed on. The bar's SUBJECT moved from placement to the synapse graph because placement is a pinned null (`tests/test_placement_charge_null.py`), which is disclosed in the row. The required drive-only measurement is recorded there at pilot time, not here.** **This gates every Phase B task.** Acceptance: the chosen task must be one that placement could *plausibly* affect but that is *not defined in terms of* placement. A task placement cannot affect makes SC-005 vacuous in the always-passes direction; a task only placement affects makes it circular. State which candidate tasks were rejected and on which side they failed.

- [X] T004 **DECIDE what "a deliberately wrong prior" concretely means** **DATED NOTE 2026-09-09 (specs/022 T063): DECIDED in `decisions.md` — the DIRECTION-REVERSED LAMINAR TABLE (inter-band edges reversed, intra-band unchanged, polarity re-applied from the new source, fan-in re-normalised), with a VALIDITY PRECONDITION measured before the campaign; the three named candidates are rejected there with what each would have falsified.** — shuffled layer labels, inverted E/I, or uniform-across-layers — and record it in `specs/011-h01-neurogenesis-prior/decisions.md`. Acceptance: state what each candidate would falsify and why the chosen one falsifies the thing SC-005's second red is *for*. Closes the design half of **SC-005 RED 2**.

- [ ] T005 **DECIDE whether grown positions are drawn from measured somata or sampled from a fitted density**, recorded in `specs/011-h01-neurogenesis-prior/decisions.md`. Acceptance: drawing reuses real coordinates and **cannot generalise past the measured volume**; sampling needs a density model, which plan.md decision D1's reasoning then applies to (the source's absolute counts are irreproducible at ±10–20% while fractions hold to ±0.015). Whichever is chosen, state how FR-005's out-of-volume rule follows from it.

---

## Phase 3 (Plan Phase A): The layered circuit gets an input port — US4

**Goal**: the layered cortical circuit stops being an isolated simulation and becomes something the
rest of the brain can drive.

**Independent test**: two distinguishable inputs produce different responses; the same input twice
produces identical responses.

**Ships alone as**: a drivable layered circuit. This is P0/P6 work needed whether or not H01 is ever
adopted, and it removes the reason the module is EXEMPT-permanent in the dormancy gate.

- [ ] T006 [US4] Add a **PATTERNED** input and a response readout to `src/snn_data_lake/cortex/cortical_microcircuit.py`. Closes **FR-015**. The circuit is **extended, never duplicated** (FR-016) — do not create a parallel layered class.

  **RESCOPED 2026-08-16 (specs/019 `[B2.02]`), because this task was written against a false premise and would have rebuilt something that exists.** It said the module accepts *"a scalar background rate and **no signal**"*. Measured at HEAD: `thalamic_drive(thalamic_hz)` is a REAL input port added 2026-08-04 — its docstring calls it *"THE INPUT PORT… the signal a caller injects"* — it is **anatomically targeted** (`CONN_PROB_TH` nonzero for L4e/L4i/L6e/L6i ONLY, so a caller cannot inject into L2/3 or L5), it contributes **exactly 0.0 at 0 Hz** so an undriven run stays bit-identical, and it **refuses a negative rate**. It is also NOT the background rate: `bg_rate` is a separate constructor argument and a separate Poisson bank.
  **What is genuinely missing is a PATTERN, not a port.** The existing port carries ONE SCALAR, so a caller can modulate drive INTENSITY across the whole thalamic projection but cannot address populations differentially. That — a per-target drive vector plus a readout — is this task's real content.
  **T008's RED must be rescoped with it:** "remove the input port" no longer constructs a red for a port that predates this phase. The red is *remove the PATTERN* (collapse the per-target drive to one scalar) and confirm two distinguishable inputs stop being distinguishable.
  **T009 is unaffected and still correct**: the exemption reason did say "no input port" and that clause is now retired — see `substrate/dormancy_gate.py`'s corrected entry. Note the exemption itself stays until a live driver exists (specs/019 `[B2.08]`), because the honest reason is that NOTHING DRIVES IT, not that the port is absent.

- [ ] T007 [US4] Add the two-input criterion to `cortical_microcircuit.py`'s self-check: two distinguishable inputs produce different responses, one input twice produces identical responses. Closes **SC-007**. Gate: `.venv-lake/Scripts/python.exe -m snn_data_lake.cortex.cortical_microcircuit`.

- [ ] T008 [US4] **CONSTRUCT AND RUN SC-007's RED**: remove the input port and confirm the first half of T007's criterion FAILS. Record the failure message in the module. **If the criterion still passes with the port removed, it is measuring something other than the port** and must be rewritten before this phase closes.

- [ ] T009 [US4] Remove `cortical_microcircuit` from `substrate/dormancy_gate.py`'s EXEMPT list (its exemption reason — "it has no input port" — no longer holds) and confirm the dormancy gate stays green. **NOT [P]**: shared gate config. Gate: `.venv-lake/Scripts/python.exe -m snn_data_lake.substrate.dormancy_gate`.

**Checkpoint**: Phase A is independently shippable here.

---

## Phase 4 (Plan Phase B): The drift instrument — US3, first half

**Goal**: build the instrument that can tell whether *any* future seeding earns its place, and validate
it against the **existing unseeded** growth.

**This ordering must not be reversed.** An instrument validated only against the thing it was built to
bless is not an instrument. It measures nothing interesting yet, and that is the point.

**Independent test**: the three-arm comparison distinguishes seeded / random / wrong-prior arms.

- [ ] T010 [US3] Create the `DriftRecord` — per statistic: seeded value, current value, distance, direction — in `src/snn_data_lake/cortex/` per [data-model.md §4](./data-model.md). Closes **FR-008**. **Sign is excluded from drift by design** (Dale is a structural invariant, not a tunable); including it would score the invariant as a hardcode. *DATED NOTE 2026-09-09 (specs/022 T063): **the STRUCTURE is BUILT and FR-008 is NOT CLOSED.** `cortex/prior_drift_harness.py::DriftRecord` carries this data-model's five fields verbatim (`_fields == ("statistic", "seeded", "current", "distance", "direction")`, pinned by a test), one record is produced for EVERY arm (C5), sign never enters either statistic, and the harness ran the whole machinery over 5 tables x 8 seeds. But the STATISTICS are newly chosen graph-weight quantities (band-pair weight share, band fan-in E fraction), NOT this spec's placement-side ones (layer occupancy, layer geometry, the excitatory CELL fraction), which are a pinned null on the object brain (`tests/test_placement_charge_null.py`) and which T063 explicitly did not measure. And since 2026-09-09 the harness has NO LEARNER -- `substrate/kernel_plasticity` was deleted on its own criterion (t(d_f) = -13.0154) -- so every distance it records is a measured zero and drift is not measurable at all. FR-008 stays OPEN.*

- [ ] T011 [US3] Implement the three-arm comparison harness (seeded / random control / deliberately-wrong-prior, per T004) against the **existing unseeded** growth, scored on the task chosen in T003. Closes **FR-010**. Requires no GPU and no live brain beyond the population under test (contract C5). *DATED NOTE 2026-09-09 (specs/022 T063): **BUILT AND RUN; FR-010's three arms exist and were scored.** `cortex/prior_drift_harness.py` carries SEEDED, RANDOM (a WITHIN-SIGN degree-preserving rewire) and WRONG (T004 re-decided by amendment 2.12 as the BAND-PERMUTED-TARGETS table), plus two arms this spec did not ask for and the row needed: PLANTED (the positive control -- without it a null reading cannot be told from a blind instrument) and FROZEN. Every arm is refused before it runs if it violates a nuisance property. Scored on the T003 task at K = 8, D = 4, W = 1.0 over 8 seeds (`tools/t063_campaign.json`). What it did NOT compare against is this row's "**existing unseeded** growth": the arms are five WIRINGS of one unit set, all built from the H01 artifact, and no growth-produced population entered the comparison. FR-010's arm structure is satisfied; its comparison-against-growth clause is not, and is recorded here rather than assumed closed.*

- [ ] T012 [US3] Make zero drift report as a **failure of the seeding approach**, not as stability. Closes **FR-009**. The message must name pinning as the cause. *DATED NOTE 2026-09-09 (specs/022 T063): **the predicate is BUILT and it RAN; FR-009 is NOT CLOSED.** `is_seeding_failure()` reads zero distance across ALL statistics as a failure of the seeding approach, and on the campaign the message named the pin: "seeding approach failed: PINNED (lr = 0 by construction): executed 120, attempted 179,571, effective 0, drift 0", on every one of the 8 seeds. A CORRECTION THIS ROW OWES, from T063's Codex read (defect 4): "the message must name pinning as the cause" is only safe for a VERIFIED lr = 0 intervention -- a zero can equally come from cancellation, from a below-quantization update or from absent spike pairs, and naming pinning for any of those is a false attribution. The harness therefore attributed pinning ONLY to its own constructed pin. Since the learner's deletion (2026-09-09) there is no intervention to verify and no learning arm to contrast, so that slot reads "harness defect: no learner ran" and the word PINNED is refused for every arm -- which is 2.9's own clause ("if effective change is 0 on every arm the learner is DELETED and red 1's reading is a harness defect, not a finding"). FR-009 stays OPEN: its subject is a seeded population that a LEARNER cannot move, and there is no learner.*

- [ ] T013 [US3] **CONSTRUCT AND RUN SC-005's RED 1**: pin the seeded values so learning cannot move them. Drift reads zero and the criterion must FAIL, naming pinning. Record the failure message. *DATED NOTE 2026-09-09 (specs/022 T063): **CONSTRUCTED AND RUN, and the message is recorded** -- the FROZEN arm at `lr = 0`, on all 8 campaign seeds: "seeding approach failed: PINNED (lr = 0 by construction): executed 120, attempted 179,571, effective 0, drift 0" (`tools/t063_campaign.json`, key `red1_messages`). ITS CONTROL HELD, which is what makes it a red rather than a coincidence: the SEEDED arm at the declared lr read effective change 528,383 entries / L1 231,865.22 and BOTH statistics above zero on every seed (S1 distance 0.107-0.172). **The red is now UNREACHABLE and the row cannot be closed by re-running it.** Its subject was a pin standing against a LEARNING arm; the learner was deleted on its own criterion 2026-09-09, so there is no intervention to verify and every arm's zero has one cause. The slot reads the no-learner refusal instead, with the control that the message never says PINNED.*

- [ ] T014 [US3] **CONSTRUCT AND RUN SC-005's RED 2**: seed from the deliberately incorrect prior defined in T004. The seeded arm underperforms the random control and the criterion must FAIL, naming the prior. **Both reds are mandatory — a criterion that can only fail one way is not measuring what this feature claims.** *DATED NOTE 2026-09-09 (specs/022 T063): **CONSTRUCTED AND RUN, and it DID NOT FIRE — for a reason worth more than the red was.** T004's prior was re-decided twice: the DIRECTION-REVERSED laminar table failed its validity precondition on this connectome by 70x (PSI 0.032389 against RANDOM's 2.281271, because 0.726520 of the band-pair mass is DIAGONAL and a transpose cannot move it), and its replacement, the BAND-PERMUTED-TARGETS table, was VALID on all 8 seeds (PSI 6.175870 against 2.271100-2.302899). Red 2's predicate is that WRONG must UNDERPERFORM the random control: t(random - wrong) = **-2.5459** (FPR 0.0391). The sign is inverted -- **the deliberately wrong prior OUTPERFORMED both the random control and the true prior** (wrong 0.25-0.50, random 0.00-0.375, seeded 0.00-0.375) -- so underperformance is not merely unestablished, its opposite is what was measured. Per T063 2.5 nothing is attributed to the prior from it. The red's own reachability was shown separately on a constructed input, so "it did not fire" is a reading of the data and not of a dead predicate.*

**Checkpoint**: A + B ships as a drivable circuit plus an instrument that can judge any future seeding.

---

## Phase 5 (Plan Phase C): Placement, sign and the beatable seed — US1 + US2 + US3 second half

**Goal**: the feature's core result — placed, signed, drift-measured neurogenesis.

**Independent test**: every grown unit carries a layer and a position; every cell's outgoing
connections carry one sign; drift is non-zero and the seeded arm holds against the control.

### The constants module

- [ ] T015 [US1] Create `src/snn_data_lake/cortex/cortical_prior.py` **and register it in `src/snn_data_lake/__init__.py` REGIONS and `run_selfchecks.py` in this same task**. Creating it unregistered reds both `tools/p4c_ratchet.py --assert` and the dormancy gate. **P4c is now mechanized and this is the first feature planned after that** — the build lands inside the governed tree from the start. **NOT [P]**: shared config. Closes **FR-006**.

- [ ] T016 [US1] Populate `cortical_prior.py` with the derived constants, **each carrying value + units + provenance + accuracy bound**; a constant missing any one is a **construction error, not a warning** (contract C1). Contents per [data-model.md §3](./data-model.md): `LAYER_CIRCLES` (six circles, **microns**), `LAYER_E_FRACTION` (**fractions, not counts** — plan D1), `SIGNED_TYPES`/`UNSIGNED_TYPES`, `GP_STRENGTH_E`/`GP_STRENGTH_I`, `SOURCE_VOLUME_NM`. **`fnr_i = 0.35` and the 1.70:1-to-2.04:1 operator-precedence ambiguity must travel with every E/I number.** Module is pure data + pure functions: no I/O on import, no module-scope side effects.

- [ ] T017 [US1] Add `cortical_prior.py`'s self-check: four-attribute completeness for every constant, plus degenerate-input handling (empty layer, single-member layer, fraction exactly 0.0 and 1.0, a layer absent from the source). Closes **SC-008**. **CONSTRUCT AND RUN ITS RED**: strip the accuracy bound from one constant — the criterion must fail and name it.

### Placement

- [ ] T018 [US1] Implement layer assignment per contract C2 in `cortex/cortical_prior.py`: first-match-wins in the geometry's own order, **depth RADIAL in the x-y plane** (`z` is never consulted — it spans the full range in every layer), out-of-volume units flagged `extrapolated` and counted. Closes **FR-004**, **FR-005**. Use `true_x/true_y/true_z` in **nanometres**, **never the `_vx` fields** — the cell and synapse grids differ by exactly 2×, and a 2× error survives a smell test a 1000× error does not.

- [ ] T019 [US1] Add the **scale assertion** to layer assignment: if the position scale is inconsistent with the geometry scale, the call **fails loudly**. It must not return the fall-through layer. Closes **FR-011**.

- [ ] T020 [US1] **CONSTRUCT AND RUN SC-002's RED — this one is mandatory and is the whole reason SC-002 exists.** Supply the boundary geometry in nanometres where microns are expected. **Every** unit lands in the fall-through layer, which **is not an error state**, so without this red a units error ships looking like a valid assignment. The criterion must FAIL. Record the message.

- [ ] T021 [US1] Wire placement into `src/snn_data_lake/cortex/neurogenesis_engine.py` so every grown unit gets position and layer **at creation**, not by a later annotation pass. Closes **FR-001**. Add the growth report: units grown, per-layer distribution, count placed outside the measured volume, fraction in the fall-through layer (contract C4).

- [ ] T022 [US1] Add the placement criterion to `neurogenesis_engine.py`'s self-check: every unit carries a layer and position, the distribution spans more than one layer, the fall-through layer holds no more than the stated fraction. Closes **SC-001**. **RED**: grow with layer assignment disabled — units carry no layer, criterion fails.

### Sign

- [ ] T023 [US2] Implement sign assignment per contract C3 in `cortex/cortical_prior.py`: sign derived **only** from cell type, an unsigned type returns **unsigned** (never a default to excitatory), sign **immutable** after creation. Closes **FR-002**, **FR-003**. Excitatory = {pyramidal neuron, excitatory/spiny neuron with atypical tree, spiny stellate neuron}; inhibitory = {interneuron}; the other seven types are unsigned.

- [ ] T024 [US2] Make `layer == L1` + excitatory **unrepresentable**, not merely unlikely, in `neurogenesis_engine.py`. **The source measures zero excitatory somata and 540 interneurons in layer 1** — the circuit's published parameterisation ("layer 1 has no cell bodies") is half wrong and we follow the data.

- [ ] T025 [US2] Add the sign criteria to `neurogenesis_engine.py`'s self-check: every cell's outgoing connections carry one sign (**SC-003**), and after a learning run no cell changed sign while magnitudes did (**SC-004**). **SC-004 must ALSO fail on a run where no magnitude changed**, so it cannot pass vacuously on a brain that never learned. **REDS**: a cell with mixed outgoing signs (SC-003); a sign flip during learning, and separately a no-magnitude-change run (SC-004).

### The circuit mapping

- [ ] T026 [US2] Map the source's seven regions onto the circuit's populations in `cortex/cortical_microcircuit.py` per plan decision D2: Layer 2 + Layer 3 → L2/3 (union); L4, L5, L6 direct; **a NEW inhibitory-only L1 population** (the deliberate departure); **white matter EXCLUDED and its exclusion COUNTED, not silent**. Closes **FR-016**. **NOT [P]** with T006–T009.

- [ ] T027 [US2] Add the single-mechanism criterion: exactly one layered-population mechanism exists in the tree. Closes **SC-010**. **RED**: add a second — the criterion fails.

### Persistence

- [ ] T028 [US1] Add a record tag for grown placement state to `src/snn_data_lake/substrate/home.py` and persist `PlacementState` (positions, layers, signs) to the **one** lake. Foreign records passed through, never dropped. Closes **FR-006** persistence half, contract C7. **The prior does NOT persist** — it is reference data versioned with the code, and writing it to the lake would make the lake a second copy of something the repo already owns (P4b).

- [ ] T029 [US1] **PROVE persistence across a PROCESS BOUNDARY**: write in one process, exit, read back in a **new** one. Add it to the self-check. **An in-memory round-trip must NOT count as passing** — "it persists" is a claim about a new process. **RED**: an in-memory-only round-trip fails the criterion.

### The seed proves itself

- [ ] T030 [US3] Seed the grown population from `cortical_prior` at construction — **read once, into mutable state, never re-read on a live path**. A value re-read every tick is pinned by construction. Closes **FR-008** (the P1 movability mechanism from plan.md).

- [ ] T031 [US3] Point the Phase B drift instrument at the now-seeded population and run the three-arm comparison. Closes **SC-005**. Both reds from T013 and T014 must still fire. **Acceptance: drift non-zero AND the seeded arm at least matches the random control.** A seeded arm that drifts but underperforms fails. *DATED NOTE 2026-09-09 (specs/022 T063): **RUN, and this row's own acceptance clause is the one that decides it — the seeded arm DRIFTED BUT UNDERPERFORMED, which this row says FAILS.** Campaign `tools/t063_campaign.json`, 5 tables x 8 seeds at K = 8, D = 4, W = 1.0. Drift non-zero: YES on every seed (S1 distance 0.107-0.172, effective change 528,383 entries). At least matches the random control: NO, t(seeded - random) = **-0.2277**. Red 1 fired; red 2 did NOT (the wrong prior outperformed the control, T014's note). SC-005 is therefore recorded NOT MET on the reading, with the instrument PROVEN NOT BLIND (PLANTED t = +9.2601) so the negative is about the arms and not the harness. **SCOPE, and it is narrower than "SC-005 closed":** the random control retains H01's degrees, signs and weight magnitudes, so this reads only wiring information BEYOND those marginals, on T063's task, at that width; and the statistics that drifted are graph-weight quantities, not this spec's placement-side ones. Since the learner's deletion the same run is no longer repeatable -- drift is not measurable without a learner -- so this row is NOT closed and is not re-runnable as written.*

- [ ] T032 [US1] **THE P3 NEUTER-A/B**: pin positions and layers to a single value and show the brain's **own output** differs from the placed run on a majority of ticks. Closes **SC-006**. **RED**: identical outputs mean placement is inert and the criterion fails. **A census of correctly placed neurons is not evidence** — this task is what separates this feature from dormancy with better provenance.

- [ ] T033 [US1] Add the determinism criterion: two runs from the same seed produce byte-identical placements, signs and seeded values, over **≥20 runs hashing floats**. Closes **SC-009**. **RED**: introduce a placement decision that depends on iteration order.

**Checkpoint**: **A + B + C is the feature's core result and is complete on its own terms.**

---

## Phase 6 (Plan Phase D): Connection strengths — US5

**Ships last, and may never ship.** Its bias is the least correctable and its quantity is a structural
proxy for a physiological one. The feature does not depend on it.

- [ ] T034 [US5] Seed initial connection strengths per population from `GP_STRENGTH_E`/`GP_STRENGTH_I` in `cortex/cortical_prior.py`. **Three caveats must travel in provenance**: "GP strength" is a per-AXON MAXIMUM over partners (an upper-tail order statistic), the population is 79.8M axon **fragments** against ~15k somata, and it **excludes every axon synapsing onto an AIS** — removing chandelier cells from exactly the tail where the E-vs-I asymmetry lives. **No fitted distributional form exists; do not invent one.**

- [ ] T035 [US5] Add the inhibitory-tail criterion: the inhibitory population has the heavier multi-connection tail. Closes **SC-011**. **RED**: seed both from one pooled distribution — the criterion fails. (Above N=10, excitatory is (4,4,0,1,0,0,0,0,1,0) against inhibitory (144,82,58,54,28,19,14,11,7,5).)

- [ ] T036 [US5] Record in provenance that **no weight units exist anywhere in the source** — no conductance, millivolts, PSP amplitude or release probability — so any multiplicity→weight conversion needs a per-synapse constant from outside and is therefore **derived, not measured**. Closes **FR-007**.

---

## Phase 7: Polish & cross-cutting

- [ ] T037 [P] Add `tests/test_cortical_prior.py`: units, provenance completeness, accuracy bounds, degenerate inputs.

- [ ] T038 [P] Add `tests/test_placement.py`, including the unit-mismatch red from T020 as a permanent regression test.

- [ ] T039 [P] Add `tests/test_drift.py`, including both SC-005 reds as permanent regression tests. *DATED NOTE 2026-09-09 (specs/022 T063): **the file exists under another name and only ONE of the two reds survives as a permanent regression test.** `tests/test_prior_drift_harness.py` is the file (39 tests, green): the DriftRecord's verbatim fields, `is_seeding_failure` as a failure-not-stability reading, red 2's predicate with the control that it does NOT fire on a mere failure to establish superiority, the seeded-vs-wrong direct contrast, the invalid-wrong-arm refusal, the nuisance refusals, the arms' constructions, the D ladder, the projection, the settling clause and the campaign guard. **RED 1 IS NOT AMONG THEM ANY MORE.** Its subject was a verified `lr = 0` pin standing against a learning arm; the learner was deleted 2026-09-09 on its own criterion, so the pin has nothing to stand against. Its slot is now a NO-LEARNER REFUSAL test with the control that the refusal never says PINNED, plus `no_learner_holds` -- red 1's control inverted and measured per arm. A permanent regression test for red 1 as this row means it requires a learner in the tree, and there is none. Recorded, not softened.*

- [ ] T040 Confirm the ungoverned build count did **not rise**: `tools/p4c_ratchet.py --assert` at 248 or below, with `cortex/cortical_prior.py` counted as governed. If it rose, T015's registration was incomplete.

- [ ] T041 Update `docs/POLICIES.md` and `CLAUDE.md` **if and only if** this feature changed something they assert — specifically P0's "cortical column construction — OPEN, not live" and the P4c note about `cortical_microcircuit` being EXEMPT because it has no input port, both of which Phase A makes false. **Do not touch them otherwise.** Verify with `tools/citation_guard.py`.

- [ ] T042 Record in `specs/011-h01-neurogenesis-prior/decisions.md` what was measured and did **not** work, if anything. A measured null is a result and deleting on a null is the licence working, not failing.

---

## Dependencies

```
Phase 0 (T000, T000b)  ── BLOCKING. Nothing starts until the plan is re-checked
   └─▶ Phase 1 (T001-T002)
   └─▶ Phase 2 decisions (T003-T005)
          ├─▶ T003 gates ALL of Phase B
          ├─▶ T004 gates T014
          └─▶ T005 gates T018, T021
Phase A (T006-T009) ──────────── independent of B/C/D; ships alone
Phase B (T010-T014) ──▶ needs T003, T004
Phase C (T015-T033) ──▶ needs Phase B (the instrument must exist first) + T005
Phase D (T034-T036) ──▶ needs Phase C
Phase 7 (T037-T042) ──▶ needs whatever shipped
```

**Story order**: US4 (Phase A) → US3-first-half (Phase B) → US1 + US2 + US3-second-half (Phase C) →
US5 (Phase D). Note US1/US2/US3 are **co-equal P1** in spec.md and interleave in Phase C; the
sequencing inside Phase C is by file dependency, not by story priority.

## Parallel opportunities

Genuinely parallel — different files, no shared state:

- **T002** with T001
- **T037, T038, T039** — three separate new test files
- Within Phase C, **T023 (sign)** and **T018 (placement)** touch different functions in
  `cortical_prior.py` and can be developed in parallel, but must be **committed separately** (one
  change per commit).

**Explicitly NOT parallel**: T015, T021, T026, T028 — every one touches shared config
(`__init__.py` REGIONS, `run_selfchecks.py`, `cortical_microcircuit.py`, `home.py`). T006–T009 and
T026 all touch `cortical_microcircuit.py` and must serialise.

## Implementation strategy

**MVP is Phase A alone (T001–T009).** A drivable layered circuit is real, standalone value, and it is
P0/P6 work the project needs whether or not this external data is ever adopted.

**A + B** adds an instrument that can judge any future seeding — also valuable standalone, because it
makes the *next* seeding attempt falsifiable whoever makes it.

**A + B + C is the feature's core result and is complete on its own terms**: placed, signed,
drift-measured neurogenesis.

**D may never ship**, and the feature is not diminished by that.

## Task count

**42 tasks**, by story label: unlabelled 11 (Setup 2, Foundational decisions 3, Polish 6), **US4 4**,
**US3 7**, **US1 12**, **US2 5**, **US5 3**.

**Twelve are red-construction tasks** — T008, T013, T014, T017, T020, T022, T025, T027, T029, T032,
T033, T035. That is 29% of the feature and it is deliberate: in this project **a criterion whose red
has never been run is not evidence**, and this tree has now caught **seven** criteria that measured a
defect and scored it as the feature. The two SC-005 reds are what stop the headline criterion
rewarding movement rather than improvement, and T020 is what stops a silent units error shipping as a
valid layer assignment.
