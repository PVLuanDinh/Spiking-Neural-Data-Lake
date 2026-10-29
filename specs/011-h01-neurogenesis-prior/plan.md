# Implementation Plan: Order neurogenesis by cortical architecture, from a prior that must be beatable

**Branch**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/011-h01-neurogenesis-prior/spec.md`

## Summary

Give every grown neuron a place, a layer and a fixed sign, seeded from the H01 human-cortex
reconstruction — and build the instrument that proves the seed is *beatable* before building the
seeding itself. Four phases, dependency-ordered: **the circuit's input port** (needed regardless of
H01), **the drift instrument** (or the seeding has nothing to prove itself against), **placement and
sign**, **connection strengths** (last, because its bias is the least correctable).

The technical approach is deliberately small: one new governed module holding every externally
sourced constant with its units, provenance and accuracy bound; a placement function ported from the
source's own 40-line pure-numpy layer test; and a drift record that makes the seed falsifiable in
both directions. No new dependency and no new gate.

## Technical Context

**Language/Version**: Python >= 3.14, **free-threaded (cp314t, `Py_GIL_DISABLED=1`) — a hard
requirement** enforced by a runtime guard in `src/snn_data_lake/__init__.py`.

**Primary Dependencies**: torch, numpy, mujoco==3.11.0, snntorch==1.0.0 (all already core). **This
feature adds none** — the four source files are JSON and CSV, parseable with `json` and `csv` from
the standard library, and the layer test is ~40 lines of arithmetic. Stated as a fact about the
work, not as a virtue: under P7 a dependency that bought something would be accepted without
argument, and none is needed here.

**Storage**: Two distinct kinds, and conflating them would violate P4b.
- **Reference data** (the H01 constants) is *not* learned state. It lives with the code as versioned
  constants in one governed module. It must **not** enter `brain_lake.bin`.
- **Grown state** (the positions, layers and signs of units the brain actually grew) *is* learned
  state and **must** persist to the one lake under its own record tag, with a cross-process read-back
  proof.

**Testing**: `run_selfchecks.py` (280 checks, module `__main__` self-checks), `pytest` (>=70%
coverage; currently 93.02%), `ruff check .`; CI lint job additionally runs citation_guard,
dependency_ledger (`--assert` + `--self-check`), red_reachability `--assert`, doc_command_guard
`--assert` *(corrected 2026-10-29: never a lint step; it runs in run_selfchecks's tools stage, which the self-checks CI job runs (`ci.yml:87`; no lint step))*, p6_gate (`--assert` + `--self-check`), p4c_ratchet (`--assert` + `--self-check`).

**Target Platform**: `.venv-lake` (python 3.14.6 free-threading, torch 2.11.0+cu128, cuda True,
RTX 5070). **CI has no GPU** and installs CPU torch on the same free-threaded interpreter, so every
criterion here must be reachable without a card.

**Project Type**: Library — a brain organised by anatomical region under `src/snn_data_lake/`.

**Performance Goals**: Placement must not dominate a growth tick. Layer assignment is six circle
tests per unit — O(1) per unit, ~40 lines of arithmetic — so the target is that a seeded growth cycle
stays within measurement noise of an unseeded one. Ingestion of the 11.2 MB source table happens
**once, offline, at constant-generation time**, never on a live path.

**Constraints**: Same-seed byte-identical placement (determinism is load-bearing project-wide and
the interpreter has no GIL). No GPU dependency. The five brainstem safety files are report-only.

**Scale/Scope**: The source describes 49,377 somata / 15,471 fully-attributed neurons. This brain is
targeted at 1M+ units. **The gap between those two numbers is the feature's sharpest design question
and it is answered in Phase 0, not deferred.**

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Policy | Assessment | Verdict |
|---|---|---|
| **P0** self-improvement by every avenue | "Neural growth" is a named avenue and the avenue list is a *build list, not a status ledger*. Growth that orders nothing is open P0 work. This feature builds an avenue rather than reporting one. | **PASS** |
| **P1** no hardcode | **The binding constraint.** An E:I ratio lifted from one patient is a magic constant. Admissible only as an initialisation prior that learning moves; see *How a seeded value stays movable* below. | **PASS, conditionally** |
| **P2** adopt-or-merge, retire the inferior | `cortex/cortical_microcircuit.py` already carries the layered structure. It becomes the substrate; no second layered mechanism is created. No new gate — all ten existing instruments were checked and none is duplicated. | **PASS** |
| **P3** no dormancy | Placement that changes no outcome is dormancy with better provenance. SC-006 is the neuter-A/B; the new module registers in REGIONS + `run_selfchecks.py`. | **PASS** |
| **P4 / P4b / P4c** one brain, one lake, every build | Reference data with the code; grown state to the lake with a cross-process proof; the new module lands **inside** the governed tree from the start. | **PASS** |
| **P5** GPU where it helps | Placement is O(1) per unit on the CPU; the crossover is nowhere near it. Ingestion is offline. No GPU claim is made. | **PASS** |
| **P6** mutual reach and use | US4 builds the microcircuit's missing input port — the reason it is EXEMPT-permanent today. That is P6 work whether or not H01 is adopted. | **PASS** |
| **P7** any dependency that helps | **No certification is offered.** None is needed; that is a fact about the work. | **PASS** |
| **P8** invent the method, with a reachable red | Eleven criteria, each with its red; SC-005 carries **two**, one per direction. All are scheduled in Phase 2. Delays are refused outright rather than approximated. | **PASS** |

### How a seeded value stays movable (P1, the load-bearing answer)

A constant learning cannot reach is a hardcode however well cited. Three mechanisms, together:

1. **Seeded values are written into mutable state, never read at use-time.** The constants module is
   consulted **once**, at population construction, to produce initial values. No live path reads it
   again. A value that is re-read every tick is pinned by construction.
2. **Sign is the deliberate exception, and it is the only one.** Dale's Principle says sign is a
   *property of the cell* — it must NOT drift, and SC-004 asserts it does not. This is not a P1
   violation but a structural invariant, and it is declared here so the exception is explicit rather
   than discovered.
3. **The drift record makes movability observable rather than asserted.** Every seeded statistic
   carries `(seeded, current, distance, direction)`. A zero-distance report is a **failure of the
   seeding approach** (FR-009), which is what stops "movable in principle" from passing as "moved".

## Key design decisions

### D1 — What the prior means once the population exceeds the measured one

**Decision: the prior is a set of per-layer PROPORTIONS that scale, not absolute counts and not a
first-N seed.**

**Rationale, and it comes from a measured property of the data rather than from preference.** Three
independent computations over the same cells gave white-matter counts of 12,872 / 13,839 / 14,923 —
absolute per-layer pool sizes are not reproducible better than ~10–20%. The *excitatory fraction* is
robust across those same three to ±0.015. So the data itself says which quantity survives
re-derivation: **fractions do, counts do not.** Seeding with a quantity the source cannot reproduce
would import its irreproducibility as if it were signal.

**Alternatives considered and rejected:**
- *Density per unit volume, with the volume growing.* Requires the brain to carry a physical volume
  and a volume-growth law. That law has no constraint in any data we hold, so it would introduce a
  free parameter to justify a parameter — and P1 would then bite the growth law instead.
- *A fixed seed only the first N units draw from.* The prior stops mattering after N, so the drift
  measurement silently loses its subject partway through a run while still reporting a number. That
  is a criterion quietly becoming vacuous, which this project has already caught five times.

**Consequence to carry:** proportions make the drift measurement well-defined at *any* population
size, which is what SC-005 needs.

### D2 — Mapping the source's seven regions onto the circuit's populations

**Decision, recorded as a decision and not an equivalence:**

| Source region | Circuit population | Rationale |
|---|---|---|
| Layer 2 + Layer 3 | L2/3 (e + i) | The source separates them; the circuit merges them, as its published parameterisation does. Union, not a choice between them. |
| Layer 4 | L4 (e + i) | Direct. |
| Layer 5 | L5 (e + i) | Direct. |
| Layer 6 | L6 (e + i) | Direct. |
| **Layer 1** | **NEW inhibitory-only population** | **A deliberate departure.** The circuit's published parameterisation assumes layer 1 has no cell bodies. The source measures **zero excitatory somata and 540 interneurons** there. The source is followed, and this is the single largest structural change the feature makes to the circuit. |
| White matter | **excluded** | Not a cortical layer. Its 670 neurons are a projection zone, not a layer population. Excluded explicitly and counted, rather than silently dropped. |

### D3 — Units grown beyond the measured volume

The six circles partition the whole plane by construction (first-match-wins with a fall-through), so
a layer is *always* assignable. What is undefined is whether the assignment is **meaningful** outside
the measured block.

**Decision: assign by the same rule, tag the unit `extrapolated`, and report the count in the growth
report.** Silently clamping would manufacture confidence; refusing to place would stop growth. The
tag makes the extrapolation visible to anything that consumes placement, and the count makes it
visible to the operator.

### D4 — The drift instrument is built BEFORE the seeding

The feature's premise is that the seed must be beatable. If seeding lands first, it lands with
nothing to prove itself against and the natural next step is to declare it good. Phase order enforces
the opposite.

## Phases

Each phase is independently shippable, lands as one change per commit with the full gate green, and
is honest on its own if the later ones never land.

### Phase A — The circuit gets an input port *(US4)*

Give `cortex/cortical_microcircuit.py` a way to receive a signal and expose a response. This is P0/P6
work that stands alone: it removes the reason the module is EXEMPT-permanent in the dormancy gate,
and it is needed whether or not any external reconstruction is ever adopted.

**Red:** remove the port and the "two inputs, two responses" criterion fails — confirming the
criterion measures the port and not something incidental.

**Ships alone as:** a layered cortical circuit the rest of the brain can drive.

### Phase B — The drift instrument *(US3, first half)*

Build the drift record and the three-arm comparison harness (seeded / random control / deliberately
wrong prior) against the *existing* unseeded growth. It measures nothing interesting yet — and that
is the point: it must be shown to work on a population whose seeding is not in question.

**Reds:** both of SC-005's, constructed here and re-run in Phase C — a pinned seed (drift zero) and a
harmful prior (underperforms the random control).

**Ships alone as:** an instrument that can tell whether any future seeding earns its place.

### Phase C — Placement, layer and sign *(US1 + US2, and US3's second half)*

The constants module, the ported layer test, per-unit position/layer/sign at creation, Dale enforced
structurally, grown state persisted to the lake with a cross-process read-back, and the drift
instrument from Phase B pointed at the now-seeded population.

**The unit-mismatch red (SC-002) is constructed and run here**, because it is the failure that
otherwise produces a plausible-looking result: feed the boundary geometry in the wrong length unit
and *every* unit lands in the fall-through layer.

**Ships alone as:** placed, signed, drift-measured neurogenesis — the feature's core result.

### Phase D — Connection strengths *(US5)*

Seed initial strengths per population from the measured GP-strength distribution, with the AIS
exclusion and the per-fragment/per-axon caveat carried in provenance.

**Ships last, and may never ship.** Its bias is the least correctable and its quantity is a
structural proxy for a physiological one. The feature is complete on its own terms without it.

## Project Structure

### Documentation (this feature)

```text
specs/011-h01-neurogenesis-prior/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
├── checklists/
│   └── requirements.md  # from /speckit-specify
└── tasks.md             # /speckit-tasks output — NOT created here
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── cortex/
│   ├── cortical_microcircuit.py     # MODIFIED (Phase A): gains an input port and a response
│   │                                #   readout. Becomes the substrate — NOT duplicated.
│   ├── neurogenesis_engine.py       # MODIFIED (Phase C): grown units gain position, layer, sign
│   │                                #   at creation; grown state persists to the lake.
│   └── cortical_prior.py            # NEW (Phase C): the ONE place every externally sourced
│                                    #   constant lives, each with units, provenance and accuracy
│                                    #   bound. Replaceable as a unit. Registered in REGIONS and
│                                    #   run_selfchecks.py from the start (P4c).
├── substrate/
│   └── home.py                      # MODIFIED (Phase C): a record tag for grown placement state
└── __init__.py                      # MODIFIED: REGIONS gains cortical_prior

tests/
├── test_cortical_prior.py           # NEW: units, provenance, accuracy bounds, degenerate inputs
├── test_placement.py                # NEW: layer assignment incl. the unit-mismatch red
└── test_drift.py                    # NEW: the three-arm comparison, both SC-005 reds

run_selfchecks.py                    # MODIFIED: registers cortical_prior
```

**Structure Decision**: The brain is organised by anatomical region, so the new constants module goes
in `cortex/` — it describes cortical architecture, and P4c's "the directory is a fact about the
filesystem, never a statement about whether it is part of the organism" cuts the other way here: this
belongs to cortex *by function*. It lands inside the governed tree registered from the start, which
is what P4c now requires and what `tools/p4c_ratchet.py` now asserts. **This feature must not raise
the ungoverned count** (currently 248 total / 150 in `reasoning/`).

## Complexity Tracking

No constitution violations require justification. Two things are recorded here because they are
departures a reviewer would otherwise flag:

| Departure | Why | Alternative rejected because |
|---|---|---|
| Sign is exempt from P1's movability requirement | Dale's Principle makes sign a structural invariant, not a tunable parameter. SC-004 asserts it does *not* drift. | Letting sign drift would discard the one biological invariant the feature exists to add. |
| The circuit gains a layer-1 inhibitory population its published parameterisation lacks | The source measures 540 interneurons and zero excitatory somata in layer 1. The published assumption is half wrong. | Keeping the published assumption would contradict the data the feature is built on, while claiming to follow it. |
