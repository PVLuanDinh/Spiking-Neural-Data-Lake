# Implementation Plan: Charge the Dormant

**Branch**: `001-charge-the-dormant` | **Date**: 2026-07-31 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-charge-the-dormant/spec.md`

## Summary

Nine learning rules are either never called or called-and-inert. Give each a disposition backed by
a recorded neuter-A/B: wired-and-charged, retired, or exempt-with-reason. One instrument change
(class-granular exemption keys) unblocks the only disposition the tooling currently cannot express.
Behavioural fixes are measured and committed separately from instrument changes, so a change to the
measuring device never rides along with a change to the thing measured.

## Technical Context

**Language/Version**: Python 3.11 (gate venv `.venv-lake`; GPU venv `.venv-cu128`)

**Primary Dependencies**: numpy, torch (both unconditional core deps). No new dependency is added
by this feature — net-new dependency count for this project stays zero.

**Storage**: the one durable lake (`<NORD_HOME>/brain_lake.bin`). Every measurement here points
`NORD_HOME` at a temp dir; the operator's live lake is never written.

**Testing**: `run_selfchecks.py` (277 self-checks), `pytest` (≥70% line coverage gate),
`ruff check .`. Evidence artifacts via `tools/coverage_score.py record` / `--rerun`.

**Target Platform**: Windows 11 + RTX 5070 locally; CI is Linux/Windows with **no GPU**, so
everything here must run GPU-free.

**Project Type**: single Python package (`src/snn_data_lake/`) plus `tools/` instruments and a
`docs/coverage/` evidence ledger.

**Performance Goals**: no regression to the live tick (currently ~27.8 ms instrumented after the
`_coherence` fusion). Any wiring that lands must be re-measured against that.

**Constraints**: seed 0 and a temp `NORD_HOME` for every measurement; repo root AND `src` on
`PYTHONPATH` so the eprop-snn perception tier actually loads; the perception tier recorded in each
artifact header, because at least one finding is tier-dependent.

**Scale/Scope**: 9 rules, 8 source files, 1 instrument file, 11 ledger files. No new module.

## Constitution Check

*GATE: must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Gate | Status |
|---|---|---|
| **P0** honesty guards | No disposition may claim a capability without measurement; scaffolding must be labelled | PASS — every disposition requires an artifact (FR-002) |
| **P1** no-hardcode | `TimeCellClock`'s de facto constant threshold is itself a P1 violation to fix, not to preserve | PASS — recorded as a P1 finding, fixed with its own A/B |
| **P2** non-additive | Retirement requires port-before-delete; no second mechanism may be added beside an existing one | PASS — FR-005; `AffectiveEpisodicStore` retirement blocked on the port check |
| **P3** no dormancy | Reachability is not charge; no dead consumer may be added to silence a gate | PASS — wiring only where it is a real call site (D5), never to satisfy the gate |
| **P4** one brain | Every measurement constructs via `make_brain()` only | PASS |
| **P5** GPU by measurement | Nothing here belongs on the GPU; the suite must stay GPU-free | PASS |
| **P6** reachable and used | Each wired rule must be consumed by another faculty in a way that changes that consumer's output | PASS — that is exactly what the A/B measures |
| **Invariant 1** count before leaning | The class-granular key change is an instrument change; count its output before any rule depends on it | PASS — Phase A reports counts before Phase B acts |
| **Invariant 2** never silence | No bare-verb exemptions; no narrowing to dodge a true positive | PASS — FR-004, and the reason the key change is needed at all |
| **Invariant 3** negative is dangerous | A "never called" verdict must not become a retirement without a port check | PASS — D5 |
| **Invariant 5** verify the verifier | The instrument change must be exercised, not merely written | **RISK** — see Complexity Tracking |

**Post-Phase-1 re-check**: unchanged. The design adds no new module, no new dependency, and no new
mechanism beside an existing one.

## Project Structure

### Documentation (this feature)

```text
specs/001-charge-the-dormant/
├── plan.md              # This file
├── spec.md              # WHAT and WHY
├── research.md          # Phase 0 — D1..D6 decisions
├── data-model.md        # Phase 1 — entities
├── quickstart.md        # Phase 1 — how to reproduce every verdict
├── contracts/
│   └── verdict-contract.md   # the disposition/verdict interface
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 — produced by /speckit-tasks
```

### Source Code (repository root)

```text
src/snn_data_lake/
├── amygdala/affective_appraisal.py        # AffectiveAppraisal.learn        (called, inert)
├── basal_ganglia/latent_q.py              # LatentQ.update, _TabularQ.update (never called)
├── cerebellum/forward_model.py            # ForwardModel.update              (never called)
├── cortex/predictive_coding.py            # PredictiveCodingNetwork.learn    (exempt, landed)
├── cortex/time_cells.py                   # TimeCellClock.learn_interval     (called, learns nothing)
├── hippocampus/affective_memory.py        # AffectiveEpisodicStore.consolidate (retire candidate)
├── neuromodulation/developmental_envelope.py  # DevelopmentalEnvelope.update (never called)
├── plasticity/intrinsic_plasticity.py     # IntrinsicPlasticity.adapt        (called, no charge)
└── substrate/method_dormancy_gate.py      # INSTRUMENT — class-granular keys

tools/
├── charge_probe.py                        # graded observable, verdict tiers
└── coverage_score.py                      # record / --rerun / score --check

docs/coverage/                             # 11 category ledgers + artifacts/
```

**Structure Decision**: No new files in `src/`. This feature edits existing rules in place and adds
evidence artifacts under `docs/coverage/artifacts/`. The one instrument change is confined to
`substrate/method_dormancy_gate.py`.

## Phasing (ordering constraint, not optional)

- **Phase A — instrument**: class-granular exemption keys. Report the counts BEFORE and AFTER.
  No disposition may be written in this phase.
- **Phase B — dispositions that need no behavioural change**: the two permanent exemptions.
- **Phase C — measure the three called-but-inert rules**: one A/B each, artifacts recorded.
- **Phase D — decide wire/retire for the three never-called rules**: port check first.
- **Phase E — re-band**: only where these findings bear on a band, citing the measurement.

Behavioural commits never share a commit with instrument commits.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Invariant 5 risk: the class-granular key path may land unexercised | `dormant` is currently 0, so no live entry traverses the exemption branch | Waiting for a real dormant entry would leave a correct disposition unexpressible indefinitely. Mitigation: a unit test that drives the lookup directly with a synthetic def table, so the path is exercised by the suite rather than by luck. |
