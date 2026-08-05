# Implementation Plan: Policy Compliance (P0-P8)

**Branch**: `010-policy-compliance-audit` | **Date**: 2026-08-04 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/010-policy-compliance-audit/spec.md`

## Summary

Bring the tree into compliance with `docs/POLICIES.md` as extended on 2026-08-04 (P4b, P7, P8) and
with constitution v1.1.0. **Verify-first remediation, not a rewrite**: each finding is adversarially
re-verified against the tree before any code changes, and a finding that does not survive
verification drives nothing.

The work splits into three independently shippable phases matching the spec's user stories. The
ordering is by *reachability of harm*, not by size: Phase A is the only defect that reaches a human
(the brain reports a consciousness verdict no brain state can move), Phase B is the only one that
breaks a clean install, Phase C is claim-vs-tree drift.

**The central design decision, recorded here because it is the one most likely to be got backwards:
the remedy for a criterion that cannot fail is not to make it report a lower number.** It is to make
it read the system under test, delete the markers that have no reachable red, stop pinning the seed
that passes on the one marker that does, and stop reporting a verdict there is no measurement
behind. Building the replacement criterion is a **follow-on feature**. This feature removes a false
finish line; it does not claim to have built the real one.

## Technical Context

**Language/Version**: Python 3.11–3.13

**Primary Dependencies**: `torch`, `numpy`, `mujoco==3.11.0` (unconditional core). Optional extras:
`arm`, `lake`, `gpu`, `nmnist`, `llm`, `genn`, `exe`, `hf`, `nengo`, `desktop`, `train`, `dev`,
`all`. **Per P7 this plan carries no "no new dependency" certification** — see Constitution Check.

**Storage**: one durable spiking lake, `brain_lake.bin`, tag-partitioned (P4b). No second store may
be introduced by this feature.

**Testing**: `pytest` (≥70% line coverage), `python run_selfchecks.py` (278 module self-checks),
`python -m ruff check .` (E, F, I, UP, B), plus `tools/citation_guard.py` in the CI lint job.

**Target Platform**: Windows 11 and Linux; CI matrix {ubuntu, windows} × {py3.11, 3.12, 3.13}.
**CI has no GPU.**

**Project Type**: Python library + always-on daemon + one stdlib HTTP frontend
(`substrate/cockpit.py`, served at `:8137`).

**Performance Goals**: no regression to the live tick. The cockpit polls at 1 Hz and observing the
brain must not become the load — a ~1 s census once sat inside `snapshot()` and starved the tick
loop while every gate stayed green.

**Constraints**: the suite MUST remain runnable without a GPU. Determinism preserved:
`random.Random(seed)` per instance, `zlib.crc32` for persisted values. ASCII-only `print()`.
Adding any module requires registering it in `__init__.py` REGIONS **and** `run_selfchecks.py`, or
the dormancy gate fails.

**Scale/Scope**: ~400 files under `src/`, 10 anatomical regions, 278 self-checks, 9 policies. Audit
surface is the tracked tree; the five `brainstem/` safety-envelope files are **report-only**.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against constitution v1.1.0, principles **P0–P8** and the Standing Invariants.

| Principle | Assessment |
|---|---|
| **P0** self-improvement | PASS. This feature removes a false capability report, which P0's honesty guard directly requires ("a measurement that cannot fail is not a measurement"). It closes no avenue and opens none. |
| **P1** no-hardcode | PASS with a note. Phase A removes a hardcoded `seed=0` that pins a passing result — that is P1's signature defect (a constant where a computed value belongs) appearing inside a measurement. |
| **P2** non-additive | **GATE — see below.** Any new instrument must not duplicate an existing one. |
| **P3** no dormancy | PASS. No faculty is force-wired. Where Phase A deletes a marker, the deletion is the P3-sanctioned disposition (retired), not a wire. |
| **P4 / P4b** one brain, one lake | PASS. No new durable store; no second `Brain`. Phase C touches only claims. |
| **P5** GPU | PASS. No GPU path added; the GPU-independence constraint is respected and is itself one of the things Phase B verifies. |
| **P6** faculty-to-faculty | NOT ADVANCED by this feature, and deliberately so. Severing a self-report does not create a faculty-to-faculty consumer. Recorded as unchanged rather than claimed as progress. |
| **P7** dependency | PASS — **and this plan deliberately does NOT certify "no new dependency."** Per constitution v1.1.0: "a Constitution Check certifying 'no new dependency' is no longer evidence of anything under P7." Phase B may legitimately ADD a declaration (`scikit-learn` is imported and undeclared); adding it is the fix, not a violation. If a hand-rolled mechanism is found that a library covers, adopting the library is a permitted remedy subject to retiring the hand-rolled one. |
| **P8** invent / reachable red | **GATE — see below.** |

### Gate 1 (P2): no duplicate instrument

Before proposing any new gate, the existing five were checked: `substrate/dormancy_gate.py`
(per-module reachability), `substrate/method_dormancy_gate.py` (per-method learning verbs),
`substrate/import_manifest.py` (cross-tier import survival), `substrate/replication_guard.py`
(stack-class uniqueness), `tools/citation_guard.py` (doc→`.py` citation resolution).

**None checks either property this feature needs.** The dependency ledger (declared ↔ imported) and
red-reachability of a criterion are both unoccupied. Two new instruments are therefore P2-legitimate.
They are scoped to not overlap each other: one answers *"is this package accounted for"*, the other
*"can this criterion fail"*.

**Resolution: PASS.** Recorded rather than assumed, because "we needed a new gate" is exactly the
claim P2 says to check before believing.

### Gate 2 (P8): every criterion this plan introduces has a reachable red

| New criterion | Its RED (the input that makes it FAIL) | Fails today? |
|---|---|---|
| dependency ledger gate | a package imported under `src/` and absent from project metadata; or a declared package imported nowhere without a reason | **YES** — both directions |
| red-reachability gate | a registered criterion for which no failing input is recorded | **YES** — at least two markers |

Both gates fail on today's tree, which is the required property: a gate that passes on first run
against a tree with known violations is not measuring what it claims. Per the project's own bar,
**the gate's output is counted before the rule leans on it** — expected first-run output is recorded
in `contracts/` so a rubber-stamp-sized result kills the gate rather than the tree.

**Resolution: PASS.**

### Standing Invariants

- Invariant 1 (count the gate's output first): honoured — expected first-run counts are recorded in
  `contracts/` before either gate is adopted as asserting.
- Invariant 2 (never silence a check): the two markers with no reachable red are **deleted**, not
  exempted or re-thresholded. An exemption here would hide exactly the defect class the gate exists
  to catch.
- Invariant 3 (negative verdict is the dangerous one): where red-reachability cannot be determined,
  the criterion is reported UNKNOWN, not deleted.
- Invariant 5 (verify the verifier): each new gate ships with a test that makes it FAIL, per the
  established pattern (`citation_guard`'s asserts "were tested by making them fire").
- Invariant 7 (safety envelope): report-only in this feature. **No violations justified; Complexity
  Tracking table is empty.**

## Project Structure

### Documentation (this feature)

```text
specs/010-policy-compliance-audit/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output — gate contracts + expected first-run output
│   ├── dependency-ledger.md
│   └── red-reachability.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (from /speckit-specify)
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

Existing layout; this feature adds two instruments under `tools/` and edits in place.

```text
src/snn_data_lake/
├── cortex/                  # access_consciousness.py (bit-exactness claim, Phase A)
├── hippocampus/
├── basal_ganglia/
├── amygdala/
├── cerebellum/
├── brainstem/               # SAFETY ENVELOPE — report-only, never edited by this feature
├── thalamus/
├── neuromodulation/
├── plasticity/
├── reasoning/               # consciousness_probe.py, spike_chat_lake.py (Phase A)
│                            # laminar_local_readout.py, vsa_structure_mapping.py (Phase B)
├── substrate/               # cockpit.py, brain_telemetry.py PROVENANCE (Phase C)
└── __init__.py              # package docstring claim (Phase C)

tools/                       # new instruments land here, NOT in the package
├── citation_guard.py        # existing
├── charge_probe.py          # existing
├── dependency_ledger.py     # NEW (Phase B)
└── red_reachability.py      # NEW (Phase C/A)

tests/                       # one failing-case test per new gate
pyproject.toml               # dependency declarations + self-description (Phase B)
.github/workflows/ci.yml     # register new gates in the lint job
```

**Structure Decision**: new instruments go in `tools/`, **not** into `src/snn_data_lake/`. This
follows the precedent set by `citation_guard.py` and its recorded rationale: `run_selfchecks.py`
only executes `python -m snn_data_lake.<mod>` package modules, so qualifying would mean moving a
repo-hygiene check into the package and churning the canonical 278 count quoted across several
documents. These gates check the repo, not a brain faculty, so they belong beside `citation_guard`
and register as CI lint steps.

## Phase ordering

| Phase | User story | Ships independently | Rationale |
|---|---|---|---|
| **A** | US1 — brain stops claiming what it cannot measure | yes | Only defect that reaches a human. No dependency on B or C. |
| **B** | US2 — clean install works | yes | Only defect that breaks arrival. Independent of A. |
| **C** | US3 + US4 — claims match tree; cockpit honesty | yes | Depends on nothing; sequenced last because it misleads rather than breaks. |

Findings from the running full-tree audit **slot into these phases by policy** rather than requiring
a re-plan: P8/honesty findings → Phase A; P7/dependency findings → Phase B; doc, cockpit, P0–P6
findings → Phase C. A finding that fits none of the three is recorded as out-of-scope with its
blocker named (spec FR-012) rather than forcing a fourth phase.

## Complexity Tracking

> No Constitution Check violations. Table intentionally empty.

The one judgement call worth recording even though it is not a violation: this feature **adds two
new instruments** to a tree that already has five. That is defensible under P2 only because neither
existing gate checks either property (Gate 1 above), and it is constrained by the standing rule that
each must be counted before it is adopted as asserting. If either gate's first run returns an
allowlist-sized list rather than a work list, **the gate is dropped, not the tree** — the precedent
is the broad citation rule that died at 475 violations while its narrow form survived at 21.
