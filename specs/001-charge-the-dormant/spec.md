# Feature Specification: Charge the Dormant

**Feature Directory**: `specs/001-charge-the-dormant`

**Created**: 2026-07-31

**Source**: Phase 5 of `docs/COVERAGE_TO_100.md`; governed by `.specify/memory/constitution.md` (P0–P6)

## Problem

Nine learning rules in the brain either never run, or run and change nothing. Each was
adjudicated at runtime on 2026-07-31 and adversarially reviewed, so the facts are measured
rather than read off a call graph. Today the ledger's bands for the affected faculties rest on
evidence weaker than it appears, and this project's own policy is that a capability reported
without measurement is fake.

The work is to give each of the nine a **disposition backed by measurement**: wired-and-charged,
retired, or exempt-with-reason. Success is emphatically NOT "the gate reports zero" — a gate can
always be made to report zero by silencing it, which the constitution forbids.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Asking "does this faculty do anything?" returns a measured answer (Priority: P1)

The owner picks any of the nine and asks whether it affects behaviour. The answer is a recorded
A/B: the rule is neutered, the brain re-run at the same seed, and the report says whether a
decision changed, whether an internal read changed and by how much, or whether nothing changed —
never merely whether the rule is imported or called.

**Why this priority**: This is the feature. Every other outcome depends on the answer being
measured rather than argued.

**Acceptance Scenarios**:

1. **Given** a rule with a live call site, **When** it is neutered and the brain re-run at the
   same seed, **Then** the verdict names which of the three outcomes occurred.
2. **Given** a rule never called on a live path, **When** the report is produced, **Then** it
   says "never called" explicitly — that is a different finding from "changed nothing".
3. **Given** a rule whose behaviour depends on which perception tier is loaded, **When** the
   report is produced, **Then** it names the tier measured and does not generalise beyond it.

### User Story 2 - A dead faculty is resolved without being quietly buried (Priority: P1)

For each rule found dead, exactly one disposition lands, carrying its reason: wired into a live
path and proven to charge; retired, with any unique capability ported first; or exempted, with a
disposition and a reason recorded at the exemption site.

**Acceptance Scenarios**:

1. **Given** a rule is exempted, **When** the exemption is written, **Then** it names that
   specific rule and cannot suppress unrelated findings elsewhere in the tree.
2. **Given** a rule is retired, **When** the retirement lands, **Then** any capability unique to
   it was ported first, and the port is demonstrated.
3. **Given** a rule is wired, **When** the wiring lands, **Then** a recorded A/B shows the
   brain's output differs with and without it.

### User Story 3 - A faculty that learns nothing is reported as learning nothing (Priority: P2)

Three of the nine are called on the right object, update real state, and change no outcome. The
owner is told so plainly, including where a module's own documentation overstates what it does.

**Acceptance Scenarios**:

1. **Given** learned state that reaches only telemetry, **When** the report is produced,
   **Then** it identifies the consumer that was expected to read it and did not.
2. **Given** a module documents that it replaced a fixed constant, **When** measurement shows
   the value never moves, **Then** the contradiction is recorded against that claim.

### Edge Cases

- A rule that charges in one perception tier and not another must not get a single verdict; the
  tier is part of the finding.
- An incomplete neuter (state re-established before it is read) biases toward "no change" — the
  direction that makes a live faculty look dead. Any neuter used must be shown to be complete.
- Where a probe cannot distinguish "never fired" from "fired and did not matter", it must report
  unknown rather than a confident negative.
- Removing one contributor from a sum of eleven rarely changes which option wins, so no verdict
  may rest on that test alone.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Each of the nine rules MUST end with exactly one disposition: wired-and-charged,
  retired, or exempt-with-reason.
- **FR-002**: Every disposition MUST cite a recorded A/B artifact capturing command, exit code
  and output, which re-runs to the same result.
- **FR-003**: A verdict MUST distinguish "never called", "changed a decision", "changed an
  internal read", and "changed nothing".
- **FR-004**: An exemption MUST identify its specific rule and MUST NOT be expressible in a form
  that also suppresses unrelated findings.
- **FR-005**: A retirement MUST port any unique capability before deletion and record where.
- **FR-006**: Where measurement contradicts a module's own documented claim, the contradiction
  MUST be recorded against that claim.
- **FR-007**: Faculty bands whose supporting evidence these findings weaken MUST be re-examined;
  any band change MUST cite the measurement justifying it.
- **FR-008**: No check may be weakened, narrowed, or silenced to reach a disposition.
- **FR-009**: The three quality gates MUST remain green throughout.

### Key Entities

- **Rule under adjudication**: a named learning method, its module, its current disposition.
- **A/B artifact**: a captured run — command, exit code, commit, verbatim output — re-executable
  by a later reader.
- **Disposition**: wired / retired / exempt, plus reason and backing evidence.
- **Ledger row**: the faculty whose band cites this rule as evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Nine of nine rules carry a disposition naming the artifact that backs it; zero
  remain "uncertain".
- **SC-002**: Every recorded artifact re-runs to the same verdict.
- **SC-003**: Zero exemptions exist that can suppress a finding other than the one they name.
- **SC-004**: For each of the three "called but changes nothing" rules, the report states the
  expected consumer and why the value does not reach it.
- **SC-005**: Every affected band is either unchanged with its reason restated, or changed with
  the measurement cited.
- **SC-006**: The three gates are green at completion and none was modified to make them so.

## Assumptions

- The nine rules and their 2026-07-31 runtime verdicts are the starting facts; they were
  adversarially reviewed and are not re-derived, though better measurement may overturn any.
- "The route the daemon actually runs" means the perception tier the always-on daemon loads, not
  the tier a bare probe harness produces. A finding measured only under the probe harness is
  scoped to it.
- Re-banding is in scope only where these findings bear on it; a general ledger re-audit is not.

## Out of Scope

- Phase 6 (capability) and Phase 7 (bounding the growing substrate) of the same plan.
- Any change to what the coverage percentage means, or to the scoring rubric.
- Retiring a mechanism whose unique capability has not been ported.
