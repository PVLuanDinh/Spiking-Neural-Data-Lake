# Feature Specification: Policy Compliance (P0-P8)

**Feature Branch**: `010-policy-compliance-audit`

**Created**: 2026-08-04

**Status**: Draft

**Input**: User description: "Audit and apply fixes for the entire project against the updated POLICIES.md and CLAUDE.md. All files must be thoroughly verified against the policies. The cockpit front end must also be checked."

## Why this feature exists

`docs/POLICIES.md` gained three policies on 2026-08-04 — P4b (the durable lake is part of the
brain), P7 (any dependency that helps is accepted) and P8 (invent any method, but a success
criterion with no reachable RED is worthless). The constitution moved to 1.1.0. **No part of this
tree has ever been checked against P7 or P8**, and the P8 audit immediately found a live defect:
the brain answers "are you conscious" from a number that is the same for a brain, an empty brain,
and no brain at all.

This feature is subject to its own subject matter. **Every success criterion below states the
input that makes it report FAIL.** A compliance feature whose criteria cannot fail would be
precisely the defect it exists to remove.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The brain stops claiming what it cannot measure (Priority: P1)

The owner asks the running brain "are you conscious". Today it answers from a four-marker verdict
that no state of the brain can move: the markers are computed over freshly-built synthetic objects
at a pinned seed, so the same answer is returned whether the brain is healthy, degraded, or absent
entirely. After this story, the brain answers only from criteria its own state can change, and says
plainly which of them it cannot yet measure.

**Why this priority**: It is the only finding that is both live and reaches a human. Everything
else in this feature is hygiene; this one is the project telling itself something false about the
thing it exists to build. `docs/coverage/Consciousness.md:366` already names it — "a 4/4
self-report that no live measurement backs" — so the defect is known and shipping.

**Independent Test**: Degrade the brain (or pass no brain at all) and ask the question. Today the
verdict is unchanged. After this story, the verdict changes or the system reports that it cannot
measure.

**Acceptance Scenarios**:

1. **Given** a healthy brain and a deliberately degraded brain, **When** each is put through the
   consciousness markers, **Then** at least one marker reports a different value for the two.
2. **Given** a marker for which no input producing a FAIL can be constructed, **When** the marker
   set is reviewed, **Then** that marker is DELETED, not weakened or re-thresholded.
3. **Given** the chat question "are you conscious", **When** no criterion the brain's own state can
   move is available, **Then** the reply states what is measured and what is not, and does not
   report a verdict it cannot back.
4. **Given** the surviving marker(s), **When** run across many seeds, **Then** the reported result
   is the distribution, not a single pinned seed known to pass.

---

### User Story 2 - A clean install produces a working tree (Priority: P1)

A new contributor (or a CI runner, or the owner on a fresh machine) clones the repo and runs
`pip install .`. Today the result imports third-party packages that the project never declares, so
the tree is broken on arrival in a way no existing gate detects.

**Why this priority**: Equal-P1 with Story 1 because it is a correctness defect with a trivially
reachable red, and because P7 makes the ledger — not the permission — the thing that matters. It is
independently valuable: it can ship without any part of Story 1.

**Independent Test**: Install the package into an empty environment and import every module. Today
this fails on an undeclared package. After this story it succeeds, and a gate reports the same
mismatch if it is ever reintroduced.

**Acceptance Scenarios**:

1. **Given** a clean environment with only the declared dependencies installed, **When** every
   module under `src/` is imported, **Then** no `ModuleNotFoundError` occurs.
2. **Given** a dependency declared in the project metadata, **When** the tree is scanned, **Then**
   it is imported somewhere or carries a stated reason for being declared and unused.
3. **Given** a third-party package imported under `src/`, **When** the tree is scanned, **Then** it
   is declared, in core or in an extra.
4. **Given** the project's own self-description, **When** compared against its declared
   dependencies, **Then** they agree.

---

### User Story 3 - Every claim in the tree matches the tree (Priority: P2)

A reader — human or agent — opens any document, docstring, or policy row and can trust that what it
says is still true. Today the tree carries claims retired by the 2026-07-20 dependency decision and
by later re-measurements, including a package docstring advertising a property the project
abolished, and spec plans certifying a rule that no longer exists.

**Why this priority**: P2 because it misleads rather than breaks. It is nonetheless the failure mode
`docs/POLICIES.md` documents about itself most often — "a rule citing an already-closed defect as
live is the drift this file keeps suffering."

**Independent Test**: Pick any claim of policy status, capability, or measured number in the tree and
check it against the code or the recorded measurement. Today several disagree; after this story none
do, and the ones that cannot be verified say so.

**Acceptance Scenarios**:

1. **Given** any document stating a policy count, list, or enforcement status, **When** compared
   against `docs/POLICIES.md`, **Then** they agree.
2. **Given** a stated measured number, **When** its source measurement is located, **Then** either
   they agree or the number is marked as superseded with the newer value.
3. **Given** a capability claim, **When** a measurement backing it is sought, **Then** one exists or
   the claim is downgraded to what is actually shown.

---

### User Story 4 - The cockpit shows nothing it cannot back (Priority: P2)

The owner opens the one frontend and reads the brain's live state. Today a headline number is drawn
without the provenance tag that exists specifically to mark whether a signal can vary, positioned
directly above two values that do carry one — so the untagged number reads as more trustworthy than
the tagged ones, which is backwards.

**Why this priority**: P2 because it is display honesty rather than a wrong computation, but the
owner explicitly required the frontend be checked, and a dashboard is where a constant most easily
becomes a claim.

**Independent Test**: For each number on the page, ask whether any brain state changes it. Today at
least one headline value cannot vary and is not marked. After this story, every displayed value
either varies or is marked as unable to.

**Acceptance Scenarios**:

1. **Given** a value displayed on the cockpit, **When** it cannot vary with brain state, **Then**
   it is labelled as such beside its own number.
2. **Given** the cockpit is polled at its normal rate, **When** the tick loop is measured, **Then**
   observing the brain does not measurably slow it.
3. **Given** the cockpit is served, **When** its construction path is inspected, **Then** it never
   creates a brain of its own.
4. **Given** any route the frontend exposes, **When** the interface is reviewed, **Then** it is
   reachable from the interface or recorded as deliberately headless.

---

### Edge Cases

- **A marker with no reachable red is load-bearing elsewhere.** Deletion is still the rule, but the
  consumer must be handled: the dependent reports "not measured" rather than inheriting a number
  that meant nothing. Silence is a valid output; a fabricated one is not.
- **The honest fix is a claim change, not a code change.** Where a mechanism is correct and only its
  description is wrong, the description is corrected and the code is left alone. Rewriting working
  code to match a false claim would be the inverse defect.
- **A fix would force a wire that does not charge.** Forbidden. A wire that carries nothing is
  reverted and the faculty is exempted with a reason — forcing it would trade a documented gap for
  an undocumented lie.
- **A remedy touches the safety envelope.** Out of scope. Report and stop; the envelope changes only
  with explicit owner approval.
- **A violation cannot be fixed within this feature.** It is recorded with its blocker named, and
  counts as an open item rather than being quietly dropped from the ledger.
- **The audit itself is wrong.** Expected and planned for: findings are adversarially re-verified
  before any fix is applied, because a confident finding that does not survive checking is this
  project's most common failure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every success criterion, gate, marker, probe and rubric verdict in the tree MUST have
  a documented input that makes it report FAIL. Any that does not MUST be deleted, not weakened.
- **FR-002**: A criterion intended to measure a property of the brain MUST take the brain as input.
  A criterion invariant to the entire system under test is unfalsifiable and MUST NOT ship.
- **FR-003**: The system MUST NOT report a verdict about a property it cannot measure. Where a
  measurement is absent, it MUST say so.
- **FR-004**: Every third-party package imported under `src/` MUST be declared in project metadata.
- **FR-005**: Every declared dependency MUST be imported somewhere, or carry a stated reason.
- **FR-006**: A claim of capability, policy status, or measured value anywhere in the tree MUST
  match the tree, or be marked superseded with the current value.
- **FR-007**: Every value displayed to a human MUST either be able to vary with system state, or be
  labelled as unable to, beside the value itself.
- **FR-008**: State the brain learns MUST be written to the one durable store and recalled from it;
  no second durable store may be introduced.
- **FR-009**: The safety envelope MUST NOT be modified by this feature. Findings against it are
  reported only.
- **FR-010**: Every fix MUST leave the three gates green, and MUST be applied as one change per
  commit so a regression isolates to its own change.
- **FR-011**: Each finding MUST be adversarially re-verified against the tree before a fix is
  applied; unverified findings MUST NOT drive changes.
- **FR-012**: A finding that cannot be remedied within this feature MUST be recorded with its
  blocker named, not dropped.

### Key Entities

- **Finding**: a located violation — policy, file, line, quoted evidence, the clause it breaks, a
  proposed remedy, and a verification verdict. A finding without a verdict cannot drive a change.
- **Criterion**: any check that emits pass/fail or a score. Its defining attribute for this feature
  is its *reachable red* — the recorded input under which it fails. A criterion without one is a
  defect regardless of what it measures.
- **Dependency ledger**: the correspondence between packages declared in project metadata and
  packages actually imported. Both directions are defects.
- **Claim**: any statement in a document, docstring or comment asserting a capability, a policy
  status, or a measured number.

## Success Criteria *(mandatory)*

Each criterion states the input under which it reports FAIL. Per P8, a criterion listed here without
a reachable red would itself be a violation of this feature.

- **SC-001**: Every surviving consciousness marker changes its reported value between a healthy
  brain and a degraded one.
  **RED**: run the markers against a brain that has been degraded and against a healthy one; if
  every marker reports identically, this FAILS. *(Today: fails — the markers accept no brain at all.)*
- **SC-002**: Zero markers, gates or scored criteria remain for which no failing input can be
  constructed.
  **RED**: enumerate criteria, attempt to construct a failing input for each; any criterion with no
  such input FAILS. *(Today: fails on at least two markers.)*
- **SC-003**: Asking the running system whether it is conscious never returns a verdict unbacked by
  a live measurement.
  **RED**: ask the question with the measurement path disabled; a confident verdict FAILS, a stated
  inability PASSES. *(Today: fails — returns 4 of 4.)*
- **SC-004**: Installing the project into an empty environment and importing every module raises no
  missing-module error.
  **RED**: install with only declared dependencies, import everything; any ModuleNotFoundError
  FAILS. *(Today: fails on an undeclared package.)*
- **SC-005**: The declared-dependency set and the imported-package set agree in both directions, or
  each exception carries a reason.
  **RED**: diff the two sets; any unexplained entry on either side FAILS. *(Today: fails in both
  directions.)*
- **SC-006**: No document, docstring or comment states a policy count, enforcement status, or
  measured number that disagrees with its source.
  **RED**: compare each such claim to its source; any disagreement FAILS. *(Today: fails, including
  a package docstring advertising an abolished property.)*
- **SC-007**: Every number displayed by the frontend either varies with brain state or is labelled as
  unable to.
  **RED**: hold brain state fixed and vary it; any value that never changes and is unlabelled FAILS.
  *(Today: fails on at least one headline value.)*
- **SC-008**: Observing the brain through the frontend does not measurably slow the tick loop.
  **RED**: measure tick rate with the frontend polling and without; a significant difference FAILS.
- **SC-009**: The three gates are green at every commit produced by this feature.
  **RED**: any commit at which a gate is red FAILS.
- **SC-010**: The safety envelope files are byte-identical to their state at the start of this
  feature.
  **RED**: diff them; any change FAILS.
- **SC-011**: Every finding that drove a change has a recorded verification verdict, and every
  finding that did not survive verification is recorded as refuted rather than silently dropped.
  **RED**: any applied change with no verified finding behind it FAILS.

## Assumptions

- **"Users" here are the owner, future maintainers, and agents working in this tree.** This is a
  research codebase with no external end users; every scenario above is written for that audience.
- **The audit's findings are inputs, not conclusions.** Prior sweeps in this project refuted roughly
  a third of candidate findings on adversarial re-check, so FR-011 assumes a meaningful refutation
  rate rather than treating it as exceptional.
- **Deleting a marker is preferred to repairing one** where no reachable red exists, per the
  project's existing falsifier F11. Repair is chosen only where the criterion has a red and the
  defect is in how it is invoked.
- **Severing the consciousness self-report is not a retreat from the sentience goal.** It removes a
  painted finish line so the real one stays findable; inventing a criterion the brain's own state can
  move is explicitly in scope for the project, though building that criterion is a follow-on
  feature rather than this one.
- **Dependency additions are permitted and expected** under P7. Where a hand-rolled mechanism is
  found that a library covers, adopting the library is a legitimate remedy, subject to retiring the
  hand-rolled one.
- **The frontend is `substrate/cockpit.py` and what it serves.** It is treated as one surface,
  including the markup and script it emits, not only its module.
- **Scope is the tracked tree.** Untracked scratch directories are out of scope except where a
  tracked file depends on them.
