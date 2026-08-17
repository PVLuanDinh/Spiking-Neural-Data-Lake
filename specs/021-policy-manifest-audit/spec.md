# Feature Specification: Make POLICIES.md the operational spine — policy manifest + release audit

**Feature Branch**: `021-policy-manifest-audit`

**Created**: 2026-08-17

**Status**: Draft

**Input**: Owner: "Make sure every policy in POLICIES.md is written with clear build instructions
around all Claude processes for SNDL, everything, spec-kit, CLAUDE.md, everything has to build around
the POLICIES.md so that each build is audited before release, every agent reach it as well."

## Why this exists

`docs/POLICIES.md` is the authoritative governing text and **nothing mechanically enforces most of
it**. Five findings, all measured at HEAD on 2026-08-17:

1. **NINE policies are declared; THREE runnable gate commands are named in the entire document.** Six
   policies carry no instruction any process can execute. The constitution says as much in prose —
   P0, P1, P2, P5 and the e-prop invariant are "review-enforced", which means enforced by whoever
   remembers.
2. **Spec-kit does not reach it.** Of four templates in `.specify/templates/`, only
   `constitution-template.md` mentions POLICIES.md. `plan`, `spec` and `tasks` never do — so every
   future feature is generated from documents that do not point at the governing text. The
   constitution REQUIRES that "every plan produced under Spec Kit MUST pass a Constitution Check
   against P0–P8 and the Standing Invariants before implementation", and **nothing runs that check**.
3. **The same claims live on three surfaces** — POLICIES.md, the constitution (12 references), and
   CLAUDE.md. The CLAUDE.md copy rotted three times in eight days and was DELETED rather than
   corrected a fourth time, because CLAUDE.md is gitignored so no commit, CI job or gate can ever see
   it. P2 applied to documentation: a claim living in two places gets corrected in one.
4. **There is no release concept and no pre-release audit.** Nothing asks "has this build been
   audited against every policy" before it lands.
5. **Hard budget constraint.** POLICIES.md measures **28,559 of a 29,100-word ceiling** enforced by
   `substrate/doc_budget_gate.py` — **541 words of headroom**. Build instructions for nine policies do
   not fit; writing them inline FAILS the budget gate before helping anything. The other four
   budgeted docs have 23–108 words each and cannot absorb spillover.

**Owner decision recorded (2026-08-17):** a compressed, machine-only rewrite of POLICIES.md that only
the model can parse was CONSIDERED AND REJECTED. It removes the owner's ability to audit the auditor
— the owner grants exemptions, makes P2 adjudications, and approves safety-envelope changes — and
compression does not produce reliability: every failure this document has had was a claim outliving
what it described, which survives compression at higher density. Counter-evidence measured the same
day: the worked cases a compression pass cuts first (the `mlp_q` feedback-alignment incident, the
`p6_gate` baseline laundering) are precisely what changed two decisions. **The manifest is the
machine-first artifact instead, DERIVED from POLICIES.md so it cannot drift.**

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Show me, honestly, which policies nothing enforces (Priority: P1)

A reviewer runs one command and sees every policy with its enforcement kind, its runnable gate
command, and whether that command exists and passes. **The first honest run shows six policies RED
for having no command at all.**

**Why this priority**: it is the "count its output FIRST" half of this project's own standing
invariant, and it delivers value even if no policy text ever changes — today nobody can answer "what
enforces P1?" without reading three documents. Every later story consumes this output.

**Independent Test**: run it; confirm the three policies with commands are distinguishable from the
six without, and that the six are reported RED rather than omitted.

**Acceptance Scenarios**:

1. **Given** the tree at HEAD, **When** the audit runs, **Then** all nine policies appear with an
   enforcement kind and a command-or-NONE, and the totals reconcile against the count of policies
   declared in POLICIES.md.
2. **Given** a policy with no runnable command, **When** the audit runs, **Then** it is reported RED
   with the reason "no gate", never silently skipped and never counted as passing.
3. **Given** a first run that reports every policy green, **When** a reviewer inspects it, **Then**
   that is treated as evidence the audit is BLIND, not that the tree is clean.

---

### User Story 2 - A build cannot be certified without being audited (Priority: P2)

A release audit runs every manifest row's command and refuses to certify when any policy is missing a
command, red, or unrunnable. "Audited before release" becomes a thing that happens rather than a thing
that is promised.

**Why this priority**: this is the owner's actual request. It depends on Story 1 only for its input.

**Independent Test**: point it at a tree where one gate is made to fail; confirm it refuses to
certify and names which policy.

**Acceptance Scenarios**:

1. **Given** any policy red or unrunnable, **When** the release audit runs, **Then** it refuses to
   certify and names the policy and the reason.
2. **Given** a gate command that does not exist, **When** it runs, **Then** "cannot run" is reported
   distinctly from "ran and failed" — collapsing those is a failure this project has already had.
3. **Given** every policy green, **When** it runs, **Then** it certifies AND records what it ran, so
   a certification can be re-derived rather than trusted.

---

### User Story 3 - Every process that generates work points at the policy (Priority: P3)

The spec-kit templates that generate future features reference the manifest, and the Constitution
Check the constitution already mandates becomes runnable. Agents receive the manifest.

**Why this priority**: it stops NEW work being generated blind. It is P3 only because Stories 1–2 must
exist first for there to be anything to point at.

**Independent Test**: generate a new feature from the templates and confirm the policy check appears
and can be executed, not merely read.

**Acceptance Scenarios**:

1. **Given** a new spec/plan/tasks generated from the templates, **When** a reviewer reads it,
   **Then** it points at the manifest and carries a runnable policy check.
2. **Given** a subagent dispatched for any substantive task, **When** it starts, **Then** it receives
   the MANIFEST, not a prose summary — a summary is the copy that drifts.

---

### User Story 4 - The governing document gets smaller (Priority: P4)

Duplication with the constitution is removed and incident forensics move to a companion file, so
POLICIES.md keeps the rule and a pointer while the evidence is preserved elsewhere.

**Why this priority**: lowest because it changes no enforcement. But the 541-word ceiling currently
arbitrates **whether policy can be written at all**, which is the shape P1 would flag in code.

**Acceptance Scenarios**:

1. **Given** the deduplication, **When** the budget gate runs, **Then** POLICIES.md is measurably
   smaller and no measurement or worked case has been lost — each is reachable from its pointer.
2. **Given** the ceiling still binds after deduplication, **When** it is raised, **Then** the reason
   is stated in the same commit.

---

### Edge Cases

- A policy declares a gate command that **does not exist** → distinct from "ran and failed".
- A gate is **slow** (the full suite is ~30 min) → the audit must not become unrunnable in practice,
  or it will be skipped and its certification will mean nothing.
- The manifest and POLICIES.md **disagree** → the manifest is derived, so disagreement means the
  derivation broke; it must fail loudly rather than serve stale rows.
- POLICIES.md gains a **tenth policy** → it appears in the manifest without anyone editing a list.
- The audit finds **zero policies** → a dead walk and a clean tree look identical; must fail loudly.
- A policy is **legitimately review-only** (P1's "brain-native" is not fully machine-decidable) →
  `review-only` must be a first-class honest kind, not a synonym for green.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST produce a manifest with one row per policy declared in POLICIES.md,
  carrying: id, what it binds, its runnable gate command (or NONE), its audit question, and its
  enforcement kind (`absolute` / `ratchet` / `review-only`).
- **FR-002**: The manifest MUST be DERIVED from POLICIES.md, never hand-maintained in parallel.
  Derivation is what makes drift structurally impossible rather than merely discouraged.
- **FR-003**: System MUST fail loudly when the manifest cannot be derived, or when its row count does
  not reconcile against the policies declared in POLICIES.md.
- **FR-004**: A release audit MUST run every row's command and MUST refuse to certify when any policy
  is missing a command, red, or unrunnable.
- **FR-005**: "Cannot run" MUST be reported distinctly from "ran and failed".
- **FR-006**: A certification MUST record what it ran, so it can be re-derived rather than trusted.
- **FR-007**: The audit MUST fail loudly on an empty or truncated policy walk.
- **FR-008**: `review-only` MUST be a first-class kind that is neither green nor a violation — it
  states honestly that no machine decides this policy.
- **FR-009**: The spec-kit `plan`, `spec` and `tasks` templates MUST point at the manifest, and the
  Constitution Check the constitution already mandates MUST become runnable.
- **FR-010**: Agent dispatch MUST deliver the MANIFEST rather than a prose summary of it.
- **FR-011**: POLICIES.md MUST NOT grow. Duplication with the constitution is removed and worked cases
  move to a companion file with a pointer left in place; no measurement or incident may be lost.
- **FR-012**: The manifest and audit MUST be built and their output COUNTED before any policy text is
  edited. Three prior policy proposals died at that bar.
- **FR-013**: System MUST NOT change the CONTENT or meaning of any policy, MUST NOT touch the five
  `brainstem/` safety-envelope files, and MUST NOT relax any existing gate.

### Key Entities

- **Policy**: one governing rule. Attributes: id, binds-what, enforcement kind, gate command or NONE.
- **Manifest**: the derived, machine-readable set of policy rows. Regenerated, never edited.
- **AuditResult**: per policy — `PASS` / `FAIL` / `CANNOT_RUN` / `NO_GATE` / `REVIEW_ONLY`.
- **Certification**: the recorded output of a release audit, including what was run.

## Success Criteria *(mandatory)*

- **SC-001**: 100% of policies declared in POLICIES.md appear in the manifest; the count reconciles.
- **SC-002**: The first release audit reports **six policies with no gate** as RED. An all-green first
  run is treated as proof the audit is blind.
- **SC-003**: A reviewer can answer "what enforces this policy, and did it pass?" for any policy in
  under one minute, from one command, without reading three documents.
- **SC-004**: A build with any policy red cannot be certified — demonstrated by a constructed failure.
- **SC-005**: A newly generated spec/plan/tasks carries a runnable policy check.
- **SC-006**: POLICIES.md is measurably smaller than 28,559 words, with zero measurements or worked
  cases lost — each reachable from its pointer.
- **SC-007**: Adding a tenth policy to POLICIES.md makes it appear in the manifest with no list edited.
- **SC-008**: The manifest+audit exist and their counts are published in a commit that precedes any
  policy-text edit — verifiable from commit order.

## Assumptions

- **This feature does NOT build the six missing gates.** It makes their absence visible and audited.
  Building each is follow-on work with its own measurement; some (P1 "brain-native") may never be
  fully machine-decidable, which is why `review-only` is a first-class kind.
- **"Release" means landing on `main`**, consistent with the P4c amendment that a build which has not
  reached main is not a build this project can claim.
- **The manifest is generated AND committed**, with a check that it is in sync with POLICIES.md —
  the `p4c_ratchet` baseline pattern. A generated-but-uncommitted artifact cannot be reviewed.
- The audit may run gates in a fast subset by default with the full set on demand, provided
  "not run" is never reported as "passed".
- `docs/POLICIES.md` remains the authority; the manifest is derived from it and never overrides it.
- The 29,100 ceiling is a hand-set "+5% over measured" snapshot, not a law, and may be raised in the
  same commit with its reason stated.
