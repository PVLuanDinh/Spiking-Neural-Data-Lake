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
   policies carry no instruction any process can execute. *(Superseded 2026-08-18, HEAD
   f61e4a94: CI now runs FIVE `--assert` gates plus three `--self-check`s — specs/020 added
   `gradient_claim_gate` the day after this was measured — and the enforcement table's
   review-only set is FOUR policies: P0, P1, P2, P5, with e-prop PARTIALLY mechanized by two
   gates. The lesson is the finding itself: a pinned enforcement count rots in days, so the
   manifest DERIVES these counts and this spec stops asserting them.)* The constitution says as much in prose —
   P0, P1, P2, P5 and the e-prop invariant are "review-enforced", which means enforced by whoever
   remembers.
2. **Spec-kit does not reach it.** There are FIVE templates in `.specify/templates/`, and NONE
   references `POLICIES.md` — so every future feature is generated from documents that do not
   point at the governing text. *(Corrected 2026-08-18: the draft said four templates with
   constitution-template the exception; measured with the exact token, its one "policies" hit
   is a generic deployment-policies comment, and `checklist-template.md` exists. The finding is
   STRONGER than first written.)* The
   constitution REQUIRES that "every plan produced under Spec Kit MUST pass a Constitution Check
   against P0–P8 and the Standing Invariants before implementation", and **nothing runs that check**.
3. **The same claims live on three surfaces** — POLICIES.md, the constitution (12 references), and
   CLAUDE.md. *(Update 2026-08-18: specs/020 Phase 8 converted the constitution's e-prop and
   P7/P8 ENFORCEMENT-TABLE rows to pure pointers, closing that class for those rows. Known
   residue, recorded in specs/020-eprop-cosine-scope/census-baseline.md's Phase 8 CLOSED
   section: the per-principle P7/P8 status lines near constitution.md:218/:250 still say
   "specified but not built" — false since 2026-08-05, and exactly this story's material.)* The CLAUDE.md copy rotted three times in eight days and was DELETED rather than
   corrected a fourth time, because CLAUDE.md is gitignored so no commit, CI job or gate can ever see
   it. P2 applied to documentation: a claim living in two places gets corrected in one.
4. **There is no release concept and no pre-release audit.** Nothing asks "has this build been
   audited against every policy" before it lands.
5. **Hard budget constraint.** POLICIES.md measures **28,559 of a 29,100-word ceiling** enforced by
   `substrate/doc_budget_gate.py` — **541 words of headroom**. *(Superseded 2026-08-18: now
   **29,750 of 29,750 — ZERO headroom** — the ceiling was raised three times during specs/020
   Phase 8, each raise with its reason in the `doc_budget_gate.py` constant, which also records
   the debt this story exists to pay: "the e-prop block owes a CONSOLIDATION pass, after which
   this ceiling comes back DOWN." The constraint is HARDER than when written, not softer.)* Build instructions for nine policies do
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

## Clarifications

### Session 2026-08-18

- Q: When the first audit reports `safety_envelope` as a runnable gate no policy claims, fix by
  adding an enforcement-table row — before or after the first counted run? -> A: AFTER. Baseline
  the unclaimed census on the first counted run (the detector demonstrates its positive on real
  drift, not a fixture); the enforcement-table rows land as a later post-count doc commit with
  US4's doc lane, ratcheting the baseline to EMPTY. *(Corrected by /speckit-analyze 2026-08-18:
  the census is not 1 — measured THREE at HEAD (`safety_envelope`, `doc_budget_gate`,
  `ingest_guard`, each 0 POLICIES.md mentions), and the instrument itself joins the runnable
  surface the moment its own CI step lands, so it carries a NAMED, printed self-disposition.
  The baseline is the SET the first counted run reports — a number pinned in prose was the
  drift class this feature audits for, caught in its own spec.)*
- Q: How does agent dispatch DELIVER the manifest (FR-010) — hook or templates? -> A: BOTH,
  split by role: the existing POLICIES guard hook (user scope) becomes the mechanized injector,
  RUNNING `--manifest` fresh at dispatch — a thin caller with zero embedded policy content, so
  its ungated location holds nothing that can rot; the templates carry the repo-tracked
  canonical command the hook calls.
- Q: Does US4 touch the e-prop invariant block (biggest recorded debt, frozen yesterday after
  nine oscillating rounds)? -> A: COUNT FIRST. Measure per-section relocatable mass before any
  cut; spend from the largest non-e-prop sections first; touch e-prop only if SC-006 is
  otherwise unreachable. Byte-preserving rule wherever the cut falls: RELOCATION, never
  rewording — moved text lands byte-exact in the companion file, a pointer stays behind
  (Phase 8's oscillation came from rewrites, not moves).
- Q: Does `--certify` treat REVIEW_ONLY rows as blocking, certifiable-with-disclosure, or
  blocking-unless-attested? -> A: CERTIFIABLE-WITH-DISCLOSURE. Mechanized gates must PASS with
  zero NO_GATE / CANNOT_RUN / unclaimed-gate drift; the certificate lists every review-only
  policy as "not machine-decided" as a first-class part of the record — never green, never a
  fake red. SC-002's first-run refusal stays reachable through the real reds (UNCLAIMED_GATE:
  safety_envelope until its row lands, plus any true NO_GATE).
- Q: When does `--certify` run, given this project has no versioned releases? -> A: CI JOB ON
  EVERY PUSH TO MAIN — runs `--certify`, publishes the certification JSON as a job artifact,
  refusal loud in the job summary. Non-blocking initially (main is not gated on the
  safety_envelope row landing); flipping the job to required once it goes green is a recorded
  later owner decision.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Show me, honestly, which policies nothing enforces (Priority: P1)

A reviewer runs one command and sees every policy with its enforcement kind, its runnable gate
command, and whether that command exists and passes. **The first honest run shows every
review-only policy prominently as not-machine-decided — never green** — six when this was
written, FOUR (P0, P1, P2, P5) by the enforcement table at 2026-08-18; the audit derives the
number, never asserts it.

**Why this priority**: it is the "count its output FIRST" half of this project's own standing
invariant, and it delivers value even if no policy text ever changes — today nobody can answer "what
enforces P1?" without reading three documents. Every later story consumes this output.

**Independent Test**: run it; confirm mechanized policies are distinguishable from review-only
ones (the tool derives both counts), that review-only rows appear prominently as NOT
MACHINE-DECIDED rather than omitted, and that any `absolute`/`ratchet` policy without a command
is reported RED as NO_GATE.

**Acceptance Scenarios**:

1. **Given** the tree at HEAD, **When** the audit runs, **Then** all nine policies appear with an
   enforcement kind and a command-or-NONE, and the totals reconcile against the count of policies
   declared in POLICIES.md.
2. **Given** a policy of kind `absolute` or `ratchet` with no runnable command, **When** the audit
   runs, **Then** it is reported RED as NO_GATE, never silently skipped and never counted as
   passing; a `review-only` policy is REVIEW_ONLY — disclosed, never green, never a fake red
   (FR-004's predicate).
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
4. **Given** a push to main, **When** CI runs, **Then** the certification job runs `--certify`
   and publishes the record — a landing build cannot avoid being audited, even while the job
   is non-blocking.

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
   Relocation order is decided by a per-section mass COUNT taken first; the e-prop block is
   touched only if the SC-006 target is otherwise unreachable, and every move is byte-exact
   (relocation, never rewording).
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
- A runnable gate **no policy claims** (`UNCLAIMED_GATE` — live instances at HEAD:
  `safety_envelope`, `doc_budget_gate`, `ingest_guard`, each 0 POLICIES.md mentions) -> reported
  distinctly; the first counted run baselines the MEASURED SET, the claiming table rows are a
  post-count doc edit (see Clarifications), the baseline then ratchets to empty. The census
  needs a defined gate predicate and a named self-disposition for the instrument's own gate —
  both stated in the module and printed in every report, never silent.
- A policy is **legitimately review-only** (P1's "brain-native" is not fully machine-decidable) →
  `review-only` must be a first-class honest kind, not a synonym for green.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST produce a manifest with one row per policy declared in POLICIES.md,
  carrying: id, what it binds, its runnable gate command (or NONE), and its enforcement kind
  (`absolute` / `ratchet` / `review-only`). *(Amended 2026-08-18: the draft also required "its
  audit question" — struck with the reason recorded: POLICIES.md holds no derivable source for
  it, and hand-authoring one per row is the parallel hand-maintained copy FR-002 forbids. The
  audit question is operationalized by the row's command + kind.)*
- **FR-002**: The manifest MUST be DERIVED from POLICIES.md, never hand-maintained in parallel.
  Derivation is what makes drift structurally impossible rather than merely discouraged.
- **FR-003**: System MUST fail loudly when the manifest cannot be derived, or when its row count does
  not reconcile against the policies declared in POLICIES.md. The reconciliation identity:
  parent `## P<n>` headings + TABLE-ONLY policies (declared in the enforcement table with no
  heading of their own — a first-class row kind, not an error) == policy table rows. Measured at
  HEAD f61e4a94: 9 parents + 1 table-only (`e-prop`) == 10 rows; the naive headings==rows form
  is arithmetically FALSE on the real tree and was caught by /speckit-analyze before build.
- **FR-004**: A release audit MUST run every row's command and MUST refuse to certify when any policy
  is missing a command, red, or unrunnable. The predicate, stated once: a policy whose enforcement
  kind is `absolute` or `ratchet` and which carries no command is NO_GATE and blocks; a policy
  whose kind is `review-only` is REVIEW_ONLY and never blocks — "missing a command" means the
  former, never the latter. REVIEW_ONLY rows do NOT block certification —
  they are disclosed in the certificate (FR-008); an UNCLAIMED_GATE drift finding DOES block
  until the gate is claimed (Clarifications, 2026-08-18).
- **FR-005**: "Cannot run" MUST be reported distinctly from "ran and failed".
- **FR-006**: A certification MUST record what it ran, so it can be re-derived rather than trusted.
- **FR-007**: The audit MUST fail loudly on an empty or truncated policy walk.
- **FR-008**: `review-only` MUST be a first-class kind that is neither green nor a violation — it
  states honestly that no machine decides this policy.
- **FR-009**: ALL FIVE spec-kit templates (`plan`, `spec`, `tasks`, `constitution`, `checklist`)
  MUST point at the manifest (measured 2026-08-18: none references `POLICIES.md`; the draft
  scoped this to three on a false premise), and the Constitution Check the constitution already
  mandates MUST become runnable.
- **FR-010**: Agent dispatch MUST deliver the MANIFEST rather than a prose summary of it —
  mechanized by the existing session-start/dispatch guard hook regenerating `--manifest` output
  at dispatch time (thin caller, no embedded content; user-scope location disclosed as ungated),
  with the canonical command repo-tracked in the templates (FR-009).
- **FR-011**: POLICIES.md MUST NOT grow (measured from implementation start, 29,750 words at
  2026-08-18 — it grew ~1,200 words between this spec's writing and its start, recorded as
  consolidation debt in `doc_budget_gate.py`; words as measured by `doc_budget_gate`'s counter —
  the budget gate is the arbiter, not `wc -w`). Duplication with the constitution is removed and worked cases
  move to a companion file with a pointer left in place; no measurement or incident may be lost.
- **FR-012**: The manifest and audit MUST be built and their output COUNTED before any policy text is
  edited. Three prior policy proposals died at that bar.
- **FR-013**: System MUST NOT change the CONTENT or meaning of any policy, MUST NOT touch the five
  `brainstem/` safety-envelope files, and MUST NOT relax any existing gate.
- **FR-014**: Certification runs as a CI job on every push to main, publishing the
  certification record as a job artifact with any refusal loud in the job summary;
  non-blocking initially, with flip-to-required a recorded owner decision once green
  (Clarifications, 2026-08-18).

### Key Entities

- **Policy**: one governing rule. Attributes: id, binds-what, enforcement kind, gate command or NONE.
- **Manifest**: the derived, machine-readable set of policy rows. Regenerated, never edited.
- **AuditResult**: per policy — `PASS` / `FAIL` / `CANNOT_RUN` / `NO_GATE` / `REVIEW_ONLY`.
- **Certification**: the recorded output of a release audit, including what was run.

### Audit against tree state (2026-08-18, HEAD f61e4a94)

Written 2026-08-17; ten commits landed the next day (specs/020 complete + Phase 8), touching
exactly this spec's subject matter. Re-measured before starting; supersessions are inline above.
Three facts the plan MUST ingest:

1. **POLICIES.md's own head matter — the enforcement table plus the gate-output table, located
   by header anchor, never line numbers — is a hand-maintained proto-manifest** — per-policy rows, mechanized-vs-review columns, gate names, even its own
   drift-correction narrative. P2 binds: the manifest DERIVES from or SUPERSEDES that table;
   a second hand-edited copy beside it is the additive failure this project names.
2. **Two gates postdate this spec**: `substrate/gradient_claim_gate.py` (asserts in CI) and
   `substrate/safety_envelope.py` (registered in `run_selfchecks.py`). FR-004's expectation
   that the manifest discovers gates rather than listing them is what makes this drift free.
3. **Templates re-verified blind at HEAD**: 0 exact `POLICIES.md` references in ALL FIVE
   templates (the earlier "constitution-template: 1" was a case-insensitive substring match on
   generic "policies"). Finding 2 is stronger than written.

## Success Criteria *(mandatory)*

- **SC-001**: 100% of policies declared in POLICIES.md appear in the manifest; the count reconciles.
- **SC-002**: The first release audit reports **every review-only policy prominently as NOT
  MACHINE-DECIDED — never green** — and refuses to certify on the real reds (the UNCLAIMED_GATE
  census — three members measured at HEAD, re-derived by the tool on every run — plus any true
  NO_GATE); the derived count reconciles against POLICIES.md's enforcement table on the same
  tree (four — P0, P1, P2, P5 — at 2026-08-18; six when this spec was written). An all-green
  first run is treated as proof the audit is blind.
- **SC-003**: A reviewer can answer "what enforces this policy, and did it pass?" for any policy in
  under one minute, from one command, without reading three documents.
- **SC-004**: A build with any policy red cannot be certified — demonstrated by a constructed failure.
- **SC-005**: A newly generated spec/plan/tasks carries a runnable policy check.
- **SC-006**: POLICIES.md is measurably smaller than 28,559 words — as counted by
  `substrate/doc_budget_gate.py`'s own counter, the named arbiter, since two plausible counters
  disagree by 502 words on this very file (42% of the target) — with zero measurements or worked
  cases lost, each reachable from its pointer. *(Re-anchored 2026-08-18: current size is
  29,750, so this target now means recovering at least 1,191 words. It aligns with the
  consolidation debt already recorded in `doc_budget_gate.py`; when the size comes down, the
  CEILING comes down with it — a shrink that leaves the old ceiling standing just re-opens the
  headroom this budget exists to deny.)*
- **SC-007**: Adding a tenth policy to POLICIES.md makes it appear in the manifest with no list edited.
- **SC-008**: The manifest+audit exist and their counts are published in a commit that precedes any
  policy-text edit — verifiable from commit order.

## Assumptions

- **This feature does NOT build the missing gates** (six at writing, four at 2026-08-18 — the audit derives the count). It makes their absence visible and audited.
  Building each is follow-on work with its own measurement; some (P1 "brain-native") may never be
  fully machine-decidable, which is why `review-only` is a first-class kind.
- **"Release" means landing on `main`**, consistent with the P4c amendment that a build which has not
  reached main is not a build this project can claim.
- **The manifest is generated AND committed**, with a check that it is in sync with POLICIES.md —
  the `p4c_ratchet` baseline pattern. A generated-but-uncommitted artifact cannot be reviewed.
- The audit may run gates in a fast subset by default with the full set on demand, provided
  "not run" is never reported as "passed".
- `docs/POLICIES.md` remains the authority; the manifest is derived from it and never overrides it.
- The ceiling (29,100 at writing; 29,750 since Phase 8) is a hand-set "+5% over measured" snapshot, not a law, and may be raised in the
  same commit with its reason stated.
