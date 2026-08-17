# Feature Specification: Scope the e-prop cosine clause, and prove the criterion can discriminate before it binds

**Feature Branch**: `020-eprop-cosine-scope`

**Created**: 2026-08-17

**Status**: Draft

**Input**: User description: "Any learning rule that represents the human brain should be allowed (no backprop). Why does it have to be cosine > 0.5? Does it violate P1 in the POLICIES.md?" -> owner directed: spec it.

## Why this exists

The e-prop cross-cutting invariant carries a numeric admission bar -- `cosine > 0.5` against an
independent reverse-mode gradient -- and three separate problems sit on it. All four findings below
were measured against HEAD on 2026-08-17, not recalled.

1. **Its SCOPE is unwritten, and the shorthand has hardened into a constraint nobody wrote.** The
   invariant reads *"Any backprop consideration pivots to e-prop ... verified cosine > 0.5"*. That is a
   PIVOT rule -- it governs work that was about to use backprop. It is not a requirement on every
   learning rule. The governed tree ships roughly 22 local rules (R-STDP, BCM, Oja/Sanger,
   Tsodyks-Markram, Vogels iSTDP, Benna-Fusi, ReSuMe, tempotron, triplet-STDP, Nessler STDP-EM, weight
   perturbation, NACA and others) and **none carries a gradient cosine**. None is in violation. Yet
   the nickname "e-prop only" is now being read as a rule: `specs/019 [B2.04]`'s own title says
   "(e-prop only)". **Owner ruling 2026-08-17: any brain-plausible non-backprop rule is allowed.**

2. **The bar has already failed to discriminate once, and POLICIES.md records the incident.** In the
   `mlp_q` feedback-alignment case the hidden-gradient cosine swept to `[-0.472, +0.279]` with
   `|cos| > 0.5` in **0 of 8** draws. That was not evidence the rule was wrong. The measurement
   reduced to `cos(W2[a], B[a])` -- it asked whether the feedback is PARALLEL to the forward weights,
   *"which is the weight symmetry feedback alignment exists to deny"*. POLICIES.md's own conclusion:
   *"An instrument no alignment rule could ever pass is not evidence against alignment rules."* It
   came within one decision of amending the invariant for the wrong reason.

3. **The instrument that would certify any of this is currently suspect.** `cortex/eprop_kernel` has
   ZERO occurrences of `ebar`/`zbar` and computes its readout gradient as `g_out += err.t() @ z` on
   raw `z`, while BOTH shipped encoders (`cortex/eprop_shd`, `cortex/eprop_nmnist`) use the
   kappa-filtered form (15 occurrences each). The only mechanization the e-prop invariant has
   validates a factorization neither shipped encoder runs.

4. **The number is undeived.** `0.5` is *permitted* under P1's honesty guard case (a) -- a fixed point
   on a normalized, dimensionless output, cosine being bounded in `[-1,1]`. **This is NOT a P1
   violation, and the spec says so plainly so the question is not re-litigated.** But no sweep, null
   distribution, or measured separation justifies `0.5` specifically. It is fixed-and-permitted, not
   fixed-and-justified.

**The binding process constraint:** this project's own cross-cutting invariant is *"adding a policy
means building its gate and COUNTING its output FIRST"* -- three prior policy proposals died at that
bar. Therefore **no POLICIES.md wording may change until the checks below exist and their output has
been counted against the real tree.** That ordering is a requirement of this feature, not a
suggestion, and it is what makes User Story 1 the MVP.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tell me which rules the bar even applies to, and count them (Priority: P1)

A reviewer looking at any learning rule in the tree can determine, from a re-derived census rather
than from memory, whether that rule owes a gradient cosine at all. The census reports the count of
rules that claim gradient approximation, the count that do not, and the count currently mis-scoped
(treated as owing a cosine while claiming no gradient, or the reverse).

**Why this priority**: It is the "count its output FIRST" half of the project's own gate rule, and it
delivers value even if the invariant's wording never changes -- today nobody can answer "does this
rule owe a cosine?" without re-reading the module. It is also the input every later story needs.

**Independent Test**: Run the census against the tree and confirm a known non-claimant (an STDP
variant) is classified as owing nothing, and a known claimant (the e-prop reference) is classified as
owing a cosine. Both classifications must be derived from the tree, not from a hand-maintained list.

**Acceptance Scenarios**:

1. **Given** the governed tree at HEAD, **When** the census runs, **Then** every learning rule is
   classified as owing-a-cosine or owing-none with zero unclassified, and the totals are printed.
2. **Given** a rule that performs a purely local update and claims no gradient approximation, **When**
   the census classifies it, **Then** it is reported as owing no cosine and is NOT a violation.
3. **Given** a new rule added to the tree that claims gradient approximation, **When** the census
   re-runs, **Then** the owing-a-cosine count rises without any list being hand-edited.

---

### User Story 2 - Refuse to let a criterion condemn a rule it could never pass (Priority: P2)

Before the cosine bar is treated as binding on a rule, the criterion is shown able to DISCRIMINATE
for that rule: a constructed aligned case clears the bar and a constructed non-aligned case falls
below it. If no constructible input makes the measurement fail, the instrument is reported as
falsified -- not the rule.

**Why this priority**: This is the `mlp_q` lesson made mechanical. Without it the bar can produce a
confident false verdict, and it already did once. It depends on Story 1 only for scoping, so it can
be built and demonstrated on a single rule.

**Independent Test**: Reconstruct the `mlp_q` feedback-alignment configuration, run the
discriminating-power check against it, and confirm the check reports INSTRUMENT-FALSIFIED rather than
RULE-FAILS. That is a historical case with a known correct answer, so it is a reachable red for the
checker itself.

**Acceptance Scenarios**:

1. **Given** a rule that owes a cosine, **When** the check runs, **Then** it reports a PAIR -- the
   aligned measurement and the non-aligned measurement -- and both are recorded, never just a verdict.
2. **Given** a rule whose measurement cannot be driven below the bar by any constructed input,
   **When** the check runs, **Then** it reports the INSTRUMENT as falsified and the rule as
   unjudged.
3. **Given** a rule whose measurement cannot be driven ABOVE the bar by any constructed input (the
   `mlp_q` shape), **When** the check runs, **Then** it likewise reports the instrument falsified and
   does not record a rule failure.

---

### User Story 3 - Measure the rule that actually ships (Priority: P3)

The gradient the bar compares against is computed from the same factorization the shipped encoders
run. **Owner decision 2026-08-17: reconciliation is REQUIRED in this feature, not disclosed and
deferred.** A correctly-scoped bar measured by the wrong instrument is not an improvement, so the
feature is not done while the reference gradient and the shipped rule disagree.

**Why this priority**: P3 is a BUILD-ORDER position, not a statement that it is optional -- this
story must ship for the feature to be complete. It sits third because Stories 1 and 2 are the
evidence for it: Story 2's discriminating-power check is expected to FLAG this mismatch on its own,
and fixing an instrument while holding the detector that catches the defect is strictly better than
editing it first and hoping. Do not reorder this ahead of Story 2 without a reason.

**Independent Test**: Perturb a shipped encoder's update rule and confirm the criterion's output
moves. That is a reachable red for the reconciliation itself, and it is the check that inspection
cannot substitute for -- two factorizations can look equivalent and score differently.

**Acceptance Scenarios**:

1. **Given** the reference gradient and a shipped encoder, **When** their factorizations are
   compared, **Then** they agree -- or, if agreement proves impossible, the divergence and its
   measured consequence are recorded on the criterion and that record is part of the deliverable.
2. **Given** a perturbation to a shipped encoder's rule, **When** the cosine criterion runs, **Then**
   the measurement moves -- proving the criterion is attached to the rule that ships.
3. **Given** the reconciled criterion, **When** it is re-run against the encoders unchanged, **Then**
   any movement in the reported cosine versus the pre-reconciliation value is disclosed rather than
   silently absorbed -- the number changing IS the finding.

---

### User Story 4 - Say plainly what the number is (Priority: P4)

The `0.5` figure is either derived from a measured separation between aligned and null gradients, or
recorded as an undeived convention with its P1 case-(a) justification stated in the same place. It is
not left implicit.

**Why this priority**: Lowest because it changes no verdict today -- but leaving an unexplained
number on an admission bar is how the next reviewer re-opens the P1 question from scratch, which is
exactly what happened to produce this spec.

**Independent Test**: A reader of the invariant can state where `0.5` came from without reading code.

**Acceptance Scenarios**:

1. **Given** the invariant text, **When** a reviewer asks why the bar is `0.5`, **Then** either a
   measured separation or an explicit "convention, permitted under P1(a), not derived" answer is
   present.

---

### Edge Cases

- A rule that learns but has **no time axis**, so a reverse-mode temporal gradient is undefined for
  it. It cannot owe a temporal-gradient cosine; the census must classify it rather than crash or
  silently omit it.
- A rule that is **default-off with zero live callers**. It should not be forced into compliance work
  it cannot charge -- but it must not vanish from the census either, or the count understates.
- A rule whose reference gradient **cannot be computed in the gate environment** (no GPU, missing
  optional dependency). Reporting "no cosine available" must be distinguishable from "cosine measured
  and failed" -- collapsing those two is the failure this project has hit before.
- The census finds **zero** rules owing a cosine. That must be reported as a suspicious result and
  fail loudly, because an empty census and a broken walk look identical.
- A rule claims gradient approximation in prose but its code performs no gradient-like update at all.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST classify every learning rule in the governed tree as either owing a
  gradient cosine or owing none, with zero unclassified, and MUST derive the classification from the
  tree rather than a hand-maintained list.
- **FR-002**: System MUST base "owes a cosine" on whether the rule CLAIMS to approximate a gradient,
  because that is the claim the measurement can check. A purely local rule making no such claim owes
  no cosine and MUST NOT be reported as a violation.
- **FR-003**: System MUST, for any rule the bar is applied to, produce a discriminating-power result
  consisting of BOTH an aligned measurement and a non-aligned measurement, and MUST record both.
- **FR-004**: System MUST report INSTRUMENT-FALSIFIED (and leave the rule unjudged) when no
  constructed input can move the measurement across the bar in either direction.
- **FR-005**: System MUST distinguish "no cosine available in this environment" from "cosine measured
  and below the bar" in its output.
- **FR-006**: System MUST fail loudly rather than report success when the census finds zero rules, or
  when its walk cannot enumerate the tree.
- **FR-007**: System MUST print its counts and MUST be runnable against the real tree before any
  governing-document wording is changed.
- **FR-008**: The governing document MUST state the scope of the cosine clause explicitly, and MUST
  state that a rule which cannot pass the criterion by construction has falsified the instrument
  rather than itself.
- **FR-009**: The governing document MUST record the provenance of the numeric bar -- derived, or
  convention-with-justification.
- **FR-010**: System MUST NOT change any learning rule's implementation, MUST NOT relax the
  backprop/weight-transport prohibition, and MUST NOT touch the safety envelope.
- **FR-011**: The reference gradient used by the criterion MUST be RECONCILED with the factorization
  the shipped encoders run -- reconciliation is in scope for this feature, decided by the owner on
  2026-08-17. A disclosure alone is not sufficient. If reconciliation proves impossible, the
  divergence and its measured consequence MUST be recorded on the criterion itself and the feature is
  not complete until that record exists.
- **FR-012**: The reconciliation MUST be demonstrated by measurement, not by inspection: perturbing a
  shipped encoder's update rule MUST move the criterion's output. A reconciliation that changes no
  measured value has not been shown to attach the criterion to the shipped rule.

### Key Entities

- **Learning rule**: a mechanism that updates weights or thresholds from experience. Attributes: the
  module it lives in, whether it claims gradient approximation, whether it is reachable from a live
  path.
- **Gradient claim**: a rule's assertion that its update approximates a gradient. This is what makes
  the cosine measurable; its absence is a legitimate state, not a gap.
- **Discriminating-power result**: the aligned/non-aligned measurement pair for one rule, plus the
  verdict -- rule-judged, or instrument-falsified.
- **Census**: the counted classification of every learning rule at a point in time, re-derived per
  run.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of learning rules in the governed tree are classified, with zero unclassified, and
  the total reconciles against an independent count of learning rules.
- **SC-002**: For every rule the bar is applied to, both an aligned and a non-aligned measurement are
  on record -- no rule is judged on a single number.
- **SC-003**: The historical feedback-alignment case is classified as instrument-falsified rather than
  rule-failed, matching the decision already reached by hand.
- **SC-004**: A reviewer can answer "does this rule owe a cosine, and why?" for any rule in under one
  minute without reading the rule's source.
- **SC-005**: The count of currently mis-scoped rules is published before any governing-document
  wording changes -- verifiable from commit order.
- **SC-006**: Zero learning rules are newly reported as violations by this work. A rule that was
  compliant before and is reported failing after indicates the scoping is wrong, not the rule.
- **SC-007**: The checks fail loudly on an empty or truncated walk, demonstrated by a constructed
  case.
- **SC-008**: Any change in a shipped encoder's measured cosine caused by reconciling the reference
  gradient is reported with its before and after values. Zero silently-absorbed movements.

## Assumptions

- The owner ruling of 2026-08-17 stands: **any brain-plausible non-backprop learning rule is
  permitted**. This feature clarifies scope; it does not widen or narrow which rules are allowed.
- `cosine > 0.5` does **not** violate P1 -- it falls under P1's stated exception (a), a fixed point on
  a normalized dimensionless output. This is settled and is not re-adjudicated here.
- This feature does **not** force existing rules into compliance. It publishes the count of
  mis-scoped rules; remediating any individual rule is follow-on work with its own measurement.
- The prohibition on backprop, BPTT and weight transport is unchanged and out of scope.
- `specs/019 [B2.04]` depends on this landing first, because its defence is a cosine measured by the
  instrument Story 3 reconciles. Reconciliation being IN SCOPE here (owner, 2026-08-17) means B2.04
  gains a trustworthy instrument as part of this feature rather than inheriting a suspect one.
- **SC-008 is the risk this decision accepts**: reconciling the reference gradient may MOVE the
  reported cosine for the shipped encoders. If a rule that measured above the bar now measures below
  it, that is a real finding about the rule and must be reported as such -- not absorbed, and not
  treated as a reason to abandon the reconciliation. This spec does not predict the direction.
- Rules living outside the governed tier are counted by the census but are not required to reach
  compliance in this feature.
- The gate environment has no GPU, so any measurement must be reproducible on CPU or must report
  itself unavailable rather than failing the build.
