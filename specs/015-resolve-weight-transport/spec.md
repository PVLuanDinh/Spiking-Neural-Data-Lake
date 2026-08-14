# Feature Specification: Resolve the e-prop Weight Transport at `mlp_q`

**Feature Branch**: `015-resolve-weight-transport`
**Created**: 2026-08-14
**Status**: Draft
**Input**: Owner-proposed framework (Predictive E-prop + Spiking Forward DFA + three-factor
plasticity), with citations verified by retrieval on 2026-08-14 and one correction applied.

---

## Why this exists

Lane 017 (`017-tier0-invariant-and-p0`, 18 commits, pushed) closed Tier 0 items #1 and #3 and
corrected the e-prop adjudication's SCOPE from 2 sites to 4. **Item #2 — actually RESOLVING the
transport — was left open by owner decision** ("record scope, decide nothing"), because
`basal_ganglia/mlp_q.py` states in its own docstring that the choice "is a policy call, not an
edit." The owner has now supplied a framework and asked for this lane. **This feature is Tier 0
item #2, and closing it finishes Tier 0.**

### The four sites, and which this feature touches

| site | status at HEAD | in scope |
|---|---|---|
| `cerebellum/jepa_world_model.py` | CLOSED 2026-08-04; `_feedback_gz()` returns the identity half alone | **no** |
| `hippocampus/ng_snn.py::fit` | ruled NOT a violation — a direct solve, no gradient propagated | **no** |
| `basal_ganglia/mlp_q.py` | SELF-DECLARES transport; off the live tick (`mlp_q=False`) but runs every gate via the arm's self-check, and is one constructor argument from live | **PRIMARY** |
| `cortex/predictive_coding.py` | `w1.T @ s_e0` on EVERY wake tick; recorded as an open question with no verdict | **decision only** |

### The finding this feature exists to act on

`mlp_q` already measured feedback alignment and rejected it: cosine over 8 draws lands in
**[-0.472, +0.279], |cos| > 0.5 in 0 of 8**, against the invariant's `cos > 0.5` bar. But the module
records what that quantity IS — *"essentially cos(W2[a], B[a]) under the relu mask (`err` cancels),
so it is a property of the DRAW."*

**For a fixed random `B`, `cos(W2, B)` is near zero by construction.** The criterion measured WEIGHT
SYMMETRY — exactly what feedback alignment denies by design — and never tested whether the rule
learns. **It cannot discriminate, so the refutation resting on it is not evidence.**

The contrast proves it: at the jepa site the same-named criterion measured **0.971**, far above the
bar, because there both sides were computed through one forward pass. One criterion name, two
quantities, opposite verdicts.

### The owner's framework, verified — with the correction that matters

- **arXiv 2403.08804 is real**: *"Forward Direct Feedback Alignment for Online Gradient Estimates of
  Spiking Neural Networks"*, Bacho & Chu, 6 Feb 2024. Proposes SFDFA.
- **Correction**: the framework described SFDFA as using *"fixed, random feedback connections."* The
  paper says it **estimates** them — *"SFDFA estimates the weights between output and hidden neurons
  as feedback connections by propagating directional derivatives as spike grades during the
  inference."* Feedback CONVERGES toward the forward weights. That is the "Forward" in Forward-DFA.
- **The correction helps.** Converging feedback should make `cos(W2, B)` RISE during training, so
  SFDFA can satisfy the project's EXISTING invariant where fixed-random FA structurally cannot.
  **The invariant may not need amending at all** — which matters, because POLICIES.md says
  invariants are never relaxed by any policy.
- **The bioRxiv preprint is real**: *"Predictive E-prop…"*, Noè, Yamamoto, Katori, Sato,
  doi 10.64898/2026.02.12.705507. **But it does not address weight transport**: full-text search
  returns 0 hits for "weight transport", "weight symmetr", "symmetric", "feedback alignment" and
  "random feedback", against 30 for BPTT. It solves TEMPORAL credit assignment. Its silence on the
  transport objection is not endorsement.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The criterion can tell two rules apart (Priority: P1)

A reviewer asks "how do we know feedback alignment fails here?" and gets a criterion that could have
said otherwise. Today the answer is a number that no alignment rule could ever pass.

**Why this priority**: every later decision rests on it. Adopting or refuting a mechanism on an
instrument that cannot fail is the defect this project's policy set exists to prevent, and the
015 lane found the identical shape in the e-prop kernel cosine.

**Independent Test**: feed the criterion a rule known to be a good gradient estimate and one known
to be garbage; confirm it separates them. A criterion that scores both the same is deleted.

**Acceptance**:
1. **Given** the existing cosine, **When** its computation is stated in full, **Then** the record
   says which quantity it compares and why a fixed random draw cannot pass it.
2. **Given** a repaired criterion, **When** run against the transport gradient itself, **Then** it
   scores high; **When** run against a permuted/garbage feedback, **Then** it scores low.

### User Story 2 - The discriminating measurement (Priority: P1)

The project needs to know whether converging feedback behaves differently from fixed-random
feedback in THIS module, on THIS task — not in a paper.

**Why this priority**: it is the evidence the adoption decision consumes, and it is cheap because
the sweep harness already exists in the module's `main()`.

**Independent Test**: run both variants under one seed set and read the cosine trajectory across
training epochs rather than at the end only.

**Acceptance**:
1. **Given** the prediction registered BEFORE the run, **When** the sweep completes, **Then** the
   result is compared against the prediction as written, not against a revised one.
2. **Given** fixed-random feedback, **When** cosine is read each epoch, **Then** its trajectory is
   recorded whatever it shows.
3. **Given** converging feedback, **When** cosine is read each epoch, **Then** its trajectory is
   recorded whatever it shows.
4. **Given** either variant, **Then** held-out accuracy on both the non-linear and the linear task is
   recorded beside the cosine, so a rule that aligns but stops learning is visible.

### User Story 3 - One mechanism survives, or none does (Priority: P1)

If the measurement supports the new rule, it BECOMES the credit-assignment path and the transport
line is deleted. If it does not, the lane says so and stops.

**Why this priority**: this is the actual Tier 0 deliverable. It is also where the project's two
strongest failure modes meet — keeping both paths behind a flag (P2's additive failure), and
softening a pre-committed null after seeing the number.

**Independent Test**: count the surviving credit-assignment mechanisms in the module. If two are
reachable, it was not a resolution.

**Acceptance**:
1. **Given** a supporting measurement, **When** the new rule lands, **Then** exactly one
   credit-assignment path is reachable and the transport line is gone.
2. **Given** a null or contrary measurement, **When** the lane concludes, **Then** nothing is
   adopted, the null is recorded with its numbers, and the site's disposition is restated.
3. **Given** either outcome, **Then** the e-prop adjudication reflects what was decided.

### User Story 4 - The live site gets an adjudication (Priority: P2)

`cortex/predictive_coding.py` transposes a weight on every wake tick and has no verdict. The owner's
framework does not reach it.

**Why this priority**: it is the LIVE site, so it matters more than `mlp_q` in operational terms —
but it needs a decision, not an implementation, and the decision is the owner's.

**Independent Test**: a reader can state the argument in both directions from the record alone and
say which was chosen and why.

**Acceptance**:
1. **Given** the adjudication, **Then** the case FOR calling it a violation is stated (exact weight
   symmetry by construction — the same array, transposed).
2. **Given** the adjudication, **Then** the case AGAINST is stated (in Rao-Ballard `W1` is the
   GENERATIVE weight, `x_hat = W1 @ r1`, so `W1^T` is one matrix in two structural roles rather than
   a separate forward pathway being copied).
3. **Given** the adjudication, **Then** it does NOT rest on the Predictive E-prop preprint's silence.

### Edge Cases

- **The repaired criterion passes everything.** If it cannot be made to fail, it is deleted, not
  weakened — and the site keeps its current disposition.
- **The new rule aligns but does not learn.** Cosine rising while held-out accuracy falls is a
  refutation, not a success; both must be read together.
- **The new rule learns but does not align.** Then adoption WOULD require amending the invariant,
  which this lane may not do on its own authority — it becomes an owner decision with the numbers
  attached.
- **A single seed decides it.** The existing discipline is eight draws; one draw is a coin flip.
- **The measurement is confounded by the task.** `mlp_q`'s fixtures are a small XOR and a linear
  rule. A result that holds only there is reported as holding only there.
- **The trajectory is read from too few epochs.** Convergence claims need enough epochs to show a
  trend, and the epoch count must be stated with the result.

---

## Requirements *(mandatory)*

- **FR-001**: The record MUST state exactly which quantity the existing cosine compares, and why a
  fixed random feedback draw cannot pass it. The degeneracy MUST be recorded, not silently repaired.
- **FR-002**: Any criterion this feature keeps or adds MUST be shown to separate a known-good
  estimate from a known-bad one. A criterion that scores both alike MUST be deleted.
- **FR-003**: The predicted outcome of the discriminating measurement MUST be written down BEFORE
  the measurement is run, and the comparison MUST be against that written prediction.
- **FR-004**: The measurement MUST report the cosine trajectory ACROSS training, not a single
  end-of-run value, for both the fixed-random and the converging variant.
- **FR-005**: The measurement MUST report held-out accuracy on both fixtures beside the cosine, so
  alignment and learning can be judged together.
- **FR-006**: The measurement MUST sweep at least the eight feedback draws the module already uses,
  and report the range rather than a central value alone.
- **FR-007** *(deletion on a null)*: If the converging variant does not clear the bar, NOTHING is
  adopted, the null is recorded with its numbers, and the site's disposition is restated. This is
  pre-committed here, before any number is known, and MUST NOT be renegotiated afterwards.
- **FR-008** *(P2)*: If a new rule is adopted, exactly ONE credit-assignment path MUST remain
  reachable in the module. A second path behind a flag or a constructor argument does not satisfy
  this.
- **FR-009**: If a new rule is adopted, the capability the retired rule had MUST be measured before
  the deletion, and any loss recorded.
- **FR-010** *(P1)*: Any feedback scale introduced MUST be derived from an existing quantity or
  justified in place. No new bare constant.
- **FR-011**: The e-prop adjudication MUST be updated to reflect what was decided, including a null.
- **FR-012**: `cortex/predictive_coding.py` MUST receive an adjudication stating the argument in
  both directions. This feature MUST NOT pre-decide it, and MUST NOT rest it on the preprint's
  silence about transport.
- **FR-013** *(safety)*: The five `brainstem/` safety-envelope files MUST NOT be modified.
- **FR-014**: No dependency may be introduced. The module is stdlib with hand-written backprop.
- **FR-015** *(determinism)*: Any randomness MUST come from a per-instance generator seeded
  explicitly. No module-level seeding.
- **FR-016** *(measurement discipline)*: Every measurement command MUST carry a positive control in
  the same command, and no negative result may be reported from a command not shown able to produce
  a positive. Denominators MUST be asserted.
- **FR-017** *(citations)*: Any claim attributed to a paper MUST reflect what the paper says. The
  SFDFA correction (estimated, not fixed-random feedback) MUST be preserved wherever the mechanism
  is described.
- **FR-018** *(cites)*: References into actively-edited files MUST use a symbol or a distinctive
  marker, never a line number.

### Key Entities

- **The criterion** — the quantity compared, the two things compared, and the threshold. Its
  identity is the subject of FR-001/FR-002.
- **The feedback pathway** — how error reaches the hidden layer. Currently the transposed forward
  read-out weights. Candidates: fixed-random, converging-estimated.
- **The trajectory** — cosine and held-out accuracy as functions of training progress, per draw.
- **The adjudication record** — the e-prop verdict, which currently enumerates four sites and
  carries verdicts on two.

---

## Success Criteria *(mandatory)*

- **SC-001**: A reader can state, from the record alone, what the old cosine measured and why no
  alignment rule could pass it.
- **SC-002**: Every criterion this feature keeps has been demonstrated to fail on a constructed
  input, and that failure is recorded.
- **SC-003**: The prediction was recorded before the measurement, and both are in the record
  regardless of whether they agree.
- **SC-004**: The trajectory result distinguishes the two feedback variants, or explicitly reports
  that it could not — a null is a complete result.
- **SC-005**: At the end, the number of reachable credit-assignment mechanisms in `mlp_q` is exactly
  one, and it is stated which.
- **SC-006**: If a capability was lost, it is named and quantified.
- **SC-007**: The e-prop adjudication's four sites each carry either a verdict or an explicit
  "open, and why", with no site left implied.
- **SC-008**: The `brainstem/` diff across the whole feature is empty.
- **SC-009**: No new dependency appears in the dependency ledger.

---

## Non-Goals

1. **The jepa and ng_snn sites** — already adjudicated; not reopened.
2. **Implementing Predictive E-prop as a training regime** — a separate and much larger lane. This
   feature uses it only as context, and only where it actually applies.
3. **Any change to the five `brainstem/` safety-envelope files.**
4. **Amending the cross-cutting cosine invariant.** If the measurement forces that question, it is
   raised as an owner decision with the numbers attached — this feature does not amend it.
5. **Changing `predictive_coding.py`'s code.** This feature produces its adjudication only.

---

## Assumptions

- The module's existing fixtures (small non-linear and linear tasks, eight feedback draws) are the
  right scale for the discriminating measurement, because the prior FA result used them and a
  comparison must share its fixtures to be a comparison.
- "Converging feedback" is implementable in this module's stdlib, hand-written form. If it is not,
  that is itself a reportable finding rather than a reason to substitute a different mechanism.
- The owner decides any invariant amendment. This feature supplies evidence, not authority.
- Tier 0 is finished when item #2 has a decision on the record — adoption or null — not when a
  particular mechanism ships.

---

## Risks

- **R-001 — the repaired criterion is tuned to the answer.** Repairing a criterion while knowing
  which result is wanted is how a criterion gets built to pass. Mitigation: FR-003's pre-registered
  prediction, and FR-002's requirement that the criterion separate a known-good from a known-bad
  rule chosen independently of the candidate.
- **R-002 — the null gets renegotiated.** FR-007 is pre-committed before any number exists precisely
  because this is the decision that softens under a result nobody likes.
- **R-003 — the paper is treated as the evidence.** A verified citation says what a method does
  elsewhere, never that it works here. Only the measurement in this module decides.
- **R-004 — the live site is settled by analogy.** `predictive_coding.py` is a different mechanism
  on a different objective; a result at `mlp_q` does not transfer to it, in either direction.
