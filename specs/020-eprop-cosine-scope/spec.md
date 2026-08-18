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

3. **A reference instrument disagrees with the rule it sits beside — but this row overstated it, and
   the correction is recorded rather than edited away.** `cortex/eprop_kernel` has ZERO occurrences of
   `ebar`/`zbar` and computes its readout gradient as `g_out += err.t() @ z` on raw `z`, while BOTH
   shipped encoders (`cortex/eprop_shd`, `cortex/eprop_nmnist`) use the kappa-filtered form (15
   occurrences each). That much is confirmed.

   **What this row originally concluded was MEASURABLY FALSE** (Phase 0, see research.md R0). It said
   *"the only mechanization the e-prop invariant has validates a factorization neither shipped encoder
   runs."* There are **three** mechanizations, all registered TORCH_GATED in `run_selfchecks.py` and
   all running in the CI torch job: `eprop_shd::run_smoke` asserts against its OWN shipped rule
   (measured cos 0.9992), `eprop_nmnist::run_smoke` likewise (0.9997), and `eprop_kernel::_self_check`
   against `_TinyNet` (0.9714). **Each encoder already guards itself on its own form.** The kernel's
   `_TinyNet` is an inline test fixture, not the encoders' guard.

   The reconciliation is still worth doing and is still FR-011, on a narrower and honest basis: it
   costs 13 insertions / 7 deletions with every gate green, and it turns a **currently false in-file
   claim into a true one** -- `_self_check` asserts the readout gradients are "IDENTICAL between
   e-prop and BPTT by construction", which is false for the present form (cos_readout 0.986830,
   relative L2 7.423e-01) and exactly true under the filtered form (cos 1.000000).

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

## Clarifications

### Session 2026-08-17

- Q: How should the checker decide whether a given learning rule "claims to approximate a gradient" and therefore owes a cosine? → A: D — declaration AND structural inference, cross-checked, with any disagreement reported as a finding rather than resolved silently.
- Q: Where should these checks run — as a gate that fails the build in CI, or as an instrument you run on demand? → A: D — split by cost: the census and cross-check assert in CI (static, deterministic); the cosine measurement and discriminating-power check run as an on-demand instrument.
- Q: Should the CI half assert that the classification is correct today (and stay that way), or only that it never gets worse than today's baseline? → A: C — split: ABSOLUTE on "every rule is classified" (zero unclassified, always), RATCHET on declaration/inference disagreements (may fall, never rise).
- Q: Should this feature derive the 0.5 bar from a measurement, or measure only to check whether 0.5 is defensible and leave the number alone? → A: C — derive to VALIDATE, do not move. Measure the aligned-vs-null separation and report whether 0.5 sits in the separating region; if it does not, that is a recorded finding for a separate decision. Moving the bar and fixing the instrument in one feature would destroy attribution on any flipped verdict.
- Q: Should the census classify learning rules across the whole repository, or only those inside the governed package tiers? → A: A — census the WHOLE tree; the absolute and ratchet bounds assert on the GOVERNED tier only; ungoverned rules are counted and REPORTED, never asserted. This is the only reading under which FR-001, FR-013, SC-010 and the ungoverned-tier assumption are all simultaneously true, and it reuses the verdict contract `substrate/eprop_transport_gate` already ships (governed asserts, ungoverned reports) rather than inventing a second one.
- Q: FR-010 forbids changing any learning rule's implementation while FR-011 requires editing the reference gradient — which change does FR-010 prohibit? → A: B — FR-010 keeps its blanket prohibition and names FR-011's target as the SINGLE explicit exception; any further exception requires a spec change. A category ("fixtures are outside it") was rejected because a class of exceptions can be widened silently, and there is exactly one known exception today. FR-010a added: the prohibition must be VERIFIED from the diff with a positive control, since FR-010 was cited by zero tasks.
- Q: How should a reader tell that the on-demand measurements have gone stale — that nobody re-ran them since the code they describe changed? → A: C — record a CONTENT DIGEST of the measured inputs beside the output; STALE exactly when the digest stops matching the tree. Commit-SHA rejected (fires on every unrelated commit, training readers to ignore it); timestamp-plus-threshold rejected (the threshold is a P1 magic constant). FR-015a added: the staleness signal owes a reachable red.
- Q: What time budget should the CI-asserting census have, given it runs on every CI job? → A: C — NO asserted budget. Measure the cost, print it in the census output, treat a large regression as a reported finding. A wall-clock assert on shared runners would redden on runner variance, the same environment-sensitivity that already forced the numeric half out of CI; and a fixed second-count is a P1 magic constant arbitrating whether the build passes. Applies to the ratchet form too, so D was rejected for the same reason as A and B.
- Q: When a ratchet reaches zero and can never fire again, how is that success state told apart from a ratchet that simply never ran? → A: B — REPORT the two states distinctly: SATISFIED (bound resting at its floor) vs GREEN (bound above its floor). The ratchet is KEPT. Retiring it for an absolute `== 0` was considered and declined: it would redden CI the moment any new rule arrives without a declaration, which tightens the gate against unrelated work rather than describing the state honestly.

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

### User Story 4 - Say plainly what the number is, without moving it (Priority: P4)

The aligned-vs-null separation is measured and `0.5`'s position relative to it is reported, alongside
its P1 case-(a) justification. **The bar itself is NOT changed by this feature** (owner decision
2026-08-17): if the measurement shows `0.5` is poorly placed, that is a recorded finding handed to a
separate decision.

**Why this priority**: Lowest because it changes no verdict -- by design. Leaving an unexplained
number on an admission bar is how the next reviewer re-opens the P1 question from scratch, which is
exactly what produced this spec; but moving the bar here would collide with the instrument
reconciliation in Story 3 and leave any flipped verdict unattributable.

**Independent Test**: A reader of the invariant can state where `0.5` came from, and what the
measured separation was, without reading code.

**Acceptance Scenarios**:

1. **Given** the invariant text, **When** a reviewer asks why the bar is `0.5`, **Then** the measured
   aligned-vs-null separation and `0.5`'s position within it are present, along with the P1 case-(a)
   justification for a fixed point on a normalized output.
2. **Given** a measurement showing `0.5` is outside the separating region, **When** this feature
   completes, **Then** the bar is UNCHANGED and the finding is recorded for a separate decision.

---

### Edge Cases

- A rule that learns but has **no time axis**, so a reverse-mode temporal gradient is undefined for
  it. It cannot owe a temporal-gradient cosine; the census must classify it rather than crash or
  silently omit it.
- A rule that is **default-off with zero live callers**. It should not be forced into compliance work
  it cannot charge -- but it must not vanish from the census either, or the count understates.
- A rule whose reference gradient **cannot be computed in the gate environment** (no GPU, missing
  optional dependency). Reporting "no cosine available" must be distinguishable from "cosine measured
  and failed" -- collapsing those two is the failure this project has hit before. This is also why
  the numeric half is on-demand rather than CI-asserting.
- A rule whose **module does not import** in the gate environment. It must still be classified by the
  CI half, which therefore cannot rely on importing the module to inspect it.
- The census finds **zero** rules owing a cosine. That must be reported as a suspicious result and
  fail loudly, because an empty census and a broken walk look identical.
- The disagreement count reaches zero and the ratchet becomes a bound that can never fire again. That
  is the success state, but it must be recognisable as such rather than sitting indefinitely as a
  green check nobody can distinguish from an unexercised one. **RESOLVED by FR-017a**: the output
  names SATISFIED (at floor) distinctly from GREEN (above floor), for both ratchets.
- A rule claims gradient approximation in prose but its code performs no gradient-like update at all
  -- the declaration and the structural inference disagree, which is a reportable finding rather than
  a classification.
- A rule performs a gradient-like update but carries no declaration. This is the case a
  declaration-only scheme would silently miss, and it must surface as a disagreement, not as
  "owes nothing".
- The two signals agree on every rule in the tree. That is the expected healthy state, but it must be
  reported as a count rather than as silence, or a cross-check that stopped running looks identical
  to one that found nothing.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST classify every learning rule in the **WHOLE tree — governed and ungoverned
  alike** — as either owing a gradient cosine or owing none, with zero unclassified, and MUST derive
  the classification from the tree rather than a hand-maintained list. (Clarified 2026-08-17; this
  said "the governed tree", which contradicted FR-013, SC-010 and the ungoverned-tier assumption
  below.)
- **FR-001a**: **CENSUS scope and ASSERT scope are different, and the split is required rather than
  incidental.** The census WALKS the whole tree; the absolute bound (FR-016) and both ratchets
  (FR-016a, FR-017) assert on the **GOVERNED tier only**, and ungoverned rules are counted and
  REPORTED. This is the verdict contract `substrate/eprop_transport_gate` already ships — it asserts
  on governed findings and reports the rest — so the feature reuses an accepted failure contract
  instead of introducing a second one. Measured scale reason: `reasoning/` alone is ~108 modules
  against roughly 22 governed local rules, so asserting over both would make the initial baselines
  ~5x larger and would gate this feature on remediating a subtree it does not own.
- **FR-002**: System MUST base "owes a cosine" on whether the rule CLAIMS to approximate a gradient,
  because that is the claim the measurement can check. A purely local rule making no such claim owes
  no cosine and MUST NOT be reported as a violation.
- **FR-002a**: The claim MUST be determined from TWO signals -- an explicit declaration carried by the
  rule itself, and a structural inference derived from its code -- which are cross-checked. **AMENDED
  after Phase 0 measurement (research.md R1):** the two are NOT co-equal. Structural inference
  measured **~31% error in both directions**, with false negatives on exactly the load-bearing cases
  (`mlp_q`, `jepa_world_model`, `gf.MLP`, `predictive_coding`), because this tree hand-rolls backprop
  (no token to key on), keeps P8 controls beside the rules they control, and computes the gradient in
  a different function from the one that applies it. The **DECLARATION is authoritative**; the
  inference is a **recall net that may answer only CLAIM or UNKNOWN, never NO-CLAIM.** Its job is to
  catch a rule with no declaration, never to overrule one that has it.
- **FR-002b**: Where the inference says CLAIM and no declaration exists, the system MUST report that
  as a finding -- a missing declaration -- and MUST NOT classify the rule by default. The inference
  MUST NOT be able to produce a NO-CLAIM verdict at all, because a confident negative from a signal
  with measured 31% error is the dangerous verdict Standing Invariant 3 forbids.
- **FR-002c**: The declaration MUST be a module-scope constant readable by `ast.literal_eval` without
  importing the module, keyed by qualified symbol, carrying **FIVE** dispositions -- `LOCAL`,
  `APPROXIMATES_GRADIENT`, `EXACT_GRADIENT`, `CONTROL`, `NOT_A_RULE`. Per-symbol is required, not
  stylistic: `cortex/eprop_kernel.py` holds the rule, its BPTT control and a surrogate
  `autograd.Function` in one file, so a per-module verdict cannot describe it. Two dispositions are
  insufficient for the same reason -- `reasoning/gf.py::MLP` is an owner-EXEMPT CONTROL, and a
  two-value classifier condemns it.
- **FR-002e**: `NOT_A_RULE` exists because FOUR dispositions were **measured insufficient during the
  first real declaration pass**, not because a fifth looked tidy. Five enumerated symbols could not be
  described truthfully by any of the four: `eprop_kernel::_first_layer_grad` (a DIAGNOSTIC returning
  summed `|grad|`), `gf::OpenGf.solve_unbounded` (iterative-deepening SEARCH), `gf::OpenGf._learn` and
  `gf::ActiveLearner.learn` (both grow a SYMBOLIC GRAMMAR, touching no weight), and
  `eprop::_batch_grad` (a COMBINATOR summing whatever `grad_fn` it is handed, so
  `APPROXIMATES_GRADIENT` is false on the BPTT call and `EXACT_GRADIENT` false on the e-prop call).
  The enumeration deliberately over-captures for recall, so non-rules in the census are a predictable
  consequence of that choice. Without this value the only options were to label a search `LOCAL`
  (false) or leave it `UNDETERMINABLE` -- true, but that conflates *"I cannot decide"* with *"there is
  nothing here to decide"*, and FR-016a ratchets `UNDETERMINABLE` DOWN as debt these are not.
- **FR-002g**: `DIRECT_SOLVE` exists for a rule that owes no cosine and is **not local**, and it is
  named with `docs/POLICIES.md`'s own word rather than an invented synonym. `hippocampus/ng_snn.py::fit`
  is a genuine learning rule adjudicated there as *"not a violation. Greedy matching pursuit with an
  analytical beta per block is a **direct solve**: no gradient is propagated through a network and no
  weight is transported. **It IS non-local, and the policy should say so distinctly rather than
  lumping it in** — 'non-local' and 'backprop' are different claims, and only the second is what this
  rule forbids."* Declaring it `LOCAL` would perform exactly the lumping-in that adjudication forbids;
  leaving it undeclared would report a settled question as unresolved. The value MUST stay scoped to
  that meaning and MUST NOT become a general non-local bucket: `cortex/predictive_coding` was checked
  against it and remains `LOCAL`, because POLICIES.md records its weight update as *"a local Hebbian
  outer product … No transposed weight appears in the learning rule at all"* — its `w1.T` is inference
  settling an activation, not the rule.
- **FR-002h**: The governing-document amendment (FR-008/FR-008a) MUST state, as a GENERAL clause,
  the criterion that decides whether an autograd trainer is inside the cosine clause's scope -- and
  that criterion already exists in this tree, adjudicated and measured. It is **reachability from a
  live path**, not "probe versus rule". `hippocampus/developmental_loop_brain`'s module docstring
  carries it verbatim from `specs/010`: its readout "trains by Adam + .backward()" and is *permitted
  because "That path is TRAIN-TIER OFFLINE, not a live learning path ... no `_LIVE_ROOT` or
  `_ENTRYPOINT` in `substrate/dormancy_gate.py` reaches it"*, with the binding constraint that wiring
  it to a live consumer *"converts the permitted offline trainer into an e-prop violation that
  nothing would announce."* **Re-measured 2026-08-17 with both controls holding** (`cortex.brain`
  reachable, a fabricated name not): `developmental_loop_brain` and `reasoning/_loop_common` are
  UNREACHABLE, so both are permitted offline trainers today.

  The amendment MUST promote that from a per-module note in one docstring to a clause the census can
  cite, and MUST say what happens when reachability CHANGES -- because the constraint's own wording
  admits that nothing currently announces the flip. A criterion recorded in exactly one module's
  prose is one the next reader of a different module cannot apply.

  **A PREMISE THIS REQUIREMENT ORIGINALLY CARRIED WAS FALSE AND IS STRUCK.** It claimed "three
  near-identical linear readouts ... all three train a plain `nn.Linear` by explicit `.backward()`",
  and offered their three differing verdicts as evidence. Measured: only TWO are that
  (`developmental_loop_brain` and `reasoning/_loop_common` -- the verbatim-copy P2 debt already
  recorded as AUDIT_BACKLOG M21). The third, `cortex/autapse_neuron::_LinearReadout.train`, is a
  NAME COLLISION: a single-layer logistic delta rule in pure stdlib (`w[k] -= lr * g * x[k]`, no
  torch imported anywhere in the module, no hidden layer, no backward pass). It is REACHABLE and it
  is `LOCAL`, and it was never evidence of an inconsistent scope. Recorded rather than deleted,
  because a requirement argued from an unmeasured premise is the defect this feature exists to catch.
- **FR-002f**: `NOT_A_RULE` MUST NOT become the drawer `UNDETERMINABLE` is ratcheted to prevent. TWO
  guards, and they are structural rather than numeric: **(a)** it MUST be reachable ONLY by an
  explicit declaration carrying a reason -- the inference MUST be incapable of producing it and no
  default may resolve to it, so the classifier can sweep nothing into it; **(b)** its count MUST be
  reported as a measure of **enumeration imprecision, not tree health**, and the output MUST say so,
  because a rising count means the recall-biased heuristic is capturing more non-rules and wants
  tightening -- never that the code got worse.
- **FR-002d**: The declaration MUST NOT be a method, and MUST avoid names containing any
  `method_dormancy_gate.LEARN_VERBS` substring. Measured traps: `declare_constraint` (con-TRAIN-t),
  `get_adapter` (ADAPT-er), `plasticity_claim` (PLAST), `recalibrate`, `fit_claim`, `update_claim`.
  Measured clean: `GRADIENT_CLAIM`, `EPROP_GRADIENT_CLAIM`, `classify`, `verify_claim`.
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
- **FR-008**: The governing documents MUST state the scope of the cosine clause explicitly, and MUST
  state that a rule which cannot pass the criterion by construction has falsified the instrument
  rather than itself.
- **FR-008a**: BOTH governing surfaces MUST be amended together: `docs/POLICIES.md` (authoritative)
  and the constitution's Standing Invariant 8, which restates the same clause verbatim. Amending only
  one leaves the other contradicting it, and the constitution's own Governance section states that
  where the two disagree the constitution is the defect. A scope fix that lands in one surface only
  has created the divergence it was written to prevent.
- **FR-009**: The governing document MUST record the provenance of the numeric bar, including the
  measured aligned-vs-null separation and whether `0.5` falls inside the separating region.
- **FR-009a**: The system MUST measure the separation between aligned gradients and a null control,
  and MUST report where `0.5` sits relative to it. **The null is named exactly: a WITHIN-MATRIX
  PERMUTATION of the candidate gradient** (permute `W_in` coordinates among themselves and
  off-diagonal `W_rec` among themselves, then cosine against the UNCHANGED reference). "Permuted or
  shuffled" was under-specified and Phase 0 measured that the two give OPPOSITE verdicts on the same
  gradients: against a permutation null `0.5` sits +4.5 to +63 sd ABOVE it (P(null>0.5) <= 3e-5);
  against a shuffled-label null it sits 0.32 to 2.55 sd BELOW the null mean (P = 0.65 to 0.98). Either
  satisfied the requirement as previously worded, which let the implementer choose the answer.

  **CORRECTED 2026-08-17 by measurement, and the correction is in the direction that matters.** The
  recorded range "+4.5 to +63 sd (P(null>0.5) <= 3e-5)" does **NOT** contain the one fixture the
  instrument can run in every environment: on `reasoning/eprop::Net.grad_eprop` vs `::Net.grad_bptt`,
  built exactly as this requirement words it, `0.5` sits at **+3.06 sd** with P(null>0.5) = 0.0000
  and the aligned cosine at +5.93 sd. Either the Phase 0 sample excluded this module or the range is
  wrong; the range is not re-derivable from anything recorded, so it is treated as UNRELIABLE and
  this measured figure supersedes it for this module only. The VERDICT is unaffected -- the
  separation is real and large -- but a spec that states a range its own anchor fixture falls outside
  of would have let an implementer "confirm" a number by finding a different one.
- **FR-009d**: The system MUST measure the SIGN COHERENCE of the candidate and reference gradients
  (`|mean| / rms`) and MUST refuse to report a separation when coherence is high, returning
  `INSTRUMENT_FALSIFIED` instead. **Reason, measured, not anticipated**: a within-matrix permutation
  preserves the per-coordinate-family mean, so for a sign-coherent gradient the null is not a null.
  On a constructed coherent pair (coherence 0.99) the same code puts the null mean at **+0.980**,
  P(null>0.5) at **1.0000**, and the aligned cosine **1.75 sd BELOW** the null -- the separation
  statistic INVERTS, and a criterion reading it would condemn an aligned rule and bless a shuffled
  one. The real fixture is safe (coherence 0.117 / 0.199), which is exactly why this cannot be left
  to be discovered later: the mandated null works on the case in front of us and fails silently on a
  case that has not been added yet. This is P8 clause 1 turned on the instrument -- the same move
  FR-004 already makes for the rule.
- **FR-009e**: The permutation null's DRAW COUNT and RNG SEED MUST be pinned constants recorded in
  the FR-015 artifact. Measured across five seeds at 400 draws, `0.5` sits between **+2.97 and +3.27
  sd** and P(null>0.5) moves between **0.0000 and 0.0025** -- small, but enough that an unpinned
  artifact reports a different number on an unchanged tree, which is precisely the staleness FR-015
  exists to make visible.
- **FR-009c**: Three further nulls -- isotropic matched-norm, within-UNIT permutation, and random
  relabel -- MUST be REPORTED and MUST NOT be asserted on. Each answers a different question and one
  of them reverses the verdict; publishing them pre-empts the reader who recomputes the bar mentally.
  The untrained twin is REJECTED as a control: it moves the net, the forward path and the reference
  at once and cannot be interpreted.
- **FR-009b**: This feature MUST NOT change the value of the bar. If the measurement shows `0.5` is
  poorly placed, that is recorded as a finding for a separate decision. Rationale, recorded so a
  later reader does not mistake it for timidity: SC-008 already accepts that reconciling the
  reference gradient may move measured cosines, and moving the BAR in the same feature would leave
  any flipped verdict with two candidate causes and no way to attribute it.
- **FR-010**: System MUST NOT change any learning rule's implementation, MUST NOT relax the
  backprop/weight-transport prohibition, and MUST NOT touch the safety envelope.
  **ONE EXPLICIT EXCEPTION, and only one (clarified 2026-08-17):** the reference gradient reconciled
  under FR-011 — `cortex/eprop_kernel.py`'s `_TinyNet.eprop_grads`, an inline test fixture, measured
  at 13 insertions / 7 deletions — together with the recorded-number corrections FR-011 forces in that
  module's comments and `basal_ganglia/mlp_q.py`'s duplicate of the same figure. **Any further
  exception REQUIRES a spec change and MUST be named here.** A category was deliberately NOT defined
  ("fixtures and reference gradients are outside FR-010"): there is exactly one known exception today,
  and a class of exceptions is one a later reader can widen without anyone noticing. The two clauses
  after the first are absolute and have no exception of any kind.
- **FR-010a**: FR-010 MUST be VERIFIED, not merely asserted. The check MUST confirm from the diff
  that no file in the safety envelope changed, and that no learning-rule body outside FR-010's named
  exception changed. It MUST carry a positive control in the same run — a check never shown to
  produce a finding cannot evidence its absence — and it is the only thing standing between "we did
  not intend to touch the safety envelope" and "we did not touch it". FR-010 was cited by ZERO tasks
  when this was found.
- **FR-011**: The reference gradient used by the criterion MUST be RECONCILED with the factorization
  the shipped encoders run -- reconciliation is in scope for this feature, decided by the owner on
  2026-08-17. A disclosure alone is not sufficient. If reconciliation proves impossible, the
  divergence and its measured consequence MUST be recorded on the criterion itself and the feature is
  not complete until that record exists.
- **FR-012**: The reconciliation MUST be demonstrated by measurement, not by inspection: perturbing a
  shipped encoder's update rule MUST move the criterion's output. A reconciliation that changes no
  measured value has not been shown to attach the criterion to the shipped rule.
- **FR-013**: The census and the declaration/inference cross-check MUST assert in CI. They MUST be
  static and deterministic, MUST NOT require a GPU, and MUST NOT require the rule's module to be
  importable -- roughly 30% of the ungoverned subtree does not import in the gate environment, and a
  classifier that skips what it cannot import silently under-counts.
- **FR-013a**: The CI half MUST MEASURE and PRINT its own wall-clock cost, and MUST NOT assert on it.
  Rationale recorded so the absence of a budget is not mistaken for an oversight: this gate runs on
  every cell of a 6-cell CI matrix with the Windows legs at 2x, against a recorded ~250 min/day burn
  and a minutes wall due around mid-September, so the cost must be VISIBLE. It is not ASSERTED
  because wall-clock on shared runners is not deterministic -- the same environment-sensitivity that
  already forced the numeric half out of CI (FR-014) -- and because a fixed second-count arbitrating
  whether the build passes is a P1 magic constant. A large regression is a reported finding, not a
  red. This replaces plan.md's "comparable to the existing AST gates" and "cannot be expensive",
  which set no measurable expectation.
- **FR-013b**: `--measure` MUST obtain gradients by running each fixture in a SUBPROCESS, and the
  gate module MUST NOT import any rule-carrying module in any mode. **This is a hard boundary, not a
  performance preference, and it is written down because 89 src modules already assert it**: each
  carries the stamp `Read statically by substrate/gradient_claim_gate.py; never imported by it.`
  An in-process `--measure` would make that sentence FALSE on the seven modules it touches
  (`reasoning/eprop`, `cortex/eprop_kernel`, `cortex/eprop_shd`, `cortex/eprop_nmnist`,
  `basal_ganglia/mlp_q`, `hippocampus/growable_readout`, `hippocampus/developmental_loop_brain` --
  all seven verified to carry it), which is this project's most-repeated defect: a claim outliving
  what it described. Subprocess isolation additionally makes `UNAVAILABLE` (FR-005) a NATURAL
  outcome rather than a special case -- a module that will not import fails its subprocess and is
  reported unavailable, which is what the verdict means. The measured ~5x cost of subprocess
  spawning is not a cost worth weighing: `--measure` never runs in CI (FR-014).
- **FR-014**: The cosine measurement and the discriminating-power check MUST run as an on-demand
  instrument and MUST NOT assert in CI. Rationale recorded so it is not mistaken for laxity: CI runs
  a different torch build from the local environment, and this project has already documented that a
  build-sensitive numeric assertion can be green in one and red in the other.
- **FR-015**: The on-demand instrument MUST record its output in a form a later reader can find and
  re-run, so that "nobody has run it lately" is visible rather than indistinguishable from "it passed".
  **Mechanism named 2026-08-17: a CONTENT DIGEST of the measured inputs** — the bytes of the modules
  the measurement actually read — stored beside the recorded output. The record is STALE exactly when
  that digest no longer matches the tree, and quiet otherwise. A commit-SHA comparison was rejected
  because it marks the provenance stale on every unrelated commit, and a warning that fires constantly
  is one readers learn to skip; a timestamp-plus-threshold was rejected because the threshold is a
  magic constant arbitrating behaviour, which is a P1 event in its own right.
- **FR-015a**: The staleness signal MUST have a REACHABLE RED: changing one measured input MUST flip
  the record to STALE, demonstrated by construction rather than asserted. Without that, a digest that
  is never recomputed is indistinguishable from a digest that always matches -- which is the same
  defect one level up.
- **FR-016**: The CI half MUST assert ABSOLUTELY that the number of unclassified learning rules **in
  the GOVERNED tier** (FR-001a) is zero. This bound does not move: a rule the system cannot classify
  is a hole in the census, and a census with a hole cannot support any claim about scope. **"Classified" INCLUDES the state
  `UNDETERMINABLE`** -- classified must not be read as decided. Phase 0 found that without this,
  "zero unclassified, always" forces an undecidable rule out as NO-CLAIM, which is the confident
  negative Standing Invariant 3 forbids, and which is measurably wrong on at least five governed
  modules -- all of them load-bearing.
- **FR-016a**: `UNDETERMINABLE` MUST itself be ratcheted DOWNWARD: it may fall, never rise. Otherwise
  the state added to preserve honesty becomes the drawer everything difficult is swept into.
- **FR-019**: `substrate/eprop_transport_gate.EXEMPT` MUST be RETIRED and its one entry
  (`reasoning.gf.train`, dispositioned "CONTROL, not a learning rule") MUST move into the declaration
  registry as a `CONTROL` disposition on `reasoning/gf.py`. Without this the feature SHIPS a P2
  violation of its own making: two hand-maintained records of one symbol's disposition, which a later
  change to `gf.py`'s status would have to land in both. P2 adjudication in plan.md.
- **FR-019a**: The retirement MUST port `stale_exempt`'s honesty property FIRST -- every exemption is
  asserted to still match a REAL finding, so a blessing that blesses nothing fails loudly rather than
  reading as coverage. That property is what makes the incumbent's dict trustworthy and MUST survive
  the move. Port before retire, per P2.
- **FR-019b**: The adoption MUST be verified by an A/B on the real tree: the incumbent's findings and
  census MUST be byte-identical before and after the swap. Note the keying hazard -- `_exempt_key`
  builds its key from the ENCLOSING FUNCTION name (`reasoning.gf.train`, a bare `train`) while the
  registry is keyed per QUALIFIED symbol (`MLP.train`); a naive match silently misses.
- **FR-017**: ~~The CI half MUST assert the count of declaration/inference disagreements **in
  the GOVERNED tier** (FR-001a) as a RATCHET~~ **NOT IMPLEMENTED AS WRITTEN — superseded by the
  SC-012 adjudication (census-baseline.md), recorded 2026-08-18 after an adversarial review
  found the spec and tree silently divergent.** The shipped gate ratchets MISSING_DECLARATION
  and UNDETERMINABLE instead. Reason: disagreement is the DESIGNED state for every correctly
  declared local rule (72 of the 97 — `infer()` can only answer CLAIM or UNKNOWN), so ratcheting
  it would red the build on each new correctly-declared rule while measuring nothing. The
  census stores both signals per rule; no disagreement report is emitted.
- **FR-017a**: **Every ratchet MUST report which of two distinct states it is in** (clarified
  2026-08-17), and this applies to BOTH ratchets -- disagreements (FR-017) and `UNDETERMINABLE`
  (FR-016a):
  - **SATISFIED** -- the count rests at its floor, so the bound can never fire again. This is the
    feature's success state and MUST be named as such.
  - **GREEN** -- the count is above its floor, so the bound is live and still doing work.

  A ratchet resting at zero and a ratchet nobody ever exercised both print green today, and a success
  state indistinguishable from a dead check is the shape Standing Invariant 5 calls worse than no gate
  because it manufactures confidence. The ratchets are KEPT rather than converted to absolute bounds:
  an absolute `== 0` would redden CI the moment any new rule arrives without a declaration, tightening
  the gate against unrelated work instead of describing the state honestly. Green on this bound means "no worse", never "the scope is
  correct", and the reported output MUST say so in those terms so a reader does not over-read it.
- **FR-018**: The ratchet baseline MUST be machine-generated, and the writer MUST REFUSE to record a
  baseline higher than the current count. A baseline that can be raised by hand-editing a literal is
  not a ratchet -- one existing gate in this tree keeps its baseline as a source-level dict with no
  mechanical refusal, so raising its ceiling is a one-line edit, and this feature MUST NOT repeat
  that shape.

### Key Entities

- **Learning rule**: a mechanism that updates weights or thresholds from experience. Attributes: the
  module it lives in, whether it claims gradient approximation, whether it is reachable from a live
  path.
- **Gradient claim**: a rule's assertion that its update approximates a gradient. This is what makes
  the cosine measurable; its absence is a legitimate state, not a gap. Carried by TWO signals -- a
  declaration on the rule and a structural inference from its code -- which are cross-checked; their
  disagreement is a reportable state in its own right, distinct from claim and no-claim.
- **Discriminating-power result**: the aligned/non-aligned measurement pair for one rule, plus the
  verdict -- rule-judged, or instrument-falsified.
- **Census**: the counted classification of every learning rule at a point in time, re-derived per
  run. Walks the whole tree; its governed and ungoverned subtotals are reported separately because
  only the governed half is asserted on (FR-001a).
- **Provenance record**: the recorded output of the on-demand instrument, plus a CONTENT DIGEST of
  the inputs it measured. Attributes: the measurements, the digest, and the derived STALE / CURRENT
  state. Its purpose is to make non-execution visible; a record whose digest is never recomputed is
  indistinguishable from one that always matches, which is why FR-015a requires a reachable red.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of learning rules in the WHOLE tree are classified, with zero unclassified in the
  governed tier, and the total reconciles against an independent count of learning rules. The
  governed and ungoverned subtotals are reported separately, so a reader can see which half the
  assertion covers (FR-001a).
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
- **SC-009**: The CI-asserting half completes without a GPU and produces an identical classification
  on two consecutive runs of the same tree.
- **SC-010**: The CI-asserting half classifies 100% of learning rules including those whose modules
  do not import in the gate environment -- verified by confirming at least one known non-importing
  module appears in the census.
- **SC-011**: Attempting to record a ratchet baseline worse than the current count is REFUSED,
  demonstrated by a constructed attempt rather than asserted in prose.
- **SC-012**: The disagreement count falls to zero, or every remaining disagreement is named with the
  reason it is unresolved. A number with no accompanying list is not an acceptable end state.
- **SC-013**: The aligned-vs-null separation is measured and published, and the numeric bar is
  unchanged from its pre-feature value -- verifiable by diff.
- **SC-014**: Exactly ONE hand-maintained record of any symbol's gradient disposition exists in the
  tree when this feature completes. Verifiable: `eprop_transport_gate.EXEMPT` is gone and its content
  is reachable only through the declaration registry.
- **SC-015**: The incumbent transport gate's findings and census are byte-identical before and after
  it adopts the registry -- zero behaviour change from a pure data merge.
- **SC-016**: The diff for this feature touches ZERO files in the safety envelope and zero
  learning-rule bodies outside FR-010's single named exception -- demonstrated by a check that also
  produces a positive finding on a constructed violation, so its clean result is evidence rather than
  silence.
- **SC-017**: The recorded provenance reports STALE when any measured input changes, demonstrated by
  changing one input and observing the flip -- a digest that is never recomputed reads identical to
  one that always matches.
- **SC-018**: Each ratchet's output names SATISFIED or GREEN, and a reader can tell a bound resting
  at its floor from one that is still live without reading the baseline file.
- **SC-019**: The CI half's wall-clock cost appears in its own output on every run, and no CI job
  fails on that number -- verifiable by confirming the figure is printed and that no assertion reads
  it.
- **SC-021**: The governing text states the live-path-reachability criterion as a general clause, and
  the TWO genuine autograd copies named in FR-002h (`developmental_loop_brain`, `reasoning/_loop_common`)
  carry the SAME disposition, derived from that clause rather than from three separate reason strings.
- **SC-020**: No symbol is labelled with a disposition that is false about it. Demonstrated rather
  than asserted: the classifier cannot produce `NOT_A_RULE` without a declaration (shown by a
  constructed case), and every symbol carrying it names in its reason what it does instead of
  learning.
- **SC-022**: The instrument refuses to report a separation it cannot support. Demonstrated by a
  CONSTRUCTED sign-coherent pair on which the mandated null provably inverts (null mean +0.980,
  P(null>0.5) = 1.0000, aligned 1.75 sd BELOW the null): `--measure` returns
  `INSTRUMENT_FALSIFIED` there, and `RULE_JUDGED` on the real fixture, in the same run. A coherence
  guard that never fires on anything is indistinguishable from no guard.
- **SC-023**: The gate imports no rule-carrying module in any mode, proven by a SUBPROCESS assertion
  with its own positive control in the same test, and the 89 modules stamped `never imported by it`
  are still telling the truth after Phase 4 lands.
- **SC-024**: Re-running `--measure` on an unchanged tree reproduces the artifact byte-for-byte, and
  the artifact records the draw count and seed that make that true.

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
- **The risk this decision accepts** (measured by SC-008): reconciling the reference gradient may MOVE the
  reported cosine for the shipped encoders. If a rule that measured above the bar now measures below
  it, that is a real finding about the rule and must be reported as such -- not absorbed, and not
  treated as a reason to abandon the reconciliation. This spec does not predict the direction.
- Rules living outside the governed tier are counted by the census but are not required to reach
  compliance in this feature. **Mechanized 2026-08-17 as FR-001a** rather than left as an assumption:
  the census walks the whole tree, the bounds assert on governed only, and the ungoverned count is
  reported. An assumption that no requirement encodes is one a later implementer can satisfy either
  way.
- The gate environment has no GPU, so any measurement must be reproducible on CPU or must report
  itself unavailable rather than failing the build.
- The clause appears on TWO governing surfaces -- `docs/POLICIES.md` and the constitution's Standing
  Invariant 8 -- and both are in scope. This was not in the original draft and was found by reading
  the constitution during clarification; it is recorded here because a one-surface amendment is the
  precise failure mode the constitution's Governance section warns about.
