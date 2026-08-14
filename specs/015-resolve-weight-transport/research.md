# Phase 0 Research — Resolve the e-prop Weight Transport at `mlp_q`

All facts re-derived at HEAD (`015-resolve-weight-transport`, one commit off `main` = `462da205`) in
`.venv-lake`. Every enumeration below carried a positive control in the same command (FR-016).

---

## D-1 — SFDFA DOES NOT TRANSFER LITERALLY, AND THAT IS NOT THE END OF IT

**Decision**: report plainly that Bacho & Chu's SFDFA cannot be implemented in this module as
specified — and adopt its ESSENTIAL mechanism, which does transfer and has a name.

**Why the literal method does not transfer.** SFDFA estimates output→hidden weights *"by propagating
directional derivatives as spike grades during the inference"*. Read against `MlpQ`:

- `MlpQ` is a **rate** MLP Q-head. `_forward` computes a relu pre-activation and activation; there is
  no membrane, no threshold, no spike, and no time index anywhere in `update`.
- There are therefore **no spike grades to propagate**, and no intra-neuron dependency between
  post-synaptic spikes for the paper's main contribution to be about.
- The paper's contribution is precisely the spiking part — *"how exact local gradients of spikes can
  be computed in an online manner while taking into account the intra-neuron dependencies between
  post-synaptic spikes"*. Strip the spikes and the contribution is gone.

**Per the spec's Assumptions this is a reportable finding, not a licence to substitute.** So what
follows is stated as what it is: a DIFFERENT, older mechanism that shares SFDFA's essential property.

**The essential property, isolated.** What makes SFDFA interesting here is not that it is spiking. It
is that the feedback weights are **estimated and converge toward the forward weights**, rather than
being (a) read off the forward weights (transport) or (b) frozen at a random draw (vanilla FA). That
property is what makes it compatible with a `cos > 0.5` invariant.

**The rate-domain mechanism with that property is Kolen-Pollack**, and `MlpQ.update` is unusually
well-suited to it. The read-out update already computes everything needed:

```
w2i[j] += lr * err * h[j]          # the existing read-out step
```

A feedback matrix `B` receiving the **same increment** converges to `W2` without ever reading it —
`err` and `h[j]` are both already local to that synapse. The transport line then becomes
`dh[j] = err * B[i][j] * relu'(pre[j])`, structurally identical to the FA twin the module already
benchmarked, differing only in that `B` is no longer frozen.

**Verified absent from the tree** (so this is not a P2 duplicate): 0 hits across `src/`, `tools/`,
`tests/` for `Kolen`, `Pollack`, `weight mirror`, `weight_mirror`, `Akrout`, `feedback_matrix`,
`B_fb`, `learned feedback` — against a control of 45 files mentioning `feedback`.

**Honest naming obligation (FR-017).** If this is adopted, it is adopted as Kolen-Pollack-style
converging feedback, cited as such. Calling it "SFDFA" would be the citation defect this lane exists
to remove, one level up.

**Open sub-decision for the plan**: the decay term. Textbook Kolen-Pollack pairs the shared increment
with weight decay on both matrices. Decay introduces a constant, which P1 forbids in bare form. Two
routes, to be settled in Phase 1 rather than assumed: derive the decay from an existing quantity
(e.g. tie it to `lr`), or run decay-free from a zero/small `B` init, where the shared-increment
argument alone gives convergence. **Whichever is chosen, the choice is measured, not asserted.**

---

## D-2 — WHAT THE REPAIRED CRITERION MEASURES

**Decision**: the criterion becomes **cosine between the HIDDEN-LAYER UPDATE the rule produces and
the hidden-layer update the transport gradient would have produced, on the same input and the same
pre-update weights** — read per-epoch, not once at the end.

**Why the current one cannot discriminate.** The module records it: *"essentially `cos(W2[a], B[a])`
under the relu mask (`err` cancels)"*. Because `err` is a scalar multiplying both sides, it divides
out, and what remains is the angle between a forward weight vector and a feedback vector. For a
**fixed random** `B` that is near zero by construction in any reasonable width. The criterion asks
"is the feedback parallel to the forward weights", which is the weight symmetry FA exists to deny.
It cannot separate a good alignment rule from a bad one because it never looks at what either rule
DOES.

**Why the replacement is the right quantity.** The cross-cutting invariant asks for cosine > 0.5
against an independent gradient — i.e. does this rule move the weights in roughly the direction the
true gradient would. `dh` IS the hidden-layer update direction. Comparing `dh_rule` against
`dh_transport` measures exactly the invariant's subject, and it does not degenerate: the relu mask
and `err` appear on both sides but do not cancel the comparison, because the two `dh` vectors differ
in the feedback vector only where the mask admits them.

**Note it will still be a monotone function of `cos(W2, B)` under this module's arithmetic** — the
mask zeroes the same coordinates on both sides. That is disclosed rather than hidden, and it is
exactly why the fix is not "a different formula" but **per-epoch readout plus a feedback that can
move**: the number becomes informative because `B` develops, not because the algebra changed.

**Alternatives weighed:**

- *(a) Keep the endpoint cosine, change nothing else* — rejected. It is the degenerate quantity.
- *(b) Per-epoch only, same quantity* — this is most of the fix and is adopted, but on its own it
  would still be reported against a rule (`B` fixed) that cannot move, so it would draw a flat line
  and prove nothing new.
- *(c) Angle between the resulting WEIGHT MATRICES after training* — rejected: confounds the update
  rule with the optimisation trajectory, and cannot be read per-epoch cheaply.
- *(d) Loss-based only (held-out accuracy)* — rejected as the criterion, kept as a co-report
  (FR-005). Accuracy cannot distinguish "aligned" from "learned by another route", which is the
  distinction the invariant is about.

**The known-bad reference is chosen independently of the candidate (R-001)**: a **permutation null**
— the same `dh` vector with its coordinates shuffled. It has identical magnitude and identical
element distribution, differing only in direction, so a criterion that cannot separate it from the
real thing is measuring magnitude and not direction. This is fixed before any candidate is run and
does not depend on which rule wins.

---

## D-3 — THE PRE-REGISTERED PREDICTION (FR-003)

**Written before any run. The comparison afterwards is against THIS text.**

Fixtures are the module's own: 400 train / 300 held-out, `hidden=32`, `lr=0.1`, 30 passes, model seed
0, `bseed` 1..8. Cosine read at each of the 30 passes.

| variant | predicted cosine trajectory | predicted endpoint |
|---|---|---|
| transport (incumbent) | 1.0 at every epoch by identity | 1.0 |
| fixed-random `B` (vanilla FA) | **flat**, no trend, inside roughly ±0.35 for width 32 | \|cos\| < 0.5 on **8 of 8** draws |
| converging `B` (Kolen-Pollack) | **rises monotonically in trend** from ~0 | \|cos\| > 0.5 on **at least 6 of 8** draws by pass 30 |
| permutation null | flat near 0 | \|cos\| < 0.5 on 8 of 8 |

**Accuracy predictions, so "aligns but stops learning" is falsifiable**: converging `B` clears the
module's shipped gates on every draw — held-out > 0.85 on the non-linear fixture and > 0.30 over the
linear control — landing within roughly ±0.02 of the FA twin's measured 0.990 (XOR) and 0.993
(linear).

**What refutes adoption**, decided now: converging `B` failing to reach 6 of 8, OR reaching it while
dropping below the shipped accuracy gates on any draw. Either fires FR-007.

---

## D-4 — MEASUREMENT LOCATION

**Decision**: build the sweep in a **scratch probe in `tests/probe_015/`** (see the correction note at the end of this decision)
first; promote into the module's `main()` **only** what survives as a permanent criterion.

**Rationale.** `main()` re-derives its numbers on every gate run, so anything landed there is paid
for on every commit by every future lane — a 30-pass × 8-draw × 2-variant × per-epoch sweep is a
cost the gate should not carry to answer a one-time question. P2 also binds: the FA twin is already
*"a SCRATCH benchmark and deliberately not shipped"* precisely so a second live head does not exist.
The measurement follows the same discipline the module already chose for the same reason.

What DOES land permanently is the repaired criterion for whichever rule ships — one rule, one
criterion, re-derived every gate run.

**LOCATION CORRECTED DURING IMPLEMENTATION, 2026-08-14.** This decision first put the probe under
`specs/015-resolve-weight-transport/probe/`. `p4c_ratchet` went **RED** on it: a `.py` under
`specs/` opens a **new zone**, measured `specs: 1 > baseline 0`, and the tool says in its own
output *"Do NOT run --update-baseline to absorb it: that path refuses a rise by design."* Confirmed
by control — removing the file returned p4c to exit 0.

The probe now lives in `tests/probe_015/`. That is not a workaround: `p4c_ratchet` already
excludes `tests/**` as *"the instrument that measures a build, not a build"*, which is exactly what
this probe is. **The rejected alternative was adding a `specs/` exclusion** — that would be
silencing a check to fit the plan (Standing Invariant 2), and the 013 lane declined the same route
for the same reason. Verified: pytest does not collect it (no `test_` prefix) and it still runs
standalone.

The reasoning in this decision is unchanged — the probe stays OUT of `mlp_q.main()`, which
re-derives on every gate run. Only its directory moved.

---

## D-5 — COMMIT SEQUENCE, WITH THE DECISION POINT EXPLICIT

1. **C1 — record the degeneracy** (FR-001). Docs/docstring only. States what the old cosine compared
   and why no alignment rule could pass it. Lands FIRST so the repair is not retro-justified.
2. **C2 — the repaired criterion + its reachable red + the permutation null** (FR-002). No rule
   change yet. The criterion must be shown to separate good from bad BEFORE it judges anything.
3. **C3 — the scratch probe and the measurement** (FR-004/005/006). Produces the trajectory table.
   Changes no shipped behaviour.
4. **DECISION POINT.** Compare against D-3 as written.
   - **Supports adoption** → C4 replaces `dh`'s feedback with the converging `B`, DELETES the
     transport line, and counts survivors (FR-008). C5 updates the e-prop adjudication.
   - **Null** → **the lane stops at C4-null**, which commits the recorded null with its numbers,
     restates the site's disposition, and updates the adjudication to say the resolution was
     attempted and refuted. **The null is a deliverable, not a failure** (FR-007).
5. **C6 — the `predictive_coding` adjudication** (see D-6).

**Only C4-adopt touches module behaviour.** C1–C3 and C5 are records and instruments.

---

## D-6 — WHERE THE `predictive_coding` ADJUDICATION LANDS

**Decision**: a **separate final commit**, blocked by nothing and blocking nothing.

It is a decision artifact, not code (FR-012), and it concerns a different mechanism on a different
objective. **Deliberately independent of the `mlp_q` outcome** — R-004: a result at `mlp_q` does not
transfer to it in either direction, so sequencing it after adoption would invite exactly that
inference. It ships whether the `mlp_q` lane adopts or nulls.

**It must not rest on the Predictive E-prop preprint.** Measured on the full text: **0** hits for
`weight transport`, `weight symmetr`, `symmetric`, `feedback alignment`, `random feedback`, against
**30** for `BPTT`. The preprint solves temporal credit assignment. Its silence about transport is
silence, not endorsement.

**Both directions to be stated, neither pre-decided:**

- **FOR a violation**: `self.w1.T` is the same array transposed — exact weight symmetry by
  construction, which is the objection the invariant names, and it runs on every wake tick.
- **AGAINST**: in Rao-Ballard `W1` is the GENERATIVE weight (`x_hat = W1 @ r1`), so `W1^T @ s_e0` is
  one matrix in its two structural roles rather than a separate forward pathway being copied — the
  standard locality argument for predictive coding.

---

## Open questions carried into Phase 1

- **The decay term** (D-1). Settled by measurement in Phase 1, not by assumption.
- **Whether the repaired criterion has a reachable red at all.** If it cannot be made to fail, it is
  DELETED rather than weakened, and the site keeps its current disposition (spec Edge Cases). This
  is deferred to measurement per Standing Invariant 6 rather than resolved by argument here.
