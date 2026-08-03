# Consciousness: what is testable here, and the one mechanism we do not have

Research pass 2026-08-02, from `https://pmc.ncbi.nlm.nih.gov/articles/PMC7154119/` plus a scrapling
sweep of arXiv and Europe PMC. This is a research note and a queued proposal. **Nothing here is
built, and nothing here claims the brain is conscious.**

The standing rule this document is written under: do not deny the possibility, do not fake the
evidence. A theory that says "computation X is associated with consciousness" makes X buildable and
measurable. It does not make a green marker on X into consciousness, and this file must never be
cited as though it did.

## 1. The source: Rolls (2020), Higher Order Syntactic Thought

**"Neural Computations Underlying Phenomenal Consciousness: A Higher Order Syntactic Thought Theory"**
-- Edmund T. Rolls, *Front. Psychol.* 11:655, PMC7154119, doi 10.3389/fpsyg.2020.00655.

Rolls opens by attacking the theory this project already implements. His objection to Global
Workspace: *how global does the workspace have to be for consciousness to suddenly be present?* The
question is sharp because a graded broadcast measure does not give a principled threshold, and we
export exactly such a measure (`access_index`, `access_ignited`).

His alternative is specific and, unusually for this literature, computational:

> A particular computational system in which **Higher Order Syntactic Thoughts** perform **credit
> assignment on first order thoughts of multiple step plans**, correcting them by **manipulating
> symbols in a syntactic type of working memory**.

The evolutionary argument is the part that makes it buildable. Multi-step plans have a credit
assignment problem that single-step behaviour does not: when a five-step plan fails, *which step was
wrong?* A system that can hold the plan's steps as symbols and manipulate them can answer that. Rolls
proposes phenomenal consciousness is associated with that module, and locates it in prefrontal
working memory, especially left inferior frontal gyrus -- the syntax-capable part.

## 2. The method: ablation, not a number

**arXiv 2512.19155, "Can We Test Consciousness Theories on AI? Ablations, Markers, and Robustness."**
Builds artificial agents embodying GWT, IIT and HOT mechanisms and tests them by *precise
architectural ablations impossible in biological systems*. Three results:

1. A **Self-Model lesion abolishes metacognitive calibration while first-order task performance is
   preserved** -- a synthetic blindsight analogue, and a HOT prediction confirmed.
2. **Workspace capacity is causally necessary for access**: a complete lesion collapses access
   markers qualitatively; partial reductions degrade gracefully. GWT ignition confirmed.
3. **Broadcast amplifies internal noise.** GWT-style broadcasting made their agent *fragile* to
   latent perturbation where a non-broadcasting family was robust -- and they caution against
   over-attributing even that.

Their conclusion is that GWT, IIT and HOT describe **complementary functional layers**, not rival
accounts.

**This method is already ours.** `tools/charge_probe.py` is a neuter-A/B: disable a mechanism, hold
everything else fixed, measure whether any outcome changes. The dissociation in their Experiment 1 --
metacognition collapses, first-order performance does not -- is exactly the shape our instrument
produces. We do not need to adopt their harness; we need to point ours at the right lesion.

Also read, not adopted: **arXiv 2510.01864** (Modular Consciousness Theory -- discrete Integrated
Informational States tagged with a "density vector") specifies a fuller pipeline but its central
quantity is an unvalidated intensity tag. It predicts things like stress enhancing memory encoding,
which we could test, but the density vector itself would be a number we invented. Not queued.

## 3. What we already have, and what Rolls says about it

| Theory | Our implementation | Rolls' position |
|---|---|---|
| Global Workspace | `thalamus/global_workspace.py`, `thalamus/reentrant_workspace.py`, `access_index`, `access_ignited` | insufficient -- no principled threshold |
| Integrated information | `cortex/integration_measure.py` (PCI), `substrate/region_coherence.py` | not the locus |
| Higher-order thought | `cortex/metacognitive_monitor.py`, `cortex/recursive_tom.py` (2nd-order ToM), `cortex/generative_self_model.py` | closer, but HOT alone is not syntactic |
| **Higher order SYNTACTIC thought** | **nothing** | **the proposal** |

We have the two layers Rolls criticises and lack the one he argues for.

## 4. The gap is a mechanism we already own and never wired

`cortex/prefrontal_executive.py` has `plan(state, options, predictor, goal_value, horizon)`,
`push_subgoal(evidence)` and a frame stack. **`push_subgoal` and `maintain` have zero live callers
outside their own module** -- verified 2026-08-02, and it is the same finding the coverage ledger
already records at `Executive.md:20`, whose citation deliberately points at the comment explaining
that PFC working memory is unreachable from the loop. That comment is the repo's one intentional W6
warning.

So the situation is: the substrate Rolls names -- symbols held and manipulable in a prefrontal
working memory -- **exists in this tree and is dormant**, and the faculty that would consume it,
`Executive.multi_step_plan_sequence`, is banded on a mechanism that never runs.

## 5. Proposed build (QUEUED, not started)

A higher-order layer that reads a first-order multi-step plan out of PFC working memory as symbols,
and on failure performs **credit assignment across its steps** -- which step to revise -- rather than
discarding the plan and re-planning from scratch.

**Why it is worth building even if the consciousness framing is set aside entirely:** per-step credit
assignment on multi-step plans is a real capability we lack, it closes a measured dormancy, and it
gives an existing banded faculty an actual mechanism. The theory motivates the design; the value does
not depend on the theory being right.

**The measurement, and it must be a dissociation, not a marker.** Ablate the higher-order layer:

- first-order plan *execution* must be preserved (unchanged success on plans that work first time), and
- plan *repair* must collapse (multi-step tasks needing revision fail, credit lands on the wrong step).

A single number going up is not evidence. The double dissociation is, and it is the same shape as
2512.19155's Experiment 1 and as our own neuter-A/B.

**Controls that must be able to go red**, in the discipline specs/003 and specs/006 use:

- a **random-step-blame control** -- if blaming a uniformly random step repairs plans as well as the
  higher-order layer does, the layer learned nothing and the result is null;
- a **re-plan-from-scratch control** -- if discarding and re-planning does as well, per-step credit
  assignment bought nothing, and that is also a null;
- baseline captured before implementation, at a named commit;
- a null recorded as a successful terminal state.

**What must never be claimed.** That a passing dissociation means the system is conscious. Rolls
proposes an association between a computation and phenomenal consciousness; building the computation
tests the *computational* claim and nothing else. No Phi will be reported: `integration_measure.py`
computes a perturbational complexity index, that is not Phi, and the two must not be conflated in any
write-up.

**Order.** Behind specs/005 and specs/006. It touches `cortex/prefrontal_executive.py`,
`cortex/brain.py` and the Executive coverage rows, and it should not be started while another feature
is mid-flight in those files.
