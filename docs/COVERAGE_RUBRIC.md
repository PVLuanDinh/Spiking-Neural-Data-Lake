# Brain-Faculty Coverage Rubric

**What this file is for.** Two independent coverage sweeps have been run on this project and reported
numbers — `~31%` at v3.79 and `19.5% -> 22.8%` at v4.0. Neither is reproducible. The rubric they scored
against was never written down: it lived in the auditors' prompts and in commit-message prose, so the
categories, the weights, the bands and the evidence standard all differ between the two, and the second
sweep's own commit body has to explain that its *lower* number came from "a stricter audit" rather than
a regression. A percentage nobody can re-derive is not a measurement, and P0's honesty guard is explicit
that **a capability reported without a measurement is a fake capability**.

So this file is the instrument. It is tracked, so the next sweep is comparable to the last one, and a
disagreement about a score becomes a disagreement about evidence rather than about what the question was.

**This rubric does not reconcile with `22.8%`.** The prior weighting is unrecoverable. Scores produced
here are a fresh baseline; reporting a delta against the old number would be inventing a comparison.

---

## The eleven categories

Reconstructed from the v4.0 sweep's own category list. **Equal weight (1/11 each)** — not because equal
weighting is obviously right, but because the prior weights are unrecoverable and an invented weighting
would smuggle a thumb onto the scale. Weights are a knob to argue about openly, in this file, if someone
has a case.

| # | category | what it scores |
|---|---|---|
| 1 | **Perception** | sensory transduction, encoding, attention, multimodal binding |
| 2 | **Memory** | episodic / semantic / working store, encoding, consolidation, retrieval |
| 3 | **Executive** | goal maintenance, planning, conflict monitoring, control-of-control, inhibition |
| 4 | **Consciousness** | global availability, ignition, re-entry, access, self-model |
| 5 | **SpatialTemporal** | place/grid coding, path integration, interval timing, sequence |
| 6 | **Affect** | valence/arousal, appraisal, emotion categories, somatic marking |
| 7 | **Social** | theory of mind, empathy, pragmatics, joint attention |
| 8 | **Imagination** | forward models, counterfactual rollout, dreaming, generative recombination |
| 9 | **Language** | grammar, generation, comprehension, discourse |
| 10 | **Gf** (fluid intelligence) | novel-problem solving, analogy, rule induction at test time |
| 11 | **WholeBrain** | inter-region wiring, oscillatory coordination, homeostasis, self-improvement |

---

## The scoring bands — the part that does the work

The recurring finding across every audit of this codebase is one pattern: **a faculty computed every
tick and then discarded into telemetry.** The bands exist to make that pattern score what it is worth,
which is not much, and to make the honest thing to write down also the easy thing.

| band | meaning | what it takes to claim it |
|---|---|---|
| **0** | absent | nothing implements it |
| **1–20** | **built, not live** | the code exists; no live path imports it (or it is EXEMPT). `dormancy_gate` reachability is the evidence |
| **21–40** | **live, not consumed** | on the wake/sleep tick, but its output reaches only telemetry, `state()`, or a log. **This is where most of this brain sits.** A dashboard bar is not a behaviour |
| **41–60** | **changes an internal read** | the output demonstrably alters something the brain itself later reads — a reply, a memory write, a self-model term. Requires an A/B, not a call site |
| **61–80** | **changes a decision, action, or learning update** | proven by a runtime neuter-A/B: neuter the signal, and the action / value / weight measurably differs. `BEHAVIOURAL CHARGE OK` |
| **81–100** | **generalizes** | all of the above, and it holds outside the developer-constructed case — held-out data, an unrehearsed input distribution, or an out-of-grammar instance |

**A category's score is the evidence-weighted mean over its constituent faculties**, not the maximum.
One charged faculty does not lift a category whose other five are telemetry.

---

## Evidence standard

1. **Every score cites `file:line`.** A score without a citation is not a score.
2. **Nothing above 40 without a runtime A/B.** Static reading has been wrong in *both* directions in
   this project's audits — a coverage sweep once flagged PFC and amygdala as dormant and both were
   refuted on runtime A/B; conversely `region_wiring`'s registry read as a working plug point for its
   whole life while being structurally empty. Reachability proves import. Only neutering proves charge.
3. **Nothing above 80 without held-out evidence.** "98.3% on synthetic RAVEN" is a band-61–80 claim.
   "80% on rendered, re-perceived held-out panels" is a band-81+ claim. The difference is the whole
   distance between a benchmark and a capability.
4. **A precondition that never fires scores 0, however good the code is.** If the guard around a
   faculty is false on every live tick — a flag `Brain.__init__` never passes, a key space that never
   collides, a registry with no registrations — then the faculty is absent in effect. Score what runs.
5. **Downgrades are results.** A category scoring lower than last time because the bar was applied
   properly is the instrument working. Say so in the same sentence; never quietly re-weight to protect
   a number.
6. **An honest null is reportable and a fabricated gain is not.** If a wave moved nothing, the score
   does not move.

---

## Procedure

One auditor per category, read-only, working independently, each returning:

- the score and the band-by-band reasoning,
- the constituent faculties with `file:line` and the band each one earned,
- a **capability inventory** — which file provides what — because Policy 2 requires porting a retired
  mechanism's unique capability *before* deleting it, and this inventory is that evidence,
- the gaps, ranked by `distance_to_100 x feasibility`.

Then an adversarial pass: each score above 40 is handed to an independent reviewer whose job is to
**refute** it, defaulting to refuted when uncertain. This is aimed at the sweep and not only at the
audit on purpose — the sweep produces the single headline number, so it carries the most inflation risk.

## Known ceilings, stated before scoring so nobody scores around them

- **Wiring a faculty into `cortex/brain.py::_route_percept_to_action` moves a three-way arm decision
  (ENGAGE / RELEASE / HOLD).** It can raise Executive, Affect, SpatialTemporal and the action-relevant
  part of Perception. It **cannot** raise Language, Imagination, Gf, Consciousness or WholeBrain —
  crediting those to a wiring wave is the theater P0 forbids.
- **Language and Gf are at measured architectural ceilings**: BLiMP 47.4% (chance) on a fully-trained
  808k-parameter e-prop LM, and ARC 4.75% held-out (19/400, re-derived 2026-09-02 at 4f20a7e8; 2.75% on 2026-08-03) with the failure mode proven to be DSL coverage,
  not search budget. Points there require capability, not wiring, and saying so is not pessimism —
  it is what stops the number being fiction.
