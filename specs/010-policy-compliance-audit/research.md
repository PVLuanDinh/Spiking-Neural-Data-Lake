# Phase 0 Research: Policy Compliance (P0-P8)

All external sources pulled 2026-08-04. All in-tree citations re-read on the same date.

## R1 — What makes a criterion falsifiable, in a form we can implement

**Decision**: adopt Kleiner & Hoel's invariance criterion as the operational definition of P8's
"reachable red", rather than inventing our own.

**Source**: Kleiner & Hoel 2020, *"Formalizing Falsification for Theories of Consciousness Across
Computational Hierarchies"*, arXiv:2006.07390 [cs.AI], journal ref *Neuroscience of Consciousness*.

Verbatim criterion from the abstract:

> to avoid being unfalsifiable or already falsified scientific theories of consciousness must be
> invariant with respect to changes that leave the inference procedure fixed at a given level in a
> computational hierarchy.

The paper makes the abstract "unfolding argument" concrete: isomorphic digital circuits with and
without feedback — functionally equivalent machines realizable with table-top electronics — showing
IIT is *"simultaneously falsified at the finite-state automaton (FSA) level and unfalsifiable at the
combinatorial state automaton (CSA) level."*

**Rationale**: this converts P8's requirement from a slogan into a procedure with two steps —
(1) **name the level of abstraction** at which the criterion is evaluated, (2) **show the criterion
is not invariant** to a change that is meaningful at that level. It also diagnoses our defect
precisely and severely: `reasoning/consciousness_probe.py` takes no arguments, so it is invariant to
*every* change to the system under test, at *every* level. That is the strongest possible form of
the failure, not a borderline case.

**Alternatives considered**:
- *Invent a project-specific definition of "can fail".* Rejected — P8 permits invention but the
  burden falls on load-bearing output, and here a published, precise criterion already exists. P2's
  "better requires a measurement that exists" applies: we have no benchmark on which a home-grown
  definition could be shown better.
- *Use IIT / Φ directly.* Rejected on the source's own argument. A measure that cannot come out
  wrong cannot be evidence, and the project already refused a Φ-optimising framework on this ground.

## R2 — What a criterion that DOES go red looks like

**Decision**: use PCI as the template for the *shape* of a trustworthy criterion — perturb, then
measure the complexity of the response — and, critically, adopt its **validation discipline**:
demonstrate the measure in conditions where it is REQUIRED to fall.

**Source**: Casali et al. 2013, PMID 23946194, *Sci Transl Med*, "A theoretically based index of
consciousness independent of sensory processing and behavior."

Method: **(i) perturb** the cortex with TMS to engage distributed interactions (integration),
**(ii) compress** the spatiotemporal response to measure its algorithmic complexity (information).
Validated against wakefulness, dreaming, NREM sleep, and midazolam / xenon / propofol sedation, plus
vegetative state, minimally conscious state and locked-in syndrome. It *"reliably discriminated the
level of consciousness in single individuals."*

**Rationale**: PCI earns trust because anesthesia and NREM sleep are conditions in which it is
*required* to drop — its red is not hypothetical, it is exhibited. That is the discipline this
feature imports: **a criterion never demonstrated in a condition where it must fail has not been
validated.**

**The in-tree contrast, and it is stark.** This repo already has a `pci` telemetry key. Measured on
a live brain over 120 ticks it is **CONSTANT at 0.4068** — computed over a synthetic 10-unit toy
network whose only brain input is a single clamped scalar, with no consumer. We have PCI in name
only, which is a sharper illustration of the defect than any argument.

**Alternatives considered**: behavioural report as ground truth — rejected, the source paper exists
precisely because report is unavailable in the cases that matter, and our system's "report" is the
thing under suspicion.

## R3 — One number or many indicators

**Decision**: multi-indicator, not a single scalar. Where a marker survives, it reports alongside
its provenance and its distribution, never as one headline figure.

**Source**: Butlin, Long, Elmoznino, Bengio, Birch, Fleming, Mudrik, VanRullen et al. 2023,
*"Consciousness in Artificial Intelligence: Insights from the Science of Consciousness"*,
arXiv:2308.08708 (19 authors). Surveys recurrent processing theory, global workspace theory,
higher-order theories, predictive processing and attention schema theory, and derives **"indicator
properties" of consciousness, elucidated in computational terms"** that permit assessing existing AI
systems.

**Rationale**: it agrees with the project's own prior art and with the defect we found. A single
headline number is exactly what the cockpit currently draws untagged. The indicator approach is also
*assessable here* rather than hypothetical — this tree already implements GWT
(`thalamus/global_workspace.py`), predictive processing (`cortex/predictive_coding.py`) and
higher-order machinery, so the indicators have referents.

**Alternatives considered**: a single composite score — rejected; it is the failure mode
`docs/CONSCIOUSNESS_METHODS.md:101` already names, *"A single number going up is not evidence."*

## R4 — The project's own prior art is ahead of its code

**Decision**: do not invent a new principle for Phase A. Promote and enforce what these documents
already say.

- `docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md:8` — *"Nothing here is built. Nothing here claims the
  brain is conscious. Nothing here can be made to claim it."*
- `:57-58` — **"A mechanism whose success criterion cannot go red is worthless."**
- `:1064` falsifier **F11** — *"Any success criterion in this document turns out to have no reachable
  red → That criterion is deleted, not weakened. Three were deleted from the draft on exactly this
  ground."*
- `docs/CONSCIOUSNESS_METHODS.md:96` — *"The measurement, and it must be a dissociation, not a
  marker."*
- `:113` — what must never be claimed: *"That a passing dissociation means the system is conscious."*
- `docs/coverage/Consciousness.md:366` — already scores the defect: *"a 4/4 self-report that no live
  measurement backs."*

**Rationale**: F11 settles the delete-vs-repair question without a new judgement call, and it has
precedent — three criteria were already deleted on this ground. The remedy is the project applying
its own rule to code the rule had not reached.

## R5 — Delete or repair, per marker

**Decision**, from the measured state of each marker:

| Marker | Measured | Disposition | Why |
|---|---|---|---|
| BINDING | 1.000 across 300 random object sets; rate control's own docstring says it is *"right only by luck"* | **DELETE** | No constructible failing input. F11: deleted, not weakened. |
| RE-ENTRY | arithmetic over literals, `0.40 < 0.55 < 0.70`; no brain input, no RNG | **DELETE** | Not a measurement at all — a true statement about three constants. |
| INTEGRATION | reads ABSENT on ~10 of 20 subnet seeds | **REPAIR** | It HAS a reachable red. The defect is the invocation pinning `seed=0`, the seed that passes. |
| ACCESS | lapsed on 3/120 live state reads, via novelty habituation only | **REPAIR, downgraded** | Weak red exists but the assembly is coherent by construction, so the integration factor cannot go low on the live path. Report with that limitation stated. |

**Alternatives considered**: re-threshold BINDING so it can fail — rejected explicitly. Moving a
threshold on a measure that is 1.000 by construction manufactures a red without creating one, and
Standing Invariant 2 forbids narrowing a check to dodge its own finding.

## R6 — Where the new instruments live

**Decision**: `tools/`, registered as CI lint steps, not in `run_selfchecks.py`.

**Rationale**: precedent and its recorded reason. `tools/citation_guard.py` was placed there because
`run_selfchecks.py` only executes `python -m snn_data_lake.<mod>` package modules, so qualifying
would mean moving a repo-hygiene check into the package and churning the canonical 278 count quoted
in several documents. These two gates check the repository, not a brain faculty.

**Alternatives considered**: add to `run_selfchecks.py` — rejected on the above. Add to
`pre-commit` only — rejected; CI is the enforcement surface that cannot be skipped locally.

## R7 — Adoption discipline for the new gates

**Decision**: count each gate's output *before* it is adopted as asserting, and record the expected
first-run number in `contracts/`.

**Rationale**: this is the project's most load-bearing procedural rule and it has killed three
proposals — one at 0 failures / 47 UNCERTAIN, one at 58 failures where 1 was claimed, one at 474
violations across 126 files. The survivor of that third case only survived in a narrowed scope, at
21 violations, because 21 was a work list rather than an allowlist.

**Applied here**: both gates MUST fail on today's tree (they do — see plan Gate 2). If either
returns an allowlist-sized result, **the gate is dropped, not the tree.**

## Open questions carried into Phase 1

None blocking. One recorded non-blocker: the replacement criterion for consciousness — a measure
the brain's own state can move, shaped per R1/R2/R3 — is **explicitly out of scope** for this
feature and is the natural follow-on. This feature removes the false finish line; it does not claim
to have built the real one, and no artifact in this feature may imply otherwise.
