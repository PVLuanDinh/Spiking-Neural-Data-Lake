<!--
Sync Impact Report
==================
Version change: (none) -> 1.0.0
Rationale: MAJOR/initial. Ratifies the seven policies already governing this repo
(docs/POLICIES.md P0-P6) as the project constitution. No principle was invented, renamed, or
redefined here; this file is a governance surface over an existing authority.

Principles (all pre-existing, adopted verbatim in substance):
  + P0 Autonomous self-improvement, by every available avenue
  + P1 Brain-native / no-hardcode
  + P2 Adopt-better or merge, then retire the inferior (non-additive)
  + P3 No dormancy
  + P4 One brain / no replication
  + P5 Use the GPU for what it measurably speeds up
  + P6 Every build must reach every other, and actually be used

Added sections:
  + Standing Invariants (never relaxed by any policy)
  + Development Workflow and Quality Gates

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md      ⚠ pending (Constitution Check section should cite P0-P6)
  - .specify/templates/spec-template.md      ⚠ pending
  - .specify/templates/tasks-template.md     ⚠ pending
  - docs/POLICIES.md                          ✅ unchanged and REMAINS AUTHORITATIVE (see Governance)

Deferred TODOs: none. RATIFICATION_DATE taken from the first commit of docs/POLICIES.md
(2026-07-24); LAST_AMENDED_DATE from its most recent change (2026-07-30).
-->

# Spiking Neural Data Lake Constitution

This constitution restates the policies in `docs/POLICIES.md`. That file remains the
authoritative, longer-form text with worked cases and measurements. Where the two ever
disagree, `docs/POLICIES.md` wins and this file is the defect.

## Core Principles

### P0. Autonomous self-improvement, by every available avenue (PURPOSE)

The brain MUST improve itself autonomously: self-writing code, knowledge acquisition, and neural
growth. A missing avenue is a gap to close, not a boundary. Safety is what makes this pursuable,
not what limits it — the self-build is sandbox-verified and byte-exact rolled back, and the brain
may rewrite what it thinks, never what stops it.

The honesty guards are the operative half and are NON-NEGOTIABLE:
- Growth is NOT improvement. A grown unit that changes no outcome is dormant.
- A grown unit that is not KEPT is not even growth.
- A self-written duplicate is bloat.
- **A capability reported without measurement is fake.**
- Scaffolding a not-yet-measured capability is PERMITTED, on three conditions: label it as
  infrastructure, claim no band or avenue until measured, and name the gap at the site.

### P1. Brain-native / no-hardcode

Behaviour-arbitrating values MUST be learned weights or running-baseline-relative forms. A magic
constant MUST NOT be kept fixed where a computed value belongs (three documented exceptions live
in `docs/POLICIES.md`). A constant sitting where a computed value belongs is this project's
signature defect; a value that MOVES but arbitrates nothing is its mirror image, and movement
MUST NOT be read as life.

### P2. Adopt-better or merge, then retire the inferior (NON-ADDITIVE)

When two mechanisms do the same job, exactly ONE survives. Either ADOPT (the better becomes the
path, delete the other) or MERGE (build the one mechanism superseding both, retire both). A
"merge" that keeps both paths behind a flag is additive with extra steps. A/B before deleting,
and port a retired mechanism's unique capability BEFORE deleting it. "Better" requires a
benchmark that exists.

### P3. No dormancy

Every module MUST be wired-and-charged, retired, or EXEMPT with a reason. Dormancy is per-METHOD,
not per-module: if a live module computes a learning signal, the rule consuming it MUST also be
called from a live path. Mechanized by `substrate/dormancy_gate.py` and
`substrate/method_dormancy_gate.py`.

Honesty guard: **reachability is NOT charge.** Import proves import. Only a neuter-A/B proves a
wire carries behaviour, and a consumer MUST NOT be faked to satisfy a gate.

### P4. One brain / no replication

Use the canonical `Brain -> LiveBrain -> AutonomousArm` composition; never re-implement it.
Construct only via `cortex.brain.make_brain()`. Mechanized by `substrate/replication_guard.py`.

### P5. Use the GPU for everything it actually speeds up

"Actually" is decided by MEASUREMENT, not expectation. The measured CUDA crossover is
N≈24,576–32,768 while live populations are 24–240, so the train/gpu tiers belong on the card and
the live tick does not. Re-measure when populations grow. The deterministic CPU gate stays
authoritative and the suite MUST remain runnable without a GPU, because CI has none.

### P6. Every build must reach every other, and actually be used

A build satisfies P6 when it is reachable from at least one other FACULTY and at least one other
faculty consumes it in a way that changes THAT CONSUMER's output. Provisional and
review-enforced: its gate is specified but not built, so review is its enforcement.

## Standing Invariants (never relaxed by any policy)

1. **Count a gate's output before any rule leans on it.** Implement the check, run it, publish
   what it finds, and only then let a policy depend on it.
2. **Never silence a check to make it pass.** No exemption that hides the class of defect the
   check exists to catch; no re-recording of a moving number; no narrowing a heuristic to dodge
   a true positive. An exemption MUST carry a disposition and a reason.
3. **The negative verdict is the dangerous one**, because it is the one that retires a working
   faculty. Where a probe cannot distinguish "did not fire" from "fired and did not matter", it
   MUST report UNKNOWN rather than a confident negative.
4. **Absence of a warning is not evidence of absence.** A check reports a floor, not a census,
   unless proven otherwise.
5. **Verify the verifier.** A green gate that structurally cannot see the thing it checks is
   worse than no gate, because it manufactures confidence.

## Development Workflow and Quality Gates

Three gates MUST stay green, and a change is not done until they are:

```
python run_selfchecks.py          # every core module's __main__ self-check
pytest                            # >=70% line coverage
python -m ruff check .            # E, F, I, UP, B
```

Evidence discipline for the coverage ledger (`docs/coverage/`):
- Every score cites `file:line`; a score without a citation is not a score.
- Nothing above band 40 without a runtime A/B whose recorded exit code passed.
- Nothing above band 80 without held-out evidence.
- A faculty that cannot be measured stays in the ledger as UNSCORED, counting 0. A row MUST NOT
  be deleted to raise a score.

## Governance

`docs/POLICIES.md` is the authoritative policy text; this constitution is its governance
surface and MUST NOT contradict it. Amendments to a principle here REQUIRE the corresponding
amendment in `docs/POLICIES.md`, with the measurement or worked case that motivated it.

Versioning: MAJOR for a backward-incompatible principle removal or redefinition, MINOR for a new
or materially expanded principle, PATCH for clarification.

Compliance: every plan produced under Spec Kit MUST pass a Constitution Check against P0–P6 and
the Standing Invariants before implementation. Complexity MUST be justified against P2 and the
ponytail rule that the best code is the code never written. `CLAUDE.md` and `docs/POLICIES.md`
are the runtime guidance files.

**Version**: 1.0.0 | **Ratified**: 2026-07-24 | **Last Amended**: 2026-07-30
