# Feature Specification: Capability and Bounding (Phases 6–7)

**Feature Directory**: `specs/002-capability-and-bounding`

**Created**: 2026-07-31

**Source**: Phases 6 and 7 of `docs/COVERAGE_TO_100.md`; governed by `.specify/memory/constitution.md`

## Problem

Two phases remain in the ladder, and they differ from 1–5 in kind: both are **measurement-bound**.
No amount of code review settles them, and P0's guard — a capability reported without measurement is
fake — means neither can be closed by argument.

- **Phase 6, capability**: the two ceiling categories attacked where they are actually blocked. A
  real-scale AR spiking LM for Language, perceptual predicate induction for Gf. Held-out numbers, or
  an honest null.
- **Phase 7, bounding**: whether the brain can run continuously at all.

## Findings already in hand (2026-07-31)

**Phase 7 is partly answered, by measuring the operator's live daemon rather than a synthetic run.**

`brain_lake.bin` over 15.3 hours of continuous operation: 75.07 MB → 151.13 MB. Mean **4.98 MB/h**;
recent window **3.64 MB/h**. Extrapolated: **119 MB/day, 31–43 GB/year.**

A compaction path exists and is live: `KnowledgeLog.consolidate` (`substrate/home.py:152`), confirmed
LIVE with charge in the Phase 5 adjudication. **It bounds only half the lake.** Its own docstring:
*"replace the durable FACT records … preserving any non-FACT (spiking) records in the one lake
untouched."* So symbolic facts are pruned and merged; spiking records — and the `_T_GROWTH` blobs
written every sleep tick — are append-only with no prune path.

**The honest phase-7 statement: the brain cannot currently run indefinitely.** Not because a store
grows, but because exactly one half of the one lake has a prune policy and the other does not.

**Phase 6's Language half already has its honest null**, recorded in `docs/DEPTH_CEILINGS.md`: a
10.3M-parameter AR spiking LM trained to checkpoint step 1399 (~10% of schedule); BLiMP did not move;
every model row sits between the 29.4% length prior and the 53.0% char-level bar, at or below chance.

## Requirements

- **FR-001**: Phase 7 MUST state, with a measured rate, whether each durable store is bounded.
- **FR-002**: A prune policy for the spiking half MUST be specified before it is built, and MUST NOT
  discard records the brain later reads (the growth substrate is restored at construction).
- **FR-003**: Phase 6 MUST report held-out numbers or an explicit null. A partial training run
  reported as a capability is forbidden by P0.
- **FR-004**: Any capability claim MUST name the tier, the held-out split, and the control it beat.
- **FR-005**: Neither phase may be marked complete on a projection. A rate is evidence; an
  extrapolation from it is not a completed measurement.

## Success Criteria

- **SC-001**: Every durable store has a measured growth rate and a stated bound-or-unbounded verdict.
- **SC-002**: The spiking half of the lake either gains a prune policy with a measured effect, or is
  documented as unbounded with the operational consequence stated in GB/year.
- **SC-003**: Language and Gf each carry a held-out number or an explicit null, with controls.

## Assumptions

- Growth measured on the operator's live daemon generalises to other deployments only in shape, not
  in rate; the rate depends on tick frequency and ingest volume.

## Out of Scope

- Building the spiking prune policy. Specifying it is in scope; implementing it changes what the
  brain remembers and needs its own A/B and its own phase.
- Retraining the AR spiking LM to completion. That is GPU-hours, and the honest null already stands.

## Status

**Phase 7: the measurement is done, the fix is not.** The unbounded half is identified and rated.

**Phase 6: blocked on GPU-hours, not on analysis.** Language has its null. Gf's predicate induction
needs a training run this session cannot contain, and no honest claim can be made without it.
