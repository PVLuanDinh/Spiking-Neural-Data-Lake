# Data Model: whole-brain neurogenesis

**Feature**: specs/007-whole-brain-neurogenesis | **Phase**: 1 | **Date**: 2026-08-03

Entities are described by what they must hold and what must be true of them, not by a class layout.
The plan deliberately does not choose between a protocol, a base class and a registry -- see
research.md Decision 1 -- so this document constrains the data, not the shape.

---

## GrowthSeam

A place in the brain where a population can widen. Sixteen exist today; one has a working floor.

| Field | Meaning | Validation |
|---|---|---|
| `region` | Which of the 15 populated regions owns it (or `reasoning`, which is outside the region map) | MUST be a populated region or `reasoning`. The 2 empty regions cannot host a seam. |
| `width` | Current live unit count | Non-negative integer. Monotone non-decreasing within a run. |
| `floor` | The derived halting condition | MUST be derivable from the region's own measured dispersion. A seam whose floor is absent is `admitted = False` and does not grow. |
| `admitted` | Whether this seam may grow at all | `False` until its floor has been built AND A/B'd. This field is what prevents 15 dormant growers. |
| `grown` / `refused` | Counters | Both MUST be reported. A seam with `refused == 0` over a run where growth occurred has an untested gate and MUST NOT be reported as earning. |
| `safety` | Whether the seam is on the effector envelope | If true, growth is refused unconditionally. Never widened by this feature. |

**State transitions**: `unadmitted -> admitted` requires a passing floor A/B, and is one-way within
a run. `admitted -> growing -> saturated` is the normal path; `saturated` means the floor now
refuses every candidate site, and is a SUCCESS state, not a failure.

**Critical invariant**: a seam's units are zero-filled at creation and therefore emit nothing.
`width` increasing is NOT evidence of anything. Only `charge` (below) is.

---

## GrownUnit

Capacity added at runtime.

| Field | Meaning | Validation |
|---|---|---|
| `emits` | What the unit contributes downstream | **Zero at creation, by construction.** A unit that has not been trained since creation cannot contribute, and any design assuming otherwise is refuted by research.md. |
| `charged` | Whether this unit changes an outcome | Established ONLY by neuter-and-measure against a matched A/A control. Defaults to `False`. |
| `earned` | Which floor admitted it, and at which site | Recorded so a refusal can be distinguished from a never-asked. |

**The distinction that carries the feature**: `emits != 0` is necessary and not sufficient for
`charged`. A unit can emit a nonzero value into a consumer that discards it -- which is exactly what
`brain.py:1603`'s int cast does to the timing refinement today, and what `max(rb.values())` does to
faculty identity in the sibling defect.

---

## DemandSignal

The evidence that a seam's current width is insufficient.

| Field | Meaning | Validation |
|---|---|---|
| `quantity` | What is measured | MUST be computed by the region about itself. |
| `reference` | What it is compared against | MUST be the region's own running baseline or measured dispersion. **A fixed cutoff, a top-p quantile, or any constant is a P1 violation** -- a quantile needs `p`, and `p` is the constant relocated. |
| `self_extinguishing` | Whether satisfying demand reduces it | MUST be true. Measured: a benefit test that is never satisfied ran 2000 growths to width 2064 without going red once. |

**Only one instance exists today**: `resolvable()` backed by `first_passage_sd()`. It is the sole
measured constant-free self-extinguishing quantity in the tree, and it produces every refusal in the
system -- neutering the held-out benefit arm is byte-identical to leaving it on (4 grown, 196
refused, both arms).

---

## ConsumptionRecord

The feature's ledger, and its acceptance evidence. One row per claimed producer/consumer pair.

| Field | Meaning | Validation |
|---|---|---|
| `producer` / `consumer` | The two ends | MUST be in different modules. Same-module consumption satisfies nothing. `REGIONS["cortex"]` contains `time_cells`, `brain` and `executive_control`, so a cortex-internal pair is disqualified even though it crosses files. |
| `observable` | What moved | MUST NOT be a growth counter, the grown component's own return value, or any field on the mechanism's own side of the neuter. FR-002 names these. |
| `ab_delta` | The measured movement | Recorded with the exact command. |
| `aa_delta` | The matched A/A control | **MUST be 0.** A nonzero A/A means the harness has run nondeterminism and the A/B is unreadable. Phase 0 caught two false positives this way. |
| `identity_sensitive` | Whether the consumer depends on WHICH producer, not just magnitude | FR-005. A consumer documented "permutation-invariant" -- as `executive_control.conflict()` is, with an assert enforcing it -- can never satisfy this. |

**Validation rule for the whole feature**: a `ConsumptionRecord` with `aa_delta != 0`, or whose
`observable` is on the forbidden list, is not evidence and MUST NOT be counted toward SC-001 or
SC-003.

---

## RegionInventory

What `tools/growth_census.py` produces.

| Field | Validation |
|---|---|
| `produces` | Whether the region grows its own population, distinguishing OWN growth code from a BORROWED mechanism it merely embeds. Four regions currently grow only by embedding `GrowingLearner`. |
| `consumes` | Whether another region's growth changes this region's output, backed by a `ConsumptionRecord`. |
| `both` | The feature's headline count. |

**Reporting rule**: always `"N of 15, 2 regions empty and excluded"`, with the `reasoning` package
reported separately because it sits outside the region map. The denominator is never silently
shrunk to flatter the ratio.

**Known measurement floor**: name-based scanning undercounts, because `GrowingLearner` recruits
inline in `observe` rather than in a `grow*` method. The census reports a floor, not a census, and
must say so in its own output.
