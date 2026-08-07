# Phase 1 Data Model

**Feature**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06

Five entities. Two of them are *state* and live in the lake; two are *reference* and live with the
code; one is a *measurement* produced on demand. Keeping that split is a P4b requirement, not a
stylistic one.

---

## 1. `GrownUnit` — **learned state, persists to the lake**

A neuron the brain grew, with the place it was born into.

| Field | Type | Units | Notes |
|---|---|---|---|
| `unit_id` | int | — | stable within a brain |
| `x`, `y`, `z` | float | **nanometres** | never voxels; see the validation rules |
| `layer` | enum | — | `L1, L2, L3, L4, L5, L6, WHITE_MATTER` — the source's seven, before the circuit mapping |
| `sign` | enum | — | `EXCITATORY`, `INHIBITORY`. **Immutable after creation.** |
| `cell_type` | str | — | the source's morphological type that decided the sign |
| `extrapolated` | bool | — | true if placed outside the measured volume (plan D3) |
| `born_tick` | int | — | when it was grown |

**Validation rules**
- `sign` is present iff `cell_type` is in a signed class. An unsigned type produces **no signed
  unit** — never a default to excitatory (FR-002).
- `layer == L1` implies `sign == INHIBITORY`. The source measures **zero excitatory somata in
  layer 1** (research D8); an excitatory L1 unit is a bug, not an edge case.
- Coordinates are nanometres. Any ingestion path that could yield voxels must convert and assert
  before constructing.

**State transitions**: `position`, `layer`, `cell_type` and `sign` are **write-once at creation**.
Only `extrapolated` is derived, and it is derived at the same moment. Nothing later may rewrite them —
the P1 movability requirement applies to *seeded statistics*, not to a cell's identity.

---

## 2. `PlacementState` — **learned state, the lake record**

The persisted collection of `GrownUnit`s under its own record tag in `brain_lake.bin`.

- **Must** survive process death: written by one process, read back by a **new** one. "It persists" is
  a claim about a new process (P4b), so the proof is a cross-process read-back, not an in-memory
  round-trip.
- Foreign records are passed through, never dropped.
- An in-memory index over it is a **cache**, never the source of truth.

---

## 3. `CorticalPrior` — **reference data, versioned with the code**

Every externally sourced constant, in **one module**, so a second dataset replaces all of them in one
edit (FR-006). **Must not enter the lake** — it is not something the brain learned.

Each entry carries four things, and an entry missing any of them is invalid:

| Attribute | Why it is mandatory |
|---|---|
| `value` | — |
| `units` | the 2× voxel-grid trap and the micron/nanometre trap both live here (research D2) |
| `provenance` | which file, which field, which population it was measured over |
| `accuracy_bound` | a number that loses its bound becomes a fake measurement (research D7) |

**Contents**

- `LAYER_CIRCLES` — six circles (centre, radius) in **microns**, ordered; first-match-wins.
- `LAYER_E_FRACTION` — per layer. **Fractions, not counts** (plan D1, research D9).
- `SIGNED_TYPES` / `UNSIGNED_TYPES` — the Dale mechanism (research D4).
- `GP_STRENGTH_E` / `GP_STRENGTH_I` — N=1..20, with the AIS-exclusion caveat in provenance.
- `SOURCE_VOLUME_NM` — 3.864 × 2.362 × 0.168 mm, for the extrapolation test.

**Degenerate inputs that must be handled explicitly** (P1 checks at zero/empty/silence/single):
an empty layer, a single-member layer, a fraction of exactly 0.0 or 1.0, and a layer absent from the
source entirely.

---

## 4. `DriftRecord` — **measurement, produced on demand**

Per seeded statistic. This is the entity the feature's headline criterion reads.

| Field | Notes |
|---|---|
| `statistic` | which seeded quantity |
| `seeded` | the value at construction |
| `current` | the value now |
| `distance` | how far it moved |
| `direction` | toward or away from the seed |

**Rules**
- `distance == 0` across all statistics is a **failure of the seeding approach** (FR-009), reported as
  such — not as stability, and not as success.
- Sign is **excluded** from drift by design (Dale; plan D1 mechanism 2). Including it would score the
  invariant as a hardcode.

---

## 5. `CircuitPopulation` — **the substrate, extended not duplicated**

The existing layered circuit's populations, mapped from the source's seven regions per plan D2.

- L2/3, L4, L5, L6 each with an e and an i pool (existing)
- **L1 inhibitory-only** (new; the deliberate departure)
- White matter: **excluded and counted**, not silently dropped

**Exactly one layered-population mechanism exists after this feature** (FR-016, SC-010). This entity
is the existing circuit with an input port and a layer-1 population, never a parallel structure.

---

## Relationships

```
CorticalPrior ──seeds──▶ GrownUnit ──aggregates into──▶ CircuitPopulation
      │                      │
      │                      └──persists to──▶ PlacementState (the lake)
      │
      └──baseline for──▶ DriftRecord ◀──current value from──┘
```

The prior is read **once**, at construction. Nothing on a live path reads it again — that is what
makes a seeded value movable rather than pinned (plan, P1 mechanism 1).
