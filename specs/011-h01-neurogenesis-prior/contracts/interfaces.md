# Phase 1 Contracts

**Feature**: `011-h01-neurogenesis-prior` | **Date**: 2026-08-06

This project is a library organised by brain region. The interfaces below are the ones other modules
and the gates depend on; they are stated as contracts rather than signatures so the plan does not
pre-empt the implementation.

---

## C1 — The prior module (`cortex/cortical_prior.py`)

**Exposes**: the constants of [data-model.md §3](../data-model.md), each as a record carrying value,
units, provenance and accuracy bound.

**Contract**
- Every exported constant carries all four attributes. **A constant missing any is a construction
  error, not a warning.**
- The module is **pure data plus pure functions**. No I/O on import, no module-scope side effects, no
  network, no lake access.
- It is read **once** per population construction. It exposes no mutable state and holds no reference
  to a brain.
- Replacing the source dataset is **one edit to this file** and touches nothing else.

**Self-check must demonstrate** (it is registered in `run_selfchecks.py` from the start, P4c):
the four-attribute completeness of every constant, with a **run red** for a constant whose bound has
been stripped.

---

## C2 — Layer assignment

**Given** a position and the layer geometry, **returns** a layer and whether the point was
extrapolated.

**Contract**
- Input positions are **nanometres**; the geometry is **microns**. The conversion happens **here,
  once, explicitly**, and the direction is asserted.
- **The scale assertion is the contract's most important clause.** If the position scale is
  inconsistent with the geometry scale, the call **fails loudly**. It must not return the
  fall-through layer, because a wrong-unit input otherwise assigns *every* point to the fall-through
  bucket and produces a result that looks valid (research D2).
- Assignment is **first-match-wins in the geometry's own order**, matching the source. No
  reclassification pass.
- Depth is **radial in the x-y plane**. `z` is never consulted (research D3).
- A point outside the measured volume still receives a layer and is flagged `extrapolated` (plan D3).
- Deterministic: same input, same output, no iteration-order dependence.

---

## C3 — Sign assignment

**Given** a cell type, **returns** a sign or "unsigned".

**Contract**
- Sign is a **property of the cell**, derived only from its type. Never from its connections, never
  from context.
- An unsigned type returns **unsigned** — never a default. The caller excludes it from the signed
  population.
- `L1` + excitatory is **unrepresentable**, not merely unlikely.
- Once assigned, sign is **immutable**. No learning path may change it (FR-003, SC-004).

---

## C4 — The growth interface (extends the existing engine)

**Contract**
- Every unit is given position, layer and sign **at creation**, not by a later annotation pass
  (FR-001).
- The growth report states, per cycle: units grown, the per-layer distribution, the count placed
  outside the measured volume, and the fraction that landed in the fall-through layer.
- Growth remains deterministic for a fixed seed (SC-009).
- **This extends the one growth mechanism. It does not add a second** — feature 007 owns the
  per-region growth interface; if this contract requires a parallel one, this feature is wrong.

---

## C5 — The drift instrument

**Given** a seeded population and its prior, **returns** a `DriftRecord` per seeded statistic.

**Contract**
- Works on **any** seeded population, including one seeded randomly or from a deliberately wrong
  prior — that is what makes the three-arm comparison possible.
- Reports **zero drift as a failure** of the seeding approach, never as stability (FR-009).
- **Excludes sign** from drift by design (Dale is an invariant, not a tunable).
- Requires **no GPU** and no live brain beyond the population under test.
- Built and shown to work **before** any seeding exists (plan, Phase B) — an instrument validated only
  against the thing it was built to bless is not an instrument.

---

## C6 — The circuit's input port

**Given** an input signal, the layered circuit **produces a response**.

**Contract**
- Two distinguishable inputs produce **different** responses; one input twice produces **identical**
  responses (SC-007).
- The port is the *only* new surface — the circuit is extended, never duplicated (FR-016).
- No GPU requirement; CI has no card.
- Removing the port must make the criterion fail. **A criterion that survives the removal of the thing
  it tests is measuring something else.**

---

## C7 — Persistence

**Contract**
- Grown placement state persists to the **one** lake under its own record tag.
- The proof is a **cross-process read-back**: written by one process, read by a **new** one. An
  in-memory round-trip proves nothing (P4b).
- Foreign records are passed through, never dropped.
- **The prior does not persist.** It is reference data versioned with the code, and writing it to the
  lake would make the lake a second copy of something the repo already owns.
