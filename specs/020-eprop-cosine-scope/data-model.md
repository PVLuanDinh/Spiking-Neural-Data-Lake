# Phase 1 Data Model

**Feature**: 020-eprop-cosine-scope | **Date**: 2026-08-17

All entities are in-memory or on-disk *gate metadata*. **None is brain state** — nothing here goes
into `brain_lake.bin` (P4b does not bind; see plan.md Constitution Check).

---

## LearningRule

One learning rule, identified per SYMBOL rather than per module — required because
`cortex/eprop_kernel.py` holds a rule, its BPTT control and a surrogate `autograd.Function` in one
file, so a per-module verdict cannot describe it.

| field | type | notes |
|---|---|---|
| `qualname` | str | `"<module>::<Class>.<method>"` or `"<module>::<function>"`. The identity key. |
| `module_rel` | str | repo-relative path. Derived, never typed. |
| `governed` | bool | inside REGIONS/TORCH_TIER/TRAIN_TIER vs the ungoverned subtree. |
| `declared` | Disposition \| None | from the module's `GRADIENT_CLAIM` constant; `None` = absent. |
| `inferred` | `CLAIM` \| `UNKNOWN` | **never `NO_CLAIM`** — see FR-002b. |
| `state` | ClassificationState | the resolved verdict. |

**Validation**: `qualname` unique across the census; every `LearningRule` resolves to exactly one
`state`; a rule with `declared is None` and `inferred == CLAIM` MUST resolve to `MISSING_DECLARATION`,
never to a claim-bearing state.

---

## Disposition (declared, authoritative)

Four values. Two are insufficient: `reasoning/gf.py::MLP` is an owner-EXEMPT **control**, and a
two-value classifier condemns it.

| value | meaning | owes a cosine? |
|---|---|---|
| `LOCAL` | local rule, asserts no gradient approximation | no |
| `APPROXIMATES_GRADIENT` | claims to approximate a gradient | **yes** |
| `EXACT_GRADIENT` | computes a gradient exactly (the independent reference) | no — it IS the reference |
| `CONTROL` | exists to lose / to be the thing compared against | no |

---

## ClassificationState

| value | how reached | asserted as |
|---|---|---|
| `CLASSIFIED_LOCAL` | declared `LOCAL`, inference not `CLAIM` | — |
| `CLASSIFIED_CLAIMANT` | declared `APPROXIMATES_GRADIENT` | owes a cosine |
| `CLASSIFIED_REFERENCE` | declared `EXACT_GRADIENT` | — |
| `CLASSIFIED_CONTROL` | declared `CONTROL` | — |
| `MISSING_DECLARATION` | no declaration, inference says `CLAIM` | **RATCHET** (may fall, never rise) |
| `UNDETERMINABLE` | no declaration, inference `UNKNOWN`, and no basis to decide | **RATCHET DOWN** (FR-016a) |

**`UNDETERMINABLE` counts as CLASSIFIED** (FR-016). Classified ≠ decided. Without this, "zero
unclassified, always" forces an undecidable rule out as a confident NO-CLAIM — measurably wrong on ≥5
load-bearing modules and forbidden by Standing Invariant 3.

**State transitions**: `UNDETERMINABLE` → `CLASSIFIED_*` when a declaration is added.
`MISSING_DECLARATION` → `CLASSIFIED_*` likewise. No transition may increase either ratcheted count.

---

## DiscriminationResult

Produced by the on-demand half, per rule the bar is applied to. **Always a pair** — no rule is judged
on one number (FR-003).

| field | type | notes |
|---|---|---|
| `qualname` | str | the rule under test |
| `aligned` | float | cosine of the candidate against the independent reference |
| `null_sample` | list[float] | the within-matrix permutation null (FR-009a) |
| `verdict` | `RULE_JUDGED` \| `INSTRUMENT_FALSIFIED` \| `UNAVAILABLE` | |
| `reason` | str | why, in words, for the non-`RULE_JUDGED` cases |

`INSTRUMENT_FALSIFIED` when no constructed input moves the measurement across the bar **in either
direction** — the rule is then left UNJUDGED (FR-004). `UNAVAILABLE` is distinct from a failed
measurement (FR-005): collapsing those two is a failure this project has already had.

---

## SeparationReport

| field | type | notes |
|---|---|---|
| `null_kind` | str | fixed: `within_matrix_permutation` |
| `aligned_sd_above_null` | float | where `0.5` sits |
| `p_null_exceeds_bar` | float | P(null > 0.5) |
| `reported_only` | dict | the three non-asserted nulls (isotropic, within-unit, random-relabel) |

The bar is **frozen** (FR-009b): this entity records placement, it never sets it.

---

## RatchetBaseline

| field | type | notes |
|---|---|---|
| `missing_declaration` | int | may fall, never rise |
| `undeterminable` | int | may fall, never rise |
| `n_rules_seen` | int | **liveness floor** — a dead walk reports zero of everything |

**Machine-generated only.** The writer MUST refuse to record a worse value (FR-018). Measured
precedent for why: one existing gate stores its baseline as a source-level dict, and a ceiling raised
*together with the regression it excuses* passes both its `--assert` and its `--self-check` — verdict
**laundered clean**.
