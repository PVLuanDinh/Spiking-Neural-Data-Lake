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

**Five** values. Two are insufficient: `reasoning/gf.py::MLP` is an owner-EXEMPT **control**, and a
two-value classifier condemns it. Four were insufficient too — found by USE, not design (FR-002e).

| value | meaning | owes a cosine? |
|---|---|---|
| `LOCAL` | local rule, asserts no gradient approximation | no |
| `APPROXIMATES_GRADIENT` | claims to approximate a gradient | **yes** |
| `EXACT_GRADIENT` | computes a gradient exactly (the independent reference) | no — it IS the reference |
| `CONTROL` | exists to lose / to be the thing compared against | no |
| `NOT_A_RULE` | not a learning rule at all — a diagnostic, a search, a symbolic-grammar operation, or a combinator over someone else's gradient | no — there is nothing to approximate |
| `DIRECT_SOLVE` | a genuine rule that solves in closed form — **non-local**, but propagates no gradient and transports no weight | no — nothing for the cosine to measure |

`LOCAL` and `DIRECT_SOLVE` both answer "no", and the difference between them is the point:
`LOCAL` **asserts locality**, `DIRECT_SOLVE` explicitly does not. POLICIES.md requires that distinction
be recorded "distinctly rather than lumping it in", because *non-local* and *backprop* are different
claims and only the second is forbidden.

`NOT_A_RULE` is **declaration-only by construction** (FR-002f): the inference cannot emit it and no
default resolves to it, so it is not a drawer the classifier can sweep into. Its count measures
**enumeration imprecision** — the recall-biased walk over-capturing — and never tree health.

---

## ClassificationState

| value | how reached | asserted as |
|---|---|---|
| `CLASSIFIED_LOCAL` | declared `LOCAL`, inference not `CLAIM` | — |
| `CLASSIFIED_CLAIMANT` | declared `APPROXIMATES_GRADIENT` | owes a cosine |
| `CLASSIFIED_REFERENCE` | declared `EXACT_GRADIENT` | — |
| `CLASSIFIED_CONTROL` | declared `CONTROL` | — |
| `CLASSIFIED_NOT_A_RULE` | declared `NOT_A_RULE` | — (reported as an imprecision meter) |
| `CLASSIFIED_DIRECT_SOLVE` | declared `DIRECT_SOLVE` | — (owes no cosine; locality NOT asserted) |
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
| `coherence` | tuple[float, float] | `|mean|/rms` of candidate and reference (FR-009d) |
| `verdict` | `RULE_JUDGED` \| `INSTRUMENT_FALSIFIED` \| `UNAVAILABLE` | |
| `reason` | str | why, in words, for the non-`RULE_JUDGED` cases |

`coherence` is not decoration: it is the PRECONDITION on whether `null_sample` means anything. A
within-matrix permutation preserves the per-family mean, so at high coherence the null sits above the
bar and the separation statistic inverts (measured: null mean +0.980, aligned 1.75 sd BELOW it). A
`DiscriminationResult` carrying a `null_sample` and no `coherence` is a number with no licence.

`INSTRUMENT_FALSIFIED` when no constructed input moves the measurement across the bar **in either
direction** — the rule is then left UNJUDGED (FR-004). `UNAVAILABLE` is distinct from a failed
measurement (FR-005): collapsing those two is a failure this project has already had.

---

## SeparationReport

| field | type | notes |
|---|---|---|
| `null_kind` | str | fixed: `within_matrix_permutation` |
| `bar_sd_above_null` | float | where **`0.5`** sits, in sd above the null mean |
| `aligned_sd_above_null` | float | where the **aligned cosine** sits, in sd above the null mean |
| `p_null_exceeds_bar` | float | P(null > 0.5) |
| `draws` | int | pinned; unpinned it is not reproducible (FR-009e) |
| `seed` | int | pinned; measured spread across 5 seeds is +2.97 to +3.27 sd |
| `reported_only` | dict | the three non-asserted nulls (isotropic, within-unit, random-relabel) |

**Two fields, not one, and the split is the point.** This entity first carried a single
`aligned_sd_above_null` while the printed line it feeds reads `0.5 sits ... sd above null`. Those are
different quantities — measured on the anchor fixture, `0.5` sits at **+3.06 sd** and the aligned
cosine at **+5.93 sd** — so one field feeding both readings would have been right in the report and
wrong in the JSON, or the reverse.

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
