# Phase 1 Data Model: The Brain-Native Assembly-Calculus Language Organ

**Feature**: `specs/005-language-organ` | **Spec**: [spec.md](./spec.md) | **Phase 0**:
[research.md](./research.md)

Entities, their fields, what is persisted versus per-trial, and the invariants an implementation must
assert rather than assume. Every quantity that arbitrates behaviour names its derivation (P1, R5).

---

## Entity: `Area`

One population of excitatory units with local inhibition. The organ's only structural building block.

### Persistent state (survives restart; part of the saved organ)

| field | type | derivation | notes |
|---|---|---|---|
| `name` | str | fixed identity | one of `PHON`, `LEX`, `SCENE`, `SEQ` |
| `n` | int | **DERIVED** from `k` and the separability requirement (R5) | asserted at construction, never a literal |
| `k` | int | **DERIVED** from the acquired lexicon size | the cap; local inhibition. Grows with what the organ has learned |
| `w_in` | dict[(src_area, src_unit, dst_unit)] -> float, or a sparse array | random at construction from the per-instance rng, then Hebbian | inbound synapses per source area |
| `w_rec` | same shape, source = self | recurrent synapses | what makes an assembly stable |
| `p` | float | **DERIVED** from `n` and `k` (R5) | connection density; asserted in range |
| `beta` | float | **DERIVED** from observed convergence (R5) | Hebbian rate. OPEN: exact derivation is a contract-time choice |
| `n_exposed` | int | counter | how many exposure items have touched this area |

### Per-trial state (cleared by `reset()`, never persisted)

| field | type | notes |
|---|---|---|
| `active` | frozenset[int] | the currently firing units - the assembly |
| `drive` | array[n] | this step's summed input; recomputed, never carried |
| `rounds` | int | projection rounds used this step |

### Invariants

1. `len(active) <= k` always; `== k` whenever any drive is nonzero. The cap is not advisory.
2. `k >= 1` and `k < n`. A cap equal to the area is no inhibition at all.
3. `n`, `k` and `p` are each asserted against their derivation at construction. A construction that
   cannot satisfy its own derivation **refuses** rather than clamping to a nearby value - a clamp is a
   literal wearing a computation's clothes.
4. Every random draw comes from the instance's own rng. Two `Area`s built from the same seeded rng in the
   same order are identical, field for field.
5. `w_rec` is never negative. Inhibition in this model is the cap, not a negative weight; a negative
   recurrent weight would be a second inhibition mechanism, which is the additive move P2 forbids.
6. Hebbian updates touch **only** synapses whose two endpoints were both active. A rule that touches
   inactive synapses is not local.

### State transitions

| from | trigger | to | notes |
|---|---|---|---|
| empty | `fire(units)` | active | external drive; the sensory entry at `PHON` |
| active | `project_into(target)` | target active | one projection round; Hebbian applied when plasticity is on |
| active | round with unchanged `active` | **converged** | the termination condition. Not a count |
| active | guard bound reached | **refused** | increments the refusal counter; reported, never silently treated as converged (R5) |
| any | `reset()` | empty | per-trial state only; weights untouched |

---

## Entity: `LanguageOrgan`

The four areas plus the schedule. Occupies the existing `Brain._lm_serve` slot.

### Persistent state

| field | type | derivation | notes |
|---|---|---|---|
| `areas` | dict[str, Area] | four, fixed | `PHON`, `LEX`, `SCENE`, `SEQ` |
| `schedule` | plastic structure over (step, src_area, dst_area) | **LEARNED** | inter-area inhibition: which areas may project into which at each step. **Must not be hardcoded** - a written schedule is a written grammar (plan Risk 5) |
| `lexicon` | dict[word] -> assembly id in `LEX` | acquired | the enumerable record of what the organ can ever have learned; R1's coverage computation reads it |
| `seed` | int | given | the organ's own; `random.Random(seed)` per instance |
| `n_items` | int | counter | exposure items consumed |
| `refusals` | int | counter | projections that hit the guard bound |

### Per-trial state

`prompt`, the current sentence's per-step assemblies, and each area's `active` set. All cleared by
`reset()`.

### Invariants

1. **Determinism**: two organs constructed with the same seed and given the same exposure in the same
   order are identical, and produce identical output. Asserted by a test (SC-012).
2. **No module-level randomness.** `random.seed(...)` and `np.random.seed(...)` at module scope are
   forbidden; every draw comes from the instance.
3. **Plasticity is switchable off for evaluation and off by default during scoring** (FR-015). An organ
   whose weights move while being benchmarked has been taught its own test.
4. **The schedule changes with exposure.** A test asserts it: if the schedule after exposure is identical
   to the schedule at construction, nothing was learned about order or role, whatever the accuracy says.
5. **`lexicon` is enumerable and honest.** A word the organ has not been exposed to is absent, and the
   organ returns "no opinion" for it rather than a fabricated assembly - mirroring
   `rose_language.ground()`, which returns `None` for an honest lexical gap and never fakes one.
6. **No literal arbitrates behaviour.** Each of `n`, `k`, `p`, `beta` names its derivation at its
   definition site; the projection loop terminates on convergence.
7. Output is ASCII. Counts print as `N of M`.

### State transitions

| from | trigger | to |
|---|---|---|
| constructed | `expose(sentence, scene)` | one joint update of lexical and syntactic structure; `n_items += 1` |
| any | `acceptability(sentence)` | a real number, or `None` for out-of-lexicon. **Read-only** - no weight moves |
| any | `generate(scene)` | a word sequence produced by rolling the schedule forward |
| any | `role_of(word, sentence)` | the acquired syntactic role, or `None` |
| any | `save(path)` / `load(path)` | atomic JSON via `home.save_state` / `load_state` |
| any | `reset()` | per-trial state cleared; weights and lexicon untouched |

---

## Entity: Grounded exposure item

The unit of exposure. Sourced from the knowledge base, not authored (R6).

| field | type | source |
|---|---|---|
| `words` | list[str] | rendered from the triple |
| `scene` | (head, relation, tail) | `reasoning/world_kb_stream.py::load_kb()` |
| `referents` | dict[word] -> KB symbol | `reasoning/rose_language.py::ground()`, `None` for a gap |
| `split` | `TRAIN` or `TEST` | **positional and deterministic** |

### Invariants

1. The split is **positional**, not a shuffle - a shuffle nobody recorded is a split nobody can
   reproduce. `specs/003` records that a positional split over ordered data is a *temporal* holdout with
   different statistics from a random one; the same caveat applies and must be stated wherever a held-out
   number is reported.
2. A TEST item is never exposed, in any arm.
3. The **shuffled** control arm permutes `scene` against `words` and holds everything else identical -
   the same items, the same order, the same split, the same seed.
4. An item whose words are entirely ungroundable is still a valid item; it teaches word statistics and
   nothing else, and the organ must not silently drop it. Dropping items that do not suit the mechanism
   is selection on the outcome.

---

## Entity: Restricted pair set (`coverage.md`, frozen before any scoring)

The BLiMP subset both arms are scored on. **Written once, committed, and not regenerated after either
arm has been scored** (FR-023).

| field | type | notes |
|---|---|---|
| `rule` | str | the stated property, e.g. "every word of both sentences is in the organ's exposure lexicon" |
| `lexicon_source` | str | the exposure plan the lexicon was computed from - the KB inventory, not a trained organ |
| `paradigms_in` | int | reported as `P of 67`; floor **8** |
| `pairs_in` | int | reported as `N of M`; floor **200** |
| `pair_ids` | list | the frozen membership. Both arms score exactly these |
| `frozen_at_commit` | str | the pin |

### Invariants

1. Membership is a function of the rule and the lexicon **only**. No arm's score is an input.
2. Both arms score exactly `pair_ids`. Not "the same rule applied separately" - the same list.
3. Missing either floor produces the **not-P2-eligible** outcome, recorded. It does not produce a wider
   corpus and a retry.
4. `paradigms_in` and `pairs_in` print as `P of 67` and `N of M`, so a shrinking denominator is visible.

---

## Entity: Verdict record

| field | type | notes |
|---|---|---|
| `outcome` | `ADOPT` \| `NULL` \| `NOT-P2-ELIGIBLE` | carried by the **exit code**, not the prose |
| `organ_sum`, `organ_norm` | float | both scorings |
| `incumbent_sum`, `incumbent_norm` | float | both scorings, same pairs |
| `bigram_sum`, `bigram_norm` | float | the control that currently beats the incumbent |
| `random_ctl`, `constant_ctl`, `length_ctl`, `unexposed_ctl` | float | admission controls, same pairs, same invocation |
| `pairs`, `paradigms` | int | `N of M`, `P of 67` |
| `commit` | str | the pin |
| `artifact` | path | via `tools/coverage_score.py record --kind ab` |

### Invariants

1. **ADOPT requires all of**: organ > incumbent under *both* scorings, AND organ > bigram control under
   both, AND all four admission controls failing as specified. Any one unmet is `NULL`.
2. The band ceiling reads the **recorded exit code**, never the claim. Editing the record cannot change
   the exit code; that is the anti-forgery core of the ledger.
3. The recorded command must be `--rerun`-able - `coverage_score rerun` does `a["command"].split()`, so a
   `python -c` one-liner records fine and dies on rerun, which is the forgery the mechanism exists to
   catch dressed as compliance.
4. `NULL` is terminal and successful. It does not license a re-chosen pair set, a substituted metric, or
   a parameter sweep followed by a re-run (FR-005).

---

## Entity: Density report (arXiv 2607.26648 check)

| field | type | notes |
|---|---|---|
| `H` | int | units in the recurrent state |
| `M` | int | distinguishable sentence states the organ must carry - **derived and shown**, not asserted |
| `rho_floor` | float | `H_b^{-1}(log2(M) / H)` |
| `rho_measured` | float | mean active fraction over the evaluation run |
| `dense_statement` | str | the plain statement required by FR-021 |

### Invariants

1. `rho_measured < rho_floor` is an **anomaly**, not a success: the organ is carrying less information
   than it needs to. It must be reported as such.
2. The report states that **compute is dense regardless of activity sparsity**, and that the low floor
   here follows from the organ's small `M` - i.e. from it doing less than a language model - so scaling
   toward open-domain prediction raises the floor and removes the sparsity (R7).
3. No field of this report is an acceptance criterion (FR-021).

---

## What is NOT an entity here

- **A new durable store.** One lake, `brain_lake.bin`. The organ persists as STATE via atomic JSON.
- **A second `Brain`.** P4.
- **A configurable area topology.** Four areas; a fifth only on a measurement that shows four cannot
  express something.
- **A "language organ registry" or plugin surface.** There is one organ, and if a second is ever built,
  P2 says one of them gets deleted.
