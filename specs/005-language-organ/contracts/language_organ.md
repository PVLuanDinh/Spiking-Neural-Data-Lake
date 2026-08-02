# Contract: `cortex/language_organ.py`

**Feature**: `specs/005-language-organ` | **Data model**: [../data-model.md](../data-model.md)

The organ's surface, and what an implementation must guarantee at each entry point. Obligations are
numbered so tasks and tests can cite them. Every obligation is checkable.

Module tier: **numpy-gated CORE**, matching the incumbent it replaces (`run_selfchecks.py:369`). It must
not raise the tier - no torch on the organ's path.

---

## 0. Module-level obligations

- **C0.1** No module-level `random.seed()` or `np.random.seed()`. Every draw comes from an instance rng.
- **C0.2** No `loss.backward()`, no autograd, no batch gradient, no weight transport (reading a forward
  weight transposed on a backward pass). Hebbian and error-modulated Hebbian only.
- **C0.3** Every quantity that arbitrates behaviour names its derivation in a comment at its definition
  site (P1, R5). A bare numeric literal in a behaviour-arbitrating position is a defect.
- **C0.4** All `print()` output is ASCII. Counts print as `N of M`, never `N/M`.
- **C0.5** `main()` is a self-check that returns 0 on pass and nonzero on fail, registered in
  `run_selfchecks.py`, and it asserts something that can fail. A self-check that asserts only that
  construction did not raise is the empty-pipe pattern `spiking_lm_serve.py` already demonstrates.

---

## 1. `Area(name, k, rng, *, n=None, p=None, beta=None)`

One population with local inhibition.

**Preconditions**

- `k >= 1`, and `k` is supplied by the organ from its acquired lexicon size, never defaulted to a literal.
- `rng` is the organ's instance rng. An `Area` never constructs its own.

**Derivations** (each asserted; a construction that cannot satisfy its own derivation **refuses**)

| quantity | rule |
|---|---|
| `n` | from `k` and the number of assemblies the area must keep distinguishable, by the birthday-collision argument. Comment states the argument |
| `p` | from `n` and `k`, so expected drive spread is discriminable by the cap. A density that makes all units receive near-identical drive makes the cap a coin flip |
| `beta` | from observed convergence. **OPEN at contract time** - whichever derivation is chosen is written here and at the definition site before implementation proceeds |

**Contract obligations**

- **C1.1** `len(active) <= k` at all times; `== k` whenever any drive is nonzero.
- **C1.2** `1 <= k < n`. A cap equal to `n` is no inhibition.
- **C1.3** Recurrent and inbound weights are non-negative. Inhibition is the cap, not a negative weight.
- **C1.4** Two `Area`s built from equal-state rngs, in the same order, with the same arguments, are
  identical field for field.
- **C1.5** A refused construction changes nothing and raises with a message naming which derivation could
  not be satisfied. It does **not** clamp to a nearby value.

---

## 2. `Area.project_into(target, *, plastic) -> bool`

The single primitive. Returns whether the target's assembly changed.

**Semantics**

1. Each unit of `target` accumulates summed weight from this area's currently active units.
2. Local inhibition keeps the top `target.k` most-driven units; ties broken deterministically by unit
   index, never by an unseeded shuffle.
3. If `plastic`, Hebbian strengthening is applied **only** to synapses whose presynaptic unit was active
   in this area AND whose postsynaptic unit is in the new selection.
4. Returns `True` if `target.active` changed.

**Contract obligations**

- **C2.1** With `plastic=False`, **no weight anywhere changes.** This is what makes read-only scoring
  possible and it must be asserted by a test that snapshots weights across a scoring run (FR-015).
- **C2.2** Tie-breaking is deterministic and documented. A tie broken by a fresh random draw destroys
  same-seed reproducibility in exactly the case hardest to notice.
- **C2.3** The update touches no synapse with an inactive endpoint. Locality is the claim; a rule that
  reaches further is not local, whatever it is called.
- **C2.4** No normalization step may be introduced that reads a global statistic over all synapses. That
  is non-local, and it is how a Hebbian rule quietly becomes something else.

---

## 3. `LanguageOrgan.project_until_stable(src, dst, *, plastic) -> (rounds, converged)`

**Termination is by convergence, not by count** (FR-007, R5).

**Contract obligations**

- **C3.1** Terminates when `project_into` returns `False` - the assembly stopped changing.
- **C3.2** A guard bound exists only to prevent non-termination, is **derived** (not a literal), and is
  documented as a guard.
- **C3.3** Reaching the guard bound returns `converged=False`, increments `self.refusals`, and is
  **reported**. It is never silently treated as convergence. This is the silent-degradation pattern the
  repo already has three recorded instances of, and it is the most likely way this organ acquires a fake
  capability.
- **C3.4** `refusals` appears in whatever the organ exposes as state, so a rising refusal count is
  visible rather than buried.

---

## 4. `LanguageOrgan.expose(words, scene) -> None`

One grounded exposure item. **The joint update** (FR-006).

**Contract obligations**

- **C4.1** Lexical structure (`PHON <-> LEX`, `LEX <-> SCENE`) and syntactic structure (`SEQ`, and the
  inter-area schedule) are updated in the **same pass** over the item. No stage boundary, no
  freeze-then-thaw, no two-phase curriculum. This is 2406.11977's finding and the spec's FR-006.
- **C4.2** The inter-area schedule is **learned**, not written. A test asserts the schedule after
  exposure differs from the schedule at construction; if it does not, nothing was learned about order or
  role, whatever the downstream accuracy says.
- **C4.3** A word with no KB referent is still exposed. Dropping ungroundable items is selection on the
  outcome.
- **C4.4** `lexicon` gains an entry for each newly acquired word, and `k` is re-derived as the lexicon
  grows (R5). Growth must not alter what was already learned - the same non-destructiveness `specs/003`
  requires of population growth (its FR-001), and for the same reason.
- **C4.5** `n_items` increments by exactly one.

---

## 5. `LanguageOrgan.acceptability(sentence) -> float | None`

The scorer handed to `blimp_bench.score()`.

**Contract obligations**

- **C5.1** **Read-only.** Plasticity is off for the whole call; `C2.1` is what makes this true. A test
  snapshots every weight before and after a full scoring pass and asserts equality (FR-015).
- **C5.2** Returns `None` - never a fabricated number - when any word of the sentence is outside the
  lexicon. `blimp_bench.cached_logprob` maps `None` to the `UNSCORABLE` sentinel, which always loses its
  comparison; that is the honest behaviour and it must not be circumvented by returning a low float
  instead, because `per_char` would then divide it by length and resurrect the length bug that
  `UNSCORABLE`'s pass-through exists to prevent.
- **C5.3** The returned value is **monotone in acceptability** - higher is more acceptable. `score()`
  requires only an ordering, not a log-probability, so a convergence-quality measure is admissible.
- **C5.4** Deterministic: the same sentence on the same organ returns the same value every call.
- **C5.5** It is passed to `blimp_bench.score()` **unmodified**. The instrument is read-only for this
  feature; an instrument edited by the party being measured is not an instrument.

---

## 6. `LanguageOrgan.role_of(word, sentence) -> str | None` and `generate(scene) -> list[str]`

The grounded-acquisition surface. Supporting evidence, **not** the deciding benchmark (R1).

**Contract obligations**

- **C6.1** `role_of` returns an acquired role or `None`. It is measured against the most-frequent-role
  baseline computed by the same harness on the same exposure, and **both numbers are printed**. A win
  against nothing is otherwise invisible.
- **C6.2** `generate` produces a word sequence from a scene by rolling the learned schedule forward. It
  is never a template fill; a test asserts that at least one generated sentence is not a verbatim
  exposure item (the "novel sentence" claim of 2507.11788), and if none is, the claim is withdrawn.
- **C6.3** Both are read-only.
- **C6.4** Held-out items used here are never exposed in any arm.

---

## 7. `LanguageOrgan.save(path)` / `load(path)`

**Contract obligations**

- **C7.1** Atomic JSON via `substrate/home.py`'s `save_state` / `load_state`, following `specs/003`'s
  precedent. **Not** the one lake - weights stay STATE and there remains exactly one durable store.
- **C7.2** A round trip is exact: `save` then `load` yields an organ producing identical output.
- **C7.3** A restored organ whose derived quantities disagree with its persisted ones **refuses** and
  says which, rather than silently truncating learned state. `specs/003` names this edge case for the
  same reason.
- **C7.4** Restart survival is required if the organ is adopted: the daemon relaunches ~20 times in a
  working session, so an organ that forgets on restart has not acquired anything durable.

---

## 8. Integration with `cortex/brain.py`

**Contract obligations**

- **C8.1** The organ is built in the existing `Brain._lm_serve` slot (`brain.py:3499`). No new branch, no
  new construction path, no second `Brain` (P4).
- **C8.2** `Brain._language` (`brain.py:3541`) keeps its shape: it calls the organ, applies the coherence
  guard, and returns `""` on failure so `respond()` falls back to the narrator's fresh self-grounded
  summary (`brain.py:3077`). The fallback is what makes a bad generation harmless and it stays.
- **C8.3** `_lm_serve` returns `None` on any failure, as today, so the brain degrades rather than raising.
- **C8.4** On ADOPT, the organ's output **changes the reply** on the Gf-fluid branch. A generator that
  changes no reply is dormant, not progress (P3, FR-024).
- **C8.5** Persistence writes replace the `language_acq.npz` writes at `brain.py:2607` and `:3656`. They
  are replaced, not added alongside.
- **C8.6** ASCII enforcement at `respond()` stays, and the organ additionally emits ASCII at the source.

---

## 9. Registration and gates

- **C9.1** `cortex.language_organ` is registered in `run_selfchecks.py` in the **numpy-gated CORE** tier.
- **C9.2** On ADOPT, `cortex.language_acquisition` is removed from `run_selfchecks.py:369` in the same
  change that deletes the file. The module count change is **stated**, not silent (SI-2).
- **C9.3** All four gates green before and after: `run_selfchecks.py`, `pytest --cov-fail-under=70`,
  `ruff check .`, `ruff format --check src tests tools`. No assertion loosened.
- **C9.4** Interpreter is `.venv-lake\Scripts\python.exe`.
