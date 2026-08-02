# Implementation Plan: The Brain-Native Assembly-Calculus Language Organ

**Feature Directory**: `specs/005-language-organ` | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/005-language-organ/spec.md`, Phase 0 from
[research.md](./research.md)

**Baseline commit for every figure in this plan**: **not yet pinned.** HEAD moved during authoring
(`5599e6ab` -> `30d8e76a`) because a second agent is committing concurrently. T001 pins it, on a tree
verified clean for `cortex/language_acquisition.py`, `cortex/spiking_lm_serve.py`, `cortex/brain.py`,
`reasoning/blimp_bench.py` and `tools/`. Every number quoted in this plan is quoted from the tracked
document that recorded it, never from an unverified working tree - `specs/003` recorded the incident this
avoids, where a research pass measured an uncommitted rewrite and labelled it the baseline.

## Summary

Replace the brain's language architecture. `cortex/language_acquisition.py` - a linear readout over a
summed bag of 2,992 parameters, not clearing the 50.0% BLiMP chance floor (the 51.0% figure is
**withdrawn** by `docs/DEPTH_CEILINGS.md:555` as stale; the incumbent's score on this feature's frozen
pair set is **measured at T014**, not quoted), beaten by a character
bigram on identical text (52.8%), plateauing at 53.2% after a 4,119x exposure increase, with all four
alternative causes falsified by ablation - is replaced by an assembly-calculus language organ built from
the six principles of arXiv 2507.11788: excitatory units, brain areas, random synapses, Hebbian
plasticity, local inhibition (the per-area cap), and inter-area inhibition (the learned disinhibition
schedule).

Phase 0 changed the approach in four ways the spec's first draft did not anticipate:

1. **The benchmark's coverage question is a gate, not a detail, and it comes before any organ code.**
   BLiMP's vocabulary vastly exceeds any grounded lexicon; an organ that cannot score most pairs ties
   them, and ties count as wrong, so it loses by construction. R1 fixes a lexical-coverage restriction
   rule with stated floors (8 of 67 paradigms, 200 pairs) **committed before either arm is scored**, and
   an explicit not-P2-eligible outcome if the floors are missed. This is the cheapest place the feature
   can end, so it is scheduled first.
2. **The restriction must be computed from the exposure plan, not from a trained organ**, or the
   before-scoring guarantee is impossible. This forces the lexicon inventory ahead of construction.
3. **The organ is slower and this is unconditional.** A projection is a dense matrix product with a
   top-k, iterated to convergence, per word, per open area pair - replacing a 34x88 matrix-vector
   product. R7 states it up front; FR-022 requires the number.
4. **Four constants and a loop bound are P1 hazards at the heart of the architecture.** The AC
   literature states `n`, `k`, `p`, `beta` and a round count as literals. R5 derives each and replaces
   the round count with convergence detection plus a *reported* refusal counter, because a silently
   accepted iteration cap is the silent-degradation pattern.

**A null result is a successful terminal state** (FR-005). If the organ does not beat the incumbent AND
the char-bigram control on the restricted set under both scorings, nothing is deleted, the organ is not
wired into `respond()`, the outcome is recorded, and the feature closes COMPLETE.

## Technical Context

**Language/Version**: Python 3.11 and 3.13 (both in CI). Interpreter for all commands:
`.venv-lake\Scripts\python.exe` - **not** the `python` on PATH.

**Primary Dependencies**: numpy (the organ's areas). No torch on the organ's path: the incumbent it
replaces is in the numpy-gated CORE tier (`run_selfchecks.py:369`), and the organ must not raise the
tier. `reasoning/blimp_bench.py` (RESEARCH, `:351`) is the instrument.

**Storage**: `home.save_state` / `load_state` atomic JSON, following `specs/003`'s precedent
(`time_cells.json`) and deliberately **not** the one lake. Weights stay STATE; there remains exactly one
durable store (`brain_lake.bin`), and this feature adds none.

**Testing**: `python run_selfchecks.py`; `pytest` (testpaths `tests`, `--cov-fail-under=70`);
`python -m ruff check .`; `ruff format --check src tests tools`. All four are CI gates and all four are
binding before and after every slice.

**Target Platform**: ubuntu-latest and windows-latest, CPU only. CI has no GPU. Output is ASCII
(Windows cp1252).

**Project Type**: single Python package (`src/snn_data_lake/`), ten anatomical regions, plus `tools/`
probes and a tracked coverage-artifact ledger.

**Performance Goals**: none are acceptance criteria. FR-022 requires the `Brain.respond` latency delta to
be **measured and reported** whichever way it falls. A regression is a result. FR-021 forbids any
sparsity, density or energy target from being an acceptance criterion, and requires the measured firing
density to be reported against the derived arXiv 2607.26648 floor.

**Constraints**:
- No `loss.backward()`, no autograd, no batch corpus gradient, no weight transport. Hebbian and
  error-modulated-Hebbian only; the standing rule pivots any backprop consideration to e-prop.
- No numeric literal may arbitrate behaviour (P1). See R5's derivation table.
- Construction through `make_brain()`; the organ occupies the existing `Brain._lm_serve` slot (P4).
- `random.Random(seed)` or a numpy `Generator` **per instance**; module-level seeding is forbidden.
- Counts printed as `N of M`, never `N/M` - the artifact fingerprint harvests bare ratios positionally.
- Recording harnesses live in tracked `tools/`, never `scratchpad/` (gitignored;
  `tools/citation_guard.py` gates it).
- Scoring is strictly read-only in every arm; plasticity must not fire during evaluation.
- Probes isolate `NORD_HOME`, because the always-on daemon writes `~/.snn_data_lake` continuously.

**Scale/Scope**: one new module plus one new probe, one module deleted on ADOPT, one adjudicated. Areas
sized by R5's derivation from a KB lexicon in the low thousands of words - three orders below this repo's
measured CUDA crossover (N ~ 24k), so the tick stays on CPU and P5 is satisfied by saying why.

## Constitution Check

Evaluated against `.specify/memory/constitution.md` (P0-P6 plus the Standing Invariants), which defers to
`docs/POLICIES.md` where the two differ.

| principle | assessment | verdict |
|---|---|---|
| **P0** autonomous self-improvement | The organ acquires language online from exposure, with no gradient training and no pretrained weights - it is the knowledge-acquisition avenue, and it keeps learning as the brain converses (subject to R8's `expose()` port question). P0's honesty guard is respected: the census of what the brain has is not a measure of what it can do, so the feature is scored by the A/B and not by the module's existence. | **PASS** |
| **P1** brain-native / no hardcode | The five literals at the centre of the AC literature are the main risk. R5 derives `k` from the acquired lexicon, `n` from `k` plus separability, `p` from `n` and `k`, `beta` from observed convergence, and replaces the round count with convergence detection. Each derivation names itself at its definition site. **Honest limit recorded in R5**: the derivations themselves encode choices, and claiming the organ has "no constants" would be fake compliance. | **PASS, with the limit stated** |
| **P2** adopt-better or merge, then retire | This is the feature's spine. One survivor: `language_acquisition.py` is deleted on ADOPT (FR-010), never flag-gated. "Better" rests on a benchmark that already exists (`blimp_bench.py`, FR-001) - the guard that refused the Thousand Brains workspace swap. Unique capabilities ported or recorded as dropped before deletion (FR-011). A/B before deleting. | **PASS** |
| **P3** no dormancy | On ADOPT the organ is on the live `respond()` path (FR-024); a generator that changes no reply is dormant, not progress. `cortex/spiking_lm_serve.py` is adjudicated to one of the three dispositions regardless of the organ's verdict (FR-012) - it is dormant today with a passing CORE self-check, which is the pattern this policy exists to catch. | **PASS** |
| **P4** one brain / no replication | No new `Brain`, no new durable store, no re-implemented composition. The organ is built inside the existing `_lm_serve` slot; construction stays `make_brain()`. Every probe constructs through it. | **PASS** |
| **P5** GPU where it measurably helps | Areas sized in the low thousands, three orders below the measured CUDA crossover (N ~ 24k). Tick stays on CPU, and the reason is stated rather than assumed. | **PASS** |
| **P6** every build reaches every other and is used | The organ consumes `world_kb_stream` (scenes), `rose_language.ground()` (referents), and the existing local plasticity rule; it is consumed by `Brain._language` -> `Brain.respond`. That is faculty-to-faculty in both directions, not merely wiring into the single action gate. | **PASS** |
| **SI-1** count a gate's output before a rule leans on it | US1 is *only* this. The degenerate controls run on the exact restricted set the verdict uses, before the baseline is taken. | **PASS** |
| **SI-2** never silence a check to make it pass | FR-020. No assertion loosened, no self-check count silently reduced; the tier change at deletion is stated. | **PASS** |
| **SI-5** never proceed on an instrument that cannot see its subject | R1's coverage floors, with an explicit not-P2-eligible exit. This is the invariant that makes the feature's first task a possible last task. | **PASS** |
| **Evidence standard** | Every FR names an observable. Band > 40 needs a recorded-exit-code A/B, produced through `coverage_score.py record --kind ab` (FR-016). Band > 80 is acknowledged unreachable through the sanctioned path (F14) and is not budgeted for. A null leaves the number where it is (FR-025). | **PASS** |

**No violations requiring justification. Complexity Tracking is therefore empty.**

Two things are recorded as *accepted limits* rather than passes-with-debt, because pretending otherwise
is the failure mode this repo keeps hitting: P1 compliance is derivation-based and the derivations
encode choices (R5); and the retire of `spiking_lm_serve.py` may legitimately resolve to *exempt* rather
than *retire*, which is a permitted P3 disposition and not a dodge, provided the reason and the consumer
count are recorded.

## Project Structure

### Documentation (this feature)

```
specs/005-language-organ/
|- spec.md                      # the specification (written)
|- research.md                  # Phase 0: R1-R9 (written)
|- plan.md                      # this file
|- data-model.md                # Phase 1: entities and invariants
|- contracts/
|  |- language_organ.md         # the organ's surface and its obligations
|  |- lang_ab.md                # the A/B harness contract and exit codes
|- quickstart.md                # runnable validation scenarios
|- tasks.md                     # numbered, dependency-ordered
|- baseline.txt                 # WRITTEN BY T001-T006, not by hand
|- coverage.md                  # WRITTEN BY T004: the restriction rule and its counts
|- verdict.md                   # WRITTEN BY T031: ADOPT or NULL, with the numbers
|- checklists/requirements.md   # spec quality checklist (written)
```

### Source code (repository root)

```
src/snn_data_lake/
|- cortex/
|  |- language_organ.py         # NEW - Area + LanguageOrgan + main() self-check
|  |- language_acquisition.py   # DELETED on ADOPT; untouched on NULL
|  |- spiking_lm_serve.py       # adjudicated by T034 regardless of verdict
|  |- brain.py                  # _lm_serve payload swapped; _language unchanged in shape
|  |- predictive_coding.py      # consumer count confirmed before the deletion, not assumed
|- reasoning/
|  |- world_kb_stream.py        # READ ONLY - the scene source
|  |- rose_language.py          # READ ONLY - ground() for referents
|  |- blimp_bench.py            # READ ONLY - the instrument. Do not modify the scorers.
tools/
|- lang_coverage.py             # NEW - computes and freezes the restricted pair set
|- lang_organ_ab.py             # NEW - the four arms + the head-to-head + the controls
|- blimp_baseline.py            # READ ONLY - the recorded "before" harness
tests/
|- test_language_organ.py       # NEW
|- test_lang_coverage.py        # NEW
run_selfchecks.py               # registration swapped in the correct tier
docs/coverage/Language.md       # updated to the measured outcome, up, down or unchanged
docs/DEPTH_CEILINGS.md          # Ceiling 2 "After" row, only if there is an after
```

**`reasoning/blimp_bench.py` is read-only for this feature.** Its scorers, its `UNSCORABLE` sentinel and
its dual-scoring output are the instrument's honesty, and an instrument edited by the party being
measured is not an instrument. If the organ genuinely cannot be scored without changing it, that is R1's
not-P2-eligible outcome, not a licence to edit.

## Execution Order

Spec priority order and execution order coincide here - stated explicitly because in `specs/003` they did
not, and the coincidence is worth verifying rather than assuming.

1. **Phase 1, Setup and instrument honesty (US1).** Pin the commit, confirm the gates, inventory the
   exposure lexicon, compute and FREEZE the restricted pair set, run the degenerate controls on it.
   **Hard gate: if the floors in R1 are missed, stop and record not-P2-eligible.**
2. **Phase 2, Baseline (US2).** Both scorings of the incumbent and the char-bigram control on the frozen
   set, plus `respond` latency and the live `_language` hit rate.
3. **Phase 3, The organ (US3).** Area, projection, schedule, exposure loop; the joint arm plus the
   staged, shuffled and frozen arms; held-out role and referent measurement; the derived-constant
   assertions; the density-versus-floor report.
4. **Phase 4, The verdict (US4).** One command, both arms, both scorings, controls inline, an exit code
   that distinguishes ADOPT from NULL, recorded through `coverage_score.py record --kind ab`.
5. **Phase 5, The retire (US5), ADOPT only.** Port or record-as-dropped, wire, delete, re-register,
   re-run the gates, update the ledger.
6. **Phase 6, Cross-cutting.** `spiking_lm_serve.py` adjudication (runs regardless), docs, coverage
   ledger, latency and density reporting.

The one ordering subtlety, and the reason phase 1 cannot be merged into phase 3: **the restriction rule
depends on the lexicon, and it must be fixed before either arm is scored (FR-023).** So it is computed
from the *exposure plan* - the KB's word inventory - and frozen to a file, before any organ exists. This
is the only order in which the before-scoring guarantee is achievable.

## Risks

Ranked by the damage a silent occurrence would do.

1. **The lexically-covered subset is too small, and someone widens the corpus until it passes.** This is
   the p-hacking path and it would look like diligence. Mitigation: the floors are stated in R1 *before*
   the measurement, `coverage.md` is written and committed before any scoring, and the permitted outcome
   on a miss is not-P2-eligible. Widening exposure is legitimate only if decided before the restriction
   is computed, never in response to it.
2. **A green A/B that could not have gone red.** The historical failure mode of this repo. Mitigation:
   four controls on the exact restricted set, inline with the verdict, in the same invocation
   (FR-002/SC-001); ties count as wrong (FR-014); read-only scoring (FR-015); the recorded exit code and
   not the prose sets the band ceiling (FR-016).
3. **The organ scores well from its random graph before learning anything.** *Movement is not
   activation*, in its most flattering form. Mitigation: the frozen (plasticity-disabled) arm must sit at
   the frequency baseline; if it does not, the readout and not the acquisition is carrying the result and
   the claim is void.
4. **The grounding turns out decorative.** If scenes shuffled against sentences reach the same score, the
   organ learned word statistics and the whole grounded framing is unfounded. Mitigation: the shuffled
   arm is mandatory and reported inline.
5. **A hardcoded inter-area schedule produces correct-looking output for the wrong reason.** The subtlest
   trap in the architecture: writing the grammar into the disinhibition order and then reporting that the
   organ learned grammar. Mitigation: the schedule is learned; `contracts/language_organ.md` makes it a
   contract obligation; a test asserts the schedule changes with exposure.
6. **A silently accepted iteration cap.** The silent-degradation pattern - a stage fails, swallows it,
   writes a placeholder, exits 0. Mitigation: the guard bound is derived, reaching it is a **refusal**,
   and refusals are counted and reported (R5).
7. **Latency regression discovered by the operator rather than the report.** Mitigation: FR-022, measured
   both sides, stated in the summary and not only an appendix.
8. **Deleting `language_acquisition.py` orphans something.** `predictive_coding.py` is the known case and
   is believed to have other consumers. Mitigation: **count them, do not assert them** (T033).
9. **Retiring `spiking_lm_serve.py` orphans the `snn_lm/1` export format**, whose only torch-free reader
   it is. Mitigation: T034 counts readers and producers as `N of M`; exempt-with-reason is a permitted
   outcome.
10. **HEAD moves under the work.** A second agent is committing concurrently. Mitigation: T001 pins and
    every figure names its pin; nothing outside `specs/005-language-organ/` is written until Phase 3, and
    `.specify/feature.json` is deliberately **not** touched because it is shared state a concurrent agent
    also reads.
11. **The staged arm cannot be constructed fairly** - "semantics to convergence" needs a criterion that
    does not itself smuggle in syntax. Mitigation: if it cannot, report that instead of a contrast (R4).
12. **BLiMP data absent in some environment, harness SKIPs, skip read as a pass.** Mitigation: the
    harness asserts a pair count and a skip is never counted as a passed A/B (FR-002, US1 scenario 3).

## Complexity Tracking

Empty. No constitutional violation requires justification.

One deliberate scope *reduction* worth recording under the ponytail rule that the best code is the code
never written: the organ is **one new module** (`cortex/language_organ.py`) holding both `Area` and
`LanguageOrgan`, not a package. Four areas, not a configurable topology. No registry, no plugin surface,
no abstraction over "kinds of language organ" - there is one, and if a second is ever wanted, P2 says one
of them will be deleted anyway.
