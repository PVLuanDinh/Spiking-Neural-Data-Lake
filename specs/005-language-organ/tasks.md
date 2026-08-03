# Tasks: The Brain-Native Assembly-Calculus Language Organ

> ## EXECUTION STATUS, 2026-08-02 -- the feature stopped at T011, correctly
>
> **Outcome: `NOT-P2-ELIGIBLE`.** `tools/lang_coverage.py` exited **2** at commit `95c3f806`: the
> exposure lexicon covers **0 of 67** paradigms and **0 of 67000** pairs, against floors of 8 and 200.
> See [`verdict.md`](./verdict.md) and [`coverage.md`](./coverage.md).
>
> **Phase 1 and Phase 2 ran. Phases 3, 4, 5 and 6 did not**, per the hard stop this file states before
> the work: *"if `tools/lang_coverage.py` exits 2 ... the feature stops at T011. It does not proceed to
> a wider corpus and a re-run."* Of Phase 7, T057 and T059 ran (they run regardless of the verdict) and
> T058 is written to [`shared-edits.md`](./shared-edits.md) because `docs/coverage/*.md` is shared state
> a concurrently running agent also reads.
>
> An unchecked box below is **not-run-because-the-feature-stopped**, not forgotten. The reason is
> recorded once here rather than repeated on each line.

**Input**: Design documents from `/specs/005-language-organ/`

**Prerequisites**: `spec.md`, `research.md`, `plan.md`, `data-model.md`, `contracts/language_organ.md`,
`contracts/lang_ab.md`, `quickstart.md`

**Baseline commit**: **NOT pinned at authoring time.** HEAD moved during authoring
(`5599e6ab` -> `30d8e76a`) because a second agent is committing concurrently. **T001 pins it.** No task
may quote a figure attributed to a commit T001 has not recorded.

**Interpreter**: `.venv-lake\Scripts\python.exe` throughout, written below as `PY`. **Not** the `python`
on PATH.

**Tests**: INCLUDED, not optional. The constitution makes four gates binding (`run_selfchecks.py`,
`pytest` at `--cov-fail-under=70`, `ruff check`, `ruff format --check`), and Standing Invariant 1 makes
the admission controls a gate rather than a nicety. Note also F13: **no test currently imports
`cortex.language_acquisition`** - a module carrying the live reply path with zero coverage is part of why
its failure went unmeasured for as long as it did.

**Organization**: grouped by user story. Spec priority order and execution order **coincide** here -
unlike `specs/003`, where they did not. That coincidence was checked (plan Execution Order), not assumed.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an incomplete task)
- **[Story]**: US1-US5 map to the spec's user stories
- Exact file paths in every description

## Path Conventions

Single project. Source at `src/snn_data_lake/`, probes at `tools/`, tests at `tests/`, evidence at
`specs/005-language-organ/`.

## Two hard stops, stated before the work

- **After T009**: if `tools/lang_coverage.py` exits 2, the outcome is `NOT-P2-ELIGIBLE`, T010 records it,
  and **the feature stops at T011**. It does not proceed to a wider corpus and a re-run.
- **After T038**: if the verdict is `NULL`, Phase 7 (the retire) does **not** execute. T040-T041 close the
  feature COMPLETE. A null is a successful terminal state (FR-005).

---

## Phase 1: Setup

**Purpose**: pin the tree so every later figure has an owner.

- [X] T001 Pin the baseline: run `git diff --quiet HEAD -- src/snn_data_lake/cortex/language_acquisition.py src/snn_data_lake/cortex/spiking_lm_serve.py src/snn_data_lake/cortex/brain.py src/snn_data_lake/reasoning/blimp_bench.py src/snn_data_lake/reasoning/world_kb_stream.py tools/` and **ABORT if dirty**; record `git rev-parse HEAD` and the clean-diff result into `specs/005-language-organ/baseline.txt` (FR-003)
- [X] T002 [P] Confirm all four gates green at the pinned commit - `PY run_selfchecks.py`, `PY -m pytest`, `PY -m ruff check .`, `PY -m ruff format --check src tests tools` - and record pass/fail for each into `specs/005-language-organ/baseline.txt` (FR-020)
- [X] T003 [P] Record the environment into `specs/005-language-organ/baseline.txt`: interpreter path and version, presence and file count of `data/lang/blimp/` (expect 67 of 67 `.jsonl`), presence of `data/lang/wikitext/`, and whether a `learn_daemon` process is running (it writes `~/.snn_data_lake` continuously and every probe below must isolate `NORD_HOME`)

**Checkpoint**: the tree is pinned and the gates are green. Every later figure names this commit.

---

## Phase 2: User Story 1 - The metric can see what it claims to measure (Priority: P1)

**Goal**: prove the instrument can FAIL on the exact pair set the verdict will use, and freeze that set,
before any organ exists.

**Independent Test**: run the three degenerate scorers on the restricted set. `random_scorer` within 3
points of 50%, `constant_scorer` at 0%, `length_scorer`'s leakage printed. Any one passing where it must
fail disqualifies the set.

**Why first**: this is the cheapest place the feature can end. R1's coverage floors can return
`NOT-P2-ELIGIBLE` before a line of organ code is written, and Standing Invariant 1 requires counting a
gate's output before any rule leans on it.

### Tests for User Story 1

- [X] T004 [P] [US1] Create `tests/test_lang_coverage.py` with `test_rule_is_score_independent`: assert the restricted membership computed by `tools/lang_coverage.py` is a function of the lexicon and rule only - recomputing it after perturbing any arm's score leaves `pair_ids` byte-identical (contract A2, FR-023)
- [X] T005 [P] [US1] Add `test_floors_exit_two` to `tests/test_lang_coverage.py`: with an artificially tiny lexicon, assert the tool exits **2** with `NOT-P2-ELIGIBLE` rather than emitting a set (contract A4)
- [X] T006 [P] [US1] Add `test_absent_data_skips_not_passes` to `tests/test_lang_coverage.py`: with `--blimp` pointed at an empty directory, assert exit 0 with a `SKIP` line, and assert the emitted record is marked such that no downstream consumer can read it as a passed A/B (contract A8, US1 scenario 3)

### Implementation for User Story 1

- [X] T007 [US1] Create `tools/lang_coverage.py` per `specs/005-language-organ/contracts/lang_ab.md` section A: compute the exposure lexicon from `reasoning/world_kb_stream.py::load_kb()` rendered into sentences (**not** from a trained organ - A1), apply the stated coverage rule, and emit `pair_ids`. Isolate `NORD_HOME` to a per-invocation temp dir (A10). ASCII only; counts as `N of M` (A3)
- [X] T008 [US1] In `tools/lang_coverage.py`, run `blimp_bench.random_scorer`, `constant_scorer` and `length_scorer` on the **restricted** set in the same invocation, printing `random` (must be within 3 points of 50%), `constant` (must be 0%) and `length` leakage as a printed number, not a thresholded assert - its value IS the size of the leak (A6, A7, FR-002, SC-001)
- [X] T009 [US1] In `tools/lang_coverage.py`, enforce the floors from `research.md` R1 by exit code: `>= 8 of 67` paradigms and `>= 200` pairs, else exit **2** with `NOT-P2-ELIGIBLE` and the reason (A4). A control passing where it must fail also exits 2 (A7)
- [X] T010 [US1] Run `PY tools/lang_coverage.py` and write its full output to `specs/005-language-organ/coverage.md`, including the rule verbatim, `P of 67` paradigms, `N of M` pairs, the frozen `pair_ids`, the four control numbers, and the pinned commit from T001. **Commit this file before any arm is scored** (FR-023, A5)
- [X] T011 [US1] **HARD STOP CHECK.** If T010 exited 2, write the `NOT-P2-ELIGIBLE` outcome and its reason into `specs/005-language-organ/verdict.md`, update `docs/coverage/Language.md` to note that the swap was found not P2-eligible with no band change, and **stop - do not proceed to Phase 3**. Widening the corpus in response to this exit is choosing the pair set to get a verdict and is forbidden (research.md R1)
- [X] T012 [US1] Run all four gates; confirm green with no assertion loosened (FR-020)

**Checkpoint**: the instrument is proven able to go red on the exact set that will decide the feature,
and that set is frozen and committed. If not, the feature ended here, correctly.

---

## Phase 3: User Story 2 - A baseline exists (Priority: P2)

**Goal**: a commit, a command, and a number for what language competence was before anything changed.

**Independent Test**: re-running each recorded command at the pinned commit reproduces its recorded
figure.

- [ ] T013 [P] [US2] Run `PY tools/blimp_baseline.py 25` and append the full three-scorer table (acquired-LM, ROSE, char-bigram) to `specs/005-language-organ/baseline.txt`. Historical anchors for comparison, from `docs/DEPTH_CEILINGS.md:118-127`: acquired-LM **51.0%**, char-bigram **52.8%**, ROSE **no signal**. A material difference is itself a finding and must be noted
- [ ] T014 [P] [US2] Score the **incumbent** `cortex/language_acquisition.py` on the frozen `pair_ids` from `specs/005-language-organ/coverage.md` under **both** scorings (sum-logprob and per-character), read-only, and append to `specs/005-language-organ/baseline.txt`. This, not the full-set 51.0%, is the number the verdict compares against (B1, B2)
- [ ] T015 [P] [US2] Score the **char-bigram control** fitted on the identical exposure text on the same frozen `pair_ids` under both scorings; append to `specs/005-language-organ/baseline.txt`. The incumbent currently loses to this, which is why beating the incumbent alone is not enough (B6.2)
- [ ] T016 [P] [US2] Measure and record into `specs/005-language-organ/baseline.txt`: `Brain.respond` wall-clock latency over >= 20 turns at >= 3 seeds via `make_brain()`, and the live `_language` hit rate (the fraction of replies taking the acquired-LM branch rather than the narrator fallback; `docs/coverage/Language.md` records 9 of 10 after 40 ticks). Isolate `NORD_HOME` (FR-022, SC-010)
- [ ] T017 [US2] Verify reproducibility: re-run every command recorded in `specs/005-language-organ/baseline.txt` at the pinned commit and confirm each figure returns. Record any that does not, with why (SC-002)
- [ ] T018 [US2] Run all four gates; confirm green (FR-020)

**Checkpoint**: every later claim has a before it can be a delta from, attributable to a named commit.

---

## Phase 4: User Story 3 - The organ acquires language from grounded exposure, jointly (Priority: P3)

**Goal**: build the organ and show, on held-out grounded items, that it acquired meaning, role and order
- and that the controls do not.

**Independent Test**: held-out role accuracy above the most-frequent-role baseline the same harness
computes, with `shuffled` and `frozen` at that baseline.

### Tests for User Story 3

- [ ] T019 [P] [US3] Create `tests/test_language_organ.py` with `test_determinism`: two `LanguageOrgan`s built with the same seed and given identical exposure in the same order produce identical output; assert no module-level `random.seed` or `np.random.seed` exists in `src/snn_data_lake/cortex/language_organ.py` (FR-018, C0.1, SC-012)
- [ ] T020 [P] [US3] Add `test_readonly_scoring` to `tests/test_language_organ.py`: snapshot every weight in every area, run a full `acceptability()` pass over >= 50 sentences, assert byte-identical weights afterwards (FR-015, C2.1, C5.1)
- [ ] T021 [P] [US3] Add `test_cap_is_enforced` to `tests/test_language_organ.py`: assert `len(area.active) <= k` always and `== k` whenever any drive is nonzero, and that `1 <= k < n` (C1.1, C1.2)
- [ ] T022 [P] [US3] Add `test_hebbian_is_local` to `tests/test_language_organ.py`: assert an update touches no synapse with an inactive endpoint, and that no normalization reads a global statistic over all synapses (C2.3, C2.4, FR-008)
- [ ] T023 [P] [US3] Add `test_derived_constants` to `tests/test_language_organ.py`: assert `n`, `k`, `p`, `beta` each satisfy their stated derivation at construction, and that a construction which cannot satisfy a derivation **refuses** rather than clamping (FR-007, C1.5, C0.3)
- [ ] T024 [P] [US3] Add `test_convergence_not_count` to `tests/test_language_organ.py`: assert `project_until_stable` terminates on an unchanged assembly; assert reaching the derived guard bound returns `converged=False`, increments `refusals`, and surfaces in the organ's reported state - never silently treated as convergence (C3.1-C3.4)
- [ ] T025 [P] [US3] Add `test_schedule_changes_with_exposure` to `tests/test_language_organ.py`: assert the inter-area disinhibition schedule after exposure differs from the schedule at construction. A hardcoded schedule is a hardcoded grammar and produces correct-looking output for the wrong reason (C4.2, plan Risk 5)
- [ ] T026 [P] [US3] Add `test_lexical_gap_returns_none` to `tests/test_language_organ.py`: assert `acceptability()` returns `None` - never a low float - for an out-of-lexicon sentence, mirroring `rose_language.ground()`'s honest gap. A low float would be divided by length in `per_char` and resurrect the length bug `UNSCORABLE`'s pass-through exists to prevent (C5.2, FR-014)
- [ ] T027 [P] [US3] Add `test_save_load_roundtrip` to `tests/test_language_organ.py`: assert a `save`/`load` round trip yields an organ producing identical output, and that a restored organ whose derived quantities disagree with its persisted ones **refuses** and says which (C7.2, C7.3)

### Implementation for User Story 3

- [ ] T028 [US3] Create `src/snn_data_lake/cortex/language_organ.py` with `Area` per `specs/005-language-organ/contracts/language_organ.md` sections 1-2: cap-k local inhibition, non-negative recurrent and inbound weights, per-instance rng, deterministic tie-breaking by unit index, and `project_into(target, *, plastic)` applying Hebbian strengthening only to synapses with both endpoints active
- [ ] T029 [US3] In `src/snn_data_lake/cortex/language_organ.py`, implement the R5 derivations with the derivation named in a comment at each definition site: `k` from acquired lexicon size, `n` from `k` plus the birthday-collision separability argument, `p` from `n` and `k`, `beta` from observed convergence. Assert each; refuse rather than clamp (FR-007, C0.3, C1.5)
- [ ] T030 [US3] In `src/snn_data_lake/cortex/language_organ.py`, implement `LanguageOrgan` with the four areas `PHON`, `LEX`, `SCENE`, `SEQ`, `project_until_stable` terminating on convergence with a derived guard bound and a reported `refusals` counter, and a **learned** inter-area disinhibition schedule (contract sections 3, 4; data-model entity `LanguageOrgan`)
- [ ] T031 [US3] In `src/snn_data_lake/cortex/language_organ.py`, implement `expose(words, scene)` as a **single joint pass** updating lexical and syntactic structure together - no stage boundary, no freeze-then-thaw (FR-006, C4.1). Re-derive `k` as the lexicon grows without altering what was already learned (C4.4). Reuse an existing local plasticity rule rather than authoring a fourth; name which one and why in the module docstring (research.md R3, P2)
- [ ] T032 [US3] In `src/snn_data_lake/cortex/language_organ.py`, implement `acceptability(sentence)` (read-only, monotone, `None` on lexical gap), `role_of(word, sentence)`, and `generate(scene)` per contract sections 5-6; and `save`/`load` via `substrate/home.py`'s `save_state`/`load_state` atomic JSON - **not** the one lake (C7.1)
- [ ] T033 [US3] In `src/snn_data_lake/cortex/language_organ.py`, implement `main()` as a self-check that asserts something able to fail - at minimum determinism, cap enforcement, and that exposure moves the schedule. A self-check asserting only that construction did not raise is the empty-pipe pattern `spiking_lm_serve.py` already demonstrates (C0.5)
- [ ] T034 [US3] Register `cortex.language_organ` in `run_selfchecks.py` in the **numpy-gated CORE** tier (alongside the entry at `:369`), matching the tier of the module it will replace. Do not raise the tier (C9.1)
- [ ] T035 [US3] Build the grounded exposure set in `tools/lang_organ_ab.py`: render `reasoning/world_kb_stream.py::load_kb()` triples into (words, scene) items, resolve referents with `reasoning/rose_language.py::ground()` (`None` for an honest gap), and split **positionally and deterministically** into TRAIN and TEST. State in the output that a positional split over ordered data is a temporal holdout with different statistics from a random one (data-model entity `Grounded exposure item`)
- [ ] T036 [US3] In `tools/lang_organ_ab.py`, implement the four arms - `joint`, `staged`, `shuffled`, `frozen` - on identical items, order, split and seed, per `contracts/lang_ab.md` section B. Do not drop ungroundable items in any arm; dropping items that do not suit the mechanism is selection on the outcome (C4.3)
- [ ] T037 [US3] In `tools/lang_organ_ab.py`, report held-out role accuracy as `N of M` correct against the most-frequent-role baseline computed by the same harness on the same exposure - **both numbers printed**, so a win against nothing is visible (C6.1, SC-004). Assert at least one generated sentence is not a verbatim exposure item, and if none is, withdraw the novel-generation claim in the output (C6.2)
- [ ] T038 [US3] Run all four gates; confirm green with the new module's self-check included and no assertion loosened (FR-020)

**Checkpoint**: the organ exists, is deterministic, learns jointly, and its controls have been run. If
`shuffled` matches `joint` the grounding is decorative; if `frozen` scores well the readout and not the
learning is carrying it. Either is a real finding and is reported, not adjusted away.

---

## Phase 5: User Story 4 - The A/B decides, and may say no (Priority: P4)

**Goal**: one command, both arms, both scorings, controls inline, an exit code that distinguishes ADOPT
from NULL.

**Independent Test**: the recorded artifact `--rerun`s and reproduces its verdict.

- [ ] T039 [US4] In `tools/lang_organ_ab.py`, load the frozen `pair_ids` from `specs/005-language-organ/coverage.md`, assert the loaded list matches it, and refuse otherwise. Score every arm on **exactly** that list - not the same rule applied separately, the same list (B1)
- [ ] T040 [US4] In `tools/lang_organ_ab.py`, report every arm under **both** scorings side by side, and print all four admission controls (`random`, `constant`, `length`, and the unexposed-organ control) inline in the same invocation (B2, B3, FR-002, FR-013)
- [ ] T041 [US4] In `tools/lang_organ_ab.py`, snapshot organ weights before and after the scoring pass and hard-fail on any change; route unscorable sentences to `None` so `blimp_bench` sentinels them and ties count as wrong (B4, B5, FR-014, FR-015)
- [ ] T042 [US4] In `tools/lang_organ_ab.py`, implement the ADOPT conjunction exactly as `contracts/lang_ab.md` B6: `joint` beats **both** the incumbent and the bigram control under **both** scorings, AND all four controls fail as specified, AND `frozen` and `shuffled` sit at the frequency baseline. Any one unmet is `NULL`. `NULL` exits **0** - a null is not an error (B7, FR-005). Provide **no** flag that can change the verdict (B8)
- [ ] T043 [US4] In `tools/lang_organ_ab.py`, print the joint-versus-staged contrast with direction and magnitude whichever way it falls; if `staged` matches or beats `joint`, state in the summary that the joint claim is **not supported here** (B9, FR-006, SC-005)
- [ ] T044 [US4] In `tools/lang_organ_ab.py`, print the density block: `H`, derived `M`, `rho_floor = H_b^{-1}(log2 M / H)`, `rho_measured`, and the plain statement that compute is dense regardless of activity sparsity and that the low floor follows from the organ's small `M`. Report `rho_measured < rho_floor` as an **anomaly**, not a success (B11, FR-021, SC-009)
- [ ] T045 [US4] In `tools/lang_organ_ab.py`, print `Brain.respond` latency for both arms in the same units with the delta, in the summary and not only an appendix; every `Brain` via `make_brain()`; isolate `NORD_HOME`; `--seeds N` reports mean and spread so a one-draw result is not mistaken for a stable one (B12, B13, B16, FR-022)
- [ ] T046 [US4] Record the verdict: `PY tools/coverage_score.py record --kind ab --name lang_organ_ab -- tools/lang_organ_ab.py`, then `PY tools/coverage_score.py --rerun lang_organ_ab` and confirm it reproduces. A `python -c` one-liner records fine and dies on rerun; the recorded command must be a plain script invocation (FR-016, B15, SC-006)
- [ ] T047 [US4] Write the outcome, all arm scores under both scorings, all four control numbers, the pair and paradigm counts as `N of M` and `P of 67`, and the pinned commit into `specs/005-language-organ/verdict.md`
- [ ] T048 [US4] Run all four gates; confirm green (FR-020)

**Checkpoint**: there is a verdict, produced by an instrument that was shown able to go red, recorded
where the exit code and not the prose sets the band ceiling.

### If the verdict is NULL - the feature closes here, successfully

- [ ] T049 [US4] **NULL path only.** Confirm nothing was changed: `git status --porcelain src/snn_data_lake/cortex/` shows `language_acquisition.py` present and unmodified, and the organ is **not** wired into `Brain._lm_serve`. Update `docs/coverage/Language.md` to record the null with no band change - a row is never deleted to raise a score (FR-005, FR-025, SC-008)
- [ ] T050 [US4] **NULL path only.** Report the feature **COMPLETE**, not blocked. Then skip Phase 6 entirely and execute only T057-T060 of Phase 7

---

## Phase 6: User Story 5 - Exactly one language generator survives (Priority: P5) - ADOPT ONLY

**Goal**: the P2 obligation. One survivor.

**Independent Test**: a tree-wide search finds no reachable second reply generator, and the four gates
are green with nothing loosened and no self-check count silently reduced.

- [ ] T051 [US5] Adjudicate `expose()`-from-live-conversation before deleting anything: determine whether the organ can learn from a conversational turn that grounds to nothing, and either port the capability into `src/snn_data_lake/cortex/language_organ.py` or **record it as a deliberately dropped capability with the reason** in `specs/005-language-organ/verdict.md`. Silently losing it is what FR-011 forbids
- [ ] T052 [US5] Port persistence: replace the `language_acq.npz` writes at `src/snn_data_lake/cortex/brain.py:2607` and `:3656` and the reload in `_lm_serve` with the organ's `save`/`load`. Replaced, not added alongside (C8.5, FR-011). Confirm restart survival - the daemon relaunches ~20 times in a working session (C7.4)
- [ ] T053 [US5] Count, do not assert, the other consumers of `src/snn_data_lake/cortex/predictive_coding.py` and record the count as `N of M` in `specs/005-language-organ/verdict.md`, so deleting the incumbent is known not to orphan it (plan Risk 8)
- [ ] T054 [US5] Swap the payload of `Brain._lm_serve` in `src/snn_data_lake/cortex/brain.py:3499` to build `LanguageOrgan`, returning `None` on any failure as today. Leave `Brain._language` (`:3541`) and the Gf-fluid branch of `respond()` (`:3077`) structurally unchanged, including the coherence guard and the narrator fallback (C8.1-C8.3)
- [ ] T055 [US5] **DELETE** `src/snn_data_lake/cortex/language_acquisition.py` and remove its registration at `run_selfchecks.py:369` in the same change. Not flag-gated, not behind a mode switch. **State the module-count change** rather than letting it shrink silently (FR-010, C9.2, SI-2). Then `grep -rn "language_acquisition" src/ tools/ tests/ run_selfchecks.py` and resolve every remaining reference
- [ ] T056 [US5] Verify the organ **changes the reply** on the live Gf-fluid branch over >= 20 turns at >= 3 seeds via `make_brain()`, and record the hit rate as `N of M` against T016's baseline. A generator that changes no reply is dormant, not progress (FR-024, C8.4, P3)

**Checkpoint**: one generator. Counting survivors is the test - if two implementations of the
reply-generation job are still reachable, it was not a merge.

---

## Phase 7: Polish & Cross-Cutting Concerns

T057 and T058 run **regardless of the verdict**. T059-T060 close the feature either way.

- [X] T057 Adjudicate `src/snn_data_lake/cortex/spiking_lm_serve.py` to one of P3's three dispositions - wired, retired, or **exempt with a stated reason**. Count its readers and producers as `N of M` (it is the only torch-free `snn_lm/1` reader, so retiring it orphans `src/snn_data_lake/training/spiking_lm.py`'s export path) and record the decision and the count in `specs/005-language-organ/verdict.md`. Default is RETIRE; exempt-with-reason is permitted; leaving it undecided is not (FR-012, SC-011)
- [ ] T058 [P] Update `docs/coverage/Language.md` to the measured outcome - up, down, or unchanged - citing `file:line` for every changed row and naming the recorded artifact. A downgrade is a result and is stated in the same sentence; a row is never deleted to raise a score (FR-025, SC-013)
- [X] T059 [P] Add the "After" row to `docs/DEPTH_CEILINGS.md` Ceiling 2 **only if there is an after** - i.e. only on ADOPT. On NULL, add a line recording that the organ was measured and did not beat the incumbent, with the artifact name
- [X] T060 Run `PY tools/citation_guard.py` and all four gates; confirm every `file:line` cited in `specs/005-language-organ/` and in the updated docs resolves, no tracked file cites an untracked `.py` as a runnable reproduction path, and nothing was loosened (FR-017, FR-020)

---

## Phase 4b/5b: SNNLP (arXiv 2401.17911) - added 2026-08-02 by owner scope change

**Status: NONE OF T061-T070 RAN.** Every one of them is downstream of the T011 hard stop - the encoder
A/B needs a `PHON` area and the sentiment arm needs an organ to score, and the feature stopped at
`NOT-P2-ELIGIBLE` before either existed. They are specified here with IDs and dependencies so the work
is defined rather than deferred into memory.

**These tasks are numbered separately but they are NOT a separate phase in execution.** T061-T066
belong inside Phase 4 (they build parts of the organ) and T067-T070 inside Phase 5 (they print
alongside the verdict). The dependency block below places each one.

**The constraint that governs all of them** (FR-027, FR-029, contract C0): **BLiMP remains the sole
arbiter and sentiment cannot rescue a `NULL`.** No task here may touch the ADOPT conjunction, the frozen
pair set, the four admission controls or the char-bigram bar. A task in this section that changes a
verdict criterion is a defect, not a feature.

### The PHON encoder A/B (inside Phase 4)

- [ ] T061 [P] Add `test_phon_encoder_is_pluggable` to `tests/test_language_organ.py`: assert `PHON`'s text-to-spike encoder is a named, swappable component and that the four arms of contract C1 differ **only** in it. A hardcoded encoder makes the A/B unrunnable (C0.6)
- [ ] T062 [P] Add `test_poisson_encoder_is_deterministic` to `tests/test_language_organ.py`: assert the Poisson rate-coding arm draws from the organ's instance rng, never a module-level or fresh `Random`; two same-seed organs produce identical spike trains (C0.8, FR-018)
- [ ] T063 In `src/snn_data_lake/cortex/language_organ.py`, implement the pluggable `PHON` encoder surface plus the four arms named in `contracts/lang_ab.md` C1: `snnlp`, `poisson`, `graphemefold` (reusing `cortex/language_acquisition.py::_encode_ctx`'s position-tagged grapheme fold, not reimplementing it - P2), and `rbfpop` (reusing `cortex/population_encoding.py::PopulationEncoder`). The `rbfpop` arm must **state which scalar embedding of the word form it used**, because that choice is then doing part of the work (C1 table)
- [ ] T064 In `tools/lang_organ_ab.py`, run the four encoder arms on identical exposure, order, split and seed, and print all four with the winner named. **An existing repo encoding winning is a legitimate and expected outcome**: it is then adopted and SNNLP is recorded as measured-and-not-adopted (FR-028, C1.1, SC-014)
- [ ] T065 In `src/snn_data_lake/cortex/language_organ.py`, name the **winning** encoder and its measured number in the module docstring (C0.7). Do not quote arXiv 2401.17911's ~13% figure as this organ's result - it is CITED, UNVERIFIED (research.md R10) and C1.2 forbids any criterion referencing it
- [ ] T066 Confirm no energy or sparsity figure from arXiv 2401.17911 appears as an acceptance criterion or as a claimed property of this organ anywhere in `src/`, `tools/` or `specs/005-language-organ/` (C1.3, FR-021). This repo has no hardware to measure them on

### The sentiment capability metric (inside Phase 5)

- [ ] T067 Create the SST-2 loader beside the organ - `src/snn_data_lake/cortex/sst2_loader.py`, **not** in `reasoning/` - following `reasoning/blimp_bench.py`'s precedent exactly: the one-time download command in the module docstring, the loader shipped, the data NOT committed (`data/` is gitignored, `.gitignore:43`). Absent data prints `SKIP` and exits 0, recorded so it can never read downstream as a passed capability (C2.1, C2.2, C2.5, SC-016)
- [ ] T068 **STOP-AND-REPORT GATE.** If the SST-2 download is unavailable or its licence is unclear, **stop and report**. Do not substitute a self-authored sentiment corpus - a corpus authored alongside the mechanism it scores is the weakest evidence available and `research.md` refuses it twice already (R1, R6). This is a hard stop of the same kind as T011 (C2.6, FR-030)
- [ ] T069 In `tools/lang_organ_ab.py`, print the sentiment block **below the verdict line**, with the **bag-of-words control inline**. If the organ does not beat the bag of words, the block must say **in that same block** that the sentiment number is not evidence of anything - not omit the control, not drop the number (C2.3, FR-030)
- [ ] T070 Add `test_sentiment_cannot_rescue_a_null` to `tests/test_language_organ.py`: force a `NULL` BLiMP block alongside a maximal sentiment block and assert the verdict is still `NULL`. Prose promising this is not enough; B8 already forbids a parameter that changes the verdict and this is the same guarantee for a second input (C0, FR-029, SC-015)

---

## Dependencies

```
T001 (pin)
 |-> T002, T003            [P] gates and environment
 |
 +-> Phase 2 (US1)
      T004, T005, T006     [P] tests, all in tests/test_lang_coverage.py -- write together, they are one file
      T007 -> T008 -> T009 -> T010 -> T011 (HARD STOP) -> T012
      |
      +-> Phase 3 (US2)  [requires the frozen set from T010]
           T013, T014, T015, T016  [P] -> T017 -> T018
           |
           +-> Phase 4 (US3)
                T019..T027  [P] tests (all tests/test_language_organ.py)
                T028 -> T029 -> T030 -> T031 -> T032 -> T033 -> T034
                T035 -> T036 -> T037
                T038 (gates)
                |
                +-> Phase 5 (US4)
                     T039 -> T040 -> T041 -> T042 -> T043, T044, T045 -> T046 -> T047 -> T048
                     |
                     +-- NULL --> T049 -> T050 -> (skip Phase 6) -> T057..T060
                     |
                     +-- ADOPT -> Phase 6 (US5)
                                  T051, T053  [P]
                                  T052 -> T054 -> T055 -> T056
                                  |
                                  +-> Phase 7: T057, T058, T059  [P] -> T060
```

**Blocking relationships worth stating explicitly**:

- **T010 blocks everything downstream.** No arm may be scored before the pair set is frozen and
  committed (FR-023). This is not a convenience ordering; it is what makes the comparison honest.
- **T011 can end the feature.** It is a stop, not a checkpoint.
- **T014 and T015 depend on T010**, not merely on Phase 1 - they score the *frozen* set, and the
  full-set figures in T013 are context, not the comparison.
- **T042 depends on T015** - the ADOPT conjunction needs the bigram control's number on the frozen set.
- **T055 depends on T051, T052 and T053.** Delete last, after porting or recording-as-dropped, and after
  counting what else would be orphaned. Deleting before proving is what P2's honesty guard forbids.
- **T057 depends on nothing in Phases 4-6.** `spiking_lm_serve.py` is dormant regardless of the verdict.

**Where the SNNLP tasks sit in the graph** (added 2026-08-02):

```
Phase 4 (US3)
  T019..T027  [P] tests   ... plus T061, T062  [P] (same file, tests/test_language_organ.py)
  T028 -> T029 -> T030 -> T063 -> T031 -> T032 -> T033 -> T034
                          ^^^^ the pluggable PHON encoder must exist before expose() uses it
  T035 -> T036 -> T037 -> T064 -> T065 -> T066
                          ^^^^ the encoder A/B needs the four arms (T063) and the harness (T036)
Phase 5 (US4)
  T039 -> T040 -> T041 -> T042 (the VERDICT -- computed from the BLiMP block ALONE)
                            |
                            +-> T067 -> T068 (STOP GATE) -> T069 -> T070
                                sentiment prints BELOW the verdict line and cannot reach it
```

- **T063 blocks T031.** `expose()` drives `PHON`, so the encoder surface must exist before the joint
  update is written against it. Retrofitting pluggability after `expose()` is written is how a
  hardcoded encoder survives.
- **T064 depends on T036**, not merely T063: the four encoder arms are scored through the same arm
  harness the joint/staged/shuffled/frozen arms use, on identical exposure, order, split and seed.
- **T068 can end the sentiment work** the way T011 can end the feature. It is a stop, not a checkpoint.
- **T070 depends on T042 and T069 both existing** - it asserts the wall between them holds. It is the
  only mechanized form of FR-029; every other statement of it is prose.
- **Nothing in T061-T070 blocks T042.** That is the point of the ordering: the verdict is reachable
  with the entire SNNLP addition removed, which is what makes it non-deciding in fact rather than by
  assertion.

## Parallel Opportunities

- **T002, T003** - independent reads.
- **T004, T005, T006** - same file (`tests/test_lang_coverage.py`); write them in one pass rather than
  three, but they have no ordering between them.
- **T013, T014, T015, T016** - four independent measurements appending to `baseline.txt`; serialize the
  appends or write to separate sections and concatenate.
- **T019 through T027** - nine tests, all in `tests/test_language_organ.py`. One pass, no ordering.
- **T051 and T053** - independent investigations.
- **T058 and T059** - different files.

Everything in `src/snn_data_lake/cortex/language_organ.py` (T028-T033) is **serial**: one file, and each
task depends on the structure the previous one established.

## Implementation Strategy

**The MVP is Phase 2 (US1) alone**, and this is not a formality. Phase 2 answers the question that
decides whether the feature is buildable at all: can the existing benchmark express an assembly-calculus
organ on enough of BLiMP to carry a verdict? If it cannot, T011 records `NOT-P2-ELIGIBLE`, the feature
stops having cost one probe and no source change, and that is the correct outcome rather than a failure.
P2's guard is explicit that the answer in that case is *build the benchmark first*, not *pick the more
fashionable mechanism* - and building a benchmark alongside the mechanism it scores is the weakest
evidence available, which is why it is out of scope here rather than a fallback.

**Incremental delivery**: Phase 2 (is this measurable?) -> Phase 3 (what was it before?) -> Phase 4 (does
the organ acquire anything, and do the controls stay flat?) -> Phase 5 (does it win?) -> Phase 6 (retire
the loser). Each phase leaves the tree green and each produces evidence that survives the next phase
failing.

**Three outcomes are all successes**: `NOT-P2-ELIGIBLE` at T011, `NULL` at T042, and `ADOPT` through
Phase 6. Only two things count as failure - a verdict produced by an instrument that could not have gone
red, and a deletion made before the A/B that justified it.
