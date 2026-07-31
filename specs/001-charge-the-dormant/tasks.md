# Tasks: Charge the Dormant

**Feature**: `specs/001-charge-the-dormant` | **Plan**: [plan.md](./plan.md) | **Spec**: [spec.md](./spec.md)

**Ordering constraint from plan.md**: Phase A (instrument) â†’ B (exemptions) â†’ C (measure inert) â†’
D (wire/retire) â†’ E (re-band). **Behavioural changes never share a commit with instrument changes.**

**Tests**: not requested as TDD. Test tasks appear only where the constitution requires a runnable
check (Invariant 5: verify the verifier).

---

## Phase 1: Setup

- [ ] T001 Create the measurement harness at `scratchpad/ab_method.py`: patch a named
      `module.Class.method` to a no-op, run N ticks at seed 0 via `make_brain()`, and emit per-tick
      hashes over (actions, coherence, elapsed, bound) plus the numeric state leaves, as `N of M`
      counts never `N/M` ratios (see contracts/verdict-contract.md Â§1)
- [ ] T002 Verify T001 against a known answer: reproduce the recorded `AffectiveAppraisal.learn`
      result (behavioural hashes identical, telemetry hash differs) in the eprop-snn tier, using the
      prerequisites in `quickstart.md`

---

## Phase 2: Foundational (BLOCKING â€” Phase A of plan.md)

**No disposition may be written in this phase.** Instrument only.

- [X] T003 Record the BEFORE counts: run `python -m snn_data_lake.substrate.method_dormancy_gate`
      and capture `{dormant, uncertain, open_exempt, n_defs}` into
      `specs/001-charge-the-dormant/counts-before.txt` (Invariant 1: count before leaning)
- [X] T004 Extend `_parse_all` in `src/snn_data_lake/substrate/method_dormancy_gate.py` to track the
      enclosing `ClassDef`, keying defs by `(module, cls, name)` so two same-named defs in one module
      stop collapsing to one entry with the wrong lineno
- [X] T005 Extend the exemption lookup in the same file to try `module.Class.method` â†’
      `module.method` â†’ `method`, preserving every existing bare key (contracts Â§3)
- [X] T006 [P] Add `tests/test_method_dormancy_keys.py` driving the lookup with a synthetic def table:
      a qualified key must exempt only its own class, a bare key must still work, and two same-named
      defs in one module must produce two distinct entries with correct linenos â€” this is the
      mitigation for the Invariant-5 risk recorded in plan.md Complexity Tracking
- [X] T007 Record the AFTER counts into `specs/001-charge-the-dormant/counts-after.txt` and state in
      the commit whether any count moved and why
- [X] T008 Commit Phase A alone: instrument change + its test + both count files. No dispositions.

**Checkpoint**: gate self-check green, `uncertain` still reported not asserted, no band moved.

---

## Phase 3: User Story 2 â€” dispositions that need no behavioural change (Priority: P1) â€” Phase B

**Goal**: the two rules that are dormant AND correct to be dormant get permanent exemptions that
name exactly themselves.

**Independent test**: each exemption suppresses its own rule and nothing else; removing the
exemption makes exactly one entry reappear.

- [X] T009 [US2] Add `basal_ganglia.latent_q._TabularQ.update` to `EXEMPT_METHOD` in
      `src/snn_data_lake/substrate/method_dormancy_gate.py` with kind `permanent` and the reason that
      it is the self-check's tabular control, constructed only inside `main()` â€” a control that ran
      live would be the bug
- [X] T010 [US2] Re-key the existing `cortex.predictive_coding.learn` entry to the class-qualified
      `cortex.predictive_coding.PredictiveCodingNetwork.learn` now that the format allows it
- [X] T011 [US2] Prove no over-suppression: assert `LatentQ.update` still appears as an open finding
      after T009, and record that assertion in `tests/test_method_dormancy_keys.py`
- [X] T012 [US2] Commit Phase B alone

**Checkpoint**: two rules disposed; `LatentQ.update` demonstrably still visible.

---

## Phase 4: User Story 3 â€” report what learns nothing (Priority: P2) â€” Phase C

**Goal**: each of the three called-but-inert rules carries a recorded A/B naming its expected
consumer and why the value does not reach it.

**Independent test**: each artifact re-runs to the same verdict under `--rerun`.

- [ ] T013 [P] [US3] Record the `AffectiveAppraisal.learn` A/B in BOTH tiers via
      `tools/coverage_score.py record`, `--by` naming the tier; the artifact must show behavioural
      hashes identical in eprop-snn and divergent in stdlib
      (`src/snn_data_lake/amygdala/affective_appraisal.py:116`)
- [ ] T014 [P] [US3] Record the `TimeCellClock.learn_interval` A/B showing the plastic threshold does
      not move across the run (`src/snn_data_lake/cortex/time_cells.py:146`)
- [ ] T015 [P] [US3] Record the `IntrinsicPlasticity.adapt` A/B showing no charge on the route taken
      (`src/snn_data_lake/plasticity/intrinsic_plasticity.py:31`)
- [ ] T016 [US3] Write the disjoint-wiring finding into `docs/coverage/README.md`: for
      `AffectiveAppraisal.learn`, the charging pool is `w_aro` while the sole non-telemetry consumer
      reads `valence` (index `[0]`), so the learned signal and the consumed signal never meet
- [ ] T017 [US3] Record the P1 contradiction against `cortex/time_cells.py`'s own header claim to have
      replaced a hand-coded interval constant, citing the measurement from T014
- [ ] T018 [US3] Verify all three artifacts reproduce: `--rerun` each, exit 0
- [ ] T019 [US3] Commit Phase C (artifacts + docs only â€” no source behaviour changed yet)

**Checkpoint**: SC-004 satisfied for all three; no band moved yet.

---

## Phase 5: User Story 1 â€” every rule answers "does it do anything?" (Priority: P1) â€” Phase D

**Goal**: the three never-called-but-real rules get wire-or-retire on evidence, port check first.

**Independent test**: for each, either a recorded A/B shows the wire charges, or a port record shows
the unique capability landed elsewhere before deletion.

- [ ] T020 [P] [US1] Port check for `AffectiveEpisodicStore.consolidate`
      (`src/snn_data_lake/hippocampus/affective_memory.py:173`): identify any capability unique to it
      and whether a consumer exists today; record the finding either way (P2 port-before-retire)
- [ ] T021 [P] [US1] Port check for `ForwardModel.update`
      (`src/snn_data_lake/cerebellum/forward_model.py:111`)
- [ ] T022 [P] [US1] Port check for `DevelopmentalEnvelope.update`
      (`src/snn_data_lake/neuromodulation/developmental_envelope.py:54`)
- [ ] T023 [US1] Decide wire-or-retire per rule using research.md D5: WIRE only where a real call
      site exists, never to satisfy the gate; otherwise RETIRE with the port recorded, or EXEMPT with
      kind `wire`/`retire` so the gate keeps reporting it as a standing TODO
- [ ] T024 [US1] For each rule wired: record the A/B proving it charges. A wire that measures
      CHANGED_NOTHING is a dead consumer and MUST be reverted (P3, data-model.md validation)
- [ ] T025 [US1] For `LatentQ.update`: decide and dispose, noting the module is imported at
      `brainstem/autonomous_loop.py:30` while every method of the class is cold
- [ ] T026 [US1] Commit Phase D â€” behavioural changes only, separate from every instrument commit
- [ ] T027 [US1] Re-run `--rerun` across all ledger artifacts after the behavioural changes and fix
      or disclose any that stop reproducing (research.md R2)

**Checkpoint**: nine of nine disposed; SC-001 satisfied.

---

## Phase 6: Polish & Cross-Cutting â€” Phase E

- [ ] T028 Re-band only where a measurement from Phase C or D bears on a band, citing that
      measurement in the row's `measured_by` cell (FR-007); where a band does not move, restate why
- [ ] T029 [P] Update `docs/COVERAGE_TO_100.md` phase 5 with the outcome, including any rule left
      OPEN and the reason
- [ ] T030 [P] Update `docs/POLICIES.md` P3 candidate set, which has not been re-adjudicated since it
      was written
- [ ] T031 Run the three gates plus `score --check`; all green, and record that none was modified to
      make them so (SC-006)
- [ ] T032 Final honesty pass: confirm no exemption added in this feature can suppress a finding it
      does not name (SC-003), and no artifact was re-recorded to make a number agree

---

## Dependencies

```
Phase 1 (T001-T002)  â†’  Phase 2 (T003-T008)  â†’  Phase 3 (T009-T012)
                                              â†’  Phase 4 (T013-T019)
                                              â†’  Phase 5 (T020-T027)  â†’  Phase 6 (T028-T032)
```

- **T004 blocks T005** (same file, key format must exist before lookup uses it).
- **T009 blocks T011** (cannot prove non-suppression before the exemption exists).
- **T020-T022 block T023** (port check precedes the wire/retire decision â€” P2).
- **T024 blocks T027** (behavioural change precedes the ledger-wide re-verification).
- Phase 4 is independent of Phase 3 and may run in parallel with it.

## Parallel opportunities

- **T006** runs alongside T004/T005 (different file).
- **T013, T014, T015** are three independent A/Bs on three different modules.
- **T020, T021, T022** are three independent port checks.
- **T029, T030** are different documents.

## Implementation strategy

**MVP = Phase 2 + Phase 3.** That alone converts two open questions into recorded dispositions and
fixes the instrument defect blocking them, with zero behavioural risk.

**Increment 2 = Phase 4**: the three inert rules become measured and documented, still without
touching behaviour.

**Increment 3 = Phase 5**: the only phase that changes what the brain does, and therefore the only
one that can invalidate recorded artifacts â€” hence T027.

**Stop condition**: if a port check (T020-T022) shows a retirement would drop a capability with no
home, that rule stays OPEN with the reason recorded. An unresolved rule with a stated reason is a
valid outcome; a silenced one is not.

