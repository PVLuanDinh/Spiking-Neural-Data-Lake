---
description: Task list for 015-resolve-weight-transport
---

# Tasks: Resolve the e-prop Weight Transport at `mlp_q`

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/criterion.md](contracts/criterion.md),
[quickstart.md](quickstart.md)

**Branch**: `015-resolve-weight-transport` (off `main` = `462da205`) | **Gate env**: `.venv-lake`

## Format: `[ID] [P?] [Story] Description`

- **[P]** = parallelizable (different files, no dependency on an incomplete task)
- **[US1]..[US4]** = user story from spec.md; Setup / Foundational / Polish carry no label

---

## THE GATE BATCH — every commit, including docs-only

```bash
.venv-lake/Scripts/python.exe run_selfchecks.py && .venv-lake/Scripts/python.exe -m pytest -q && .venv-lake/Scripts/python.exe -m ruff check . && .venv-lake/Scripts/python.exe -m ruff format --check . && .venv-lake/Scripts/python.exe tools/p4c_ratchet.py --assert && .venv-lake/Scripts/python.exe tools/p6_gate.py --assert && .venv-lake/Scripts/python.exe tools/citation_guard.py --assert && .venv-lake/Scripts/python.exe tools/red_reachability.py --assert && .venv-lake/Scripts/python.exe tools/dependency_ledger.py --assert && .venv-lake/Scripts/python.exe -m snn_data_lake.substrate.import_manifest
```

**There is NO docs-only carve-out in POLICIES.md.** Lane 017 assumed one; it was a finding.
**COUNT the exit codes — a partial log reads as a pass.** That happened in lane 017.

## SAFETY SCOPE — binding on every task

This lane carries **NO** safety-envelope approval. The five `brainstem/` files are not touched by any
task. `src/snn_data_lake/basal_ganglia/mlp_q.py` is the **only** source file any task may change. A
task requiring otherwise is an error in this list — stop and report.

## MEASUREMENT DISCIPLINE — binding on every measurement task (FR-016)

Every measurement command carries a **positive control in the same command**. No negative result
("flat", "none", "unchanged", "clean") may be reported from a command not shown able to produce a
positive. **Assert the denominator.**

---

## Phase 1: Setup — baselines before anything changes

- [X] T001 Create `specs/015-resolve-weight-transport/evidence/` for recorded reds, tables and probe output
- [X] T002 [P] Record the pre-change gate batch (all ten commands, all exit codes) to `specs/015-resolve-weight-transport/evidence/baseline-gate.txt`, asserting **9 exit lines are present** before reading a verdict
- [X] T003 [P] Record the module's own shipped numbers by running `python -m snn_data_lake.basal_ganglia.mlp_q` to `specs/015-resolve-weight-transport/evidence/baseline-mlpq.txt` — control: assert the output is non-empty before reading it
- [X] T004 [P] Record baseline `tools/p4c_ratchet.py` and `tools/p6_gate.py` counts to `specs/015-resolve-weight-transport/evidence/baseline-ratchets.txt`, so a later "unchanged" claim is measured
- [X] T005 [P] Record `git diff --name-only main..HEAD -- src/snn_data_lake/brainstem/` (expected empty) to `specs/015-resolve-weight-transport/evidence/baseline-envelope.txt`

**Checkpoint**: baselines exist for gate, module output, ratchets, envelope.

---

## Phase 2: User Story 1 — the criterion can tell two rules apart (P1) — COMMIT C1

**Goal**: put the degeneracy on the record BEFORE any repair, so the repair cannot be
retro-justified.

**Independent test**: a reader can state from the docstring alone what the old cosine compared and
why no alignment rule could pass it.

- [X] T006 [US1] Record the degeneracy in `src/snn_data_lake/basal_ganglia/mlp_q.py`'s module docstring: the cosine reduces to `cos(W2[a], B[a])` because `err` is a shared scalar that cancels and the relu mask zeroes the same coordinates, so for a FIXED RANDOM `B` it is near zero **by construction** — it measured weight SYMMETRY, which is what feedback alignment denies by design, and never tested whether the rule learns
- [X] T007 [US1] Record in the same docstring the contrast that proves it: the same-named criterion measured **0.971** at the jepa site, far ABOVE the bar, because there both sides were computed through one forward pass. One criterion name, two quantities, opposite verdicts
- [X] T008 [US1] State in the same docstring that the prior FA refutation therefore **does not stand on this criterion** — without asserting the opposite conclusion, which is what the rest of this lane measures
- [X] T009 [US1] Re-derive every `file:line` cite touched by this commit AFTER the last edit; prefer SYMBOL or MARKER over line numbers, then run the gate batch and assert 9 exit codes

**Checkpoint**: degeneracy on the record, no behaviour changed. **COMMIT C1.**

---

## Phase 3: User Story 1 (continued) — the repaired criterion qualifies (P1) — COMMIT C2

**Goal**: the criterion is shown to separate good from bad BEFORE it judges any candidate.

**Independent test**: feed it a known-good estimate and a known-bad one; confirm it separates them.

**This phase must complete before Phase 4. A criterion that has not qualified cannot judge.**

- [X] T010 [US1] Implement the repaired criterion per [contracts/criterion.md](contracts/criterion.md) — `cos(dh_rule, dh_transport)` on the same input and same pre-update weights — in `tests/probe_015/criterion.py`, stdlib only
- [X] T011 [US1] Implement the permutation null in `tests/probe_015/criterion.py`: `dh_transport` with coordinates shuffled by a **per-instance** `random.Random(seed)` — identical magnitude, identical element multiset, different direction. Never module-level seeding
- [X] T012 [US1] Qualification check 1 (identity): transport against itself scores **1.0**. If not, the harness is wrong and nothing downstream means anything — stop and report. Record to `evidence/criterion-identity.txt`
- [X] T013 [US1] Qualification check 2 (control): the permutation null scores LOW. **If it does not, the criterion measures magnitude and is DELETED, not weakened** (FR-002) — record that outcome and stop. Record to `evidence/criterion-permutation.txt`
- [X] T014 [US1] Qualification check 3: construct the input that makes the criterion FAIL and **record the actual output** to `evidence/criterion-red.txt`. A criterion with no reachable red is deleted (P8 guard 1)
- [X] T015 [US1] Write the disclosed degeneracy into `tests/probe_015/criterion.py`'s own docstring: the new quantity remains a **monotone function of `cos(W2[a], B[a])`** under this module's arithmetic; the repair is NOT a new formula but (a) per-epoch readout and (b) a feedback that can MOVE. A reader must be able to predict that fixed-random still scores near zero and know that is correct, not a residual bug
- [X] T016 [US1] Run the gate batch, assert 9 exit codes, and confirm `p4c_ratchet` is unchanged against T004. **The probe lives in `tests/probe_015/`, NOT under `specs/`**: a `.py` under `specs/` opens a NEW ZONE and p4c refuses a rise by design (measured 2026-08-14: `specs: 1 > baseline 0`). `tests/**` is already excluded as "the instrument that measures a build, not a build", which is exactly what this probe is

**Checkpoint**: criterion qualified on all three checks, degeneracy disclosed. **COMMIT C2.**

---

## Phase 4: User Story 2 — the discriminating measurement (P1) — COMMIT C3

**Goal**: produce the trajectory table the decision consumes. Changes no shipped behaviour.

**Independent test**: the table distinguishes the two feedback variants, or explicitly reports that
it could not — a null is a complete result.

- [X] T017 [US2] Implement the Kolen-Pollack converging feedback in `tests/probe_015/sweep.py`: `B[a][j] += lr * err * h[j]`, the SAME increment the read-out step already applies, so `B` tracks `W2` without ever reading it. `err` and `h[j]` are already local to that synapse
- [X] T018 [US2] Implement the decay decision as a measured comparison in `tests/probe_015/sweep.py` — decay derived from `lr` vs decay-free from a small init (P1: no new bare constant). **Measure which, do not assume**
- [X] T019 [US2] Build the sweep matrix in `tests/probe_015/sweep.py`: `{transport, fixed_random, converging, permutation_null}` × `bseed 1..8` × `epoch 1..30`, using the module's own fixtures (400 train / 300 held-out, hidden=32, lr=0.1, 30 passes, model seed 0, `B` from `random.Random(bseed)` as `uniform(-1,1)/sqrt(hidden)`)
- [X] T020 [US2] Record cosine AND held-out accuracy on BOTH fixtures for every row, so *aligns-but-stops-learning* and *learns-but-does-not-align* are both visible (FR-005)
- [X] T021 [US2] Assert the row count is exactly **960** (4 × 8 × 30) and that the transport variant reads **1.0 at every epoch** — if either fails the probe is broken and no verdict may be read from it
- [X] T022 [US2] Write the trajectory table to `specs/015-resolve-weight-transport/evidence/trajectory.md`, reporting 8 draws as a **RANGE** not a mean (FR-006) — the module already records this quantity "is a property of the DRAW"
- [X] T023 [US2] Confirm the probe is in `tests/probe_015/`, NOT under `specs/` (a `.py` there opens a p4c zone the ratchet refuses), and is NOT registered in `__init__.py` or `run_selfchecks.py`; it must never be promoted to `src/`
- [X] T024 [US2] Run the gate batch, assert 9 exit codes

**Checkpoint**: 960-row trajectory recorded, probe unpromoted. **COMMIT C3.**

---

## Phase 5: THE DECISION POINT — compare against the prediction AS WRITTEN

**No code in this phase.** It reads [research.md](research.md) D-3 and decides which of Phase 6a or
6b executes. The bar and epoch count come from D-3 and **are not renegotiated here.**

- [X] T025 [US3] Compare the trajectory against research.md D-3 **as written**, not a revised version, and record the comparison to `specs/015-resolve-weight-transport/evidence/decision.md`
- [X] T026 [US3] Apply the bar: converging `B` reaches `|cos| > 0.5` on **≥ 6 of 8** draws by pass 30 **AND** holds the shipped accuracy gates (> 0.85 held-out non-linear, > 0.30 over the linear control) on every draw
- [X] T027 [US3] Record per-item attribution — which draws cleared, which did not, and the accuracy beside each (P8 guard 3). "It worked" is not attribution

**Checkpoint**: **GO → Phase 6a. NO-GO → Phase 6b.** Exactly one executes.

---

## Phase 6a: User Story 3 — ADOPT (P1) — COMMIT C4-adopt *(only on GO)*

**Goal**: one credit-assignment path, and it is the converging one.

- [X] T028 [US3] Replace the feedback vector inside the existing `dh` step in `src/snn_data_lake/basal_ganglia/mlp_q.py` with the converging `B`, and **DELETE the transport line**
- [X] T029 [US3] Cite the mechanism as **Kolen-Pollack** in `src/snn_data_lake/basal_ganglia/mlp_q.py`, NOT as SFDFA. SFDFA is defined over spikes and this module has none; calling it SFDFA would be the citation defect this lane exists to remove (research.md D-1)
- [X] T030 [US3] Count survivors in `src/snn_data_lake/basal_ganglia/mlp_q.py`: grep for the retired transport form and assert **zero** hits, with a positive control proving the grep can match (FR-008). A flag selecting between transport and `B` fails this
- [X] T031 [US3] Re-run the module's shipped gates on every draw and record to `evidence/post-adoption-gates.txt` — > 0.85 held-out non-linear, > 0.30 over the linear control
- [X] T032 [US3] Name and quantify any capability lost (FR-009) in `src/snn_data_lake/basal_ganglia/mlp_q.py`'s docstring — measured, not asserted
- [X] T033 [US3] Settle the decay term with the measurement that decided it (P1), recorded in the docstring beside the rule
- [X] T034 [US3] Land the repaired criterion permanently in the module's `main()` for the shipped rule only — one rule, one criterion, re-derived every gate run (research.md D-4)
- [X] T035 [US3] Re-derive every cite AFTER the last edit, then run the gate batch and assert 9 exit codes

**Checkpoint**: one mechanism, transport deleted, gates green. **COMMIT C4-adopt.**

---

## Phase 6b: User Story 3 — NULL (P1) — **DID NOT EXECUTE**

**The decision was GO** (evidence/decision.md). Phase 6a ran instead. Tasks T036-T039 are
not applicable and are left unchecked deliberately -- an unexecuted branch is not an unfinished one.

**The null is a deliverable, not a failure** (FR-007, pre-committed in research.md D-3 before any
number existed). **The lane stops after Phase 7.**

- [ ] T036 [US3] Record the null with its numbers in `specs/015-resolve-weight-transport/evidence/null.md` — which draws failed, by how much, and on which of cosine or accuracy
- [ ] T037 [US3] Restate the site's disposition in `src/snn_data_lake/basal_ganglia/mlp_q.py`'s docstring: the transport stands, the resolution was attempted and refuted, and the two named resolutions remain
- [ ] T038 [US3] Record that the repaired criterion still stands even though the candidate did not — the degeneracy fix is independent of the adoption outcome
- [ ] T039 [US3] Re-derive every cite AFTER the last edit, then run the gate batch and assert 9 exit codes

**Checkpoint**: null recorded, disposition restated. **COMMIT C4-null.**

---

## RESOLVED: Phases 7 and 8 -- 017 was merged in, conflict hand-fixed, both phases executed

**Discovered during implementation, 2026-08-14.** Phases 7 and 8 amend the e-prop adjudication in
`docs/POLICIES.md`. That adjudication's 4-site SCOPE CORRECTION, its `mlp_q` OPEN QUESTION entry and
its `predictive_coding` OPEN QUESTION entry were all added by lane **017-tier0-invariant-and-p0**.

This branch was created off `main`, and 017 is **not** an ancestor of it. Measured on this branch:
`docs/POLICIES.md` is main's 1705-line version, with **0 hits** for "SCOPE CORRECTED", 0 for
"OPEN QUESTION, added 2026-08-13" and **0 for `mlp_q`**. There is nothing here to amend, and writing
the entries fresh would collide with 017 on merge rather than update them.

**The plan missed this dependency.** 015 resolves a site that 017 scoped, so 015 genuinely depends
on 017 — that should have been in plan.md's Technical Context and was not.

**A merge was attempted and ABORTED**: `src/snn_data_lake/basal_ganglia/mlp_q.py` conflicts, because
both lanes edited the same docstring region (017 fixed its stale cross-reference; 015 recorded the
degeneracy and the adoption). Resolving a cross-lane source conflict is a decision, not a mechanical
step, so it is surfaced rather than taken unilaterally.

**Options, for the owner:**
1. Merge 017 into 015, hand-resolve the `mlp_q.py` docstring conflict, then run Phases 7-8 here.
2. Merge both into `main` first, then run Phases 7-8 as a follow-up lane.
3. Run Phases 7-8 on branch 017 instead, where the adjudication already lives.

Everything in Phases 1-6a and 9 is complete and gated. **The code change that closes Tier 0 item #2
has landed**; what remains is recording it in the policy text.

---

## Phase 7: Update the adjudication (P1) — COMMIT C5 *(BLOCKED — see above)*

- [X] T040 [US3] Update the e-prop adjudication in `docs/POLICIES.md` with what was decided — adoption OR null. Cite by MARKER, not line number: this lane's predecessor rotted its own cites three times
- [X] T041 [US3] Confirm all four sites in the adjudication carry either a verdict or an explicit "open, and why", with none left implied (SC-007)
- [X] T042 [US3] Run the gate batch, assert 9 exit codes

**Checkpoint**: adjudication reflects reality. **COMMIT C5.**

---

## Phase 8: User Story 4 — the `predictive_coding` adjudication (P2) — COMMIT C6

**Ships EITHER WAY.** Independent of Phase 5's outcome — a `mlp_q` result does not transfer to a
different mechanism on a different objective, in either direction (R-004).

- [X] T043 [US4] State the case FOR a violation in `docs/POLICIES.md`: `self.w1.T` is the same array transposed — exact weight symmetry by construction — and it runs on every wake tick
- [X] T044 [US4] State the case AGAINST in the same entry: in Rao-Ballard `W1` is the GENERATIVE weight (`x_hat = W1 @ r1`), so `W1^T` is one matrix in its two structural roles rather than a separate forward pathway being copied — the standard locality argument for predictive coding
- [X] T045 [US4] Record that the adjudication does NOT rest on the Predictive E-prop preprint: measured **0** full-text hits for weight transport / weight symmetr / symmetric / feedback alignment / random feedback, against **30** for BPTT. **Silence is not endorsement** (FR-012)
- [X] T046 [US4] Produce the verdict, or record explicitly that it remains open and what would settle it. **Do not pre-decide it from the `mlp_q` outcome**
- [X] T047 [US4] Run the gate batch, assert 9 exit codes

**Checkpoint**: the live site has an adjudication. **COMMIT C6.**

---

## Phase 9: Polish & scope verification

- [X] T048 [P] Verify `git diff --name-only main..HEAD -- src/snn_data_lake/brainstem/` is **EMPTY** — this lane carries no envelope approval
- [X] T049 [P] Verify `git diff --name-only main..HEAD -- src/` — **CORRECTED after the 017 merge**: it lists TWO files, `mlp_q.py` and `self_mod_harness.py`. The second arrived with 017's "It IS constructed" docstring fix, which post-dates this task's "nothing else" wording. Both accounted for; nothing unexpected in the diff
- [X] T050 [P] Verify `tools/dependency_ledger.py --assert` shows no new entry, and that `mlp_q.py` imports no torch, autograd or numpy in the update path
- [X] T051 [P] Verify the probe remains in `tests/probe_015/`, is not under `specs/`, and appears in no gate registration
- [X] T052 [P] Verify every criterion added by this lane has a RECORDED red under `specs/015-resolve-weight-transport/evidence/` — enumerate them and confirm none is missing (P8 guard 1)
- [X] T053 [P] Sweep the lane's own diff for module-level `random.seed(` or `torch.manual_seed(` — determinism invariant
- [ ] T054 Run `graphify update .` — **REFUSED BY THE TOOL, deliberately not forced.** It reports "new graph has 14036 nodes but existing graph.json has 14037. Refusing to overwrite — you may be missing chunk files from a previous session." That is a safety refusal against a partial rebuild; `--force` would discard a node on a guess. Left for a session that can rebuild from complete chunks
- [ ] T055 Final full gate batch from a clean checkout, asserting 9 exit codes, plus the `brainstem/` diff one last time before push

---

## Dependencies & Execution Order

```
Phase 1 (Setup, baselines)
   |
Phase 2 (US1: degeneracy on record, C1)      <- FIRST, so the repair is not retro-justified
   |
Phase 3 (US1: criterion qualifies, C2)        <- BLOCKS Phase 4 absolutely
   |
Phase 4 (US2: the measurement, C3)
   |
Phase 5 (DECISION POINT)                      <- reads research.md D-3 as written
   |
   +-- GO ----> Phase 6a (C4-adopt)
   |
   +-- NO-GO -> Phase 6b (C4-null)            <- the null is a deliverable
   |
Phase 7 (adjudication updated, C5)
   |
Phase 8 (US4: predictive_coding, C6)          <- ships either way
   |
Phase 9 (Polish)
```

### User story dependencies

- **US1** spans C1 and C2. C2 blocks everything downstream — an unqualified criterion cannot judge.
- **US2** (C3) depends on US1's qualified criterion.
- **US3** is the decision point plus exactly one of 6a/6b.
- **US4** (C6) is independent of US3's outcome by design (R-004). It could in principle run at any
  point; it is placed last only so the adjudication edit lands once.

### Parallel opportunities

- **Phase 1**: T002–T005 all parallel (independent evidence files)
- **Phase 9**: T048–T053 all parallel (independent read-only checks)
- **Phases 2–8**: essentially sequential — each is one commit against one file, and the ordering is
  the point

### NOT parallelizable

- Phase 3 before Phase 4 — absolute. The criterion qualifies before it judges.
- Phase 5 gates 6a/6b — exactly one executes.

---

## Implementation Strategy

### MVP

**Phases 1–3 (through C2)** is the smallest honest increment: the degeneracy recorded and a criterion
that has been shown to discriminate. **That alone closes the spec's central finding**, and it stands
whether or not any candidate is later adopted.

### What could stop the lane early, legitimately

- **T012 fails** (identity ≠ 1.0) — the harness is wrong; fix it before anything else.
- **T013 fails** (permutation null scores high) — the criterion measures magnitude and is DELETED.
  The lane reports that and the site keeps its disposition.
- **Phase 5 NO-GO** — FR-007 fires, C4-null commits, Phase 7 records it, Phase 8 still ships.

None of these is a failure of the lane. Each is a result.

---

## Notes

- **SFDFA does not transfer.** `MlpQ` is a rate MLP — no membrane, no threshold, no spike, no time
  index in `update`. What transfers is the ESSENTIAL property (feedback estimated and converging),
  and the rate mechanism for that is Kolen-Pollack. Verified absent from the tree: 0 hits for
  Kolen/Pollack/weight-mirror/Akrout across `src/`, `tools/`, `tests/` against a control of 45 files
  mentioning "feedback".
- **The prediction is already written** (research.md D-3) and must not be revised. The comparison in
  T025 is against that text.
- **Cite by symbol or marker.** Editing a file invalidates every citation INTO it, and
  `citation_guard` structurally cannot catch that — it resolves whether a cited FILE is tracked,
  never whether a cited LINE still says what the citation claims.
