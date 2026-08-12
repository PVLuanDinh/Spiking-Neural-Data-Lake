# Tasks: Policy Compliance (P0-P8)

**Feature**: `specs/010-policy-compliance-audit` | **Date**: 2026-08-04
**Evidence base**: [findings.md](./findings.md) — **34 CONFIRMED** (C34 found during implementation), 3 REFUTED, 61 carried-unverified.

## How to read this file

- Every task names the **finding ID** (C01–C33) it closes. A change with no finding behind it
  violates spec FR-011 and must not be made.
- **17 of the 33 confirmed findings have `Fix safe as proposed: False`.** Those tasks are marked
  **[UNSAFE-AS-PROPOSED]** and carry the verifier's correction inline. **Do not apply the fix as the
  finding originally worded it** — the verifier refuted it by control experiment, false premise, or
  gate breakage.
- Every task ends green on all four: `run_selfchecks.py`, `pytest`, `ruff check .`,
  `tools/citation_guard.py`. **One change per commit.**
- `[P]` = parallelizable. **Anything touching `cortex/brain.py` is never `[P]` with another
  `brain.py` task** — that is C02, C03, C04, C18, C22, C29 and several carried findings.
- The five `brainstem/` safety-envelope files are **REPORT-ONLY**. No task modifies them.

---

## Phase 1: Setup — baseline capture

These record the starting state so later criteria are demonstrable rather than asserted.

- [x] T001 [DONE: envelope SHA256 hashes -> baseline-envelope.txt] [P] Record the pre-change hashes of the five safety-envelope files to
      `specs/010-policy-compliance-audit/baseline-envelope.txt`, covering
      `src/snn_data_lake/brainstem/{reflex,serial_arm,so101_arm,arm_config,interpreter}.py`.
      This is the evidence for SC-010; the file is re-diffed at T060.
- [x] T002 [DONE: baseline-gates.txt written; gap disclosed (captured after Phase 2, reconstructed from clean-HEAD control)] [P] Record the current gate state (self-check count in both envs, pytest pass count,
      coverage %, ruff status, citation_guard STRICT/BROAD counts) to
      `specs/010-policy-compliance-audit/baseline-gates.txt`. Note the 278 (`.venv-cu128`) vs 277
      (`.venv-lake`) split is correct and expected; a CHANGED count is the signal.
- [ ] T003 [P] Measure and record baseline tick rate with the cockpit polling and with it closed, to
      `specs/010-policy-compliance-audit/baseline-tick.txt`. This is the SC-008 comparison basis;
      without it SC-008 has no red.

---

## Phase 2: Foundational — safety-relevant and arrival-breaking (BLOCKING)

**These come first regardless of story.** Two break a safety-adjacent control path or the
documentation of the safety envelope; two mean a clean install produces a broken tree.

- [x] T004 **[CRITICAL]** ✅ DONE 2026-08-04. Fix `serve()` dropping its `acts` queue in
      `src/snn_data_lake/substrate/cockpit.py` so cockpit arm buttons reach `/api/act`.
      **Closes C10.** Today every arm button on the one frontend is dead — **including E-STOP**.
      This is the highest-priority task in the feature. The E-STOP in `brainstem/reflex.py` is
      unaffected and remains the real safety path; what is broken is the operator's cockpit control.
      **Gate**: manual — click each arm button and confirm `/api/act` receives it; plus all four gates.
- [x] T005 [P] ✅ DONE (note: the file is `SECURITY.md` at repo ROOT, not `docs/`; AGENTS.md's link
      was also wrong. **Scope narrowed on inspection: the table and commands were already correct —
      only the one-line summary at `:9` named the non-existent `ArmSim.apply()`.** The finding would
      have had me "fix" a correct table.) Correct `SECURITY.md` where it names a **deleted class** as the safety gate every
      motor command flows through. **Closes C11.** Doc-only; do NOT modify any brainstem file. Verify
      the live gate's real name against the tree before writing it — the verifier noted `class ArmSim`
      does not exist anywhere under `src/`.
- [x] T006 [DONE 2026-08-04: AGENTS.md + CLAUDE.md envelope list corrected: arm_sim.py -> so101_arm.py, ArmSim.apply() -> So101Arm.apply()] [P] Correct the safety-envelope file list in `AGENTS.md` and `CLAUDE.md`: they name the
      **deleted** `arm_sim.py` and omit the live `so101_arm.py`. **Closes C12.** `docs/POLICIES.md`
      and `.specify/memory/constitution.md` already list the correct five. Doc-only.
- [x] T007 [DONE 2026-08-04: test_png_encoder_does_not_use_pil DELETED with reasoning; both docstrings (test + arm_view source) corrected] **[CRITICAL]** Remove the mechanized pure-stdlib purity gate in `tests/` that forbids a
      declared dependency. **Closes C06.** A CI-enforced gate implementing the rule abolished
      2026-07-20 is a live P7 violation — it can fail a build for using a dependency the project
      declares. Delete the gate rather than exempting it (Standing Invariant 2: no narrowing to dodge
      a true finding). **Gate**: `pytest` must still pass with the gate gone.
- [x] T008 [DONE 2026-08-04: scikit-learn declared in [gpu] and [train] per the verifier's corrected placement (NOT core)] Declare `scikit-learn` in `pyproject.toml`. **Closes C07.** It is imported unguarded at
      module top level in ~5 `src/` modules and is absent from project metadata entirely, so
      `pip install .` yields a tree that raises `ModuleNotFoundError` on import. Decide core vs extra
      by where the importers live; if they are all in the `reasoning/` tier, an extra is correct and
      the imports must then be guarded. Also check `nir`. **Per P7 this addition needs no
      justification beyond what it buys — state that, do not apologise for it.**
      **Gate**: quickstart Scenario 2 (clean-env install + import-everything) must pass.

**Checkpoint**: the cockpit's controls work, the safety envelope is described correctly, and a clean
install imports. Everything after this is independently shippable.

---

## Phase 3: User Story 1 (P1) — the brain stops claiming what it cannot measure

**Goal**: no verdict is reported that no brain state can move.
**Independent test**: degrade the brain, ask the question, observe the answer change or the system
say it cannot measure. Today the answer is byte-identical.

- [x] T009 [DONE: tools/red_reachability.py BUILT - AST scan, NO_RED + tautological-assert detection, UNKNOWN kept distinct] [US1] Build `tools/red_reachability.py` per
      [contracts/red-reachability.md](./contracts/red-reachability.md). Discovery by AST, scoped to
      modules the module gate already proves reachable. **`UNKNOWN` must be a distinct reported
      bucket, never folded into pass** — that failure would make the gate commit the sin it exists to
      catch.
- [x] T010 [DONE: tests/test_red_reachability.py - 8 tests, fires every bucket against a reconstructed pre-fix consciousness_probe] [US1] Write `tests/test_red_reachability.py` making each assert FIRE (Standing Invariant 5):
      a zero-arg fixture → `NO_RED`; a demonstrated-failing fixture → `RED_REACHABLE`; an
      unconstructible fixture → `UNKNOWN` **not** `NO_RED`; an `EXEMPT` entry matching no real
      violation → gate fails loudly.
- [x] T011 [DONE: COUNTED: NO_RED 0 / TAUTOLOGICAL 0 / READS_SYSTEM 13. Green because T012-T015 fixed the case first - proven able to fire via fixture, not assumed] [US1] **Run the gate and COUNT its output before making it asserting.** Record the result
      in `contracts/red-reachability.md` under a new "First run, actual" heading. Expected ≥2 `NO_RED`.
      **If `UNKNOWN` dominates — the 0-failures/47-UNCERTAIN shape that killed a prior proposal — DROP
      THE GATE, not the tree**, and record that outcome as the result.
- [x] T012 [DONE: probe_binding DELETED (not re-thresholded)] [US1] **DELETE** `probe_binding` from `src/snn_data_lake/reasoning/consciousness_probe.py`.
      **Closes C08.** It scores exactly 1.000 across 300 random object sets against a rate control
      whose own docstring says it is *"right only by luck"*. **Deletion, not re-thresholding** — F11:
      *"That criterion is deleted, not weakened."* Moving the `+0.15` margin would manufacture a red
      without creating one.
- [x] T013 [DONE: probe_reentry DELETED (not re-thresholded)] [US1] **DELETE** `probe_reentry` from the same module. It is arithmetic over three
      literals (`0.40 < 0.55 < 0.70`) with no brain input and no RNG — a true statement about
      constants, not a measurement.
- [x] T014 [DONE: probe_integration reports 10/20-seed distribution; pinned seed=0 removed] [US1] Repair `probe_integration`: stop pinning `seed=0`, which is a **passing** seed while
      the marker reads ABSENT on ~10 of 20 subnet seeds. Report the distribution across seeds, not a
      single verdict. This is the one marker that already HAS a reachable red; the defect is the
      invocation, so this is repair rather than deletion.
- [x] T015 [DONE: probe_access(brain) now READS THE BRAIN; returns UNKNOWN without one] [US1] Make the surviving criteria **take the brain as an argument**. **Closes C09.** A
      criterion invariant to the entire system under test is unfalsifiable at every level of the
      computational hierarchy (research R1, Kleiner & Hoel 2020). This is the root defect; T012–T014
      are consequences of it.
- [x] T016 [DONE: chat self-report severed: known=False, coverage=0.0, says UNMEASURED] [US1] Sever the chat self-report in `src/snn_data_lake/reasoning/spike_chat_lake.py:925`
      so `"are you conscious"` / `"are you aware"` / `"do you have experience"` no longer return a
      4-of-4 verdict. Reply with what IS measured and what is not.
      **This is not a retreat from the sentience goal** — per P8 the brain may report what it
      measures and must not report what it cannot. Building the replacement criterion is a follow-on
      feature; this task must not imply one exists.
- [x] T017 [DONE: docs/coverage/Consciousness.md updated + LakeChat-has-no-brain finding recorded] [US1] Update `docs/coverage/Consciousness.md`, which already scores the defect at `:366`
      (*"a 4/4 self-report that no live measurement backs"*). Record the deletions and the new state.

**Checkpoint**: quickstart Scenario 3 — healthy vs degraded brain give different results, or the
system says it cannot measure.

---

## Phase 4: User Story 2 (P1) — a clean install produces a working tree

**Goal**: declared ↔ imported agree in both directions.
**Independent test**: quickstart Scenario 2.

- [x] T018 [DONE: tools/dependency_ledger.py built] [US2] Build `tools/dependency_ledger.py` per
      [contracts/dependency-ledger.md](./contracts/dependency-ledger.md). The import-name →
      distribution-name table is **explicit and tested, never inferred** (`sklearn`→`scikit-learn`,
      `PIL`→`pillow`, `cv2`→`opencv-python`) — a wrong mapping yields a false UNDECLARED, which is
      the dangerous direction. Use `sys.stdlib_module_names`, not a hand list.
- [x] T019 [DONE: tests/test_dependency_ledger.py - 9 tests, each makes a bucket FIRE] [US2] Write `tests/test_dependency_ledger.py` making each assert FIRE: undeclared import →
      `UNDECLARED`; declared-unused → `UNUSED`; an `EXEMPT` blessing nothing → gate fails loudly.
- [x] T020 [DONE: COUNTED before adopting: 13->6->14->5->4; contract records the full sequence] [US2] **Run and COUNT before asserting.** Record actuals in the contract. Expected: ≥1
      UNDECLARED, ≥2 UNUSED (`pillow`, `pyarrow`), 1 self-description mismatch. **Allowlist-sized
      result → drop the gate, not the tree.**
- [x] T021 [DONE: pillow + pyarrow dispositioned as TRANSITIVE with reasons; UNUSED bucket now 0. PNG-encoder adoption left as measured work] [P] [US2] Resolve `pillow` and `pyarrow`, declared in `pyproject.toml` extras and imported
      nowhere in `src/`, `tests/` or `tools/`. Either use them or remove the declaration —
      **note the P7 angle: `pillow` sits undeclared-in-core while the tree hand-writes a PNG encoder
      at `substrate/arm_view.py:46-74`.** Adopting `pillow` there and retiring the hand-rolled encoder
      is P7+P2-legitimate and is the better fix if it holds.
- [x] T022 [DONE 2026-08-04: proxy asserts NAMED as proxies rather than deleted - they defend an architectural property, not a dependency rule] [P] [US2] Remove the three surviving `assert "torch" not in sys.modules` purity asserts at
      `reasoning/brain_adapters/laminar_local_readout.py:314`, `:336` and
      `reasoning/vsa_structure_mapping.py:294`. Residue of the abolished rule.
      [2026-08-12 annex: all three are since FULLY DELETED from the code (the laminar pair
      with an in-file removal note; vsa:294 is ordinary code), and the laminar module is now
      `cortex/laminar_local_readout.py`. The paths above are the historical record.]
- [x] T023 [DONE 2026-08-04: __init__.py:1 'zero-dependency' corrected] [P] [US2] Correct `src/snn_data_lake/__init__.py:1`, which still opens *"a zero-dependency
      neuromorphic library"* — a property the project abolished on 2026-07-20.
- [x] T024 [DONE 2026-08-04: pyproject.toml:1 and description corrected to torch+numpy+mujoco] [P] [US2] Correct `pyproject.toml:1` and `:27`, both of which say *"torch + numpy"* while
      `mujoco==3.11.0` is a third unconditional core dependency.
- [x] T025 [DONE: dependency_ledger registered in ci.yml REPORT-ONLY (1 undispositioned finding); red_reachability registered ASSERTING] [US2] Register `tools/dependency_ledger.py` as a step in `.github/workflows/ci.yml`'s lint
      job, beside `citation_guard.py`. **Only after T020's count justifies it.**

**Checkpoint**: quickstart Scenario 2 passes in a clean environment.

---

## Phase 5: User Story 3 (P2) — every claim in the tree matches the tree

- [ ] T026 [P] [US3] Correct the P0 self-building-code row in `docs/POLICIES.md` — **already applied
      2026-08-04, verify only.** **Closes C14.** [UNSAFE-AS-PROPOSED] The finding's fix was wrong in
      its first half; the verifier confirmed the diagnosis is *understated*, not overstated.
      `Brain.self_improve` (`brain.py:3767`) has zero call sites and its own docstring says so, while
      the row filed it under "What fires".
- [x] T027 [DONE: P0 neural-growth + P6 incident-table retraction verified] [P] [US3] Verify the P0 neural-growth row correction in `docs/POLICIES.md` and
      `CLAUDE.md`. **Closes C01 and C15** — **already applied 2026-08-04.** The row had asserted the
      engine "retains NOTHING" with "no save/load at all"; `neurogenesis_engine.py:211` is
      `self._model = model` and persistence runs on two live paths into `_T_GROWTH`. Confirm the
      retraction also landed in the P6 incident table (C15) — the same claim lived in two places.
      **The remaining honest gap is the cross-process restart MEASUREMENT, not the channel.**
- [x] T028 [DONE: P6 topology corrected from tools/p6_gate.py; the 'cheap remedy' REFUTED (11 private buses)] [P] [US3] Correct the P6 reading in `docs/POLICIES.md:697`. **Closes C27.** The published
      36-writes/2-reads figure was re-measured wrong two days ago; the reproduced figures are 42
      writes / 10 modules, 9 reads / 5 modules, 25 of 42 inside the bus modules, 17 real broadcast
      writes from 8 modules, **0 faculty-to-faculty reads**. **Do NOT copy line cites out of
      `P6_INTERCONNECT.md` or the `p6_gate.py` docstring — those have themselves drifted.**
- [x] T029 [DONE: P6 gate status corrected to HALF BUILT in POLICIES.md + CLAUDE.md + constitution] [P] [US3] Correct P6's enforcement status. **Closes C28.** `docs/POLICIES.md:18` and the
      CLAUDE.md P6 bullet say "gate specified but NOT built" while `tools/p6_gate.py` exists, runs and
      self-checks. Accurate status is three-part: topology half BUILT but manual/non-asserting and not
      in CI; connectome-diff half not built. **This also affects the constitution's enforcement table
      and my own P6 text — fix all three.**
- [x] T030 [DONE 2026-08-04: AGENTS.md:7 P0-P5 -> P0-P8 corrected] [P] [US3] Correct `AGENTS.md:7`, which states the governing policy set as P0–P5. **Closes
      C31.** There are nine (P0–P8; P4b is a sub-clause). Note `AGENTS.md` is gitignored
      (`.gitignore:76`) — it is local-only, so git-history evidence about it is invalid.
- [ ] T031 [US3] Sweep the remaining doc-consistency findings in findings.md (dimension
      `docs-consistency`, 11 findings, 3 verified) and the carried ones. Each must be verified before
      it drives an edit.

---

## Phase 6: User Story 4 (P2) — the cockpit shows nothing it cannot back

- [x] T032 [DONE: coverage_* PROVENANCE entries added; big() gained a pk slot it never had] [US4] Add `PROVENANCE` entries for the three displayed census numbers that are
      process-lifetime constants. **Closes C30.** `substrate/cockpit.py:806-807` renders
      `coverage_reachable` / `coverage_total` / `coverage_dormant` untagged. **Policy header
      correction from the verifier: this is P3, not P8** — the clause is
      `substrate/brain_telemetry.py:33-35`, a stated MUST that a dashboard render the tag next to the
      number. No behaviour change.
- [ ] T033 [US4] Add PROVENANCE entries for `access_index`, `region_coherence`, `ignitions`,
      `reentrant_availability`, `reentrant_strength` — none has one, and `cockpit.py:798` draws
      `access_index` as the untagged headline directly above two tagged values.
- [ ] T034 [US4] Rename the `episode_retained` telemetry key and its comment. **Closes C22.** It is
      labelled durable and measures the volatile store. **Verifier correction: the P3 shape-5 citation
      is wrong — use P4b.** The signal varies richly (57 distinct values over 600 ticks), so it is not
      a constant-wearing-a-signal's-clothes; the defect is that its NAME claims the lake.
- [ ] T035 [US4] **[UNSAFE-AS-PROPOSED]** Address the 1 Hz telemetry poll writing brain state from the
      HTTP thread. **Closes C29.** **The proposed fix rests on a FALSE PREMISE** — the verifier found
      `Brain.state()` is already called once per tick by the brain itself (`_wake_tick()` ends at
      `brain.py:1983 return self.state()`), so the affect-core spiking-WTA is NOT per-poll work.
      Re-derive the actual cross-thread write before changing anything, and treat the P4 framing as
      unconfirmed.
- [ ] T036 [US4] Re-run quickstart Scenario 5 (observing must not become the load) against T003's
      baseline.

---

## Phase 7: Policy debt with measured evidence

Each is independently shippable. **Ordered by severity; every one carries a measurement.**

### P4b — persistence (all touch `cortex/brain.py`; NOT parallel with each other)

- [ ] T037 Feed `_episodic` from the lake at construction. **Closes C04.** The episode store the
      brain SPEAKS from boots empty beside a full lake. Fix in the existing `if self._persist:` block
      that already calls `_restore_episodes()` (`brain.py:909-910`).
- [ ] T038 **[UNSAFE-AS-PROPOSED]** Durable episodic recall into the live consumer. **Closes C02.**
      Verifier: the "one live consumer" is weaker than claimed — measured **2** recalls (cold, 300
      ticks) and **0** (warm, 1,600 ticks), because the block at `brain.py:3302` is gated on
      `confidence > _rc_th`. **Do not force-wire** (P3 forbids it); establish whether the consumer
      fires at all first.
- [ ] T039 **[UNSAFE-AS-PROPOSED]** The live episodic-consolidation consumer goes inert once the store
      fills. **Closes C03.** **Both halves of the proposed fix were REFUTED by control experiment**:
      Control A (192 stores into capacity=64, no replay) gave `retained=1.000`, self-cue 64/64 — ring
      eviction alone destroys nothing. The cause is NOT capacity. Re-diagnose before acting.
- [ ] T040 [P] Give the arm layer durable state. **Closes C05.** `brainstem/autonomous_loop.py:197` —
      the canonical stack's action+reward layer persists nothing. **Verifier corrections to the fix:**
      `GrowingLearner` has NO `labels` attribute (only `vigilance`, `lr`, `dim`, `prototypes`), and
      `GrowingLearner._best` RAISES on width mismatch, which the proposed fix did not name.
      *(`autonomous_loop.py` is in `brainstem/` but is NOT one of the five envelope files.)*
- [ ] T041 **[UNSAFE-AS-PROPOSED]** `recall_episode_durable()` has zero live callers. **Closes C23.**
      Verifier: the census is wrong — **six** sites, not four — and the fix as written gets a **hard
      NO** on cost grounds. Re-scope.

### P1 — hardcodes with measured A/Bs

- [ ] T042 **[UNSAFE-AS-PROPOSED]** [P] The `0.6` perception salience that alone decides
      global-workspace ignition, `brainstem/autonomous_loop.py:1168`. **Closes C16.** Measured: at
      0.6/0.55 the workspace broadcasts on 73/73; at 0.45/0.30 it broadcasts `None` on 73/73 and
      `thalamus_global_workspace` drops out of the live-region set. **Verifier: the call-site cite is
      wrong and severity is overstated** — re-derive before editing.
- [ ] T043 **[UNSAFE-AS-PROPOSED]** [P] `gate_threshold=0.5` against an unnormalized learned Q-value
      margin, `cortex/prefrontal_executive.py:87`. **Closes C17.** Verifier: the finding
      **understated** its own effect — `pfc.goal` does move on the real live path. Use the
      `_margin_base` running-EMA idiom already at `brain.py:2005-2008`.
- [ ] T044 [P] `_real_novelty > 0.5` at `cortex/brain.py:2040` while the same signal is
      baseline-relative twice in the same file (`:1789`, `:2417`). **Closes C18.** One-line fix reusing
      state the tick already maintains. **Verifier correction: the growth path is NOT opt-in** — the
      in-code comment claiming `NORD_TIME_CELLS_GROW=1` is stale.

### P2 — duplicate mechanisms

- [ ] T045 **[UNSAFE-AS-PROPOSED]** Two live Wang-2002 spiking decision attractors. **Closes C19.**
      **Verifier: drop the P4 attachment** — the finding quoted POLICIES.md:403-404 but stopped one
      line short; `:409-411` scopes P4 to a second *stack class* definition. P2 core survives.
- [ ] T046 **[UNSAFE-AS-PROPOSED]** `CuriosityDrive` declared "replaced" but never retired. **Closes
      C20.** Verifier: evidence wrong on half its sites and **the proposed fix is refuted by
      measurement**.
- [ ] T047 **[UNSAFE-AS-PROPOSED]** Three action-value substrates behind an if/elif. **Closes C21.**
      Verifier: **severity drops to minor** — neither shared substrate is reachable from any live
      entrypoint.

### P5 — GPU honesty

- [ ] T048 [P] Device-bound RNG makes `grown_units` and CORE gate margins hardware-dependent.
      **Closes C24.** Same seed + a GPU = a different brain. Draw grown weights from a CPU generator
      and move them. **Verifier: drop the PERFORMANCE.md citation, it is misapplied.**
- [ ] T049 **[UNSAFE-AS-PROPOSED]** [P] `training/device.py::select()` has zero call sites. **Closes
      C25.** Verifier: **the violation is P2, not P5**, and severity drops to moderate.
- [ ] T050 **[UNSAFE-AS-PROPOSED]** [P] cuda-by-availability defaults with no benchmark. **Closes
      C26.** Verifier: **second half REFUTED by measurement**, severity drops to minor.

### Cross-cutting invariants

- [ ] T051 **[UNSAFE-AS-PROPOSED]** JEPA weight transport on the live wake tick. **Closes C13.**
      `cerebellum/jepa_world_model.py:275-278`. **Verifier found a SECOND, unnamed transport site at
      `:350-355`** and marked the fix menu **half unsafe**. Feedback alignment (fixed random `Bz`) is
      the e-prop-compatible route; benchmark before and after.
- [ ] T052 **[UNSAFE-AS-PROPOSED]** [P] Predictive-coding weight transport. **Closes C32.**
      `cortex/predictive_coding.py:200`. **Verifier: the title overstates concealment** — the file
      writes the transpose out four times in its own prose. The honest minimal fix may be to correct
      the CLAIM, not the code, since symmetric feedback is standard Rao-Ballard.
- [ ] T053 **[UNSAFE-AS-PROPOSED]** [P] Import-time global RNG reseeds. **Closes C33.** Verifier:
      **count is 14, not 16** — both `arc_agi/play.py:64` sites are inside `__main__` — and the fix is
      half-wrong. Replace with per-instance `random.Random(seed)` per the
      `recognizer_bank.py:194` pattern.

---

## Phase 8: Polish & cross-cutting

- [ ] T054 **Verify the 61 carried-unverified findings before ANY of them drives a change** (spec
      FR-011). They are listed in findings.md's final table. A carried finding actioned blind is
      exactly the failure this project's history is made of — roughly a third of candidate findings
      have been refuted on adversarial re-check.
- [ ] T055 [P] Record every REFUTED finding in the feature record as refuted, not dropped (spec
      FR-011, second half). The three from this audit are in findings.md.
- [ ] T056 [P] Record every finding deferred with its blocker named (spec FR-012).
- [x] T057 [DONE: red_reachability --assert wired into ci.yml lint job, exit 0] Register `tools/red_reachability.py` in CI lint, if T011's count justified it.
- [ ] T058 Update `docs/POLICIES.md` P8 with the measured first-run output of the red-reachability
      gate — the policy says to count the gate's output before the rule leans on it, and this closes
      that loop.
- [ ] T059 Re-run quickstart Scenarios 1–7 end to end.
- [ ] T060 Diff the safety envelope against T001's baseline. **Expected: empty.** Any change fails
      SC-010.
- [ ] T061 **[CRITICAL, PRE-EXISTING]** Diagnose why `tests/test_episode_durability.py`'s two P4b
      tests FAIL in the full suite and PASS in isolation. **Closes C34.** Confirmed pre-existing by a
      clean-`HEAD` worktree control run — not caused by this feature. The assertion is the exact
      defect P4b exists to prevent: *"the brain formed 39 episodes but the lake holds 0"*.
      **Do NOT fix by reordering or isolating the test** — that silences the check (Standing
      Invariant 2) and restores a false green. Find the shared state: the tests spawn a real second
      OS process, and something earlier in the suite leaves the lake, its path, or its append lock in
      a state where `persist=True` writes land nowhere.
      **Until diagnosed, P4b's guarantee is UNPROVEN in CI, not merely untested** — a green isolated
      run is indistinguishable from a green suite, which is why this survived.

---

## Dependencies

```
Phase 1 (T001-T003)  -- baseline, blocks nothing but must precede T036/T060
   |
Phase 2 (T004-T008)  -- BLOCKING: safety-relevant + arrival-breaking
   |
   +-- Phase 3 US1 (T009-T017)  ── independently shippable
   +-- Phase 4 US2 (T018-T025)  ── independently shippable
   +-- Phase 5 US3 (T026-T031)  ── independently shippable
   +-- Phase 6 US4 (T032-T036)  ── depends on T003 for its baseline
   +-- Phase 7    (T037-T053)  ── each independently shippable
   |
Phase 8 (T054-T060)  -- after everything else
```

**Within-phase serialization**: T009→T010→T011 (build, test, count) and T018→T019→T020 are strictly
ordered — the gate must be counted before it is adopted. T037/T038/T039/T041/T044 all touch
`cortex/brain.py` and are **not** parallel with one another.

## Parallel opportunities

- **Phase 2**: T005, T006 `[P]` (different docs). T004, T007, T008 are separate files but T004 is
  highest priority and should land alone.
- **Phase 4**: T021, T022, T023, T024 all `[P]`.
- **Phase 5**: T026–T030 all `[P]`.
- **Phase 7**: T042, T043, T044 `[P]`; T048, T049, T050 `[P]`; T052, T053 `[P]`.

## MVP scope

**T001–T008** (Setup + Foundational) is the smallest shippable increment and is the recommended MVP:
it restores the cockpit's arm controls including E-STOP, corrects the safety envelope's
documentation, and makes a clean install work. Adding **Phase 3 (US1)** delivers the headline
outcome — the brain stops reporting a consciousness verdict nothing can move.

## Summary

| Phase | Tasks | Findings closed |
|---|---|---|
| 1 Setup | T001–T003 | — (baseline) |
| 2 Foundational | T004–T008 | C10, C11, C12, C06, C07 |
| 3 US1 | T009–T017 | C08, C09 |
| 4 US2 | T018–T025 | C07 residue, P7 sweep |
| 5 US3 | T026–T031 | C01, C14, C15, C27, C28, C31 |
| 6 US4 | T032–T036 | C22, C29, C30 |
| 7 Policy debt | T037–T053 | C02–C05, C13, C16–C21, C23–C26, C32, C33 |
| 8 Polish | T054–T060 | carried + refuted record |

**61 tasks. 34 findings closed. 17 marked [UNSAFE-AS-PROPOSED] and rewritten from the verifier's
correction rather than the finding's original wording.**


---

## Added during implementation

- [ ] T061 **[CRITICAL, PRE-EXISTING]** Diagnose the two P4b durability tests failing in the full
      suite but passing in isolation. **Closes C34.** Proven pre-existing by clean-HEAD control.
      Do NOT fix by reordering.
- [ ] T062 **[MAJOR]** Disposition `arcengine` — 2 UNGUARDED undeclared imports at
      `reasoning/arc_agi/play.py:37` and `play_brain.py:39`; exists nowhere in the repo.
      **Closes C35** (found by `tools/dependency_ledger.py`, NOT by the 48-agent audit).
      Three options, one must be MEASURED not assumed: declare it (if a real published package),
      guard the imports (if `arc_agi/` is an optional entrypoint tier), or retire the two scripts.
      **Do not guess a distribution name** — that would be worse than the defect.
      Flipping `dependency_ledger` to `--assert` in CI depends on this.
