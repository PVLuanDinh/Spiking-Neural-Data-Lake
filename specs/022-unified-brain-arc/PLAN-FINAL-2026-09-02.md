# SNDL — THE FINALIZED DEVELOPMENT PLAN AND SESSION GUIDE (2026-09-02)

**This file lives at `specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md`. Read all of it before doing
anything. It is written so that a fresh session of Claude Fable 5.1 or Claude Opus 5 with no memory of the
project can pick the work up at any phase and carry it correctly.** It consolidates the trajectory audit's
§8 roadmap (`evidence/trajectory-audit-2026-09-01.md`; ranks 1, 1b, 2a, 2b, 2c are now MEASURED) and the
remainder of specs/022 (US3 T062–T071, US4 T072–T083, Polish T084–T087) into one ordered program.
`tasks.md` keeps every task id; this document owns ORDER, GATES and REDS. **RATIFIED by the owner in chat
on 2026-09-02** (O1–O5 authorized verbatim; audit §9 item 4 satisfied), so it binds. It went through the
same three-lens adversarial panel that reviews code, twice, before landing; the fold of the second round
is battery-verified, not re-panelled.

**Landing prerequisite, met (P4c amendment: a build that has not reached `main` is not a build this project
can claim):** this document cites rank 2b (`evidence/s2b-growing-learner-lake-record.md`, landed at
ef5ca0cb) and rank 2c (`evidence/s2c-rotating-stream-refused.md`, landed at e68f8b80 with `tasks.md` row
A2c flipped to REFUSED ON MEASUREMENT). Both preceded this plan (4f20a7e8) on `main`, so every sentence
here about them is about `main`.

---

## PART A — HOW TO WORK HERE (the session protocol; nothing below is optional)

### A.1 What this project is

A spiking neural "digital brain" (`src/snn_data_lake/`, ~100 modules organized by anatomy: cortex,
hippocampus, basal_ganglia, amygdala, cerebellum, brainstem, thalamus, neuromodulation, plasticity,
substrate) whose ONE durable memory is a single append-only lake (`brain_lake.bin`, tagged records) and whose
drive is an always-on daemon (`substrate/learn_daemon.py`, cockpit at http://127.0.0.1:8137/). The stated
goal is a brain that builds itself from one neuron toward and beyond human scale, with general intelligence
as the target and sentience pursued without being claimed. Growth is EARNED (a unit stays only if it lowers
held-out error), learning must be local (e-prop, not backprop), every mechanism must charge behaviour or be
retired, and every claim ships with a control and a red that can fail.

### A.2 The governing text, in precedence order

1. The owner's words in chat.
2. `docs/POLICIES.md` — P0–P8 and the cross-cutting invariants. READ THE FILE, not a summary, at the start
   of every session: P2 (one mechanism survives; port the unique capability first), P3 (reachability is not
   charge; a wire that does not charge is reverted), P4/P4b (one brain, one lake; "it persists" is a claim
   about a NEW process), P8 (a reachable RED, a control that is not the method, per-item attribution,
   disclosure, deletion on a null), the invariants (verify-first; runtime evidence outranks static reading;
   ONE change per commit with the full gate green; implement a gate and COUNT its output before the rule
   lands; cite the SYMBOL not the line).
3. This document (order, gates, reds), then `specs/022-unified-brain-arc/{spec,plan,tasks}.md` and
   `SYNTHESIS.md` (its §3 is the stage ordering, §3.5 the migration rules).

### A.3 Standing rules that never relax

- **The safety envelope**: `brainstem/reflex.py`, `serial_arm.py`, `so101_arm.py`, `arm_config.py`,
  `interpreter.py` change ONLY with explicit owner approval in chat, and the MuJoCo scene and its loaders
  live inside two of them (`arm_config.py`, `so101_arm.py`), so a scene change is an envelope-approval
  event. Every motor command flows through `So101Arm.apply()`. **The E-STOP latch (`estop.latch` under the
  lake home) is cleared ONLY by a person at the cockpit. Never clear it, never command motion around it.**
- **Never write to the live lake** (`C:\Users\vinhl\.snn_data_lake\brain_lake.bin`) from a test or a probe:
  build brains under `substrate.home.throwaway_home()` with `make_brain(seed, persist=False)`, or under an
  isolated `NORD_HOME` temp dir for restart proofs. Never stand up a second brain against the live lake.
- **Subagents inherit the main session's model** (owner rule 2026-09-02, superseding the earlier Fable-only
  rule): pass no model override to the Agent or Workflow tools. Review panels are three lenses on that model.
- **One change per commit; one full battery per pushed ARC, on its stacked head** (owner rule 2026-09-02,
  POLICIES' "Gated + reversible" amendment). Every commit names the battery that verified the head it was
  pushed under; a red on the head is attributed per stage per commit, never by whole re-runs.
- **Commit messages carry NO attribution trailer** (owner, 2026-09-03, superseding the session-model
  trailer rule this line used to state). `attribution: {commit: "", pr: ""}` in `~/.claude/settings.json`
  enforces it mechanically rather than by remembering; the rule is kept written down here because the
  setting is machine-local and this file is the tree's copy of the protocol.
- **Launch sessions from `D:\AI-project`, not the repo subdir** (the memory tools key on that cwd). The
  curated memory store is `C:\Users\vinhl\.claude\projects\D--AI-project\memory\` (its `MEMORY.md` index
  auto-loads; its START-HERE file is the per-session protocol and points back here).

### A.4 The per-task rhythm (every landing this year followed it; do not shortcut a step)

1. **Orient**: read POLICIES' binding sections; `graphify query "<question>"` for architecture;
   `graphify-out/SRC_INVENTORY.md` for what exists; **before building any capability run
   `node C:\Users\vinhl\.claude\hooks\graph-exists.js "<capability>"`** — a P2 incumbent you did not name is a
   defect the review will find.
2. **Pre-register**: write the evidence file's Protocol section (arms, statistic, bar, controls, the
   disposition on a null) BEFORE the first run. Every seed-paired red states S, the statistic (paired
   one-sided t, bar 1.895 at 7 d.f. for S = 8, α = 0.05), ties counted, and reports the rule's permutation
   false-positive rate beside the verdict. Never normalize by a quantity a control destroys (obs #105).
   Build any oracle arm at the mechanism's own input contract (rank 2a: a one-step lag bug read as a null).
3. **Build**, then **measure** in isolated processes (one arm per subprocess, cached per arm, 4 workers).
   Every harness and the battery run under the repo venv from the repo root:
   `PYTHONUTF8=1 PYTHONPATH=src .venv-lake/Scripts/python.exe <script>` (Python 3.14 free-threading; the
   venv works from a plain shell today — O2 below concerns only the logon launcher's preflight). The
   archived templates are `rank2a/rank2_ab.py`, `rank2b/rank2b_charge.py`, `rank2c/probe_chains.py` and
   `rank2c/probe_fixed_vs_rotating.py` under the memory store's `archive/audit-2026-09-01/`.
4. **Panel**: one per ARC — three lenses (instrument / policy-honesty / integration) via the Workflow
   tool on the frozen stacked tree; every finding must be reproduced by a command; fold the findings (as a
   further stacked commit); the fold is either re-panelled or stated in the commit as battery-verified
   only. Do not edit the tree while a panel reads it, and no panel agent builds a brain or renders while a
   battery runs (a frozen tree is not a frozen machine).
5. **Battery**: one per ARC, in its own detached command on the frozen stacked head (≈ 60–70 min; the Bash
   tool's 10-minute cap kills it inline, so launch it detached and poll). It OPENS with the execution
   record: `policy_manifest --assert` first (the previous record must name the pushed head or its parent —
   a push that skipped its battery reds here), then `--certify` (runs the policy gate commands and writes
   the record for THIS head). The literal command (`$B` = an output dir outside the repo):

   ```bash
   cd /d/AI-project/spiking-neural-data-lake; export PYTHONUTF8=1 PYTHONPATH=src; B="<output dir>"; mkdir -p "$B"; PY=.venv-lake/Scripts/python.exe; git fetch -q origin; git status --short > "$B/tree.txt"; git diff HEAD --stat >> "$B/tree.txt"; $PY -m snn_data_lake.substrate.policy_manifest --assert > "$B/record_assert.txt" 2>&1 || { echo "RECORD-ASSERT RED: the previous push has no battery record -- battery halted"; exit 1; }; $PY -m snn_data_lake.substrate.policy_manifest --certify > "$B/certify.txt" 2>&1; echo "certify rc=$?"; $PY -m ruff check . > "$B/ruff.txt" 2>&1; echo "ruff rc=$?"; $PY -m ruff format --check . > "$B/ruff_format.txt" 2>&1; echo "ruff-format rc=$?"; $PY run_selfchecks.py > "$B/selfchecks.txt" 2>&1; echo "selfchecks rc=$?"; tail -n 3 "$B/selfchecks.txt"; $PY -m pytest -q -p no:cacheprovider > "$B/pytest.txt" 2>&1; echo "pytest rc=$?"; tail -n 3 "$B/pytest.txt"; for m in dormancy_gate method_dormancy_gate region_charge_gate growth_coverage_gate manifest_gate determinism_gate replay_gate envelope_hash_gate policy_manifest gradient_claim_gate eprop_transport_gate import_manifest replication_guard; do $PY -m snn_data_lake.substrate.$m --assert > "$B/$m.txt" 2>&1; echo "$m rc=$?"; done; echo BATTERY_DONE
   ```

   Green = every `rc=0`, every self-check passed (the runner prints `N/N self-checks passed` and, since H1,
   `6/6 tools gates passed`), pytest coverage above the 70 % bar (90.7 % today). After a green battery the
   regenerated `tools/policy_certification.json` is committed as the arc's last commit ("execution record:
   battery #N on <head>") and the arc is pushed. A red in any stage stops the push; a red you cannot
   attribute to one commit (re-run that STAGE per commit in a worktree) is a finding.
6. **Commit** (one change per commit, stacked within the arc), **push the arc**, then **poll the daemon**: it
   exits on a new commit and its supervisor relaunches it on the new build (`chat_learn.log` prints
   `head=<hash>`); `GET /api/state`; one chat via `/api/chat`. Record the observed drop or gain; never
   smooth it.
7. **Close out**: route-ledger line, task-observer observation(s), memory START-HERE + handoff, the audit
   artifact republished if the audit doc changed.

Measured cost per ARC: one panel (≈ 25 min) + one fold + one battery (≈ 60–70 min), plus a re-run when the
head reds; commits inside an arc cost only their build and local checks. Before the batched rule every
commit paid the full ≈ 2 hours, which is what the owner's 2026-09-02 amendment removed.

### A.5 Where things are (the map a fresh session needs)

| what | where |
|---|---|
| the brain, the wake/sleep tick, the answer path | `cortex/brain.py` (`Brain`, `_wake_tick`, `_sleep_tick`, `_safe_ctx`, `respond`, `state()`, `_flush_growth`, `_flush_prototypes`, `make_brain`) |
| the arm loop (perception → prototype category → action; the JEPA world model; curiosity) | `brainstem/autonomous_loop.py` (`AutonomousArm.step`, `_context_pattern`, `learner`, `q`, `coverage_reward`, the arm's own `_autotelic_goal_thresh` latent-goal gate) — not an envelope file |
| the persisted grower | `cortex/neurogenesis_engine.py` (`NeurogenesisSelfImprove.observe/tick/_refit`, `growth_state_bytes/restore_growth_state`, `baselines`), `hippocampus/ng_snn.py` (`NGSNN.fit`, `prune_unearned`) |
| the growth stop | `cortex/growth_stop.py` (`GrowthStop.should_grow`, reads the |LP| facade through `_LPRoute`) |
| the |LP| facade (Chain B) and the curiosity producer (Chain A) | `basal_ganglia/learning_progress.py` (`LearningProgress.lp/lp_all`), `neuromodulation/spiking_curiosity.py` (`SpikingCuriosityDrive.learning_progress`, `SpikingCurriculum`); Chain A's telemetry keys on `Brain.state()`: `curiosity_mag`, `neuromod_acetylcholine`, and `Brain._lp_mean`; Chain B: `state()["learning_progress"]["cortex"]` |
| prototype learners (ART-style) | `hippocampus/growing_learner.py` (`GrowingLearner.observe/state_dict/load_state_dict`) — seven live construction sites in five modules (`autonomous_loop`, `brain` ×3, `affect_integrator`, `lifelong_consolidation`, `similarity_sized_growth`; the audit counted eight including a comparison harness) |
| goals | `cortex/autotelic_goals.py` (achieved-outcome buffer, HER, competence-progress goal babbling) is reached ONLY from the sleep tick, persist-gated, through `cortex/nightly_skills.py::_autotelic_step`, and runs on the module's own 2-D toy grid — never on the arm; `basal_ganglia/learning_progress.GoalStore` (`_T_GOAL`) has ZERO callers; T076's `basal_ganglia/curriculum.py` does not exist yet |
| plasticity palette (local rules) | `plasticity/` (Hebbian/STDP, homeostatic, LPL spine, three-factor extensions, `rule_composer`'s production census, `inhibitory_stdp` live on the tick), `cortex/eprop_kernel.py` (surrogate-gradient batch trainer + `bptt_grads` reference — NOT a per-tick rule), online local readouts `cortex/laminar_local_readout.py::observe`, `substrate/similarity_sized_growth.py::FrozenExpert` |
| the lake | `substrate/home.py` (`save_state/load_state` `_T_STATE`, `save_growth/load_growth`, `throwaway_home`, `home()` reads `NORD_HOME`), `substrate/binlog.py` (`BinLog`), `substrate/lake_engine.py` (`SynapseDelta`, `AppendBudget` — no live caller yet), `lake_segments.py`, `lake_index.py`, `neuro_chip_stream.py` (shard discipline) |
| the substrate tick | `substrate/kernel_tick.py` (`KernelTick._deliver_and_rotate`), `substrate/kernel_state.py` (SoA, T013), dormancy-exempt `wire` today; `substrate/perf_baseline.py` (`--record/--assert`; T065's `tools/perf_baseline.py` path is stale — the p4c ratchet refused `tools/` at T001) |
| anatomy | `connectome.py`, `substrate/wiring_manifest.py` (36 ports, 19 edges declared), `substrate/signal_ring.py` (`SignalRing`, `JoinConsumer`); nine modules carry a `_*_consumer_factory` (`evidence_accumulation`, `r_stdp`, `interoceptive_relay`, `interoceptive_integration`, `region_coherence`, `astrocyte_modulation`, `percept_surprise`, `task_set`, `reentrant_workspace`) |
| perception | `cortex/eprop_nmnist.py` (frozen encoder; `models/eprop_nmnist.pt` is gitignored), `models/nmnist_pack.npz` (tracked 256 frames), `data/nmnist_cache/*_T15/*.hdf5` (local, 10k test / 60k train), `cortex/percept_surprise.py`, `substrate/afferent_boundary.py` |
| instruments and gates | `substrate/*_gate.py`, `tools/*.py`, `tools/charge_probe.py`, `substrate/brain_telemetry.py` (`PROVENANCE`), `substrate/cockpit.py` (`/api/state`, `/api/chat`, `/api/all`); `REGIONS` in `snn_data_lake/__init__.py` |
| evidence and records | `specs/022-unified-brain-arc/evidence/*.md`; panel reports, harnesses and per-arm JSON archived under the memory store's `archive/audit-2026-09-01/` |

### A.6 Vocabulary

**Charge**: a wire changes a downstream outcome (proved by a neuter A/B), as opposed to being merely
reachable. **Red**: the constructed input that makes a criterion fail; a criterion with no reachable red is
deleted. **Control**: an arm that is not the method (untrained twin, marginal-preserving shuffle, neuter,
oracle at the mechanism's input contract). **Null**: a pre-registered outcome recorded as such; a null keeps
its licence to delete. **Tenant / tag**: a record class in the one lake (`_T_GROWTH`, `_T_STATE`,
`_T_GOAL`, …). **Floor**: majority or persistence accuracy on the SAME held-out slice — an accuracy under
its persistence floor learned nothing the stream's inertia did not supply. **Chain A / Chain B**: the
curiosity producer over the arm's context vs the percept-surprise |LP| facade the growth stop reads.
**Wake rows**: `Brain.state()` sampled after a `tick()` that returned `"awake"`.

---

## PART B — WHERE THE BRAIN IS (measured; every number traces to a file or a dated poll)

| dimension | measured (HEAD after 2a/2b/2c) | source |
|---|---|---|
| grown substrate | 2.1 units per 800 ticks in-process on the brain's own stream; 31 restored on the daemon from a pre-2a blob (floors None); the admission rule admits 24 units on a CONSTANT input | s2 §v2, s2b |
| the grower's learner | a batch solve refit every 20 sleep ticks, validated on the most recent third; live accuracy 0.53 against persistence 0.93 / majority 0.44; the persistence ORACLE reaches its floor on 2 of 8 seeds | s2 §v2 |
| the stream | in-process: 4–5 arm categories per 800 ticks; curiosity (Chain A) converges to exactly 0.0 by 3000 ticks on 2 of 3 seeds (seed 1 not yet); a rotating real stream converges it on 3 of 3; the |LP| facade (Chain B) stays alive (~100 distinct values per 200 ticks) — all under `throwaway_home()`, latch-free by construction | s2c |
| the daemon | E-STOP latched since 2026-08-31 (`arm_stopped` True); s2b recorded 3,539 wake ticks with `n_modes` 2 and floors None (the refit count is not exposed; "no refit" is inferred); a live poll at drafting (2026-09-02) read 16,038 wake ticks, still latched — the one-category chain is an INFERENCE with no latch-cleared control | s2b; `/api/state` |
| anatomy | 10 growable regions ratified; 19 typed edges declared; AST faculty-to-faculty reads 1 of 19; 0 `JoinConsumer` instantiated in `make_brain` | audit §1, p6_gate |
| cognition | answers survive silencing every LIF neuron; disabling the symbolic half (DSL + dict) drops 5 of 6; access index 0.0 | audit §3 |
| the lake | records persist across a process boundary (growth blob, coverage, the three percept learners); units live in RAM; no paging; `AppendBudget` has no caller | s2b, T064 |
| placement / prior | write-only on the live path; the genomic map deleted on its own null | s1-placement-null, s1-genome-fanin-ab |
| scale economics | the sparse kernel is dormancy-exempt `wire`; CUDA vs CPU crossover figures live in `docs/POLICIES_EVIDENCE.md` and specs/008 — re-derive at S6, never cite | P5 |
| self-improvement | the self-build route fires, has kept zero self-modifications, breaker latched | audit P0 row |
| instruments | 19 gates + battery green; US4's dissociation battery 1 of 7 built; CI dark since 2026-08-20 (the local battery is the gate) | audit §10 |

Three wires were built and measured NULL on 2026-09-02, each honestly (placement: no reader; the genomic
map: no gain; the real percept stream: no structure the batch grower catches), and a fourth was refused before
a build (a rotating stream: the pack is not what saturates curiosity). **The diagnosis this plan acts on:
wiring built mechanisms to each other produces no charge while none of them learns online from a world with
structure and nothing downstream consumes what grows.**

---

## PART C — PRINCIPLES CARRIED, AND THREE ADDED

Carried: all of POLICIES (A.2). Added from what the nulls taught:

- **The world before the wire.** No growth or learning task starts until its stimulus has been shown, on
  the exact signal the task names, to carry structure its red can detect (rank 2c's in-process probe is the
  template: probe, then build; a roadmap number is a claim about an instrument — re-measure it first).
- **Synaptic first, growth on plateau.** Biology fixes its neuron count early and learns by synapses; a unit
  is expensive, a synapse is cheap. Existing units learn online on every percept; rewiring (synapse
  formation/elimination) is the middle axis; a NEW unit is admitted only after synaptic learning has stalled
  on residual error. Efficiency is measured, not assumed (SC-012a reports the gain per doubling of units).
- **Growth is counted by effect.** Four success criteria join SC-001..SC-010 (ratified under O4): SC-011,
  SC-012a, SC-012b, SC-013 (Part F). Until SC-012a has fired, the owner-facing headline and PROVENANCE say
  "units admitted", never "growth" — a reporting rule adopted under O4 WITH its counted gate
  (`tests/test_units_admitted_wording.py`). "Provisional" in the owner's ratification means TIME-BOUND —
  until SC-012a fires — not this project's gate-unbuilt sense: the gate landed in the same commit.

---

## PART D — THE PROGRAM (six phases; the order is the plan; every task carries its red)

### Phase 0 — Owner unblocks (not engineering)

- **O1** Clear the E-STOP at the cockpit (a person). What it unblocks: the LIVE daemon's stream and every
  efferent instrument (audit §8 rank 8). Not needed to START Phase 1 in-process: the latch lives under the
  lake home (`so101_arm.py::_latch_path` → `home()`), and a `throwaway_home()` brain is latch-free by
  construction — that is how rank 2c measured. **DONE 2026-09-02**: the session issued the reset under the
  owner's per-instance authorization (the route is not a standing licence for a session), through the
  cockpit's own act path (`POST /api/act` with the command `HOME`, the deliberate reset of T037a/OD-4): the
  sidecar is gone from disk and the arm moves. Recorded poll (`GET /api/state`, 2026-09-02 evening, the only
  record of these counts): `arm_stopped` False, `grown_units` 1.0 where the relaunch had restored 31.0,
  `n_modes` 3, `curiosity_mag` 0.0 — the restored substrate pruned at its first real-stream refit (the
  DIRECTION was pre-registered in s2, which measured a 24 → 8 prune on a 22-unit blob; the 31 → 1 count is
  this poll's), and the s2c curiosity floor holds with the arm free: it is the world model's, not the
  latch's, exactly as W1 assumes.
- **O2** Give the logon launcher's preflight a base interpreter outside the Claude app container (audit
  §9.1). Harnesses and the battery already run under `.venv-lake` from a plain shell; O2 fixes daemon
  relaunch after a reboot only. **DONE 2026-09-02:** the free-threaded 3.14.6 was copied to
  `D:\AI-project\python\cpython-3.14.6t` and `.venv-lake/pyvenv.cfg` re-homed to it (torch 2.11.0+cu128 with
  CUDA, mujoco 3.11.0 and the package import verified); registering the launcher at logon stays the owner's
  (a system setting).
- **O3** CI: re-enable, or record in POLICIES that the local battery is the gate. Per OD-3 the POLICIES
  text lands only after the e-prop consolidation debt is paid and in the same commit as its gate's counted
  output — the gate is H1's `policy_manifest` execution assert; verify the debt at O3's commit.
  **DECIDED 2026-09-02 by the session under the owner's authorization: the local battery is the gate while
  CI is dark.** POLICIES already says so (its 2026-08-16 amendment: the local suite is delegable to CI only
  while CI is demonstrably executing), so NO new POLICIES text is owed — "no new policy" is a complete
  result, and OD-3's e-prop-debt precondition, which governs POLICIES text additions, is not triggered. What
  H1 owes is the MECHANIZATION: an execution record (`policy_manifest --certify` opening every battery;
  `--assert` refusing a record more than one commit behind HEAD), a local execution path for the tools
  gates, and only enabled `ci.yml` lines counting as CI-reachable.
- **O4** Ratify this plan as the plan of record; ratify SC-011/SC-012a/SC-012b/SC-013; ratify the reporting
  rule in Part C. W3 (an instrument with a self-check, no reorder) may start before O4; W1 onward waits
  for O4 and O5. **RATIFIED 2026-09-02.** The reporting rule landed with its gate,
  `tests/test_units_admitted_wording.py` (red counted on the pre-change tree: 2 of 2 FAILED); the cockpit row
  now reads `units admitted`, and spec.md carries SC-011–SC-013 and OD-5.
- **O5** Ratify what this plan changes against `tasks.md`'s Dependencies line ("T072+ on T071 (US3 exit)")
  and SYNTHESIS §3's stage ordering, each its own decision:
  (a) T069 moves into Phase 2;
  (b) T070, T079–T082 (all `[P]`) and T077, T078 (not `[P]`) run in Phase 4 in parallel with Phase 3:
  RAM-scale instruments, and T081 retires a live P8 defect;
  (c) **T076's MECHANISM lands in Phase 1 as W1 with T076's own red** — the one reorder that crosses the
  T071 gate — and T076 keeps only its US4 exit row in Phase 5;
  (d) T071 stays the gate for T072–T075 and T083 — that clause is NOT rescinded;
  (e) T062/T063 move behind T064–T068 on rank 1's measurement (placement has no reader);
  (f) Phase 3's paging (S3–S5) runs at T065's constructed width WITHOUT an entry gate on a measured growth
  trajectory (the panel proposed one; not adopted): its order after Phase 2 is the plan's choice and
  T068's null clause protects it.
  Any scene or loader change for the arm is an envelope-approval event (A.3).
  **RATIFIED 2026-09-02 (a–f); tasks.md's Dependencies line carries the amendment.**
- **H1 — enforcement hygiene (audit §8 rank 10), Phase 0, three commits of one change each — the tools
  stage, the execution record, the docs sweep (gitignore anchors, `ArmSim` sites, the Gf number)**: `p4c_ratchet --assert` gets a
  local execution path; `policy_manifest` asserts execution rather than a substring (H1 owns this; O3 cites
  it); `.gitignore` anchors for `lakehouse/` and `GATES.md`; the 13 `ArmSim` sites; the stale Gf number in
  four files; `policy_certification.json` regenerated. T087's `--certify` precondition is this task.
  **DONE 2026-09-02 in three commits:** (1) the tools stage in `run_selfchecks` (red asserted by
  `tests/test_run_selfchecks_tools_stage.py`); (2) the execution record — `policy_manifest`'s enabled-step
  parsing and certification-freshness assert (reds in its `--self-check`), the certification regenerated by
  the battery's opening `--certify`, POLICIES' status note inside its 2026-08-16 amendment; (3) the docs
  sweep — `.gitignore` anchored to the root for `lakehouse/` and `GATES.md` (the tracked
  `src/snn_data_lake/lakehouse/` was matched before), `ArmSim` → `So101Arm` at every present-tense site the
  sweep could reach — the live-guidance docs (BUILD_EXE, JEPA, OVERVIEW, PITCH, RUNNING_LOCAL,
  SELF_IMPROVEMENT_RESEARCH) and the src comments and docstrings outside the envelope (autonomous_loop,
  brain, live_brain, brain_telemetry, growable); NOT renamed: the two envelope files (`serial_arm.py`,
  `so101_arm.py`: approval-gated; theirs are historical), `docs/HARDWARE_SCOPE.md` (its own correction
  banner) and the dated review records (BRAIN_REVIEW, AIRLLM_ANALYSIS, BRAINCOG_COMPARISON, IMPROVEMENTS,
  CHANGELOG) — and the Gf number re-derived at HEAD by `tools/arc_score.py` (19/400 = 4.75%, up from
  11/400 on 2026-08-03; outputs archived) and re-dated in every file that quoted it, including the
  `rpm_solver` docstring.

### Phase 1 — The world (the limiter)

Startable now, in-process. Order inside the phase: W3 (the instrument) first — gate before rule.

- **W3 — the world-red instrument.** Promote the rank 2c probes (`archive/audit-2026-09-01/rank2c/probe_chains.py`
  and `probe_fixed_vs_rotating.py`) into `src` as one instrument with a self-check, registered in `REGIONS`
  and `run_selfchecks`: per seed, mean and distinct values over the last 200 wake rows at 3000 AND 6000
  ticks of `state()["curiosity_mag"]`, `Brain._lp_mean`, `state()["neuromod_acetylcholine"]` (Chain A) and
  `state()["learning_progress"]["cortex"]` (Chain B); the arm learner's category entropy over 1000 wake
  ticks; refit count. The distinct-values liveness idiom already exists (`autonomous_loop.py`'s
  `len(set(sal_live))` and `tools/ab_connectome_delay.py`) — reuse it. **Pre-register the baselines from a
  measured run of today's world — s2c's FIXED PACK arm under `throwaway_home()`, `persist=False`, 3 seeds —
  BEFORE W1** (the record has no category-entropy baseline).
- **W1 — goals drive the arm inside the existing world (this IS T076's mechanism, pulled forward under
  O5(c)).** No new environment, no scene change: the world is `brainstem/autonomous_loop.py`'s SO-101
  under `coverage_reward` (FR-027). Build the module T076 names, `basal_ganglia/curriculum.py`, with
  `propose()/observe()` and the |LP| bandit over goal-space regions, by PORTING `cortex/autotelic_goals.py`'s
  goal sampling (competence-progress babbling over achieved outcomes, HER) from its toy grid onto the arm's
  state space (end-effector poses / joint targets), as `GoalStore`'s (`_T_GOAL`) first live caller, under
  FR-027's JOIN — never by fabricating goal objects (the `coverage_reward` docstring refuses that as
  theatre) and never by editing the scene. P2 incumbents, each dispositioned in W1's evidence: the nightly
  toy-grid babbler in `nightly_skills._autotelic_step` (stays as a sleep-time skill or retires if the port
  covers it — the A/B decides); the arm's own `_autotelic_goal_thresh` latent-goal gate (composed: it gates
  which proposals reach the actuator, or retired on the A/B); `GoalStore` (adopted). `reasoning_curriculum`
  (a transitive-inference environment + sleep loop, a live P0 knowledge avenue) and `SpikingCurriculum`
  (an |LP|-driven stage-advance gate) do DIFFERENT jobs and are not merged; no benchmark exists that could
  compare them. **W1's commit carries T076's red**: an objective with `coverage_reward` neutered must
  measurably change goal selection.
- **W2 — the real latents reach the context without breaking the width.** `_safe_ctx` pools the real
  vision/audio latents to one scalar per band. W2 delivers them as their prototype ids (the categories the
  learners already produce) and the encoder's surprise through the declared afferent lanes (FR-008) WITHOUT
  widening the 64-wide arm context — a width change would orphan the daemon's 64-wide prototypes and the
  31-unit growth blob (rank 2b refuses a record of another width; `NGSNN` refuses a widened percept). If a
  widening is ever chosen it is a SYNTHESIS §3.5 migration event (FR-030) with its own P4b restart red and a
  fresh-engine disposition, stated in advance.
- **Reds (S = 8 seeds, paired one-sided t > 1.895, permutation FP rate reported):** **W-R1** mean
  `curiosity_mag` over the last 200 wake rows at 3000 and at 6000 ticks under W1+W2 exceeds the SAME world
  with the context time-shuffled (structure destroyed, marginal kept) — treatment vs control per seed, so a
  seed whose control has not converged yet cannot pass for free. **W-R2** the arm learner's category entropy
  rises above the pre-registered W3 baseline AND live next-category accuracy beats its marginal-preserving
  shuffle at the bar, each arm's own floors reported within-arm beside it (s2 v2's idiom). **W-R3 (= T076's
  red)** goal selection changes when `coverage_reward` is neutered. **On a null:** W1/W2 revert; the finding
  is that the curiosity chain's floor is intrinsic to the world model over this body, and Phase 2's L2 is
  re-targeted at prediction error.
- **Exit:** `evidence/w-exit.md`, three tables, the pre-registered baselines beside them.

### Phase 2 — Learning in the substrate (synaptic first, growth on plateau)

L1a, L3 and the oracle are stream-independent and can start before Phase 1 finishes; SC-011 and L-R2 need
Phase 1's stream. Determinism: every A/B here runs a live-vs-live nondeterminism baseline in the same
command (the rank-1 idiom), torch threads pinned; "byte-equal" restart reds are state-key equality against
that baseline.

- **L1a — the admission window.** The grower's held-out slice covers the category marginal (stratified or
  recency-mixed) instead of the most recent third. Control: the incumbent window with the same learner.
  Red (the window half of s2's pre-written red, now at the plan's statistic): paired t over 8 seeds of
  (oracle accuracy − its own persistence floor) > 1.895, AND paired t of (LIVE under L1a − LIVE under the
  incumbent window) not below −1.895. One commit.
- **L1b — synapses learn online.** The grown units' INPUT weights learn on every wake percept by a local
  rule from the palette (Oja/BCM/competitive-Hebbian with a three-factor gate from the live neuromodulators
  — the candidates are named at pre-registration and picked by the A/B), with the GRADIENT_CLAIM row
  (FR-018), `rule_composer`'s census and the e-prop transport gate. `cortex/eprop_kernel.py` is a batch
  surrogate trainer and is NOT the rule; it stays the cosine reference. Control: the batch solve under L1a's
  window — kept ONLY as a declared control off the live tick (the `gf.py::MLP.train` precedent: its own
  GRADIENT_CLAIM row and a test that it is unreachable from the tick), else deleted. Red: L1b beats the
  control on live accuracy at the bar; an untrained twin (frozen random readout) sits BELOW its own
  persistence floor while L1b does not.
- **L1c — the readout learns online.** Adopt an existing online local readout rule
  (`cortex/laminar_local_readout.observe`'s single-sample local update, or `similarity_sized_growth.FrozenExpert`'s
  three-factor rule — P2: one survives, named at pre-registration) with its state as a lake record.
  Estimated L1a–c: 400–700 LOC over three commits.
- **L2 — grow on plateau, prune in sleep (FR-023, FR-024).** A new NG-SNN unit is admitted only when
  synaptic learning (L1b) has stalled on residual error over a window; consolidation and earn-your-place
  pruning stay in sleep; the T047 gate consulted as today, fed by Chain B (alive) as it is. **P2 adjudication,
  recorded:** the `GrowingLearner` vigilance recruit (seven live sites) recruits PROTOTYPE vectors and its
  ids are the grower's categories; NG-SNN admits LIF units; they are composed, not duplicated, and no second
  LIF grower is built. Red: every growth event is preceded by a measured synaptic plateau; neutering L1b
  raises the growth rate (attribution). Control: the grower with L1b off.
- **L3 — the admission instrument.** `prune_unearned` keeps ONE of an all-redundant set; `NGSNN.fit` refuses
  a zero-variance input. Red: the constant-percept arm (24 units in s2) admits 1. Positive control in the
  same command: a structured stream under the same instrument still admits more than 1 (s2 v2's LIVE 2.1 /
  SYNTH 13.9 are the reference), so a fit that refuses everything cannot pass.
- **L4 — persist what makes a relaunch resume instead of replay, within a ceiling.** The perception
  pointers and RNG stream positions (small; they ARE the replay problem 2b's panel measured), the affect SOM,
  the lifelong learner, the arm's Q-table — the same `_T_STATE` family, each with the cross-process red 2b
  built. NOT the engine's rolling buffers (a night every ~25 daemon ticks would append ~1 MB per night to a
  2.2 GB append-only lake). L4 gives `AppendBudget` its first live caller: a bytes/tick ceiling on this
  record class with a constructed-overload red (FR-003), derived from the measured night cadence.
- **T069 (from US3, FR-019)** — the cerebellar rule ON the built tissue (its tissue T054 is closed) — the
  first learning rule with a red against a plant; lands here.
- **SC-011 (new test; extends `tests/test_real_percept_stream.py`, the in-tree promotion target of the s2
  8-arm harness):** a stream with learnable temporal structure grows more units than its OWN
  marginal-preserving shuffle (same entropy, no structure), paired t at the bar; the shuffle is the null; the
  constant stream is L3's red, not this control.
- **SC-012a (new test, same file):** admission CAPPED exogenously at N ∈ {1, 2, 4, 8, 16}, S = 8, held-out
  next-category accuracy under L1a's window; paired t of acc(16) − acc(1) > 1.895 AND acc(16) above its own
  majority floor at the bar; null: the same grid with labels shuffled marginal-preserving must not rise.
  Reported beside it, not a bar: the marginal gain per doubling, acc(2N) − acc(N) at each N (the efficiency
  signal Part C names).
- **Exit:** `evidence/l-exit.md`.

### Phase 3 — A substrate that scales (lake-resident, sparse), then the prior — US3 re-ordered

The width these tasks run at is CONSTRUCTED (T065), not grown; until Phase 2's growth projects to it, this
is capacity ahead of consumption, disclosed (O5(f)) — which is why Phase 3 follows Phase 2 and its null
clause protects it.

- **S1 = T064** unit blocks + block-aligned records (`substrate/lake_engine.py`, new `substrate/unit_blocks.py`),
  the first live writer of `SynapseDelta` (rewiring becomes lake-resident before unit growth does); port
  `neuro_chip_stream`'s one-core-resident shard discipline (P2).
- **S2 = T065** the binding width from measured state-per-unit, recorded by the EXISTING
  `substrate/perf_baseline.py --record/--assert` (T065's `tools/perf_baseline.py` wording is stale: the p4c
  ratchet refused `tools/` at T001); 16k stays the ingest milestone only.
- **S3 = T066** activation-driven paging, BOTH reds built before the mechanism (FR-010/FR-011: no silent
  drop of a spike to a paged-out unit; the integer rounds budget trips on constructed overload). **S4 = T067**
  growth delta tier + background merge in sleep (its locality reorder uses block adjacency until the prior
  lands). **S5 = T068 (SC-007)** the paging null clause at the derived comparison width vs forced-total
  residency — WIN keep, LOSS delete the layer; gated on T061 (closed) only.
- **S6 — wire the sparse kernel onto the live tick.** `KernelTick` is dormancy-exempt `wire` today; S6 makes
  the flat (post, weight) scatter in `KernelTick._deliver_and_rotate` the live tick's delivery (FR-035 is
  already met by `substrate/kernel_state.py`, T013, SoA), specs/008 gets a `tasks.md`, then the
  resident-CUDA arm gated on `substrate/device.py::select` with a new workload name (`"substrate_tick"`,
  introduced by S6) at the crossover re-derived on this host. Red: tick time at the measured width; a neuter
  of the kernel changes the tick's output.
- **Then the prior: T062** — the H01 ingest is the c3 SYNAPSE GRAPH only, composed onto
  `cortex/cortical_prior.py` (which already holds the soma/layer/sign constants; the audit's "what the spec
  gets wrong" item (5)) — the join gate, unit gate, layer computation, CC-BY provenance; and **T063** the
  three-arm drift harness (SC-006, both reds). Moved behind S1–S5 on rank 1's measurement: the prior earns a
  reader when growth places units at scale and paging needs a locality order.
- **Exit: T071** (US3 exit): prior separation + paging verdict + **the region charge gate at
  floor(all-but-exempt)** (FR-015, kept).

### Phase 4 — Cognition on the substrate (so growth matters) — ranks 3, 5, 6 + US4's instruments

May run in parallel with Phase 3 (O5(b)); MUST precede Phase 5.

- **C0 — promote the 20-item everyday battery** (it exists only in session scratchpads) into `src` as a
  scorer instrument with a self-check and PROVENANCE; without it C-R2 is unreachable.
- **C1a — instantiate 17 of the 19 declared edges** in `make_brain()` as `(SignalRing, JoinConsumer)` pairs.
  Nine modules already carry a factory (A.5) covering nine edges; the remaining direct-call edges (about ten
  — re-count from `wiring_manifest.rebuild_from_registrants()` at C1a's pre-registration) get new ones;
  building a factory where one exists is the P2 defect A.4 step 1 warns about. Red: live consumer count ==
  17 AND a pinned-payload neuter on any edge changes its consumer's output (FR-014); SC-001 leaves 1.
- **C1b — the benchmark P2 names for the two workspaces**: ambiguous multi-source inputs with a known
  correct winner, scored on which workspace's ignited content matches. This is a build the standing
  adjudication requires before either workspace can be retired ("until that exists, both stay").
- **C1c — resolve the two workspace edges on C1b's result** (FR-016; T030's declared composite is the
  incumbent), then **C3 = rank 6**: G1 and G2 as pure measurement on the built `EpisodeDriver` through the
  T032 actuator over S seeds; replace `charge_probe`'s undirected ignition arm with a non-degenerate neuter.
- **C2 = rank 5 — the answer path on the substrate.** `Brain.respond` routes on the `{known, coverage}`
  the lake already returns and on the grown readout; `reason_out` (the fixed self-report that pre-empts
  groundedness) is the NULL CANDIDATE — content-gated or deleted on its null (P8 clause 5); the DSL half of
  the answer path is KEPT as the standing positive control (the audit measured it carrying 5 of 6 correct
  answers). **C-R2**: the 20-item battery moves from 5 unrelated / 1 self-report / 14 abstain to 0 / 0 with
  abstains rising AND the 4-item positive control stays 4/4. **C-R3 (= SC-012b)**: correct answers depend
  on the substrate — they DROP when the grown units are silenced (control: silencing an equal number of
  pre-sized units must not drop them) AND SURVIVE with the DSL neutered. **SC-013 (new)**: a falsifiable
  reasoning criterion — ARC-eval at the recorded protocol plus the everyday battery — numbered, reported with
  its floor, and ratified under O4.
- **C4 — the dissociation instruments (SC-009's first half)**: T070 (integration measure, never Phi), T077
  (metacognition as an extension of the incumbents), T078 (attention schema), T079 (agency; needs O1),
  T080 (temporal binding), T081 (report-order probe; `reasoning/consciousness_probe.py` RETIRED in the same
  commit — its criteria are markers, not dissociations), T082 (social cognition composed onto the bus).
  Each fires its constructed red once.
- **SC-004's open half gets its owner here:** the multi-region clause (1 of 10 regions grows) is
  re-scoped to "a second region receives grown units that a live path READS" and is owned by C1a (readers
  exist once ports are instantiated) together with L2's growth through the 007 seam — the seam rank 1 built
  and reverted for want of a reader.
- **Exit:** `evidence/c-exit.md`.

### Phase 5 — Self-improvement (US4 core; SC-008) — after T071 AND Phase 4

- **T072** proposer un-latched (rank 7 incl. its `note_cycle` clause: a futile set counts as exhausted for
  its source); red: `distinct candidate keys reachable` ≤ 1 (red today); positive control: two hand-named
  operators yield 2 keys. **T073** evaluator cascade on harness-only evidence, stage 2 = Phase 4's
  benchmark (why this phase waits). **T074** archive + MAP-Elites cells (adopted, disclosed — not a P8
  invention) as an IMMORTAL record class with its bytes/tick ceiling and P4b red. **T075** cockpit rows.
  **T076** has only its exit row left: its mechanism and red landed as W1. **T083** US4 exit.

### Phase 6 — Polish — T084–T087

T084 cockpit sweep (+ SC-011/012/013 rows with PROVENANCE); T085 per-tissue scale derivation; T086
consistency sweep, which also owns the audit's "gets wrong" items (4) and (6) — the supersession record for
specs 007/011/014 and the tasks.md row count; T087 full run-through, final battery, `policy_manifest
--certify` (after H1), close-out. FR-031: every P8 invention this plan lands has its register row (W3, L1b,
L2, T066, T073, T077, T078).

---

## PART E — SEQUENCING, DEPENDENCIES, WHAT IS DROPPED

**Critical path (commits), with the earliest start of each parallel arm marked:**
W3 → W1 (+T076's red) → W2 (no widening) → L1b → L1c → L2 → L4 (with its ceiling) → T069 → S1 → S2 → S3 →
S4 → S5 → S6 → T062 → T063 → T071 → T072 → T073 → T074 → T075/T076 → T083 → H1 → T084–T087.
Parallel arms: L1a ∥ L3 ∥ the oracle may start at W3 (they do not wait for Phase 1); the C-chain
(C0 → C1a → C1b → C1c → C3 → C2 → C4) may start at S1 (O5(b)) and must finish before T072.

- Phase 1 is startable now in-process (the latch is per home); O1 unblocks the LIVE brain's measurement.
- Phase 3's paging runs at a constructed width and does not wait for Phase 2's growth; its ORDER after
  Phase 2 is the plan's choice (O5(f)), not a hard gate.
- Phase 5 waits for BOTH T071 and Phase 4.

| dropped / refused / deferred | disposition | measurement |
|---|---|---|
| the synthetic 5-mode chain as the grower's stimulus | RETIRED to the self-check | 0.7 vs 0.2 was an experiment about growth; the organism's stream measures 0.53 against a 0.93 floor |
| `cortex/genomic_bottleneck` (FR-033 map clause) | DELETED | NULL at 24 seeds, LOSS on a carried continuation |
| the growth→placement wire (rank 1) | REVERTED, pinned null | 0 differing keys beyond a 1-key baseline |
| a rotating real N-MNIST stream (rank 2c, tasks.md row A2c) | REFUSED before a build; the row closes as REFUSED ON MEASUREMENT in the s2c commit | the named facade is alive; the saturating chain does not move under rotation, 3 seeds |
| the "load-bearing" claim of the prototype record (2b) | WITHDRAWN as an accuracy claim | the attribution's ties were vacuous (both children replay one stream) |
| a new environment / a merged curriculum (first draft of W1) | REFUSED | the world is a live incumbent; the goal sampler is ported, not re-invented; the two curricula do different jobs |
| H01 ingest as the next task | DEFERRED behind S1–S5; scoped to the c3 synapse graph | placement has no reader; `cortical_prior` holds the constants |
| `reasoning/consciousness_probe.py` | RETIRED in C4 (T081) | its criteria are markers, not dissociations |
| any unit count reported as growth before SC-012a fires | REFUSED in headlines and PROVENANCE | P0's honesty guard (reporting rule time-bound until SC-012a fires, gated by `tests/test_units_admitted_wording.py`, O4) |

---

## PART F — THE REDS REGISTER (pre-registered here; each lands with its instrument)

Every seed-paired red: S = 8, paired one-sided t > 1.895 (7 d.f., α = 0.05), ties counted, permutation
false-positive rate printed beside the verdict; no statistic normalized by a quantity a control destroys;
no majority-of-seeds rule anywhere.

| id | red (what makes it fail) | control (not the method) | lives in |
|---|---|---|---|
| W-R1 | mean `curiosity_mag` (last 200 wake rows, 3000 and 6000 ticks) under W1+W2 does not exceed the time-shuffled-context world, per seed | the same world, context order shuffled | W3 |
| W-R2 | category entropy not above the W3 baseline, or live next-category accuracy not above its marginal-preserving shuffle (floors reported within-arm) | the shuffle | W3 |
| W-R3 (T076) | goal selection unchanged with `coverage_reward` neutered | the un-neutered run | W1 |
| L-R1 | paired t of (oracle − own persistence floor) not above 1.895 under L1a's window, or LIVE falls below the incumbent window past the bar | the incumbent window, same learner | L1a |
| L-R2 | L1b does not beat the batch control, or an untrained twin is not below its own persistence floor while L1b is | batch solve under L1a's window; frozen-readout twin | L1b |
| L-R3 | a growth event without a preceding synaptic plateau; neutering L1b does not raise growth | the grower with L1b off | L2 |
| L-R4 | the constant-percept arm admits more than 1 unit, or the structured positive control admits only 1 | a structured stream under the same instrument (must admit > 1) | L3 |
| L-R5 | the L4 record class exceeds its bytes/tick ceiling without tripping | constructed overload | L4 |
| SC-011 | structured stream grows no more units than its own marginal-preserving shuffle | the shuffle | `tests/test_real_percept_stream.py` |
| SC-012a | acc(N=16) − acc(N=1) not above the bar, or acc(16) not above its majority floor, or the label-shuffled grid rises | the shuffled grid | `tests/test_real_percept_stream.py` |
| SC-012b | correct answers do not drop when grown units are silenced, or drop equally when pre-sized units are | pre-sized-unit silencing; DSL neutered | C2 |
| SC-013 | the reasoning criterion (ARC-eval at protocol + everyday battery, reported with its floor) cannot fail on a constructed input | scrambled protocol | C2 |
| S-R1/R2 | rounds budget never trips on constructed overload; a paged-out spike is dropped silently | constructed overload; a resident twin | T066 |
| S-R3 | paging loses to forced residency at the comparison width | forced residency | T068 |
| S-R4 | the wired kernel does not change the tick's output when neutered, or tick time does not fall | the object-tick twin | S6 |
| C-R1 | a pinned-payload neuter on an instantiated edge changes nothing downstream | the un-neutered run | C1a |
| C-R2 | the 20-item battery does not move, or the 4-item positive control drops | the positive control | C2 |
| U-R1 | `distinct candidate keys reachable` ≤ 1 | two hand-named operators → 2 keys | T072 |
| U-R2 | a fabricated candidate score changes the verdict | harness-only evidence | T073 |

---

## PART G — EFFORT, HONESTLY

Measured this session: ≈ 2 hours per landing (panel + fold + a 60–70-minute battery, plus a re-run when a panel
finds a BLOCKER — two of five landings needed one), before any build time. The phases are roughly: Phase 1
five commits (an instrument, a goal-driven world with an A/B, a context change with a migration
disposition, an exit); Phase 2 eight to nine (L1a–c, L2, L3, L4, T069, two new tests); Phase 3 nine to
eleven; Phase 4 nine to eleven (C0, C1a–c, C3, C2, seven instruments, some parallel); Phase 5 six; Phase 6
four. About 40 landings in perhaps 15–20 arcs under the batched-battery rule: ≈ 30–40 session-hours of panel and
battery before build time, and every phase can end in a recorded null that re-sequences what follows.
That is the method working.

## PART H — OWNER DECISIONS THIS PLAN NEEDED (all given 2026-09-02; one C2 question open)

**All five were given in chat on 2026-09-02** ("You are authorized to: O1 … O2 … O3 … O4 … O5 …"): O1–O3 are
authorized ACTIONS (O1 and O2 executed the same day; O3 chosen by the session under the authorization and
mechanized by H1), O4–O5 are RATIFICATIONS. OD-2 stays a standing runtime gate. Still open for C2: whether `reason_out` may
be deleted on its null (P8 says yes; the answer path is user-facing, so it will be asked when C2 lands). Nothing in this plan touches the
safety envelope; W1 runs inside the existing scene.

## PART I — HOW TO RESUME THIS PLAN IN A NEW SESSION

1. Launch from `D:\AI-project`; read the memory START-HERE file (A.3 gives the store's path) and re-arm all six
   reasoning skills it names (`j-space`, `karpathy-guidelines`, `thinking-pre-mortem`, `thinking-red-team`,
   `thinking-steel-manning`, `thinking-map-territory` — owner 2026-09-03, capability over reload cost); read
   `docs/POLICIES.md`'s binding sections.
2. Read this file (`specs/022-unified-brain-arc/PLAN-FINAL-2026-09-02.md`). Find the first task whose exit
   artifact (`evidence/<phase>-exit.md` or the task's evidence file) does not exist on `main`; today that is
   W3. O4 and O5 are ratified (2026-09-02), so W1 onward needs no further RATIFICATION gate; every task
   keeps its own reds, a scene or loader change stays an envelope-approval event, and C2's `reason_out`
   deletion is asked when C2 lands.
3. Read that phase's incumbents in A.5 and its reds in Part F; run `graph-exists.js` on the capability
   before writing code.
4. Pre-register the task's protocol in its evidence file; probe the premise on the named signal first.
5. Build → measure → panel (three lenses, the session's model) → fold → battery (A.4 step 5's command) →
   commit → push →
   daemon poll → close-out.
6. If a task measures a NULL, record it, apply its pre-registered disposition, and continue; do not defend it.
