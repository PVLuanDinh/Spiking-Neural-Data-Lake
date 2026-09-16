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
- **FABLE 5.1 ORCHESTRATES, OPUS 5 EXECUTES** (owner rule 2026-09-08, superseding the 2026-09-02 inherit
  rule). Orchestration is the main session and the workflow script it authors -- routing, planning, folding a
  panel, and the final verification that never delegates; the OWNER sets that model, since Claude cannot flip
  its own main model mid-session. Execution is every spawned worker: `Agent(...)` and every Workflow
  `agent(...)` pass **`model: 'opus'`** explicitly, panel lenses and verifiers included. Omitting `model:` is
  no longer correct -- it inherits, which is the rule this replaces.
- **One change per commit; one full battery per pushed ARC, on its stacked head** (owner rule 2026-09-02,
  POLICIES' "Gated + reversible" amendment). Every commit names the battery that verified the head it was
  pushed under; a red on the head is attributed per stage per commit, never by whole re-runs.
- **Commit messages carry NO attribution trailer** (owner, 2026-09-03, superseding the session-model
  trailer rule this line used to state). `attribution: {commit: "", pr: ""}` in `~/.claude/settings.json`
  enforces it mechanically rather than by remembering; the rule is kept written down here because the
  setting is machine-local and this file is the tree's copy of the protocol.
- **Suggested-fix chips and open tasks are a COMPLETION GATE** (owner, 2026-09-03, strengthening the
  2026-08-31 rule that made them pre-authorized). **This bullet is the authoritative text of the
  rule**; the curated memory file `feedback-chip-fixes-preauthorized` points here and carries only
  machine-local gloss. Chips are fixed AUTONOMOUSLY, and **no arc, task or session is declared done
  while a chip or an open task remains.** Filing a chip is deferral, not closure — the session that
  raises one owns it. Two honest-sounding reasons had been ending arcs with chips open ("out of this
  arc's one-change-per-commit scope" and "a `src/` change owes its own panel"); both were true and
  neither made the work done.
  - **Fix them under the full rhythm, not around it** — a chip touching `src/` still owes its own arc,
    so the gate is that the queue empties, not that the rhythm is skipped to empty it.
  - **Before saying done, ENUMERATE** open chips, background tasks without a terminal verdict, and
    anything the session itself deferred; report that list or report it empty. A panel fold stated in
    its commit as battery-verified only is CLOSED by that disclosure per A.4 step 4, not open — that
    is the one deferral this document already grants and this rule does not withdraw it.
  - **The rule must never reduce the number of chips FILED.** An unfiled defect is strictly worse than
    an open one: a dismissal at least leaves a reason, and an open chip is visible. If a defect is
    found and the queue cannot absorb it this arc, FILE IT and escalate the queue to the owner — do
    not decline to notice. Making an empty queue a completion condition pays a session to look away,
    and that is the failure this clause exists to price.
  - **Dismiss a chip only when it is genuinely closed** (fixed, or measured moot), with the reason,
    and cite the chip id at the fix site. The only clean exit that is not a fix is an explicit owner
    deferral — ask, do not assume scope.
  - **UNMECHANIZED, and the debt is named rather than left implicit.** The durable record is
    START-HERE's `Open chips:` line plus the arc's route-ledger entry: a session reporting the queue
    empty writes that line empty in the same act. The gate this owes, when built, reads those two
    records against the chip system and reds on disagreement — and per the cross-cutting invariant it
    is built and its output counted BEFORE it is promoted to a rule.
  - The A.3 carve-outs above are unaffected — the five `brainstem/` envelope files and the E-STOP.
    **Extended here:** anything irreversible outside the repo (a force-push, a published artifact, an
    external deletion) is likewise raised, never auto-landed.
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
   cd /d/AI-project/spiking-neural-data-lake; export PYTHONUTF8=1 PYTHONPATH=src; B="<output dir>"; mkdir -p "$B"; PY=.venv-lake/Scripts/python.exe; git fetch -q origin; git status --short > "$B/tree.txt"; git diff HEAD --stat >> "$B/tree.txt"; $PY -m snn_data_lake.substrate.policy_manifest --assert > "$B/record_assert.txt" 2>&1 || { echo "RECORD-ASSERT RED: the previous push has no battery record -- battery halted"; exit 1; }; $PY -m snn_data_lake.substrate.policy_manifest --certify > "$B/certify.txt" 2>&1; echo "certify rc=$?"; $PY -m ruff check . > "$B/ruff.txt" 2>&1; echo "ruff rc=$?"; $PY -m ruff format --check . > "$B/ruff_format.txt" 2>&1; echo "ruff-format rc=$?"; $PY run_selfchecks.py > "$B/selfchecks.txt" 2>&1; echo "selfchecks rc=$?"; tail -n 3 "$B/selfchecks.txt"; $PY -m pytest -q -p no:cacheprovider > "$B/pytest.txt" 2>&1; echo "pytest rc=$?"; tail -n 3 "$B/pytest.txt"; for m in dormancy_gate method_dormancy_gate region_charge_gate growth_coverage_gate manifest_gate determinism_gate replay_gate envelope_hash_gate policy_manifest gradient_claim_gate eprop_transport_gate import_manifest replication_guard capability_bar immortal_retention conclusion_protocol; do $PY -m snn_data_lake.substrate.$m --assert > "$B/$m.txt" 2>&1; echo "$m rc=$?"; done; echo BATTERY_DONE
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
| the grower's learner | a batch solve refit every 20 sleep ticks, validated on the most recent third; live accuracy 0.53 against persistence 0.93 / majority 0.44; the persistence ORACLE reaches its floor on 2 of 8 seeds — *superseded 2026-09-06: L1a's marginal-covering window; L1c's online delta readout beside the batch solve (+0.14 live accuracy in the cells, dormant on the daemon's silent unit), `evidence/l1c-online-readout.md`* | s2 §v2 |
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
  *DECIDED 2026-09-08 (the session's call under the owner's "close any openings" authorization; a separate
  docs commit as this plan sequenced it): SC-012a FIRED on 2026-09-07 on a QUALIFIED reading — held-out
  accuracy rises with admitted units under the capped grid, but forced width does not beat the uncapped
  incumbent and nothing reaches the persistence floor (its own DISPOSITION above). The time-bound
  condition is therefore met and the wording is NOT lifted: a count may be labelled "growth" only when
  that count is shown to change an outcome the brain is measured on (P0's honesty guard, P3's "growth
  is not improvement"), and SC-012a's reading establishes accuracy-with-count, not improvement-over-the-
  incumbent. The rule is re-based from time-bound to that permanent condition; the gate is unchanged in
  mechanism and its docstring states the condition. The wording lifts the day a row shows the
  improvement — and that row names the gate it retires. THIS DEVIATES FROM A PRE-REGISTERED
  CONSEQUENCE and says so: `evidence/sc-012a-width.md` §2.3 registered "if it fires: the Part C
  reporting rule lifts". The fire's veto conjunct (c) turned out to have NO REACHABLE RED
  (`evidence/l-exit.md`), so the fire rests on (a) and (b) alone and its pre-registered consequence
  rests on a veto that could not veto; the lift is HELD on that null rather than taken on a fire whose
  own record impugns it, until (c) has a reachable red or its replacement null lands with its gate
  first — and it then lifts only under the permanent condition above. Not a relabelling of a met
  condition as unmet: (a) and (b) are MET, the fire STANDS, the consequence is what is held.*

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
  *DATED NOTE 2026-09-10: discharged, not rescinded -- T071 CERTIFIED (evidence/s2-exit.md; Part E's note of the same date).*
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
- **W1 — RETIRED 2026-09-04 BY OWNER DECISION ON ITS OWN MEASUREMENT (P8 deletion-on-a-null).**
  `basal_ganglia/curriculum.py` was built, measured, panelled, folded, landed at `ec5e9e4`/`dadc9b6`,
  and is now DELETED. The reason is not that it failed to work — its 9 wiring criteria were green and
  14/14 mutants red — but that the capability it existed to provide was never demonstrated:
  (a) **the actuation half was built and refused on its own data.** A bias toward `propose()`'s cell,
  pushed through the arm's existing `_ext_drive` port and scored on the objective it would serve
  (distinct cells covered, 400 ticks × 8 seeds): goal vs incumbent paired **t −0.34**, vs a
  magnitude-preserving shuffled control **t −0.52**, and a FRONTIER arm aimed at the nearest UNREACHED
  cell **t −0.50** — none resolvable against 1.895, with the incumbent alone spanning 50–107 cells
  across seeds and a 10% effect needing ~96 seeds.
  (b) **Part F's unrelated-scalar control came back RED**: substituting any unrelated live scalar for
  the objective changes 0.863–0.906 of proposals against a SHUFFLE BOUND of 0.879 — no measurable
  objective-specificity.
  (c) **a structural defect in the goal space**: `seen()` proposed only cells ALREADY REACHED while the
  objective pays for reaching somewhere NEW, and the FRONTIER arm shows repairing that does not rescue
  the wire.
  The full record survives in `evidence/w1-goal-curriculum.md`, which is deliberately NOT deleted:
  P8 calls deletion on a null the licence working, and the evidence is what stops this being rebuilt.
  **T076's mechanism is therefore WITHDRAWN, not delivered** — see its row in Phase 5 (Part D) and
  `tasks.md`'s T076 row.
  The superseded specification follows, kept so the deletion can be audited against what was asked.

- ~~**W1 — goals drive the arm inside the existing world (this IS T076's mechanism, pulled forward under
  O5(c)).**~~ No new environment, no scene change: the world is `brainstem/autonomous_loop.py`'s SO-101
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
- **W2 — REFUSED ON MEASUREMENT 2026-09-05, BEFORE A BUILD (the id lane; the surprise half entered no
  probe cell then — Q2 ran its own probe the same day and CLOSED BY MEASUREMENT for its delivery, G-FRAG: `evidence/q2-surprise-context.md`;
  cell and is Q2; the rank-2c outcome; record: `evidence/w2-context-latents.md`).** The premise was probed in-process with no tree change (nine cells:
  incumbent / W2 / W2 with the context's succession destroyed per cycle, seeds 0–2, 3000 ticks) against a
  pre-registered decision rule, and failed it on the direction pinned in advance:
  (a) **the arm's world model reads no succession** — `JepaWorldModel`'s latent COLLAPSES on every
  context this body sees (trained/fresh target-latent spread ×0.005–0.11 on nine of nine models; its
  live-model `latent_std > 0.0` self-check bars clear on all of them; the module's own 0.05 floor, which
  never sees the live context, would fail all of them) and its prediction error is the same to the
  fourth decimal under the natural order and under a per-cycle permutation — on the W2 context (three
  seeds) and, in the fold, on the incumbent context (three more; record §4.3b); the tracked packs carry no
  first-order succession either (ordered next-label MI at the median of 200 shuffles). W-R1 as written
  is therefore unreachable on this chain and this world.
  (b) **curiosity under W2 moves — 0.09–0.13 on seeds 0 and 2 against an incumbent reading 0.0000 at
  3000, and 0.0000 on seed 1 against the incumbent's 0.0200 — but not above its own time-shuffled
  control on 3 of 3** (2 of 3, the pinned direction reversed on those two): with the world model blind to
  succession, recruitment of new arm prototypes is sufficient for a reading and its absence returns the
  gauge to its 4-dp floor (seed 1: `raw_cur` exactly 0 over the last 200 rows), though recruitment does
  not order the six cells.
  (c) **the id lane fragments the arm's alphabet** (51–102 units against 5–7) and the grower's next-category
  accuracy falls to 1.3–1.7 %, below its own persistence and majority floors on every seed — W-R2's
  entropy half passes by recruitment alone and its accuracy half nulls.
  Nothing was built. Two incumbent defects are FILED as their own arcs (the JEPA collapse with its red
  already specified; `_cur_mag`'s held gauge). **The null sentence below is CORRECTED by measurement**, and
  two owner questions are open (Part H). The superseded specification follows for audit.
- ~~**W2 — the real latents reach the context without breaking the width.**~~ `_safe_ctx` pools the real
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
  re-targeted at prediction error. **CORRECTED BY MEASUREMENT 2026-09-05 (W2 record §4.6):** the floor is
  intrinsic to the world MODEL — a latent that collapses on every context this body sees — and to a world
  with no first-order succession, not to the body; the prediction error this sentence names is the
  collapsed one, so the L2 re-target was HELD pending Q1 (Part H). **Q1 RATIFIED (a) by the owner
  2026-09-05:** the JEPA-collapse arc runs first, and no task whose red reads
  `JepaWorldModel.prediction_error` starts before that arc is green.
- **Exit:** `evidence/w-exit.md`, three tables, the pre-registered baselines beside them.

### Phase 2 — Learning in the substrate (synaptic first, growth on plateau)

L1a, L3 and the oracle are stream-independent and can start before Phase 1 finishes; SC-011 and L-R2 need
Phase 1's stream. **Recorded 2026-09-05 (Phase 1 exit):** Phase 1 delivered NO stream change, and the
tracked packs were measured to carry no first-order succession (W2 record §3.1c) — so SC-011's positive
arm ("a stream with learnable temporal structure") has no measured stimulus on today's world and, per Part
C, must show one before it runs; L-R2's red needs category structure, not succession, and runs on the
incumbent stream. Determinism: every A/B here runs a live-vs-live nondeterminism baseline in the same
command (the rank-1 idiom), torch threads pinned; "byte-equal" restart reds are state-key equality against
that baseline.

- **L1a — the admission window.** *LANDED 2026-09-05 (`3979001`; `evidence/l-parallel-arms.md`): one marginal-covering
  split for all three grower windows; the oracle went from answering the majority (5 of 8) to answering persistence
  (within 0.067 of its floor on 8 of 8; deficit −0.43 → −0.02, not above — the plan's strict red NOT MET, kept as a
  strict xfail) and beats LIVE on 8 of 8; LIVE's raw held-out gain (t 2.27) is confounded with the slice the window
  redefined (learner-only on the same rows: does not clear, t 1.00); landed on the oracle's reading, the session's
  call, owner may revert.* The grower's held-out slice covers the category marginal (stratified or
  recency-mixed) instead of the most recent third. Control: the incumbent window with the same learner.
  Red (the window half of s2's pre-written red, now at the plan's statistic): paired t over 8 seeds of
  (oracle accuracy − its own persistence floor) > 1.895, AND paired t of (LIVE under L1a − LIVE under the
  incumbent window) not below −1.895. One commit.
- **L1b — synapses learn online.** *MEASURED AND DELETED ON ITS NULL 2026-09-05 (`evidence/l1b-synaptic-online.md`,
  P8 clause 5): a norm-preserving Hebbian (Oja-in-direction) and a BCM step on the grown units' input rows, 8 seeds × 800
  ticks on the same rows — neither beat the frozen rows (t -1.96 / -1.36), nor cleared the majority floor, nor (the
  letter) rose above persistence (8 of 8 below); Oja FELL past the bar — a measured harm whose mechanism is the input
  rows moving under a readout the closed growth gate holds stale (constant-class rows; the incumbent shows the same
  mode), NOT a measured loss of feature diversity (unmeasured, retracted); the ACh gate was withdrawn at Stage 0
  (decays to 0.03 where the rule acts). The rule, its hook and the gate wire are gone (P3); the LIVE instrument stays.
  What it points at: the readout must co-adapt first — L1c precedes any second input-weight rule.* The grown units' INPUT weights learn on every wake percept by a local
  rule from the palette (Oja/BCM/competitive-Hebbian with a three-factor gate from the live neuromodulators
  — the candidates are named at pre-registration and picked by the A/B), with the GRADIENT_CLAIM row
  (FR-018), `rule_composer`'s census and the e-prop transport gate. `cortex/eprop_kernel.py` is a batch
  surrogate trainer and is NOT the rule; it stays the cosine reference. Control: the batch solve under L1a's
  window — kept ONLY as a declared control off the live tick (the `gf.py::MLP.train` precedent: its own
  GRADIENT_CLAIM row and a test that it is unreachable from the tick), else deleted. Red: L1b beats the
  control on live accuracy at the bar; an untrained twin (frozen random readout) sits BELOW its own
  persistence floor while L1b does not.
- **L1c — the readout learns online.** *LANDED 2026-09-06 (`evidence/l1c-online-readout.md`): the delta rule — laminar’s L5 delta rule ported onto NGSNN.beta — learns on every wake percept
  from the pair the live instrument just scored; live accuracy over the stale incumbent +0.143 (t 4.13, p 0.0156, 8 seeds, same rows),
  still below the live persistence floor on every seed; the growth gate's evidence is the DECOUPLED batch instrument (a closed-gate
  refit scores the last open-gate solve's beta -- unchanged from the incumbent by construction, verified bit for bit), because the
  online readout is recency-shaped and read LOWER on the gate's marginal slice on every closed-gate seed; the instrument is
  persisted in the growth record (the cells' panel found the first landing re-coupled it on every relaunch). CHARGED in the
  cells (2-7 units), DORMANT on the deployed brain (one silent unit: charge 0.0 over 929 percepts, live 0.0 vs persistence
  0.997) until a unit fires on the live stream -- the open P3 item handed to L2 / SC-012a. A first refusal on an ungated
  brain-free control was withdrawn after its panel; the ridge candidate did not land by its pre-registered control's letter
  (near-vacuous on a one-class window) and lost the P2 wash; the perceptron lost the tiebreak on t (by mean it read ahead).
  The session's call under the authorization.* Adopt an existing online local readout rule
  (`cortex/laminar_local_readout.observe`'s single-sample local update, or `similarity_sized_growth.FrozenExpert`'s
  three-factor rule — P2: one survives, named at pre-registration) with its state as a lake record.
  Estimated L1a–c: 400–700 LOC over three commits.
- **L2 — grow on plateau, prune in sleep (FR-023, FR-024).** *BUILT, MEASURED and DELETED ON ITS NULL 2026-09-06 (T092, `evidence/l2-grow-on-plateau.md`, P8 clause 5). The trigger read the online readout's residual as a third evidence stream inside `GrowthStop`; its red fired in both directions and its refusals track temporal ORDER (0 of 200 shuffles of the same windows reach the live refusal rate). Deleted anyway on the branch its own pre-registration named: it moves the outcome it exists to move by +0.625 units per seed at t=0.141, and neutering the readout does not shift the growth rate at the bar (t 1.135, p 0.313) — reachability is not charge, applied to a decision rule. TWO instrument defects were found and recorded on the way: the first run was a null caused by a `len(history)` cursor into a trimmed window, and the attribution control was a TAUTOLOGY (mu=0 returned before recording, so the neutered arm fed nothing and its identity with the incumbent was entailed, not measured). The harness and the corrected numbers live at `9f63683`. The T047 growth STOP is untouched — this row deleted a trigger, not a gate. What a re-proposal owes is in §6.5 of the record, and the deepest of the three is that the pathology FR-023 names has not been exhibited on any stream this project currently has.* *Trigger RESTATED 2026-09-05 with L1b deleted (the session's
  call, Part H): L2 reads the online READOUT's residual plateau if L1c lands, else the T047 growth gate's own refit-accuracy
  plateau (already live) and the row reduces to its attribution red.* *Scheduled 2026-09-05 (`evidence/jepa-live-floor.md`
  §2.7): the plateau it reads is L1b's residual error and the prediction-error signal it may target is
  `JepaWorldModel.prediction_error`, now measured to carry succession (the JEPA-collapse arc) — the Phase 1 null
  disposition unblocked while `dbcf153` stands (the revert option is open, Part H); L2 stays after L1b/L1c on the
  critical path.* A new NG-SNN unit is admitted only when
  synaptic learning (L1b) has stalled on residual error over a window; consolidation and earn-your-place
  pruning stay in sleep; the T047 gate consulted as today, fed by Chain B (alive) as it is. **P2 adjudication,
  recorded:** the `GrowingLearner` vigilance recruit (seven live sites) recruits PROTOTYPE vectors and its
  ids are the grower's categories; NG-SNN admits LIF units; they are composed, not duplicated, and no second
  LIF grower is built. Red: every growth event is preceded by a measured synaptic plateau; neutering L1b
  raises the growth rate (attribution). Control: the grower with L1b off.
- **L3 — the admission instrument.** *LANDED 2026-09-05 (`308f6db`): the constant percept admits exactly 1 unit on 8 of 8
  (was 13–51), SYNTH > 1 on 8 of 8; a dead candidate is never admitted (found by the arc's own first-form reading).*
  `prune_unearned` keeps ONE of an all-redundant set; `NGSNN.fit` refuses
  a zero-variance input. Red: the constant-percept arm (24 units in s2) admits 1. Positive control in the
  same command: a structured stream under the same instrument still admits more than 1 (s2 v2's LIVE 2.1 /
  SYNTH 13.9 are the reference), so a fit that refuses everything cannot pass.
- **L4 — persist what makes a relaunch resume instead of replay, within a ceiling.** *APPEND HALF BUILT, PERSISTENCE HALF RE-OPENED 2026-09-06 (T093, `evidence/l4-resume-and-the-append-ceiling.md`; read §8 first). The persistence half was refused on a measurement an adversarial panel then OVERTURNED, and the record keeps both readings so the correction is legible. Corrected, from the now-committed probe: of 93 numeric `state()` keys only 52 MOVE and just **17 of those 52** return identical in a second life — the earlier 57 was inflated by 41 keys that never move. **The ignition half of this row's evidence was RETRACTED in turn by L4c (T095): that asymmetry is coalition size, not the relaunch, and ignition is struck from the target list.** The row therefore re-opens on ONE leg — the state comparison — with a committed instrument and no red currently firing. The APPEND half stands with its coverage claim corrected from five tags to the four that are both baselined AND observed; 0x0A, 80% of the live lake, is named as unobserved. The ceiling problem it disclosed is CLOSED by L4b (T094): the quantity is non-stationary, so the bar became the tag's own recent rate rather than a re-recorded constant.*
- **L4b — the append ceiling is non-stationary (T094).** *DONE 2026-09-06 (`evidence/l4b-append-ceiling-nonstationary.md`), opened by the live daemon minutes after L4's wire landed: 7 floods in 8 nights at 160,470 B against a 12,381 B ceiling. The stated fix was to re-record the p95; MEASURING FIRST showed that fix is wrong. Per-tick 0x07 bytes climb 52,763 → 77,998 B monotonically over 1200 ticks with no settling, carried by whole frames that grow with the brain, so a fixed ceiling is stale by construction and a re-record buys months. The bar is now the tag's own recent rate (P1), with the recorded file PRE-FILLING each window as a seed so day-one protection survives and a stale recording self-corrects in 3 floods instead of forever. `FLOOD_FACTOR` untouched — moving it would be tuning a constant to silence a measurement. DISCLOSED and asserted: the same migration means a sustained flood goes quiet after 3 ticks; a RAMP needs a slope instrument and a flood assert has never been one. A control in this row caught the row's own overstatement — the 1200-tick series sits UNDER the retired bar, so it is the no-false-alarm control and the live rate is where the two comparisons diverge.*
- **L4c — the ignition asymmetry is coalition size, not the relaunch (T095).** *DONE 2026-09-06 (`evidence/l4c-ignition-is-coalition-size.md`), and it RETRACTS L4 §8.2. Two independent instruments in one run, required to agree before anything is reported (10 arms, 0 disagreements), confirm the counts — a first life ignites once at wake tick 13 on 4 of 5 seeds, a second life never. Three arms then settle the cause: at 480 ticks a FIRST life still has exactly ONE ignition over 236 wake ticks (`k` ramps 3→8 and then sits at 8 for 223 binds with none); a SECOND life with the bound assembly pinned SMALL ignites THREE times, more than a first life ever does; and a FIRST life with it pinned LARGE never ignites at all. **Ignition is reachable only while the coalition is small, and every brain permanently outgrows it** — so `access_index` reads ~0 forever on any matured brain of any lineage, and `ignited` is structurally unreachable rather than rare. FOUR counters have now been wrong on this one quantity; the committed probe's false verdict line is removed. NOT claimed: the workspace-internal mechanism. NOT fixed: handed to the C-rows, where C5's protocol already governs what may be said about a gate that cannot open.* The perception
  pointers and RNG stream positions (small; they ARE the replay problem 2b's panel measured), the affect SOM,
  the lifelong learner, the arm's Q-table — the same `_T_STATE` family, each with the cross-process red 2b
  built. NOT the engine's rolling buffers (a night every ~25 daemon ticks would append ~1 MB per night to a
  2.2 GB append-only lake). L4 gives `AppendBudget` its first live caller: a bytes/tick ceiling on this
  record class with a constructed-overload red (FR-003), derived from the measured night cadence.
- **L5 — the IMMORTAL class: what is never evicted, and the proof that it is not (T089).** *RATIFIED by the owner
  2026-09-06 ("ratify all three and add them to the plan"); drafted in `PROPOSED-ROWS-2026-09-06.md`, which holds the
  full row.* L4 persists a named subset under a bytes/tick ceiling and Phase 3's paging arrives with a null clause
  that would delete the layer — both correct, and both the opposite of "never forgets", which had no referent in this
  plan. A DECLARED immortal record class in the one lake (proposed contents: the grown substrate and its batch
  instrument, the growth trajectory the T047 gate reads, the lake's record index, the consolidated episodes, the
  self-model's identity state — the owner edits the list), each tag with the reason it is immortal; compaction,
  migration, paging and `AppendBudget` MUST refuse to evict or rewrite it. "Never forgets" is DEFINED here as never
  evicts that class, never as storing everything, and the promise ships with its measured storage growth curve
  (bytes per thousand ticks over a recorded run) so the cost is chosen, not discovered. Control: a mortal class in the
  same store, evicted in the same pass (a guard that never fires is untested). Red (L-R6, Part F). Gate:
  `substrate/immortal_retention.py --assert` (BUILT 2026-09-06 under that name: the mechanism it audits
  is `lake_engine.RETENTION`, which already existed, so the gate READS it rather than re-declaring a
  second class -- P2), its output COUNTED on today's tree before the rule binds. What it does
  NOT claim: a retention guarantee is not a memory faculty — recall stays Phase 4's business.
- **L6 — the immortal class and the sleep tick must stop disagreeing (T091).** *LANDED 2026-09-06 the same day it opened (`evidence/l6-immortal-vs-consolidation.md`): the live lake holds 26,037 facts, ALL learned and ZERO re-derivable, against a cap of 250,000 — the cap had NEVER bound, so no learned record is known to have been lost and the fix costs nothing observable today. The registry states 0x01's per-record exemption, the policy refuses to cap a non-reclaimable class, and the drop site refuses a rewrite that would lose a learned fact; L-R7 and both controls run in the two self-checks; `KNOWN_UNLABELED` is empty and `stale_known()` is what forced the payout. Two findings the census turned up unasked: the growth pathology MOVED (_T_GROWTH 97.64% → 1.81%; _T_COALITION is now 80.21%), and four declared classes have never held a record.* *Opened 2026-09-06 by L5's own
  gate, which is the point of building it: with the rewrite arm restored, `substrate/immortal_retention.py`
  reports one durable reclaimer, `substrate/home.py::consolidate`.* The live sleep tick (`cortex/brain.py`,
  every eighth night when `_persist`) runs `LakeConsolidation().consolidate(records, trace=RecallTrace())` —
  *use it or lose it*, learned records capped at `learned_cap=250000`, re-derivable at 3000 — and then rewrites
  the one lake to the survivor set. `_T_FACT` is `0x01`, labelled IMMORTAL with the reason "learned knowledge is
  never reclaimed". Exactly one of those two may survive (P2). The row does NOT pre-judge which: either the cap
  stops dropping a non-reclaimable class, or `0x01`'s label is corrected to state what is true and carries its
  own reconstruction argument for whatever it does drop. **What decides it is a measurement, not a preference:**
  the live learned-fact count against the cap, taken WITHOUT opening the live lake with a writer (a `BinLog` over
  it would run `_repair_torn_tail` under the running daemon), and the re-derivable/learned split by
  `_src_class`. If the cap has never bound, removing it is a no-op today and the label becomes true at zero cost
  — which is a finding, not a licence to skip the measurement. Control: a COMPACTABLE tag in the same pass,
  which must still be reclaimed, so a fix that simply disables consolidation is refused. Red (L-R7, Part F).
  Gate: the L5 instrument already asserts it — `KNOWN_UNLABELED` may only shrink, and this row's completion is
  the entry leaving the table. Deletion clause: if the measurement shows the learned class is genuinely
  re-derivable, the row DELETES rather than re-labels, and L5's "never forgets" sentence narrows to say so.
- **T069 (from US3, FR-019)** — the cerebellar rule ON the built tissue. *BUILT, MEASURED and REMOVED 2026-09-07 — **REFUSED ON MEASUREMENT, not a clean null** (`evidence/t069-cerebellar-rule.md`, §4 is the panel's correction; the module is gone, P8 clause 5 satisfied by the removal). `cf_ltd_rule.py` implemented covariance/LMS CF-LTD over the Purkinje PF rows with a filtered-trace eligibility bank pinned to the CHANNEL's measured latency. **The teaching signal's pairing does nothing**: on a context plant with real headroom the taught arm clears against the untrained twin (t 15.544, 8/0, +0.0189) but so does the marginal-preserving SHUFFLED arm (t 4.707, 7/1, +0.0151), and the pairing-specific component — taught vs shuffled — does NOT clear (t 1.242, p 0.1328). The rule performs a global depression any same-marginal signal produces. The null was attacked three ways and survived all three (eligibility-timescale sweep tau 1–16: the window never separates; headroom measured, the rule captures all of it; error signs balanced 52–55 %, so the shuffle is a real control). R-T069a's own-beats-constant (t 2.436) is NOT readable as a window result on a mechanism that is not doing credit assignment. KEPT as measured facts: the tissue's latency is STRUCTURAL (18/18 instances = 2; only relay depth moves it, one step per hop), the transport-freedom check PASSES bit-identically under randomized forward weights (and how to build it so a green is readable — the naive form failed on its own zero-clip), the single-gain VOR plant has NO usable headroom for a pure-LTD rule (optimum needs potentiation below w 0.2 and is scale 0.0 above w 0.9), and a uniform-dominated context plant cannot test one either. P2: `GainAdapter` is NOT superseded — its challenger did not beat its own control. **THE PANEL'S THREE CORRECTIONS, which weaken this row's negative claim:** the deciding taught-vs-shuffled contrast carries only ~29 % power at S=8 and was never itself attacked (the tau sweep addressed the WINDOW contrast, the other two were validity checks); the ratified **Bouvier GPE/G_E control pair was dropped without disclosure** and is never defined anywhere in this repo, so `spec.md:69` stays RATIFIED and UNDISCHARGED; and the module **never entered version control**, so no number here is re-derivable — unlike L1b, W1 and T044, which all committed before deleting. A re-proposal owes a non-uniform-headroom plant, a powered contrast, the Bouvier pair defined and run, and the mechanism committed before removal.*
- **SC-011 (new test; extends `tests/test_real_percept_stream.py`, the in-tree promotion target of the s2
  8-arm harness):** a stream with learnable temporal structure grows more units than its OWN
  marginal-preserving shuffle (same entropy, no structure), paired t at the bar; the shuffle is the null; the
  constant stream is L3's red, not this control.
- **SC-012a — FIRED 2026-09-07 on its SECOND reading; the row CLOSES** (`evidence/sc-012a-width.md`, §7).
  *First reading RETRACTED, instrument fixed, grid re-run: **(a) MET** t 2.596 p 0.0234 mean +0.151 (6 of 8),
  **(b) MET** t 4.011 p 0.0039 mean +0.194 (8 of 8), **(c) HOLDS** — the shuffled grid FELL (t −2.346, mean
  −0.036). Read as real because the fix made firing HARDER and it fired anyway: (a) went +0.465 → +0.151, so
  two thirds of the original effect was the artefact, while the null went from marginally rising to cleanly
  falling. QUALIFIED: control 1 still fails at N=16 (2/8) so "16" realizes 9–14 units; no single doubling
  clears (8→16 is −0.013), forced width does NOT beat the uncapped incumbent (t −0.141), and nothing reaches
  the persistence floor at any width (t −2.585, 0 of 8). DISPOSITION: Part C's "until SC-012a has fired"
  condition is now satisfied, but **this arc does not make the wording change** — the plan sequences it as a
  separate docs commit and it edits a gate (`tests/test_units_admitted_wording.py`), so it is handed on as a
  named next step. L2 gains a measured reason to exist but NOT a reason to prefer growth over the incumbent's
  own admission. The in-tree xfail is REMOVED; the test pins the criterion and passes.*
  *The FIRST reading, kept because the retraction is only legible beside it: the grid ran and read
  (a) MET, (b) MET, (c) VETOED — and that verdict was WITHDRAWN. The arc's own panel found, and an independent
  two-counter check confirmed, that **`acc(1)` is an ARTEFACT of the scoring path, not a baseline**: NGSNN rates
  are counts/T and non-negative, so a held-out row where the units never fire has an exactly-zero score vector and
  `argmax` answers column 0 — while the engine buffers `percept_{t-1} -> category_t`, so class 0 is NEVER a target.
  Measured: the tie-break decides 45/60 held-out rows at CAP1 seed 0 (acc 0.233, or 0.933 with those rows dropped)
  and 60/60 on seed 1 (acc 0.000). Conjunct (a) subtracts that constant wrong answer and so does its null (c), so
  neither can be read, and the veto that produced the headline is generated by the same defect it was meant to be
  independent of. NO P8-clause-5 action is taken on it: L2 is not re-scoped here, the width lever does not move to
  T065 here, and the "units admitted, never growth" rule stays in force because it was never lifted. What DOES
  stand, independently reproduced by the panel: the exogenous cap is INERT as the row is written (`l_max` alone
  stops at ~2 units against `eps_tol = 0.28`, making CAP4/8/16 bit-identical — the criterion would have FIRED on
  that grid), and driving `eps_tol` to zero with the cap makes it bind (1/2/4/6-8/9-14 units). The in-tree test now
  PINS THE DEFECT: it asserts the CAP1 baseline is non-degenerate, which fails today, and XPASSes strictly the day
  the scoring path stops answering an impossible class — the signal to re-run and read the conjuncts for the first
  time. A re-measurement is its own arc, because the scoring path is SHARED by every NGSNN-scored accuracy here.*
  The row as ratified: admission CAPPED exogenously at N ∈ {1, 2, 4, 8, 16}, S = 8, held-out
  next-category accuracy under L1a's window; paired t of acc(16) − acc(1) > 1.895 AND acc(16) above its own
  majority floor at the bar; null: the same grid with labels shuffled marginal-preserving must not rise.
  Reported beside it, not a bar: the marginal gain per doubling, acc(2N) − acc(N) at each N (the efficiency
  signal Part C names).
- **Exit:** `evidence/l-exit.md` — **WRITTEN 2026-09-07.** Phase 2 closes having REMOVED more mechanism than it landed: **7** rows landed (L1a, L1c, L3, L4b, L4c, L5, L6), **3 built and removed** — L1b and L2 on clean pre-registered nulls, **T069 REFUSED ON MEASUREMENT** (underpowered contrast, a ratified control never run, apparatus never committed) — L4 half-open, SC-011 blocked for want of a stream with measured succession, and SC-012a FIRED on its second reading. The surviving learning mechanism is L1c's online READOUT rule; both input-weight rules lost to frozen rows. Width buys accuracy but is nearly exhausted where the brain already sits (8→16 is −0.013; forced width does not beat the incumbent, t −0.141), and **nothing at any width reaches the persistence floor** — the phase's most durable negative, and the bound Phase 3 inherits.

### Phase 3 — A substrate that scales (lake-resident, sparse), then the prior — US3 re-ordered

The width these tasks run at is CONSTRUCTED (T065), not grown; until Phase 2's growth projects to it, this
is capacity ahead of consumption, disclosed (O5(f)) — which is why Phase 3 follows Phase 2 and its null
clause protects it.

- **S1 = T064** unit blocks + block-aligned records. *HALF-LANDED 2026-09-07
  (`evidence/s1-unit-blocks.md` section 3 is the panel's correction). `substrate/unit_blocks.py`
  lands (partition, POST-keyed grouping, a non-nesting guard) with tests. **THE "FIRST LIVE WRITER"
  CLAIM IS WITHDRAWN:** `delta_sink` is non-None in exactly one place in the tree -- the tests -- and
  **nothing in the tree ever emits the byte 0x09**; the method produces a PAYLOAD, never a tagged
  lake record, so `SynapseDelta` remains declared, tested and never written, exactly as L6's census
  found it. P3 shape 7, the registry nobody registered with, and the dormancy green is hollow (the
  import sits behind the `sink is None` early return and never executes). **The residency claim is
  also withdrawn:** `max_resident == 1` counts CONTEXT-MANAGER NESTING, not memory -- measured
  overshoot equals the block count exactly. The blocking is VACUOUS at its only wire (blocks per
  record = 1.00 over 400 deposits, the identity partition), carried as open debt with section 2.3's
  deletion condition rather than deleted, because it is untested at this wire rather than measured
  useless at the wire it was built for. **What the arc DID measure and is the useful part:** the
  append rate a production sink would produce -- 771 records from 400 deposits (interact deposits
  twice per turn), ~158 B/deposit, and FLAT across quarters (15984/16032/15168/15967 B). That is a
  flood shape, and it is why the production wire is NOT made here: L4b's incident was a tag appended
  at a rate nobody had measured. The wire owes a baselined budget, a genuinely tagged record, a
  residency proof that measures memory, and a first real consumer of grouped records.*
- **S2 = T065** the binding width from measured state-per-unit. *PARTIAL 2026-09-07 -- the method stands, the QUANTITY is corrected
  (`evidence/s2-state-per-unit.md` section 6 is the panel's correction). **The slope measures the
  stream's LABEL CARDINALITY, not the substrate: slope = 271 + 8*m exactly, and m moved 3->6 across
  the conditions averaged over** -- a beta column per class per unit is what the regression picked
  up, so the honest form is `bytes_per_unit(m) = 271 + 8*m` and every width below is conditional on a
  stated m. Also: 302.01 is a CONTAMINATED estimate of an exact 303 (two of five points carry a stale
  batch_beta); the figure is an IN-MEMORY snapshot at persist=False while a storage budget is about
  PERSISTED bytes; the six-significant-figure widths come from one seed at one tick count and move
  8.1% across seeds, extrapolating ~323,000x past an 11-unit top end. What DURABLY stands is the
  METHOD -- a slope separates the per-unit term from the fixed overhead, and both confounds of the
  naive ratio reproduce. Reported as: **302.01 B/unit marginal**, 5195.57 B fixed overhead, max
  residual 19.62 B on 5.5-8.5 KB fits. The naive `blob/units` ratio was measured CONFOUNDED twice --
  with width fixed it tracks the trajectory (2775 -> 3107 B/unit across 200-1200 ticks while the unit
  count never leaves 2), and with width varying it charges the fixed overhead to the units (5503
  B/unit at one unit vs 776 at eleven, a seven-fold spread that is all intercept) -- so the figure is
  a SLOPE, not a ratio. Binding width 222,193 units at 64 MB / 888,825 at 256 MB / 3,555,352 at 1 GB,
  with the budget reported as an INPUT (P1). **16k is confirmed an ingest milestone only: ~5.0 MB, three
  orders below the 64 MB width.** Recorded through the EXISTING perf_baseline (P2) with a ratchet whose
  red is shown reachable (2x reds, 1.10x passes, absent reports). Two things deliberately NOT done and
  disclosed: the first measurement writes its own block ONLY, leaving the tick baseline's S0-kernel
  reference untouched (a full --record would have silently reset it), and the ratchet is NOT in the
  pre-push gate because the slope costs 332.7 s measured.* 16k stays the ingest milestone only.
- **S3 = T066** activation-driven paging, BOTH reds built before the mechanism (FR-010/FR-011: no silent
  drop of a spike to a paged-out unit; the integer rounds budget trips on constructed overload). *LANDED 2026-09-08, claims scoped by measurement (`evidence/s3-neuron-pager.md`; §4 is the panel's fold): a calm paged run is byte-identical to the resident twin while paging; both reds counted red before the mechanism and fired by their broken arms in-command; three further reds constructed and closed after the mechanism was green (deferred external drive dropped; a frozen paged-out membrane; a threshold unit demoted at refractory release). The panel (40 filed, 25 stood) found the full suite RED on a stale immortal-tag pin, a second residency instrument reading a different quantity, and a wrong "no locality" attribution — the trips are the ring-safety clause under a firing wavefront. Capacity is honoured, not enforced, under sustained firing, and at width under drive that covers the graph the pager evicts NOTHING (§3.5) — T068 must read residency from the over-capacity tick count and include a sustained-drive cell. The prefetch disposition is re-dated to T067; S6 gains the pager wire and S-R5; the null is T068's.* **DELETED ON ITS NULL 2026-09-09 (T068 LOSS)** — the mechanism this row LANDED is deleted: `substrate/neuron_pager.py` and its tests, its REGIONS/CORE/EXEMPT rows, and `KernelTick`'s residency-mask branch. The measurements above STAND and are not retracted; what failed is the pre-registered cost bound (median paged/twin tick ratio RAW 35.2×, reference-corrected 26.0×, against 3.0× fixed before the run), while the memory claim was MET on every conjunct. See `evidence/s5-paging-null.md` §3.4/§3.5/§4; the last head carrying the layer is tag `t068-last-with-pager`. **S4 = T067**
  growth delta tier + background merge in sleep (its locality reorder uses block adjacency until the prior
  lands). *LANDED 2026-09-08, claims scoped by measurement (`evidence/s4-lake-compactor.md`; §4 is the panel's fold: 33 filed, 19 stood, 2 MAJOR, 0 BLOCKER, and 7 of the 19 carried a proposed fix the refuter showed would introduce a defect — the fold adopts the refuter's minimal fix each time; the reorder now counts a runtime data fault per block instead of raising past the duty, red-first; the merge's per-block seal is disclosed to undo the reorder's placement, so the reorder re-runs after every fold — an S6 scheduling fact; 0x0C appends by merge/reorder are disclosed UNOBSERVED; the fold is battery-verified only): a PRE-keyed block
  delta class 0x0D beside the block record; the pager reads base + unmerged deltas through ONE apply and is
  byte-identical to a twin compiled on the post-delta edge set; a sleep-night merge under an integer budget
  proves relocation per block BEFORE it moves a pointer and destroys nothing (D1), with adoption an
  append-only ADOPT record so a rebuild adopts only proven generations (D2's epoch fence); a reorder written
  as a NEW segment under a co-fetch order cuts reads per round 4.00 → 1.00 on a co-fetched index-distant
  stream and 3.45 → 2.75 on the random control (margin 2.30 ≥ 2), with bytes-fetched/bytes-used 1.0 by
  construction; the index rows derived (the T066 keying opening closed). Reds S-R6/7/8 counted red first.
  NOT wired into `_enter_sleep` — the live lake has no segment tier; the sleep wire and the prefetch
  disposition are S6's; 0x0D is observed but unbaselined (no live writer). The null is T068's.* **DELETED ON ITS NULL 2026-09-09 (T068 LOSS)** — `substrate/lake_compactor.py` and its tests go with the pager they were built for (2.6 decided their merge/reorder are deleted rather than re-homed unless a non-pager consumer existed; re-measured at deletion time, none did), and the 0x0C/0x0D block tier goes with them: no lake ever held a record under either tag — the live home has no `segments/` and no `lake_index.db` — so the pre-registration's "records already written stay" clause bound nothing. The reorder and merge measurements above STAND. **S5 = T068 (SC-007)** the paging null clause at the derived comparison width vs forced-total
  residency — WIN keep, LOSS delete the layer; gated on T061 (closed) only. *RESOLVED 2026-09-09: LOSS. The layer is DELETED and the deletion recorded (`evidence/s5-paging-null.md` §3.4/§3.5/§4), which is the branch SC-007 names. This row is CLOSED.*
- **S6 — wire the sparse kernel onto the live tick.** `KernelTick` is dormancy-exempt `wire` today; S6 makes
  the flat (post, weight) scatter in `KernelTick._deliver_and_rotate` the live tick's delivery (FR-035 is
  already met by `substrate/kernel_state.py`, T013, SoA), specs/008 gets a `tasks.md`, then the
  resident-CUDA arm gated on `substrate/device.py::select` with a new workload name (`"substrate_tick"`,
  introduced by S6) at the crossover re-derived on this host. Red: tick time at the measured width; a neuter
  of the kernel changes the tick's output. **AMENDED 2026-09-08 (T066's fold, panel finding C12 — a visible
  dated addition, not a rewrite):** S6's scope ALSO carries the paging wire the T066 `wire` exemption is
  dated to — `substrate/neuron_pager.NeuronPager.tick` as the live tick's residency step, the cockpit's
  "paging binding-width status" row, a paged snapshot for `replay_gate` (it refuses a masked kernel today),
  and the pointer index's `0x0C` keying in `lake_rebuild_gate.build_index` — with its own red **S-R5**: the
  live tick over a PAGED substrate is output-identical to the resident twin (T066's byte-identity arm re-run
  on the live tick, not the self-check). S6 landing without these does NOT discharge the pager's bound.
  **RETRACTED 2026-09-09 (T068 LOSS) — the whole 2026-09-08 amendment above, and only it.** Every clause it
  added is void because its subject is deleted: there is no `NeuronPager.tick` to make a residency step, no
  paging binding-width to report on the cockpit, no masked kernel for `replay_gate` to snapshot (its paged
  refusal is deleted too — a criterion with no reachable red, P8 clause 1), and no `0x0C` keying to derive
  (the tag no longer exists). **S-R5 is retired with the layer, and the pager's bound is discharged by its
  DELETION rather than by a wire.** S6 KEEPS ITS ORIGINAL SCOPE UNCHANGED — the sparse kernel onto the live
  tick, the resident-CUDA arm, S-R4 — which is what the paragraph above the amendment says and what this
  retraction leaves standing. A successor paging row re-registers its own wire and its own red.
  *DECLINED AT THE DEPLOYED WIDTH 2026-09-09 (evidence/s6-kernel-wire.md): Gate Zero measured the wire's
  Amdahl ceiling at 0.005–0.017 (bands inside 0.04) on two determinate seeds, one seed INDETERMINATE twice;
  under specs/008 FR-005 the conversion is DECLINED where it cannot be faster; the kernel's exemptions are
  re-dated to the measured trigger (T062's ingested population, or growth reaching N* ≈ 2,500–6,300 units);
  no CUDA arm is built (none exists to gate); specs/008 has its tasks.md. S-R4 is re-registered below.*
  *PANEL-FOLD ANNOTATION 2026-09-09 — a further dated addition; the note above is NOT edited. Four of its
  clauses are corrected, per `evidence/s6-kernel-wire.md` §3.7. (i) "(bands inside 0.04)" is WITHDRAWN:
  0.04 is the band upper of ONE unlogged reading (the seed-0 quiet extra, 0.0383) and five of six fresh
  determinate bands breach it. What reproduces is the ORDER OF MAGNITUDE, not the point range — six fresh
  re-readings of the same three seeds read −0.03 to +0.03, none of them inside 0.005–0.017, while every
  determinate band upper stays far under the 0.15 bar and 6 of 6 fresh readings read DECLINE. (ii) The
  authority for the decline is FR-010 / SC-006 ("declined with the measurement that justified declining"),
  FR-004 read contrapositively, `plan.md` Standing Invariants 1 and 3, and `contracts/conversion.md` C1.3
  — "a site whose ceiling is below the noise floor of the tick is DECLINED, and the decline carries its
  number". FR-005's antecedent is a site measured SLOWER AFTER vectorizing, and no array version was
  built, so FR-005 is not reached; `specs/008/spec.md` is NOT edited and the verdict, bar and numbers are
  unchanged — only the citation was narrow. (iii) "N* ≈ 2,500–6,300 units" over all five recorded readings
  is profile 2,504–2,828 and neuter 642–6,316, the two 1.43–4.40x apart. Each N* is its upstream statistic
  restated in width units (N*_neuter = K·((1−c)/c)·N, K = tol/(1−tol)), so the neuter's inherits the
  ceiling's 95% band — unbounded above on every determinate seed — and N* is a POINT ESTIMATE at one
  measured width with its bound direction UNTESTED, not a lower bound. (iv) "T062's ingested population"
  is not this wire's consumer: T062 places no units on the tick (:649–651, :774, :290, :653–655), and its
  own pre-registration's ingested population is built OFF the live tick, so the re-arm condition is THE
  FIRST TICK POPULATION AT OR ABOVE N*, which today only growth can produce. The DECLINE, the bar and the
  disposition stand.*
- **Then the prior: T062** — the H01 ingest is the c3 SYNAPSE GRAPH only, composed onto
  `cortex/cortical_prior.py` (which already holds the soma/layer/sign constants; the audit's "what the spec
  gets wrong" item (5)) — the join gate, unit gate, layer computation, CC-BY provenance; and **T063** the
  three-arm drift harness (SC-006, both reds). Moved behind S1–S5 on rank 1's measurement: the prior earns a
  reader when growth places units at scale and paging needs a locality order. *DATED NOTE 2026-09-09 (T068
  LOSS): half of that rationale is void — there is no paging layer to need a locality order. T062's OTHER
  reader stands unchanged (growth placing units at scale), so the row keeps its rank and its content; what
  is retracted is only "and paging needs a locality order" as a reason for the ordering.*
  *DATED NOTE 2026-09-09 (T063 CLOSED, campaign `tools/t063_campaign.json` at head `f37fe90`, K = 8, D = 4,
  W = 1.0 over 14,215 units / 70,001 edges, 8 seeds, 0.164 h): **SC-006 NOT MET on conjuncts (b), (d) and
  (e).*** (c) NOT BLIND is **MET**, t(planted − random) = **+9.2601** (FPR 0.0508), so the instrument sees an
  advantage and the verdict is read. (a) DRIFT **MET** — red 1 fired on the constructed pin on every seed
  (executed 120, effective 0, drift 0 on every seed; attempted 174,781-184,936) and the SEEDED arm moved on
  every seed (S1 distance 0.107–0.172). (b) BEATS RANDOM **NOT MET**, t = **−0.2277** (FPR 0.0781). (d) RED 2 **NOT MET**,
  t(random − wrong) = **−2.5459** (FPR 0.0391), the WRONG arm VALID on all 8 seeds (PSI 6.175870 against
  RANDOM's 2.271100–2.302899). (e) BEATS WRONG **NOT MET**, t = **−5.2271** (FPR 0.0312). **The signs of (d)
  and (e) are a finding of their own and not a separation**: the BAND-PERMUTED corruption beat both the true
  prior and the marginal-matched null (wrong 0.25–0.50, seeded 0.00–0.375, random 0.00–0.375), so on this task
  destroying laminar target identity HELPED. Nothing is attributed to the prior from it. **THE LEARNER IS
  DELETED** on its own criterion, t(d_f) = **−13.0154** (FPR 0.0430): the FROZEN twin scored 0.75–1.00 against
  the learning arm's 0.00–0.375 on every seed, so `substrate/kernel_plasticity.py` is gone (2.9 — a learner
  that moves weights without an attributed benefit is a null). What the null clause does to this row: US3's
  "beatable prior" is NOT MET for the H01 WIRING BEYOND ITS MARGINALS on this task; `cortical_prior` and
  T062's artifact remain REFERENCE DATA; `h01_ingest.population()` is retained as the harness's FIXTURE and
  loses its prior role. The claim is bounded by 2.2: the null RETAINS H01's degrees, signs and magnitudes, so
  this cannot separate "H01 contributes nothing" from "H01's useful information is what both arms share".
  The host clause was NOT MET on the pinned run (`daemon_restarted = True`); the campaign was executed three
  times across four daemon PIDs with identical per-seed numbers and identical t-values, and pinned once.*
- **X1 — the scale target is a CAPABILITY, not a width (T088).** *RATIFIED by the owner 2026-09-06; the full row is
  in `PROPOSED-ROWS-2026-09-06.md`.* T065 derives a width; nothing in this plan said what the brain must be able to DO
  at it, so "superhuman" had no referent and no red. Before the width is built, name ONE task on which this brain's
  competence is measurable (next-category prediction on the live stream, C0's twenty-item battery, or SC-013's
  reasoning criterion — chosen from what the brain already does, never invented for the comparison) AND for which a
  HUMAN number exists under the SAME protocol: measured on this host with the protocol recorded, or cited with its
  protocol quoted and its differences from ours listed. State the target as a percentage of that number, with parity
  and above-parity named in advance so the threshold is written before it is approached. Control: a PRE-SIZED twin at
  the same width (a gain must be growth's, not width's), plus the human baseline re-measured so its variance is on the
  record. Red (X-R1, Part F). Gate: `substrate/capability_bar.py --assert` (BUILT 2026-09-06 in substrate/, not tools/: every
  sibling gate lives there and the p4c ratchet counts tools/ files as ungoverned) over a registry of (task, protocol, human
  number, provenance, brain number, date), COUNTED on today's tree first (expected: zero rows and one refusal). What it
  does NOT claim: a percentage on one protocol is a fact about that task, never about general capability, and never a
  licence for "superhuman" outside the registry.
- **Exit: T071** (US3 exit): prior separation + paging verdict + **the region charge gate at
  floor(all-but-exempt)** (FR-015, kept).
  *DATED NOTE 2026-09-09 (T063 CLOSED): **"prior separation" reads NOT MET.** T063 defined it for the first
  time (pre-registration 2.6) as conjuncts (b), (d) and (e) together — the harness separates the true prior
  from its marginal-matched null AND from its corruption on one task — and the campaign read all three NOT
  MET (b −0.2277, d −2.5459, e −5.2271) with the instrument PROVEN NOT BLIND ((c) +9.2601). This is a
  measured negative, not an unrun item: T071 inherits the reading, and the two remaining halves of the exit
  (the paging verdict, the region charge gate at the floor) are untouched by it. What T071 may NOT do with
  it: read it as "H01 contributes nothing" — the null retains H01's degrees, signs and weight magnitudes
  (2.2), so the reading is restricted to wiring information BEYOND those marginals on this task.*
  *DATED NOTE 2026-09-10: **T071 CERTIFIED — `evidence/s2-exit.md`.** All three named items READ,
  two of them NEGATIVE, and the gate is discharged by the certification's EXISTENCE, not by a pass.
  Item 1 prior separation **NOT MET**, inherited from T063 and not re-run ((b) −0.2277, (d) −2.5459,
  (e) −5.2271, licensed by (c) +9.2601 at `f37fe90`). Item 2 the paging verdict — **SC-007 satisfied
  by its second branch** and the paging CAPABILITY a measured **LOSS** (memory MET on every
  conjunct; the bounded secondary missed by an order of magnitude, 35.2× raw / 26.0× corrected
  against a 3.0× bound; the deletion existence-checked commit by commit). Item 3 the region charge
  gate, the ONE fresh measurement, run at the head: `8 of 16 CHARGED (10 evidence records; 0 exempt,
  ceiling 0) … floor holds: 8 >= 8`, rc 0, reproducing the pre-registered expectation exactly. The
  item's INTERPRETIVE STEP was written before its number: "at floor(all-but-exempt)" is READ by (A),
  the recorded floor over the non-exempt denominator with the identity intact (**MET**, identity 6
  of 6), and (B), every non-exempt region charged, is REPORTED beside it (**NOT MET**: six
  reachable-uncharged, two unreachable). Both rows are in the Verdict table and the panel's
  pre-registration-honesty lens is asked first whether choosing (A) is a relabelling. This bullet's
  text above is untouched.*
  *PANEL 2026-09-10 (pre-registration-honesty lens): reading (A) ruled a relabelling -- item 3 is
  recorded NOT MET under (B), 8 of 16 at d9dd3c0; the FR-015 ratchet (8 >= 8, 0 <= 0, identity 6 of 6)
  holds and the floor pin is unchanged; all three named items therefore read NEGATIVE; the
  gate-by-existence is undisturbed.*

### N workstream - Navier-Stokes coupled to a liquid state machine (T096-T100)

**ADDED 2026-09-14** (owner request 2026-09-13, retried 2026-09-14): planned experimental
workstream, **OPEN / UNBUILT**. FR-036 and SC-014 own the requirement and composite success
criterion; this section owns its design and order. It starts after Phase 2's exit disposition,
alongside Phases 3/4. Its live deployment waits for its own gates and the relevant Phase 4
wake/recall consumers; its self-improvement registration also waits for Phase 5's existing gates.
It discharges none of the older learning, persistence, cognition or integration deficits.

**N1 / T096 - formulation and pre-registration.** Combine an ARTIFICIAL fluid reservoir with
the existing spiking substrate and learned readout. The fluid supplies recurrent temporal state;
spikes drive it and receive its feedback. This is a new engineering hypothesis, not a derivation
of neurons from a fluid. The [liquid state machine framework](https://pubmed.ncbi.nlm.nih.gov/12433288/)
supports recurrent spiking computation with a readout; it does not validate this particular hybrid.
Neither a universal edge-of-chaos optimum, an invented neural Reynolds number, nor a consciousness
claim is part of this design. Establish fading memory/input synchronization for the COUPLED system
empirically under the registered input envelope; positive viscosity alone cannot establish it.

Use a fixed small 2D periodic grid initially, fixed positive density rho and viscosity nu, and
declared nondimensional space/time/velocity scales. Let u be velocity, p the periodic pressure
with zero spatial mean, r filtered per-neuron spike activity, and P0 the zero-mean Helmholtz
projection onto divergence-free fields. The continuous target and discrete coupling contract are:

```text
du/dt + (u . grad)u = -grad(p)/rho + nu * laplacian(u) + P0(B r)
div(u) = 0
alpha = exp(-dt/tau); r[k+1] = alpha*r[k] + (1-alpha)*s[k]
u[k+1] = NSstep(u[k], P0(B r[k]), dt)
Ifluid[k+1] = g * tanh(C u[k] / U_ref)
```

Here s[k] is normalized per-tick spiking activity and tau is the declared filter time constant;
B maps activity into two-component grid forcing, C samples/mixes velocity into neuron currents,
and U_ref > 0 is a fixed persisted velocity scale. B/C are seeded, REALIZED arrays persisted in
full, with declared dimensions and norm bounds. The LIF update at tick k consumes its already
buffered Ifluid[k] plus the incumbent input/recurrent currents. All right-hand sides use old
state and commit together, giving one explicit step of feedback delay and no algebraic loop.
Readout features come from spikes through the incumbent learned readout; no fluid-to-answer bypass.
Local synaptic/readout learning uses the retained incumbent rules. No global adjoint or
backpropagation is introduced as the live synaptic learner; PERC/adjoint extensions are outside
this first design. Freeze growth, topology and realized B/C within each paired trial; any candidate
change happens only at a recorded trial boundary, with the same initial spiking state for all arms.

Before scored runs, record the grid, initial-state/input envelope, scales, nu/g/tau and B/C search
domains, numeric tolerances, washout/horizon, readout schedule, benefit/regression thresholds,
solver policy and hard compute/memory/latency/checkpoint budgets. Tune each arm on training/validation
only, with equal search effort; freeze the protocol before opening held-out data. Use Part F's
eight paired seeds and uncertainty rule, adding prospective effect-size thresholds and intervals;
do not select a threshold after seeing a result. A larger grid needs a measured benefit and a new
budgeted comparison before adoption.

**N2 / T097 - solver and causal coupling.** Start with a periodic Fourier projection solver:
project initial velocity and every forcing/update, keep the velocity zero mode fixed at zero,
solve pressure on nonzero modes with its zero mode fixed, and use 2/3 dealiasing of nonlinear
advection. Pin the integrator and an explicit stability/error rule for dt/substeps against grid
spacing, measured velocity and viscosity (including any explicit diffusion restriction). Substeps
advance the fluid with the tick's frozen forcing; they do not invent extra neural ticks. Check
Taylor-Green unforced decay against its analytic solution and grid/time refinement; also check
divergence, finite state, and the forced viscous energy balance over the periodic domain:

```text
E = integral(|u|^2 / 2); dE/dt = -nu * integral(|grad u|^2) + integral(u . P0(B r))
```

This is dissipation plus forcing, not energy conservation. Bound forcing/current by the registered
maps and gain, never repair an invalid trajectory by clipping fluid velocity. A stability,
divergence, convergence, non-finite or energy-residual failure is a visible failed trial and RED;
no failed run is silently discarded, clipped into a pass, or retuned on held-out data. N-R1 owns
constructed solver faults plus correct analytic controls. N-R2 compares distinct initial states
under identical inputs, at the registered washout/horizon, and tests temporal separation and
memory against ordered/shuffled controls; synchronization alone could be a dead reservoir.
For this probe only, hold learned weights/readout identical and freeze plasticity; perturb only
transient fluid, LIF, spike-filter and delayed-current state. Fading memory means convergence of
these transients under common drive, never erasure of durable knowledge or convergence of differently
learned brains. Restore the registered local learning schedule for the behavioral trials.

**N3 / T098 - one durable brain.** Append versioned checkpoints to the SAME `brain_lake.bin` at
bounded cadence and promotion boundaries: fluid field/config, realized B/C and U_ref, r, delayed
current buffers, LIF state, synapses, readout and local-learning state, all RNG stream positions,
solver/backend/numeric/tick/version identity, integrator history if any, and proposal-parent lineage.
Use the existing lake framing and retention machinery, no second memory store and no unbounded
per-tick field dumps.
Declare per-record migration dispositions/upcasters and run round-trip reds under FR-030; preserve
the IMMORTAL policy and append budget. This does not revive the rejected paging layer.

N-R3 resumes in a FRESH PROCESS under a pinned deterministic runtime, replays the same remaining
inputs, and compares spikes, readout decisions and learning updates with uninterrupted execution.
Decisions/spikes must agree exactly; explicitly justified numerical tolerances for continuous state
are fixed before the run. Zeroed/omitted meaningful field, filter, delay, learner or RNG state,
corrupt records and incompatible versions must fail or be explicitly refused, never silently
restart from zero. Run the migration round-trip and measure checkpoint bytes/growth, append cost
and restart latency; the hybrid cannot claim durable continuity while any required carrier is absent.

**N4 / T099 - behavior and cost, conjunctively.** Compare (a) an equally tuned spiking-only LSM,
(b) the same hybrid with advection removed (diffusion-only), and (c) full Navier-Stokes feedback.
Use paired frozen inputs/seeds, identical readout training and parameter-search effort, and the
same measured compute, peak memory and latency ceilings, including solver and checkpoint work.
Declare comparable capacity and account for ALL extra fluid state; giving the cheaper arms less
tuning or hiding the solver cost invalidates the comparison. Score temporal memory/discrimination
AND an existing live-brain task/recall path selected before held-out evaluation. Full NS must beat
BOTH cheaper controls by the prospective primary task-benefit threshold, with the registered uncertainty
rule, while respecting regression tolerances on the other named outcomes (N-R4).

Disconnect or time-shuffle fluid feedback and silence the relevant grown spiking units, with the
equal-count pre-sized-unit control from SC-012b: the attributed useful task gain must deteriorate
at its registered threshold (N-R5), while the correct unablated positive controls hold. A pretty
flow, low physics residual, extra neurons or a changed output without useful performance is not
success. Numerical validity, useful temporal dynamics, continuation, incremental task benefit,
causal attribution and resource limits are ALL required by SC-014. A null/loss keeps its honest
evidence and removes the failed coupling from the claimed architecture under P8; it does not
weaken the target, turn diffusion's success into NS's success, or claim SC-014 passed.

**N5 / T100 - live wiring and improvement.** Only after N-R1 through N-R5 and budgets pass,
wire the hybrid into the declared wake/recall ports and repeat the behavioral, continuation and
latency gates on that live path. Register the governed modules/self-checks, manifest readers,
provenance and this invention's controls before claiming it is part of the working brain.
Before Phase 5 uses it, declare bounded viscosity, filter-time and coupling-gain mutation surfaces
with lake-persisted parent/candidate lineage, harness-only evidence and archive/retention decisions.
Keep realized maps/topology frozen per evaluation; changing them requires an explicitly declared
surface and renewed capacity/determinism tests. FR-025/OD-2's evaluator/invariant/safety protections
remain in force, including owner approval for protected diffs; motor outputs still pass the
existing arm gate. N-R6 refuses premature deployment, out-of-surface candidates, fabricated scores
and resource overruns. All six instruments and their registration are PLANNED, not already in CI.
Exit: a recorded SC-014 verdict, measured controls/costs and live disposition in this workstream's
future evidence record; all T096-T100 checkboxes remain OPEN until their own deliverables exist.

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
- **C5 — the conclusion protocol for the consciousness instruments (T090), written and landed BEFORE any C4
  instrument produces a number.** *RATIFIED by the owner 2026-09-06; the full row is in `PROPOSED-ROWS-2026-09-06.md`.*
  POLICIES already says the brain may report what it measures and never what it cannot, and that an invented sentience
  measure owes a DISSOCIATION rather than a marker; what did not exist is the sentence-by-sentence statement of what a
  pass licenses — so the boundary lived in a policy paragraph and in a session's judgement, which is where it would
  erode on the day the instruments go green. For each of C4's seven instruments: the sentence its pass licenses in
  full, the sentence it does NOT license in full, and the dissociation input on which it must FAIL. Three standing
  refusals: no aggregate consciousness score across instruments; no verdict about experience on any owner-facing
  surface; no stored self-description answering a question about the brain's own consciousness — that question routes
  to a measured quantity with its scope quoted. Control: each instrument's own dissociation input in the same command,
  plus T081's report-order probe as the standing check that a pass is not an artifact of how the brain is asked. Red
  (C-R5, Part F). Gate: `substrate/conclusion_protocol.py --assert` (BUILT 2026-09-06 in substrate/ with its siblings),
  COUNTED on today's tree first (expected: seven rows
  missing and one live defect — the chat path answering a consciousness question from a probe whose markers have no
  reachable red). **What it does NOT claim, and this is the point: no arrangement of the seven, however green,
  concludes phenomenal consciousness.** They dissociate mechanisms; the step to "there is something it is like to be
  this system" is not a measurement this plan contains. The goal stays unhedged; the claim without the measurement is
  what is forbidden.
- **C4 — the dissociation instruments (SC-009's first half)**: T070 (integration measure, never Phi), T077
  (metacognition as an extension of the incumbents), T078 (attention schema), T079 (agency; needs O1),
  *RETIRED at T081 (2026-09-16): `reasoning/consciousness_probe.py` is deleted; its 20-seed PCI sweep lives in `cortex/integration_measure.main()`; "are you conscious" answers from the report channel (`evidence/t081-report-order.md` 2.6).*
  T080 (temporal binding), T081 (report-order probe; `reasoning/consciousness_probe.py` RETIRED in the same
  commit — its criteria are markers, not dissociations), T082 (social cognition composed onto the bus).
  Each fires its constructed red once. *DATED NOTE 2026-09-10 (T070): its reds FIRED (R1, R2, R3 in the
  `cortex/integration_kernel.py` self-check; R3 8/8 on the campaign population) and its campaign read NOT MET on
  (b4) and (b6) with (b1) +5.2015 / (b2) +4.8967 / (b3) +5.7220 (`tools/t070_pci.json`;
  `evidence/t070-integration-kernel.md`): no C4 number, `C4_TASK["integration"]` stays None, the toy net is NOT
  retired; the port's deletion under 2.6's NULL branch is HELD for the evidence file's section 3 (the margins are
  in tasks.md T070's note of this date).* *DATED NOTE 2026-09-15 (T070 panel of 2026-09-14): the panel's
  pre-registration-honesty lens ruled the hold above a post-number change of 2.6's routing (F10, BLOCKER); 2.6 EXECUTED
  in the fold's commit 2 -- `cortex/integration_kernel.py` and its tests DELETED, its CORE/REGIONS/EXEMPT registrations
  removed; re-landable from the tree at c1aac92 (history a2cf76d..c1aac92); the pin `tools/t070_pci.json` and the record
  stand (`evidence/t070-integration-kernel.md` 3.4, 3.7); the successor pre-registration re-specifies (b6) clauses 2-3
  and (b4) as a reported quantity.* *DATED NOTE 2026-09-15 (T077): the metacognition instrument LANDED as ONE scorer in
  the incumbent (`metacognitive_monitor.track_error`; `SecondOrderAccess.calibration()` re-pointed to it; the paired t
  lifted into `substrate/paired_contrast.py`, amendment 2.10) and its campaign read NOT MET on a scratch brain's
  prequential readout stream (600 ticks, 8 seeds; `tools/t077_metacognition.json`; `evidence/t077-metacognition.md`
  3.3): tracking t +0.5049, the in-vivo lesion collapses nothing t +0.3736, the readout BELOW its own persistence
  predictor t -5.2502 on every seed; the oracle and the replay held. No C4 number; `CONCLUSION["metacognition"].reading`
  stays None; nothing of the incumbents deleted (the C5 refusals and P8 clause 5 as read in its 2.0). Phase 4 still
  waits on T078-T082.* *DATED NOTE 2026-09-15 (the T077 panel fold): 27 findings stood (2 BLOCKER / 10 MAJOR / 15 MINOR), 9 refuted, every filed fix wrong; the refuters' minimal fixes landed as eleven stacked commits (`475fbe4`..`a5eff15`) and the record's 3.5 items 9-26, 3.7 ((b5) withdrawn under P8 clause 1 -- the oracle is `_auroc2`'s orientation identity, entailed by (a)) and 3.8; VERDICT NOT MET stands, no bar moved. C-R6's "constructed reds R0-R5" reads: R0 / R1 / R3 / R4 fired before the campaign, R2 / R5 were rebuilt after it on `contrast`; the "+5-tick lag control" of its row shifts by five FED reads (5-19 ticks on the run, 3.5 item 19).* *DATED NOTE 2026-09-15 (T078 Part I): pre-registered at `02d677e` with the gates FIRST (G3 stop / G4 void, SYNTHESIS :65) and Part II conditional; the Codex design read refused twice by its usage limit (2.0, 2.10); built red-first (`3485853`..`c955e27`, 2.10's code `455236d`, the latent packer `677ee16`); the campaign read G3 STOP x G4 PASS: (g3c) the estimator sees t +12.6296, FPR 0.0469, MET; (g3) task binding t -3.0213, FPR 0.0469, NOT MET; (g4a) u_t decodes the label t +1.4903, FPR 0.0430, NOT MET; (g4b) the matched content decodes better t +9.2116, FPR 0.0508, MET; leak +0.0151 accuracy points. STOPPED AT G3 (2.6): the live loop's action path does not consume the attention control state above the shuffle floor one tick later, so NO ACS is built, NO third factor, NO wire to brain.py; the row stays `[ ]`; no surface symbol is declared and none is needed; the C5 row's reading stays None; the successor row owes the brain-side stimulus port (docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md 5.2) first. Phase 4 still waits on T079-T082; the successor row that owes the brain-side stimulus port is named in the record's 3.6.* *DATED NOTE 2026-09-15 (the T078 Part I panel fold; `evidence/t078-attention-schema.md` 3.4 iii-iv, 3.5 items 8-21, 3.7): SCOPE of the STOP sentence above -- the barred X in `t078_g3` is the recorded focus band (the audio attend call's; identically argmax(gate); four bands plus the NONE head of the first awake tick) against the next committed action; the only state-wide reading, `gate16`, returned `readable=False` on every seed, and the 15-field `u_t` was never read against the action, so the row says nothing about the rest of `U_FIELDS` (record 3.5 item 15). The DETECTION FLOOR of the (g3) contrast is reported post-hoc in the record's 3.4 (iv) and carried as opening (8) in 3.6; the verdict G3 STOP x G4 PASS, every conjunct, the bar and 2.6's STOP branch are unchanged. The fold is battery-verified only (obs #35).* *DATED NOTE 2026-09-15 (T079; `evidence/t079-agency.md` section 3; pin `tools/t079_agency.json`, head `743176b`, seeds 0-7 x 1,200 ticks x 2 arms on a scratch brain, 1376 s): VERDICT NOT MET. (p) MET; (r) MET on every seed and arm; (g0) the estimator sees -- every joint's intact coupling clears its cause-blind twin (shoulder_pan t +91.3188; shoulder_lift t +21.1497; elbow_flex t +14.6628; wrist_flex t +57.5714; wrist_roll t +71.1811), MET; (a1) the shoved joint collapses, t +13.9998, FPR 0.0430, MET; (a2) preserved elsewhere within the instrument's own dead-band 0.20: shoulder_pan t +18.6279 MET; shoulder_lift t +0.8723 NOT MET; wrist_flex t +4.6820 MET; wrist_roll t +1.3668 NOT MET -- NOT MET on two of four; (a3) the status machine: elbow_flex 'lost' on 8/8 shoved seeds but 4 other losses (shoulder_lift on seed 1 (shoved), shoulder_lift on seed 2 (shoved), wrist_flex on seed 2 (shoved), wrist_roll on seed 6 (shoved)), NOT MET. Per 2.6's NOT MET branch: NO reading, NO symbol printed (`agency_controllability` is NAMED in the conclusion row and printed nowhere, so refusal 3's arming arm sees it), the per-joint efference feed STAYS as the one feed (P2), `AgencyMonitor`'s P3 disposition is EXEMPT with this record as its reason, `route_around` stays unwired; nothing deleted (R1-R6 fired). The row is marked [X] NOT MET.* *DATED NOTE 2026-09-16 (T080): the temporal-binding instrument (`thalamus/binding_instrument.py`; reds R1-R6 fired first) read PASS on a scratch brain fed an ALIGNED bimodal stream (`tools/t080_temporal_binding.json` + `tools/t080_latents.npz`, run 2 on `c3a9fb5`; `evidence/t080-temporal-binding.md` 3.3-3.4): the live cross-modal binder reports ONE object for features presented inside one tick window on 0.26-0.42 of ticks against a membership-matched null (the same audio features and the same ten taught objects, their pairing permuted across ticks) whose q95 is 0.12-0.14 -- (b1) t +13.5395, FPR 0.0469, 8 of 8 seeds; (g0) t +65.35; (b2) 8 of 8 inside the offline null's family-wise band (run 1's per-seed band refused seed 3 at 0.1531 vs 0.1424; amended in 2.11 after the numbers, disclosed). The binding window is the tick (1 on 6 seeds, 2 and 5 on the others -- reported). `CONCLUSION["temporal_binding"]` is CHARGED at 0.2309 with the symbol `crossmodal_bound_rate` (`Brain.state()`, the cockpit row with both sentences, a PROVENANCE line saying the daemon's own independent streams hold it at a chance rate); Phase 4 now waits on T081 and T082. DATED 2026-09-16 (the panel fold; `evidence/t080-temporal-binding.md` 3.5 item 11): the "same view" conjunct is NOT MET AS RENDERED -- the cockpit's `cross-modal bound rate` row is appended below the ACCESS plate's 110 px visible box and its two sentences reach the DOM only in the row's title string, so a viewer sees neither; the row's data and the /api/state PROVENANCE line (both sentences) stand; the clipping is shared with the PCI and integration rows and is owed a cockpit row of its own.* *DATED NOTE 2026-09-16 (T081): the report-order probe (`cortex/report_order_probe.py`; reds R1-R6 fired first, each reddened by a mutated scorer) read PASS on a scratch brain's own stream (`tools/t081_report_order.json`, head `f5cf8b8`, seeds 0-7 × 600 ticks, ten clones of the same pre-step state per awake tick, 349.8 s; `evidence/t081-report-order.md` section 2 and its 2.10 amendment): lesioning the pools whose content the report NAMES moved the reported content on f_named(1.0) = 1.0000 of applicable ticks on every seed, while lesioning a pool the report does not name moved it on f_scr(1.0) = 0.0000 -- (c1)'s seed-paired difference is +1.0 eight times, a ZERO-VARIANCE sample read MET by 2.3's pre-registered degenerate rule (t 0.0000, sign-flip FPR 0.0000, recorded as degenerate rather than as a large t), and (c2) counted 0 scrambled ticks that moved the report on 8 of 8. (p), (r) and (g0) all held. REPORTED, not barred: the dose ladder is a STEP -- 0.25 / 0.5 / 0.75 moved nothing and 1.0 moved everything -- so the flip dose is 1.0 on every seed BY CONSTRUCTION (2.10 (b)) and 2.3's "does the live charge margin predict the flip dose?" cannot be read from this pin. `CONCLUSION["report_order"]` is CHARGED at 1.0 with the symbol `report_order_margin` (`Brain.state()` via the instrument's own `_t081_top_margin`, the cockpit row, a PROVENANCE line with both sentences). THE RETIREMENT LANDED on T034's condition: `reasoning/consciousness_probe.py` DELETED after its 20-seed PCI marker sweep was ported to `cortex/integration_measure`, the chat's "are you conscious" re-pointed at the report channel, `test_red_reachability` at `report_order_probe.`. The EpisodeDriver of the row's wording was NOT routed (2.0, fact 6: it drives a kernel the composite does not include; its exemption row carries the dated current reading). The "same view" conjunct is NOT MET AS RENDERED for this row too -- it is appended below the ACCESS plate's 110 px visible box exactly as the cross-modal row above is, the clipping is the same shared defect, and the /api/state PROVENANCE line carries both sentences. Also disclosed: `report_order_margin` is <= 1 but has no lower bound at 0 -- the slot order is the admission rank, not the charge order, so it read negative on 24 of 100 ticks (range [−0.0664, 0.2083], mean 0.0541); it is not clamped. Phase 4 now waits on T082.*
- **SC-004's open half gets its owner here:** the multi-region clause (1 of 10 regions grows) is
  re-scoped to "a second region receives grown units that a live path READS" and is owned by C1a (readers
  exist once ports are instantiated) together with L2's growth through the 007 seam — the seam rank 1 built
  and reverted for want of a reader.
- **Exit:** `evidence/c-exit.md`.

### Phase 5 — Self-improvement (US4 core; SC-008) — after T071 AND Phase 4

*DATED NOTE 2026-09-10: the T071 conjunct is DISCHARGED (evidence/s2-exit.md); Phase 5 still waits on Phase 4 -- T070, T077-T082 all OPEN.*

*DATED NOTE 2026-09-15 (T070 panel of 2026-09-14): T070 ran 2026-09-10 (campaign run 2 at `c1aac92`, pin `ab09b9b`;
evidence/t070-integration-kernel.md 3.3: verdict NOT MET; tasks.md T070 `[X]` at `c56a730`); the port's fate per 3.4's
adjudication: DELETED under 2.6 in `994ff0b`. Phase 5 still waits on Phase 4 -- T077-T082 open, no c-exit.*

- **T072** proposer un-latched (rank 7 incl. its `note_cycle` clause: a futile set counts as exhausted for
  its source); red: `distinct candidate keys reachable` ≤ 1 (red today); positive control: two hand-named
  operators yield 2 keys. **T073** evaluator cascade on harness-only evidence, stage 2 = Phase 4's
  benchmark (why this phase waits). **T074** archive + MAP-Elites cells (adopted, disclosed — not a P8
  invention) as an IMMORTAL record class with its bytes/tick ceiling and P4b red. **T075** cockpit rows.
  **T076** — its mechanism landed as W1 and was RETIRED 2026-09-04 on its own measurement (owner decision, P8 deletion-on-a-null); the red was reachable and the mechanism was green, but the capability was never demonstrated. Its exit row is WITHDRAWN with it. `evidence/w1-goal-curriculum.md` is the record. **T083** US4 exit.

### Phase 6 — Polish — T084–T087

T084 cockpit sweep (+ SC-011/012/013 rows with PROVENANCE); T085 per-tissue scale derivation; T086
consistency sweep, which also owns the audit's "gets wrong" items (4) and (6) — the supersession record for
specs 007/011/014 and the tasks.md row count; T087 full run-through, final battery, `policy_manifest
--certify` (after H1), close-out. FR-031: every P8 invention this plan lands has its register row (W3, L1b,
L2, T066, T073, T077, T078).

---

## PART E — SEQUENCING, DEPENDENCIES, WHAT IS DROPPED

**Critical path (commits), with the earliest start of each parallel arm marked:**
W3 → ~~W1 (+T076's red)~~ RETIRED 2026-09-04 → ~~W2 (no widening)~~ REFUSED ON MEASUREMENT 2026-09-05 → [the JEPA-collapse arc — SHIPPED at `dbcf153` (record `evidence/jepa-collapse.md`): red (a) cleared brain-free 3/3 and in-brain 3/3 by direction; the in-brain `latent_std` floor (b) met 1/3, then RETIRED-AND-REPLACED by measurement in `evidence/jepa-live-floor.md` (2026-09-05: the consecutive-window gauge reads the stimulus; the replacement criterion met by construction on the calibration seeds, read evidentially on seeds 3–4 — MET on seed 4 (×0.133), NOT MET on seed 3 (×0.062 against the ablation's ×0.056): (b) under the replacement is RED on 1 of 5 seeds and OPEN; a reservoir batch that fixed the repeated-world red did NOT land — it loses the planted order read — and stands as a strict-xfail red in `tests/test_jepa_collapse.py`)] → L1b → L1c → L2 → L4 (with its ceiling) → T069 → S1 → S2 → S3 →
S4 → S5 → S6 → T062 → T063 → T071 → T072 → T073 → T074 → T075/T076 → T083 → H1 → T084–T087.
Parallel arms: L1a ∥ L3 ∥ the oracle may start at W3 (they do not wait for Phase 1); the C-chain
(C0 → C1a → C1b → C1c → C3 → C2 → C4) may start at S1 (O5(b)) and must finish before T072.

- *DATED NOTE 2026-09-09 on the ordering above: **S6 is DISCHARGED BY MEASUREMENT, not by a wire.** Gate Zero
  read the kernel wire's Amdahl ceiling at the deployed width and the conversion was DECLINED under specs/008
  FR-005 (evidence/s6-kernel-wire.md; the S6 bullet's annotation carries the numbers). The critical path is
  unchanged in ORDER — **T062 is next** — and the kernel's exemptions are now dated to it.*
- *DATED NOTE 2026-09-10 on the critical path above: **T071 CERTIFIED 2026-09-10**
  (`evidence/s2-exit.md`) — the ordering `… → T062 → T063 → T071 → T072 → …` advances past T071 and
  **"Phase 5 waits for BOTH T071 and Phase 4"** now waits on Phase 4 alone. O5(d)'s clause is
  DISCHARGED, not rescinded: T072–T075 and T083 are unblocked because all three of T071's named
  items were READ and recorded with their measurements — prior separation NOT MET, the paging
  verdict a LOSS with SC-007 satisfied by its second branch, and the region charge gate MET at its
  floor under reading (A) with reading (B) reported NOT MET beside it. **The gate is discharged by
  the certification's EXISTENCE, not by a pass** — the s1-exit precedent, where US2 exited with its
  headline criterion UNMET and T068, gated on it, ran. T070 remains OPEN and is not a gate for this
  exit (O5(b)).*
  *PANEL 2026-09-10 (pre-registration-honesty lens): reading (A) ruled a relabelling -- item 3 is
  recorded NOT MET under (B), 8 of 16 at d9dd3c0; the FR-015 ratchet (8 >= 8, 0 <= 0, identity 6 of 6)
  holds and the floor pin is unchanged; all three named items therefore read NEGATIVE; the
  gate-by-existence is undisturbed.*
- Phase 1 is startable now in-process (the latch is per home); O1 unblocks the LIVE brain's measurement.
- Phase 3's paging runs at a constructed width and does not wait for Phase 2's growth; its ORDER after
  Phase 2 is the plan's choice (O5(f)), not a hard gate.
- Phase 5 waits for BOTH T071 and Phase 4.
- **ADDED 2026-09-14:** N1 -> N2 -> N3 -> N4 -> N5 (T096-T100) is a planned arm after
  Phase 2's exit disposition; N4's brain comparison waits for its named Phase 4 consumer.
  N5's live wire waits for SC-014's numerical/behavioral/persistence/resource evidence; its
  mutation registration waits for T073/T074 and Phase 5's standing prerequisites. A failed
  hybrid does not delay unrelated work; it receives the recorded null/loss disposition above.

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
| any unit count reported as growth before that count is shown to change a measured outcome | REFUSED in headlines and PROVENANCE | P0's honesty guard (reporting rule re-based 2026-09-08 from "until SC-012a fires" — it fired, QUALIFIED — to the permanent condition; gated by `tests/test_units_admitted_wording.py`, O4; Part C carries the dated decision) |

---

## PART F — THE REDS REGISTER (pre-registered here; each lands with its instrument)

Every seed-paired red: S = 8, paired one-sided t > 1.895 (7 d.f., α = 0.05), ties counted, permutation
false-positive rate printed beside the verdict; no statistic normalized by a quantity a control destroys;
no majority-of-seeds rule anywhere.

| id | red (what makes it fail) | control (not the method) | lives in |
|---|---|---|---|
| N-R1 | analytic decay/convergence, finite/divergence or forced-energy tolerance fails; a constructed projection/advection/timestep fault is missed | correct Taylor-Green and forced reference runs, refinement | N2 / T097, PLANNED |
| N-R2 | initial-state dependence survives the registered washout, or ordered temporal information is no better than its shuffle | same-input distinct-state twins and learnable temporal positive control | N2 / T097, PLANNED |
| N-R3 | fresh-process spikes/decisions/updates differ, a meaningful omitted/corrupt carrier is missed, or migration loses state | uninterrupted twin, valid versioned round-trip and corruption refusal | N3 / T098, PLANNED |
| N-R4 | full NS fails the prospective benefit over either equally tuned LSM or diffusion-only control, or regresses past a registered tolerance | both cheaper controls under paired protocol and measured budgets | N4 / T099, PLANNED |
| N-R5 | disconnecting/shuffling feedback or silencing relevant grown units does not reduce attributed useful task gain | intact feedback and equal-count pre-sized-unit silence | N4 / T099, PLANNED |
| N-R6 | live path bypasses a gate, resources exceed budget, or an invalid candidate/premature promotion is admitted | compliant live candidate, measured cost control and refused invalid diff | N5 / T100, PLANNED |
| W-R1 | mean `curiosity_mag` (last 200 wake rows, 3000 and 6000 ticks) under W1+W2 does not exceed the time-shuffled-context world, per seed | the same world, context order shuffled | W3 |
| W-R2 | category entropy not above the W3 baseline **at a NAMED window, PER SEED**, or live next-category accuracy not above its marginal-preserving shuffle (floors reported within-arm). **AMENDED 2026-09-03 on W3's measurement:** entropy is NOT a stable property of a run -- the same three seeds on one world moved in OPPOSITE directions between the first-1000 and last-1000 wake-row windows, spanning 0.0 to 2.6 bits -- so a bar that does not name its window and apply per seed is measuring the window. Pre-registered window = the FIRST 1000 wake rows; the 6000-tick column is the clean one (at 3000 the two windows overlap by ~517 rows). `evidence/w3-world-baselines.md` §4 | the shuffle | W3 |
| W-R3 (T076) | **the CURRICULUM'S OWN PROPOSED GOAL** unchanged with `coverage_reward` neutered, OR changed when an unrelated scalar of comparable magnitude is neutered. **AMENDED 2026-09-03 before the build:** the original row -- "goal selection unchanged" -- is VACUOUS, measured: on the incumbent path with no W1 at all, neutering `coverage_reward` already changes 18/60 actions and takes `_z_goal` 60/60 -> 0/60. A red satisfiable without the mechanism it tests is not a red for that mechanism (P8 clause 1). The observable is therefore `propose()`'s returned goal cell, which the incumbent cannot produce, and the row now carries the SPECIFICITY control the original lacked | the un-neutered run, AND an unrelated-scalar neuter that must NOT move it | W1 |
| L-R1 | paired t of (oracle − own persistence floor) not above 1.895 under L1a's window, or LIVE falls below the incumbent window past the bar | the incumbent window, same learner | L1a — **read 2026-09-05: the first half NOT MET (t −2.51; the oracle sits AT its floor, from −0.43; 0 / 6 / 2 with a 1e-6 tie tolerance), the second half MET; the raw LIVE gain (t +2.27) is confounded with the slice the window redefined (majority-floor-relative t 1.32; learner-only on the same rows: t = 1.002, exact p = 0.1719, 5 / 3 / 0, mean +0.088); L1a LANDED on the oracle's majority-answer → persistence-answer reading — the session's call, owner may revert `3979001`; strict xfail in the tree, guarded on the perception packs** |
| L-R2 | L1b does not beat the batch control, or an untrained twin is not below its own persistence floor while L1b is | batch solve under L1a's window; frozen-readout twin | L1b — **read 2026-09-05: NOT MET on both candidates (Oja t -1.96, FELL past the bar; BCM t -1.36); the letter's second conjunct NOT MET (both candidates below persistence 8 of 8, t -3.81 / -3.22; the twins, re-drawn at each open refit, t -13.6 / -23.6) and the lowered majority bar pre-registered beside it NOT MET; the rule DELETED on its null** |
| L-R3 | a growth event without a preceding synaptic plateau; neutering L1b does not raise growth | the grower with L1b off | L2 |
| L-R4 | the constant-percept arm admits more than 1 unit, or the structured positive control admits only 1 | a structured stream under the same instrument (must admit > 1) | L3 — **MET 2026-09-05 (1 alive unit on 8 of 8, answering the majority; SYNTH 8–29; re-read under the N-independent equality form: the same)** |
| L-R5 | the L4 record class exceeds its bytes/tick ceiling without tripping | constructed overload | L4 |
| SC-011 | structured stream grows no more units than its own marginal-preserving shuffle | the shuffle | `tests/test_real_percept_stream.py` |
| SC-012a | acc(N=16) − acc(N=1) not above the bar, or acc(16) not above its majority floor, or the label-shuffled grid rises | the shuffled grid | `tests/test_real_percept_stream.py` — **FIRED 2026-09-07 on the SECOND reading. First reading (a) t 5.024 / (b) t 3.936 / (c) VETOED t 1.928 was RETRACTED: `acc(1)` was an ARTEFACT — all-zero rate rows resolving to column 0, a class the stream never emits (45/60 rows at seed 0, 60/60 at seed 1) — and (a) and (c) both subtracted it. `NGSNN.decide` fixed the scoring path (`evidence/scoring-path-abstention.md`); the grid was re-run and reads (a) t 2.596 p 0.0234 +0.151, (b) t 4.011 p 0.0039 +0.194, (c) the shuffled grid FELL (t −2.346). The fix CUT the effect two-thirds and it still clears while the null cleanly reversed. Qualified: "16" realizes 9–14 units, no single doubling clears, forced width does not beat the incumbent, nothing reaches persistence. xfail REMOVED; instrument checks still raise `Failed` so a scoring regression cannot be filed as a criterion result in EITHER direction** |
| SC-012b | correct answers do not drop when grown units are silenced, or drop equally when pre-sized units are | pre-sized-unit silencing; DSL neutered | C2 |
| SC-013 | the reasoning criterion (ARC-eval at protocol + everyday battery, reported with its floor) cannot fail on a constructed input | scrambled protocol | C2 |
| S-R1/R2 | rounds budget never trips on constructed overload; a paged-out spike is dropped silently | constructed overload; a resident twin | T066 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| S-R3 | paging loses to forced residency at the comparison width | forced residency | T068 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| S-R4 | the wired kernel does not change the tick's output when neutered, or tick time does not fall | the object-tick twin | S6 — **re-registered 2026-09-09 with its widths named** (evidence/s6-kernel-wire.md): conjunct A (a neuter of the kernel changes the tick's output) at the TRIGGER WIDTH where the kernel first produces a live output; conjunct B (tick time falls) at N*; the object-tick twin as a SEPARATE PROCESS. **Not runnable at the deployed width** — the measured Amdahl ceiling there (0.005–0.017, bands inside 0.04) sits an order of magnitude under the 0.15 bar, so conjunct B has no reachable red at 177–189 units. The row is armed, not retired: it fires when the trigger width is reached. **2026-09-09 CORRECTED (panel fold, C7/C12):** this cell read "sits under the instrument's own floor (~0.008) and an order of magnitude under the 0.15 bar"; those seven words are deleted and recorded here rather than removed silently. The clause quoted ONE reading's sham point estimate as a property of the instrument. The sham is recomputed per run by the gate's own definition ("the sham arm removes NO work, so whatever it reads is the instrument's own floor") and it flips sign: the five recorded values are −0.0009 / +0.0005 / −0.0078 / +0.0185 / −0.0140 and the gate's "ceiling exceeds 2x the floor" flag prints True on 3 of them, so the sentence also contradicted itself — 0.017 was never under 0.008. No replacement constant is published. The DECLINE is unaffected: it is read from 2.3's rule (upper 95% bound < 0.15), never from the floor. **Also withdrawn (C3/C8):** "bands inside 0.04" — 0.04 is the band upper of one unlogged reading (the seed-0 quiet extra, 0.0383), and five of six fresh determinate bands breach it; the reproducible statement is that every determinate band upper sits far under the 0.15 bar. **Reading note 2026-09-09 (C25):** "conjunct A/B" here name the PASS conditions; the red is the negation of either, exactly as this column's first clause states. |
| S-R5 | the live tick over a PAGED substrate is not output-identical to the resident twin, or the pager's bound is declared discharged while `NeuronPager.tick` is not the live tick's residency step (added 2026-09-08, T066's fold) | the resident twin on the live tick; the un-wired pager | S6 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| S-R6 | a merge or the fresh-tier read loses or mis-applies a block delta: the paged run reading base + deltas (or merged records) is not byte-identical to the resident twin built directly on the post-delta edge set, a relocation proof passes on a dropped delta, a stale base_generation delta is applied, a pre-existing segment file changes, or a rebuild adopts a record no ADOPT record fenced (added 2026-09-08, T067) | the resident twin on the post-delta edge set; the DROPPING fold; the proof-neutered fold | T067 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| S-R7 | the merge budget never trips on constructed overload (deferred == 0 with more dirty blocks than budget_blocks), or the merge is reachable from the wake tick (added 2026-09-08, T067) | the calm merge; the BUDGET-BLIND merge; the wake-tick ratchet | T067 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| S-R8 | a reorder changes or loses a record, or reads per round do not fall to 1 on a co-fetched index-distant stream, or fall as much on a random-fetch stream, or fall under the identity order (added 2026-09-08, T067) | the random-fetch stream; the DROPPING reorder; the identity order | T067 **RETIRED WITH THE LAYER 2026-09-09 (S-R3 FIRED)** -- T068 measured the paging layer a LOSS and the layer is deleted (evidence/s5-paging-null.md 3.4/3.5); a red whose mechanism no longer exists cannot go red, and P8 clause 1 deletes such a criterion rather than weakening it. The row STAYS as the record of what was pre-registered. S-R3 is the one that FIRED: paging did not beat forced residency under the pre-registered rule -- the memory claim was met on every conjunct and the 3.0x cost bound was missed at 35.2x raw / 26.0x corrected. |
| C-R1 | a pinned-payload neuter on an instantiated edge changes nothing downstream | the un-neutered run | C1a |
| C-R2 | the 20-item battery does not move, or the 4-item positive control drops | the positive control | C2 |
| U-R1 | `distinct candidate keys reachable` ≤ 1 | two hand-named operators → 2 keys | T072 |
| X-R1 | the brain at the constructed width does not beat its own PRE-SIZED twin at the bar, or a brain number is reported without its protocol and provenance; **and the refusal arm:** no human number under a matching protocol can be obtained → the comparison is REFUSED and "superhuman" is not a claim this project may make on that task | the pre-sized twin at the same width; the human baseline re-measured | X1 |
| L-R6 | a constructed compaction / migration / paging / budget pass evicts or rewrites an IMMORTAL record, or the cross-process restore does not return it byte-identical, or the storage growth curve is projected rather than measured | a MORTAL class in the same store, evicted in the same pass | L5 |
| L-R7 | a constructed reclaimer that destroys a record of a `reclaimable=False` class without consulting `retention_of` is not caught, or a real one is silenced by a `KNOWN_UNLABELED` entry carrying no owning row and no retirement condition | a COMPACTABLE tag reclaimed in the same pass, which must still be permitted -- a fix that stops all consolidation is refused by its own control | L6 |
| L-R8 | the append budget flags a class's ORDINARY growth (the measured 52,763 -> 77,998 B climb), or fails to flag a real spike once the window is all-real, or a stale seed alarms for more than a handful of ticks | the same series under the retired fixed bar, and a cold tag that must still be bounded from its first tick | L4b |
| C-R5 | an instrument that cannot be made to FAIL on a constructed input (a marker, not a dissociation) is not deleted; or a surface prints one of the seven readings without its licensed sentence and scope; or the brain answers a consciousness question with anything but a measured quantity and its scope; or any aggregate score is computed across instruments | each instrument's own dissociation input in the same command; T081's report-order probe | C5 *DATED NOTE 2026-09-10 (T070): the dissociation's constructed red R2 -- a partition that did NOT cut reads above the null -- FIRED on the fixture; on the campaign (b6) read NOT MET (t(INTEGRATED - null) +5.0732; partitioned <= null 6/8; rate-matched <= null 4/8; `tools/t070_pci.json`); no surface prints a T070 reading -- the cockpit PCI row stays unfilled and `PROVENANCE["pci"]` reads NOT PRODUCED.* *DATED NOTE 2026-09-15 (T070 panel of 2026-09-14): the panel's pre-registration-honesty lens ruled the record's hold a post-number change of 2.6's routing (F10, BLOCKER); 2.6 EXECUTED in the fold's commit 2 -- `cortex/integration_kernel.py` and its tests DELETED, its CORE/REGIONS/EXEMPT registrations removed; re-landable from the tree at c1aac92 (history a2cf76d..c1aac92); the pin `tools/t070_pci.json` and the record stand (`evidence/t070-integration-kernel.md` 3.4, 3.7); the successor pre-registration re-specifies (b6) clauses 2-3 and (b4) as a reported quantity.* |
| C-R6 (T077) | **added 2026-09-15 at T077's pre-registration (`evidence/t077-metacognition.md` 2.3-2.4; FR-031's register row).** The metacognition instrument reads NOT MET or REFUSES when: the second-order read's type-2 discrimination (auroc2) over the scratch brain's prequential readout stream does not clear chance at the bar (b1); the shuffled pairing is not below it (b2, the C5 dissociation input); calibration does NOT collapse when the second-order read is removed in vivo (b3); first-order accuracy is NOT preserved under that lesion by two positive equivalence contrasts with the run's own seed dispersion as margin (b4); the readout is not above its own persistence predictor (b6); or any seed's stream cannot resolve tracking on its own permutation null (q95 above 0.5 + rho, rho = 0.25 declared) -- a REFUSAL, not a reading. Constructed reds R0-R5: a perturbed recorded read is caught; 2 errors in 400 refused; a relay reads a zero contrast; the shuffle sits inside the band while the oracle sits outside; the monitor's own synthetic world is not relay-proof; a lesion whose read equals the intact read reads a zero contrast | the shuffled pairing; the oracle (conf := outcome); the offline first-order passthrough; the lesioned twin; the +5-tick lag; the per-input neuters by exact replay | `cortex/metacognitive_monitor.py` (`track_error`, the campaign), `tests/test_metacognition_instrument.py` |
| C-R7 (T078) | **added 2026-09-15 at T078's pre-registration (`evidence/t078-attention-schema.md` 2.3-2.4; FR-031's register row, invention #8).** PART I, the gates first: the row STOPS when the live loop's attention control state at tick t (the focus band) carries no mutual information about the action committed at t+1 above its own shuffle floor (G3: paired t over 8 seeds of MI - q95 not above 1.895), and is REFUSED -- not stopped -- when the estimator's positive control MI(action_t; action_t+1) does not clear its own floor (a blind estimator) or any seed's reading is below 10 pairs per observed cell; VOIDS when the candidate `u_t` decodes the perceived frame's label above its label-shuffle band and the matched 15-d content decode cannot be told from it (G4). Constructed reds R1-R6: an i.i.d. pair reads at or below its floor; the positive control reads AT the floor on an i.i.d. action stream; 145 pairs over 39 cells refuse; a planted label decodes above its band, permuted labels inside it; two equal control states with different latents give a bit-identical `u_t`; a one-hot-of-label `u_t` VOIDS and a noise `u_t` passes -- each reddened in fact by a mutated scorer. PART II (built only on G3 PASS and G4 not VOID; 2.11): RED-1 the pinned third factor changes nothing; RED-2 raw-residual precision equals schema precision (the invention is DELETED); RED-3; the severed report keeps the prediction. *DATED 2026-09-15 (the T078 Part I panel fold): this cell mirrors `evidence/t078-attention-schema.md` 2.3-2.4 and was written with them at `02d677e`; amendment 2.10 (`1094234`, code `455236d`), dated and appended BEFORE any brain recording existed and justified on the builder's constructed control-axis test, replaced the readability predicate for the (g3c) POSITIVE CONTROL only with `N_MIN_PER_CELL` pairs per ACTION CLASS on the next-tick axis -- `mi_read(..., min_rule="per_class")` on the control call inside `t078_g3` and nowhere else; the PRIMARY reading keeps this cell's per-observed-cell rule as pre-registered. Under this cell's unamended wording the control refuses on all 8 seeds (126-146 observed action x action cells on 540-549 pairs; `run_t078_campaign`'s "REFUSED at (r)" block) and the verdict would have read REFUSED rather than STOP; this was the one surface still stating the superseded rule (record 3.5 items 2 and 14).* | the permutation floors; the label-shuffle bands; the same-tick MI; the action-autocorrelation control; the matched-dimension content decode; the determinism twin; in Part II the capacity-matched sham and 9.2's four collapse controls | `cortex/attention_schema.py` (`mi_read`, `decode_heldout`, `control_state`, the recorder, the campaign), `tests/test_attention_schema.py` |
| C-R8 (T079) | **added 2026-09-15 at T079's pre-registration (`evidence/t079-agency.md` 2.3-2.4; FR-028's agency item, register #6; the C5 row `agency`).** The row is REFUSED when the determinism twin differs, when any rotatable joint is unscored on any seed or arm (n < warmup, command variance under the floor, a latched body, a tap error), or when the intact coupling of any joint does not clear its cause-blind twin (u_j shuffled across passages, 2,000 draws, q95) at t > 1.895 over 8 seeds -- a blind estimator, never a null; it reads NOT MET when, with one joint (elbow_flex) shoved by an external force of 2 x the motor step after every passage through the envelope's own test-only exafference port, the paired collapse of that joint's coupling (intact - shoved) is not above the bar, or any unshoved joint's drop exceeds the instrument's own dead-band (0.20) at the bar, or the hysteresis machine does not read 'lost' on the shoved joint on 8/8 seeds with no other loss; PASS otherwise. Constructed reds R1-R6: a decoupled channel reads at its floor and a coupled one above; one high reading never revives 'lost' and one low never loses 'ok'; an unexercised channel is unscored, never lost; the token parser refuses a foreign joint; the per-joint fold wraps the seam; the dissociation fires on a constructed body and not on its intact twin -- each reddened in fact by a mutated scorer. | the shuffle floor (the cause-blind twin); the intact arm on the same seed and stream; the four unshoved joints; the module's own constructed trajectory; the shove ladder 3 / 6 / 12 deg (reported); the determinism twin | `cortex/agency_monitor.py` (`t079_record_brain`, `t079_read`, `t079_floor`, `t079_contrasts`, `t079_reds`, the campaign), `substrate/efferent_boundary.py` (`proprioception_deltas`, `intended_deltas`, the per-joint `last_copy`), the feed switch in `brainstem/autonomous_loop.apply_observed`, `tests/test_agency_monitor.py` |
| C-R9 (T080) | **added 2026-09-16 at T080's pre-registration (`evidence/t080-temporal-binding.md` 2.3-2.4; FR-028's binding item, register #6; the C5 row `temporal_binding`).** The row is REFUSED when the determinism twin differs, when any seed does not teach the binder all K = 10 objects within the warm-up cap or any scored awake tick lacks a modality, when the per-modality recognition decomposition disagrees with the binder's recorded report on any tick, when the classifiers' congruence on the ALIGNED stream does not exceed the NULL stream's at t > 1.895 (the alignment did not reach the brain -- a blind apparatus, never a null), or when the in-vivo NULL arm's bound fraction falls outside the offline null's [q005, q995] band on any seed; it reads NOT MET when the ALIGNED arm's bound fraction (one reported object on a tick whose vision digit and audio word were presented together) does not exceed the q95 of the membership-matched null (the same audio features and the same taught K, their pairing permuted across ticks, 2,000 draws -- the DERIVED margin) at t > 1.895 over 8 seeds; PASS otherwise. Reported, never barred: the binding window from lag arms (1, 2, 5 ticks), the scrambled-phase arm, the false-binding fraction, the yield. Constructed reds R1-R6: every null draw keeps the audio multiset; the decomposition equals `bind()` on 200 pairs; the window reader reads a constructed window; the positive control refuses an independent stream; the dissociation reads NOT MET on within == null and MET on within > null; the agreement band refuses an outlier -- each reddened in fact by a mutated scorer. | the offline membership-matched null; the in-vivo permuted arm; the classifiers' congruence; the lag arms; the scrambled-phase arm; the determinism twin | C4 |
| C-R10 (T081) | **added 2026-09-16 at T081's pre-registration (`evidence/t081-report-order.md` 2.3-2.4; FR-028's report-order item, spec.md US4 scenario 3; the C5 row `report_order`).** The row is REFUSED when the determinism twin differs, when on any seed a clone of the arm's workspace competition state re-stepped does not reproduce the report channel's record on every awake tick, when the named or the scrambled lesion is applicable on fewer than half the awake ticks, or when the report carries fewer than 2 distinct contents or no tick-to-tick change (a constant report cannot be read for its generation order -- a blind apparatus, never a null); it reads NOT MET when the fraction of ticks whose reported CONTENT changes under the NAMED lesion (the reported content's pool charge zeroed and its submissions dropped on a clone) does not exceed the fraction under the SCRAMBLED lesion (the same operation on a randomly chosen unreported pool) at t > 1.895 over 8 seeds, or when the scrambled lesion changes the reported content on any tick of any seed (a count, exact); PASS otherwise. Reported, never barred: the dose ladder (0.25 / 0.5 / 0.75 / 1.0), the flip dose against the live top-slot charge margin, the strength changes. Constructed reds R1-R6: the clone leaves the original and its subscribers untouched; the dissociation reads on a constructed workspace; a canned reporter reads NOT MET (the scorer must read the report channel, not the state); a hallucinating reporter reads NOT MET; a shared-RNG clone refuses; the ladder reader reads a constructed flip dose -- each reddened in fact by a mutated scorer. `reasoning/consciousness_probe.py` is retired in the surfaces commit on PASS and NOT MET alike (its 20-seed PCI sweep ported into `integration_measure` first); the EpisodeDriver is not routed (measured: no path to the composite's report; a dated CURRENT READING on its exemption row). | the same-state clone re-stepped; the scrambled lesion; the canned and hallucinating reporters; the dose ladder; the determinism twin | C4 |
| C-R11 (T082) | **added 2026-09-16 at T082's pre-registration (`evidence/t082-social-cognition.md` 2.3-2.4; FR-034; the C5 row `social_cognition`).** The row is REFUSED when the determinism twin differs, when any scripted statement fails the belief grammar or any episode's prediction is missing, or when the scripted other agent is not a false-believing one (fewer than 30 % unwitnessed episodes, or an unwitnessed episode on which its look equals the world -- a blind apparatus, never a null); it reads NOT MET when the composed recursive tracker's first-order prediction of the agent's look does not beat an EQUAL-CAPACITY WORLD MODEL (the same class fed the same statements with the absence markers ignored) at t > 1.895 over 8 seeds, or when that control's prediction differs from the world content on any episode (the replacement must remove the gain exactly -- a count); PASS otherwise. Reported, never barred: the charge A/B of the declared wire `recursive_tom.belief_out -> brain.social_risk_in` (live vs pinned caution trajectories, the constant-consumer specificity control), the spurious nested-divergence count under the merged feed, the social-risk trajectory. Constructed reds R1-R6: the grammar refuses a non-statement; the control is a world model on a constructed script; the dissociation reads on mixed and all-witnessed scripts; the charge A/B refuses a payload-ignoring consumer; the merge reproduces the two retired trackers' readings on 200 witness scripts; the symmetric feed kills the spurious divergence the old subject-only feed reconstructs -- each reddened in fact by a mutated scorer. The merge (ONE tracker), the wire and the charge evidence land in the build on every branch (P2, FR-034). | the equal-capacity world model; the world predictor; the all-witnessed script; the constant consumer; the old subject-only feed; the determinism twin | C4 |
| U-R2 | a fabricated candidate score changes the verdict | harness-only evidence | T073 |
| H-R1 | **added 2026-09-09 (T063, SC-006/FR-022 — the register carried no row for either).** Two reds, both must be constructible and RUN: **red 1** — a seeded population that never moves off its seed (the FROZEN `lr = 0` arm: effective weight change 0 and both statistics' distance 0, the verdict failing and naming the pin the harness itself applied, never naming pinning for a cancellation, a below-quantization update or an absent spike pair); **red 2** — the deliberately WRONG prior does not UNDERPERFORM its within-sign degree-matched null (paired one-sided t over acc_random − acc_wrong ≤ 1.895), or the reversed-laminar table fails its validity precondition (its band-pair weight-share matrix no farther from SEEDED's, in PSI, than RANDOM's realised matrix is) — in which case red 2 is NOT READ and no attribution to the prior is made. The bar the arms must clear is SC-006's two DIRECT contrasts, seeded-vs-random and seeded-vs-wrong, under Part F's rule above | the PLANTED positive control (a table wired to make the task easy under the same nuisance constraints; it must beat RANDOM at the same t bar or the harness is BLIND and no verdict is read) AND the FROZEN twin (the same seeded table at `lr = 0`, which is also the learner's own retention criterion) | `cortex/prior_drift_harness.py` (with `substrate/kernel_plasticity.py`, the learner the row cannot run without) — widths NAMED BY THE PILOT RULE: the full ingested width and band-balanced induced subgraphs at 1/2, 1/4, 1/8, W = the LARGEST whose measured tick cost projects the campaign under 6 hours on this host, recorded with the projection before the campaign. ***FIRED 2026-09-09, and the row is RESOLVED.*** Both reds were constructed and RUN at K = 8, D = 4, W = 1.0 (14,215 units / 70,001 edges, 8 seeds, `tools/t063_campaign.json` at head `f37fe90`). **RED 1 FIRED** on the FROZEN `lr = 0` arm on every seed and named the pin the harness applied — executed 120, effective 0, drift 0 on every seed; attempted 174,781-184,936 — with its control holding (the SEEDED arm moved: S1 distance 0.107–0.172 on every seed). **RED 2 DID NOT FIRE**: the WRONG arm was VALID (PSI 6.175870 against RANDOM's 2.271100–2.302899, measured on all 8 seeds), and t(random − wrong) = **−2.5459** — underperformance is NOT ESTABLISHED because the corruption OUTPERFORMED the null, so nothing is attributed to the prior. **The PLANTED positive control held**, t = **+9.2601** (FPR 0.0508), so the harness is not blind and the verdict is read. **The bar was not cleared on either direct contrast**: seeded−random **−0.2277**, seeded−wrong **−5.2271**. **The FROZEN twin's second job resolved too**: t(d_f) = **−13.0154**, so `substrate/kernel_plasticity.py` — cited in this cell as "the learner the row cannot run without" — is **DELETED** on that criterion, and the harness is kept as a no-learner instrument whose conjunct (a) now reads NOT READ. This row's reds are therefore not re-runnable as written: red 1's subject was a pin against a LEARNING arm, and its slot now reads the no-learner refusal instead. ***APPENDED 2026-09-10 (T063 panel of 2026-09-09), the cell above kept VERBATIM:*** the sentence "This row's reds are therefore not re-runnable as written" OVER-REACHES and is corrected here rather than removed. Only RED 1 lost its subject: its pin stood against a learning arm and there is no learning arm (the harness's own self-check docstring at this head reads "WHAT IS GONE: red 1"). RED 2's contrast — the paired one-sided t over acc_random − acc_wrong against the 1.895 bar — needs NO learner, is conjunct (d) of the verdict, is named among what SURVIVES in the module docstring, and `--campaign` still computes it. Two qualifications ride with that: red 2 as literally written names the REVERSED-LAMINAR table for its validity precondition, and that table was retired on measurement by amendment 2.12 in favour of the BAND-PERMUTED-TARGETS arm, so what remains runnable is red 2's PREDICATE under 2.12's arm, not this cell's 2026-09-09 wording; and an evidential re-reading requires `--campaign`, which WRITES `tools/t063_campaign.json` and must not be taken against that pin, which stays the record of the learner's run. No self-check number is entered here: §3.5(9) already fixes that the offline reading is a predicate on a constructed input, not an evidential reading. ***APPENDED 2026-09-10, second sentence:*** the surviving no-learner instrument, run on `data/h01/h01_c3_graph.npz` at the pinned K = 8, D = 4, W = 1.0 over seeds 0-7, reads (c) NOT MET at t = +0.8143 and `verdict()` prints SC-006 = NO VERDICT with (a)(b)(d)(e) NOT READ — so an evidential re-reading of (b)(c)(d)(e) on H01 needs a NEW pre-registration of the capacity guard, not a re-run of this row. The pinned campaign's own (c) = +9.2601 is untouched by that: it is the learner-ON run and it stands. |

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

## PART H — OWNER DECISIONS THIS PLAN NEEDED (all given 2026-09-02; the W2 questions and the two revert options resolved 2026-09-05; the 2026-09-05 standing authorization makes every later call the session's)

**Open since 2026-09-05 (`evidence/w2-context-latents.md` §5):** **Q1** — the Phase 1 null disposition
sends L2 at a prediction error that is measured collapsed; ratify (a) the JEPA-collapse arc BEFORE L2 and
before any task whose red reads `prediction_error`, (b) a different learning signal for L2, or (c) the
sentence as written (not recommended). **Q1 RATIFIED (a) BY THE OWNER IN CHAT, 2026-09-05** ("Q1 (a),
start the JEPA-collapse arc"): the JEPA-collapse arc runs now, before L2 and before any task whose red
reads `JepaWorldModel.prediction_error`; its record is `evidence/jepa-collapse.md`. **Landed 2026-09-05
at `dbcf153`:** the world model reads succession (planted world 3/3 at 12 cycles; the live stream 3/3
seeds, 9/9 block-pairs); the in-brain `latent_std` floor met on 1/3 seeds and kept OPEN; the ship
decision on (a) alone was NOT pre-registered and is the session's, disclosed in the record's §13.4 —
the owner may revert `dbcf153`. **Follow-up, owner-instructed ("Do all this", 2026-09-05), landed in
`evidence/jepa-live-floor.md`:** (b) is RETIRED-AND-REPLACED (the consecutive-window gauge measures the
stimulus — an untrained twin reads 0.01–0.07, the trained model over distinct patterns 0.036–0.045 on every
seed; the replacement, the trained/fresh spread ratio over distinct patterns ≥ ×0.10 with the ablation as its
red, is MET by construction on the calibration seeds and read evidentially on seeds 3–4 — MET on seed 4 (×0.133), NOT MET on seed 3 (×0.062 against the ablation's ×0.056): (b) under the replacement is RED on 1 of 5 seeds and OPEN); the
reservoir batch built to turn the repeated-world red green did NOT land (it loses the planted order read —
P8 clause 5), so that arc's PASS was not reached and **the revert option on `dbcf153` STAYS OPEN**. L2 is
scheduled by the same arc (Phase 2's L2 row): its trigger is L1b's synaptic plateau and the signal it may
target is `JepaWorldModel.prediction_error` (while `dbcf153` stands), after L1b/L1c, no task reordered. **Q2** — with the id
lane refused, whether W2's surprise half (one scalar into the context, no alphabet fragmentation) gets its
own probe or W2 closes whole. **CLOSED BY MEASUREMENT 2026-09-05 (owner "Do all this";
`evidence/q2-surprise-context.md`):** its own probe ran (12 cells, seeds 0–2) and closed on a pre-registered
unit-count bar — G-FRAG NOT MET by the letter (+2 / +2 / +3 units against a +2 bar; attribution not
established: the zero-information pinned control fails the same bar on the same seed and the shuffled
control passes it); G-CONSUMER met by the letter only ((iii), no margin — not load-bearing); G-INFO as first
scored was UNMEASURED (a misaligned instrument) and was re-measured aligned (record §3.2). Closed FOR THIS
DELIVERY (`visual[3] = s ∈ [0, 1]`, 4–30× the band's native range); a range-matched delivery is the
reopening an owner could name. No lane; W2 stays REFUSED. (An earlier draft's "HARMS the grower" was
unearned at n = 3 and is withdrawn — the record panel.)

**OWNER 2026-09-05 (chat, after the Phase 2 parallel-arms fold `1eb37df`): "Unblock everything, you are
authorized to do everything to finish the plan."** Standing authorization, recorded here and in the memory store.
What it resolves: the revert option on `dbcf153` is CLOSED — the JEPA-collapse ship on (a) alone STANDS; the
revert option on `3979001` is CLOSED — L1a STANDS on the oracle's majority-answer → persistence-answer reading;
every later "owner may revert / owner question" row is the SESSION's decision, made at pre-registration and
disclosed in the record as "the session's call under the 2026-09-05 authorization" — no arc waits on chat; C2's
`reason_out` deletion on its null is pre-authorized (P8 clause 5); L1b's rule and L1c's P2 survivor are chosen by
the A/B named at pre-registration, as their rows say; Q2 stays CLOSED BY MEASUREMENT (a range-matched delivery is
a session-choosable reopening). What it does NOT do: a measured red is not closed by an authorization — the JEPA
live-floor (b) red on 1 of 5 seeds and the L1a-alone attribution cell stay open as recorded; the per-arc rhythm
(A.4) is the plan's method and is not lifted; the safety envelope stays out of scope (nothing in this plan
touches it; an envelope change, if a row ever needed one, is asked per instance); E-STOP clears stay a person's
per-instance act at the cockpit.

**OWNER 2026-09-06 (chat): "ratify all three and add them to the plan."** X1 (Phase 3), L5 (Phase 2) and C5
(Phase 4) are RATIFIED and land in the phases above, in Part F (X-R1, L-R6, C-R5) and in `tasks.md` as T088–T090. **L6 / T091 / L-R7 was opened on 2026-09-06 by L5's own instrument**, which is what a gate is for: with its rewrite arm restored the census found that the sleep tick reclaims a class the registry labels IMMORTAL. It is carried as dated debt in `immortal_retention.KNOWN_UNLABELED`, which may only shrink, and the row's completion is that entry leaving the table.
They were drafted at the owner's request after the plan was measured against the owner's stated goal (a brain that
grows neurons to reason and think, reaches phenomenal consciousness, and keeps a durable memory that never forgets,
a spiking network replicating a superhuman brain): the plan covered the growth, the spiking substrate, the reasoning
bar and the one lake, but had NO scale target expressed as a capability, BOUNDED retention rather than forbidding
forgetting, and by its own honesty rule cannot conclude phenomenal consciousness from a passing instrument. The full
text of each row, with its control, red, gate and what it does not claim, is `PROPOSED-ROWS-2026-09-06.md` (re-headed
RATIFIED). Each binds only after its gate exists and its output has been counted, per the cross-cutting invariant.

**All five were given in chat on 2026-09-02** ("You are authorized to: O1 … O2 … O3 … O4 … O5 …"): O1–O3 are
authorized ACTIONS (O1 and O2 executed the same day; O3 chosen by the session under the authorization and
mechanized by H1), O4–O5 are RATIFICATIONS. OD-2 stays a standing runtime gate. Still open for C2: whether `reason_out` may
be deleted on its null (P8 says yes; the answer path is user-facing, so it will be asked when C2 lands). Nothing in this plan touches the
safety envelope; W1 runs inside the existing scene.

**OWNER AMENDMENT 2026-09-14:** "Combine the Navier-stokes equation with Liquid state machines to
the final plan" (requested 2026-09-13; retry requested 2026-09-14). This adds the N workstream,
FR-036, SC-014 and T096-T100 as planned experimental work. The amendment records a design and
falsifiable gates; it does not record implementation, a passing measurement, or a consciousness
finding. Custom architecture is acceptable. Existing phase exits, lake retention and protected
self-modification/safety approval boundaries retain their authority.

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
