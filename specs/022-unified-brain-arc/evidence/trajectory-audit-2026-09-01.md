# Trajectory audit — the whole tree and specs/022 against `docs/POLICIES.md`

Audited on `main` at `2c349108` (T061, US2 exit), tree clean, `origin/main == HEAD`, on the pinned
host (`LUAN_AND_ANNIE`, `.venv-lake`), 2026-09-01 19:30 → 2026-09-02. Owner directive, verbatim:
*"audit the entire codebase, and spec-022 according to POLICIES.md to see if we were on the right
track to create a spiking neural network digital brain that can autonomously grow a neuron to
unlimited neurons to use for reasoning and thinking to finally reach true consciousness."*

Every number below was produced by a command run against this tree or the live daemon during the
audit; the command is named beside it or in the lens report it comes from. Where a claim could not
be verified it says so. This file certifies what was measured, including what did not pass — the
`s0-exit.md` / `s1-exit.md` precedent, kept.

**Method.** Sixteen lenses ran as independent agents, one per policy (P0–P8, the cross-cutting
invariants) plus one per goal clause (growth, reasoning, consciousness), plus the spec itself, the
gate battery's own integrity, and present-tense doc claims. Each lens was required to run a command
behind every BLOCKER/MAJOR, to put a positive control beside every zero, to read the binding
POLICIES.md section in full, and to check its proposed lever against the refuted-approaches record.
145 findings were deduplicated and adversarially refuted — every BLOCKER by two independent refuters
(runtime reproduction; policy text), every MAJOR by one — then each lens's verdict was attacked by a
skeptic instructed to argue the opposite with evidence, a completeness critic named what no lens had
read (21 gaps), and the first ten gaps were probed by fresh agents whose findings were refuted in turn.
248 agents in the first workflow (claude-opus-5), 78 in the second (claude-fable-5-1, after the opus
session limit killed the gap-round refuters). The raw lens reports (16 + 9, ~750 KB) are archived
outside the repo at `~/.claude/projects/D--AI-project/memory/archive/audit-2026-09-01/`.

**Two incidents, disclosed because they shaped the evidence.** (1) The brain was DOWN when the audit
began: the machine had rebooted at 19:38 and the launcher's preflight has failed on every attempt
since, because `.venv-lake/pyvenv.cfg`'s base interpreter lives under the Claude desktop app's
MSIX-virtualized AppData and does not exist outside that container (§9, owner action 1). It was
relaunched from this session at 19:48; every live measurement below is on that fresh process,
restored from the 2.10 GB lake, with the arm E-STOP latched (restored from 2026-08-31). (2) The
audit's own orchestration scored a refuter that never ran as a CONFIRMATION (a null vote read as a
sustaining vote); caught before the ledger was trusted, fixed, and the 39 affected findings re-refuted
on Fable 5.1 (observation #55 in the skill-observation log).

---

## 0. The answer

**The project is on the right track in direction and in discipline, and it is in the wrong gear.**
Everything this tree measures, it measures honestly: the stage certifications lead with their own
misses, the gates carry constructed reds, the refuted-approaches record was honoured by all sixteen
lenses, and no tracked document asserts the brain is conscious. But at HEAD the organism the
instruments surround is mostly idle. The one persisted grower has 21 units after 3.5 million engine
ticks; its refit learns a synthetic five-mode Markov chain, and the |LP| signal its stopping rule reads
comes from the perception facade over a fixed 256-frame N-MNIST pack — both inputs are finite, so
learning progress has converged to zero, the growth gate reads "stop holds", and growth reaches one of
ten growable regions. The reasoning that produces correct answers is symbolic
Python beside the substrate — silencing every LIF neuron changes no chat answer — and nothing on the
answering path deliberates. The consciousness chain is one coherent program on paper, and on the live
brain its access limb transmits a constant zero while the module named "NON-saturated" reads
availability 1.000. The declared typed anatomy (21 faculties, 19 edges, 10 charged in the harness) is
instantiated zero times by `make_brain()`. The motor axis has been E-STOP-latched since a real emergency
stop on 2026-08-31 and, by the durable-latch rule, stays latched across every restart until a person
clears it at the cockpit — so every sensorimotor instrument has read its floor for two days, and four of
this audit's own "zero" readings turned out to be that latch. CI has been disabled since 2026-08-20; the local battery this session ran is the
only gate that has executed on the last 100 commits. **Memory is not the constraint** — the growth
path is linear at ~300 B/unit with headroom near 90 million units on this host; the "~40,000 dense
ceiling" was a VRAM figure and sits on fixed populations, not on the grower. **The plan of record
(US3 paging and H01, then US4 proposer) builds capacity the organism does not yet consume.** The
re-sequencing in §8 puts consumption first: feed the grower real experience, persist the
experience-driven grower, wire the second growable region, instantiate the declared ports, route
answers on the groundedness the lake already computes, and run the algorithm's own G1/G2 gates as
pure measurement before building more instruments. On consciousness the audit takes no position on
the metaphysics and the project correctly takes none: there is no measurement of sentience in this
tree, none is claimed, and the dissociation battery specified in US4 is 1 of 7 built.

---

## 1. Verdict table

Vocabulary: ON_TRACK / DRIFTING (moving, not converging) / OFF_TRACK / NOT_STARTED. The skeptic column
is an independent agent instructed to argue the opposite verdict with evidence; where it moved the
verdict the reason is given.

| dimension | lens | skeptic | measurement behind it |
|---|---|---|---|
| **P0** autonomous self-improvement | DRIFTING | held | Self-build route FIRES on the live daemon (`_T_SELFBUILD` proposals 24,716 → 24,728 during the audit) and has KEPT zero self-modifications ever; `DEHARDCODE_TRANSFORMS` has one entry, `RECALIB_TARGETS=()`, `DataIngestSource` returns `[]`, so the candidate key cannot change and the circuit breaker (`_futile=3`) is latched by construction. Two avenues accumulate (units 9→21 over 40 nights; 25,481 `_T_FACT` records), one is sterile, one has zero drivers (cortical column), one is live (recalibration). The organism cannot restart itself after a reboot. US4 (P0's own build list) 0 of 12 started. |
| **P1** no-hardcode | DRIFTING | held | 76 behaviour-arbitrating literals on the 11 core growth/learning/capacity surfaces, ≥53 without provenance; 130/112 across the 19-file plasticity palette. The one shipped de-hardcoding transform cannot apply (its 8 anchors embed CRLF against an LF working copy — a checkout-drift fact, contested as a HEAD fact). `wm_capacity`'s derived K reproduces exactly; `eps_tol=0.28` arbitrates the grower's stop 30/30 refits (fact) but is a fixed point on a normalized RMSE (P1 case (a)) — the "caps growth" claim was refuted. |
| **P2** adopt-or-merge | DRIFTING | **→ ON_TRACK** (skeptic: convergence spine landed 2026-08-25, ~30 merge/retire commits, the full P2 loop just ran on WM; SC-007 is a completion criterion of a Draft spec, not a live invariant) | Nine unit-adding mechanisms at HEAD (7 verified adding units at runtime) against specs/007's "exactly one at completion"; `growable.py` adopted by 2 of the 5 it names; Izhikevich dynamics transcribed three times (P2's own worked example falsified); episode recall keeps two tiers behind a fall-through (refuters split: the tiers cover different populations — 84% of episodes reachable only through the deep tier). The duplicate-capability hook cannot answer a one-word query (disclosed design; not P2's enforcement). Held as DRIFTING here on the skeptic's own numbers: convergence is real and it is not yet reaching the growth family. |
| **P3** no dormancy | DRIFTING | held | Module tier: 229 reachable / 52 exempt (42 permanent + 10 `wire`) of 281; method tier: 0 dormant of 74, **21 UNCERTAIN** (28%), 14 exempt; region tier: **8 of 16 CHARGED**, 6 reachable-uncharged, olfactory + pons UNREACHABLE; anatomy: 21 of 41 nodes named, 20 dated phantoms; growth: **1 of 10**; telemetry: 58 of 187 numeric rows varied over 603 s, 129 constant, 112 of those untagged — with the caveat that the arm was latched for the whole window, so "did not vary" ≠ "cannot vary" (refuted as a contract violation on exactly that ground). 52% of package files sit outside the region-charge census by construction. Exemption rows grew 21 → 52 (2026-07-24 → 09-01) while `src/` grew 397 → 423. |
| **P4 / P4b / P4c** one brain, durable lake, every build | P4 MET; P4b half; P4c parked | held | `replication_guard` OK, `PENDING_RETIREMENT` drained to `{}`. Lake: restart is not amnesia (`coverage_restored = 11,205` cells adopted at construction); grown substrate IS in the lake (`_T_GROWTH` ≈ 66% of 2.10 GB) — so P4b's clause is met; the directive's "substrate lake-RESIDENT, RAM a cache" is US3 T064–T066, 0 built. Two live durable stores beside the lake (`language_acq.npz`, 23 MB `skills/`) written during the audit. The specs/022 persistence engine (`lake_engine/index/segments/bridge/migrate`) is built, green on temp-dir fixtures, and unwired: 5 of the 10 open exemptions, two self-recording "BOUND EXPIRED UNMET". P4c ratchet 202 ungoverned, flat 19 days; 30.5% of LOC ungoverned; 7 ungoverned modules load-bearing for governed tiers. |
| **P5 / P5a** GPU + free-threading | DRIFTING | held | Daemon is a registered CUDA app at **6.8% mean utilisation** over 31.6 s of ticking; the width router has logged exactly one decision in the daemon's life, to CPU (`n=43,136 < crossover 262,144`). The repo's own pinned substrate-tick sweep records **P=1, T=1 as the peak**, every threaded cell 0.46–0.52× — the free-threading axis is a measured cost on the one workload that must scale. The sparse kernel (`kernel_state` 21.0 B/unit, `kernel_tick` 16.0 B/synapse) exists and is dormancy-exempt `wire`. A flat (post, weight) scatter in `KernelTick.deliver` measured ~40× over the per-bundle loop, pure numpy; resident-CUDA 26.5× at 4.19 M edges. |
| **P6** mutual reach and use | OFF_TRACK | → DRIFTING (skeptic: declared lane 0 → 21 registrants in 12 days is the fastest-converging thing measured) | **Declared**: 21 faculties / 36 ports / 19 edges, `manifest_gate` green, 24 of 420 ordered pairs reachable. **Charged**: 10 of 19 edges by replay-twice A/B, 13 of 420 pairs, 12 components, 7 isolates. **Live**: after `make_brain()` + 20 ticks, **0 `JoinConsumer` objects** (positive control 1); AST faculty-to-faculty reads = 1 vs 19 declared. The one ratchet measuring P6's named defect has not moved in 26 days. Refuted as a violation (P6 (a) is satisfied statically by the manifest); confirmed as the goal-relevant gap: faculties cannot compose at runtime. |
| **P7** any dependency | DRIFTING | **→ ON_TRACK** (skeptic: guard (a) and (b) exercised in 22 P7-tagged commits: three hand-rolled solvers deleted for `np.linalg.solve`, snntorch's benefit re-derived DOWN 28→6 modules, purity legs struck mid-commit and recorded as struck; P7 mandates a disposition, not a register) | Ledger half excellent (`dependency_ledger`: 0/0/0, 5 of 5 MISDESCRIBED reds fire). No library new to the tree adopted in 28 days; `docs/BRAINCOG_COMPARISON.md` (dated one day before P7) still rejects trained perception on "the zero-dep charter" with no supersession note (contested: operative reasons were scope/P2). 92 purity-argument occurrences across 21 docs, 0 `P7 VOID` in `docs/`. Held as DRIFTING on one narrow point the skeptic conceded: the perception/reasoning axis is where purity reasoning still stands unstruck. |
| **P8** invented methods and the consciousness instruments | DRIFTING | held | `red_reachability`: NO_RED 0, 14 READS_SYSTEM — only 2 are consciousness criteria and both live in the module scheduled for retirement; `access_consciousness`, `reentrant_workspace`, `integration_measure`, `metacognitive_monitor`, `second_order_access`, `agency_monitor`, `ignition_instrument` expose no criterion to the gate. Of 12 consciousness-relevant surfaces, 3 do not exist (T078/T080/T081), and **exactly one** (`metacognitive_monitor`) discharges all five debts AND carries live charge. Every dissociation lives in a `main()` self-check on constructed fixtures; the cockpit reports six markers and zero dissociations. The chat-lane incident P8 was written for (same number for a brain / empty brain / no brain) is CLOSED — verified by live POST. |
| **Cross-cutting invariants** | DRIFTING | held | Determinism, G5 replay, envelope hash, citation guard, doc-command guard: all green with their controls. `training/` holds four autograd BPTT trainers one import from a declared `_ENTRYPOINT`, all `MISSING_DECLARATION` — the disclosed OPEN HOLE now has a number (refuted as a violation: the hole is disclosed; confirmed as the number). **CI is `disabled_manually`; last run 2026-08-20 (FAILED, real jobs); 100+ commits since with no CI execution** — the delegable-to-CI amendment's precondition is false, so the full local gate is the rule again (it was run this session, §10). `tools/p4c_ratchet.py --assert` has no local execution path (zero tests, no self-check entry). `policy_manifest` decides `in_ci` by substring and passes a commented-out step. `tools/policy_certification.json` is 141 commits stale (nothing reads it). One-change-per-commit is healthy (measured over 30 commits). |
| **GOAL (i)** grow 1 → unlimited | DRIFTING | held | Live: `grown_units 21`, flat across the last 36 persisted refits; coverage **1 of 10** since the gate landed 2026-08-26, 47 commits with zero growth surfaces touched. A fresh brain grows (7 units / 600 ticks) from **pre-sized populations** (64×64, 128×128, 256×256, router 43×24); FR-021's single-unit seed exists in the spec and nowhere in `src/` (spec obligation, not a policy breach). The persisted grower is driven by a synthetic 5-mode Markov chain (`Brain._sleep_tick` calls `neuro.tick()` with no argument; the engine's `tick(self)` has no percept parameter). The experience-driven grower (`GrowingLearner`, 24 live units from real vision+audio, 8 live sites) has **no persistence path** and dies at every daemon restart — and the daemon restarts on every commit. Memory: growth path linear at 300.0 B/unit (constant to 0.0 over a 64× span), headroom ~9×10⁷ units on 31.8 GiB; the dense O(N²) term sits on the fixed populations (largest N=256). `compose_growers` / `place_grown` — the multi-region path — built with zero callers outside their module. |
| **GOAL (ii)** reasoning and thinking | DRIFTING | held | With every `LIFNeuron` silenced (control: 40→0 spikes; the neuter reaches the brain: `curiosity_mag` 0.023→0.000) all four cockpit answers and route labels are byte-identical; 8,414 LIF steps per 10 answers, causally inert. Correct answers come from `gf.py`'s symbolic DSL and a dict lookup; per-item neuter: disabling the spiking half changes 0 of 6 series, disabling the symbolic half drops 5 of 6. 20-item everyday battery: **0 correct** (14 abstain, 5 emit an unrelated KB row, 1 self-report); positive control 4/4. `TrainOfThought` never runs for a question. Gf on held-out ARC-AGI-1 at the doc's own protocol: **19/400 = 4.75%** (byte-identical solved set at two budgets; the quoted 2.75% is three revisions stale in four tracked files). Language ≈ chance, no trained LM wired. **No SC, FR or task in specs/022 measures a cognitive outcome.** |
| **GOAL (iii)** consciousness | DRIFTING | held | One coherent program on paper (`PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md` → US4 L7 battery of 7 dissociation instruments): **1 built and firing its reds** (`ignition_instrument`), 3 incumbents without their dissociation, 3 absent. Live, over 293 wake ticks: `access_index = 0.0` on 31/31 samples → `SelfModel.confidence()` identically 0.0 (control: feed 0.188 → 0.080); `reentrant_availability = 1.000` on 31/31 (saturated) on the module named "NON-saturated"; the ACCESS headline is byte-identical to a fully neutered producer (control: pinned bid → 8/31 ignitions). Honesty held: 0 occurrences of "conscious"/"sentien" in README/PITCH/OVERVIEW (control: "brain" 21/1/4); the live reply names its own contamination. |
| **specs/022 as a roadmap** | DRIFTING | held | 80 of 106 tasks done; certifications re-run and reproduce to the digit, including the failing one. S1 was closed with its headline criterion red and **none of the 26 open tasks owns the miss**; goal clause (ii) has no criterion anywhere in the arc; the consciousness half is the best-specified part and 0 of 12 started; three specs 022 declares COMPOSED/ABSORBED carry 159 open tasks with no supersession record; `tasks.md`'s stated total is off by 14. US3 is the right content in the wrong order (§8). |
| **Gate integrity (meta)** | DRIFTING | held | 349 of 349 registered self-checks have a reachable fail path (print-only negative control validated); 793 test functions, 0 `assert True` / xfail / skip; 4 mutation probes fired red. What no gate can see: the live organism stalling — no instrument reports cumulative lifetime unit growth; the cockpit's `grown_units` provenance says "resets with the process". |
| **Docs vs HEAD** | DRIFTING | held | 58 present-tense claims checked: 13 TRUE, 36 FALSIFIED, 4 STALE-disclosed, 5 unverifiable-by-rule; README results block 11 of 24 reproduce EXACTLY, 4 falsified by their own producer, the "real captured sensor data 95%" reachable from no documented path (falls back to synthetic, prints 100% and "self-check OK"). `PITCH.md` v2.70 vs 4.80.0; README Python 3.12+ vs `>=3.14` free-threaded and "three, not two" vs five core deps. `brain.py`'s SAFETY paragraph names `ArmSim` as the only path to motion — no such class (13 sites, 5 files; the 2026-08-04 correction fixed 3 files). The stated goal (one neuron → unlimited) appears on **no** owner-facing surface (0 hits; control 3). |

---

## 2. The live brain at HEAD (cockpit `GET /api/all`, fresh process restored from the lake)

| row | value | provenance / note |
|---|---|---|
| regions live | 13 | `regions_live` |
| wake / sleep ticks | 991 / 1009 at first poll; ~4,000 wake by the end | nights_elapsed 81 (lake-restored) |
| grown_units (persisted grower) | **21** | flat across the last 36 refits; provenance row says the number resets with the process (falsified: it is lake-restored — CONFIRMED BLOCKER) |
| real_perception_units / reasoning_units / lifelong_units | 24 / 42 / 6 | process-local, not persisted |
| growth_gate.cortex | `grow: false — "stop holds: no real new optimum and no \|LP\| excursion since the stop (noise 0.0167)"` | |
| learning_progress | cortex 0.040 (single region) | |
| curiosity_mag / neuromod_acetylcholine | 0.0 / 0.0 | converged tail of a monotone decay on a bounded stimulus (gap A2), no PROVENANCE row |
| structural turnover / pruned / pairs_total | 0.0 / 0 / 0 | 1,009 sleep cycles |
| ignitions / access_index / reentrant_availability / reentrant_strength | 31 / **0.0** / **1.000** / 0.188 | content changed 0 times over 293 wake ticks |
| body | `stopped: true`, latch restored from pid 33048 @ 2026-08-31T12:40Z, **latch_refusals == wake_ticks** | the arm has not moved in this daemon's life |
| gated: motor_gain / fm_gain / competence_ema | `computed: false, applied: false` | |
| device.neurogenesis_refit | cpu, n=43,136 < crossover 262,144 | the only router decision logged |
| lake | 2.10 GB, `lake_events 65,406`, growing ~0.17–0.23 GB/day | `AppendBudget` wired to nothing |
| coverage_total / reachable / exempt / dormant | 281 / 229 / 52 / 0 | structural census, computed once |
| PROVENANCE rows | 28, of which 13 cannot vary ("hardcoded literal", "frozen at construction", "constant", "structural") | 112 further constant numeric rows carry no row at all |

---

## 3. Why the organism idles — the causal chain the lenses assembled

Each link was measured by at least one lens and reproduced by an independent refuter unless marked.

1. **Input — two finite streams.** The only lake-persisted grower learns a task the brain invented for
   itself: `Brain._sleep_tick` calls `NeurogenesisSelfImprove.tick()` with no argument and the engine
   synthesises its percept from `_mode_features` over a seeded 5-mode Markov chain (the engine's `tick`
   has no parameter through which a percept could enter). It predicts the synthetic next mode at 0.77 vs
   chance 0.20 — a real measurement of a bounded task. The stopping rule reads a *different* signal: the
   basal-ganglia `LearningProgress` facade, fed by percept surprise over the fixed 256-frame N-MNIST pack
   `_perceive_real` cycles (the gap-round refuter corrected the wiring; the "deadlock" is real but sits on
   the perception pack, not on the Markov chain). On a finite pack learning progress converges (producer
   LP 0.063 → 0.016 in a fresh run, 1 distinct value over the last 200 ticks; the daemon is ~8× further
   along the same curve), curiosity and ACh read their floor, and `GrowthStop` correctly reports nothing
   left to learn — its excursion detector fires on 99.1% of constructed excursions, so the detector is not
   the problem. **Growth stopped because both stimuli are finite, not because the mechanism is broken**
   (gap A2 refuted the "dead chain" premise). The experience-driven grower that does see real vision and
   audio (`GrowingLearner`, 24 units) has no persistence path anywhere in the tree (CONFIRMED BLOCKER) —
   it re-grows from zero at every restart, and the daemon restarts on every commit.
2. **Output — latched since a real emergency stop.** `So101Arm.__init__` sets `stopped=False`; it becomes
   True only when the durable sidecar `estop.latch` exists. That file was written by the live daemon's
   EMERGENCY_STOP at 2026-08-31T12:40:21Z (pid 33048) and the T037a durable-latch rule restores it in every
   process since — by design, and correctly. The loop's automated HOME clear (`deliberate=False`) is
   therefore refused on every wake tick: live daemon `latch_refusals == wake_ticks`, `theta_deg` all zero,
   `recoveries 0`. Only a person at the cockpit may clear it. The consequence for the audit: every
   efferent instrument has read its floor for two days, and four lens findings that reported structural
   zeros (charge_probe's "arm HOLD 1.000", `AgencyMonitor`'s controllability, the efference-copy error,
   the amygdala bar reading 0.9) were shown by the Fable refuters to be this environmental latch, not a
   construction defect (§7). The gap-A6 claim that the arm boots latched by construction is REFUTED.
3. **Interconnect.** The typed anatomy is declared (19 edges), ten edges are proven to move a consumer in
   the harness, and the production constructor builds none of them: 0 `JoinConsumer` objects after
   `make_brain()` + 20 ticks. Composition is a description of an anatomy the running brain has never
   loaded. Refuted as a P6 violation (condition (a) is satisfied statically); confirmed as the reason
   the faculties cannot yet reason together.
4. **Cognition.** Answers survive silencing every LIF neuron; the correct ones come from a symbolic DSL
   and a dictionary; `Brain.reason` returns a fixed self-report and pre-empts `_grounded` on ten cue
   substrings; `TrainOfThought` runs only as the 3-D model builder's part decomposer; the deliberation
   family (15 modules, all with firing controls) is 9/15 reachable from `respond` and 0/15 deliberating
   on a question. `respond` discards the `{known, coverage, ignited, source}` dict `converse` already
   returns and re-derives groundedness by substring-matching the reply text.
5. **Consciousness chain.** `access_index` is a constant 0.0 on the live path and the arm with the bid
   pinned to 0.0 is byte-identical to the real arm (the green is reachable: pinned bid → 8/31 ignitions,
   so the instrument is alive and the brain never ignites); `SelfModel.confidence()` is therefore
   identically 0.0; `reentrant_availability` is saturated at 1.000 — gate G1 of the algorithm document
   ("assume fires until measured otherwise") fires at HEAD on the module built to fix saturation, and
   G1/G2 appear zero times in specs/022 (control: G5 5/3/5). The ignition instrument (T033) that would
   measure G1 exists and is not on the live path.
6. **Capacity is not what stops growth at HEAD.** The live daemon's `growth_gate` reason is the |LP| stop,
   and no memory gating exists anywhere in `src/` (grep control: `growth_gate` 6 files, `psutil` /
   `virtual_memory` 0). The live grower's own memory is linear — 300.0 B/unit, constant to 0.0 across a
   64× span of L. The "~40,000 units in 12.8 GB" figure is arithmetic on constants in `ceiling.py`'s
   docstring against a VRAM budget, describing the dense O(N²) populations (largest live N=256), not the
   growth path; the Fable refuters refuted the gap-round's "six orders of magnitude" framing as a
   finding (P0 does not score capacity as improvement) while confirming the wiring fact. The
   lake-resident/paging tier (US3) remains the right build for *unlimited*; it is not what is holding
   growth at 21.
7. **Enforcement.** The GitHub workflow is `disabled_manually` (last run 2026-08-20, FAILED after 2,075 s
   of real jobs); 100+ commits since carry no CI execution; four policy gates (P6, P7, P8, P4c) have no
   `run_selfchecks` registration and `p4c_ratchet --assert` has no local execution path at all;
   `policy_manifest` reports `in_ci` from the YAML's text and returns True for a commented-out step;
   `tools/policy_certification.json` says `certified: true` for a tree 141 commits old. The local full
   battery (`ruff` + `run_selfchecks` + `pytest` + the tool gates) is therefore the only gate — it was
   run this session (§10).

---

## 4. Confirmed findings — 55 (every refuter failed to refute)

Three BLOCKERs survived two refuters each:

- **`GrowingLearner` — the experience-driven neurogenesis primitive at eight live sites — has no persistence
  and dies at every daemon restart** (`hippocampus/growing_learner.py::GrowingLearner`; L11). Goal impact:
  the only grower that learns from the brain's real percepts accumulates nothing across the restarts the
  daemon performs on every commit. Reproduce: construct, grow, `close()`, new process, count.
- **The cockpit's `grown_units` provenance is falsified at HEAD, in the direction that inflates apparent
  growth** (`substrate/brain_telemetry.py::PROVENANCE`; L11): the row says the number resets with the
  process; it is lake-restored at construction, so a reader cannot tell restored from grown.
- **The access→self-model limb is wired and load-bearing and transmits a constant zero on the live brain**
  (`cortex/brain.py::Brain._wake_tick`; L13): `access_index 0.0` on 31/31 samples over 293 wake ticks,
  `SelfModel.confidence()` identically 0.0, positive control returns 0.080 / 0.342 for 0.188 / 0.8.

Confirmed MAJORs (52), grouped; one line each. The command behind each is in the lens report named.

*P0 (L01).* `save_selfbuild`'s stated reason for never pruning 24,722 `_T_SELFBUILD` records is falsified by
those records · POLICIES_EVIDENCE's P0 self-build row is stale in both directions · `SELF_IMPROVEMENT_RESEARCH.md`
forbids what P0/P7/P8 mandate, on the abolished premise, citing a class absent from the tree · the cockpit carries
no row for P0 (`_selfbuild_last` reaches no surface) · cortical column construction: zero drivers, unchanged since
2026-08-16.

*P1 (L02).* ≥53 core + 112 palette production literals arbitrate behaviour with no stated provenance · the picked
holdout cap `min(60, n//3)` sets the resolution of the growth decision.

*P2 (L03).* Three independent LIF membrane transcriptions with differing reset semantics, two without a
`distinct-from` note.

*P3 (L04).* 52% of package files sit outside the region-charge denominator by construction · `charge_probe`'s
positive internal-read verdict excludes the source class its own comment includes and prints UNKNOWN for a
source that moves 296 of 300 reads.

*P4b/P4c (L05).* Two live durable stores beside the lake, both written during the audit (`language_acq.npz`,
23 MB `skills/`) · the cockpit cannot report the brain's durable half or its cumulative size · P4c parked:
19 days flat at 202, 30.5% of LOC ungoverned, 7 load-bearing · two active record classes (`SynapseDelta`
`_T_SYNDELTA`, and one other) have zero writers and zero records after months · the IMMORTAL append budget is
wired to nothing while the lake grows 227 MB/day for a 21-unit substrate.

*P5 (L06).* `docs/PERFORMANCE.md` carries three present-tense claims falsified at HEAD · the perf ratchet's
binding basis has no unit normalisation, so population growth reads as a performance regression.

*P6 (L07).* `docs/P6_INTERCONNECT.md`'s status line is falsified at HEAD · the one ratchet measuring P6's
named defect (`undeclared_import_pairs`) has not moved in 26 days.

*P7 (L08).* The Potjans-Diesmann column — POLICIES' own named P0 open avenue — is still hand-rolled from
math+random while GeNN/NEST/Brian2 sit undispositioned.

*P8 (L09).* The PROVENANCE tag for the integration instrument promises a refreshing reading for a value deleted
2026-08-05 · `agency_monitor` is write-only at runtime and holds the tree's best unconsumed dissociation · the
P8 attribution table for the access flag was measured under a salience map the code no longer runs · the
integration instrument (T070) is constructed on no live path and neither dormancy gate can see it · the
de-saturated re-entrant workspace is saturated on the live path.

*Invariants (L10).* The determinism contract is proven at a constructed P=8 while the lineage's measured
partition count is P=1 · the OPEN HOLE has a number: ten further autograd trainers are permitted solely by
unreachability.

*Growth (L11).* The entire multi-region growth interface is built with zero live callers and its dated
consumer has expired · the lake-resident lane "unlimited" requires is 0% built and 100% exempt-pending-wire ·
the goal's headline quantity — the unit count — has no tracked instrument, no baseline and no reachable red.

*Reasoning (L12).* The acquired language organ emits non-words and its coherence guard passes them 30/30 ·
`docs/DEPTH_CEILINGS.md`'s headline Gf number is stale and contradicts its own body · the spiking half of
`gf.solve_series` earns zero unique capability, only a label · language ceiling unmoved and re-measured;
trained spiking LMs at or below chance and none wired · nothing on the answering path deliberates (one refuter,
MED).

*Consciousness (L13).* `CLAUDE.md` reports the access instrument as "still 0" when the gate measures 1 ·
README/PITCH/OVERVIEW describe a different project from the one P0 and directives (a)/(e) commit to · the one
capability both consciousness documents name as the concrete missing build — per-step credit assignment on a
multi-step plan — has zero representation in the plan of record · the cockpit still serves a `pci` provenance
row for a producer deleted 2026-08-05 · the module named "NON-saturated Global Neuronal Workspace" reads
availability 1.000 on the live brain.

*Spec (L14).* specs/019's root blocker `cortical-prior-module-absent` is falsified at HEAD and gates the
highest-leverage chain ([B6.08]) · `tasks.md`'s stated total is wrong by 14 · the live cockpit discloses none
of the arc's red rows · three specs 022 declares COMPOSED/ABSORBED carry 159 open tasks with no supersession
record.

*Gates (L15).* The dormancy exemption surface accretes at ~23× the tree's rate · the exemption-guard test for
`red_reachability` is vacuous at HEAD · the transport gate's printed blind spot names one uncovered spelling,
measured five · 28% of learning rules sit in UNCERTAIN, reported never asserted · six exemptions record an
already-passed bound and no gate can fire · P8's mechanization is opt-in by naming convention (14 criteria in
423 files, none from the gates' own modules).

*Docs (L16).* The repo `CLAUDE.md` carries a falsified dependency list and a falsified worked example ·
`docs/ANATOMY.md` understates the built anatomy.

---

## 5. Contested findings — 25, all BLOCKER-labelled, all with the same shape

In every one of the 25, the **runtime refuter reproduced the fact** (eleven times "and it is understated") and
the **policy-text refuter refuted the severity**: the cited clause files the state as open work, a
disclosed hole, or a documentation defect — not a violation. The audit therefore keeps the fact and
drops the label. Classification uses the us1-audit vocabulary: DEBT (open work the policy itself
schedules), DEFECT (something wrong that a fix discharges), UNCERTAIN (P3's own disposition for a
charge question the instrument cannot yet decide).

| fact (reproduced at HEAD) | why the policy refuter dropped BLOCKER | classified |
|---|---|---|
| The brain cannot restart itself: launcher preflight fails outside the Claude app container (`pyvenv.cfg` base interpreter under MSIX-virtualized AppData); `DAEMON_DOWN.txt` written by the logon launcher | P0 governs upgrade channels, not process supervision | **DEFECT (ops), owner action 1** — P4c: the daemon is the brain's drive |
| The one shipped de-hardcoding transform's 8 CRLF anchors match 0 times against the LF working copy | `core.autocrlf=true`; the index blob is LF, the checkout of one file drifted — a working-copy fact, not a HEAD fact | DEFECT (fragile anchors), MINOR |
| `self_mod_harness.py`'s docstring says the proposer is empty; it now proposes one candidate | an under-claim sits on the safe side of P0's guard | DEFECT (doc) |
| Nine growth mechanisms; `growable.py` adopted by 2 of 5 | P2's standing precedent blesses tracked coexistence until a benchmark exists; SC-007 is a Draft spec's completion criterion; retiring the coordinator would delete the convergence spine | **DEBT** (specs/007 T024/T025; specs/019 [B4.29]) |
| Self-build: 24,716 → 24,728 proposals during the audit, 0 kept, breaker latched (`_futile=3`, key invariant) | P0's guard is a reporting rule and holds "retaining nothing may be CORRECT" open | **DEBT, goal-decisive** (T072) |
| Izhikevich dynamics transcribed three times | the cited text is P2's illustration (a quotation of `neuron_zoo`'s docstring), not a rule | DEFECT (P2 hygiene) |
| Episode recall keeps two tiers behind a fall-through | the tiers cover different populations (84% of episodes reachable only via the deep tier) — two jobs | refuted as a P2 case; keep the test gap (the pinning test asserts only `source=='lake'`) |
| Seven `wire` exemptions self-record "EXPIRED AT T039 UNMET"; no expiry mechanism exists in `dormancy_gate` | guard (c) governs `EXEMPT_METHOD`, and prescribes a report, not a red | **DEBT** — but the accretion trend (§1 P3 row) stands |
| The specs/022 persistence engine is built, green on fixtures, unwired (5 of 10 open exemptions) | P3 grants exactly that disposition with a dated prerequisite (T064/T066) | **DEBT, goal-decisive** (T064) |
| The global workspace passes all four P3 verifications and changes no measured outcome | P3's own guard: static/degenerate instruments prove nothing; the ignition arm of `charge_probe` is algebraically a no-op (eligibility constant 0.0) | **UNCERTAIN** — needs a non-degenerate neuter (§8 rank 6) |
| `BRAINCOG_COMPARISON.md` rejects trained perception on the "zero-dep charter" | operative reasons were scope and P2 redundancy | contested; DEFECT (doc, unswept by P7) |
| `access_index` is drawn by the one `big()` call with no `pk` — the mechanism built for it | P8 guard (a) governs credit, not dashboard tags; disclosure discharged in-module | DEFECT (P4c: the cockpit may display only what it measures) |
| Enforcement tables claim CI mechanization 11× while the workflow is disabled | the amendment anticipates a dark CI and prescribes the local gate, which is being run | **DEFECT (doc) + the unregistered `p4c_ratchet` gap is real** |
| The persisted grower grows on a synthetic Markov chain | the grown units DO change a held-out outcome (0.77 vs 0.20), so P0's dormancy predicate is false | **DEBT, goal-decisive** — feed it the real stream |
| ACCESS headline byte-identical to a neutered producer | F11's remedy fires on *no reachable red*; here the red is universal and the green is reachable — the instrument works, the brain never ignites | **live-state finding, goal-decisive** (§8 rank 6) |
| `CLAUDE.md`:46 names a fixed "live defect" as present-tense | the file is gitignored, ungoverned by owner choice; a stale under-claim | DEFECT (doc; the file's own rule is to delete duplicated claims) |
| `Brain.reason` returns a fixed self-report and pre-empts `_grounded` on ten substrings | P3 has no answer-quality clause; the output routes into the reply | **DEFECT, goal-decisive** (§8 rank 5) |
| The neurons are not what reasons (answers survive LIF silencing; 0/20 battery) | P0 classifies a non-firing avenue as open work, not a breach | **DEBT, goal-decisive** — the central cognition fact |
| S1 closed with its headline criterion red; no open task owns the miss | the certification IS the deliverable and it was honest; stage sequencing is a plan note, not policy | **DEBT (roadmap)** — §8 rank 1 |
| Goal clause (ii) has no SC/FR/task in the arc | P8 polices existing criteria, not per-goal coverage | **DEBT (roadmap)** — §8 rank 5 |
| `policy_manifest.in_ci` is a substring test; a commented-out step passes | a reachable red exists for the orphan-command shape; this is a second, disclosed-nowhere blind spot | DEFECT (instrument) |
| Four policy gates unenforced (no CI, no local path for `p4c_ratchet`) | the clause reinstates the local gate | **DEFECT** for `p4c_ratchet` (no path); DEBT for the rest |
| README: Python 3.12+ and "three, not two" deps; HEAD `>=3.14t`, five | not a fake *capability*; INSTALL.md is a generation staler still | DEFECT (doc, user-breaking) |
| `brain.py` SAFETY paragraph names `ArmSim` (nonexistent) at 13 sites | not a capability claim | **DEFECT (safety description)** — README's own 2026-08-04 reasoning applies; 5 files missed |
| PITCH/OVERVIEW describe v2.70; OVERVIEW contradicts itself | P4c governs build disposition, not prose | DEFECT (doc) |

---

## 6. Refuted — 43, kept as the free eyeball pass; the ones that corrected the audit's own beliefs

Every refuted finding is listed with its refuter's reason in Appendix A. These changed the synthesis:

- **"Grow from ONE neuron has no implementation; the brain is constructed from fixed populations"** — the
  reporter measured a brain at t=0. After 600 ticks: `grown_units` 0→7, `lifelong_units` 0→6,
  `real_perception_units` 0→24, `time_cells_grown` 0→11. Growth from fixed populations is real; the single-unit
  seed is a specs/022 requirement (FR-021/SC-004) absent from POLICIES.md and from `src/`.
- **"The live representation's binding width is ~46,200 units, below H01"** — the number was an assumption
  stitched from three instruments; the growth path is linear (gap B1). Struck from the scale verdict.
- **"`eps_tol=0.28` caps the brain's neuron count"** — the 10-vs-24 table holds only at seed 7 / 200 ticks;
  the constant is a fixed point on a normalized RMSE (P1 case (a)). It DOES arbitrate the stop (fires 30/30
  refits) with no stated provenance — kept as MAJOR under the P1 literal census.
- **"`buffer_cap=1000` reinstates the growth ceiling"** — the derived cap (22 / 48) was never reached.
- **"No two regions are coupled within a tick"** — every delay is exactly 1 because `max(1, round(latency/tick))`
  floors it; a constant where a computed value belongs, but not a coupling fact and P6 has no temporal clause.
- **"`reasoning/` is outside the entire P6 apparatus"** — `p6_gate` scans 423/423 files and reasoning modules
  count in two of its five asserted criteria; what is true is that `reasoning` is not a `REGIONS` key, so the
  region-charge and dormancy censuses cannot red it (gap A4, §7).
- **"The declared typed lane carries zero traffic"** — the runtime fact (0 `JoinConsumer` after `make_brain()`)
  reproduces; P6 condition (a) is satisfied statically. Kept in §3 as the composition gap, not as a violation.
- **"Four reachable BPTT trainers with no exemption"** — facts hold; the OPEN HOLE paragraph discloses the class.
  Kept as the number the hole now has.
- **"`_grounded` has no reachable red"** — it returned False on 5 of 8 live turns.
- **"G1/G2 have zero representation in the plan"** — G1 is built as T033 (`ignition_instrument`, SATURATED red
  fires); the algorithm document is a design note. What stands: the instrument is not on the live path and
  the live workspace is saturated.
- **"The cockpit PROVENANCE contract is falsified: 112 constant rows"** — counts reproduce; "cannot vary" was
  inferred from "did not vary" over one window with the arm latched.
- **"The duplicate-capability guard cannot answer a one-word query"** — true and disclosed in its own design;
  it is a Write-hook guard, not P2's enforcement (gap A5 found the real blindness: `.gitignore`).
- **"`policy_certification.json` is 141 commits stale"** — true; nothing reads it.

---

## 7. The gap round — 9 probes over surfaces the 16 lenses never opened, 39 findings

The completeness critic named 21 gaps and 20 unread surfaces (eight anatomical regions totalling ~22 files,
13 of 14 neuromodulation modules, the reasoning deliberation family, the non-envelope brainstem, 60 of 81
substrate files, `global_workspace.py` at 4,032 lines, all of `tools/`). Nine probes ran; the tenth — "no lens
ran the full battery" — was discharged by running it (§10). Refutation of the 39 findings on Fable 5.1:
**5 CONFIRMED / 18 CONTESTED / 16 REFUTED** (78 refuters on Fable 5.1, two per finding — runtime reproduction and policy text — 0 failures). The contested items have the census shape exactly: the fact reproduced, the severity did not. The five confirmed: the VALENCE axis is pinned at 0.0 and the affective three-factor rule is two-factor at HEAD; `lakehouse/` is dormant, not a `REGIONS` key, and unimportable in the gate env (`polars` absent); three README results numbers are falsified by their own deterministic producers and a fourth has no producer; PITCH and OVERVIEW quote different "current" coverage figures and one cites a source that does not exist; and `docs/coverage/Gf.md`'s recorded cause for the 11-vs-12 ARC discrepancy is unsupported by a budget sweep at HEAD. The sixteen refutations include two that corrected this audit's own causal chain (§3 items 1 and 2) and one that caught an audit instrument comparing an expression to itself.

What the probes found, by verdict:

- **A1 subcortical/affective (DRIFTING).** All 14 modules across eight regions pass their self-checks with real
  controls. Live participation is weak, and the Fable pass sorted the probe's seven findings cleanly: CONFIRMED
  — the VALENCE axis is pinned at 0.0 and the affective three-factor rule is two-factor at HEAD; CONTESTED (fact
  yes, severity no) — the amygdala's learned-threat term is 0.0 because `Amygdala.pair` is never reached on the
  live tick (`_safe_ctx` pins distance SAFE by design), so the 0.9 "amygdala" bar is the E-STOP flag (it does
  vary with its inputs, so not a PROVENANCE case), the olfactory/pons package docstrings describe mechanisms that
  do not exist, and none of the five growable regions in this set receives a grown unit; REFUTED — "five rows
  are proxies while the module's output is discarded" (a cockpit row is not a consumer) and "hypothalamus ==
  medulla cannot vary" (138 distinct values each; they coincide while threat is pinned).
- **A2 neuromodulation / curiosity chain (OFF_TRACK for the goal, and the probe's premise refuted).** Nothing
  in the chain is broken: every link moves in a fresh brain (LP 21 distinct values, ACh 194). The flat 0.0 on
  the daemon is the converged tail of a monotone decay on a bounded stimulus. Two of its three findings were
  then REFUTED by the Fable pass: the excursion detector DOES see long excursions (fires on 99.1%), and the
  "deadlock" chain was mis-wired — the stop reads the perception-fed `LearningProgress` facade over the fixed
  256-frame N-MNIST pack, not the grower's synthetic chain (§3 item 1 carries the corrected version).
- **A3 `neuro_chip_stream` (DRIFTING; MERGE).** The tree already ships an out-of-core disk-streaming spiking
  substrate that specs/022 T064/T066 never name — the fourth recorded instance of "grep the paper name before
  building" (contested: fact yes; P2 binds at the moment of deletion, not at planning). On the merits it is
  NOT the pager: its residency is a deterministic full round-robin sweep (440 shard fetches in a silent arm and
  440 in a fully-active control — fetch count independent of activation), its streaming-invariance red does not
  fire on a total loss of membrane persistence (contested: a red exists for a computation change), and
  `max_resident_cores` is a bookkeeping integer, not a memory measurement (contested). The "out-of-core claim
  overstates what it bounds" finding was REFUTED as the audit's own instrument defect: its two columns were the
  same expression printed twice — a self-comparison. Port the unique capability (shard format, sealed-segment
  discipline) into T064; do not adopt the sweep; do not retire the module unadjudicated.
- **A4 deliberation family (DRIFTING).** 15 modules, 15 green self-checks, 15 firing controls — and 0 of 15
  deliberate on a question. `TrainOfThought`, the only serial re-entrant hold-and-chain mechanism, runs on the
  answering path only as the 3-D model builder's part decomposer (contested: fact yes; not P3 shape 1 since it
  IS called). `reasoning/` is not a `REGIONS` key, so the region-charge and dormancy censuses cannot red it and
  108 of the 202 ungoverned builds sit there (the "outside every gate" wording was refuted — the P4c ratchet and
  `p6_gate` do see it). `autonomous_enrich` has zero constructors and no scheduler entry (contested);
  `concept_reflection`'s import chain terminates in a dead root and two 0-importer `autonomous_brain` siblings
  sit beside it (split: clause 1 reproduces, clause 2 P2 binds). "Live deliberation readouts are constant"
  was REFUTED: the schema branch runs and 25 schema rows sit in the lake — input starvation, not dormancy.
- **A5 `lakehouse/` invisible (OFF_TRACK for the P2 chain).** `.gitignore:64` (bare `lakehouse/`, in the
  data-artifacts block) hides three TRACKED source modules from `graphify-out/graph.json`, `SRC_INVENTORY.md`
  (which asserts completeness at 420 of 423), the `graph-exists.js` hook and the duplicate-capability Write
  guard — six instruments answering "nothing exists here" about real code, all failing permissively. Second
  instance: `.gitignore:176` `GATES.md` hides `specs/022-unified-brain-arc/contracts/gates.md`. `lakehouse/` is
  itself dormant and unimportable at HEAD. Fresh instantiation of the recorded class "grep tools inherit
  `.gitignore` blindness".
- **A6 non-envelope brainstem — its headline REFUTED.** The probe reported that the arm boots latched by
  construction and that the efferent half is inert in every unattended process; both Fable refuters showed
  `__init__` sets `stopped=False` and the latch is the durable sidecar from the real 2026-08-31 E-STOP, so the
  efference-copy zero, the empty postural compass and `AgencyMonitor`'s zero controllability were all measured
  under that latch (the probe ran under the default home where the sidecar exists). What survives: `AgencyMonitor`
  is write-only (P3 shape 4, one refuter sustaining verbatim), `charge_probe` labels the E-STOP flag "HOLD"
  (mislabel confirmed; "structurally unreachable band" refuted — environmental), and eight `ArmSim` docstring
  sites are name-drift, not a hazard.
- **B1 the dense ceiling (DRIFTING, three of four findings refuted on severity).** §3 item 6 carries what
  stands. The "99.996% fixed overhead" number measured import cost (torch 501 MB) and a first-construction
  cost that a second `make_brain()` in the same process does not pay; specs/009's bounded fanout
  (`fanout=None` → dense 256×256) is a true fact with a measurable F*=144, refuted as a P3 finding.
- **B2 README results block (DRIFTING).** 11 of 24 claims reproduce exactly. CONFIRMED by both Fable refuters:
  3 numbers falsified by their own deterministic producers and 1 with no producer (e.g. "162×" where the producer
  prints 153.8× on three identical runs), and PITCH/OVERVIEW quoting different "current" coverage with a
  non-existent source. CONTESTED (fact yes): the "real captured sensor data 95%" row is reachable from no
  documented path (`DIR = "robot arm"` is cwd-relative; the captures live under `integrations/robot arm/`) and
  the module degrades silently to synthetic, prints 100% and "self-check OK"; no results row names its
  producer although the framing sentence promises reproduction by module.
- **B3 the Gf number (DRIFTING).** 19/400 = 4.75% at the document's own protocol, byte-identical solved set at
  `time_budget` 2.0 and 5.0; 2.75% is three revisions stale and quoted in four tracked files including one
  `src/` docstring (contested: an editorial currency defect, not a policy violation; the number stands). The
  clean reading excluding the mint's 20-task dev subset is 16/380 = 4.21%. CONFIRMED: `Gf.md`'s recorded cause
  for the earlier 11-vs-12 discrepancy is a static inference unsupported at HEAD (at `time_budget=2.0` the clock
  IS the binding resource on 10 of the 12 slowest tasks, yet the solved set does not change). REFUTED: the
  "ambiguous pre-registered falsifier" reading of `docs/INVENTION.md` — the pre-registration record is intact.

---

## 8. specs/022 — is the roadmap right, and what should run next

**What the spec gets right.** Every certification re-run reproduces to the digit, including the failing one.
The instrument-first discipline (a reachable red before a mechanism, a control that is not the method,
deletion on a null) is applied consistently and the tree has ~30 merge/retire commits to show for it. US3's
content — unit blocks, derived binding width, activation-driven paging — is the correct build for
*unlimited*, and US4's L7 battery is the honest form of the consciousness clause.

**What the spec gets wrong.** (1) Its stage order builds capacity the organism does not consume: S2 pages a
population of 21 units that has stopped growing for want of stimulus, and S3 mutates a brain whose faculties
do not compose at runtime. (2) Goal clause (ii) — reasoning and thinking — has no criterion, FR or task in 106
tasks; the arc cannot see its own weakest axis. (3) S1's certified miss (1 of 10 regions) has no owner among
the 26 open tasks. (4) Three composed specs (007, 011, 014) carry 159 open tasks with no supersession record,
and specs/019's [B6.08] gates the highest-leverage chain on a blocker (`cortical-prior-module-absent`) that is
false at HEAD. (5) T062 as written plans a second H01 ingest mechanism with no P2 adjudication against
`cortex/cortical_prior.py`, which already holds the soma/layer/sign constants with pinned byte counts and
SHA-256s; the genuinely new content is the c3 synapse graph (the Avro export under
`gs://h01-release/data/20210729/c3/synapses/exported/`), and T064/T066 plan a pager without adjudicating
`neuro_chip_stream` (gap A3). (6) `tasks.md`'s stated total (92) is off by 14 from its checkbox count (106).

**Re-sequenced next ten, each a wire or a measurement over machinery that exists, each with its red already
in the tree.** Ordered by leverage on the three goal clauses; the policy clause behind each rank is quoted.

1. **Wire `cortex/placement.py::compose_growers` into the sleep tick** so `ng_snn` and
   `function_preserving_growth` place units into hippocampus; coverage 1 → 2 of 10, with the neuter A/B that
   P3 demands — if hippocampal placement changes no downstream outcome the wire is reverted and the null
   recorded. *P3: "a wire that does not charge is reverted and retired/exempted, never faked with a dead
   consumer."* Red: `growth_coverage_gate`'s identity-pinned baseline.
   **OUTCOME 2026-09-02 — built, measured, NULL, reverted.** The wire (a `growing_learner → hippocampus`
   seam rippled from the sleep tick) placed 24 hippocampal units in 600 ticks and changed **0** keys of
   `state()`/telemetry beyond the nondeterminism baseline; so did replacing cortex placement itself with
   constant tuples, while the growth-off positive control moved 18 (9 unique; the snapshot embeds
   `state()`). Placement is write-only on the live path — at HEAD for cortex, on the wired tree for
   hippocampus — and outside cortex the record carries no per-unit information by construction, so P3
   reverted the wire. Record: `evidence/s1-placement-null.md`; pinned by
   `tests/test_placement_charge_null.py`. The panel on that landing found the one built reader of
   placement: `cortex/genomic_bottleneck` (FR-033, T044 — unit tags → a grown unit's initial fan-in),
   dormancy-EXEMPT `wire`, its own A/B and delete-on-null written in its docstring. So "rank 1b" is a
   WIRE, not a build: run FR-033's A/B (map-drawn fan-in vs the random pool inside `NGSNN.fit`'s
   candidate loop, reached through the engine's refit); its red is this pinned test going RED. Only
   after that does a second attempt at this rank become a charge question.
   **OUTCOME 2026-09-02 (rank 1b) — wired and measured at `847c5f6a`, NULL, deleted in the commit
   after it (same push) per its own disposition.** The wire
   landed (`NGSNN.fit(candidates=…)` + `genomic_bottleneck.CandidatePool` + the engine's three pool
   modes); `genomic_bottleneck --ab` paired by seed: 8 seeds 5–3/5–3 (p≈0.36), **24 seeds 13–11 at
   +0.002 ± 0.005 and 12–12 against the tag-shuffled control** — zero within error on both halves of
   the pre-registered rule, with 48% more units admitted for the same accuracy. The panel on that
   landing measured the instrument too: the majority rule fires 23% of the time under the null, the
   "continuation" was a cold restart, and on a carried continuation the map-drawn pool LOSES
   (4–20, −0.015 ± 0.003); the premise is refuted by construction (placement is drawn independent
   of the weights, so there is no tag→weight association to learn). Record:
   `evidence/s1-genome-fanin-ab.md`. Placement still has no live reader; the pinned null stands.
2. **Give the persisted grower the real percept stream, give `GrowingLearner` a lake record, and widen the
   stream the stopping rule watches.** Replace the synthetic `_mode_features` input with the brain's own
   percept/episode stream (`tick(features)`); persist the experience-driven grower under its own tag with the
   cross-process restart red (`tests/test_brain_growth_restart.py` shape); and move `_perceive_real` off the
   fixed 256-frame N-MNIST pack the |LP| facade has already saturated on (a rotating or streamed real dataset —
   P7 accepts the dependency). *P0: "a grown unit that is not KEPT is not even growth."* *P4b: "'it persists'
   is a claim about a NEW process."* Red: the |LP| facade must leave its floor (1 distinct value → >1) with the
   same seeds; if it does not, the stimulus was not the limit and the finding is recorded.
3. **Instantiate the declared anatomy in `make_brain()`**: one `(ring, JoinConsumer)` per declared edge, using the
   factories the charge harness already calls; criterion "live `JoinConsumer` count == declared edge count and
   the pinned-payload neuter on any one changes the consumer's live output" — red today (reads 0). *P6: "the
   real remedy is a construction-path change (P4's `make_brain()` owning one workspace and handing it down)."*
4. **T064 unit blocks through the already-built `lake_index` + `lake_segments`**, giving `_T_SYNDELTA` its first
   live writer and `AppendBudget` a tick; discharges 5 of the 10 open dormancy exemptions; port
   `neuro_chip_stream`'s shard discipline first (P2). *P4b directive: substrate lake-resident, RAM a cache.*
   Then T065/T066 at the derived width.
5. **Fix the answering path**: route `Brain.respond` on the `{known, coverage}` the lake already returns
   instead of substring-matching the reply, and delete or content-gate the `reason_out` branch (P8 "deletion on
   a null"). Pre-registered red: the 20-item battery must move from 5 unrelated / 1 self-report / 14 abstain to
   0 / 0 with abstains rising, AND the 4-item positive control must stay 4/4. Then **add the missing SC**: a
   falsifiable reasoning criterion (ARC-eval at the recorded protocol + the everyday battery), so the arc can
   see its weakest clause.
6. **Run G1 and G2 as pure measurement** on the built `EpisodeDriver` against `ReentrantWorkspace` through the
   T032 theta actuator, over S seeds, threshold derived from the measured coalition-drive overlap; a RED is the
   valuable outcome (it cancels three planned builds). In the same pass, replace `charge_probe`'s degenerate
   ignition arm with a non-degenerate neuter so the global workspace's charge question leaves UNCERTAIN.
   *P8: "the criterion is written before the result."*
7. **T072 scoped to unlatch the breaker**: candidate keys must be able to differ (hand-named operators +
   `DEHARDCODE_TRANSFORMS` as generators over a per-symbol allowlist); red = "distinct candidate keys reachable
   at HEAD ≤ 1" (red today); `note_cycle` treats a futile set as exhausted for its source. *P0: "count
   improvements by their effect, never by their number."* T075 (cockpit rows for the breaker) with it.
8. **Clear the E-STOP at the cockpit (a person), then re-run every efferent instrument** — `charge_probe`'s
   drive arms, `AgencyMonitor`, the efference-copy boundary, the amygdala/brainstem bars — none of which has
   produced a non-floor reading since 2026-08-31. No envelope change is implied: the durable latch worked as
   designed. Record the re-measured values beside the latched ones so the two-day floor is not mistaken for
   structure again. *Envelope: never relaxed; the reflex E-STOP is cleared only by a person.*
9. **Cut `KernelTick.deliver` to a flat (post, weight) scatter** (~40×, pure numpy, one module) and give
   specs/008 a `tasks.md`; only then the resident-CUDA arm gated on `select("substrate_tick", n=E)`.
   *P5a: "the crossover is a number, not a doctrine."*
10. **Enforcement hygiene in one commit each**: give `tools/p4c_ratchet.py --assert` a local execution path;
    make `policy_manifest` assert execution (an enabled `run:` step + a recorded execution SHA) rather than a
    substring; re-enable CI or record the local-gate regime as the rule; anchor `.gitignore`'s `lakehouse/` and
    `GATES.md`; fix the 13 `ArmSim` sites; re-derive the Gf number in four files; delete `policy_certification.json`
    or regenerate it. Refuted approaches avoided: no prose-shape purity grep (the broad-citation-guard shape
    that died on its own output), no hand-correction pass on the gitignored `CLAUDE.md`.

T062–T071 and T072–T083 then run in their existing order over an organism that is being fed, moves, composes,
and reasons on the lake it has.

---

## 9. Owner decisions and actions (things this audit could not do)

1. **Make the brain self-restarting.** `.venv-lake/pyvenv.cfg` reads `home = C:\Users\vinhl\AppData\Roaming\uv\python\cpython-3.14.6+freethreaded-...` and `executable = C:\Users\vinhl\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Roaming\uv\...`. Outside the Claude desktop app's container the `home` path does not exist, so `learn_daemon.bat` fails preflight forever after a reboot (the logon launcher is still failing while a session-launched daemon runs beside it, and keeps writing `DAEMON_DOWN.txt`). Fix from a plain PowerShell (not from inside a Claude session): install the free-threaded 3.14.6 to a real path (`uv python install cpython-3.14.6+freethreaded` or the python.org 3.14t installer), repoint `home`/`executable` in `pyvenv.cfg`, run `learn_daemon.bat` from that shell, and confirm `DAEMON_DOWN.txt` disappears. Then register the launcher at logon (`Get-ScheduledTask` finds none today).
2. **The arm.** A real emergency stop at 2026-08-31 12:40:21Z (pid 33048) wrote the durable `estop.latch`; every process since has honoured it, exactly as the T037a rule intends, and the loop's automated HOME clear is refused on every tick. Only a person at the cockpit (http://127.0.0.1:8137/) may clear it. Until it is cleared the efferent, agency and efference instruments read their floor (§8 rank 8). No envelope decision is needed.
3. **CI.** Re-enable the workflow (last run 2026-08-20, FAILED on real jobs) or record in `docs/POLICIES.md` that the local full battery is the gate — the enforcement tables currently claim what CI is not doing.
4. **Roadmap order.** §8 proposes moving five consumption tasks ahead of US3/US4. That is a change to the plan of record and is the owner's call; the audit's recommendation is stated with the measurement behind each rank.
5. **A dependency register?** P7's judgement half is being exercised in commits (the skeptic's finding); whether it needs a `docs/DEPENDENCY_REGISTER.md` is a P7 process choice — the audit records both views.

---

## 10. Method disclosure, the battery, and what was not verified

**Battery.** Run once by the orchestrator in its own commands on the frozen tree at `2c349108` while the
audit agents were still active (so the timings are inflated by contention) — battery #127, ALL GREEN:
`ruff check .` clean (1 s); `run_selfchecks.py` **348/348** (2,534 s; the gate-home E-STOP sidecar was left
by one check and cleared between checks, attributed by mtime-in-slot to `brainstem.autonomous_loop` —
information, not a defect); `pytest -q` green, **coverage 90.74%** (1,655 s); `p4c_ratchet`, `p6_gate`,
`dependency_ledger`, `red_reachability`, `citation_guard`, `doc_command_guard` all OK. The thirteen
policy gates were additionally run at the start of the audit (their outputs are what the lenses read)
and `policy_manifest --assert` reconciled 10 rows with 0 unclaimed. Because CI is dark, this local run
is the only execution of the gate on the last 100 commits; it is green.

**Refutation asymmetry, disclosed.** Across 25 contested items the reproduce-lens refuter sustained 25 of 25
and the policy-lens refuter refuted 25 of 25 on severity. The lenses over-applied "BLOCKER = violation" to
states the policies themselves file as open work; the facts survived every reproduction. Readers should
weight this audit's factual findings highly and its severity labels as the lenses' first draft.

**Not verified.** No lens or refuter ran the full battery (forbidden to avoid collisions; run once by the
orchestrator, above). Whether the two `learn_daemon.py` PIDs observed are parent/child or two brains against
one lake was not established. The knowledge-acquisition avenue's *improvement* (as distinct from its 25,481
records) was not neuter-tested. The GCS H01 synapse export's schema was located, not read. The E-STOP latch
confounds every "constant row" measurement on the live daemon; the lenses that inferred "cannot vary" from
"did not vary" were refuted on that ground and the corrected reading is in §1.

**Instrument defects found in the audit's own tools, logged.** A shell `grep` returning empty non-
deterministically on files with confirmed matches (observation #54; mitigated with absolute-path grep and a
same-command positive control on every negative); a null refuter vote scored as a confirmation (#55); a
gap-probe table whose two columns were the same expression printed twice (a self-comparison, caught by its
Fable refuter); a gap probe run under the default home where the 2026-08-31 `estop.latch` exists, so the
"structural zero" it reported was the latch (caught by both refuters); and a lens that measured a brain at
t=0 and reported every growth counter as absent (caught by its refuter). Every one of these read in the
direction that inflates findings — the same direction as every instrument defect this project has logged.

**Re-proposal guard.** All 16 lenses and 9 probes checked their levers against the refuted-approaches record;
the critic found 0 re-proposals of a recorded negative and 2 intra-audit contradictions (a purity-grep lever
that has the rejected broad-citation-guard shape; a `CLAUDE.md` hand-correction that the file's own rule
forbids) — both dropped from §8.

## Appendix A — the ledger, one line per finding

Format: `[severity · status] title — lens — policy clause (abbreviated) — where`. Status: CONFIRMED = no refuter refuted it; CONTESTED = refuters split (in every such case the runtime refuter sustained the fact and the policy-text refuter dropped the severity); REFUTED = every refuter refuted it, kept as the eyeball pass; MINOR items were not refuted by design. The full claims, reproduction commands, outputs and every refuter's reasoning are in the archived `findings_appendix.md` beside the lens reports.


### A.1 Census — CONFIRMED (55)

- [BLOCKER · CONFIRMED] GrowingLearner — the experience-driven neurogenesis primitive used at eight live sites — has no persistence anywhere; P4b amnesia on every restart — `L11_goal_growth_unlimited` — P4b: "A restart is not amnesia. A brain constructed over an… — src/snn_data_lake/hippocampus/growing_learner.py::GrowingLearner ; src/snn_data_lake/cort…
- [BLOCKER · CONFIRMED] The cockpit's grown_units provenance is falsified at HEAD, in the direction that inflates the apparent growth rate — `L11_goal_growth_unlimited` — Cross-cutting: "never fake a capability, a metric, or a con… — src/snn_data_lake/substrate/brain_telemetry.py::PROVENANCE['grown_units'] ; src/snn_data_…
- [BLOCKER · CONFIRMED] The access→self-model limb is wired and load-bearing but transmits a constant zero on the live brain — `L13_goal_consciousness_program` — P3 dormancy family, the "live but inert" shape; and P0's ho… — src/snn_data_lake/cortex/brain.py::Brain._wake_tick (the access_index=min(...) feed) and …
- [MAJOR · CONFIRMED] The stated reason for never pruning 24,722 _T_SELFBUILD lake records is falsified by those records — `L01_p0_self_improvement` — P4b (the durable lake is part of the brain); P0: "the censu… — src/snn_data_lake/substrate/home.py::save_selfbuild (and the _T_GROWTH/_T_SELFBUILD no-pr…
- [MAJOR · CONFIRMED] docs/POLICIES_EVIDENCE.md's P0 self-building-code row is stale in both directions at once — `L01_p0_self_improvement` — P0: "Their current firing state is a measurement, not a con… — docs/POLICIES_EVIDENCE.md (section 'P0: the avenue status table as of 2026-08-16', self-b…
- [MAJOR · CONFIRMED] The P0 roadmap authority forbids what P0/P7/P8 mandate, on an abolished premise, and cites a class absent from src/ — `L01_p0_self_improvement` — P0: "Where an avenue of self-improvement is missing, that i… — docs/SELF_IMPROVEMENT_RESEARCH.md (section 'Honest scope', the L0 row, and three ArmSim.a…
- [MAJOR · CONFIRMED] The brain's own state report carries no row for P0 — _selfbuild_last reaches no surface — `L01_p0_self_improvement` — P4c: "What the cockpit REPORTS must be what the brain MEASU… — src/snn_data_lake/substrate/cockpit.py; src/snn_data_lake/substrate/brain_telemetry.py; s…
- [MAJOR · CONFIRMED] >=53 core + 112 plasticity-palette production-region literals arbitrate behaviour with no stated provenance — `L02_p1_no_hardcode` — P1: "Every behaviour-arbitrating value must be a learned we… — src/snn_data_lake/cortex/prefrontal_executive.py (16), src/snn_data_lake/thalamus/reentra…
- [MAJOR · CONFIRMED] The picked holdout cap min(60, n//3) sets the resolution of the growth decision — `L02_p1_no_hardcode` — P1: "not a magic constant"; the honesty guard's case (a) do… — src/snn_data_lake/cortex/neurogenesis_engine.py::NeurogenesisSelfImprove._refit
- [MAJOR · CONFIRMED] Cortical column construction: still zero drivers at HEAD, unchanged since 2026-08-16 — `L01_p0_self_improvement` — P0 avenue table: "Wiring a live caller is the open work"; P… — src/snn_data_lake/cortex/cortical_microcircuit.py::thalamic_drive; src/snn_data_lake/subs…
- [MAJOR · CONFIRMED] Three independent LIF membrane transcriptions with differing reset semantics, two carrying no distinct-from note — `L03_p2_duplicates` — P2 line 181: leaving both live additively "dilutes the sign… — src/snn_data_lake/cortex/lif_neuron.py::LIFNeuron (canonical), src/snn_data_lake/cerebell…
- [MAJOR · CONFIRMED] 52% of package source files sit outside the region-charge census denominator by construction — `L04_p3_dormancy` — P3 region-charge rule: "THE DENOMINATOR is the anatomical i… — src/snn_data_lake/substrate/region_charge_gate.py (the NON_ANATOMY exclusion and the REGI…
- [MAJOR · CONFIRMED] Two live durable stores beside the lake, both written DURING this audit: language_acq.npz and 23 MB of skills/*.npz — `L05_p4_one_brain_lake` — P4b: "What the brain learns goes IN the lake ... Not a side… — src/snn_data_lake/cortex/brain.py (self._acq_lm.save(state_path("language_acq.npz")) and …
- [MAJOR · CONFIRMED] The cockpit — the brain's own report of its state — cannot report the brain's durable half or its cumulative size — `L05_p4_one_brain_lake` — P4c: "the page is the brain's own report of its state" and … — src/snn_data_lake/substrate/cockpit.py (no lake readout) + src/snn_data_lake/substrate/br…
- [MAJOR · CONFIRMED] P4c is parked, not falling — 19 days flat at 202, with 30.5% of tree LOC ungoverned and 7 of those modules load-bearing for the governed tiers — `L05_p4_one_brain_lake` — P4c: "A file that exists in this tree is either part of the… — tools/p4c_ungoverned_baseline.json (last lowered 2026-08-13 at 37545dec) + scratchpad/gat…
- [MAJOR · CONFIRMED] Two active record classes have zero live writers and zero records after months — including the per-synapse channel — `L05_p4_one_brain_lake` — P4b RETENTION RULE: "Every record class in the lake is labe… — src/snn_data_lake/substrate/lake_engine.py::SynapseDelta (_T_SYNDELTA 0x09) and src/snn_d…
- [MAJOR · CONFIRMED] docs/PERFORMANCE.md — the project's performance reference — carries three present-tense claims falsified at HEAD, one self-contradicting — `L06_p5_gpu_scale` — Cross-cutting: runtime evidence outranks static reading; an… — docs/PERFORMANCE.md
- [MAJOR · CONFIRMED] The IMMORTAL append budget is wired to nothing while the lake grows a measured 227 MB/day for a 21-unit substrate, with no reclaimer in existence — `L05_p4_one_brain_lake` — P4c RETENTION RULE: "IMMORTAL classes carry a measured byte… — src/snn_data_lake/substrate/lake_engine.py::AppendBudget (and dormancy_gate.EXEMPT['lake_…
- [MAJOR · CONFIRMED] The perf ratchet's binding basis has no unit normalisation, so population growth reads as a performance regression — `L06_p5_gpu_scale` — Cross-cutting: "implement the gate and count its output bef… — src/snn_data_lake/substrate/perf_baseline.py::_norm, tools/perf_baseline.json
- [MAJOR · CONFIRMED] docs/P6_INTERCONNECT.md's status line is falsified at HEAD — `L07_p6_interconnect` — P0 honesty guard (a present-tense claim in a doc that misst… — docs/P6_INTERCONNECT.md (header and section 0)
- [MAJOR · CONFIRMED] The PROVENANCE tag for the integration instrument promises a refreshing reading for a value that was deleted 27 days ago — `L09_p8_instrument_debts` — P8 clause 4 (unprompted disclosure of a precondition that m… — src/snn_data_lake/substrate/brain_telemetry.py::PROVENANCE['pci'] and src/snn_data_lake/s…
- [MAJOR · CONFIRMED] The Potjans-Diesmann cortical column — POLICIES' own named P0 open avenue — is still hand-rolled from math+random at HEAD, and no library was ever ev… — `L08_p7_dependencies` — P7 incident table: "cortex/cortical_microcircuit.py reimple… — src/snn_data_lake/cortex/cortical_microcircuit.py
- [MAJOR · CONFIRMED] The one ratchet that measures P6's named defect has never moved in 26 days — `L07_p6_interconnect` — P6 mechanization: "It ships as a RATCHET, not an arrival: b… — tools/p6_gate.py::BASELINE (undeclared_import_pairs)
- [MAJOR · CONFIRMED] charge_probe's positive internal-read verdict excludes the source class its own comment includes, and prints a false sentence — `L04_p3_dormancy` — P3 honesty guard: "a call site proves a caller, not a charg… — tools/charge_probe.py::main (the guard `if read_ticks > 0 and tbl is not None:`)
- [MAJOR · CONFIRMED] agency_monitor is write-only at runtime and holds the tree's best unconsumed dissociation — `L09_p8_instrument_debts` — P3 no dormancy, per-METHOD amendment; P8 clause 5 (deletion… — src/snn_data_lake/cortex/agency_monitor.py (T079 open)
- [MAJOR · CONFIRMED] The determinism contract is proven at a constructed P=8; the lineage's measured partition count is P=1, where the thread-invariance red cannot fire a… — `L10_crosscutting_invariants` — Cross-cutting invariants, THE DETERMINISM CONTRACT: "its sw… — src/snn_data_lake/substrate/determinism_gate.py, tools/kernel_partition_baseline.json
- [MAJOR · CONFIRMED] The P8 per-item attribution table for the access flag was measured under a salience map the code no longer runs — `L09_p8_instrument_debts` — P8 clause 3: 'Attribution — which part of the invention pro… — src/snn_data_lake/cortex/brain.py::Brain._update_access (docstring, the 10-row reader cen…
- [MAJOR · CONFIRMED] The declared OPEN HOLE now has a number: ten further autograd trainers are permitted solely by unreachability, with no detector to announce a flip an… — `L10_crosscutting_invariants` — Cross-cutting invariants, e-prop OPEN HOLE: "nothing detect… — src/snn_data_lake/reasoning/_loop_common.py::LinearReadout.train_step, reasoning/hjepa_to…
- [MAJOR · CONFIRMED] The integration instrument (T070) is constructed on no live path, and neither dormancy gate can see it — `L09_p8_instrument_debts` — P3 ('reachability is not charge'); P8 clause 5 ('An invente… — src/snn_data_lake/cortex/integration_measure.py::PerturbationalComplexity
- [MAJOR · CONFIRMED] The de-saturated re-entrant workspace is saturated on the live path, and the doc that keeps both ignition mechanisms says the opposite — `L09_p8_instrument_debts` — P8 clause 1 (a criterion that cannot go red) applied to the… — src/snn_data_lake/thalamus/reentrant_workspace.py, src/snn_data_lake/cortex/brain.py::Bra…
- [MAJOR · CONFIRMED] The entire multi-region growth interface is built with zero live callers, and its dated consumer has expired — `L11_goal_growth_unlimited` — P3: "A faculty is not 'done' until it is one of three dispo… — src/snn_data_lake/cortex/placement.py::place_grown ; src/snn_data_lake/cortex/placement.p…
- [MAJOR · CONFIRMED] The lake-resident lane that 'unlimited' requires is 0% built and 100% exempt-pending-wire — `L11_goal_growth_unlimited` — P4b: "the one durable lake is not storage the brain writes … — src/snn_data_lake/substrate/kernel_state.py ; src/snn_data_lake/substrate/lake_engine.py …
- [MAJOR · CONFIRMED] The goal's headline quantity — the unit count — has no tracked instrument, no baseline and no reachable red — `L11_goal_growth_unlimited` — P8: "A success criterion with a reachable RED ... a measure… — tools/perf_baseline.json ; src/snn_data_lake/substrate/growth_coverage_gate.py::BASELINE_…
- [MAJOR · CONFIRMED] The acquired language organ emits non-words and its coherence guard passes them 30/30 — `L12_goal_reasoning_thinking` — P8: a criterion whose red is unreachable for the inputs it … — src/snn_data_lake/cortex/brain.py::Brain._language / ::Brain._coherent, src/snn_data_lake…
- [MAJOR · CONFIRMED] docs/DEPTH_CEILINGS.md's headline Gf number is stale by ~2 points and contradicts its own body — `L12_goal_reasoning_thinking` — Verify-first: 'report the true number including a regressio… — docs/DEPTH_CEILINGS.md
- [MAJOR · CONFIRMED] The spiking half of gf.solve_series earns zero unique capability -- only a label — `L12_goal_reasoning_thinking` — P8 clause 5: 'Deletion on a null. An invented method that m… — src/snn_data_lake/reasoning/gf.py::solve_series (induce_shift / _exact_readout)
- [MAJOR · CONFIRMED] CLAUDE.md reports the access-consciousness instrument as "still 0" when the gate measures 1 — `L13_goal_consciousness_program` — Verify-first, and P6's own guard "Re-derive with it rather … — CLAUDE.md (line 24) vs tools/p6_gate.py output at HEAD
- [MAJOR · CONFIRMED] README, PITCH and OVERVIEW describe a different project from the one P0 and directives (a)/(e) commit to — `L13_goal_consciousness_program` — P0: "This is the project's purpose, and it outranks the res… — README.md, docs/PITCH.md, docs/OVERVIEW.md
- [MAJOR · CONFIRMED] The one capability both consciousness documents name as the concrete missing build — per-step credit assignment on a multi-step plan — has zero repre… — `L13_goal_consciousness_program` — P0: "Where an avenue of self-improvement is missing, that i… — specs/022-unified-brain-arc/{spec.md,tasks.md,SYNTHESIS.md} vs docs/CONSCIOUSNESS_METHODS…
- [MAJOR · CONFIRMED] specs/019's root blocker 'cortical-prior-module-absent' is falsified at HEAD and is gating the highest-leverage growth chain — `L14_spec022_arc` — Cross-cutting invariant: "Runtime evidence outranks static … — specs/019-consolidated-open-work/tasks.md::[B6.08]; src/snn_data_lake/cortex/cortical_pri…
- [MAJOR · CONFIRMED] The cockpit still serves a PCI provenance row promising a periodic integration reading whose producer was deleted 27 days ago — `L13_goal_consciousness_program` — P4c: "What the cockpit REPORTS must be what the brain MEASU… — src/snn_data_lake/substrate/brain_telemetry.py (PROVENANCE entry "pci") and src/snn_data_…
- [MAJOR · CONFIRMED] tasks.md's own stated total is wrong by 14, inflating apparent completion by 12 points — `L14_spec022_arc` — Cross-cutting invariant, verify-first: "report the true num… — specs/022-unified-brain-arc/tasks.md::Implementation-strategy
- [MAJOR · CONFIRMED] The live cockpit discloses none of the arc's red rows, including the growth red and the pinned scaffold set it is required to show — `L14_spec022_arc` — P4c: "What the cockpit REPORTS must be what the brain MEASU… — src/snn_data_lake/substrate/cockpit.py; specs/022-unified-brain-arc/tasks.md::T084 (open)
- [MAJOR · CONFIRMED] Three specs 022 declares COMPOSED/ABSORBED carry 159 open tasks with no supersession record; and specs/019 does not know 022 exists — `L14_spec022_arc` — P2 non-additive rule, applied to plans the way POLICIES app… — specs/022-unified-brain-arc/spec.md::Assumptions; specs/007-whole-brain-neurogenesis/spec…
- [MAJOR · CONFIRMED] Language ceiling unmoved and re-measured; trained spiking LMs are at or below chance and none is wired to the chat lane — `L12_goal_reasoning_thinking` — Verify-first + honest framing: 'never fake a capability, a … — src/snn_data_lake/reasoning/blimp_bench.py, models/ar_spiking_lm.npz, models/eprop_lm.npz…
- [MAJOR · CONFIRMED] The dormancy exemption surface is accreting at ~23x the rate of the tree (21 -> 52 rows against 397 -> 423 src files) — `L15_gate_integrity_meta` — P3 — "No dormancy"; the gate's own contract, "a 'retire'/'w… — src/snn_data_lake/substrate/dormancy_gate.py::EXEMPT
- [MAJOR · CONFIRMED] The exemption-guard test for red_reachability is vacuous at HEAD, and its length check is mis-shaped for the entry format the module documents — `L15_gate_integrity_meta` — P8 — "a metric that cannot fail is not"; the gate's own doc… — tests/test_red_reachability.py::test_stale_exemption_is_reported
- [MAJOR · CONFIRMED] The transport gate's printed blind spot names one uncovered spelling; measured, there are five — `L15_gate_integrity_meta` — e-prop invariant — "its green never means 'compliant' and i… — src/snn_data_lake/substrate/eprop_transport_gate.py::main (BLIND SPOT block), ::_findings…
- [MAJOR · CONFIRMED] 28% of the tree's learning rules sit in an UNCERTAIN bucket that is reported and never asserted — the exact bucket a silently-stopped learning rule o… — `L15_gate_integrity_meta` — P3 amendment — "dormancy is per-METHOD, not per-module" — src/snn_data_lake/substrate/method_dormancy_gate.py
- [MAJOR · CONFIRMED] Nothing on the answering path deliberates: the train of thought and the spiking working memory are both off the chat lane — `L12_goal_reasoning_thinking` — P6: 'a faculty satisfies P6 when ... at least one other fac… — src/snn_data_lake/reasoning/spike_chat_lake.py::LakeChat.think, src/snn_data_lake/cortex/…
- [MAJOR · CONFIRMED] Six exemptions record an already-passed retirement bound; the gate prints them as STANDING TODO and no gate in the tree asserts on an expired bound — `L15_gate_integrity_meta` — P3 amendment; the gate's own contract, "a 'retire'/'wire' e… — src/snn_data_lake/substrate/dormancy_gate.py::main (open_dispositions branch), src/snn_da…
- [MAJOR · CONFIRMED] P8's mechanization is opt-in by naming convention: 14 criteria found in 423 files, and none of the gates' own verdict functions are inside the census — `L15_gate_integrity_meta` — P8 — "Mechanization: BUILT AND ASSERTING -- tools/red_reach… — tools/red_reachability.py::CRITERION_PREFIXES, ::_is_criterion
- [MAJOR · CONFIRMED] The repo CLAUDE.md -- read by every agent session -- carries a falsified dependency list, a falsified worked example of its own honesty guard, and th… — `L16_docs_reality_drift` — POLICIES.md, Cross-cutting invariants -- 'Verify-first + ho… — D:\AI-project\spiking-neural-data-lake\CLAUDE.md (the dependency paragraph at the top, an…
- [MAJOR · CONFIRMED] docs/ANATOMY.md -- the brain-completeness document, i.e. the one that answers the growth question -- understates the brain at HEAD and omits a whole … — `L16_docs_reality_drift` — POLICIES.md, Cross-cutting invariants -- 'report the true n… — docs/ANATOMY.md (the Allen top-level coverage table, its 'Corrected 2026-07-28' summary p…
- [MAJOR · CONFIRMED] The module named "NON-saturated Global Neuronal Workspace" reads availability 1.000 on the live brain, and the P2 collapse it declared is undischarged — `L13_goal_consciousness_program` — FR-028: "a 100% ignition rate is a saturated detector". P2:… — src/snn_data_lake/thalamus/reentrant_workspace.py::ReentrantWorkspace (availability, acce…

### A.2 Census — CONTESTED (25)

- [BLOCKER · CONTESTED] The brain cannot restart itself: DAEMON_DOWN.txt is live at HEAD and the venv base interpreter resolves only inside the Claude app container — `L01_p0_self_improvement` — P0: "The brain must not be a fixed artifact that a human up… — .venv-lake/pyvenv.cfg (executable= line), learn_daemon.bat (preflight loop), DAEMON_DOWN.…
- [BLOCKER · CONTESTED] The only shipped de-hardcoding mechanism cannot apply at HEAD — all 8 patch anchors embed CRLF against an LF tree — `L02_p1_no_hardcode` — P1 corollary: "never keep a constant fixed — when a hardcod… — src/snn_data_lake/basal_ganglia/brain_native_selfbuild.py::DEHARDCODE_TRANSFORMS (entry m…
- [BLOCKER · CONTESTED] substrate/self_mod_harness.py's module docstring is falsified at HEAD and points P0 work at an already-finished task — `L01_p0_self_improvement` — Cross-cutting invariant: "runtime evidence outranks static … — src/snn_data_lake/substrate/self_mod_harness.py (module docstring)
- [BLOCKER · CONTESTED] Nine growth mechanisms where the spec requires exactly one; the goal's first capability has never been converged — `L03_p2_duplicates` — P2: "When two mechanisms do the same job, exactly one survi… — src/snn_data_lake/substrate/growable.py (the declared adopt-target), plus cortex/time_cel…
- [BLOCKER · CONTESTED] P0's headline avenue is latched shut by construction: 24,716 proposals, 0 kept, 3 rolled back — `L01_p0_self_improvement` — P0 honesty guard: "a grown unit that is not KEPT is not eve… — src/snn_data_lake/basal_ganglia/brain_native_selfbuild.py::BrainNativeSelfBuildProposer.s…
- [BLOCKER · CONTESTED] P2's own worked example is falsified at HEAD: the Izhikevich dynamics are transcribed three times — `L03_p2_duplicates` — P2, verbatim: neuron_zoo "does NOT re-implement any dynamic… — src/snn_data_lake/cortex/neuron_zoo.py::_IzhikevichCell.step, src/snn_data_lake/cortex/he…
- [BLOCKER · CONTESTED] Lake episode recall keeps both implementations behind a fall-through `if`, and no test can tell them apart — `L03_p2_duplicates` — P2 merge guard, verbatim: "'Merging' that keeps both code p… — src/snn_data_lake/cortex/brain.py::Brain.recall_episode_durable (window tier -> hippocamp…
- [BLOCKER · CONTESTED] The lake/kernel growth substrate is exempt-'wire' on seven self-declared EXPIRED-UNMET rows, and no red can fire — `L04_p3_dormancy` — P3 disposition 3: "exempt — legitimately not on the live ti… — src/snn_data_lake/substrate/dormancy_gate.py::EXEMPT (rows lake_segments, lake_bridge, la…
- [BLOCKER · CONTESTED] The whole specs/022 persistence engine is built, green and UNWIRED — 5 of 10 open dormancy exemptions, two recording their own bound as EXPIRED UNMET — `L05_p4_one_brain_lake` — Cross-cutting P3: every module is "wired+charged, retired, … — src/snn_data_lake/substrate/dormancy_gate.py::EXEMPT (rows lake_engine, lake_index, lake_…
- [BLOCKER · CONTESTED] The global workspace passes all four of P3's verifications and changes no measured outcome — `L04_p3_dormancy` — P3: "Being imported, being called, being in the snapshot, a… — src/snn_data_lake/cortex/brain.py::Brain._update_access (and ::Brain._wake_tick, which ca…
- [BLOCKER · CONTESTED] docs/BRAINCOG_COMPARISON.md rejects an entire capability class on the rule P7 abolished, and describes the tree in a present tense falsified at HEAD — `L08_p7_dependencies` — P7: "A rejection that cites dependency count, install weigh… — docs/BRAINCOG_COMPARISON.md
- [BLOCKER · CONTESTED] The cockpit's own anti-lying provenance mechanism was built for access_index and was never applied to it — `L09_p8_instrument_debts` — P8 honesty guard (a) 'Invention is credited at measurement,… — src/snn_data_lake/substrate/cockpit.py (big/tag/PROV) and src/snn_data_lake/substrate/bra…
- [BLOCKER · CONTESTED] The enforcement tables claim CI mechanization eleven times while the workflow is disabled_manually and 100 commits behind, and the instrument that wo… — `L10_crosscutting_invariants` — Cross-cutting invariants, gated+reversible amendment: "the … — docs/POLICIES.md (enforcement tables, lines 34/42/43/44/45/60/61/62/835/844/992), .github…
- [BLOCKER · CONTESTED] The one persisted grower grows on a synthetic internal Markov chain, not on the brain's experience — `L11_goal_growth_unlimited` — P0 honesty guard: "a grown unit that changes no outcome is … — src/snn_data_lake/cortex/brain.py::Brain._sleep_tick ; src/snn_data_lake/cortex/neurogene…
- [BLOCKER · CONTESTED] The tree's headline consciousness number is byte-identical to a fully neutered producer on the live path — `L09_p8_instrument_debts` — P8 clause 1 + the P8 source line it derives from: "A mechan… — src/snn_data_lake/cortex/brain.py::Brain._update_access and src/snn_data_lake/cortex/acce…
- [BLOCKER · CONTESTED] CLAUDE.md names a present-tense P8 "live defect" in the consciousness reply that is falsified four ways at HEAD — `L13_goal_consciousness_program` — P8 honesty guard: "a 'capability' reported without a measur… — CLAUDE.md (line 46, the P8 bullet); falsified against src/snn_data_lake/reasoning/conscio…
- [BLOCKER · CONTESTED] The one chat route named 'reasoning' does not reason about the question, and it preempts a correct answer — `L12_goal_reasoning_thinking` — P3 dormancy shape 2, 'return value discarded -- a live call… — src/snn_data_lake/cortex/brain.py::Brain.reason and ::Brain.respond
- [BLOCKER · CONTESTED] At HEAD the neurons are not what reasons: the answers survive silencing every LIF neuron — `L12_goal_reasoning_thinking` — P0 avenue list is a BUILD list ('an avenue that does not fi… — src/snn_data_lake/cortex/brain.py::Brain.respond, src/snn_data_lake/reasoning/gf.py::solv…
- [BLOCKER · CONTESTED] The S1 stage gate was closed with its headline criterion red, and none of the 26 remaining tasks owns the miss — `L14_spec022_arc` — tasks.md Dependencies: "each stage's exit gates are the nex… — specs/022-unified-brain-arc/tasks.md::T061, ::Dependencies, ::T068; specs/022-unified-bra…
- [BLOCKER · CONTESTED] Goal clause (ii) - reasoning and thinking - has no success criterion, FR, or task anywhere in the 106-task arc — `L14_spec022_arc` — P8 clause 1: "A success criterion with a reachable RED"; P0… — specs/022-unified-brain-arc/spec.md::Success-Criteria (SC-001..SC-010); specs/022-unified…
- [BLOCKER · CONTESTED] policy_manifest, the meta-gate that certifies gate registration, decides in_ci by substring and passes a commented-out step — `L15_gate_integrity_meta` — "a gate that has only ever been seen green is indistinguish… — src/snn_data_lake/substrate/policy_manifest.py::cross_check
- [BLOCKER · CONTESTED] Four policy gates are unenforced at HEAD: the CI workflow is disabled_manually, its last run failed, and 103 commits are unverified — while POLICIES … — `L15_gate_integrity_meta` — "'CI will catch it' is a claim about a run that has EXECUTE… — docs/POLICIES.md (enforcement table rows P6/P7/P8/P4c), .github/workflows/ci.yml, tools/p…
- [BLOCKER · CONTESTED] The published install contract is falsified: README says Python 3.12+ and 'three, not two' dependencies; HEAD requires 3.14 free-threaded and lists f… — `L16_docs_reality_drift` — POLICIES.md, Cross-cutting invariants -- 'Verify-first + ho… — README.md (the python-3.12%2B badge, the 'Install' section, and the 'core dependencies --…
- [BLOCKER · CONTESTED] cortex/brain.py's SAFETY paragraph names ArmSim as the sole path to motion; that class does not exist, and the 2026-08-04 correction pass fixed 3 fil… — `L16_docs_reality_drift` — POLICIES.md, Cross-cutting invariants -- 'Verify-first + ho… — src/snn_data_lake/cortex/brain.py::<module docstring>, src/snn_data_lake/brainstem/autono…
- [BLOCKER · CONTESTED] docs/PITCH.md and docs/OVERVIEW.md describe a v2.70-era project with no staleness marker; OVERVIEW contradicts itself and repeats the exact framing P… — `L16_docs_reality_drift` — POLICIES.md P4c (via README's own gloss: "'it is only a ben… — docs/PITCH.md, docs/OVERVIEW.md

### A.3 Census — REFUTED (43)

- [BLOCKER · REFUTED] The duplicate-capability guard cannot answer a one-word query and fails permissively, so P2 has no working automated enforcement — `L03_p2_duplicates` — Cross-cutting invariant: "implement the gate and count its … — C:\Users\vinhl\.claude\hooks\graph-exists.js (called by the Tier-0 sndl-graphify-guard.ps…
- [BLOCKER · REFUTED] eps_tol=0.28 is an undisclosed picked constant that caps the brain's neuron count on the live path — `L02_p1_no_hardcode` — P1: "Every behaviour-arbitrating value must be a learned we… — src/snn_data_lake/cortex/neurogenesis_engine.py::NeurogenesisSelfImprove.__init__ (eps_to…
- [BLOCKER · REFUTED] The substrate is snapshotted INTO the lake, not resident in it — the UnitBlock class the contract requires exists in zero lines of code — `L05_p4_one_brain_lake` — contracts/lake-records.md: "UnitBlock | IMMORTAL (current +… — specs/022-unified-brain-arc/contracts/lake-records.md (record-class table) + src/snn_data…
- [BLOCKER · REFUTED] The cockpit PROVENANCE contract is falsified at HEAD: 112 constant rows untagged, 14 of 17 region bars frozen — `L04_p3_dormancy` — P3: "shapes 5 and 6 are why substrate/brain_telemetry.py::P… — src/snn_data_lake/substrate/brain_telemetry.py::PROVENANCE (contract) and src/snn_data_la…
- [BLOCKER · REFUTED] The entire scale path is written and dormant; the LIVE representation's binding width (~46,200 units) sits below H01's 57,000 — `L06_p5_gpu_scale` — P3: "no brain module may be silently dormant"; and P5a: "th… — src/snn_data_lake/substrate/kernel_state.py::KernelState, src/snn_data_lake/substrate/ker…
- [BLOCKER · REFUTED] specs/009 diagnoses its own scale ceiling as the NEURON representation, and specs/008 — the spec that owns it — has never been broken into tasks — `L06_p5_gpu_scale` — P2: "port the unique capability first" / adopt-or-merge, th… — specs/009-sparse-connectivity/tasks.md, specs/009-sparse-connectivity/research.md, specs/…
- [BLOCKER · REFUTED] reasoning/ -- the region that would carry thinking -- is outside the entire P6 apparatus — `L07_p6_interconnect` — P6: "A faculty must not be an island." — src/snn_data_lake/reasoning/ (94 files); src/snn_data_lake/reasoning/rpm_solver.py, src/s…
- [BLOCKER · REFUTED] No two regions are coupled within a tick: 0 of 91 connectome edges is delay-0 — `L07_p6_interconnect` — P6: "at least one other faculty consumes its output in a wa… — src/snn_data_lake/substrate/region_wiring.py::same_tick_dependency_map; src/snn_data_lake…
- [BLOCKER · REFUTED] Four reachable, unadjudicated BPTT+Adam trainers sit on the tier POLICIES declares in scope with no exemption, and the only detector is masked so it … — `L10_crosscutting_invariants` — Cross-cutting invariants, e-prop SCOPE clause: "There is NO… — src/snn_data_lake/training/mnist.py::train_model, src/snn_data_lake/training/trainer.py::…
- [BLOCKER · REFUTED] A tracked certification artifact reads certified:True for a tree 141 commits stale and recorded as dirty, and nothing asserts its freshness — `L10_crosscutting_invariants` — Cross-cutting invariants, gated+reversible amendment: "'CI … — tools/policy_certification.json, src/snn_data_lake/substrate/policy_manifest.py::CERTIFIC…
- [BLOCKER · REFUTED] "Grow from ONE neuron" has no implementation; the brain is constructed from fixed pre-sized populations — `L11_goal_growth_unlimited` — P0: "growing its own neural substrate. Where an avenue of s… — src/snn_data_lake/cortex/brain.py::Brain.__init__ ; src/snn_data_lake/cortex/growth_progr…
- [BLOCKER · REFUTED] buffer_cap=1000 reinstates, one level down, the artificial growth ceiling the module's docstring claims to have removed — `L11_goal_growth_unlimited` — P1 corollary: "never keep a constant fixed -- when a hardco… — src/snn_data_lake/cortex/neurogenesis_engine.py::NeurogenesisSelfImprove.__init__ ; src/s…
- [BLOCKER · REFUTED] The gate deciding a reply is a crystallized FACT is a two-string denylist with no reachable red for ungroundedness — `L12_goal_reasoning_thinking` — P8: 'A mechanism whose success criterion cannot go red is w… — src/snn_data_lake/cortex/brain.py::Brain._grounded
- [BLOCKER · REFUTED] The declared typed lane carries zero traffic in the running brain — `L07_p6_interconnect` — P6 condition (a): satisfied when "it is reachable from at l… — src/snn_data_lake/cortex/brain.py::make_brain; src/snn_data_lake/substrate/learn_daemon.p…
- [BLOCKER · REFUTED] G1 and G2 — the algorithm's own blocking gates — have zero representation in the plan of record, and G1 is measurably firing on the live brain — `L13_goal_consciousness_program` — docs/PHENOMENAL_CONSCIOUSNESS_ALGORITHM.md §3: "Blocking ga… — specs/022-unified-brain-arc/spec.md, tasks.md, SYNTHESIS.md (G1/G2 absent); measured agai…
- [MAJOR · REFUTED] Brain.self_improve and godel_machine.attempt_on_repo still have zero call sites, 28 days after both were recorded as such — `L01_p0_self_improvement` — P3: "every module is wired+charged, retired, or EXEMPT-with… — src/snn_data_lake/cortex/brain.py::Brain.self_improve; src/snn_data_lake/reasoning/godel_…
- [MAJOR · REFUTED] PATIENCE = 3 is a disclosed pick that fits none of P1's exceptions and is what is currently stopping growth — `L02_p1_no_hardcode` — P1 honesty guard: "a constant may stay fixed in exactly thr… — src/snn_data_lake/cortex/growth_stop.py::PATIENCE (and ::_WINDOW, derived from it)
- [MAJOR · REFUTED] Two AER spike-telemetry stores with an identical query contract, identical answers, and no adjudication or distinct-from note — `L03_p2_duplicates` — P2: "exactly one survives"; merge guard: "Count the survivo… — src/snn_data_lake/substrate/spike_telemetry_hub.py::SpikeTelemetryHub vs src/snn_data_lak…
- [MAJOR · REFUTED] The workspace duo's deciding benchmark is fully specified and owned by no task in the active spec — `L03_p2_duplicates` — P2 STANDING ADJUDICATION, verbatim: "a set of ambiguous mul… — src/snn_data_lake/thalamus/global_workspace.py vs src/snn_data_lake/thalamus/reentrant_wo…
- [MAJOR · REFUTED] Six runnable developmental-loop drivers, and the tree's two authorities record contradictory dispositions for one of them — `L03_p2_duplicates` — P2 route 1: "one is better on evidence; it becomes THE path… — src/snn_data_lake/reasoning/autonomous_brain.py, autonomous_brain_recovered.py, developme…
- [MAJOR · REFUTED] Half the behavioural-charge observable is pinned: arm HOLD 1.000 in both arms in every run — `L04_p3_dormancy` — P3: the charge tier is "proven by the runtime neuter-A/B pr… — tools/charge_probe.py::run (the arm_stopped / act-hold observable)
- [MAJOR · REFUTED] The measured width router reaches 1 production call site of 59; the crossover is computed and not consumed — `L06_p5_gpu_scale` — P5a: "A caller that routes on AVAILABILITY ... instead of t… — src/snn_data_lake/substrate/device.py::resolve_device, src/snn_data_lake/substrate/device…
- [MAJOR · REFUTED] The ONE merged e-prop training kernel — the tier P5 names as where the GPU pays — defaults to CPU and never enters the router — `L06_p5_gpu_scale` — P5: "Where the GPU pays, and should be used without hesitat… — src/snn_data_lake/cortex/eprop_kernel.py::EpropEncoderBase, src/snn_data_lake/cortex/epro…
- [MAJOR · REFUTED] Acetylcholine is a mathematical constant; its wake-tick consumer is an identity multiply and it is registered as live — `L04_p3_dormancy` — P3 dormancy shape 5: "constant wearing a signal's clothes —… — src/snn_data_lake/neuromodulation/acetylcholine.py::Acetylcholine.step, consumed at src/s…
- [MAJOR · REFUTED] 21 of 74 learning rules have no resolved live call site (UNCERTAIN, hand-adjudicated) — `L04_p3_dormancy` — P3 amendment: "A learning-verb method on a reachable module… — src/snn_data_lake/substrate/method_dormancy_gate.py (census output) — affected rules incl…
- [MAJOR · REFUTED] The crossover that routes the whole tree is measured with host->device transfer excluded; CUDA loses at that width once transfer is paid — `L06_p5_gpu_scale` — P5 honesty guard (a): "A GPU path ships with its benchmark,… — src/snn_data_lake/substrate/device.py::_bench
- [MAJOR · REFUTED] The faculty-wiring registry P6 was written from still has zero registrants — `L07_p6_interconnect` — P6 source line: "substrate/region_wiring.py, whose docstrin… — src/snn_data_lake/substrate/region_wiring.py::_WIRED, ::wire, ::charge_all; src/snn_data_…
- [MAJOR · REFUTED] The shared bus payload carries no submitter identity, so faculty-to-faculty consumption through it is unprovable in principle — `L07_p6_interconnect` — P6 condition (b): "at least one other faculty consumes its … — src/snn_data_lake/cortex/brain.py (wiring comment block, F1/F2); src/snn_data_lake/thalam…
- [MAJOR · REFUTED] numba won its benchmark 16x and P7 explicitly put it back on the table; the re-measurement P5 calls inevitable has no trigger — `L08_p7_dependencies` — POLICIES_EVIDENCE P7 reframe: "The one item here that P7 pu… — docs/POLICIES_EVIDENCE.md (P5a relocated block) — no consumer in pyproject.toml or src/
- [MAJOR · REFUTED] P7 guard (b) has no artifact: docs/ carries 92 purity arguments and 0 of the 10 P7 VOID annotations; P8 got a register, P7 did not — `L08_p7_dependencies` — P7 honesty guard: "Symmetrically, a rejection ships with it… — docs/ (21 files) — and the absent docs/DEPENDENCY_REGISTER.md
- [MAJOR · REFUTED] P5a asserts the free-threading axis pays; the repo's own pinned substrate-tick sweep measures it losing, and the policy carries no re-derive caveat o… — `L06_p5_gpu_scale` — P5a: "that work spreads across cores with no GIL serializin… — docs/POLICIES.md (### P5a), tools/kernel_partition_baseline.json, src/snn_data_lake/subst…
- [MAJOR · REFUTED] structural_rewiring runs 1203 cycles on the live sleep tick and does nothing, published as a constant 0.0 with no provenance — `L11_goal_growth_unlimited` — P3 dormancy shape 5 ("constant wearing a signal's clothes")… — src/snn_data_lake/plasticity/structural_rewiring.py ; src/snn_data_lake/substrate/brain_t…
- [MAJOR · REFUTED] second_order_access never runs on the autonomous tick — the metacognitive access gate exists only while a human is typing — `L09_p8_instrument_debts` — P3 no dormancy (presence is not participation); P8 clause 1… — src/snn_data_lake/cortex/second_order_access.py; call site src/snn_data_lake/cortex/brain…
- [MAJOR · REFUTED] cortex/spiking_lm_serve.py is an empty pipe carrying no P3 disposition — `L12_goal_reasoning_thinking` — P3: a faculty is not done until it is wired+charged, retire… — src/snn_data_lake/cortex/spiking_lm_serve.py
- [MAJOR · REFUTED] The Gf Phase-3 mint has no consumer inside the solver; the recorded P2 migration obligation is still open — `L12_goal_reasoning_thinking` — P2 adopt-or-merge, then retire the inferior (non-additive) … — src/snn_data_lake/reasoning/arc_mint.py, src/snn_data_lake/reasoning/arc_residual.py, src…
- [MAJOR · REFUTED] plan.md's Constitution Check claims the feature IS P0's avenue list; two of P0's five avenues appear in none of the 106 tasks — `L14_spec022_arc` — P0: "the avenue list is a BUILD LIST, not a status ledger";… — specs/022-unified-brain-arc/plan.md::Constitution-Check-P0; src/snn_data_lake/cortex/cort…
- [MAJOR · REFUTED] T062 plans a second H01 ingest mechanism with no P2 adjudication against cortical_prior.py, and no P4b disposition for putting the prior in the lake — `L14_spec022_arc` — P2: "When two mechanisms do the same job, exactly one survi… — specs/022-unified-brain-arc/tasks.md::T062, ::T064; src/snn_data_lake/cortex/cortical_pri…
- [MAJOR · REFUTED] P6 consumption is 1 read against 19 declared edges, and US4 is scheduled to mutate a brain whose faculties do not read each other — `L14_spec022_arc` — P6: "a faculty satisfies P6 when ... at least one other fac… — specs/022-unified-brain-arc/spec.md::SC-001; specs/022-unified-brain-arc/tasks.md::T072-T…
- [MAJOR · REFUTED] AgencyMonitor runs on every live tick with no reader — a declared, undischarged P3 case in the consciousness chain — `L13_goal_consciousness_program` — P3: "No dormancy" — reachability is not charge; the dormanc… — src/snn_data_lake/cortex/agency_monitor.py::AgencyMonitor, written from src/snn_data_lake…
- [MAJOR · REFUTED] Growth is REACHABLE everywhere and CHARGED nowhere — no growth edge appears in the region-charge evidence — `L11_goal_growth_unlimited` — P3: "reachability proves a module is imported, NOT that its… — tools/region_charge_evidence.json ; src/snn_data_lake/substrate/region_charge_gate.py
- [MAJOR · REFUTED] The project's stated goal is absent from every owner-facing surface: the repo publishes ~40 numbers and not one of them is on the growth axis P0 names — `L16_docs_reality_drift` — POLICIES.md P0 -- 'This is the project's purpose, and it ou… — README.md (opening framing and the three 'Results (measured)' tables), docs/PITCH.md, doc…
- [MAJOR · REFUTED] docs/TRAINING.md's 'read this first on a fresh clone' paragraph is false in three of its four sentences, including a runtime-refutable claim that a f… — `L16_docs_reality_drift` — POLICIES.md, Cross-cutting invariants -- 'Runtime evidence … — docs/TRAINING.md, section 'Regenerating the perception weights -- read this first on a fr…
- [MAJOR · REFUTED] No instrument reports cumulative lifetime unit growth — the headline quantity of the project's stated goal resets with the process — `L15_gate_integrity_meta` — P0 — "The brain must autonomously improve itself, by every … — src/snn_data_lake/cortex/brain.py (state dict), cockpit /api/all provenance

### A.4 Census — MINOR (22)

- [MINOR] The one surviving self-build candidate's empirical premise is falsified on the live daemon — `L01_p0_self_improvement`
- [MINOR] _REFRESH_GAIN cannot be replaced through the module symbol that names it — `L02_p1_no_hardcode`
- [MINOR] SAFE_DISTANCE = CONTACT * 3.0 duplicates an undisclosed multiplier across two files — `L02_p1_no_hardcode`
- [MINOR] A 2.0 GB near-duplicate of the ONE lake sits beside it on disk — `L03_p2_duplicates`
- [MINOR] spikes.medulla and spikes.hypothalamus are an undisclosed live duplicate — `L04_p3_dormancy`
- [MINOR] 20 of 41 connectome nodes are named by no module in their own region — `L04_p3_dormancy`
- [MINOR] Sleep consolidation rewrites the ENTIRE lake, an O(total-history) cost under a never-forget mandate — `L05_p4_one_brain_lake`
- [MINOR] specs/022 T065 cites tools/perf_baseline.py, which does not exist — `L06_p5_gpu_scale`
- [MINOR] The in-degree/out-degree gate P6 specifies is not built -- I built it, and its output kills it (reported as a null) — `L07_p6_interconnect`
- [MINOR] substrate/arm_view.py::encode_png still hand-writes a PNG encoder while pillow sits declared; the deferral has outlived P7 by 28 days — `L08_p7_dependencies`
- [MINOR] docs/AIRLLM_ANALYSIS.md states the core has "two declared dependencies"; HEAD has five — `L08_p7_dependencies`
- [MINOR] POLICIES.md's line cite for falsifier F11 does not resolve at HEAD — `L09_p8_instrument_debts`
- [MINOR] The live consciousness reply renders a float as an emotion name and makes a capability claim with no same-moment reading — `L09_p8_instrument_debts`
- [MINOR] Four growth counters are published to the cockpit with no PROVENANCE row — `L11_goal_growth_unlimited`
- [MINOR] The one unbounded, linear-memory grower in the tree is invisible to the growth-coverage census and carries no anatomy — `L11_goal_growth_unlimited`
- [MINOR] The reasoning tier is largely outside the e-prop invariant's asserted scope — `L12_goal_reasoning_thinking`
- [MINOR] The endogenous attention goal port is still goal=None, unchanged for 30 days — `L13_goal_consciousness_program`
- [MINOR] graphify query cannot orient work on the consciousness chain — `L13_goal_consciousness_program`
- [MINOR] spec.md cites a SYNTHESIS.md clause that no longer exists in SYNTHESIS.md — `L14_spec022_arc`
- [MINOR] specs/BACKLOG_RANKED.md is derived at a stale HEAD, is partly closed and partly live, and is duplicated by specs/019 — `L14_spec022_arc`
- [MINOR] Four ratchets are holding 'no worse' with no improving move for 6-19 days, while the one being actively worked tightened yesterday — `L15_gate_integrity_meta`
- [MINOR] docs/BRAIN_REVIEW.md carries no date, commit or version anchor, so its accuracy at HEAD is luck rather than verifiability — `L16_docs_reality_drift`

### A.5 Gap round (39; CONFIRMED 5, CONTESTED 18, REFUTED 16)

- [BLOCKER · CONFIRMED] Three README results numbers are falsified by their own producers and a fourth has no producer at all — all four drift in the flattering direction — `GAP_B2_readme_results_block_unverified` — POLICIES.md, Cross-cutting invariants: "never fake a capabi… — README.md (rows at :92, :95, :96, :112); propagated to docs/OVERVIEW.md:94 and docs/PITCH…
- [MAJOR · CONFIRMED] The brain's VALENCE axis is pinned at 0.0 and its three-factor rule is two-factor at HEAD — `GAP_A1_subcortical_affective_regions_unread` — P3 dormancy shape #5 ("constant wearing a signal's clothes"… — src/snn_data_lake/amygdala/affective_appraisal.py::AffectiveAppraisal.appraise (driven fr…
- [MAJOR · CONFIRMED] lakehouse/ is dormant AND unexecutable at HEAD: zero importers, not a REGIONS key, and the module cannot be imported in .venv-lake — `GAP_A5_lakehouse_absent_from_generated_inventory` — P3 — "No dormancy"; and the project's own gloss, reachabili… — src/snn_data_lake/lakehouse/medallion.py, src/snn_data_lake/lakehouse/delta_demo.py
- [MAJOR · CONFIRMED] Two owner-facing documents quote different current test coverage, and one cites a source that does not exist — reviving the exact practice README alr… — `GAP_B2_readme_results_block_unverified` — POLICIES.md, Cross-cutting invariants: "Verify-first + hone… — docs/PITCH.md:139,223,252 (84%) and docs/OVERVIEW.md:151 (~85%), against pyproject.toml:3…
- [MAJOR · CONFIRMED] The recorded cause of the earlier 11-vs-12 ARC discrepancy (time_budget / machine speed) is unsupported by a 2.5x budget sweep at HEAD — `GAP_B3_gf_number_contradiction_unmatched_protocol` — Cross-cutting invariants: "Runtime evidence outranks static… — docs/coverage/Gf.md (the 'My held-out number disagrees with the tracked one' paragraph, a…
- [BLOCKER · CONTESTED] spikes.amygdala is a structural constant with no PROVENANCE tag, and on the live cockpit it displays the arm E-STOP latch — `GAP_A1_subcortical_affective_regions_unread` — P3: "a signal that cannot vary must say so beside its own n… — src/snn_data_lake/cortex/live_brain.py::LiveBrain.spike_snapshot (with src/snn_data_lake/…
- [BLOCKER · CONTESTED] TrainOfThought never runs for a question — on the answering path it is only the 3-D model builder's part decomposer — `GAP_A4_deliberation_family_unread` — P3 dormancy shape #1: "method never called — the module is … — src/snn_data_lake/reasoning/train_of_thought.py::TrainOfThought.think
- [BLOCKER · CONTESTED] .gitignore:64 hides three tracked source modules from the entire P2 duplicate-prevention chain, which then fails permissively — `GAP_A5_lakehouse_absent_from_generated_inventory` — P2 — "When two mechanisms do the same job, exactly **one** … — .gitignore (line 64, bare `lakehouse/`) — root cause. Affected: graphify-out/graph.json, …
- [BLOCKER · CONTESTED] tools/charge_probe.py reports a null that belongs to its instrument, and its top rubric band is structurally unreachable for every source — `GAP_A6_brainstem_non_envelope_and_autonomous_loop` — Cross-cutting invariants: "before believing a signal is liv… — tools/charge_probe.py::run and ::main
- [BLOCKER · CONTESTED] The 'real captured sensor data' headline is unreachable from every documented path and degrades silently to a synthetic dataset while printing self-c… — `GAP_B2_readme_results_block_unverified` — POLICIES.md, Cross-cutting invariants: "never fake a capabi… — src/snn_data_lake/cortex/gesture_recognition.py::_load (DIR = "robot arm", cwd-relative);…
- [BLOCKER · CONTESTED] The project's headline fluid-intelligence number is 1.73x stale at HEAD, and four files (one in src/) quote it as the current capability — `GAP_B3_gf_number_contradiction_unmatched_protocol` — Cross-cutting invariants: "Verify-first + honest framing: .… — docs/DEPTH_CEILINGS.md (Ceiling 1 score table row 'evaluation (held out, never inspected)…
- [MAJOR · CONTESTED] The amygdala never learns on the live tick; its threat readout is a proven constant 0.0 — `GAP_A1_subcortical_affective_regions_unread` — P3 amendment: "A learning-verb method on a reachable module… — src/snn_data_lake/amygdala/fear_conditioning.py::Amygdala.threat (consumer src/snn_data_l…
- [MAJOR · CONTESTED] olfactory and pons package docstrings describe mechanisms that do not exist at HEAD — `GAP_A1_subcortical_affective_regions_unread` — P3's honesty guard ("never faked with a dead consumer") app… — src/snn_data_lake/olfactory/__init__.py and src/snn_data_lake/pons/__init__.py
- [MAJOR · CONTESTED] Half the declared growth surface is exactly this unread anatomy, and none of it grows — `GAP_A1_subcortical_affective_regions_unread` — P0 ("The brain must autonomously improve itself, by every a… — src/snn_data_lake/cortex/growth_program.py::GROWABLE (measured by src/snn_data_lake/subst…
- [MAJOR · CONTESTED] The plan of record specifies a from-scratch pager without adjudicating the shard-streaming out-of-core mechanism the same package already ships — `GAP_A3_neuro_chip_stream_is_an_unadjudicated_pager` — P2: "any unique capability of the retired mechanism ported … — specs/022-unified-brain-arc/tasks.md::T066 (and T064); src/snn_data_lake/substrate/neuro_…
- [MAJOR · CONTESTED] The bit-for-bit streaming-invariance red does not fire on a total loss of membrane-state persistence -- the fixture's own drive is supra-threshold — `GAP_A3_neuro_chip_stream_is_an_unadjudicated_pager` — P8 "a REACHABLE RED"; and specs/022 T066's "paged-out-spike… — src/snn_data_lake/substrate/neuro_chip_stream.py::main (invariance check) and ::Streaming…
- [MAJOR · CONTESTED] max_resident_cores is a self-reported bookkeeping integer, not a memory measurement, and its scope excludes the O(fan-out) in-RAM AER queue — `GAP_A3_neuro_chip_stream_is_an_unadjudicated_pager` — Cross-cutting invariant: runtime evidence outranks static r… — src/snn_data_lake/substrate/neuro_chip_stream.py::StreamingChip._step_core (self._residen…
- [MAJOR · CONTESTED] autonomous_enrich has zero constructors in src/ and no scheduler entry — the between-sessions knowledge-growth avenue is closed — `GAP_A4_deliberation_family_unread` — P3: a faculty is not done until it is "wired + charged", "r… — src/snn_data_lake/reasoning/autonomous_enrich.py::AutonomousEnricher
- [MAJOR · CONTESTED] concept_reflection's import chain terminates in a dead root, and two 0-importer autonomous_brain siblings sit beside it unadjudicated — `GAP_A4_deliberation_family_unread` — P2: adopt-or-merge, then retire the inferior — "exactly one… — src/snn_data_lake/reasoning/concept_reflection.py::ConceptReflector
- [MAJOR · CONTESTED] The generated existence ground truth asserts a completeness it does not have (420 of 423), and agents are instructed to trust it in place of the tree — `GAP_A5_lakehouse_absent_from_generated_inventory` — Cross-cutting invariant — "runtime evidence outranks static… — graphify-out/SRC_INVENTORY.md (header) — generated by ~/.claude/hooks/graph-exists.js::--…
- [MAJOR · CONTESTED] Second, unreported instance of the same class: .gitignore:176 `GATES.md` hides the specs/022 contracts file from graph.json — `GAP_A5_lakehouse_absent_from_generated_inventory` — P2 (same duplicate-prevention chain); the hidden artifact i… — .gitignore (line 176, `GATES.md`) → hides specs/022-unified-brain-arc/contracts/gates.md
- [MAJOR · CONTESTED] AgencyMonitor is write-only, and the controllability signal it writes is structurally zero because the body is frozen — `GAP_A6_brainstem_non_envelope_and_autonomous_loop` — P3, dormancy shape 4: "write-only consumer -- a faculty fed… — src/snn_data_lake/brainstem/autonomous_loop.py::AutonomousArm.step (representation-only d…
- [MAJOR · CONTESTED] The results block's own reproduction contract names no producer for any row — `GAP_B2_readme_results_block_unverified` — POLICIES.md, Cross-cutting invariants: "Verify-first ... pr… — README.md:41 and README.md:78 (framing) against the three tables at README.md:90-113
- [BLOCKER · REFUTED] spikes.hypothalamus and spikes.medulla are one byte-identical signal, untagged, while the analogous duplicate IS tagged — `GAP_A1_subcortical_affective_regions_unread` — P3: "a signal that cannot vary must say so beside its own n… — src/snn_data_lake/cortex/live_brain.py::LiveBrain.spike_snapshot
- [BLOCKER · REFUTED] GrowthStop._lp_excursion cannot see a sustained |LP| excursion longer than PATIENCE; the live growth stop rests on it — `GAP_A2_neuromodulation_curiosity_chain_unread` — P3 dormancy family shape #5: "constant wearing a signal's c… — src/snn_data_lake/cortex/growth_stop.py::GrowthStop._lp_excursion
- [BLOCKER · REFUTED] The entire 'thinking' tier sits outside every dormancy and charge gate — reasoning/ is not a declared region — `GAP_A4_deliberation_family_unread` — P3: "Every part of the brain must never remain dormant; all… — src/snn_data_lake/__init__.py::REGIONS
- [BLOCKER · REFUTED] The efferent half of the brain is inert in every unattended process; the live daemon has refused its own re-arm 4035 consecutive times — `GAP_A6_brainstem_non_envelope_and_autonomous_loop` — P3: "a wire that does not charge is reverted and retired/ex… — src/snn_data_lake/brainstem/autonomous_loop.py::AutonomousArm.step (section 1b self-clear…
- [BLOCKER · REFUTED] Growth is six orders of magnitude below its own measured memory ceiling, and no memory constraint explains the gap — `GAP_B1_dense_ceiling_never_measured_at_head` — P0 — "The brain must autonomously improve itself, by every … — src/snn_data_lake/cortex/neurogenesis_engine.py::NeurogenesisSelfImprove.growth_state_byt…
- [MAJOR · REFUTED] Five region rows are proxies while that region's own live, varying module output is computed and discarded — `GAP_A1_subcortical_affective_regions_unread` — P3 dormancy shape #2 ("return value discarded — a live call… — src/snn_data_lake/cortex/live_brain.py::LiveBrain.spike_snapshot (with cortex/crossmodal_…
- [MAJOR · REFUTED] curiosity_mag and neuromod_acetylcholine render a CONVERGED signal as a flat 0.0 with no PROVENANCE row, and it misdirected four audit lenses — `GAP_A2_neuromodulation_curiosity_chain_unread` — P3 shape #5's remedy, verbatim: "a signal that cannot vary … — src/snn_data_lake/substrate/brain_telemetry.py::PROVENANCE (and cortex/brain.py::Brain.st…
- [MAJOR · REFUTED] The growth deadlock: growth is gated on |LP| motion, and |LP| motion requires the new learnable structure only growth supplies — `GAP_A2_neuromodulation_curiosity_chain_unread` — P0 title clause: "The brain must autonomously improve itsel… — src/snn_data_lake/cortex/brain.py::Brain._safe_ctx (the stimulus source) and cortex/growt…
- [MAJOR · REFUTED] The out-of-core scale claim overstates what the mechanism actually bounds: resident bytes are the largest single shard, not the disk — `GAP_A3_neuro_chip_stream_is_an_unadjudicated_pager` — Owner directive (never fake achievement) and the standing s… — src/snn_data_lake/substrate/neuro_chip_stream.py (module docstring)
- [MAJOR · REFUTED] Live deliberation readouts are constant after 347 nights: schemas 0.0, tom_live null, recursive_tom_false_beliefs [] — `GAP_A4_deliberation_family_unread` — P3 dormancy shape #5: "constant wearing a signal's clothes … — src/snn_data_lake/cortex/brain.py::Brain._enter_sleep
- [MAJOR · REFUTED] Eight ArmSim docstring sites in autonomous_loop.py (15 present-tense in src/), and one asserts a behaviour that is false at HEAD — `GAP_A6_brainstem_non_envelope_and_autonomous_loop` — Cross-cutting invariants: "never fake a capability, a metri… — src/snn_data_lake/brainstem/autonomous_loop.py -- module docstring, ::AutonomousArm.__ini…
- [MAJOR · REFUTED] The efference-copy boundary reads exactly zero error, and the postural compass is empty, so _recovery_command degenerates to the fixed token the 2026… — `GAP_A6_brainstem_non_envelope_and_autonomous_loop` — P3, dormancy shape 5: "constant wearing a signal's clothes … — src/snn_data_lake/substrate/efferent_boundary.py::EfferentBoundary.observe (fed from src/…
- [MAJOR · REFUTED] The audit's operative scale bound is a VRAM figure read as a host-RAM figure — `GAP_B1_dense_ceiling_never_measured_at_head` — Cross-cutting invariant: "Verify-first + honest framing: pr… — specs/009-sparse-connectivity/plan.md ("caps the population near 40,000 in 12.8 GB of VRA…
- [MAJOR · REFUTED] specs/009's bounded-fanout remedy is reachable on the live path and never charged — `GAP_B1_dense_ceiling_never_measured_at_head` — P3 — "Every part of the brain must never remain dormant; al… — src/snn_data_lake/hippocampus/episode_consolidation.py::EpisodeConsolidator (fanout param…
- [MAJOR · REFUTED] The brain's memory is ~99.996% fixed overhead, and no ceiling arithmetic in the project accounts for it — `GAP_B1_dense_ceiling_never_measured_at_head` — Cross-cutting invariant: "Verify-first + honest framing ...… — src/snn_data_lake/cortex/brain.py::make_brain (1,160.76 MB at zero grown units) and the l…
- [MAJOR · REFUTED] docs/INVENTION.md's pre-registered primary falsifier states its red against a superseded count, and contradicts itself two sections apart — `GAP_B3_gf_number_contradiction_unmatched_protocol` — Cross-cutting invariants: "Adding a policy: implement the g… — docs/INVENTION.md (section 3.6 'Falsifiers and control arms' vs section 3.5), docs/POLICI…

