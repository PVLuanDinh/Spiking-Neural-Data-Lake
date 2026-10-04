# G6 / T114 -- sleep and durable memory (pre-registration, wave 1; frozen by the lead's P0 commit; section 2.16 governs)

Plan row: PLAN-FINAL G workstream, G6 / T114, parts (a)-(e) with every dated note (2026-09-25 the engine, the docs panel's
fold, Astra owner-saved 2026-09-27), and Part F's G-R6. Owner text binding this row: POLICIES P4d's 2026-09-27 quotation
(*"The brain should still be able to sharpen its understanding while sleeping, using synaptic plasticity, just like how the
fly brain is still able to learn new skills while being a frozen snapshot"*; *"The sleep cycle should be similar to actual
human sleep and lasts longer to rewire the entire brain to improve capability."*) and its operational reading from
`evidence/astra-architecture-2026-09-27.md` (every plastic circuit eligible; change selective by strength, activity and
tags; judged by held-out retention and cost). PLAN-FINAL orders G6 "after R2/R3" (T103/T104): every DECIDING test below
waits for them (section 2.10); this pre-registration and the incumbent-pruner instruments proceed now. The orchestrator
freezes this text after review; until then every item is a draft and every unmeasured number is marked **(projection)**.

## 1. Territory at the base `e1b3ae3c` (read and measured 2026-10-03, worktree `claude/g6`)

1. **The live sleep path has five prune-shaped operations** (census by grep over `src/`, every hit's call site read; the
   instrument in 2.3 mechanizes it). Two hosts: `Brain._sleep_tick` (every asleep tick) and `Brain._enter_sleep` (once
   per night, at sleep onset).

   | id | operation (symbol) | host, cadence | neuter handle today | outcome it can move | resource it bounds |
   |---|---|---|---|---|---|
   | P-SR | `sleep/structural_rewiring.py::StructuralRewiring.rewire` (DEEP-R prune of sign-identity violators + regen) | `_sleep_tick`, every tick, `_sleep_caps["structural"]` | `Brain.neuter_sleep("structural")` (T058) | the interaction learner's rows (`brain._ixn`) | live connection count (conserved budget) |
   | P-DP | `sleep/sleep_replay.py::AttractorNet.downscale_prune` (zero 0 < w < 1e-3, renormalize) | `_sleep_tick`, every tick | none (instance-level neuter in the instrument, disclosed) | `_consolidation_acc` (attractor recall, 3 patterns x 12 trials) | nonzero off-diagonal synapses (552) |
   | P-EY | `growth/ng_snn.py::NGSNN.prune_unearned` (earn-your-place, single pass, held-out RMSE) | `NeurogenesisSelfImprove._refit`, every `refit_every = 20` sleep ticks | none (class-level neuter inside the scratch process; the model is built inside `_refit`) | the grower's held-out accuracy (`predict_accuracy`) | grown units |
   | P-LL | `sleep/lifelong_consolidation.py::LifelongConsolidator._prune_to_budget` (lowest utility to `budget = 6`) | `_enter_sleep` via `learn_task`, once per learned task | `prune_on = False`; built-in control `prune_mode = "random"` | `_ll_retention` (task-0 test, 40 items) | prototype units |
   | P-LAKE | `sleep/lake_consolidation.py::LakeConsolidation.consolidate` + `KnowledgeLog.consolidate` (RECORD pruner over `_T_FACT`) | `_enter_sleep`, every 8th night, persist only | none | -- (storage) | lake records |

   P-LAKE prunes durable RECORDS, not active-network units. `evidence/l6-immortal-vs-consolidation.md` section 3.1
   counted 26,037 learned `_T_FACT`, 0 re-derivable, neither cap ever bound: "LakeConsolidation has therefore never
   dropped a single record." It is listed for completeness and is **out of (a)'s neuter-A/B scope** (its job is P4b
   storage under the L5/L6 retention labels, and PLAN's own enumeration of "sleep already prunes" does not name it);
   the scope call is put to the owner (section 2.14 Q1). Not prune operations, kept out of the census with reasons:
   the Benna-Fusi `relax` and the interaction replay (consolidation, no removal); the episode attractor's ring eviction
   (store-time capacity eviction, not a sleep step); `sleep/developmental_pruning.py` (a BENCHMARK, `dormancy_gate`
   permanent exemption, no live importer); `reasoning/developmental_loop.py::prune_dead` (research tier, instrument
   only); `substrate/home.py::_prune_state_history` (state-record history, not a sleep step).
2. **Prior measurements on these pruners.** P-DP: 2026-08-05 neuter-A/B on `_consolidation_acc`, 10 seeds, 900 ticks,
   A-B t(9) = -0.572, A-C t(9) = -1.032; the comment at the `downscale_prune` call site bounds that null at "no effect
   larger than ~0.10 accuracy" and names the unattempted power fix (average over all nights, not the last). Its
   harness-alive control: neutering `AttractorNet.sleep` pins the outcome at exactly 0.333333 on 7 of 7 seeds. P-SR:
   refused twice (2026-08-26; T054-ii 2026-08-31), wired at T058 default-on; its live substrate is all-excitatory and
   the Benna-Fusi rebind restores signs, so live turnover is ~0 by design; PLAN: "no structural verdict unless measured
   turnover exceeds zero." P-EY: `prune_unearned`'s docstring names its own KNOWN CEILING (single pass; "the upgrade
   path is to loop this until it removes nothing") -- that loop is (b)'s greedy arm. P-LL: synthetic separable tasks
   (T = 3, 2 classes each, BUDGET = 6, test 20 per class).
3. **No mask search and no annealer exist.** `grep -rn anneal src` fires on known positives (`reasoning/rsnn_delays_torch.py`
   `anneal_sigma`, `faculties/developmental_envelope.py`, `random_boolean_network.py`'s "annealed approximation") and
   none is a search over masks; `grep "joint.mask|mask_search"` returns nothing. PLAN: "No simulated annealing exists in
   the tree." A1-A3 (T105-T107) cover lake access only and are not this row.
4. **One scheduler, no pressure input.** `Brain.tick` reads `CircadianClock.consolidation_gate()` (2 hits in
   `engine/brain.py`, the positive control) and nothing else decides sleep; `grep -i pressure` over
   `sleep/circadian_clock.py` and `engine/brain.py` returns 0. Sleep is clock-driven, not solve-driven (engine
   inventory 2026-10-03).
5. **No engine record types exist yet.** `substrate/home.py` tags: `_T_FACT 0x01 .. _T_GOAL 0x0b`; no unit-journal,
   problem or sleep-verdict record exists on main. The solved predicate, budget pause and problem ids are spec-only
   (G2/G3); feed, readout neurons and the return path are spec-only (G3/G5); the reservoir is instrument-only (G1).
6. **Cost, measured on the real compute** (one throwaway brain, `NORD_HOME` in TEMP, exposed dev seed 12001, 200 ticks,
   default persistence, `.venv-lake` 3.14t; scratch script `g6_cost_probe.py`, out of tree): `make_brain` 2.73 s; 76.8
   ms/tick (98 awake at 70.0 ms, 102 asleep at 11.3 ms); `_enter_sleep` 915.5 ms per night (8 nights) dominates; the
   pruners are cheap (P-DP 0.06 ms, P-SR 0.09 ms, P-EY 1.31 ms per call, P-LL 0.01 ms). Readings at tick 200, each
   load-bearing for the design below: P-SR turnover 0.0, pruned 0 over 102 cycles; P-DP pruned 0 on its last call;
   `_consolidation_acc` = 0.333333 (the harness-alive control's floor value -- at 200 ticks the outcome is not yet off
   its floor); P-EY 5 calls, last `_pruned` 0, 3 grown units; P-LL 6 units, `_ll_retention` 1.0 (ceiling).
7. **Disclosed method gap.** The brief asks for `graphify query/explain`; `graphify-out/` is absent in this worktree and
   present only in the main checkout, which PROTOCOL section 4 forbids entering. Substitute: a read of
   `graphify-out/SRC_INVENTORY.md` (generated at e1b3ae3c; its sleep section lists the 12 `sleep/` modules) plus the
   grep census in item 1, each negative proved on a known positive.

## 2. Pre-registration

### 2.1 Questions, and what a result licenses

- **(a)** Does each live sleep pruner earn its place by a neuter-A/B on its own outcome? A KEEP licenses only "this
  pruner, on this workload, beats its no-prune twin on benefit or on cost at non-inferior capability"; never a
  sleep-pruning benefit for the engine (P4d's "Not claimed").
- **(b)** Does a joint-mask search (simulated annealing as the named example, P4d: "Either Simulated Annealing or
  something different") find smaller feasible masks over a solved problem's new neurons and their feed, readout and
  return synapses than the incumbent, greedy and equal-budget random search, with savings that survive their costs and
  no retention break on untouched tests?
- **(c)** Does a learning-load sleep pressure, as a second INPUT to the one scheduler, give a budget-matched,
  pressure-specific retention gain over clock-only and over its own shuffled twin?
- **(d)** After a solve, does sleep save the surviving new neurons and the problem's data so that a fresh process
  re-solves the problem with no regrowth and no new teaching?
- **(e)** Does an extended, staged, resumable sleep bout over the eligible learned brain give a registered held-out
  capability gain over equal-compute replay and plasticity controls, with measured brain-wide participation, no
  fabricated evidence, and its gain and stage surviving a fresh-process restore?

### 2.2 Statistic, bar and inference (Part F, unchanged)

S = 8 paired seeds per contrast; paired one-sided t > 1.895 (7 d.f., alpha = 0.05); ties counted (a zero difference is
a data point, never dropped); the exact sign-flip false-positive rate over all 2^8 = 256 masks printed beside every
verdict (adopted: `instruments/paired_contrast.py::paired_t`, `permutation_fpr`); no statistic normalized by a quantity
a control destroys; no majority-of-seeds rule. Per-seed PRECONDITIONS (off-floor, nonzero activity) are checked on
every seed and a failure makes the verdict REFUSED, which is stricter than any majority rule, not a substitute for the
t. Learning is OFF during every evaluation. **Multiplicity, disclosed:** (a) reads four independent per-pruner
dispositions at alpha = 0.05 each with no family-wise correction, because each pruner's disposition is its own
retirement decision; the four FPRs are printed side by side.

### 2.3 Part (a): the incumbent-pruner census and per-pruner neuter-A/B

**Instrument** `instruments/sleep_pruner_census.py` (new). Two halves:
1. **Census (static + runtime).** An AST walk from `Brain._sleep_tick` and `Brain._enter_sleep` through attribute
   calls collects every callee whose name is in the prune vocabulary {`prune*`, `*_prune*`, `downscale*`, `rewire`,
   `evict*`, `consolidate` on a record list}; the runtime half wraps each found site and counts calls and removals on a
   scratch brain. Output: the section 1 item 1 table regenerated, never hand-maintained.
2. **Per-pruner A/B harness.** One throwaway brain per (seed, arm) at `NORD_HOME` in TEMP, default persistence, a
   fixed teaching script injected at every wake period (the four `tests/test_sleep_capabilities.py` TEACH pairs, so the
   interaction ring is fed), **900 ticks** per run (the 2026-08-05 length; ~36 nights at the measured 25 ticks/day).
   Neuters, each surgical and disclosed (no brain.py edit in P1): P-DP by replacing `b.attractor.downscale_prune` on
   the instance with a no-op returning zero telemetry; P-EY at CLASS level inside the scratch process
   (`NGSNN.prune_unearned` returning 0 and leaving `last_keep = None`), because the model is constructed inside
   `_refit` and no instance exists to patch -- the cost probe had to wrap it the same way; P-LL by `prune_on = False`.
   P-SR's neuter, `neuter_sleep("structural")`, is R3's arm (below). C5 is the check that each patch reaches only its
   intended site.

**Per-pruner readings, in order; the first that fails decides.**
- **R0 repeatability:** the live arm run twice on the same seed gives byte-identical outcome and activity series
  (else REFUSED: nondeterministic instrument).
- **R1 activity:** total removals over the run (summed over every call, never the last call) > 0 on every seed of the
  live arm. Else **INERT** (P3 shape: live, never acts): no benefit verdict is read; disposition per 2.8.
- **R2 sensitivity (the instrument's positive control, read before any benefit verdict):** a SABOTAGE twin removes the
  SAME count per call as the live arm, chosen adversarially (P-DP: the strongest synapses; P-EY: the units whose removal
  costs the most held-out RMSE; P-LL: `prune_mode = "random"`, the module's own control, plus highest-utility-first),
  and must lose to the live arm at the bar. If it does not, the outcome cannot see which items a pruner removes and the
  pruner's A/B is **REFUSED (blind outcome)**. For P-DP the harness-alive control also runs: `AttractorNet.sleep`
  neutered must pin the outcome at 0.333333 while the live arm's per-run outcome is off that floor on every seed. It is
  a CONTROL on G6's workload, never a verdict on `AttractorNet.sleep`; that verdict is R3's F3 arm (sleep learning
  OFF), which reports its null beside P-DP's census.
- **R3 benefit:** live outcome > neutered (no-prune) outcome, paired one-sided t > 1.895.
- **R4 efficiency = capability NON-INFERIORITY, the only tested half:** t of (live - neutered + delta) > 1.895, where
  delta is ONE outcome quantum, the smallest change the instrument can register (P1 case (a), a fixed point of the
  measurement's resolution): P-DP 1/36 per night, P-EY 1/|x_te| of each refit, P-LL 1/40. The resource saving
  (neutered minus live: synapses, units) is REPORTED beside it and NOT tested: once R1 passes, a no-prune twin holds at
  least the live arm's resource BY CONSTRUCTION, so a resource test could not go red (P8 clause 1; such a criterion is
  deleted, not kept as decoration). The bar stays discriminating: the 2026-08-05 P-DP point estimates (-0.025, -0.044)
  against delta = 0.0278 would fail it.

**Outcomes.** P-DP: `_consolidation_acc` averaged over every night after the first store (the named power fix). P-EY:
the refit's held-out `_acc` averaged over refits (rows `prune_unearned` never sees), units at end as the resource.
P-LL: mean accuracy over every learned task's test (task 0's retention reported beside it), units at end. P-SR: the
activity census only (turnover through R3's detector); its outcome and verdict are R3's.

**Disposition.** KEEP on R3 or R4. Otherwise RETIRE (P3 disposition 2: deleted, its capability ported first if it has
one) or an owner-visible justified P3 exemption with an expiry. **A pruner is never kept as a CAP on the brain** (owner
2026-09-30, verbatim: "The growing brain has unbounded growth"): R4 credits cost per capability, not a ceiling.

**One measurer per pruner (debate, section 2.15; main's ruling 2026-10-03).** P-SR's verdict is R3's: R3's structural
OFF arm (`_sleep_caps["structural"] = False`) IS P-SR's neuter, read only if mean turnover > 0 on all 8 confirmatory
seeds, else "NOT READ (turnover 0 on k/8)", after a constructed sign-flip positive control (the
`tests/test_sleep_capabilities.py` flip idiom) reads > 0. G6 runs no second P-SR A/B; it ADOPTS R3's verdict and, on
zero turnover, brings retire-versus-dated-exemption to the owner (Q4). Known P-SR restore gap (R2's reading):
`Brain._rewirer` (`_rng`, `_signs`, `_budget`, `_cycles`) is process-local and not in `interaction_learner.json`.
P-LL stays G6's: R3's port (HIST-I) carries only `LifelongConsolidator`'s interleave and oldest-task retention probe
onto the NG-SNN grower, and R3's C18 runs the whole policy with the prune ON as a comparator, never varying it (R3,
2026-10-03); so P-LL's prune A/B (ON / OFF / random / highest-utility-first on its outcome) is a different question
and runs in G6's census instrument. If R3's C18 later retires the module, P-LL's verdict stands as its record. G6 runs
P-DP, P-EY and P-LL in full.

**Carried finding (G7 and G3 concur, 2026-10-03).** The live P-EY prune drops a pruned unit's placement
(`neurogenesis_engine._refit`, "follow the survivors") and the growth blob keeps only survivors, so a pruned percept
unit's weights survive only in `_T_GROWTH_DELTA` (0x08) prefixes, which `lake_engine.RETENTION[0x08]` marks
COMPACTABLE (reclaimable before the newest FULL). The gap is LATENT, not realized: main ran
`python -m snn_data_lake.gates.immortal_retention --assert` (2026-10-03), which reports "findings: 0" -- no 0x08
reclaimer exists today and the COMPACTABLE licence is one "nobody has exercised" -- so nothing has been lost; the first
0x08 reclaimer would make the live incumbent break P4d's retention rule ("a prune removes a unit from the active
network, never its lake record"). G3's
IMMORTAL per-unit journal (2.6) closes it for PROBLEM-grown units only; the live percept grower's units stay
chain-only (G3 reports this gap to main). (a) records it beside P-EY's verdict; it is not repaired by a G6 arc.

### 2.4 Part (b): neuron-mask search by joint-mask validation

**Instrument** `instruments/joint_mask_search.py` (new): one search contract, four arms behind it.
- **Mask:** binary over (i) the solved problem's new units and their feed, readout and return synapses (the
  **solved-problem arm**, PLAN 2026-09-25), and separately (ii) older learned circuits, inter-batch links, readouts and
  inter-network projections (the **eligible-brain arm**, Astra 2026-09-27), under the same anatomical, safety,
  validation and retention limits. The `brainstem/` envelope and protected scaffolds are never in a mask.
- **Protocol (PLAN's five steps):** fit and replay on training episodes; score each candidate mask with the readout
  REFIT on training data only; select on validation constraints; judge the frozen winner ONCE on untouched future and
  rare-task tests; charge search and restart costs.
- **Splits (G2's contract, `instruments/taught_problems.py::Problem.episodes(seed, split)`):** `teach` is the ONLY split
  sleep may replay or a mask may be FIT on; `validation` selects masks and gives the post-prune solved re-check at save;
  `future` is read ONCE: the frozen winner's judgment and (d)'s fresh-process re-solve. Learning OFF is G2's definition
  (no `correct` / `learn_turn` / `interact`, no tick; plasticity frozen, state stepped on reads). Evaluation reads call
  the LiveBrain's `brain.brain.route(text)` (pure inference; `SpikingInteractionLearner.respond`, no plasticity, no
  RNG), NEVER `Brain.respond`: R2 confirmed `Brain.respond` calls `chat` -> `learn_turn` on every turn, which writes an
  r2.v1 `turn` event and deposits, while `route` writes no event on a persisting brain; reads are never
  replay-eligible (eligibility keys on kinds teach + turn), and any journaled validation/future presentation carries
  `split = heldout`, which R2's reader refuses for the ring and the redo (D19). Check C21 guards it. **Disclosed limit (G2 confirms, and widens it):** the incumbent's read
  is a pure function of (weights, current text), exactly; G2's held-out splits are never-taught PRESENTATIONS of the same
  finite items. So for ANY STATELESS reader with learning OFF, validation and future are the same reads in another order
  and cannot disagree, on (ii) AND on (i). Consequences, registered: (1) for a stateless reader, (b)'s "judge the frozen
  winner once on untouched future" cannot fire independently of validation; it is REPORTED as a repeat-read
  determinism check and is not counted as a red (P8 clause 1); it discriminates only under a STATEFUL reader (grown
  memory units, the reservoir). (2) (b)'s discriminating untouched test for every reader is retention on the OTHER
  problem and prior tasks (PLAN's rare-task test), plus (i) under a stateful reader. (3) (d)'s fresh-process re-solve on
  `future` stays a real red: a restart changes state between save and read, which is what it tests.
- **Feasibility is a constraint, never a penalty:** a proposal whose masked network fails the problem's solved
  predicate (G2's `Problem.solved(reads)`, read verbatim) on validation, or breaks the old-task retention bound (G3's
  `growth_loop.retention_ok(before, after)`, read verbatim AT ITS FREEZE -- today a projection in G3's draft, one item),
  is rejected before it can be accepted.
- **Where each arm's pre-sleep state comes from (debate):** the SOLVED-PROBLEM arm from G3's loop (a SOLVED
  `ProblemRecord` with its grown units, `role in {E, I}`, `batch_id`, and the feed synapses as a declared array after
  G1(c)); the ELIGIBLE-BRAIN arm from R3's `PreSleepState` twin-brain fork (same seed, same A-then-B wake teaching), so it
  is paired with R3's arms and charged in R3's compute unit. R3's terms: `PreSleepState` is rebuilt from seed in a
  fresh process and home per arm (never deep-copied), verified by a `PreSleepDigest` (learner weights, ring, learner RNG
  state, stack weights, `_rewirer is None`); the compute unit is PRESENTATIONS (one forward drive of one stored
  stimulus), with deposits, consolidate calls, structural edits and wall time reported beside and never claimed
  matched; a mask evaluation converts as one validated item = one presentation and one refit = rows x passes.
- **Mask domains (G7's precision: `origin` separates born from legacy, it does not identify scaffold; scaffold and
  neuromodulation regions are refused at admission):** the solved-problem arm masks only units with `origin = born` and
  that `problem_id` (via `placement.labels_by(records, key)`, keys {region, batch_id, problem_id, layer, origin}). The
  eligible-brain arm masks born units of any problem PLUS the declared LEGACY learned circuits that are in today's
  active network without labels (enumerated by module: `brain._ixn` rows, the NGSNN grower's units, `AttractorNet.W`),
  each reported as its own population `legacy:<module>` in the participation census. Never in any mask: the
  `brainstem/` envelope, any non-learned scaffold, the reflex/safety paths. The
  incumbent arm on the taught learner is G3's consumer-agnostic one-pass prune (`prune_unearned`'s discipline; signature
  at G3's P2), on the percept grower `prune_unearned` itself.
- **Objective:** cost C = active units + active synapses of the masked network (both counts reported; per-tick compute
  measured beside them).
- **Arms at EQUAL evaluation budget B** (one evaluation = one refit + validation score): no-prune; the incumbent
  (`prune_unearned`, one pass); greedy (`prune_unearned` iterated until it removes nothing, its docstring's named
  upgrade; its measured evaluation count on the instance DEFINES B, so the budget is data-derived); equal-budget
  random search (B random feasible masks, best on validation); simulated annealing (B proposals, single-element flips,
  initial temperature T0 = median uphill delta-C of the first sweep / ln 2 so the median uphill move starts at
  acceptance 1/2, cooled geometrically to T0 / B over the budget -- both constants derived from the measured deltas
  and B; the cooling SHAPE is a disclosed session choice).
- **Endpoint (primary):** C of the frozen winner, feasible on the untouched test. SA beats greedy AND beats random:
  paired one-sided t on (C_other - C_SA) > 1.895 each. Secondary: SA vs incumbent and vs no-prune.
- **Savings after costs:** net = (per-tick compute saved) x H - (search + restart compute), H = the measured median
  wake-period length in ticks of the same run (the masked network serves at least until the next sleep re-searches).
  Red if net is not > 0 at the bar. H is a **session call**, disclosed; the break-even horizon H* = cost / saving is
  printed beside it, at the measured scale and at the projected engine scale.
- **P2 on the result:** SA wins -> it becomes THE unit pruner for grown neurons and `prune_unearned`'s single pass is
  retired after its capability is ported; greedy wins -> the incumbent adopts its fixed-point loop and SA is deleted;
  neither beats the incumbent -> SA is deleted (P8 deletion on a null). P-EY's (a) verdict and (b)'s incumbent arm are
  one measurement family on one pruner: (a) reads it on the live sleep path today, (b) on the engine's grown units.

### 2.5 Part (c): learning-load sleep pressure

**Instrument** `instruments/sleep_pressure.py` (new).
- **Pressure:** p_t = sum over plastic circuits of |delta w| at tick t, divided by the number of plastic synapses
  (width-normalized), per tick (rate-normalized). Load L = sum of p_t since the last sleep onset. Request sleep when L
  exceeds the running baseline B_L, the mean load per completed wake period, persisted (P1: relative to a persisted
  running baseline, the `_lg_base` pattern; the instrument keeps it in memory, the live wire persists it as named state).
- **One scheduler:** pressure is a second INPUT to the circadian gate, never a second scheduler. In P1 the instrument
  drives `_wake_tick` / `_sleep_tick` / `_enter_sleep` / `_enter_wake` under a schedule policy, exactly as
  `tests/test_sleep_capabilities.py` drives `_sleep_tick` today: a HARNESS, not a scheduler. The live input to
  `CircadianClock` lands only after the deciding test (2.13 arc P4), with check C13 (harness == live gate on a fixed
  pressure trace).
- **Budget matching by RESCHEDULING (PLAN: "budget-matched rescheduling"):** a run is a fixed number of clock cycles
  (nights), not a tick count. A pressure or solve request during wake ADVANCES the clock's next scheduled onset to the
  request tick; the advanced night REPLACES that scheduled night (which is then skipped) and lasts the clock's own night
  length for it. A request never adds or removes a night; a request arriving when the current cycle's onset has already
  been advanced, or while asleep, is recorded and absorbed. So every arm has identical total sleep ticks AND identical
  onset count; only placement differs. A run whose budgets differ is refused (C12), and C12's planted red is an arm
  given one extra sleep tick -- never a pressure arm obeying this rule.
- **Arms:** clock-only (the incumbent gate); pressure; shuffled-pressure (the pressure arm's own onset count placed at
  uniformly permuted times, same seed); solve-trigger (sleep on "solved", the owner's rule: the gate subscribes to G3's
  `GrowthLoop(..., solved_sink=...)`, called once per OPEN -> SOLVED transition; G3's loop only emits, the REQUEST is
  G6's); a schedule ORACLE (sleep exactly after each planted learning burst) as the workload's positive control.
- **Replay and the baseline as carriers (agreed with R2):** the deciding run takes R2's hydrated replay by constructing
  `make_brain(seed, persist=True)` over a throwaway home whose learner events were written through `learn_turn`'s one
  sink (construction-time hydration; never `hydrate` on a live learner); until R2 lands, runs disclose the process-local
  ring (R2: the replay POLICY is unchanged, 64-event recency window, k = 8). The pressure baseline is G6's durable
  carrier: named state `sleep_pressure.json` {baseline_load_per_wake, n_wake_periods, load_since_last_onset} at
  `_enter_wake` and close, declared in R2's carrier table with its omission arm in R2-c6.
- **Workload:** a teaching curriculum with uneven load (planted bursts followed by conflicting material), through the
  interaction learner and R2's hydrated replay. If the oracle does not beat clock-only at the bar, the workload cannot
  show a timing benefit and (c) is **REFUSED**.
- **Endpoint:** held-out retention of taught items with learning OFF, at each burst's horizon and at the run's end.
  Primary: pressure > shuffled-pressure (pressure-SPECIFIC). Secondary: pressure > clock-only; solve-trigger vs both.

### 2.6 Part (d): save

- **What sleep writes after a solve** (PLAN (d), verbatim scope): the surviving new neurons (unit rows with their G7
  labels, feed synapses, readout and return weights, lineage to the problem id and to the sleep that kept or pruned
  each unit) and the problem's data (typed teaching episodes in `_T_EPISODE`, its solved predicate, its validation
  record), under a declared problem tag (= G3's `_T_GOAL` problem record, D11) and retention label (FR-003:
  reconstructible deltas). A problem paused
  UNSOLVED is saved OPEN with G3's budget ledger. A prune removes a unit from the active network, never its lake
  record (the retention rule).
- **No new tag by G6 (agreed with G3, G7, R2).** ONE problem record, G3's: event rows under GoalStore's `_T_GOAL`
  (0x0B, IMMORTAL). G3 writes kind `unit` WHEN GROWN, keyed by a stable `unit_id` minted at admission and never reused,
  carrying the unit's OWN weights (taught learner: input weights, threshold/kind, output-synapse row per label neuron;
  NGSNN unit: W row, b, beta column), placement + G7 labels, `problem_id`, `batch_id`, `role`, `origin` -- never a
  digest plus a ref into the COMPACTABLE 0x08 chain (a digest verifies but cannot rebuild). G6 writes kind
  `sleep_verdict` {`unit_id`, `problem_id`, `sleep_id`, `verdict` in {"kept", "pruned"}, `mask_digest`,
  feed/readout/return synapse refs, `retention_label`} -- G7's four gate fields verbatim -- and, through G3's
  `sleep_save` row kind {`sleep_id`, `mask_digest`, post-prune solved re-check on validation, kept/pruned unit ids},
  one per problem saved in that sleep, plus kind `sleep_bout` {`sleep_id`, `trigger` in {clock, pressure, solved},
  `problem_ids`, `night`, start/end tick, bout report digest}, `sleep_id = "<session>-n<night>"` so it joins R2's
  per-tick `_T_EPISODE` kind `sleep` progress events on (session, night); G7's `resolve_sleep(sleep_id)` is true iff a
  `sleep_bout` row carries that id. Ownership:
  G6 owns the CONTENT of `sleep_bout`, `sleep_verdict` and `sleep_save`; G3 owns the codec, the fold and the `problem` and
  `unit` kinds, and builds the fold's pass-through of unknown kinds in its P3 (P4b: foreign records preserved), so G6's
  kinds never depend on G3's codec knowing them. Fold = latest per unit; a prune never deletes a `unit` row.
- **The problem's teaching data** = R2's r2.v1 `_T_EPISODE` event ids by reference (its validation items
  `split = heldout`, so replay can never train on them); G6 writes no teaching and no new episode kind.
- **WRITE-AHEAD:** every unit in the mask's domain gets its `sleep_verdict` row BEFORE the mask is applied to the
  active network, so the pruned half of the saved-record clause is checkable at all (G7's finding).
- **One validator (P2):** G-R6's "a kept or pruned unit's saved record does not name the sleep that kept or pruned it"
  READS G7's `gates/unit_label_gate.py::check_saved(record, *, resolve_problem, resolve_sleep, first_sign)`; G6 builds no
  second validator.
- **Red (G-R6 (d)):** a FRESH PROCESS (new interpreter, same throwaway home) does not re-solve the saved problem with
  no regrowth and no new teaching (the solved predicate on the untouched test fails). Control: the uninterrupted twin
  (same process continues) gives byte-equal answers on the saved problem.

### 2.7 Part (e): extended staged sleep

- **What it is:** repeated NREM-inspired replay/consolidation and REM-inspired integration/refinement stages, through
  the one scheduler; names label engineering arms, never biological equivalence or a literal human-hour timer. Replay
  only prior eligible lake experience; internally generated variations are marked SYNTHETIC and can never become
  observations, labels or rewards; no sleep operation issues a motor command or advances wake chronology. Pause on
  registered budget, sustained validation plateau, interference limit or wake request.
- **Ownership split (agreed with R2 and R3; main's ruling 2026-10-03):** ONE sleep-arm campaign, R3-owned, run in R3's
  `instruments/sleep_plasticity_ab.py`: fixed-topology plasticity ON/OFF, structural ON/OFF, short/medium/extended
  budgets on the flat incumbent first; staged-vs-flat and stage-order shuffle registered by R3 and executed once G6's
  runner exists. G6 contributes the eligible-brain structural arm (2.4's search, plugged into R3's harness on its
  `PreSleepState`), the participation census and the fabrication guard, and READS G-R6's (e) reds from that campaign
  (P2 applied to experiments). R2 owns persistence and the interrupted-versus-uninterrupted check (a sleep-progress
  event per learner-affecting sleep tick + a deterministic redo of the night from the start-of-night state); G6
  DECLARES its carriers in R2's format {class, where, boundary, owner, omission_red}.
- **The runner is G6's build (main's ruling: the BUILD does not wait on R3's verdicts; only G6's deciding tests do).**
  `instruments/staged_sleep_bout.py::run_bout(brain, stages, *, rng_seed) -> BoutReport`, R3's interface: `stages` an
  ordered list of `(name in {"nrem", "rem", "flat"}, budget)`, `budget = {"presentations": int,
  "structural_candidates": int | None}`; `flat` = one stage with the summed presentations (the equal-compute control);
  order a parameter (the shuffle control); pause reasons {budget, plateau, interference, wake_request}; `BoutReport` =
  per-stage presentations, deposits, consolidate calls, structural edits, participating populations, pause reason, and
  the carriers handed to R2. `nrem` = replay of eligible lake items through the real learner; `rem` =
  integration/refinement over already-replayed eligible items, every recombination marked SYNTHETIC. Evaluation
  (learning OFF) stays R3's. Declared carriers, mapped onto R2's sleep-progress event (`_T_EPISODE` kind `sleep`,
  r2.v1, written at the END of each learner-affecting sleep tick; R2's redo rebuilds the start-of-night state from the
  end-of-previous-night checkpoint plus the day's learner events -- no extra checkpoint at `_enter_sleep` -- then
  re-executes the night's completed ticks and verifies each against `weights_digest`/`rng_digest`):
  `stage_index`, `stage_name`, `presentations_spent` and `pending` (G6's serialized structural candidates) durable per
  sleep tick (accepted by R2 as filed); the runner RNG reconstructible by the redo because every draw comes from ONE
  runner RNG seeded by `rng_seed` and no draw is conditional on wall time or on another RNG (a `rem` presentation draws
  more than once, so draw count != presentations; R2 adds slot `bout_rng_draws`, verified on redo); `pause_reason`
  durable at the bout's last tick (R2 adds the slot; empty on every other tick). Pause criteria never condition a draw
  inside a completed tick.
- **G6's own (e) instruments:** a **participation census** (which populations and synapses changed during a bout,
  counted by population label; red: no measured change outside the solved problem's units = "lacks measured
  eligible-brain participation"); a **fabrication guard** (a synthetic item carrying an observed/label/reward flag is
  refused); the **restore read** (gain and stage after a fresh-process restore, read from R2's interruption check).

### 2.8 Null dispositions (each a CAUSAL claim whose instrument owes its positive control FIRST)

| null | named cause | instrument that owes its positive control before the task | control |
|---|---|---|---|
| (a) pruner INERT | the pruner's trigger never fires on this workload | the activity counter | C2: a planted pruner removing k items reads k |
| (a) pruner REFUSED | the outcome cannot see which items are removed | the per-pruner outcome | R2 sabotage twin must lose; P-DP also its 0.333333 floor control |
| (a) pruner no R3/R4 | the pruner neither helps nor is harmless here | the A/B harness | C5 surgical neuter; R0 repeatability |
| (b) SA no better | no better feasible mask exists, or the search is broken | the search | C8: SA beats random on a planted fixture with a known smaller mask |
| (b) savings vanish | pruning saves less compute than the search costs | the cost meter | C10: a zero-saving mask reads net < 0 |
| (c) no pressure gain | timing does not matter on this workload | the workload | the schedule oracle must beat clock-only (else REFUSED) |
| (d) no re-solve | the save omits a carrier | the save/restore harness | C14: omitted feed synapses must make the re-solve fail |
| (e) no gain | extended sleep adds nothing over its controls | R3's campaign | R-R3's own controls (read, not re-run) |

### 2.9 What a null deletes, and what it does not

A null deletes the IMPLEMENTATION that failed and nothing the owner drew. Owner-drawn and never deleted by a G6 null
(P4d's null rule): the sleep cycle that prunes and saves after a solve, the save into the one lake, recall from it.
Deleted on its own null: SA (or any named search method; the owner's "Either Simulated Annealing or something
different"); the pressure input (the clock stays); an incumbent pruner that misses (a) (retired or exempted, the owner
told); an extended-bout length that does not beat the shorter budget. If every pruner retires and no search wins, the
owner-drawn "prune" step stands UNBUILT and reported, never faked with a pruner that measures nothing.

### 2.10 Dependencies and waits

- **R2 (T103)**: hydrated, restart-safe replay -- (c)'s workload (sleep's benefit flows through replay), (e)'s carriers,
  persistence and restore read, the r2.v1 event ids (d) references. **R3 (T104)**: the ONE sleep-arm campaign (e)
  reads; P-SR's verdict (adopted, 2.3); `PreSleepState` for (b)'s eligible-brain arm. DEPENDENCIES.md lists these two; main's ruling (2026-10-03) adds that
  G6 BUILDS the staged-bout runner without waiting on R3's verdicts and that R3's staged/shuffle arms wait on that build.
- **Found dependencies beyond DEPENDENCIES.md (reported to main):** (b)'s solved-problem arm needs G3's loop
  (`ProblemRecord`, grown units with `role`/`batch_id`, the feed array after G1(c)) and G5's readout and return
  synapses; (b)'s masks and (e)'s census need G7's `placement.labels_by` (keys incl. `origin`); feasibility and the
  solve-trigger need G2's `Problem.solved(reads)` and splits (`instruments/taught_problems.py`), G3's `growth_loop.retention_ok` at its freeze and
  `GrowthLoop(solved_sink=...)`; (d) needs G3's P3 (`_T_GOAL` `unit` rows with weights, the unknown-kind pass-through,
  the `sleep_save` codec) and G7's `check_saved`.
- **Why (a)'s confirmatory read also waits (PLAN: G6 "after R2/R3"):** R2 changes how a persisting brain is built
  (construction-time hydration of the replay ring), so a P-DP/P-EY verdict read on today's brain would describe a
  workload R2 is about to change; and P-SR/P-LL's verdicts are R3's.
- **Proceeds now:** this text; (a)'s census and A/B harness on dev seeds; (b)'s search on a planted fixture; (c)'s
  harness and oracle workload on the incumbent learner; the staged-bout runner (after P1-P3); (d)'s field agreements.
  **Waits:** every deciding run of (a), (b), (c), (d), (e).
- **Home safety (PROTOCOL 4b):** every probe and run that builds a Brain uses a throwaway `NORD_HOME`; never the live
  home, never the daemon. The section 1 cost probe set `NORD_HOME` to a fresh TEMP dir before import; the live home's
  newest write is still 2026-09-27 01:19 (checked after the probe).

### 2.11 Checks, each with a constructed, REACHABLE red (a red that does not fire halts its arc)

C1 census finder: a planted subclass adding a prune call to `_sleep_tick` must appear; deleting `downscale_prune` from
the walk must red. C2 activity counter: a planted pruner removing exactly k items reads k. C3 sensitivity: R2's
sabotage twin (per pruner). C4 harness-alive: `AttractorNet.sleep` neutered pins 0.333333. C5 surgical neuter:
neutering pruner p leaves every other pruner's activity series byte-identical up to the first tick p would have
removed something. C6 repeatability (R0). C7 feasibility: with the feasibility check disabled the search accepts an
unsolving mask and the frozen-winner judge must red. C8 SA beats random on a planted fixture (k needed units, m
redundant duplicates, a known minimal mask). C9 a mask removing an old task's unit breaks the retention bound and must
red. C10 a zero-saving mask reads net < 0. C11 the schedule oracle beats clock-only (workload sensitivity). C12 a
planted arm with one extra sleep tick is refused as budget-unmatched. C13 harness schedule == live gate sequence on a
fixed pressure trace (built at P4). C14 save omitting the kept units' feed synapses: the fresh process must fail to
re-solve. C15 a saved unit record without its `sleep_id` is refused (G7's gate; G6 supplies the planted record). C16 a
synthetic replay item flagged observed is refused. C17 a bout touching only the solved problem's units reds the
participation census. C18 write-ahead: a crash injected between the `sleep_verdict` write and the mask apply loses no
verdict, and a crash before the write leaves the active network unchanged. C19 runner equal compute: a `flat` stage and
the staged list it sums report the same total presentations, and a planted stage list with one extra presentation is
refused as compute-unmatched. C20 carrier omission: each carrier G6 declares to R2 has an omission arm in R2-c6 that
must red (R2 runs it; G6 supplies the declaration). C21 read purity: every evaluation asserts zero new learner events
and byte-identical learner weights across the read; a planted evaluator that reads through `Brain.respond` must red.

### 2.12 Seeds (G6's block 12000-12999; 12345 and 12000 EXCLUDED)

`12345` is already a seed on main (`random.Random(12345)` in six modules, the c0 battery's refuter), and `12000` is a
size constant (`NORD_TRAIN`); both are excluded and reported to main. Grep positive control: `4242` fires in 16 files;
`7100-7103` do NOT fire on main (they live on the G1(c)/G2 branches), disclosed, not counted as a passing control. No
other in-block token is a seed (bytes, word counts, timings). Sets (dev = exposed; confirmatory read once; reserve for
a repaired failure; reds for constructed reds):

| part | dev | confirmatory (S = 8) | reserve | reds |
|---|---|---|---|---|
| (a) | 12001-12008 | 12011-12018 | 12021-12028 | 12031-12038 |
| (b) | 12101-12108 | 12111-12118 | 12121-12128 | 12131-12138 |
| (c) | 12201-12208 | 12211-12218 | 12221-12228 | 12231-12238 |
| (d) | 12301-12308 | 12311-12318 | 12321-12328 | 12331-12338 |
| (e) | 12401-12408 | 12411-12418 | 12421-12428 | 12431-12438 |

Paired-seed rule: within a part every arm runs every seed of the set; a contrast reads the per-seed difference. Seed
12001 was used once already (the cost probe, section 1 item 6) and stays dev.

### 2.13 Cost at real scale, and the arc split

**Cost.** Measured: 76.8 ms/tick, 2.73 s per brain build (section 1 item 6). **(projection)** 900-tick run ~ 69 s +
build ~ 72 s. (a) per confirmatory seed: live, P-DP neuter, P-DP sabotage, P-DP floor control, P-EY neuter, P-EY
sabotage, plus P-LL's neuter and two sabotage twins (P-LL stays G6's, D5): 9 runs x 8 seeds = 72 runs ~ 86 min
serial, ~ 22 min at 4 workers; R0 repeatability on two dev seeds. (b) on the fixture is seconds per
instance at today's 3-unit grower. (c) and (e) scale with R2/R3's replay cost, unmeasured. The engine-scale projection
(owner 2026-09-30: judge at the scale the engine grows to) is owed when G3's grown populations exist.

**Arc split** (each its own arc under PLAN A.4; files claimed on the board):
- **P0 docs:** this file. `specs/022-unified-brain-arc/evidence/g6-sleep-durable-memory.md`.
- **P1 (a) instruments:** `src/snn_data_lake/instruments/sleep_pruner_census.py`, `tests/test_sleep_pruner_census.py`
  (C1-C6 on fixtures and one short scratch brain). No edit to `engine/brain.py`, `substrate/home.py` or `sleep/`.
- **P2 (b) search on fixtures:** `src/snn_data_lake/instruments/joint_mask_search.py`, `tests/test_joint_mask_search.py`
  (C7-C10).
- **P3 (c) harness:** `src/snn_data_lake/instruments/sleep_pressure.py`, `tests/test_sleep_pressure.py` (C11, C12).
- **P3b the staged-bout runner (R3's interface; no wait on R3's verdicts):**
  `src/snn_data_lake/instruments/staged_sleep_bout.py`, `tests/test_staged_sleep_bout.py` (C16, C17, C19; C20's
  declaration). Instrument tier until a deciding test adopts it into `sleep/`.
- **P4 wires, after R2/R3 and the deciding reads:** the pressure and solve-request inputs to the one gate in
  `sleep/circadian_clock.py` + `Brain.tick` (HOT: brain.py edit order R2 -> R1 -> R3 -> G6, OWNERS.md; G6 claims
  only after R3 releases; C13); the `sleep_pressure.json` carrier; a `neuter_sleep` cap for P-DP if (a) keeps it; the
  `sleep`/`sleep_verdict`/`sleep_save` rows through G3's codec (C14, C15, C18).
- **Deciding runs:** (a) confirmatory, (b) on G3's populations, (c), (d), (e) -- each after its waits, each one read.

### 2.14 Open questions for the owner

1. Is the RECORD pruner `LakeConsolidation` (never dropped a record, L6 3.1) in G6(a)'s scope, or does it stay the L5/L6
   retention row's?
2. Savings-after-costs horizon H: one measured wake period (this draft's session call) or the network's expected
   lifetime? H decides whether any pruning at today's 3-unit scale can pay for its search.
3. A pruner kept only on R4 (efficiency at non-inferior capability): acceptable under "unbounded growth", read as cost
   per capability rather than a cap?
4. P-SR, if INERT (likely; turnover 0 over 102 cycles measured here, 0 on 10/10 sleep ticks in R3's probe): retire
   now, or a dated P3 exemption until a substrate whose writer crosses zero exists (its own docstring's criterion)?
   R3 cites this question rather than asking a second time.
5. The live percept grower's pruned units survive only in COMPACTABLE `_T_GROWTH_DELTA` prefixes (2.3's carried
   finding; G3 and G7 concur). LATENT today (`immortal_retention --assert`: findings 0, no 0x08 reclaimer); P4d's
   retention rule is met for problem-grown units by G3's IMMORTAL journal, not for the live percept grower once a
   reclaimer exists. Queued by main as owner Q5 with three options (board `OWNER_QUESTIONS.md`).
6. The eligible-brain arm masks LEGACY learned circuits (`brain._ixn` rows, the percept grower, `AttractorNet.W`)
   beside born units: inside "every plastic circuit eligible", or should the arm stay on labeled (born) units only?

### 2.15 Debate log (wave 1, 2026-10-03)

| # | from -> to | challenge | answer | resolution |
|---|---|---|---|---|
| D1 | G6 -> R3 | G6(e) and R3's 2026-09-27 note list the same arm set | R3 accepts ONE campaign, R3-owned (`instruments/sleep_plasticity_ab.py`); G6 contributes eligible-brain arm, participation census, fabrication guard; reads G-R6 (e) from it | RESOLVED; main ruled the same; 2.7 |
| D2 | R3 -> G6 | the mask arm must take its pre-sleep state from R3's twin-brain fork (`PreSleepState`), not its own teaching | ACCEPT for the eligible-brain arm; the solved-problem arm needs a solved G3 problem, which A-then-B does not produce. R3: PS rebuilt from seed per arm in a fresh process + home, `PreSleepDigest`; compute unit = presentations; mask evaluations convert | RESOLVED; 2.4 |
| D3 | G6 -> R3 | StructuralRewiring is both G6(a)'s pruner and R3's structural incumbent | R3 owns the verdict (turnover > 0 on all 8 seeds else NOT READ k/8; sign-flip positive control first); G6 adopts it and brings retire-vs-exemption to the owner | RESOLVED; 2.3, Q4 |
| D4 | G6 -> R3 | who builds the staged-bout runner | R3 is not building it; G6 builds it with R3's `run_bout` interface; build does not wait on R3's verdicts | RESOLVED; main ruled the same; 2.7, P3b |
| D5 | G6 -> R3 | `LifelongConsolidator` (P-LL) is both G6(a)'s pruner and, per main, an R3-ported arm | R3: its port carries only the interleave + retention probe; its C18 runs the whole policy prune-ON as a comparator, never varying the prune; G6 keeps P-LL's prune A/B | RESOLVED; 2.3 |
| D5b | R3 -> G6 | `nrem`/`rem` definitions | R3 ACCEPTS, with its C10 (synthetic rem items never become lake events, ring entries, labels or held-out) | RESOLVED; 2.7, C16 |
| D5c | G6 -> R3 | G6's P-DP harness-alive control neuters `AttractorNet.sleep`, which is also R3's F3 arm | the control stays on G6's workload as a CONTROL only; F3 owns the verdict on `AttractorNet.sleep` | CLOSED from G6's side; R3 may refute (no reply at draft time); 2.3 |
| D6 | G6 -> R2 | how G6's harness gets hydrated replay | no mid-life hydrate; `make_brain(seed, persist=True)` over a home with learner events; never `hydrate` a live learner | RESOLVED; 2.5 |
| D7 | G6 -> R2 | who persists the bout's carriers | R2 owns persistence + interrupted-vs-uninterrupted (start-of-night checkpoint at `_enter_sleep` + verified redo); G6's declarations accepted as filed, mapped onto R2's `_T_EPISODE` kind `sleep` event | RESOLVED; 2.7 |
| D7b | R2 -> G6 | the redo reaches the runner RNG only if draw count == presentations_spent | REFUTED in part: `rem` draws more than once per presentation; the redo still reproduces it (one seeded runner RNG, no wall-time/foreign-RNG conditioning); R2 accepted and adds slots `bout_rng_draws` and `pause_reason` | RESOLVED; 2.7 |
| D7d | R2 -> G6 | correction: no extra checkpoint at `_enter_sleep`; start-of-night state = previous night's checkpoint + the day's learner events; `session` is per process, `'<session>-n<night>'` is a stable join key | ACCEPT | RESOLVED; 2.6, 2.7 |
| D7c | G6 (self) | two record kinds named `sleep` (R2's `_T_EPISODE` progress event; G6's bout row on `_T_GOAL`) | G6's renamed `sleep_bout`, `sleep_id = "<session>-n<night>"`; G3 and G7 told | RESOLVED; 2.6 |
| D8 | G6 -> R2 | the pressure baseline | G6's own durable carrier, named state `sleep_pressure.json` at `_enter_wake` and close | RESOLVED; 2.5 |
| D9 | R2 -> G6 | no parallel teaching writer; a new record kind must be registered in learner_events' registry | G6 writes no teaching and no episode kind; problem data = r2.v1 event ids by reference, validation items split=heldout | RESOLVED; 2.6 |
| D10 | G6 -> G3 | sleep verdict needs a unit record | G3's IMMORTAL `unit` rows under `_T_GOAL`; G6's `sleep_verdict`, `sleep_save`, `sleep` kinds; content G6, codec/fold G3 with unknown-kind pass-through | RESOLVED; 2.6 |
| D11 | G3 -> G6 | does G6(d)'s "declared problem tag" collide with GoalStore `_T_GOAL` (O6) | CONCEDED: one problem record, G3's; G6 references it | RESOLVED; 2.6 |
| D12 | G6 -> G3 | `_T_GROWTH_DELTA` is positional and COMPACTABLE, so it cannot be the unit journal; a digest + ref cannot rebuild | G3 conceded: `unit` rows carry the weights themselves; live percept grower stays 0x08-only (gap reported to main) | RESOLVED for problem units; percept gap -> Q5 |
| D13 | G3 -> G6 | the retention bound is a projection until G3's dev seeds fix it | ACCEPT: G6 depends on `retention_ok` at its freeze | RESOLVED; 2.4 |
| D14 | G7 -> G6 | a pruned unit's placement is dropped by `_refit`; the pruned half of the saved-record clause is uncheckable | ACCEPT: write-ahead `sleep_verdict` before the mask applies (C18) | RESOLVED; 2.6 |
| D15 | G7 -> G6 | G-R6's saved-record clause should read G7's `check_saved`, not a second validator | ACCEPT (P2) | RESOLVED; 2.6 |
| D16 | G7 -> G6 | `origin` separates born from legacy, it does not identify scaffold | ACCEPT: solved-problem arm = born + that problem; eligible-brain arm = born + enumerated legacy learned circuits; envelope/scaffold excluded by enumeration | RESOLVED; 2.4, Q6 |
| D17 | G6 -> main | seed 12345 in-block, already a seed on main | main put 12345 in SEEDS.md's taken list; G6 excludes 12345 and 12000 | RESOLVED; 2.12 |
| D19 | G2 -> G6 (courtesy) / G6 -> G2 | G6 reads G2's predicate `solved(reads)` and splits teach/validation/future | ADOPTED; G6 disclosed (ii)'s future read may repeat validation. G2: right and wider -- any stateless reader with learning OFF makes validation and future identical on (i) and (ii); reads are `route` calls, journaled nowhere today; G2 campaign runs persist=False | RESOLVED; 2.4 (future judgment reported, not counted as a red, for stateless readers). R2 confirmed `route` writes no event on a persisting brain and named the trap (`Brain.respond` -> `chat` -> `learn_turn` writes a `turn` event); reads get no event, any journaled held-out presentation is `split = heldout` and refused for ring/redo; C21 added |
| D18 | G6 -> main | the 0x08 retention finding; found dependencies | main VERIFIED the finding as LATENT (`immortal_retention --assert`: findings 0), queued owner Q5; dependencies recorded in DEPENDENCIES.md; brain.py edit order R2 -> R1 -> R3 -> G6 (OWNERS.md) | RESOLVED; 2.3, 2.13, Q5 |

### 2.16 Amendments at freeze (these GOVERN where they differ from 2.1-2.15)

Agreed in the lead read (board `p0_reports\G6.md`, L1-L5; rulings 9, 9a, 10, 12 in `DEPENDENCIES.md`), then folded
with the Codex critique (findings 16-24, `codex_p0\b2.critique.md`; review record 2.17). Each item names the section
it overrides. Nothing here is measured.

- **A1 -- the keep rule (self-defect 1, L1, Codex 18 and 20; overrides 2.2's multiplicity paragraph and 2.3's
  Disposition).** G6 reads THREE per-pruner dispositions (P-DP, P-EY, P-LL); P-SR's is R3's (2.3), so 2.2's "four" is
  wrong. A pruner is **KEPT iff R4 (non-inferiority) is MET AND either R3 (benefit) or E (efficiency, A11) is MET.**
  Verdict words: "KEEP (beneficial)" on R3, "KEEP (efficient)" on E, both printed when both hold. Removals alone
  never credit efficiency (Codex 18): an active, harmless pruner that saves no measured work is NOT kept. Why R4 is
  the gate: with delta ONE FIXED NUMBER per pruner (A2), R3 MET implies R4 MET (adding the same constant to every
  per-seed difference raises the mean and leaves the variance unchanged, so t rises); this holds ONLY for a constant
  margin (Codex 20 corrected the lead's first wording), which is why A2 fixes delta before data. The second stage is
  an OR of two superiority tests (R3, E); its false-positive rate is bounded by the union bound 2 x alpha = 0.10 under
  the joint null and is printed as such, beside the exact sign-flip FPR of each of R3, R4 and E. The three
  dispositions are read without a family-wise correction (each is its own retirement decision), FPRs side by side.
- **A2 -- delta is one fixed number per pruner, registered before data (self-defect 2, Codex 20; overrides 2.3 R4's
  quanta, its outcomes paragraph and its last sentence).** Each outcome is pooled over a FIXED registered window so
  its quantum is a constant: P-DP: pooled recall correct/total over nights 2-36 of the 900-tick run (35 nights x 36
  trials = 1260), delta_DP = 1/1260; P-EY: pooled held-out correct/total over the refits in the run (the refit
  schedule is fixed, every 20 sleep ticks; the per-refit sizes vary), delta_EY = 1/T_ref with T_ref the LARGEST pooled
  total over the eight dev seeds 12001-12008 of the live arm, frozen and printed before any confirmatory run (the
  smallest step any run of that size registers; a run with a smaller total has a coarser quantum, so the margin is
  never looser than the step it can see on the reference run); P-LL: pooled correct/total over the T = 3 tasks x 40
  items, delta_LL = 1/120 (equal task sizes, so pooled = the mean over tasks). A run whose window is incomplete (the
  run ends before night 36, the attractor's first store comes after night 1, or a refit is missing) is REFUSED,
  never re-quantized. The sentence "the 2026-08-05 P-DP
  point estimates ... would fail it" is DELETED: those were last-night readings, the pooled outcome was never measured,
  and no per-seed spread is in hand.
- **A3 -- one ordered decision function for (b) (self-defect 3, Codex 16; replaces 2.4's Endpoint and P2 on the
  result).** `decide_b(per-seed readings) -> disposition`, steps in order, the first that applies decides:
  1. **Instrument gates:** C7-C10 and C22 must have fired on their constructed inputs, else REFUSED.
  2. **Common gates, per competitor X in {incumbent, greedy, random, SA}:** X is ELIGIBLE iff, on every confirmatory
     seed, its frozen winner (i) satisfies `Problem.solved` on the untouched `future` combinations (A4), (ii) passes
     the untouched retention judgment on `retention_final` (A14), and (iii) its net savings after costs (2.4) vs
     no-prune are > 0 at the bar (paired one-sided t > 1.895). No-prune is the reference and is never adopted as a
     pruner.
  3. **Adjudicate among the eligible:** SA is adopted as THE unit pruner (the single pass retired after its capability
     is ported) iff SA is eligible and beats EVERY other eligible competitor on C (paired one-sided t on
     (C_X - C_SA) > 1.895 for each). Else greedy's fixed-point loop is adopted into the incumbent (SA deleted) iff
     greedy is eligible and beats the incumbent on C, and random does not beat greedy. Else, if random is eligible and
     beats every eligible structured competitor (incumbent, greedy, SA), **random-search win**: SA is deleted (it lost
     to its own control), greedy is not adopted, the incumbent stays unchanged, and the finding "a smaller feasible
     mask exists than the structured searches find" is recorded as a carried opening for a later search method (the
     control is never itself adopted, P8). Else, if the incumbent is eligible, it stays as is and SA is deleted.
  4. **Nothing eligible:** SA is deleted; the owner-drawn prune step for grown units stands UNBUILT and is reported
     (2.9); the incumbent's ineligibility is reported beside P-EY's (a) verdict.
  Greedy's first pass IS the incumbent pass, so C_greedy <= C_incumbent by construction; that contrast's reachable red
  is the single pass already being a fixed point (every difference zero -> degenerate -> NOT MET, A9). Every greedy
  pass re-checks feasibility on validation and greedy stops at the first infeasible pass or the fixed point.
  **C22 (new check):** `decide_b` on constructed readings must return: a size win with a `future` failure -> SA not
  adopted; a size win with negative net savings -> SA not adopted; a random-search win -> incumbent unchanged, SA
  deleted, opening recorded; nothing eligible -> UNBUILT.
- **A4 -- a real held-out split for (b) (L2, ruling 12, Codex 24; overrides 2.4's disclosed limit consequence (1) for
  C1-d).** G2's C1-d is the solved-problem arm's PRIMARY problem in ruling 12's construction ONLY: XOR plus two
  distractor slots of FOUR words each (16 distractor combinations), partitioned disjointly into teach >= 8 /
  validation 2 / `future` 3 / `future_d` 3, with G2's union certificate over teach + validation + future + future_d
  and G2's B13 disjointness check over all four splits. Every C1-d reference in this file means this construction;
  the earlier three-words-per-slot, three-split construction is obsolete. On C1-d the frozen winner's one judgment on the `future` combinations is a
  COUNTED red for every reader, stateless included: masks are selected on validation combinations only, and a winner
  that fails `Problem.solved` on future combinations reds (b) for that arm. On (ii) and (i)-S0C1 (the same-items cells)
  the future judgment stays the reported repeat-read determinism check of 2.4, not counted.
- **A5 -- (d) gets its own untouched set and (b)'s trajectory (self-defect 4, L3, ruling 12, Codex 23; overrides
  2.4's "future is read ONCE: the frozen winner's judgment and (d)'s fresh-process re-solve", 2.6's Red and 2.12's
  (d) and (e) rows).** Dependence, stated
  for the split count (G2's request): (b) and (d) run on ONE trajectory -- (d) saves exactly (b)'s frozen winner -- but
  neither future read selects anything (masks are chosen on validation only; (d) chooses nothing), so a repeat read
  is not selection leakage. It does make (d)'s red non-independent evidence on items (b) already read. Fix (lead
  ruling 12, G2's Option B): C1-d carries 4 words per distractor slot (16 distractor combinations), partitioned teach
  >= 8 / validation 2 / `future` 3 / `future_d` 3, disjoint (G2's split construction and B13 disjointness check,
  extended by G2). (b)'s frozen winner is judged once on `future`; (d)'s fresh-process re-solve reads `future_d` once;
  no item is read by both. G6's first proposal (parity halves of a 2-combination `future`) is withdrawn.
  **Recording blocks for the shared trajectories (Codex 23).** (d) has NO search of its own: its confirmatory save
  under test is, per seed, the sleep that applied (b)'s frozen winner on that same seed, so (d)'s confirmatory seeds
  ARE (b)'s, 12111-12118 (reserve 12121-12128, shared), and the saved artifact is identified by (seed, `sleep_id`,
  `mask_digest`) recorded by (b). G6's (d) block is reclassified: 12301-12308 dev (save/restore harness on fixtures),
  12331-12338 reds (C14, C18 constructions); 12311-12328 RELEASED. (e) adopts R3's one campaign and R3's recording
  block (R3's 16xxx sets, as R3 registers them); G6 runs no second sleep campaign. G6's (e) block is reclassified:
  12401-12408 dev (runner, census and guard on fixtures), 12431-12438 reds (C16, C17, C19 constructions);
  12411-12428 RELEASED. Main is asked to record both releases in SEEDS.md.
- **A6 -- the shuffle obeys the rescheduling rule (self-defect 5;
  overrides 2.5's shuffled-pressure arm).** The shuffled-pressure arm takes the pressure arm's k advanced cycles and
  its k advance offsets (ticks before the scheduled onset), chooses k of the run's N cycles uniformly without
  replacement (same seed, its own RNG stream), and gives them the k offsets in uniformly permuted order. At most one
  onset per cycle, every night replaces its scheduled night, so total sleep ticks and onset count equal the pressure
  arm's by construction and C12 applies unchanged.
- **A7 -- load counts wake learning only, per circuit relative to its own baseline, with a registered cold start
  (self-defect 6, Codex 21; overrides 2.5's Pressure).** L_c = the sum over wake ticks since the last `_enter_wake`
  of |delta w| over plastic circuit c's synapses, divided by c's synapse count; sleep-tick plasticity is excluded.
  Each circuit's baseline B_c is its own mean L_c per completed wake period, persisted. **Eligibility (cold start):**
  a circuit enters the pressure only once it has K_MIN = 2 completed wake periods since its admission (or since the
  run's start) AND B_c > 0; until then it contributes nothing (never 0/0, never x/0). A circuit admitted mid-run (a born
  population, a new grower unit set) starts its own count at admission. A circuit whose B_c is 0 after K_MIN periods
  stays ineligible until a later completed period makes B_c > 0 (its first nonzero period then counts toward the mean
  only; eligibility is re-checked at the next `_enter_wake`). Pressure = the MEDIAN over eligible circuits of L_c / B_c
  (one outlying circuit cannot force a request); sleep is requested when it exceeds 1. With no eligible circuit, no
  request is ever raised and the clock alone schedules (the fallback is the incumbent gate). K_MIN = 2 and the median
  are disclosed session choices, fixed before data. Circuits are enumerated as in 2.4's legacy list (`brain._ixn` rows,
  the NGSNN grower, `AttractorNet.W`) plus born units once they exist. The `sleep_pressure.json` carrier becomes
  {per-circuit baseline_load_per_wake, per-circuit n_wake_periods, per-circuit admission tick, per-circuit
  load_since_last_wake_onset}. **New checks:** C23 a constructed trace with a zero-baseline circuit yields no NaN, no
  infinite pressure and no request from that circuit; C24 a circuit admitted mid-run contributes nothing before its
  K_MIN-th completed wake period and a constructed first update of size 1e6 x its later mean raises no request.
- **A8 -- stale log rows and the sabotage selector (self-defects 7-9).**
  D7: the "start-of-night checkpoint at `_enter_sleep`" is superseded by D7d (no extra checkpoint; start-of-night state
  = previous night's checkpoint + the day's learner events), as 2.7 already states. D10: G6's kinds are `sleep_verdict`,
  `sleep_save` and `sleep_bout` (renamed per D7c), not `sleep`. P-EY's sabotage twin (2.3 R2) picks the units whose
  removal costs the most TRAINING-row RMSE (`x_tr`), never the held-out rows that are its outcome.
- **A9 -- degenerate contrasts (rulings 9 and 9a).** A zero-variance
  paired contrast (every per-seed difference identical, zero included) is printed "degenerate" and read by its kind.
  SUPERIORITY contrasts (R2 live beats sabotage, R3 benefit, A3 greedy > incumbent, (b)'s SA > greedy / random, (c)'s
  pressure > shuffled / clock and the oracle > clock, (e)'s reads) read NOT MET (ruling 9), disclosed as stricter than
  `instruments/paired_contrast.py::contrast`'s zero-variance rule (G1(c) section 2.3 is the pushed precedent); for R2
  this means a sabotage twin identical to the live arm on every seed makes the pruner's A/B REFUSED. The
  NON-INFERIORITY bar R4 reads a degenerate sample by its common value c of (live - neutered + delta) (ruling 9a): c > 0
  MET, c <= 0 NOT MET, "degenerate" printed beside it. So a pruner whose neuter changes nothing on any seed (every
  live - neutered = 0, c = delta > 0) meets R4; under A1 it is still KEPT only if R3 or E (A11) is also met, and only
  after R1 (it acts) and R2 (its outcome sees what it removes). E is a superiority contrast, so a degenerate E reads
  NOT MET.
- **A10 -- waits and owner questions (L4, L5).** 2.10's waits stand: (a)'s confirmatory read waits on R2/R3 (PLAN);
  (a)'s dev runs and the census build may run after the lead's "campaign DONE". Q1-Q6 (2.14) go to the owner batch;
  Q5 is already queued. Q3 now reads: a pruner kept on R4 + E (efficient, not beneficial) -- acceptable under
  "unbounded growth", read as cost per capability rather than a cap?
- **A11 -- E, the efficiency condition (Codex 18; replaces 2.3 R4's "resource saving REPORTED ... NOT tested").** The
  earlier claim that a no-prune twin holds at least the live arm's resource BY CONSTRUCTION is WITHDRAWN: pruning
  changes later growth (a pruned grower can regrow), so the twins' resource ordering is not guaranteed, and a zeroed
  synapse in a dense recurrence (`AttractorNet` iterates its full matrix after `downscale_prune`) saves no executed
  work. E is therefore an independently reachable superiority contrast over the paired 900-tick workload: **executed
  work** W = the deterministic count of synapse and unit operations the pruned component actually performs, summed
  over every tick of the run (an instrumented counter at the component's own forward step, never inferred from
  weights or removals); E MET iff paired one-sided t on (W_neutered - W_live) > 1.895. Beside it, reported and not
  tested: the time-integral of active units and nonzero synapses over the run (not the end count), and per-tick wall
  time. Reachable reds, registered: a dense recurrence whose zeroed synapses are still iterated (P-DP today, so E is
  expected NOT MET for P-DP unless the counter shows otherwise); a pruner whose removals are regrown. C25 (new
  check): a constructed component that zeroes weights but iterates its dense matrix reads W_live == W_neutered
  (degenerate -> NOT MET), and one that skips pruned units reads W_live < W_neutered by exactly the skipped count.
- **A12 -- transactional mask application (Codex 17; replaces 2.6's WRITE-AHEAD paragraph and C18).** Writing a
  pruned verdict before the mask applies (G7's D14, kept) is not by itself crash-consistent. Protocol, per sleep that
  applies a mask (G6 owns the content of the new kinds; G3's codec carries them as unknown kinds; R2 owns the
  checkpoint): (1) write `sleep_mask_prepared` {`sleep_id`, `mask_digest`, kept/pruned unit ids, synapse refs};
  (2) write every domain unit's `sleep_verdict` row, each carrying `mask_digest`; (3) apply the mask to the active
  network; (4) the next learner checkpoint records `applied_mask_digest`; (5) write `sleep_mask_commit` {`sleep_id`,
  `mask_digest`, post-apply state digest}. **Restore:** a prepared mask WITH a commit is applied state; a prepared
  mask WITHOUT a commit is resolved by the restored checkpoint -- if its `applied_mask_digest` equals the prepared
  digest the operation is COMPLETED (the missing commit row is written, idempotent), otherwise it is ROLLED BACK (the
  active network is the checkpoint's pre-mask state; the prepared row and its verdict rows stay in the lake, never
  deleted, P4b, and are folded as `uncommitted`). **Fold** = the latest COMMITTED verdict per unit; an uncommitted
  verdict never changes active membership, labels or synapses, and G7's `check_saved` resolves a unit's keeping or
  pruning sleep from committed verdicts only (G7 and R2 to confirm at G6's P4, filed as debate items D20, D21). **C18
  (replaced):** a crash is injected at each of the seven boundaries (before 1; between 1 and 2; inside 2; between 2
  and 3; between 3 and 4; between 4 and 5; after 5) and the COMPLETE restored state (active membership, unit labels,
  feed/readout/return synapses, folded verdicts, learner weights digest) must equal the registered reference for that
  boundary (pre-mask for the first five, post-mask for the last two); a planted restore that folds uncommitted
  verdicts must red. (Step 4, the prepared row and `mask_digest` are amended: see the DATED PRE-DATA AMENDMENT
  2026-10-03 at the end of 2.16.)
- **A13 -- equal-budget accounting and SA initialization (Codex 19; overrides 2.4's Arms).** One EVALUATION = one
  proposed mask scored (refit on teach + validation score), feasible or not; every infeasible proposal and every
  refit counts against B for every arm, and calibration consumes B. B = greedy's measured evaluation count on the
  instance, as before. **Random:** exactly B draws, each unit kept independently with probability q = the fraction
  kept by greedy's winner on that instance (data-derived, disclosed); the best feasible draw on validation wins; if no
  draw is feasible the arm returns the all-active (no-prune) mask. **SA:** starts from the all-active mask; a proposal
  flips one uniformly chosen mask element; the first min(n, floor(B/2)) proposals (n = mask elements) are the
  calibration sweep, run at acceptance 1 for downhill and feasible moves, and are part of B; T0 = median |delta C|
  over that sweep's proposals / ln 2 (absolute deltas, so T0 is defined when the sweep has no uphill move; every flip
  changes C by at least one unit, so |delta C| >= 1 and T0 > 0); cooling geometric to T0 / B over the remaining
  proposals; infeasible proposals are rejected and counted. Small budget: if B < 4, SA and random are REFUSED on that
  instance (reported, the seed's paired difference not formed, which REFUSES the contrast under 2.2's per-seed
  precondition rule).
- **A14 -- old-task retention: a selecting set and an untouched judging set (Codex 22; overrides 2.4's feasibility
  bullet and its disclosed limit consequence (2)).** For every old task the retention bound reads, two disjoint sets
  are frozen before any search: `retention_select` (read by `retention_ok` inside the search to reject masks) and
  `retention_final` (read ONCE, on the frozen winner, as the untouched rare-task judgment of A3 step 2). For a prior
  problem with G2 splits, `retention_select` = its validation split and `retention_final` = its future split; for a
  legacy task without splits (the interaction learner's taught pairs, P-LL's tasks), its held-out items are split in
  two by a fixed registered rule (index parity in the task's frozen item order). **State reset:** every read is
  learning-OFF from the same post-search state; for a stateful reader each set is read from a snapshot restored
  immediately before it, so no set's reads carry state into another's. Neither `retention_final`, `future` nor
  `future_d` may influence mask selection, refitting or search stopping (the search stops only on B or the fixed
  point); C26 (new check): a planted search that reads `retention_final` or `future` during selection must red (the
  harness asserts zero reads of those sets before the winner is frozen).

**DATED PRE-DATA AMENDMENT 2026-10-03 (D21, agreed with R2; no data read).** R2 writes no per-tick checkpoint (its
r2.v1 checkpoint is written at `_enter_wake` and `close` only), so `applied_mask_digest` rides where R2's redo can
prove the apply. Amends A12:
1. A12 step 4 becomes: "R2's P2 `sleep` progress event for that tick records `applied_mask_digest` (mid-night
   applies); the r2.v1 checkpoint at `_enter_wake`/`close` carries the same field". Restore reads the interrupted
   night's latest progress event, else the checkpoint.
2. `sleep_mask_prepared` gains `tick` and kept bits on its synapse refs (reader: R2's redo takes each re-executed
   tick's mask from this row, keyed on (`sleep_id`, `tick`), and verifies its digest against the event). The verdict
   rows link to it by `mask_digest` and carry no tick.
3. `mask_digest` is defined: sha256 hex of canonical JSON (sorted keys, separators (",", ":")) of {sleep_id, tick,
   sorted kept ids, sorted pruned ids, sorted [kind, pre, post, kept_bit] synapses}; one function
   `mask_digest(prepared_row)`, which R2 imports and does not reimplement; `applied_mask_digest` is the digest of the
   mask, not of the network state. Build order (the lead's ruling 17a, 2026-10-03): the function ships in
   G6's P2 module `instruments/joint_mask_search.py` (fixtures only, no waits), not in P4, so R2's P2, which imports
   it, is not blocked behind G6's P4 (which lands after R2/R3); a later move of the function re-points R2's import in
   the same commit.
4. The P2 file list (2.13 arc split) becomes: `src/snn_data_lake/instruments/joint_mask_search.py` (the search AND
   `mask_digest(prepared_row)`), `tests/test_joint_mask_search.py` (C7-C10, C22, plus a `mask_digest` check: the
   digest is invariant to the order of ids and synapses in the row, and changes when any one kept bit flips). P4 no
   longer defines the digest; it imports it from P2.
C18's seven boundaries are unchanged: "between 3 and 4" (applied, no progress event yet) restores PRE-mask, which
R2's redo gives (no event, no redo); "between 4 and 5" restores the applied mask and restore writes the commit.

**DATED PRE-DATA AMENDMENT 2026-10-04 (A2's windows; lead ruling 24; no deciding data read).** Additive to A2.
Measured on the eight exposed dev seeds 12001-12008 run as the frozen live arm (900 ticks, TEACH at every wake
onset, throwaway homes): `_enter_wake` readings per run 37 (12001) and 36 (the other seven); `_stored_any` True at
the second reading on all eight (False at the first on 12001 and 12006); P-EY refits measured = scheduled on all
eight (22 or 23); P-EY pooled held-out totals 1070, 1060, 1108, 1057, 1081, 1111, 1064, 1076; P-LL 3 of 3 tasks
learned on all eight. Indexing nights by `Brain._nights` would refuse seven of the eight (no `_nights == 36` reading),
so A2's words are read as follows:
1. Night k = the k-th `_enter_wake` reading, counted from 1. The P-DP window is k = 2..36 (35 readings x 36 trials =
   1260; delta_DP = 1/1260 unchanged).
2. A P-DP run is REFUSED if it has fewer than 36 readings, or if `_stored_any` is False at the k = 2 reading (every
   window night reads stored patterns). This replaces A2's "the attractor's first store comes after night 1", which
   read literally would refuse 12001 and 12006 although their whole window reads stored patterns.
3. T_ref = 1111 (the largest of the eight pooled totals, seed 12006), so delta_EY = 1/1111, fixed before any
   confirmatory run.

### 2.17 Review record

- **Lead read:** 2026-10-03, CLI lead session 2d85b496; board `p0_reports\G6.md` section "=== LEAD READ ===" (L1-L5),
  rulings 9, 9a, 10, 12 in board `DEPENDENCIES.md`.
- **Advisor read:** CLI lead session, Fable, 2026-10-03 (the L1 justification; its "R3 implies R4" holds for a constant
  margin only, corrected per Codex 20 and made true by A2).
- **Codex critique:** gpt-6.1-sol; critique `codex_p0\b2.critique.md` (findings 16-24 are G6's: 2 BLOCKER, 6 MAJOR,
  1 MINOR); reply file `codex_p0\b2.reply.txt`, sha256 `e40e3fb21341ddd031ee018d8127745c9f6f3f34265118f787ffcbd1aac7edf9`
  (as reported by the lead; G6 did not re-hash it while the host was held quiet). Dispositions: 16 -> A3 (ordered
  `decide_b`, C22); 17 -> A12 (prepared/commit, restore rule, C18 replaced); 18 -> A1 + A11 (E, executed-work counter,
  C25); 19 -> A13; 20 -> A2 (fixed delta) + A1 (constant-margin reason); 21 -> A7 (cold start, median, C23, C24); 22 ->
  A14 (two retention sets, C26); 23 -> A5 ((d) on (b)'s seeds and artifact; (e) on R3's block; releases); 24 -> A4
  (ruling 12's construction only). All nine accepted; none refuted.
- **Open after freeze:** D20 (G7: `check_saved` reads committed verdicts only) and D21 (R2: the checkpoint carries
  `applied_mask_digest`) are G6's requests to those rows, to be settled before G6's P4 lands; until they agree, A12's
  restore rule is G6's design, not yet theirs.

## 3. Results

None. Nothing is measured beyond section 1; no deciding test has run.
