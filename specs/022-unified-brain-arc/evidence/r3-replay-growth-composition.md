# R3 / T104 -- eligible replay composed with the incumbent learner and grower; earned growth on held-out data; the 2026-09-27 sleep-plasticity arms (pre-registration, wave 1; frozen by the lead's P0 commit)

Plan row: PLAN-FINAL R workstream, R3 / T104 (FR-039, SC-017) with its dated note "2026-09-27 (sleep plasticity)"; Part F's
R-R3 and its controls; the Combined-architecture paragraph ("new units must earn retention on untouched data, not on a
replayed training example"); POLICIES P1, P2, P3 (+ amendment), P4b, P4d (the owner's 2026-09-27 sleep lines) and P8.
Status: written by the R3 row agent on branch `claude/r3` (base `e1b3ae3c`); the lead (the single integrator) commits it as
R3's P0 after the review recorded in 2.17. Every number not marked MEASURED is a **[PROJECTION]**. Orchestrator rulings
of 2026-10-03 (board `DEPENDENCIES.md`, "Orchestrator rulings") are folded in and cited where they decide something.
Sections 2.15 (after the lead read) and 2.16 (after the Codex critique) hold amendments; each supersedes the text it
names, and where an amendment and older text disagree the amendment governs.

## 1. Territory, taken at `e1b3ae3c` (2026-10-03)

Static facts were read in this worktree; the runtime facts come from ONE scratch probe on R3's dev seed 16000 (out of
tree, throwaway `NORD_HOME` set before import, `NORD_PERSIST=0`, no restore, single process; afterwards
`C:\Users\vinhl\.snn_data_lake` still read its 2026-09-27 01:19 mtime, so the real home was not written).

1. **The live sleep tick.** `engine/brain.py::Brain._sleep_tick` runs, in order: `neuro.tick()` (the grower; a refit
   every `refit_every` = 20 sleep ticks), the drives' rest, `_episode_consolidator.replay(steps=60)`, then the T058
   block -- (a) `brain._ixn.replay(k=8)` re-drives recorded wake interactions through the full deposit path, (b) the
   structural slow clock `sleep/structural_rewiring.py::StructuralRewiring.rewire` on the learner's rows, (c) the
   Benna-Fusi rest (`_plasticity.relax` written back) -- then `attractor.sleep(seed, episodes=3, learn=True)` +
   `downscale_prune()`, then `buffer.replay` (value table). `Brain.neuter_sleep` switches `_sleep_caps`
   {replay, benna_fusi, structural} and "phase"; nothing switches `attractor.sleep`.
2. **The learner.** `readout/interaction_learning.py::SpikingInteractionLearner` (one R-STDP neuron per intent; the brain's
   has 9 labels, n_in 52, t_post 6). Its replay ring `_replay` holds the last `_REPLAY_CAP` = 64 wake interactions
   (stimulus, label, utterance key); `replay(k)` samples uniformly with the instance `rng` and re-drives through
   `interact(_record=False)`. G0(c) persists its continuation state including the ring (`O_ring` carrier) in
   `interaction_learner.json` (`_T_STATE`); the end of a night (`_enter_wake`) is the exact learner boundary
   (`evidence/g0c-learner-persists.md` 1.3). Its own disjoint corpus is `_CONV_TRAIN` / `_CONV_TEST` (5 intents x 4
   train + 4 test lines; test lines are no-keyword paraphrases).
3. **The grower.** `growth/neurogenesis_engine.py::NeurogenesisSelfImprove`: `observe(percept, category)` on wake
   buffers (percept_{t-1} -> category_t) in `_buf_x/_buf_y` (cap 1000, FIFO), scores the LIVE instrument
   (`live_n`, `live_correct`) and lets `OnlineReadout("delta")` learn; `_refit` grows on top of the retained NGSNN,
   prunes unearned units, scores a `marginal_holdout` slice, and consults the growth gate (`GrowthStop`, T047). The buffer
   is process-local and NOT in `growth_state_bytes`. L2's grow-on-plateau trigger was DELETED on its null
   (`evidence/l2-grow-on-plateau.md` 6) and that null stands here.
4. **No percept-transition record exists.** `substrate/home.py` tags are 0x01-0x0b; none holds a wake percept. R2's
   hydration journals interaction-learner turns only (R2 message, 2026-10-03). **Orchestrator ruling (1), 2026-10-03:
   the percept-transition record is ASSIGNED TO R2 as an added part.**
5. **A live incumbent of the composed job (P2).** `sleep/lifelong_consolidation.py::LifelongConsolidator` is constructed
   in `Brain.__init__` and `_enter_sleep` runs `learn_task` on synthetic `_make_tasks`: GrowingLearner prototype growth
   + INTERLEAVED old-task replay (`ReplayBuffer`) + prune to a capacity ceiling (budget 6) + an oldest-task retention
   probe (`lifelong_retention` on state). Its job (grow + interleaved replay + retention) is R3's composition on a
   different learner. **Orchestrator ruling, 2026-10-03: register the port as an arm; one mechanism survives.** G3 holds
   it is not a G3 growth-loop incumbent (prototype recruitment, a capacity ceiling, not a spend budget) and defers the
   adjudication to R3 (debate log 4.2).
6. **Structural turnover is ~0 by design** (plan R3 note). MEASURED: turnover 0.0 on 10/10 sleep ticks (seed 16000, this
   probe); G6 measured 0.0, pruned 0, over 102 cycles at 200 ticks (seed 12001); R2 measured 0.0 / 0 / 0 on 6 ticks
   (seed 15000).
7. **`AttractorNet.sleep(learn=False)`** runs the same episodes, steps and RNG draws as `learn=True` and skips only the
   covariance STDP + `_normalize` (static read of `sleep/sleep_replay.py::AttractorNet.sleep`): a compute-matched OFF.
   Its patterns are stored only on wake ticks (`Brain._wake_tick` runs `attractor.store(_patterns[_wake_ticks % 3],
   reps=1)` then sets `_stored_any`); without them `_consolidation_acc` is never read. F3 therefore makes those two
   calls itself (2.3) and never runs `_wake_tick`, so no R3 protocol touches the motor loop.
8. **Restart carrier (R2, MEASURED seed 15000):** `Brain._rewirer` (`_rng`, `_signs`, `_budget`, `_cycles`) is
   process-local and rebuilt fresh after a restart; on a turnover-0 schedule weights and learner RNG still restore equal
   on 6/6 ticks (only `_cycles` differs). R2 puts the rewirer state into its r2.v1 checkpoint.
9. **The statistic exists once:** `instruments/paired_contrast.py` (`T_BAR` 1.895, `paired_t` ties counted,
   `permutation_fpr` exact over 2^S sign flips, `contrast` with its degenerate rule).
10. **Seed block 16000-16999 is unused.** `grep -rlw 4242 specs src` 14 files and `7000` 9 files (positive controls);
    every hit in `\b16[0-9]{3}\b` over specs/src/tests/tools is a byte count, a paper id, a sequence term or a corpus
    size (`PLAN-FINAL` "16032 B", `developmental_loop.py` "2311.16141", `spike_chat_lake.py` "16807",
    `tools/capacity_curve.py` 16000 = a corpus length, `t078_attention_schema.json` counts); none is a seed.

**MEASURED, dev probe (seed 16000, `e1b3ae3c`, 2026-10-03; session A = {GREET, GOAL}, B = {MEMORY, CONSCIOUS, ARM}, 3
reps of `learn_turn(line, firewall_intent=label)` per session, A then B, no restore):**

| quantity | value |
|---|---|
| import / `make_brain` | 0.49 s / 2.19 s |
| teach A (24 turns) / B (36 turns) | 0.006 s / 0.009 s |
| held-out acc A after A (n 8) | 1.000 |
| held-out acc A after B | 0.250 (interference exists) |
| held-out acc B after B (n 12) | 1.000 |
| ring length after B | 60 (both sessions inside the 64 cap) |
| 10 `_sleep_tick` calls | 0.30 s; replayed 8 on the last tick; turnover 0.0 on all 10 |
| held-out acc A / B after the 10 sleep ticks | 0.500 / 0.917 |
| grower: 600 SyntheticPerceptStream percepts (k 5) + 20 refits | 2.58 s; 22 units; held-out 0.80 |

The A trace (1.000 -> 0.250 -> 0.500) is R3's interference positive control and the first sign that the incumbent ring
replay is reachable on held-out data. One seed decides nothing.

## 2. Pre-registration (DRAFT)

### 2.1 The question and what a result licenses

- **Question (R-R3).** Does sleep replay, run through the incumbent learner and composed with the incumbent grower,
  give a held-out benefit (acquisition on new sessions, retention on old ones) that is attributable to the replay or to
  admitted units rather than to extra updates, stays inside an interference bound, and survives a fresh-process restore?
  And (2026-09-27 note) does each live sleep-plasticity incumbent -- the replay deposit, the Benna-Fusi rest, the
  structural slow clock, `attractor.sleep` learning -- earn its place against its equal-compute OFF arm?
- **A MET family licenses only** its own worded claim at its own budget on its own workload (2.5): never "sleep
  improves the brain", never a structural, stage-order or extended-duration benefit that its own contrast did not
  read, never a reservoir or engine claim (G9 only), never a growth gain from a unit count (P4d "Not claimed").
- **What is not this row:** the hydration reader and its R-R2 restart proofs (R2), the staged-bout runner (G6 P4), the
  solved-problem joint mask (G6(b)), growth for G2's capacity cell (G3), the census of sleep PRUNERS (G6(a)).

### 2.2 The pre-sleep state PS(s), shared by every interaction-learner family

- **Build, never copy.** Each arm builds PS(s) itself, in its own process and its own scratch home
  (`<tmp>/r3/<family>/<arm>/<s>`, `NORD_PERSIST=1`): `make_brain(seed=s)`; the intents I = `_CONV_INTENTS` (all five
  are learner labels); `random.Random(s * 7919 + 3).shuffle(I)` -> A = first 2, B = last 3, fixed BEFORE any teaching,
  indexing or selection; teaching = R = 3 reps per session, each rep the session's `_CONV_TRAIN` lines in a seeded order,
  each line `b.brain._ixn.learn_turn(line, firewall_intent=label)`; session A, then session B; no wake or sleep tick in
  between. Held-out = `_CONV_TEST` for A and B (disjoint strings; utterance keys checked, K3).
- **The fork is verified, not assumed (K1):** at bout start every arm records a `PreSleepDigest` = sha256 of the
  learner weights, the ring (stimulus, label, key), the learner `rng` state, the plasticity-stack weights and
  `_rewirer is None`; all arms of one seed must carry the same digest. Rebuild-from-seed is chosen over an in-process
  deepcopy because the stack holds a lock and shared lists (G0(c) 1.2) and a copy would not travel the restore path.
- **The bout.** `b._enter_sleep()`; n x `b._sleep_tick()` with the arm's per-tick action (2.3); `b._enter_wake()` (the
  exact boundary); `b.close()`. Then a FRESH process builds `make_brain(seed=s)` on the same home (restore) and
  evaluates. G6's eligible-brain structural arm consumes this fixture unchanged (debate 4.3).
- **Compute ledger (the unit every equal-compute claim is made in): PRESENTATIONS** = forward drives of one stored
  stimulus through the learner (`interact` or `respond`) during the bout. Deposits, consolidate calls, structural edits
  and wall time are reported beside it and never claimed matched. G6's mask arm reports its evaluations converted to
  presentations (one validation item scored = one presentation; one refit = rows x passes), wall beside.
- **Evaluation, learning OFF:** `route_text` on every held-out line in the restored process; no `learn_turn`, no
  `interact`, no replay; the learner weights and `rng` state are digested before and after (K4). Endpoints:
  E_all (20 items, quantum 0.05), E_A (8 items, 0.125), E_B (12 items, 0.0833); the in-process post-bout value is
  recorded beside the restored one (K5). Secondary, never decisive: the mean correct-label margin (drive of the
  correct label minus the best other) in the learner's membrane units.

### 2.3 Families, arms and contrasts

Every arm of a family runs from PS(s) of the same seed; arms differ only after PS(s). k = 8 presentations per sleep tick
(the incumbent's own `replay(k=8)`). Budgets in sleep ticks: short 10, medium 40, extended 160 **[PROJECTION; no human-
hour timer (P4d reading); owner question O6]**. "fwd" = the instrument draws k ring entries per tick with a twin RNG
cloned from the learner's `rng` at bout start and calls `respond` only (presentation-matched, no plasticity).

| family | arm | caps (replay / benna_fusi / structural) | per-tick action | budget |
|---|---|---|---|---|
| F1 synaptic, topology fixed | ON | on / on / OFF | -- | medium |
| | OFF (control) | OFF / OFF / OFF | fwd | medium |
| | R (replay deposit only) | on / OFF / OFF | -- | medium |
| | BF (rest only) | OFF / on / OFF | fwd | medium |
| | Rec (recency-only, extra-updates control) | OFF / on / OFF | k draws from session-B ring entries only, `interact(_record=False)` | medium |
| | ORACLE (sensitivity control) | as OFF | as OFF; session A re-taught once (8 turns) at wake before the bout | medium |
| | A-ONLY (interference positive control) | as OFF | as OFF; PS(s) built with session A only | medium |
| F2 structural | S-ON | on / on / on | -- | medium |
| | S-OFF | on / on / OFF | -- | medium |
| F3 attractor | AT-ON | as F1-ON | `attractor.sleep` unchanged | medium |
| | AT-OFF | as F1-ON | `attractor.sleep` wrapped to `learn=False` (compute-matched, 1.7) | medium |
| F4 budget (flat) | SHORT / MED / EXT | as F1-ON | -- | 10 / 40 / 160 |
| | MASSED (extra-updates control) | OFF / on / OFF (benna matches EXT) | the same n x 8 draws as EXT re-driven at WAKE (`interact(_record=False)`, twin RNG) before `_enter_sleep`; fwd during the bout | 160 |
| F5 staged (WAITS on G6's runner) | STAGED / FLAT / SHUFFLED | as F1-ON | `run_bout(brain, stages, rng_seed)` (4.3) | matched presentations |
| F6 composed, learner (WAITS on R2's reader) | HIST-U / HIST-I / RING / OFF | on / on / OFF | see 2.3.1 | medium |
| F7 composed, grower (WAITS on R2's percept record + G3 P2) | see 2.3.2 | -- | -- | -- |

F3 precondition: the attractor needs stored patterns, so F3's PS(s) ends with W = 30 calls of the wake tick's own two
lines, `b.attractor.store(b._patterns[i % 3], reps=1)` and `b._stored_any = True` (a CONSTRUCTED pre-sleep state,
disclosed; `_wake_tick` is never called), identical across its arms; `_stored_any` False reads NOT READ. G6(a) owns
`downscale_prune`; F3 reads only the sleep LEARNING. G6's harness-alive control also neuters `AttractorNet.sleep` on
G6's own workload; it is a control there, never a verdict -- the verdict is F3's (debate 4.3).
(Amended 2026-10-04: W is no longer 30 -- see "DATED PRE-DATA AMENDMENT 2026-10-04 (F3's store count W)" at the end.)

MASSED ledger, written out: deposits matched to EXT (n x 8 draws, delivered at wake instead of in sleep); bout
presentations matched by fwd draws; total forward passes therefore 2x EXT's -- disclosed, never claimed equal; benna
on in both, so C9 isolates WHEN the deposits happen, not the rest phase (which C4 attributes).

**2.3.1 F6, the composed test on the learner (needs R2).** Session B is lengthened to R = 6 reps (72 turns > 64), so
session A is EVICTED from the ring and only lake history can bring it back. The bout starts in a fresh process restored
from the end of wake (the real restart case). Arms at equal presentations, benna on, structural off: RING (the incumbent
after R2 lands: R2's DEFAULT hydration -- the last 64 eligible teach/turn events in lake order, deterministic, no RNG,
equal to an uninterrupted process's ring (R2, 2026-10-03) -- sampled uniformly by `replay(k)` with the restored learner
RNG; with session B at 72 turns that ring holds B only), HIST-U (uniform over eligible history A u B via R2's
`substrate/learner_events.read_learner_events(kinds=("teach","turn"), exclude_sessions=<held-out>)`), HIST-I
(interleaved: alternate A-session and B-session draws, within-session order kept, never joining two sessions as a
transition -- LifelongConsolidator's interleave rule PORTED, 2.6), OFF (fwd). Drawn event_ids are joined against the
held-out session set (K3). Hydration and replay must not advance `_step`, `_nights`, `_wake_ticks` or motor output
(R2's check, re-read here, K8).

**2.3.2 F7, the composed test on the grower (needs R2's percept record and G3's P2).** Curriculum: task A =
`SyntheticPerceptStream(seed=s, k_modes=5)`; task B = `SyntheticPerceptStream(seed=s+500, k_modes=8)` with its categories
offset by 5, so A and B are retention-compatible (distinct percept modes, distinct categories) and B needs new units
**[PROJECTION; the P4 design step on dev seeds confirms A is forgotten once B fills the 1000-row buffer]**. Separate
untouched evaluation sessions per task (stream seeds s+520 for A and s+540 for B, never buffered; all derived seeds
listed in 2.10). Replay enters through
`NeurogenesisSelfImprove.replay_rows(rows, *, source_ids) -> int`: rows join the NEXT refit's TRAINING rows only, are
excluded from `marginal_holdout`'s held-out slice, never pass through `observe()` (so `observed`, `live_n`,
`live_correct`, `_prev` are untouched) and never change refit cadence (added after G3's P2 as R3's stacked commit, G3
reviews; 4.2). Arms, equal refit training-row count per refit: replay {NONE (the extra rows are duplicates drawn from
the current buffer), HIST-U, HIST-I (the ported LifelongConsolidator interleave)} x growth {ON (incumbent), OFF}.
Growth OFF is NOT a closed gate: under a closed gate `_refit` scores the retained model and returns before `fit`, so
beta is never re-solved and replay rows would have no consumer (C14 would then read consumption as growth). OFF instead
pins the engine's existing `l_max` at the post-A unit count at the A->B boundary (an instrument-side setting of
`_l_max`, disclosed): `_refit` still runs `fit(init_W=prev)`, re-solving beta on the same training rows (replay
included) while adding no net units; units `prune_unearned` drops may be replaced up to the pinned count, never past it
**[PROJECTION: that `fit` re-solves beta with zero additions when the seeded W already holds `l_max` units is a static
reading of `growth/ng_snn.py::NGSNN.fit`; P4 verifies; no G3 interface is needed]**. Check K15 holds the two arms'
consumed replay-row counts equal; PRESIZED (an NGSNN pre-sized at the growth-ON+HIST-I arm's final unit count, same
rows, same refits); LIFELONG (LifelongConsolidator's own policy on the same rows, the P2 comparator). Evaluation:
`model.predict` on the untouched A and B sessions after `restore_growth_state` in a fresh process; the online readout
does not learn during evaluation.
The rows come from R2's percept records (R2 draft schema, 2026-10-03): `_T_EPISODE` kind `percept`, schema r2.v1
(event_id, stream, session, seq, x as exact float64, category, n_protos, encoder, provenance, valid, split), written by a
default-None `transition_sink` in `observe` and read through the same `read_learner_events(("percept",), ...)`; pairs
(percept_{t-1}, category_t) are formed WITHIN a session only, never across a restart. R2's hydration refills the buffer
at construction, so after a restart the NONE arm's buffer is the hydrated recent buffer (the post-R2 incumbent). F7
writes its synthetic task streams through that writer into the arm's scratch lake only, encoder
`synthetic.stream.k<k>`, provenance `instrument`, `n_protos` = the stream's category count, read with
`read_learner_events(("percept",), ..., allow_instrument=True)`; the reader refuses that flag on the production home
with a visible ValueError, and the Brain's hydration never passes it (R2 conceded, debate 4.1). Edit windows on
`growth/neurogenesis_engine.py`, each a stacked commit the others review and main sequences the claim handoff: R2
(`transition_sink` + hydration, inert with no sink or lake), then R3 (`replay_rows`), then G7 (persistence arrays in
`restore_growth_state`); G3's P2 touches no line of it and only reviews, and every one of these commits re-runs G3's
fingerprint test on SyntheticPerceptStream 9000-9003 (G3's condition, debate 4.2).

### 2.4 Endpoints, statistic, bars

- **Statistic (Part F):** per contrast, d_s = E(arm, s) - E(control, s) over exactly S = 8 confirmatory seeds, through
  `paired_contrast.contrast(name, d, T_BAR)`: MET iff t > 1.895 (one-sided, 7 d.f.), ties counted, the exact 2^8
  sign-flip false-positive rate printed beside it; a list shorter than 8 halts, never averages what completed. The
  incumbent degenerate rule (a zero-variance sample is MET iff its common value > 0, reported "degenerate") is kept and
  disclosed. No statistic is normalized by a quantity a control destroys; no majority-of-seeds rule.
- **Beside every contrast:** mean, sd, the one-sided 95% lower bound (mean - 1.895 sd / sqrt 8), the endpoint quantum,
  and the minimal detectable mean at this sd (1.895 sd / sqrt 8). Design effect Delta = 0.10 on E_all (two of 20
  items) **[PROJECTION; reported, not an acceptance threshold]**.
- **Interference bound (non-inferiority, a stated statistic -- L2 6.3's lesson):** d'_s = E_A(arm, s) - E_A(control, s)
  + eps with eps = 0.125 = one A item (P1 case (a): a fixed point on a normalized accuracy, the smallest change E_A can
  register; not tuned). WITHIN BOUND iff `contrast("NI", d', T_BAR)` is MET; otherwise INTERFERENCE RED. The boundary
  is strict and stated now: the bound is "the arm loses LESS than one A item relative to its control, shown at
  t > 1.895"; an arm that loses exactly one A item on every seed gives a degenerate d' of 0, which `contrast` reads NOT
  MET, so it reads INTERFERENCE RED. For F7 eps = one quantum of its A evaluation session, same rule.
- **Turnover rule (F2):** a structural verdict is read only if the S-ON arm's mean per-tick turnover is > 0 on all 8
  confirmatory seeds; else "NOT READ (turnover 0 on k/8)". G6 adopts this detector (debate 4.3).

| contrast | family | d | reads |
|---|---|---|---|
| C1 (primary F1) | F1 | E_all(ON) - E_all(OFF) | synaptic sleep plasticity benefit |
| C1-NI | F1 | E_A(ON) - E_A(OFF) + eps | interference bound |
| C2 | F1 | E_all(ON) - E_all(Rec) | MET: old-experience replay carries the gain; NOT MET: recency-sharpening only |
| C3 / C4 | F1 | E_all(R) - E_all(OFF) / E_all(BF) - E_all(OFF) | per-item attribution (P8 3) |
| C-IF (positive control) | F1 | E_A(A-ONLY) - E_A(OFF) | interference exists for replay to fix |
| C-SENS (positive control) | F1 | E_A(ORACLE) - E_A(OFF) | the harness can see a retention gain at S = 8 |
| C5 (+NI) | F2 | E_all(S-ON) - E_all(S-OFF) | structural benefit (turnover rule first) |
| C6 | F3 | mean over nights of `_consolidation_acc`, AT-ON - AT-OFF | attractor sleep learning |
| C7 / C8 (+NI) | F4 | E_all(MED) - E_all(SHORT) / E_all(EXT) - E_all(MED) | a longer budget beats the shorter |
| C9 | F4 | E_all(EXT) - E_all(MASSED) | MET: a sleep-duration benefit; NOT MET: "extra updates alone" |
| C10 / C11 | F5 | E_all(STAGED) - E_all(FLAT) / E_all(STAGED) - E_all(SHUFFLED) | stage benefit / stage-order credit (C11 read only if C10 MET) |
| C12 (+NI on E_B) | F6 | E_A(HIST-U) - E_A(RING) | hydrated history protects the evicted session |
| C13 | F6 | E_all(HIST-I) - E_all(HIST-U) | interleaving beats uniform |
| C14 | F7 | E_all(HIST-I, growth ON) - E_all(HIST-I, growth OFF) | growth adds under replay |
| C15 | F7 | E_all(HIST-I, growth ON) - E_all(PRESIZED) | the gain is GROWTH, not capacity (L2's null idiom) |
| C16 | F7 | drop(silence admitted units) - drop(silence an equal count of pre-B units matched on output-weight L2 norm and firing on the items) | admitted units earn retention (G3's `silence_units` + matched picker, imported) |
| C17 (+NI on E_B) | F7 | E_A(HIST-I, ON) - E_A(NONE, ON) | replay, not extra rows, protects A |
| C18 | F7 | E_all(HIST-I, ON) - E_all(LIFELONG) | P2: the composed path beats the incumbent policy |

### 2.5 Decision tables (rows read top to bottom; the first that applies decides; each a pure function with a red per row, K11)

**F1:** any check in 2.7 red -> HALT (instrument defect, no verdict) | C-SENS NOT MET -> REFUSED (blind harness) | C1 NOT
MET -> NULL "sleep synaptic plasticity not shown on held-out at the medium budget" | C1 MET, C1-NI NOT MET -> RED
(interference beyond bound) | C1 MET, C1-NI MET -> MET, worded by C2 and attributed by C3 / C4; C-IF NOT MET adds "no
interference was present to fix" and C2 is then reported, not read.

**F2:** the turnover detector's positive control dark -> REFUSED (blind turnover) | turnover rule fails -> NOT READ
(turnover 0 on k/8); G6 puts retire-vs-dated-exemption to the owner (G6's Q4; not asked twice) | C5 NOT MET -> NULL | C5
MET, NI NOT MET -> RED | C5 MET, NI MET, restore check K5 red -> NOT CLAIMED (gain not restorable; R-R3's restart
clause) | else MET.

**F3:** `_stored_any` False -> NOT READ | C6 NOT MET -> NULL | C6 MET -> MET (the attractor's own retention task only).

**F4:** a longer budget is credited only if its step (C7, then C8) is MET with NI MET; EXT is a "sleep-duration benefit"
only if C9 is also MET, otherwise "a gain from extra updates alone" (not an accepted benefit, plan R3 note). Cost
(presentations, deposits, wall) is printed per budget.

**F5 / F6 / F7:** the same shape; F6 MET needs C12 and its NI; F7's composed claim ("replay composed with growth gives an
attributed held-out benefit") needs C14, C15, C16, C17 and its NI all MET plus restore equality; C18 decides P2.

### 2.6 Controls (none is the method itself)

- **Compute-matched plasticity OFF** (F1-OFF, F3 AT-OFF): presentations happen, plasticity does not.
- **Recency-only replay** (F1-Rec) and **massed wake** (F4 MASSED): the same updates spent without old experience / without
  sleep -- the plan's "a gain from extra updates alone" control.
- **Positive controls before any verdict:** C-SENS (a real re-teaching at wake must register on E_A), C-IF (interference
  must exist), the turnover detector on a constructed sign flip (the `tests/test_sleep_capabilities.py` `_flip_site`
  idiom), F3's historical positive (neutering attractor sleep pinned `_consolidation_acc` at 0.333 on 7/7 seeds,
  brain.py's 2026-08-05 record) re-run once on dev seeds.
- **F7:** growth OFF, an equal-count PRESIZED twin, equal-count matched silencing (dead units excluded from both sets;
  a pool too small to match halts -- G3's C9 rule, conceded in debate 4.2), random replay order inside HIST-U.
- **P2 comparator:** LifelongConsolidator's own policy (C18). Its unique capability -- interleaving old-task replay
  while learning a new task, and an oldest-task retention probe -- is PORTED first as HIST-I and E_A; then one survives:
  if C18 is MET the composed path is THE path and LifelongConsolidator's live instance is retired (its prune half is
  G6(a)'s census); if NOT MET, R3 does not land a second composition and the owner is told (O4).

### 2.7 Checks, each with a constructed, reachable red (a red that does not fire halts its arc)

| id | check | constructed red (must fire) |
|---|---|---|
| K1 | A/A fork: every arm of one seed carries the same `PreSleepDigest`; two ON twins give byte-equal post-bout weights and equal E | build one arm with the split RNG offset by 1 |
| K2 | presentation counts equal across a family's arms (per tick and in total) | drop one fwd draw per 10 ticks |
| K3 | no held-out key in teaching, the ring, any drawn replay item or (F6) any drawn event_id | plant one `_CONV_TEST` line in session B's teaching |
| K4 | learning OFF in evaluation: weights and `rng` digests equal before / after | evaluate through `interact` instead of `route_text` |
| K5 | restore equality: in-process post-bout E equals restored E on every seed | delete `interaction_learner.json` before the fresh process; (F2) the constructed rewirer sign-flip from R2's red O_rw |
| K6 | fwd arms are plasticity-free: weights byte-identical across an OFF bout with benna off | route one fwd draw through `interact` |
| K7 | turnover detector sighted | the A/A stable-sign schedule must read exactly 0.0 and the planted flip > 0 |
| K8 | no wake chronology or motor output during a bout, hydration or replay: `_step`, `_nights`, `_wake_ticks`, the grower's `observed` / `live_n`, the arm command count unchanged | inside a bout, bump `_step` by one and feed the grower one `observe()` (no motor-loop call is ever used as a red) |
| K9 | (F7) replay rows never in the held-out slice and never through `observe()` | route one replay row through `observe()` |
| K10 | (F5) synthetic REM items are marked internal, never written as lake events, ring entries, labels or held-out | write one synthetic item to the scratch lake |
| K11 | each decision table is a pure function and every row has a constructed input that reaches it | a table row with no reaching input |
| K12 | seed rule: a contrast refuses n != 8 and any seed outside 16000-16999 | pass 7 values; pass seed 300 |
| K13 | AttractorNet compute match: AT-ON and AT-OFF consume the same RNG draws per night | wrap with `episodes - 1` |
| K15 | (F7) growth-OFF consumes the same replay rows per refit as growth-ON (both re-solve beta) | route the OFF arm through the closed-gate branch: its consumed-row count falls to 0 |
| K14 | safety (board PROTOCOL 4b): every Brain is built under a fresh throwaway `NORD_HOME` (or `substrate.home.throwaway_home()`); `C:\Users\vinhl\.snn_data_lake` is never written (its mtime is read before and after every run); the daemon is never started | build one arm with `NORD_HOME` unset in a sandboxed `HOME` and watch the mtime check fire |

### 2.8 Null dispositions (each a causal claim; the instrument owes its positive control first)

- **F1 null** claims "the incumbent sleep deposit + rest add no held-out routing at the medium budget on this corpus";
  its instrument owes C-SENS and C-IF MET first, or the reading is REFUSED / reworded. Disposition: the replay and
  benna_fusi caps are P3 neuter-A/B misses on held-out capability (not R-R3 reds: R-R3's 2026-09-27 clause fires only
  on a CLAIMED benefit). R3 reports them; the owner decides retire vs dated exemption (O1). The caps' other measured job
  (reaching `consolidate()` on the sleep tick, T058) is unaffected.
- **F2 null / NOT READ** -> G6 puts it to the owner (G6 Q4); R3 does not delete `structural_rewiring.py`.
- **F3 null** -> reported to G6(a) beside its `downscale_prune` census; the owner decides.
- **F4** an EXT gain that C9 does not separate from massed wake is "extra updates alone", never credited as sleep.
- **F6 / F7 null (R-R3 RED):** the R3 composition code (HIST-U/HIST-I draw policies, `replay_rows`) is DELETED (P8 5);
  R2's hydration stays (its own R-R2 test). Admitted units that fail C15 or C16 mean "growth under replay is not shown";
  L2's null stands and is not relabelled.

### 2.9 What a null deletes and what it does not

- Deletes (P8 5): only R3's own invented code that measured nothing -- the F6/F7 draw policies, `replay_rows`, and any
  live wire R3 added for them.
- Never deletes: an owner-drawn part (P4d); **the owner's 2026-09-27 sleep requirement** ("still be able to sharpen its
  understanding while sleeping, using synaptic plasticity") -- a null reverts or declines to claim the mechanism and the
  owner decides on the evidence, the requirement stands (treated as P4d treats an owner-drawn part; O1); an incumbent
  cap (P3 owner decision, O1); the grower; R2's reader; sleep itself.

### 2.10 Seeds (block 16000-16999; nothing outside it)

| set | use |
|---|---|
| 16000-16007 | dev / design rehearsal (EXPOSED; the 2026-10-03 probe used 16000) |
| 16100-16107 | F1 confirmatory (read once) |
| 16110-16117 | F2 confirmatory |
| 16120-16127 | F3 confirmatory |
| 16130-16137 | F4 confirmatory |
| 16140-16147 | F5 confirmatory (reserved until G6's runner) |
| 16200-16207 | F6 confirmatory (reserved until R2's reader) |
| 16210-16217 | F7 confirmatory (reserved until R2's percept record and G3's P2) |
| 16300-16347 | reserve (six sets of 8; a set replaces a failed one only under the repair rule below) |
| 16500-16799 | F7 DERIVED stream seeds only: task B = s+500, A evaluation = s+520, B evaluation = s+540 (dev 16000-16007 -> 16500-16547; confirmatory 16210-16217 -> 16710-16757; reserve sets map the same way); never a primary seed |
| 16900-16999 | constructed reds and positive controls only |

Paired-seed rule: one seed drives every arm of a contrast from PS(s); F7's derived stream seeds belong to the primary
seed's record and never coincide with another primary seed. A TIMEOUT or a halted cell re-runs the SAME seed after the defect is fixed and recorded; a reserve set
replaces a confirmatory set only for a recorded harness defect found before any of its contrasts was read.

### 2.11 Dependencies (board `DEPENDENCIES.md`, orchestrator rulings 2026-10-03)

- **Proceed now (wave 2 build):** F1-F4 (the flat incumbent; G0(c)'s persistence; no R2, no G6).
- **F5 waits on G6's staged-bout runner** (`instruments/staged_sleep_bout.py`, G6 P4; its BUILD does not wait on R3's
  verdicts -- ruling (2)).
- **F6 waits on R2's reader** (`substrate/learner_events.read_learner_events`), whose signature R2 sends if it changes.
- **F7 waits on R2's added percept-transition part (ruling (1)) and on G3's P2** (the lifecycle refactor with its
  byte-identity proof; `silence_units` + matched picker in `growth/candidate_lifecycle.py`).
- **R-R3's restart clause for structural state waits on R2's r2.v1 checkpoint** carrying the rewirer state (1.8).
- G6's deciding tests wait on R3's campaign (G6 reads G-R6 (e)'s reds from it).

### 2.12 Cost at real scale (MEASURED units from 1; everything else a PROJECTION)

- One interaction-learner arm-seed: build 2.19 s + teach < 0.02 s + `_enter_sleep` (unmeasured) + n x 0.03 s + close +
  a fresh process (import 0.49 s + build 2.19 s + 20-item evaluation) ~ 6-7 s at the medium budget, ~ 10 s at extended
  **[PROJECTION]**.
- F1 7 arms x 8 = 56 units ~ 6.5 min; F2 2 x 8 ~ 2 min; F3 2 x 8 (+30 constructed stores each) ~ 2 min; F4 4 x 8 ~ 5 min:
  the F1-F4 confirmatory campaign ~ 16 min in ONE process at a time; the dev rehearsal the same again; reds < 2 min
  **[PROJECTION]**.
- F7: 600 percepts + 20 refits took 2.58 s, so A then B at ~1500 percepts each ~ 13 s per arm-seed; 8 arms x 8 ~
  14 min **[PROJECTION]**. F6 ~ F1's scale.
- Host load: sequential, one process, never beside a full pytest file; no live seeds.

### 2.13 Arc split and the files each part touches (claims on the board hold the new paths)

- **P0** this file (the orchestrator freezes it; its dated PLAN-FINAL note is the orchestrator's).
- **P1** NEW `src/snn_data_lake/instruments/sleep_plasticity_ab.py` (PS(s) builder + digest, the F1-F4 arms, the
  ledger, the decision tables, checks K1-K8 and K11-K13 with their reds), NEW `tests/test_sleep_plasticity_ab.py`;
  registrations claimed at build time (hot files, message holders first): `gates/dormancy_gate.py` EXEMPT row
  (benchmark harness, never on the live tick), `src/snn_data_lake/__init__.py` REGIONS if instruments are listed
  there. No edit to `engine/brain.py`, `readout/interaction_learning.py` or `growth/`.
- **P2** the F1-F4 dev rehearsal, then the confirmatory campaign on one clean head; section 3 of this file.
- **P3** (after R2's reader) NEW `src/snn_data_lake/instruments/replay_growth_composition.py` (F6 arms; imports R2's
  reader, writes none), NEW `tests/test_replay_growth_composition.py`.
- **P4** (after R2's percept part and G3's P2) EDIT `growth/neurogenesis_engine.py` (`replay_rows` only, in the window
  after R2's commit and before G7's; G3 and R2 review; G3's fingerprint test re-run),
  the F7 arms + C18 in `replay_growth_composition.py`; if F7 is MET, the live consumer -- one call in
  `Brain._sleep_tick` (`engine/brain.py`, HOT; claim and message R2 / G3 / G6 first) -- in the same arc, so the composed
  replay is never built-and-uncharged (P3); LifelongConsolidator's retirement per C18 (owner told).
- **P5** (after G6's runner) F5 arms in `sleep_plasticity_ab.py`; check K10.
- Each arc: census after symbol-adding commits, the Codex gpt-6.1-sol panel (orchestrator), the fold stated as re-panelled
  or battery-verified; the orchestrator runs the one battery.

### 2.14 Scope and what is not claimed

Not claimed by any reading here: biological equivalence of NREM/REM-named stages; that more sleep computation improves
the brain; fluid intelligence; a growth gain from a unit count; brain-wide eligibility from a cycle count (G6's
participation census is the evidence for that). The workload is the learner's own in-tree corpus (small: quantum 0.05 on
E_all); a larger constructed intent workload is a problem instrument and is NOT built here without G2 (O5).

### 2.15 Amendments after the lead read (2026-10-03) -- they supersede the text they name

Source: R3's self-defects 1-7 (board `p0_reports\R3.md`), the lead read L1-L7 of 2026-10-03 and orchestrator rulings 9
and 9a (board `DEPENDENCIES.md`); P-6 and P-8 also carry Codex findings 27 and 21 (2.16). Each item names the paragraph it replaces; where an item and the older text disagree, the item
governs. Nothing here has been run; every number is a static reading or a **[PROJECTION]**.

**P-1 (self-defect 1) Check names.** The checks of 2.7 and the contrasts of 2.4 both used C1-C15 ("C14" and "C15" meant
both). The checks are renamed K1-K15, each keeping its number (2.7 C1 -> K1, ..., C14 safety -> K14, C15 F7 row-count ->
K15); every "2.7 Cn" reference in 2.2, 2.3, 2.7, 2.13 and 4 has been renamed Kn in place. Contrast names C1-C18 and C-IF / C-SENS / C-NI are
unchanged. 2.3.2's "Check C15" is K15; its "C14 would then read consumption as growth" is the CONTRAST C14.

**P-2 (self-defect 2) Where the fork digest is taken (replaces K1's first clause).** ORACLE (wake re-teach), MASSED (wake
re-drive), F3 (30 constructed stores) and A-ONLY (session A only) differ from the shared state BY DESIGN before their
bouts, so a bout-start digest would red on them every time. K1 now reads:
- `PreSleepDigest` D0 is taken at the END of the shared PS(s) build -- after session B's last `learn_turn`, before any
  arm-specific action -- and every arm of one seed that starts from PS(s) must carry the same D0.
- A-ONLY builds PS_A(s) (session A only, same seed rule): its own D0_A, checked by a second A-ONLY build (twin equality).
- A second digest D1 is taken at bout start and printed for every arm; D1 equality is REQUIRED only across arms whose
  pre-bout actions are identical (F1 ON / OFF / R / BF / Rec; F2 S-ON / S-OFF; F3 AT-ON / AT-OFF; F4 SHORT / MED / EXT).
- Red unchanged: one arm built with the split RNG offset by 1 must fail D0 equality.

**P-3 (self-defect 3; lead L1, REQUIRED form) The safety check's red (replaces K14's red).** No planted red ever builds a
Brain outside a throwaway home. The live-lake check is a function of a path, `live_lake_untouched(path, before)`, which
reads every mtime under `path` before and after a run. Production use passes `C:\Users\vinhl\.snn_data_lake`, read
before and after every run of every arm. The red passes a DECOY directory created under the scratch area, touches one
file in it between the two reads, and must report the change; no Brain, no `make_brain`, no import of the engine. The
rest of K14 stands (every Brain under a fresh throwaway `NORD_HOME` or `substrate.home.throwaway_home()`; the daemon is
never started).

**P-4 (self-defect 4) Derived seed range (replaces 2.10's 16500-16799 row).** The F7 derived stream seeds of the reserve
sets reach 16300 + 540 + 47 = 16887, so the derived range is **16500-16899** (dev 16000-16007 -> 16500-16547;
confirmatory 16210-16217 -> 16710-16757; reserves 16300-16347 -> 16800-16887). It touches no primary set and stays below
the reds' 16900-16999. K12's seed rule is unchanged.

**P-5 (self-defect 5; lead L2) F3 is an in-process pattern-completion reading (replaces C6's row and F3's table).**
Static reading at `e1b3ae3c`: `_consolidation_acc = self.attractor.accuracy(self._patterns, 100)` (`engine/brain.py`
3432-3433) scores recall of the 3 STORED patterns from 12 partial cues each (36 trials, quantum 1/36 = 0.0278), with the
cue seed fixed at 100, and is computed on each sleep tick AFTER `downscale_prune()`. It is not a held-out endpoint.
- C6's endpoint: per seed, the mean of `_consolidation_acc` over the bout's sleep ticks (the brain.py note's own
  "average across nights" power lever), in process; d_s = AT-ON - AT-OFF; 2.4's statistic.
- F3 makes NO R-R3 claim: no held-out, retention or restart reading. Whether the attractor's weights survive a restore is
  not read here, and K5 does not apply to F3.
- F3 MET licenses only: "attractor sleep learning raises in-process pattern completion of its stored patterns at the
  medium budget". F3 NULL is reported beside G6(a)'s `downscale_prune` census as before.
- Disclosed: AT-OFF still runs `downscale_prune()` (both arms prune identically), so C6 reads learning composed with an
  identical prune; the prune's own verdict is G6(a)'s.

**P-6 (self-defect 6; lead L3) Claims and multiplicity (adds to 2.4 and 2.5).** 18 contrasts are registered, each read
one-sided at t > 1.895 (exact 2^8 sign-flip FPR printed). A family claims ONLY through its pre-designated primary:
| family | claim made only through |
|---|---|
| F1 | C1, with C1-NI; gated by C-SENS (REFUSED if NOT MET) |
| F2 | C5, with its NI; the turnover rule first |
| F3 | C6 (P-5's in-process reading) |
| F4 | the C7 -> C8 chain (each step with its NI), then C9 for the "sleep-duration" wording |
| F5 | C10; C11 read only if C10 MET |
| F6 | C12, with its NI on E_B |
| F7 | the conjunction C14 AND C15 AND C16 AND C17 (+ its NI) plus restore equality; C18 decides P2 only |
- C2 only chooses the WORDING of an F1 MET; C3 / C4 (per-item attribution) and C13 (interleave vs uniform) are REPORTED,
  never a separate claim. C-IF and C-SENS gate readings and are never claims.
- No pooling across families and no family-wise correction is claimed: each family's claim carries its own one-sided
  error rate, and a conjunction (F4, F7) is stricter than any of its parts.
- (Codex finding 27) The census is derived mechanically from the list of test IDs actually read, printed by category:
  superiority contrasts (of C1-C18), positive controls (C-IF, C-SENS), NI bars (C1-NI, C5-NI, C7-NI, C8-NI, C12-NI,
  C17-NI, F7-NI), each with its verdict; then "claims made: m" with the claim list. "18" names only the numbered
  superiority contrasts, never the total number of tests read.

**P-7 (self-defect 7) Cost (adds to 2.12).** `_enter_sleep` runs `LifelongConsolidator.learn_task` on synthetic tasks
(1.5) and its time is UNMEASURED, so 2.12's per-arm-seed figures are incomplete. The dev rehearsal (16000-16007, after
"campaign DONE") prints wall per phase (build, teach, `_enter_sleep`, bout, close, restore, evaluation) per arm-seed;
2.12 is restated from those numbers before the lead schedules the confirmatory run (lead L5: the F1-F4 campaign is one
single process the lead schedules).

**P-8 (rulings 9 and 9a; lead L7) Degenerate contrasts (replaces 2.4's degenerate sentence).**
- **Superiority contrasts (ruling 9):** every contrast that claims a gain (C1-C18, C-IF, C-SENS) reads a zero-variance
  (degenerate) d as NOT MET, reported "degenerate" with its common value. This is STRICTER than
  `paired_contrast.contrast`'s incumbent rule (a zero-variance sample is MET iff its common value > 0) and is disclosed
  as such; G1(c) 2.3 is the precedent.
- **Non-inferiority bars (ruling 9a):** every NI reading (C1-NI, C5's NI, C7 / C8's NI, C12's and C17's NI on E_B, F7's
  NI) reads a degenerate d' by its common value: > 0 -> MET (WITHIN BOUND; a lossless arm gives a constant d' = +eps),
  <= 0 -> NOT MET, with "degenerate" printed beside it. This is the incumbent `contrast` rule, applied to NI bars only,
  and is disclosed as such.
- **INTERFERENCE RED** = NI NOT MET: a degenerate common d' <= 0, or a non-degenerate t at or below 1.895. 2.4's
  boundary sentence stands (an arm that loses exactly one A item on every seed gives degenerate d' = 0 -> RED).
- The instrument applies both rules in a wrapper around `contrast` (superiority: sd == 0 -> NOT MET; NI: `contrast`
  unchanged), without editing `paired_contrast.py`; the wrapper's two branches each carry a constructed input in K11.
- (Codex finding 21) `contrast` reports FPR = 0 for a degenerate sample without enumerating flips, so the printed
  sign-flip rate is not the operative one. The wrapper ENUMERATES all 2^8 = 256 sign flips of d (or d') and applies the
  ACTUAL verdict rule (superiority: ruling 9; NI: ruling 9a) to each; the operative FPR is the fraction of flips that
  read MET. (Eight equal positive differences: a 7+/1- flip gives t = 3.0 > 1.895 and reads MET, so that rate is not
  0.) `contrast`'s own rate is printed beside it, labelled "reference (incumbent t-test rule)". K11 adds two reporting
  fixtures with hand-computed expected rates: constant positive and constant zero.

**P-9 (lead L4-L6).** C-SENS stays the guard against the small corpus (E_all quantum 0.05): C-SENS NOT MET -> REFUSED, as
in 2.5. O5 (a larger intent workload) and O1-O6 go to the owner's batch through the lead. The F1-F4 build starts only
after the lead's "campaign DONE".

### 2.16 Amendments after the Codex critique (gpt-6.1-sol, 2026-10-04) -- they supersede the text they name

R3's findings in the critique are 5, 6 and 7 (BLOCKERS, all on F7), 15-21 (MAJOR) and 27 (MINOR). 21 and 27 are folded
into P-8 and P-6 above. Every finding is ACCEPTED; none is refuted. Where a finding's code claim was checkable by a
static read at `e1b3ae3c`, it was re-read in this worktree and is cited.

**X-F7 (findings 5, 6, 7 and 18) F7's workload, splits and growth-OFF control are DEFERRED to a dated pre-data F7
amendment.** F7 already waits on R2 P3 and G3 P2 (2.11). Its concrete construction (task definitions, stream and split
seeds, the admission interface, the growth-OFF mechanism) is therefore frozen in a separate amendment to this file. That
amendment is written after those upstream results land and BEFORE any F7 data, dev included, and goes through the same
lead read and Codex critique. What the amendment must satisfy is frozen now; an amendment that cannot satisfy an item
withdraws the claim that item protects, and says so.
- **R7a (finding 5) Evaluation is a fresh draw of the SAME task.** CONFIRMED by a static read:
  `growth/neurogenesis_engine.py::_markov_matrix(k_modes, peak, seed)` seeds the preferred-successor permutation, so a
  stream with another seed is another task. 2.3.2's "stream seeds s+520 for A and s+540 for B" are WITHDRAWN as
  evaluation sessions. Each task is a frozen definition (transition matrix + percept codebook), seeded apart from the
  trajectory RNG. Evaluation and admission sessions are new trajectories from that same definition. If the incumbent
  stream cannot separate the two seeds, the instrument builds its own constructed stream: P1 case, disclosed, never
  an edit to `neurogenesis_engine.py` beyond `replay_rows`.
- **R7b (finding 7) A and B must be jointly solvable.** CONFIRMED: `_mode_features` places mode i at band centre
  (i + 0.5) / k with no task term, so k = 5 and k = 8 centres interleave in one band and their jitter clouds can
  overlap. 2.3.2's "retention-compatible" claim is WITHDRAWN. The amendment builds disjoint percept support OR an
  explicit task-context input. Before confirmation, an independent task oracle on dev seeds (a classifier given the
  true codebooks and transition matrices, which is not the method) must solve A u B jointly at a level frozen in the
  amendment; otherwise F7 is REFUSED (ambiguous workload) and nothing is read as forgetting.
- **R7c (finding 6) Untouched admission data, by source ID.** CONFIRMED: `observe` lets the live `OnlineReadout("delta")`
  learn on wake, so a buffered row has already moved the readout before any refit. Excluding replay rows from
  `marginal_holdout` therefore does not make that slice untouched. Every generated item gets an immutable source ID and
  split at GENERATION, before any `observe`, deposit or replay: TRAIN (observed; replay-eligible), ADMIT (scores unit
  admission only; never observed, deposited or replayed), JUDGE (final evaluation only).
  - K9 is widened from list membership to source IDs: no TRAIN or replayed ID may appear in ADMIT or JUDGE.
  - New red: plant one previously observed or replayed ID into ADMIT; K9 must fire.
  - If the incumbent's admission scoring cannot be pointed at ADMIT without a second refit path, C16 reads retention on
    JUDGE only, and the claim "units earn admission on untouched data" is NOT made (disclosed).
- **R7d (finding 18) Growth OFF changes only additions.** CONFIRMED: `growth/ng_snn.py` 437,
  `n_val = 0 if self.l_max is not None else max(8, int(0.2 * N))`, so pinning `l_max` also switches off NGSNN's internal
  validation split. 2.3.2's pinned-`l_max` growth-OFF arm is WITHDRAWN as specified. ON, OFF and PRESIZED must run the
  same fit/validation partitions and the same stopping protocol, and OFF may disable only unit additions. K15 is widened
  from equal consumed-row COUNTS to equal consumed source IDs with equal roles (fit / validation / admission). If no
  mechanism meets this without a second fit path, C14 is withdrawn and the F7 conjunction is restated in the amendment,
  before data.

**X-15 (finding 15) Deciding tests wait on accepted upstream RESULTS, not on code landing (replaces 2.11's F6 / F7 /
restart bullets).** Builds may proceed earlier, but:
- F6's confirmatory run waits on R2's P1 hydration results, ACCEPTED in R2's record (its R-R2 checks c1-c5).
- F2's restart reading waits on the accepted r2.v1 rewirer-state carrier.
- F7 waits on R2's P3 hydration results, ACCEPTED (R2-c7), on G3's P2 landing and on the X-F7 amendment.
- If R2 later nulls a hydration result, every R3 reading that consumed it is withdrawn.
- F5 still waits only on G6's runner BUILD (ruling 2), never on G6's verdicts.

**X-16 (finding 16) F6's two tasks are two real boots (replaces 2.3.1's teaching sentence).** R2's `session` is a boot
counter, so sessions A and B taught in one process share one session ID, and HIST-I could not separate them. F6
therefore builds PS_F6(s) across real boots, each a fresh process on the arm's scratch home:
1. boot 1: `make_brain(s)`, teach A, `close()`;
2. boot 2: restore, teach B (R = 6), `close()`;
3. boot 3: restore, bout, `close()`;
4. boot 4: restore, evaluate.
The instrument keeps its own map session ID -> task (A, B) from the boot that taught it. That map is never written to
the lake or the schema, so R2's boot counter carries no task semantics.
- New check K16: the A boot's and the B boot's session IDs differ, and every drawn event maps to exactly one task. Red:
  teach A and B in one boot; K16 must fire.
- Disclosed: PS_F6 has two more restarts than F1's PS(s), so F6 numbers are never compared with F1's.
- The session counter's definition is R2's (R2 finding 13); F6 reads whatever R2 freezes and K16 checks the result.

**X-17 (finding 17) K8 split into a local check and a boundary check (replaces K8).** `_enter_wake` increments
`_nights` (`engine/brain.py` 3876, "a completed subjective night"), so "`_nights` unchanged during a bout" would have
halted every valid arm.
- K8a, local: around every replay call, every hydration, every per-tick instrument action and every fwd draw,
  `_step`, `_nights`, `_wake_ticks`, the grower's `observed` / `live_n` and the arm command count are unchanged. Red:
  bump `_step` once inside one replay call and feed the grower one `observe()`.
- K8b, boundary: from `_enter_sleep` to the end of `_enter_wake`, `_nights` rises by exactly 1, `_wake_ticks` and the arm
  command count are unchanged, and `_step` is printed. Red: call `_enter_wake` twice.
- Disclosed: `_enter_wake` also runs its periodic persistence and self-build on night-count schedules (brain.py 3685,
  3726, 3881). Every arm of one family runs the same single night from the same `_nights`, so those run identically
  across arms. They are part of the bout and not attributed to any arm.

**X-19 (finding 19) A NOT MET is never an affirmative explanation (replaces the C2 and C9 rows of 2.4, the F4 sentence
of 2.5 and the F4 bullet of 2.8).**
- C2 NOT MET reads "no benefit over recency-only replay demonstrated", not "recency-sharpening only".
- C9 NOT MET reads "no sleep-duration benefit over massed wake demonstrated"; EXT's gain is then not credited to
  sleep. It is not called "extra updates alone".
- An affirmative explanation would need its own registered contrast and control, and none is registered here.

**X-20 (finding 20) One joint F7 / P2 table (replaces C18's disposition in 2.6 and 2.8). The first row that applies
decides:**
1. Any check red -> HALT.
2. The F7 acceptance conjunction (the X-F7 amendment's restatement of C14 AND C15 AND C16 AND C17 + its NI + restore
   equality) is NOT all MET -> F7 NULL. R3's composition code is deleted (P8 5); `LifelongConsolidator` is RETAINED
   whatever C18 reads, and C18 is reported only.
3. F7 MET and C18 MET -> ADOPT (P2 route 1). The composed path becomes THE path, with its one live consumer in
   `Brain._sleep_tick` in the same arc (P3). `LifelongConsolidator`'s live instance is retired; its unique capability,
   the interleave and the oldest-task probe, is already ported as HIST-I and E_A. The owner is told (O4).
4. F7 MET and C18 NOT MET -> NOT ADOPTED. Superiority is not shown, so the incumbent stays. R3 lands no live consumer
   and deletes its composition code rather than parking it (P3). A MERGE is put to the owner only as a proposal (O4).
At most one mechanism is live after any row.

### 2.17 Review record

- **Lead read:** 2026-10-03, board `p0_reports\R3.md` section "LEAD READ" (L1-L7), plus orchestrator rulings 9 and 9a
  (board `DEPENDENCIES.md`); folded as 2.15.
- **Advisor read:** CLI lead session, Fable, 2026-10-03; it reached R3 through the lead read.
- **Codex critique:** model gpt-6.1-sol, 2026-10-04. Reply file
  `<lead scratchpad>\codex_p0\b3.reply.txt`, sha256
  `1c8abc4a99bfa18c7e1d97d042bb1240c8b4cf8f893f88277bec5c38fae52e57` (re-hashed by R3 before this fold). R3's
  findings: BLOCKERS 5, 6, 7 -> X-F7 (deferred to the dated pre-data F7 amendment, with requirements R7a-R7c frozen
  now); MAJOR 15 -> X-15; 16 -> X-16; 17 -> X-17; 18 -> X-F7 R7d; 19 -> X-19; 20 -> X-20; 21 -> P-8; MINOR 27 -> P-6.
  All accepted, none refuted.
- Not yet reviewed: the X-F7 amendment itself, which does not exist yet and gets its own lead read and Codex critique
  before any F7 data.

## 3. Open questions for the owner

- **O1.** The 2026-09-27 sleep requirement: R3 proposes it is never deleted by a null (like an owner-drawn part, P4d),
  while a specific incumbent cap that misses its neuter-A/B on held-out capability is a P3 retire-or-exempt decision
  that is yours. Accept?
- **O2.** The percept-transition record (assigned to R2 by the orchestrator): every wake percept is a ~64-float vector,
  a high-rate stream; its retention class and append budget need your call before it is journaled.
- **O3.** The structural slow clock reads turnover 0 on three rows' schedules; G6's Q4 asks retire vs dated exemption.
  R3 cites it rather than asking twice.
- **O4.** LifelongConsolidator: if the ported composition beats its policy (C18) it is retired; if it does not, R3 lands
  no second composition. Confirm one survives either way.
- **O5.** The in-tree corpus gives E_all quantum 0.05 (20 items). Accept it for F1-F4, or authorize a constructed larger
  intent workload (G2 builds problem instruments)?
- **O6.** Budgets in sleep ticks (10 / 40 / 160), not human hours. Accept?

## 4. Debate log (wave 1; each challenge, its sender, the answer, the resolution)

### 4.1 R2 (a97af13910daffbad)

- R3 asked: which records R2 hydrates; the reader's signature; replay provenance. R2: interaction-learner turns only
  (kinds `teach`, `turn`; fields kind, event_id, stream, session, seq, source, target, provenance, encoder, schema,
  valid, split); one reader `read_learner_events(kinds, *, path=None, include_sessions=None, exclude_sessions=None) ->
  (events, refusals)` in `substrate/learner_events.py` (R2 claims it); replay writes no lake event, ring entries carry
  event_ids, replays are counted per event_id and the last tick's drawn ids are exposed; hydration and replay advance no
  chronology. RESOLVED: R3 imports the reader and writes none; F6 joins drawn ids against its held-out sessions.
- R2 raised: R-R3's restart clause needs restore exact through the sleep tick; `Brain._rewirer` is process-local.
  R2 MEASURED (seed 15000) equality on a turnover-0 schedule and a constructed red O_rw; R2 adds the rewirer state to
  r2.v1. RESOLVED: R3's check K5 cites O_rw as its red source; F2's restart reading waits on r2.v1 (2.11).
- R3 told R2 the A trace (1.000 -> 0.250 -> 0.500) so F6's eviction design (B > 64 turns) is visible to R2.
- R2 (after the orchestrator assigned it the percept record) sent its draft percept schema (2.3.2). R3 ADOPTS it and
  raised two challenges: (a) `n_protos` eligibility is defined for the arm's GrowingLearner; F7's synthetic streams need
  an instrument encoder/provenance that R2's eligibility accepts only in a scratch lake (else F7 must bypass the reader,
  which would make a second path); (b) three rows edit `neurogenesis_engine.py`. R2 CONCEDED (a) with a better
  mechanism -- a consumer flag `allow_instrument=False` by default, refused on the production home with a ValueError,
  never passed by the Brain's hydration, red = a planted instrument percept refused by hydration -- and the
  arm-vocabulary check applies only to encoder `arm.percept gained width=64`. (b) the order is R2 -> R3 -> G7 (G3's P2
  no longer touches the file); R2 releases its claim at its commit. R2 CONFIRMED its hydration fills `_buf_x/_buf_y`
  and `_categories` but not `_prev`, `_prev_cat`, `observed`, `live_n` or `live_correct`, so no cross-restart pair
  exists. RESOLVED.
- R3 asked R2 (advisor point, after the draft): what is R2's DEFAULT ring hydration (which eligible turns fill the 64
  slots after a restart)? R2: RECENCY -- the last 64 eligible teach/turn events in lake order across sessions,
  rebuilt to equal an uninterrupted process's ring (post-watermark events re-applied through `learn_turn`), labels
  rewritten to the latest authority, deterministic with no RNG; held-out and invalid events never enter it; any other
  policy is R3's to register. RESOLVED: F6's RING arm is that default and C12 reads what it was designed to read.
- R2 (2026-10-04, after its own Codex fold) sent its FROZEN reader:
  `read_learner_events(kinds, *, path=None, include_sessions=None, exclude_sessions=None, allow_instrument=False) ->
  (events, refusals)` (`path=None` = the brain's home lake; `allow_instrument=True` raises ValueError on the production
  home). Percept pairing changed: each `percept` event carries `pseq`, a per-session counter advanced on every write
  attempt. A pair is admitted only when both endpoints are eligible, share one session and are pseq-adjacent; a
  refused, held-out or failed-write endpoint breaks the chain. R2's checkpoint cursor now covers teach/turn/sleep events
  with one shared seq. A failed event write refuses that turn's deposit. R2 accepts Codex 15's gating of F6/F7 on its
  ACCEPTED results. RESOLVED: F6 and F7 call this signature. The X-F7 amendment forms F7 pairs ONLY through R2's
  `hydrate_transitions` or R2's exact pseq-adjacency rule, never a second pairing rule (P2), and its K9 red set includes
  an interposed held-out percept that must yield no bridging pair.
  (Amended 2026-10-04: the `hydrate_transitions` citation -- see "DATED PRE-DATA AMENDMENT 2026-10-04" at the end.)

### 4.2 G3 (aab6003013a8b76f6)

- G3 asked (a)-(d). R3: F1-F4 touch nothing in growth/; F7 needs `replay_rows(rows, *, source_ids) -> int` on
  NeurogenesisSelfImprove. G3: its P2 refactor lands FIRST with its byte-identity proof; R3 adds `replay_rows` after it
  as R3's own stacked commit that G3 reviews; G3 adds nothing to NeurogenesisSelfImprove's public surface; `growth_loop`
  does NOT supersede `_refit` (the percept consumer's entry), so R3's target does not move. RESOLVED.
- Silencing helper: `growth/candidate_lifecycle.py::silence_units(consumer, unit_ids)` + a matched-control picker;
  R3 imports them (C16). RESOLVED (signature sent when G3's P2 lands).
- Seeds: no overlap (G3 9000-9999). RESOLVED.
- G3 then changed its P2 to touch NO line of `neurogenesis_engine.py` (the lifecycle routes through `ng_snn.py` +
  `candidate_lifecycle.py`); G3 only reviews the R2 -> R3 -> G7 windows, on the condition that each stacked commit
  re-runs G3's fingerprint test (shipped in G3's P2) and R2's hydration is inert with no lake. ACCEPTED.
- R3 told G3 (advisor point): F7's growth-OFF arm pins the existing `l_max` at the post-A count rather than holding
  the gate closed, so it still re-solves beta on replay rows; no new G3 interface is needed.
- R3 challenge 1 (LifelongConsolidator as a P2 incumbent): G3 holds it is a different job for G3 (prototype growth, a
  capacity ceiling, not a spend budget) and defers the adjudication to R3; adopts its retention-probe idiom only if R3's
  adjudication keeps it. RESOLVED: R3 owns the adjudication (C18; orchestrator ruling concurs).
- R3 challenge 2 (C9's equal-count control could include dead units): G3 CONCEDED -- the control is matched on
  output-synapse L2 norm and firing on the evaluated items, dead units excluded, a too-small pool halts. RESOLVED; R3's
  C16 uses the same rule.

### 4.3 G6 (a37039ad149d21ac3)

- G6 collision 1 (R3's 2026-09-27 note and G6(e) list the same arms). Resolution, accepted by both and by the
  orchestrator (ruling (3)): ONE sleep campaign, R3-owned, in `sleep_plasticity_ab.py`; G6 adds the eligible-brain
  structural arm, the participation census and the fabrication guard, and reads G-R6 (e)'s reds from it. R3's challenge:
  G6's arm must take PS(s) from R3's fixture -- G6 ACCEPTED for its eligible-brain arm; its solved-problem arm takes
  its state from G3's loop (outside this fixture). RESOLVED.
- G6 collision 2 (StructuralRewiring is G6's pruner and R3's structural incumbent): R3 owns the verdict and its turnover
  rule; G6 adopts R3's detector and puts retire-vs-dated-exemption to the owner (G6 Q4). RESOLVED.
- G6 collision 3 (the staged-bout runner): G6 builds it at its P4 (`instruments/staged_sleep_bout.py`, claimed by G6)
  with R3's interface `run_bout(brain, stages, *, rng_seed) -> BoutReport`; its BUILD does not wait on R3's verdicts
  (orchestrator ruling (2)). RESOLVED.
- G6 asked: (i) how PS(s) is forked -- answered in 2.2 (rebuild-from-seed in a fresh process and home, verified by
  `PreSleepDigest`; `_rewirer` is None at every bout start); (ii) the compute unit -- PRESENTATIONS (2.2), G6 converts its
  mask evaluations. G6 proposed 'nrem' = replay of eligible lake items through the real learner and 'rem' =
  integration/refinement over already-replayed eligible items, recombinations marked SYNTHETIC -- R3 ACCEPTS, adding
  check K10. (Answers sent; awaiting G6's acknowledgement.)
- G6 asked (after the orchestrator's LifelongConsolidator ruling): does R3's ported arm vary
  `LifelongConsolidator._prune_to_budget` (G6(a)'s pruner P-LL)? R3: NO. HIST-I ports only the interleave and the
  oldest-task retention probe onto the NG-SNN grower (whose prune is `prune_unearned`, also in G6(a)'s census); C18 runs
  LifelongConsolidator's whole policy, prune ON as it stands, as a comparator. So G6(a) keeps P-LL's prune A/B (ON / OFF
  / random on `_ll_retention`) in its own census instrument -- a different question (does the prune earn its place) from
  C18's (does the composed path beat the incumbent policy). If C18 later retires the module, P-LL's verdict stands as its
  record. (Answer sent.)
- G6 ACKNOWLEDGED all of the above and recorded it (G6 debate rows D1-D5b), and found one more overlap: G6's
  harness-alive positive control also neuters `AttractorNet.sleep` on G6's own workload. Resolution: it stays in G6's
  harness as a CONTROL on G6's outcome, never a verdict on `AttractorNet.sleep`; the verdict is F3's (2.3). RESOLVED;
  G6 closed its round.

### 4.4 main (orchestrator)

- R3 reported: the unassigned percept record, the G6/R3 cycle, the plan duplication, the LifelongConsolidator P2 finding.
  Rulings (2026-10-03): (1) percept record ASSIGNED TO R2; (2) G6's runner build does not wait; R3's staged arms wait on
  it, budget arms run now; (3) one campaign, R3-owned, recorded by the orchestrator in PLAN-FINAL as a dated note; the
  LifelongConsolidator port is registered as an arm, one survives. RESOLVED.

### 4.5 Unresolved

- None between rows: every challenge sent or received in wave 1 was resolved (4.1-4.4).
- Open for the owner: O1-O6.

## 5. Results

Empty until the frozen campaign runs.

## DATED PRE-DATA AMENDMENT 2026-10-04 (citation of R2 P3's hydrate_transitions; no data read)

Written after the P0 freeze (`f83c7c74`) and before any R3 data, dev included; no run of any kind preceded it.
Source: R2's message of 2026-10-04 (R2's frozen text fixes only the function's name and input).
- 4.1's last row cites "R2's `hydrate_transitions`". The citation is now **"R2 P3's `hydrate_transitions(events)`, R2
  2.10"**: a pure function over the output of `read_learner_events(("percept",), ...)`. Its return value, and whether it
  is a method or a free function, are fixed at R2's P3 build. R2 sends them, with any dated amendment of its own, before
  the X-F7 amendment cites them; X-F7 cites the exact form.
- Nothing else changes: F7 forms pairs only through that function or R2's exact pseq-adjacency rule, and the
  interposed-held-out red stands.

## DATED PRE-DATA AMENDMENT 2026-10-04 (F3's store count W; approved by the lead the same day)

Written before any F3 CONFIRMATORY seed (16120-16127) has run. DISCLOSED: dev data WERE read before this amendment.
The F1-F4 dev rehearsal (16000-16007, EXPOSED, P1 `c8f33213`'s instrument) is what showed the defect it repairs.
- **The defect (dev, MEASURED).** At W = 30 constructed stores, F3's attractor sits at ceiling. AT-OFF and AT-NONE
  read 36/36 on 8/8 dev seeds, and AT-ON reads 0.989-1.000 (per-seed means of `_consolidation_acc` over the bout).
  C6 = AT-ON - AT-OFF therefore can never read MET (d <= 0 by construction): the MET row of F3's table is unreachable,
  and a null would describe the ceiling, not the attractor. 2.6's historical positive control (neutering
  `AttractorNet.sleep` pinned `_consolidation_acc` at 0.333 on 7/7 seeds, brain.py's 2026-08-05 record) did NOT
  reproduce at W = 30: AT-NONE read 1.000. That record came from natural wake-tick storage over ~37 nights, not from W
  back-to-back constructed stores.
- **The repair: W is chosen on DEV seeds by a frozen rule.** W = the smallest value in the ladder {1, 2, 3, 6, 12} at
  which AT-NONE (the dev-only positive control, attractor sleep removed) reads below 35/36 on ALL 8 dev seeds
  16000-16007. The ladder runs only AT-NONE, on dev seeds only, in ascending order, and stops at the first W that
  qualifies. W is then frozen in this file by a dated note, before any F3 confirmatory seed runs. Every F3 arm (AT-ON,
  AT-OFF) uses that W. The stores stay the wake tick's own two lines, `_patterns[i % 3]`, `reps=1`.
- **If no W in the ladder qualifies,** F3 is REFUSED (blind harness) and reported as such. The ladder is not extended
  after the fact.
- **Nothing else changes:** C6's endpoint (P-5), its statistic (P-8), K13's compute match, F3's table and seeds
  16120-16127. Choosing W on AT-NONE never reads AT-ON or AT-OFF, so the confirmatory contrast C6 is not tuned on its
  own outcome.

## DATED PRE-DATA AMENDMENT 2026-10-06 (F3's W FROZEN by the dev ladder)

Written before any F3 CONFIRMATORY seed (16120-16127) has run; additive (a zero-removed-lines diff).
- **Ladder result (MEASURED, 2026-10-06, `claude/r3` fb7e3a0f's instrument, one process, dev seeds 16000-16007, AT-NONE only).**
  W = 1 is the first rung of {1, 2, 3, 6, 12}; the per-seed mean `_consolidation_acc` was 0.3333 on all 8 seeds, below
  35/36, so the rung QUALIFIES and the ladder STOPPED (no later rung was run). Result file `f3_ladder.json`, sha256
  `3c965479794534e302858dfa2886c6875df9d415e4c534d196478feef3a2c76c` (exit 0; live lake mtime unchanged).
- **W = 1 is FROZEN** for every F3 arm (AT-ON, AT-OFF); `F3_STORES = 1` lands in the next commit. Nothing else changes.
- **DISCLOSED RISK, not resolved by this note.** 0.3333 is exactly 1/3, the 3-pattern chance level, identical on all 8
  seeds: AT-NONE may be at a floor rather than merely degraded. The frozen rule selects W on AT-NONE only and never reads
  AT-ON or AT-OFF, so whether C6 = AT-ON - AT-OFF is reachable at W = 1 (AT-ON above chance) is UNMEASURED. A confirmatory
  F3 reading is a null-or-MET reading under the pre-registered C6 table either way; an AT-ON reading on dev to check the
  ceiling-and-floor reachability would be a NEW pre-data choice, raised to the lead rather than taken.

## DATED PRE-DATA NOTE 2026-10-06 (F3's AT-ON positive control at W = 1, dev seeds; approved by the lead the same day)

Written before any F3 CONFIRMATORY seed (16120-16127) has run; additive. The lead approved reading AT-ON on the EXPOSED dev
seeds 16000-16007 at the frozen W = 1 as the positive control that C6 is reachable; it changes nothing frozen.
- **Reading (MEASURED, one process, `run --family F3 --set dev --arms AT-ON`, 8 arm-seeds, 40 sleep ticks each, live_lake_untouched
  True on all 8).** Per-seed mean `_consolidation_acc` of 36 stored items (chance = 12): 14.45, 12.00, 12.00, 14.20, 12.00,
  12.10, 13.15, 12.53 -> as fractions 0.401, 0.333, 0.333, 0.394, 0.333, 0.336, 0.365, 0.348 on seeds 16000-16007. AT-ON is
  ABOVE the AT-NONE floor (0.3333, all 8 seeds, the ladder) on 5 of 8 seeds (16000, 16003, 16005, 16006, 16007; tick maxima
  up to 20/36) and EXACTLY at the floor on 3 (16001, 16002, 16004). It is therefore NOT at chance everywhere: C6's MET row is
  REACHABLE at W = 1, but the margin is small and seed-dependent. Run log sha256 `ca4fa2393dd54a63f7dc46e1eb65ad6f1d4267e73700852e4649585af3657779`.
- **Not read here:** AT-OFF at W = 1 (the confirmatory contrast's other arm; "AT-OFF still prunes", self-defect 5). Whether
  C6 = AT-ON - AT-OFF clears t > 1.895 on 8 seeds is the confirmatory question and stays unread. No W change follows.

## DATED PRE-DATA AMENDMENT 2026-10-07 (X-F7: F7's workload, splits, admission and growth-OFF control)

Written before any F7 data of any kind (dev included) exists; additive (a zero-removed-lines diff). Reviewed: lead read and two Codex gpt-6.1-sol rounds (reply texts in the lead's scratchpad, lens-r3-amend.out and lens-r3b.out); every finding was folded, none refuted. It satisfies R7a-R7d and K15/K9 as frozen in 2.16 (K17 is new here), or withdraws the claim an item protects (stated per item). Everything not marked MEASURED is a [PROJECTION]. Upstream forms still owed to it are cited as OPEN: R2 P3's `hydrate_transitions(events)` (return value; method or free function) and R2's percept `pseq` rule; G3 P2's matched-control picker (`silence_units(consumer, unit_ids)` exists at `growth/candidate_lifecycle.py:110`; the picker does not). G3's `grown_code.admit_rows` (grown_code.py:191) is the unit-LABEL gate, not an ADMIT split, and is not used here.

### X1. Tasks are frozen definitions, not stream seeds (R7a, R7b)

- A task T = (codebook, transition matrix), built ONCE per primary seed s by an instrument-side class `TaskDefinition`
  in `instruments/replay_growth_composition.py` (a P1 case, disclosed: the incumbent `SyntheticPerceptStream` seeds the
  matrix and the trajectory from one RNG, `_markov_matrix(k, peak, seed)` and `self._chain`, so a second seed is another
  TASK, which is the R7a defect; no edit to `neurogenesis_engine.py` beyond `replay_rows`).
- Task A: k_A = 5 modes, categories 0-4. Task B: k_B = 8 modes, categories 5-12. The transition matrix is the
  incumbent `_markov_matrix(k, 0.75, seed)` (imported, not copied) with the STRING seed
  `f"r3.f7|def|{task}|{s}"` (`random.Random` accepts it; deterministic). Trajectories use separate string seeds (X2).
- **Disjoint percept support (R7b, first option).** The percept is 64 wide (the buffer's width). Task A's modes occupy
  dims 0-31 and task B's dims 32-63; the other half of the vector is exactly 0.0. Within its half, mode i sits at the
  incumbent `_mode_features` band centre (i + 0.5) / k with the incumbent seeded jitter 0.12, so a jitter cloud never
  leaves its band by more than the incumbent's own. A percept of A and a percept of B therefore differ on at least 32
  dims by construction; the mode clouds of ONE task still overlap as in the incumbent (that difficulty is what makes
  growth matter and is not changed).
- **Task oracle (R7b's gate, not the method).** A nearest-codebook decoder that is GIVEN the true codebooks and the true
  transition matrices: decode the current mode from the percept, answer the matrix row's argmax. It is built from the
  definition only and never trains. Frozen level: on dev seeds 16000-16007 with 1,000 fresh TRAIN-style rows per task,
  the oracle's accuracy on A, on B and on A u B must each be >= 0.70 (Bayes optimum = peak = 0.75; 0.70 leaves 0.05 for
  mode-decoding error) **[PROJECTION: the dev oracle run confirms; the number is frozen here, before it runs, and is not
  tuned afterwards; a miss on any dev seed -> F7 REFUSED (ambiguous workload), nothing read as forgetting]**.
- Forgetting must exist or nothing is earned. Gate, frozen here: per dev seed, in the NONE / growth-ON arm, the
  accuracy on A-JUDGE after stage 1 (A trained) minus the accuracy on A-JUDGE after stage 2 (B filled the 1000-row
  buffer) must be >= 0.10 absolute. Denominator: A-JUDGE has 599 pairs (X2), so one pair = 0.00167 and 0.10 = 60 pairs;
  at accuracy ~0.7 the binomial s.e. of one such accuracy is ~0.019, so 0.10 is ~5 s.e. of a single reading. (The
  earlier wording "two quanta at 8 items" was an arithmetic slip -- an 8-item quantum is 0.125 -- and is not used.) If the
  gate fails on any dev seed: "retention has nothing to protect", F7 REFUSED, C17 not read.

### X2. Splits by immutable source ID, assigned at generation (R7c)

- A trajectory is a seeded draw of n percepts from a definition; id `s{S}/{task}/{role}/{idx}`. The earlier "stream
  seeds s+520 / s+540" evaluation sessions are WITHDRAWN (2.16 R7a). **Frozen seed mapping:** every random stream is
  `random.Random(<string>)` with the strings `r3.f7|def|{task}|{s}` (matrix), `r3.f7|traj|{task}|{role}|{s}` (one
  trajectory per task and role, roles TRAIN / ADMIT / JUDGE), and `r3.f7|replay|{s}` (every replay draw, shared by all
  arms of one seed), where `s` is the PRIMARY seed (so K12's block rule applies to it unchanged) and task is A or B.
  Six trajectory streams and two definition streams per primary seed; string namespaces make a cross-role collision
  impossible by construction, so 2.10's reserved derived range 16500-16899 is simply unused by F7 (disclosed; no
  numeric derived seeds exist). The build prints all eight strings per seed and the instrument asserts the set has
  eight distinct members.
- **Frozen sizes (pairs = percepts - 1, formed inside one trajectory; counts chosen now, before any F7 run).** TRAIN: 1500
  percepts per task (1499 pairs; the incumbent's `buffer_cap` 1000 then forgets A while B trains). ADMIT: 300 percepts per
  task (299 pairs). JUDGE: 600 percepts per task (599 pairs; E quantum 1/599). The F7 non-inferiority eps of 2.4 ("one
  quantum of its A evaluation session") is therefore 1/599.
- **Frozen schedule.** Stage 1: `drive` A-TRAIN, 1500 observe+tick steps (refit_every 20, 75 refits), no replay. **Stage
  boundary (Codex round 2, finding 4; checked: `observe` keeps `_prev` across calls, `neurogenesis_engine.py` 314-343, so a
  continuous drive WOULD buffer the pair (A's last percept -> B's first category)).** The boundary is a REAL PROCESS RESTART,
  the case the engine already defines: `growth_state_bytes` is taken, a fresh process runs `restore_growth_state`, and R2's
  hydration refills the buffer and the category set from the lake; `_prev`, `_prev_cat` and the pending prediction are
  not restored, so no cross-boundary pair can form (R2 2.3.2 / debate 4.1: pairs are never formed across a restart). The
  model and the buffer are kept. This is upstream-compatible and needs no engine edit. If R2's hydration is not
  available when F7 builds, F7 waits (2.11 / X-15), it does not fall back to setting private fields. **New check K17
  (F7):** every pair the engine buffers or is handed has both endpoints in ONE trajectory (source ids), and the
  restart boundary is present; red: drive B-TRAIN continuously after A-TRAIN in one process and K17 must fire on the pair
  (A_last, B_first).
- Stage 2: `drive` B-TRAIN, 1500 steps; at every refit during stage 2 the instrument calls `replay_rows` once, immediately
  before that refit's tick, with **R = 64 rows at every refit** (the hydrated buffer holds up to 1000 A pairs at the start of stage 2, so NONE's
  draw of 64 is always supplied), identical across NONE / HIST-U / HIST-I and across growth ON / OFF / PRESIZED at
  every refit. **Draw rules, frozen, all WITHOUT replacement inside one refit's draw, RNG
  `random.Random(f"r3.f7|replay|{s}")` consumed in refit order, one stream per arm family restart-equal:**
  NONE draws 64 rows uniformly from the pairs currently in the buffer (duplicates of rows the refit already trains on).
  Eligible history for HIST-U and HIST-I = every TRAIN pair already observed at that refit (all A pairs of stage 1 + the
  B pairs observed so far in stage 2; a pair becomes eligible the tick it is buffered, never earlier; ADMIT and JUDGE
  never). HIST-U draws 64 uniformly from that pool. HIST-I draws b = min(32, number of eligible B pairs) from B and a = 64 - b from A
  (the first stage-2 refit follows 20 B percepts = 19 B pairs: b = 19, a = 45; from the second refit on, 39 or more B
  pairs exist: b = 32, a = 32; A always has >= 1499 eligible pairs, so the shortfall is always filled from A and
  a is never short);
  within each task the drawn pairs are sorted by source index ascending (within-task order kept), then merged by
  strict alternation A, B, A, B, ... starting with A, with the remainder (when a > b) appended in order. After stage 2:
  `growth_state_bytes`, a fresh process, `restore_growth_state`, then `model.predict` on A-JUDGE and B-JUDGE (readout not
  learning). These numbers are design constants, not tuned to any F7 outcome; the dev design step may report cost and
  the forgetting gate but may change a constant only by a further dated amendment. K15 compares each refit's drawn
  multiset and, for HIST-I, its order against the rules above; reds in X3 plus: (iv) HIST-I with b = 32 at the first
  refit (a draw the pool cannot supply) must fire.
- Roles. TRAIN: written through R2's `transition_sink` and observed; replay-eligible. ADMIT: scores unit admission
  ONLY; never observed, deposited or replayed. JUDGE: final evaluation only. Pairs (percept_{t-1}, category_t) form WITHIN
  a trajectory through R2 P3's `hydrate_transitions` or R2's pseq-adjacency rule (OPEN; never a second pairing rule);
  JUDGE and ADMIT items are pairs inside their own trajectory.
- **K9 (widened, 2.16 R7c):** no source ID of a TRAIN or replayed item appears in ADMIT or JUDGE; every id carries its
  role; red 1: plant one TRAIN id into ADMIT; red 2: plant one replayed id into JUDGE; red 3 (R2's): an interposed
  held-out percept yields no bridging pair. Each must fire.
- **Admission scoring.** The incumbent `_refit` scores admission on `marginal_holdout` of the OBSERVED buffer, i.e. on
  rows the live `OnlineReadout("delta")` already learned from (R7c's defect). F7 does NOT assume a pointable ADMIT
  slice. Default: C16 reads retention on JUDGE only, and the claim "units earn admission on untouched data" is NOT made
  (disclosed). That claim is made only if a mechanism exists that points the existing `_refit` admission score at ADMIT
  rows WITHOUT a second refit path; if one exists at the build, it is added by a further dated amendment before data.

### X3. Growth OFF changes only additions (R7d) and K15

- ON, OFF and PRESIZED run the same fit/validation partitions and the same stopping protocol. The incumbent pinned-`l_max`
  OFF arm is WITHDRAWN (it switches NGSNN's internal validation split off, `ng_snn.py` 437, `n_val = 0 if self.l_max is
  not None ...`).
- **OFF mechanism: a cap on additions, not on `l_max`.** [OPEN, needs G3's/the build's static read of `NGSNN.fit` growth
  loop]: the instrument wraps the growth step so that, at the A->B boundary, a candidate unit that would be ADDED is
  discarded (the add decision is read and refused), with `l_max` and the validation split untouched. If the fit
  loop's structure admits no such refusal without a second fit path, **C14 is withdrawn** and the F7 conjunction becomes
  C15 AND C16 AND C17 (+NI) AND restore equality, restated here BEFORE data; C15's PRESIZED twin (same rows, same refits,
  an NGSNN built at the ON arm's final unit count) then carries the "growth vs capacity" question alone.
- **K15 (widened, multiset form):** per refit, the three arms' consumed rows are compared as MULTISETS of
  (source id, role), role in {fit, validation}, with multiplicity (so [a, a, b] differs from [a, b, b]); where order
  matters (HIST-I's alternation, the fit/validation partition order) the ORDERED sequence is compared too; admission rows
  are compared the same way where the mechanism exists. Reds, each must fire: (i) route the OFF arm through the closed-gate
  branch (it consumes no replay rows); (ii) substitute one row for a duplicate of another in one arm ([a, a, b] against
  [a, b, b], same set, same count); (iii) swap two HIST-I draws in one arm (same multiset, different order).

### X4. The F7 acceptance conjunction (restated; 2.16 X-20 row 2 reads it)

F7 MET = C14 (if kept) AND C15 AND C16 AND C17 AND C17-NI AND restore equality (a fresh process after
`restore_growth_state`, `model.predict` on JUDGE of A and B, readout not learning), each at t > 1.895 over 8 seeds, the
P-8 degenerate rules, the 2^8 operative FPR printed. C16 additionally needs its matched pool (dead units excluded; a pool
too small to match HALTS). C18 is reported and decides P2 only through X-20 rows 3-4.

### X5. What this amendment does not decide

Refit cadence (`refit_every` 20) and `buffer_cap` (1000) are the incumbent's, inherited and not chosen; sizes, the replay
row count and the schedule are frozen in X2. HIST-I's alternation rule is the ported LifelongConsolidator interleave
(within-task order kept, two tasks never joined as a transition). F7 waits as 2.11/X-15
say: R2 P3's ACCEPTED results, G3 P2, this amendment frozen, then the dev oracle run, then confirmation (16210-16217).

## DATED AMENDMENT 2026-10-07 (E_B non-inferiority guard for F1, F2 and F4; written AFTER the exposed dev data was read)

Written after the F1-F4 dev rehearsal (16000-16007, EXPOSED) was read, and before any CONFIRMATORY seed (16100-16137) has run; additive (a zero-removed-lines diff); it ADDS a guard and removes, loosens or re-weights nothing. Reviewed: lead read and two Codex gpt-6.1-sol rounds; every finding folded. Its code half (EPS_B, C1/C5/C7/C8-NI-B, the new table rows with K11 inputs) is the commit that follows this one.

**Disclosure first.** This amendment was written after the F1-F4 dev rehearsal (seeds 16000-16007, EXPOSED) had been read
by R3, and it was prompted by that reading. The reading (MEASURED from the 144 dev records, restored-process E, paired
over 8 dev seeds): F1 ON - OFF on E_all +0.019 (t 0.704) hides R (replay only) - OFF +0.100 (t 7.48, 8/8) and BF (rest
only) - OFF -0.375 (t -5.56); ON - OFF on E_B is -0.094 (t -3.21) while on E_A it is +0.19 (t 3.55). F1's frozen NI bar
(2.4, C1-NI) reads E_A only, so a loss on the session taught second is not guarded.

**The gap.** The pre-registration's interference bound protects only the OLD session. A sleep arm that buys a gain on E_A
(or E_all) by damaging session B would pass C1-NI. P1 case (a) of the bound's own rationale ("the arm loses less than one
item relative to its control") applies equally to B; F6 already carries "+NI on E_B" (C12) and F7 "NI on E_B" (C17), so F1
and F4 were the inconsistent rows.

**The added guard (E_B-NI).**
- eps_B = 1/12 = 0.0833, one E_B item (the smallest change E_B can register; the same rule as eps = 0.125 for E_A).
- For a contrast with arm X and control Y, d'_s = E_B(X, s) - E_B(Y, s) + eps_B over the same 8 seeds, through the same
  P-8 NI wrapper (degenerate d': MET iff the common value > 0; otherwise t > 1.895). NOT MET = INTERFERENCE RED on B.
- Where it applies (all additive, each beside the existing E_A bar, never instead of it): C1-NI-B (F1: ON vs OFF),
  C7-NI-B (MED vs SHORT), C8-NI-B (EXT vs MED). It is a conjunct of the family's MET row: F1 MET needs C1, C1-NI AND
  C1-NI-B; F4's chain steps C7/C8 need their NI-B too. C3 / C4 / C2 / C9 stay reported, never claims (P-6). F2's C5-NI
  gets its B bar the same way if turnover ever reads > 0.
- Decision-table effect (F1, rows read top to bottom): any K red HALT | C-SENS NOT MET REFUSED | C1 NOT MET NULL | C1 MET
  and (C1-NI NOT MET or C1-NI-B NOT MET) RED | else MET. The NULL row is unchanged; the E_B bar can only turn a MET
  into a RED, never a NULL or RED into a MET.
- Census (P-6, Codex 27): C1-NI-B, C7-NI-B, C8-NI-B join the printed NI bars; "claims made: m" is unchanged by them
  (they gate claims, they are not claims).

**What it would have read on the exposed dev data (not a confirmatory reading, not tuned; stated so the guard cannot be
accused of fitting dev).** F1 ON - OFF on E_B + eps_B: mean -0.0104, sd 0.0826, t -0.357 -> NOT MET (RED on B), had C1
been MET. F4 on dev: MED - SHORT +eps_B t 7.94, EXT - MED t 4.58 (both MET); E_A bars t 4.25 and 2.93. The guard
therefore fires on F1's dev reading and passes F4's.

**What it does not do.** It does not change C1's statistic or bar, any seed set, any budget, W, or any positive control;
it does not read a per-cap verdict (the BF-alone loss is attribution, C4); it does not turn the held HOLD on the null
disposition into a routing claim.

**Reachability (K11).** The new row has a constructed reaching input: C1 MET with a C1-NI-B NOT MET (a constructed
d' with mean < 0 on E_B) must read RED; the instrument's decision-table test adds it. A constructed C1 MET with both
NI bars MET must still read MET.

## DATED NOTE 2026-10-07 (record corrections; additive, changes no frozen number, seed, bar or clause)

- **Review status, stated plainly.** Commits fcdb9d84, 72128b30, a44437fd, f8730471 (the FINAL X-F7 text, which differs from the draft Codex round 2 read), d68bc492 and edc86a96 are TESTS-BY-AUTHOR ONLY: no Codex panel has read them and no battery has run on them. Section 2.17's "Not yet reviewed: the X-F7 amendment" is therefore still true of the final text. A fold-lens panel over 10b59f95^..edc86a96 is requested through the lead; this note is replaced by its disposition when it returns.
- **Section 2.10, row 16500-16799.** The derived stream seeds (task B = s+500, evaluations s+520/s+540) are WITHDRAWN by X-F7 X2 (string-seed streams, no numeric derived seeds). The range 16500-16799 is unused by F7 and holds no primary seed; the 16900-16999 reds range and every confirmatory set are unchanged.
- **Carried owner-open (no board post; none changes code, a decision table or a run now).** O1 (sleep never deleted by a null), O2 (percept-transition record retention class and append budget; gates F7's journaling, not F1-F4), O3 (structural slow clock, cited from G6's Q4), O4 (LifelongConsolidator survival), O5 (E_all quantum 0.05 vs a larger constructed workload; F1-F4 run at 20 items as frozen), O6 (budgets in ticks). Each is posted with ask-owner only when its answer would change a run.

## DATED NOTE 2026-10-07 (fold-lens record; additive)

Codex fold lens over 10b59f95^..edc86a96 returned 3 MAJOR + 2 MINOR; folded in 5e3a7932 (absent lake root; my edc86a96 regressed it), 5254306d (KE: A/B must equal the seed's frozen split; malformed identities HALT), 5a604c8e (dev never claims), bd69d3cb (empty record list), 5845937e (dev-record manifest). **Disclosed deviation: 5254306d folds two findings (M1 and m1) in one commit because they share the same lines.** These commits and their predecessors in that range are "battery-verified only" (author tests, no further panel) unless the owner grants another round; the first run after "machine free" is test_two_real_arm_seeds_restore_exactly, alone.

## 5. Results -- F1-F4 CONFIRMATORY (2026-10-07, seeds 16100-16137; read once, nothing re-run)

Run at 5254306d..b917f5b8 (author-tested only; fold range "battery-verified only"), each family in one process with a fresh NORD_HOME on D:, live lake stamp 3318607701 bytes / mtime 2026-09-27 01:19:19 identical before and after every process (LAKE UNCHANGED x 19: 10 reds, K7, and run+decide of each of 4 families; plus the pytest runner stamp). Gate before the runs: test_two_real_arm_seeds_restore_exactly PASS (RC 0, 31 s); constructed reds K1 K2 K3 K4 K5 K6 K8a K8b K8a-obs K13 all FIRED and K7 sighted (planted 0.00214 vs aa 0.0). Records: 72 + 16 + 16 + 32 JSON files, sha256 manifests `r3-conf-F{1,2,3,4}-records.sha256` beside this file, copies on the board at r3_conf_1007\. Command per family: `python -m snn_data_lake.instruments.sleep_plasticity_ab run|decide --family F<n> --set F<n> --out <dir>` (F2 reads the k7 positive control).

- **F1: NULL** (table row "NULL: sleep synaptic plasticity not shown on held-out at the medium budget"). C-SENS MET (t 3.100), C-IF MET (3.208); C1 NOT MET (t -0.370, mean -0.0125); C1-NI MET (4.583); C1-NI-B NOT MET (t -3.035, mean -0.1042: a B loss, the guard fired); attribution reads only: C2 MET 3.631, C3 MET 2.487, C4 NOT MET -6.081. Claim NOT MADE. Recorded as a null; not re-run.
- **F2: NOT READ** (turnover 0 on a confirmatory seed). C5 degenerate NOT MET; C5-NI, C5-NI-B MET (degenerate). Claim NOT MADE.
- **F3: NULL** (C6 NOT MET, t 1.303, mean 0.0306). Claim NOT MADE.
- **F4: "EXT credited: a sleep-duration benefit (C9 MET)"**. C7 MET 4.255, C7-NI 7.514, C7-NI-B 5.916, C8 MET 2.000 (lb95 0.0026), C8-NI 3.055, C8-NI-B 5.292, C9 MET 5.465; 7 of 7 primary; K checks all 0. Claim stands per the final guarded row. C8's t is 2.000 against the 1.895 bar, a thin margin; this note reports it, it does not interpret it.

## DATED NOTE 2026-10-07 (owner decisions that settle the confirmatory dispositions; additive)

F2 (NOT READ, turnover 0 on 8/8 S-ON seeds): OWNER_DECISIONS_2026-10-04 item 11 already decided the structural rewirer -> "dated P3 exemption until a zero-crossing writer exists" (G6 records it; no new question). F1 and F3: item 15 (R3 O1) holds, "the 2026-09-27 sleep requirement is never deleted by a null", so the requirement stands either way; the retire-vs-exemption question for the replay cap, the benna_fusi rest and attractor.sleep learning is owner question #124. Nothing is applied before the answer.
