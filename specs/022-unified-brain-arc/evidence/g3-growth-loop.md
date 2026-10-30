# G3 / T111 -- growth where G2's capacity cell licenses it (DRAFT pre-registration, wave 1; not frozen)

Plan row: PLAN-FINAL G workstream, G3 / T111 (the row, its dated notes: the encoding contract, the trigger, ONE grower,
candidates follow the diagnosis, the curriculum, the arms, L2's disclosure, the loop and the budget pause (owner
2026-09-25), the growth unit (G7's T1), the reusable column contract (Astra, owner-saved 2026-09-27), the feed
(connection 2)), Part F's G-R3 and its controls, POLICIES P1, P2, P3 (+ amendment), P4b, P4d and P8. Status: DRAFT,
written by the G3 row agent on 2026-10-03 at base `e1b3ae3c`; the orchestrator freezes it after review. Every number
marked **[PROJECTION]** is unmeasured. Nothing here is committed.

## 1. Territory, taken at `e1b3ae3c` (2026-10-03; runtime facts measured on exposed dev seed 9000 only)

1. **The grower is live and stateless.** `engine/brain.py` builds `growth/neurogenesis_engine.py::NeurogenesisSelfImprove`
   with `growth_gate=self._growth_gate_verdict` (the T047 stop, LIVE) and `online_readout="delta"`; it grows through
   `growth/ng_snn.py::NGSNN.fit(init_W, init_b)` (random multi-scale candidate pool, Eq. 6-7 contribution score with the
   1e12 bonus whose float32 spacing ties passing candidates, the plan's 2026-09-23 finding still owed) and
   `prune_unearned`, and places every born unit with `growth/placement.py::place` -> (depth, layer, Dale sign), read by
   nothing. Its percept consumer has a byte fingerprint: `neurogenesis_engine.growth_fingerprint` (crc32 over W, b, beta,
   n_units, accuracy, d_in, m).
2. **The stop rule is live and region-keyed.** `growth/growth_stop.py::GrowthStop` (T047, P8 invention #4) decides per
   REGION whether the next refit may grow, from (n_units, held-out acc) rows and an injected |LP| reader;
   `should_grow(region)` takes no tuning input (the executive-independence contract). Nothing in the tree is
   PROBLEM-keyed: `grep -i 'UNSOLVED|solved_predicate|growth_budget'` over `src/` finds only growth_stop's prose
   ("stopping as UNSOLVED" in the NDP papers) and `reasoning/gf.py::grow_budget` (iterative deepening, unrelated).
2a. **A neighbouring grow-and-prune loop is live (disclosed; R3 adjudicates it).** `sleep/lifelong_consolidation.py::
   LifelongConsolidator` (constructed in `engine/brain.py`, run by `_enter_sleep` on synthetic tasks) grows ART
   prototypes through `growth/growing_learner.py::GrowingLearner` on input novelty, interleaves old-task replay, and
   prunes to a FIXED capacity ceiling (its `budget = 6`) after each task. "Budget" there is a capacity ceiling, not a
   growth spend. G3 holds it a different job (a prototype recruiter, not the NG-SNN lifecycle; no posed problem, no
   solved predicate); its job (grow + interleaved replay + retention) is R3's composition, so R3 runs and adjudicates it.
   If R3 finds it the same job as G3's loop, G3's loop is the merge target.
3. **A lake-resident goal record exists with no live writer.** `growth/learning_progress.py::GoalStore` writes `_T_GOAL`
   (0x0b) rows (embedding, achievement_function_ref, |LP| history, outcomes) and replays them in a fresh view; it is
   instantiated only in its own self-check.
4. **The taught learner is the incumbent G2's problems are posed to.** `readout/interaction_learning.py::
   SpikingInteractionLearner` as `engine/live_brain.py` builds it: nine intents, `n_in=52`, `t_post=6`; one R-STDP neuron
   per label; drive `v = sum_i w[i] * PSP(t_post - rank_i)` over a rank-order code of `text_features` (crc32 buckets mod
   n_in); argmax with ties to the lowest index. Teaching enters through G0's path `learn_turn(text, None,
   feedback=label)`. Its checkpoint (G0(c)) pins the constructor's attribute set, so adding an attribute is a schema
   change.
5. **The encoding contract's red is real (measured).** Widening the code re-buckets every word: cat/dog/sun/sky sit in
   buckets 28/41/31/51 at 52, 11/30/34/19 at 53, 10/5/17/37 at 62; widening 52 -> 62 changed the first-52-synapse code
   of 6 of 6 probe texts, while the frozen encoder reproduced 6 of 6 (scratch probe `g3_encoding_red.py`, out of tree).
6. **The certificate is cheap and exact (measured).** In the learner's own code (seed 9000, the four G2(ii) texts): a
   Kesler LP (4 items x 9 labels x 52 dims, scipy 1.18 HiGHS) reads XOR INFEASIBLE in 6.2 ms and the separable twin
   FEASIBLE in 2.3 ms; the Farkas multipliers lambda = 1 on the four GREET/GOAL constraints, checked in exact `Fraction`
   arithmetic on the (synapse, rank) one-hot lift, sum to zero (0.68 ms), and doubling one multiplier makes the check
   fail. 400 teach epochs of the bare incumbent on XOR took 0.175 s and read at most 2 of 4 (scratch probe
   `g3_cost_probe.py`).
7. **What G3 waits on.** G2(ii)'s generator and construction check (`instruments/inseparable_mapping.py`) exist only on
   the frozen `claude/g1c` stack; G2(i)'s delayed-cue 2x2 is being designed (G2 row agent). The g1 LSM reservoir
   (`reservoir/recurrent_pool.py`, N 256, input only through `w_in`, N_IN 64) is instrument-only on main and rewritten
   on `claude/g1c` (KernelLsmPool). G7's label gate (under `gates/`) and the position/region/problem extension of
   `placement`'s record are G7's; G4's T3 must fit a distance kernel before any attachment arm uses one.
8. **Seed block.** 9000-9999 (SEEDS.md). Scan of `specs/` and `src/` (positive control: 4242 found in 13 files): no
   recording or campaign seed in 9000-9999. Hits are commit hashes, line counts (`doc_budget_gate`), H01 fixture ids
   (`h01_ingest` "9001"-"9005"), a growth-table count (`ng_snn` 9984), and three self-check RNG OFFSETS `9000 + seed`
   (`faculties/wholistic_loop.py`, `faculties/metacognitive_monitor.py`, T080's position null) -- derived RNG streams,
   not recordings; disclosed, not reused.

## 2. Pre-registration (DRAFT; frozen by the orchestrator; later changes are dated strengthenings before any G3 run)

### 2.1 The question and what a result licenses

**Question.** Where G2's capacity cell certifies that the taught learner's current code cannot represent a mapping, does
problem-licensed growth -- grow a batch, freeze growth, attempt with learning ON, test the solved predicate, repeat,
paused at a budget -- solve it prospectively with fewer neurons than presized features selected at the same search
budget and than random growth, without breaking an old mapping, and does the loop stop honestly?

- **A PASS licenses only** "targeted growth on this learner reaches G2's solved predicate on this problem with fewer
  neurons than its controls, prospectively, learning OFF at evaluation". Never fluid intelligence, never an engine
  claim (G-R9 only), never "growth from a unit count" (P4d's not-claimed list).
- **The loop clauses** license only "the loop pauses and reports honestly on these constructed problems".

### 2.2 The problem interface G3 consumes (G2 constructs; G3 never builds a problem)

`Problem` (G2's `instruments/taught_problems.py`; (i)'s generator `instruments/delayed_cue.py`, (ii) wrapped from
`instruments/inseparable_mapping.py`) = {id; intents; items; `episodes(seed, split)` for split in teach / validation /
future, each episode a tuple of turns (text, role context|probe, label or None); `solved(reads)` -- G2's predicate read
through the learner's REAL route, never through an exact-arithmetic shadow; the budget VALUES (2.6)}. Validation and
future are disjoint never-taught PRESENTATION sets of the same finite items (4 texts for (ii), 8 episodes for (i)),
drawn from distinct RNG streams -- not novel items. Learning OFF = no `learn_turn`/correction AND no tick (the sleep tick
replays, rewires and relaxes). The capacity cells are (i)'s S0C1 (label p XOR q in the current probe) and (ii); S1C0 is
the state cell, S1C1 needs both; every (i) cell carries its explicit-history form (`"cue filler probe"`, inside the
6-rank horizon); G2's cell U (labels from a hash of (seed, split, index)) is the constructed DATA / UNSOLVABLE problem.
For G2(ii) the predicate is G2's: all four mappings right on every held-out presentation of its validation episodes. The float-vs-exact
inconsistency G2 found (dev seed 7100: an exact argmax reads 2 of 4 while the float `route_text` reads 4 of 4 at a
constructed near-tie) is a registered check (2.9 C3), not a surprise: the certificate reasons exactly, the predicate
reads the float learner, and a disagreement is logged and never licenses growth on its own.

**Lead read 2026-10-03 -- L1, the primary capacity problem is G2's C1-d.** C1-d = G2(ii)'s
XOR core plus 2 distractor slots of 4 words each (ruling 12, option B); the distractor COMBINATIONS are split into disjoint teach / validation
/ future sets (G2's construction, G2's module). Novel WORDS are unreachable by construction (a bucket no taught text
hits has a synapse that never receives a teaching signal, nor does any grown unit read it), so the held-out data are
novel combinations of taught words, not novel words. Candidate SELECTION reads validation combos only; the primary
endpoint's solve is JUDGED on future combos, which no selection, training or attempt read touched. The capacity
certificate holds over the UNION of teach + validation + future combos (G2, confirmed 2026-10-03: a superset of an
inseparable set is inseparable, so the per-combo Farkas witness with every other item at multiplier 0 is a union
witness, checked by G3's exact `Fraction` check unchanged; distractor buckets are distinct, >= 8 and disjoint from the
core by G2's B1, whose planted red is a collision). Red owed by C1-d (C19 below): a lookup memorizer that reads every
taught and validation combo right must fail the future combos. (ii) and (i) S0C1 stay as SECONDARY cells, read on
same-item presentations with the 2.8 disclosure kept for them.
- **Splits (ruling 12, option B).** 4 words per distractor slot = 16 combos; G2 constructs teach >= 8, validation 2,
  future 3 and future_d 3 (G6(d)'s own untouched set), stratified over the 4 (p, q) cells, the draw rejected if any
  distractor word is missing from teach; G2 may raise counts only. Buckets: 2 p + 2 q + 8 distractor words = 12
  distinct buckets >= 8, disjoint from (i), (ii) and A_ii. G3's split: candidate selection and the attempt's stable-10
  solve read VALIDATION (2 combos) only; the primary endpoint's judgement reads FUTURE (3 combos) once per solve; G3
  never reads future_d (G6's). Disclosed: two validation combos make selection coarse (few presentations per (p, q)
  cell), so a validation solve that fails future can occur. The future read never steers the loop (that would make it
  a selection set): the loop stops at the validation solve as in 2.6, and an arm whose future read then fails scores
  the failure penalty of 2.8's restricted score, the same as an arm unsolved within the budget, and is counted as a
  FAILURE in the solve rate reported beside it.
  The open-time assertion budget >= N_codes reads the teach set's distinct codes from G2's frozen construction (with
  teach = 8 combos it stays under the 52-unit first budget; a larger teach set that crosses it halts per D5).
- **Task A for B = C1-d (ruling 14).** A = A_ii on disjoint words, its buckets disjoint from C1-d's 12; G2 registers the
  pairing. Shared-intent retention stays primary as in the curriculum paragraph below.
- **Ruling 9a** (degenerate superiority contrasts NOT MET; non-inferiority bars by their common value) applies to every
  C1-d reading as stated in 2.8.

**Curriculum (A then B; G2 constructs A, D-G2-3).** A = a separable mapping taught first, B = the capacity cell, on
DISJOINT word sets (same texts relabelled would be a cross-task contradiction, not a retention test). For B = (ii):
A = (ii)'s construction on a disjoint word quadruple, separably labelled. For B = (i) S0C1: A = (ii)'s separable twin.
Intents SHARED with B (GREET/GOAL) are the primary retention test (the harder one); a disjoint-intent A is reported
beside it. Replay, sleep, modulation and exposure are matched across every arm; evaluation is learning OFF.

### 2.3 The encoding contract (built first; P1 of the arcs)

The frozen code is byte-identical forever: `text_features(text, 52)` -> `rank_code` -> 52 integer spike ranks, encoder
version recorded. Grown units add a SEPARATE segment: each grown unit j contributes one synapse per label neuron, read
as `PSP(t_post_g - t_j)` from the unit's own first-spike latency `t_j` (integer steps of its own integration window;
no spike = no contribution). The original 52 ranks are never re-ranked with the grown segment (`rank_code` over the
concatenation would re-rank, the same failure as widening `n_in`). The segment carries source unit ids, spike times,
units, encoder version and episode boundaries (Astra's patterned-input clause). Implementation: a default-OFF grown-
segment port on `SpikingInteractionLearner` (None -> byte-identical drive, deposit and checkpoint), because a wrapper
would re-implement the deposit (P4). Agreed in the debate (section 4, D-R2, D-G5):
- **Sequencing.** R2 holds the claim on `readout/interaction_learning.py` and lands its P1 first (one upcaster chain,
  schema r2.v1); G3's port is the NEXT schema step on that chain (an r2.v1 checkpoint restores with the port None),
  never a second migration path. Main sequences the claim handoff.
- **Port structure.** Every label-neuron input synapse is tagged by segment {frozen52, grown, reservoir}; ablation is a
  mask argument (all-on == today), so "grown neutered" and "reservoir neutered" are arguments, not code paths. Each
  grown/reservoir synapse names its source unit_id, resolvable to G7's label record (no second label store). Segment
  order is fixed by unit_id, never insertion order.
- **Learning OFF freezes plasticity, never state (D-G2-1).** Grown MEMORY-kind units (the state branch) step on
  `route` reads of context turns, not only on `learn_turn`, and reset at episode boundaries; for them the segment is a
  function of (units, the episode's turns so far), so a redo needs the episode id and turn order (open with R2, D-R2-5).
- **Determinism.** A capacity unit's latency is a pure function of (the frozen code, the unit's own parameters): no
  RNG, no state carried between stimuli. Admitted units' INPUT weights are frozen after admission (trace training acts
  on candidates before admission); afterwards only the label neurons' synapses onto the segment learn (R-STDP). A
  'unit' row is written synchronously at admission (write-ahead), before the next teaching event, and
  `grown_code.apply_unit_row(learner, row)` moves the segment from one version to the next with no RNG and no other
  state (R2's redo calls it in lake order). `grown_version` = crc32 over the admitted units' parameters in unit_id order, changing on
  every admission, silencing or weight change. Teaching events carry `encoder = "text_features n_in=52"` (eligibility
  for redo AND replay keys on it) with `grown_version` beside it: R2's REDO re-deposits each event under the version
  current at its log position (reconstructible because unit admissions are IMMORTAL journal rows in the same ordered
  lake); sleep REPLAY deposits through the current units by design and records the version it used. (A version-locked
  replay would make every pre-growth event of task A ineligible after the first growth and starve matched replay.)
- **Lead read 2026-10-03 -- D9, Dale's sign on the grown and reservoir segments.** Every
  output synapse of a grown or reservoir unit onto a label neuron is clipped to its unit's sign at EVERY deposit
  (E >= 0, I <= 0), learning ON and in redo alike. The sign is read once per unit_id at admission from G7's label
  (`label_of`) and cached; the cache is sound because G7's `check_saved` makes the sign write-once per unit_id (a
  re-save whose sign differs from the birth entry's is refused). Two checkers of two objects: G3's clip covers the
  WEIGHT (its red names the synapse, C17), G7's `check_saved` covers the LABEL (its red names the unit). Without the
  clip R-STDP could drive an I unit's synapse positive, making E/I cosmetic and voiding T1-struct's e_s.
- **Not in G3's step:** G5's teaching-signal argument and drive accessor (they touch learn_turn/interact beyond the
  port; G5's own default-off step on R2's chain, or folded by main).

### 2.4 The trigger: a three-way diagnosis matched to the learner's argmax

Over the problem's taught items, on the learner's CURRENT code (frozen segment + grown segment):
- **k** = the number of taught items whose full current code equals another item's code under a different label (the
  contradiction count). **[Reading of the plan's "k = 0 on clean data"; a session call, owner question O1.]**
- **k > 0, and k = 0 once each item's code is augmented with its explicit episode history** (G2(i)'s explicit-history
  control) -> a STATE deficit: candidates are memory kinds from `growth/neuron_zoo.py` (ALIF, autapse).
- **k > 0 with history too** -> a DATA diagnosis: no growth; the problem is reported "contradictory" and kept OPEN for
  new teaching.
- **Same pass (D-G2-5):** the capacity certificate also runs on the history-augmented code, so a cell needing both
  (G2's S1C1) reads STATE AND CAPACITY and licenses both candidate kinds; no second pass after memory growth.
- **k = 0 and a Farkas witness** -> a CAPACITY deficit, growth licensed. The witness is multipliers lambda >= 0 (sum > 0)
  over the multiclass constraints `w_y.phi_i - w_c.phi_i >= 1` (all c != y_i) whose combination vanishes on the
  (synapse, latency) one-hot lift. Checked independently of the LP that found it, in exact `Fraction` arithmetic. A
  lift witness implies inseparability for ANY PSP kernel (the PSP code is a linear map of the lift), so it is sound;
  when the lift admits no witness the certificate does not fire (conservative).
- **k = 0 and a separating witness** -> NOT a capacity deficit: a weight setting that reads every taught item right
  through the real `route_text` (G2's witness pattern). The certificate-gated loop uses this direction to stop growing.
- **k = 0, no lift witness AND no verified real-reader separator -> UNRESOLVED (Codex b1 G3 #8).** The certificate is
  conservative (no lift witness does not imply a separator exists for the fixed PSP projection), so this branch is
  registered explicitly: growth is NOT licensed, the loop attempts without growth under the attempt cap (2.6, A_max),
  and on reaching it persists the terminal state UNRESOLVED in the ledger (never solved, problem kept OPEN).
- **Every no-growth path is finite and persisted.** DATA ("contradictory": no attempts, OPEN); separating witness
  (attempts without growth, cap -> LEARNING-RULE DEFICIT); UNRESOLVED (attempts without growth, cap -> UNRESOLVED).
  Each terminal state is a ledger row; none reports solved.

### 2.5 ONE grower; candidates follow the diagnosis

- **The shared lifecycle.** `NGSNN.fit`'s candidate lifecycle (draw, score on the residual, admit, place, prune) is
  factored so a second consumer can call it. The percept consumer must be BYTE-IDENTICAL first: `growth_fingerprint`
  and `trajectory` equal on `SyntheticPerceptStream` seeds 9000-9003 and on `online_readout`'s self-check engines,
  before and after the refactor. The chat learner consumes the lifecycle only after that check is green (the row's
  "equivalence for the percept consumer"). The float32 1e12-bonus tie is preserved byte-for-byte by the refactor; its
  fix is its own stacked commit with its own before/after fingerprint [owner question O5]. `_refit` stays the percept
  consumer's entry (growth_loop does not supersede it); the lifecycle also exposes `silence_units(consumer, unit_ids)`
  with its matched-control picker (C9; R3 imports it) and a consumer-agnostic prune with `prune_unearned`'s one-pass
  held-out discipline (`NGSNN.prune_unearned` keeps its signature as a thin call; G6's incumbent and greedy arms use
  it). G3's problem loop never prunes during growth: post-solve pruning is G6's. R3's `replay_rows` entry on
  `NeurogenesisSelfImprove` lands after P2 as R3's own stacked commit.
- **Candidate sources.** Incumbent: NGSNN's random multi-scale draws. Capacity deficit: candidate rows TRAINED on the
  multiclass residual with eligibility traces -- per candidate j, `dW_ji = -eta * L_j * e_ji`, `e_ji` the PSP trace of
  input i at j's membrane, `L_j = sum_c B_jc (y_c - p_c)` with B a FIXED random feedback matrix drawn at the
  candidate's birth (never a transposed forward weight: the e-prop invariant; `gates/eprop_transport_gate.py` must
  report zero governed findings on the new module). Train on taught items, SELECT on validation presentations, JUDGE
  on future presentations. State deficit: neuron_zoo memory kinds, selected the same way.
- **Lead read 2026-10-03 -- D4, p_c defined.** The learner is an argmax R-STDP reader with no
  probabilities, so the trace rule's `p_c` is REGISTERED as `softmax(v / T)` over the label neurons' drives `v` (2.3's
  drive, frozen + grown segments, on the item being trained), at a fixed temperature T = 1 drive unit. A session call,
  disclosed: T is fixed now, never tuned on data; a different T is a new pre-registration. `p` enters only the
  candidates' trace rule; the learner's own read stays argmax (ties to the lowest index). Under C1-d, "validation" in
  this bullet means validation COMBOS and "future" means future combos (L1).
- **Not built:** replacing the chat learner with an NG-SNN (stays a benchmark).

### 2.6 The loop and the budget pause (owner 2026-09-25)

Grow one batch -> freeze growth -> attempt (teach the problem's items, learning ON, matched exposure E per attempt)
-> test `solved` on validation -> solved: stop, record the spend; unsolved: grow again, until the budget.
- **The attempt (G2's values, D-G2-7).** E = 400 epochs of the problem's `teaching_sequence` (G2's one value for every
  problem and arm, a projection there), validation read at every epoch end with learning OFF (G2's B12: a validation
  leaves the learner byte-identical, so reading it often changes nothing). An attempt SOLVES under G2's own definition:
  `solved` at each of STABLE = 10 consecutive epoch-end validation reads; it ends early at the first such run, otherwise
  at E. Growth granularity is set by the batch, not by E.
- **Two loop arms.** LITERAL (grow after every unsolved attempt; the owner's rule) and CERTIFICATE-GATED (grow only while
  the capacity certificate still fires on the augmented code; otherwise attempt again without growth). L2's
  grow-on-plateau null (`evidence/l2-grow-on-plateau.md` 6.5) is disclosed: a literal-loop gain the presized twin
  matches is L2's null again, not a growth gain. G3 does NOT discharge L2's re-proposal debt.
- **Lead read 2026-10-03 -- D3, the attempt cap.** Units are not the only spend: the
  certificate-gated loop, finding a SEPARATING witness, attempts again without growth, and with no cap it would
  re-attempt for ever if the learner's rule cannot reach a separator that exists. Registered: at most A_max = 3
  consecutive no-growth attempts per budget (each a full attempt of E epochs with G2's stable-10 early end). Reaching
  A_max stops the loop with the state LEARNING-RULE DEFICIT: never reported solved, growth not resumed by it, the problem
  kept OPEN, the ledger recording attempts and the witness. The same cap binds the UNRESOLVED branch (2.4), whose
  terminal state is UNRESOLVED. The literal loop never attempts without growth, so the cap binds the gated arm only.
  A_max = 3 is a session call fixed now (matched to K = 3 budgets), disclosed. Its red is C18.
- **Budget: G3 owns the MECHANISM, G2's Problem record owns the VALUES (P2, D-G2-2).** A problem's growth spend (units
  admitted, plus candidate evaluations reported beside it) is capped at `beta * median(spend of SOLVED problems with
  POSITIVE spend)` from a persisted running baseline (P1). A paused problem never enters the baseline (it spends up to
  its budget, D5) and neither does a zero-spend solve (task A solves without growth; counting it would collapse B's budget
  to 0 -- G2's refutation). Values as G2 registers them: first problem (no baseline) = the frozen code's width, 52
  units (spend / n_in = 1, P1 case (a)) [O3]; `beta` **[PROJECTION; fixed on dev seeds]**. At problem open the ledger
  asserts budget >= N_codes (the distinct taught codes: 4 for (ii), 8 for (i), 8 with history), so a pause never
  starves a problem below the size at which a memorizing solution exists.
- **Lead read 2026-10-03 -- D5, a batch that would cross the budget.** Admission is per WHOLE
  batch: a batch whose admission would take the unit spend above the budget is refused whole (never truncated, since a
  truncated column is a different unit), and the problem pauses at spend <= budget. With the 52-unit first budget and a
  9-unit column batch (8 E + 1 I) the loop admits at most 5 batches (45 units); single neurons (batch 1) can reach 52.
  The failure penalty of the primary score is the next admissible grid size above the budget (2.8, Codex b1 G3 #6:
  54 for batch 9, 53 for batch 1), not `budget + batch`. If G2's C1-d gives more distinct taught codes than the
  first budget (N_codes > 52), the open-time assertion budget >= N_codes halts that problem as a construction/budget
  mismatch reported to G2 and the lead, never run with a starved budget.
- **At the limit:** the brain reports UNSOLVED, stops growing, keeps the problem OPEN, never reports solved.
- **Retry rule (pre-registered).** A retry after SLEEP draws NO fresh budget (attempts only, growth frozen). New TEACHING
  of the same problem (a new teaching episode, typed in `_T_EPISODE`) draws one fresh budget, at most K = 3 budgets per
  problem lifetime (G2's values; owner question O2), after which the problem stays OPEN with growth exhausted --
  never grown without bound.
- **One stop rule (P2, P4d).** The problem-keyed pause is MERGED into `growth/growth_stop.py`, not a second stop module:
  `GrowthStop` gains a problem key whose verdict is `not solved and spend < budget`, observed by the gate itself; the
  region-keyed percept verdicts stay byte-identical on growth_stop's own harness (merge check). If the debate shows the
  two jobs are distinct, the draft argues it here instead [owner question O4].
- **The ledger (P2, P4b).** The problem record (id, budget, spend, attempts, OPEN / SOLVED / UNSOLVED-paused /
  contradictory / LEARNING-RULE DEFICIT / UNRESOLVED, the baseline) EXTENDS `GoalStore`'s `_T_GOAL` rows (achievement_function_ref = the solved predicate's
  dotted name; outcomes = attempts) rather than a new record class; a fresh process rebuilds the ledger and the
  baseline from the lake. Read by G6 through `growth_loop.ledger(problem_id)` -> `ProblemRecord` {problem_id, state,
  budget, spend_units, spend_evals, attempts, baseline_at_open} and `growth_loop.retention_ok(before, after)` (planned
  names, fixed at P3).
- **The unit journal (P4d: journaled when grown; agreed with G6 and G7, D-G6, D-G7).** No new tag: event rows under
  the same IMMORTAL `_T_GOAL` (0x0B) record class. G3 owns the codec, the fold and the kinds "problem" and "unit";
  "unit" is written at admission, keyed by a stable unit_id minted then and never reused, and carries the unit's
  WEIGHTS THEMSELVES (input row, threshold and kind, the output synapse per label neuron; for an NGSNN unit the W row,
  b and beta column), its role (E or the batch's shared I), batch_id, problem_id, G7's label fields, origin
  ("born"), encoder version and RNG position, with a digest beside them as a check only. Never a ref into the
  `_T_GROWTH_DELTA` (0x08) chain, which is positional and COMPACTABLE (`lake_engine.RETENTION[0x08]`: prefixes before
  the newest FULL are reclaimable), so a pruned unit's only record there could be reclaimed. G6 writes kinds "sleep_bout",
  "sleep_verdict" and "sleep_save" on this class (G3's P3 adds the "sleep_save" API); the fold passes unknown kinds
  through untouched (P4b). A prune never removes a "unit" row. Before the first write P3 checks
  `lake_engine.RETENTION` and `lake_migrate.TAG_DISPOSITIONS` for 0x0B and re-measures `_T_GOAL`'s AppendBudget with
  the unit rows (an IMMORTAL class's bytes/tick ceiling; a constructed flood red is owed). **Disclosed gap, reported
  to main, not fixed here:** the LIVE percept grower's units (no posed problem) stay only in the compactable 0x08
  chain, survivors only [O8].
- **The solve event.** `GrowthLoop(..., solved_sink=None)` calls the injected sink once per OPEN -> SOLVED transition
  with the `ProblemRecord`; default None requests nothing. G6's scheduler gate subscribes (the sleep REQUEST is G6's;
  the loop only emits "solved").

### 2.7 The growth unit (G7's T1) and the reusable column contract (Astra)

- **Unit arms, same neuron budget:** a column-sized batch (k excitatory LIF units + one shared inhibitory unit; k is an
  arm: 4, 8, 12, primary 8 -- shared with G7) vs single neurons vs a presized network of the final size vs random
  attachment. COL and SINGLE share ONE site rule (the map site where error builds up) and ONE targeting rule (attach
  to the label neurons with the largest residual); they differ only in unit size, so T1 isolates the unit. U counts
  EVERY admitted unit, E and the shared I alike (a column of 8 E + 1 I adds 9).
- **Placement and labels (G7 owns both).** Labels come only from G7's `placement.place_labeled(n, *, site,
  problem_id, batch_id, seed)` (full records: position (ap, dv, x_um, y_um, depth), layer, sign, region, problem_id,
  origin "born"); `column_batch.py` keeps batch_id and the E/I role and NO label of its own (no second store). Every
  grown unit, SHAM units included, passes G7's `gates/unit_label_gate.py::require_admissible(units, *,
  resolve_problem)` at ADMISSION, before its row joins the learner; presized arms are constructed networks, not
  admissions, and are not gated; `check_saved` is G6's. `problem_id` = G3's ledger key = G2's `Problem.id` (one id,
  stable across restarts). Within-site x/y offsets are disclosed uninformative and attachment is KERNEL-FREE (layer
  target only) until G4's T3 fits a kernel; a hand-picked width is refused (plan order).
- **Lead read 2026-10-03 -- roles and one store (agreed with G7).** G7's `place_labeled`
  gains `roles=None`: None draws exactly as `place()` does today (the pinned cortex crc does not move); with roles
  given ("E" / "I" per unit) the Dale sign IS the role (E +1, I -1) and the layer is drawn from H01's joint
  P(layer | sign) with `assign_sign`'s measured E-fractions, depth inside that layer's band. G3 passes an explicit
  roles tuple on EVERY grown batch (a single = `("E",)`, a column = `("E",)*8 + ("I",)`, BATCH9 the same call as the
  column), so no G3 admission rides the roles=None path. `column_batch.py` keeps NO E/I role of its own: batch_id and
  its members only, the role read off the label's sign through `label_of` (one store, P2; a role/sign disagreement
  cannot exist by construction). The bullet above's "the E/I role" in `column_batch.py` is superseded by this one.
- **Astra's matched challengers:** LIF vs ALIF, adaptive vs fixed inhibition, a separately driven dendritic apical
  context port vs a disconnected one, each against its simpler equal-capacity, equal-compute arm. A-then-B followed by
  an A+B composition task: shared collaborating batches vs task-exclusive batches on held-out transfer; neuter the
  cross-batch projections as the causal control. A causal context/feedback port never carries a target or a future
  observation (a forbidden-channel check, built like G1(c)'s R-a2b). Inference works with learning OFF. Output spikes
  leave through the named feed port. Accepted weights, topology, adaptation, traces, delays, RNG and lineage get a lake
  disposition (journaled when grown, P4d; durable or reconstructible).

### 2.8 Endpoints, statistic and bar

Part F: S = 8 paired seeds, paired one-sided t > 1.895 (7 d.f., alpha 0.05), ties counted (a zero difference stays in
n), the exact sign-flip false-positive rate (all 2^8 masks) printed beside each verdict, no majority-of-seeds rule.
Paired seed s fixes the learner's init, the candidate RNG, the teaching schedule and the presentation order for every
arm.

- **Primary (G-R3 clause 1): a RESTRICTED SOLVE/FAILURE SCORE (Codex b1 G3 #6).** R(arm, s) = the observed number of
  hidden units at which an attempt first SOLVES (2.6: G2's stable-10 definition on validation) AND the following
  future read is `solved`, learning OFF; an arm that fails (unsolved within the budget, a no-growth terminal state, or a
  validation solve whose future read fails) scores the FAILURE PENALTY = the next admissible grid size above the
  budget (54 for batch 9, 53 for batch 1 at the 52-unit budget). The penalty is an assigned score, not an observed
  size, and no conservatism is claimed for it. Targeted = the certificate-gated loop with trace-trained candidates.
  Contrasts, each paired: D_sel = R(presized-selected) - R(targeted) and D_rand = R(random growth) - R(targeted).
  Presized arms are evaluated on the growth arm's size grid (multiples of the batch), each size a FRESH presized
  network, selected at the same cumulative candidate-evaluation count the targeted loop spent to reach that size.
  **Exposure matching (Codex b1 G3 #7).** A presized network at a grid size is taught the targeted arm's ACTUAL
  recorded teaching sequence up to the moment the targeted arm reached that size (every epoch of every attempt,
  including the gated loop's no-growth attempts and every early end at a stable-10 solve), replayed in the same order;
  `m x E` is not used. Candidate-training exposure is matched separately: the presized-selected arm's candidates get
  the same total candidate evaluations and candidate-training epochs as the targeted arm's candidates up to that size.
  PASS needs t > 1.895 on BOTH contrasts and mean D >= the size-grid floor (L4 below). **What a PASS licenses.** The
  restricted score mixes size and failure, so a PASS licenses "targeted growth solves within the budget with a lower
  restricted score". The words "FEWER NEURONS" are licensed only when both arms of a contrast SOLVE (uncensored) on at
  least 6 of the 8 seeds and the paired difference of observed sizes on the seeds where both solve is itself positive
  (mean D >= the floor); otherwise the result is reported as a solve-rate difference, never as a neuron saving. Solve
  rates and observed solve sizes are reported separately for every arm.
- **Disclosure: "prospective" on finite items (D-G2-8).** G2's validation and future splits are never-taught
  PRESENTATIONS of the same 4 or 8 items, not novel items. For a capacity unit (stateless) with learning OFF, a read is a
  pure function of (weights, text), so a validation solve ENTAILS the future solve; the future read guards only against
  a state leak (G2's B12) and, for memory units, presentation-order effects. G-R3's "prospectively" is therefore read as
  "after the fact, learning OFF, on presentations the selection never saw", and the plan's "select on validation, judge
  on future data" is weak on these problems: candidate selection on validation is selection on the taught items'
  codes. Generalization to unseen items is out of G2's and G3's scope [O9].
- **The six arms (the row's list), each with its role.** INCUMBENT (no growth: must stay unsolved, re-confirming G2's
  cell on G3's seeds); PRESIZED FROZEN (units from the lifecycle's incumbent random source -- the same draw RANDOM GROWTH
  uses -- at the final size, read through the same grown-segment port, never selected; NOT G2's `presized_expansion`,
  which replaces the code with a ReLU front-end taught by `interact` and would confound architecture with timing; G2's
  control is reported beside as G-R2's, D-G2-9); PRESIZED SELECTED
  (the SAME candidate source as targeted -- trace-trained against the bare learner's residual -- with equal candidate
  evaluations and candidate-training epochs, chosen all at once before the first attempt; so D_sel isolates growing
  where error builds from presizing); TARGETED (above); RANDOM GROWTH (the same schedule and batch, untrained,
  unselected random-source units; D_rand isolates targeting); SHAM EXPANSION (identical bookkeeping: units appended,
  placed, labelled and ledgered, output synapses held at zero; must match the incumbent, so bookkeeping alone buys
  nothing).
- **Degenerate-contrast rule (superseded by ruling 9 / 9a, next bullet).** Accuracy at matched final size (each arm at the targeted arm's final size, learning OFF on future presentations)
  is SECONDARY, reported with its own degenerate flag ("tied at ceiling"); it can never pass or fail clause 1 alone,
  because on a four-text problem every arm that solves reads 4 of 4.
- **Lead read 2026-10-03 -- ruling 9 supersedes the degenerate-contrast rule above.** A
  zero-variance paired contrast (all eight differences equal, whatever their sign) reads NOT MET, never PASS. This is
  disclosed as STRICTER than `paired_contrast.contrast`'s zero-variance rule (which would read mean > 0 as t = +inf);
  it is the one convention every row feeding this integration uses (G1(c) 2.3 is the pushed precedent). On the
  size grid a degenerate contrast is likely (every seed solving at the same batch count), so a uniform win of one
  batch on all eight seeds reads NOT MET here -- the cost of the convention, disclosed, not a reason to change it.
  The secondary accuracy reading keeps its own "tied at ceiling" flag.
  **Ruling 9a (amends 9) scopes this:** degenerate -> NOT MET binds the SUPERIORITY contrasts only (D_sel, D_rand, the
  loop pair, C10's held-out contrast, the feed's U contrasts, Astra's mechanism-vs-simpler-arm contrasts). A
  NON-INFERIORITY bar reads a degenerate sample by its common value: > 0 MET, <= 0 NOT MET. In G3 that is T1-size's
  d_s, T1-struct's d_s and e_s (2.10) and C11 when read as a paired contrast against the NI margin delta_NI (e =
  delta_NI - F, per seed; delta_NI > 0 strictly, D7). C10's contrast is superiority (true-label minus shuffled-label
  held-out score). Disclosed: under T1-struct's granularity, equal U on every seed gives d_s = log 1.2 > 0 on every seed, an
  entailed MET -- which is why its U half stays behind the granularity guard (REFUSED below the threshold) and only
  e_s reads at any size.
- **Lead read 2026-10-03 -- L4, the primary floor is the size grid's resolution, fixed now.**
  "mean D >= one batch" is not a projection to be fixed on data: D is measured on the growth arm's size grid, whose
  resolution is the batch b (b = 9 units for the primary column arm, 8 E + 1 I; b = 1 for a single-neuron arm), so a
  difference below b is not measurable. The floor is mean D >= b, fixed now, never set on dev or confirmatory data.
- **Lead read 2026-10-03 -- L1 narrows the "prospective" disclosure above.** On the primary
  cell C1-d, validation and future are disjoint sets of novel distractor COMBINATIONS, so a validation solve no longer
  entails the future solve and the future read is a real held-out judgement ("prospectively" = learning OFF, on
  combinations no selection, training or attempt read). Novel WORDS remain out of scope (unreachable by construction,
  2.2). The same-items disclosure above stays in force for the secondary cells (ii) and (i) S0C1.
- **The loop pair.** The literal loop's U is reported beside the gated loop's; each is compared with the presized and
  random arms at matched final size (secondary, same statistic).
- **The baseline rule is exercised (secondary; D-G2-10).** In a single-problem campaign every B is the first
  positive-spend problem, so its budget is always the 52-unit first value and `beta x baseline` never runs. A
  SEQUENCE reading on the same seeds poses, after B = (ii) solves with spend s_B, a second capacity problem (G2's (i)
  S0C1) whose budget must equal beta x s_B as read from the rebuilt ledger in a fresh process; a ledger unit check
  (constructed solved spends {10, 20, 30} plus one zero-spend solve -> budget beta x 20) is its red-bearing companion.
  The sequence reading is its own family with its own isolated homes and ledger chain (2.12a), never sharing state with
  the primary family.

### 2.9 Checks, each with a constructed, reachable red (a red that does not fire halts its arc)

| id | check | constructed red (must fire) | control (not the method) |
|---|---|---|---|
| C1 | old codes unchanged: the 52-rank code of every A and B item byte-identical before and after growth | widen `n_in` 52 -> 62 (measured: 6/6 texts change) | the frozen encoder read twice (measured 6/6 identical) |
| C2 | the certificate fires on XOR (k 0, exact lift witness) and NOT on the separable twin (separating witness reads 4/4 through `route_text`) | twin labels pushed through the XOR path; one doubled multiplier (measured: fails) | the LP finder's status beside the exact check (measured: 2 on XOR, 0 on the twin) |
| C3 | exact-vs-float agreement logged: the certificate's verdict and the learner's real reads are both recorded per item | G2's dev-seed-7100 near-tie weights: exact 2/4, float 4/4 must be logged as a disagreement, never as a solve licence | the same weights off the tie |
| C4 | contradiction routes to DATA, not growth | G2's cell U (labels hashed per presentation; k > 0 with history): zero units grown | its clean twin grows |
| C5 | state routes to memory kinds | a delayed-cue item pair (k > 0, k = 0 with history; G2(i)) must draw neuron_zoo memory kinds, not capacity rows | the capacity cell draws capacity rows |
| C6 | lifecycle equivalence for the percept consumer | reverse the admission tie-break: `growth_fingerprint` must change | the unrefactored engine on the same seeds |
| C7 | no weight transport in the trace rule | a variant whose L_j reads the transposed readout weights: `eprop_transport_gate` must flag it | the fixed random-B rule: zero findings |
| C8 | the loop never reports solved on an unsolvable problem and pauses at its budget (G-R3, G-R9 (iii)) | (a) G2's unsolvable cell U run through the LITERAL loop with its diagnosis gate DISABLED for this red (C4 would otherwise grow nothing; the pause must hold even when the trigger misfires): must pause when the next whole batch would exceed the budget, with spend <= budget (Codex b1 G3 #12; 45 of 52 at batch 9, 52 at batch 1), report UNSOLVED, keep OPEN (no chance figure is claimed, D6 below the table); (b) a capacity cell with its budget overridden, in the red's fixture only, below the smallest U seen on dev seeds: same | a solvable twin reports SOLVED under budget |
| C9 | the gain needs the new units (rewritten, Codex b1 G3 #1; see "C9's comparison population" below the table) | silence the grown units at evaluation: B's future-combo `solved` must fail (near-entailed by the certificate, disclosed: silencing every added unit largely restores the original deficit, so the informative readings are the PER-ITEM effects and the matched control); silencing an equal count of MATCHED ACTIVE PRESYNAPTIC SOURCES from the registered comparison population must not fail it | the unsilenced arm |
| C10 | the candidates' training signal matters (rewritten, Codex b1 G3 #2) | candidates trace-trained on SHUFFLED teaching labels, with the diagnosis (computed once from true labels and frozen), the candidate budget, the selection rule (validation combos, TRUE labels) and the admission schedule (the true-label arm's batch count on that seed) all held constant: the learner's held-out (future-combo) score after admission must be lower than with true-label-trained candidates (paired t, superiority; ruling 9a) | true-label-trained candidates |
| C11 | old mapping retention | A's future accuracy drop after B, learning OFF, <= the factual bound F_bound (D7); a deliberately overwriting arm (A's texts relabelled during B) must exceed it | A untouched |
| C12 | the label gate is called | a grown unit with no label must be refused at admission (G7's red, reached through G3's admission path) | a fully labeled unit is admitted |
| C13 | budget ledger survives a restart (P4b) | a fresh process reading a lake whose spend row was dropped must refuse to reproduce the ledger | the uninterrupted twin's ledger, byte-equal |
| C14 | the forbidden context channel | a context port that reads a target or a future observation must raise | the declared causal port |
| C16 | the grown-segment port's positive control (2.3; also T2's precondition, D-G4-3): the bare learner, reading G2's `witness_units(problem)` (two hand threshold units over the frozen code) through the port, learns B (XOR or S0C1) to G2's solved predicate, learning OFF at evaluation, on fixture seeds 9900-9907 | the same units with the grown segment masked off (all-off ablation): B must stay unsolved; and, for SPECIFICITY (Codex b1 G3 #9), witness units replaced by PROVABLY INCAPABLE planted units of equal count and norm (each unit's latency a function of a single word slot, so the augmented code stays inseparable), whose incapacity is VERIFIED before the run by the exact lift certificate firing on the augmented code: B must stay unsolved. G2's random norm-matched `decoy_units(problem, n, seed)` are a MEASURED COMPARATOR, reported, never a red (random threshold features can make XOR learnable) | the port None (byte-identical incumbent), unsolved on B |
| C15 | (P5) the feed is wired | the norm-matched random-feed arm must differ from the fed arm on the pool's carriers | the pool's own determinism red at N 256 (G1), and at EVERY tested N once the N > 256 arm is built [ANSWERED 2026-10-29: see DATED ADDENDUM 2026-10-29 (owner growth target) at the end of this file.] |
| C17 | (lead read 2026-10-03; D9): Dale's sign holds on every grown/reservoir output synapse | a planted deposit that would flip an I unit's synapse positive (and an E unit's negative) must be clipped, and a planted synapse whose stored value violates its unit's sign must be refused at restore; the red names the synapse | the same deposits on sign-consistent synapses pass unchanged |
| C18 | (lead read 2026-10-03; D3): the gated loop's attempt cap | (Codex b1 G3 #10) a SEPARABLE problem (C2's twin) in the red's fixture only, with PLANTED, VERIFIED-INCORRECT initial reads (label-neuron weights set so the twin reads at most 2 of 4, checked before the run) and EVERY plasticity path disabled (R-STDP deposit, modulation, correction and tick; asserted by the learner's weights and checkpoint bytes being identical before and after one attempt): the fixture's unsolved state is ASSERTED first, then the gated loop must stop after exactly A_max = 3 no-growth attempts, state LEARNING-RULE DEFICIT, never solved, zero units grown; a twin variant with neither lift witness nor verified separator must end UNRESOLVED the same way | the same twin with learning on solves without growth |
| C19 | (lead read 2026-10-03; L1): the held-out judgement on C1-d is real | a lookup memorizer (a constructed reader storing every teach and validation combo's label) reads teach + validation right and must FAIL the future combos' `solved` | a reader that computes the XOR core from the words solves the future combos |

**Lead read 2026-10-03 -- D6, C8(a)'s chance figure.** The earlier "2^-32 on 32 validation
presentations" was never reproduced and assumed two labels; with the learner's nine intents a uniform guess per read
is 1/9, and the stable-10 rule needs ten consecutive solved epoch-end reads. The number is WITHDRAWN, and no
probability bound for the LEARNER is registered in its place (Codex b1 G3 #11: a uniform-guesser figure does not bound
a trained, validation-inspected learner on fixed crc-indexed labels). No chance figure is printed with C8 and no
inference about the learner is drawn from one. C8 needs none: it reds on any solved report and on any growth past the
budget.

**Lead read 2026-10-03 -- D7, C11's retention bound from dev seeds.** "Minus one item" allowed
a 25 % drop on a four-text A and was a projection. Registered instead, as TWO quantities (Codex b1 G3 #5):
- **F_bound, the factual retention bound (C11).** Fixed on exposed dev seeds 9000-9015 before any confirmatory seed, as
  the INCUMBENT's A-drop (no growth, the same B teaching exposure as the targeted arm, learning OFF reads of A's future
  presentations) -- mean + 2 sd over the 16 dev seeds, in items, rounded up. It may be 0. C11 passes when the targeted
  arm's drop <= F_bound; the overwriting arm (A's texts relabelled during B) must exceed it, and if it does not on the
  dev seeds the bound is unreachable and C11 halts its arc (P8 clause 1).
- **delta_NI, the statistical non-inferiority margin.** delta_NI = max(F_bound, ONE ITEM of A's future set), so it is
  STRICTLY POSITIVE: a zero margin would turn two equally lossless arms (constant e_s = 0) into NOT MET under ruling 9a
  and a deletion. delta_NI is the margin of every NI contrast on retention: T1-size's and T1-struct's e_s (2.10) and C11
  read as a paired contrast. One item is the smallest measurable retention difference (the quantum of an items count),
  fixed now. The same fix is sent to G7 (T1's owner) so T1's rule uses delta_NI, not F_bound.
- **Wording.** Failure to establish non-inferiority within delta_NI is reported as "retention not shown non-inferior",
  never as demonstrated interference.

**C9's comparison population (Codex b1 G3 #1; lead guidance considered).** Task A is separable and solves without
growth, so there are no A-grown units, and the shared GREET/GOAL label neurons are not an A-exclusive population. The
lead's two suggested populations were checked by design and REFUSED, each with its reason: (a) units grown on an earlier
capacity problem read that problem's word buckets, which are disjoint from C1-d's 12 by construction (ruling 12), so
they are silent on B's evaluated items and silencing them is vacuous (the D-R3-4 failure); (b) units of the
presized-frozen arm live in a different network, so silencing them cannot control a silencing of the targeted network.
Registered instead: the comparison population is C1-d's DISTRACTOR INPUT SOURCES in the same network -- the 8 frozen-code
input lines of the distractor-word buckets (one line per bucket onto each label neuron), each active on the B items
that contain its word (two of the eight on any one item), carrying learned output synapses onto the label neurons, and
irrelevant to the XOR rule by construction. They are presynaptic sources of the same
kind the port reads (a grown unit is one more input source onto the label neurons), so "silence a source" means the
same operation for both. Matching: each distractor source is paired, nearest neighbour without replacement, with a
grown unit on (output-synapse L2 norm onto the label neurons, firing rate on B's future items), dead sources and dead
units excluded; k = the number of matched pairs within a tolerance fixed now (norm ratio within [0.5, 2], rate
difference <= 0.25). C9 silences the k matched grown units and, separately, the k matched distractor sources. Effects
are reported PER ITEM (each future-combo presentation, each (p, q) cell). **Feasibility check before confirmation:** on
dev seeds 9000-9007, k >= 1 on every seed and k covers at least half of the grown units on at least 6 of 8 seeds;
otherwise C9 is declared infeasible before any confirmatory seed, reported to the lead, and the primary family does not
confirm until a population is ruled. The secondary cells ((ii), (i) S0C1) have no distractor slots: there C9 reports
the per-item and per-batch silencing effects only, labelled "no matched control".

### 2.10 G7's T1, Astra's batches and the feed (each its own family; same statistic)

- **T1 (G7 owns the decision rule; G3 runs it, D-G7).** Burden on the column: per seed, d_s = log 1.2 - log(U_COL /
  U_SINGLE) and e_s = delta_F - (F_COL - F_SINGLE), F = C11's retention drop and delta_F = C11's frozen bound; PASS iff
  both contrasts are MET (`paired_contrast`, t > 1.895, S = 8), otherwise RED. More than half the seeds censored in
  either arm -> T1 REFUSED (reported, neither pass nor red). PRESIZED at the COL arm's final size runs on T1's own
  seeds, so "a presized network matching both means growth earns nothing" is a paired reading. A T1 dev power pilot
  on 9040-9047 estimates sd(d_s), sd(e_s); projected power < 0.8 at S = 8 -> main is asked for a Part F amendment
  (only G1 may pre-register S > 8 today) before any confirmatory seed. A red deletes the column part of decision C and
  the owner is told (G7 section 6).
- **Lead read 2026-10-03; joint G3 + G7; PENDING the owner's O10) -- the T1 split supersedes
  the T1 bullet above.** Why: on G2's problems the minimal U is about 2-4 (XOR = OR - AND, a linear read of two
  threshold units; disjoint XOR blocks collapse to the same two shared units because one block is active per item;
  parity is capped at n <= 6 by the 6-rank horizon), so a 9-unit column has U_COL >= 9 >= 2 x U_SINGLE and d_s < 0 on
  every seed: the red is entailed by granularity, not measured. Growing SINGLE in batches of nine instead entails the
  mirror PASS (both arms quantize to 9) and changes the record's question, so it cannot replace T1 by itself. Registered:
  - **T1-size** (the record's question): COL (8 E + 1 I) vs SINGLE (batch 1), G7's burden-on-the-column rule unchanged,
    behind a GRANULARITY GUARD: the U contrast is REFUSED (neither pass nor red, reported, nothing deleted) unless the
    median U_SINGLE on the dev power pilot (9040-9047) is >= 80 (k = 4: >= 40). 80 is CHOSEN and disclosed: the worst
    quantization overshoot is k = 8 units and 8 / U <= 0.10 keeps it at half the 20 % bound. On G2's present problems
    T1-size therefore reads UNTESTED this wave; reported to the owner as untested, never as a deletion of decision C's
    column part.
  - **T1-struct** (a separate contrast): COL (8 E + 1 I, shared I with E->I and I->E wiring) vs BATCH9 (nine independent
    single units, also 8 E + 1 I through the same `place_labeled` call, same site and targeting rules; BATCH9's I takes
    frozen-code input and projects only onto label neurons). Its U half carries the same granularity guard (both arms
    quantize to 9). Its retention half e_s = delta_NI - (F_COL - F_BATCH9) (delta_NI > 0 from D7) reads at ANY size,
    because at equal count and equal E/I composition it isolates the column's structure (in COL vs SINGLE it is
    confounded by neuron count). A T1-struct red (the column's retention is NOT SHOWN non-inferior to an equal-count
    unstructured batch within delta_NI -- never worded as demonstrated interference) DELETES the column STRUCTURE (the
    shared I and its wiring) under G7's burden-on-the-column rule, and the owner is told. Dale clipping (D9, C17) is its
    precondition: without it E/I is cosmetic. T1's e_s in the bullet above reads delta_NI in place of delta_F (sent to
    G7, T1's owner).
  - **T1 waits on G4's T3 at FAMILY level (Codex b1 G3 #3; DEPENDENCIES.md G7 row).** No T1 arm (T1-size or T1-struct)
    runs before G4's T3 result, alongside G2's problems, G3's loop and the owner's O10; the kernel-free exemption of
    2.7 / 2.13 does NOT apply to T1. Any exception needs a board amendment.
  - Statistic, seeds and the > half-censored REFUSED rule as in the T1 bullet above (S = 8, 9300-9307, PRESIZED paired),
    with ruling 9a for degenerate samples: d_s and e_s are non-inferiority bars, so a degenerate sample reads by its
    common value (> 0 MET), never ruling 9's superiority NOT MET. The S > 8 amendment question (O7) stands for T1-size
    only.
  - Until the owner answers O10, T1-size is "untested" and nothing is deleted on its account.
- **Astra:** reusable batches must beat task-exclusive equal-size, equal-compute batches on prospective A+B transfer,
  the gain must vanish under cross-batch projection neutering, retention must hold, and each added mechanism must beat
  its simpler matched arm; any failure is G-R3 red for that mechanism.
- **Feed (connection 2):** every admitted unit keeps its identity and problem lineage and projects into the G1
  reservoir through declared input synapses; arms: fed; the same units read directly by the learner's label neurons
  (no reservoir); a feed from norm-matched random units. The readout in all three is the taught learner's label
  neurons through a separate latency segment of the reservoir's first spikes (the incumbent spiking, brain-tuned
  readout for the posed-problem job; G5's tuning-rule arms run on it, and a rule replaces R-STDP only by winning G-R5).
  Red: fed does not beat direct and random feed (U, paired). A further arm lets a fed unit
  become a reservoir unit (N > 256); it re-runs the pool's determinism red at the new N and waits for the owner to
  ratify a growth target inside the reservoir (POLICIES P4d). [ANSWERED 2026-10-29: see DATED ADDENDUM 2026-10-29 (owner growth target) at the end of this file.]

### 2.11 Null dispositions (each a causal claim; its instrument owes its positive control first)

- **Before any confirmatory run:** C16 (the port's positive control: a learner that cannot solve B even when handed
  the witness units through the port makes every growth arm unreadable, and G4's T2 then reads REFUSED), C2's positive
  control (the certificate reads the separable twin feasible with a
  route_text witness; measured on 9000) and C6's equivalence must be green, and G2's construction check (G-R2) must
  have passed: a G3 null on a problem whose construction is wrong is G2's red, not G3's.
- **Primary null** (targeted does not beat presized-selected or random): the trace-trained candidate source is DELETED
  (P8 clause 5); the incumbent random source stays. The owner-drawn part ("the brain grows new neurons to solve")
  is NOT deleted: its implementation is not claimed, and the owner decides on the evidence (P4d). A literal-loop gain
  matched by the presized twin is recorded as L2's null again.
- **Loop null** (the gated loop does not beat the literal loop at matched size): the certificate gate is retired from
  the loop (it stays as the trigger diagnosis); one loop variant survives (P2).
- **T1 red:** the column part of C is deleted, the owner told (G7 section 6). **Lead read 2026-10-03:** under the T1 split
  (2.10) a T1-size REFUSED reading deletes nothing; a T1-struct red (retention not shown non-inferior within delta_NI)
  deletes the column STRUCTURE only, the owner told. **Astra mechanism red:** that mechanism
  is deleted, its simpler arm survives. **Feed null:** its implementation is reverted and not claimed; the reservoir,
  an owner-drawn part, is not deleted (P4d); the owner decides.

### 2.12 Seeds (block 9000-9999)

Dev, exposed: 9000-9099 (design measurements, the projections' fixing, smoke runs; 9040-9047 the T1 power pilot).
Confirmatory, read once: 9100-9107 (S = 8, the primary family and C8-C11); reserve 9108-9115. T1 + Astra family,
PRESIZED included: 9300-9307, reserve 9308-9315 (G7 released its T1 seeds 13110-13127 to this, pending main).
Feed family (after G1(c)): 9200-9207, reserve 9208-9215. Constructed reds and fixtures: 9900-9999. Exposed and
inherited, read only as exposed: G2's learner dev seeds 7100-7103 (C3's near-tie). Never run: 300-361, 6000, 6001,
6100-6107, 7000-7007.

### 2.12a Campaign isolation (Codex b1 G3 #4)

- **One isolated home per (seed, arm).** Every run of every family gets its own fresh directory under the campaign's
  scratch root, holding its own home (NORD_HOME / `substrate.home.throwaway_home()`), its own lake path and its own
  output path. The parent sets them and passes them EXPLICITLY to every child process (environment AND argument); a
  child that does not receive them refuses to start rather than falling back to a default.
- **The live lake is refused.** Before any write each process asserts that its resolved lake path lies under its own
  scratch directory and is not `C:\Users\vinhl\.snn_data_lake` (or any path under it); a violation raises. After the
  campaign the live lake's mtime is checked unchanged and printed in the results record.
- **Paired arms start from the SAME frozen ledger state.** For seed s, one ledger snapshot (the problem/unit journal
  and the positive-spend baseline as of the moment B opens) is written once, digested (sha256), and COPIED byte-for-byte
  into every arm's home before that arm runs; each arm asserts the digest before its first step. No arm inherits
  another arm's spend, units or baseline, so every arm of a seed sees the same budget.
- **The sequence experiment is a SEPARATE family** (2.8's baseline rule: (ii) then (i) S0C1 with `beta x s_B` read from
  the rebuilt ledger). It has its own homes per (seed, arm), its own ledger chain that deliberately carries the first
  problem's spend into the second, and its own results section; it never shares a home or a snapshot with the primary
  family.
- **Red (part of C13's fixture):** a child launched without its home argument must refuse; a child whose home
  resolves to the live lake path must raise before any write; two arms whose ledger snapshot digests differ must refuse
  to pair.

### 2.13 Dependencies and what waits

- G2's capacity cells ((ii), and (i)'s S0C1), task A, the cell U and their budget values, all from G2's
  `instruments/taught_problems.py`, and G-R2's construction check: the CONFIRMATORY campaign waits; P1-P4 build on
  fixtures (the learner's own `text_features`, built in G3's tests, never a copy of G2's module; re-pointed to G2's
  module when it and `claude/g1c` land on main).
- G7's label gate and record extension: P4's admission path waits for the gate; P1-P3 do not.
- G4's T3: the distance-kernel attachment arms wait; kernel-free arms of the PRIMARY family do not. The T1 family
  (T1-size and T1-struct) waits on T3 as a whole (Codex b1 G3 #3; 2.10), and on the owner's O10.
- G1(c) landing and a reservoir input port: P5 waits entirely.
- G5: none for G3's deciding test (the learner's label neurons are the readout; G5's tuning-rule arms run on them).
- R2: P1's port waits for R2's P1 (its upcaster chain on `interaction_learning.py`); main sequences the claim.
- Consumers of G3 (they wait on G3, not the reverse): G6 reads `ledger`, `retention_ok` at its freeze, the unit
  journal and `solved_sink`; R3 imports `silence_units` and adds `replay_rows` after P2; G7 writes its two
  `neurogenesis_engine.py` persistence changes after P2.

### 2.14 Cost at real scale

Measured (seed 9000, one unit each): 400 teach epochs of the bare learner 0.175 s; one Kesler LP 2-6 ms; one exact
Farkas check 0.7 ms; `place(10)` 0.03 ms. **[PROJECTIONS]** one gated-loop run on G2(ii) with trace-trained candidates
(C = 32 candidates per step, 50 candidate-training epochs): 2-10 s per seed-arm; the primary family (S 8 x ~7 arms x the
size grid) 5-20 min serial; T1 + Astra about 3x that; the feed family bounded by the pool's 25 ms per step (1,600
presentations ~40 s per seed-arm, ~10-30 min). No GPU needed at these widths (`infra.device.select` routes CPU below its
measured crossover).

### 2.15 Arc split and the files each part touches (claims on the board hold the new paths)

- **P0** this file; the plan's dated G3 note (orchestrator).
- **P1** encoding contract + certificate: NEW `growth/grown_code.py`, `growth/capacity_certificate.py`,
  `tests/test_grown_code.py`, `tests/test_capacity_certificate.py`; EDIT `readout/interaction_learning.py` (the
  default-off grown-segment port as the next schema step on R2's upcaster chain, AFTER R2's P1; claim handed over by
  main) and G0(c)'s checkpoint pin test as R2 left it. Registrations: `src/snn_data_lake/__init__.py` (REGIONS),
  `run_selfchecks.py`, `gates/dormancy_gate.py` (EXEMPT until G9 wires the engine).
- **P2** the shared lifecycle + the trace-trained source: NEW `growth/candidate_lifecycle.py` (`NGSNN.fit` routes
  through it; one lifecycle survives; R3 agreed either form), `tests/test_candidate_lifecycle.py`; EDIT
  `growth/ng_snn.py` ONLY (`NGSNN.fit` and `prune_unearned` route through the lifecycle). G3 edits NO line of
  `growth/neurogenesis_engine.py` (R2 holds its claim): the percept consumer's byte identity is checked through its
  existing `growth_fingerprint` + `trajectory` on `SyntheticPerceptStream` 9000-9003, shipped as one pytest file the
  later stacked commits re-run. After P2, in windows sequenced among them and main, each as its own stacked commit G3
  reviews for that proof: R2's default-None `transition_sink` + hydration (inert with no lake attached), R3's
  `replay_rows`, G7's additive persistence arrays and `origin="derived"` on restore.
- **P3** the loop, the budget, the ledger and the unit journal: NEW `growth/growth_loop.py`,
  `tests/test_growth_loop.py`; EDIT `growth/growth_stop.py` (problem key), `growth/learning_progress.py` (GoalStore:
  the "problem" and "unit" kinds, the "sleep_save" API for G6, unknown-kind pass-through).
- **P4** the growth unit, T1 and Astra arms, the campaign instrument: NEW `growth/column_batch.py`,
  `instruments/growth_campaign.py`, their tests; `growth/placement.py` and `gates/unit_label_gate.py` are G7's (called,
  never edited).
- **P5** the feed: NEW `growth/grown_feed.py`, `tests/test_grown_feed.py`; EDIT `reservoir/recurrent_pool.py` (a
  declared input port) after G1(c) lands.
- **Lead read 2026-10-03 -- D8 and L5.** `growth/placement.py`'s board claim is G7's (its
  claim file names G3 only as the party to the agreed extension); G3 never edits it and calls `place_labeled(...,
  roles=...)` once G7 lands it. P1's port waits for R2's P1 on `readout/interaction_learning.py`, confirmed by the
  OWNERS.md landing order (R2 P1 -> G3's port, one upcaster chain). `column_batch.py` (P4) carries no E/I role (2.7).
  C17's clip lands with the port (P1) on the grown segment and with `grown_feed.py` (P5) on the reservoir segment.
- P8 owes: `docs/INVENTION_REGISTER.md` rows for the three inventions (the lift-certificate trigger, the
  trace-trained candidate source, the problem-keyed budget pause), each with its red and its control, in the arc that
  lands it; `gates/growth_coverage_gate.py`'s census re-derived for every new growth call site.
- Each arc: registrations in the commit that adds a module, census after symbol-adding commits, a Codex gpt-6.1-sol
  panel (the orchestrator sends it), fold stated as re-panelled or battery-verified. Then the campaign, the results
  record, ONE battery on the stacked head, the closing record last.

### 2.16 Scope

Built: the items of 2.15. Not built: live wiring into `make_brain()` or the daemon (G9's engine); sleep's save and
prune (G6); readout neurons (G5); a growth target inside the reservoir (owner); [ANSWERED 2026-10-29, see DATED ADDENDUM 2026-10-29 (owner growth target)]; a distance kernel (G4); the label gate
(G7); any problem construction (G2).

## 3. Open questions for the owner

- O1. "k = 0 on clean data" read as the contradiction count beside an exact Farkas witness (2.4) -- confirm.
- O2. Retry rule (values in G2's Problem record): sleep draws no fresh budget; new teaching draws one; K = 3 budgets
  per problem lifetime.
- O3. The first problem's budget = the frozen code's width (52 units), P1 case (a) (G2 accepted the value).
- O4. One stop rule: the problem-keyed pause merged into `GrowthStop` (vs argued as a different job).
- O5. The float32 1e12-bonus tie in `NGSNN.fit`: fixed in G3's lifecycle arc as its own commit, or left to a grower arc.
- O6. RESOLVED in debate (D-G6): one record class, GoalStore's `_T_GOAL` (G3's codec and kinds; G6's sleep kinds).
- O7. T1's burden direction (G7's rule): the column must SHOW it needs < 20% more neurons and forgets no more, and a
  red deletes the column part of C -- so an underpowered T1 deletes it. The owner should see that before the freeze;
  a T1 S > 8 needs a Part F amendment (only G1 may pre-register S > 8 today).
- O8. The LIVE percept grower's units live only in the compactable `_T_GROWTH_DELTA` chain (survivors only), short of
  P4d's "journaled when grown ... never its lake record" for units with no posed problem. G3 does not change the live
  grower (byte-identical); reported to main for a row to own.

- O9. G2's held-out reads are never-taught presentations of the same finite items, so for capacity units a validation
  solve entails the future solve. G-R3's "prospectively" is read as "learning OFF, on presentations selection never
  saw"; generalization to unseen items is out of scope for G2 and G3. Confirm, or ask G2 for novel-item problems.
  **Lead read 2026-10-03:** narrowed by L1 -- the primary cell C1-d judges on novel
  distractor COMBINATIONS (a real held-out split); novel WORDS are unreachable by construction; O9 now asks the owner
  only to confirm "novel combinations, not novel words" as G-R3's reading of "prospectively", and the same-items
  reading for the secondary cells.
- O10 (= G7's O9; lead read 2026-10-03). T1-size has no reading on G2's problems (granularity
  guard, 2.10): may T1-struct's retention contrast stand in for the record's T1 while no problem class reaches median
  U_SINGLE >= 80? Until answered, T1-size is "untested" and nothing is deleted on its account.
- Session calls disclosed by the lead-read and Codex folds (for the owner's information): p_c = softmax(v / T), T = 1 (D4); the
  attempt cap A_max = 3 (D3), also binding UNRESOLVED; the guard threshold 80 / 40 (T1-size); F_bound = incumbent
  mean + 2 sd on dev 9000-9015 and delta_NI = max(F_bound, one item) (D7); ruling 9 / 9a for degenerate samples
  (superiority NOT MET; non-inferiority by the common value); the restricted score's failure penalty = the next grid
  size above the budget and the ">= 6 of 8 uncensored solves" condition for the words "fewer neurons" (2.8); C9's
  comparison population = C1-d's distractor input sources with the matching tolerance (norm ratio [0.5, 2], rate
  difference <= 0.25) and its feasibility rule (2.9).

## Review record

- **Lead read:** 2026-10-03, CLI lead session 2d85b496; board `p0_reports\G3.md`, section "=== LEAD READ ===" (L1-L5),
  and the orchestrator rulings 9, 9a, 10, 12 and 14 in the board's `DEPENDENCIES.md`.
- **Advisor read:** CLI lead session, Fable, 2026-10-03 (relayed through the lead's read).
- **Codex critique:** model gpt-6.1-sol; reply `codex_p0\b1.reply.txt` on the lead's board scratchpad (critique
  extract `codex_p0\b1.critique.md`, section "G3", findings 1-12: 4 BLOCKER, 7 MAJOR, 1 MINOR), sha256 of the reply
  file `52b335d74ef7dea045a21b3e68ae15d4fb286c5bdd7f3ebec66f2e2e843c8f4c`. Disposition, each folded in the text above (the fold is itself unreviewed: lead-read and
  Codex-text-checked only, no re-panel):
  1. BLOCKER C9 population: FIXED (2.9 "C9's comparison population"): C1-d's distractor input sources, matched, per-item
     effects, feasibility check on dev 9000-9007 before confirmation. The lead's two suggested populations were REFUSED
     with reasons (disjoint buckets make prior-problem units silent on B; presized-frozen units are another network).
  2. BLOCKER C10: FIXED (C10 row): diagnosis, candidate budget, selection and admission schedule held constant; endpoint
     = held-out (future-combo) score, true- vs shuffled-label-trained candidates, superiority.
  3. BLOCKER T1 / G4 T3: FIXED (2.10, 2.13): the T1 family waits on T3 at family level; no kernel-free exemption for T1.
  4. BLOCKER isolation: FIXED (new 2.12a): one isolated home / lake / output per (seed, arm), passed explicitly to
     children, live lake refused, every paired arm from one digested ledger snapshot, the sequence experiment separate;
     reds added to C13's fixture.
  5. MAJOR zero NI margin: FIXED (D7, 2.10): F_bound (factual, may be 0) vs delta_NI = max(F_bound, one item) > 0;
     "not shown non-inferior" wording; sent to G7.
  6. MAJOR censoring: FIXED (2.8): a restricted solve/failure score, penalty = next grid size above the budget (54 / 53),
     no conservatism claim, "fewer neurons" only with >= 6 of 8 uncensored solves in both arms, solve rates and observed
     sizes reported separately.
  7. MAJOR exposure: FIXED (2.8): presized arms replay the targeted arm's actual recorded teaching sequence (no-growth
     attempts and early ends included); candidate-training exposure matched separately.
  8. MAJOR undefined diagnosis: FIXED (2.4, 2.6): UNRESOLVED branch, no growth, attempt cap, persisted terminal state;
     every no-growth path finite.
  9. MAJOR C16 decoys: FIXED (C16 row): provably incapable planted units, verified by the exact certificate, as the
     specificity red; random decoys a measured comparator.
  10. MAJOR C18 lr = 0: FIXED (C18 row): planted verified-incorrect reads, every plasticity path disabled and asserted by
      byte identity, unsolved state asserted first; an UNRESOLVED twin added.
  11. MAJOR D6 bound: FIXED (D6): no learner probability bound is registered or printed.
  12. MINOR C8(a): FIXED (C8 row): pause when the next whole batch would exceed the budget, spend <= budget.

## 4. Debate log (wave 1, 2026-10-03; every resolution agreed by message, recorded on both sides)

| id | sender | challenge | answer | resolution |
|---|---|---|---|---|
| D-R3-1 | R3 | sequencing of R3's `replay_rows(rows, *, source_ids)` on `NeurogenesisSelfImprove`; does `growth_loop` supersede `_refit` | G3's P2 (byte-identical) lands first; R3 adds `replay_rows` as its own stacked commit G3 reviews; `_refit` stays the percept entry | agreed (2.5, 2.15) |
| D-R3-2 | R3 | home of a silencing helper both rows need | `candidate_lifecycle.silence_units(consumer, unit_ids)` + its matched-control picker | agreed (2.5) |
| D-R3-3 | R3 | P2: is `sleep/lifelong_consolidation.py::LifelongConsolidator` (live grow + replay + prune-to-capacity) a growth incumbent G3 owes | different job for G3 (ART prototype recruiter, capacity ceiling, no posed problem, no solved predicate); R3 runs and adjudicates it; if R3 finds it the same job, G3's loop is the merge target | conceded as a disclosure (1.2a) |
| D-R3-4 | R3 | C9's equal-count control drawn at random can include dead units and pass vacuously | conceded | C9 now norm- and rate-matched, dead units excluded, too-small pool halts |
| D-G6-1 | G6 | does G3's loop change `prune_unearned`'s signature or location | no: unchanged for the percept consumer; G3's problem loop never prunes; the lifecycle exposes a consumer-agnostic one-pass prune for G6's arms | agreed (2.5) |
| D-G6-2 | G6 | one problem record, whose | G3's GoalStore `_T_GOAL` extension; G6 drops its separate problem tag; G6's kinds "sleep_bout" (renamed from "sleep": R2 uses `_T_EPISODE` kind "sleep"), "sleep_verdict", "sleep_save" on G3's class, unknown-kind pass-through | agreed (2.6) |
| D-G6-3 | G6 | FINDING: the `_T_GROWTH_DELTA` chain is positional and COMPACTABLE, so it cannot hold a pruned unit's only record; a digest + ref cannot rebuild | conceded | "unit" rows carry the weights themselves, stable unit_id, IMMORTAL; live-grower gap -> O8 |
| D-G6-4 | G3 -> G6 | G6's feasibility read "G3's retention bound verbatim" while it is a projection | G6 conceded | G6 depends on `retention_ok`'s FREEZE |
| D-G6-5 | G6 | the solve event's symbol | `GrowthLoop(..., solved_sink=None)`, one call per OPEN -> SOLVED | agreed (2.6) |
| D-G7-1 | G7 | who edits `neurogenesis_engine.py` persistence (additive arrays, `origin="derived"`) | G7, as its own stacked commit after G3's P2, G3 reviews | agreed (2.15) |
| D-G7-2 | G7 | gate API and call point | `require_admissible(units, *, resolve_problem)` at ADMISSION, sham included, presized not gated; labels only from `place_labeled` | agreed (2.7) |
| D-G7-3 | G7 | problem_id | G3's ledger key == G2's `Problem.id`, stable | agreed (2.7) |
| D-G7-4 | G7 | SINGLE needs a position; arms must isolate the unit | one site rule, one targeting rule, unit size the only difference | agreed (2.7) |
| D-G7-5 | G7 (R-a) | U must count the shared inhibitory unit | conceded | 2.7 |
| D-G7-6 | G7 (R-b) | T1's decision direction ambiguous under Part F | G7's burden-on-the-column rule adopted (G7 owns T1) | 2.10; O7 |
| D-G7-7 | G7 (R-c) | censoring at budget + batch compresses the T1 ratio | conceded for T1: > half censored -> REFUSED; for the primary, censoring disclosed as conservative against targeted | 2.8, 2.10 |
| D-G7-8 | G7 (R-d) | PRESIZED on 9100-9107 is unpaired with T1 on 9300-9307 | conceded: PRESIZED also on 9300-9307; T1 pilot 9040-9047; G7 releases 13110-13127 pending main | 2.10, 2.12 |
| D-G7-9 | G3 -> G7 | an S > 8 for T1 is not a pre-registration line | G7 agreed: a Part F amendment request to main | O7 |
| D-R2-1 | R2 | both rows edit `SpikingInteractionLearner` (constructor, checkpoint) | R2's P1 first with one upcaster chain; G3's port is the next schema step; main sequences the claim | agreed (2.3, 2.15); main RULED the order 2026-10-03: R2 P1, then G3's port |
| D-R2-2 | R2 | redo determinism with the port ON | the segment is a pure function of (frozen code, units at `grown_version`) | agreed (2.3) |
| D-R2-3 | G3 -> R2 | refusing every event whose grown version differs would make all of task A ineligible for replay after growth | split slot: `encoder` (frozen) keys eligibility; `grown_version` beside it; redo version-locked by log position, replay through current units | R2 conceded; R2's reader reads `_T_GOAL` "unit" rows in one ordered pass (2.3) |
| D-R2-4 | R2 | redo needs unit rows at lake positions consistent with the teaching events, and a pure version step | rows written synchronously at admission (write-ahead); admitted input weights frozen; `grown_code.apply_unit_row(learner, row)`, no RNG | agreed (2.3) |
| D-R2-5 | G3 -> R2 | (from G2's round) memory-kind units are episode-stateful, stepping on context-turn reads | a redo under a memory unit needs the episode id and turn order | OPEN by agreement: when G3 pre-registers the state branch it sends R2 the read call site and the state memory units carry across reads; R2 then adds a context-read event kind to its `learner_events.py` registry, or memory-unit deposits are excluded from redo eligibility |
| D-R2-6 | R2 | R2's `transition_sink` + hydration in `NeurogenesisSelfImprove.observe` (R2 holds that file's claim): collision with G3? | G3's P2 now edits NO line of `neurogenesis_engine.py` (only `ng_snn.py` + the new lifecycle); R2, R3, G7 sequence their windows there; G3 reviews for its byte-identity proof, which requires R2's hydration inert with no lake attached | agreed (2.15); R3 concurs |
| D-G5-1 | G5 | reword "G5's readout neurons replace it" | conceded: G5's tuning-rule arms run on the learner's label neurons; a rule replaces R-STDP only by winning G-R5 | 2.10 |
| D-G5-2 | G5 | port interface: segment tags + ablation mask, source unit_id -> G7 label, unit_id order; a teaching-signal argument and a drive accessor | (1), (2), (5) in G3's port; (3), (4) are G5's own default-off step on R2's chain (G5 agreed) | 2.3 |
| D-G2-1 | G2 | the Problem interface; learning OFF freezes plasticity, never state (context turns arrive by `route` reads) | conceded: G2's `taught_problems.Problem`; memory units step on reads | 2.2, 2.3 |
| D-G2-2 | G2 | (a) BASELINE COLLAPSE: A solves with zero growth, the median of solved spend is 0, B's budget 0; (b) two registrations of the budget numbers is additive (P2) | both conceded: baseline over positive-spend solves only; values live in G2's Problem record, G3 owns the mechanism; G2's budget >= N_codes check adopted | 2.6 |
| D-G2-3 | G2 | task A: same texts relabelled is a contradiction; which A, shared or disjoint intents | A_ii for B = (ii); (ii)'s twin for B = (i) S0C1; shared intents primary, disjoint beside | 2.2 |
| D-G2-4 | G2 | where the unsolvable cell U lives | G2's module (one problem module); G3's C4/C8(a) consume it; C8(b) is a budget override in G3's red fixture only | 2.9 |
| D-G2-5 | G2 | S1C1 needs state AND capacity; the three-way rule reads state only | conceded: the certificate also runs on the history-augmented code in the same pass | 2.4 |
| D-G4-1 | G4 | who calls the attachment rule; scale; no kernel before T3; T2's endpoint and seeds | grown units only (NGSNN dense rows byte-identical); scale is G7's (native um sheet); agreed; U on G4's 10400-10407 through `growth_campaign` with the prior as an arm argument | agreed (2.7) |
| D-G2-6 | G2 | G2's draft written; one refute round (its 2.2, 2.6-2.9) | read; answers below | -- |
| D-G2-7 | G2 (a) | is E = 400 the attempt length G3 wants | yes, with early end at G2's stable-10 solve; growth granularity is the batch's. G3's own finding while answering: presized arms must get the targeted loop's CUMULATIVE exposure (m x E at m batches) | 2.6, 2.8 |
| D-G2-8 | G3 -> G2 | validation and future are presentations of the same items, so for a stateless deterministic reader a validation solve entails the future solve | disclosed in G3's 2.8; G2's 2.6 states it (G2's O1 = G3's O9) | O9 |
| D-G2-9 | G2 (b) | should PRESIZED FROZEN reuse `presized_expansion` (P2: one frozen random expansion) | refuted for G3's arm: it must share the growth arms' architecture (port + lifecycle random source) or D confounds architecture with timing; raised back to G2: `presized_expansion` may itself duplicate the tree's existing random-feature expansion (NGSNN feature mode) -- draw from it or declare it a control instrument | G2 conceded the question, settled by measurement before its freeze: NGSNN feature mode as the presized front-end on dev 8000-8003; adopted (presized_expansion a thin adapter) if it meets G2's B7 on all four, else a declared CONTROL instrument with the measured reason; G3's arm unchanged |
| D-G2-10 | G3 (own finding during G2's round) | in a one-problem campaign `beta x baseline` never runs, so its red is unreachable there | a secondary SEQUENCE reading ((ii) then (i) S0C1) plus a ledger unit check | 2.8 |
| D-G2-11 | G3 -> G2 (d) | G2's B6 requires explicit history NOT to solve S1C1-with-history on every seed, a capacity bound exact only in integer arithmetic; B4 treats the same float caveat as a reported measurement, B6 makes it a clause-4 RED | sent: treat B6's second clause like B4 (a float solve at a near-tie is reported, with the exact-arithmetic check beside it) or register why not | G2 conceded: a capacity-cell solve by the incumbent or by explicit history reads "construction wrong" with B2(c)'s witness margin; clause 4 keeps only "history fails S1C0"; a solve of an exact-bound state cell reads HALT |
| D-G4-3 | G4 (settled with G2) | T2's positive control needs no new problem: B = an existing capacity cell, planted prerequisite units = G2's `witness_units`, filler = `decoy_units`; "the bare learner can solve B once it reads those units" is the positive control of G3's port | accepted | C16 registered; it is also T2's precondition |
| D-G4-2 | G3 -> G4 | T2 on G3's problems may be insensitive (grown units read the frozen code directly), so a null could delete the map part of C on an instrument's blindness | G4 conceded: a spatially clustered positive control first (10410-10417), REFUSED if it does not separate; the problem is G2's, the cluster-planting site hook is G3's `growth_campaign` | registered in G4's draft |

Probe safety (main's PROTOCOL 4b, received mid-draft): G3's two scratch probes built no Brain (a bare
`SpikingInteractionLearner`, `placement.place`, scipy); `C:\Users\vinhl\.snn_data_lake\brain_lake.bin` mtime is
2026-09-27 01:19, unchanged.

## DATED PRE-DATA AMENDMENT 2026-10-04 (G7's asks; no data read)

Frozen passages above are unchanged; each line below adds to the section it names. Source: G7's Codex fold (b3 #8/#9),
agreed by message 2026-10-04.
- (2.7, 2.15 P1/P4) `admit_rows(rows, ticket)` is the SINGLE append point of the grown segment: it refuses any row whose
  unit_id the `AdmissionTicket` returned by G7's `require_admissible` does not cover, and a ticket is consumed once, so
  the label gate precedes every append by construction. Red: an append without a ticket, with a ticket not covering a
  row, or with a spent ticket must raise.
- (2.6 unit journal) `known_unit(unit_id) -> bool` reads the IMMORTAL `_T_GOAL` "unit" birth rows; G7's gate refuses an
  id already known (a duplicate across batches) or repeated within a batch. Red: re-admitting a journaled id must be
  refused.
- (2.6, 2.7 restore) `classify_restored(rows, journal) -> {unit_id: origin}` runs at restore (restore never calls
  `require_admissible`): born iff a journal birth entry exists AND the restored label equals it field for field;
  derived iff the full label set is present and NO birth entry exists; legacy iff the full label set is lacking. A
  restored row that HAS a birth entry but a DIFFERENT label is REFUSED as a tampered born unit, naming the unit and the
  differing fields -- never classed derived (G3's refusal, G7 agreed as D21). The persisted origin array is a
  recomputed cache, never evidence. Reds: a planted label change on a born unit must be refused; a planted origin array
  that disagrees with the journal must be overwritten by the recomputation.
- (2.9) G7's census reds any derived or legacy row in a grown segment that has no open debt row; G3's segments are
  registered as GROWN_SEGMENTS for it.

## DATED PRE-DATA AMENDMENT 2026-10-06 (ruling 18 / D-R2-5: the state branch; no data read)

Frozen passages above are unchanged; this adds to 2.3's "Learning OFF freezes plasticity, never state" paragraph and
to 2.15 P4. It answers R2's wait (ruling 18): any R2 verdict covering grown MEMORY-role units waits on THIS text and on the
reconstruction test below passing.
- **What state is carried.** A grown MEMORY-kind unit (a `growth/neuron_zoo.py` kind drawn by the STATE branch, 2.4/2.5) keeps
  its own dynamical state across the CONTEXT turns of one episode (its membrane and, for ALIF, the adaptive threshold; for the
  autapse kind, its self-connection state; the exact variables are the kind's, named in the unit row at admission). The state
  is reset to the unit's birth state at an episode boundary and is a function of (the unit's parameters, the episode's turns
  so far, in turn order) and nothing else: no RNG, no wall clock. Capacity-kind units carry none (2.3, Determinism).
- **The read call site.** The learner's two entry points that present a turn: `SpikingInteractionLearner.route_text(utterance)`
  (readout/interaction_learning.py:615, a context-role or probe read) and `learn_turn(utterance, firewall_intent, feedback)`
  (:621). Each steps the memory units exactly once per turn through ONE hook added with the P1 port (none exists on this
  branch today; the hook is the only writer of the unit state).
- **The context-read journal.** One row per stepped turn, written BEFORE the step (write-ahead, as the "unit" rows are):
  `{episode_id, turn_index, encoded_ranks_digest (sha256 of the turn's `text_features` n_in=52 code), grown_version}`. It
  is the event kind R2 adds to its `learner_events.py` registry (D-R2-5's first branch); this amendment fixes its fields,
  R2 owns its schema and reader (one schema owner, P2). Bytes are accounted against the 0x0B/learner-event append budgets
  at the campaign, not assumed.
- **The reconstruction test (owed before any R2 verdict on memory units).** `grown_code.reconstruct_state(units, context_rows,
  episode_id)` rebuilds every memory unit's state at an episode's last turn from the journal alone, and its result must equal
  the live state EXACTLY (float64 equality) on G2's (i) delayed-cue fixture episodes. Reds: (a) a planted DROPPED context row
  must be detected and the episode refused as `stale_state`, never reconstructed approximately; (b) two planted swapped
  turn_index values must be refused; (c) a row whose digest differs from the replayed turn's code must be refused.
  Control: the untouched journal reconstructs equal.
- **The interim rule (until R2's kind and this test have landed).** Deposits made while a memory unit is live are EXCLUDED
  from redo eligibility, counted per episode and reported once, never silently re-derived (ruling 18's text). D-R2-5's
  second branch, so stated, is the fallback and not a choice: if R2 declines the kind, the exclusion stands and R2's verdict
  on those units reads NOT RECOVERABLE.
- **Scope.** No memory unit is grown in any confirmatory run before this lands; C5 only checks that the STATE branch DRAWS
  memory kinds. This amendment decides no threshold and reads no data.

## DATED PRE-DATA AMENDMENT 2026-10-06 (b) (cp5 fold disclosures; no data read)

- **0x0B "baselined" is not "bounded".** `tools/lake_append_baseline.json` tag 11 (13,959 B) is one fixture burst, a SEED p95 for the
  per-TICK flood check only (a 9-unit batch burst against 8x that bar). It does not bound the immortal class's CUMULATIVE growth
  (52 weight-bearing rows per problem, forever); owner decision 13 stands: report the measured rate at the first daemon restart
  before calling the growth acceptable.
- **A run is a run only once it journals.** The "run" ledger row (2.6 retry rule) is written at a loop run's FIRST journal write
  (a spend row or an attempt row), never at its start. A run refused before any write (a wrong-size batch with no evaluations, a
  repeated id) leaves no run row, so the next run still draws growth under the same budget; a refusal that already paid
  evaluations journals them (spend units = 0) and counts as a run, so the next run under that budget is a retry.
- **Ids are not coerced.** The ledger refuses a non-int, bool or negative `unit_id`/`batch_id` (the same types G7's gate refuses).

## DATED PRE-DATA AMENDMENT 2026-10-06 (c) (the context-read row's types, agreed with R2; no data read)

Adds to the state-branch amendment above; nothing is removed. Agreed with R2 (kind `ctxread`, registered in R2's `learner_events.py`):
- Slot types: `episode_id` = the ASCII-decimal learner seq of the episode's first event; `turn_index` = ASCII decimal, 0-based within the
  episode; `grown_version` = ASCII decimal of `grown_code.grown_version`'s unsigned crc32; `encoded_ranks_digest` = a lowercase hex
  sha256 token (the one non-decimal slot). The row shares `Brain._learner_seq` and the boot's session with the other learner events.
- The row is written BEFORE the step, through R2's sink `Brain._record_ctxread(...)` (G3's memory-step hook calls it; G3 never writes
  the lake directly) and inherits the `split` and `valid` of the turn it journals, so a held-out turn's row is held-out.
- R2's reader refuses only per-event reasons (missing_slot, duplicate_id, seq_regression ...); a SWAPPED turn_index or a dropped row is
  `reconstruct_state`'s red, not the reader's (the test says so).
- Cost: one more immortal row per stepped turn of a live memory unit; it joins decision 13's claim-with-a-check (R2 reports the
  measured B/turn at the first daemon restart).

## DATED PRE-DATA AMENDMENT 2026-10-06 (d) (a stated crash window; no data read; G7's residual)

A process crash between a batch's birth rows and its run/spend rows leaves born units journaled with no spend row and no run row (write-ahead
order: birth rows, run row, spend row, then the learner's admit). This is the same window the spend row already had; the run row is
deliberately NOT moved ahead of the birth rows, because a refusal inside `record_units` (a repeated id) must leave no run row (2.6, amendment
(b)). Recovery is a ledger/journal reconciliation, not a rule: the next fold reads birth rows with no spend as an unsettled batch, to be
reported, never silently re-derived. No reconciliation code exists yet; the window is disclosed, not closed.

**OPEN ITEM (carried, 2026-10-06, agreed with G7):** a reconciler for unsettled batches (birth rows with no spend row, amendment (d)) is owed by an
arc that wires the loop live (G9's engine); until then the window is stated, not closed. No task of this row is DONE while this item is open.

## DATED PRE-DATA AMENDMENT 2026-10-06 (e) (what the gate integration is and is not; G7's census findings; no data read)

- **No production adapter yet.** `GrowthLoop` hands `learner.admit(units, labels, ticket)` a ticket and expects the adapter to append
  through `grown_code.admit_rows`. The only `admit` implementations in `src` today are the self-check's scripted learner and test
  doubles; the real adapter is the learner port on `readout/interaction_learning.py`, after R2's P1 (OWNERS landing order), and the
  engine wiring is G9's. Until both land, the single-append row is satisfied by declaration plus a charge test on `admit_rows`
  itself, and the gate carries a dormancy EXEMPT 'wire' row (lead ruling 29) that retires when G9 wires the engine.
- **Pre-validation (moved, not stated).** `grown_code.apply_unit_row` is dry-run on every unit BEFORE any birth row, so a refusal
  the segment would make (non-finite weights, Dale on outputs) writes nothing and pays only the search's evaluations; the labels'
  ids/signs are cross-checked against the units in `Ledger.record_units` before any write. After the birth rows a refusal is not
  reachable from this loop's own admission.
- **Ticket enforcement is detection, not prevention.** The loop cannot un-admit; when G7's read-only `AdmissionTicket.consumed`
  lands, a ticket left unspent after `learner.admit` raises `LedgerError`, loudly, after the fact.

## DATED PRE-DATA AMENDMENT 2026-10-06 (f) (the state branch is PARKED; lead decision B; no data read)

Ties to the state-branch amendment (98d92203) and its ctxread row types (21ca1bc7). Nothing in them is removed or changed.
- **Parked:** `grown_code.reconstruct_state` and the memory-step hook (the one writer of a memory unit's state, calling R2's
  `Brain._record_ctxread`, 97acf5df on claude/r2). No memory-unit STEP is defined anywhere (grown units are stateless latency units;
  nothing fixes what current a context turn feeds a stateful kernel, how many steps per turn, or its reset), so any step is a new
  design decision to be pre-registered first, and no confirmatory run grows a memory unit this wave: a reconstruction with no
  consumer is speculative.
- **What unparks it:** the first confirmatory run that grows a MEMORY-role unit, or G9's engine arc, whichever comes first; then a
  dated amendment defines the step, G7 and R2 refute it, and the reconstruction test (with the three reds of 98d92203) is built.
- **Meanwhile:** the interim exclusion rule of 98d92203 stands: deposits made while a memory unit is live are excluded from redo
  eligibility, counted and reported; R2's verdict on grown MEMORY-role units stays excluded (NOT RECOVERABLE-excluded).

## DATED ADDENDUM 2026-10-07 (P5 build disclosures and a corrected measurement; no confirmatory data read)

- **Branch and commits.** `claude/g3-p5` from batch 1 (8ea4d59c). `bdf7d8a2`: `_Pool.percept_fed(x, feed)`, the declared
  input port on `reservoir/recurrent_pool.py` (one external drive per pool unit added to `W_in x`; a zero feed is
  byte-identical to `percept`; a wrong-length or non-finite feed is REFUSED). `57cb62c5`: `growth/grown_feed.py`
  (`feed_drive`, `clip_feed`, `check_feed`, `random_feed`) and `tests/test_grown_feed.py`; the module is registered in
  `snn_data_lake/__init__.py` and takes a dormancy EXEMPT `wire` row (consumers: the feed arms of P4/P5 and G5's Part D;
  retires when G9 wires the engine). `71aee53f`: `candidate_lifecycle.select_unearned`, the one pure unearned-unit
  rule (G6's P2 finding, settled by message): `heldout_prune` calls it, behaviour identical.
- **What 2.9 C15 and C17 are, as built.** C15: the norm-matched random-feed arm (per-column L2 norm and Dale sign kept,
  seeded) must give different pool carriers from the fed arm; the instrument's positive control is that a copy of the fed
  matrix gives IDENTICAL carriers and a no-feed arm differs. C17 on the reservoir segment: `clip_feed` flips a sign-violating
  deposit to the unit's side without mutating its input; `check_feed` refuses a violating or non-finite stored synapse and
  names the unit and the pool unit. Constructed fixtures on the pre-registered 9900-9999 block only; the confirmatory feed
  family (9200-9207, reserve 9208-9215) is NOT run and waits for the lead's written go.
- **Scope of what is built.** The port, the projection and its two reds. NOT built, as pre-registered: the fed / direct /
  random-feed READOUT contrast of 2.10 (the U contrasts need G2's problems and the taught learner's label neurons) and the
  N > 256 fed-unit-becomes-reservoir arm (owner ratification, POLICIES P4d) [ANSWERED 2026-10-29: see DATED ADDENDUM 2026-10-29 (owner growth target) at the end of this file.] No claim about feed benefit is made.
- **A fixture is not a measurement.** The tests run on a hand-sized KernelLsmPool (N 3) and W_in of 0.01; they prove the
  wiring and the reds, not the engine's N = 256 behaviour. The pool's own N = 256 determinism red was not re-run by this
  arm beyond `tests/test_recurrent_pool.py` passing.
- **Correction (the 0x0B append baseline).** Fold 18321afc pinned tag 11 at 15,241 B from "seed 9040, 5 batches:
  15,238 / 15,223 / 15,235 / 15,241 / 15,201 B", but its script was not recorded, so the five numbers had no command. Re-run
  from the test's own construction (random.Random(9040), one rng and one id counter across batches, 9 units per batch,
  `Ledger.open_problem`, `_born` + `record_spend`, bytes = file size delta per batch), twice, deterministic:
  15,236 / 15,216 / 15,240 / 15,242 / 15,197 B. The lead reproduced it and re-pinned tag 11 to 15,242 (int1b 732a72e8). The
  fold's figures are HISTORICAL; the flood test's 8x bar is unaffected.

## DATED ADDENDUM 2026-10-07 (b) (G2's confirmatory (i) read; gradient declarations; no G3 data read)

- **C11 for B = S0C1 is UNREAD, never MET, until Q-AII resolves.** G2's confirmatory (i) (8200-8207, claude/g2 e6a4a75e,
  record 5.1) has the canonical incumbent solving task A in C1-d's shape 8/8 (`inc.ac1d`; lr 0 never solves it, so it is
  learned) but the curriculum-A twin for B = S0C1 (`inc.aii`) 0/8, parked as Q-AII. The retention row (task A for B,
  2.9 C11, option (a) of ruling 15) therefore has no readable A for S0C1; C11 on C1-d is separately gated by G2's
  `pre.c1d` 0/8 (REPRESENTABLE, NOT SHOWN LEARNABLE). Neither is relaxed or relabelled here.
- **What G2's 8/8 is not.** `pre.S0C1` is a presized frozen random-feature expansion (H = 64) feeding the canonical
  learner: no grown segment, no witness units, no decoys. It supports only "a nonlinear layer of H = 64 solves S0C1"; it is
  NOT C16's grown-segment-masked-off arm, which stays G3's own unrun cell.
- **Gradient-claim declarations (dfedf10c).** `growth_loop.trace_train` is declared LOCAL (own PSP eligibility trace x one
  broadcast scalar from a FIXED random vector drawn at birth; no transposed forward weight; this pre-registration states the
  update and never claims it tracks a gradient). Accepted by the lead. The guard stands: the day any text says it tracks a
  gradient it is reclassified a claimant and owes the cosine > 0.5 bar. `capacity_certificate._constraints` is NOT_A_RULE.

## DATED ADDENDUM 2026-10-07 (c) (batch-2 panel record; no G3 data read)

- **Review status of the batch-2 range bdf7d8a2..e61ac66e (additive; nothing above changes).** Codex gpt-6.1-sol read e793afc9
  (correctness + P8 lens: 1 MAJOR, 2 MINOR), 84b1f840 (p2p3 retry: 3 MINOR) and 150daa76 (safety retry: 2 MINOR + the 15,241 B baseline already
  re-pinned on int1b 732a72e8). Folds, each red-first on the parent: 989e5b00, be6da35d, 84b1f840 (correctness); 79a36615, 150daa76
  (p2p3); e4529118, e61ac66e (safety). 6fed31ed (G7 finding 4: the loop checks the segment grew by exactly the admitted units) lies
  outside every panel SHA. e4529118 and e61ac66e were never re-read by Codex: **battery-verified only** (tests/test_grown_feed.py,
  test_growth_loop.py; dormancy_gate and method_dormancy_gate `--assert` green with grown_feed importing growth_loop). 6fed31ed is
  the same.
- **Addendum (c), later the same day.** The `growth.grown_feed` `__main__` self-check and its run_selfchecks CORE registration
  (91a1cc8e, with the E501 comment fix dc843042) were added after every panel and are **battery-verified only**; Codex never read them.
- **Corrected wording (Codex int2 closing report; additive).** Zero-feed identity is verified on the N=3 fixture only (identical output,
  carrier bytes and fingerprints), not at N=256. Finite `random_feed` output requires a REPRESENTABLE column norm: a finite, sign-valid
  column whose L2 norm exceeds the float max is REFUSED, not scaled (the batch-2 fold that fixed it is battery-verified only). The
  adapter refusal in `growth_loop` (6fed31ed) DETECTS corruption after admission and cannot undo it.

## DATED PRE-DATA AMENDMENT 2026-10-15 (row-plan 1.0 against fbb1f38e; no confirmatory or reserve data read)

### A1. Precondition: QUALIFIED, not a certificate of seed non-exposure

Amends 2.1-2.16 and the dated disclosures only as specified below; every earlier byte is retained. Basis: G3 row-plan
1.0 REPORT.md and census.txt, relayed at #2752 and independently reviewed at #2760 and #2781. Source claims below
were checked by this lane's own git grep/show at fbb1f38e, against registration/build tree 5e608d6c where needed.
The latter's 2026-10-08 branch addendum is supplemental history, absent from this evidence on main; it is not a
replacement registration or a completed experiment. Main's old DRAFT header is historical, not a new licence.

The heading means this lane read no confirmatory/reserve outcome or newly generated protected-seed construction.
The supplied two-tree TEXT audit (src/tests/tools/specs) found no G3 trained-arm or deciding campaign result.
It does NOT prove that no protected seed was read. `g7-unit-label-gate.md`, amendment (f), "T1's judging set",
records, verbatim:
"Future items counted in distinct items, reproduced at seeds 8000 and 9300" using
`tp.PROBLEMS[id].items(seed, "future")`. This is recorded construction-only exposure at T1 seed 9300, not a trained
T1 result. It requires an integrator provenance disposition before that family can be called pristine.
`test_the_batch_seed_never_leaves_the_g3_block` contains seed 9300/batch 7 -> placement seed 9307; the feed tests
contain protected numbers as refusal inputs. Test text alone establishes neither execution nor outcome exposure.
The old campaign's `preflight(seeds, go=...)` accepted a caller-supplied go; missing main code cannot exclude its use.
External rows/, binary lakes/checkpoints, untracked logs, other clones/machines and intermediate execution artifacts
are UNEXAMINED. This lane did not audit them. The supplied audit is a qualified basis for prospective documentation,
never permission to rerun an exposed cohort or read a protected family. Fresh provenance/go remains required.

ORDERING DISCLOSURE: this is pre-outcome on that qualified basis but POST-implementation for the landed adapter,
chat-grown/return carriers, reconciliation, latch and bout paths. It cannot retrospectively precede their build/tests.
In particular the #1037 prospective-amendment-before-new-checks ordering was breached by the already-landed latch
implementation and tests; this amendment records the breach, not a repaired chronology or a test result.

### A2. Changed or absent clauses: mechanism actually landed

Symbols below belong to src/snn_data_lake at fbb1f38e unless a historical tree is explicit. Presence is not charge.

- **2.1/2.16 scope.** `growth_loop.GrowthLoop` now coexists with `CanonicalGrowthAdapter`,
  `readout/posed_problem.PosedActor` and `Brain._g9_apply_source`. Bound ordinary operations exist; a deciding
  G3 growth/engine result does not follow. `GrowthLoop.__init__` calls the canonical class's `require_ready`,
  whose body unconditionally raises `LedgerError("G9_A_UNREADY: ...")`, including after engine binding.
  The surviving generic loop and public admission APIs do not supply the registered deciding experiment.
- **2.2 problems.** `taught_problems.Problem`, `read_split`, `solved` and `solves_stably` survive unchanged.
  `CanonicalGrowthAdapter.__init__` accepts only the exact registered ii.xor/ii.twin objects and canonical stack.
  C1-d primary and i.S0C1 secondary campaign coverage is ABSENT; ii.xor is not promoted to primary. C1-d remains
  REPRESENTABLE, NOT SHOWN LEARNABLE in the G2 record; its learned positive and C16 are prerequisites, not results.
- **2.2/2.9 C11 retention.** `CanonicalGrowthAdapter.read_retention` reads old-task selection sets with snapshot/
  restore through `RetentionSets`; a final read with a meter raises `G9_RETENTION_FINAL_UNRECORDED` until durable
  consumption exists. Selection is not untouched final judgement. A_ii/S0C1 repair gates remain as A3 states.
- **2.3 real port.** `SpikingInteractionLearner._argmax_label` adds `posed_problem.chat_grown_drives` when
  `_g9_context` is present; `_deposit` calls `chat_grown_deposit`. `actor_context` carries problem_id, unit rows,
  return afferents, t_post_g and out_eligible; `sync_actor_context` transfers actual learned output weights back.
  `grown_code.segment_psp` and `GrownReturnPort` supply the latency/return primitives. This is not proof of the
  registered complete frozen52/grown/reservoir tagging, masks or six-arm port. Frozen text_features(text,52) and
  separate unit_id-ordered segments remain required; None drive/deposit/checkpoint identity and C16 are UNVERIFIED
  on the actual campaign consumer. Ablations must cover that consumer's real state and return contribution.
- **2.3 redo/version.** `learner_events.KIND_SPECS` adds posed_problem; `Brain._g9_apply_source` and `PosedActor`
  bind typed source effects to the common writer and restore path. `grown_version` covers ordered unit parameters,
  active flags and output weights, not the entire return queue or column gamma/batch membership. Bind those actual
  carriers and their replay/redo domain before data; crc presence is not the registered log-position redo proof.
  Keep input weights frozen after admission, redo at each event's version and replay through current units.
- **2.3 Dale/state branch.** `chat_grown_deposit` calls shared `clip_to_sign`; `apply_unit_row`, `clip_feed`,
  `check_feed` and `unit_label_gate.check_saved` retain their limited checks. C17 still covers every actual deposit
  and restore, not just isolated clipping. ctxread has typed schema slots in `KIND_SPECS`; `Brain._record_ctxread`
  and `grown_code.reconstruct_state` remain ABSENT in src/tests/tools at both comparison trees. `GrownUnit` is a
  stateless latency unit. The 10-06(f) parked step, three reconstruction reds and memory redo exclusion still stand;
  no schema-only recovery claim, no confirmatory memory growth, and no C5 pass is asserted.
- **2.4 diagnosis.** `capacity_certificate.current_keys`/`diagnose` survive byte-identically and accept grown units.
  The canonical adapter instead passes `history_texts=None, units=()` and reports canonical-frozen-code-only;
  the helper's grown-code separator is unbound and no-witness cases may remain UNRESOLVED. Full current-code,
  history and actual return-aware diagnosis is owed. Frozen-only diagnosis cannot replace that gate.
- **2.5 lifecycle/source.** `candidate_lifecycle` draw/score/select/prune, `NGSNN.fit`/`prune_unearned`, and
  `NeurogenesisSelfImprove._refit`/`growth_fingerprint` are byte-identical across the compared trees; retained
  equivalence tests are substantive tests, not a new execution. Preserve the 1e12 tie during this refactor.
  `growth_loop.trace_train`/`label_probabilities` retain the LOCAL fixed-random-B, own-eligibility rule at T=1.
  Actual grown-source search/selection/admission runners are ABSENT. R2 transition_sink/hydrate_transitions are
  absent at both trees, an adjacent unbuilt obligation, not a newly measured G3 null.
- **2.6 attempts.** `CanonicalGrowthAdapter.attempt` teaches the requested pre-frozen epochs, then calls
  `tp.solved` once on validation. It does not test epoch-end stable-10 or stop at its first run. `GrowthLoop._attempt`
  records configured epochs, not a demonstrated actual-exposure history. Restore the registered E=400 maximum,
  STABLE=10 consecutive epoch-end solves, early stop and exact recorded exposure before deciding data, or obtain
  a further reviewed prospective scientific amendment. A final all-correct read is not renamed stable-10.
- **2.6 finite budgets.** `GrowthLoop.run`, `_paused`, GROWTH_KINDS, A_MAX=3, `Ledger.set_state`, `new_teaching`,
  `record_run`, `GrowthStop.observe_problem`/`should_grow_problem` and solved_sink retain the finite generic paths.
  Preserve DATA/open, LEARNING_RULE_DEFICIT and UNRESOLVED without scoring readiness refusal as scientific failure.
  `Ledger.open_problem` now calls G2's named unset-beta refusal when a positive-spend baseline exists and beta is
  None; `taught_problems.BETA` remains None. First/no-baseline budget=52 does not require beta. SEQUENCE or any
  actual positive-spend baseline requires its dated dev fixing; sleep grants no fresh budget, teaching at most three.
- **2.6/10-04/10-06 births.** `Ledger.record_units`, `_birth_row`, `known_unit`, `birth_records`, `first_sign`,
  `make_admission` and `grown_code.admit_rows` retain weight-bearing immortal births and one-use ticket checks.
  `CanonicalGrowthAdapter.admit`/`Brain._g9_apply_source` add actor admission with typed feed/return descriptors.
  Capacity birth fields are not every promised kind/RNG/memory carrier or an R2 installation receipt.
  NEW `batch_reconciler.reconcile_batches`, called by `GrowthLoop.run` and `Brain._g9_reconcile`, scans the same
  journal for sequence/birth/batch/spend inconsistencies and refuses by name. This supersedes "no reconciler" in
  10-06(d); it appends no repair, removes no birth and does not close every crash window. Ticket/segment checks
  after admission still detect corruption without undoing it (10-06(e), 10-07(c)).
- **2.7/2.10 columns.** `place_labeled`, `label_of`, `labels_by`, `require_admissible` and `column_latencies`
  survive. `growth/column_batch.py` and its test are ABSENT on main, present at 5e608d6c. COL/SINGLE/BATCH9 assembly,
  site/targeting, k={4,8,12} primary8, counting every E/I and gamma/config versioning remain build obligations.
  G7 amendments (e)/(f) RECORD A2 resolving its O9 (= G3 O10): the T1 split is accepted and retention stands in.
  Original OWNER_DECISIONS_2026-10-04.md is UNREAD here; this is a G7-recorded resolution, not fresh direct-owner
  verification or a renewed request for the same ruling. G7 (i)/(j) retain Q-COL and version debt, and the entire
  T1 family still waits on T3. Kernel-free primary arms do not wait on T3 solely because T1 does.
- **2.8/2.9/2.10 instruments.** `instruments/growth_campaign.py` and its test are ABSENT on main, present at
  5e608d6c; its old primary helper consumed supplied outcome dictionaries, not arm runs. Main's changed
  `paired_contrast.contrast` alone can mark positive degenerate differences MET; G3 superiority must instead
  read NOT MET for every zero-variance sample. Non-inferiority uses its common value (>0 MET, otherwise NOT MET).
  The six-arm, T1 and Astra runners, real C1-C19 measurements and causal context experiment remain ABSENT.
- **2.10 feed.** `grown_feed` and `recurrent_pool` are byte-identical across these trees. NEW registered
  `feed_readout.arm_latencies`/`pool_first_spikes` return direct or hosted-pool first-spike latencies;
  `_build_seed` permits dev/fixture seeds only, and the helper refuses N>256. `feed_u_contrasts` enforces the two
  nondegenerate superiority contrasts on supplied U values; none is a trained arm runner or feed benefit.
  N=3 fixture identity is not N=256 proof. G9's additional actual return/feed binding does not supply G3's campaign.
- **2.11-2.16 guards/cost/closing.** Campaign preflight/go, child-home and paired-snapshot helpers are ABSENT on
  main; common live-home/snapshot guards and an explicit Ledger path survive, without proving arm isolation.
  `tools/lake_append_baseline.json` is byte-identical, tag11 fixture p95=15242 B, explicitly not a cumulative bound.
  Old timings/projections are not measured costs for new posed events, return carriers, retries or scaled nights.
  Measure actual append classes, evaluations, teaching, final reads and night cost on the authorized workload.
  Old panel/battery statements retain their historical scope. Main's real tests and registrations do not include
  the missing campaign/column modules. Stacked review, validation, closing record and engine credit remain owed.

### A3. Owner rulings and their precise authority

The following words are quotations, not new scientific results. Supplied board numbers refer to read text dumps.

- **#1037/#1039, codex-sndl-coord-1010 relaying the owner's direct answer:** "Adopt these three decisions".
  The exact question adopted was: "Three existing owner-only decisions block final validation. My proposed
  choices are: R2 stops all learning writes after a write failure until restart, with its old failed-write test
  updated in a dated amendment; G6 includes every plastic circuit in the eligible-brain arm; R3 retires the three
  capabilities that failed their frozen tests, while keeping the sleep requirement. Which disposition should I
  record?" The coordinator's application in owner-dispositions-adopted-2026-10-11.md is separately attributed:
  "all kinds on the common R2 learner-event writer latch after an uncertain failed write until successful restart
  reconciliation; write the dated prospective amendment of the old FAILWRITE schedule before implementing or running
  its new checks." G3 inherits the actual `Brain._append_learner_event` all-learning latch and rollback boundary,
  not per-kind retry after a failed write. A latched arm is unreadable, never silently scored unsolved or retried.
  Main already implements/tests the latch: the ordering breach is disclosed in A1. R3's named interaction
  replay/deposit, Benna-Fusi rest/rebinding and attractor.sleep learning caps are retired; sleep and distinct
  wake/storage/recall/episodic-replay jobs survive. `Brain._r2_sleep_step` is structural-only. Explicit public
  `SpikingInteractionLearner.replay` remains callable: retirement of a live cap is not API deletion or a P8 licence.
- **#518, Claude advisor at the owner's instruction:** owner's verbatim "X1: A". The relay's condition is:
  "the amended G8 translation test must load existing g0c.v1 checkpoints and translate them into r2.v1, so old
  checkpoints keep working". r2.v1 approval is not proof of this: `Brain._r2_boot` refuses unbound legacy migration
  despite a surviving learner upcaster. Translation remains an UNVERIFIED integration tension, not deleted.
- **#2106, claude-cont-1013 verbatim relay of the owner's selection:** "A: journal the bout" -- "Add a new R2 event
  kind with deterministic restart redo, and a dated post-data amendment loosening K10c to allow exactly that write.
  Bout effects become durable brain learning." #2105 was the integrator's question, not an owner decision.
  `staged_sleep_bout.run_bout` dispatches persistent bouts through `Brain._r2_run_bout`; redo uses `learner_bout`.
  G6's labelled POST-DATA 2026-10-13 K10c amendment remains POST-DATA. G3 cannot reopen F5 or claim no bout writes.
- **#2135, integrator claude-cont-1013 verbatim relay:** "Fix the stack first". The relay requires a measured
  canonical E/I/Benna-Fusi/homeostatic repair with neuter A/B and a reachable red before G2(ii), G3 C11 on S0C1 and
  G9 proceed. A bare learner or omission of A is no substitute; no release of that gate is inferred here.
- **#2565, integrator claude-cont-1013 verbatim relay:** "Bouts scale with labels". The relay's rule is
  "bouts = 7 x rows" for label rows sharing REM. Its disclosed trade-off is B's overlap lost as A recovers;
  the night grows with vocabulary. #2564 is the integrator's question, not the answer. For G3's eventual matched
  curriculum, freeze the participating label-row population L and one identical 7*L-bout schedule, RNG,
  replay/REM budget, modulation and teaching exposure across the paired arms before data. Report L, bouts and
  actual work per arm; no hidden per-arm night extension or match to nominal epochs. L and the operational
  schedule remain UNVERIFIED here. This is not a C11 PASS or automatic release of #2135's prerequisites.

#123 is G5's agent note about named brain-producer credit, not a direct owner answer or G3 connection-5 credit.
#1779 is frozen R2 operation preparation, not G3 integration proof; #2719 is an R2 review, not scientific authority.
G7-recorded A2 is handled in A2 above with its original UNREAD, without invented owner words.

### A4. Campaign instrument: ABSENT; rebuild on main in its own build arc

G3 cannot run as registered at fbb1f38e. Recovering branch helpers does not restore the missing arm runners or open
the canonical readiness/final-retention refusals. The instrument will be rebuilt on main in its own build arc
against THIS amendment, with its tests and registrations; this documentation lane does not build it or bypass gates.
Before any protected seed, that arc owes reviewed, stacked implementation and the following registered contents:

- Six arms: INCUMBENT, PRESIZED FROZEN, PRESIZED SELECTED, TARGETED, RANDOM GROWTH and SHAM; also literal/gated
  loop comparison, SEQUENCE in separate homes, T1 COL/SINGLE/BATCH9 with paired PRESIZED and random attachment,
  Astra shared/task-exclusive A+B, cross-batch neuter, LIF/ALIF, fixed/adaptive inhibition and causal/disconnected
  context; feed fed/direct/norm-matched random-feed. N>256 stays owner-target gated.
- C1-d remains primary: select/attempt on validation combos, future judged once after validation stable-10,
  never steering the loop; future_d never read. Shared GREET/GOAL A-then-B retention primary, disjoint beside it.
  Secondary ii/S0C1 retain the finite-same-item disclosure. C16 learned witness/masked/incapable controls on
  9900-9907, C2/C6 and G2 construction/learnability precede confirmation; no presized H64 substitution for C16.
- Restricted score, next-grid failure penalties (54 for batch9, 53 for batch1 at52), BOTH D_sel/D_rand t>1.895
  and mean>=batch; report failure/solve rate and observed sizes separately. "Fewer neurons" additionally needs
  >=6/8 uncensored solves in both compared arms and positive observed paired mean>=batch on jointly solved seeds.
  Match actual cumulative teaching incl. no-growth attempts/early ends, candidate evaluations/training separately,
  and A3's actual night work. Never substitute m*E or feed helpers for these outcomes.
- Every C1-C19 red/control, stable-10 and finite-path checks, learnt C9 feasibility, C10 true/shuffled control,
  C11 overwriting control and dev-fixed F_bound/delta_NI. S=8 paired t, ties counted, all 2^8 sign-flip masks
  reported; superiority degeneracy NOT MET and NI by common value as above. T1 keeps >half censored REFUSED,
  U_SINGLE median>=80 (k4:40) granularity guard, positive delta_NI and its T3/Q-COL waits. REFUSED deletes nothing.
- Seed allocations stay 9000-9099 exposed dev, T1 power pilot9040-9047, C9 feasibility9000-9007 and retention-bound
  fixing9000-9015; fixtures9900-9999; primary9100-9107/reserve9108-9115, feed9200-9207/reserve9208-9215,
  T1/Astra9300-9307/reserve9308-9315: seed9300 is construction-exposed by G7 (f)'s item counts, not a trained result.
  These are allocations, not run permission; A1's exposure still requires integrator disposition before any run.
  Old G2 dev7100-7103 remains exposed only; the never-run list of 2.12 remains. Freeze provenance/go,
  fresh explicit home/lake/output per seed/arm, child environment AND argument, identical paired ledger SHA and
  separate SEQUENCE chains. Missing-home, live-home and differing-SHA reds precede writes; live lake untouched.

No unreadable arm triggers 2.11's scientific-null deletion. Its conditional dispositions remain: retire the measured
failed trace source/gate/column structure/Astra mechanism/feed implementation as registered, never an owner-drawn
growth or reservoir part on missing code or a refusal. L2's re-proposal debt is not discharged.

### A5. Reds retained: public constructions, with missing consumers distinguished

These are STATIC constructions, not fired reds or measured effects. The canonical loop is closed. Where a runner
or learned consumer is missing, the construction is a build obligation; failure of today's instrument to exercise
it is not proof that no input can. No C1-C19 criterion is deleted or weakened under P8 by this amendment.

- **C1:** public learner n_in=52 versus62 on the registered texts constructs the encoder red; two52 reads control.
  `segment_psp`/actual port before-after identity remains owed.
- **C2:** `capacity_certificate.check_farkas` accepts supplied multipliers/texts/labels: double one multiplier or
  use separable-twin labels; `separating_witness` through route_text controls it.
- **C3:** set the public bare learner's registered near-tie weights and compare actual route_text to exact witness,
  with an off-tie control. The actual grown-reader disagreement logger is missing, so no agreement claim follows.
- **C4:** `capacity_certificate.diagnose` accepts contradictory items/history; generic `GrowthLoop` accepts a DATA
  adapter. Clean capacity twin controls it; canonical refusal is not this scientific red.
- **C5:** delayed-cue/history inputs construct STATE diagnosis through `diagnose`; capacity inputs control it.
  Memory drawing/step/reconstruction consumer remains parked/unbuilt; all three dropped/swapped/digest ctxread
  reds survive as explicit unpark obligations, not satisfied checks.
- **C6:** `candidate_lifecycle.select_block(reverse=True)` reverses the tie on fixed dev streams;
  `growth_fingerprint` and original trajectory control it. Their retained tests were not run in this lane.
- **C7:** supply transposed-forward-weight source text to `gates/eprop_transport_gate`; fixed random-B controls.
- **C8:** public generic `GrowthLoop` accepts an unsolvable LITERAL adapter with diagnosis disabled or a fixture
  budget override; under-budget solvable twin controls. Keep next-whole-batch pause, OPEN and no solved report.
- **C9:** `silence_units` accepts real unit ids, `match_sources`/`match_feasible` accept norm/rate populations.
  Build the actual learned C1-d reader and active distractor matches; unsilenced control, per-item effects, k>=1
  every dev seed and >=half covered on >=6/8 remain required. Missing solves cannot be called a silencing result.
- **C10:** `trace_train` takes supplied candidate labels; shuffle these while true diagnosis, true validation
  selection, budget and admission schedule stay frozen. True-label twin controls; future superiority is owed.
- **C11:** public teaching can deliberately relabel A's texts to construct overwrite; untouched A controls it.
  `retention_ok` over invented numeric drops is not empirical retention. Readable A, F_bound, strictly positive
  delta_NI and the real once-only final reader remain owed before a retention verdict.
- **C12:** `require_admissible`/`admit_rows` accept missing-label, duplicate-id, uncovered/spent-ticket or tampered
  input; fully labelled ticketed batch controls. Detection after admission does not undo installation.
- **C13:** `Ledger`/`reconcile_batches` accept a constructed journal with a missing intermediate spend/sequence
  row or unmatched births; untouched same-codec twin controls. Not every arbitrary trailing truncation is
  detectable. Missing-home/live-home/differing-SHA child inputs remain constructible for the rebuilt guard.
- **C14:** a supplied context adapter can attempt target/future access, with past-only input as control.
  `posed_source`/`GrownReturnPort` types do not semantically detect every forbidden channel; rebuild that causal
  channel instrument and its violating adapter before reading this check. Neither absence nor type safety passes it.
- **C15:** `feed_drive`/`random_feed`/`percept_fed` accept distinct norm/sign-matched feed matrices; identical copy
  and no-feed arms control. N=3 fixture proof does not certify the registered N=256 determinism/campaign control.
- **C16:** `delayed_cue.witness_units` and public source/admission/learner interfaces permit witness versus masked
  or provably incapable units. Learn stable-10 and B future solve through the actual port; certify incapable
  equal-count/norm units by the augmented-code witness first. Random decoys remain comparator, never the red.
- **C17:** `clip_to_sign`/`apply_unit_row`/`clip_feed`/`check_feed` accept sign-flipping deposits or violating
  saved synapses, with sign-consistent control. Exercise the actual chat-grown/return/restore consumers too.
- **C18:** public generic loop accepts SEPARABLE/UNRESOLVED adapters; construct verified-wrong real reads, freeze
  ALL plasticity and assert full weight/checkpoint identity. Learning-on twin controls; scripted diagnoses alone
  do not prove the registered three-attempt real-learner red.
- **C19:** a public route callable can memorize teach/validation combinations and fail unknown future combinations;
  XOR-core callable controls. Build its held-out runner; primary learnability still gates confirmation.

The G7 (h)-(j) coupling checks also remain: gamma=0 identity, bidirectional liveness mutants and structural causal
ordering; the same-step mutant's equivalence at PSP(0)=0 is disclosed there, not empirical evidence of a fired red.
Configuration versioning, Dale and the rebuilt isolation guards retain their constructed violating inputs.

### A6. Openings carried, each UNVERIFIED

- **UNVERIFIED seed provenance:** external executions/binaries and T1 seed9300 construction-only exposure; fresh
  integrator disposition/go before any protected family. The heading grants no categorical non-exposure claim.
- **UNVERIFIED primary/readout:** C1-d learned positive; actual C16/None/masked/incapable port; full frozen52/
  grown/reservoir segmentation; all-consumer Dale and return ablations; current augmented/history diagnosis.
- **UNVERIFIED attempts/campaign:** six-arm/T1/Astra/feed runners, stable-10 actual exposure, future-once handling,
  C9 feasibility, C10 causal comparison, finite real-reader caps and registered per-arm restart/isolation controls.
- **UNVERIFIED retention/night:** readable A_ii and #2135 release, final durable once-only judgement, F_bound and
  delta_NI, actual sharing-row count L and matched 7*L bouts; #2565's A/B trade-off is not a G3 C11 pass.
- **UNVERIFIED T1/configuration:** Q-COL gamma, gamma/batch version carrier, family-level T3 and power pilot;
  G7 records O10 answered, but the original A2/delegation remains UNREAD here.
- **UNVERIFIED sequence/birth/restart:** BETA when a baseline applies, complete birth/RNG/kind carriers,
  learner-installation receipt, repair beyond named refusal and full versioned redo/return-queue equivalence.
- **UNVERIFIED memory/P3:** defined memory step, ctxread producer/reconstruction/three reds and exclusion counts;
  adjacent R2 transition hydration. Parked is not recoverable or MET.
- **UNVERIFIED translation:** #518's existing g0c.v1 -> r2.v1 guarantee versus Brain's unbound-legacy refusal.
- **UNVERIFIED scale/cost:** real N=256 feed control and trained contrasts, N>256 owner target, actual immortal
  append rates/retention reads/candidate/retry/night costs. Historic tag11 fixture15242 B is neither a cumulative
  bound nor contradicted by a different workload's tick. Decision13's live rate remains owed to the integrator.
- **UNVERIFIED delivery/closing:** future instrument registration, stacked independent reviews/panels, mapped
  battery and closing record; this docs verification establishes no campaign or whole-engine acceptance.
- **UNREAD originals cited above or by the historical registration:** OWNER_DECISIONS_2026-10-04.md;
  OWNER_QUESTIONS O1-O11/Q-COL and board#2; p0_reports/G3.md and DEPENDENCIES.md lead rulings9/9a/10/12/14/18/29;
  Codex P0 reply/critique; g3_encoding_red.py/g3_cost_probe.py; G7#204; #2133/#2134; #2556/#2562;
  unsupplied original #1/#124/#430/#2112/#2689. Relayed excerpts do not make those originals independently read.

*Port note (arc 6, ported onto c4c10b28, tests not run): this addendum is dated 2026-10-08 and was written on `claude/g3-b3` from 117aada0; the commit ids below are that branch's. It is appended after A1-A6 unchanged, and A4's "campaign instrument ABSENT" is the state before this port.*

## DATED ADDENDUM 2026-10-08 (P4a / P4b built learner-independent; owner-answer readings; no G3 data read)

- **Branch and commits.** `claude/g3-b3` from origin/main 117aada0. 4e744848: `growth/column_batch.py` + `tests/test_column_batch.py` (P4a).
  c4d0e921: `instruments/growth_campaign.py` + `tests/test_growth_campaign.py` (P4b, learner-independent part). Additive; nothing above changes.
- **P4a as built.** Arms COL / SINGLE / BATCH9 (batch sizes 9 / 1 / 9 at k = 8; U counts every unit), `coupling(arm, gamma)` with gamma a
  REQUIRED parameter (pending Q-COL; nothing decided), `as_column` / `column_source` (k E then one I, output synapses clipped to the
  sign), role-free `ColumnBatch` from label rows (2.7 D8), `worst_residual` (the one shared targeting rule). Reds that fire (read off the 2026-10-08 run on claude/g3-b3; this port has not run them): BATCH9 equals
  COL at gamma 0 byte-identically; a planted early-I fixture makes COL's E latency None where BATCH9's is 4.
- **P4b as built.** Seed families and `preflight` (a confirmatory family is refused unless it is the exact registered block AND the lead's
  written go is entered; `GO` is None), `failure_penalty` (54 for batch 9, 53 for batch 1), `restricted_score` (2.8), `primary` (D_sel, D_rand
  under ruling 9, floor = one batch (L4), "fewer neurons" only when both arms solve on >= 6 seeds and the both-solve size difference is >=
  the floor), `non_inferiority` (ruling 9a), and 2.12a's isolation (fresh home per (seed, arm), live-lake refusal through home.py's guard,
  `pair_snapshots`, `child_home`/`launch`). The statistic is `paired_contrast`'s, the guard `substrate/home.py`'s.
- **NOT built, stated plainly.** The arm RUNNERS, C16, C9-C11 runners, the T1 / Astra / feed runners and the real-learner adapter: all need the
  grown-segment port on `readout/interaction_learning.py`, which waits on R2's r2.v1 checkpoint step (owner question #2). No arm of 2.8 has run;
  no number in this file is a growth result. `growth_loop`'s only `admit` implementers remain the scripted learner and test doubles.
- **Readings each owner answer leaves (no answer assumed).**
  - **#2 (R2 r2.v1) no:** the port cannot take R2's chain; G3 either re-proposes the port as a standalone schema step (a new owner item) or the
    growth campaign stays unrunnable; the confirmatory families read NOT RUN.
  - **#2 yes:** the port lands as the next schema step, C16 runs first on 9900-9907, then the dev smoke on the lead's machine-free go.
  - **O11 (C1-d not shown learnable):** option (a) leaves the primary cell UNTESTED and the campaign runs on the secondary cells (ii), (i) S0C1
    with the 2.8 same-items disclosure; C19 and C10's C1-d form read NOT RUN. Option (b) needs a new pre-data C1-d design from G2 first.
  - **Q-AII:** A (A_ii leaves) means C11 for B = S0C1 stays UNREAD and C11 reads only on C1-d with A_c1d (itself gated by O11); B (bare-learner
    incumbent) changes 2.5's incumbent definition and re-registers 2.9 C11's baseline before any C11 reading.
  - **Q-COL (gamma) and O10 (T1 split):** T1-struct and T1-size read NOT RUN regardless; nothing in P4a decides gamma.

## DATED ADDENDUM 2026-10-29 (owner growth target for the N > 256 arm; pre-data; no G3 data read)

- **Owner answer, verbatim** (relayed by the launcher from the owner's chat, transcript e49a47ce, 2026-10-29): "Grow steadily to build new
  neurons to eventually gain cognition and consciousness". It answers the open owner question of 2.16 (annotated there) and rules the N > 256
  arm of 2.8 (:518-520, "waits for the owner to ratify a growth target inside the reservoir").
- **Ruling (the launcher's binding reading).** The target is STEADY growth with NO CEILING. Reservoir growth is not capped at any N.
- **Tested N values are measurement points, not caps.** Today no test BUILDS or admits a pool above 256. The tests that fix N around the gate are
  `tests/test_feed_readout.py:143-159` and `:162-178` (a planted declared size of 257 only; no larger pool is built or admitted) and
  `:181-195` (N = 256, the bound at which the size gate passes). They pin the CURRENT refusal gate `growth/feed_readout.py:89-90`
  (`pool.n > 256`), not a measurement point. Why those N: 256 is the engine's existing pool size (`reservoir/recurrent_pool.py:73`), the last
  N whose determinism red has been run; 257 is the smallest N above it, the minimal probe that the gate fires. The N values the arm will MEASURE
  at are named by the building session in a further dated line in this file BEFORE any seed runs, with the reason for each; none is named here.
  `tests/test_g3_arms.py` fixes no N.
- **Determinism red.** The pool's own determinism red (C15, :421) re-runs at EVERY tested N, not only the first.
- **Pacing (POLICIES P4d, docs/POLICIES.md:575-577 at this branch's base 8fe7bf57).** "Pause at a budget (owner decision): each problem's
  growth spends a budget relative to a persisted running baseline (P1); at the limit the brain reports UNSOLVED, stops growing, and keeps the
  problem OPEN to retry after sleep or new teaching." The budget paces growth per problem and pauses it; it is not a ceiling on the reservoir.
- **No code may treat a tested N as a maximum.** A measurement point is a place the arm looks, never a bound the pool may not pass.
- **Owed to the integrator (P3 / P4d conflict).** The refusal `growth/feed_readout.py:89-90` is a hard maximum on N, and its message
  ('N>256 growth target unratified') is now false; its two tests above pin it. They are NOT changed here (docs only); they are listed in the g3-arms REPORT.md, Round 3, as OWED for the build that admits N > 256.
