# G2 / T110 -- problems the current weights provably cannot solve, taught through G0's channel (DRAFT pre-registration, wave 1; not frozen)

Plan row: PLAN-FINAL G workstream, G2 / T110 (items (i) and (ii), the 2026-09-25 note "each problem pre-registers its
SOLVED predicate and its growth budget before any run"), Part F's G-R2, and the G3 row's budget session calls (dated
2026-09-25: baseline over SOLVED problems only; the first budget pre-registered with its P1 case; the retry rule
pre-registered). Inherited and NOT re-run here: item (ii)'s generator and G-R2's construction check A1-A5
(`instruments/inseparable_mapping.py`, `evidence/g1c-carried-state.md` 2.8), which run only inside G1(c)'s campaign on
seeds 7000-7007. This file owns: item (i) (the delayed-cue 2x2), the controls G1(c) 2.8 leaves to T110 (the presized
nonlinear layer and the delayed-cue explicit-history control), the taught campaign for (i) and (ii), each problem's
solved predicate and growth budget, and the shared problem contract G3, G5, G6 and G9 read. Branch `claude/g2`, base
`c8dd93fa` (the G1(c) stack). Status: DRAFT; the orchestrator freezes it after review; the freeze is two-step (2.13).

## 1. Territory, taken at `c8dd93fa` (2026-10-03)

1. **The incumbent and G0's channel.** `SpikingInteractionLearner` (one R-STDP neuron per intent, n_in 52, t_post 6). Its
   read, `route_text` -> `respond`, is a pure function of (weights, current text): no plasticity, no RNG, no carried
   state. G0's correction path is `LiveBrain.correct(text, intent)` == `learn_turn(text, None, feedback=intent)`, reached
   from the canonical `Brain.correct`; the learned route is `LiveBrain.route(text)` == `respond(text_features(text, 52))`.
   `Brain.__init__` attaches a `PlasticityRuleStack(n_neurons=9, seed=brain seed)` as the learner's `stack` (the
   `_plasticity` attach), so the CANONICAL incumbent is the bare class plus that stack; `naca` and `readout_zoo` are None.
   `inseparable_mapping` teaches the bare class (LiveBrain's form, no stack).
2. **The code.** `text_features` hashes each word (crc32 % 52) and adds `1 + 0.1 (pos + 1) + 0.01 len(word)`, so the LATER
   word ranks first; `rank_code` breaks ties by index, so empty buckets take the next ranks in index order; with t_post 6
   only ranks 0-5 have a nonzero PSP. The learner therefore reads at most the six latest words (a 6-word horizon), and
   unequal word lengths can reorder ranks only past a 10-character difference.
3. **(ii) exists** (G1(c) P2, `869ff4a1`): texts, XOR and twin labels (GREET / GOAL), A1-A5, TEACH_EPOCHS 400, the A5
   witness, the stream. Its docstring states that the presized nonlinear and delayed-cue controls are T110's.
4. **Incumbents searched (P2), each a different subject class, none adopted, each disclosed `distinct-from`:**
   `growth/autapse_neuron.py::_delayed_match_stream` (one autaptic LIF, a spike stream, last-bit recall);
   `instruments/episode_driver.py` (KernelTick trials, sample/delay/probe, same/different oracle);
   `reasoning/developmental_loop_wm.py` (explicit working memory in the developmental loop). None is a text-channel
   problem for the taught learner. No presized nonlinear layer in front of the chat learner exists (`presized` greps to
   `gates/capability_bar.py`'s twin bookkeeping and `inseparable_mapping`'s pointer only; the search was proven on
   `inseparable_mapping`). The learner's opt-in `readout_zoo` passes each intent's LINEAR read through a point-neuron pool:
   a per-intent transform after the linear score, not a hidden layer; distinct.
5. **Route cues.** `engine/live_brain.py::_ROUTE_CUES` keys GOAL, MEMORY, ATTENTION, SAFETY, CONSCIOUS, INTEROCEPT and
   EXPLORE; only GREET and ARM are not keys. A `Brain.respond(..., human=True)` presentation turns a learned cue-intent
   route into G0(b)'s clarify turn. ARM is never used (arm-adjacent).
6. **Firewall.** Every text of the generator below, and (ii)'s four texts, interpret as kind `reject` under
   `sensorimotor/nl_interface.NLInterface` (none `arm` or `lake`); the detector fires on known positives ("go home",
   "open the gripper" -> `arm`). Measured 2026-10-03.
7. **Seeds.** No recording seed in 8000-8999 appears in specs/, src/ or tests/ (the 8xxx hits are decimals, budgets and
   settle counts); the same grep found 23 seed lines each for the positive controls 7000 and 4242.
8. **Design measurements (2026-10-03, out of tree, EXPOSED scratch seeds 8900 and 8901 only; they decide nothing).** Run
   under `throwaway_home()` (the live home's newest file is 2026-09-27, unchanged). Intents MEMORY / EXPLORE were used
   here; the registered pair is GREET / GOAL (2.2, changed in the debate).
   - `make_brain(seed=8900, persist=False)`: 1.78 s; learner n_in 52, t_post 6, seed 8900, stack PlasticityRuleStack.
   - Channel identity: after S0C0's teaching through `brain.correct` / `brain.brain.route`, the factory learner (bare
     class + `PlasticityRuleStack(n_neurons=9, seed)`) taught the same sequence had byte-equal weights; the bare class
     WITHOUT the stack did not (a natural red).
   - Incumbent readings (factory, 8900): S0C0 solved at the first epoch end; S1C0 plain max 0.5 over 150 epochs;
     S1C0 with explicit history solved after 1 epoch; S0C1 plain max 0.75 over 300 epochs (last 0.5); S1C1 with history
     max 0.75 over 300 epochs.
   - **Presized control, a load-bearing design finding.** An UNCENTERED frozen random expansion (z = W phi, rank-coded,
     read by the incumbent's class) FAILED (ii)'s XOR at H 16, 32, 64 and 128 (2-3 of 4) and (ii)'s twin at H 32 and 64
     (seed 8900). The MEDIAN-CENTERED form (2.4) solved every capacity cell at H 32 and 64 on both seeds (stable within 0-3
     epochs), was marginal at H 16 (S1C1 + history unstable on 8900; (ii) XOR needed 160 epochs on 8901), and stayed at or
     below 0.5 on S1C0 plain. The centering is recorded here, before any dev or check seed, so it cannot be read as
     post-hoc tuning; the proof on the dev seeds is owed before the freeze (2.13).
   - Cost: about 7 ms per epoch (8 episodes plus a 32-presentation validation) for the factory incumbent; a presized
     run at H 64 about 2-3 s per 100 epochs.

## 2. Pre-registration (DRAFT)

### 2.1 The question and what a result licenses

**Question.** Do G2's problems have the property the engine needs: the CURRENT weights (the canonical taught learner,
growth OFF) provably cannot solve them, for a NAMED deficit (state, capacity, or both), while a control that adds exactly
the missing thing does, all taught through G0's channel?

**A PASS** for a problem licenses only: "P is a valid G2 problem with diagnosis D", usable by G3, G5, G6 and G9 with its
predicate and budget. It never licenses a claim about growth, the reservoir, a readout, or the engine.
**A RED** says the construction is wrong (2.10). Nothing owner-drawn is touched.

### 2.2 Item (i): the delayed-cue 2x2

- **Vocabulary** (crc32 % 52 buckets): cue `jar` 14 / `pen` 19; filler `box` 38; probe first word `oak` 21 / `elm` 23,
  probe second word `fox` 50 / `owl` 49. Seven distinct buckets, all >= 8 (so every text's empty-bucket filler ranks are
  buckets 0-5), none of (ii)'s 28, 31, 41, 51; all three letters (ranks follow position).
- **Episode** (three turns, one per item (c, p, q) in {0,1}^3): context `CUE[c]`; context `box` (the delay, D = 1 filler
  turn); probe `"P[p] Q[q]"`. Labels exist only on the probe. 8 items.
- **The four cells = four labellings of the SAME 8 episodes** (bit 0 -> GREET, bit 1 -> GOAL; the pair (ii) uses, so G3's
  A-then-B curricula share label neurons by default):

  | cell | label bit | needs state | needs capacity |
  |---|---|---|---|
  | S0C0 | p | no | no (positive cell) |
  | S1C0 | c | yes | no |
  | S0C1 | p XOR q | no | yes |
  | S1C1 | c XOR p | yes | yes |

- **Explicit-history form** (the plan's control; 2.4): the probe read and taught as `"CUE[c] box P[p] Q[q]"`, four words,
  inside the 6-word horizon; the cue sits at rank 3.
- **Constructed unsolvable cell U** (for G3's DATA branch and G-R3 / G-R9 (iii)'s "the loop reports solved on a
  constructed unsolvable problem"; G3 consumes it): the same episodes, each presentation's label a bit of
  `crc32(f"{seed}:{split}:{index}")`, independent of every turn and of history. *(SUPERSEDED by 2.15.6, Codex #3: U is
  a deterministic opposite-label-pair fixture for stateless readers; U_S serves stateful ones; the former 2^-32 sentence
  is withdrawn.)*
- **Curriculum A problems for G3 (debate, G3's choice):** for B = (ii), `A_ii` = (ii)'s word-order construction on the
  disjoint quadruple `ram` 15, `cup` 16, `emu` 20, `cod` 26 (distinct buckets, none in 0-1, disjoint from (i) and (ii)),
  labelled separably (u), intents GREET / GOAL; for B = (i) S0C1, A = (ii)'s twin as is. `A_ii` carries (ii)'s A1 as a
  generator check (B1) and A4's learned clause as a campaign arm (the incumbent solves `A_ii` on every seed, B5's form).

### 2.3 Why the current weights cannot solve the S1 and C1 cells (two strengths, never read as one)

- **State cells (S1C0, S1C1 plain): exact in floating point.** The probe text is identical across the two cue values, and
  the read is a pure function of (weights, text), so in any read the learner gives the same answer to a probe whatever the
  cue was. With learning OFF every probe text appears with both labels equally often, so accuracy is at most 0.5 in every
  read, with no rounding caveat. The runtime instrument is the identity check B3; its red is a stateful stand-in.
- **Capacity cells (S0C1 on the probe code, S1C1 on the history code): exact in integer-rank arithmetic only.** Each
  relevant bit sets its own word's (synapse, rank), the rest of the code is identical, so every neuron's drive is additive
  in the bits; the argmax cannot answer all four XOR combinations of the two bits (A2's argument, for each value of the
  third bit), so accuracy is at most 0.75. The learner sums in floating point, so at a constructed near-tie a rounding
  can flip a winner (G1(c) P2's panel: dev seed 7100, (ii) read 4 of 4 through the real read at an exact-arithmetic 2 of
  4). Here the bound is therefore a MEASUREMENT (B4), as A3 is for (ii).

### 2.4 The controls (each the method's complement, never the method)

- **Positive cell S0C0** taught to the incumbent: proves the channel teaches (the analogue of (ii)'s twin).
- **Explicit history** (`ExplicitHistory(reader, window=3)`): the same turn stream, but the reader's probe input is the
  episode's three turns concatenated. It adds exactly the cue and no capacity. It must solve S1C0 (the plan: "proven on
  the generator before the freeze", 2.13) and must NOT solve S1C1 (capacity still missing; specificity).
- **Presized nonlinear layer** (`presized_expansion(problem, H, seed, texts=None)`, one implementation; G5 reuses it as
  its matched-size control; `texts` defaults to the problem's distinct stimulus texts, byte-identical, and lets G5 add a
  non-adapted arm with medians over a stream-level text set): a FROZEN front-end in front of the incumbent's own class. z = W phi, phi the learner's own
  52-dim PSP-of-rank code, W H x 52 drawn N(0, 1) from `random.Random(seed * 7919 + H)`; x_j = max(0, z_j - m_j), m_j the
  median of unit j's drive over the PROBLEM's distinct stimulus texts (4 probe texts; 8 history texts; (ii)'s 4 texts),
  computed once, label-free; x is read by `SpikingInteractionLearner(n_in=H, t_post=6)` with the canonical stack, taught
  by `interact(x, label)`. **Disclosed:** it consumes (expanded stimulus, label), not `learn_turn(text)`, so it is a
  different front-end on the same teaching sequence, not the same channel; it proves that A nonlinear layer of size H
  solves the cell, not that a SMALL one does (at H 64 top-6 codes give distinct items near-distinct synapses). It must
  solve S0C1, S1C1 with history, (ii) XOR and (ii)'s twin, and must NOT solve S1C0 plain (no state; exact bound).
  H = **64 [PROJECTION]**, rule in 2.13. **P2 (G3, debate):** the tree already has a frozen random-feature expansion,
  `growth/ng_snn.py`'s feature mode (each LIF driven by the constant current A = W f over T steps). Before the freeze, on
  the dev seeds, NGSNN's feature mode at H units (its own draw, its LIF-rate nonlinearity, the code as f) is run as the
  front-end on every capacity cell; if it meets B7 on all four dev seeds it is ADOPTED (`presized_expansion` becomes a
  thin adapter over it, no second expansion); otherwise the centered expansion stays as a CONTROL instrument with a
  recorded disposition (instrument tier, never wired, exists to win or lose; the `gf.py` MLP.train precedent) and the
  measured reason. G3's PRESIZED FROZEN arm does NOT use it (it must share the growth arms' architecture); this control is
  reported beside G3's as G-R2's.

### 2.5 The channel, the incumbent and its identity check

The campaign teaches the CANONICAL incumbent through G0's calls: probe and context turns through the read
(`route`), teaching through the correction (`correct` == `learn_turn(text, None, feedback=label)`). It runs on the
factory (bare class + `PlasticityRuleStack(n_neurons=9, seed)`), because B8 proves it equal to
`make_brain(seed, persist=False).brain._ixn` under `brain.correct` / `brain.brain.route`. Context turns are READ, never
taught: through `chat()` / `respond()` every turn is self-labelled from the firewall and deposited, a fast-weight channel
(shared low empty buckets; the stack's global scaling) that the exact state bound excludes. G9 presents through its own
protocol; a `human=True` respond presentation of GOAL routes meets G0(b)'s clarify turn (1.5), so G9 and G5 read with
`human=False` or through `route` (G5 agreed, debate log).

### 2.6 The problem contract (`instruments/taught_problems.py`; G3, G5, G6, G9 import it)

- `Problem` = {id; intents; items; `delay_turns` (D = 1 for (i), 0 for (ii); G5's harness reports memory span as
  `delay_turns` x its own ticks per turn; the turn-to-tick mapping and reset-vs-carry at episode boundaries belong to the
  substrate's harness, never to the problem); `episodes(seed, split)` for split in `teach` / `validation` / `future`,
  each episode a tuple of turns (text, role `context` | `probe`, label or None), the episode being THE boundary unit; `teaching_sequence(problem, seed, epochs)` (the
  recorded per-episode (index, probe text, label) list an attempt teaches; G5's time-shifted control shifts it);
  `solved(reads)`; the budget fields (2.7)}. (ii) is wrapped as a Problem over `inseparable_mapping` (one-turn episodes).
- **Splits.** `teach`: E epochs, each the items once in `random.Random(seed).sample` order (one stream, as (ii)'s
  `_schedule`). `validation` and `future`: 4 passes over the items, orders from `random.Random(seed + 7_777_777)` and
  `random.Random(seed + 8_888_888)`. Held-out means NEVER-TAUGHT PRESENTATIONS of the same finite items (4 texts for (ii),
  8 episodes for (i)), not novel items; generalization to unseen items is out of G2's scope. **Disclosed (G6, debate):**
  for a STATELESS reader with learning OFF (the incumbent, any capacity-only grown readout), a read is a pure function of
  (weights, current text), so validation and future are the same reads in another order and cannot disagree, on (ii)
  and on (i). Future discriminates only for a stateful reader (grown memory units, the reservoir) or when something
  changes between the two reads; "judge on future data" is a repeat read otherwise. Reads are `route` calls and are not
  journaled; if R2 journals reads, validation and future presentations carry R2's held-out tag.
- **Learning OFF** (validation, future): no `correct` / `learn_turn` / `interact`, no `tick` (the sleep tick replays the
  ring, rewires and relaxes). Learning OFF freezes PLASTICITY, never STATE: context turns arrive through the read, so a
  stateful engine must step its state on reads (G3 adopted this for grown memory units, debate log). B12 checks that a
  validation leaves the learner's full state byte-identical.
- **The solved predicate (P1 case (a): accuracy 1.0, a fixed point on a normalized score).** `solved` holds iff every
  probe of the read is answered correctly through the real read: (ii) 16 presentations (the plan's "all four mappings
  answered correctly on every held-out presentation of its validation episodes"), (i) 32. A read with no presentations is
  refused. "Solves" in a control clause = `solved` at each of the last STABLE = 10 epoch-end validation reads of the E-epoch
  attempt (A4's stability); "never solves" = `solved` false at every epoch-end read.

### 2.7 The growth budget (the values live here; G3 owns the mechanism and cites them)

- **Unit:** grown units admitted to the active network for this problem, counted from the lake by G7's provenance label
  (the problem that grew a unit); candidate evaluations reported beside it (G3).
- **budget(P) = beta x baseline**, baseline = the median growth spend of SOLVED problems with POSITIVE spend, persisted
  and rebuilt from lake records (G3's ledger). A zero-spend solve never enters the baseline (a separable problem solved
  without growth would otherwise set the median to 0 and every later budget to 0; found in the debate, G3 conceded).
  beta = **2 [PROJECTION]**, a fixed point on the normalized ratio spend / baseline (P1 case (a)); its value is set on G3's
  dev seeds before G3's freeze and entered here by a dated amendment.
- **First problem** (no positive-spend solve exists): 52 units, the frozen code's width; the ratio spend / n_in = 1 is a
  fixed point on a normalized output (P1 case (a)); G3's value, adopted. **Lookup-bound check (strengthening):**
  budget >= N_codes(P), the number of distinct taught codes with history (4 for (ii) and A_ii, 8 for every (i) cell); G3
  asserts it at problem open. *(Amended by 2.15.6, Codex #4: a cost-budget pacing floor only, with no sufficiency claim.)*
- **Retry.** After SLEEP: no fresh budget (attempts only, growth frozen). After NEW TEACHING (a new teaching episode of the
  same problem, typed in `_T_EPISODE`): one fresh budget at the current baseline, at most K = **3 [PROJECTION, owner
  question O4]** per problem lifetime, after which the problem stays OPEN with growth exhausted. No schedule grows a
  problem without bound; budgets pace growth per problem, never cap the brain (owner 2026-09-30, "unbounded growth").
- **Reachability of the steady-state rule (G3, debate).** In a one-problem campaign every problem is the first
  positive-spend problem, so beta x baseline never runs; G3 registers a SEQUENCE reading ((ii), then (i) S0C1 with
  budget = beta x (ii)'s spend from a fresh-process ledger) and a ledger unit check, so the rule has a reachable red.
- **Attempt length** E = **400 epochs [PROJECTION]**, one value for every G2 problem and arm: max(TEACH_EPOCHS = 400,
  2 x the largest dev epochs-to-stable of any positive control, rounded up to a multiple of 50).

### 2.8 Checks, each with a constructed, reachable red (a red that does not fire halts its arc)

| id | check | constructed red (must fire) |
|---|---|---|
| B1 | vocabulary and code: seven distinct buckets, all >= 8, none of (ii)'s; every word of every presented text at a rank < t_post with ranks strictly by position (read off the learner's own `encode`); every text <= 6 words; every text firewall kind not `arm`/`lake` | a word in bucket < 8 (`eel` 4); a bucket collision (`ash`/`plum` 10); a 14-letter first word that reorders ranks; a 7-word history text (the cue at rank 6, PSP 0); `go home` in the vocabulary |
| B2 | structure on the learner's code: (a) the probe code is identical across cue values; (b) additivity phi00 + phi11 = phi01 + phi10 for S0C1 (probe code) and S1C1 (history code), exact on integer ranks, within 1e-12 in float, for each value of the third bit; (c) a witness weight setting reads S0C0 and S1C0-with-history 8 of 8 through the real read and the XOR cells at most 6 of 8 | (a) a probe that carries the cue word; (b) a variant whose q word's position depends on p; (c) the XOR labelling placed in the separable slot |
| B3 | state identity: in every validation read of the incumbent on S1C0 / S1C1 plain, each probe text gets the same answer under both cue values | the explicit-history reader standing in for the incumbent (dev 8900: it solves S1C0) |
| B4 | the incumbent never solves S1C0, S0C1 or S1C1 (plain) within E epochs, on every seed; S0C1's max accuracy reported (a float exceedance of 0.75 is reported as the A3-type measurement, not a halt) | the presized control standing in for the incumbent on S0C1 (the reader must see the solve) |
| B5 | the incumbent solves S0C0 on every seed | lr 0 (fails to solve; the stack's `stabilize` may still move weights, so the red is "not solved", never "unchanged"); shuffled teaching labels (each epoch a re-arrangement never equal to the truth, as (ii)'s `_shuffled`) |
| B6 | explicit history solves S1C0 and S0C0 and does NOT solve S1C1-with-history, on every seed | window 1 (the probe alone); the cue replaced by a constant word in the history text; for the "does NOT solve" clause, the presized front-end with history standing in for explicit history (it solves S1C1, the reader must see it) |
| B7 | the presized control solves S0C1, S1C1-with-history, (ii) XOR and (ii)'s twin, and does NOT solve S1C0 plain, on every seed | the identity front-end (no hidden layer: the incumbent) on S0C1; shuffled teaching labels; for the "does NOT solve" clause, the explicit-history reader standing in for the presized control on S1C0 (it solves it) |
| B8 | channel identity: per campaign seed, one `make_brain(seed, persist=False)` twin under `throwaway_home()` over the full S0C0 attempt; weights and every read equal the factory's at every epoch end | the bare class without the stack (dev 8900: differs) |
| B9 | the predicate reader: a planted all-right read is solved, one planted wrong answer is not, an empty read raises | a reader that ignores one presentation must miss the planted wrong answer |
| B10 | U's labels: balanced within 6 binomial sd per read, and at least one item carries both labels across the 4 validation passes on every registered seed | U's labels replaced by S0C0's (a function of the item) |
| B11 | budget: budget >= N_codes for every G2 problem; a zero-spend solve does not enter the baseline | a ledger with one zero-spend solve under the unfixed rule gives budget 0 (must red) |
| B12 | learning OFF: a validation leaves weights, `_authority`, the replay ring and the stack's state byte-identical | a validation that calls `correct` once |

(ii) adds, on this block's seeds and through the canonical learner (a strengthening; A1-A5 stay G1(c)'s): the
incumbent never solves (ii) XOR, solves the twin; the presized control solves both (B7).

### 2.9 The taught campaign and the G-R2 reading

- **Seeds** S = 8 (Part F), each seed s fixing for EVERY arm and cell the learner seed, the stack seed, the teach,
  validation and future orders and the presized draw (the paired-seed rule). The clauses are deterministic per-seed
  predicates that must hold on EVERY seed (A3 / A4's rule): one failure is a counterexample, so no t statistic and no
  majority rule is read.
- **Order (each stage halts on its own red not firing):** B9, B1-B2, B10-B12 (no learner) -> B8 on every seed -> B3's red
  -> the arms: incumbent on S0C0, S1C0, S0C1, S1C1; explicit history on S0C0, S1C0, S1C1; presized on S0C1, S1C1 with
  history, S1C0 plain; the incumbent on `A_ii` (it must solve it, B5's form; a failure reads clause 2's row for `A_ii`
  only); then (ii) (incumbent XOR and twin; presized XOR and twin) if 2.12's dependency is met.
- **Reading (a pure function; first row that applies decides; one constructed red per row):**

  | inputs | G-R2 reads |
  |---|---|
  | B8 fails on any seed, or any check's red does not fire | HALT (shared instrument; no reading) |
  | B1, B2 or B10-B12 fails | RED: construction wrong (the generator) |
  | B3 fails, or the presized control solves S1C0 plain (both impossible under the exact state bound) | HALT: a reader carries hidden state; the instrument is broken, not the construction |
  | the incumbent solves S1C0, S0C1 or S1C1 on any seed (B4), or explicit history solves S1C1 with history (B6's specificity clause) | RED: construction wrong (clause 1's analogue; a capacity-cell solve is a float near-tie in the code, reported with B2(c)'s witness margin, never a control failure) |
  | the incumbent fails S0C0 on any seed (B5) | RED: the channel does not teach (clause 2's analogue); halt, to the owner (it would contradict G0's PASS) |
  | explicit history fails S1C0 (B6) | RED: the delayed-cue explicit-history control fails (clause 4) |
  | the presized control fails any capacity cell (B7) | RED: the presized nonlinear positive control fails (clause 3) |
  | otherwise | PASS: S0C0 positive; S1C0 STATE; S0C1 CAPACITY; S1C1 STATE + CAPACITY |

  (ii) reads the same table on its rows (incumbent solves XOR -> clause 1; twin not learned -> clause 2; presized fails ->
  clause 3), or NOT RUN under 2.12. **Two instruments on clause 1 for (ii):** G1(c)'s A3 certifies it on the bare learner
  (7000-7007); this canonical re-read is REPORTED BESIDE that verdict. A disagreement (a float solve here on any seed) is
  the near-tie carried opening (g1c 2.8), goes to the owner, and never reads RED against G1(c)'s record. Clauses 2 and 3
  (twin, presized) are G2's alone and read normally.
- **Output for G3:** the frozen problems, the per-seed table (every epoch-end accuracy per arm and cell, max accuracy,
  first solved read) and one machine-readable verdict per problem, written outside the repo; the results record is
  section 5.

### 2.10 Null dispositions (each a causal claim; its instrument shows its positive control first)

- "The construction is wrong" is the claim a RED makes. Its instruments owe their reds before any incumbent read: the
  identity check (B3's stateful stand-in), the predicate reader (B9's planted reads), the never-solves reader (B4's
  presized stand-in). A RED with a blind instrument reads HALT, never RED.
- A RED deletes or repairs the CONSTRUCTION (the generator, a control), under P8 (deleted, not weakened); the problem is
  not offered to G3, G5, G6 or G9 until a re-registered construction passes. No owner-drawn part (P4d: the two networks,
  the reservoir, the return path) is touched by any G2 result. A RED on S0C0 impugns G0's channel and goes to the owner.
- A PASS adds nothing to the engine: it certifies problems.

### 2.11 Seeds (block 8000-8999)

8000-8003 development (exposed: E, H, the explicit-history proof, A_ii's reading); 8004-8099 further development if a
dev reading must be repeated (each use disclosed); 8100-8199 held, unassigned; 8200-8207 the confirmatory taught
campaign (read once); 8300-8307 its reserve (one repair after inspection; a second repair halts G2 and goes to the
owner); 8500-8599 planted reds; 8900-8999 scratch (8900, 8901 already exposed, 1.8).
Never run here: 300-361, 6000, 6001, 6100-6107, 7000-7007 (G1(c)'s); 7100-7103 are not read.

### 2.12 Dependencies and what waits

- **(ii)'s taught campaign WAITS for G1(c)'s construct-stage verdict** (A1-A5 on 7000-7007). If it reds or was not run,
  (ii) is NOT RUN here and goes to the owner (g1c 2.8's last bullet). Nothing here imports `G2II_CHECK_SEEDS` into code
  that runs.
- **(i) waits on nothing upstream.**
- **Downstream:** G3's confirmatory campaign waits for a PASS on a capacity cell ((i) S0C1 or (ii)); G5's Part D reads the
  predicate per cell (S1 cells as STATE, S0C1 as CAPACITY); G6 and G9 read the predicate; G4's T2 positive control reads
  `witness_units(problem)` (debate log).

### 2.13 Freeze, cost and arc split

- **Two-step freeze (g1c 2.8's precedent).** This text, then a dated PRE-DATA amendment from dev seeds 8000-8003 only,
  before any of 8200-8307 is read: E; H = 2 x the smallest power of two in {16, 32, 64} at which the presized control
  solves every capacity cell on all four dev seeds within E / 2 epochs (128 tried if none; a failure at 128 redesigns the
  control before the freeze); the P2 adopt-or-control reading of NGSNN's feature mode (2.4); the explicit-history proof
  (S1C0 solved, S1C1 not, all four dev seeds); A_ii's A1/A4 readings; beta from G3.
- **Cost at real scale [PROJECTION, scaled from 8900 / 8901].** (i): 8 seeds x 10 arm-cells x 400 epochs; incumbent and
  history runs about 3 s each, presized runs about 8-10 s each, so about 7 min in one process. (ii): 8 x 4 runs, about
  3.5 min. B8 twins: 8 brains x 1.8 s plus the attempt, about 1 min. Reds and checks: about 2 min. Total about 15 min in
  one process, about 5 min at 4 workers. No pytest suite, no recording, no live seed was run for this estimate.
- **Arc split (one arc per part, stacked on `claude/g2`; each arc: census, Codex gpt-6.1-sol panel via the orchestrator,
  fold stated as re-panelled or battery-verified).**
  - P0: this file; the plan's dated note on the G2 row (orchestrator-only: PLAN-FINAL).
  - P1: `instruments/delayed_cue.py` (generator, cells, U, A_ii, `ExplicitHistory`, `witness_units` and `decoy_units`
    for G4's T2 control, B1-B3, B10 with reds) + `tests/test_delayed_cue.py`. `witness_units(problem)` returns the
    problem's two hand threshold units over the frozen code (for (ii) it delegates to `inseparable_mapping`'s A5 witness,
    never a copy); its check: 4 of 4 (8 of 8) through the two units, the XOR labelling at most 0.75 under any linear
    readout of the code. `decoy_units(problem, n, seed)`: n random threshold units over the same code, the caller's seed.
  - P2: `instruments/taught_problems.py` (Problem contract, splits, learning-OFF check, predicate, budget fields, (ii)'s
    wrapper; B9, B11, B12) + `tests/test_taught_problems.py`.
  - P3: `instruments/taught_campaign.py` (the factory and B8 twin, `presized_expansion` and B7, the arms, the G-R2 reading
    with a red per row, the campaign driver, preflight refusals: dirty tree, OUT inside the repo, a live seed) +
    `tests/test_taught_campaign.py`.
  - Each new module's registrations land in the commit that adds it: `src/snn_data_lake/__init__.py` (REGIONS),
    `run_selfchecks.py`, `gates/dormancy_gate.py` (EXEMPT row, governed benchmark), and `tests/test_g8_move_gate.py`'s
    `_MOVED` pin (292 + 3). These four are hot shared files (claimed only at build time).
  - P4: dev measurement and the dated amendment (freeze); P5: the campaign, the results record, the closing review, ONE
    battery by the orchestrator, closing record last.

### 2.14 Scope

Built: 2.13's modules. Not built: any change to the learner, the channel, G0's provenance, the brain, the growth loop or a
readout; G3's certificate, loop, ledger and grown-segment port; G9's presentation protocol. Not claimed: that small
growth suffices for a capacity cell (the presized control's H is a positive control, not a bound), any engine or
reservoir benefit, generalization to unseen items.

### 2.15 Lead read, rulings and Codex fold (2026-10-03)

Lead read: board `p0_reports\G2.md` L1-L7; orchestrator rulings 7, 9, 9a, 10, 12 (C1-d option B) and 14 (A_ii as A for
C1-d); the Codex critique, folded (2.16). Each item below amends the numbered section it
names; where it contradicts the earlier text, this section governs until the fold merges it in.

**2.15.1 C1-d, the held-out capacity cell (L1, L7; answers O1 and G3's D1).** S0C1 stays as the same-items cell bounding
the incumbent. C1-d becomes the capacity cell G3 decides on, because S0C1's held-out is a repeat read for a stateless reader
(2.6). C1-d adds novel COMBINATIONS of taught words.
- **Items.** One-turn episodes (`delay_turns` 0). Each episode is one probe text `"P[p] Q[q] R1[a] R2[b]"`:
  - p, q in {0,1}; a, b in {0,1,2,3} (ruling 12, option B: 4 words per distractor slot);
  - four words at fixed positions, so the ranks follow position (ranks 0-3 by the later-first rule). The empty-bucket
    fillers take ranks 4-5 from buckets 0-1, the same in every text.
  - Label = p XOR q (bit 0 -> GREET, bit 1 -> GOAL). The distractors R1, R2 never affect the label.
  - 4 (p,q) cells x 16 distractor combos (a,b) = 64 items.
- **Vocabulary (B1-C1d).** 12 words: 2 P, 2 Q, 4 R1 and 4 R2. All 12 crc32 % 52 buckets are distinct, >= 8 and disjoint
  from (i) {14, 19, 21, 23, 38, 49, 50}, (ii) {28, 31, 41, 51} and A_ii {15, 16, 20, 26}. The free set has 29 buckets:
  {8-13, 17, 18, 22, 24, 25, 27, 29, 30, 32-37, 39, 40, 42-48}. All words have three letters.
  - **Disclosed:** the 12 words and their buckets are NOT listed yet. Computing crc32 is a run, and the host is quiesced
    for G1(c)'s campaign. They are chosen by the rule above at P1 and entered by the dated pre-data amendment (2.13),
    before any 8000-8307 seed is read. B1 reads the list off the learner's own `encode`.
  - B1's reds extend to C1-d: a bucket collision between two C1-d words; a C1-d word in a bucket of (i), (ii) or A_ii; a
    word in a bucket < 8.
- **Curriculum pairing (ruling 14).** For B = C1-d, task A is `A_ii` (2.2) on its own words, intents GREET / GOAL shared.
  `A_ii`'s buckets {15, 16, 20, 26} lie outside the free set, so they are disjoint from C1-d's 12 by construction. B1
  asserts the empty intersection; its red is a C1-d word planted in bucket 15.
- **Splits (replace 2.6's split rule for C1-d only; ruling 12).** The 16 distractor combos are partitioned per seed s into
  four disjoint splits: teach 8, validation 2, future 3 and future_d 3. The counts may only go up.
  - Each combo carries all four (p,q) cells, so every split is stratified over (p,q), and every judging split keeps at least
    2 combos per (p,q) cell: teach 32 items, validation 8, future 12, future_d 12.
  - RNG: `random.Random(f"g2.c1d-split:{s}")` draws the partition. A draw is REJECTED (redrawn from the same stream) if
    any of the 8 distractor words is absent from teach, so no read ever drives an untaught synapse.
  - Teach epochs order the 32 teach items as in 2.6. Each read is 4 passes:
    - validation: 32 presentations;
    - future and future_d: 48 presentations each.
    Validation and future are ordered by 2.6's offset streams (+7 777 777, +8 888 888); future_d by
    `random.Random(seed + 9_999_999)`.
  - Consumers state their split (ruling 12): G3 selects on validation and judges on future; G6(b) judges on future; G6(d)
    re-solves on future_d, its own untouched set.
  - The splits are disjoint item sets, so a validation solve no longer entails a future or future_d solve (the 2.6
    disclosure does not apply to C1-d).
  - Novel WORDS (substitution) are unreachable by construction: a non-colliding bucket's synapse never receives teaching
    signal, so a held-out split over novel words would measure the initial weights. Not offered.
- **Certificate (B2(b)-C1d).** For EVERY distractor combo (a,b), the four core codes satisfy phi00 + phi11 = phi01 + phi10.
  This is exact on integer ranks (fixed word count, fixed positions, buckets >= 8 and distinct) and within 1e-12 in float.
  - The union of teach, validation and future is therefore inseparable: one combo's Farkas witness, with multiplier 0 on
    every other item, is a witness for the union. No linear reader of the 52-code is right on every item the predicate
    reads, so the cell is a capacity cell on its whole item set.
  - Reds: a variant whose Q position depends on p (B2(b)'s red); a collision planted between an R word and a P word
    (additivity must break for the combos that contain it).
- **Split check (B13).** The combos are pairwise disjoint across the four splits, the split sizes are at least 8 / 2 / 3 / 3,
  each split holds all four (p,q) cells, and every distractor word appears in teach. Reds:
  - a partition with one combo in both future and future_d;
  - a partition missing R1[3] from teach;
  - a future_d of 2 combos.
- **Held-out red and positive (B14).**
  - The red is a whole-code lookup MEMORIZER: a dict from each teach item's code to its label, with a default answer on an
    unseen code. It must solve teach and must NOT solve future or future_d (their codes never occur in teach).
  - The positive is the hand OR-minus-AND witness, `witness_units(C1-d)`: two threshold units over the P[1] and Q[1]
    synapses only (OR: drive >= one bit; AND: both bits), with label = OR - AND, read through a linear output. It must
    read every presentation of validation, future and future_d correctly. It proves that the split admits a rule that
    generalizes.
  - If the memorizer solves future or future_d, or the witness fails any split, the split construction is wrong (RED:
    generator).
- **Arms (L4; appended to 2.9's order after the (i) arms, before (ii)).**
  - The incumbent on C1-d must never solve it on validation or future (B4's form; a float exceedance of 0.75 is reported
    as the A3-type measurement).
  - The presized control on C1-d must solve validation AND future (B7's form). Medians are computed over the TEACH
    split's texts only, so no validation or future text enters the front-end (label-free, but transductive otherwise).
  - **future_d is read by NO learner arm of G2's** (Codex #1). The incumbent, the presized control and the explicit-history
    reader never read it, and neither does epoch-end logging. Only B13's split check and B14's memorizer and hand witness
    touch it; these are construction checks with no taught weights.
    - The campaign driver ENFORCES this: `episodes(seed, "future_d")` called from any learner arm raises, and the driver's
      preflight refuses an arm list that names future_d.
    - B15 (constructed red): a learner arm configured to read future_d must be refused before any read.
  - B13 and B14 run with no learner, before B8.
  - **The positive must be LEARNED (Codex #2).** The hand witness proves the split admits a generalizing rule
    (representability), not that a learner given the missing capacity acquires it. C1-d's PASS therefore requires the
    presized control, taught through the registered reader, to solve validation AND future on every seed.
    - The random expansion mixes distractor drive into every hidden unit, so this solve is not guaranteed the way the
      same-items solve is. The pre-data amendment may REDESIGN the control on dev seeds 8000-8003, before any
      confirmatory seed is read.
    - If no learned control solves future, the hand witness never substitutes. C1-d reads "REPRESENTABLE, NOT SHOWN
      LEARNABLE" (below), and G3's confirmatory run on C1-d waits for a separate learned positive.
- **Reading (2.9).**
  - The incumbent solving C1-d reads RED (construction wrong, B2(b)-C1d margin reported).
  - The memorizer solving future or future_d, or the witness failing any split, reads RED (generator).
  - The presized control failing C1-d's validation or future while B14's witness passes reads **REPRESENTABLE, NOT SHOWN
    LEARNABLE**. C1-d is NOT offered to G3's confirmatory campaign (G3's C16, learned-positive-before-confirmation,
    extends to C1-d) until a learned positive passes under a dated, re-registered control.
  - The presized control failing while the witness also fails reads RED (generator), as above.
  - A PASS (learned positive on validation and future, incumbent never solves, B13-B15 pass) adds "C1-d CAPACITY,
    held-out by novel combinations".
- **Contract.** C1-d is a `Problem` in `delayed_cue.py` / `taught_problems.py` (one problem module, P2, ruling 8):
  `episodes(seed, split)` returns its split items for split in `teach` / `validation` / `future` / `future_d`; `solved`
  reads the split's full presentation count (32 or 48); its budget fields follow 2.7.
  - B11 for C1-d is a COST-BUDGET check (2.15.6, Codex #4): budget >= 32 (the teach-code count), with the union count of
    64 reported beside it. No sufficiency is claimed from either count.

**2.15.2 Self-defects D1-D6 (L2), folded as written.**
- D1 (2.13): no row bumps `tests/test_g8_move_gate.py`'s `_MOVED` pin or any selfcheck or atlas count (ruling 7). Each P1-P3
  commit message states its module delta (+1 each, +3 in all); the integrator re-pins once per integration. The
  registrations themselves (REGIONS, `run_selfchecks.py`, the dormancy EXEMPT row) still land with each module.
- D2 (2.13 P1, `witness_units`): "XOR at most 0.75 under ANY linear readout" is not an enumerable check. It is restated as
  IMPLIED by B2(b)'s additivity identity, which is exact on integer ranks. The float side is a measurement: a
  feasibility LP on the float codes (infeasible, or the minimal violation is reported), with the witness margin
  reported. It is not a separate checkable claim.
- D3 (2.7, 2.12): see 2.15.3.
- D4 (1.8, 2.13): H = 64 was measured with MEMORY / EXPLORE, not the registered GREET / GOAL. The pre-data amendment
  re-derives H on GREET / GOAL on dev seeds 8000-8003 by 2.13's rule. The 1.8 measurement is design evidence only.
- D5 (2.2): the empty-bucket fillers of an n-word text are buckets 0..(5 - n): 0-4 for a 1-word context turn, 0-3 for a
  2-word probe, 0-1 for a 4-word history or C1-d text. B1 asserts this exact form off the learner's `encode`.
- D6 (P8 "what an invented method owes", item 3): per-item attribution is N/A for a construction certificate. G2 claims
  no gain. Its attribution analogue is B2(c)'s and B14's witness margins: which bits a solve rides on. Stated, not
  skipped.

**2.15.3 Dependencies (L5; amends 2.12).**
- The retry trigger (2.7, "new teaching") depends on R2's typed `_T_EPISODE` record, and the budget unit ("grown units
  ... counted by G7's provenance label") depends on G7's label gate.
- This wave's fallback, if either has not landed by G3's confirmatory campaign:
  - the retry trigger is a teaching-session counter on the Problem record, incremented by the caller at each new
    teaching session;
  - the budget unit is G3's ledger count of units admitted for the problem id.
  - Both are disclosed in G3's results as the fallback, and are replaced by R2's record and G7's label in the commit that
    lands them.
- C1-d waits on nothing upstream.

**2.15.4 Degenerate contrasts (rulings 9 and 9a; Codex #5).** G2's clauses are deterministic per-seed predicates that
must hold on every seed (2.9). G2 reads no paired contrast, so neither ruling binds anything today. If a later amendment
adds a paired contrast, a degenerate (zero-variance) sample is read as follows, and "degenerate" is printed beside the
verdict:
- a SUPERIORITY contrast (arm beats control) reads NOT MET (ruling 9). This is disclosed as stricter than
  `paired_contrast.contrast`'s zero-variance rule (G1(c) 2.3 precedent);
- a NON-INFERIORITY bar (d' = arm - control + margin) is decided by its common value (ruling 9a): > 0 reads MET,
  <= 0 reads NOT MET.

**2.15.5 Cost (L6; amends 2.13) [PROJECTION, scaled from 1.8's 7 ms per epoch at about 56 learner operations].**
- One C1-d epoch (option B, ruling 12) is 32 teach presentations plus a 32-presentation validation read, about 64
  operations, so about 8 ms. 400 epochs come to about 3.5 s for the incumbent and about 12-15 s for the presized control at
  H 64 (1.8's 2-3 s per 100 epochs, scaled by about 1.25). The memorizer, the witness and B13 cost under a second.
- 8 seeds x (incumbent + presized + the future reads) is about 2.5-3 min.
- Campaign total: about 18 min in one process, about 6 min at 4 workers. Nothing was run for this estimate.

**2.15.6 Codex fold: U and B11 (Codex #3, #4).**
- **U is a deterministic contradiction fixture for STATELESS readers (replaces 2.2's U bullet and B10; the 2^-32 claim
  is withdrawn).**
  - The label of item i at validation pass k (k = 0..3) is `h(seed, i) XOR (k mod 2)`, where h is a fixed crc32 bit of
    `f"{seed}:{i}"`. So every item carries GREET in two passes and GOAL in two, explicit opposite-label pairs.
  - The future split repeats the rule with the future offset stream's order. Teach labels follow the same rule over the
    teach passes.
  - Any reader whose answer is a function of (weights, current text) with learning OFF gives each item one answer across
    the four passes, so it is wrong on exactly two of them: never solved, exactly.
  - B10 now checks that every validation and future item appears with both labels. Its red: U's labels replaced by
    S0C0's (one label per item).
  - **Scope:** U certifies impossibility for stateless readers only. A reader carrying presentation state could track the
    pass parity. U is therefore NOT a fixture for G3's stateful branch (grown memory units) or the reservoir.
- **U_S, the stateful-branch fixture (separate).**
  - The validation and future labels of each READ are drawn fresh from
    `random.Random(f"g2.US:{seed}:{split}:{read_index}")`: one fair bit per presentation, read_index counting that
    split's reads within the run.
  - The reader never receives these labels. Teaching uses the separate teach stream, and learning is OFF on every read.
    So its answers are fixed by inputs that are independent of that read's label draw.
  - Over the draw of that stream, P(all 32 presentations correct) = 2^-32 per read, whatever the reader's state. This
    probability is stated over the specified label-stream experiment only; it is not a property of the reader.
  - B10-S checks that each read's labels come from its own stream index (no reuse across reads). Its red: two reads
    sharing one index.
  - G3 uses U for stateless branches and U_S for stateful ones; a "solved" on either reads as G3's dishonest-report red
    under its own C8.
- **B11 is a COST-BUDGET check, not a sufficiency check (amends 2.7's lookup-bound sentence).**
  - B11 asserts budget >= N_teach_codes (4 for (ii) and A_ii, 8 for each (i) cell, 32 for C1-d), with the union code
    count reported beside it.
  - It is a pacing floor only. It does NOT establish that the admitted candidate architecture can implement a
    memorizing or any other solution within that budget. A paused problem is never described as paused "above a
    sufficient budget".
  - 2.7's sentence "so a paused problem was never starved below the size at which a memorizing solution exists" is
    withdrawn.

### 2.16 Review record

- **Lead read:** 2026-10-03, board `p0_reports\G2.md` (L1-L7); orchestrator rulings 7, 8, 9, 9a, 10, 12 and 14
  (board `DEPENDENCIES.md`).
- **Advisor read:** the CLI lead session, Fable, 2026-10-03 (L7, the 12-bucket listing for C1-d).
- **Codex critique:** model gpt-6.1-sol. Reply file `scratchpad\codex_p0\b1.reply.txt` (lead session 161935bb...),
  sha256 `52b335d74ef7dea045a21b3e68ae15d4fb286c5bdd7f3ebec66f2e2e843c8f4c`. Section "G2", findings 1-5, all folded:
  1. BLOCKER: future_d was read by the incumbent. FIXED (2.15.1): no learner arm reads future_d, the driver refuses it,
     B15 is the red.
  2. MAJOR: the hand witness could stand in for a learned positive. FIXED (2.15.1): PASS requires the learned presized
     solve on validation and future; otherwise "REPRESENTABLE, NOT SHOWN LEARNABLE", and G3's C16 extends to C1-d.
  3. MAJOR: U's unsolvability and 2^-32. FIXED (2.15.6): U is an exact stateless contradiction fixture; U_S is the
     stateful fixture, whose probability is stated over its label-stream experiment only.
  4. MAJOR: B11 claimed sufficiency. FIXED (2.15.6): it is a cost-budget check with no sufficiency claim, and 2.7's
     sentence is withdrawn.
  5. MINOR: the 9a omission. FIXED (2.15.4).
- **Still owed before the freeze is complete** (the two-step freeze, 2.13): the pre-data amendment from dev seeds
  8000-8003, after the G1(c) campaign is DONE. It carries the 12 C1-d words and buckets, E, H on GREET / GOAL, the
  presized-on-C1-d dev reading (or a redesigned learned control), the NGSNN adopt-or-control reading, the
  explicit-history proof, A_ii's readings, and beta from G3.

### 2.17 Dated PRE-DATA amendment, 2026-10-06 (dev seeds 8000-8003 only; 2.13's second freeze step)

Read at head `9fc94e24` by `taught_campaign.dev_measurement` (no Brain built; `brain_lake.bin` mtime 2026-09-27 01:19:19
before and after) plus the same `_dev_row` instrument run without `all()` short-circuiting for R4, R5, R6. No seed of
8200-8307 and no live seed was read. This amendment carries NO beta and claims nothing about budget values (G3's
dev-run gap, accepted: a one-problem run never reaches beta x baseline; beta is G3's own dated amendment). Nothing in the
registration above is changed; this section adds readings and fixes the constants the registration named.

| reading | rule (2.13) | dev result | disposition |
|---|---|---|---|
| R1 H | 2 x smallest H in {16, 32, 64, 128} solving every same-items capacity cell on all four seeds within E/2 | H 16 fails (presized_history S1C1 on 8000), H 32 solves all four seeds on S0C1, S1C1 (history), ii.xor, ii.twin | **H = 64** (code constant unchanged) |
| R2 E | max(400, 2 x largest epochs-to-stable, rounded up to 50) | largest epochs-to-stable of any positive = 5 | **E = 400** (code constant unchanged) |
| R3 explicit history | S1C0 and S0C0 solved, S1C1 never, all four seeds | S1C0 stable at epoch 0-1, S0C0 stable at 2-5, S1C1 max accuracy 0.75 never stable, on 8000-8003 | **PROVEN** |
| R4 NGSNN feature mode (H 64) | adopt iff B7 on every seed | S0C1 stable on 8000 and 8002, not on 8001 and 8003 (max 0.75); S1C0 never (max 0.5) on all four | **NOT ADOPTED**: the centred `presized_expansion` stays a declared CONTROL instrument (2.4), measured reason: NGSNN's expansion meets B7 on 2 of 4 seeds |
| R6 presized control on C1-d | learned solve on validation AND future | at H 16, 32, 64, 128 on all four seeds: never stable-solved, future never solved (future accuracy 0.0-0.75; max validation accuracy 0.62-1.0 transient) | **C1-d reads REPRESENTABLE, NOT SHOWN LEARNABLE** (2.15.1, ruling 28). No redesign here (O11 is the owner's) |
| R5 A_c1d (incumbent, H 64, E 400) | solves validation and future on every seed | stable at epoch 0 and future solved on 8000-8003 | **PASS** |
| R5 A_ii (incumbent, E 400) | solves on every seed | not stable-solved on any seed: max accuracy 1.0 (transient) on 8000-8002, 0.5 on 8003 | **FAILS the registered requirement.** Not substituted, not tuned: reported to the lead (a_rows_ok = false) |

Realised split sizes at seed 8000 (distinct texts; learning presentations per read are 4 per text as registered): C1-d and
A_c1d each: teach 32 distinct codes (`n_teach_codes`), validation 8 texts (32 presentations), future 12 texts (48
presentations). `future_d` is not readable by any learner arm (2.15.1) and was not read here.

Raw rows: scratchpad `g2_dev.jsonl` (92 rows, the registered run, derived line `H 64, E 400, history_proof true,
ngsnn_adopt false, c1d_learned_at [], a_rows_ok false, largest 5`) and `g2_dev_full.jsonl` (32 rows, the complete R4/R5/R6
reads). Module self-checks at the same head: `taught_problems` OK (B9, B11, B12, 5 of 5 reds), `delayed_cue` OK (18 of 18
reds), `taught_campaign` OK (scratch seed 8900).

## 3. Open questions for the owner

- **O1** *(narrowed by 2.15.1, lead read 2026-10-03.)* Held-out means novel COMBINATIONS of taught words (C1-d), not novel
  words. Novel words are unreachable by construction (an untaught synapse). The same-items cells keep never-taught
  presentations, where a stateless reader's future read is a repeat read. Accept, or should G2 seek a different encoder
  for novel-word generalization (out of scope here)?
  *(Original text:)* Held-out for finite problems means never-taught presentations of the same items, not novel items (2.6). For a
  stateless reader, "judge on future data" (the G3 row) is then a repeat read; only stateful readers and retention on
  other problems are judged independently. Accept, or should G2 add a problem family with unseen items (generalization)?
- **O2** G2's problems are posed through `correct` and `route` (G0(a)'s correction and the learned read), never through
  `chat()`, whose firewall self-label on every context turn adds a fast-weight channel the exact state bound excludes
  (2.5). Accept?
- **O3** Intents GREET / GOAL for (i) as for (ii) (shared label neurons, G3's harder retention test); GOAL is a route cue,
  so a `human=True` respond presentation meets G0(b)'s clarify turn. Accept, or prefer GREET plus a non-cue intent? (ARM is
  the only other non-cue intent and is excluded.)
- **O4** Budget: first problem 52 units (n_in), beta (dev), K = 3 fresh budgets per problem lifetime from new teaching,
  none from sleep (2.7). Accept K?
- **O5** The presized control proves solvability by a nonlinear layer of size H (projection 64), not by a small one; G9's
  presized growth-OFF twin (G-R9 (i)) must be sized to the engine's final size, or a capacity cell cannot separate
  growth from presizing. Noted for G9.

## 4. Debate log

| # | from | challenge | answer | resolution |
|---|---|---|---|---|
| 1 | G2 -> G3 | G3 2.6's baseline (median spend of SOLVED problems) collapses to 0 when the separable A solves with zero growth, so B's budget is 0 | -- | G3 CONCEDED: baseline over solved problems with POSITIVE spend (2.7) |
| 2 | G2 -> G3 | two registrations of the budget numbers (G2 row and G3 2.6) is P2-additive | -- | G3 CONCEDED: values live in G2's Problem record; G3 owns the mechanism; the lookup-bound check adopted as a ledger assertion |
| 3 | G2 -> G3 | S1C1 reads STATE only under G3's three-way k rule, but needs capacity too | -- | G3 CONCEDED: the certificate runs on the history-augmented code in the same pass (STATE and CAPACITY) |
| 4 | G3 -> G2 | the Problem interface (items, episode boundaries, validation, future, solved, budget) | accepted as 2.6; learning OFF freezes plasticity never state, context turns arrive through reads | G3 added it to its 2.3 (memory units step on reads, reset at episode boundaries) |
| 5 | G3 -> G2 | who constructs curriculum A | G2 does (a problem instrument); S0C0 cannot be A for S0C1 (same texts relabelled) | G3 chose A_ii for B = (ii), (ii)'s twin for B = S0C1, intents SHARED (GREET / GOAL), disjoint reported beside; G2 moved (i) to GREET / GOAL accordingly |
| 6 | G3 -> G2 | which cell is the capacity cell; a DATA control | S0C1 and (ii); S1C0 state; S1C1 both; U as DATA / unsolvable | G3: U in G2's module, consumed by its C4 and C8(a) |
| 7 | G5 -> G2 | (i) is unsolvable by any readout on today's stateless substrate | partly refuted: only the S1 cells; S0C1 is solvable by a nonlinear readout of the current turn; G2's problems go to the chat learner, never NGSNN / OnlineReadout | G5 ACCEPTED: Part D reads S1 cells as STATE, S0C1 as CAPACITY; Part A does not touch G2's problems |
| 8 | G5 -> G2 | interface for Part D (split, predicate, recorded teaching sequence) | 2.6's `episodes`, `solved`, `teaching_sequence` | G5 ACCEPTED |
| 9 | G5 -> G2 | double credit if both use the single R-STDP layer | no shared run: G-R2 certifies problems on 8200-8207; Part D runs its own controls on its seeds, never citing G2's run | AGREED |
| 10 | G5 -> G2 | reuse the presized expansion as Part D's matched-size control (P2) | accepted as `presized_expansion(problem, H, seed)`; disclosed: per-problem threshold set, positive control only at certified H, no help on S1 cells | CLOSED: G5 takes the stimulus vector and carries all three disclosures into its Part D |
| 11 | G4 -> G2 | build a two-stage conjunction problem for T2's positive control | partly refuted: no new taught A needed (a taught A cannot fix which features its grown units carry); B = an existing capacity cell, the planted earlier units = `witness_units(problem)`; the "learner solves B with them" half is G3's grown-segment port's positive control | G4 ACCEPTED; G4 tells G3 about the port half |
| 12 | G2 -> G4 | planting exactly the witness makes T2's effect "wire to the two units that solve it"; decoys? | -- | G4: YES, `decoy_units(problem, n, seed)` (random threshold units over the same code), seeds from G4's block 10418-10429; the large planted effect is the intended size of a positive control, a dilution series reported beside it |
| 13 | G6 -> G2 | for (ii), future presents the same 4 texts as validation; if the read is order-insensitive, "judge the frozen winner on future" cannot fire independently | conceded and widened: exact for ANY stateless reader on (ii) and (i); future discriminates only for stateful readers; reads are not journaled (R2's tag if ever) | disclosed in 2.6 and O1; G6 reads (i) under a stateful reader and cross-problem retention as its untouched tests |
| 14 | G5 -> G2 | do not put ticks per turn in the Problem; add `delay_turns`; keep the episode as the boundary unit | conceded: `delay_turns` added (2.6); the mapping and reset-vs-carry are the harness's arms; a D sweep needs re-certification (the 6-word horizon caps explicit history at D <= 3) | CLOSED |
| 15 | G5 -> G2 | B6's "history must not solve S1C1" rests on a float MEASUREMENT; routing a solve there to clause 4 misattributes a construction near-tie as a control failure | conceded: a capacity-cell solve (incumbent or history) reads "construction wrong" with B2(c)'s witness margin; clause 4 keeps only "history fails S1C0"; a solve of an exact-bound state cell reads HALT (hidden state) | CLOSED (2.9's table) |
| 16 | G5 -> G2 | optional `texts=` on the expansion for a non-adapted arm | accepted, default byte-identical | CLOSED |
| 17 | G3 -> G2 | E = 400 per attempt? | G3 ACCEPTED (an attempt ends early at the stable-10 solve; growth granularity is the batch); G3 fixed its presized arms to the loop's cumulative exposure | CLOSED |
| 18 | G3 -> G2 | P2: `presized_expansion` may be a second frozen random-feature expansion beside NGSNN's feature mode; G3's PRESIZED FROZEN arm will not use it (architecture confound) | conceded the P2 question: NGSNN feature mode is run on the dev seeds as the front-end; adopted if it meets B7, else the centered expansion stays as a declared CONTROL instrument with the measured reason (2.4) | CLOSED on the rule; the reading is a pre-freeze dev measurement |
| 19 | G3 -> G2 | B6's float inconsistency (as row 15); state that a validation solve ENTAILS the future solve for stateless readers | both already conceded (rows 13, 15; 2.6, 2.9) | CLOSED; G3 raises "no unseen items" as its O9 (= this file's O1) |

**Scope accepted in the debate, for the orchestrator to ratify** (consumer-driven additions in G2's directory, each
accepted on P2's one-module grounds; requests from siblings, not owner instructions): the unsolvable cell U (G3); the
curriculum A problems `A_ii` and (ii)'s twin as A (G3); `witness_units` and `decoy_units` (G4's T2); `presized_expansion`
as a shared control with `texts=` (G5); `delay_turns` (G5); the NGSNN feature-mode adopt-or-control reading (G3).

## 5. Results

(Empty until the campaign's results record.)
